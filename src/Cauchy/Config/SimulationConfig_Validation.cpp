#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Input/StateInput.hpp"
#include "DNDS/Config/JsonSchema.hpp"
#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include <set>

namespace DNDS::Cauchy
{
    namespace
    {
        void PrimitiveValid(const std::array<real, 5> &values)
        {
            DNDS_check_throw_info(std::all_of(values.begin(), values.end(), [](real value)
                                              { return std::isfinite(value); }) &&
                                      values[0] > 0 && values[4] > 0,
                                  "Cauchy primitive input needs finite values and positive rho,p");
        }
    }
    void SimulationConfig::ValidateDomain(const DomainConfig &domain, int size)
    {
        const auto &settings = domain.settings;
        settings.implicit.ValidateComponents(settings.equation.eos == "perfectGas" ? 5 : INT_MAX - 1);
        settings.motion.Validate(settings.mesh.dimension);
        DNDS_check_throw_info(!settings.implicit.requireConvergedReconstruction || settings.numerics.reconstruction != "variational" || settings.numerics.reconstructionTolerance > 0,
                              "strict implicit VR acceptance requires a positive reconstructionTolerance");
        DNDS_check_throw_info(!domain.placement.name.empty() && !domain.placement.ranks.empty(), "Cauchy domain needs a name and ranks");
        std::set<int> ranks;
        for (int rank : domain.placement.ranks)
            DNDS_check_throw_info(rank >= 0 && rank < size && ranks.insert(rank).second, "invalid/duplicate Cauchy domain rank");
        const auto &equation = settings.equation;
        ResolveReferenceScales(equation);
        DNDS_check_throw_info(std::isfinite(equation.gamma) && equation.gamma > 1 && std::isfinite(equation.gasConstant) && equation.gasConstant > 0 &&
                                  std::isfinite(equation.viscosity) && equation.viscosity >= 0 && std::isfinite(equation.prandtl) && equation.prandtl > 0,
                              "invalid Cauchy constitutive constants");
        DNDS_check_throw_info(equation.transport != "inviscid" || equation.viscosity == 0, "inviscid configuration has unused nonzero viscosity");
        if (equation.eos == "perfectGas")
            EquationSystem::Resolve(equation);
        DNDS_check_throw_info(settings.mesh.dimension == 2 || settings.mesh.dimension == 3, "Cauchy geometric dimension must be 2 or 3");
        DNDS_check_throw_info(settings.execution.hostThreads >= 0 && settings.execution.hostThreads <= 1024, "invalid CPU worker count");
        DNDS_check_throw_info(settings.execution.device != "cuda" || settings.execution.deviceOrdinal >= 0,
                              "invalid CUDA device ordinal");
        const auto &numerics = settings.numerics;
        DNDS_check_throw_info(numerics.vrBoundaryValue == "physicalTrace" || numerics.vrBoundaryValue == "physicalMirrored",
                              "vrBoundaryValue must be physicalTrace or physicalMirrored");
        numerics.ValidatePositivity();
        DNDS_check_throw_info(numerics.reconstructionIterations > 0 && numerics.reconstructionResidualCheckInterval > 0 && std::isfinite(numerics.reconstructionTolerance) && numerics.reconstructionTolerance >= 0,
                              "invalid Cauchy numerical controls");
        numerics.cfv.Validate();
        numerics.limitedVR.Validate();
        const auto &riemann = numerics.riemann;
        DNDS_check_throw_info((numerics.flux != "Roe_M3" && numerics.flux != "Roe_M5") || riemann.fixScale > 0,
                              "low-dissipation Roe flux requires a positive fixScale divisor");
        DNDS_check_throw_info(std::isfinite(riemann.fixScale) && riemann.fixScale >= 0 && std::isfinite(riemann.incrementScale) && riemann.incrementScale >= 0 &&
                                  std::isfinite(riemann.dLambda) && riemann.dLambda >= 0 && std::isfinite(riemann.referenceMach) && riemann.referenceMach > 0 &&
                                  std::isfinite(riemann.pressureStencilMin) && riemann.pressureStencilMin > 0,
                              "invalid Cauchy Riemann controls");
        PrimitiveValid(settings.initialPrimitive);
        DNDS_check_throw_info(std::isfinite(settings.initialTemperature) && settings.initialTemperature >= 0 &&
                                  (equation.eos != "perfectGas" || settings.initialMassFractions.empty()),
                              "invalid constant initial thermodynamic input");
        const auto validateProfile = [](const FlowProfileSettings &profile)
        {
            DNDS_check_throw_info(profile.kind == "constant" || profile.kind == "isentropicVortex" || profile.kind == "doubleMachReflection" || profile.kind == "cylinderWake", "unknown physical field profile");
            DNDS_check_throw_info(std::isfinite(profile.period) && profile.period > 0 && std::isfinite(profile.strength) && std::isfinite(profile.perturbation) &&
                                      std::all_of(profile.center.begin(), profile.center.end(), [](real value)
                                                  { return std::isfinite(value); }),
                                  "invalid field profile parameters");
        };
        validateProfile(settings.initialProfile);
        for (auto length : settings.mesh.periodicLengths)
            DNDS_check_throw_info(std::isfinite(length) && length >= 0, "periodic lengths must be finite and nonnegative");
        for (const auto &[name, zone] : settings.mesh.boundaryIds)
            DNDS_check_throw_info(!name.empty() && zone != 0 && zone != 1 && zone >= -6,
                                  "mesh.boundaryIds reserves 0 for interior, 1 for null, and -1..-6 for periodic boundaries");
        for (const auto &[zone, condition] : settings.boundaries)
        {
            size_t parsed = 0;
            const auto id = std::stoll(zone, &parsed);
            DNDS_check_throw_info(parsed == zone.size() && id >= INT_MIN && id <= INT_MAX && std::to_string(id) == zone,
                                  "boundary key must be a canonical integer zone ID");
            PrimitiveValid(condition.primitive);
            DNDS_check_throw_info(condition.massFractions.empty() ||
                                      (equation.eos != "perfectGas" && (condition.kind == "farfield" || condition.kind == "analytic")),
                                  "boundary mass fractions require a prescribed mixture state");
            validateProfile(condition.profile);
            if (condition.kind == "totalConditionInlet")
                DNDS_check_throw_info(!condition.state.is_null() &&
                                          Physics::StateInput::Parse(condition.state).form == Physics::StateInput::Form::TotalPT,
                                      "totalConditionInlet requires a totalPT or totalPT_phy state vector");
            if (condition.kind == "noSlipWall" && condition.thermal == "isothermal")
                DNDS_check_throw_info(condition.temperature >= numerics.temperatureFloor, "wall temperature lies below the configured temperature floor");
            if (condition.kind == "farfield" && equation.eos == "perfectGas" && condition.state.is_null())
                DNDS_check_throw_info(condition.primitive[4] / (condition.primitive[0] * equation.gasConstant) >= numerics.temperatureFloor,
                                      "farfield temperature lies below the configured temperature floor");
            DNDS_check_throw_info(std::isfinite(condition.temperature) && condition.temperature > 0 &&
                                      std::all_of(condition.velocity.begin(), condition.velocity.end(), [](real v)
                                                  { return std::isfinite(v); }),
                                  "invalid boundary state");
            DNDS_check_throw_info(condition.kind == "noSlipWall" || (condition.thermal == "adiabatic" && (condition.kind == "slipWall" || condition.velocity == std::array<real, 3>{0, 0, 0})),
                                  "wall thermal/velocity parameters supplied to a non-wall condition");
        }
    }
    void NumericsSettings::ValidatePositivity() const
    {
        DNDS_check_throw_info(std::isfinite(temperatureFloor) && temperatureFloor > 0 && std::isfinite(densityFloor) && densityFloor > 0 &&
                                  std::isfinite(pressureFloor) && pressureFloor > 0 && std::isfinite(sensibleEnergyFloor) && sensibleEnergyFloor >= 0 &&
                                  std::isfinite(positivityBetaCutoff) && positivityBetaCutoff >= 0 && positivityBetaCutoff < 1 &&
                                  (positivityReconstruction == "Euler" || positivityReconstruction == "Segment") &&
                                  (positivityClippedReferencePolicy == "fallbackToBase" || positivityClippedReferencePolicy == "backtracking") &&
                                  positivityBetaPower >= 0 && positivityBetaPower < INT_MAX,
                              "invalid reconstruction positivity controls");
        DNDS_check_throw_info(!positivity || positivityReconstruction != "Segment" || (positivityCompressToMean && positivityBetaPower == 1),
                              "Segment compatibility does not implement Euler order-reference or beta-power controls");
    }
    void DITRTimeSettings::Validate() const
    {
        DNDS_check_throw_info((variant == "U2R2" || variant == "U2R1") && std::isfinite(fraction) && fraction > 0 && fraction < real(2) / 3 &&
                                  std::isfinite(theta) && theta > 0 && (!iterations || *iterations > 0) && (!minIterations || *minIterations >= 0) &&
                                  (!iterations || !minIterations || *minIterations <= *iterations) &&
                                  (!relativeTolerance || (std::isfinite(*relativeTolerance) && *relativeTolerance >= 0)) &&
                                  (!absoluteTolerance || (std::isfinite(*absoluteTolerance) && *absoluteTolerance >= 0)),
                              "invalid/unavailable DITR positive-stage-scale controls");
    }
    ImplicitSettings DITRTimeSettings::ResolveControls(const ImplicitSettings &common) const
    {
        Validate();
        auto result = common;
        if (iterations)
            result.iterations = *iterations;
        if (minIterations)
            result.minIterations = *minIterations;
        if (relativeTolerance)
            result.relativeTolerance = *relativeTolerance;
        if (absoluteTolerance)
            result.absoluteTolerance = *absoluteTolerance;
        DNDS_check_throw_info(result.iterations > 0 && result.minIterations >= 0 && result.minIterations <= result.iterations &&
                                  std::isfinite(result.relativeTolerance) && result.relativeTolerance >= 0 &&
                                  std::isfinite(result.absoluteTolerance) && result.absoluteTolerance >= 0,
                              "DITR overrides conflict with the common implicit iteration/tolerance controls");
        return result;
    }
    void PhysicalTimePositivitySettings::Validate() const
    {
        DNDS_check_throw_info(std::isfinite(relax) && relax > 0 && relax <= 1 && std::isfinite(scale) && scale > 0 &&
                                  std::isfinite(increaseLimit) && increaseLimit >= 1,
                              "invalid Euler physical timestep positivity/growth controls");
    }
    void ImplicitSettings::Validate() const
    {
        ValidateComponents(5);
    }
    void ImplicitSettings::ValidateComponents(index components) const
    {
        DNDS_check_throw_info(components > 0 && components < INT_MAX, "invalid implicit equation dimension");
        const auto validMatrixTable = [](const ImplicitMatrixTableSettings &table)
        {
            return (table.layout == "aos" || table.layout == "soa" || table.layout == "aosoa32") && table.rowPaddingMultiple > 0;
        };
        const bool usesAMG = linear.method == "amgx" || (linear.method == "gmres" && linear.preconditioner == "amgx");
        DNDS_check_throw_info(validMatrixTable(linear.matrixStorage.diagonal) && validMatrixTable(linear.matrixStorage.inverse) &&
                                  validMatrixTable(linear.matrixStorage.offDiagonal) &&
                                  (linear.matrixStorage.indices == "auto" || linear.matrixStorage.indices == "int32" || linear.matrixStorage.indices == "int64"),
                              "invalid independent linear matrix storage settings");
        DNDS_check_throw_info(!usesAMG || (linear.matrixStorage.indices != "int64" && linear.matrixStorage.diagonal.layout == "aos" &&
                                           linear.matrixStorage.offDiagonal.layout == "aos"),
                              "copy-free AmgX requires primary int32 sparse indices and aos diagonal/off-diagonal blocks");
        DNDS_check_throw_info((normOrder == 1 || normOrder == 2) &&
                                  (residualReference == "initial" || residualReference == "runningMaximum"),
                              "invalid implicit residual norm/reference/components");
        std::vector<std::string> selected;
        for (const auto &component : residualComponents)
        {
            DNDS_check_throw_info(!component.empty() && std::find(selected.begin(), selected.end(), component) == selected.end(), "implicit residual components must be distinct nonempty names");
            selected.push_back(component);
        }
        const auto nonnegative = [](real value)
        { return std::isfinite(value) && value >= 0; };
        std::vector<int> updated;
        for (const auto component : multigrid.updateComponents)
        {
            DNDS_check_throw_info(component >= 0 && component < components && std::find(updated.begin(), updated.end(), component) == updated.end(), "pMG update components must be distinct equation indices");
            updated.push_back(component);
        }
        DNDS_check_throw_info((diagonal == "entries" || diagonal == "block") && (!roeDissipation || diagonal == "block") &&
                                  (boundaryJacobian == "physical" || boundaryJacobian == "eulerScalar") &&
                                  (jacobianAction == "assembled" || jacobianAction == "directEvaluated" || jacobianAction == "semiMatrixFree") &&
                                  !updatePolicy.empty() &&
                                  iterations > 0 && minIterations >= 0 && minIterations <= iterations && consoleInterval >= 0 &&
                                  nonnegative(relativeTolerance) && nonnegative(absoluteTolerance) &&
                                  std::isfinite(pseudoCFL) && pseudoCFL > 0 && std::isfinite(maxPseudoCFL) && maxPseudoCFL >= pseudoCFL &&
                                  std::isfinite(pseudoGrowth) && pseudoGrowth >= 1 && std::isfinite(relaxation) && relaxation > 0 && relaxation <= 1,
                              "invalid implicit nonlinear/diagonal/pseudo-time controls");
        DNDS_check_throw_info((linear.cudaJacobiMapping == "thread" || linear.cudaJacobiMapping == "componentGroup") &&
                                  (linear.cudaDiagonalInverse == "batchedSVD" || linear.cudaDiagonalInverse == "guardedLU") &&
                                  linear.diagonalInverseRefinement >= 0 && linear.diagonalInverseRefinement <= 3 &&
                                  (!linear.diagonalInverseRefinement ||
                                   (diagonal == "block" && (linear.method == "sgs" || linear.method == "jacobi" ||
                                                            (linear.method == "gmres" && (linear.preconditioner == "sgs" || linear.preconditioner == "jacobi"))))),
                              "block-inverse refinement requires 0..3 corrections and block-diagonal SGS/Jacobi");
        DNDS_check_throw_info((linear.method == "sgs" || linear.method == "jacobi" || linear.method == "ilu" || linear.method == "amgx" || linear.method == "gmres") && (linear.preconditioner == "sgs" || linear.preconditioner == "ilu" || linear.preconditioner == "jacobi" || linear.preconditioner == "amgx") &&
                                  linear.jacobiLocalSweeps >= 1 && linear.jacobiLocalSweeps <= 2 &&
                                  (linear.componentScaling == "none" || linear.componentScaling == "meanState") &&
                                  (linear.stopping == "residual" || (linear.stopping == "fixedCount" && linear.method != "gmres")) &&
                                  linear.iterations > 0 && linear.residualCheckInterval > 0 && linear.subspace > 0 && linear.subspace < INT_MAX &&
                                  nonnegative(linear.relativeTolerance) && nonnegative(linear.absoluteTolerance) && sgs.sweeps > 0 && ilu.fill >= -1 && ilu.fill <= 5 && ilu.preconditionerCorrections > 0 &&
                                  (ilu.ordering == "natural" || ilu.ordering == "rcm") &&
                                  !((linear.method == "ilu" || linear.method == "amgx" || linear.method == "jacobi" || (linear.method == "gmres" && (linear.preconditioner == "ilu" || linear.preconditioner == "jacobi" || linear.preconditioner == "amgx"))) && sgs.exchangeBetweenSweeps),
                              "invalid implicit linear/SGS/ILU controls");
        DNDS_check_throw_info((positivity.rhsFPPMode == "none" || positivity.rhsFPPMode == "rhs" || positivity.rhsFPPMode == "dTau") &&
                                  (positivity.rhsFPPReduction == "cell" || positivity.rhsFPPReduction == "minimumLimited" || positivity.rhsFPPReduction == "minimumAll") &&
                                  nonnegative(positivity.rhsFPPScale) && std::isfinite(positivity.rhsFPPRelax) && positivity.rhsFPPRelax > 0 && positivity.rhsFPPRelax <= 1 &&
                                  std::isfinite(positivity.incrementRelax) && positivity.incrementRelax > 0 && positivity.incrementRelax <= 1 &&
                                  positivity.compressionPower > 0 && positivity.compressionPower <= INT_MAX / 2,
                              "invalid implicit Euler-PP controls");
        DNDS_check_throw_info(multigrid.levels >= 0 && multigrid.levels <= 2 && multigrid.cycles > 0 &&
                                  multigrid.preSmooth >= 0 && multigrid.postSmooth >= 0 && multigrid.coarseSmooth > 0 &&
                                  (!multigrid.levels || (multigrid.preSmooth <= iterations && multigrid.postSmooth <= iterations && multigrid.coarseSmooth <= iterations)) &&
                                  std::isfinite(multigrid.damping) && multigrid.damping > 0 && multigrid.damping <= 1,
                              "invalid polynomial FAS controls");
        for (const auto &[name, level] : multigrid.coarseLevelSettings)
        {
            DNDS_check_throw_info((name == "1" || name == "2") && level.preSmooth >= -1 && level.postSmooth >= -1 &&
                                      level.preSmooth <= iterations && level.postSmooth <= iterations,
                                  "invalid coarse level key or smoothing count");
            const index number = name == "1" ? 1 : 2;
            if (number == multigrid.levels)
            {
                const index pre = level.preSmooth < 0 ? multigrid.coarseSmooth : level.preSmooth;
                const index post = level.postSmooth < 0 ? 0 : level.postSmooth;
                DNDS_check_throw_info(pre <= iterations - post && pre + post > 0, "invalid bottom-level correction count");
            }
            ResolveCoarseLevel(number);
        }
    }
    ImplicitSettings ImplicitSettings::ResolveCoarseLevel(index level) const
    {
        DNDS_check_throw_info(level == 1 || level == 2, "coarse smoother level must be 1 or 2");
        auto result = *this;
        result.multigrid.levels = 0;
        result.multigrid.coarseLevelSettings.clear();
        const auto found = multigrid.coarseLevelSettings.find(std::to_string(level));
        if (found != multigrid.coarseLevelSettings.end())
        {
            Config::ValidatePartialJsonSchema(found->second.solver, MultigridSmootherSchema(),
                                              "/implicit/multigrid/coarseLevelSettings/" + std::to_string(level) + "/solver");
            Json merged = result;
            merged.merge_patch(found->second.solver);
            result = merged.get<ImplicitSettings>();
        }
        result.Validate();
        return result;
    }
    DomainSettings ResolveSweepOrdering(DomainSettings settings)
    {
        if (settings.numerics.cfv.SORInstead)
            settings.numerics.cfv.coreFirst = true;
        return settings;
    }
    void ExecutionSettings::ValidateMPIThreadSupport(int provided) const
    {
        DNDS_check_throw_info(hostThreads == 0 || provided >= MPI_THREAD_FUNNELED,
                              "Cauchy CPU workers require at least MPI_THREAD_FUNNELED; disable workers or request a higher MPI thread level");
    }
    CFV::ReconstructionDeviceKernel ExecutionSettings::ReconstructionKernel() const
    {
        using Kernel = CFV::ReconstructionDeviceKernel;
        if (reconstructionKernel == "fp64")
            return Kernel::FP64;
        if (reconstructionKernel == "fp64Shared")
            return Kernel::FP64Shared;
        if (reconstructionKernel == "fp64Wide")
            return Kernel::FP64Wide;
        if (reconstructionKernel == "fp64Static")
            return Kernel::FP64Static;
        if (reconstructionKernel == "fp64MatrixShared")
            return Kernel::FP64MatrixShared;
        if (reconstructionKernel == "fp64Tensor")
            return Kernel::FP64TensorCore;
        if (reconstructionKernel == "fp64TensorShared")
            return Kernel::FP64TensorCoreShared;
        if (reconstructionKernel == "fp64TensorAsync")
            return Kernel::FP64TensorCoreAsync;
        if (reconstructionKernel == "fp32")
            return Kernel::FP32;
        if (reconstructionKernel == "fp32Accum")
            return Kernel::FP32FullAccumulation;
        if (reconstructionKernel == "fp32FloatFloatAccum")
            return Kernel::FP32FloatFloatAccumulation;
        if (reconstructionKernel == "tf32")
            return Kernel::TF32TensorCore;
        DNDS_check_throw_info(false, "unknown reconstruction kernel: " + reconstructionKernel);
        return Kernel::FP64;
    }
    ResolvedExecutionPlan ResolvedExecutionPlan::Resolve(const DomainConfig &domain)
    {
        const auto &s = domain.settings;
        std::vector<std::string> missing;
        auto require = [&](bool found, const std::string &operation)
        { if (!found) missing.push_back(operation); };
        const bool cuda = s.execution.device == "cuda";
        const auto reconstructionKernel = s.execution.ReconstructionKernel();
        require(cuda || reconstructionKernel == CFV::ReconstructionDeviceKernel::FP64,
                "nondefault reconstruction kernel requires CUDA");
#ifndef DNDS_USE_CCCL_COMPUTE
        require(reconstructionKernel != CFV::ReconstructionDeviceKernel::FP32FloatFloatAccumulation,
                "fp32FloatFloatAccum reconstruction requires DNDS_USE_CCCL_COMPUTE");
#endif
        const auto &kernels = s.execution.kernels;
        require(cuda || kernels.Default(), "operator compute/layout experiments require CUDA");
        require(kernels.rhsMapping == "thread" || kernels.rhsMapping == "warpQuadrature", "RHS kernel mapping " + kernels.rhsMapping);
        require(kernels.rhsBlockSize == 64 || kernels.rhsBlockSize == 128 || kernels.rhsBlockSize == 256, "RHS block size must be 64, 128 or 256");
        require(!kernels.rhsDoubleBuffer || (kernels.polynomialArithmetic == "fp64" &&
                                             kernels.dissipationArithmetic == "fp64" && kernels.fluxArithmetic == "fp64"),
                "double buffering currently qualifies FP64 only");
        require(kernels.rhsMapping == "thread" || (kernels.polynomialArithmetic == "fp64" &&
                                                   kernels.dissipationArithmetic == "fp64" && kernels.fluxArithmetic == "fp64"),
                "warp quadrature currently qualifies FP64 arithmetic only");
        for (const auto *layout : {&kernels.stateInputLayout, &kernels.reconstructionInputLayout})
            require(*layout == "aos" || *layout == "aosoa32", "kernel input layout " + *layout);
        for (const auto *precision : {&kernels.polynomialArithmetic, &kernels.dissipationArithmetic, &kernels.fluxArithmetic})
        {
            require(*precision == "fp64" || *precision == "fp32" || *precision == "f32f32", "kernel arithmetic " + *precision);
#ifndef DNDS_USE_CCCL_COMPUTE
            require(*precision != "f32f32", "f32f32 kernel arithmetic requires DNDS_USE_CCCL_COMPUTE");
#endif
        }
        require(kernels.fluxArithmetic == "fp64" || kernels.dissipationArithmetic == "fp64",
                "full flux arithmetic already includes dissipation; select only one reduced flux policy");
        std::string reducedArithmetic;
        for (const auto *precision : {&kernels.polynomialArithmetic, &kernels.dissipationArithmetic, &kernels.fluxArithmetic})
            if (*precision != "fp64")
            {
                require(reducedArithmetic.empty() || reducedArithmetic == *precision,
                        "one RHS launch requires one common reduced arithmetic type");
                reducedArithmetic = *precision;
            }
        require(s.numerics.limiter != "LVR" || reconstructionKernel == CFV::ReconstructionDeviceKernel::FP64,
                "LVR currently requires the fp64 reconstruction kernel");
        require(s.execution.device == "cpu" || cuda, "residual/BC/reconstruction backend " + s.execution.device);
#ifndef DNDS_USE_CUDA
        require(!cuda, "backend cuda requires a CUDA-enabled build");
#endif
        require(s.execution.layout == "aos" || s.execution.layout == "soa" || s.execution.layout == "aosoa32", "flow layout " + s.execution.layout);
        require(s.execution.communication == "mpi-direct", "transport " + s.execution.communication);
        require(s.execution.communicationExecution == "caller" || s.execution.communicationExecution == "worker", "communication execution " + s.execution.communicationExecution);
        if (s.execution.communicationExecution == "worker")
        {
            int provided = MPI_THREAD_SINGLE;
            MPI_Query_thread(&provided);
            require(provided == MPI_THREAD_MULTIPLE, "communication worker requires MPI_THREAD_MULTIPLE");
        }
        require(s.execution.transferPreflight == "collective" || s.execution.transferPreflight == "abortOnFailure", "transfer preflight " + s.execution.transferPreflight);
        require(s.execution.fieldGrouping == "equationBlocks" || s.execution.fieldGrouping == "coupled", "field grouping " + s.execution.fieldGrouping);
        require(s.equation.system == "compressible", "equation " + s.equation.system);
        const bool mixture = s.equation.eos == "idealGasMixture";
        const bool rans = s.equation.rans.model != "none", extended = mixture || rans;
        require(!kernels.rhsDoubleBuffer || !extended, "double buffering currently qualifies flow5 only");
        require(kernels.rhsMapping == "thread" || !extended, "warp quadrature currently qualifies flow5 only");
        require(reducedArithmetic.empty() || !extended, "experimental reduced RHS arithmetic requires perfect-gas flow5");
        require(kernels.fluxArithmetic == "fp64" || s.numerics.viscousGradient == "polynomial",
                "reduced full flux currently requires conservative polynomial gradients");
        if (kernels.dissipationArithmetic != "fp64" && Riemann::RiemannFlux::Available(s.numerics.flux))
        {
            const auto solver = Riemann::RiemannFlux::Resolve(s.numerics.flux).Kind();
            require(solver >= Riemann::Solver::Roe && solver <= Riemann::Solver::Roe_M9,
                    "reduced dissipation currently requires the Roe family");
        }
        const auto usesAMGX = [](const ImplicitSettings &implicit)
        { return implicit.linear.method == "amgx" || (implicit.linear.method == "gmres" && implicit.linear.preconditioner == "amgx"); };
        const auto requireAMGX = [&](const ImplicitSettings &implicit, const std::string &level)
        {
            require(!usesAMGX(implicit) || (cuda && implicit.jacobianAction == "assembled" && Algebra::AmgXPreconditioner::Available()),
                    level + " AmgX requires assembled CUDA storage and the single-device provider in this build");
        };
        requireAMGX(s.implicit, "finest");
        for (index level = 1; level <= s.implicit.multigrid.levels; ++level)
            requireAMGX(s.implicit.ResolveCoarseLevel(level), "pMG level " + std::to_string(level));
        if (cuda)
        {
            // Resolve the selected numerical profile without probing hardware
            // or silently downloading fields. Actual device ordinals and view
            // residency are checked by context creation and host launchers.
            require(s.execution.layout == "aos" && s.execution.fieldGrouping == "coupled",
                    "CUDA requires combined AoS equation fields");
            require(s.equation.eos == "perfectGas", "backend cuda requires perfectGas EOS");
            require(!rans || s.equation.rans.model == "SA" || s.equation.rans.model == "KOmegaWilcox" ||
                        s.equation.rans.model == "KOmegaSST",
                    "CUDA RANS profile " + s.equation.rans.model);
            require(s.numerics.reconstruction != "variational" || !s.numerics.cfv.SORInstead,
                    "CUDA variational reconstruction requires Jacobi (SORInstead=false)");
            const auto requireLinear = [&](const ImplicitSettings &implicit, const std::string &level)
            {
                const auto &linear = implicit.linear;
                const bool amgx = usesAMGX(implicit);
                require(linear.method == "jacobi" || linear.method == "amgx" ||
                            (linear.method == "gmres" && (linear.preconditioner == "jacobi" || amgx)),
                        level + " CUDA linear solve requires Jacobi, AmgX defect correction or GMRES with Jacobi/AmgX");
            };
            requireLinear(s.implicit, "finest");
            for (index level = 1; level <= s.implicit.multigrid.levels; ++level)
                requireLinear(s.implicit.ResolveCoarseLevel(level), "pMG level " + std::to_string(level));
        }
        require(rans || s.initialRANS.empty(), "initialRANS without a RANS model");
        if (rans)
        {
            require((s.equation.rans.model == "SA" || s.equation.rans.model == "KOmegaWilcox" ||
                     s.equation.rans.model == "KOmegaSST" || s.equation.rans.model == "RealizableKE"),
                    "RANS spatial profile " + s.equation.rans.model);
            require(s.equation.transport == "NewtonianFourier", "RANS requires viscous transport");
        }
        require(s.equation.eos == "perfectGas" || mixture, "EOS " + s.equation.eos);
#ifndef DNDS_USE_CANTERA
        require(!mixture, "idealGasMixture requires Cantera in this build");
#endif
        require(s.equation.transport == "inviscid" || s.equation.transport == "NewtonianFourier", "constitutive law " + s.equation.transport);
        require(s.equation.source == "none" || (mixture && s.equation.source == "chemical"), "volume source " + s.equation.source);
        if (mixture)
        {
            const auto &material = s.equation.mixture;
            require(!material.mechanismFile.empty(), "mixture mechanismFile");
            require(std::isfinite(material.rho0) && material.rho0 > 0 && std::isfinite(material.U0) && material.U0 > 0 &&
                        std::isfinite(material.L0) && material.L0 > 0 && std::isfinite(material.T0) && material.T0 > 0,
                    "mixture reference scales");
            require(s.equation.viscosity == 0 && s.equation.viscosityModel == "constant" &&
                        s.equation.viscosityReferenceTemperature == 1 && s.equation.sutherlandConstant == 0,
                    "mixture transport uses its material, not a perfect-gas viscosity law");
        }
        if (extended)
        {
            require(s.initialProfile.kind == "constant", "nonconstant extended equation initialization");
            const auto flux = Riemann::RiemannFlux::Resolve(s.numerics.flux);
            require((!mixture || flux.Kind() == Riemann::Solver::Rusanov) && !s.numerics.faceFallback,
                    "extended equation thermodynamic flux preparation/face fallback");
            require(!s.numerics.riemann.useStencil || (!flux.UsesTransverseSignal() && !flux.UsesPressureStencil()),
                    "extended equation wave/pressure stencil auxiliary inputs");
            require(s.numerics.limiter == "none" || s.numerics.limiter == "Barth" || s.numerics.limiter == "WBAP",
                    "extended equation limiter");
            require(s.equation.transport != "NewtonianFourier" || s.numerics.viscousGradient == "polynomial", "extended equation viscous gradient");
            require(!s.implicit.roeDissipation, "extended equation Roe Jacobian");
        }
        require(s.initialField.Empty() || s.initialProfile.kind == "constant", "spatial overrides require a constant background profile");
        require(Riemann::RiemannFlux::Available(s.numerics.flux), "numerical flux " + s.numerics.flux);
        require(s.numerics.reconstruction == "constant" || s.numerics.reconstruction == "gradient" || s.numerics.reconstruction == "variational", "reconstruction " + s.numerics.reconstruction);
        require(s.implicit.sourceJacobianMode == "center" || s.implicit.sourceJacobianMode == "rhsQuadrature", "source Jacobian mode " + s.implicit.sourceJacobianMode);
        require(s.implicit.sourceJacobianMode != "rhsQuadrature" || s.numerics.reconstruction == "variational", "rhsQuadrature source Jacobian requires variational reconstruction");
        require(s.implicit.sourceJacobianMode != "rhsQuadrature" || s.equation.source != "none" || rans, "rhsQuadrature source Jacobian requires an active volume source");
        require(s.numerics.gradient == "leastSquares" || s.numerics.gradient == "greenGauss" || s.numerics.gradient == "ggmp", "gradient " + s.numerics.gradient);
        require(s.numerics.viscousGradient == "primitive" || s.numerics.viscousGradient == "polynomial", "viscous gradient " + s.numerics.viscousGradient);
        require(s.numerics.limiter == "none" || s.numerics.limiter == "Barth" ||
                    s.numerics.limiter == "WBAP" || s.numerics.limiter == "LVR",
                "limiter " + s.numerics.limiter);
        require(s.numerics.limiter != "Barth" || s.numerics.reconstruction == "gradient",
                "Barth needs direct gradient reconstruction");
        require(s.numerics.limiter != "WBAP" || s.numerics.reconstruction != "constant",
                "WBAP needs gradient or variational reconstruction");
        require(s.numerics.limiter != "LVR" || (s.numerics.reconstruction == "variational" && s.numerics.cfv.maxOrder >= 2), "LVR needs degree2/3 variational reconstruction");
        require(!s.implicit.multigrid.levels || (s.numerics.reconstruction == "variational" && s.numerics.cfv.maxOrder >= 2), "polynomial FAS requires high-order variational reconstruction");
        for (const auto &[zone, condition] : s.boundaries)
        {
            require(condition.kind == "extrapolate" || condition.kind == "farfield" || condition.kind == "totalConditionInlet" || condition.kind == "slipWall" || condition.kind == "noSlipWall" || condition.kind == "analytic" || condition.kind == "doubleMachBottom" || condition.kind == "interface", "BC zone " + zone + ": " + condition.kind);
            require(condition.kind != "doubleMachBottom" || condition.profile.kind == "doubleMachReflection", "DMR bottom requires its analytic shock profile");
            require(condition.thermal == "adiabatic" || condition.thermal == "isothermal", "thermal BC " + condition.thermal);
            require(condition.wallVelocityFrame == "mesh" || condition.wallVelocityFrame == "ground", "wall velocity frame " + condition.wallVelocityFrame);
            require(!extended || (condition.kind != "doubleMachBottom" &&
                                  (condition.kind != "analytic" ||
                                   condition.profile.kind == "constant")),
                    "extended equation boundary profile " + condition.kind);
            require(condition.kind != "noSlipWall" || s.equation.transport == "NewtonianFourier", "no-slip wall needs viscous transport");
        }
        if (!missing.empty())
        {
            std::string message = "Cauchy domain " + domain.placement.name + " has missing capabilities:";
            for (const auto &item : missing)
                message += "\n  " + item;
            DNDS_check_throw_info(false, message);
        }
        const std::string backend = cuda ? "cuda/" : "cpu/";
        return {{backend + (mixture ? "idealGasMixture/flow5+species" : "perfectGas/flow5") + (rans ? "+" + s.equation.rans.model : ""), backend + s.numerics.flux, backend + s.equation.transport,
                 "cfv/" + s.numerics.reconstruction, "cfv/" + s.numerics.limiter, "runtime/mpi-direct"}};
    }
}
