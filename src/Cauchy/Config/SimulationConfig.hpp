#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Math/LimitedVR.hpp"
#include "Cauchy/Core/DomainPlacement.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"
#include "Cauchy/Motion/RigidMotion.hpp"
#include "Physics/RANS/Model.hpp"
#include "Physics/StateInput.hpp"
#include "Physics/Chemistry/ReactorStepSettings.hpp"
#include <map>
#include <optional>

namespace DNDS::Cauchy
{
    using Json = nlohmann::ordered_json;
    /// Species names are model-defined keys, not configuration member names.
    inline Json SpeciesMassFractionsSchema()
    {
        return {{"type", "object"}, {"additionalProperties", {{"type", "number"}}}};
    }
    struct MixtureSettings
    {
        std::string mechanismFile, phaseName;
        real rho0 = 1, U0 = 1, L0 = 1, T0 = 1;
        DNDS_DECLARE_CONFIG(MixtureSettings)
        {
            DNDS_FIELD(mechanismFile, "Cantera mechanism path or installed mechanism name");
            DNDS_FIELD(phaseName, "Cantera phase; empty selects the default phase");
            DNDS_FIELD(rho0, "SI reference density");
            DNDS_FIELD(U0, "SI reference velocity");
            DNDS_FIELD(L0, "SI reference length");
            DNDS_FIELD(T0, "SI reference temperature");
        }
    };
    struct EquationSettings
    {
        std::string system = "compressible", eos = "perfectGas", transport = "inviscid", source = "none";
        real gamma = 1.4, gasConstant = 1., viscosity = 0., prandtl = 0.72;
        real acmAlpha = 0, acmBeta2 = 1, acmEntropyFixRatio = .05;
        std::optional<real> acmPressureMean;
        std::string viscosityModel = "constant";
        real viscosityReferenceTemperature = 1., sutherlandConstant = 0.;
        MixtureSettings mixture;
        /// Null preserves existing mixture scale fields (including their
        /// configured unit-system defaults). An object requires all four scales.
        Json referenceScales = nullptr;
        Physics::RANS::ModelSettings rans;
        DNDS_DECLARE_CONFIG(EquationSettings)
        {
            DNDS_FIELD(system, "Coupled equation family");
            DNDS_FIELD(acmAlpha, "Scheme-A artificial-compressibility parameter");
            DNDS_FIELD(acmBeta2, "Positive artificial wave speed squared in code velocity units");
            DNDS_FIELD(acmEntropyFixRatio, "Relative Harten width for mechanical Roe flux; zero disables smoothing");
            config.field_optional(&T::acmPressureMean, "acmPressureMean", "Fixed free-gauge pressure mean in code units; null uses the initialized mean");
            DNDS_FIELD(eos, "Thermodynamic closure");
            DNDS_FIELD(transport, "inviscid or NewtonianFourier");
            DNDS_FIELD(source, "none or chemical reactive volume source; attached RANS sources are independent");
            DNDS_FIELD(gamma, "Perfect-gas heat capacity ratio");
            DNDS_FIELD(gasConstant, "Specific gas constant in the chosen units");
            DNDS_FIELD(viscosity, "Code-unit molecular viscosity coefficient; reference value for the selected perfect-gas law");
            DNDS_FIELD(viscosityModel, "constant, sutherland, or constantKinematic; mixtures use material transport instead");
            DNDS_FIELD(viscosityReferenceTemperature, "Sutherland reference temperature in code units, TRef_phys/T0");
            DNDS_FIELD(sutherlandConstant, "Sutherland constant in code units, C_phys/T0");
            DNDS_FIELD(prandtl, "Molecular Prandtl number");
            config.field_section(&T::mixture, "mixture", "Ideal-gas mixture mechanism and reference scales");
            config.field_json_schema(&T::referenceScales, "referenceScales", "Unit scales, separate from physical referenceState", []
                                     {
                const Json positive{{"type", "number"}};
                return Json{{"oneOf", Json::array({Json{{"type", "null"}}, Json{{"type", "object"}, {"required", {"rho0", "U0", "L0", "T0"}},
                    {"additionalProperties", false}, {"properties", {{"rho0", positive}, {"U0", positive}, {"L0", positive}, {"T0", positive}}}}})}}; });
            config.field_section(&T::rans, "rans", "Shared turbulence closure; none disables the transported RANS block");
        }
    };
    struct KernelExecutionSettings
    {
        std::string stateInputLayout = "aos", reconstructionInputLayout = "aos";
        std::string polynomialArithmetic = "fp64", dissipationArithmetic = "fp64", fluxArithmetic = "fp64";
        std::string rhsMapping = "auto", rhsReduction = "auto";
        std::optional<int> rhsGridBlocks, rhsBlockSize;
        bool rhsDoubleBuffer = false;
        std::optional<bool> rhsHoistFaceRows;
        int polynomialPPThreads = 0;
        DNDS_DECLARE_CONFIG(KernelExecutionSettings)
        {
            DNDS_FIELD(stateInputLayout, "Experimental CUDA mean/linear input shadows: aos or aosoa32; primary storage is unchanged");
            DNDS_FIELD(reconstructionInputLayout, "Experimental CUDA RHS coefficient input shadow: aos or aosoa32");
            DNDS_FIELD(polynomialArithmetic, "Point-local polynomial compute: fp64, fp32 or f32f32; coefficients remain FP64");
            DNDS_FIELD(dissipationArithmetic, "Gas-agnostic Riemann dissipation arithmetic: fp64, fp32 or f32f32");
            DNDS_FIELD(fluxArithmetic, "Point physics and full flux arithmetic: fp64, fp32 or f32f32; residual sums remain FP64");
            DNDS_FIELD(rhsMapping, "Modal RHS mapping: auto uses a qualified device preset, otherwise the arithmetic default; ctaSegments or thread overrides it");
            DNDS_FIELD(rhsReduction, "auto selects segmented CTA reduction; explicit serial, segmented, boundedTree, or warpTree with a cross-warp fallback");
            config.field_optional(&T::rhsGridBlocks, "rhsGridBlocks", "CUDA RHS maximum grid blocks; null selects automatically, 0 explicitly uses the mapping default cap");
            config.field_optional(&T::rhsBlockSize, "rhsBlockSize", "CUDA RHS threads per block: null selects automatically, or use 64, 128, 256 or 512");
            DNDS_FIELD(rhsDoubleBuffer, "Experimental two-bank FP64 CTA staging");
            config.field_optional(&T::rhsHoistFaceRows, "rhsHoistFaceRows", "FP64 modal face-thread row binding; null selects automatically");
            DNDS_FIELD(polynomialPPThreads, "CUDA polynomial PP block size: 0 selects the device/shape default (512 for measured A100 P3 flow5, otherwise 128); explicit 128/256/512 or the existing environment override is preserved");
        }
        bool Default() const
        {
            return stateInputLayout == "aos" && reconstructionInputLayout == "aos" &&
                   polynomialArithmetic == "fp64" && dissipationArithmetic == "fp64" && fluxArithmetic == "fp64" && (rhsMapping == "auto" || rhsMapping == "thread") &&
                   rhsBlockSize.value_or(128) == 128 && !rhsDoubleBuffer && !rhsHoistFaceRows.value_or(false) &&
                   (rhsReduction == "auto" || rhsReduction == "serial") && rhsGridBlocks.value_or(0) == 0 && polynomialPPThreads == 0;
        }
    };
    struct ExecutionSettings
    {
        std::string device = "cpu", layout = "aos", communication = "mpi-direct";
        std::string fieldGrouping = "coupled";
        std::string transferPreflight = "collective";
        std::string communicationExecution = "caller";
        /// Automatic or explicit CFV CUDA sweep policy. Quantized variants keep solution
        /// coefficients in FP64; only already-inverted sweep maps/products
        /// change precision. CPU and existing solver defaults remain FP64.
        std::string reconstructionKernel = "auto";
        // Stage-once scan launch settings belong to numerics.cfv.cudaSweep;
        // Cauchy selects the operator policy and never launches a CFV kernel.
        KernelExecutionSettings kernels;
        CFV::ReconstructionDeviceKernel ReconstructionKernel() const;
        int hostThreads = 0;
        int deviceOrdinal = 0;
        bool checkAccess = true;
        bool overlap = true;
        bool haloFreshnessChecks = false;
        /// Check the actual MPI process capability before constructing workers.
        /// FUNNELED permits CPU workers; MPI calls still belong to the controller.
        /// SINGLE forbids worker threads even if workers themselves never call MPI.
        void ValidateMPIThreadSupport(int provided) const;
        DNDS_DECLARE_CONFIG(ExecutionSettings)
        {
            DNDS_FIELD(device, "Required numerical backend; no implicit fallback");
            DNDS_FIELD(deviceOrdinal, "CUDA device ordinal for this process; explicit rank/device mapping, ignored on CPU");
            DNDS_FIELD(layout, "Physical flow table: aos, soa, aosoa32");
            DNDS_FIELD(fieldGrouping, "equationBlocks: separate flow/RANS/species fields; coupled: one physical table with the same logical blocks");
            DNDS_FIELD(communication, "Payload transport");
            DNDS_FIELD(transferPreflight, "MPI launch validation: collective (default) or abortOnFailure (local checks, fatal MPI abort, no per-launch global agreement); setup remains collective");
            DNDS_FIELD(hostThreads, "CPU worker count; zero means inline execution");
            DNDS_FIELD(communicationExecution, "MPI payload execution: caller (default) or worker (CPU Direct/HostPacked, requires MPI_THREAD_MULTIPLE)");
            DNDS_FIELD(reconstructionKernel, "auto selects fp64Pipelined for measured A100 P3 flow5 relaxation, otherwise fp64. AMG reconstruction uses its solver setting instead. Explicit comparison kernels retain their C5/M19 precomputed non-LVR fixture; fp64/fp64Pipelined support the production shapes and LVR. Launch dimensions: numerics.cfv.cudaSweep. Precision is never reduced automatically; nondefault kernels require CUDA.");
            config.field_section(&T::kernels, "kernels", "Operator-local CUDA input layout and compute arithmetic policies");
            DNDS_FIELD(checkAccess, "Validate declared scheduled accesses at submission");
            DNDS_FIELD(overlap, "Overlap input halo exchanges with owned-only core work; false waits then runs unsplit kernels");
            DNDS_FIELD(haloFreshnessChecks, "Extra collective halo-publication audits; default off, also disabled by implicit.collectiveChecks=false");
        }
    };
    struct MeshSettings
    {
        std::string file;
        int dimension = 2;
        int elevation = 0, bisect = 0;
        real scale = 1, rotationZDegrees = 0;
        std::array<real, 3> periodicLengths{0, 0, 0};
        std::map<std::string, int> boundaryIds;
        DNDS_DECLARE_CONFIG(MeshSettings)
        {
            DNDS_FIELD(file, "CGNS input path; an injected mesh needs no file");
            DNDS_FIELD(dimension, "Geometric dimension, independent of flow component count");
            DNDS_FIELD(elevation, "Native geometry order elevation: 0 keeps input order, 1 elevates to O2", DNDS::Config::range(0, 1));
            DNDS_FIELD(bisect, "Native uniform bisection passes after partition and optional elevation", DNDS::Config::range(0, 4));
            DNDS_FIELD(scale, "Positive uniform input-mesh coordinate scale, applied once after native preparation and before FV geometry");
            DNDS_FIELD(rotationZDegrees, "Input-mesh rotation around Z in degrees, applied before scale; injected meshes must already be transformed");
            DNDS_FIELD(periodicLengths, "Input-file-frame translation from PERIODIC_n to its donor; zero disables that direction; mesh scale/rotation transform the frames too");
            config.field_schema(&T::boundaryIds, "boundaryIds", "Explicit CGNS boundary name to integer zone",
                                []() -> Json
                                { return {{"type", "object"}, {"additionalProperties", {{"type", "integer"}}}}; });
        }
    };
    struct RiemannSettings
    {
        real fixScale = 1, incrementScale = 1, dLambda = 0, referenceMach = .1, pressureStencilMin = 1e300;
        bool useStencil = true;
        DNDS_DECLARE_CONFIG(RiemannSettings)
        {
            DNDS_FIELD(fixScale, "Entropy and H-correction scale");
            DNDS_FIELD(incrementScale, "Roe contact dissipation scale");
            DNDS_FIELD(dLambda, "Transverse H-correction signal floor; exact scalar input when useStencil is false");
            DNDS_FIELD(referenceMach, "Positive AUSM+-up reference Mach cutoff");
            DNDS_FIELD(pressureStencilMin, "AUSMPW+ stencil pressure upper bound; exact scalar input when useStencil is false");
            DNDS_FIELD(useStencil, "Supply cell-face-cell wave/pressure stencils to fluxes that use them; false keeps the explicit scalar inputs");
        }
        Riemann::FluxSettings KernelSettings() const { return {dLambda, fixScale, incrementScale, referenceMach, pressureStencilMin}; }
    };
    struct RANSStateSettings
    {
        real saUpperLimit = 1e5, omegaLowerFactor = .01;
        std::map<std::string, real> referenceConservative;
        DNDS_DECLARE_CONFIG(RANSStateSettings)
        {
            DNDS_FIELD(saUpperLimit, "Euler SA final mean upper bound multiplied by density");
            DNDS_FIELD(omegaLowerFactor, "Euler omega floor factor multiplying the prescribed reference rhoOmega");
            config.field_schema(&T::referenceConservative, "referenceConservative", "Named conservative RANS reference; rho is required for reconstruction PP primitive-reference conversion, without fallback. RANS values are required for a nonzero omega floor", SpeciesMassFractionsSchema);
        }
    };
    /// Spatial reconstruction solver; distinct from the relaxation CUDA kernel.
    /// AMG is an explicit single-rank/device alternative with fixed V-cycles.
    struct ReconstructionSolverSettings
    {
        std::string method = "relaxation", amgxStorage = "native";
        int amgxCycles = 1;
        bool UsesAMG() const { return method == "amgx"; }
        void Validate() const
        {
            DNDS_check_throw_info(method == "relaxation" || UsesAMG(), "unknown reconstruction solver");
            DNDS_check_throw_info((amgxStorage == "native" || amgxStorage == "upload") && amgxCycles > 0,
                                  "invalid reconstruction AMG storage/cycle controls");
        }
        DNDS_DECLARE_CONFIG(ReconstructionSolverSettings)
        {
            DNDS_FIELD(method, "relaxation uses the existing Jacobi/SOR operator; amgx uses fixed projected-system defect corrections on one CUDA rank");
            DNDS_FIELD(amgxStorage, "native shares the prepared finest numeric matrix; upload retains the provider-copy comparison route");
            DNDS_FIELD(amgxCycles, "Fixed V-cycles per physical component per reconstruction correction; no inner residual stopping");
        }
    };
    struct NumericsSettings
    {
        std::string reconstruction = "gradient", gradient = "greenGauss", flux = "rusanov", limiter = "none";
        CFV::LeastSquaresWeights leastSquares;
        int reconstructionIterations = 1;
        int reconstructionResidualCheckInterval = 4;
        real reconstructionTolerance = 0;
        ReconstructionSolverSettings reconstructionSolver;
        bool characteristicLimiter = true, faceFallback = true;
        bool characteristicDiagnostics = false;
        bool reconstructionBoundaryFromPolynomial = true;
        /// VR boundary equation target only. Direct gradients and flux exterior
        /// states retain their own contracts; interface samples are unchanged.
        std::string vrBoundaryValue = "physicalTrace";
        bool firstOrderConvection = false;
        bool ransMeanFaceValues = false;
        bool ransP1Overlay = false;
        std::string viscousGradient = "primitive";
        bool viscousLimited = true, noGRPOnWall = false, noRsOnWall = false;
        bool positivity = false;
        real temperatureFloor = 1e-12;
        real positivityBetaCutoff = 1e-3;
        std::string positivityReconstruction = "Euler";
        std::string positivityClippedReferencePolicy = "fallbackToBase";
        bool positivityCompressToMean = true;
        int positivityBetaPower = 1;
        real densityFloor = 1e-12, pressureFloor = 1e-12;
        real sensibleEnergyFloor = 1e-12;
        real wallDistanceFloor = 1e-10;
        RANSStateSettings ransState;
        CFV::ReconstructionSettings cfv;
        CFV::LimitedVRSettings limitedVR;
        RiemannSettings riemann;
        /// Direct linear/constant FV uses center-point flux/source evaluation.
        /// cfv.intOrder still controls geometric measures and VR setup.
        int RHSIntegrationOrder() const { return reconstruction == "variational" ? cfv.intOrder : 1; }
        void ValidatePositivity() const;
        DNDS_DECLARE_CONFIG(NumericsSettings)
        {
            DNDS_FIELD(reconstruction, "constant, gradient, or variational reconstruction");
            DNDS_FIELD(gradient, "leastSquares, greenGauss, or ggmp");
            config.field_section(&T::leastSquares, "leastSquares", "Weights for the P1 weighted least-squares objective");
            DNDS_FIELD(flux, "Numerical inviscid flux family");
            DNDS_FIELD(limiter, "none, direct-gradient Barth, WBAP, or native LVR; positivity is checked separately");
            config.field_section(&T::reconstructionSolver, "reconstructionSolver", "Relaxation or explicit single-device AMG reconstruction solve");
            DNDS_FIELD(reconstructionIterations, "Maximum local VR relaxation sweeps or AMG defect corrections per residual");
            DNDS_FIELD(reconstructionResidualCheckInterval, "Observe the VR update norm every N sweeps and at the final sweep when reconstructionTolerance is positive; each observation is an explicit host stopping boundary");
            DNDS_FIELD(reconstructionTolerance, "Relative reconstruction update tolerance for relaxation or AMG; zero uses the fixed correction count");
            DNDS_FIELD(characteristicLimiter, "Use equation-provided cell-mean face characteristics for 3WBAP");
            DNDS_FIELD(characteristicDiagnostics, "Opt-in finite/inverse-matrix diagnostics after characteristic production; not an execution-order check");
            DNDS_FIELD(faceFallback, "Fall back to the cell mean at inadmissible face values; disable for non-PP accuracy studies");
            DNDS_FIELD(reconstructionBoundaryFromPolynomial, "Project current face polynomial through BC each VR sweep; false uses mean-based boundary values");
            DNDS_FIELD(vrBoundaryValue, "VR-only boundary target: physicalTrace or physicalMirrored (Euler reflected-state target); interface values remain prescribed samples");
            DNDS_FIELD(firstOrderConvection, "P0 face operator: cell means and jump-only normal viscosity; separate reconstruction remains available for RANS/volume sources. Used by bottom pMG");
            DNDS_FIELD(ransMeanFaceValues, "Use cell-mean RANS scalars in face fluxes while retaining face gradients, flow/species reconstruction, and physical or interface exterior states");
            DNDS_FIELD(ransP1Overlay, "Use the configured mean-gradient reconstruction for RANS volume points and for RANS face points when ransMeanFaceValues is off. The full VR sweep remains active");
            DNDS_FIELD(viscousGradient, "primitive retains the cell-gradient scheme; polynomial uses cached polynomial derivatives and Euler's normal face-jump penalty");
            DNDS_FIELD(viscousLimited, "Use the filtered flux polynomial for polynomial viscosity; false uses the unlimited VR guess");
            DNDS_FIELD(noGRPOnWall, "Disable the polynomial normal jump penalty on no-slip walls, retaining the polynomial gradient");
            DNDS_FIELD(noRsOnWall, "Euler wall option: use the same physical wall state on both convective sides, retaining the previously formed viscous gradient");
            DNDS_FIELD(positivity, "Mean-preserving reconstruction and implicit correction PP; flow5 SSPRK also uses conservative flux blending");
            DNDS_FIELD(temperatureFloor, "Minimum temperature enforced through the equation's internal-energy bound");
            DNDS_FIELD(positivityBetaCutoff, "Set numerically tiny reconstruction beta to zero; matches legacy's mean fallback");
            DNDS_FIELD(positivityReconstruction, "Euler common quadrature beta or Segment for reproducing earlier Cauchy archives");
            DNDS_FIELD(positivityClippedReferencePolicy, "Clipped-composition reconstruction energy policy: fallbackToBase (Euler compatibility) or backtracking");
            DNDS_FIELD(positivityCompressToMean, "Euler reference is the mean; false uses legacy progressive order reduction");
            DNDS_FIELD(positivityBetaPower, "Euler extra beta exponent, so the total exponent is 1+power");
            DNDS_FIELD(densityFloor, "Positive inclusive density admissibility floor");
            DNDS_FIELD(pressureFloor, "Positive inclusive pressure admissibility floor for the initial perfect-gas equation");
            DNDS_FIELD(sensibleEnergyFloor, "Mixture internal-energy density margin above its backend reference temperature; no gamma conversion");
            DNDS_FIELD(wallDistanceFloor, "Positive floor for nodal wall-distance quadrature samples used by RANS");
            config.field_section(&T::ransState, "ransState", "Model-specific scalar repairs after the common flow correction");
            config.field_section(&T::cfv, "cfv", "Independent mathematical CFV configuration");
            config.field_section(&T::limitedVR, "limitedVR", "Native v2/v3 target-penalty controls");
            config.field_section(&T::riemann, "riemann", "Numerical flux correction controls, independent of EOS");
        }
    };
    struct FlowProfileSettings
    {
        std::string kind = "constant";
        real strength = 5, period = 10;
        std::array<real, 3> center{5, 5, 0};
        real perturbation = 0;
        DNDS_DECLARE_CONFIG(FlowProfileSettings)
        {
            DNDS_FIELD(kind, "constant, isentropicVortex, doubleMachReflection, or cylinderWake");
            DNDS_FIELD(strength, "Isentropic vortex strength; five is the standard non-PP vortex");
            DNDS_FIELD(period, "Periodic vortex box length");
            DNDS_FIELD(center, "Initial vortex center");
            DNDS_FIELD(perturbation, "Localized transverse initial wake perturbation amplitude");
        }
    };
    struct BoundarySettings
    {
        std::string kind = "extrapolate", thermal = "adiabatic";
        std::string wallVelocityFrame = "mesh";
        std::array<real, 5> primitive{1, 0, 0, 0, 1};
        Json state = nullptr;
        /// Setup-only resolved [p0,T0,unit direction,primitive RANS,Y...].
        /// Original tagged state is serialized; this prepared vector is not a
        /// second user configuration and is populated by physical input setup.
        std::vector<real> resolvedTotalState;
        /// Setup-resolved velocity, relative pressure and turbulence scalars.
        std::vector<real> resolvedACMState;
        std::array<real, 3> velocity{0, 0, 0};
        std::map<std::string, real> massFractions;
        std::map<std::string, real> ransConservative;
        real temperature = 1;
        FlowProfileSettings profile;
        DNDS_DECLARE_CONFIG(BoundarySettings)
        {
            DNDS_FIELD(kind, "extrapolate, farfield, slipWall, noSlipWall; ACM velocityInlet or pressureOutlet; compressible totalConditionInlet, analytic, doubleMachBottom or interface");
            DNDS_FIELD(thermal, "Wall thermal law: adiabatic or isothermal");
            DNDS_FIELD(wallVelocityFrame, "mesh: velocity relative to moving wall; ground: prescribed absolute ground velocity");
            DNDS_FIELD(primitive, "Farfield rho,u,v,w,p; three velocities on any geometry");
            config.field_json_schema(&T::state, "state", "Complete tagged prescribed state; null uses primitive and scalar fields", Physics::StateInput::Schema);
            config.field_schema(&T::massFractions, "massFractions", "Prescribed species mass fractions; omitted last species fills the remainder", SpeciesMassFractionsSchema);
            config.field_schema(&T::ransConservative, "ransConservative", "Named conservative RANS values in code units for prescribed boundaries", SpeciesMassFractionsSchema);
            DNDS_FIELD(velocity, "Prescribed wall velocity in wallVelocityFrame; farfield/analytic data always use ground axes");
            DNDS_FIELD(temperature, "Positive isothermal wall temperature");
            config.field_section(&T::profile, "profile", "Space/time input for analytic boundary values");
        }
    };
    /// Matrix physical layout is independent of the equation-state layout.
    struct ImplicitMatrixTableSettings
    {
        std::string layout = "aos";
        index rowPaddingMultiple = 1;
        index rowStridePaddingMultiple = 1;
        size_t allocationAlignmentBytes = 0;
        DNDS_DECLARE_CONFIG(ImplicitMatrixTableSettings)
        {
            DNDS_FIELD(layout, "aos, soa, or aosoa32; one logical row is one dense block; within-block order belongs to the matrix descriptor");
            DNDS_FIELD(rowPaddingMultiple, "Positive SoA row-allocation padding multiple; ignored for aos and fixed-tile aosoa32");
            DNDS_FIELD(rowStridePaddingMultiple, "Positive AoS row-stride multiple in real elements; rounds physical block pitch without changing logical block width");
            DNDS_FIELD(allocationAlignmentBytes, "Requested allocation alignment in bytes, zero uses the backend default; nonzero must be a power of two");
        }
    };
    struct ImplicitMatrixStorageSettings
    {
        ImplicitMatrixTableSettings diagonal, inverse, offDiagonal;
        std::string indices = "auto", amgxStorage = "upload";
        DNDS_DECLARE_CONFIG(ImplicitMatrixStorageSettings)
        {
            config.field_section(&T::diagonal, "diagonal", "Assembled diagonal blocks");
            config.field_section(&T::inverse, "inverse", "Prepared local diagonal inverses");
            config.field_section(&T::offDiagonal, "offDiagonal", "Assembled off-diagonal blocks");
            DNDS_FIELD(amgxStorage, "upload: public provider copies; native: one compatible shared device matrix allocation without numeric upload copies");
            DNDS_FIELD(indices, "Primary sparse index storage: auto selects int32 for amgx and int64 otherwise; int32/int64 are explicit alternatives. AmgX requires native int32 and compatible aos blocks without adapter repacking.");
        }
    };
    struct ImplicitLinearSettings
    {
        std::string method = "gmres", preconditioner = "sgs";
        std::string stopping = "residual";
        std::string componentScaling = "none";
        index iterations = 20;
        index residualCheckInterval = 1;
        int subspace = 10;
        int jacobiLocalSweeps = 1;
        int amgxAggregateSize = 0;
        int diagonalInverseRefinement = 0;
        std::string cudaDiagonalInverse = "auto";
        std::string cudaJacobiMapping = "thread";
        ImplicitMatrixStorageSettings matrixStorage;
        real relativeTolerance = .1, absoluteTolerance = 1e-12;
        DNDS_DECLARE_CONFIG(ImplicitLinearSettings)
        {
            DNDS_FIELD(method, "sgs or jacobi stationary sweeps, ilu or amgx stationary defect correction, or restarted left-gmres; amgx requires assembled CUDA storage");
            DNDS_FIELD(amgxAggregateSize, "Flow AmgX target aggregate size: 0 selects 4 in 2D or 8 in 3D. Explicit choices are 2, 4 and 8.");
            DNDS_FIELD(jacobiLocalSweeps, "One or two local Jacobi sweeps per iteration; two freezes remote iterate values between sweeps, with one input exchange");
            DNDS_FIELD(diagonalInverseRefinement, "Optional guarded block-inverse corrections: 0 (default) preserves the existing inversion, 1..3 refines within retained SVD subspaces without changing the cutoff. Requires block-diagonal SGS/Jacobi; unavailable for ILU/AmgX.");
            DNDS_FIELD(cudaDiagonalInverse, "auto selects guardedLU on A100 and batchedSVD elsewhere; explicit providers are preserved. Guarded 5/6/7 LU retains the CPU condition guard and filtered SVD fallback. CPU inversion is unchanged");
            DNDS_FIELD(cudaJacobiMapping, "CUDA assembled Jacobi mapping: thread (reference/default) or componentGroup (experimental eight lanes per cell); CPU row arithmetic is unchanged");
            config.field_section(&T::matrixStorage, "matrixStorage", "Independent matrix-table and native sparse-index formats; no hot-path format conversion");
            DNDS_FIELD(preconditioner, "Fixed sgs, zero-started jacobi, rank-local ilu or single-device amgx for GMRES; amgx requires assembled CUDA storage; unused by stationary methods");
            DNDS_FIELD(stopping, "residual uses tolerance checks; fixedCount executes exactly iterations for SGS/ILU, omits true-residual reporting and checks the final update is finite; tolerances are ignored");
            DNDS_FIELD(componentScaling, "GMRES metric: none or meanState (Euler volume-averaged absolute conservative magnitudes with a common momentum scale and kinetic energy scale; requires a compressible flow5 block)");
            DNDS_FIELD(iterations, "Maximum linear iterations per nonlinear correction");
            DNDS_FIELD(residualCheckInterval, "Observe stationary residual stopping every N iterations and at the final iteration; GMRES checks its convergence estimate at this cadence, with mandatory restart/breakdown checks and unchanged Arnoldi products");
            DNDS_FIELD(subspace, "Restarted GMRES subspace size");
            DNDS_FIELD(relativeTolerance, "Relative true-residual tolerance");
            DNDS_FIELD(absoluteTolerance, "Euclidean tolerance on the cell-integrated true residual; zero disables the absolute cutoff, useful for small physical volumes");
        }
    };
    struct ImplicitSGSSettings
    {
        int sweeps = 1;
        bool exchangeBetweenSweeps = false;
        DNDS_DECLARE_CONFIG(ImplicitSGSSettings)
        {
            DNDS_FIELD(sweeps, "Fixed number of zero-started SGS sweeps per GMRES preconditioner application");
            DNDS_FIELD(exchangeBetweenSweeps, "Expose a forward-halo exchange between the lower and upper SGS scans");
        }
    };
    struct ImplicitILUSettings
    {
        int fill = 0;
        int preconditionerCorrections = 1;
        bool coreFirst = true;
        std::string ordering = "natural";
        DNDS_DECLARE_CONFIG(ImplicitILUSettings)
        {
            DNDS_FIELD(fill, "Local ILU stencil expansion 0..5, or -1 for complete LU");
            DNDS_FIELD(preconditionerCorrections, "Fixed zero-started ILU defect corrections per GMRES preconditioner call; corrections after the first exchange the iterate halo. Stationary ILU instead uses linear.iterations");
            DNDS_FIELD(coreFirst, "Order core before dependent cells before symbolic fill");
            DNDS_FIELD(ordering, "Rank-local factor ordering: natural or rcm (CorrectRCM), followed by optional stable core grouping; does not reorder mesh/DOFs");
        }
    };
    struct ImplicitPositivitySettings
    {
        std::string rhsFPPMode = "rhs";
        std::string rhsFPPReduction = "cell";
        real rhsFPPScale = 1, rhsFPPRelax = .5, incrementRelax = 1;
        int compressionPower = 1;
        DNDS_DECLARE_CONFIG(ImplicitPositivitySettings)
        {
            DNDS_FIELD(rhsFPPMode, "RHS forecasted positivity preserving: none, rhs (scale solve defect), or dTau (shrink pseudo-step); active only with numerics.positivity");
            DNDS_FIELD(rhsFPPReduction, "RHS-FPP factor scope: cell keeps local alpha; minimumLimited applies domain minimum only to alpha<1 cells; minimumAll applies it everywhere");
            DNDS_FIELD(rhsFPPScale, "Euler RHS-FPP trial increment multiplier");
            DNDS_FIELD(rhsFPPRelax, "Euler RHS/dTau density and energy relaxation in (0,1]");
            DNDS_FIELD(incrementRelax, "Euler accepted-correction relaxation in (0,1]");
            DNDS_FIELD(compressionPower, "Positive integer exponent multiplier in Euler alpha compression");
        }
    };
    /// Partial solver controls are merged into the domain implicit defaults
    /// once, before constructing a coarse smoother. They cannot alter physics,
    /// the hierarchy or the finest convergence policy.
    inline Json MultigridSmootherSchema()
    {
        return {{"type", "object"}, {"additionalProperties", false}, {"properties", {{"diagonal", {{"type", "string"}}}, {"jacobianAction", {{"type", "string"}}}, {"roeDissipation", {{"type", "boolean"}}}, {"pseudoCFL", {{"type", "number"}}}, {"maxPseudoCFL", {{"type", "number"}}}, {"relaxation", {{"type", "number"}}}, {"linear", ImplicitLinearSettings::schema()}, {"sgs", ImplicitSGSSettings::schema()}, {"ilu", ImplicitILUSettings::schema()}, {"positivity", ImplicitPositivitySettings::schema()}}}};
    }
    struct MultigridLevelSettings
    {
        index preSmooth = -1, postSmooth = -1;
        Json solver = Json::object();
        DNDS_DECLARE_CONFIG(MultigridLevelSettings)
        {
            DNDS_FIELD(preSmooth, "Coarse nonlinear corrections before descent; -1 inherits preSmooth, or coarseSmooth at the bottom");
            DNDS_FIELD(postSmooth, "Corrections after ascent; -1 inherits postSmooth, or zero at the bottom; bottom uses their sum");
            config.field_json_schema(&T::solver, "solver", "Partial independent smoother controls; unspecified fields inherit domain implicit defaults", MultigridSmootherSchema);
        }
    };
    struct PolynomialMultigridSettings
    {
        bool scratchPool = false;
        int levels = 0;
        index cycles = 50, preSmooth = 1, postSmooth = 1, coarseSmooth = 4;
        real damping = 1;
        bool reuseCoarseRate = true;
        /// Match Euler's omission of coarse RHS/dTau PP by default.
        /// Reconstruction/increment PP remain independently active.
        bool coarseRHSFPP = false;
        std::vector<index> updateComponents;
        std::map<std::string, MultigridLevelSettings> coarseLevelSettings;
        DNDS_DECLARE_CONFIG(PolynomialMultigridSettings)
        {
            DNDS_FIELD(levels, "0 disables pMG; 1 adds direct-gradient O2; 2 also adds first-order convection, retaining viscosity");
            DNDS_FIELD(cycles, "Maximum FAS V-cycles per physical stage");
            DNDS_FIELD(preSmooth, "Bounded nonlinear corrections before restriction");
            DNDS_FIELD(postSmooth, "Bounded nonlinear corrections after prolongation");
            DNDS_FIELD(coarseSmooth, "Bounded nonlinear corrections at the coarsest level");
            DNDS_FIELD(scratchPool, "Reserve four ghost-capable DoF scratch slots for synchronous pMG values/residuals/corrections; keep parent sources/references and solver/reconstruction workspaces independent");
            DNDS_FIELD(reuseCoarseRate, "Reuse a version-validated direct-gradient coarse tau rate for its first correction; independent of finest measured-rate reuse");
            DNDS_FIELD(coarseRHSFPP, "Opt in to RHS/dTau positivity limiting on coarse levels; default false forces coarse rhsFPPMode=none after level overrides, matching Euler. Reconstruction and increment PP remain unchanged");
            config.field_map_of<MultigridLevelSettings>(&T::coarseLevelSettings, "coarseLevelSettings", "Independent coarse-level controls keyed by 1 or 2; finest uses domain implicit and the common pre/post counts");
            DNDS_FIELD(damping, "Physical coarse-correction multiplier in (0,1]");
            DNDS_FIELD(updateComponents, "Distinct logical equation indices updated by pMG; empty updates all. Other means remain frozen; finest convergence still uses residualComponents");
        }
    };
    struct ImplicitSettings
    {
        std::string diagonal = "entries", jacobianAction = "assembled";
        std::string sourceJacobianMode = "center";
        std::string boundaryJacobian = "physical";
        std::string updatePolicy = "auto";
        index iterations = 100, minIterations = 1;
        index consoleInterval = 0;
        real relativeTolerance = 1e-8, absoluteTolerance = 1e-11;
        real algebraicRateTolerance = 1e-8;
        int normOrder = 2;
        std::string residualReference = "runningMaximum";
        std::vector<std::string> residualComponents{"auto"};
        std::vector<std::string> ResolveResidualComponents(bool acm) const
        {
            if (residualComponents != std::vector<std::string>{"auto"})
                return residualComponents;
            return acm ? std::vector<std::string>{"u", "v", "w"} : std::vector<std::string>{"rho"};
        }
        real pseudoCFL = 10, maxPseudoCFL = 1e4, pseudoGrowth = 1.3, relaxation = 1;
        int nIterCellSmoothScale = 15;
        bool roeDissipation = false, collectiveChecks = true;
        bool requireConvergedReconstruction = false;
        bool rejectUnconvergedStages = false;
        ImplicitLinearSettings linear;
        ImplicitSGSSettings sgs;
        ImplicitILUSettings ilu;
        ImplicitPositivitySettings positivity;
        PolynomialMultigridSettings multigrid;
        void Validate() const;
        /// Resolve one coarse smoother by merging its partial solver settings
        /// into domain defaults and validating the resulting typed controls.
        /// The returned settings disable recursive multigrid. Level 1 is O2.
        ImplicitSettings ResolveCoarseLevel(index level) const;
        /// Operator-level validation after resolving an equation's dimension.
        /// Configuration checks name syntax first; material setup resolves
        /// component names against the actual equation schema before execution.
        void ValidateComponents(index components) const;
        DNDS_DECLARE_CONFIG(ImplicitSettings)
        {
            DNDS_FIELD(diagonal, "entries or full block diagonal storage");
            DNDS_FIELD(jacobianAction, "assembled (canonical) or directEvaluated off-diagonal Ax; semiMatrixFree is a compatibility spelling. Diagonal blocks and ILU factors remain explicit");
            DNDS_FIELD(sourceJacobianMode, "center: retained one-point mean source linearization; rhsQuadrature: opt-in variational source derivatives at the RHS source quadrature");
            DNDS_FIELD(boundaryJacobian, "Scalar-dissipation boundary diagonal: physical composes the frozen exterior-state derivative; eulerScalar keeps only the spectral diagonal, matching Euler useRoeJacobian=false. Roe dissipation retains its own boundary linearization");
            DNDS_FIELD(updatePolicy, "Registered update policy name; auto resolves singleStageImplicit or ditrSequential2Stage from the time method");
            DNDS_FIELD(iterations, "Maximum nonlinear corrections per physical stage");
            DNDS_FIELD(minIterations, "Minimum corrections before accepting a stage");
            DNDS_FIELD(consoleInterval, "Rank-zero nonlinear residual/VR progress interval; zero disables it");
            DNDS_FIELD(relativeTolerance, "Selected-component stage residual tolerance relative to its configured reference");
            DNDS_FIELD(absoluteTolerance, "Per-component absolute volume-integrated stage residual tolerance");
            DNDS_FIELD(algebraicRateTolerance, "Absolute volume-weighted RMS tolerance on the spatial algebraic rate at each implicit solve (code units)");
            DNDS_FIELD(normOrder, "Volume-integrated residual norm order: 1 or 2");
            DNDS_FIELD(residualReference, "runningMaximum (Euler default) or initial per-component reference within one solve; reset at solve entry");
            DNDS_FIELD(residualComponents, "Named nonlinear convergence components. [auto] resolves to density for compressible flow or momentum for ACM. Empty selects all. ACM continuity is also checked independently");
            DNDS_FIELD(pseudoCFL, "Initial local pseudo-time CFL for each stage");
            DNDS_FIELD(maxPseudoCFL, "Maximum pseudo-time CFL");
            DNDS_FIELD(pseudoGrowth, "CFL growth factor after each nonlinear correction");
            DNDS_FIELD(nIterCellSmoothScale, "Euler geometric pseudo-time smoothing passes at FV setup; zero disables it");
            DNDS_FIELD(relaxation, "Multiplier on each computed correction in (0,1]");
            DNDS_FIELD(roeDissipation, "Use full equation-owned characteristic dissipation; requires block diagonal");
            DNDS_FIELD(collectiveChecks, "Optional solver and Runtime halo safety agreements, default on; false also disables extra halo freshness audits. Routing/protocol and numerical collectives remain. Runtime local launch failure then aborts instead of rank-asymmetric recovery");
            DNDS_FIELD(requireConvergedReconstruction, "Require the requested VR tolerance as well as the mean equation target; false preserves Euler-style fixed-sweep acceptance");
            DNDS_FIELD(rejectUnconvergedStages, "Reject a finite admissible stage or coupled DITR solve at its iteration cap; false advances the final iterate like Euler while retaining an unconverged report");
            config.field_section(&T::linear, "linear", "Linear solve controls");
            config.field_section(&T::sgs, "sgs", "Symmetric relaxation controls");
            config.field_section(&T::ilu, "ilu", "Rank-local factorization controls");
            config.field_section(&T::positivity, "positivity", "Euler-derived implicit PP controls");
            config.field_section(&T::multigrid, "multigrid", "Same-cell polynomial FAS hierarchy");
        }
    };
    struct BoxInitializer
    {
        real x0 = 0, x1 = 0, y0 = 0, y1 = 0, z0 = 0, z1 = 0;
        Json v = nullptr;
        DNDS_DECLARE_CONFIG(BoxInitializer)
        {
            DNDS_FIELD(x0, "Strict lower x bound in ground-frame code coordinates");
            DNDS_FIELD(x1, "Strict upper x bound");
            DNDS_FIELD(y0, "Strict lower y bound");
            DNDS_FIELD(y1, "Strict upper y bound");
            DNDS_FIELD(z0, "Strict lower z bound, including for 2D geometry");
            DNDS_FIELD(z1, "Strict upper z bound");
            config.field_json_schema(&T::v, "v", "Required complete physical state", Physics::StateInput::Schema);
        }
    };
    struct PlaneInitializer
    {
        real a = 0, b = 0, c = 0, h = 0;
        Json v = nullptr;
        DNDS_DECLARE_CONFIG(PlaneInitializer)
        {
            DNDS_FIELD(a, "x coefficient in a*x+b*y+c*z+h>0, ground-frame code coordinates");
            DNDS_FIELD(b, "y coefficient");
            DNDS_FIELD(c, "z coefficient");
            DNDS_FIELD(h, "Additive plane offset, preserving Euler implementation sign");
            config.field_json_schema(&T::v, "v", "Required complete physical state", Physics::StateInput::Schema);
        }
    };
    struct ExprtkInitializer
    {
        std::vector<std::string> exprs;
        std::string stateType = "primRhoP";
        DNDS_DECLARE_CONFIG(ExprtkInitializer)
        {
            DNDS_FIELD(exprs, "Ordered program lines; x[3], t, iCell, inRegion, UExprtk; return zero");
            DNDS_FIELD(stateType, "Euler state input type, optionally physical _phy units");
        }
    };
    struct InitialFieldSettings
    {
        /// Volume quadrature order used to conservatively average programmed
        /// point states into cell means. Six preserves the historical path.
        int integrationOrder = 6;
        std::vector<BoxInitializer> boxInitializers;
        std::vector<PlaneInitializer> planeInitializers;
        std::vector<ExprtkInitializer> exprtkInitializers;
        bool Empty() const { return boxInitializers.empty() && planeInitializers.empty() && exprtkInitializers.empty(); }
        DNDS_DECLARE_CONFIG(InitialFieldSettings)
        {
            DNDS_FIELD(integrationOrder, "Cell-average quadrature order in [0,6]; default 6 preserves the historical initialization");
            config.field_array_of<BoxInitializer>(&T::boxInitializers, "boxInitializers", "Ordered quadrature-point overrides; later matches win");
            config.field_array_of<PlaneInitializer>(&T::planeInitializers, "planeInitializers", "Applied after boxes, in list order");
            config.field_array_of<ExprtkInitializer>(&T::exprtkInitializers, "exprtkInitializers", "Applied after planes; inRegion selects the updated point state");
        }
    };
    struct DomainSettings
    {
        EquationSettings equation;
        ExecutionSettings execution;
        MeshSettings mesh;
        RigidMotionSettings motion;
        NumericsSettings numerics;
        ImplicitSettings implicit;
        std::array<real, 5> initialPrimitive{1, 0, 0, 0, 1};
        Json initialState = nullptr, referenceState = nullptr;
        std::map<std::string, real> initialMassFractions;
        std::map<std::string, real> initialRANS;
        real initialTemperature = 0;
        FlowProfileSettings initialProfile;
        InitialFieldSettings initialField;
        std::map<std::string, BoundarySettings> boundaries;
        /// Resolved from global output controls by the Simulation frontend.
        bool boundaryFacialFluxes = false;
        /// Extra halo audits are opt-in and respect the umbrella check switch.
        bool CheckHaloFreshness() const { return execution.haloFreshnessChecks && implicit.collectiveChecks; }
        DNDS_DECLARE_CONFIG(DomainSettings)
        {
            config.field_section(&T::equation, "equation", "Equation and constitutive laws");
            config.field_section(&T::execution, "execution", "Explicit execution capabilities");
            config.field_section(&T::mesh, "mesh", "Mesh input");
            config.field_section(&T::motion, "motion", "Prescribed rigid frame and mesh velocity; absolute flow state in mesh axes");
            config.field_section(&T::numerics, "numerics", "Discrete operators");
            config.field_section(&T::implicit, "implicit", "Nonlinear CPU implicit stage controls");
            DNDS_FIELD(initialPrimitive, "Constant initial rho,u,v,w,p");
            config.field_json_schema(&T::initialState, "initialState", "Complete tagged initial state; null uses existing initial fields", Physics::StateInput::Schema);
            config.field_json_schema(&T::referenceState, "referenceState", "Complete physical reference state; never inferred from unit scales or initial values", Physics::StateInput::Schema);
            config.field_schema(&T::initialMassFractions, "initialMassFractions", "Initial species mass fractions; omitted last species receives the remainder", SpeciesMassFractionsSchema);
            config.field_schema(&T::initialRANS, "initialRANS", "Complete named conservative RANS initial values in code units", SpeciesMassFractionsSchema);
            DNDS_FIELD(initialTemperature, "Positive constant initial temperature overrides initialPrimitive pressure; zero uses pressure");
            config.field_section(&T::initialProfile, "initialProfile", "Quadrature-averaged initial physical field");
            config.field_section(&T::initialField, "initialField", "Ordered spatial overrides of a constant background; conservatively averaged at cell quadrature");
            config.field_map_of<BoundarySettings>(&T::boundaries, "boundaries", "Boundary-zone ID to condition");
        }
    };
    /// Cauchy's ordered reconstruction uses one logical core-first order under
    /// both overlap policies. Standalone CFV retains natural-order All support.
    /// Resolve before publishing root config and when constructing FV directly.
    DomainSettings ResolveSweepOrdering(DomainSettings settings);
    struct DITRTimeSettings
    {
        std::string variant = "U2R2";
        real fraction = .55, theta = .9146;
        /// Deprecated compatibility overrides, not a second iteration policy.
        /// Missing/null values inherit each domain's common implicit controls.
        std::optional<index> iterations, minIterations;
        std::optional<real> relativeTolerance, absoluteTolerance;
        void Validate() const;
        ImplicitSettings ResolveControls(const ImplicitSettings &common) const;
        DNDS_DECLARE_CONFIG(DITRTimeSettings)
        {
            DNDS_FIELD(variant, "U2R2 or U2R1 interpolation; U2R1 is available for reaction-dominated stepping");
            DNDS_FIELD(fraction, "Intermediate time fraction; CPU positive-scale profile requires (0,2/3)");
            DNDS_FIELD(theta, "Positive H+theta*I intermediate-equation weight");
            config.field_optional(&T::iterations, "iterations", "Deprecated maximum-corrections override; null inherits implicit.iterations");
            config.field_optional(&T::minIterations, "minIterations", "Deprecated minimum paired-corrections override; null inherits implicit.minIterations");
            config.field_optional(&T::relativeTolerance, "relativeTolerance", "Deprecated I+H relative-target override; null inherits implicit.relativeTolerance");
            config.field_optional(&T::absoluteTolerance, "absoluteTolerance", "Deprecated I+H absolute-target override; null inherits implicit.absoluteTolerance");
        }
    };
    struct PhysicalTimePositivitySettings
    {
        bool enabled = false;
        real relax = .8, scale = 1, increaseLimit = 2;
        void Validate() const;
        DNDS_DECLARE_CONFIG(PhysicalTimePositivitySettings)
        {
            DNDS_FIELD(enabled, "Euler physical dt PP bound from a zero-reconstruction trial before time-equation preparation");
            DNDS_FIELD(relax, "Euler density/internal-energy relaxation in (0,1]");
            DNDS_FIELD(scale, "Positive trial-increment multiplier dtPPLimitScale");
            DNDS_FIELD(increaseLimit, "Maximum timestep growth from the last accepted positive step; at least one");
        }
    };
    struct TimeSettings
    {
        std::string method = "ssprk3";
        std::string sourceSplitting = "coupled";
        bool steady = false;
        real dt = 1e-3;
        index steps = 1;
        real tEnd = 0;
        /// Compatibility input only. Resolution copies a nonnegative value to
        /// tEnd and rewrites the resolved root; native code uses tEnd.
        real finalTime = -1;
        real cfl = 0, wallSeconds = 0;
        real steadyTolerance = 0;
        index steadyMinSteps = 100;
        int maxRetries = 0;
        DITRTimeSettings ditr;
        PhysicalTimePositivitySettings positivity;
        Physics::Chemistry::ReactorStepSettings reactor;
        void ValidateSourceIntegration() const
        {
            DNDS_check_throw_info(sourceSplitting == "coupled" || sourceSplitting == "strang", "unknown chemical source splitting method");
            DNDS_check_throw_info(sourceSplitting != "strang" || !steady, "Strang source splitting requires physical transient time");
            reactor.Validate();
        }
        DNDS_DECLARE_CONFIG(TimeSettings)
        {
            DNDS_FIELD(method, "Coordinated physical-time method");
            DNDS_FIELD(sourceSplitting, "Chemical source integration: coupled, or strang constant-volume half-steps around source-free transport; inert domains are unchanged");
            DNDS_FIELD(steady, "Euler steady override: BDF1 infinite-dt limit, one outer solve and frozen physical time");
            DNDS_FIELD(dt, "Fixed positive physical timestep");
            DNDS_FIELD(steps, "Nonnegative accepted step target");
            DNDS_FIELD(tEnd, "Positive physical-time target; zero uses the step target");
            DNDS_FIELD(finalTime, "Deprecated alias for tEnd; leave at -1 in new inputs");
            DNDS_FIELD(cfl, "Positive CFL enables adaptive global stepping, capped by dt");
            DNDS_FIELD(wallSeconds, "Collective graceful wall stop; zero disables it");
            DNDS_FIELD(maxRetries, "C++ frontend maximum whole-step dt halvings after rejection; zero reports failure. Direct/Python callers own retries");
            DNDS_FIELD(steadyTolerance, "Relative accepted-state residual target; zero means physical transient");
            DNDS_FIELD(steadyMinSteps, "Minimum accepted steps before a steady stop");
            config.field_section(&T::ditr, "ditr", "Coupled U2R2/U2R1 time equations");
            config.field_section(&T::positivity, "positivity", "Euler physical timestep positivity/growth policy");
            config.field_section(&T::reactor, "reactor", "Existing Euler constant-volume source-step controls");
        }
    };
    struct ImplicitOutputSettings
    {
        index consoleInterval = 1;
        index historyInterval = 1;
        index checkpointInterval = 0;
        index visualizationInterval = -1;
        DNDS_DECLARE_CONFIG(ImplicitOutputSettings)
        {
            DNDS_FIELD(consoleInterval, "Nonlinear iteration console cadence; zero suppresses ordinary iteration lines");
            DNDS_FIELD(historyInterval, "Nonlinear iteration CSV cadence; stage/step end rows are unconditional");
            DNDS_FIELD(checkpointInterval, "Restartable nonlinear-iteration checkpoint cadence; zero disables it");
            DNDS_FIELD(visualizationInterval, "Nonlinear-iteration visualization cadence; -1 inherits checkpointInterval and zero disables it");
        }
    };
    struct DiagnosticOutputSettings
    {
        bool allocations = false;
        index allocationWarmupSteps = 1;
        bool perRank = false;
        std::string level = "info";
        index maxBytes = 64 * 1024 * 1024;
        index rotatedFiles = 3;
        DNDS_DECLARE_CONFIG(DiagnosticOutputSettings)
        {
            DNDS_FIELD(allocations, "Track Runtime allocations from simulation setup and write rank-local memory reports after warm-up and at final output");
            DNDS_FIELD(allocationWarmupSteps, "Accepted steps before the working-set report; zero reports immediately after output initialization");
            DNDS_FIELD(perRank, "Write one rotating operational diagnostic file per global rank");
            DNDS_FIELD(level, "Operational diagnostic level: trace, debug, info, warn, error, critical or off");
            DNDS_FIELD(maxBytes, "Per-rank diagnostic rotation size in bytes");
            DNDS_FIELD(rotatedFiles, "Number of rotated per-rank diagnostic files retained");
        }
    };
    struct OutputSettings
    {
        std::string prefix = "cauchy";
        /// Deprecated compatibility alias. A nonnegative value is resolved to
        /// checkpointInterval before execution.
        index interval = -1;
        index checkpointInterval = 1;
        index visualizationInterval = -1;
        real checkpointTimeInterval = 0;
        real visualizationTimeInterval = -1;
        real endpointSnapRelativeTolerance = smallReal;
        index consoleInterval = 1;
        index historyInterval = 1;
        index historySegmentRows = 10000;
        int historyPrecision = 10;
        std::string consoleColor = "auto", consoleBold = "auto";
        bool eventLog = true;
        int hdfDeflateLevel = 0;
        bool boundaryFacialFluxes = false;
        bool checkpointReconstruction = false;
        bool asynchronous = false;
        std::vector<int> forceZones;
        ImplicitOutputSettings implicit;
        DiagnosticOutputSettings diagnostics;
        DNDS_DECLARE_CONFIG(OutputSettings)
        {
            DNDS_FIELD(prefix, "Output file prefix");
            DNDS_FIELD(interval, "Deprecated alias for checkpointInterval; leave at -1 in new inputs");
            DNDS_FIELD(checkpointInterval, "Accepted-step checkpoint cadence; zero disables periodic checkpoints");
            DNDS_FIELD(visualizationInterval, "Accepted-step visualization cadence; -1 inherits checkpointInterval and zero disables it");
            DNDS_FIELD(checkpointTimeInterval, "Physical-time checkpoint cadence; zero disables it");
            DNDS_FIELD(visualizationTimeInterval, "Physical-time visualization cadence; -1 inherits checkpointTimeInterval and zero disables it");
            DNDS_FIELD(endpointSnapRelativeTolerance, "Relative dt tolerance for snapping physical steps to output times; Euler default is 1e-10");
            DNDS_FIELD(consoleInterval, "Accepted-step console cadence; zero suppresses ordinary step lines");
            DNDS_FIELD(historyInterval, "Accepted-step CSV cadence; final step rows remain unconditional");
            DNDS_FIELD(historySegmentRows, "Maximum history rows per CSV segment; each segment has the same header");
            DNDS_FIELD(historyPrecision, "Scientific-notation digits after the decimal point for real CSV cells");
            DNDS_FIELD(consoleColor, "Console color policy: auto, always or never");
            DNDS_FIELD(consoleBold, "Console emphasis policy, independent of color: auto, always or never");
            DNDS_FIELD(eventLog, "Write plain-text rotating events-rank<N>.log alongside ordinary CLI progress by default; no tee is needed. Uses diagnostics rotation limits");
            DNDS_FIELD(boundaryFacialFluxes, "Write observed face-averaged numerical flux and tangential force in boundary VTKHDF");
            DNDS_FIELD(hdfDeflateLevel, "VTKHDF compression level 0..9; uncompressed is the initial portable baseline");
            DNDS_FIELD(checkpointReconstruction, "Store optional compatible polynomial warm guesses for exact finite-sweep continuation; default omits them");
            DNDS_FIELD(asynchronous, "Bounded background accepted-step output; requires MPI_THREAD_MULTIPLE. Internal output and shutdown drain the queue");
            DNDS_FIELD(forceZones, "Boundary zones on which pressure and viscous body force are integrated");
            config.field_section(&T::implicit, "implicit", "Nonlinear iteration history, console and internal artifact cadence");
            config.field_section(&T::diagnostics, "diagnostics", "Operational logging sinks independent of scientific history");
        }
    };
    struct DomainConfig
    {
        DomainRankGroup placement;
        DomainSettings settings;
    };
    /// Directed point-interpolation connection. Configure two entries for a
    /// two-way interface. Zone IDs are scoped to their respective domains.
    struct InterfaceSettings
    {
        std::string donor, receiver, routing = "coarseBoxes";
        std::string refreshCadence = "stageFrozen";
        int donorZone = 0, receiverZone = 0, coarseBoxesPerRank = 4;
        real maxGap = 1e-8;
        DNDS_DECLARE_CONFIG(InterfaceSettings)
        {
            DNDS_FIELD(donor, "Donor domain name");
            DNDS_FIELD(receiver, "Receiver domain name");
            DNDS_FIELD(donorZone, "Selected donor boundary zone");
            DNDS_FIELD(receiverZone, "Interface receiver boundary zone");
            DNDS_FIELD(routing, "coarseBoxes or allGatherBoundary correctness baseline");
            DNDS_FIELD(refreshCadence,
                       "stageFrozen or nonlinearIteration value refresh policy");
            DNDS_FIELD(coarseBoxesPerRank, "Maximum coarse routing boxes per donor rank");
            DNDS_FIELD(maxGap, "Maximum physical point-to-surface projection distance");
        }
    };
    struct ConfigOverride
    {
        std::string pointer;
        Json value;
    };
    struct ResolvedExecutionPlan
    {
        std::vector<std::string> kernels;
        /// Enumerate every unavailable branch before domain construction. A
        /// backend's presence alone does not imply its equation/BC kernels exist.
        static ResolvedExecutionPlan Resolve(const DomainConfig &domain);
    };
    /// Immutable resolved root. Profiles contain DomainSettings patches. Resolve
    /// schema defaults -> selected profiles -> root defaults -> domain overrides
    /// -> CLI JSON pointers. Provenance uses pointers into the expanded root.
    /// Runtime clocks/history/controller values never mutate configuration.
    class SimulationConfig
    {
        struct Data
        {
            Json root;
            std::map<std::string, std::string> provenance;
            std::vector<DomainConfig> domains;
            TimeSettings time;
            OutputSettings output;
            std::vector<InterfaceSettings> interfaces;
        };
        /// Check a resolved domain before constructing collective resources.
        static void ValidateDomain(const DomainConfig &domain, int simulationSize);
        std::shared_ptr<const Data> _data;
        explicit SimulationConfig(std::shared_ptr<const Data> data) : _data(std::move(data)) {}

    public:
        /// @brief Canonical schema for the layered user input accepted by
        /// Resolve(). It describes patches, not the fully expanded Root().
        static Json InputSchema();
        static SimulationConfig Resolve(const Json &input, int simulationSize, const std::vector<ConfigOverride> &overrides = {});
        /// Production entry: coordinate local parsing/capability failures, then
        /// prove exact resolved-root/provenance agreement before mesh collectives.
        static SimulationConfig ResolveCollective(const Json &input, MPI_Comm simulation, const std::vector<ConfigOverride> &overrides = {});
        const Json &Root() const { return _data->root; }
        const std::map<std::string, std::string> &Provenance() const { return _data->provenance; }
        const std::vector<DomainConfig> &Domains() const { return _data->domains; }
        const TimeSettings &Time() const { return _data->time; }
        const OutputSettings &Output() const { return _data->output; }
        const std::vector<InterfaceSettings> &Interfaces() const { return _data->interfaces; }
        std::vector<DomainRankGroup> RankGroups() const;
    };
}
