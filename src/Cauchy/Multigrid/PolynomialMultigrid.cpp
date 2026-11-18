#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Multigrid/PolynomialMultigridOperations.hpp"

#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Implicit/NonlinearIteration.hpp"
#include "Cauchy/Implicit/NonlinearSolve.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Solver/FAS.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Algebra/DofScratchPool.hpp"
#include "Cauchy/Core/LocalStage.hpp"
#include <tuple>
#include <cstdlib>
#include <cstdio>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        struct EquationProfile
        {
            using State = EquationDofs;
            using Operations = PolynomialMultigridOperations;
            using Solver = NonlinearSolver;
            using Workspace = FiniteVolume::Workspace;
            using ReconstructionState = FiniteVolume::ReconstructionState;
            using Report = NonlinearIterationReport;
            static Algebra::DofBundle Bundle(const State &state) { return state.Fields(); }
            static State BindScratch(const State &prototype, const Algebra::DofBundle &fields) { return prototype.RebindFields(fields); }
            static std::vector<real> Norms(const State &state) { return std::vector<real>(size_t(state.ComponentNumber())); }
            static State NewState(const FiniteVolume &fv, const State &prototype)
            {
                Algebra::DofBundle fields(fv.Metrics().Space());
                for (index iField = 0; iField < prototype.PhysicalFieldNumber(); ++iField)
                    fields.Add(prototype.Fields().Name(iField), fv.Metrics().BuildField(prototype.PhysicalField(iField).Owned().Layout()));
                std::vector<EquationDofBinding> bindings;
                for (index iBlock = 0; iBlock < prototype.BlockNumber(); ++iBlock)
                {
                    const auto location = prototype.LocateBlock(iBlock);
                    bindings.push_back({prototype.Schema().blocks[iBlock].name, prototype.Fields().Name(location.physicalField), location.firstComponent});
                }
                // The new operator owns another DofSpace with the same cell
                // order. Construct that association explicitly, never relax
                // EquationDofs' ordinary same-space rebinding contract.
                return {prototype.Schema(), std::move(fields), std::move(bindings)};
            }
            static auto BuildWorkspace(const FiniteVolume &fv, State &state, const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers)
            {
                return fv.BuildWorkspace(state, workers);
            }
            static auto BuildSolver(const FiniteVolume &fv, State &state, std::shared_ptr<ReconstructionState> reconstruction,
                                    std::shared_ptr<Workspace> workspace,
                                    R::EagerEngine &engine, const R::ExecutionStream &stream, const DomainSettings &settings,
                                    const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers,
                                    std::shared_ptr<const real> pressureGauge)
            {
                const Algebra::ComponentSelection selection(state.ComponentNumber(), state.Schema().ResolveComponents(settings.implicit.ResolveResidualComponents(fv.Equation().IsACM())));
                return std::make_unique<Solver>(fv, state, std::move(reconstruction), std::move(workspace), engine, stream, settings, selection, workers, std::move(pressureGauge));
            }
            static void CheckTransfer(const State &a, const State &b)
            {
                a.Validate();
                b.Validate();
                a.RequireSchema(b.Schema());
                DNDS_check_throw_info(a.Space().OwnedRows() == b.Space().OwnedRows(), "polynomial FAS equation identity transfer shape mismatch");
            }
            static void SetBoundaryInputs(const FiniteVolume &fv, Workspace &workspace, const BoundaryInputs &input)
            {
                fv.SetBoundaryInputs(workspace, input);
            }
        };

        /// One p-hierarchy implementation for the canonical logical state.
        /// Physics, source quadrature and PP remain in the level stage solvers;
        /// time assembly and finest convergence remain outside the V-cycle.
        template <class Profile>
        struct PolynomialHierarchy
        {
            using State = typename Profile::State;
            using Operations = typename Profile::Operations;
            using Solver = typename Profile::Solver;
            using Report = typename Profile::Report;
            using Level = Multigrid::StateFASLevel<State>;
            using Transfer = Multigrid::StateFASTransfer<State>;
            using Cycle = Multigrid::StateFASCycle<State, Operations>;
            // Declared before hierarchy resources so it is destroyed after all
            // streams, solvers and scratch which may refer to its runtime.
            std::optional<DomainExecutionContext> executionOwner;
            Operations operations;
            struct SpatialRate
            {
                State rate;
                std::optional<typename Operations::Binding> stateStamp;
                real time = 0, scale = 0;
                std::optional<Runtime::Array<real>> derivative;
                OptionalRef<const Runtime::Array<real>> publishedDerivative;
                SpatialRate(const State &input, Solver &solver)
                    : rate(Operations::NewVector(input)), derivative(solver.BuildSourceDerivative(input)) {}
            };
            static void IdentityTransfer(const State &input, State &output, const MPIInfo &mpi,
                                         const Operations &operations, bool collectiveChecks = true)
            {
                // Only construction-validated equal cell orders may use this
                // explicit cross-space copy. Ordinary algebra stays strict.
                const auto run = [&]
                {
                    Profile::CheckTransfer(input, output);
                    Operations::RequireDistinct(input, output);
                    operations.CopyIdentityRows(input, output);
                    // Explicit level-edge publication: synchronous FAS
                    // callbacks return ready values and release pool borrows.
                    operations.Synchronize();
                };
                RunLocalStage(mpi, collectiveChecks, run,
                              "polynomial multigrid identity transfer failed");
            }
            struct Coarse
            {
                FiniteVolume discrete;
                State state;
                std::shared_ptr<typename Profile::ReconstructionState> reconstruction;
                std::shared_ptr<typename Profile::Workspace> workspace;
                std::unique_ptr<Solver> solver;
                static FiniteVolume PrepareLevel(ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings,
                                                 const FiniteVolume &fine, R::ExecutionContext context)
                {
                    FiniteVolume level(std::move(mesh), settings, fine.Equation());
                    return context.device == R::Device::Host() ? level : level.Placed(context.device, context);
                }
                Coarse(ssp<Geom::UnstructuredMesh> mesh, DomainSettings settings, const FiniteVolume &fine,
                       const State &prototype, R::EagerEngine &engine, const R::ExecutionStream &stream,
                       const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers,
                       const std::optional<DomainExecutionContext> &context, std::shared_ptr<const real> pressureGauge)
                    : discrete(PrepareLevel(std::move(mesh), settings, fine, stream.Context())), state(Profile::NewState(discrete, prototype))
                {
                    DNDS_check_throw_info(discrete.Metrics().Cells().SameOrder(fine.Metrics().Cells()),
                                          "polynomial FAS levels changed cell identity/order");
                    // A level solver validates its physical background at
                    // construction. Restriction has not run yet: seed the
                    // owned state explicitly instead of exposing zero means.
                    IdentityTransfer(prototype, state, MPIInfo(discrete.Metrics().Communicator()), Operations(stream));
                    reconstruction = discrete.BuildReconstructionState(state);
                    workspace = Profile::BuildWorkspace(discrete, state, workers);
                    if (context)
                        solver = std::make_unique<Solver>(discrete, state, reconstruction, workspace, *context, settings,
                                                          Algebra::ComponentSelection(state.ComponentNumber(), state.Schema().ResolveComponents(settings.implicit.ResolveResidualComponents(discrete.Equation().IsACM()))), pressureGauge);
                    else
                        solver = Profile::BuildSolver(discrete, state, reconstruction, workspace, engine, stream, settings, workers, std::move(pressureGauge));
                }
            };
            FiniteVolume fine;
            State state, value;
            // Retain the solver's shared implementation, not the caller's
            // movable facade. Stored level callbacks borrow this stable member.
            Solver fineSolver;
            ImplicitSettings settings;
            std::unique_ptr<NonlinearSolveSetup> nonlinearSetup;
            MPIInfo mpi;
            std::vector<std::unique_ptr<Coarse>> coarse;
            std::unique_ptr<Cycle> cycle;
            std::unique_ptr<Algebra::DofScratchPool> scratchPool;
            std::vector<std::array<index, 4>> scratchDescriptors;
            std::vector<Level> levelDefinitions;
            std::vector<Transfer> transferDefinitions;
            std::vector<real> smootherCFL, smootherBaseCFL, smootherMaxCFL;
            real mass = 0, rhsScale = 1, time = 0;
            real continuityRatio = 0;
            int stage = 0;
            Report *report = nullptr;
            // Opt-in local timings only: no barriers/reductions or clock calls
            // in ordinary runs. Profiles belong to this synchronous hierarchy.
            bool profileEnabled = std::getenv("DNDS_CAUCHY_PROFILE_PMG") != nullptr;
            bool profileAllRanks = std::getenv("DNDS_CAUCHY_PROFILE_ALL_RANKS") != nullptr;
            std::vector<NonlinearProfile> smootherProfiles;
            std::vector<real> operatorSeconds;
            std::vector<index> operatorCalls;
            std::vector<index> cachedRateUses;
            std::shared_ptr<SpatialRate> fineRate;
            std::vector<std::shared_ptr<SpatialRate>> cachedRates;
            PolynomialHierarchy(const FiniteVolume &discrete, State &input, Solver &solver, const DomainSettings &controls)
                : fine(discrete), state(input), value(Operations::NewVector(input)), fineSolver(solver.RetainForMultigrid()),
                  settings(controls.implicit), mpi(discrete.Metrics().Communicator())
            {
                settings.ValidateComponents(Operations::ComponentNumber(input));
                DNDS_check_throw_info(settings.multigrid.levels > 0 && controls.numerics.reconstruction == "variational" &&
                                          controls.numerics.cfv.maxOrder >= 2,
                                      "polynomial FAS requires high-order variational reconstruction");
                auto iterationSettings = settings;
                iterationSettings.iterations = settings.multigrid.cycles;
                nonlinearSetup = std::make_unique<NonlinearSolveSetup>(
                    iterationSettings, fine.Equation().Schema(), mpi.rank,
                    "singleStageImplicit",
                    UpdatePolicyRequirements{
                        1,
                        NonlinearDetail::ResidualObservation::BeforeUpdate,
                        false});
            }
            Level MakeLevel(State input, Solver &solver, index level)
            {
                auto *relax = &solver;
                // The finest convergence evaluation already advances its VR
                // guess: consume that exact rate for the first correction,
                // rather than performing a second set of configured VR sweeps.
                // Direct-gradient levels may also reuse their preceding tau
                // evaluation because they have no iterative VR history.
                const auto cached = !level || settings.multigrid.reuseCoarseRate ? std::make_shared<SpatialRate>(input, solver) : nullptr;
                if (cached)
                    cachedRates.push_back(cached);
                if (!level)
                    fineRate = cached;
                const auto smooth = [this, relax, level, cached](auto &u, const auto &source, index count, const std::vector<index> &components)
                {
                    for (index correction = 0; correction < count; ++correction)
                    {
                        OptionalRef<const State> rate;
                        if (cached && cached->stateStamp)
                        {
                            cached->stateStamp->Check(u, true);
                            DNDS_check_throw_info(cached->time == time && cached->scale == rhsScale,
                                                  "FAS spatial rate belongs to another time/equation scale");
                            rate.Emplace(cached->rate);
                            if (profileEnabled)
                                ++cachedRateUses[level];
                        }
                        std::shared_ptr<NonlinearProfile> timing;
                        NonlinearSolveControl control{1, smootherCFL[level], {}, components};
                        if (profileEnabled)
                            control.profile = timing = std::make_shared<NonlinearProfile>();
                        const auto step = relax->CorrectOnce(u, source, mass, rhsScale, time, stage,
                                                             std::move(control), rate, rate ? cached->publishedDerivative : OptionalRef<const Runtime::Array<real>>{});
                        if (cached)
                            cached->stateStamp.reset();
                        if (profileEnabled)
                        {
                            for (index phase = 0; phase < NonlinearProfile::PhaseNumber; ++phase)
                                smootherProfiles[level].seconds[phase] += timing->seconds[phase];
                            smootherProfiles[level].residualEvaluations += timing->residualEvaluations;
                        }
                        NonlinearDetail::AccumulateWork(*report, step);
                    }
                    return count;
                };
                return {std::move(input),
                        [this, relax, level, cached](auto &u, auto &out)
                        { TimedOperator(*relax, u, out, mass, level, level ? cached.get() : nullptr); },
                        [smooth](auto &u, const auto &source, index count)
                        { return smooth(u, source, count, {}); },
                        [this, relax, cached](auto &u, const auto &delta)
                        {
                            if (cached)
                                cached->stateStamp.reset();
                            const auto [limited, alpha] = relax->ApplyCorrection(u, delta, settings.multigrid.damping, settings.multigrid.updateComponents);
                            report->incrementLimited += limited;
                            report->minimumIncrementAlpha = std::min(report->minimumIncrementAlpha, alpha);
                        },
                        smooth,
                        [this, relax, level, cached](auto &u, auto &out)
                        { TimedOperator(*relax, u, out, 0, level, level ? cached.get() : nullptr); }};
            }
            void TimedOperator(Solver &solver, State &input, State &output, real equationMass, index level, SpatialRate *cached = nullptr)
            {
                const real start = profileEnabled ? MPI_Wtime() : 0;
                OptionalRef<Runtime::Array<real>> destination;
                if (cached && cached->derivative)
                    destination.Emplace(*cached->derivative);
                solver.EvaluateEquation(input, output, equationMass, rhsScale, time,
                                        cached ? OptionalRef<State>(cached->rate)
                                               : OptionalRef<State>{},
                                        destination);
                if (cached)
                {
                    cached->publishedDerivative = destination;
                    cached->stateStamp.emplace(input);
                    cached->time = time;
                    cached->scale = rhsScale;
                }
                if (profileEnabled)
                {
                    operatorSeconds[level] += MPI_Wtime() - start;
                    ++operatorCalls[level];
                }
            }
            void BuildCycle()
            {
                const auto &control = settings.multigrid;
                typename Cycle::AcquireScratch acquireScratch;
                if (control.scratchPool)
                {
                    scratchPool = std::make_unique<Algebra::DofScratchPool>();
                    const auto prototype = Profile::Bundle(levelDefinitions.front().state);
                    for (index slot = 0; slot < 4; ++slot)
                        scratchPool->Reserve(prototype);
                    scratchDescriptors.resize(levelDefinitions.size());
                    for (index level = 0; level < index(levelDefinitions.size()); ++level)
                    {
                        const auto &fv = level == 0 ? fine : coarse[level - 1]->discrete;
                        for (index slot = 0; slot < 4; ++slot)
                            scratchDescriptors[level][slot] = scratchPool->Prepare(slot, levelDefinitions[level].state.Space(),
                                                                                   [&](auto &field)
                                                                                   { return fv.BindHalo(field); });
                    }
                    acquireScratch = [this](index level, typename Cycle::ScratchKind kind, index lane)
                    {
                        const index slot = 2 * lane + (kind == Cycle::ScratchKind::Residual ? 1 : 0);
                        auto borrow = std::make_shared<Algebra::DofScratchPool::Borrow>(scratchPool->Acquire(scratchDescriptors[level][slot]));
                        return typename Cycle::Scratch{Profile::BindScratch(levelDefinitions[level].state, borrow->Fields()), borrow};
                    };
                }
                Multigrid::FASCycleSettings sweeps{control.preSmooth, control.postSmooth, control.coarseSmooth, control.updateComponents};
                sweeps.preSmoothByLevel.assign(size_t(control.levels), control.preSmooth);
                sweeps.postSmoothByLevel.assign(size_t(control.levels), control.postSmooth);
                for (const auto &[name, level] : control.coarseLevelSettings)
                {
                    const index number = name == "1" ? 1 : 2;
                    if (number < control.levels)
                    {
                        if (level.preSmooth >= 0)
                            sweeps.preSmoothByLevel[number] = level.preSmooth;
                        if (level.postSmooth >= 0)
                            sweeps.postSmoothByLevel[number] = level.postSmooth;
                    }
                    else if (number == control.levels)
                        sweeps.coarseSmooth = (level.preSmooth < 0 ? control.coarseSmooth : level.preSmooth) +
                                              (level.postSmooth < 0 ? 0 : level.postSmooth);
                }
                cycle = std::make_unique<Cycle>(levelDefinitions, transferDefinitions, std::move(sweeps), [this](const auto &run)
                                                { RunLocalStage(mpi, settings.collectiveChecks, [&]
                                                                {
                        run();
                        // FAS callbacks promise completion, particularly
                        // before reserved scratch is returned/reborrowed.
                        // Multiple algebra launches inside this segment stay
                        // queued; observation is at the recursion boundary.
                        operations.Synchronize(); }, "polynomial multigrid local phase failed"); }, std::move(acquireScratch), operations);
            }
            void Build(const ssp<Geom::UnstructuredMesh> &mesh, R::EagerEngine &engine, const R::ExecutionStream &stream,
                       const DomainSettings &controls, const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers = std::nullopt)
            {
                operations = Operations(stream);
                smootherCFL.assign(size_t(settings.multigrid.levels + 1), settings.pseudoCFL);
                smootherProfiles.resize(smootherCFL.size());
                operatorSeconds.resize(smootherCFL.size());
                operatorCalls.resize(smootherCFL.size());
                cachedRateUses.resize(smootherCFL.size());
                smootherBaseCFL = {settings.pseudoCFL};
                smootherMaxCFL = {settings.maxPseudoCFL};
                levelDefinitions.push_back(MakeLevel(state, fineSolver, 0));
                const auto identity = [mpi = mpi, checks = settings.collectiveChecks, operations = operations](const State &input, State &output)
                {
                    IdentityTransfer(input, output, mpi, operations, checks);
                };
                for (int level = 0; level < settings.multigrid.levels; ++level)
                {
                    auto lower = PolynomialMultigrid::CoarseSettings(controls, level);
                    smootherBaseCFL.push_back(lower.implicit.pseudoCFL);
                    smootherMaxCFL.push_back(lower.implicit.maxPseudoCFL);
                    coarse.push_back(std::make_unique<Coarse>(mesh, lower, fine, state, engine, stream, workers, executionOwner, fineSolver.PressureGauge()));
                    levelDefinitions.push_back(MakeLevel(coarse.back()->state, *coarse.back()->solver, level + 1));
                    // These equations use cell-volume-normalized rates. Every
                    // level has the same schema and cell order. Identity
                    // restriction thus commutes with M for both I and the
                    // constant ACM pressure projection, even on curved cells.
                    transferDefinitions.push_back({identity, identity, identity, true});
                }
                BuildCycle();
            }
            void BindEquation(const State &source, real equationMass, real equationScale, real equationTime, int equationStage, real pseudoCFL)
            {
                const auto validate = [&]
                {
                    Operations::CheckCompatible(state, source);
                    Operations::RequireDistinct(state, source);
                    DNDS_check_throw_info(std::isfinite(equationMass) && equationMass >= 0 && std::isfinite(equationScale) && equationScale > 0 &&
                                              std::isfinite(equationTime) && (pseudoCFL == 0 || (std::isfinite(pseudoCFL) &&
                                              pseudoCFL >= settings.pseudoCFL && pseudoCFL <= settings.maxPseudoCFL)), "invalid polynomial FAS equation coefficients"); };
                if (settings.collectiveChecks)
                {
                    MPI::LocalStage(mpi, validate);
                    std::array<real, 4> minimum{equationMass, equationScale, equationTime, pseudoCFL}, maximum = minimum;
                    MPI::AllreduceInPlace<MPI::Reduction::Min>(minimum, mpi.comm);
                    MPI::AllreduceInPlace<MPI::Reduction::Max>(maximum, mpi.comm);
                    DNDS_check_throw_info(minimum == maximum, "polynomial FAS coefficients differ between ranks");
                }
                else
                    validate();
                for (auto &cached : cachedRates)
                    cached->stateStamp.reset();
                mass = equationMass;
                rhsScale = equationScale;
                time = equationTime;
                stage = equationStage;
            }
            auto Measure(const State &source)
            {
                TimedOperator(fineSolver, state, value, mass, 0, fineRate.get());
                auto norms = Profile::Norms(state);
                std::vector<real> sum;
                const bool acm = fine.Equation().IsACM();
                const auto accumulate = [&]
                {
                    sum = fineSolver.MeasureRateNorm(source, settings.normOrder, std::cref(value),
                                                     acm ? OptionalRef<const State>(fineRate->rate) : OptionalRef<const State>{});
                };
                // Only the diagnostic preflight is optional. The residual
                // norm below is a mathematical collective in both modes.
                // Unchecked failures propagate to the fail-fast frontend.
                if (settings.collectiveChecks)
                    MPI::LocalStage(mpi, accumulate);
                else
                    accumulate();
                MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                    MPI::ContiguousView<real>{sum.data(), sum.size()}, mpi.comm);
                const real volume = sum[norms.size()];
                DNDS_check_throw_info(volume > 0 && std::isfinite(volume), "invalid FAS global volume");
                if (acm)
                    continuityRatio = std::sqrt(sum.back() / volume) / settings.algebraicRateTolerance;
                for (index j = 0; j < index(norms.size()); ++j)
                {
                    norms[j] = settings.normOrder == 1 ? sum[j] : std::sqrt(sum[j]);
                    DNDS_check_throw_info(std::isfinite(norms[j]), "nonfinite FAS finest residual");
                }
                return norms;
            }
            Report ApplyCycle(const State &source, real mass, real rhsScale, real time, int stage, real pseudoCFL,
                              OptionalRef<const State> rate = {}, OptionalRef<const Runtime::Array<real>> derivative = {})
            {
                BindEquation(source, mass, rhsScale, time, stage, pseudoCFL);
                if (!cycle)
                    BuildCycle();
                if (rate)
                {
                    // Same contract as CorrectOnce: an external method owns
                    // time/stage/VR provenance. Internal Solve passes only the
                    // immediately preceding measured state, checked below.
                    if (rate.Get() != &fineRate->rate)
                    {
                        operations.Copy(fineRate->rate, *rate);
                        operations.Synchronize();
                    }
                    fineRate->publishedDerivative = derivative;
                    fineRate->stateStamp.emplace(state);
                    fineRate->time = time;
                    fineRate->scale = rhsScale;
                }
                const real cfl = pseudoCFL == 0 ? settings.pseudoCFL : pseudoCFL;
                for (index level = 0; level < index(smootherCFL.size()); ++level)
                    smootherCFL[level] = std::min(smootherMaxCFL[level], smootherBaseCFL[level] * (cfl / settings.pseudoCFL));
                Report work;
                work.residual = Profile::Norms(state);
                work.pseudoCFL = cfl;
                report = &work;
                try
                {
                    const auto counts = cycle->Apply(source);
                    work.multigridCycles = work.iterations = 1;
                    work.levelCorrections = counts.corrections;
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("Cauchy multigrid cycle failed");
                    cycle.reset();
                    report = nullptr;
                    throw;
                }
                report = nullptr;
                return work;
            }
            Report Solve(const State &source, real mass, real rhsScale,
                         real time, int stage,
                         NonlinearSolveControl solveControl,
                         OptionalRef<const NonlinearSolveSetup> setup = {})
            {
                BindEquation(source, mass, rhsScale, time, stage, 0);
                fineSolver.BindControllerState(solveControl);
                auto residual = Profile::Norms(state);
                Report initial;
                initial.residual = residual;
                const real start = profileEnabled ? MPI_Wtime() : 0;
                const auto &activeSetup = setup ? *setup
                                                : *nonlinearSetup;
                auto session = NonlinearSession::Prepare(
                    activeSetup, stage,
                    {std::move(solveControl), std::move(initial), fineSolver.ReconstructionTolerance(), [&]
                     { residual = Measure(source); },
                     [&](auto &norms)
                     { norms = residual; },
                     [&]
                     { return fineSolver.ReconstructionStats(); },
                     [&](int coupledStage, auto &total, bool applyCorrection)
                     {
                         if (!applyCorrection)
                             return;
                         DNDS_check_throw_info(
                             coupledStage == 0,
                             "p-FAS single-stage policy requested another stage");
                         fineRate->stateStamp->Check(state, true);
                         NonlinearDetail::AccumulateWork(
                             total,
                             ApplyCycle(source, mass, rhsScale, time, stage,
                                        total.pseudoCFL,
                                        std::cref(fineRate->rate), fineRate->publishedDerivative));
                     },
                     {},
                     [this]
                     { return continuityRatio; }});
                const auto result = SolveNonlinear(session);
                if (profileEnabled && (!mpi.rank || profileAllRanks))
                {
                    Json levels = Json::array();
                    for (index level = 0; level < index(smootherProfiles.size()); ++level)
                        levels.push_back({{"level", level}, {"smoother_seconds", smootherProfiles[level].seconds}, {"smoother_rhs_calls", smootherProfiles[level].residualEvaluations}, {"cached_rate_uses", cachedRateUses[level]}, {"operator_seconds", operatorSeconds[level]}, {"operator_calls", operatorCalls[level]}});
                    const auto record = Json{{"rank", mpi.rank}, {"local_seconds", MPI_Wtime() - start}, {"cycles", result.multigridCycles}, {"levels", levels}}.dump();
                    std::fprintf(stdout, "CAUCHY_PMG_PROFILE %s\n", record.c_str());
                }
                return result;
            }
            NonlinearSession Start(
                const NonlinearSolveSetup &setup, const State &source,
                real mass, real rhsScale, real time, int stage,
                NonlinearSolveControl solveControl,
                std::shared_ptr<void> lifetime)
            {
                BindEquation(source, mass, rhsScale, time, stage, 0);
                fineSolver.BindControllerState(solveControl);
                auto residual = std::make_shared<std::vector<real>>(
                    Profile::Norms(state));
                Report initial;
                initial.residual = *residual;
                return NonlinearSession::Prepare(
                    setup, stage,
                    {std::move(solveControl), std::move(initial),
                     fineSolver.ReconstructionTolerance(),
                     [this, residual, source = source.NumericAlias()]
                     { *residual = Measure(source); },
                     [residual](auto &norms)
                     { norms = *residual; },
                     [this]
                     { return fineSolver.ReconstructionStats(); },
                     [this, source = source.NumericAlias(), mass, rhsScale, time,
                      stage](int coupledStage, auto &total, bool applyCorrection)
                     {
                         if (!applyCorrection)
                             return;
                         DNDS_check_throw_info(
                             coupledStage == 0,
                             "p-FAS single-stage policy requested another stage");
                         fineRate->stateStamp->Check(state, true);
                         NonlinearDetail::AccumulateWork(
                             total,
                             ApplyCycle(source, mass, rhsScale, time, stage,
                                        total.pseudoCFL,
                                        std::cref(fineRate->rate), fineRate->publishedDerivative));
                     },
                     std::move(lifetime), [this]
                     { return continuityRatio; }});
            }
            void SetBoundaryInputs(const BoundaryInputs &input)
            {
                for (auto &lower : coarse)
                    Profile::SetBoundaryInputs(lower->discrete, *lower->workspace, input);
            }
        };
    }
    struct PolynomialMultigrid::Data : PolynomialHierarchy<EquationProfile>
    {
        using PolynomialHierarchy::PolynomialHierarchy;
    };

    DomainSettings PolynomialMultigrid::CoarseSettings(const DomainSettings &fine, index level)
    {
        DNDS_check_throw_info(level >= 0 && level < 2, "polynomial coarse level must be zero or one");
        auto lower = fine;
        lower.implicit = fine.implicit.ResolveCoarseLevel(level + 1);
        // Euler's coarse loop omits EvaluateCellRHSAlpha before the solve.
        // Match that default; explicit opt-in retains coarse RHS/dTau PP.
        // Reconstruction and final-increment admissibility are independent.
        if (!fine.implicit.multigrid.coarseRHSFPP)
            lower.implicit.positivity.rhsFPPMode = "none";
        // P0 drops reconstructed gradients only from its viscous face flux
        // (which retains the face-jump term). Keep reconstruction available
        // for all volume-source gradients, including RANS and chemistry.
        lower.numerics.reconstruction = "gradient";
        // The coarse direct-gradient operator has no variational system to
        // solve. Do not inherit the finest VR-only AMG selector; the separate
        // flow linear-solver settings above remain available on every level.
        lower.numerics.reconstructionSolver = ReconstructionSolverSettings{};
        lower.numerics.firstOrderConvection = level > 0;
        lower.numerics.cfv.maxOrder = 1;
        if (lower.numerics.limiter == "LVR")
            lower.numerics.limiter = "WBAP";
        if (level)
            lower.numerics.limiter = "none";
        return lower;
    }

    PolynomialMultigrid::PolynomialMultigrid(ssp<Geom::UnstructuredMesh> mesh, const FiniteVolume &fine, EquationDofs &state,
                                             NonlinearSolver &solver, Runtime::EagerEngine &engine, Runtime::ExecutionStream stream, const DomainSettings &settings,
                                             std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
    {
        MPI::LocalStage(MPIInfo(fine.Metrics().Communicator()), [&]
                        { _data = std::make_shared<Data>(fine, state, solver, settings); });
        _data->Build(mesh, engine, stream, settings, workers);
    }
    PolynomialMultigrid::PolynomialMultigrid(
        ssp<Geom::UnstructuredMesh> mesh, const FiniteVolume &fine,
        EquationDofs &state, NonlinearSolver &solver,
        DomainExecutionContext context, const DomainSettings &settings)
    {
        MPI::LocalStage(MPIInfo(fine.Metrics().Communicator()), [&]
                        { _data = std::make_shared<Data>(fine, state, solver, settings); });
        _data->executionOwner.emplace(context);
        _data->Build(mesh, context.Engine(), context.ComputeStream(), settings, context.ChemicalWorkers());
    }
    NonlinearIterationReport PolynomialMultigrid::Solve(const EquationDofs &source, real mass, real rhsScale, real time, int stage,
                                                        NonlinearSolveControl control)
    {
        auto session = _data->fineSolver.ReserveForMultigrid();
        return _data->Solve(source, mass, rhsScale, time, stage, std::move(control));
    }
    NonlinearSession PolynomialMultigrid::PrepareSolve(
        const NonlinearSolveSetup &setup, const EquationDofs &source,
        real mass, real rhsScale, real time, int stage,
        NonlinearSolveControl control)
    {
        struct Lifetime
        {
            std::shared_ptr<Data> data;
            Runtime::SessionAccess::Lease session;
        };
        auto lifetime = std::make_shared<Lifetime>(
            Lifetime{_data, _data->fineSolver.ReserveForMultigrid()});
        return _data->Start(setup, source, mass, rhsScale, time, stage,
                            std::move(control), std::move(lifetime));
    }
    NonlinearIterationReport PolynomialMultigrid::ApplyCycle(
        const EquationDofs &source, real mass, real rhsScale, real time, int stage, real pseudoCFL,
        OptionalRef<const EquationDofs> rate, OptionalRef<const Runtime::Array<real>> derivative)
    {
        auto session = _data->fineSolver.ReserveForMultigrid();
        return _data->ApplyCycle(source, mass, rhsScale, time, stage, pseudoCFL, rate, derivative);
    }
    void PolynomialMultigrid::SetBoundaryInputs(const BoundaryInputs &input)
    {
        auto session = _data->fineSolver.ReserveForMultigrid();
        _data->SetBoundaryInputs(input);
    }
    std::function<void(const BoundaryInputs &)>
    PolynomialMultigrid::PrepareBoundaryInputBinding() const
    {
        using Receiver = std::pair<FiniteVolume, std::shared_ptr<FiniteVolume::Workspace>>;
        std::vector<Receiver> receivers;
        receivers.reserve(_data->coarse.size());
        for (const auto &level : _data->coarse)
            receivers.emplace_back(level->discrete, level->workspace);
        // Retain only receiver resources. Capturing the hierarchy would also
        // retain its execution context and form a cycle when this binding is
        // stored in that context's pending interface dependency.
        return [receivers = std::move(receivers)](const BoundaryInputs &input)
        {
            for (const auto &receiver : receivers)
                receiver.first.SetBoundaryInputs(*receiver.second, input);
        };
    }
    std::pair<size_t, index> PolynomialMultigrid::ScratchPoolStatistics() const
    {
        const auto *pool = _data->scratchPool.get();
        return pool ? std::pair<size_t, index>{pool->RetainedBytes(), pool->PeakBorrows()} : std::pair<size_t, index>{0, 0};
    }
}
