#pragma once

#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"

namespace DNDS::Cauchy
{
    class DomainExecutionContext;
    class NonlinearSolveSetup;
    class NonlinearSession;
    /// Live nonlinear setup state. This object contains only small,
    /// JSON-serializable values; field payloads and runtime resources belong to
    /// the checkpoint bundle and execution context respectively.
    struct NonlinearSolveState
    {
        int schemaVersion = 1;
        index attempt = 0, iteration = 0;
        int stage = 0, coupledStage = -1;
        bool converged = false, terminal = false, capped = false, rejected = false;
        std::vector<real> residual, initialResidual, referenceResidual;
        std::vector<std::string> convergenceComponents;
        real residualTargetRatio = 0, pseudoCFL = 0;
        index linearIterations = 0, multigridCycles = 0, rhsLimited = 0, incrementLimited = 0;
        std::vector<index> levelCorrections;
        real minimumRHSAlpha = 1, minimumIncrementAlpha = 1;
        int reconstructionIterations = 0;
        real reconstructionRelativeUpdate = 0;
        bool reconstructionUpdateMeasured = false;

        Json ToJson() const;
        static NonlinearSolveState FromJson(const Json &value);
    };
    void to_json(Json &value, const NonlinearSolveState &state);
    void from_json(const Json &value, NonlinearSolveState &state);

    /// Sink-neutral nonlinear progress snapshot. Numerical solvers publish the
    /// same values to history/console observers; no file or MPI policy lives in
    /// this type. Vectors follow the equation schema's logical component order.
    struct NonlinearProgress
    {
        int stage = 0, coupledStage = -1;
        index iteration = 0;
        bool converged = false, terminal = false;
        std::vector<real> residual, initialResidual, referenceResidual;
        real residualTargetRatio = 0, pseudoCFL = 0;
        index linearIterations = 0, rhsLimited = 0, incrementLimited = 0;
        real minimumRHSAlpha = 1, minimumIncrementAlpha = 1;
        int reconstructionIterations = 0;
        real reconstructionRelativeUpdate = 0;
        bool reconstructionUpdateMeasured = false;
        /// Local wall duration of the completed nonlinear iteration. This is
        /// sink-only telemetry, not restart state or a collective maximum.
        real iterationWallSeconds = 0;
        /// Worst selected component's norm divided by its own reference.
        real relativeConverged = 0;
    };
    /// Optional CPU profile. Values are local wall seconds, not
    /// reduced across ranks. Total encloses the disjoint named sections;
    /// its remainder includes setup work and timer overhead. The caller
    /// shares this output with the solve/session, including unwinding, and
    /// serializes external reads with solve progression.
    struct NonlinearProfile
    {
        enum Phase
        {
            Entry,
            Residual,
            Convergence,
            PseudoTimePP,
            Jacobian,
            LinearSolve,
            Correction,
            Total,
            PhaseNumber
        };
        std::array<real, PhaseNumber> seconds{};
        index residualEvaluations = 0;
    };
    struct NonlinearSolveControl
    {
        index correctionLimit = -1; // -1 uses the configured stage cap
        real initialPseudoCFL = 0;  // 0 starts at the configured stage CFL
        /// Null means no clock queries or extra communication. A provided
        /// output resets per call and retains partial timings on exceptions.
        std::shared_ptr<NonlinearProfile> profile;
        /// Empty updates all components. A subset solves the principal
        /// linear system with all other means frozen, including through PP.
        /// This never changes the separately configured convergence selection.
        std::vector<index> updateComponents;
        /// Called synchronously after a completed nonlinear correction is
        /// measured. It must not enter unrelated MPI collectives. The observer
        /// receives copied scalar/vector data and may buffer it for a safe
        /// frontend flush boundary.
        std::function<void(const NonlinearProgress &)> progress;
        /// Optional live setup publication. Iterate updates it before the
        /// progress observer, so checkpoint/console/CSV see one state source.
        /// Shared ownership also covers externally phased prepared sessions;
        /// dropping the caller's control object must not invalidate publication.
        std::shared_ptr<NonlinearSolveState> state;
        /// Called at a restart-safe boundary after the current correction (or
        /// at a terminal observation) and after the next pseudo-CFL is
        /// published. It may synchronously serialize the supplied small state.
        std::function<void(const NonlinearSolveState &)> quiescent;
        /// Continue from the live/restored `state` rather than initializing a
        /// new residual reference and iteration cursor.
        bool resume = false;
    };
    template <class Norms>
    struct BasicNonlinearIterationReport
    {
        bool converged = false;
        index iterations = 0, linearIterations = 0, rhsLimited = 0, incrementLimited = 0;
        index multigridCycles = 0;
        std::vector<index> levelCorrections;
        /// All component norms remain visible even for a selected-component
        /// stop. Reference may be a running maximum; initial never changes.
        Norms initialResidual{}, referenceResidual{}, residual{};
        real residualTargetRatio = 0;
        real minimumRHSAlpha = 1, minimumIncrementAlpha = 1, pseudoCFL = 0;
        /// Final equation evaluation's reconstruction, not an observation
        /// workspace or a sum/max over every inner reconstruction evaluation.
        int reconstructionIterations = 0;
        real reconstructionRelativeUpdate = 0;
        bool reconstructionUpdateMeasured = false;
        Linear::LinearIterationReport lastLinear;
    };
    using NonlinearIterationReport = BasicNonlinearIterationReport<std::vector<real>>;
    /// CPU whole-equation stage adapter over the shared nonlinear iteration
    /// policy, equation FV and coupled linear driver. The explicit residual
    /// selection is resolved by the equation frontend; it cannot silently use
    /// the historical flow5 defaults for a larger system. Every component's
    /// norm is reported. Storage grouping/layout may differ between operands.
    ///
    /// The equation profile matches Workspace: no Roe fallback.
    /// Polynomial FAS uses CorrectOnce as its bounded smoother. A capped finite solve
    /// remains unconverged and leaves acceptance to the time orchestrator.
    /// Callers serialize bound state/workspace/engine access, coordinate all
    /// domain ranks and keep owners on the setup through worker joins.
    /// Move-only session; explicit pMG aliases share exclusion tickets.
    /// Each execution call locks the bound stream's actual runtime and rejects
    /// closed/expired execution. No EagerEngine object address is retained.
    /// Optional local-phase failures follow the flow5 adapter's same policy:
    /// off mode aborts; candidate acceptance and solve-entry remain coordinated.
    class NonlinearSolver
    {
        struct Data;
        std::shared_ptr<Data> _data;
        std::weak_ptr<const Runtime::SessionAccess::Ticket> _nestedSession;
        explicit NonlinearSolver(std::shared_ptr<Data> data) : _data(std::move(data)) {}
        std::pair<index, real> ApplyCorrectionImpl(EquationDofs &state, const EquationDofs &increment, real scale, const std::vector<index> &components);
        NonlinearIterationReport Execute(EquationDofs &state, const EquationDofs &source,
                                         real mass, real rhsScale, real time, int stage,
                                         NonlinearSolveControl control, bool singleCorrection,
                                         OptionalRef<const EquationDofs> cachedRate,
                                         OptionalRef<const Runtime::Array<real>> sourceDerivative = {},
                                         OptionalRef<const NonlinearSolveSetup> setup = {});
        NonlinearSession Start(EquationDofs &state, const EquationDofs &source,
                               real mass, real rhsScale, real time, int stage,
                               NonlinearSolveControl control,
                               const NonlinearSolveSetup &setup);

    public:
        NonlinearSolver(const NonlinearSolver &) = delete;
        NonlinearSolver &operator=(const NonlinearSolver &) = delete;
        NonlinearSolver(NonlinearSolver &&) noexcept = default;
        NonlinearSolver &operator=(NonlinearSolver &&) noexcept = default;
        /// Explicit internal hierarchy alias; ordinary session copies are forbidden.
        NonlinearSolver RetainForMultigrid() const { return NonlinearSolver(_data); }
        Runtime::SessionAccess::Lease ReserveForMultigrid();
        NonlinearSolver(const FiniteVolume &discrete, EquationDofs &state,
                        std::shared_ptr<FiniteVolume::ReconstructionState> reconstruction,
                        std::shared_ptr<FiniteVolume::Workspace> workspace,
                        Runtime::EagerEngine &engine, Runtime::ExecutionStream stream, const DomainSettings &settings,
                        const Algebra::ComponentSelection &residualSelection,
                        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt,
                        std::shared_ptr<const real> pressureGauge = {});
        /// Solver-orchestration constructor. Retains the complete domain
        /// execution owner so stream/runtime/worker resources outlive every
        /// operation record and any hierarchy alias.
        NonlinearSolver(const FiniteVolume &discrete, EquationDofs &state,
                        std::shared_ptr<FiniteVolume::ReconstructionState> reconstruction,
                        std::shared_ptr<FiniteVolume::Workspace> workspace,
                        DomainExecutionContext context, const DomainSettings &settings,
                        const Algebra::ComponentSelection &residualSelection,
                        std::shared_ptr<const real> pressureGauge = {});
        /// Finest domain's fixed pressure target, reusable on same-volume pMG levels.
        std::shared_ptr<const real> PressureGauge() const;
        NonlinearIterationReport Solve(EquationDofs &state, const EquationDofs &source, real mass, real rhsScale, real time,
                                       int stage = 0, NonlinearSolveControl control = {});
        /// Prepare one externally phased finest nonlinear solve. The returned
        /// session retains this solver's exclusion ticket and all borrowed
        /// numerical owners until coordinated completion.
        NonlinearSession PrepareSolve(
            const NonlinearSolveSetup &setup, EquationDofs &state,
            const EquationDofs &source, real mass, real rhsScale, real time,
            int stage = 0, NonlinearSolveControl control = {});
        /// One bounded correction retaining split allocation/layout and
        /// EOS/source/PP dispatch. No nested solve.
        NonlinearIterationReport CorrectOnce(EquationDofs &state, const EquationDofs &source, real mass, real rhsScale, real time,
                                             int stage = 0, NonlinearSolveControl control = {},
                                             OptionalRef<const EquationDofs> cachedRate = {},
                                             OptionalRef<const Runtime::Array<real>> sourceDerivative = {});
        /// Evaluate dU/dt with aux preparation before reconstruction/RHS and
        /// retain that publication for a subsequent cached CorrectOnce.
        /// Pointwise volume normalization writes owned values only. State/rate
        /// must be distinct; caller coordinates state/time/cache usage on ranks.
        void EvaluateRate(EquationDofs &state, EquationDofs &rate, real time, OptionalRef<Runtime::Array<real>> derivative = {});
        /// Allocate a caller-owned integrated quadrature source derivative for one retained rate.
        std::optional<Runtime::Array<real>> BuildSourceDerivative(const EquationDofs &state) const;
        /// Observe rank-local volume-weighted L1/L2-squared contributions and
        /// final volume for a method's stopping residual. Optional reference
        /// subtracts a same-space rate for FAS. Uses the solver's managed
        /// backend operator; domain MPI reduction remains with the controller.
        std::vector<real> MeasureRateNorm(const EquationDofs &rate, int normOrder,
                                          OptionalRef<const EquationDofs> reference = {}, OptionalRef<const EquationDofs> rawACMRate = {}) const;
        /// Frozen auxiliary publication from the last rate/operator evaluation.
        /// A subsequent cached correction preserves it; after a mean update it
        /// describes the previous linearization, not the updated state. Read-only
        /// diagnostics may inspect its buffers/generations after joining work.
        AuxiliaryValues PreparedAuxValues() const;
        /// Same normalized stage operator as the field adapter, without
        /// changing owned means. Polynomial FAS uses this same operator.
        /// Optional rate follows the field adapter's distinct-output contract.
        void EvaluateEquation(EquationDofs &state, EquationDofs &value, real mass, real rhsScale, real time,
                              OptionalRef<EquationDofs> rate = {}, OptionalRef<Runtime::Array<real>> derivative = {});
        /// Caller agrees scale/selection and releases conflicting views before
        /// entry. Candidate EOS/raw-species checks precede any state mutation.
        /// With PP enabled, use Euler relaxed/common increment compression,
        /// final energy repair and transported mean-species repair. A repair
        /// which would alter a frozen component rejects instead of unfreezing.
        /// RANS always applies its Euler exponential scalar update and final
        /// model bounds after the optional common compression. Old RANS means
        /// must be admissible, and nonfinite corrections are never repaired.
        std::pair<index, real> ApplyCorrection(EquationDofs &state, const EquationDofs &correction, real scale = 1,
                                               const std::vector<index> &updateComponents = {});
        std::pair<int, real> ReconstructionStats() const;
        const NonlinearSolveState &State() const;
        void RestoreState(const NonlinearSolveState &state);
        /// Bind the shared live setup to an outer nonlinear policy such as
        /// pMG, applying the same new-attempt versus resume semantics as Solve.
        void BindControllerState(NonlinearSolveControl &control);
        Json StateJson() const { return State().ToJson(); }
        bool ReconstructionConverged() const;
        real ReconstructionTolerance() const;
    };
}
