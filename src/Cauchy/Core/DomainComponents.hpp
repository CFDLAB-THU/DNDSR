#pragma once

#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Core/TimeStageOperators.hpp"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Implicit/NonlinearSolve.hpp"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/EquationTimeStepper.hpp"
#include "Cauchy/State/MeanStateCheck.hpp"
#include "Cauchy/State/StateDiagnostics.hpp"
#include "Cauchy/Discretization/FlowErrorOperator.hpp"

#include <array>
#include <limits>

namespace DNDS::Cauchy
{
    /// Long-lived physical/discretization definition of one domain. Mutable
    /// solution history and backend scratch deliberately do not live here.
    struct DomainProblem
    {
        std::string name;
        EquationSystem equation;
        FiniteVolume finiteVolume;
        std::optional<AdmissibilityPolicy> admissibility;

        DomainProblem(std::string name, ssp<Geom::UnstructuredMesh> mesh,
                      const DomainSettings &settings, const TimeSettings &time,
                      EquationSystem equation, Diagnostics::SetupProgress progress = {});
    };

    /// Restart-relevant mutable numerical state. Every member either appears
    /// in a checkpoint or defines the active trial/cursor from which an exact
    /// internal checkpoint is formed. No engine, stream or temporary workspace
    /// is retained here.
    struct DomainState
    {
        EquationDofs accepted, trial, residual, initial;
        /// Domain-owned pressure target shared with the stage and pMG solvers.
        /// Only checkpoint publication can change its value after setup.
        std::shared_ptr<real> pressureGauge;
        std::shared_ptr<FiniteVolume::ReconstructionState> reconstruction;
        std::optional<EquationDofs> reconstructionBackup;
        std::optional<Runtime::Array<real>> limitedVRBackup;
        std::unique_ptr<EquationTimeStepper> timeStepper;
        std::shared_ptr<NonlinearSolveState> coupledNonlinear = std::make_shared<NonlinearSolveState>();
        bool restoreReconstruction = false;
        StateEpoch epoch, trialEpoch;
        uint64_t nextTrial = 0;
        real time = 0, trialTime = 0, trialDt = 0, lastAcceptedDt = 0,
             progressTime = 0;
        bool trialReady = false, internalResume = false;
        std::optional<EquationTimeStepper::StageGroup> stageGroup;
        int coordinatedStage = 0;
        bool coordinatedPrepared = false;

        DomainState(const DomainProblem &problem, const DomainSettings &settings,
                    const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers,
                    const std::vector<real> &initialConservative,
                    Diagnostics::SetupProgress progress = {});
    };

    /// Small host-side description of the currently prepared mathematical
    /// group. Serialized DIRK stages have one member; DITR has two. No field
    /// ownership or mutable method scratch escapes through this metadata.
    struct DomainStageGroup
    {
        int stage = 0;
        int coupledStages = 1;
        bool implicit = false;
        std::array<real, 2> times{};
    };

    /// Two-stage nonlinear operator bundle consumed by the registered DITR
    /// update policy. Durable midpoint/end reconstruction state remains owned
    /// by Domain/time-method storage; these callables are synchronous borrows.
    struct DomainCoupledStageOperator
    {
        EquationTimeStepper::Evaluate evaluate;
        std::function<NonlinearIterationReport(
            const ODE::CoupledStageEquation<EquationDofs> &,
            const EquationDofs &, real)>
            correct;
        std::function<std::pair<int, real>()> reconstruction;
        std::function<void(const BoundaryInputs &)> bindBoundary;
        std::shared_ptr<FiniteVolume::ReconstructionState> state;
    };

    /// Shared packet-handle publication, retained by queued receiver bindings.
    /// Updates are serialized on the domain compute lane or at a quiescent
    /// controller boundary. Numerical payloads remain owned by BoundaryInputs.
    struct DomainInterfaceState
    {
        std::optional<BoundaryInputs> active;
        std::array<std::optional<BoundaryInputs>, 2> coupled;
    };

    /// Rebuildable backend-resolved execution resources. This owner contains
    /// no accepted numerical state. Its context handle is declared last and is
    /// released first; solver-held copies keep that same runtime alive through
    /// solver teardown. Their final local context release drains the engine
    /// before workspace fields and, subsequently, DomainState/Problem vanish.
    /// Managed operation records retain buffers required during that drain.
    struct DomainExecution
    {
        std::shared_ptr<FiniteVolume::Workspace> workspace,
            observationWorkspace;
        std::shared_ptr<FiniteVolume::ReconstructionState>
            observationReconstruction;
        std::optional<EquationDofs> observationInput, observationState;
        std::unique_ptr<NonlinearSolver> implicit;
        std::unique_ptr<PolynomialMultigrid> multigrid;
        /// Lazy backend reductions/explicit arithmetic; unused blend and CFL
        /// scratch families remain unallocated for the selected time policy.
        mutable std::optional<TimeStageOperators> timeStages;
        mutable std::optional<MeanStateCheck> meanCheck;
        mutable std::optional<StateDiagnostics> stateDiagnostics;
        mutable std::optional<FlowErrorOperator> flowError;
        mutable std::optional<AuxiliaryValues> timeAuxiliary;
        mutable std::optional<MeanGradients> timeMeanGradients;
        mutable std::vector<Runtime::TransferBinding> timeMeanHalo;
        std::shared_ptr<FiniteVolume::MeanWorkspace> timePPWorkspace;
        std::optional<EquationDofs> timePPResidual;
        std::shared_ptr<DomainInterfaceState> interfaces =
            std::make_shared<DomainInterfaceState>();
        std::function<DomainCoupledStageOperator(EquationDofs &)>
            coupledFactory;
        Runtime::ArrayStorage::Shared coupledMiddleOwner;
        uint64_t coupledPreparedTrial = std::numeric_limits<uint64_t>::max();
        std::optional<EquationDofs> coupledResidual;
        std::unique_ptr<NonlinearSolveSetup> nonlinearSetup;
        DomainExecutionContext context;

        /// Construction-only placement happens after host setup/initialization
        /// and before workspace binding. Thereafter problem/state bindings are
        /// stable; no numerical-loop migration or host fallback is permitted.
        DomainExecution(DomainProblem &problem, DomainState &state,
                        const ExecutionSettings &settings,
                        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                        Diagnostics::SetupProgress progress = {});

        Runtime::EagerEngine &Engine() const { return context.Engine(); }
        DomainExecutionContext &Context() { return context; }
        const DomainExecutionContext &Context() const { return context; }
        const Runtime::ExecutionStream &ComputeStream() const
        {
            return context.ComputeStream();
        }
        const Runtime::ExecutionStream &CommunicationStream() const
        {
            return context.CommunicationStream();
        }
        const std::optional<Physics::Chemistry::ChemicalWorkerPool> &
        ChemicalWorkers() const
        {
            return context.ChemicalWorkers();
        }
        void Synchronize() const { context.Synchronize(); }
    };
}
