#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Core/StepResult.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/State/CPU/StatePropertiesKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Input/StateInput.hpp"
#include "Cauchy/Operators/Source/ChemicalSource.hpp"
#include "Cauchy/Implicit/NonlinearOperators.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        DomainSettings SpatialSettings(const DomainSettings &settings, const TimeSettings &time)
        {
            time.ValidateSourceIntegration();
            auto result = settings;
            if (time.sourceSplitting == "strang" && settings.equation.source == "chemical")
                result.equation.source = "none";
            return result;
        }
        PhysicsResources ResolveEquation(const std::string &name, const DomainSettings &settings, const MPIInfo &mpi)
        {
            std::optional<PhysicsResources> setup;
            MPI::LocalStage(mpi, [&]
                            {
                ResolvedExecutionPlan::Resolve({{name, {0}}, settings});
                // Each numerical OMP worker needs an exclusive thermochemical
                // context. Ordered engine lanes reuse this bounded team rather
                // than cloning again for every engine thread or point.
                setup.emplace(PhysicsResources::Resolve(settings.equation, OMP::DistributedThreads()));
                index width = 0;
                for (const auto &block : setup->equation.Schema().blocks) width += index(block.components.size());
                settings.implicit.ValidateComponents(width);
                setup->equation.Schema().ResolveComponents(settings.implicit.ResolveResidualComponents(setup->equation.IsACM())); });
            return std::move(*setup);
        }
        ResolvedPhysicalInputs ResolveInputs(const DomainSettings &settings, const PhysicsResources &setup, const MPIInfo &mpi)
        {
            std::optional<ResolvedPhysicalInputs> result;
            MPI::LocalStage(mpi, [&]
                            {
                settings.execution.ValidateMPIThreadSupport(MPI::GetMPIThreadLevel());
                result = ResolvePhysicalStateInputs(settings, setup.equation, setup.workers); });
            return std::move(*result);
        }
    }
    Domain::Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time,
                   Diagnostics::SetupProgress progress)
        : Domain(name, mesh, settings, time, ResolveEquation(name, settings, mesh->getMPI()), std::move(progress)) {}
    Domain::Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time, PhysicsResources setup,
                   Diagnostics::SetupProgress progress)
        : Domain(name, mesh, ResolveInputs(settings, setup, mesh->getMPI()), time, setup, std::move(progress)) {}
    Domain::Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const ResolvedPhysicalInputs &inputs, const TimeSettings &time, PhysicsResources setup,
                   Diagnostics::SetupProgress progress)
        : _problem(std::move(name), mesh, inputs.settings, time,
                   std::move(setup.equation), progress),
          _state(_problem, inputs.settings, setup.workers,
                 inputs.initialConservative, progress),
          _execution(_problem, _state, inputs.settings.execution,
                     std::move(setup.workers), progress)
    {
        auto setupStage = progress.Stage("implicit/setup");
        // All spatial operators, including coarse FAS and approximate
        // Jacobians, see the same source-free transport profile when splitting.
        // Physical config/material metadata still describe the chemical PDE.
        const auto settings = SpatialSettings(inputs.settings, time);
        _sourceStrang = time.sourceSplitting == "strang" && inputs.settings.equation.source == "chemical";
        _reactorControls = time.reactor;
        _positivity = settings.numerics.positivity;
        _state.accepted.RequireSchema(_problem.equation.Schema());
        DNDS_check_throw_info(_problem.finiteVolume.Metrics().Dimension() == settings.mesh.dimension, "Cauchy config/mesh dimension differs");
        _problem.admissibility.emplace(_problem.finiteVolume.PrepareAdmissibility(_execution.ChemicalWorkers()));
        if (_problem.equation.IsACM() && !_problem.finiteVolume.HasPressureBoundary())
        {
            auto &engine = _execution.Engine();
            const auto &lane = _execution.ComputeStream();
            const NonlinearOperators operators(_problem.finiteVolume.Metrics(), *_problem.admissibility, lane.Context());
            const auto requested = settings.equation.acmPressureMean;
            _state.pressureGauge = std::make_shared<real>(requested ? *requested : operators.PressureMean(engine, lane, _state.accepted));
            if (requested)
            {
                operators.ProjectPressureMean(engine, lane, _state.accepted, *requested);
                EquationDofOperations::Copy(engine, lane, _state.trial, _state.accepted);
            }
        }
        DNDS_check_throw_info(Admissible(_state.accepted), "Cauchy initial means violate density or minimum-temperature internal-energy bounds");
        ConfigureTimeStep(time, settings.implicit.positivity.compressionPower);
        InitializeImplicit(settings, time, mesh);
    }
    bool Domain::Admissible(const EquationDofs &state) const
    {
        try
        {
            DNDS_assert(_problem.admissibility);
            int valid = 0;
            const MPIInfo mpi(Communicator());
            MPI::LocalStage(mpi, [&]
                            {
                if (!_execution.meanCheck)
                    _execution.meanCheck.emplace(_problem.finiteVolume.Metrics().Space(), *_problem.admissibility,
                                                 _execution.ComputeStream().Context());
                valid = bool(_execution.meanCheck->Evaluate(_execution.Engine(), _execution.ComputeStream(), state)); });
            return MPI::AllreduceValue<MPI::Reduction::Min>(valid, mpi.comm);
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy domain operation failed");
            throw;
        }
    }
    bool Domain::RefreshDerived(DerivedFields &fields, bool onlyIfStale) const
    {
        const auto &workers = _execution.ChemicalWorkers();
        return workers ? RefreshDerivedFields(_state.accepted, _problem.equation, fields, workers->Acquire(), onlyIfStale)
                       : RefreshDerivedFields(_state.accepted, _problem.equation, fields, onlyIfStale);
    }
    real Domain::StableStep(real cfl) const
    {
        AuxiliaryInputs inputs;
        if (_execution.interfaces->active)
            inputs.boundary.Emplace(*_execution.interfaces->active);
        if (!_execution.timeAuxiliary)
        {
            AuxiliaryRequest request;
            request.meanThermodynamics = request.cellSpectralSum = true;
            std::optional<AuxiliaryValues> prepared;
            MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                            { prepared.emplace(_problem.finiteVolume.BuildAuxiliaryValues(request)); });
            std::vector<R::TransferBinding> halos;
            for (index iField = 0; iField < _state.accepted.PhysicalFieldNumber(); ++iField)
            {
                auto input = _state.accepted.PhysicalField(iField);
                halos.push_back(_problem.finiteVolume.BindHalo(input));
            }
            _execution.timeMeanHalo = std::move(halos);
            _execution.timeAuxiliary = std::move(prepared);
        }
        if (_problem.equation.HasRANS() && _problem.equation.RANS().ComponentNumber() == 2)
        {
            auto state = _state.accepted.NumericAlias();
            if (!_execution.timeMeanGradients)
                _execution.timeMeanGradients.emplace(_problem.finiteVolume.BuildMeanGradients(state));
            _problem.finiteVolume.PrepareAuxValues(state, *_execution.timeAuxiliary, *_execution.timeMeanGradients,
                                                   _execution.Engine(), _execution.ComputeStream(), _execution.CommunicationStream(), _state.time,
                                                   _execution.ChemicalWorkers(), inputs);
        }
        else
        {
            OperatorSchedule schedule(_execution.Engine(), _execution.ComputeStream(), _execution.CommunicationStream(),
                                      _problem.finiteVolume.Settings().execution.overlap);
            auto transfers = schedule.BeginAfterCompute(_execution.timeMeanHalo);
            auto preparation = _problem.finiteVolume.BeginAuxValues(_state.accepted, *_execution.timeAuxiliary,
                                                                    _state.time, _execution.ChemicalWorkers(), inputs);
            if (schedule.RunsCoreSeparately())
                preparation.Apply(_execution.Engine(), _execution.ComputeStream(), CFV::ReconstructionRegion::Core);
            schedule.Await(transfers);
            preparation.Apply(_execution.Engine(), _execution.ComputeStream(), schedule.DependentRegion());
            schedule.Join();
            preparation.Finish(*_execution.timeAuxiliary);
        }
        real local = 0;
        MPI::LocalStage(MPIInfo(Communicator()), [&]
                        {
            _problem.finiteVolume.ValidateAuxValues(_state.accepted, *_execution.timeAuxiliary, _state.time);
            DNDS_check_throw_info(_execution.timeAuxiliary->cellSpectralSum, "CFL selection requires cell spectral sums");
            if (!_execution.timeStages)
                _execution.timeStages.emplace(_problem.finiteVolume.Metrics(), _execution.ComputeStream().Context());
            local = _execution.timeStages->MinimumStableStep(_execution.Engine(), _execution.ComputeStream(),
                _execution.timeAuxiliary->cellSpectralSum->Owned(), cfl); });
        return MPI::AllreduceValue<MPI::Reduction::Min>(local, Communicator());
    }
    void Domain::BeginTrial(real dt)
    {
        _state.trialReady = false;
        int valid = std::isfinite(dt) && dt > 0 && (_steady || (std::isfinite(_state.time + dt) && _state.time + dt > _state.time));
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, _problem.finiteVolume.Metrics().Communicator());
        DNDS_check_throw_info(valid, "invalid Cauchy physical timestep");
        real min = dt, max = dt;
        min = MPI::AllreduceValue<MPI::Reduction::Min>(min, _problem.finiteVolume.Metrics().Communicator());
        max = MPI::AllreduceValue<MPI::Reduction::Max>(max, _problem.finiteVolume.Metrics().Communicator());
        DNDS_check_throw_info(min == max, "Cauchy timestep differs between domain ranks");
        DNDS_check_throw_info(Admissible(_state.accepted), "accepted Cauchy mean is inadmissible");
        // Owned read scopes may coexist with a retained accepted-state reader.
        // Drain earlier engine work, then queue both copies on the same lane.
        // Trial initialization is an explicit publication boundary before the
        // method may take immediate reconstruction/history reservations.
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        {
            _execution.Synchronize();
            EquationDofOperations::Copy(_execution.Engine(), _execution.ComputeStream(), _state.trial, _state.accepted);
            EquationDofOperations::Copy(_execution.Engine(), _execution.ComputeStream(), _state.initial, _state.accepted);
            _execution.Synchronize(); });
        _lastFallbacks = 0;
        _lastAlpha = _lastBeta = 1;
        _lastAlphaCells = _lastBetaCells = 0;
        _lastCharacteristicFallbacks = 0;
        _state.trialEpoch = _state.epoch;
        _state.trialDt = _steady ? 0 : dt;
        _state.trialEpoch.trial = ++_state.nextTrial;
        // The polynomial guess is numerical state for explicit fixed-sweep VR
        // too. Preserve it before either time driver can overwrite the trial.
        PreserveReconstructionGuess();
        if (_sourceStrang)
        {
            ChemicalHalfStep(_state.accepted, _state.trial, .5 * dt);
            MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                            { EquationDofOperations::Copy(_execution.Engine(), _execution.ComputeStream(), _state.initial, _state.trial);
                              _execution.Synchronize(); });
            // Euler's physical split does not reuse multistep transport
            // history across source jumps. Accepted split history is empty.
            if (_state.timeStepper)
                _state.timeStepper->ResetHistory();
        }
    }
    void Domain::ChemicalHalfStep(const EquationDofs &input, EquationDofs &output, real dt)
    {
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        {
            _execution.Synchronize();
            const auto &workers = _execution.ChemicalWorkers();
            DNDS_check_throw_info(_problem.equation.HasMixture() && workers, "chemical source step requires its explicit mixture worker");
            AdvanceChemicalSource(_problem.equation, input, output, *workers, dt, _reactorControls); });
    }
    void Domain::FinishChemicalStep()
    {
        if (!_sourceStrang)
            return;
        // The transport method has finished using its initial state. Reuse
        // that allocation as a distinct source output, then copy into trial.
        ChemicalHalfStep(_state.trial, _state.initial, .5 * _state.trialDt);
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        { EquationDofOperations::Copy(_execution.Engine(), _execution.ComputeStream(), _state.trial, _state.initial);
                          _execution.Synchronize(); });
        DNDS_check_throw_info(Admissible(_state.trial), "chemical source candidate is inadmissible; accepted state unchanged");
    }
    StepResult Domain::PrepareStep(real dt)
    {
        DNDS_check_throw_info(!_problem.finiteVolume.HasInterfaces(),
                              "interface domains must advance through their Simulation");
        if (auto start = BeginStep(dt); !start)
            return start;
        while (HasNextGroup())
        {
            PrepareGroup();
            if (auto session = PrepareGroupSolve())
            {
                SolveNonlinear(*session);
                if (auto result = FinishGroupSolve(*session); !result)
                    return result;
            }
            else if (auto result = ApplyExplicitGroup(); !result)
                return result;
            if (auto result = CompleteGroup(); !result)
                return result;
        }
        return {};
    }
    int Domain::GroupNumber() const
    {
        return _state.timeStepper ? _state.timeStepper->GroupNumber() : 3;
    }
    void Domain::BindInterfaceInputs(const BoundaryInputs &input, int coupledStage)
    {
        PrepareInterfaceReceiver(coupledStage)(input);
    }
    StepResult Domain::BeginStep(real dt)
    {
        StepResult result;
        _execution.Context().ClearEvaluationDependency();
        if (_state.internalResume)
        {
            int valid = _steady || dt == _state.trialDt;
            valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, Communicator());
            DNDS_check_throw_info(valid && _state.timeStepper,
                                  "exact internal resume requires its stored physical dt");
            _state.coordinatedStage = _state.timeStepper->ActiveGroup().stage;
        }
        else
        {
            BeginTrial(dt);
            _state.coordinatedStage = 0;
            if (_state.timeStepper)
                _state.timeStepper->BeginStep(
                    _state.initial, _state.trial, _state.time, dt,
                    [this, &result](auto &state, auto &rate, real time, int stage)
                    {
                        EvaluateImplicitRate(state, rate, *_execution.workspace, time, stage);
                        // DITR retains the initial rate as a term in both
                        // implicit equations. DIRK checks its explicit stage
                        // through the ordinary CompleteGroup rate callback.
                        if (_algebraicRateComponent >= 0)
                            result = ValidateAlgebraicRate(rate, "initial state");
                    },
                    _steady);
        }
        _lastImplicit.clear();
        _lastCoupledIterations = 0;
        _lastCoupledTargetRatio = 0;
        _state.coordinatedPrepared = false;
        _state.stageGroup.reset();
        for (auto &input : _execution.interfaces->coupled)
            input.reset();
        return result;
    }
    void Domain::PrepareCoupledInterface(int coupledStage, bool reconstructed)
    {
        DNDS_check_throw_info(_state.stageGroup && _state.stageGroup->coupledStages == 2 &&
                                  (coupledStage == 0 || coupledStage == 1) &&
                                  _coupledMiddle,
                              "invalid coupled interface preparation");
        if (reconstructed)
            ActivateCoupledInterface(coupledStage);
        auto &state = _state.timeStepper->GroupState(
            *_state.stageGroup, _state.trial, coupledStage);
        auto &reconstruction = coupledStage == 0 ? *_coupledMiddle->state
                                                 : *_state.reconstruction;
        _problem.finiteVolume.Execute(
            state, reconstruction, *_execution.workspace,
            _execution.Context(),
            reconstructed
                ? FiniteVolumeEvaluation::Reconstruction(
                      _state.stageGroup->times[size_t(coupledStage)])
                : FiniteVolumeEvaluation::Sampling());
    }
    InterfaceDonorPublication
    Domain::PublishCoupledInterfaceDonor(int coupledStage, bool reconstructed) const
    {
        DNDS_check_throw_info(_state.stageGroup && _state.stageGroup->coupledStages == 2 &&
                                  (coupledStage == 0 || coupledStage == 1) &&
                                  _coupledMiddle,
                              "invalid coupled interface publication");
        const auto &state = _state.timeStepper->GroupState(
            *_state.stageGroup, _state.trial, coupledStage);
        auto &reconstruction = coupledStage == 0 ? *_coupledMiddle->state
                                                 : *_state.reconstruction;
        return _problem.finiteVolume.PublishInterfaceDonor(
            state, reconstruction, *_execution.workspace, reconstructed);
    }
    void Domain::ActivateCoupledInterface(int coupledStage)
    {
        DNDS_check_throw_info(coupledStage == 0 || coupledStage == 1,
                              "invalid coupled interface activation");
        const auto &input =
            _execution.interfaces->coupled[size_t(coupledStage)];
        if (_problem.finiteVolume.HasInterfaces())
            DNDS_check_throw_info(input,
                                  "coupled stage has no frozen interface packet");
        if (!input)
            return;
        BindInterfaceInputs(*input);
        if (coupledStage == 0 && _coupledMiddle &&
            _coupledMiddle->bindBoundary)
            _coupledMiddle->bindBoundary(*input);
    }
    DomainStageGroup Domain::PrepareGroup()
    {
        _execution.Context().ClearEvaluationDependency();
        DNDS_check_throw_info(
            _state.coordinatedStage < GroupNumber() &&
                !_state.coordinatedPrepared && !_state.trialReady,
            "domain group is out of order");
        _state.coordinatedPrepared = true;
        DomainStageGroup group;
        group.stage = _state.coordinatedStage;
        _state.trialEpoch.stage = group.stage;
        _state.trialEpoch.coupledStage = 0;
        if (_state.timeStepper)
        {
            _state.stageGroup.emplace(
                _state.internalResume ? _state.timeStepper->ActiveGroup()
                                      : _state.timeStepper->PrepareGroup(_state.trial));
            const auto &prepared = *_state.stageGroup;
            group.coupledStages = prepared.coupledStages;
            group.implicit = prepared.implicit;
            group.times = prepared.times;
            PrepareGroupOperators(_state.internalResume);
        }
        else
        {
            static constexpr real times[]{0, 1, .5};
            group.times[0] = _state.time + times[group.stage] * _state.trialDt;
        }
        return group;
    }

    std::optional<NonlinearSession> Domain::PrepareGroupSolve()
    {
        DNDS_check_throw_info(_state.coordinatedPrepared,
                              "domain group was not prepared");
        if (!_state.stageGroup || !_state.stageGroup->implicit)
            return std::nullopt;
        if (_state.stageGroup->coupledStages == 2)
            return CreateCoupledNonlinearSession(_state.internalResume);
        const auto equation = _state.timeStepper->PrepareEquation(
            *_state.stageGroup, _state.trial, 0);
        const auto &coefficients = equation.coefficients;
        _state.progressTime = coefficients.time;
        SetTimeStage(_state.stageGroup->stage);
        auto control = ProgressControl();
        control.resume = _state.internalResume;
        return _execution.multigrid
                   ? _execution.multigrid->PrepareSolve(
                         *_execution.nonlinearSetup, equation.resOther,
                         coefficients.inverseDt, coefficients.rhsScale,
                         coefficients.time, _state.stageGroup->stage,
                         std::move(control))
                   : _execution.implicit->PrepareSolve(
                         *_execution.nonlinearSetup, equation.state,
                         equation.resOther, coefficients.inverseDt,
                         coefficients.rhsScale, coefficients.time,
                         _state.stageGroup->stage, std::move(control));
    }

    StepResult Domain::FinishGroupSolve(NonlinearSession &session)
    {
        DNDS_check_throw_info(_state.stageGroup && session.Complete(),
                              "domain nonlinear group is incomplete");
        _state.timeStepper->ValidateGroup(*_state.stageGroup, _state.trial);
        if (_state.stageGroup->coupledStages == 2)
        {
            _lastCoupledIterations = session.Report().iterations;
            _lastCoupledTargetRatio = session.Report().residualTargetRatio;
        }
        if (!RecordImplicitStage(session.Report()))
        {
            // Failure-only diagnostics survive the common stage-group entry.
            // Numerical rejection leaves accepted state/history untouched;
            // reporting requires no extra reduction or exception recovery.
            const auto &last = session.Report();
            const Json details{
                {"method", _state.timeStepper->Method()}, {"steady", _steady}, {"stage", _state.stageGroup->stage}, {"dt", _steady ? 0 : _state.trialDt}, {"coupled_iterations", _lastCoupledIterations}, {"coupled_target_ratio", _lastCoupledTargetRatio}, {"last_solve_converged", last.converged}, {"nonlinear_iterations", last.iterations}, {"linear_iterations", last.linearIterations}, {"pseudo_cfl", last.pseudoCFL}, {"initial_residual", last.initialResidual}, {"reference_residual", last.referenceResidual}, {"residual_target_ratio", last.residualTargetRatio}, {"last_residual", last.residual}, {"vr_sweeps", last.reconstructionIterations}, {"vr_relative_update", last.reconstructionRelativeUpdate}};
            return StepResult::Rejected(
                "implicit stage group reached its correction cap; accepted state/history unchanged; " + details.dump());
        }
        return {};
    }

    StepResult Domain::ApplyExplicitGroup()
    {
        DNDS_check_throw_info(_state.coordinatedPrepared &&
                                  (!_state.stageGroup || !_state.stageGroup->implicit),
                              "domain group is not explicit");
        if (_state.stageGroup)
            return {}; // The DIRK method prepared this explicit stage's state.
        static constexpr real history[]{0, .75, 1. / 3};
        static constexpr real weight[]{1, .25, 2. / 3}, times[]{0, 1, .5};
        const int stage = _state.coordinatedStage;
        return UpdateStage(_state.trialDt, history[stage], weight[stage],
                           _state.time + times[stage] * _state.trialDt);
    }

    StepResult Domain::CompleteGroup()
    {
        StepResult result;
        DNDS_check_throw_info(_state.coordinatedPrepared,
                              "no prepared domain group to complete");
        // The coupled stopping check already measured both rates in a fresh
        // converged DITR solve. A resumed terminal cursor or an explicitly
        // accepted cap must verify the immutable rates before publication.
        if (_algebraicRateComponent >= 0 && _state.stageGroup->coupledStages == 2 &&
            (_state.internalResume || !_lastImplicit.back().converged))
            for (int stage = 0; stage < 2; ++stage)
            {
                const auto &rate = _state.timeStepper->CompletedCoupledRate(*_state.stageGroup, stage);
                if (auto checked = ValidateAlgebraicRate(rate, stage == 0 ? "DITR midpoint" : "DITR endpoint"); !checked)
                    return checked;
            }
        if (_state.timeStepper)
            _state.timeStepper->CompleteGroup(
                *_state.stageGroup, _state.trial,
                [this, &result](auto &state, auto &rate, real time, int stage)
                {
                    EvaluateImplicitRate(state, rate, *_execution.workspace, time, stage);
                    if (_algebraicRateComponent >= 0)
                        result = ValidateAlgebraicRate(rate, stage == 0 ? "explicit initial stage" : "completed implicit stage");
                });
        if (!result)
            return result;
        _state.coordinatedPrepared = false;
        _state.internalResume = false;
        _state.stageGroup.reset();
        if (++_state.coordinatedStage == GroupNumber())
        {
            FinishChemicalStep();
            if (!Admissible(_state.trial))
                return StepResult::Rejected("coordinated candidate is inadmissible");
            _state.trialTime = _steady ? _state.time : _state.time + _state.trialDt;
            _state.trialReady = true;
        }
        return {};
    }
    void Domain::CommitStep()
    {
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        { DNDS_check_throw_info(_state.trialReady && (!_state.timeStepper || _state.timeStepper->HasPendingTrial()), "no prepared Cauchy candidate"); });
        auto epoch = _state.trialEpoch;
        ++epoch.acceptedStep;
        epoch.stage = 0;
        epoch.coupledStage = 0;
        PublishTrial(_state.trialTime, epoch);
        if (_state.timeStepper)
        {
            _state.timeStepper->Commit();
            if (_sourceStrang)
                _state.timeStepper->ResetHistory();
        }
        _state.lastAcceptedDt = _state.trialDt;
    }
    void Domain::CommitReserved(EquationDofs::WriteScope &destination, EquationDofs::ReadScope &source)
    {
        // Simulation reserved every domain's destination/source and agreed
        // readiness globally before entering this publication. No state-field
        // allocation occurs here; backend launch/descriptor bookkeeping may.
        EquationDofOperations::CopyReserved(_execution.Engine(), _execution.ComputeStream(),
                                            _state.accepted, _state.trial, destination, source);
        _state.time = _state.trialTime;
        _state.epoch = _state.trialEpoch;
        ++_state.epoch.acceptedStep;
        _state.epoch.stage = 0;
        _state.epoch.coupledStage = 0;
        _state.trialReady = _state.restoreReconstruction = false;
        if (_state.timeStepper)
        {
            _state.timeStepper->Commit();
            if (_sourceStrang)
                _state.timeStepper->ResetHistory();
        }
        _state.lastAcceptedDt = _state.trialDt;
    }
    void Domain::PublishTrial(real time, StateEpoch epoch)
    {
        // Agree local immediate-access availability before any accepted values
        // change. A retained debug/Python view must not allow a partial commit
        // on only the other ranks. Keep this lease through the actual copy.
        std::optional<EquationDofs::WriteScope> destination;
        std::optional<EquationDofs::ReadScope> source;
        _execution.Synchronize();
        const auto comm = _problem.finiteVolume.Metrics().Communicator();
        MPIInfo mpi(comm);
        try
        {
            MPI::LocalStage(mpi, [&]
                            { destination.emplace(_state.accepted.WriteOwned(_execution.ComputeStream().Context()));
                              source.emplace(_state.trial.ReadOwned(_execution.ComputeStream().Context())); });
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy domain operation failed");
            // Successful ranks acquired but never used their destination.
            // Cancel before unwinding; a normal exceptional write destructor
            // deliberately poisons possibly partial writes.
            if (destination)
                destination->CancelUnmodified();
            throw;
        }
        EquationDofOperations::CopyReserved(_execution.Engine(), _execution.ComputeStream(),
                                            _state.accepted, _state.trial, *destination, *source);
        _state.time = time;
        _state.epoch = epoch;
        _state.trialReady = false;
        _state.restoreReconstruction = false;
    }
    const EquationDofs &Domain::Residual()
    {
        _state.trialReady = false;
        DNDS_check_throw_info(Admissible(_state.accepted), "accepted Cauchy mean is inadmissible");
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        {
            _execution.Synchronize();
            EquationDofOperations::Copy(_execution.Engine(), _execution.ComputeStream(), _state.trial, _state.accepted);
            _execution.Synchronize(); });
        _problem.finiteVolume.Execute(_state.trial, *_state.reconstruction, *_execution.workspace, _execution.Context(),
                                      FiniteVolumeEvaluation::Spatial(_state.residual, _state.time));
        _lastFallbacks = _problem.finiteVolume.LastFaceFallbacks(*_execution.workspace);
        _lastCharacteristicFallbacks = _problem.finiteVolume.LastCharacteristicFallbacks(*_execution.workspace);
        return _state.residual;
    }
    std::array<real, 14> Domain::Diagnostics(const DomainSettings &settings, const std::vector<int> &forceZones)
    {
        const auto snapshot = CaptureStateSnapshot(true);
        EvaluateObservation(snapshot);
        std::array<real, 14> result{};
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        {
            if (!_execution.stateDiagnostics)
                _execution.stateDiagnostics.emplace(_problem.finiteVolume.Metrics(), _problem.equation,
                    _execution.ComputeStream().Context(), _execution.ChemicalWorkers());
            result = _execution.stateDiagnostics->Evaluate(_execution.Engine(), _execution.ComputeStream(),
                snapshot.Means(), *_execution.observationState); });
        const auto comm = _problem.finiteVolume.Metrics().Communicator();
        MPI::AllreduceInPlace<MPI::Reduction::Sum>(
            MPI::ContiguousView<real>{result.data(), 2}, comm);
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<real>{result.data() + 2, 2}, comm);
        result[4] = MPI::AllreduceValue<MPI::Reduction::Sum>(result[4], comm);
        if (_problem.equation.IsACM())
        {
            MPI::AllreduceInPlace<MPI::Reduction::Sum>(MPI::ContiguousView<real>{result.data() + 10, 2}, comm);
            result[10] = std::sqrt(result[10] / result[0]);
            result[11] /= result[0];
        }
        else
            result[13] = MPI::AllreduceValue<MPI::Reduction::Min>(result[13], comm);
        // Report the volume-integrated L2 defect, matching Euler's res0
        // convention rather than a volume-averaged residual.
        result[4] = std::sqrt(result[4]);
        const auto force = _problem.finiteVolume.LastBoundaryForce(*_execution.observationWorkspace, forceZones, _execution.ComputeStream());
        for (int d = 0; d < 3; ++d)
            result[5 + d] = force[d];
        result[8] = _problem.finiteVolume.LastReconstructionIterations(*_execution.observationWorkspace);
        result[9] = _problem.finiteVolume.LastReconstructionUpdate(*_execution.observationWorkspace);
        if (!_problem.equation.IsACM() && !_problem.equation.HasMixture() && !_problem.equation.HasRANS() && settings.initialProfile.kind == "isentropicVortex")
        {
            MPI::LocalStage(mpi, [&]
                            {
                if (!_execution.flowError)
                    _execution.flowError.emplace(_problem.finiteVolume, _execution.ComputeStream().Context()); });
            const auto error = _execution.flowError->Evaluate(_execution.Engine(), _execution.ComputeStream(),
                                                              snapshot.Means(), settings.initialProfile, settings.initialPrimitive, snapshot.Time());
            for (int i = 0; i < 3; ++i)
                result[10 + i] = error[i][0];
        }
        return result;
    }
    FiniteVolume::WorkspaceOptions Domain::ObservationWorkspaceOptions() const
    {
        return FiniteVolume::WorkspaceOptions(FiniteVolume::RHSKernelPolicy::Auto, _problem.finiteVolume.Settings().boundaryFacialFluxes);
    }
    void Domain::WriteVisualization(const std::string &prefix, int hdfDeflateLevel)
    {
        PrepareVisualization(hdfDeflateLevel).Write(prefix, _problem.finiteVolume.Metrics().Communication());
    }
    PreparedVisualization Domain::PrepareVisualization(int hdfDeflateLevel)
    {
        const auto snapshot = CaptureStateSnapshot(true);
        EvaluateObservation(snapshot);
        return _problem.finiteVolume.PrepareVisualization(*_execution.observationInput, *_execution.observationReconstruction, *_execution.observationWorkspace, snapshot.Time(), hdfDeflateLevel, _execution.ChemicalWorkers());
    }
    void Domain::WriteInternalVisualization(const std::string &prefix, int hdfDeflateLevel)
    {
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        { DNDS_check_throw_info(_state.timeStepper && _state.trialEpoch.trial &&
                                                    (!_problem.finiteVolume.PolynomialReconstruction() || _state.restoreReconstruction) &&
                                                    std::isfinite(_state.progressTime),
                                                "Cauchy internal visualization has no active implicit trial"); });
        if (!_execution.observationWorkspace)
        {
            MPI::LocalStage(mpi, [&]
                            {
                _execution.observationInput.emplace(EquationDofOperations::NewVector(_state.trial));
                _execution.observationState.emplace(EquationDofOperations::NewVector(_state.trial));
                _execution.observationInput->Fields().SetName(_problem.name + "/observation/means", DNDS_ALLOCATION_SITE());
                _execution.observationState->Fields().SetName(_problem.name + "/observation/residual", DNDS_ALLOCATION_SITE()); });
            _execution.observationWorkspace = _problem.finiteVolume.BuildWorkspace(*_execution.observationInput, _execution.ChemicalWorkers(), ObservationWorkspaceOptions());
            _execution.observationReconstruction = _problem.finiteVolume.BuildReconstructionState(*_execution.observationInput);
            if (_execution.interfaces->active)
                _problem.finiteVolume.SetBoundaryInputs(*_execution.observationWorkspace, *_execution.interfaces->active);
        }
        MPI::LocalStage(mpi, [&]
                        {
            _execution.Synchronize();
            EquationDofOperations::Copy(_execution.Engine(), _execution.ComputeStream(), *_execution.observationInput, _state.trial);
            _execution.Synchronize(); });
        if (_problem.finiteVolume.PolynomialReconstruction())
            _problem.finiteVolume.RestoreReconstructionGuess(*_execution.observationReconstruction,
                                                             _problem.finiteVolume.ReconstructionGuess(*_state.reconstruction), {}, _execution.ComputeStream());
        _problem.finiteVolume.Execute(*_execution.observationInput, *_execution.observationReconstruction,
                                      *_execution.observationWorkspace, _execution.Context(),
                                      FiniteVolumeEvaluation::Spatial(*_execution.observationState, _state.progressTime));
        _problem.finiteVolume.WriteVisualization(*_execution.observationInput, *_execution.observationReconstruction, *_execution.observationWorkspace, prefix, _state.progressTime,
                                                 hdfDeflateLevel, _execution.ChemicalWorkers());
    }
    DomainStateSnapshot Domain::CaptureStateSnapshot(
        bool includeReconstruction) const
    {
        return DomainStateSnapshot(*this, includeReconstruction);
    }
    void Domain::EvaluateObservation(const DomainStateSnapshot &snapshot)
    {
        if (!_execution.observationWorkspace)
        {
            std::optional<EquationDofs> input;
            std::optional<EquationDofs> residual;
            MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                            {
                input.emplace(EquationDofOperations::NewVector(snapshot.Means()));
                residual.emplace(EquationDofOperations::NewVector(snapshot.Means()));
                input->Fields().SetName(_problem.name + "/observation/means", DNDS_ALLOCATION_SITE());
                residual->Fields().SetName(_problem.name + "/observation/residual", DNDS_ALLOCATION_SITE()); });
            auto workspace = _problem.finiteVolume.BuildWorkspace(*input, _execution.ChemicalWorkers(), ObservationWorkspaceOptions());
            auto reconstruction = _problem.finiteVolume.BuildReconstructionState(*input);
            _execution.observationInput = std::move(input);
            _execution.observationState = std::move(residual);
            _execution.observationReconstruction = std::move(reconstruction);
            _execution.observationWorkspace = std::move(workspace);
            if (_execution.interfaces->active)
                _problem.finiteVolume.SetBoundaryInputs(*_execution.observationWorkspace, *_execution.interfaces->active);
        }
        // Start every observation from the accepted numerical guess, not the
        // last observation and not a cold zero polynomial. This preserves
        // output-cadence independence while avoiding a much less converged
        // boundary reconstruction than the one used by the implicit stage.
        // A pending/rejected trial owns a different polynomial; its backup is
        // the accepted guess until publication, just as for checkpoint writing.
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        {
            _execution.Synchronize();
            EquationDofOperations::Copy(_execution.Engine(), _execution.ComputeStream(), *_execution.observationInput, snapshot.Means());
            _execution.Synchronize(); });
        if (_problem.finiteVolume.PolynomialReconstruction())
        {
            DNDS_check_throw_info(snapshot.Reconstruction(),
                                  "observation snapshot lacks polynomial reconstruction");
            _problem.finiteVolume.RestoreReconstructionGuess(
                *_execution.observationReconstruction,
                *snapshot.Reconstruction(), {}, _execution.ComputeStream());
        }
        _problem.finiteVolume.Execute(*_execution.observationInput, *_execution.observationReconstruction,
                                      *_execution.observationWorkspace, _execution.Context(),
                                      FiniteVolumeEvaluation::Spatial(*_execution.observationState, snapshot.Time()));
    }
}
