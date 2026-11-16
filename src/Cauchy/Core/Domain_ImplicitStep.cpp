#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Core/StepResult.hpp"
#include "Cauchy/Implicit/NonlinearIteration.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    void Domain::SetTimeStage(int stage)
    {
        const bool coupled = _state.timeStepper && _state.timeStepper->Method() == "ditr";
        _state.trialEpoch.stage = coupled ? 0 : stage;
        _state.trialEpoch.coupledStage = coupled ? stage : 0;
    }
    NonlinearSolveControl Domain::ProgressControl() const
    {
        NonlinearSolveControl control;
        if (_progressObserver)
            control.progress = [observer = _progressObserver](const NonlinearProgress &progress)
            { observer(progress); };
        if (_quiescentObserver)
            control.quiescent = [observer = _quiescentObserver](const NonlinearSolveState &state)
            { observer(state); };
        return control;
    }
    Json Domain::UpdatePolicyMetadata() const
    {
        if (!_execution.nonlinearSetup)
            return {{"name", "explicit"},
                    {"coupled_stages", 0},
                    {"residual_observation", "none"},
                    {"method_stopping_residual", false}};
        return _execution.nonlinearSetup->policy->Descriptor().Metadata();
    }
    void Domain::InitializeImplicit(const DomainSettings &settings, const TimeSettings &time, const ssp<Geom::UnstructuredMesh> &mesh)
    {
        _steady = time.steady;
        _coupledReconstructionTolerance = settings.implicit.requireConvergedReconstruction &&
                                                  settings.numerics.reconstruction == "variational"
                                              ? settings.numerics.reconstructionTolerance
                                              : 0;
        _rejectUnconvergedStages = settings.implicit.rejectUnconvergedStages;
        if (time.method == "ssprk3" && !_steady)
        {
            DNDS_check_throw_info(!settings.implicit.multigrid.levels, "polynomial FAS requires an implicit time method");
            return;
        }
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        { time.ditr.Validate();
                          _implicitControls = time.method == "ditr" ? time.ditr.ResolveControls(settings.implicit) : settings.implicit;
                          _ditrComponents = _problem.equation.Schema().ResolveComponents(settings.implicit.residualComponents);
                          const bool ditr = time.method == "ditr";
                          _execution.nonlinearSetup = std::make_unique<NonlinearSolveSetup>(
                              _implicitControls, _problem.equation.Schema(), mpi.rank,
                              _implicitControls.updatePolicy,
                              UpdatePolicyRequirements{
                                  ditr ? 2 : 1,
                                  ditr ? NonlinearDetail::ResidualObservation::AfterUpdate
                                       : NonlinearDetail::ResidualObservation::BeforeUpdate,
                                  ditr}); });
        _execution.coupledFactory = [this, mesh, settings, mpi](EquationDofs &middle)
        {
            const auto reconstruction = _problem.finiteVolume.BuildReconstructionState(middle);
            // Endpoint and midpoint solves are serialized by the coupled DITR
            // loop, so they reuse one allocation-independent scratch workspace.
            // Their means, halos and durable reconstruction states stay distinct.
            const auto workspace = _execution.workspace;
            const auto solver = std::make_shared<NonlinearSolver>(
                _problem.finiteVolume, middle, reconstruction, workspace,
                _execution.Context(), settings,
                Algebra::ComponentSelection(middle.ComponentNumber(), _ditrComponents));
            std::shared_ptr<PolynomialMultigrid> multigrid;
            if (settings.implicit.multigrid.levels)
                multigrid = std::make_shared<PolynomialMultigrid>(
                    mesh, _problem.finiteVolume, middle, *solver, _execution.Context(), settings);
            return DomainCoupledStageOperator{
                [this, solver](auto &state, auto &rate, real time, int stage)
                { SetTimeStage(stage); solver->EvaluateRate(state, rate, time); },
                [this, solver, multigrid](const auto &equation, const auto &rate, real cfl)
                {
                    const auto &c = equation.coefficients;
                    _state.progressTime = c.time;
                    SetTimeStage(equation.coupledStage);
                    return multigrid ? multigrid->ApplyCycle(equation.resOther, c.inverseDt, c.rhsScale, c.time, equation.coupledStage, cfl, std::cref(rate))
                                     : solver->CorrectOnce(equation.state, equation.resOther, c.inverseDt, c.rhsScale, c.time,
                                                           equation.coupledStage, {1, cfl}, std::cref(rate));
                },
                [solver]
                { return solver->ReconstructionStats(); },
                multigrid ? multigrid->PrepareBoundaryInputBinding()
                          : std::function<void(const BoundaryInputs &)>{},
                reconstruction};
        };
        MPI::LocalStage(mpi, [&]
                        { _state.timeStepper = std::make_unique<EquationTimeStepper>(_state.trial, _steady ? "bdf1" : time.method, [mpi, checks = settings.implicit.collectiveChecks](const auto &run)
                                                                                     { if (checks) MPI::LocalStage(mpi, run); else run(); }, ODE::DITRSettings{time.ditr.fraction, time.ditr.theta, time.ditr.variant == "U2R1" ? ODE::DITRVariant::U2R1 : ODE::DITRVariant::U2R2}, QueuedEquationOperations(_execution.Context().ComputeStream())); });
        _execution.implicit = std::make_unique<NonlinearSolver>(
            _problem.finiteVolume, _state.trial, _state.reconstruction, _execution.workspace,
            _execution.Context(), settings,
            Algebra::ComponentSelection(_state.trial.ComponentNumber(), _ditrComponents));
        if (settings.implicit.multigrid.levels)
            _execution.multigrid = std::make_unique<PolynomialMultigrid>(
                mesh, _problem.finiteVolume, _state.trial, *_execution.implicit, _execution.Context(), settings);
    }
    void Domain::EvaluateImplicitRate(EquationDofs &state, EquationDofs &rate, FiniteVolume::Workspace &workspace, real time, int stage)
    {
        SetTimeStage(stage);
        DNDS_check_throw_info(_execution.implicit && _execution.workspace.get() == &workspace,
                              "implicit equation rate belongs to another workspace");
        _execution.implicit->EvaluateRate(state, rate, time);
    }
    bool Domain::RecordImplicitStage(NonlinearIterationReport report)
    {
        _lastAlphaCells += report.incrementLimited;
        _lastAlpha = std::min(_lastAlpha, report.minimumIncrementAlpha);
        const bool converged = report.converged;
        _lastImplicit.push_back(std::move(report));
        // A cap is an iteration-control decision, not a numerical failure.
        // Preserve the actual convergence flag even when Euler-style policy
        // accepts the final iterate. Invalid corrections/states still throw.
        return converged || !_rejectUnconvergedStages;
    }
    void Domain::PreserveReconstructionGuess()
    {
        if (!_problem.finiteVolume.PolynomialReconstruction())
            return;
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        {
            if (_state.restoreReconstruction)
                _problem.finiteVolume.RestoreReconstructionGuess(*_state.reconstruction, *_state.reconstructionBackup,
                                               _state.limitedVRBackup
                                                   ? OptionalRef<const Runtime::Array<real>>(*_state.limitedVRBackup)
                                                   : OptionalRef<const Runtime::Array<real>>{}, _execution.ComputeStream());
            else
            {
                const auto &guess = _problem.finiteVolume.ReconstructionGuess(*_state.reconstruction);
                if (!_state.reconstructionBackup)
                {
                    _state.reconstructionBackup.emplace(EquationDofOperations::NewVector(guess));
                    _state.reconstructionBackup->Fields().SetName(_problem.name + "/reconstructionBackup", DNDS_ALLOCATION_SITE());
                }
                QueuedEquationOperations(_execution.ComputeStream()).Copy(*_state.reconstructionBackup, guess);
                if (const auto history = _problem.finiteVolume.LimitedVRHistory(*_state.reconstruction))
                {
                    if (!_state.limitedVRBackup)
                    {
                        _state.limitedVRBackup.emplace(history->Layout(), history->GetDevice());
                        _state.limitedVRBackup->SetName(_problem.name + "/limitedVRBackup", DNDS_ALLOCATION_SITE());
                    }
                    _state.limitedVRBackup->CopyFrom(*history, _execution.ComputeStream().Context()).Wait();
                }
                _execution.Synchronize();
                _state.restoreReconstruction = true;
            } });
    }
    void Domain::PrepareGroupOperators(bool resume)
    {
        if (!_state.stageGroup || _state.stageGroup->coupledStages != 2)
            return;
        auto &middle = _state.timeStepper->GroupState(
            *_state.stageGroup, _state.trial, 0);
        const auto owner = middle.PhysicalField(0).Owned().SharedStorage();
        if (!_coupledMiddle || _execution.coupledMiddleOwner != owner)
        {
            _coupledMiddle = _execution.coupledFactory(middle);
            _execution.coupledMiddleOwner = owner;
            _execution.coupledPreparedTrial =
                std::numeric_limits<uint64_t>::max();
        }
        if (!resume &&
            _execution.coupledPreparedTrial != _state.trialEpoch.trial)
        {
            if (_problem.finiteVolume.PolynomialReconstruction())
                _problem.finiteVolume.CopyReconstructionState(*_coupledMiddle->state, *_state.reconstruction, _execution.ComputeStream());
            _execution.coupledPreparedTrial = _state.trialEpoch.trial;
        }
    }
    NonlinearSession Domain::CreateCoupledNonlinearSession(bool resume)
    {
        PrepareGroupOperators(resume);
        auto &endpoint = _state.trial;
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        {
            if (!_execution.coupledResidual)
                _execution.coupledResidual.emplace(EquationDofOperations::NewVector(endpoint)); });
        const auto &settings = _implicitControls;
        NonlinearIterationReport initial;
        initial.residual.resize(size_t(endpoint.ComponentNumber()));
        auto coupledControl = ProgressControl();
        coupledControl.state = _state.coupledNonlinear;
        coupledControl.resume = resume;
        if (!resume)
        {
            ++_state.coupledNonlinear->attempt;
            _state.coupledNonlinear->convergenceComponents = settings.residualComponents;
        }
        else
            DNDS_check_throw_info(_state.coupledNonlinear->convergenceComponents == settings.residualComponents,
                                  "restored DITR convergence components differ");
        DNDS_check_throw_info(_execution.nonlinearSetup,
                              "DITR update policy was not resolved");
        return NonlinearSession::Prepare(
            *_execution.nonlinearSetup, 0,
            {std::move(coupledControl), std::move(initial),
             _coupledReconstructionTolerance, [this]
             { _state.timeStepper->AssembleStoppingResidual(
                   *_state.stageGroup, _state.trial,
                   *_execution.coupledResidual); }, [this, mpi](auto &norms)
             {
            std::vector<real> sum(norms.size() + 1, 0);
            MPI::LocalStage(mpi, [&]
            {
                sum = _execution.implicit->MeasureRateNorm(*_execution.coupledResidual, _implicitControls.normOrder);
            });
            MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                MPI::ContiguousView<real>{sum.data(), sum.size()}, mpi.comm);
            DNDS_check_throw_info(std::isfinite(sum.back()) && sum.back() > 0, "invalid DITR global volume");
            for (index j = 0; j < index(norms.size()); ++j)
                norms[j] = _implicitControls.normOrder == 1 ? sum[j] : std::sqrt(sum[j]); }, [this]
             {
            const auto a = _coupledMiddle->reconstruction();
            const auto b = _execution.implicit->ReconstructionStats();
            return std::make_pair(std::max(a.first, b.first), std::max(a.second, b.second)); }, [this](int coupledStage, auto &iteration, bool applyCorrection)
             {
                SetTimeStage(coupledStage);
                ActivateCoupledInterface(coupledStage);
                const auto equation = _state.timeStepper->PrepareEquation(
                    *_state.stageGroup, _state.trial, coupledStage);
                const auto &c = equation.coefficients;
                _state.progressTime = c.time;
                const auto rate = _state.timeStepper->CachedRate(*_state.stageGroup);
                DNDS_check_throw_info(rate, "DITR update requires its method rate");
                NonlinearIterationReport work;
                if (applyCorrection && coupledStage == 0)
                    work = _coupledMiddle->correct(equation, *rate, iteration.pseudoCFL);
                else if (applyCorrection)
                    work = _execution.multigrid ? _execution.multigrid->ApplyCycle(equation.resOther, c.inverseDt, c.rhsScale, c.time,
                                                               coupledStage, iteration.pseudoCFL, rate)
                                      : _execution.implicit->CorrectOnce(_state.trial, equation.resOther, c.inverseDt, c.rhsScale, c.time,
                                                               coupledStage, {1, iteration.pseudoCFL}, rate);
                NonlinearDetail::AccumulateWork(iteration, work);
                _state.timeStepper->CompleteEquation(
                    *_state.stageGroup, _state.trial,
                    [this](EquationDofs &state, EquationDofs &rate, real time,
                           int stage)
                    {
                        if (stage == 0)
                            _coupledMiddle->evaluate(state, rate, time, stage);
                        else
                            EvaluateImplicitRate(state, rate,
                                                 *_execution.workspace, time,
                                                 stage);
                    }); }});
    }
}
