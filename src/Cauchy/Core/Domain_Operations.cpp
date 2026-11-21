#include "Cauchy/Core/Domain.hpp"

namespace DNDS::Cauchy
{
    struct Domain::CandidateReservation::Data
    {
        std::reference_wrapper<Domain> domain;
        std::unique_ptr<EquationDofs::WriteScope> destination;
        std::unique_ptr<EquationDofs::ReadScope> source;
        bool committed = false;

        explicit Data(Domain &owner) : domain(owner) {}
        ~Data()
        {
            if (!committed && destination)
                destination->CancelUnmodified();
        }
    };

    struct Domain::CheckpointCandidate::Data
    {
        std::reference_wrapper<Domain> domain;
        std::unique_ptr<IO::DomainCheckpointRead> candidate;
        Data(Domain &owner, std::unique_ptr<IO::DomainCheckpointRead> value)
            : domain(owner), candidate(std::move(value)) {}
    };

    struct Domain::InternalCheckpointCandidate::Data
    {
        std::reference_wrapper<Domain> domain;
        std::unique_ptr<IO::DomainInternalCheckpointRead> candidate;
        Data(Domain &owner, std::unique_ptr<IO::DomainInternalCheckpointRead> value)
            : domain(owner), candidate(std::move(value)) {}
    };

    Domain::CandidateReservation::CandidateReservation(
        std::unique_ptr<Data> data)
        : _data(std::move(data)) {}
    Domain::CandidateReservation::CandidateReservation(CandidateReservation &&) noexcept = default;
    Domain::CandidateReservation &Domain::CandidateReservation::operator=(CandidateReservation &&) noexcept = default;
    Domain::CandidateReservation::~CandidateReservation() = default;
    Domain::CheckpointCandidate::CheckpointCandidate(
        std::unique_ptr<Data> data)
        : _data(std::move(data)) {}
    Domain::CheckpointCandidate::CheckpointCandidate(CheckpointCandidate &&) noexcept = default;
    Domain::CheckpointCandidate &Domain::CheckpointCandidate::operator=(CheckpointCandidate &&) noexcept = default;
    Domain::CheckpointCandidate::~CheckpointCandidate() = default;
    real Domain::CheckpointCandidate::Time() const
    {
        DNDS_check_throw_info(_data && _data->candidate,
                              "empty checkpoint candidate");
        return _data->candidate->time;
    }
    StateEpoch Domain::CheckpointCandidate::Epoch() const
    {
        DNDS_check_throw_info(_data && _data->candidate,
                              "empty checkpoint candidate");
        return _data->candidate->epoch;
    }
    Domain::InternalCheckpointCandidate::InternalCheckpointCandidate(
        std::unique_ptr<Data> data)
        : _data(std::move(data)) {}
    Domain::InternalCheckpointCandidate::InternalCheckpointCandidate(
        InternalCheckpointCandidate &&) noexcept = default;
    Domain::InternalCheckpointCandidate &
    Domain::InternalCheckpointCandidate::operator=(
        InternalCheckpointCandidate &&) noexcept = default;
    Domain::InternalCheckpointCandidate::~InternalCheckpointCandidate() = default;
    StateEpoch Domain::InternalCheckpointCandidate::TrialEpoch() const
    {
        DNDS_check_throw_info(_data && _data->candidate,
                              "empty internal checkpoint candidate");
        return _data->candidate->trialEpoch;
    }

    MPI_Comm Domain::Communicator() const
    {
        return _problem.finiteVolume.Metrics().Communicator();
    }
    std::shared_ptr<Runtime::CommunicationContext> Domain::PrepareReductionContext()
    {
        if (!_execution.reductionContext)
        {
            const auto parent = _problem.finiteVolume.Metrics().Communication();
            _execution.reductionContext = std::make_shared<Runtime::CommunicationContext>(
                parent->Communicator(), parent->ChecksCollectivePreflight());
        }
        return _execution.reductionContext;
    }
    void Domain::RequireAcceptedClock(real time, uint64_t acceptedStep) const
    {
        DNDS_check_throw_info(
            _state.time == time && _state.epoch.acceptedStep == acceptedStep,
            "domain advanced outside its Simulation");
    }
    uint64_t Domain::TrialIdentifier() const
    {
        return _state.trialEpoch.trial;
    }
    void Domain::PrepareInterface(int coupledStage, bool reconstructed, real time)
    {
        if (_state.stageGroup && _state.stageGroup->coupledStages == 2)
        {
            PrepareCoupledInterface(coupledStage, reconstructed);
            return;
        }
        DNDS_check_throw_info(coupledStage == 0, "ordinary interface has only one stage slot");
        _problem.finiteVolume.Execute(
            _state.trial, *_state.reconstruction, *_execution.workspace,
            _execution.Context(), reconstructed ? FiniteVolumeEvaluation::Reconstruction(time) : FiniteVolumeEvaluation::Sampling());
    }
    InterfaceDonorPublication
    Domain::PublishInterfaceDonor(int coupledStage, bool prepared) const
    {
        if (_state.stageGroup && _state.stageGroup->coupledStages == 2)
            return PublishCoupledInterfaceDonor(coupledStage, prepared);
        DNDS_check_throw_info(coupledStage == 0, "ordinary interface has only one stage slot");
        return _problem.finiteVolume.PublishInterfaceDonor(
            _state.trial, *_state.reconstruction,
            *_execution.workspace, prepared);
    }
    void Domain::SetEvaluationDependency(
        QueuedEvaluationDependency dependency)
    {
        _execution.Context().SetEvaluationDependency(
            std::move(dependency));
    }
    std::function<void(const BoundaryInputs &)>
    Domain::PrepareInterfaceReceiver(int coupledStage)
    {
        const bool coupled = coupledStage >= 0 && _state.stageGroup &&
                             _state.stageGroup->coupledStages == 2;
        DNDS_check_throw_info(coupledStage >= -1 && coupledStage < (coupled ? 2 : 1),
                              "invalid interface receiver slot");
        auto coarse = _execution.multigrid
                          ? _execution.multigrid->PrepareBoundaryInputBinding()
                          : std::function<void(const BoundaryInputs &)>{};
        auto middle = coupled && coupledStage == 0 && _coupledMiddle
                          ? _coupledMiddle->bindBoundary
                          : std::function<void(const BoundaryInputs &)>{};
        return [discrete = _problem.finiteVolume,
                packets = _execution.interfaces, work = _execution.workspace,
                observation = _execution.observationWorkspace,
                mean = _execution.timePPWorkspace, coarse = std::move(coarse),
                middle = std::move(middle), coupled, coupledStage](const BoundaryInputs &input)
        {
            const auto published = coupled ? discrete.FreezeBoundaryInputs(input)
                                           : discrete.AliasBoundaryInputs(input);
            if (coupled)
                packets->coupled[size_t(coupledStage)] = published;
            discrete.SetBoundaryInputs(*work, published);
            if (observation)
                discrete.SetBoundaryInputs(*observation, published);
            if (mean)
                discrete.SetBoundaryInputs(*mean, published);
            packets->active = published;
            if (coarse)
                coarse(published);
            if (middle)
                middle(published);
        };
    }
    void Domain::DrainEvaluationDependency()
    {
        _execution.Context().DrainEvaluationDependency();
    }
    Domain::CandidateReservation Domain::ReserveCandidate()
    {
        DNDS_check_throw_info(
            _state.trialReady &&
                (!_state.timeStepper || _state.timeStepper->HasPendingTrial()),
            "domain has no complete candidate");
        auto data = std::make_unique<CandidateReservation::Data>(*this);
        _execution.Synchronize();
        try
        {
            data->destination =
                std::make_unique<EquationDofs::WriteScope>(_state.accepted.WriteOwned(_execution.ComputeStream().Context()));
            data->source =
                std::make_unique<EquationDofs::ReadScope>(_state.trial.ReadOwned(_execution.ComputeStream().Context()));
        }
        catch (...)
        {
            if (data->destination)
                data->destination->CancelUnmodified();
            data->destination.reset();
            throw;
        }
        return CandidateReservation(std::move(data));
    }
    void Domain::CommitCandidate(CandidateReservation &reservation)
    {
        DNDS_check_throw_info(
            reservation._data &&
                &reservation._data->domain.get() == this &&
                !reservation._data->committed && reservation._data->destination &&
                reservation._data->source,
            "invalid or already committed domain candidate reservation");
        CommitReserved(*reservation._data->destination,
                       *reservation._data->source);
        reservation._data->committed = true;
    }
    void Domain::SnapAcceptedTime(real time)
    {
        _state.time = _state.trialTime = time;
    }
    Domain::CheckpointCandidate Domain::PrepareCheckpoint(
        const std::string &path, const Json &resolved, CheckpointLoadMode mode)
    {
        return CheckpointCandidate(std::make_unique<CheckpointCandidate::Data>(
            *this, PrepareCheckpointRead(path, resolved, mode)));
    }
    Domain::InternalCheckpointCandidate
    Domain::PrepareInternalCheckpoint(const std::string &path,
                                      const Json &resolved)
    {
        return InternalCheckpointCandidate(
            std::make_unique<InternalCheckpointCandidate::Data>(
                *this,
                PrepareInternalCheckpointRead(path, resolved)));
    }
    InterfaceDonorPublication
    Domain::PrepareCheckpointInterfaceDonor(
        CheckpointCandidate &candidate)
    {
        DNDS_check_throw_info(
            candidate._data &&
                &candidate._data->domain.get() == this &&
                candidate._data->candidate,
            "interface checkpoint candidate belongs to another domain");
        auto &guess = candidate._data->candidate->equationGuess;
        const auto restored = _execution.ComputeStream().Record();
        const auto exchange = [&](CFV::FVField &field)
        {
            _problem.finiteVolume.BindHalo(field)
                .SubmitAfter(_execution.Engine(), _execution.CommunicationStream(), restored)
                .Wait();
        };
        for (index iField = 0; iField < _state.trial.PhysicalFieldNumber(); ++iField)
        {
            exchange(_state.trial.PhysicalField(iField));
            if (guess)
                exchange(guess->PhysicalField(iField));
        }
        // Match the accepted-time predictor in Simulation::Advance. Direct P1
        // publishes means here. It rebuilds gradients with stage-time packets
        // when the next step prepares its first reconstruction.
        if (guess)
            return _problem.finiteVolume.PublishVariationalInterfaceDonor(_state.trial, *guess);
        DNDS_check_throw_info(!_problem.finiteVolume.PolynomialReconstruction(),
                              "missing variational checkpoint donor coefficients");
        return _problem.finiteVolume.PublishInterfaceDonor(
            _state.trial, *_state.reconstruction, *_execution.workspace, false);
    }
    void Domain::ReserveCheckpoint(CheckpointCandidate &candidate,
                                   bool collectiveChecks)
    {
        DNDS_check_throw_info(candidate._data &&
                                  &candidate._data->domain.get() ==
                                      this &&
                                  candidate._data->candidate,
                              "checkpoint candidate belongs to another domain");
        ReserveCheckpointRead(*candidate._data->candidate,
                              collectiveChecks);
    }
    void Domain::CommitCheckpoint(CheckpointCandidate &candidate)
    {
        DNDS_check_throw_info(candidate._data &&
                                  &candidate._data->domain.get() ==
                                      this &&
                                  candidate._data->candidate,
                              "checkpoint candidate belongs to another domain");
        CommitCheckpointRead(*candidate._data->candidate);
    }
    void Domain::CommitInternalCheckpoint(
        InternalCheckpointCandidate &candidate)
    {
        DNDS_check_throw_info(candidate._data &&
                                  &candidate._data->domain.get() ==
                                      this &&
                                  candidate._data->candidate,
                              "internal checkpoint candidate belongs to another domain");
        CommitInternalCheckpointRead(*candidate._data->candidate);
    }
}
