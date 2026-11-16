#include "Cauchy/Core/Domain.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/IO/StateArchive.hpp"
#include "DNDS/Runtime/Array/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Runtime/Communication/IdRowTransfer.hpp"
#include <limits>
#include <set>

#include "Cauchy/IO/CheckpointMetadata.hpp"

namespace DNDS::Cauchy
{
    using namespace CheckpointMetadata;
    void Domain::ReserveCheckpointRead(IO::DomainCheckpointRead &checkpoint, bool collectiveChecks)
    {
        _execution.Synchronize();
        const auto reserve = [&]
        {
            DNDS_check_throw_info(!checkpoint.committed && !checkpoint.destination, "checkpoint publication was already reserved or committed");
            checkpoint.source.emplace(_state.trial.ReadOwned(_execution.ComputeStream().Context()));
            checkpoint.destination.emplace(_state.accepted.WriteOwned(_execution.ComputeStream().Context())); };
        if (collectiveChecks)
            MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), reserve);
        else
            reserve();
        if (_problem.finiteVolume.PolynomialReconstruction())
        {
            DNDS_check_throw_info(bool(checkpoint.equationGuess),
                                  "checkpoint lacks a unified equation reconstruction candidate");
            checkpoint.reconstruction.emplace(_problem.finiteVolume.ReserveReconstructionRestore(
                _state.reconstruction, *checkpoint.equationGuess,
                checkpoint.limitedVRHistory
                    ? OptionalRef<const Runtime::Array<real>>(*checkpoint.limitedVRHistory)
                    : OptionalRef<const Runtime::Array<real>>{},
                collectiveChecks, _execution.ComputeStream()));
        }
    }
    void Domain::CommitCheckpointRead(IO::DomainCheckpointRead &checkpoint)
    {
        // Every participating domain/rank has reserved before this point.
        // The retained scopes cover the copy on the ordinary engine lane. No
        // rank communication or host dereference of resident values occurs.
        if (checkpoint.reconstruction)
            checkpoint.reconstruction->Commit();
        EquationDofOperations::CopyReserved(_execution.Engine(), _execution.ComputeStream(),
                                            _state.accepted, _state.trial, *checkpoint.destination, *checkpoint.source);
        checkpoint.committed = true;
        checkpoint.source.reset();
        checkpoint.destination.reset();
        checkpoint.reconstruction.reset();
        _state.time = checkpoint.time;
        _state.epoch = checkpoint.epoch;
        _state.trialReady = _state.restoreReconstruction = _state.coordinatedPrepared = false;
        _state.stageGroup.reset();
        if (checkpoint.stepper)
            _state.timeStepper.swap(checkpoint.stepper);
        _state.lastAcceptedDt = checkpoint.lastDt;
        _state.nextTrial = 0;
    }
    void Domain::CommitInternalCheckpointRead(IO::DomainInternalCheckpointRead &checkpoint)
    {
        const bool polynomial = bool(_problem.finiteVolume.PolynomialReconstruction());
        DNDS_check_throw_info(_state.timeStepper &&
                                  (!polynomial || (checkpoint.endpointWarm && checkpoint.endpointFinal)) &&
                                  checkpoint.trialEpoch.acceptedStep == _state.epoch.acceptedStep,
                              "internal checkpoint cannot follow the restored accepted state");
        if (polynomial)
        {
            const auto &acceptedGuess = _problem.finiteVolume.ReconstructionGuess(*_state.reconstruction);
            if (!_state.reconstructionBackup)
            {
                _state.reconstructionBackup.emplace(EquationDofOperations::NewVector(acceptedGuess));
                _state.reconstructionBackup->Fields().SetName(_problem.name + "/reconstructionBackup", DNDS_ALLOCATION_SITE());
            }
            EquationDofOperations::Copy(_execution.Engine(), _execution.ComputeStream(), *_state.reconstructionBackup, acceptedGuess);
            _execution.Synchronize();
            if (const auto history = _problem.finiteVolume.LimitedVRHistory(*_state.reconstruction))
            {
                if (!_state.limitedVRBackup)
                {
                    _state.limitedVRBackup.emplace(history->Layout(), history->GetDevice());
                    _state.limitedVRBackup->SetName(_problem.name + "/limitedVRBackup", DNDS_ALLOCATION_SITE());
                }
                _state.limitedVRBackup->CopyFrom(*history, _execution.ComputeStream().Context()).Wait();
            }
            else
                _state.limitedVRBackup.reset();
        }
        else
        {
            _state.reconstructionBackup.reset();
            _state.limitedVRBackup.reset();
        }
        _state.timeStepper->RestoreActive(checkpoint.time, _state.trial);
        if (polynomial)
            _problem.finiteVolume.RestoreReconstructionState(*_state.reconstruction, *checkpoint.endpointWarm,
                                                             *checkpoint.endpointFinal,
                                                             checkpoint.endpointHistory
                                                                 ? OptionalRef<const Runtime::Array<real>>(*checkpoint.endpointHistory)
                                                                 : OptionalRef<const Runtime::Array<real>>{},
                                                             _execution.ComputeStream());
        if (checkpoint.ditr)
        {
            DNDS_check_throw_info(!polynomial || (checkpoint.middleWarm && checkpoint.middleFinal),
                                  "DITR internal checkpoint lacks midpoint reconstruction");
            auto &middle = _state.timeStepper->GroupState(
                _state.timeStepper->ActiveGroup(), _state.trial, 0);
            _coupledMiddle = _execution.coupledFactory(middle);
            _execution.coupledMiddleOwner = middle.PhysicalField(0).Owned().SharedStorage();
            if (polynomial)
                _problem.finiteVolume.RestoreReconstructionState(*_coupledMiddle->state, *checkpoint.middleWarm,
                                                                 *checkpoint.middleFinal,
                                                                 checkpoint.middleHistory
                                                                     ? OptionalRef<const Runtime::Array<real>>(*checkpoint.middleHistory)
                                                                     : OptionalRef<const Runtime::Array<real>>{},
                                                                 _execution.ComputeStream());
            *_state.coupledNonlinear = checkpoint.nonlinear;
        }
        else
            _execution.implicit->RestoreState(checkpoint.nonlinear);
        _state.trialEpoch = checkpoint.trialEpoch;
        _state.trialDt = checkpoint.trialDt;
        _state.progressTime = checkpoint.progressTime;
        _state.trialReady = false;
        _state.restoreReconstruction = polynomial;
        _state.internalResume = true;
        _state.coordinatedPrepared = false;
        _state.stageGroup.reset();
        _state.nextTrial = std::max(_state.nextTrial, _state.trialEpoch.trial);
        _lastImplicit.clear();
        _lastCoupledIterations = 0;
        _lastCoupledTargetRatio = 0;
    }
}
