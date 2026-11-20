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
    void Domain::WriteCheckpoint(const std::string &path, const Json &resolved, bool includeReconstruction,
                                 const Json &outputController) const
    {
        const auto prepared = PrepareCheckpoint(_problem.finiteVolume.Metrics().Communication(),
                                                {includeReconstruction, false});
        prepared.Write(path, resolved, outputController);
    }
    PreparedCheckpoint Domain::PrepareCheckpoint(std::shared_ptr<Runtime::CommunicationContext> communication,
                                                 CheckpointPreparation options) const
    {
        return CaptureCheckpoint(std::move(communication), options, Runtime::ExecutionContext::Host());
    }
    PreparedCheckpoint Domain::CaptureCheckpoint(std::shared_ptr<Runtime::CommunicationContext> communication,
                                                 CheckpointPreparation options, Runtime::ExecutionContext writerCopy) const
    {
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        const bool includeReconstruction = options.includeReconstruction && bool(_problem.finiteVolume.PolynomialReconstruction());
        MPI::LocalStage(mpi, [&]
                        {
            DNDS_check_throw_info(communication, "checkpoint preparation needs a retained I/O context");
            int relation = MPI_UNEQUAL;
            DNDS_check_throw_info(MPI_Comm_compare(mpi.comm, communication->Communicator(), &relation) == MPI_SUCCESS &&
                                      (relation == MPI_IDENT || relation == MPI_CONGRUENT),
                                  "checkpoint I/O communicator must preserve domain membership and rank order"); });
        auto snapshot = CaptureStateSnapshot(includeReconstruction);
        // A background GPU checkpoint freezes its own device values here.
        // The writer stages that immutable snapshot on its dedicated copy stream.
        const bool detached = options.detached || snapshot.Means().Device() != Runtime::Device::Host();
        MPI::LocalStage(mpi, [&]
                        {
            if (detached) snapshot = snapshot.DetachedTo(writerCopy.device, _execution.ComputeStream().Context()); });
        const bool history = bool(snapshot.PreviousState());
        const bool limitedVRHistory = bool(snapshot.LimitedVRHistory());
        int flags[]{int(includeReconstruction), -int(includeReconstruction), int(history), -int(history), int(limitedVRHistory), -int(limitedVRHistory)};
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<int>{flags, 6}, mpi.comm);
        DNDS_check_throw_info(flags[0] == -flags[1] && flags[2] == -flags[3] && flags[4] == -flags[5], "checkpoint optional sections differ between ranks");
        Json metadata{{"schema_version", 1}, {"mesh_envelope", 2}, {"flow", EquationMetadata(_problem.equation)}, {"reconstruction", _problem.finiteVolume.ReconstructionMetadata()}, {"time_method", snapshot.TimeMethod()}, {"history", history}, {"limited_vr_history", limitedVRHistory}, {"limited_vr_history_schema", 1}, {"frame", FrameMetadata(_problem.finiteVolume, snapshot.Time())}};
        metadata["restart_scope"] = "accepted_step";
        if (auto gauge = snapshot.PressureGauge())
            metadata["acm_pressure_mean"] = *gauge;
        metadata["update_policy"] = snapshot.UpdatePolicy();
        metadata["nonlinear_state"] = snapshot.NonlinearState();
        metadata["state_storage"] = DescribeEquationStorage(snapshot.Means());
        if (includeReconstruction)
            metadata["reconstruction_storage"] = DescribeEquationStorage(*snapshot.Reconstruction());
        metadata["source_integration"] = snapshot.SourceStrang() ? "strang" : "coupled";
        Runtime::Array<index> ids;
        std::optional<MeshEnvelope> envelope;
        MPI::LocalStage(mpi, [&]
                        {
            ids = CellIds(*this);
            envelope.emplace(DescribeMesh(*this)); });
        return PreparedCheckpoint(std::move(snapshot), std::move(metadata), std::move(ids),
                                  std::move(*envelope), std::move(communication), detached, {}, writerCopy);
    }
    void Domain::WriteInternalCheckpoint(const std::string &path, const Json &resolved,
                                         const NonlinearSolveState &state,
                                         const Json &outputController) const
    {
        CaptureInternalCheckpoint(_problem.finiteVolume.Metrics().Communication(), state, false).Write(path, resolved, outputController);
    }

    PreparedCheckpoint Domain::CaptureInternalCheckpoint(std::shared_ptr<Runtime::CommunicationContext> communication,
                                                         const NonlinearSolveState &state, bool detach, Runtime::ExecutionContext writerCopy) const
    {
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        const bool polynomial = bool(_problem.finiteVolume.PolynomialReconstruction());
        auto acceptedSnapshot = CaptureStateSnapshot(polynomial);
        std::optional<EquationTimeStepper::ActiveCheckpoint> active;
        MPI::LocalStage(mpi, [&]
                        {
            DNDS_check_throw_info(_state.timeStepper && (!polynomial || (_state.restoreReconstruction && _state.reconstructionBackup)) &&
                                      _state.trialEpoch.trial && state.stage >= 0 && state.iteration >= 0,
                                  "exact internal checkpoint has no active implicit trial");
            active.emplace(_state.timeStepper->CaptureActive(_state.trial)); });
        // Method-owned snapshot copies use queued algebra, just like stepping.
        // Observe them before the host-only serializer acquires the payloads.
        _execution.Synchronize();
        const bool ditr = _state.timeStepper->Method() == "ditr";
        MPI::LocalStage(mpi, [&]
                        { DNDS_check_throw_info(!ditr || (_coupledMiddle && _coupledMiddle->state),
                                                "DITR internal checkpoint lacks its midpoint reconstruction"); });
        const bool acceptedHistory = bool(acceptedSnapshot.PreviousState());
        const bool acceptedLimitedVR = polynomial && bool(acceptedSnapshot.LimitedVRHistory());
        const bool endpointLimitedVR = polynomial && bool(_problem.finiteVolume.LimitedVRHistory(*_state.reconstruction));
        const bool middleLimitedVR = polynomial && ditr && bool(_problem.finiteVolume.LimitedVRHistory(*_coupledMiddle->state));
        Json timePayloads = Json::array();
        for (index i = 0; i < index(active->fields.size()); ++i)
            timePayloads.push_back({{"name", active->names[size_t(i)]},
                                    {"storage", DescribeEquationStorage(active->fields[size_t(i)])}});
        Json timeState{{"schema_version", active->schemaVersion},
                       {"method", active->method},
                       {"pending_dt", active->pendingDt},
                       {"start_time", active->startTime},
                       {"pending_steady", active->pendingSteady},
                       {"stages_active", active->stagesActive},
                       {"stage_prepared", active->stagePrepared},
                       {"next_stage", active->nextStage},
                       {"equation", {{"inverse_dt", active->equation.inverseDt}, {"rhs_scale", active->equation.rhsScale}, {"time", active->equation.time}}},
                       {"bdf_history", active->bdfHistory},
                       {"bdf_previous_dt", active->bdfPreviousDt},
                       {"ditr_time", active->ditrTime},
                       {"ditr_dt", active->ditrDt},
                       {"payloads", std::move(timePayloads)}};
        if (active->ditrSettings)
            timeState["ditr_settings"] = {
                {"fraction", active->ditrSettings->fraction},
                {"theta", active->ditrSettings->theta},
                {"variant", active->ditrSettings->variant == ODE::DITRVariant::U2R1 ? "U2R1" : "U2R2"}};
        const Json trialEpoch{{"accepted_step", _state.trialEpoch.acceptedStep},
                              {"trial", _state.trialEpoch.trial},
                              {"stage", _state.trialEpoch.stage},
                              {"coupled_stage", _state.trialEpoch.coupledStage},
                              {"coupling_iteration", _state.trialEpoch.couplingIteration},
                              {"geometry", _state.trialEpoch.geometry},
                              {"schema", _state.trialEpoch.schema}};
        Json metadata{{"schema_version", 1},
                      {"mesh_envelope", 2},
                      {"flow", EquationMetadata(_problem.equation)},
                      {"reconstruction", _problem.finiteVolume.ReconstructionMetadata()},
                      {"polynomial_reconstruction", polynomial},
                      {"time_method", _state.timeStepper->Method()},
                      {"history", acceptedHistory},
                      {"limited_vr_history", acceptedLimitedVR},
                      {"limited_vr_history_schema", 1},
                      {"frame", FrameMetadata(_problem.finiteVolume, acceptedSnapshot.Time())},
                      {"restart_scope", "exact_internal"},
                      {"update_policy", acceptedSnapshot.UpdatePolicy()},
                      {"nonlinear_state", state.ToJson()},
                      {"trial_epoch", trialEpoch},
                      {"trial_dt", _state.trialDt},
                      {"progress_time", _state.progressTime},
                      {"time_state", timeState},
                      {"state_storage", DescribeEquationStorage(acceptedSnapshot.Means())},
                      {"endpoint_limited_vr_history", endpointLimitedVR},
                      {"middle_limited_vr_history", middleLimitedVR},
                      {"source_integration", _sourceStrang ? "strang" : "coupled"}};
        if (polynomial)
        {
            metadata["reconstruction_storage"] = DescribeEquationStorage(*acceptedSnapshot.Reconstruction());
            metadata["endpoint_warm_storage"] = DescribeEquationStorage(_problem.finiteVolume.ReconstructionGuess(*_state.reconstruction));
            metadata["endpoint_final_storage"] = DescribeEquationStorage(_problem.finiteVolume.FluxReconstruction(*_state.reconstruction, *_execution.workspace));
        }
        if (polynomial && ditr)
        {
            metadata["middle_warm_storage"] = DescribeEquationStorage(_problem.finiteVolume.ReconstructionGuess(*_coupledMiddle->state));
            metadata["middle_final_storage"] = DescribeEquationStorage(_problem.finiteVolume.FluxReconstruction(*_coupledMiddle->state, *_execution.workspace));
        }
        if (auto gauge = acceptedSnapshot.PressureGauge())
            metadata["acm_pressure_mean"] = *gauge;
        const auto context = _execution.ComputeStream().Context();
        detach |= acceptedSnapshot.Means().Device() != Runtime::Device::Host();
        PreparedCheckpoint::InternalFields internal;
        std::optional<MeshEnvelope> envelope;
        Runtime::Array<index> ids;
        MPI::LocalStage(mpi, [&]
                        {
            const auto addField = [&](std::string key, const EquationDofs &field)
            {
                auto saved = detach ? EquationDofOperations::CopyOwnedTo(field, writerCopy.device, context,
                                           {writerCopy.device == Runtime::Device::Host() && field.Device().space == Runtime::MemorySpace::CUDA}) : field.NumericAlias();
                if (detach) saved.Fields().SetName(_problem.name + "/outputSnapshot/" + key, DNDS_ALLOCATION_SITE());
                internal.equations.emplace_back(std::move(key), std::move(saved));
            };
            const auto addArray = [&](std::string key, const Runtime::Array<real> &field)
            {
                auto saved = detach ? field.To(writerCopy.device, context) : field;
                if (detach) saved.SetName(_problem.name + "/outputSnapshot/" + key, DNDS_ALLOCATION_SITE());
                internal.arrays.emplace_back(std::move(key), std::move(saved));
            };
            for (size_t i = 0; i < active->fields.size(); ++i)
                addField("internal_time_" + std::to_string(i) + "_field_", active->fields[i]);
            if (ditr)
                for (int stage = 0; stage < 2; ++stage)
                    if (const auto &derivative = _execution.coupledSourceDerivative[stage])
                        addArray("internal_source_derivative_" + std::to_string(stage), *derivative);
            if (polynomial)
            {
                const auto addReconstruction = [&](const auto &reconstruction, const std::string &prefix)
                {
                    addField(prefix + "warm_field_", _problem.finiteVolume.ReconstructionGuess(reconstruction));
                    addField(prefix + "final_field_", _problem.finiteVolume.FluxReconstruction(reconstruction, *_execution.workspace));
                    if (const auto history = _problem.finiteVolume.LimitedVRHistory(reconstruction))
                        addArray(prefix + "limited_vr_history", *history);
                };
                addReconstruction(*_state.reconstruction, "internal_endpoint_");
                if (ditr) addReconstruction(*_coupledMiddle->state, "internal_middle_");
            }
            if (detach) acceptedSnapshot = acceptedSnapshot.DetachedTo(writerCopy.device, context);
            ids = CellIds(*this);
            envelope.emplace(DescribeMesh(*this)); });
        return PreparedCheckpoint(std::move(acceptedSnapshot), std::move(metadata), std::move(ids),
                                  std::move(*envelope), std::move(communication), detach, std::move(internal), writerCopy);
    }
}
