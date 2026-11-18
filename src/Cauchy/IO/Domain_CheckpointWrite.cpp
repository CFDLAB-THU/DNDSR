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
        // Host-only serializers never receive device allocations. The detached
        // copy pins the entire source epoch until all field copies complete.
        const bool detached = options.detached || snapshot.Means().Device() != Runtime::Device::Host();
        MPI::LocalStage(mpi, [&]
                        {
            if (detached) snapshot = snapshot.DetachedHost(_execution.ComputeStream().Context()); });
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
                                  std::move(*envelope), std::move(communication), detached);
    }
    void Domain::WriteInternalCheckpoint(const std::string &path, const Json &resolved,
                                         const NonlinearSolveState &state,
                                         const Json &outputController) const
    {
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        const bool polynomial = bool(_problem.finiteVolume.PolynomialReconstruction());
        const auto acceptedSnapshot = CaptureStateSnapshot(polynomial);
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
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, false);
        serializer->WriteInt("cauchy_format", 5);
        serializer->WriteInt("cauchy_internal_format", 1);
        serializer->WriteString("resolved_config", resolved.dump());
        if (auto gauge = acceptedSnapshot.PressureGauge())
            metadata["acm_pressure_mean"] = *gauge;
        serializer->WriteString("metadata_json", metadata.dump());
        if (!outputController.is_null())
            serializer->WriteString("output_controller_json", outputController.dump());
        serializer->WriteString("domain_name", acceptedSnapshot.Name());
        serializer->WriteReal("accepted_time", acceptedSnapshot.Time());
        serializer->WriteReal("last_accepted_dt", acceptedSnapshot.LastAcceptedDt());
        serializer->WriteIndex("accepted_step", index(acceptedSnapshot.Epoch().acceptedStep));
        Runtime::WriteArray(CellIds(*this), serializer, "cell_ids", mpi);
        std::optional<MeshEnvelope> envelope;
        MPI::LocalStage(mpi, [&]
                        { envelope.emplace(DescribeMesh(*this)); });
        Runtime::WriteArray(envelope->topology, serializer, "mesh_topology", mpi);
        Runtime::WriteArray(envelope->coordinates, serializer, "mesh_coordinates", mpi);
        Runtime::WriteArray(envelope->periodicTransforms, serializer, "periodic_transforms", mpi);
        WriteEquationArrays(acceptedSnapshot.Means(), serializer, "state_field_", mpi, _execution.ComputeStream().Context());
        if (polynomial)
            WriteEquationArrays(*acceptedSnapshot.Reconstruction(), serializer, "reconstruction_field_", mpi, _execution.ComputeStream().Context());
        if (acceptedLimitedVR)
            WriteStateArray(*acceptedSnapshot.LimitedVRHistory(), serializer, "limited_vr_history", mpi, _execution.ComputeStream().Context());
        if (acceptedHistory)
        {
            serializer->WriteReal("history_dt", acceptedSnapshot.PreviousDt());
            WriteEquationArrays(*acceptedSnapshot.PreviousState(), serializer, "history_field_", mpi, _execution.ComputeStream().Context());
        }
        for (index i = 0; i < index(active->fields.size()); ++i)
            WriteEquationArrays(active->fields[size_t(i)], serializer,
                                "internal_time_" + std::to_string(i) + "_field_", mpi, _execution.ComputeStream().Context());
        if (polynomial)
        {
            WriteEquationArrays(_problem.finiteVolume.ReconstructionGuess(*_state.reconstruction), serializer,
                                "internal_endpoint_warm_field_", mpi, _execution.ComputeStream().Context());
            WriteEquationArrays(_problem.finiteVolume.FluxReconstruction(*_state.reconstruction, *_execution.workspace), serializer,
                                "internal_endpoint_final_field_", mpi, _execution.ComputeStream().Context());
            if (const auto history = _problem.finiteVolume.LimitedVRHistory(*_state.reconstruction))
                WriteStateArray(*history, serializer, "internal_endpoint_limited_vr_history", mpi, _execution.ComputeStream().Context());
            if (ditr)
            {
                WriteEquationArrays(_problem.finiteVolume.ReconstructionGuess(*_coupledMiddle->state), serializer,
                                    "internal_middle_warm_field_", mpi, _execution.ComputeStream().Context());
                WriteEquationArrays(_problem.finiteVolume.FluxReconstruction(*_coupledMiddle->state, *_execution.workspace), serializer,
                                    "internal_middle_final_field_", mpi, _execution.ComputeStream().Context());
                if (const auto history = _problem.finiteVolume.LimitedVRHistory(*_coupledMiddle->state))
                    WriteStateArray(*history, serializer, "internal_middle_limited_vr_history", mpi, _execution.ComputeStream().Context());
            }
        }
        serializer->CloseFile();
    }
}
