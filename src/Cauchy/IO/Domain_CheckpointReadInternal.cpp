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
    std::unique_ptr<IO::DomainInternalCheckpointRead> Domain::PrepareInternalCheckpointRead(const std::string &path, const Json &resolved)
    {
        _execution.Synchronize();
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        auto result = std::make_unique<IO::DomainInternalCheckpointRead>();
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, true);
        int format = 0;
        serializer->ReadInt("cauchy_internal_format", format);
        std::string serializedMetadata, serializedConfig;
        serializer->ReadString("metadata_json", serializedMetadata);
        serializer->ReadString("resolved_config", serializedConfig);
        const auto metadata = Json::parse(serializedMetadata, nullptr, false);
        const auto storedConfig = Json::parse(serializedConfig, nullptr, false);
        const bool polynomial = bool(_problem.finiteVolume.PolynomialReconstruction());
        int valid = format == 1 && metadata.is_object() && storedConfig.is_object() &&
                    metadata.value("restart_scope", std::string{}) == "exact_internal" &&
                    CompatibleConfig(storedConfig) == CompatibleConfig(resolved) && _state.timeStepper &&
                    metadata.value("time_method", std::string{}) == _state.timeStepper->Method() &&
                    metadata.value("polynomial_reconstruction", true) == polynomial &&
                    (!metadata.contains("update_policy") || metadata.at("update_policy") == UpdatePolicyMetadata()) &&
                    metadata.value("reconstruction", Json()) == _problem.finiteVolume.ReconstructionMetadata();
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, mpi.comm);
        DNDS_check_throw_info(valid, "Cauchy exact internal checkpoint metadata differs");
        const auto rowCount = StoredRows(serializer, "cell_ids");
        const auto range = EvenSplitRange(mpi.rank, mpi.size, rowCount);
        const Serializer::ArrayGlobalOffset shard{range.second - range.first, range.first};
        auto ids = Runtime::ReadArray<index, Runtime::FixedRows<1>>(serializer, "cell_ids", mpi, shard);
        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        const auto route = Runtime::BuildRowTransferById(context, ids, CellIds(*this), true);
        const auto transfer = [&](const auto &source, auto &destination)
        {
            using T = typename std::decay_t<decltype(source)>::value_type;
            Runtime::PayloadPlan(route, source.Layout(), destination.Layout(), Runtime::ElementSchema::For<T>())
                .Bind(source, destination, Runtime::TransferTransport::Direct,
                      {Runtime::TransferMode::Transient, context, 0})
                .Begin()
                .Wait();
        };
        const auto &time = metadata.at("time_state");
        result->time.schemaVersion = time.at("schema_version").get<int>();
        result->time.method = time.at("method").get<std::string>();
        result->time.pendingDt = time.at("pending_dt").get<real>();
        result->time.startTime = time.at("start_time").get<real>();
        result->time.pendingSteady = time.at("pending_steady").get<bool>();
        result->time.stagesActive = time.at("stages_active").get<bool>();
        result->time.stagePrepared = time.at("stage_prepared").get<bool>();
        result->time.nextStage = time.at("next_stage").get<int>();
        result->time.equation = {time.at("equation").at("inverse_dt").get<real>(),
                                 time.at("equation").at("rhs_scale").get<real>(),
                                 time.at("equation").at("time").get<real>()};
        result->time.bdfHistory = time.at("bdf_history").get<bool>();
        result->time.bdfPreviousDt = time.at("bdf_previous_dt").get<real>();
        result->time.ditrTime = time.at("ditr_time").get<real>();
        result->time.ditrDt = time.at("ditr_dt").get<real>();
        if (time.contains("ditr_settings"))
        {
            const auto &parameters = time.at("ditr_settings");
            const auto variant = parameters.at("variant").get<std::string>();
            DNDS_check_throw_info(variant == "U2R1" || variant == "U2R2",
                                  "unknown exact-checkpoint DITR variant");
            result->time.ditrSettings = ODE::DITRSettings{
                parameters.at("fraction").get<real>(), parameters.at("theta").get<real>(),
                variant == "U2R1" ? ODE::DITRVariant::U2R1 : ODE::DITRVariant::U2R2};
        }
        std::set<std::string> payloadNames;
        index payloadOrdinal = 0;
        for (const auto &payload : time.at("payloads"))
        {
            const auto name = payload.at("name").get<std::string>();
            DNDS_check_throw_info(!name.empty() && payloadNames.insert(name).second,
                                  "duplicate exact time-checkpoint payload");
            auto archive = StateArchive::Read(serializer,
                                              "internal_time_" + std::to_string(payloadOrdinal) + "_field_",
                                              payload.at("storage"), mpi, shard);
            std::optional<EquationDofs> field;
            MPI::LocalStage(mpi, [&]
                            {
                field.emplace(EquationDofOperations::NewVector(_state.trial));
                DNDS_check_throw_info(archive.Compatible(*field), "time-checkpoint equation payload differs"); });
            archive.Restore(*field, route, context, _execution.ComputeStream().Context());
            result->time.names.push_back(name);
            result->time.fields.push_back(std::move(*field));
            ++payloadOrdinal;
        }
        const auto restoreReconstruction = [&](const char *prefix, const Json &description)
        {
            auto archive = StateArchive::Read(serializer, prefix, description, mpi, shard);
            std::optional<EquationDofs> field;
            MPI::LocalStage(mpi, [&]
                            {
                field.emplace(EquationDofOperations::NewVector(_problem.finiteVolume.ReconstructionGuess(*_state.reconstruction)));
                DNDS_check_throw_info(archive.Compatible(*field), "internal reconstruction payload differs"); });
            archive.Restore(*field, route, context, _execution.ComputeStream().Context());
            return std::move(*field);
        };
        if (polynomial)
        {
            result->endpointWarm.emplace(restoreReconstruction("internal_endpoint_warm_field_", metadata.at("endpoint_warm_storage")));
            result->endpointFinal.emplace(restoreReconstruction("internal_endpoint_final_field_", metadata.at("endpoint_final_storage")));
        }
        const auto entries = serializer->ListCurrentPath();
        const auto restoreCellArray = [&](const std::string &name, index width) -> std::optional<Runtime::Array<real>>
        {
            if (!entries.count(name))
                return std::nullopt;
            auto source = Runtime::ReadArray<real>(serializer, name, mpi, shard);
            std::optional<Runtime::Array<real>> destination;
            MPI::LocalStage(mpi, [&]
                            { destination.emplace(Runtime::RowLayout::Dense(_problem.finiteVolume.Metrics().NumCell(), width)); });
            transfer(source, *destination);
            if (_state.trial.Device() != Runtime::Device::Host())
                MPI::LocalStage(mpi, [&]
                                { destination = destination->To(_state.trial.Device(), _execution.ComputeStream().Context()); });
            return destination;
        };
        if (polynomial)
            result->endpointHistory = restoreCellArray("internal_endpoint_limited_vr_history", 2);
        result->ditr = result->time.method == "ditr";
        if (result->ditr)
            for (int stage = 0; stage < 2; ++stage)
                if (const auto &required = _execution.coupledSourceDerivative[stage])
                {
                    auto &field = result->coupledSourceDerivative[stage];
                    field = restoreCellArray("internal_source_derivative_" + std::to_string(stage), required->Layout().Width());
                    DNDS_check_throw_info(field, "exact DITR quadrature-source restart requires a derivative for each retained stage rate");
                }
        if (polynomial && result->ditr)
        {
            result->middleWarm.emplace(restoreReconstruction("internal_middle_warm_field_", metadata.at("middle_warm_storage")));
            result->middleFinal.emplace(restoreReconstruction("internal_middle_final_field_", metadata.at("middle_final_storage")));
            result->middleHistory = restoreCellArray("internal_middle_limited_vr_history", 2);
        }
        result->nonlinear = NonlinearSolveState::FromJson(metadata.at("nonlinear_state"));
        const auto &epoch = metadata.at("trial_epoch");
        result->trialEpoch.acceptedStep = epoch.at("accepted_step").get<uint64_t>();
        result->trialEpoch.trial = epoch.at("trial").get<uint64_t>();
        result->trialEpoch.stage = epoch.at("stage").get<uint64_t>();
        result->trialEpoch.coupledStage = epoch.at("coupled_stage").get<uint64_t>();
        result->trialEpoch.couplingIteration = epoch.at("coupling_iteration").get<uint64_t>();
        result->trialEpoch.geometry = epoch.at("geometry").get<uint64_t>();
        result->trialEpoch.schema = epoch.at("schema").get<uint64_t>();
        result->trialDt = metadata.at("trial_dt").get<real>();
        result->progressTime = metadata.at("progress_time").get<real>();
        serializer->CloseFile();
        MPI::LocalStage(mpi, [&]
                        {
            DNDS_check_throw_info(result->trialEpoch.trial > 0 && std::isfinite(result->trialDt) &&
                                      result->trialDt >= 0 && std::isfinite(result->progressTime),
                                  "invalid exact internal checkpoint cursor");
            const auto found = std::find(result->time.names.begin(), result->time.names.end(), "trial");
            DNDS_check_throw_info(found != result->time.names.end() &&
                                      Admissible(result->time.fields[size_t(found - result->time.names.begin())]),
                                  "exact internal checkpoint trial is inadmissible");
            auto validationTrial = EquationDofOperations::NewVector(_state.trial);
            auto validationStepper = _state.timeStepper->Fresh();
            validationStepper->RestoreActive(result->time, validationTrial);
            _execution.Synchronize(); });
        return result;
    }
}
