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
    void Domain::ReadCheckpoint(const std::string &path, const Json &resolved)
    {
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        bool internal = false;
        {
            auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
            serializer->OpenFile(path, true);
            internal = serializer->ListCurrentPath().count("cauchy_internal_format");
            serializer->CloseFile();
        }
        auto checkpoint = PrepareCheckpointRead(path, resolved);
        auto active = internal ? PrepareInternalCheckpointRead(path, resolved) : nullptr;
        ReserveCheckpointRead(*checkpoint);
        CommitCheckpointRead(*checkpoint);
        if (active)
            CommitInternalCheckpointRead(*active);
    }
    std::unique_ptr<IO::DomainCheckpointRead> Domain::PrepareCheckpointRead(const std::string &path, const Json &resolved)
    {
        _execution.Synchronize();
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        std::unique_ptr<IO::DomainCheckpointRead> checkpoint;
        MPI::LocalStage(mpi, [&]
                        { checkpoint = std::make_unique<IO::DomainCheckpointRead>(); });
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, true);
        int version = 0;
        std::string config, name;
        real time = 0, lastDt = 0;
        index step = 0;
        serializer->ReadInt("cauchy_format", version);
        serializer->ReadString("resolved_config", config);
        serializer->ReadString("domain_name", name);
        serializer->ReadReal("accepted_time", time);
        serializer->ReadIndex("accepted_step", step);
        DNDS_check_throw_info(version >= 2 && version <= 5, "unsupported Cauchy checkpoint format");
        const auto rowCount = StoredRows(serializer, "cell_ids");
        const auto range = EvenSplitRange(mpi.rank, mpi.size, rowCount);
        const Serializer::ArrayGlobalOffset shard{range.second - range.first, range.first};
        auto ids = Runtime::ReadArray<index, Runtime::FixedRows<1>>(serializer, "cell_ids", mpi, shard);
        auto topology = Runtime::ReadArray<index>(serializer, "mesh_topology", mpi, shard);
        auto coordinates = Runtime::ReadArray<real>(serializer, "mesh_coordinates", mpi, shard);
        auto transforms = Runtime::ReadArray<real>(serializer, "periodic_transforms", mpi, {1, 0});
        std::optional<Runtime::Array<real>> flow;
        if (version < 5)
            flow.emplace(Runtime::ReadArray<real, Runtime::FixedRows<5>>(serializer, "flow", mpi, shard));
        Json metadata;
        if (version >= 4)
        {
            std::string serialized;
            serializer->ReadString("metadata_json", serialized);
            metadata = Json::parse(serialized, nullptr, false);
        }
        std::optional<Runtime::Array<real>> reconstruction;
        const auto entries = serializer->ListCurrentPath();
        if (entries.count("last_accepted_dt"))
            serializer->ReadReal("last_accepted_dt", lastDt);
        if (entries.count("reconstruction_guess"))
            reconstruction.emplace(Runtime::ReadArray<real>(serializer, "reconstruction_guess", mpi, shard));
        std::optional<Runtime::Array<real>> limitedVRHistory;
        if (entries.count("limited_vr_history"))
            limitedVRHistory.emplace(Runtime::ReadArray<real>(serializer, "limited_vr_history", mpi, shard));
        std::optional<Runtime::Array<real>> history;
        real historyDt = 0;
        if (version >= 4 && metadata.is_object() && metadata.value("history", false))
        {
            serializer->ReadReal("history_dt", historyDt);
            if (version < 5)
                history.emplace(Runtime::ReadArray<real, Runtime::FixedRows<5>>(serializer, "history_flow", mpi, shard));
        }
        std::optional<StateArchive> stateArchive, reconstructionArchive, historyArchive;
        if (version == 5)
        {
            stateArchive.emplace(StateArchive::Read(serializer, "state_field_", metadata.at("state_storage"), mpi, shard));
            if (metadata.contains("reconstruction_storage"))
                reconstructionArchive.emplace(StateArchive::Read(serializer, "reconstruction_field_", metadata.at("reconstruction_storage"), mpi, shard));
            if (metadata.value("history", false))
                historyArchive.emplace(StateArchive::Read(serializer, "history_field_", metadata.at("state_storage"), mpi, shard));
        }
        serializer->CloseFile();
        // Older BDF2 files carry the last physical dt with their history.
        // Other old files leave it unknown (zero): the first resumed step has
        // no growth cap, then its actual accepted dt establishes the reference.
        if (!entries.count("last_accepted_dt") && (history || historyArchive))
            lastDt = historyDt;
        DNDS_check_throw_info(std::isfinite(lastDt) && lastDt >= 0, "invalid stored last accepted timestep");
        if (std::isfinite(time) && time >= 0 && lastDt > time)
            DNDS_check_throw_info(lastDt - time <= 1024 * std::numeric_limits<real>::epsilon() * std::max(lastDt, time),
                                  "stored last accepted timestep exceeds the physical clock");
        if (entries.count("last_accepted_dt") && (history || historyArchive))
            DNDS_check_throw_info(lastDt == historyDt, "stored last accepted timestep conflicts with BDF history");
        auto storedConfig = Json::parse(config, nullptr, false);
        // Before the first step a mean-only snapshot is initial data. Its
        // producer has never applied a nonlinear update policy. The consumer
        // initializes its own method (including both DITR stages) from it.
        const bool initialMeansOnly = step == 0 && time == 0 && lastDt == 0 &&
                                      !entries.count("cauchy_internal_format") && !history && !historyArchive &&
                                      !reconstruction && !reconstructionArchive && !limitedVRHistory;
        int valid = name == _problem.name && !storedConfig.is_discarded() && CompatibleConfig(storedConfig) == CompatibleConfig(resolved) && std::isfinite(time) && time >= 0 && step >= 0 &&
                    (stateArchive ? stateArchive->Compatible(_state.trial) && stateArchive->fields[0].RowCount() == ids.RowCount()
                                  : flow && _state.trial.ComponentNumber() == 5 && _state.trial.PhysicalFieldNumber() == 1 &&
                                        flow->RowCount() == ids.RowCount() && flow->Layout().Width() == 5);
        valid = valid && (version < 4 || (metadata.is_object() && metadata.value("schema_version", 0) == 1 &&
                                          CompatibleEquationMetadata(metadata.value("flow", Json())) == CompatibleEquationMetadata(EquationMetadata(_problem.equation)) &&
                                          (initialMeansOnly || !metadata.contains("update_policy") || metadata.at("update_policy") == UpdatePolicyMetadata())));
        valid = valid && CompatibleFrame(metadata, _problem.finiteVolume, std::isfinite(time) ? time : 0);
        valid = valid && (!limitedVRHistory || ((reconstruction || reconstructionArchive) && limitedVRHistory->RowCount() == ids.RowCount() &&
                                                !limitedVRHistory->Layout().IsVariable() && limitedVRHistory->Layout().Width() == 2));
        valid = valid && (version < 4 || (metadata.is_object() && metadata.value("limited_vr_history", false) == bool(limitedVRHistory) &&
                                          (!limitedVRHistory || metadata.value("limited_vr_history_schema", 0) == 1)));
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, mpi.comm);
        if (!valid)
        {
            const auto differences = storedConfig.is_discarded() ? nlohmann::json() : nlohmann::json::diff(CompatibleConfig(storedConfig), CompatibleConfig(resolved));
            DNDS_check_throw_info(false, "Cauchy restart config/equation schema mismatch; normalized config differences: " + differences.dump());
        }
        const int meshEnvelope = version >= 4 ? metadata.value("mesh_envelope", 1) : 1;
        DNDS_check_throw_info(meshEnvelope == 1 || meshEnvelope == 2, "unsupported Cauchy mesh envelope");
        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        const auto targetIds = CellIds(*this);
        const auto route = Runtime::BuildRowTransferById(context, ids, targetIds, true);
        const auto transfer = [&](const auto &source, auto &destination)
        {
            using T = typename std::decay_t<decltype(source)>::value_type;
            std::optional<Runtime::Array<T>> host;
            MPI::LocalStage(mpi, [&]
                            {
                if (destination.GetDevice() != Runtime::Device::Host())
                    host.emplace(destination.Layout()); });
            auto &target = host ? *host : destination;
            Runtime::PayloadPlan(route, source.Layout(), target.Layout(), Runtime::ElementSchema::For<T>())
                .Bind(source, target, Runtime::TransferTransport::Direct, {Runtime::TransferMode::Transient, context, 0})
                .Begin()
                .Wait();
            if (host)
                MPI::LocalStage(mpi, [&]
                                { destination.CopyFrom(*host, _execution.ComputeStream().Context()).Wait(); });
        };
        std::optional<MeshEnvelope> envelope;
        std::optional<Runtime::Array<index>> mappedTopology;
        std::optional<Runtime::Array<real>> mappedCoordinates;
        MPI::LocalStage(mpi, [&]
                        { envelope.emplace(DescribeMesh(*this)); });
        const auto topologyLayout = route.DestinationLayout(topology.Layout());
        const auto coordinateLayout = route.DestinationLayout(coordinates.Layout());
        MPI::LocalStage(mpi, [&]
                        {
            mappedTopology.emplace(topologyLayout);
            mappedCoordinates.emplace(coordinateLayout); });
        transfer(topology, *mappedTopology);
        transfer(coordinates, *mappedCoordinates);
        MPI::LocalStage(mpi, [&]
                        {
            const auto topologyDifference = TopologyDifference(*mappedTopology, envelope->topology, targetIds, meshEnvelope == 1);
            const auto coordinateDifference = RowDifference(*mappedCoordinates, envelope->coordinates, targetIds);
            DNDS_check_throw_info(topologyDifference.empty() && coordinateDifference.empty() && SameRows(transforms, envelope->periodicTransforms),
                                  "Cauchy restart persistent-cell mesh topology/geometry mismatch: topology " + topologyDifference + "; coordinates " + coordinateDifference); });
        _state.trialReady = false;
        // Extended state shares the same mesh/ID validation and publication
        // transaction as historical flow5. No accepted field is changed until
        // means, optional reconstruction and BDF history have all been checked.
        if (stateArchive)
        {
            stateArchive->Restore(_state.trial, route, context, _execution.ComputeStream().Context());
            DNDS_check_throw_info(Admissible(_state.trial), "Cauchy restart contains inadmissible equation states");
            std::optional<EquationDofs> guess;
            if (_problem.finiteVolume.PolynomialReconstruction())
                MPI::LocalStage(mpi, [&]
                                { guess.emplace(EquationDofOperations::NewVector(_problem.finiteVolume.ReconstructionGuess(*_state.reconstruction))); });
            const bool needsLimitedVRHistory = bool(_problem.finiteVolume.LimitedVRHistory(*_state.reconstruction));
            int compatible = guess && reconstructionArchive && reconstructionArchive->Compatible(*guess) &&
                             metadata.value("reconstruction", Json()) == _problem.finiteVolume.ReconstructionMetadata() &&
                             (!needsLimitedVRHistory || limitedVRHistory);
            compatible = MPI::AllreduceValue<MPI::Reduction::Min>(compatible, mpi.comm);
            std::optional<Runtime::Array<real>> mappedLimitedVRHistory;
            if (compatible)
            {
                reconstructionArchive->Restore(*guess, route, context, _execution.ComputeStream().Context());
                if (needsLimitedVRHistory)
                {
                    MPI::LocalStage(mpi, [&]
                                    { mappedLimitedVRHistory.emplace(Runtime::RowLayout::Dense(_problem.finiteVolume.Metrics().NumCell(), 2), _state.trial.Device()); });
                    transfer(*limitedVRHistory, *mappedLimitedVRHistory);
                }
            }
            else if (guess)
            {
                EquationDofOperations::Fill(_execution.Engine(), _execution.ComputeStream(), *guess, 0);
                _execution.Synchronize();
            }
            std::unique_ptr<EquationTimeStepper> restoredStepper;
            if (_state.timeStepper)
            {
                MPI::LocalStage(mpi, [&]
                                { restoredStepper = _state.timeStepper->Fresh(); });
                if (!_sourceStrang && historyArchive && metadata.value("time_method", std::string{}) == restoredStepper->Method() && restoredStepper->Method() == "bdf2")
                {
                    std::optional<EquationDofs> previous;
                    MPI::LocalStage(mpi, [&]
                                    {
                        DNDS_check_throw_info(std::isfinite(historyDt) && historyDt > 0, "invalid equation restart history timestep");
                        previous.emplace(EquationDofOperations::NewVector(_state.trial)); });
                    historyArchive->Restore(*previous, route, context, _execution.ComputeStream().Context());
                    DNDS_check_throw_info(Admissible(*previous), "inadmissible equation restart history");
                    MPI::LocalStage(mpi, [&]
                                    { restoredStepper->RestoreHistory(*previous, historyDt); });
                }
            }
            checkpoint->time = time;
            checkpoint->lastDt = lastDt;
            checkpoint->epoch.acceptedStep = uint64_t(step);
            checkpoint->stepper = std::move(restoredStepper);
            checkpoint->equationGuess = std::move(guess);
            checkpoint->limitedVRHistory = std::move(mappedLimitedVRHistory);
            return checkpoint;
        }
        // Existing trial allocations and binding identities remain stable.
        auto &trial = _state.trial.CoupledField();
        transfer(*flow, trial.Owned());
        DNDS_check_throw_info(Admissible(_state.trial), "Cauchy restart contains inadmissible states");
        std::optional<EquationDofs> guess;
        std::optional<Runtime::Array<real>> mappedLimitedVRHistory;
        // Means-only/direct reconstructions have no coefficient destination.
        // A modal payload in an older archive is simply not applicable; do
        // not manufacture a scratch/alias to run the modal restore branch.
        if (_problem.finiteVolume.PolynomialReconstruction())
        {
            MPI::LocalStage(mpi, [&]
                            { guess.emplace(EquationDofOperations::NewVector(_problem.finiteVolume.ReconstructionGuess(*_state.reconstruction))); });
            auto &guessArray = guess->CoupledField().Owned();
            int compatibleReconstruction = bool(reconstruction) && (version < 4 || metadata.value("reconstruction", Json()) == _problem.finiteVolume.ReconstructionMetadata());
            const bool needsLimitedVRHistory = _problem.finiteVolume.LimitedVRHistory(*_state.reconstruction).has_value();
            compatibleReconstruction = compatibleReconstruction && (!needsLimitedVRHistory || limitedVRHistory);
            if (reconstruction)
                for (index iRow = 0; iRow < reconstruction->RowCount(); ++iRow)
                    compatibleReconstruction = compatibleReconstruction && reconstruction->RowSize(iRow) == guessArray.Layout().Width();
            compatibleReconstruction = MPI::AllreduceValue<MPI::Reduction::Min>(
                compatibleReconstruction, mpi.comm);
            if (compatibleReconstruction)
            {
                transfer(*reconstruction, guessArray);
                if (needsLimitedVRHistory)
                {
                    MPI::LocalStage(mpi, [&]
                                    { mappedLimitedVRHistory.emplace(Runtime::RowLayout::Dense(_problem.finiteVolume.Metrics().NumCell(), 2), _state.trial.Device()); });
                    transfer(*limitedVRHistory, *mappedLimitedVRHistory);
                }
            }
            else
            {
                EquationDofOperations::Fill(_execution.Engine(), _execution.ComputeStream(), *guess, 0);
                _execution.Synchronize();
            }
        }
        std::unique_ptr<EquationTimeStepper> restoredStepper;
        std::optional<CFV::FVField> restoredHistory;
        if (_state.timeStepper)
        {
            MPI::LocalStage(mpi, [&]
                            { restoredStepper = _state.timeStepper->Fresh(); });
            if (history && metadata.value("time_method", std::string{}) == restoredStepper->Method() && restoredStepper->Method() == "bdf2")
            {
                MPI::LocalStage(mpi, [&]
                                {
                    DNDS_check_throw_info(std::isfinite(historyDt) && historyDt > 0, "invalid Cauchy restart history timestep");
                    restoredHistory.emplace(_problem.finiteVolume.Metrics().BuildField(_state.accepted.CoupledField().Owned().Layout(), _state.trial.Device())); });
                transfer(*history, restoredHistory->Owned());
                const auto restoredState = _state.trial.RebindCoupled(*restoredHistory);
                DNDS_check_throw_info(Admissible(restoredState), "inadmissible Cauchy restart history");
                MPI::LocalStage(mpi, [&]
                                { restoredStepper->RestoreHistory(restoredState, historyDt); });
            }
        }
        checkpoint->time = time;
        checkpoint->lastDt = lastDt;
        checkpoint->epoch.acceptedStep = uint64_t(step);
        checkpoint->stepper = std::move(restoredStepper);
        checkpoint->equationGuess = std::move(guess);
        checkpoint->limitedVRHistory = std::move(mappedLimitedVRHistory);
        return checkpoint;
    }
}
