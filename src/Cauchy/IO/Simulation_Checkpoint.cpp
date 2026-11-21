#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Core/ConnectedPressureGauge.hpp"
#include "Cauchy/IO/CheckpointMetadata.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <filesystem>
#include <chrono>
#include <climits>

namespace DNDS::Cauchy
{
    namespace
    {
        // Actions may perform collective I/O. Do not catch and continue to a
        // control agreement: peers may not have left the failing domain action.
        template <class Action>
        void LocalDomains(const DomainPlacement &placement, Action &&action)
        {
            for (index iDomain = 0; iDomain < placement.DomainNumber(); ++iDomain)
                if (placement.DomainRank(iDomain) >= 0)
                    action(iDomain);
        }
        void AgreePath(const MPIInfo &mpi, const std::string &path)
        {
            std::string reference;
            int length = 0;
            MPI::LocalStage(mpi, [&]
                            {
                DNDS_check_throw_info(!path.empty() && path.size() <= size_t(INT_MAX), "invalid Cauchy archive path");
                if (mpi.rank == 0) length = int(path.size()); });
            MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
            MPI::LocalStage(mpi, [&]
                            { reference = mpi.rank == 0 ? path : std::string(size_t(length), '\0'); });
            MPI_Bcast(reference.data(), length, MPI_CHAR, 0, mpi.comm);
            MPI::LocalStage(mpi, [&]
                            { DNDS_check_throw_info(path == reference, "Cauchy archive paths differ between control ranks"); });
        }
        void AgreeJson(const MPIInfo &mpi, const Json &value)
        {
            const auto encoded = value.dump();
            int length = 0;
            MPI::LocalStage(mpi, [&]
                            {
                DNDS_check_throw_info(encoded.size() <= size_t(INT_MAX), "Cauchy checkpoint controller metadata is too large");
                if (mpi.rank == 0) length = int(encoded.size()); });
            MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
            std::string reference;
            MPI::LocalStage(mpi, [&]
                            { reference = mpi.rank == 0 ? encoded : std::string(size_t(length), '\0'); });
            MPI_Bcast(reference.data(), length, MPI_CHAR, 0, mpi.comm);
            MPI::LocalStage(mpi, [&]
                            { DNDS_check_throw_info(reference == encoded, "Cauchy checkpoint controller metadata differs between ranks"); });
        }
        bool DirectArchive(const DomainPlacement &placement)
        {
            return placement.DomainNumber() == 1 && placement.Group(0).ranks.size() == size_t(placement.Size());
        }
    }
    void Simulation::WriteCheckpoint(const std::string &path, bool includeReconstruction,
                                     Json outputController) const
    {
        const MPIInfo control(_placement.Control());
        AgreePath(control, path);
        AgreeJson(control, outputController);
        int flags[]{int(includeReconstruction), -int(includeReconstruction)};
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<int>{flags, 2}, control.comm);
        DNDS_check_throw_info(flags[0] == -flags[1], "simulation checkpoint reconstruction selection differs between ranks");
        MPI::LocalStage(control, [&]
                        {
            for (const auto &domain : _domains)
                if (domain) DNDS_check_throw_info(domain->Time() == _time && domain->Epoch().acceptedStep == _step,
                                                  "domain was advanced outside its Simulation before checkpoint"); });
        if (DirectArchive(_placement))
        {
            GetDomain(0).WriteCheckpoint(path, _config.Root(), includeReconstruction, outputController);
            return;
        }
        // Never overwrite children referenced by a previously published
        // manifest. Retain older generations; cleanup is an explicit user task.
        const std::filesystem::path manifest(path);
        std::filesystem::path generation;
        uint64_t identifier = control.rank == 0 ? uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count()) : 0;
        MPI_Bcast(&identifier, 1, MPI_UINT64_T, 0, control.comm);
        MPI::LocalStage(control, [&]
                        {
            generation = manifest.filename().string() + ".parts/" + std::to_string(identifier);
            if (control.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(manifest.parent_path() / generation), "checkpoint generation already exists");
                std::filesystem::create_directories(manifest.parent_path() / generation);
            } });
        Json domains = Json::array();
        MPI::LocalStage(control, [&]
                        {
            for (index iDomain = 0; iDomain < _placement.DomainNumber(); ++iDomain)
                domains.push_back({{"name", _placement.Group(iDomain).name}, {"file", (generation / ("domain" + std::to_string(iDomain) + ".h5")).generic_string()}}); });
        LocalDomains(_placement, [&](index iDomain)
                     { GetDomain(iDomain).WriteCheckpoint((manifest.parent_path() / domains[iDomain]["file"].get<std::string>()).string(),
                                                          _config.Root(), includeReconstruction); });
        const auto temporary = manifest.parent_path() / generation / "manifest.h5";
        Serializer::SerializerH5 serializer(control);
        serializer.OpenFile(temporary.string(), false);
        serializer.WriteInt("cauchy_simulation_format", 1);
        serializer.WriteString("domains_json", domains.dump());
        serializer.WriteReal("accepted_time", _time);
        serializer.WriteIndex("accepted_step", index(_step));
        if (!outputController.is_null())
            serializer.WriteString("output_controller_json", outputController.dump());
        serializer.CloseFile();
        MPI::LocalStage(control, [&]
                        { if (control.rank == 0) std::filesystem::rename(temporary, manifest); });
    }
    void Simulation::ReadCheckpoint(const std::string &path, CheckpointLoadMode mode)
    {
        const MPIInfo control(_placement.Control());
        AgreePath(control, path);
        CheckpointMetadata::AgreeLoadMode(control, mode);
        const bool resume = mode == CheckpointLoadMode::Resume;
        const std::filesystem::path manifest(path);
        // Fail missing files collectively before entering the HDF5 backend,
        // whose low-level open failure is an assertion rather than an exception.
        MPI::LocalStage(control, [&]
                        { DNDS_check_throw_info(std::filesystem::is_regular_file(manifest), "Cauchy checkpoint does not exist: " + path); });
        std::vector<std::string> paths;
        std::optional<Json> outputController;
        real time = 0;
        index step = 0;
        bool direct = false;
        Serializer::SerializerH5 serializer(control);
        serializer.OpenFile(path, true);
        const auto keys = serializer.ListCurrentPath();
        direct = keys.count("cauchy_format");
        const bool internal = keys.count("cauchy_internal_format");
        MPI::LocalStage(control, [&]
                        { DNDS_check_throw_info(keys.count("accepted_time") && keys.count("accepted_step") &&
                                                    (direct || (keys.count("cauchy_simulation_format") && keys.count("domains_json"))),
                                                "incomplete Cauchy simulation archive metadata"); });
        if (direct)
        {
            DNDS_check_throw_info(DirectArchive(_placement), "a single-domain checkpoint cannot restore a multi-domain simulation");
            DNDS_check_throw_info(!internal || _placement.DomainNumber() == 1,
                                  "exact internal checkpoint currently requires one simulation domain");
            serializer.ReadReal("accepted_time", time);
            serializer.ReadIndex("accepted_step", step);
            if (resume && keys.count("output_controller_json"))
            {
                std::string text;
                serializer.ReadString("output_controller_json", text);
                MPI::LocalStage(control, [&]
                                { outputController = Json::parse(text); });
            }
            paths.push_back(path);
        }
        else
        {
            int version = 0;
            std::string text;
            serializer.ReadInt("cauchy_simulation_format", version);
            serializer.ReadString("domains_json", text);
            serializer.ReadReal("accepted_time", time);
            serializer.ReadIndex("accepted_step", step);
            if (resume && keys.count("output_controller_json"))
            {
                std::string controllerText;
                serializer.ReadString("output_controller_json", controllerText);
                MPI::LocalStage(control, [&]
                                { outputController = Json::parse(controllerText); });
            }
            MPI::LocalStage(control, [&]
                            {
                const auto entries = Json::parse(text);
                DNDS_check_throw_info(version == 1 && entries.is_array() && entries.size() == _domains.size(), "Cauchy simulation archive domain schema differs");
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                {
                    DNDS_check_throw_info(entries[iDomain].at("name").get<std::string>() == _placement.Group(iDomain).name,
                                          "Cauchy simulation archive domain names/order differ");
                    const std::filesystem::path child(entries[iDomain].at("file").get<std::string>());
                    DNDS_check_throw_info(!child.empty() && child.is_relative(), "invalid Cauchy simulation child path");
                    paths.push_back((manifest.parent_path() / child).string());
                } });
        }
        serializer.CloseFile();
        std::vector<std::optional<Domain::CheckpointCandidate>> checkpoints;
        std::optional<Domain::InternalCheckpointCandidate> internalCheckpoint;
        MPI::LocalStage(control, [&]
                        {
            DNDS_check_throw_info(std::isfinite(time) && time >= 0 && step >= 0, "invalid Cauchy simulation archive clock");
            checkpoints.resize(_domains.size()); });
        LocalDomains(_placement, [&](index iDomain)
                     {
            MPI::LocalStage(MPIInfo(_placement.Context(iDomain)->Communicator()), [&]
                            { DNDS_check_throw_info(std::filesystem::is_regular_file(paths[iDomain]), "missing Cauchy domain checkpoint"); });
            checkpoints[iDomain].emplace(GetDomain(iDomain).PrepareCheckpoint(paths[iDomain], _config.Root(), mode));
            MPI::LocalStage(MPIInfo(_placement.Context(iDomain)->Communicator()), [&]
                            { DNDS_check_throw_info(checkpoints[iDomain]->Time() == time && checkpoints[iDomain]->Epoch().acceptedStep == uint64_t(step),
                                                    "domain checkpoint clock differs from its simulation manifest"); }); });
        if (internal && resume)
        {
            internalCheckpoint.emplace(GetDomain(0).PrepareInternalCheckpoint(path, _config.Root()));
            MPI::LocalStage(control, [&]
                            { DNDS_check_throw_info(internalCheckpoint->TrialEpoch().acceptedStep == uint64_t(step),
                                                    "internal checkpoint trial belongs to another accepted step"); });
        }
        if (_interfaces)
        {
            if (!resume)
                _pressureGauge->ProjectRestartMeans();
            // Complete the restored donor halos on the domain communicator. The
            // exchange is collective and must propagate failure directly.
            std::vector<InterfaceDomainInput> ports;
            MPI::LocalStage(control, [&]
                            { ports.resize(_domains.size()); });
            LocalDomains(_placement, [&](index iDomain)
                         {
                auto &domain = GetDomain(iDomain);
                auto &candidate = *checkpoints[iDomain];
                ports[iDomain] = {domain.PrepareCheckpointInterfaceDonor(candidate)}; });
            StateEpoch epoch;
            epoch.acceptedStep = uint64_t(step);
            _interfaces->Refresh(ports, time, epoch);
            LocalDomains(_placement, [&](index iDomain)
                         { if (const auto input = _interfaces->Inputs(iDomain)) GetDomain(iDomain).BindInterfaceInputs(*input); });
        }
        try
        {
            // All reservation work is local, including reconstruction writers.
            // This one control preflight prevents any domain from publishing if
            // another domain cannot reserve, without catching collective I/O.
            MPI::LocalStage(control, [&]
                            { LocalDomains(_placement, [&](index iDomain)
                                           { GetDomain(iDomain).ReserveCheckpoint(*checkpoints[iDomain], false); }); });
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy checkpoint reservation failed");
            // Reserved writers cancel without changing their generations.
            checkpoints.clear();
            if (_interfaces)
            {
                _interfaces->Discard();
                // Already validated accepted-packet aliases. Binding is local
                // and does not perform I/O, search, transfer or MPI agreement.
                LocalDomains(_placement, [&](index iDomain)
                             { if (const auto input = _interfaces->Inputs(iDomain)) GetDomain(iDomain).BindInterfaceInputs(*input); });
            }
            throw;
        }
        for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
            if (_domains[iDomain])
                _domains[iDomain]->CommitCheckpoint(*checkpoints[iDomain]);
        if (internalCheckpoint)
            GetDomain(0).CommitInternalCheckpoint(*internalCheckpoint);
        if (_interfaces)
            _interfaces->Commit();
        _time = time;
        _step = uint64_t(step);
        _restoredOutputController = std::move(outputController);
    }
    void Simulation::WriteVisualization(const std::string &prefix, int hdfDeflateLevel)
    {
        AgreePath(MPIInfo(_placement.Control()), prefix);
        LocalDomains(_placement, [&](index iDomain)
                     { GetDomain(iDomain).WriteVisualization(prefix + (_domains.size() == 1 ? "" : "_domain" + std::to_string(iDomain)), hdfDeflateLevel); });
    }
}
