#include "Cauchy/IO/AsyncOutput.hpp"
#ifdef DNDS_USE_CUDA
#    include "DNDS/Runtime/Execution/CUDAStream.hpp"
#endif
#include "Cauchy/IO/OutputQueue.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>

namespace DNDS::Cauchy
{
    namespace
    {
        struct DomainFiles
        {
            std::shared_ptr<Runtime::CommunicationContext> communication;
            std::optional<PreparedCheckpoint> checkpoint;
            std::optional<PreparedVisualization> visualization;
            std::string checkpointPath, visualizationPrefix;
        };
        struct OutputBatch
        {
            std::shared_ptr<Runtime::CommunicationContext> control;
            std::vector<DomainFiles> domains;
            Json resolved, outputController, domainPaths, snapshotManifest;
            std::string checkpointPath, temporaryManifest, snapshotPath;
            real time = 0;
            index step = 0;
            bool checkpoint = false, direct = false;
            void Write() const
            {
                MPI::SolverFailureScope failureScope(control->Communicator());
                for (const auto &domain : domains)
                {
                    if (domain.checkpoint)
                        domain.checkpoint->Write(domain.checkpointPath, resolved, direct ? outputController : Json());
                    if (domain.visualization)
                        domain.visualization->Write(domain.visualizationPrefix, domain.communication);
                }
                const MPIInfo mpi(control->Communicator());
                // All local file handles are closed before global publication.
                // On exception the worker aborts; never agree to recover while
                // another domain may still be inside collective file I/O.
                MPI_Barrier(mpi.comm);
                if (checkpoint && !direct)
                {
                    Serializer::SerializerH5 serializer(mpi);
                    serializer.OpenFile(temporaryManifest, false);
                    serializer.WriteInt("cauchy_simulation_format", 1);
                    serializer.WriteString("domains_json", domainPaths.dump());
                    serializer.WriteReal("accepted_time", time);
                    serializer.WriteIndex("accepted_step", step);
                    if (!outputController.is_null())
                        serializer.WriteString("output_controller_json", outputController.dump());
                    serializer.CloseFile();
                    MPI::LocalStage(mpi, [&]
                                    {
                        if (mpi.rank == 0) std::filesystem::rename(temporaryManifest, checkpointPath); });
                }
                MPI::LocalStage(mpi, [&]
                                {
                    if (mpi.rank != 0) return;
                    std::ofstream file(snapshotPath);
                    file << snapshotManifest.dump(2) << '\n';
                    file.close();
                    DNDS_check_throw_info(bool(file), "cannot publish asynchronous snapshot manifest"); });
            }
        };
    }
    struct AsyncOutput::Data
    {
        std::shared_ptr<Runtime::CommunicationContext> control;
        std::vector<std::shared_ptr<Runtime::CommunicationContext>> domains;
        std::vector<Runtime::ExecutionContext> writerCopy;
        // Destroy/join worker before releasing its MPI ordering contexts.
        std::unique_ptr<OutputQueue> queue;
        explicit Data(Simulation &simulation)
        {
            const MPIInfo mpi(simulation.Placement().Control());
            MPI::LocalStage(mpi, [&]
                            {
                int provided = MPI_THREAD_SINGLE;
                DNDS_check_throw_info(MPI_Query_thread(&provided) == MPI_SUCCESS && provided == MPI_THREAD_MULTIPLE,
                                      "asynchronous parallel HDF5 output requires MPI_THREAD_MULTIPLE; use synchronous output otherwise"); });
            control = Runtime::CommunicationContext::Duplicate(mpi.comm);
            domains.resize(size_t(simulation.Placement().DomainNumber()));
            writerCopy.resize(domains.size(), Runtime::ExecutionContext::Host());
            for (index iDomain = 0; iDomain < index(domains.size()); ++iDomain)
                if (simulation.Local(iDomain))
                {
                    const auto &metrics = simulation.GetDomain(iDomain).Discretization().Metrics();
                    domains[iDomain] = Runtime::CommunicationContext::Duplicate(metrics.Communicator());
#ifdef DNDS_USE_CUDA
                    if (metrics.Device().space == Runtime::MemorySpace::CUDA)
                        writerCopy[iDomain] = {metrics.Device(), Runtime::CreateCUDAStream(metrics.Device().ordinal)};
#endif
                }
            queue = std::make_unique<OutputQueue>([owner = control](std::exception_ptr error)
                                                  {
                // MPI thread support was checked before the worker existed.
                // AbortCaught prints locally and calls MPI_Abort independently;
                // no agreement or distributed exception recovery is attempted.
                try { std::rethrow_exception(error); }
                catch (...) { MPI::AbortCaught(owner->Communicator(), "asynchronous Cauchy output failed"); } });
        }
    };
    AsyncOutput::AsyncOutput(Simulation &simulation) : _data(std::make_unique<Data>(simulation)) {}
    AsyncOutput::~AsyncOutput() = default;
    void AsyncOutput::Drain() { _data->queue->Drain(); }
    void AsyncOutput::SubmitInternal(Simulation &simulation, const SimulationConfig &config, const OutputPlan &plan,
                                     index iDomain, const std::string &token, const NonlinearSolveState &state,
                                     bool checkpoint, bool visualization, Json outputController, Json snapshotManifest)
    {
        _data->queue->Enqueue([&]() -> OutputQueue::Action
                              {
            auto batch = std::make_shared<OutputBatch>();
            // A dislocated domain may emit independently. Its completion barrier
            // and manifest publication must never enter the simulation control communicator.
            batch->control = _data->domains.at(iDomain);
            batch->direct = true;
            batch->checkpoint = checkpoint;
            batch->resolved = config.Root();
            batch->snapshotManifest = std::move(snapshotManifest);
            batch->outputController = std::move(outputController);
            batch->snapshotPath = plan.InternalSnapshotManifestPath(iDomain, token);
            DomainFiles files;
            files.communication = batch->control;
            auto &domain = simulation.GetDomain(iDomain);
            if (visualization)
            {
                files.visualizationPrefix = plan.VisualizationPrefix(iDomain, token);
                files.visualization.emplace(domain.CaptureInternalVisualization(config.Output().hdfDeflateLevel));
            }
            if (checkpoint)
            {
                files.checkpointPath = plan.InternalCheckpointPath(iDomain, token);
                files.checkpoint.emplace(domain.CaptureInternalCheckpoint(files.communication, state, true, _data->writerCopy[iDomain]));
            }
            batch->domains.push_back(std::move(files));
            return [batch] { batch->Write(); }; });
    }
    void AsyncOutput::Submit(Simulation &simulation, const SimulationConfig &config, const OutputPlan &plan,
                             const std::string &token, bool checkpoint, bool visualization, Json outputController, Json snapshotManifest)
    {
        _data->queue->Enqueue([&, controller = std::move(outputController), manifest = std::move(snapshotManifest)]() mutable -> OutputQueue::Action
                              {
            auto batch = std::make_shared<OutputBatch>();
            batch->control = _data->control;
            batch->resolved = config.Root();
            batch->outputController = std::move(controller);
            batch->snapshotManifest = std::move(manifest);
            batch->snapshotPath = plan.SnapshotManifestPath(token);
            batch->checkpointPath = plan.CheckpointPath(token);
            batch->checkpoint = checkpoint;
            batch->time = simulation.Time();
            batch->step = index(simulation.Step());
            const auto &placement = simulation.Placement();
            batch->direct = placement.DomainNumber() == 1 && placement.Group(0).ranks.size() == size_t(placement.Size());
            const MPIInfo control(placement.Control());
            MPI::LocalStage(control, [&]
            {
                for (index iDomain = 0; iDomain < placement.DomainNumber(); ++iDomain)
                    if (simulation.Local(iDomain)) simulation.GetDomain(iDomain).RequireAcceptedClock(batch->time, simulation.Step());
            });
            std::filesystem::path generation;
            const std::filesystem::path archive(batch->checkpointPath);
            if (checkpoint && !batch->direct)
            {
                uint64_t identifier = control.rank == 0 ? uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count()) : 0;
                MPI_Bcast(&identifier, 1, MPI_UINT64_T, 0, control.comm);
                generation = archive.filename().string() + ".parts/" + std::to_string(identifier);
                MPI::LocalStage(control, [&]
                {
                    if (control.rank != 0) return;
                    DNDS_check_throw_info(!std::filesystem::exists(archive.parent_path() / generation), "checkpoint generation already exists");
                    std::filesystem::create_directories(archive.parent_path() / generation);
                });
                batch->temporaryManifest = (archive.parent_path() / generation / "manifest.h5").string();
                batch->domainPaths = Json::array();
                for (index iDomain = 0; iDomain < placement.DomainNumber(); ++iDomain)
                    batch->domainPaths.push_back({{"name", placement.Group(iDomain).name},
                        {"file", (generation / ("domain" + std::to_string(iDomain) + ".h5")).generic_string()}});
            }
            for (index iDomain = 0; iDomain < placement.DomainNumber(); ++iDomain)
            {
                if (!simulation.Local(iDomain)) continue;
                auto &domain = simulation.GetDomain(iDomain);
                DomainFiles files;
                files.communication = _data->domains[iDomain];
                if (checkpoint)
                {
                    files.checkpoint.emplace(domain.CaptureCheckpoint(files.communication, {config.Output().checkpointReconstruction, true}, _data->writerCopy[iDomain]));
                    files.checkpointPath = batch->direct ? batch->checkpointPath
                        : (archive.parent_path() / batch->domainPaths[iDomain]["file"].get<std::string>()).string();
                }
                if (visualization)
                {
                    files.visualization.emplace(domain.PrepareVisualization(config.Output().hdfDeflateLevel));
                    files.visualizationPrefix = plan.VisualizationPrefix(iDomain, token);
                }
                batch->domains.push_back(std::move(files));
            }
            // Only the immutable batch escapes. No reference to the Simulation,
            // Domain, OutputPlan, or this service is captured by the writer.
            return [batch] { batch->Write(); }; });
    }
}
