#include "Cauchy/IO/OutputCoordinator.hpp"

#include "Cauchy/IO/ProgressOutput.hpp"
#include "Cauchy/IO/AsyncOutput.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <fmt/format.h>

#include <algorithm>
#include <climits>
#include <filesystem>
#include <fstream>

namespace DNDS::Cauchy
{
    namespace
    {
        Json BoundaryArtifact(const std::string &prefix, bool facialFluxes, real observationTime)
        {
            Json artifact{{"kind", "boundary"}, {"path", prefix + "_bnd.vtkhdf"}, {"schema", "dndsr.cauchy.boundary.v1"}, {"facialFluxes", {{"available", facialFluxes}}}};
            if (facialFluxes)
                artifact["facialFluxes"].update({{"observationTime", observationTime},
                                                 {"sign", "negative_outward_rhs_flux"},
                                                 {"normalization", "rhs_quadrature_area_average"}});
            return artifact;
        }

        Json GatherJsonArrays(const MPIInfo &mpi, const Json &local)
        {
            const auto encoded = local.dump();
            DNDS_check_throw_info(encoded.size() <= size_t(INT_MAX),
                                  "Cauchy manifest contribution is too large");
            const int count = int(encoded.size());
            std::vector<int> counts(mpi.rank == 0 ? size_t(mpi.size) : 0);
            MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
                       mpi.comm);
            std::vector<int> offsets;
            std::vector<char> bytes;
            if (mpi.rank == 0)
            {
                offsets.resize(size_t(mpi.size));
                int total = 0;
                for (int rank = 0; rank < mpi.size; ++rank)
                {
                    offsets[rank] = total;
                    DNDS_check_throw_info(
                        counts[rank] <= INT_MAX - total,
                        "Cauchy manifest gather is too large");
                    total += counts[rank];
                }
                bytes.resize(size_t(total));
            }
            MPI_Gatherv(encoded.data(), count, MPI_CHAR, bytes.data(),
                        counts.data(), offsets.data(), MPI_CHAR, 0, mpi.comm);
            Json result = Json::array();
            if (mpi.rank == 0)
                for (int rank = 0; rank < mpi.size; ++rank)
                {
                    const auto contribution = Json::parse(
                        bytes.data() + offsets[rank],
                        bytes.data() + offsets[rank] + counts[rank]);
                    for (const auto &entry : contribution)
                        result.push_back(entry);
                }
            return result;
        }
    }

    struct OutputCoordinator::Data
    {
        Simulation &simulation;
        const SimulationConfig &config;
        MPIInfo mpi;
        real started;
        std::optional<real> stepStarted;
        real stepWallSeconds = 0;
        const OutputSettings &output;
        const TimeSettings &time;
        index domains;
        OutputPlan plan;
        OutputControllerState state;
        std::shared_ptr<OperationalLogger> logger;
        std::vector<std::unique_ptr<ProgressOutputController>> progress;
        std::vector<real> baselines;
        uint64_t nextCheckpointTime = 0, nextVisualizationTime = 0;
        index internalCheckpointInterval = 0,
              internalVisualizationInterval = 0;
        bool continueOutput = false, finalized = false, allocationReportWritten = false;
        std::unique_ptr<AsyncOutput> asynchronous;

        Data(Simulation &simulationIn, const SimulationConfig &configIn,
             MPIInfo mpiIn, real startedIn)
            : simulation(simulationIn), config(configIn), mpi(mpiIn),
              started(startedIn), output(config.Output()), time(config.Time()),
              domains(index(config.Domains().size())), plan(output, domains),
              progress(size_t(domains)), baselines(size_t(domains), 0),
              internalCheckpointInterval(output.implicit.checkpointInterval),
              internalVisualizationInterval(
                  output.implicit.visualizationInterval < 0
                      ? output.implicit.checkpointInterval
                      : output.implicit.visualizationInterval)
        {
            // A dislocated domain root need not be world rank zero. Let each
            // actual history writer print its own domain's progress without
            // gathering residuals or adding collectives to the output path.
            bool domainProgressWriter = false;
            for (index iDomain = 0; iDomain < domains; ++iDomain)
                domainProgressWriter |= simulation.Local(iDomain) &&
                                        simulation.Placement().DomainRank(iDomain) == 0;
            logger = std::make_shared<OperationalLogger>(output, mpi.rank, domainProgressWriter);
            const auto &restored = simulation.RestoredOutputController();
            continueOutput = restored && restored->is_object() &&
                             restored->value("prefix", std::string{}) ==
                                 output.prefix;
            state = restored ? plan.RestoreState(*restored, simulation.Time())
                             : plan.InitialState(simulation.Time());
            nextCheckpointTime = state.nextCheckpointTime;
            nextVisualizationTime = state.nextVisualizationTime;
            InitializeFiles();
            if (output.asynchronous)
                asynchronous = std::make_unique<AsyncOutput>(simulation);
            if (!continueOutput)
                Record(0, 0, {}, false, false, true);
            MaybeReportAllocations();
            // Publish callbacks only after fallible output/resource setup.
            // A failed constructor has no Data destructor to withdraw them.
            InstallObservers();
        }

        ~Data()
        {
            if (asynchronous)
            {
                if (std::uncaught_exceptions())
                    MPI::AbortIfSolverFailure("solver unwinding with asynchronous output");
                asynchronous->Drain();
            }
            // Domain callbacks capture this owner. Withdraw them before logger,
            // CSV and scheduling state can disappear.
            Visit([&](index, Domain &domain)
                  {
                      domain.SetProgressObserver({});
                      domain.SetQuiescentObserver({}); });
        }

        template <class Action>
        void Visit(Action &&action)
        {
            for (index iDomain = 0; iDomain < domains; ++iDomain)
                if (simulation.Local(iDomain))
                    action(iDomain, simulation.GetDomain(iDomain));
        }

        void ReportAllocations(const std::string &token)
        {
            // Explicit host output boundary. Rank-local files require no new
            // agreement/gather and do not interrupt a queued numerical region.
            const auto name = output.prefix + "-memory-R" + std::to_string(mpi.rank) + token + ".txt";
            std::ofstream file(name);
            DNDS_check_throw_info(bool(file), "cannot open allocation report: " + name);
            Runtime::AllocationDiagnostics::Snapshot().Dump(file);
            file << "Untracked provider allocations: AmgX/NCCL/internal pools (unknown bytes)\n";
            file.flush();
            DNDS_check_throw_info(bool(file), "failed to write allocation report: " + name);
        }
        void MaybeReportAllocations()
        {
            if (output.diagnostics.allocations && !allocationReportWritten &&
                simulation.Step() >= uint64_t(output.diagnostics.allocationWarmupSteps))
            {
                ReportAllocations(OutputPlan::StepToken(simulation.Step()));
                allocationReportWritten = true;
            }
        }

        void InitializeFiles()
        {
            MPI::LocalStage(mpi, [&]
                            {
                if (mpi.rank == 0)
                {
                    const auto parent =
                        std::filesystem::path(output.prefix).parent_path();
                    if (!parent.empty())
                        std::filesystem::create_directories(parent);
                    std::ofstream resolved(plan.ResolvedPath());
                    resolved << config.Root().dump(2) << '\n';
                    DNDS_check_throw_info(
                        bool(resolved),
                        "cannot write resolved Cauchy configuration");
                } });
            Visit([&](index iDomain, Domain &domain)
                  { MPI::LocalStage(
                        MPIInfo(domain.Discretization().Metrics().Communicator()),
                        [&]
                        {
                            progress[size_t(iDomain)] =
                                std::make_unique<ProgressOutputController>(
                                    domain.Equation().Schema(), output,
                                    domains == 1 ? std::string{} : domain.Name(), plan.HistoryBare(iDomain),
                                    simulation.Placement().DomainRank(iDomain) == 0,
                                    logger,
                                    continueOutput
                                        ? std::optional(state.history.at(
                                              size_t(iDomain)))
                                        : std::nullopt);
                        }); });
            Json localDomains = Json::array();
            for (index iDomain = 0; iDomain < domains; ++iDomain)
                if (simulation.Local(iDomain) &&
                    simulation.Placement().DomainRank(iDomain) == 0)
                {
                    const auto &domain = simulation.GetDomain(iDomain);
                    localDomains.push_back(
                        {{"index", iDomain},
                         {"name", domain.Name()},
                         {"history", plan.HistoryBare(iDomain) + ".csv"},
                         {"eventLog", output.eventLog ? Json(output.prefix + "-events-rank" + std::to_string(mpi.rank) + ".log") : Json()},
                         {"components", EquationComponentNames(
                                            domain.Equation().Schema())}});
                }
            const auto manifestDomains = GatherJsonArrays(mpi, localDomains);
            MPI::LocalStage(mpi, [&]
                            {
                if (mpi.rank == 0)
                {
                    const Json manifest{
                        {"schema", "dndsr.cauchy.run-output.v1"},
                        {"resolvedConfig", plan.ResolvedPath()},
                        {"historySchema", output.historyStateBounds ? "dndsr.cauchy.detailed-history.v3" : "dndsr.cauchy.detailed-history.v2"},
                        {"progressStatus",
                         {{"nonlinearIteration", 0},
                          {"implicitStageEnd", 1},
                          {"acceptedStepEnd", 2}}},
                        {"domains", manifestDomains}};
                    std::ofstream file(plan.RunManifestPath());
                    file << manifest.dump(2) << '\n';
                    DNDS_check_throw_info(bool(file),
                                          "cannot write Cauchy run manifest");
                } });
        }

        Json CaptureState()
        {
            std::vector<int64_t> positions(size_t(2 * domains), 0);
            std::vector<int> roots(size_t(domains), 0);
            Visit([&](index iDomain, Domain &)
                  {
                      progress[size_t(iDomain)]->Flush();
                      if (simulation.Placement().DomainRank(iDomain) == 0)
                      {
                          const auto cursor = progress[size_t(iDomain)]->HistoryPosition();
                          positions[size_t(2 * iDomain)] = cursor.block;
                          positions[size_t(2 * iDomain + 1)] = cursor.line;
                          roots[size_t(iDomain)] = 1;
                      } });
            MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                MPI::ContiguousView<int64_t>{positions.data(), positions.size()},
                mpi.comm);
            MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                MPI::ContiguousView<int>{roots.data(), roots.size()}, mpi.comm);
            state.nextCheckpointTime = nextCheckpointTime;
            state.nextVisualizationTime = nextVisualizationTime;
            for (index iDomain = 0; iDomain < domains; ++iDomain)
            {
                DNDS_check_throw_info(
                    roots[size_t(iDomain)] == 1 &&
                        positions[size_t(2 * iDomain)] >= 0 &&
                        positions[size_t(2 * iDomain + 1)] >= 0,
                    "Cauchy output history cursor has no unique domain root");
                state.history[size_t(iDomain)] = {int(positions[size_t(2 * iDomain)]), positions[size_t(2 * iDomain + 1)]};
            }
            return state.ToJson();
        }

        Json CaptureDomainState(index iDomain)
        {
            auto local = state;
            auto &domain = simulation.GetDomain(iDomain);
            const MPIInfo domainMPI(
                domain.Discretization().Metrics().Communicator());
            CsvLogPosition cursor;
            MPI::LocalStage(domainMPI, [&]
                            {
                progress[size_t(iDomain)]->Flush();
                if (domainMPI.rank == 0)
                    cursor = progress[size_t(iDomain)]->HistoryPosition(); });
            int64_t encoded[2]{cursor.block, cursor.line};
            MPI::Bcast(encoded, 2, MPI_INT64_T, 0, domainMPI.comm);
            local.history[size_t(iDomain)] = {int(encoded[0]), encoded[1]};
            local.nextCheckpointTime = nextCheckpointTime;
            local.nextVisualizationTime = nextVisualizationTime;
            return local.ToJson();
        }

        void InstallObservers()
        {
            Visit([&](index iDomain, Domain &domain)
                  {
                      domain.SetProgressObserver(
                          [this, iDomain](const NonlinearProgress &entry)
                          { progress[size_t(iDomain)]->CaptureIteration(entry, MPI_Wtime() - started); },
                          output.historyStateBounds ? output.implicit.historyInterval : 0);
                      domain.SetQuiescentObserver(
                          [this, iDomain](const NonlinearSolveState &nonlinear)
                          { InternalSnapshot(iDomain, nonlinear); }); });
        }

        void InternalSnapshot(index iDomain,
                              const NonlinearSolveState &nonlinear)
        {
            const bool ditr = time.method == "ditr";
            const index completedIteration =
                ditr || nonlinear.terminal ? nonlinear.iteration
                                           : nonlinear.iteration + 1;
            const auto step = simulation.Step() + 1;
            const auto selected = plan.SelectedSnapshots(step, nonlinear.stage, completedIteration);
            const auto intervalDue = [completedIteration](index interval)
            { return completedIteration > 0 && interval > 0 && completedIteration % interval == 0; };
            const bool checkpoint = selected.checkpoint || intervalDue(internalCheckpointInterval);
            const bool visualization = selected.visualization || intervalDue(internalVisualizationInterval);
            // Emit progress at the existing quiescent nonlinear boundary
            // before staging any output or entering file I/O.
            progress[size_t(iDomain)]->FlushIterations(checkpoint || visualization);
            if (!checkpoint && !visualization)
                return;
            const bool writer = simulation.Placement().DomainRank(iDomain) == 0;
            const auto token = OutputPlan::InternalToken(step, nonlinear.stage, completedIteration);
            if (writer)
                logger->Info(fmt::format("Writing {} D{}{}",
                                         checkpoint && visualization ? "checkpoint and visualization"
                                         : checkpoint                ? "checkpoint"
                                                                     : "visualization",
                                         iDomain, token));
            // Internal checkpoints may contain active coupled stages. Keep the
            // exact existing synchronous writer, after all older background
            // transactions finish; never share its HDF/MPI sequence with them.
            if (asynchronous)
                asynchronous->Drain();
            auto &domain = simulation.GetDomain(iDomain);
            const auto prefix = plan.VisualizationPrefix(iDomain, token);
            if (visualization)
                domain.WriteInternalVisualization(prefix,
                                                  output.hdfDeflateLevel);
            if (checkpoint)
                domain.WriteInternalCheckpoint(
                    plan.InternalCheckpointPath(iDomain, token), config.Root(),
                    nonlinear, CaptureDomainState(iDomain));
            if (!writer)
                return;
            Json artifacts = Json::array();
            if (checkpoint)
                artifacts.push_back(
                    {{"kind", "checkpoint"},
                     {"path", plan.InternalCheckpointPath(iDomain, token)},
                     {"schema", "dndsr.cauchy.checkpoint.internal.v1"}});
            if (visualization)
            {
                artifacts.push_back(
                    {{"kind", "volume"},
                     {"path", prefix + ".vtkhdf"},
                     {"schema", "dndsr.cauchy.visualization.v1"}});
                artifacts.push_back(BoundaryArtifact(prefix, output.boundaryFacialFluxes,
                                                     domain.InternalObservationTime()));
            }
            const Json manifest{
                {"schema", "dndsr.cauchy.internal-snapshot.v1"},
                {"restartScope",
                 checkpoint ? "exact_internal" : "visualization_only"},
                {"domain", iDomain},
                {"step", simulation.Step() + 1},
                {"stage", nonlinear.stage},
                {"coupledStage", nonlinear.coupledStage},
                {"iteration", completedIteration},
                {"time", domain.InternalObservationTime()},
                {"artifacts", std::move(artifacts)}};
            std::ofstream file(plan.InternalSnapshotManifestPath(iDomain, token));
            file << manifest.dump(2) << '\n';
            DNDS_check_throw_info(bool(file),
                                  "cannot write Cauchy internal snapshot manifest");
        }

        void FlushPending()
        {
            Visit([&](index iDomain, Domain &domain)
                  { MPI::LocalStage(
                        MPIInfo(domain.Discretization().Metrics().Communicator()),
                        [&]
                        { progress[size_t(iDomain)]->FlushIterations(); }); });
        }

        bool Record(real dt, index retries,
                    const std::vector<PhysicalStepLimit> &limits,
                    bool useLimits, bool printConsole, bool forceHistory)
        {
            int allReached = 1;
            Visit([&](index iDomain, Domain &domain)
                  {
                      const auto values = domain.Diagnostics(
                          config.Domains()[iDomain].settings, output.forceZones);
                      std::vector<real> stateBounds;
                      if (output.historyStateBounds && (forceHistory ||
                          (output.historyInterval > 0 && domain.Epoch().acceptedStep % uint64_t(output.historyInterval) == 0)))
                          stateBounds = domain.MeasureStateBounds(false);
                      auto &baseline = baselines[size_t(iDomain)];
                      if (baseline == 0)
                          baseline = std::max(values[4], real(1e-300));
                      const auto limit = useLimits
                                             ? limits.at(size_t(iDomain))
                                             : PhysicalStepLimit{};
                      index nonlinearIterations = 0, linearIterations = 0;
                      for (const auto &report : domain.LastImplicitStages())
                      {
                          nonlinearIterations += report.iterations;
                          linearIterations += report.linearIterations;
                      }
                      MPI::LocalStage(
                          MPIInfo(domain.Discretization().Metrics().Communicator()),
                          [&]
                          {
                              const ProgressLocation location{
                                  domain.Epoch().acceptedStep, -1, -1, -1,
                                  retries, domain.Time(), dt,
                                  MPI_Wtime() - started, time.steady};
                              for (index iStage = 0;
                                   iStage < index(domain.LastImplicitStages().size());
                                   ++iStage)
                              {
                                  auto stage = location;
                                  stage.stage = iStage;
                                  progress[size_t(iDomain)]->StageEnd(
                                      stage,
                                      domain.LastImplicitStages()[size_t(iStage)]);
                              }
                              AcceptedStepProgress accepted;
                              accepted.diagnostics = values;
                              accepted.stateBounds = std::move(stateBounds);
                              accepted.faceFallbacks = domain.LastFaceFallbacks();
                              accepted.retries = retries;
                              accepted.implicitStages =
                                  index(domain.LastImplicitStages().size());
                              accepted.cappedStages = index(std::count_if(
                                  domain.LastImplicitStages().begin(), domain.LastImplicitStages().end(),
                                  [](const auto &report) { return !report.converged; }));
                              accepted.nonlinearIterations = nonlinearIterations;
                              accepted.linearIterations = linearIterations;
                              accepted.stepWallSeconds = stepWallSeconds;
                              accepted.coupledIterations =
                                  domain.LastCoupledIterations();
                              accepted.coupledResidualTargetRatio =
                                  domain.LastCoupledTargetRatio();
                              if (!domain.LastImplicitStages().empty())
                              {
                                  accepted.residual = domain.LastImplicitStages().back().residual;
                                  accepted.pseudoCFL = domain.LastImplicitStages().back().pseudoCFL;
                              }
                              accepted.positivity = domain.LastPositivity();
                              accepted.timePPLimitedCells = limit.limitedCells;
                              accepted.timePPMinimumAlpha = limit.minimumAlpha;
                              accepted.timeGrowthLimited = limit.growthLimited;
                              progress[size_t(iDomain)]->StepEnd(
                                  location, accepted, forceHistory, printConsole);
                          });
                      const bool outerTarget =
                          time.steadyTolerance <= 0 ||
                          values[4] / baseline <= time.steadyTolerance;
                      if (time.steady)
                          allReached &=
                              !domain.LastImplicitStages().empty() &&
                              std::all_of(
                                  domain.LastImplicitStages().begin(),
                                  domain.LastImplicitStages().end(),
                                  [](const auto &stage)
                                  { return stage.converged; }) &&
                              outerTarget;
                      else
                          allReached &=
                              time.steadyTolerance > 0 &&
                              domain.Epoch().acceptedStep >=
                                  uint64_t(time.steadyMinSteps) &&
                              outerTarget; });
            allReached = MPI::AllreduceValue<MPI::Reduction::Min>(allReached,
                                                                  mpi.comm);
            return bool(allReached);
        }

        void WriteSnapshot(const std::string &token, bool checkpoint,
                           bool visualization)
        {
            const auto manifest = SnapshotManifest(token, checkpoint, visualization);
            if (asynchronous)
            {
                asynchronous->Submit(simulation, config, plan, token, checkpoint, visualization,
                                     checkpoint ? CaptureState() : Json(), manifest);
                return;
            }
            if (checkpoint)
                simulation.WriteCheckpoint(plan.CheckpointPath(token),
                                           output.checkpointReconstruction,
                                           CaptureState());
            if (visualization)
                Visit([&](index iDomain, Domain &domain)
                      { domain.WriteVisualization(
                            plan.VisualizationPrefix(iDomain, token),
                            output.hdfDeflateLevel); });
            MPI::LocalStage(mpi, [&]
                            {
                if (mpi.rank != 0)
                    return;
                std::ofstream file(plan.SnapshotManifestPath(token));
                file << manifest.dump(2) << '\n';
                file.close();
                DNDS_check_throw_info(bool(file),
                                      "cannot write Cauchy snapshot manifest"); });
        }

        Json SnapshotManifest(const std::string &token, bool checkpoint, bool visualization) const
        {
            Json artifacts = Json::array();
            if (checkpoint)
                artifacts.push_back(
                    {{"kind", "checkpoint"},
                     {"path", plan.CheckpointPath(token)},
                     {"schema", "dndsr.cauchy.checkpoint.v4"}});
            if (visualization)
                for (index iDomain = 0; iDomain < domains; ++iDomain)
                {
                    const auto prefix =
                        plan.VisualizationPrefix(iDomain, token);
                    artifacts.push_back(
                        {{"kind", "volume"},
                         {"domain", iDomain},
                         {"path", prefix + ".vtkhdf"},
                         {"schema", "dndsr.cauchy.visualization.v1"}});
                    auto boundary = BoundaryArtifact(prefix, output.boundaryFacialFluxes, simulation.Time());
                    boundary["domain"] = iDomain;
                    artifacts.push_back(std::move(boundary));
                }
            return Json{{"schema", "dndsr.cauchy.snapshot.v1"},
                        {"step", simulation.Step()},
                        {"time", simulation.Time()},
                        {"token", token},
                        {"artifacts", std::move(artifacts)}};
        }

        bool CompleteAcceptedStep(
            real dt, index retries,
            const std::vector<PhysicalStepLimit> &limits, bool useLimits,
            bool printConsole)
        {
            // Read the existing host boundary only; no new device fence or
            // timing collective. Include retries, exclude following file I/O.
            stepWallSeconds = stepStarted ? MPI_Wtime() - *stepStarted : 0;
            stepStarted.reset();
            bool checkpointTimeDue = false, visualizationTimeDue = false;
            if (!time.steady && output.checkpointTimeInterval > 0)
                while (OutputPlan::TimeAt(nextCheckpointTime,
                                          output.checkpointTimeInterval) <=
                       simulation.Time())
                {
                    checkpointTimeDue = true;
                    ++nextCheckpointTime;
                }
            if (!time.steady && plan.VisualizationTimeInterval() > 0)
                while (OutputPlan::TimeAt(nextVisualizationTime,
                                          plan.VisualizationTimeInterval()) <=
                       simulation.Time())
                {
                    visualizationTimeDue = true;
                    ++nextVisualizationTime;
                }
            const bool checkpointStepDue = output.checkpointInterval > 0 &&
                                           simulation.Step() %
                                                   uint64_t(output.checkpointInterval) ==
                                               0;
            const bool visualizationStepDue =
                plan.VisualizationInterval() > 0 &&
                simulation.Step() % uint64_t(plan.VisualizationInterval()) == 0;
            const auto selected = plan.SelectedSnapshots(simulation.Step());
            const bool checkpoint = checkpointTimeDue || checkpointStepDue || selected.checkpoint;
            const bool visualization =
                visualizationTimeDue || visualizationStepDue || selected.visualization;
            const bool reached =
                Record(dt, retries, limits, useLimits,
                       printConsole || (output.consoleInterval > 0 && (checkpoint || visualization)),
                       false);
            MaybeReportAllocations();
            if (checkpoint || visualization)
            {
                const auto token = checkpointTimeDue || visualizationTimeDue
                                       ? OutputPlan::TimeToken(simulation.Time())
                                       : OutputPlan::StepToken(simulation.Step());
                WriteSnapshot(token, checkpoint, visualization);
            }
            return reached;
        }

        int Finalize(bool wallStop, bool reached)
        {
            DNDS_check_throw_info(!finalized,
                                  "Cauchy output coordinator finalized twice");
            finalized = true;
            WriteSnapshot("-final", true, true);
            if (asynchronous)
                asynchronous->Drain();
            if (output.diagnostics.allocations)
                ReportAllocations("-final");
            Visit([&](index iDomain, Domain &)
                  { progress[size_t(iDomain)]->Flush(); });
            logger->Flush();
            const bool timeReached =
                time.tEnd <= 0 || EquationTimeStepper::ReachedTimeTarget(
                                      simulation.Time(), time.tEnd);
            const bool stepStop =
                !wallStop &&
                ((!timeReached && time.tEnd > 0) ||
                 ((time.steady || time.steadyTolerance > 0) && !reached));
            MPI::LocalStage(mpi, [&]
                            {
                if (mpi.rank == 0)
                {
                    const Json result{
                        {"time", simulation.Time()},
                        {"steps", simulation.Step()},
                        {"wallSeconds", MPI_Wtime() - started},
                        {"status", wallStop ? "wall-capped"
                                            : stepStop ? "step-capped"
                                                       : reached
                                                             ? "steady-converged"
                                                             : "target-reached"},
                        {"steadyTargetMet", reached},
                        {"steadyMode", time.steady},
                        {"physicalTimeTarget", time.tEnd},
                        {"steadyTolerance", time.steadyTolerance}};
                    std::ofstream summary(plan.SummaryPath());
                    summary << result.dump(2) << '\n';
                    DNDS_check_throw_info(bool(summary),
                                          "cannot write Cauchy summary");
                } });
            return wallStop || stepStop ? 2 : 0;
        }
    };

    OutputCoordinator::OutputCoordinator(Simulation &simulation,
                                         const SimulationConfig &config,
                                         MPIInfo mpi, real started)
        : _data(std::make_unique<Data>(simulation, config, mpi, started)) {}
    OutputCoordinator::~OutputCoordinator() = default;

    OutputPlan::StepTarget OutputCoordinator::SnapStep(real proposedDt) const
    {
        return _data->plan.SnapStep(
            _data->simulation.Time(), proposedDt, _data->time.tEnd,
            _data->nextCheckpointTime, _data->nextVisualizationTime);
    }
    void OutputCoordinator::BeginAttempt(real dt, index retries,
                                         real targetTime)
    {
        if (!_data->stepStarted)
            _data->stepStarted = MPI_Wtime();
        _data->Visit([&](index iDomain, Domain &)
                     { _data->progress[size_t(iDomain)]->BeginAttempt(
                           {_data->simulation.Step() + 1, -1, -1, -1, retries, targetTime,
                            dt, MPI_Wtime() - _data->started, _data->time.steady}); });
    }
    void OutputCoordinator::FlushPending() { _data->FlushPending(); }
    void OutputCoordinator::WarnRetry(const std::string &reason)
    {
        _data->logger->Warn("Retry candidate with half dt: " + reason);
    }
    bool OutputCoordinator::CompleteAcceptedStep(
        real dt, index retries, const std::vector<PhysicalStepLimit> &limits,
        bool useLimits, bool printConsole)
    {
        return _data->CompleteAcceptedStep(dt, retries, limits, useLimits,
                                           printConsole);
    }
    int OutputCoordinator::Finalize(bool wallStop, bool reached)
    {
        return _data->Finalize(wallStop, reached);
    }
}
