#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Core/StepResult.hpp"
#include "Cauchy/IO/OutputCoordinator.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Core/OMP.hpp"
#include "argparse.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

/// Keep every MPI-owning solver object inside this scope, before DNDS Finalize.
static int Run(int argc, char **argv, double started)
{
    DNDS::OMP::set_full_parallel_OMP();
    int status = 0;
    {
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        const DNDS::MPI::SolverFailureScope solverFailures(mpi.comm);
        try
        {
            // Own help handling so argparse never exits before MPI finalization.
            argparse::ArgumentParser arguments(
                "cauchy.exe", "1.0", argparse::default_arguments::none);
            arguments.add_argument("config")
                .default_value(std::string{})
                .help("Cauchy JSON input; paths are relative to the working directory");
            arguments.add_argument("-h", "--help").flag();
            arguments.add_argument("--check-config")
                .flag()
                .help("Print resolved configuration without constructing a mesh");
            arguments.add_argument("--emit-schema")
                .flag()
                .help("Print the canonical raw-input JSON schema without reading a case");
            arguments.add_argument("--restart")
                .default_value(std::string{})
                .help("Restart from a Cauchy checkpoint");
            arguments.add_argument("--color")
                .default_value(std::string{})
                .help("Terminal color: auto, always or never; overrides output.consoleColor");
            arguments.add_argument("--bold")
                .default_value(std::string{})
                .help("Terminal emphasis: auto, always or never; overrides output.consoleBold");
            arguments.add_argument("-k", "--overwrite_key")
                .append()
                .default_value<std::vector<std::string>>({})
                .help("Resolved-config JSON pointer to override; pair with -v, may repeat");
            arguments.add_argument("-v", "--overwrite_value")
                .append()
                .default_value<std::vector<std::string>>({})
                .help("JSON value for the corresponding -k (quote JSON strings)");
            DNDS::MPI::LocalStage(mpi, [&]
                                  { arguments.parse_args(argc, argv); });
            if (arguments.get<bool>("--help"))
            {
                if (mpi.rank == 0)
                    std::cout << arguments;
                return 0;
            }
            if (arguments.get<bool>("--emit-schema"))
            {
                if (mpi.rank == 0)
                    std::cout << DNDS::Cauchy::SimulationConfig::InputSchema().dump(2)
                              << '\n';
                return 0;
            }

            const auto configPath = arguments.get<std::string>("config");
            const auto restartPath = arguments.get<std::string>("--restart");
            const bool checkConfig = arguments.get<bool>("--check-config");
            DNDS::Cauchy::Json input;
            std::vector<DNDS::Cauchy::ConfigOverride> overrides;
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                DNDS_check_throw_info(!configPath.empty(), "provide a Cauchy config file or --help");
                DNDS_check_throw_info(!checkConfig || restartPath.empty(), "--check-config and --restart are mutually exclusive");
                std::ifstream stream(configPath);
                DNDS_check_throw_info(bool(stream), "cannot open Cauchy config");
                stream >> input;
                const auto keys = arguments.get<std::vector<std::string>>("--overwrite_key");
                const auto values = arguments.get<std::vector<std::string>>("--overwrite_value");
                DNDS_check_throw_info(keys.size() == values.size(), "overwrite keys and values must be paired");
                for (DNDS::index i = 0; i < DNDS::index(keys.size()); ++i)
                {
                    DNDS_check_throw_info(!keys[i].empty() && keys[i][0] == '/', "overwrite keys must be nonempty JSON pointers");
                    overrides.push_back({keys[i], DNDS::Cauchy::Json::parse(values[i])});
                }
                for (const auto &[flag, key] : std::vector<std::pair<std::string, std::string>>{
                         {"--color", "/output/consoleColor"}, {"--bold", "/output/consoleBold"}})
                    if (const auto value = arguments.get<std::string>(flag); !value.empty())
                        overrides.push_back({key, value}); });
            auto config = DNDS::Cauchy::SimulationConfig::ResolveCollective(
                input, mpi.comm, overrides);
            if (checkConfig)
            {
                if (mpi.rank == 0)
                    std::cout << config.Root().dump(2) << '\n';
                return 0;
            }

            DNDS::Cauchy::Simulation simulation(config, mpi.comm);
            if (!restartPath.empty())
                simulation.ReadCheckpoint(restartPath);
            DNDS::Cauchy::OutputCoordinator outputs(simulation, config, mpi,
                                                    started);
            const auto &output = config.Output();
            const auto &time = config.Time();
            const auto domainCount = DNDS::index(config.Domains().size());
            std::vector<DNDS::Cauchy::PhysicalStepLimit> limits(
                static_cast<size_t>(domainCount));
            const auto visitDomains = [&](const auto &action)
            {
                for (DNDS::index iDomain = 0; iDomain < domainCount; ++iDomain)
                    if (simulation.Local(iDomain))
                        action(iDomain, simulation.GetDomain(iDomain));
            };

            bool wallStop = false, reached = false;
            for (DNDS::index step = 0;
                 time.tEnd > 0
                     ? !DNDS::Cauchy::EquationTimeStepper::ReachedTimeTarget(
                           simulation.Time(), time.tEnd)
                     : step < time.steps;
                 ++step)
            {
                int stop = time.wallSeconds > 0 &&
                           MPI_Wtime() - started >= time.wallSeconds;
                stop = DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Max>(
                    stop, mpi.comm);
                if (stop)
                {
                    wallStop = true;
                    break;
                }

                DNDS::real dt = time.dt;
                if (time.cfl > 0)
                    visitDomains([&](DNDS::index, auto &domain)
                                 { dt = std::min(dt, domain.StableStep(time.cfl)); });
                dt = DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Min>(
                    dt, mpi.comm);
                if (time.positivity.enabled)
                {
                    visitDomains([&](DNDS::index iDomain, auto &domain)
                                 { limits[size_t(iDomain)] = domain.LimitPhysicalStep(dt); });
                    for (DNDS::index iDomain = 0; iDomain < domainCount;
                         ++iDomain)
                        if (simulation.Local(iDomain))
                            dt = std::min(dt, limits[size_t(iDomain)].dt);
                    dt = DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Min>(
                        dt, mpi.comm);
                }

                std::optional<DNDS::real> exactTarget;
                if (!time.steady)
                {
                    const auto adjusted = outputs.SnapStep(dt);
                    dt = adjusted.dt;
                    exactTarget = adjusted.exactTime;
                }
                int retries = 0;
                for (;;)
                {
                    outputs.BeginAttempt(
                        dt, retries,
                        time.steady
                            ? simulation.Time()
                            : exactTarget.value_or(simulation.Time() + dt));
                    const auto result = simulation.Advance(dt, exactTarget);
                    outputs.FlushPending();
                    if (result)
                        break;
                    if (retries++ >= time.maxRetries)
                        DNDS::MPI::Abort(
                            mpi.comm,
                            "Cauchy timestep retries exhausted: " +
                                result.reason);
                    outputs.WarnRetry(result.reason);
                    dt *= .5;
                    exactTarget.reset();
                }

                const bool printStep =
                    output.consoleInterval > 0 &&
                    simulation.Step() % uint64_t(output.consoleInterval) == 0;
                reached = outputs.CompleteAcceptedStep(
                              time.steady ? 0 : dt, retries, limits,
                              time.positivity.enabled, printStep) ||
                          reached;
                if (reached)
                    break;
            }
            status = outputs.Finalize(wallStop, reached);
        }
        catch (const std::exception &exception)
        {
            // Throws are allowed; coordinated tests may catch and recover.
            // This production try includes collectively ordered MPI work,
            // including point-to-point exchanges. A catch can report locally
            // and abort only: no retry, drain, error agreement or finalization.
            DNDS::MPI::Abort(mpi.comm, "Cauchy solver failed", exception.what());
        }
        catch (...)
        {
            DNDS::MPI::Abort(mpi.comm, "Cauchy: unknown fatal exception");
        }
        status = DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Max>(
            status, mpi.comm);
    }
    return status;
}

int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
    {
        // Initialization failed: no MPI clock, solver construction or finalize.
        std::cerr << "Cauchy: MPI initialization failed\n";
        return 1;
    }
    // Include import/partition/setup in the external wall-time budget.
    const int status = Run(argc, argv, MPI_Wtime());
    DNDS::MPI::Finalize();
    return status;
}
