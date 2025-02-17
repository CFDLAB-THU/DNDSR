#pragma once

#include <argparse.hpp>

#include "EulerSolver.hpp"

namespace DNDS::Euler
{
    constexpr static inline const char *getSingleBlockAppName(const EulerModel model)
    {
        if (model == NS)
            return "euler";
        else if (model == NS_SA)
            return "eulerSA";
        else if (model == NS_2D)
            return "euler2D";
        else if (model == NS_3D)
            return "euler3D";
        else if (model == NS_SA_3D)
            return "eulerSA3D";
        else if (model == NS_2EQ)
            return "euler2EQ";
        else if (model == NS_2EQ_3D)
            return "euler2EQ3D";
        return "_error_app_name_";
    }

    template <EulerModel model>
    int RunSingleBlockConsoleApp(int argc, char *argv[])
    {
        using namespace std::literals;
        MPIInfo mpi;
        mpi.setWorld();

        std::string defaultConfJson = "../cases/"s + getSingleBlockAppName(model) + "_default_config.json"s;
        std::string confJson = "../cases/"s + getSingleBlockAppName(model) + "_config.json";
        std::vector<std::string> overwriteKeys, overwriteValues;

        argparse::ArgumentParser mainParser(getSingleBlockAppName(model), "version"s + " commit "s + DNDS_MACRO_TO_STRING(DNDS_CURRENT_COMMIT_HASH));
        std::string read_configPath;
        mainParser.add_argument("config").default_value("");
        mainParser.add_argument("-k", "--overwrite_key")
            .help("keys to the json entries to overwrite")
            .append()
            .default_value<std::vector<std::string>>({});
        mainParser.add_argument("-v", "--overwrite_value")
            .help("values to the json entries to overwrite")
            .append()
            .default_value<std::vector<std::string>>({});
        mainParser.add_argument("--debug").flag().default_value(false);

        RegisterSignalHandler();

        try
        {
            mainParser.parse_args(argc, argv);
            if (mainParser.get<bool>("--debug"))
            {
                Debug::isDebugging = true;
                Debug::MPIDebugHold(mpi);
            }
            read_configPath = mainParser.get("config");
            if (!read_configPath.empty())
            {
                confJson = read_configPath;
                std::filesystem::path p(read_configPath);
                if (p.is_relative())
                    p = "." / p; // so that not using a wrong path if using a file in cwd
                defaultConfJson = getStringForcePath(p.parent_path()) + "/"s + getSingleBlockAppName(model) + "_default_config.json"s;
            }
            overwriteKeys = mainParser.get<std::vector<std::string>>("--overwrite_key");
            overwriteValues = mainParser.get<std::vector<std::string>>("--overwrite_value");
            if (overwriteKeys.size() != overwriteValues.size())
                throw std::runtime_error("overwrite keys and values not matching");
        }
        catch (const std::exception &err)
        {
            std::cerr << err.what() << std::endl;
            std::cerr << mainParser;
            std::abort();
        }

        try
        {

            if (mpi.rank == 0)
                log() << "Current MPI thread level: " << MPI::GetMPIThreadLevel() << std::endl;
            auto strategy = MPI::CommStrategy::Instance().GetArrayStrategy();
            Euler::EulerSolver<model> solver(mpi);
            if (mpi.rank == 0)
            {
                log() << "Reading configuration from " << confJson << std::endl;
                log() << "Using default configuration from " << defaultConfJson << std::endl;
            }
            solver.ConfigureFromJson(defaultConfJson, false);
            solver.ConfigureFromJson(defaultConfJson, true, confJson, overwriteKeys, overwriteValues);
            solver.ReadMeshAndInitialize();
            solver.RunImplicitEuler();
        }
        catch (const std::exception &e)
        {
            std::cerr << "DNDS top-level exception: " << e.what() << std::endl;
            std::abort();
        }
        catch (...)
        {
            std::cerr << "Unknown exception" << std::endl;
            std::abort();
        }
        return 0;
    }

}