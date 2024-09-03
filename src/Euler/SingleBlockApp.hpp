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
        std::string defaultConfJson = "../cases/"s + getSingleBlockAppName(model) + "_default_config.json"s;
        std::string confJson = "../cases/"s + getSingleBlockAppName(model) + "_config.json";

        argparse::ArgumentParser mainParser(getSingleBlockAppName(model), "version"s + " commit "s + DNDS_MACRO_TO_STRING(DNDS_CURRENT_COMMIT_HASH));
        std::string read_configPath;
        mainParser.add_argument("config").default_value("");

        RegisterSignalHandler();

        try
        {
            mainParser.parse_args(argc, argv);
            read_configPath = mainParser.get("config");
            if (!read_configPath.empty())
            {
                confJson = read_configPath;
                std::filesystem::path p(read_configPath);
                if (p.is_relative())
                    p = "." / p; // so that not using a wrong path if using a file in cwd
                defaultConfJson = getStringForcePath(p.parent_path()) + "/"s + getSingleBlockAppName(model) + "_default_config.json"s;
            }
        }
        catch (const std::exception &err)
        {
            std::cerr << err.what() << std::endl;
            std::cerr << mainParser;
            return 1;
        }

        try
        {
            MPIInfo mpi;
            mpi.setWorld();
            auto strategy = MPI::CommStrategy::Instance().GetArrayStrategy();
            Euler::EulerSolver<model> solver(mpi);
            if (mpi.rank == 0)
            {
                log() << "Reading configuration from " << confJson << std::endl;
                log() << "Using default configuration from " << defaultConfJson << std::endl;
            }
            solver.ConfigureFromJson(defaultConfJson, false);
            solver.ConfigureFromJson(defaultConfJson, true, confJson);
            solver.ReadMeshAndInitialize();
            solver.RunImplicitEuler();
        }
        catch (const std::exception &e)
        {
            std::cerr << "DNDS top-level exception: " << e.what() << std::endl;
            return 1;
        }
        catch (...)
        {
            std::cerr << "Unknown exception" << std::endl;
            return 1;
        }
        return 0;
    }

}