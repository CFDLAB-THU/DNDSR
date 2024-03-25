#pragma once

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
    void RunSingleBlockConsoleApp(int argc, char *argv[])
    {
        using namespace std::literals;
        std::string defaultConfJson = "../cases/"s + getSingleBlockAppName(model) + "_default_config.json"s;
        std::string confJson = "../cases/"s + getSingleBlockAppName(model) + "_config.json";
        if (argc > 1 && std::stoi(argv[1]) == 1)
        {
            defaultConfJson = "./"s + getSingleBlockAppName(model) + "_default_config.json";
            confJson = "./"s + getSingleBlockAppName(model) + "_config.json";
        }
        try
        {
            MPIInfo mpi;
            mpi.setWorld();
            auto strategy = MPI::CommStrategy::Instance().GetArrayStrategy();
            Euler::EulerSolver<model> solver(mpi);
            solver.ConfigureFromJson(defaultConfJson, false);
            solver.ConfigureFromJson(defaultConfJson, true, confJson);
            solver.ReadMeshAndInitialize();
            solver.RunImplicitEuler();
        }
        catch (std::exception &e)
        {
            std::cerr << "DNDS top-level exception: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception" << std::endl;
        }
    }

}