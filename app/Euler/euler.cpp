#include "Euler/EulerSolver.hpp"

int main(int argc, char *argv[])
{
    // std::cout << "====================\n";
    // std::cout << argc << std::endl;
    // for (int i = 0; i < argc; i++)
    //     std::cout << argv[i] << "\n";
    MPI_Init(&argc, &argv);
    // std::cout << "====================\n";
    // std::cout << argc << std::endl;
    // for (int i = 0; i < argc; i++)
    //     std::cout << argv[i] << "\n";
    // return 0;
    std::string defaultConfJson = "../cases/euler_default_config.json";
    std::string confJson = "../cases/euler_config.json";
    if (argc > 1 && std::stoi(argv[1]) == 1)
    {
        defaultConfJson = "./euler_default_config.json";
        confJson = "./euler_config.json";
    }
    try
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        auto strategy = DNDS::MPI::CommStrategy::Instance().GetArrayStrategy();
        DNDS::Euler::EulerSolver<DNDS::Euler::EulerModel::NS> solver(mpi);
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
    MPI_Finalize();
}