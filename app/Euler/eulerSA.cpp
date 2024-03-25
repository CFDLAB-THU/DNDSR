#include "Euler/SingleBlockApp.hpp"


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    DNDS::Euler::RunSingleBlockConsoleApp<DNDS::Euler::NS_SA>(argc, argv);
    MPI_Finalize();
}