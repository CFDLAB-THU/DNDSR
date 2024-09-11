#include "Euler/SingleBlockApp.hpp"

int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::Euler::RunSingleBlockConsoleApp<DNDS::Euler::NS>(argc, argv);
    MPI_Finalize();
}