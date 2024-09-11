#include "Euler/SingleBlockApp.hpp"


int main(int argc, char* argv[])
{
// #ifndef H5_HAVE_THREADSAFE
//     DNDS_assert(false);
// #endif
    // MPI_Init(&argc, &argv);
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::Euler::RunSingleBlockConsoleApp<DNDS::Euler::NS_SA>(argc, argv);
    MPI_Finalize();
}