#include "Euler/SingleBlockApp.hpp"


int main(int argc, char* argv[])
{
// #ifndef H5_HAVE_THREADSAFE
//     DNDS_assert(false);
// #endif
    // MPI_Init(&argc, &argv);
    int provided_MPI_THREAD_LEVEL;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_MPI_THREAD_LEVEL);
    if (provided_MPI_THREAD_LEVEL < MPI_THREAD_MULTIPLE)
    {
        printf("ERROR: The MPI library does not have full thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    DNDS::Euler::RunSingleBlockConsoleApp<DNDS::Euler::NS_SA>(argc, argv);
    MPI_Finalize();
}