#include "Euler/SingleBlockApp.hpp"

int main(int argc, char *argv[])
{
    // #ifndef H5_HAVE_THREADSAFE
    //     DNDS_assert(false);
    // #endif
    // MPI_Init(&argc, &argv);
    DNDS::MPI::Init_thread(&argc, &argv);
    int errc = DNDS::Euler::RunSingleBlockConsoleApp<DNDS::Euler::NS_SA>(argc, argv);
    if (errc)
        MPI_Abort(MPI_COMM_WORLD, errc);
    MPI_Finalize();
}