#include "Defines.hpp"
#include "Eigen/src/Core/products/Parallelizer.h"

namespace DNDS::OMP
{
    inline void set_full_parallel_OMP()
    {
#ifdef DNDS_USE_OMP
        // omp_set_num_threads( // note that the meaning is like "omp_set_max_threads()"
        //     DNDS::MPIWorldSize() == -1
        //         ? std::min(omp_get_num_procs(), omp_get_max_threads())
        //         : (get_env_DNDS_DIST_OMP_NUM_THREADS() == 0 ? 1 : DNDS::get_env_DNDS_DIST_OMP_NUM_THREADS()));
        omp_set_num_threads(
            (get_env_DNDS_DIST_OMP_NUM_THREADS() == 0 ? 1 : DNDS::get_env_DNDS_DIST_OMP_NUM_THREADS()));
#endif
        Eigen::setNbThreads(1); // ! sets Eigen's multithreading!
    }
}