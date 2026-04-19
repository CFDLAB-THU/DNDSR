#pragma once
/// @file OMP.hpp
/// @brief Helpers for OpenMP thread-count configuration at solver entry points.

#include "Defines.hpp"
#include "Eigen/src/Core/products/Parallelizer.h"

namespace DNDS::OMP
{
    /**
     * @brief Configure OpenMP and Eigen threading for the "distributed OMP" pattern.
     *
     * @details Sets the OpenMP thread count from the `DNDS_DIST_OMP_NUM_THREADS`
     * environment variable (falling back to 1) and disables Eigen's internal
     * parallelism so it does not oversubscribe the MPI+OpenMP layout.
     *
     * Intended to be called once near `main()` when #DNDS_USE_OMP is on.
     */
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