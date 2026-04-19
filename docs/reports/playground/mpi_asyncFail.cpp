#include <iostream>
#include <thread>
#include <map>
#include <future>
#include <string>
#include <mutex>
#include <mpi.h>
#include <cassert>
#include <omp.h>
#include <pthread.h>

std::mutex mpiMut;

// mpicxx -o main main.cpp --std=c++17

int siz, rank;

void *reduceTest1(void *arg)
{
    int a = 1;
    int aG{1};
    {
        std::lock_guard lock(mpiMut);
        // MPI_Allreduce(MPI_IN_PLACE, &aG, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Reduce(&a, &aG, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Bcast(&aG, 1, MPI_INT, 0, MPI_COMM_WORLD);
        using namespace std::chrono_literals;
        // std::this_thread::sleep_for(1s);
        double s = omp_get_wtime();
        // while (omp_get_wtime() - s < 1)
        // {
        // }
    }
    std::cout << aG << std::endl;
    assert(aG == siz);
    return NULL;
}

int main(int argc, char *argv[])
{
    int provided_MPI_THREAD_LEVEL;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_MPI_THREAD_LEVEL);
    if (provided_MPI_THREAD_LEVEL < MPI_THREAD_MULTIPLE)
    {
        printf("ERROR: The MPI library does not have full thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &siz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto reduceTest = []()
    {
        int a = 1;
        int aG{1};
        {
            std::lock_guard lock(mpiMut);
            // MPI_Allreduce(MPI_IN_PLACE, &aG, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Reduce(&a, &aG, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Bcast(&aG, 1, MPI_INT, 0, MPI_COMM_WORLD);
            using namespace std::chrono_literals;
            // std::this_thread::sleep_for(1s);
            double s = omp_get_wtime();
            // while (omp_get_wtime() - s < 1)
            // {
            // }
        }
        std::cout << aG << std::endl;
        assert(aG == siz);
    };

    for (int iter = 0; iter < 0; iter++)
    {
        auto future = std::async(reduceTest);
        // auto th = std::thread(reduceTest);

        double dtMinA = 1;
        double dtMinAAll = 1;
        {
            // std::lock_guard lock(mpiMut);
            // MPI_Allreduce(MPI_IN_PLACE, &dtMinAAll, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Reduce(&dtMinA, &dtMinAAll, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        }
        int a{0};
        int ar{0};
        // th.join();
        future.wait();
        if (rank == 0)
            std::cout << "iter " << iter << std::endl;
    }

    for (int iter = 0; iter < 1000; iter++)
    {
        int err{0};
        pthread_t thread;
        err |= pthread_create(&thread, NULL, &reduceTest1, NULL);

        double dtMinA = 1;
        double dtMinAAll = 1;
        {
            // std::lock_guard lock(mpiMut);
            // MPI_Allreduce(MPI_IN_PLACE, &dtMinAAll, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Reduce(&dtMinA, &dtMinAAll, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        }
        int a{0};
        int ar{0};
        // th.join();
        err != pthread_join(thread, NULL);
        assert(err == 0);
        if (rank == 0)
            std::cout << "iter " << iter << std::endl;
    }

    omp_set_num_threads(12);

#pragma omp parallel for
    for (int iter = 0; iter < 0; iter++)
    {
        std::cout << " " << std::endl;
        reduceTest();
        std::cout << iter << " nt " << omp_get_num_threads() << std::endl;
    }

    MPI_Finalize();
}