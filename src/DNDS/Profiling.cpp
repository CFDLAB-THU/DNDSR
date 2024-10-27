#include "Profiling.hpp"

namespace DNDS
{
    PerformanceTimer &PerformanceTimer::Instance()
    {
        static PerformanceTimer instance;
        return instance;
    }

    void PerformanceTimer::StartTimer(TimerType t)
    {
        tStart[t] = MPI_Wtime();
    }

    void PerformanceTimer::StartTimer(int t)
    {
        tStart[t + Ntype] = MPI_Wtime();
    }

    void PerformanceTimer::StopTimer(TimerType t)
    {
        this->timer[t] += MPI_Wtime() - tStart[t];
    }

    void PerformanceTimer::StopTimer(int t)
    {
        this->timer[t + Ntype] += MPI_Wtime() - tStart[t];
    }

    real PerformanceTimer::getTimer(TimerType t)
    {
        return timer[t];
    }

    real PerformanceTimer::getTimer(int t)
    {
        return timer[t + Ntype];
    }

    real PerformanceTimer::getTimerCollective(TimerType t, const MPIInfo &mpi)
    {
        real timeTotal{0};
        MPI::Allreduce(&timer[t], &timeTotal, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        return timeTotal / mpi.size; // loses one record of all reduce
    }

    real PerformanceTimer::getTimerCollective(int t, const MPIInfo &mpi)
    {
        real timeTotal{0};
        MPI::Allreduce(&timer[t + Ntype], &timeTotal, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        return timeTotal / mpi.size; // loses one record of all reduce
    }

    void PerformanceTimer::clearTimer(TimerType t)
    {
        timer[t] = 0;
    }

    void PerformanceTimer::clearTimer(int t)
    {
        timer[t + Ntype] = 0;
    }

    void PerformanceTimer::clearAllTimer()
    {
        for (int i = 0; i < Ntype_All; i++)
            clearTimer(TimerType(i)); // TODO: optimization
    }

}