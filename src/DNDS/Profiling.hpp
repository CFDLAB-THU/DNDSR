#pragma once
#include "Defines.hpp"
#include "MPI.hpp"

namespace DNDS
{
    class PerformanceTimer // cxx11 + thread-safe singleton
    {
    public:
        enum TimerType
        {
            Unknown = 0,
            RHS = 1,
            Dt = 2,
            Reconstruction = 3,
            ReconstructionCR = 4,
            Limiter = 5,
            LimiterA = 6,
            LimiterB = 7,
            Basis = 8,
            Comm = 9,
            Comm1 = 10,
            Comm2 = 11,
            Comm3 = 12,
            LinSolve = 13,
            LinSolve1 = 14,
            LinSolve2 = 15,
            LinSolve3 = 16,
            Positivity = 17,
            PositivityOuter = 18,
            __EndTimerType = 64
        };

        static const int Ntype = __EndTimerType;
        static const int Ntype_Past = 64;
        static const int Ntype_All = Ntype + Ntype_Past;

    private:
        real timer[Ntype_All] = {0};
        real tStart[Ntype_All];
        PerformanceTimer() {}
        PerformanceTimer(const PerformanceTimer &);
        PerformanceTimer &operator=(const PerformanceTimer &);

    public:
        static PerformanceTimer &Instance();
        void StartTimer(TimerType t);
        void StartTimer(int t);
        void StopTimer(TimerType t);
        void StopTimer(int t);
        real getTimer(TimerType t);
        real getTimer(int t);
        real getTimerCollective(TimerType t, const MPIInfo &mpi);
        real getTimerCollective(int t, const MPIInfo &mpi);
        template <typename T>
        real getTimerColOrLoc(T t, const MPIInfo &mpi, bool col)
        {
            return col ? getTimerCollective(t, mpi) : getTimer(t);
        }
        void clearTimer(TimerType t);
        void clearTimer(int t);
        void clearAllTimer();
    };

    class ScalarStatistics
    {
        real average = 0;
        index count = 0;
        real sigmaS = 0;

    public:
        void clear()
        {
            average = 0;
            count = 0;
            sigmaS = 0;
        }
        ScalarStatistics &update(real v)
        {
            count++;
            real newAverage = average + (v - average) / count;
            sigmaS += ((v - newAverage) * (v - average) - sigmaS) / count;
            average = newAverage;
            return *this;
        }

        std::tuple<real, real> get()
        {
            return std::make_tuple(average, std::sqrt(std::max(0., sigmaS)));
        }

        real getSum()
        {
            return average * count;
        }
    };

    inline PerformanceTimer &Timer()
    {
        return PerformanceTimer::Instance();
    }

}
