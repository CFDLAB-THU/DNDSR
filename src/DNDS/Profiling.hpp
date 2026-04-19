#pragma once
/// @file Profiling.hpp
/// @brief Wall-clock performance timer and running scalar statistics utilities.

#include "Defines.hpp"
#include "MPI.hpp"

namespace DNDS
{
    /**
     * @brief Process-wide singleton aggregating wall-clock timings by category.
     *
     * @details Provides a fixed set of named timer slots (RHS, Comm, LinSolve, ...)
     * that callers start / stop at well-known phases of the solver. Calls are
     * expected to nest correctly (each `StartTimer` matched by a `StopTimer`).
     *
     * The buffer holds two copies of each slot (`Ntype_All == 2 * Ntype`) so
     * current and previous-iteration timings can be retained for reporting.
     *
     * Thread-safe C++11 singleton; not reentrant for the same timer id.
     */
    class PerformanceTimer // cxx11 + thread-safe singleton
    {
    public:
        /// @brief Named timer slots. New categories can be added before `__EndTimerType`.
        enum TimerType
        {
            Unknown = 0,
            RHS = 1,             ///< Total RHS evaluation.
            Dt = 2,              ///< Time-step computation.
            Reconstruction = 3,  ///< Variational reconstruction.
            ReconstructionCR = 4,///< CR (compact reconstruction) branch.
            Limiter = 5,         ///< Slope / variable limiter.
            LimiterA = 6,        ///< Limiter sub-phase A.
            LimiterB = 7,        ///< Limiter sub-phase B.
            Basis = 8,           ///< Basis-function evaluation.
            Comm = 9,            ///< Catch-all MPI comm.
            Comm1 = 10,          ///< Comm phase 1 (e.g., cell-ghost).
            Comm2 = 11,          ///< Comm phase 2 (e.g., face-ghost).
            Comm3 = 12,          ///< Comm phase 3.
            LinSolve = 13,       ///< Linear solve (total).
            LinSolve1 = 14,      ///< Linear solve phase 1.
            LinSolve2 = 15,      ///< Linear solve phase 2.
            LinSolve3 = 16,      ///< Linear solve phase 3.
            Positivity = 17,     ///< Positivity preservation.
            PositivityOuter = 18,///< Outer-iteration positivity.
            __EndTimerType = 64  ///< One past the last valid id.
        };

        static const int Ntype = __EndTimerType;
        static const int Ntype_Past = 64;
        static const int Ntype_All = Ntype + Ntype_Past;

    private:
        std::array<real, Ntype_All> timer = {0};
        std::array<real, Ntype_All> tStart;
        PerformanceTimer() = default;
        PerformanceTimer(const PerformanceTimer &);
        PerformanceTimer &operator=(const PerformanceTimer &);

    public:
        /// @brief Access the process-wide singleton.
        static PerformanceTimer &Instance();
        /// @brief Record the current wall time in the "start" slot for timer `t`.
        void StartTimer(TimerType t);
        /// @brief Integer-id overload of #StartTimer.
        void StartTimer(int t);
        /// @brief Add (now - start) to the accumulated time for timer `t`.
        void StopTimer(TimerType t);
        /// @brief Integer-id overload of #StopTimer.
        void StopTimer(int t);
        /// @brief Current local (this-rank) accumulated wall time (seconds).
        real getTimer(TimerType t);
        real getTimer(int t);
        /// @brief Global maximum across ranks (collective on `mpi.comm`).
        real getTimerCollective(TimerType t, const MPIInfo &mpi);
        real getTimerCollective(int t, const MPIInfo &mpi);
        /// @brief Either #getTimerCollective (when `col == true`) or #getTimer.
        template <typename T>
        real getTimerColOrLoc(T t, const MPIInfo &mpi, bool col)
        {
            return col ? getTimerCollective(t, mpi) : getTimer(t);
        }
        /// @brief Zero the accumulated time for one timer slot.
        void clearTimer(TimerType t);
        void clearTimer(int t);
        /// @brief Zero every timer slot.
        void clearAllTimer();
    };

    /**
     * @brief Running-statistics accumulator using Welford's online algorithm.
     *
     * @details Updates mean and variance in a single pass without storing the
     * sample history. Numerically stable for long sequences. Used to report
     * residual / CFL / iteration-count statistics during solver runs.
     */
    class ScalarStatistics
    {
        real average = 0;
        index count = 0;
        real sigmaS = 0;

    public:
        /// @brief Reset counts and moments to zero.
        void clear()
        {
            average = 0;
            count = 0;
            sigmaS = 0;
        }
        /// @brief Incorporate a new sample `v`.
        ScalarStatistics &update(real v)
        {
            count++;
            real newAverage = average + (v - average) / real(count);
            sigmaS += ((v - newAverage) * (v - average) - sigmaS) / real(count);
            average = newAverage;
            return *this;
        }

        /// @brief `(mean, stddev)` pair for the samples so far.
        [[nodiscard]] std::tuple<real, real> get()
        {
            return std::make_tuple(average, std::sqrt(std::max(0., sigmaS)));
        }

        /// @brief Total sum of the samples (reconstructed from the running mean).
        [[nodiscard]] real getSum() const
        {
            return average * real(count);
        }
    };

    /// @brief Short-hand accessor to the #PerformanceTimer singleton.
    inline PerformanceTimer &Timer()
    {
        return PerformanceTimer::Instance();
    }

}
