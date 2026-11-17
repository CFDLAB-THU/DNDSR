#pragma once
/// @file OMP.hpp
/// @brief Helpers for OpenMP thread-count configuration at solver entry points.

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Core/Span.hpp"
#include "Eigen/src/Core/products/Parallelizer.h"
#include <algorithm>
#include <exception>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace DNDS::OMP
{
    /// Ordinary OpenMP team used by the designated reader before ownership is
    /// distributed. This is intentionally independent of the per-rank team.
    inline int IngressThreads() noexcept
    {
#if defined(DNDS_USE_OMP)
        if (omp_in_parallel())
            return 1;
        return std::max(1, get_env_OMP_NUM_THREADS());
#else
        return 1;
#endif
    }

    /// Explicit team size for distributed kernels. Querying this on a CPU
    /// worker gives the same policy as on the controller; OpenMP's per-thread
    /// inherited max-team setting is deliberately not the policy source.
    inline int DistributedThreads() noexcept
    {
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
        if (omp_in_parallel())
            return 1;
        return std::max(1, get_env_DNDS_DIST_OMP_NUM_THREADS());
#else
        return 1;
#endif
    }

    /** Independent row arithmetic; no lease, MPI, allocation or exception work.
     * @pre body writes only its row or thread-local state. Read-only metadata
     * is already published, and outputs cannot alias another row's input.
     * Ordered SGS/SOR scans must use subpartition policies instead.
     * One thread follows the original serial order with no OpenMP region.
     */
    template <class Body>
    void ForRows(index begin, index end, Body &&body) noexcept
    {
        const int threads = DistributedThreads();
        if (threads == 1 || end - begin <= 1)
        {
            for (index iRow = begin; iRow < end; ++iRow)
                body(iRow);
            return;
        }
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (index iRow = begin; iRow < end; ++iRow)
            body(iRow);
    }

    /** Deterministic setup-only exclusive prefix over nonnegative row counts.
     * count(i) and store(i, offset) must not throw. Static chunks preserve row
     * order; the controller scans only the bounded worker totals.
     */
    template <class Count, class Store>
    index ExclusivePrefixRowsWithThreads(index rowNumber, int requestedThreads,
                                         Count &&count, Store &&store)
    {
        DNDS_check_throw_info(rowNumber >= 0, "negative prefix row number");
        const int threads = int(std::min<index>(std::max(1, requestedThreads),
                                                std::max<index>(rowNumber, 1)));
        std::vector<uint64_t> totals(size_t(threads), 0), bases(size_t(threads), 0);
        int invalid = 0;
#if defined(DNDS_USE_OMP)
#    pragma omp parallel num_threads(threads) reduction(| : invalid)
#endif
        {
            int worker = 0;
#if defined(DNDS_USE_OMP)
            worker = omp_get_thread_num();
#endif
            const index common = rowNumber / threads;
            const index remainder = rowNumber % threads;
            const index begin = common * worker + std::min<index>(worker, remainder);
            const index end = begin + common + (worker < remainder);
            uint64_t local = 0;
            for (index row = begin; row < end; ++row)
            {
                const index value = count(row);
                if (value < 0 || uint64_t(value) > uint64_t(std::numeric_limits<index>::max()) - local)
                    invalid = 1;
                else
                    local += uint64_t(value);
            }
            totals[size_t(worker)] = local;
        }
        DNDS_check_throw_info(!invalid, "setup prefix count is negative or overflows a chunk");
        uint64_t total = 0;
        for (int worker = 0; worker < threads; ++worker)
        {
            bases[size_t(worker)] = total;
            DNDS_check_throw_info(totals[size_t(worker)] <= uint64_t(std::numeric_limits<index>::max()) - total,
                                  "setup prefix total overflows index");
            total += totals[size_t(worker)];
        }
#if defined(DNDS_USE_OMP)
#    pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int worker = 0; worker < threads; ++worker)
        {
            const index common = rowNumber / threads;
            const index remainder = rowNumber % threads;
            const index begin = common * worker + std::min<index>(worker, remainder);
            const index end = begin + common + (worker < remainder);
            uint64_t local = bases[size_t(worker)];
            for (index row = begin; row < end; ++row)
            {
                store(row, index(local));
                local += uint64_t(count(row));
            }
        }
        store(rowNumber, index(total));
        return index(total);
    }

    template <class Count, class Store>
    index ExclusivePrefixRows(index rowNumber, Count &&count, Store &&store)
    {
        return ExclusivePrefixRowsWithThreads(
            rowNumber, DistributedThreads(), std::forward<Count>(count),
            std::forward<Store>(store));
    }

    template <class Count, class Store>
    index ExclusivePrefixIngressRows(index rowNumber, Count &&count,
                                     Store &&store)
    {
        return ExclusivePrefixRowsWithThreads(
            rowNumber, IngressThreads(), std::forward<Count>(count),
            std::forward<Store>(store));
    }

    /** Independent row arithmetic with explicit numerical failure status.
     * Successful rows have no shared-counter/atomic update. Each worker keeps
     * its first failure; only failing workers enter the final error merge.
     * The lowest logical row wins, independently of OpenMP scheduling. As with
     * every failed kernel, partial output must not be published by the caller.
     * body must not throw or perform MPI, leases, or mutable shared work.
     */
    template <class Body>
    NumericalStatus ForRowsStatus(index begin, index end, Body &&body) noexcept
    {
        const int threads = DistributedThreads();
        if (threads == 1 || end - begin <= 1)
        {
            for (index iRow = begin; iRow < end; ++iRow)
                if (const auto status = body(iRow); !status)
                    return status;
            return {};
        }
        NumericalStatus result;
        index failedRow = end;
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel num_threads(threads)
#endif
        {
            NumericalStatus local;
            index localRow = end;
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp for schedule(static)
#endif
            for (index iRow = begin; iRow < end; ++iRow)
                if (local)
                    if (const auto status = body(iRow); !status)
                    {
                        local = status;
                        localRow = iRow;
                    }
            if (!local)
            {
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp critical(DNDS_OMP_ROW_FAILURE)
#endif
                {
                    if (localRow < failedRow)
                    {
                        failedRow = localRow;
                        result = local;
                    }
                }
            }
        }
        return result;
    }

    /** Deterministic component-sum reduction over contiguous row ranges.
     * @pre partials is empty (serial oracle), or contains at least
     * DistributedThreads()*sum.size() values reserved by the operator.
     * body(row, localSum) performs only arithmetic and returns a numerical
     * status. No allocations/atomics occur here. Each partition accumulates
     * in increasing row order; the final merge follows partition order.
     * OMP=1 keeps the original addition order directly in sum. On failure,
     * all numerical outputs are unspecified and must not be published.
     */
    template <class Body>
    NumericalStatus SumRows(index rows, SpanView<real> sum, SpanView<real> partials, Body &&body) noexcept
    {
        const index threads = DistributedThreads();
        if (threads == 1 || partials.empty() || rows <= 1)
        {
            for (index iRow = 0; iRow < rows; ++iRow)
                if (const auto status = body(iRow, sum); !status)
                    return status;
            return {};
        }
        const index partitions = std::min(rows, threads), width = sum.size();
        DNDS_assert(partials.size() >= partitions * width);
        const auto status = ForRowsStatus(0, partitions, [&](index part) -> NumericalStatus
                                          {
            SpanView<real> local{partials.data() + part * width, width};
            for (index j = 0; j < width; ++j) local[j] = 0;
            const index begin = part * (rows / partitions) + std::min(part, rows % partitions);
            const index end = begin + rows / partitions + (part < rows % partitions);
            for (index iRow = begin; iRow < end; ++iRow)
                if (const auto failure = body(iRow, local); !failure) return failure;
            return {}; });
        if (!status)
            return status;
        for (index part = 0; part < partitions; ++part)
            for (index j = 0; j < width; ++j)
                sum[j] += partials[part * width + j];
        return {};
    }

    /** Setup-only parallel work, allowing allocating/validating library calls.
     * Exceptions are retained until every worker has joined, then rethrown on
     * the controller (whose collective stage applies its usual failure policy).
     * No work item may call MPI. No exception crosses an OpenMP team boundary.
     * Do not use this helper for hot numerical kernels or numerical fallback.
     */
    template <class Body>
    void ForSetupRowsWithThreads(index begin, index end, int threads,
                                 Body &&body)
    {
        if (threads <= 1 || end - begin <= 1)
        {
            for (index iRow = begin; iRow < end; ++iRow)
                body(iRow);
            return;
        }
        std::mutex failureMutex;
        std::exception_ptr failure;
        index failedRow = end;
#if defined(DNDS_USE_OMP)
#    pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (index iRow = begin; iRow < end; ++iRow)
        {
            try
            {
                body(iRow);
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(failureMutex);
                if (iRow < failedRow)
                {
                    failedRow = iRow;
                    failure = std::current_exception();
                }
            }
        }
        if (failure)
            std::rethrow_exception(failure);
    }

    template <class Body>
    void ForSetupRows(index begin, index end, Body &&body)
    {
        ForSetupRowsWithThreads(begin, end, DistributedThreads(),
                                std::forward<Body>(body));
    }

    /** Reader-owned setup rows. Unlike ForSetupRows, this follows the ordinary
     * `OMP_NUM_THREADS` policy and remains available without distributed-MT.
     */
    template <class Body>
    void ForIngressRows(index begin, index end, Body &&body)
    {
        ForSetupRowsWithThreads(begin, end, IngressThreads(),
                                std::forward<Body>(body));
    }

    /**
     * @brief Restore the ordinary OpenMP team for reader-owned ingress.
     *
     * @details Reader-owned file import and serial partitioning may use a much
     * larger share of a node than each distributed MPI rank. Direct OpenMP
     * regions in that phase therefore follow `OMP_NUM_THREADS`, independently
     * of the later per-rank native-kernel policy.
     */
    inline void set_serial_ingress_OMP()
    {
#ifdef DNDS_USE_OMP
        omp_set_num_threads(std::max(1, get_env_OMP_NUM_THREADS()));
#endif
        Eigen::setNbThreads(1);
    }

    /**
     * @brief Configure OpenMP and Eigen threading for the "distributed OMP" pattern.
     *
     * @details Sets the OpenMP thread count from the @ref DNDS_DIST_OMP_NUM_THREADS
     * environment variable (falling back to 1) and disables Eigen's internal
     * parallelism so it does not oversubscribe the MPI+OpenMP layout.
     *
     * Intended to be called once near `main()` when @ref DNDS_USE_OMP is on.
     */
    inline void set_full_parallel_OMP()
    {
#ifdef DNDS_USE_OMP
        omp_set_num_threads(DistributedThreads());
#endif
        Eigen::setNbThreads(1); // ! sets Eigen's multithreading!
    }
}
