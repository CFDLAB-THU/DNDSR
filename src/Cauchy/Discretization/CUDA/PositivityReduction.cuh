#pragma once

#include "Cauchy/Discretization/CUDA/ModalPositivityKernels.hpp"
#include <cub/block/block_reduce.cuh>

namespace DNDS::Cauchy::CUDA::PositivityReductionDetail
{
    struct FirstFailure
    {
        __device__ NumericalStatus operator()(NumericalStatus a, NumericalStatus b) const
        {
            if (a)
                return b;
            if (b)
                return a;
            return b.row < a.row || (b.row == a.row && b.point < a.point) ? b : a;
        }
    };
    struct SummaryValue
    {
        NumericalStatus status;
        index count;
        real minimum;
    };
    struct MergeSummary
    {
        __device__ SummaryValue operator()(SummaryValue a, SummaryValue b) const
        {
            return {FirstFailure{}(a.status, b.status), a.count + b.count, std::min(a.minimum, b.minimum)};
        }
    };
    /** @brief Producing-kernel epilogue over all CTA lanes.
     * @details Successful idle/tail lanes supply {success,0,1}. CUB's raw temp
     * storage avoids constructing nontrivial status objects in shared memory.
     * Counts/minima are exact/order-independent, failures use deterministic
     * row/point ordering. No floating-point sum or full-cell scan is introduced.
     */
    template <int Threads>
    __device__ void PublishPositivitySummary(SummaryValue local, ModalPositivitySummary output)
    {
        using Reduce = cub::BlockReduce<SummaryValue, Threads>;
        __shared__ typename Reduce::TempStorage temporary;
        const auto result = Reduce(temporary).Reduce(local, MergeSummary{});
        if (threadIdx.x == 0)
        {
            output.status(blockIdx.x, 0) = result.status;
            output.count(blockIdx.x, 0) = result.count;
            output.minimum(blockIdx.x, 0) = result.minimum;
        }
    }
    /// Status-only counterpart for a producer without limiter diagnostics.
    template <int Threads, class StatusOutput>
    __device__ void PublishStatusSummary(NumericalStatus local, StatusOutput output)
    {
        using Reduce = cub::BlockReduce<NumericalStatus, Threads>;
        __shared__ typename Reduce::TempStorage temporary;
        const auto result = Reduce(temporary).Reduce(local, FirstFailure{});
        if (threadIdx.x == 0)
            output(blockIdx.x, 0) = result;
    }
}
