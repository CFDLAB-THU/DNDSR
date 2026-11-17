#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "DNDS/Algebra/CUDA/CountRows.cuh"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        constexpr int Threads = 128;
        template <bool Partial>
        __global__ void LimiterStatistics(StateInput beta, LimiterCountOutput partialCount,
                                          LimiterMinimumOutput partialMinimum,
                                          LimiterCountOutput count, LimiterMinimumOutput minimum)
        {
            __shared__ index counts[Threads];
            __shared__ real minima[Threads];
            index localCount = 0;
            real localMinimum = 1;
            const index rows = Partial ? beta.RowCount() : partialCount.RowCount();
            for (index iRow = index(blockIdx.x) * Threads + threadIdx.x; iRow < rows; iRow += index(gridDim.x) * Threads)
                if constexpr (Partial)
                {
                    const real value = beta(iRow, 0);
                    localCount += value < 1;
                    localMinimum = fmin(localMinimum, value);
                }
                else
                {
                    localCount += partialCount(iRow, 0);
                    localMinimum = fmin(localMinimum, partialMinimum(iRow, 0));
                }
            counts[threadIdx.x] = localCount;
            minima[threadIdx.x] = localMinimum;
            __syncthreads();
            for (int stride = Threads / 2; stride; stride /= 2)
            {
                if (threadIdx.x < stride)
                {
                    counts[threadIdx.x] += counts[threadIdx.x + stride];
                    minima[threadIdx.x] = fmin(minima[threadIdx.x], minima[threadIdx.x + stride]);
                }
                __syncthreads();
            }
            if (!threadIdx.x)
                if constexpr (Partial)
                {
                    partialCount(blockIdx.x, 0) = counts[0];
                    partialMinimum(blockIdx.x, 0) = minima[0];
                }
                else
                {
                    count(0, 0) = counts[0];
                    minimum(0, 0) = minima[0];
                }
        }
    }
    void ReduceLimiterStatistics(R::ExecutionContext context, StateInput beta,
                                 LimiterCountOutput partialCount, LimiterMinimumOutput partialMinimum,
                                 LimiterCountOutput count, LimiterMinimumOutput minimum)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "limiter reduction requires CUDA");
        R::ValidateViewDevices(context.device, beta, partialCount, partialMinimum, count, minimum);
        DNDS_check_throw_info(beta.Width() == 1 && partialCount.RowCount() == StateReductionScratchRows(beta.RowCount()) &&
                                  partialMinimum.RowCount() == partialCount.RowCount() && count.RowCount() == 1 && minimum.RowCount() == 1 &&
                                  partialCount.data != count.data && partialMinimum.data != minimum.data &&
                                  beta.data != partialMinimum.data && beta.data != minimum.data,
                              "limiter reduction shape or alias differs");
        R::CheckKernelArgumentBudget<StateInput, LimiterCountOutput, LimiterMinimumOutput, LimiterCountOutput, LimiterMinimumOutput>();
        const auto stream = static_cast<cudaStream_t>(context.stream);
        LimiterStatistics<true><<<unsigned(partialCount.RowCount()), Threads, 0, stream>>>(beta, partialCount, partialMinimum, count, minimum);
        LimiterStatistics<false><<<1, Threads, 0, stream>>>(beta, partialCount, partialMinimum, count, minimum);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "limiter statistics launch failed");
    }
    void CountFlags(R::ExecutionContext context,
                    R::ArrayView<const uint8_t, R::AccessBoundsChecks, R::FixedRows<1>> flags,
                    LimiterCountOutput partialCount, LimiterCountOutput count)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "flag reduction requires CUDA");
        R::ValidateViewDevices(context.device, flags, partialCount, count);
        DNDS_check_throw_info(partialCount.RowCount() == StateReductionScratchRows(flags.RowCount()) &&
                                  count.RowCount() == 1 && partialCount.data != count.data,
                              "flag reduction shape or alias differs");
        R::CheckKernelArgumentBudget<decltype(flags), LimiterCountOutput, LimiterCountOutput>();
        const auto stream = static_cast<cudaStream_t>(context.stream);
        Algebra::CUDA::CountRows<Threads, true><<<unsigned(partialCount.RowCount()), Threads, 0, stream>>>(flags, partialCount);
        Algebra::CUDA::CountRows<Threads, false><<<1, Threads, 0, stream>>>(partialCount.ConstView(), count);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "flag reduction launch failed");
    }
}
