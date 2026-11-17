#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Algebra/LocalProducts.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "DNDS/Device/ReductionCompletion.hpp"
#include <cuda_runtime.h>
#include <cmath>

namespace DNDS::Algebra::Detail
{
    namespace R = Runtime;
    namespace
    {
        constexpr int Threads = 256;
        template <bool Finite>
        __global__ void ReduceRows(ProductInput x, ProductInput y, ProductInput scales, ProductOutput output,
                                   ProductOutput result, ProductCompletion completed, index slot)
        {
            __shared__ real tree[Threads];
            __shared__ bool lastBlock;
            real value = Finite ? 1 : 0;
            const index count = x.RowCount() * x.Width();
            for (index i = index(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += index(gridDim.x) * blockDim.x)
            {
                const index iRow = i / x.Width(), j = i % x.Width();
                if constexpr (Finite)
                    value = std::isfinite(x(iRow, j)) ? value : 0;
                else
                {
                    const real scale = scales.RowCount() ? scales(j, 0) : 1;
                    value += (x(iRow, j) * scale) * (y(iRow, j) * scale);
                }
            }
            tree[threadIdx.x] = value;
            __syncthreads();
            for (int width = Threads / 2; width > 0; width /= 2)
            {
                if (threadIdx.x < width)
                {
                    if constexpr (Finite)
                        tree[threadIdx.x] = tree[threadIdx.x] && tree[threadIdx.x + width];
                    else
                        tree[threadIdx.x] += tree[threadIdx.x + width];
                }
                __syncthreads();
            }
            if (!threadIdx.x)
            {
                output(blockIdx.x, 0) = tree[0];
                lastBlock = Device::ArriveReductionBlock(completed(0, 0), gridDim.x);
            }
            __syncthreads();
            if (!lastBlock)
                return;
            // Acquire/release election plus this block's barrier makes every
            // partial visible. Fixed indices preserve the original tail tree;
            // no floating sum depends on block arrival order.
            tree[threadIdx.x] = threadIdx.x < output.RowCount() ? output(threadIdx.x, 0) : Finite ? 1
                                                                                                  : 0;
            __syncthreads();
            for (int width = Threads / 2; width > 0; width /= 2)
            {
                if (threadIdx.x < width)
                {
                    if constexpr (Finite)
                        tree[threadIdx.x] = tree[threadIdx.x] && tree[threadIdx.x + width];
                    else
                        tree[threadIdx.x] += tree[threadIdx.x + width];
                }
                __syncthreads();
            }
            if (!threadIdx.x)
                result(slot, 0) = tree[0];
        }
    }
    void ReduceLocalProductCUDA(R::ExecutionContext context, ProductInput x, ProductInput y,
                                ProductInput scales, ProductOutput partial, ProductOutput result, ProductCompletion completed, index slot, bool finite)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && x.RowCount() >= 0 && x.Width() > 0 &&
                                  x.RowCount() <= INT64_MAX / x.Width() && y.RowCount() == x.RowCount() && y.Width() == x.Width() &&
                                  (scales.RowCount() == 0 || scales.RowCount() == x.Width()) && scales.Width() == 1 &&
                                  partial.RowCount() >= 1 && partial.RowCount() <= Threads && partial.Width() == 1 &&
                                  result.Width() == 1 && slot >= 0 && slot < result.RowCount() && completed.RowCount() == 1,
                              "invalid CUDA local reduction views");
        R::ValidateViewDevices(context.device, x, y, scales, partial, result, completed);
        for (const auto input : {x, y, scales})
            DNDS_check_throw_info(!input.data || (input.data != partial.data && input.data != result.data), "CUDA local reduction aliases input");
        DNDS_check_throw_info(partial.data != result.data, "CUDA local reduction scratch aliases result");
        for (const auto buffer : {x.data, y.data, scales.data, static_cast<const real *>(partial.data), static_cast<const real *>(result.data)})
            DNDS_check_throw_info(static_cast<const void *>(buffer) != completed.data, "CUDA local reduction counter aliases payload");
        R::CheckKernelArgumentBudget<ProductInput, ProductInput, ProductInput, ProductOutput, ProductOutput, ProductCompletion, index>();
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        DNDS_check_throw_info(cudaMemsetAsync(completed.data, 0, sizeof(unsigned int), stream) == cudaSuccess,
                              "CUDA local reduction counter reset failed");
        if (finite)
            ReduceRows<true><<<unsigned(partial.RowCount()), Threads, 0, stream>>>(x, y, scales, partial, result, completed, slot);
        else
            ReduceRows<false><<<unsigned(partial.RowCount()), Threads, 0, stream>>>(x, y, scales, partial, result, completed, slot);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA local reduction launch failed");
    }
}
