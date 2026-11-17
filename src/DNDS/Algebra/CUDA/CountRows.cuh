#pragma once

#include "DNDS/Core/Defines.hpp"
#include <cuda_runtime.h>

namespace DNDS::Algebra::CUDA
{
    /** @brief Fixed-tree integer row reduction for diagnostic counters.
     * @details CountNonzero converts flags to 0/1; otherwise input values are
     * summed unchanged (including partial counts). Each CTA writes exactly
     * one output row. Empty input produces zero from every launched CTA.
     * The host wrapper validates width, output capacity, placement and aliasing;
     * this leaf allocates nothing and performs no host observation or atomics.
     * Numerical floating-point sums have their own reduction/order contract.
     */
    template <int Threads, bool CountNonzero, class Input, class Output>
    __global__ void CountRows(Input input, Output output)
    {
        static_assert(Threads > 0 && (Threads & (Threads - 1)) == 0);
        __shared__ index counts[Threads];
        index count = 0;
        for (index iRow = index(blockIdx.x) * Threads + threadIdx.x;
             iRow < input.RowCount(); iRow += index(gridDim.x) * Threads)
            if constexpr (CountNonzero)
                count += input(iRow, 0) != 0;
            else
                count += input(iRow, 0);
        counts[threadIdx.x] = count;
        __syncthreads();
        for (int stride = Threads / 2; stride; stride /= 2)
        {
            if (threadIdx.x < stride)
                counts[threadIdx.x] += counts[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            output(blockIdx.x, 0) = counts[0];
    }
}
