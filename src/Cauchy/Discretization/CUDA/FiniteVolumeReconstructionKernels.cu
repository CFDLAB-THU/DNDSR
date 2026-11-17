#include "Cauchy/Discretization/CUDA/FiniteVolumeReconstructionKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        using NormInput = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
        using NormOutput = R::ArrayView<real, R::AccessBoundsChecks, R::FixedRows<2>>;
        template <bool Partial>
        __global__ void UpdateNorm(NormInput current, NormInput next, NormOutput partial, NormOutput result)
        {
            __shared__ real differences[128], values[128];
            real difference = 0, value = 0;
            const index rows = Partial ? next.RowCount() : partial.RowCount();
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < rows; iRow += index(gridDim.x) * blockDim.x)
            {
                if constexpr (Partial)
                    for (index j = 0; j < next.Width(); ++j)
                    {
                        const real b = next(iRow, j), delta = b - current(iRow, j);
                        difference += delta * delta;
                        value += b * b;
                    }
                else
                    difference += partial(iRow, 0), value += partial(iRow, 1);
            }
            differences[threadIdx.x] = difference;
            values[threadIdx.x] = value;
            __syncthreads();
            for (int stride = 64; stride; stride /= 2)
            {
                if (threadIdx.x < stride)
                {
                    differences[threadIdx.x] += differences[threadIdx.x + stride];
                    values[threadIdx.x] += values[threadIdx.x + stride];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                const auto output = Partial ? partial : result;
                output(blockIdx.x, 0) = differences[0];
                output(blockIdx.x, 1) = values[0];
            }
        }
    }
    void ReconstructionUpdateNorm(R::ExecutionContext context, NormInput current, NormInput next,
                                  NormOutput partial, NormOutput result)
    {
        R::ValidateViewDevices(context.device, current, next, partial, result);
        const index blocks = std::max<index>(1, std::min<index>(256, (next.RowCount() + 127) / 128));
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && current.RowCount() == next.RowCount() &&
                                  current.Width() == next.Width() && partial.RowCount() == blocks && result.RowCount() == 1,
                              "reconstruction norm shape or backend differs");
        R::CheckKernelArgumentBudget<NormInput, NormInput, NormOutput, NormOutput>();
        const auto stream = static_cast<cudaStream_t>(context.stream);
        UpdateNorm<true><<<unsigned(blocks), 128, 0, stream>>>(current, next, partial, result);
        UpdateNorm<false><<<1, 128, 0, stream>>>(current, next, partial, result);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "reconstruction norm reduction launch failed");
    }
}
