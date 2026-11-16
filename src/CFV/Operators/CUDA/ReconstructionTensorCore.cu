#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "DNDS/Algebra/CUDA/FP64Matrix.cuh"
#include <algorithm>
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    /// Four independent cells per block, one warp per cell. Matrix products use
    /// padded8x8x4 FP64 MMA; scalar mean/BC terms retain the ordinary FV formula.
    __global__ void JacobiTensorCore(ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                                     R::PairReadView<real> mean, R::PairReadView<real> input,
                                     R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if (status(0, 0))
            return;
        __shared__ Algebra::CUDA::FP64Tile8 tiles[4];
        const auto lane = index(threadIdx.x % 32), warp = index(threadIdx.x / 32);
        auto &tile = tiles[warp];
        const auto components = output.width / plan.n;
        for (index iWork = index(blockIdx.x) * 4 + warp; iWork < selection.count; iWork += index(gridDim.x) * 4)
        {
            const auto iCell = selection.Cell(iWork);
            for (index modeBegin = 0; modeBegin < plan.n; modeBegin += 8)
                for (index componentBegin = 0; componentBegin < components; componentBegin += 8)
                {
                    Algebra::CUDA::FP64Accumulator8 product;
                    for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
                    {
                        const auto other = plan.neighbors(entry, 0);
                        if (other == UnInitIndex)
                            continue;
                        for (index kBegin = 0; kBegin < plan.n; kBegin += 4)
                        {
                            const auto mode = modeBegin + lane / 4, kA = kBegin + lane % 4;
                            tile.a[lane] = mode < plan.n && kA < plan.n ? plan.matrixAInvB(entry, kA * plan.n + mode) : 0;
                            const auto component = componentBegin + lane % 8, kB = kBegin + lane / 8;
                            real value = 0;
                            if (component < components && kB < plan.n)
                            {
                                const auto first = plan.rotated && plan.rotationActive(entry, 0)
                                                       ? vectors.First(component)
                                                       : -1;
                                if (first >= 0)
                                    for (index d = 0; d < 3; ++d)
                                        value += plan.rotation(entry, d * 3 + component - first) * input(other, (first + d) * plan.n + kB);
                                else
                                    value = input(other, component * plan.n + kB);
                            }
                            tile.b[lane] = value;
                            __syncwarp();
                            product.AddProduct(tile);
                            __syncwarp();
                        }
                    }
                    product.Store(tile);
                    __syncwarp();
                    for (index i = lane; i < 64; i += 32)
                    {
                        const auto mode = modeBegin + i / 8, component = componentBegin + i % 8;
                        if (mode >= plan.n || component >= components)
                            continue;
                        real value = (1 - plan.relax) * input(iCell, component * plan.n + mode) + plan.relax * tile.c[i];
                        const auto first = plan.rotated ? vectors.First(component) : -1;
                        for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
                        {
                            const auto other = plan.neighbors(entry, 0);
                            if (other != UnInitIndex)
                            {
                                real otherMean = mean(other, component);
                                if (first >= 0 && plan.rotationActive(entry, 0))
                                {
                                    otherMean = 0;
                                    for (index d = 0; d < 3; ++d)
                                        otherMean += plan.rotation(entry, d * 3 + component - first) * mean(other, first + d);
                                }
                                value += plan.vectorAInvB(entry, mode) * (plan.relax * (otherMean - mean(iCell, component)));
                            }
                            else
                                for (index q = plan.boundaryOffsets(entry, 0); q < plan.boundaryOffsets(entry + 1, 0); ++q)
                                    value += plan.boundaryInjection(q, mode) * (plan.relax * (boundary(q, component) - mean(iCell, component)));
                        }
                        output(iCell, component * plan.n + mode) = value;
                    }
                    __syncwarp();
                }
        }
#else
        // An explicit capability guard prevents this unsupported image from
        // being launched. Never silently run the ordinary FP64 kernel here.
        asm("trap;");
#endif
    }
    bool FP64TensorCoreAvailable(R::Device device)
    {
        if (device.space != R::MemorySpace::CUDA || cudaSetDevice(device.ordinal) != cudaSuccess)
            return false;
        cudaDeviceProp properties;
        cudaFuncAttributes attributes;
        if (cudaGetDeviceProperties(&properties, device.ordinal) != cudaSuccess || cudaFuncGetAttributes(&attributes, JacobiTensorCore) != cudaSuccess)
            return false;
        // Explicit initial support: datacenter Ampere/Hopper FP64 MMA. Do not
        // infer double tensor support for consumer/future devices from CC alone.
        // JIT can raise binaryVersion for older PTX containing only our trap
        // stub; require a real sm80-or-newer frontend image as well.
        return ((properties.major == 8 && properties.minor == 0) || (properties.major == 9 && properties.minor == 0)) &&
               attributes.binaryVersion >= 80 && attributes.ptxVersion >= 80;
    }
    void LaunchJacobiTensorCore(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                                R::PairReadView<real> mean, R::PairReadView<real> input, R::ArrayView<const real> boundary,
                                R::ArrayView<real> output, R::ArrayView<const int> status)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA tensor Jacobi requires a CUDA context");
        DNDS_check_throw_info(!plan.quantized && !plan.raw, "FP64 tensor Jacobi requires FP64 sweep maps");
        R::ValidateViewDevices(context.device, plan, selection, vectors, mean, input, boundary, output, status);
        R::CheckKernelArgumentBudget<decltype(plan), decltype(selection), decltype(vectors), decltype(mean),
                                     decltype(input), decltype(boundary), decltype(output), decltype(status)>();
        if (!selection.count)
            return;
        JacobiTensorCore<<<unsigned(std::min<index>(256, (selection.count + 3) / 4)), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(plan, selection, vectors, mean, input, boundary, output, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA FP64 tensor-core Jacobi launch failed");
    }
}
