#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "DNDS/Algebra/CUDA/FP64Matrix.cuh"
#include <algorithm>
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    template <bool Async>
    __device__ void StageWord(real *destination, const real *source)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (Async)
        {
            const unsigned address = unsigned(__cvta_generic_to_shared(destination));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(address), "l"(source) : "memory");
        }
        else
            *destination = *source;
#endif
    }
    /// One block/cell, four concurrent component tiles. Raw matrices are staged
    /// once per neighbor and reused by every warp. Async stages the next neighbor
    /// while the current product executes; no host input or extra allocation.
    template <int N, int Components, bool Async>
    __global__ void JacobiShared(ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                                 R::PairReadView<real> mean, R::PairReadView<real> input,
                                 R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if (status(0, 0))
            return;
        __shared__ real matrices[2][N * N], coefficients[2][4][8 * N];
        __shared__ Algebra::CUDA::FP64Tile8 tiles[4];
        const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        auto &tile = tiles[warp];
        const index components = Components ? Components : output.width / N;
        for (index iWork = blockIdx.x; iWork < selection.count; iWork += gridDim.x)
        {
            const auto iCell = selection.Cell(iWork);
            const index begin = plan.offsets(iCell, 0), end = plan.offsets(iCell + 1, 0);
            for (int modeBegin = 0; modeBegin < N; modeBegin += 8)
                for (index componentRound = 0; componentRound < components; componentRound += 32)
                {
                    const index componentBegin = componentRound + warp * 8;
                    auto load = [&](index entry, int slot)
                    {
                        const auto other = plan.neighbors(entry, 0);
                        for (int j = threadIdx.x; j < N * N; j += blockDim.x)
                            StageWord<Async>(&matrices[slot][j], &plan.matrixAInvB(entry, j));
                        for (int j = lane; j < 8 * N; j += 32)
                        {
                            const auto component = componentBegin + j / N;
                            if (other != UnInitIndex && component < components)
                                StageWord<Async>(&coefficients[slot][warp][j], &input(other, component * N + j % N));
                            else
                                coefficients[slot][warp][j] = 0;
                        }
                        if constexpr (Async)
                            asm volatile("cp.async.commit_group;" ::: "memory");
                    };
                    Algebra::CUDA::FP64Accumulator8 product;
                    if (begin < end)
                        load(begin, 0);
                    for (index entry = begin; entry < end; ++entry)
                    {
                        const int slot = int((entry - begin) % 2);
                        if constexpr (Async)
                            asm volatile("cp.async.wait_group 0;" ::: "memory");
                        __syncthreads();
                        if constexpr (Async)
                        {
                            if (entry + 1 < end)
                                load(entry + 1, 1 - slot);
                        }
                        if (plan.neighbors(entry, 0) != UnInitIndex)
                            for (int kBegin = 0; kBegin < N; kBegin += 4)
                            {
                                const int mode = modeBegin + lane / 4, ka = kBegin + lane % 4;
                                tile.a[lane] = mode < N && ka < N ? matrices[slot][ka * N + mode] : 0;
                                const index component = componentBegin + lane % 8;
                                const int kb = kBegin + lane / 8;
                                real value = 0;
                                if (component < components && kb < N)
                                {
                                    const auto first = plan.rotated && plan.rotationActive(entry, 0)
                                                           ? vectors.First(component)
                                                           : -1;
                                    if (first >= 0)
                                        for (index d = 0; d < 3; ++d)
                                            value += plan.rotation(entry, d * 3 + component - first) * input(plan.neighbors(entry, 0), (first + d) * N + kb);
                                    else
                                        value = coefficients[slot][warp][int(lane % 8) * N + kb];
                                }
                                tile.b[lane] = value;
                                __syncwarp();
                                product.AddProduct(tile);
                                __syncwarp();
                            }
                        __syncthreads();
                        if constexpr (!Async)
                        {
                            if (entry + 1 < end)
                                load(entry + 1, 1 - slot);
                        }
                    }
                    product.Store(tile);
                    __syncwarp();
                    for (index i = lane; i < 64; i += 32)
                    {
                        const index mode = modeBegin + i / 8, component = componentBegin + i % 8;
                        if (mode >= N || component >= components)
                            continue;
                        real value = (1 - plan.relax) * input(iCell, component * N + mode) + plan.relax * tile.c[i];
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
                        output(iCell, component * N + mode) = value;
                    }

                    __syncthreads();
                }
        }
#else
        asm("trap;");
#endif
    }
    template <int N, int Components>
    void LaunchSharedShape(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                           R::PairReadView<real> mean, R::PairReadView<real> input, R::ArrayView<const real> boundary,
                           R::ArrayView<real> output, R::ArrayView<const int> status, bool async)
    {
        // No per-block global scratch is used here. A larger grid can occupy
        // the register-limited resident blocks instead of inheriting WBAP's
        // fixed scratch/block cap from the original baseline launcher.
        const auto blocks = unsigned(std::min<index>(1024, selection.count));
        const auto stream = static_cast<cudaStream_t>(context.stream);
        if (async)
            JacobiShared<N, Components, true><<<blocks, 128, 0, stream>>>(plan, selection, vectors, mean, input, boundary, output, status);
        else
            JacobiShared<N, Components, false><<<blocks, 128, 0, stream>>>(plan, selection, vectors, mean, input, boundary, output, status);
    }
    void LaunchJacobiTensorCoreShared(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                                      R::PairReadView<real> mean, R::PairReadView<real> input, R::ArrayView<const real> boundary,
                                      R::ArrayView<real> output, R::ArrayView<const int> status, bool async)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA shared tensor Jacobi requires a CUDA context");
        DNDS_check_throw_info(!plan.quantized && !plan.raw, "FP64 shared tensor Jacobi requires FP64 sweep maps");
        R::ValidateViewDevices(context.device, plan, selection, vectors, mean, input, boundary, output, status);
        R::CheckKernelArgumentBudget<decltype(plan), decltype(selection), decltype(vectors), decltype(mean),
                                     decltype(input), decltype(boundary), decltype(output), decltype(status)>();
        if (!selection.count)
            return;
#define DNDS_LAUNCH(N)                                                                                                 \
    case N:                                                                                                            \
        if (output.width / N == 5)                                                                                     \
            LaunchSharedShape<N, 5>(context, plan, selection, vectors, mean, input, boundary, output, status, async);  \
        else if (output.width / N == 21)                                                                               \
            LaunchSharedShape<N, 21>(context, plan, selection, vectors, mean, input, boundary, output, status, async); \
        else                                                                                                           \
            LaunchSharedShape<N, 0>(context, plan, selection, vectors, mean, input, boundary, output, status, async);  \
        break
        switch (plan.n)
        {
            DNDS_LAUNCH(2);
            DNDS_LAUNCH(5);
            DNDS_LAUNCH(9);
            DNDS_LAUNCH(3);
            DNDS_LAUNCH(19);
        default:
            DNDS_check_throw_info(false, "unsupported shared tensor mode count");
        }
#undef DNDS_LAUNCH
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "shared tensor Jacobi launch failed");
    }
}
