#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "CFV/Math/ReconstructionMatrixAccess.hpp"
#include "DNDS/Algebra/CUDA/FP64Matrix.cuh"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    using RawTensorView = RawReconstructionSweepView;
    /** @brief Warp-per-cell raw B gather and local-inverse FP64 DMMA.
     * @details The first MMA accumulates unscaled, oriented neighbor B products.
     * Scalar mean/boundary contributions complete the shared raw right-hand side.
     * A second MMA applies the unchanged local SVD pseudoinverse; triangle packing
     * is decoded into shared staging, never treated as a triangular solve.
     * Modes/components beyond the logical shape are zero padding. No allocation,
     * lease acquisition, communication or host synchronization occurs here.
     */
    __global__ void RawJacobiTensorCore(RawTensorView plan, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                                        R::PairReadView<real> mean, R::PairReadView<real> input,
                                        R::ArrayView<const real> boundary, R::ArrayView<real> output,
                                        R::ArrayView<const int> status)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if (status(0, 0))
            return;
        __shared__ Algebra::CUDA::FP64Tile8 tiles[4];
        __shared__ real right[4][19 * 7];
        const index lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        const index n = plan.n, components = output.width / n;
        auto &tile = tiles[warp];
        for (index iWork = index(blockIdx.x) * 4 + warp; iWork < selection.count; iWork += index(gridDim.x) * 4)
        {
            const index iCell = selection.Cell(iWork);
            for (index modeBegin = 0; modeBegin < n; modeBegin += 8)
            {
                Algebra::CUDA::FP64Accumulator8 gather;
                for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
                {
                    const index other = plan.neighbors(entry, 0);
                    const bool rotated = plan.rotated && plan.rotationActive(entry, 0);
                    if (other == UnInitIndex || (other == iCell && !rotated))
                        continue;
                    const index matrixRow = plan.faceCoupling ? plan.faces(entry, 0) : entry;
                    const bool transpose = plan.faceCoupling && plan.faceSide(entry, 0);
                    const auto block = plan.matrices[matrixRow];
                    for (index kBegin = 0; kBegin < n; kBegin += 4)
                    {
                        const index mode = modeBegin + lane / 4, kA = kBegin + lane % 4;
                        tile.a[lane] = mode < n && kA < n ? block[transpose ? kA + n * mode : mode + n * kA] : 0;
                        const index component = lane % 8, kB = kBegin + lane / 8;
                        real value = 0;
                        if (component < components && kB < n)
                        {
                            const index first = rotated ? vectors.First(component) : -1;
                            if (first >= 0)
                                for (index d = 0; d < 3; ++d)
                                    value += plan.rotation(entry, d * 3 + component - first) * input(other, (first + d) * n + kB);
                            else
                                value = input(other, component * n + kB);
                        }
                        tile.b[lane] = value;
                        __syncwarp();
                        gather.AddProduct(tile);
                        __syncwarp();
                    }
                }
                gather.Store(tile);
                __syncwarp();
                for (index i = lane; i < 64; i += 32)
                {
                    const index mode = modeBegin + i / 8, component = i % 8;
                    if (mode >= n || component >= components)
                        continue;
                    real value = tile.c[i];
                    const index first = plan.rotated ? vectors.First(component) : -1;
                    for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
                    {
                        const index other = plan.neighbors(entry, 0);
                        if (other != UnInitIndex)
                        {
                            real otherMean = mean(other, component);
                            if (first >= 0 && plan.rotationActive(entry, 0))
                            {
                                otherMean = 0;
                                for (index d = 0; d < 3; ++d)
                                    otherMean += plan.rotation(entry, d * 3 + component - first) * mean(other, first + d);
                            }
                            value += plan.vectors(entry, mode) * (otherMean - mean(iCell, component));
                        }
                        else
                            for (index q = plan.boundaryOffsets(entry, 0); q < plan.boundaryOffsets(entry + 1, 0); ++q)
                                value += plan.injection(q, mode) * (boundary(q, component) - mean(iCell, component));
                    }
                    right[warp][component * n + mode] = value;
                }
                __syncwarp();
            }
            const auto inverse = plan.inverse[iCell];
            for (index modeBegin = 0; modeBegin < n; modeBegin += 8)
            {
                Algebra::CUDA::FP64Accumulator8 correction;
                for (index kBegin = 0; kBegin < n; kBegin += 4)
                {
                    const index mode = modeBegin + lane / 4, kA = kBegin + lane % 4;
                    tile.a[lane] = mode < n && kA < n ? inverse[plan.triangle ? Math::SymmetricTriangleEntry(mode, kA) : mode + n * kA] : 0;
                    const index component = lane % 8, kB = kBegin + lane / 8;
                    tile.b[lane] = component < components && kB < n ? right[warp][component * n + kB] : 0;
                    __syncwarp();
                    correction.AddProduct(tile);
                    __syncwarp();
                }
                correction.Store(tile);
                __syncwarp();
                for (index i = lane; i < 64; i += 32)
                {
                    const index mode = modeBegin + i / 8, component = i % 8;
                    if (mode < n && component < components)
                        output(iCell, component * n + mode) = (1 - plan.relax) * input(iCell, component * n + mode) + plan.relax * tile.c[i];
                }
                __syncwarp();
            }
        }
#else
        asm("trap;");
#endif
    }
    void LaunchRawJacobiTensorCore(R::ExecutionContext context, ReconstructionDeviceView plan,
                                   DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                                   R::PairReadView<real> mean, R::PairReadView<real> input,
                                   R::ArrayView<const real> boundary, R::ArrayView<real> output,
                                   R::ArrayView<const int> status, bool wide)
    {
        // Capability resolution belongs to the prepared plan (or the direct
        // benchmark caller). Do not requery CUDA properties/functions per sweep.
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && plan.raw && !plan.quantized &&
                                  plan.n > 0 && plan.n <= 19 &&
                                  mean.owned.Width() > 0 && mean.owned.Width() <= 7 && output.Width() == mean.owned.Width() * plan.n &&
                                  input.owned.Width() == output.Width() && output.RowCount() == mean.owned.RowCount() &&
                                  plan.matrixAInvB.Width() == plan.n * plan.n && plan.vectorAInvB.Width() == plan.n &&
                                  plan.localInverse.Width() == (plan.inverseTriangle ? plan.n * (plan.n + 1) / 2 : plan.n * plan.n),
                              "raw FP64 tensor VR requires supported FP64-MMA device, P1..P3 and at most seven components");
        R::ValidateViewDevices(context.device, plan, selection, vectors, mean, input, boundary, output, status);
        R::CheckKernelArgumentBudget<RawTensorView, decltype(selection), decltype(vectors), decltype(mean), decltype(input),
                                     decltype(boundary), decltype(output), decltype(status)>();
        if (!selection.count)
            return;
        RawJacobiTensorCore<<<unsigned(std::min<index>((selection.count + 3) / 4, wide ? 65535 : 256)), 128, 0,
                              static_cast<cudaStream_t>(context.stream)>>>(RawTensorView(plan), selection, vectors, mean, input, boundary, output, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "raw FP64 tensor Jacobi launch failed");
    }
}
