#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "CFV/Operators/CUDA/ReconstructionDevice_General.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    /// Candidate SIMT alternative to the reference and DMMA implementations.
    /// Each block stages one neighbor row, then reuses it across target modes.
    /// Keep entry, mode-product, rotation-correction and mean/BC addition order
    /// identical to the general FP64 Jacobi sweep. Barriers are block-uniform, including
    /// lanes with no output slot. No limited/LVR semantics are changed here.
    template <int FixedModes, int Components, class Layout = R::RowLayout>
    __global__ void JacobiSharedSIMT(JacobiView<real, Layout> plan, DeviceWorkSelection selection, DevicePeriodicBlocks blocks,
                                     R::PairReadView<real> mean, R::PairReadView<real> input,
                                     R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
        if (status(0, 0))
            return;
        const index n = FixedModes ? FixedModes : plan.n;
        __shared__ real staticCoefficients[Components ? FixedModes * Components : 1];
        __shared__ real staticMatrix[Components ? FixedModes * FixedModes : 1];
        extern __shared__ real dynamic[];
        real *coefficients = Components ? staticCoefficients : dynamic;
        real *matrix = Components ? staticMatrix : dynamic + output.width;
        const index slots = (output.width + blockDim.x - 1) / blockDim.x;
        for (index iWork = blockIdx.x; iWork < selection.count; iWork += gridDim.x)
        {
            const auto iCell = selection.Cell(iWork);
            real values[Components ? (19 * Components + 95) / 96 : 1]{};
            for (int slot = 0; slot < slots; ++slot)
            {
                const index i = threadIdx.x + slot * blockDim.x;
                if (i < output.width)
                {
                    if constexpr (Components)
                        values[slot] = input(iCell, i) * (1 - plan.relax);
                    else
                        output(iCell, i) = input(iCell, i) * (1 - plan.relax);
                }
            }
            for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
            {
                const auto other = plan.neighbors(entry, 0);
                if (other != UnInitIndex)
                {
                    for (index i = threadIdx.x; i < output.width; i += blockDim.x)
                        coefficients[i] = input(other, i);
                    for (index i = threadIdx.x; i < n * n; i += blockDim.x)
                        matrix[i] = plan.matrixAInvB(entry, i);
                }
                __syncthreads();
                for (int slot = 0; slot < slots; ++slot)
                {
                    const index i = threadIdx.x + slot * blockDim.x;
                    if (i >= output.width)
                        continue;
                    real &value = Components ? values[slot] : output(iCell, i);
                    const index mode = i % n, component = i / n;
                    if (other != UnInitIndex)
                    {
                        real product = 0;
                        for (index j = 0; j < n; ++j)
                        {
                            const real a = matrix[j * n + mode];
                            product += a * coefficients[component * n + j];
                        }
                        value += plan.relax * product;
                        const auto first = plan.rotated && plan.rotationActive(entry, 0) ? blocks.First(component) : -1;
                        if (first >= 0)
                        {
                            real correction = 0;
                            for (index j = 0; j < n; ++j)
                            {
                                real rotated = 0;
                                for (index d = 0; d < 3; ++d)
                                    rotated += coefficients[(first + d) * n + j] *
                                               (plan.rotation(entry, d * 3 + component - first) - (component - first == d ? 1 : 0));
                                const real a = matrix[j * n + mode];
                                correction += a * rotated;
                            }
                            value += plan.relax * correction;
                        }
                        real otherMean = mean(other, component);
                        if (first >= 0)
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
                __syncthreads(); // No lane may overwrite a still-consumed neighbor.
            }
            if constexpr (Components)
                for (int slot = 0; slot < slots; ++slot)
                {
                    const index i = threadIdx.x + slot * blockDim.x;
                    if (i < output.width)
                        output(iCell, i) = values[slot];
                }
        }
    }
    __global__ void ValidateIndicator(R::ArrayView<const real> indicator, R::ArrayView<int> status)
    {
        for (index i = index(blockIdx.x) * blockDim.x + threadIdx.x; i < indicator.rows; i += index(gridDim.x) * blockDim.x)
            if (!isfinite(indicator(i, 0)))
                atomicExch(&status(0, 0), 1);
    }
    __global__ void ValidatePenalty(R::ArrayView<const real> alpha, R::ArrayView<int> status)
    {
        for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < alpha.rows; iCell += index(gridDim.x) * blockDim.x)
            if (!isfinite(alpha(iCell, 0)) || alpha(iCell, 0) < 0 || alpha(iCell, 0) > 1)
                atomicExch(&status(0, 0), 1);
    }
    // Stream-ordered after the update: never read and write status in the same
    // grid. Check only selected results, allowing unused FP64 inputs/rows to
    // exceed the narrower compute range. Commit remains guarded by this status.
    __global__ void ValidateJacobiResult(DeviceWorkSelection selection, R::ArrayView<const real> output, R::ArrayView<int> status)
    {
        for (index iWork = blockIdx.x; iWork < selection.count; iWork += gridDim.x)
            for (index i = threadIdx.x; i < output.width; i += blockDim.x)
                if (!isfinite(output(selection.Cell(iWork), i)))
                    atomicExch(&status(0, 0), 1);
    }
    __device__ real Biway(real center, real other, real epsilon)
    {
        const real ratio = center / (fabs(other) + epsilon) * ((other > 0) - (other < 0));
        const real r2 = ratio * ratio;
        return other == 0 ? 0 : center * (1 + r2 * ratio) / (1 + r2 * r2);
    }
    /// Limiter-only projection keeps unrelated raw/inverse/sweep tables out
    /// of the aggregate CUDA argument block.
    struct WBAPView
    {
        R::ArrayView<const index> offsets, neighbors, faces;
        R::ArrayView<const real> secondary, rotation;
        R::ArrayView<const uint8_t> rotationActive;
        index n;
        real wbapCenterWeight;
        bool rotated;
        explicit WBAPView(const ReconstructionDeviceView &p)
            : offsets(p.offsets), neighbors(p.neighbors), faces(p.faces), secondary(p.secondary), rotation(p.rotation),
              rotationActive(p.rotationActive), n(p.n), wbapCenterWeight(p.wbapCenterWeight), rotated(p.rotated) {}
    };
    __global__ void WBAP(WBAPView plan, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                         R::PairReadView<real> input, R::ArrayView<real> output, R::ArrayView<real> scratch,
                         DeviceCharacteristicView transform, R::ArrayView<const real> indicator, R::ArrayView<int> status,
                         index begin, index modes, real threshold, real epsilon)
    {
        const index components = output.width / plan.n, count = modes * components;
        const auto iBlock = index(blockIdx.x);
        for (index iWork = iBlock; iWork < selection.count; iWork += gridDim.x)
        {
            if (status(0, 0))
                return;
            const auto iCell = selection.Cell(iWork);
            for (index i = threadIdx.x; i < output.width; i += blockDim.x)
                output(iCell, i) = input(iCell, i);
            __syncthreads();
            if (indicator.data && indicator(iCell, 0) < threshold)
                continue;
            // Six bounded scratch rows per active block, not per mesh cell.
            const index row = iBlock * 6;
            for (index i = threadIdx.x; i < count; i += blockDim.x)
            {
                scratch(row + 3, i) = plan.wbapCenterWeight;
                scratch(row + 4, i) = plan.wbapCenterWeight;
                scratch(row + 5, i) = 1;
            }
            __syncthreads();
            for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
            {
                const auto neighbor = plan.neighbors(entry, 0);
                if (neighbor == UnInitIndex)
                    continue;
                for (index i = threadIdx.x; i < count; i += blockDim.x)
                {
                    const index mode = i % modes, component = i / modes;
                    scratch(row, i) = input(iCell, component * plan.n + begin + mode);
                    real value = 0;
                    for (index j = 0; j < plan.n; ++j)
                        value += plan.secondary(entry, j * plan.n + begin + mode) * input(neighbor, component * plan.n + j);
                    scratch(row + 1, i) = value;
                }
                __syncthreads();
                if (plan.rotated && plan.rotationActive(entry, 0))
                {
                    for (index i = threadIdx.x; i < count; i += blockDim.x)
                    {
                        const index mode = i % modes, component = i / modes, first = vectors.First(component);
                        real value = scratch(row + 1, i);
                        if (first >= 0)
                        {
                            value = 0;
                            for (index d = 0; d < 3; ++d)
                                value += scratch(row + 1, (first + d) * modes + mode) * plan.rotation(entry, d * 3 + component - first);
                        }
                        scratch(row + 2, i) = value;
                    }
                    __syncthreads();
                    for (index i = threadIdx.x; i < count; i += blockDim.x)
                        scratch(row + 1, i) = scratch(row + 2, i);
                    __syncthreads();
                }
                const auto face = plan.faces(entry, 0);
                if (transform.count)
                {
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                    {
                        const index mode = i % modes, component = i / modes;
                        real value = 0;
                        for (index j = 0; j < transform.count; ++j)
                            value += scratch(row, (transform.first + j) * modes + mode) * transform.forward(face, j * transform.count + component);
                        scratch(row + 2, i) = value;
                    }
                    __syncthreads();
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                        scratch(row, transform.first * modes + i) = scratch(row + 2, i);
                    __syncthreads();
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                    {
                        const index mode = i % modes, component = i / modes;
                        real value = 0;
                        for (index j = 0; j < transform.count; ++j)
                            value += scratch(row + 1, (transform.first + j) * modes + mode) * transform.forward(face, j * transform.count + component);
                        scratch(row + 2, i) = value;
                    }
                    __syncthreads();
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                        scratch(row + 1, transform.first * modes + i) = scratch(row + 2, i);
                    __syncthreads();
                }
                for (index i = threadIdx.x; i < count; i += blockDim.x)
                    scratch(row + 2, i) = Biway(scratch(row, i), scratch(row + 1, i), epsilon);
                __syncthreads();
                if (transform.count)
                {
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                    {
                        const index mode = i % modes, component = i / modes;
                        real value = 0;
                        for (index j = 0; j < transform.count; ++j)
                            value += scratch(row + 2, (transform.first + j) * modes + mode) * transform.inverse(face, j * transform.count + component);
                        scratch(row + 1, i) = value;
                    }
                    __syncthreads();
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                        scratch(row + 2, transform.first * modes + i) = scratch(row + 1, i);
                    __syncthreads();
                }
                for (index i = threadIdx.x; i < count; i += blockDim.x)
                {
                    const index mode = i % modes, component = i / modes;
                    const real value = scratch(row + 2, i);
                    const real ratio = input(iCell, component * plan.n + begin + mode) / (fabs(value) + epsilon) * ((value > 0) - (value < 0));
                    scratch(row + 3, i) += ratio * ratio * ratio;
                    scratch(row + 4, i) += ratio * ratio * ratio * ratio;
                    if (value == 0)
                        scratch(row + 5, i) = 0;
                }
                __syncthreads();
            }
            for (index i = threadIdx.x; i < count; i += blockDim.x)
            {
                const index mode = i % modes, component = i / modes;
                output(iCell, component * plan.n + begin + mode) *= scratch(row + 5, i) * scratch(row + 3, i) / scratch(row + 4, i);
            }
            __syncthreads();
        }
    }
    // Publication copies logical owned entries, retaining allocation identity and
    // halo bindings. Future layout-specific coalescing/memcpyAsync comparisons
    // must preserve this device-status guard without a host status observation.
    // Physical memcpy is only equivalent for matching complete layouts/bytes;
    // it cannot blindly replace this conditional copy (plan 56, E3).
    __global__ void Commit(R::ArrayView<const real> input, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
        if (status(0, 0))
            return;
        for (index i = index(blockIdx.x) * blockDim.x + threadIdx.x; i < output.rows * output.width; i += index(gridDim.x) * blockDim.x)
            output(i / output.width, i % output.width) = input(i / output.width, i % output.width);
    }
    void LaunchJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks blocks,
                      R::PairReadView<real> mean, R::PairReadView<real> input, R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
        if (plan.raw)
        {
            LaunchRawJacobi(context, plan, selection, blocks, mean, input, boundary, output, status, {}, {}, false, false);
            return;
        }
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA Jacobi requires a CUDA context");
        R::ValidateViewDevices(context.device, plan, selection, blocks, mean, input, boundary, output, status);
        R::CheckKernelArgumentBudget<JacobiView<real>, decltype(selection), decltype(blocks), decltype(mean),
                                     decltype(input), decltype(boundary), decltype(output), decltype(status),
                                     R::ArrayView<const real>, R::ArrayView<const real>>();
        if (!selection.count)
            return;
        const GeneralSweepArguments args{context, plan, selection, blocks, mean, input, boundary, output, status, {}, {}};
        if (plan.quantized)
            RunGeneralFP32(args, false);
        else
            RunGeneralFP64(args);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA Jacobi launch failed");
    }
    void LaunchValidateJacobiResult(R::ExecutionContext context, DeviceWorkSelection selection,
                                    R::ArrayView<const real> output, R::ArrayView<int> status)
    {
        R::ValidateViewDevices(context.device, selection, output, status);
        if (selection.count)
            ValidateJacobiResult<<<unsigned(std::min<index>(selection.count, 256)), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(selection, output, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA Jacobi finite-result check failed");
    }
    void LaunchJacobiFullAccumulation(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                                      DevicePeriodicBlocks blocks, R::PairReadView<real> mean, R::PairReadView<real> input,
                                      R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<int> status, bool floatFloat)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && plan.quantized,
                              "full accumulation Jacobi requires a CUDA context and FP32 sweep maps");
#ifndef DNDS_USE_CCCL_COMPUTE
        DNDS_check_throw_info(!floatFloat, "float-float Jacobi requires DNDS_USE_CCCL_COMPUTE");
#endif
        R::ValidateViewDevices(context.device, plan, selection, blocks, mean, input, boundary, output, status);
        R::CheckKernelArgumentBudget<JacobiView<float>, decltype(selection), decltype(blocks), decltype(mean),
                                     decltype(input), decltype(boundary), decltype(output), decltype(status),
                                     R::ArrayView<const real>, R::ArrayView<const real>>();
        if (!selection.count)
            return;
        const GeneralSweepArguments args{context, plan, selection, blocks, mean, input, boundary, output, status.ConstView(), {}, {}};
        if (floatFloat)
            RunGeneralFloatFloat(args);
        else
            RunGeneralFP32(args, true);
        LaunchValidateJacobiResult(context, selection, output.ConstView(), status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "full accumulation CUDA Jacobi launch failed");
    }
    void LaunchJacobiMatrixShared(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                                  DevicePeriodicBlocks blocks, R::PairReadView<real> mean, R::PairReadView<real> input,
                                  R::ArrayView<const real> boundary, R::ArrayView<real> output,
                                  R::ArrayView<const int> status)
    {
        if (plan.raw)
        {
            LaunchRawJacobi(context, plan, selection, blocks, mean, input, boundary, output, status, {}, {}, false, true);
            return;
        }
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && !plan.quantized,
                              "optimized FP64 Jacobi requires a CUDA context, FP64 maps and a valid mapping");
        R::ValidateViewDevices(context.device, plan, selection, blocks, mean, input, boundary, output, status);
        DNDS_check_throw_info(output.Width() == mean.owned.Width() * plan.n &&
                                  input.owned.Width() == output.Width() && output.RowCount() == mean.owned.RowCount(),
                              "optimized FP64 Jacobi field shape differs");
        DNDS_check_throw_info(plan.n == 2 || plan.n == 3 || plan.n == 5 || plan.n == 9 || plan.n == 19,
                              "static FP64 Jacobi requires P1..P3 modal dimensions");
        DNDS_check_throw_info(mean.owned.Width() >= 1 && mean.owned.Width() <= INT_MAX / plan.n,
                              "invalid matrix-shared FP64 Jacobi component count");
        R::CheckKernelArgumentBudget<JacobiView<real>, decltype(selection), decltype(blocks), decltype(mean),
                                     decltype(input), decltype(boundary), decltype(output), decltype(status),
                                     R::ArrayView<const real>, R::ArrayView<const real>>();
        if (!selection.count)
            return;
        const auto stream = static_cast<cudaStream_t>(context.stream);
        // More blocks make sufficient independent cells available to every SM.
        // The reference's 256-block cap is retained by LaunchJacobi for a clean
        // comparison; changing this limit never changes arithmetic order.
        const unsigned grid = unsigned(std::min<index>(selection.count, 65535));
        {
            const auto launch = [&](auto modes)
            {
                constexpr int N = decltype(modes)::value;
                const auto launchLayout = [&](auto layout)
                {
                    using Layout = decltype(layout);
                    const auto components = [&](auto c)
                    {
                        JacobiSharedSIMT<N, decltype(c)::value, Layout><<<grid, 96, decltype(c)::value ? 0 : size_t(output.width + N * N) * sizeof(real), stream>>>(JacobiView<real, Layout>(plan), selection, blocks, mean, input, boundary, output, status);
                    };
                    switch (mean.owned.width)
                    {
                    case 5:
                        components(std::integral_constant<int, 5>{});
                        break;
                    case 6:
                        components(std::integral_constant<int, 6>{});
                        break;
                    case 7:
                        components(std::integral_constant<int, 7>{});
                        break;
                    default:
                        components(std::integral_constant<int, 0>{});
                        break;
                    }
                };
                // Projection is checked once by the wrapper, never per element.
                if (plan.matrixAInvB.kind == R::RowLayoutKind::Table && plan.vectorAInvB.kind == R::RowLayoutKind::Table &&
                    plan.boundaryInjection.kind == R::RowLayoutKind::Table)
                    launchLayout(R::TableRows{});
                else
                    launchLayout(R::RowLayout{});
            };
            if (plan.n == 2)
                launch(std::integral_constant<int, 2>{});
            else if (plan.n == 3)
                launch(std::integral_constant<int, 3>{});
            else if (plan.n == 5)
                launch(std::integral_constant<int, 5>{});
            else if (plan.n == 9)
                launch(std::integral_constant<int, 9>{});
            else
                launch(std::integral_constant<int, 19>{});
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "optimized FP64 Jacobi launch failed");
    }
    void LaunchLimitedJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                             DevicePeriodicBlocks blocks, R::PairReadView<real> mean, R::PairReadView<real> input,
                             R::ArrayView<const real> boundary, R::ArrayView<const real> target, R::ArrayView<const real> alpha,
                             R::ArrayView<real> output, R::ArrayView<int> status)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA limited Jacobi requires a CUDA context");
        R::ValidateViewDevices(context.device, plan, selection, blocks, mean, input, boundary, target, alpha, output, status);
        DNDS_check_throw_info(target.rows == output.rows && target.width == output.width && alpha.rows == output.rows && alpha.width == 1,
                              "CUDA limited Jacobi penalty/target shape mismatch");
        R::CheckKernelArgumentBudget<JacobiView<real>, decltype(selection), decltype(blocks), decltype(mean), decltype(input),
                                     decltype(boundary), decltype(output), decltype(status), decltype(target), decltype(alpha)>();
        if (!selection.count)
            return;
        // A separate stream-ordered validation pass avoids a status read/write
        // race between cell blocks and prevents partial publication on error.
        ValidatePenalty<<<unsigned(std::min<index>(256, (alpha.rows + 127) / 128)), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(alpha, status);
        if (plan.raw)
            LaunchRawJacobi(context, plan, selection, blocks, mean, input, boundary, output, status.ConstView(), target, alpha, true, false);
        else
            RunGeneralLimited({context, plan, selection, blocks, mean, input, boundary, output, status.ConstView(), target, alpha});
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA limited Jacobi launch failed");
    }
    void LaunchWBAP(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks blocks,
                    R::PairReadView<real> input, R::ArrayView<real> output, R::ArrayView<real> scratch, DeviceCharacteristicView transform,
                    R::ArrayView<const real> indicator, R::ArrayView<int> status, index begin, index modes, real threshold, index blocksCount)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA WBAP requires a CUDA context");
        R::ValidateViewDevices(context.device, plan, selection, blocks, input, output, scratch, transform, indicator, status);
        R::CheckKernelArgumentBudget<WBAPView, decltype(selection), decltype(blocks), decltype(input),
                                     decltype(output), decltype(scratch), decltype(transform), decltype(indicator),
                                     decltype(status), index, index, real, real>();
        const auto stream = static_cast<cudaStream_t>(context.stream);
        // Status persists through a queued segment; caller explicitly clears it
        // when recovering after an observed error. No host read per submission.
        if (indicator.rows)
            ValidateIndicator<<<unsigned(std::min<index>(256, (indicator.rows + 127) / 128)), 128, 0, stream>>>(indicator, status);
        if (selection.count)
            WBAP<<<unsigned(blocksCount), 128, 0, stream>>>(WBAPView(plan), selection, blocks, input, output, scratch, transform, indicator, status, begin, modes, threshold, std::pow(verySmallReal, .25));
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA WBAP launch failed");
    }
    void LaunchCommit(R::ExecutionContext context, R::ArrayView<const real> input, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA commit requires a CUDA context");
        R::ValidateViewDevices(context.device, input, output, status);
        R::CheckKernelArgumentBudget<decltype(input), decltype(output), decltype(status)>();
        // Investigate mapping and launch overhead by shape before selecting a
        // copy backend. No universal kernel-versus-memcpy speedup is assumed.
        if (!output.rows || !output.width)
            return;
        Commit<<<unsigned(std::min<index>(256, (output.rows * output.width + 255) / 256)), 256, 0, static_cast<cudaStream_t>(context.stream)>>>(input, output, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA reconstruction commit launch failed");
    }
}
