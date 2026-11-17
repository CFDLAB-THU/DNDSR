#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <limits>
#include <climits>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        constexpr int Threads = 128;
        using Cells = R::ArrayView<const real, false, R::FixedRows<8>>;
        using ActiveRows = R::ArrayView<const uint8_t, R::AccessBoundsChecks, R::FixedRows<1>>;
        __device__ StateReductionValue Merge(StateReductionValue a, StateReductionValue b)
        {
            return {fmin(a.value, b.value), a.firstInvalid < 0 ? b.firstInvalid : b.firstInvalid < 0 ? a.firstInvalid
                                                                                                     : (a.firstInvalid < b.firstInvalid ? a.firstInvalid : b.firstInvalid)};
        }
        struct MergeValues
        {
            __device__ StateReductionValue operator()(StateReductionValue a, StateReductionValue b) const { return Merge(a, b); }
        };
        struct MergeStatus
        {
            __device__ NumericalStatus operator()(NumericalStatus a, NumericalStatus b) const
            {
                if (a)
                    return b ? NumericalStatus{} : b;
                if (b)
                    return a;
                if (a.row != b.row)
                    return a.row < b.row ? a : b;
                if (a.point != b.point)
                    return a.point < b.point ? a : b;
                if (a.code != b.code)
                    return int(a.code) < int(b.code) ? a : b;
                return a.detail < b.detail ? a : b;
            }
        };
        __global__ void ClearStatusRows(StatusOutput output)
        {
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < output.RowCount(); iRow += index(gridDim.x) * blockDim.x)
                output(iRow, 0) = {};
        }
        __global__ void StatusRows(StatusInput first, StatusInput second, StatusOutput output)
        {
            NumericalStatus value;
            const index count = first.RowCount() + second.RowCount();
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < count; iRow += index(gridDim.x) * blockDim.x)
                value = MergeStatus{}(value, iRow < first.RowCount() ? first(iRow, 0) : second(iRow - first.RowCount(), 0));
            using Reduce = cub::BlockReduce<NumericalStatus, Threads, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
            __shared__ typename Reduce::TempStorage temporary;
            const auto result = Reduce(temporary).Reduce(value, MergeStatus{});
            if (!threadIdx.x)
                output(blockIdx.x, 0) = result;
        }
        __device__ void AtomicBlockMinimum(StateReductionValue value, StateReductionOutput output)
        {
            using Reduce = cub::BlockReduce<StateReductionValue, Threads, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
            __shared__ typename Reduce::TempStorage temporary;
            const auto reduced = Reduce(temporary).Reduce(value, MergeValues{});
            if (!threadIdx.x)
            {
                static_assert(sizeof(real) == sizeof(unsigned long long) && sizeof(index) == sizeof(unsigned long long));
                atomicMin(reinterpret_cast<unsigned long long *>(&output(0, 0).value),
                          static_cast<unsigned long long>(__double_as_longlong(reduced.value)));
                if (reduced.firstInvalid >= 0)
                    atomicMin(reinterpret_cast<unsigned long long *>(&output(0, 0).firstInvalid),
                              static_cast<unsigned long long>(reduced.firstInvalid));
            }
        }
        __global__ void AdmissibilityRows(StateInput means, AdmissibilityView bounds, StateReductionOutput scratch, ActiveRows activeRows)
        {
            StateReductionValue value{1, -1};
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < means.RowCount(); iCell += index(gridDim.x) * blockDim.x)
                if ((!activeRows.RowCount() || activeRows(iCell, 0)) && !bounds.Admissible(means[iCell]))
                    value = Merge(value, {0, iCell});
            AtomicBlockMinimum(value, scratch);
        }
        __global__ void StableStepRows(StateInput sums, Cells cells, real cfl, StateReductionOutput result)
        {
            StateReductionValue value{std::numeric_limits<real>::max(), -1};
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < sums.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const real radius = sums(iCell, 0);
                if (!(isfinite(radius) && radius > 0))
                    value = Merge(value, {value.value, iCell});
                else
                    value = Merge(value, {cfl * cells(iCell, 0) / radius, -1});
            }
            using Reduce = cub::BlockReduce<StateReductionValue, Threads, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
            __shared__ typename Reduce::TempStorage temporary;
            const auto reduced = Reduce(temporary).Reduce(value, MergeValues{});
            if (!threadIdx.x)
            {
                // Positive IEEE FP64 values have the same order as unsigned bits.
                // This is min, not a floating sum: block arrival order cannot
                // change the numerical answer. -1 is the largest unsigned row ID.
                static_assert(sizeof(real) == sizeof(unsigned long long) && sizeof(index) == sizeof(unsigned long long));
                atomicMin(reinterpret_cast<unsigned long long *>(&result(0, 0).value),
                          static_cast<unsigned long long>(__double_as_longlong(reduced.value)));
                if (reduced.firstInvalid >= 0)
                    atomicMin(reinterpret_cast<unsigned long long *>(&result(0, 0).firstInvalid),
                              static_cast<unsigned long long>(reduced.firstInvalid));
            }
        }
    }
    index StateReductionScratchRows(index ownedRows)
    {
        DNDS_check_throw_info(ownedRows >= 0, "negative CUDA state reduction row count");
        return std::max<index>(1, std::min<index>(256, ownedRows / Threads + (ownedRows % Threads != 0)));
    }
    void ClearNumericalStatus(R::ExecutionContext context, StatusOutput output)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA status clear needs a CUDA context");
        output.ValidateDevice(context.device);
        R::CheckKernelArgumentBudget<StatusOutput>();
        if (!output.RowCount())
            return;
        ClearStatusRows<<<unsigned(StateReductionScratchRows(output.RowCount())), Threads, 0, ::DNDS::Runtime::CUDAStream(context)>>>(output);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA status clear launch failed");
    }
    void ReduceNumericalStatus(R::ExecutionContext context, StatusInput first, StatusInput second,
                               StatusOutput scratch, StatusOutput result)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA status reduction needs a CUDA context");
        R::ValidateViewDevices(context.device, first, second, scratch, result);
        DNDS_check_throw_info(first.RowCount() <= std::numeric_limits<index>::max() - second.RowCount() &&
                                  scratch.RowCount() >= StateReductionScratchRows(first.RowCount() + second.RowCount()) && scratch.RowCount() <= 256 &&
                                  result.RowCount() == 1 && scratch.data != result.data &&
                                  scratch.data != first.data && scratch.data != second.data,
                              "CUDA status reduction workspace shape or alias differs");
        R::CheckKernelArgumentBudget<StatusInput, StatusInput, StatusOutput>();
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        StatusRows<<<unsigned(scratch.RowCount()), Threads, 0, stream>>>(first, second, scratch);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA status partial reduction launch failed");
        StatusRows<<<1, Threads, 0, stream>>>(scratch.ConstView(), {}, result);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA status final reduction launch failed");
    }
    void AdmissibleMeans(R::ExecutionContext context, StateInput means, AdmissibilityView bounds,
                         StateReductionOutput result, ActiveRows activeRows)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA admissibility requires a CUDA context");
        R::ValidateViewDevices(context.device, means, result);
        DNDS_check_throw_info(means.kind == R::RowLayoutKind::Table && result.RowCount() == 1 &&
                                  static_cast<const void *>(means.data) != result.data &&
                                  static_cast<const void *>(activeRows.data) != result.data,
                              "CUDA admissibility result shape/alias differs");
        if (activeRows.RowCount())
            activeRows.ValidateDevice(context.device);
        DNDS_check_throw_info(!activeRows.RowCount() || activeRows.RowCount() == means.RowCount(), "CUDA admissibility row mask differs");
        DNDS_check_throw_info(means.Width() >= 5 && means.Width() <= 7 && bounds.thermal.components == means.Width() &&
                                  bounds.thermal.flowFirst == 0 && bounds.thermal.independent == 0 &&
                                  bounds.thermal.speciesEnergies[0].empty() && bounds.thermal.speciesEnergies[1].empty() &&
                                  bounds.rans.components == means.Width() - 5 && (!bounds.rans.components || bounds.ransFirst == 5),
                              "CUDA admissibility supports combined perfect-gas flow5/SA/2EQ only");
        R::CheckKernelArgumentBudget<StateInput, AdmissibilityView, StateReductionOutput, ActiveRows>();
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        // The temporary all-ones value bits are an unsigned atomic identity,
        // never read as a floating value. Every block publishes exact 0 or 1;
        // one empty block also publishes 1. Invalid-row -1 is the same unsigned
        // identity. Thus no FP sum, partial table or tail launch is necessary.
        DNDS_check_throw_info(cudaMemsetAsync(result.data, 0xff, sizeof(StateReductionValue), stream) == cudaSuccess,
                              "CUDA admissibility identity reset failed");
        const auto blocks = unsigned(std::max<index>(1, std::min<index>(INT_MAX, means.RowCount() / Threads + (means.RowCount() % Threads != 0))));
        AdmissibilityRows<<<blocks, Threads, 0, stream>>>(means, bounds, result, activeRows);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA admissibility row launch failed");
    }
    void MinimumStableStep(R::ExecutionContext context, StateInput sums, Cells cells, real cfl,
                           StateReductionInput identity, StateReductionOutput result)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA CFL reduction needs a CUDA context");
        R::ValidateViewDevices(context.device, sums, cells, identity, result);
        DNDS_check_throw_info(sums.kind == R::RowLayoutKind::Table && sums.Width() == 1 &&
                                  cells.RowCount() >= sums.RowCount() && std::isfinite(cfl) && cfl > 0 &&
                                  identity.RowCount() == 1 && result.RowCount() == 1 && identity.data != result.data &&
                                  static_cast<const void *>(result.data) != sums.data && static_cast<const void *>(result.data) != cells.data,
                              "CUDA stable-step shape/control differs");
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        DNDS_check_throw_info(cudaMemcpyAsync(result.data, identity.data, sizeof(StateReductionValue), cudaMemcpyDeviceToDevice, stream) == cudaSuccess,
                              "CUDA stable-step reset failed");
        if (!sums.RowCount())
            return;
        R::CheckKernelArgumentBudget<StateInput, Cells, real, StateReductionOutput>();
        // Grid-stride iteration handles even inputs beyond the hardware grid limit;
        // there is no partial-buffer capacity constraining the launch.
        const auto blocks = unsigned(std::min<index>(INT_MAX, sums.RowCount() / Threads + (sums.RowCount() % Threads != 0)));
        StableStepRows<<<blocks, Threads, 0, stream>>>(sums, cells, cfl, result);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA stable-step row launch failed");
    }
}
