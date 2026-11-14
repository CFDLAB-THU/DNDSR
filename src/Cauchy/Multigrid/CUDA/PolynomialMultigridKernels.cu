#include "Cauchy/Multigrid/CUDA/PolynomialMultigridKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        constexpr int Threads = 128;
        /// Keep the mathematical reduction independent of deprecated CUB
        /// functor aliases; supported with toolkit and pinned source headers.
        struct MinimumIndex
        {
            DNDS_DEVICE_CALLABLE index operator()(index left, index right) const noexcept
            {
                return right < left ? right : left;
            }
        };
        template <bool Final>
        __global__ void Frozen(FASRead values, FASRead saved, FASSelection selection,
                               R::ArrayView<const index, R::AccessBoundsChecks, R::FixedRows<1>> input, FASIndexWrite output)
        {
            index first = INT64_MAX;
            const index count = Final ? input.RowCount() : values.RowCount();
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < count; iRow += index(gridDim.x) * blockDim.x)
                if constexpr (Final)
                    first = std::min(first, input(iRow, 0));
                else
                    for (index j = 0; j < values.Width(); ++j)
                        if (selection.active[j] == 0 && __double_as_longlong(values(iRow, j)) != __double_as_longlong(saved(iRow, j)))
                            first = std::min(first, iRow);
            using Reduce = cub::BlockReduce<index, Threads>;
            __shared__ typename Reduce::TempStorage storage;
            const index result = Reduce(storage).Reduce(first, MinimumIndex{});
            if (!threadIdx.x)
                output(blockIdx.x, 0) = Final && result == INT64_MAX ? -1 : result;
        }
        __global__ void Selected(FASWrite values, FASRead delta, FASSelection selection, bool add)
        {
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < values.RowCount(); iRow += index(gridDim.x) * blockDim.x)
                for (index j = 0; j < values.Width(); ++j)
                    if (add && selection.active[j] == 1)
                        values(iRow, j) += delta(iRow, j);
                    else if (!add && selection.active[j] == 0)
                        values(iRow, j) = 0;
        }
        template <class View>
        void Check(R::ExecutionContext context, View values)
        {
            values.ValidateDevice(context.device);
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && values.kind == R::RowLayoutKind::Table &&
                                      values.Width() >= 5 && values.Width() <= 7,
                                  "CUDA FAS requires combined AoS flow5/SA/2EQ");
        }
    }
    index FrozenCheckScratchRows(index rows)
    {
        DNDS_check_throw_info(rows >= 0, "negative FAS row count");
        return std::max<index>(1, std::min<index>(256, rows / Threads + (rows % Threads != 0)));
    }
    void CheckFrozenRows(R::ExecutionContext context, FASRead values, FASRead saved, FASSelection selection, FASIndexWrite partial, FASIndexWrite result)
    {
        Check(context, values);
        Check(context, saved);
        R::ValidateViewDevices(context.device, partial, result);
        DNDS_check_throw_info(values.RowCount() == saved.RowCount() && values.Width() == saved.Width() &&
                                  partial.RowCount() == FrozenCheckScratchRows(values.RowCount()) && result.RowCount() == 1 && partial.data != result.data,
                              "CUDA FAS frozen-check shape differs");
        R::CheckKernelArgumentBudget<FASRead, FASRead, FASSelection, decltype(partial.ConstView()), FASIndexWrite>();
        const auto stream = static_cast<cudaStream_t>(context.stream);
        Frozen<false><<<unsigned(partial.RowCount()), Threads, 0, stream>>>(values, saved, selection, {}, partial);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA FAS frozen-check launch failed");
        Frozen<true><<<1, Threads, 0, stream>>>({}, {}, {}, partial.ConstView(), result);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA FAS frozen-check reduction failed");
    }
    void UpdateSelectedRows(R::ExecutionContext context, FASWrite values, FASRead delta, FASSelection selection, bool add)
    {
        Check(context, values);
        if (add)
        {
            Check(context, delta);
            DNDS_check_throw_info(values.RowCount() == delta.RowCount() && values.Width() == delta.Width() &&
                                      (!values.data || values.data != delta.data),
                                  "CUDA FAS selected-add shape or alias differs");
        }
        R::CheckKernelArgumentBudget<FASWrite, FASRead, FASSelection, bool>();
        if (!values.RowCount())
            return;
        Selected<<<unsigned(FrozenCheckScratchRows(values.RowCount())), Threads, 0, static_cast<cudaStream_t>(context.stream)>>>(values, delta, selection, add);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA FAS selected update failed");
    }
}
