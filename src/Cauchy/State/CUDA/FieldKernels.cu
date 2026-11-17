#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/State/CUDA/FieldKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cmath>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        constexpr int Threads = 128;
        struct Merge
        {
            __device__ NumericalStatus operator()(NumericalStatus a, NumericalStatus b) const
            {
                return !b && (a || b.row < a.row) ? b : a;
            }
        };
        template <bool Partial>
        __global__ void Check(EquationBlockArrayView<const real> values, bool capped, std::array<real, 2> caps,
                              FieldCheckOutput partial, FieldCheckOutput result)
        {
            NumericalStatus status;
            const index rows = Partial ? values.RowCount() : partial.RowCount();
            for (index iRow = index(blockIdx.x) * Threads + threadIdx.x; iRow < rows; iRow += index(gridDim.x) * Threads)
                if constexpr (Partial)
                {
                    for (index j = 0; j < values.ComponentNumber(); ++j)
                    {
                        const real value = values(iRow, j);
                        if (!std::isfinite(value) || (capped && !(value >= 0 && value <= caps[j])))
                        {
                            if (status)
                                status = {NumericalError::InvalidState, 1, iRow};
                            break;
                        }
                    }
                }
                else
                    status = Merge{}(status, partial(iRow, 0));
            using Reduce = cub::BlockReduce<NumericalStatus, Threads, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
            __shared__ typename Reduce::TempStorage scratch;
            const auto reduced = Reduce(scratch).Reduce(status, Merge{});
            if (!threadIdx.x)
                if constexpr (Partial)
                    partial(blockIdx.x, 0) = reduced;
                else
                    result(0, 0) = reduced;
        }
    }
    index FieldCheckScratchRows(index rows) { return std::max(index(1), std::min(index(256), (rows + Threads - 1) / Threads)); }
    void CheckFieldRows(R::ExecutionContext context, EquationBlockArrayView<const real> values,
                        bool capped, std::array<real, 2> caps, FieldCheckOutput partial, FieldCheckOutput result)
    {
        R::ValidateViewDevices(context.device, values, partial, result);
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && values.first >= 0 &&
                                  values.ComponentNumber() > 0 && values.first + values.ComponentNumber() <= values.parent.Width() &&
                                  (!capped || values.ComponentNumber() == 2) && partial.RowCount() == FieldCheckScratchRows(values.RowCount()) &&
                                  result.RowCount() == 1 && partial.data != result.data,
                              "invalid CUDA field-check shape or backend");
        if (capped)
            for (real cap : caps)
                DNDS_check_throw_info(std::isfinite(cap) && cap >= 0, "invalid field-check cap");
        R::CheckKernelArgumentBudget<EquationBlockArrayView<const real>, bool, std::array<real, 2>, FieldCheckOutput, FieldCheckOutput>();
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        Check<true><<<unsigned(partial.RowCount()), Threads, 0, stream>>>(values, capped, caps, partial, result);
        Check<false><<<1, Threads, 0, stream>>>(values, capped, caps, partial, result);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA field-check launch failed");
    }
}
