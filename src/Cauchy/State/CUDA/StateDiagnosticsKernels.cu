#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/State/CUDA/StateDiagnosticsKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "DNDS/Device/ReductionCompletion.hpp"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        constexpr int Threads = 128;
        using Cells = R::ArrayView<const real, false, R::FixedRows<8>>;
        struct Merge
        {
            __device__ StateDiagnosticSummary operator()(StateDiagnosticSummary a, StateDiagnosticSummary b) const
            {
                a.mass += b.mass;
                a.energy += b.energy;
                a.squaredResidual += b.squaredResidual;
                a.squaredContinuity += b.squaredContinuity;
                a.integratedPressure += b.integratedPressure;
                a.densityMinimum = fmin(a.densityMinimum, b.densityMinimum);
                a.pressureMinimum = fmin(a.pressureMinimum, b.pressureMinimum);
                a.temperatureMinimum = fmin(a.temperatureMinimum, b.temperatureMinimum);
                if (!b.status && (a.status || b.status.row < a.status.row))
                    a.status = b.status;
                return a;
            }
        };
        struct ConstantDensity
        {
        };
        template <class Physics>
        __global__ void Rows(DiagnosticInput mean, DiagnosticInput residual, Cells cells, Physics gas,
                             DiagnosticOutput partial, DiagnosticOutput result, DiagnosticCounter completed)
        {
            StateDiagnosticSummary value;
            for (index iCell = index(blockIdx.x) * Threads + threadIdx.x; iCell < mean.RowCount(); iCell += index(gridDim.x) * Threads)
            {
                const real volume = cells(iCell, 0);
                if constexpr (std::is_same_v<Physics, ConstantDensity>)
                {
                    value.mass += volume;
                    value.densityMinimum = 1;
                    value.pressureMinimum = fmin(value.pressureMinimum, mean(iCell, 3));
                    value.squaredContinuity += residual(iCell, 3) * residual(iCell, 3) / volume;
                    value.integratedPressure += volume * mean(iCell, 3);
                    for (int d = 0; d < 3; ++d)
                        value.energy += .5 * volume * mean(iCell, d) * mean(iCell, d);
                }
                else
                {
                    FlowState flow;
                    for (int j = 0; j < 5; ++j)
                        flow[j] = mean(iCell, j);
                    const auto point = TryPerfectGasPoint(gas, flow, false);
                    if (!point)
                    {
                        if (value.status)
                            value.status = point.status.At(iCell);
                        continue;
                    }
                    value.mass += volume * flow[0];
                    value.energy += volume * flow[4];
                    value.densityMinimum = fmin(value.densityMinimum, flow[0]);
                    value.pressureMinimum = fmin(value.pressureMinimum, point.value.thermo.pressure);
                    value.temperatureMinimum = fmin(value.temperatureMinimum, point.value.thermo.temperature);
                }
                for (index j = 0; j < residual.Width(); ++j)
                    value.squaredResidual += residual(iCell, j) * residual(iCell, j) / volume;
            }
            using Reduce = cub::BlockReduce<StateDiagnosticSummary, Threads, cub::BLOCK_REDUCE_RAKING>;
            __shared__ typename Reduce::TempStorage scratch;
            __shared__ bool last;
            const auto block = Reduce(scratch).Reduce(value, Merge{});
            if (!threadIdx.x)
            {
                partial(blockIdx.x, 0) = block;
                last = Device::ArriveReductionBlock(completed(0, 0), gridDim.x);
            }
            __syncthreads();
            if (!last)
                return;
            StateDiagnosticSummary tail;
            for (index iRow = threadIdx.x; iRow < partial.RowCount(); iRow += Threads)
                tail = Merge{}(tail, partial(iRow, 0));
            const auto total = Reduce(scratch).Reduce(tail, Merge{});
            if (!threadIdx.x)
                result(0, 0) = total;
        }
    }
    index StateDiagnosticScratchRows(index rows)
    {
        DNDS_check_throw_info(rows >= 0, "negative diagnostic row count");
        return std::max(index(1), std::min(index(256), rows / Threads + (rows % Threads != 0)));
    }
    void AccumulateStateDiagnostics(R::ExecutionContext context, DiagnosticInput mean, DiagnosticInput residual,
                                    Cells cells, std::optional<PerfectGas> gas, DiagnosticOutput partial, DiagnosticOutput result, DiagnosticCounter completed)
    {
        R::ValidateViewDevices(context.device, mean, residual, cells, partial, result, completed);
        const index flowWidth = gas ? 5 : 4;
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && mean.Width() >= flowWidth && mean.Width() <= flowWidth + 2 &&
                                  mean.RowCount() == residual.RowCount() && mean.Width() == residual.Width() && cells.RowCount() >= mean.RowCount() &&
                                  partial.RowCount() == StateDiagnosticScratchRows(mean.RowCount()) && result.RowCount() == 1 &&
                                  completed.RowCount() == 1 && partial.data != result.data,
                              "invalid CUDA diagnostic views");
        if (gas)
            gas->Validate();
        const std::array<const void *, 3> outputs{partial.data, result.data, completed.data};
        for (index i = 0; i < index(outputs.size()); ++i)
        {
            for (index j = 0; j < i; ++j)
                DNDS_check_throw_info(outputs[i] != outputs[j], "CUDA diagnostic outputs alias");
            for (const auto input : {mean.data, residual.data, cells.data})
                DNDS_check_throw_info(!input || outputs[i] != input, "CUDA diagnostic output aliases input");
        }
        R::CheckKernelArgumentBudget<DiagnosticInput, DiagnosticInput, Cells, PerfectGas, DiagnosticOutput, DiagnosticOutput, DiagnosticCounter>();
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        DNDS_check_throw_info(cudaMemsetAsync(completed.data, 0, sizeof(unsigned int), stream) == cudaSuccess, "CUDA diagnostic counter reset failed");
        if (gas)
            Rows<<<unsigned(partial.RowCount()), Threads, 0, stream>>>(mean, residual, cells, *gas, partial, result, completed);
        else
            Rows<<<unsigned(partial.RowCount()), Threads, 0, stream>>>(mean, residual, cells, ConstantDensity{}, partial, result, completed);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA diagnostic reduction launch failed");
    }
}
