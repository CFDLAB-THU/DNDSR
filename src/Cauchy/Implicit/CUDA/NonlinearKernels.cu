#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Implicit/CUDA/NonlinearKernels_Detail.hpp"
#include <limits>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace D = NonlinearDetail;
    namespace
    {
        __global__ void ReduceSummary(NonlinearSummaryInput input, NonlinearSummaryOutput output)
        {
            NonlinearSummary value;
            for (index i = threadIdx.x; i < input.RowCount(); i += blockDim.x)
                value = D::Merge{}(value, input(i, 0));
            D::PublishBlock(value, output);
        }
        __global__ void Defect(NonlinearCells cells, LinearReadRows mean, LinearReadRows residual, LinearReadRows constant,
                               LinearWriteRows output, NonlinearSelection selection, real mass, real rhsScale, int order, bool measure, NonlinearSummaryOutput scratch)
        {
            NonlinearSummary local;
            const index n = mean.Width();
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < mean.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const real volume = cells(iCell, 0);
                if (measure)
                    local.norms[n] += volume;
                for (index j = 0; j < n; ++j)
                {
                    const real value = rhsScale * residual(iCell, j) + volume * (constant(iCell, j) - mass * mean(iCell, j));
                    if ((!std::isfinite(value) || !(volume > 0)) && local.status)
                        local.status = NumericalStatus{NumericalError::InvalidState, 1, iCell};
                    output(iCell, j) = selection.active[j] ? value : 0;
                    if (measure)
                        local.norms[j] += order == 1 ? std::abs(value) : value * value / volume;
                }
            }
            D::PublishBlock(local, scratch);
        }
        __global__ void MeanScale(NonlinearCells cells, LinearReadRows mean, NonlinearSummaryOutput scratch)
        {
            NonlinearSummary local;
            const index n = mean.Width();
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < mean.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const real volume = cells(iCell, 0);
                local.norms[n] += volume;
                for (index j = 0; j < n; ++j)
                {
                    const real value = volume * std::abs(mean(iCell, j));
                    if ((!std::isfinite(value) || !(volume > 0)) && local.status)
                        local.status = NumericalStatus{NumericalError::InvalidState, 1, iCell};
                    local.norms[j] += value;
                }
            }
            D::PublishBlock(local, scratch);
        }
        __global__ void Rate(NonlinearCells cells, LinearReadRows input, LinearWriteRows output, bool multiply)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < input.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const real factor = multiply ? cells(iCell, 0) : 1 / cells(iCell, 0);
                for (index j = 0; j < input.Width(); ++j)
                    output(iCell, j) = factor * input(iCell, j);
            }
        }
        __global__ void RateNorm(NonlinearCells cells, LinearReadRows rate, LinearReadRows reference, int order, NonlinearSummaryOutput scratch)
        {
            NonlinearSummary local;
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < rate.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const real volume = cells(iCell, 0);
                local.norms[rate.Width()] += volume;
                for (index j = 0; j < rate.Width(); ++j)
                {
                    const real value = rate(iCell, j) - (reference.RowCount() ? reference(iCell, j) : 0);
                    if ((!std::isfinite(value) || !(volume > 0)) && local.status)
                        local.status = {NumericalError::InvalidState, 1, iCell};
                    local.norms[j] += order == 1 ? volume * std::abs(value) : volume * value * value;
                }
            }
            D::PublishBlock(local, scratch);
        }
        __global__ void Equation(NonlinearCells cells, LinearReadRows mean, LinearReadRows residual,
                                 LinearWriteRows output, LinearWriteRows rate, real mass, real rhsScale)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < mean.RowCount(); iCell += index(gridDim.x) * blockDim.x)
                for (index j = 0; j < mean.Width(); ++j)
                {
                    const real volume = cells(iCell, 0);
                    output(iCell, j) = mass * mean(iCell, j) - rhsScale * residual(iCell, j) / volume;
                    if (rate.RowCount())
                        rate(iCell, j) = residual(iCell, j) / volume;
                }
        }
    }
    index NonlinearScratchRows(index rows)
    {
        DNDS_check_throw_info(rows >= 0, "negative nonlinear reduction row count");
        return std::max<index>(1, std::min<index>(256, rows / D::Threads + (rows % D::Threads != 0)));
    }
    void D::Finish(R::ExecutionContext context, NonlinearSummaryOutput scratch, NonlinearSummaryOutput result)
    {
        CheckLaunch();
        ReduceSummary<<<1, Threads, 0, ::DNDS::Runtime::CUDAStream(context)>>>(scratch.ConstView(), result);
        CheckLaunch();
    }
    void AssembleDefect(R::ExecutionContext context, NonlinearCells cells, LinearReadRows mean, LinearReadRows residual,
                        LinearReadRows constant, LinearWriteRows output, NonlinearSelection selection, real mass, real rhsScale,
                        int normOrder, bool measureNorm, NonlinearSummaryOutput scratch, NonlinearSummaryOutput result)
    {
        D::CheckInput(context, mean);
        D::CheckCells(context, cells, mean.RowCount());
        D::CheckRows(context, residual, mean.RowCount(), mean.Width());
        D::CheckRows(context, constant, mean.RowCount(), mean.Width());
        D::CheckRows(context, output, mean.RowCount(), mean.Width());
        D::CheckSummary(context, mean.RowCount(), scratch, result);
        DNDS_check_throw_info(std::isfinite(mass) && mass >= 0 && std::isfinite(rhsScale) && rhsScale >= 0 && (normOrder == 1 || normOrder == 2) &&
                                  (!output.data || (output.data != mean.data && output.data != residual.data && output.data != constant.data && output.data != cells.data)),
                              "invalid CUDA defect coefficients or alias");
        R::CheckKernelArgumentBudget<NonlinearCells, LinearReadRows, LinearReadRows, LinearReadRows, LinearWriteRows,
                                     NonlinearSelection, real, real, int, bool, NonlinearSummaryOutput>();
        Defect<<<unsigned(scratch.RowCount()), D::Threads, 0, ::DNDS::Runtime::CUDAStream(context)>>>(cells, mean, residual, constant,
                                                                                                      output, selection, mass, rhsScale, normOrder, measureNorm, scratch);
        D::Finish(context, scratch, result);
    }
    void MeasureMeanState(R::ExecutionContext context, NonlinearCells cells, LinearReadRows mean,
                          NonlinearSummaryOutput scratch, NonlinearSummaryOutput result)
    {
        D::CheckInput(context, mean);
        D::CheckCells(context, cells, mean.RowCount());
        D::CheckSummary(context, mean.RowCount(), scratch, result);
        R::CheckKernelArgumentBudget<NonlinearCells, LinearReadRows, NonlinearSummaryOutput>();
        MeanScale<<<unsigned(scratch.RowCount()), D::Threads, 0, ::DNDS::Runtime::CUDAStream(context)>>>(cells, mean, scratch);
        D::Finish(context, scratch, result);
    }
    void ScaleRateByVolume(R::ExecutionContext context, NonlinearCells cells, LinearReadRows input, LinearWriteRows output, bool multiply)
    {
        D::CheckInput(context, input);
        D::CheckCells(context, cells, input.RowCount());
        D::CheckRows(context, output, input.RowCount(), input.Width());
        DNDS_check_throw_info(!output.data || (output.data != input.data && output.data != cells.data), "CUDA rate output aliases input");
        R::CheckKernelArgumentBudget<NonlinearCells, LinearReadRows, LinearWriteRows, bool>();
        if (!input.RowCount())
            return;
        Rate<<<unsigned(NonlinearScratchRows(input.RowCount())), D::Threads, 0, ::DNDS::Runtime::CUDAStream(context)>>>(cells, input, output, multiply);
        D::CheckLaunch();
    }
    void MeasureRateNorm(R::ExecutionContext context, NonlinearCells cells, LinearReadRows rate,
                         LinearReadRows reference, int normOrder, NonlinearSummaryOutput scratch, NonlinearSummaryOutput result)
    {
        D::CheckInput(context, rate);
        D::CheckCells(context, cells, rate.RowCount());
        D::CheckSummary(context, rate.RowCount(), scratch, result);
        if (reference.RowCount())
            D::CheckRows(context, reference, rate.RowCount(), rate.Width());
        DNDS_check_throw_info(normOrder == 1 || normOrder == 2, "invalid CUDA rate norm order");
        R::CheckKernelArgumentBudget<NonlinearCells, LinearReadRows, LinearReadRows, int, NonlinearSummaryOutput>();
        RateNorm<<<unsigned(scratch.RowCount()), D::Threads, 0, ::DNDS::Runtime::CUDAStream(context)>>>(cells, rate, reference, normOrder, scratch);
        D::Finish(context, scratch, result);
    }
    void AssembleEquation(R::ExecutionContext context, NonlinearCells cells, LinearReadRows mean, LinearReadRows residual,
                          LinearWriteRows output, LinearWriteRows rate, real mass, real rhsScale)
    {
        D::CheckInput(context, mean);
        D::CheckCells(context, cells, mean.RowCount());
        D::CheckRows(context, residual, mean.RowCount(), mean.Width());
        D::CheckRows(context, output, mean.RowCount(), mean.Width());
        if (rate.RowCount())
            D::CheckRows(context, rate, mean.RowCount(), mean.Width());
        DNDS_check_throw_info(std::isfinite(mass) && mass >= 0 && std::isfinite(rhsScale) && rhsScale >= 0, "invalid CUDA stage equation coefficients");
        for (const auto target : {output, rate})
            DNDS_check_throw_info(!target.data || (target.data != mean.data && target.data != residual.data && target.data != cells.data),
                                  "CUDA stage equation output aliases input");
        DNDS_check_throw_info(!rate.data || output.data != rate.data, "CUDA stage equation outputs alias");
        R::CheckKernelArgumentBudget<NonlinearCells, LinearReadRows, LinearReadRows, LinearWriteRows, LinearWriteRows, real, real>();
        if (!mean.RowCount())
            return;
        Equation<<<unsigned(NonlinearScratchRows(mean.RowCount())), D::Threads, 0, ::DNDS::Runtime::CUDAStream(context)>>>(
            cells, mean, residual, output, rate, mass, rhsScale);
        D::CheckLaunch();
    }
}
