#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Core/CUDA/TimeStageKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <climits>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        constexpr int Threads = 128;
        struct Merge
        {
            __device__ TimeStageSummary operator()(TimeStageSummary a, const TimeStageSummary &b) const
            {
                a.blend.alpha = std::min(a.blend.alpha, b.blend.alpha);
                a.blend.limited += b.blend.limited;
                for (int j = 0; j < 2; ++j)
                    a.blend.safe[j] &= b.blend.safe[j];
                if (!b.status && (a.status || b.status.row < a.status.row))
                    a.status = b.status;
                return a;
            }
        };
        __device__ void Publish(TimeStageSummary value, TimeStageSummaryOutput out)
        {
            using Reduce = cub::BlockReduce<TimeStageSummary, Threads, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
            __shared__ typename Reduce::TempStorage scratch;
            const auto result = Reduce(scratch).Reduce(value, Merge{});
            if (!threadIdx.x)
                out(blockIdx.x, 0) = result;
        }
        __global__ void Finish(TimeStageSummaryOutput partial, TimeStageSummaryOutput result)
        {
            TimeStageSummary value;
            for (index iRow = threadIdx.x; iRow < partial.RowCount(); iRow += Threads)
                value = Merge{}(value, partial(iRow, 0));
            Publish(value, result);
        }
        __global__ void Blend(TimeStageCells cells, TimeStageInput initial, TimeStageInput stage,
                              TimeStageInput high, TimeStageInput low, ThermalBounds bounds, ExplicitStageCoefficients c,
                              TimeStageSummaryOutput partial)
        {
            TimeStageSummary value;
            for (index iCell = index(blockIdx.x) * Threads + threadIdx.x; iCell < stage.RowCount(); iCell += index(gridDim.x) * Threads)
            {
                FlowState base, target;
                bool finite = true;
                for (int j = 0; j < 5; ++j)
                {
                    const real constant = c.historyWeight * initial(iCell, j) + c.stageWeight * stage(iCell, j);
                    base[j] = constant + c.stageWeight * c.dt * low(iCell, j) / cells(iCell, 0);
                    target[j] = constant + c.stageWeight * c.dt * high(iCell, j) / cells(iCell, 0);
                    finite &= std::isfinite(base[j]) && std::isfinite(target[j]);
                }
                if (!finite)
                {
                    if (value.status)
                        value.status = {NumericalError::InvalidState, 1, iCell};
                    continue;
                }
                value.blend.safe[0] &= ThermallyAdmissible(target, bounds);
                const bool lowSafe = ThermallyAdmissible(base, bounds);
                value.blend.safe[1] &= lowSafe;
                const real alpha = lowSafe ? AdmissibleFraction(base, target, bounds) : real(1);
                if (alpha < 1)
                {
                    ++value.blend.limited;
                    value.blend.alpha = std::min(value.blend.alpha, alpha);
                }
            }
            Publish(value, partial);
        }
        __global__ void Flow(TimeStageCells cells, TimeStageInput initial, TimeStageInput rhs,
                             TimeStageInput low, TimeStageOutput stage, real alpha, ExplicitStageCoefficients c)
        {
            const index iCell = index(blockIdx.x) * Threads + threadIdx.x;
            if (iCell >= stage.RowCount())
                return;
            for (int j = 0; j < 5; ++j)
            {
                const real residual = alpha == 1 ? rhs(iCell, j) : (1 - alpha) * low(iCell, j) + alpha * rhs(iCell, j);
                stage(iCell, j) = c.historyWeight * initial(iCell, j) + c.stageWeight * (stage(iCell, j) + c.dt * residual / cells(iCell, 0));
            }
        }
        struct StepIncrement
        {
            TimeStageInput values;
            index row;
            real dtScale, volume;
            __device__ real operator[](index j) const { return dtScale * values(row, j) / volume; }
        };
        __global__ void PhysicalStep(TimeStageCells cells, TimeStageInput mean, TimeStageInput rhs,
                                     ThermalView thermal, real dt, real relax, real scale, int power, TimeStageSummaryOutput partial)
        {
            TimeStageSummary result;
            for (index iCell = index(blockIdx.x) * Threads + threadIdx.x; iCell < mean.RowCount(); iCell += index(gridDim.x) * Threads)
            {
                const ThermalAdmissibilityDetail::PointRow<TimeStageInput> row{mean, iCell};
                const StepIncrement delta{rhs, iCell, dt * scale, cells(iCell, 0)};
                int error = thermal.Admissible(row) ? 0 : 1;
                for (index j = 0; j < mean.Width() && !error; ++j)
                    if (!std::isfinite(delta[j]))
                        error = 2;
                real alpha = 1;
                if (!error)
                {
                    // Match CPU/Euler forecast: one thermal factor for dt,
                    // with no independent RANS scalar admissibility condition.
                    alpha = thermal.IncrementFraction(row, delta, relax, 1, power);
                    if (!(std::isfinite(alpha) && alpha >= 0 && alpha <= 1))
                        error = 3;
                }
                if (error)
                {
                    if (result.status)
                        result.status = {NumericalError::InvalidState, error, iCell};
                }
                else if (alpha < 1)
                {
                    ++result.blend.limited;
                    result.blend.alpha = std::min(result.blend.alpha, alpha);
                }
            }
            Publish(result, partial);
        }
        template <int N>
        __global__ void Extended(TimeStageCells cells, TimeStageInput initial, TimeStageInput rhs,
                                 TimeStageOutput stage, RANSStateView rans, ExplicitStageCoefficients c, TimeStageSummaryOutput partial)
        {
            TimeStageSummary result;
            for (index iCell = index(blockIdx.x) * Threads + threadIdx.x; iCell < stage.RowCount(); iCell += index(gridDim.x) * Threads)
            {
                // PP POLICY: flow keeps the conservative SSPRK increment.
                // Only the separate RANS block receives exponential repair.
                for (int j = 0; j < 5; ++j)
                    stage(iCell, j) = c.historyWeight * initial(iCell, j) + c.stageWeight * (stage(iCell, j) + c.dt * rhs(iCell, j) / cells(iCell, 0));
                std::array<real, 2> old{}, change{}, next{};
                bool finite = true;
                for (int j = 0; j < N - 5; ++j)
                {
                    old[j] = stage(iCell, 5 + j);
                    change[j] = c.historyWeight * initial(iCell, 5 + j) + c.stageWeight * (old[j] + c.dt * rhs(iCell, 5 + j) / cells(iCell, 0)) - old[j];
                    finite &= std::isfinite(change[j]);
                }
                if (!finite)
                {
                    if (result.status)
                        result.status = {NumericalError::InvalidState, 1, iCell};
                    continue;
                }
                rans.RepairIncrement(old, change);
                for (int j = 0; j < N - 5; ++j)
                    next[j] = old[j] + change[j];
                rans.RepairMean(stage(iCell, 0), next);
                for (int j = 0; j < N - 5; ++j)
                    stage(iCell, 5 + j) = next[j];
            }
            Publish(result, partial);
        }
        template <class View>
        void CheckRows(R::ExecutionContext context, View view, index rows, index width)
        {
            view.ValidateDevice(context.device);
            DNDS_check_throw_info(view.kind == R::RowLayoutKind::Table && view.RowCount() == rows && view.Width() == width,
                                  "explicit CUDA stage requires matching combined AoS rows");
        }
        void Check(R::ExecutionContext context, TimeStageCells cells, TimeStageInput initial,
                   TimeStageInput rhs, ExplicitStageCoefficients c)
        {
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && initial.Width() >= 5 && initial.Width() <= 7 &&
                                      std::isfinite(c.dt) && c.dt > 0 && std::isfinite(c.historyWeight) && c.historyWeight >= 0 &&
                                      std::isfinite(c.stageWeight) && c.stageWeight >= 0,
                                  "invalid explicit CUDA coefficients or profile");
            CheckRows(context, initial, initial.RowCount(), initial.Width());
            CheckRows(context, rhs, initial.RowCount(), initial.Width());
            cells.ValidateDevice(context.device);
            DNDS_check_throw_info(cells.RowCount() >= initial.RowCount(), "explicit CUDA geometry lacks owned rows");
        }
        void CheckSummary(R::ExecutionContext context, index rows, TimeStageSummaryOutput partial, TimeStageSummaryOutput result)
        {
            R::ValidateViewDevices(context.device, partial, result);
            DNDS_check_throw_info(partial.RowCount() == TimeStageScratchRows(rows) && result.RowCount() == 1 && partial.data != result.data,
                                  "explicit CUDA reduction needs distinct bounded scratch/result");
        }
        void CheckLaunch() { DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "explicit CUDA stage launch failed"); }
    }
    index TimeStageScratchRows(index rows) { return std::max(index(1), std::min(index(256), (rows + Threads - 1) / Threads)); }
    void LimitPhysicalStep(R::ExecutionContext context, TimeStageCells cells, TimeStageInput mean, TimeStageInput rhs,
                           ThermalView thermal, real dt, real relax, real scale, int power, TimeStageSummaryOutput partial, TimeStageSummaryOutput result)
    {
        Check(context, cells, mean, rhs, {dt, 0, 1});
        CheckSummary(context, mean.RowCount(), partial, result);
        DNDS_check_throw_info(thermal.components == mean.Width() && thermal.flowFirst == 0 && thermal.independent == 0 &&
                                  thermal.speciesEnergies[0].empty() && thermal.speciesEnergies[1].empty() && std::isfinite(relax) && relax > 0 && relax <= 1 &&
                                  std::isfinite(scale) && scale > 0 && power > 0 && power <= INT_MAX / 2,
                              "CUDA physical-step forecast needs matching perfect-gas policy and valid controls");
        R::CheckKernelArgumentBudget<TimeStageCells, TimeStageInput, TimeStageInput, ThermalView, real, real, real, int, TimeStageSummaryOutput>();
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        PhysicalStep<<<unsigned(partial.RowCount()), Threads, 0, stream>>>(cells, mean, rhs, thermal, dt, relax, scale, power, partial);
        Finish<<<1, Threads, 0, stream>>>(partial, result);
        CheckLaunch();
    }
    void MeasureExplicitBlend(R::ExecutionContext context, TimeStageCells cells, TimeStageInput initial,
                              TimeStageInput stage, TimeStageInput high, TimeStageInput low, ThermalBounds bounds, ExplicitStageCoefficients c,
                              TimeStageSummaryOutput partial, TimeStageSummaryOutput result)
    {
        Check(context, cells, initial, high, c);
        DNDS_check_throw_info(initial.Width() == 5, "explicit conservative blend is flow5 only");
        CheckRows(context, stage, initial.RowCount(), 5);
        CheckRows(context, low, initial.RowCount(), 5);
        CheckSummary(context, initial.RowCount(), partial, result);
        R::CheckKernelArgumentBudget<TimeStageCells, TimeStageInput, TimeStageInput, TimeStageInput, TimeStageInput,
                                     ThermalBounds, ExplicitStageCoefficients, TimeStageSummaryOutput>();
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        Blend<<<unsigned(partial.RowCount()), Threads, 0, stream>>>(cells, initial, stage, high, low, bounds, c, partial);
        Finish<<<1, Threads, 0, stream>>>(partial, result);
        CheckLaunch();
    }
    void UpdateExplicitFlow(R::ExecutionContext context, TimeStageCells cells, TimeStageInput initial,
                            TimeStageInput rhs, TimeStageInput low, TimeStageOutput stage, real alpha, ExplicitStageCoefficients c)
    {
        Check(context, cells, initial, rhs, c);
        CheckRows(context, stage, initial.RowCount(), 5);
        DNDS_check_throw_info(initial.Width() == 5 && alpha >= 0 && alpha <= 1 && stage.data != initial.data && stage.data != rhs.data,
                              "invalid explicit flow blend or aliased state");
        if (alpha != 1)
        {
            CheckRows(context, low, initial.RowCount(), 5);
            DNDS_check_throw_info(stage.data != low.data, "explicit low residual aliases stage");
        }
        R::CheckKernelArgumentBudget<TimeStageCells, TimeStageInput, TimeStageInput, TimeStageInput, TimeStageOutput, real, ExplicitStageCoefficients>();
        if (stage.RowCount())
            Flow<<<unsigned((stage.RowCount() + Threads - 1) / Threads), Threads, 0,
                   ::DNDS::Runtime::CUDAStream(context)>>>(cells, initial, rhs, low, stage, alpha, c);
        CheckLaunch();
    }
    void UpdateExplicitExtended(R::ExecutionContext context, TimeStageCells cells, TimeStageInput initial,
                                TimeStageInput rhs, TimeStageOutput stage, RANSStateView rans, ExplicitStageCoefficients c,
                                TimeStageSummaryOutput partial, TimeStageSummaryOutput result)
    {
        Check(context, cells, initial, rhs, c);
        CheckRows(context, stage, initial.RowCount(), initial.Width());
        CheckSummary(context, initial.RowCount(), partial, result);
        DNDS_check_throw_info((initial.Width() == 6 || initial.Width() == 7) && rans.components == initial.Width() - 5 &&
                                  stage.data != initial.data && stage.data != rhs.data,
                              "explicit extended CUDA profile or alias differs");
        R::CheckKernelArgumentBudget<TimeStageCells, TimeStageInput, TimeStageInput, TimeStageOutput, RANSStateView, ExplicitStageCoefficients, TimeStageSummaryOutput>();
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        if (initial.Width() == 6)
            Extended<6><<<unsigned(partial.RowCount()), Threads, 0, stream>>>(cells, initial, rhs, stage, rans, c, partial);
        else
            Extended<7><<<unsigned(partial.RowCount()), Threads, 0, stream>>>(cells, initial, rhs, stage, rans, c, partial);
        Finish<<<1, Threads, 0, stream>>>(partial, result);
        CheckLaunch();
    }
}
