#pragma once

#include "Cauchy/Implicit/CUDA/NonlinearKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace DNDS::Cauchy::CUDA::NonlinearDetail
{
    constexpr int Threads = 128;
    struct Merge
    {
        __device__ NonlinearSummary operator()(NonlinearSummary a, const NonlinearSummary &b) const
        {
            for (int j = 0; j < 8; ++j)
                a.norms[j] += b.norms[j];
            a.limited += b.limited;
            a.minimum = std::min(a.minimum, b.minimum);
            a.frozen = a.frozen && b.frozen;
            if (!b.status && (a.status || b.status.row < a.status.row ||
                              (b.status.row == a.status.row && b.status.detail < a.status.detail)))
                a.status = b.status;
            return a;
        }
    };
    __device__ inline void PublishBlock(NonlinearSummary value, NonlinearSummaryOutput output)
    {
        using Reducer = cub::BlockReduce<NonlinearSummary, Threads, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
        __shared__ typename Reducer::TempStorage storage;
        const auto result = Reducer(storage).Reduce(value, Merge{});
        if (!threadIdx.x)
            output(blockIdx.x, 0) = result;
    }
    void Finish(Runtime::ExecutionContext, NonlinearSummaryOutput scratch, NonlinearSummaryOutput result);
    inline void CheckSummary(Runtime::ExecutionContext context, index rows, NonlinearSummaryOutput scratch, NonlinearSummaryOutput result)
    {
        DNDS_check_throw_info(context.device.space == Runtime::MemorySpace::CUDA &&
                                  scratch.RowCount() == NonlinearScratchRows(rows) && result.RowCount() == 1 && scratch.data != result.data,
                              "nonlinear reduction needs bounded distinct resident scratch/result");
        Runtime::ValidateViewDevices(context.device, scratch, result);
    }
    template <class View>
    void CheckRows(Runtime::ExecutionContext context, View view, index rows, index width)
    {
        view.ValidateDevice(context.device);
        DNDS_check_throw_info(view.kind == Runtime::RowLayoutKind::Table && view.RowCount() == rows && view.Width() == width,
                              "nonlinear CUDA field shape or combined AoS layout differs");
    }
    inline void CheckInput(Runtime::ExecutionContext context, LinearReadRows input)
    {
        DNDS_check_throw_info(context.device.space == Runtime::MemorySpace::CUDA && input.Width() >= 4 && input.Width() <= 7,
                              "nonlinear CUDA profile requires combined AoS width 4 through 7");
        CheckRows(context, input, input.RowCount(), input.Width());
    }
    inline void CheckCells(Runtime::ExecutionContext context, NonlinearCells cells, index rows)
    {
        cells.ValidateDevice(context.device);
        DNDS_check_throw_info(cells.RowCount() >= rows, "nonlinear cell metrics do not cover owned rows");
    }
    inline void CheckThermal(const ThermalView &thermal, int n)
    {
        DNDS_check_throw_info(thermal.components == n && thermal.flowFirst == 0 && thermal.independent == 0 &&
                                  thermal.speciesEnergies[0].empty() && thermal.speciesEnergies[1].empty(),
                              "nonlinear CUDA thermal policy must be perfect gas flow5 first");
    }
    inline void CheckLaunch()
    {
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "nonlinear CUDA launch failed");
    }
}
