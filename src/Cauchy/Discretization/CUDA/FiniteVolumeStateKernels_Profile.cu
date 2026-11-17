#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
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

        __device__ void Merge(FlowErrorReduction &a, const FlowErrorReduction &b)
        {
            for (int j = 0; j < 5; ++j)
            {
                a.errors[0][j] += b.errors[0][j];
                a.errors[1][j] += b.errors[1][j];
                a.errors[2][j] = fmax(a.errors[2][j], b.errors[2][j]);
            }
            a.volume += b.volume;
            // Deterministic tie breaking also makes the status merge commutative
            // for reused/test records, although production cells are disjoint.
            const bool earlier = b.status.row < a.status.row ||
                                 (b.status.row == a.status.row && (b.status.point < a.status.point ||
                                                                   (b.status.point == a.status.point && (int(b.status.code) < int(a.status.code) ||
                                                                                                         (b.status.code == a.status.code && b.status.detail < a.status.detail)))));
            if (!b.status && (a.status || earlier))
                a.status = b.status;
        }
        struct MergeErrors
        {
            __device__ FlowErrorReduction operator()(FlowErrorReduction a, const FlowErrorReduction &b) const
            {
                Merge(a, b);
                return a;
            }
        };
        __device__ void StoreBlock(FlowErrorReduction value, FlowErrorOutput output, index row)
        {
            // Use CCCL's reduction primitive, with an explicit non-atomic tree.
            // The regular partial table is reduced by the same primitive in
            // the last arriving block. No host scalar return or hidden allocation.
            using Reduce = cub::BlockReduce<FlowErrorReduction, Threads, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
            __shared__ typename Reduce::TempStorage temporary;
            const auto reduced = Reduce(temporary).Reduce(value, MergeErrors{});
            if (!threadIdx.x)
                output(row, 0) = reduced;
        }
        __device__ NumericalResult<FlowState> CellAverage(FlowProfileQuadrature geometry, index iCell,
                                                          FlowProfileParameters profile, FlowState primitive,
                                                          PerfectGas gas, RigidFrameParameters frame, real time)
        {
            if (profile.kind == FlowProfileKind::Constant)
                return frame.LocalState(gas.Conservative(primitive));
            FlowState average{};
            real volume = 0;
            const index begin = geometry.offsets(iCell, 0), end = geometry.offsets(iCell + 1, 0);
            for (index iPoint = begin; iPoint < end; ++iPoint)
            {
                const Vector3 point{geometry.points(iPoint, 0), geometry.points(iPoint, 1), geometry.points(iPoint, 2)};
                const real weight = geometry.points(iPoint, 3);
                const auto exact = TryProfileValue(profile, primitive, gas, frame.WorldPoint(point), time);
                if (!exact)
                    return exact.status.At(iCell, iPoint - begin);
                const auto local = frame.LocalState(exact.value);
                for (int j = 0; j < 5; ++j)
                    average[j] += weight * local[j];
                volume += weight;
            }
            if (!(isfinite(volume) && volume > 0))
                return NumericalStatus{NumericalError::InvalidInput, 1}.At(iCell);
            for (auto &value : average)
                value /= volume;
            return average;
        }
        __global__ void ErrorRows(FlowProfileQuadrature geometry, StateInput means, FlowProfileParameters profile,
                                  FlowState primitive, PerfectGas gas, RigidFrameParameters frame, real time,
                                  FlowErrorOutput scratch, FlowErrorOutput result, FlowErrorCounter completed)
        {
            FlowErrorReduction sum;
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iCell < means.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const auto exact = CellAverage(geometry, iCell, profile, primitive, gas, frame, time);
                if (!exact)
                {
                    if (sum.status)
                        sum.status = exact.status;
                    continue;
                }
                FlowErrorReduction cell;
                cell.volume = geometry.cells(iCell, 0);
                for (int j = 0; j < 5; ++j)
                {
                    const real error = fabs(means(iCell, j) - exact.value[j]);
                    if (!isfinite(error))
                        cell.status = NumericalStatus{NumericalError::InvalidState, 2}.At(iCell);
                    cell.errors[0][j] = cell.volume * error;
                    cell.errors[1][j] = cell.volume * error * error;
                    cell.errors[2][j] = error;
                }
                Merge(sum, cell);
            }
            StoreBlock(sum, scratch, blockIdx.x);
            __shared__ bool last;
            if (!threadIdx.x)
                last = Device::ArriveReductionBlock(completed(0, 0), gridDim.x);
            __syncthreads();
            if (!last)
                return;
            FlowErrorReduction tail;
            for (index iRow = threadIdx.x; iRow < scratch.RowCount(); iRow += blockDim.x)
                Merge(tail, scratch(iRow, 0));
            StoreBlock(tail, result, 0);
        }
    }
    void MeasureFlowError(R::ExecutionContext context, FlowProfileQuadrature geometry, StateInput means,
                          FlowProfileParameters profile, FlowState primitive, PerfectGas gas, RigidFrameParameters frame,
                          real time, FlowErrorOutput scratch, FlowErrorOutput result, FlowErrorCounter completed)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "flow error reduction requires CUDA");
        geometry.ValidateDevice(context.device);
        R::ValidateViewDevices(context.device, means, scratch, result, completed);
        DNDS_check_throw_info(means.kind == R::RowLayoutKind::Table && means.Width() >= 5 && means.Width() <= 7 &&
                                  geometry.offsets.RowCount() == means.RowCount() + 1 && geometry.cells.RowCount() >= means.RowCount() &&
                                  scratch.RowCount() == StateReductionScratchRows(means.RowCount()) && result.RowCount() == 1 &&
                                  scratch.data != result.data && completed.RowCount() == 1 && std::isfinite(time),
                              "flow error reduction shape/placement differs");
        R::CheckKernelArgumentBudget<FlowProfileQuadrature, StateInput, FlowProfileParameters, FlowState,
                                     PerfectGas, RigidFrameParameters, real, FlowErrorOutput, FlowErrorOutput, FlowErrorCounter>();
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        DNDS_check_throw_info(cudaMemsetAsync(completed.data, 0, sizeof(unsigned int), stream) == cudaSuccess, "CUDA flow error counter reset failed");
        ErrorRows<<<unsigned(scratch.RowCount()), Threads, 0, stream>>>(
            geometry, means, profile, primitive, gas, frame, time, scratch, result, completed);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA flow error row launch failed");
    }
}
