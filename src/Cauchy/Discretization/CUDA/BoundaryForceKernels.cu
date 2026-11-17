#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/BoundaryForceKernels.hpp"
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
        struct Sum
        {
            __device__ Vector3 operator()(Vector3 a, Vector3 b) const
            {
                for (int d = 0; d < 3; ++d)
                    a[d] += b[d];
                return a;
            }
        };
        __global__ void Rows(ForceGeometry geometry, ForceInput values,
                             SpanView<const int> zones, ForceOutput partial, ForceOutput result, ForceCounter completed)
        {
            const auto &stencil = geometry.stencil(0, 0);
            Vector3 value{};
            for (index iBoundary = index(blockIdx.x) * Threads + threadIdx.x; iBoundary < geometry.faces.RowCount(); iBoundary += index(gridDim.x) * Threads)
            {
                const auto range = geometry.faces(iBoundary, 0);
                const int zone = stencil.topology.GetFaceZone(range.face);
                bool selected = false;
                for (index iZone = 0; iZone < zones.size(); ++iZone)
                    selected |= zones[iZone] == zone;
                if (!selected)
                    continue;
                const index begin = stencil.quadratureBoundaryValues ? range.quadratureBegin : iBoundary;
                const index end = begin + (stencil.quadratureBoundaryValues ? range.quadratureCount : 1);
                for (index iPoint = begin; iPoint < end; ++iPoint)
                    for (int d = 0; d < 3; ++d)
                        value[d] += values(iPoint, d);
            }
            using Reduce = cub::BlockReduce<Vector3, Threads, cub::BLOCK_REDUCE_RAKING>;
            __shared__ typename Reduce::TempStorage scratch;
            __shared__ bool last;
            const auto block = Reduce(scratch).Reduce(value, Sum{});
            if (!threadIdx.x)
            {
                for (int d = 0; d < 3; ++d)
                    partial(blockIdx.x, d) = block[d];
                last = Device::ArriveReductionBlock(completed(0, 0), gridDim.x);
            }
            __syncthreads();
            if (!last)
                return;
            Vector3 tail{};
            for (index iRow = threadIdx.x; iRow < partial.RowCount(); iRow += Threads)
                for (int d = 0; d < 3; ++d)
                    tail[d] += partial(iRow, d);
            const auto total = Reduce(scratch).Reduce(tail, Sum{});
            if (!threadIdx.x)
                for (int d = 0; d < 3; ++d)
                    result(0, d) = total[d];
        }
    }
    index BoundaryForceScratchRows(index points)
    {
        DNDS_check_throw_info(points >= 0, "negative boundary force point count");
        return std::max(index(1), std::min(index(256), points / Threads + (points % Threads != 0)));
    }
    void AccumulateBoundaryForce(R::ExecutionContext context, ForceGeometry geometry, ForceInput values,
                                 SpanView<const int> zones, ForceOutput partial, ForceOutput result, ForceCounter completed)
    {
        R::ValidateViewDevices(context.device, geometry, values, partial, result, completed);
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && geometry.stencil.RowCount() == 1 &&
                                  partial.RowCount() == BoundaryForceScratchRows(geometry.faces.RowCount()) && result.RowCount() == 1 && completed.RowCount() == 1,
                              "invalid CUDA boundary force views");
        const std::array<const void *, 3> outputs{partial.data, result.data, completed.data};
        for (index i = 0; i < index(outputs.size()); ++i)
        {
            for (index j = 0; j < i; ++j)
                DNDS_check_throw_info(outputs[i] != outputs[j], "CUDA boundary force outputs alias");
            for (const void *input : {static_cast<const void *>(geometry.stencil.data), static_cast<const void *>(geometry.faces.data), static_cast<const void *>(values.data)})
                DNDS_check_throw_info(!input || outputs[i] != input, "CUDA boundary force output aliases input");
        }
        R::CheckKernelArgumentBudget<ForceGeometry, ForceInput, SpanView<const int>, ForceOutput, ForceOutput, ForceCounter>();
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        DNDS_check_throw_info(cudaMemsetAsync(completed.data, 0, sizeof(unsigned int), stream) == cudaSuccess, "CUDA boundary force counter reset failed");
        Rows<<<unsigned(partial.RowCount()), Threads, 0, stream>>>(geometry, values, zones, partial, result, completed);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA boundary force reduction launch failed");
    }
}
