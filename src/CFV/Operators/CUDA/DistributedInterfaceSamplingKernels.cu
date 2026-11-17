#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "CFV/Operators/CUDA/DistributedInterfaceSamplingKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <climits>

namespace DNDS::CFV::CUDA
{
    namespace
    {
        constexpr int Threads = 128;
        template <bool Gradient>
        __global__ void SampleRows(Detail::DirectDonorSamplingView request)
        {
            for (index iQuery = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iQuery < request.output.RowCount(); iQuery += index(gridDim.x) * blockDim.x)
            {
                const index iCell = request.cells(iQuery, 0);
                for (index j = 0; j < request.output.Width(); ++j)
                {
                    real value = request.means(iCell, j);
                    if constexpr (Gradient)
                        for (int d = 0; d < request.dimension; ++d)
                            value += request.gradients(iCell, j * request.dimension + d) *
                                     (request.points(iQuery, d) - request.geometry(iCell, d + 1));
                    request.output(iQuery, j) = value;
                }
            }
        }
    }
    void EvaluateDirectDonors(Runtime::ExecutionContext context, Detail::DirectDonorSamplingView request)
    {
        DNDS_check_throw_info(context.device.space == Runtime::MemorySpace::CUDA, "CUDA donor sampling requires a CUDA context");
        request.ValidateDevice(context.device);
        DNDS_check_throw_info(request.cells.RowCount() == request.output.RowCount() && request.means.Width() == request.output.Width() &&
                                  request.means.kind == Runtime::RowLayoutKind::Table && request.output.kind == Runtime::RowLayoutKind::Table,
                              "CUDA donor sampling shape or AoS layout differs");
        if (request.gradient)
            DNDS_check_throw_info(request.dimension >= 1 && request.dimension <= 3 && request.points.Width() == 3 &&
                                      request.points.RowCount() == request.output.RowCount() && request.gradients.RowCount() == request.means.RowCount() &&
                                      request.gradients.Width() == request.dimension * request.means.Width() &&
                                      request.geometry.RowCount() >= request.means.RowCount() && request.gradients.kind == Runtime::RowLayoutKind::Table,
                                  "CUDA gradient donor sampling shape differs");
        Runtime::CheckKernelArgumentBudget<Detail::DirectDonorSamplingView>();
        if (!request.output.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>(INT_MAX, request.output.RowCount() / Threads + (request.output.RowCount() % Threads != 0)));
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        if (request.gradient)
            SampleRows<true><<<blocks, Threads, 0, stream>>>(request);
        else
            SampleRows<false><<<blocks, Threads, 0, stream>>>(request);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA direct donor sampling launch failed");
    }
}
