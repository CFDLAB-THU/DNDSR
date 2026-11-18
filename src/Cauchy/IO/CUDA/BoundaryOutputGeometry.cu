#include "Cauchy/IO/BoundaryOutputGeometry.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <climits>

namespace DNDS::Cauchy::BoundaryOutputDetail
{
    namespace
    {
        __global__ void GatherBoundaryRows(Rows rows, Values flux, Momentum momentum, Values quadrature, Output output)
        {
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < rows.RowCount(); iRow += index(gridDim.x) * blockDim.x)
                GatherFace(rows(iRow, 0), flux, momentum, quadrature, output, iRow);
        }
    }

    void GatherDeviceFaces(R::ExecutionContext context, Rows rows, Values flux, Momentum momentum, Values quadrature, Output output)
    {
        R::ValidateViewDevices(context.device, rows, flux, momentum, quadrature, output);
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && rows.RowCount() == output.RowCount() &&
                                  output.Width() == flux.Width() + 3,
                              "invalid CUDA boundary output bindings");
        if (!rows.RowCount())
            return;
        R::CheckKernelArgumentBudget<Rows, Values, Momentum, Values, Output>();
        constexpr int threads = 128;
        const auto blocks = unsigned(std::min<index>((rows.RowCount() - 1) / threads + 1, INT_MAX));
        DNDS_check_throw_info(cudaSetDevice(context.device.ordinal) == cudaSuccess, "boundary output device selection failed");
        GatherBoundaryRows<<<blocks, threads, 0, R::CUDAStream(context)>>>(rows, flux, momentum, quadrature, output);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "boundary output gather launch failed");
    }
}
