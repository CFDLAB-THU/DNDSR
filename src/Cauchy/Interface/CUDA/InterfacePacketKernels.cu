#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Interface/CUDA/InterfacePacketKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <climits>

namespace DNDS::Cauchy::CUDA
{
    namespace
    {
        constexpr int Threads = 128;
        __global__ void PacketRows(InterfacePacketTableView request)
        {
            for (index iPoint = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iPoint < request.rows.RowCount(); iPoint += index(gridDim.x) * blockDim.x)
            {
                const auto row = request.rows(iPoint, 0);
                const auto destination = row.faceCenter ? request.faceCenters : request.quadrature;
                for (index j = 0; j < request.receiverComponents.size(); ++j)
                    destination(row.row, request.receiverComponents[j]) = request.values(iPoint, j);
                if (!request.rotation.donorActive && !request.rotation.receiverActive)
                    continue;
                for (const index first : request.vectorStarts)
                {
                    const auto rotated = request.rotation.Apply({destination(row.row, first), destination(row.row, first + 1), destination(row.row, first + 2)});
                    for (int d = 0; d < 3; ++d)
                        destination(row.row, first + d) = rotated[d];
                }
            }
        }
    }
    void FillInterfacePacket(Runtime::ExecutionContext context, InterfacePacketTableView request)
    {
        namespace R = Runtime;
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA interface packet requires CUDA execution");
        request.ValidateDevice(context.device);
        DNDS_check_throw_info(request.rows.RowCount() == request.values.RowCount() &&
                                  request.receiverComponents.size() == request.values.Width() &&
                                  request.quadrature.Width() == request.values.Width() && request.faceCenters.Width() == request.values.Width() &&
                                  request.values.kind == R::RowLayoutKind::Table && request.quadrature.kind == R::RowLayoutKind::Table && request.faceCenters.kind == R::RowLayoutKind::Table &&
                                  request.values.data != request.quadrature.data && request.values.data != request.faceCenters.data &&
                                  (!request.quadrature.RowCount() || !request.faceCenters.RowCount() || request.quadrature.data != request.faceCenters.data),
                              "CUDA interface packet requires disjoint complete AoS values");
        R::CheckKernelArgumentBudget<InterfacePacketTableView>();
        if (!request.rows.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>(INT_MAX, request.rows.RowCount() / Threads + (request.rows.RowCount() % Threads != 0)));
        PacketRows<<<blocks, Threads, 0, ::DNDS::Runtime::CUDAStream(context)>>>(request);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA interface packet launch failed");
    }
}
