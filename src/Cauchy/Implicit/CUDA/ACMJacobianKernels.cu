#include "Cauchy/Implicit/CUDA/ACMJacobianKernels.hpp"
#include "Cauchy/Implicit/ACMJacobianRow.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        template <int C>
        __global__ void Assemble(ACMJacobianInput request, ACMJacobianStatusRows status)
        {
            for (index cell = index(blockIdx.x) * blockDim.x + threadIdx.x; cell < status.RowCount(); cell += index(blockDim.x) * gridDim.x)
                status(cell, 0) = ACM::AssembleRow<C>(request, request.stencil(0, 0), cell);
        }
    }
    void PrepareACMJacobian(R::ExecutionContext context, const ACMJacobianInput &q, ACMJacobianStatusRows status)
    {
        const index cells = q.diagonal.RowCount(), width = q.input.owned.Width();
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && width >= 4 && width <= 6 &&
                                  q.diagonal.Width() == width * width && q.input.owned.RowCount() == cells &&
                                  q.input.ghost.Width() == width && q.offDiagonal.blockSize == width &&
                                  q.offDiagonal.RowNumber() == cells && q.offDiagonal.values.Width() == width * width &&
                                  q.tau.RowCount() == cells && q.radius.Width() == 2 && q.tau.Width() == 1 &&
                                  status.RowCount() == cells && q.stencil.RowCount() == 1 &&
                                  (width == 4 || (q.lengths.RowCount() >= cells && q.wallMean.RowCount() >= cells)),
                              "invalid CUDA ACM Jacobian layout");
        R::ValidateViewDevices(context.device, q.input.owned, q.input.ghost, q.diagonal, q.offDiagonal, q.radius, q.tau, q.conditions.rows, q.stencil, status);
        if (width > 4)
            R::ValidateViewDevices(context.device, q.lengths, q.wallMean);
        R::CheckKernelArgumentBudget<ACMJacobianInput, ACMJacobianStatusRows>();
        if (!cells)
            return;
        const int blocks = int(std::min<index>(65535, 1 + (cells - 1) / 64));
        const auto stream = R::CUDAStream(context);
        if (width == 4)
            Assemble<4><<<blocks, 64, 0, stream>>>(q, status);
        else if (width == 5)
            Assemble<5><<<blocks, 64, 0, stream>>>(q, status);
        else
            Assemble<6><<<blocks, 64, 0, stream>>>(q, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA ACM sparse assembly launch failed");
    }
}
