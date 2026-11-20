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
        __global__ void AssembleACMFluxJacobianRows(ACMJacobianInput request, R::RowSelection rows, ACMJacobianStatusRows status)
        {
            for (index slot = index(blockIdx.x) * blockDim.x + threadIdx.x; slot < rows.RowCount(); slot += index(blockDim.x) * gridDim.x)
            {
                const index cell = rows.RowAt(slot);
                status(cell, 0) = ACM::AssembleRow<C>(request, request.stencil(0, 0), cell);
            }
        }
    }
    void PrepareACMFluxJacobian(R::ExecutionContext context, const ACMJacobianInput &q, R::RowSelection rows, ACMJacobianStatusRows status)
    {
        const index cells = q.diagonal.RowCount(), width = q.input.owned.Width();
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && width >= 4 && width <= 6 &&
                                  q.diagonal.Width() == width * width && q.input.owned.RowCount() == cells &&
                                  q.input.ghost.Width() == width && q.offDiagonal.blockSize == width &&
                                  q.offDiagonal.RowNumber() == cells && q.offDiagonal.values.Width() == width * width &&
                                  q.tau.RowCount() == cells && q.radius.Width() == 2 && q.tau.Width() == 1 &&
                                  status.RowCount() == cells && q.stencil.RowCount() == 1 &&
                                  (width == 4 || q.wallMean.RowCount() >= cells),
                              "invalid CUDA ACM Jacobian layout");
        R::ValidateViewDevices(context.device, q.input.owned, q.input.ghost, q.diagonal, q.offDiagonal, q.radius, q.tau, q.incoming, q.conditions.rows, q.stencil, status);
        DNDS_check_throw_info((q.profile == EquationPhysicsProfile::ACM && width == 4) ||
                                  (q.profile == EquationPhysicsProfile::ACMSA && width == 5) ||
                                  (q.profile == EquationPhysicsProfile::ACMTwoEquation && width == 6),
                              "CUDA ACM flux Jacobian component width differs from equation physics");
        if (width > 4)
            R::ValidateViewDevices(context.device, q.wallMean);
        rows.ValidateDevice(context.device);
        R::CheckKernelArgumentBudget<ACMJacobianInput, R::RowSelection, ACMJacobianStatusRows>();
        if (!rows.RowCount())
            return;
        const int blocks = int(std::min<index>(65535, 1 + (rows.RowCount() - 1) / 64));
        const auto stream = R::CUDAStream(context);
        if (q.profile == EquationPhysicsProfile::ACM)
            AssembleACMFluxJacobianRows<4><<<blocks, 64, 0, stream>>>(q, rows, status);
        else if (q.profile == EquationPhysicsProfile::ACMSA)
            AssembleACMFluxJacobianRows<5><<<blocks, 64, 0, stream>>>(q, rows, status);
        else
            AssembleACMFluxJacobianRows<6><<<blocks, 64, 0, stream>>>(q, rows, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA ACM sparse assembly launch failed");
    }
}
