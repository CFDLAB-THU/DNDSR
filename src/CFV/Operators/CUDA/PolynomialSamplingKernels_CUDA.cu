#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "CFV/Detail/PolynomialSamplingKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    template <bool Cached>
    __global__ void SamplePolynomial(PolynomialSamplingView plan, SamplingField mean, SamplingField coefficients,
                                     SamplingOutput output, int derivative)
    {
        const int modes = plan.polynomial.size - 1;
        for (index iPoint = index(blockIdx.x) * blockDim.x + threadIdx.x; iPoint < plan.cells.RowCount(); iPoint += index(gridDim.x) * blockDim.x)
        {
            const index iCell = plan.cells(iPoint, 0);
            const index iBasis = plan.basisRows.RowCount() ? plan.basisRows(iPoint, 0) : iPoint;
            const index iOutput = plan.outputRows.RowCount() ? plan.outputRows(iPoint, 0) : iPoint;
            real basis[19]; // P3 in 3D, bounded point-local scratch; no allocation.
            if constexpr (Cached)
            {
                const auto row = plan.basis[iBasis];
                for (int mode = 0; mode < modes; ++mode)
                    basis[mode] = row[mode * plan.cachedDerivatives + derivative];
            }
            else
            {
                const real scale = plan.geometry(iCell, 7);
                const std::array<real, 3> point{plan.points(iPoint, 0), plan.points(iPoint, 1), plan.points(iPoint, 2)};
                const std::array<real, 3> center{plan.geometry(iCell, 1), plan.geometry(iCell, 2), plan.geometry(iCell, 3)};
                const auto evaluation = Math::TaylorPointEvaluation::FromPoint(point, center, scale);
                for (int mode = 0; mode < modes; ++mode)
                    basis[mode] = plan.polynomial.ReconstructionDerivative(derivative, mode, evaluation, plan.moments, iCell);
            }
            const auto meanRow = mean[iCell], coefficientRow = coefficients[iCell];
            const auto result = output[iOutput];
            for (index component = 0; component < result.size(); ++component)
            {
                real value = derivative == 0 ? meanRow[component] : 0;
                for (int mode = 0; mode < modes; ++mode)
                    value += basis[mode] * coefficientRow[component * modes + mode];
                result[component] = value;
            }
        }
    }
    void LaunchPolynomialSamples(R::ExecutionContext context, PolynomialSamplingView plan,
                                 SamplingField mean, SamplingField coefficients, SamplingOutput output, int derivative)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA polynomial sampling requires a CUDA stream");
        R::ValidateViewDevices(context.device, plan, mean, coefficients, output);
        R::CheckKernelArgumentBudget<decltype(plan), decltype(mean), decltype(coefficients), decltype(output), int>();
        DNDS_check_throw_info(plan.polynomial.size >= 3 && plan.polynomial.size <= 20 && derivative >= 0 && derivative < plan.polynomial.size &&
                                  (!plan.cells.RowCount() || (plan.outputRows.RowCount() ? plan.outputRows.RowCount() == plan.cells.RowCount() : output.RowCount() == plan.cells.RowCount())) &&
                                  (!plan.basisRows.RowCount() || plan.basisRows.RowCount() == plan.cells.RowCount()) && coefficients.owned.Width() == mean.owned.Width() * (plan.polynomial.size - 1) &&
                                  output.Width() == mean.owned.Width(),
                              "CUDA polynomial sample shape mismatch");
        if (!plan.cells.RowCount())
            return;
        const auto blocks = unsigned(std::min<index>((plan.cells.RowCount() + 127) / 128, 1024));
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        if (derivative < plan.cachedDerivatives)
            SamplePolynomial<true><<<blocks, 128, 0, stream>>>(plan, mean, coefficients, output, derivative);
        else
            SamplePolynomial<false><<<blocks, 128, 0, stream>>>(plan, mean, coefficients, output, derivative);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA polynomial sampling launch failed");
    }
}
