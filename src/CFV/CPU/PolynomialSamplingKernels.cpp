#include "CFV/Detail/PolynomialSamplingKernels.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV::Detail
{
    void EvaluatePolynomialSamples(PolynomialSamplingView plan, SamplingField mean, SamplingField coefficients,
                                   SamplingOutput output, int derivative)
    {
        Runtime::ValidateViewDevices(Runtime::Device::Host(), plan, mean, coefficients, output);
        const int modes = plan.polynomial.size - 1;
        OMP::ForRows(0, plan.cells.RowCount(), [&](index iPoint)
                     {
            const index iCell = plan.cells(iPoint, 0);
            const index iBasis = plan.basisRows.RowCount() ? plan.basisRows(iPoint, 0) : iPoint;
            const index iOutput = plan.outputRows.RowCount() ? plan.outputRows(iPoint, 0) : iPoint;
            std::array<real, 19> values;
            if (derivative < plan.cachedDerivatives)
                for (int mode = 0; mode < modes; ++mode)
                    values[mode] = plan.basis(iBasis, mode * plan.cachedDerivatives + derivative);
            else
            {
                const std::array<real, 3> point{plan.points(iPoint, 0), plan.points(iPoint, 1), plan.points(iPoint, 2)};
                const std::array<real, 3> center{plan.geometry(iCell, 1), plan.geometry(iCell, 2), plan.geometry(iCell, 3)};
                const auto evaluation = Math::TaylorPointEvaluation::FromPoint(point, center, plan.geometry(iCell, 7));
                for (int mode = 0; mode < modes; ++mode)
                    values[mode] = plan.polynomial.ReconstructionDerivative(derivative, mode, evaluation, plan.moments, iCell);
            }
            const auto meanRow = mean[iCell], coefficientRow = coefficients[iCell];
            const auto result = output[iOutput];
            for (index component = 0; component < result.size(); ++component)
            {
                real value = derivative == 0 ? meanRow[component] : 0;
                for (int mode = 0; mode < modes; ++mode)
                    value += values[mode] * coefficientRow[component * modes + mode];
                result[component] = value;
            } });
    }
}
