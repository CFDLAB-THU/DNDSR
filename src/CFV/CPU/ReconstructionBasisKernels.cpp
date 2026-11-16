#include "CFV/CPU/ReconstructionBasisKernels.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::CFV::CPU
{
    Math::ReconstructionBasisView EvaluateFaceBasis(
        const ReconstructionBasisEvaluationView &input,
        index face, int side, index point, int derivativeOrder,
        Math::SmallMatrix &scratch)
    {
        const Math::TaylorBasis polynomial(input.dimension, input.degree);
        const int rows = polynomial.DerivativeNumber(derivativeOrder);
        if (point >= 0 && input.cached.data && rows <= input.cachedDerivatives)
            return {input.cached[2 * point + side].data(), rows,
                    input.modes, input.cachedDerivatives};
        const index cell = input.mesh.FaceCells(face)[side];
        const auto position = Detail::FacePointInCell(
            input.mesh, face, side, point < 0 ? input.geometry.GetFaceCent(face) : input.geometry.GetFacePoint(point));
        const auto center = input.geometry.GetCellCent(cell);
        const auto prepared = Math::TaylorPointEvaluation::FromPoint(
            {position[0], position[1], position[2]}, {center[0], center[1], center[2]},
            input.geometry.cells(cell, 7));
        const auto parameters = polynomial.KernelParameters();
        scratch.resize(rows, input.modes);
        for (int mode = 0; mode < input.modes; ++mode)
            for (int derivative = 0; derivative < rows; ++derivative)
            {
                if (point >= 0 && input.cached.data && derivative < input.cachedDerivatives)
                    scratch(derivative, mode) = input.cached(2 * point + side, mode * input.cachedDerivatives + derivative);
                else
                {
                    const real value = parameters.Derivative(derivative, mode + 1, prepared);
                    scratch(derivative, mode) = derivative == 0 ? value - input.moments(cell, mode + 1) : value;
                }
            }
        return {scratch.data(), int(scratch.rows()), input.modes,
                int(scratch.rows())};
    }
}
