#pragma once

#include "CFV/Math/TaylorBasis.hpp"
#include "DNDS/Runtime/Array/Array.hpp"

namespace DNDS::Cauchy
{
    /// Immutable cell-major quadrature bank. Source points are constructed in
    /// offsets[cell] order; there is no scatter map or ghost query here.
    struct VolumeSourceBasis
    {
        using Rows = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
        Rows points, cached, moments, geometry;
        CFV::Math::TaylorBasisParameters polynomial;
        int cachedDerivatives = 0;

        /// At most P3 in three dimensions. Reuse each derivative's mode values
        /// across every equation component at this quadrature point.
        DNDS_DEVICE_CALLABLE void Fill(index cell, index point, int derivatives, real (&values)[4][19]) const noexcept
        {
            CFV::Math::TaylorPointEvaluation direct{};
            if (derivatives > cachedDerivatives)
                direct = CFV::Math::TaylorPointEvaluation::FromPoint<real>(
                    {points(point, 0), points(point, 1), points(point, 2)},
                    {geometry(cell, 1), geometry(cell, 2), geometry(cell, 3)}, geometry(cell, 7));
            for (int d = 0; d < derivatives; ++d)
                for (int m = 0; m < polynomial.size - 1; ++m)
                    values[d][m] = d < cachedDerivatives ? cached(point, m * cachedDerivatives + d)
                                                         : polynomial.ReconstructionDerivative(d, m, direct, moments, cell);
        }
    };
}
