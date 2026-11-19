#pragma once

#include "Physics/RANS/Model.hpp"

namespace DNDS::Cauchy
{
    class EquationSystem;
    /// Reconstruction-only scalar policy, independent of flow/species PP.
    /// Nonnegative quadrature values are obtained by scaling the entire RANS
    /// polynomial about its cell mean, with a common factor for the scalar pair.
    /// Zero is the floor; no density or far-field reference enters this rule.
    struct RANSReconstructionView
    {
        Physics::RANS::ModelKind kind = Physics::RANS::ModelKind::None;
        int components = 0;

        /// Euler reconstruction fallback is component-wise: an invalid RANS
        /// quadrature value reverts to its cell mean, without flattening flow
        /// or other RANS components. SA follows its existing compile-time flag.
        template <class Mean, class Row>
        DNDS_DEVICE_CALLABLE void RepairPoint(const Mean &mean, Row &&values) const
        {
            for (int j = 0; j < components; ++j)
            {
#ifndef USE_NS_SA_NUT_REDUCED_ORDER
                if (kind == Physics::RANS::ModelKind::SA)
                    continue;
#endif
                if (values[j] < 0)
                    values[j] = mean[j];
            }
        }

        /// Replace only the RANS point values and derivatives. Displacement
        /// is measured from the cell barycenter in the parent cell frame.
        template <class Mean, class Offset, class ReadGradient, class WriteGradient, class Row>
        DNDS_DEVICE_CALLABLE void OverlayP1(int dimension, const Mean &mean, const Offset &dx,
                                           ReadGradient gradient, WriteGradient setGradient, Row &&values) const noexcept
        {
            for (int j = 0; j < components; ++j)
            {
                real value = mean[j];
                for (int d = 0; d < dimension; ++d)
                {
                    const real slope = gradient(j, d);
                    value += slope * dx[d];
                    setGradient(j, d, slope);
                }
                values[j] = value;
            }
        }

        /// Select the interior RANS face trace before applying face-frame
        /// transforms and boundary conditions. Derivatives live separately.
        template <class Mean, class Row>
        DNDS_DEVICE_CALLABLE void ApplyFacePoint(const Mean &mean, Row &&values, bool useMean) const
        {
            if (useMean)
            {
                for (int j = 0; j < components; ++j)
                    values[j] = mean[j];
            }
            else
                RepairPoint(mean, values);
        }

        /// mean(q,j) must return the cell mean for every q. Active scalar
        /// limiting requires finite nonnegative means and finite point values.
        /// SA is untouched unless USE_NS_SA_NUT_REDUCED_ORDER is enabled.
        /// The 0.9 safety factor applies only when theta < 1; tiny positive
        /// constant states therefore remain unchanged. Flow beta is independent.
        template <class Mean, class Higher>
        DNDS_DEVICE_CALLABLE real Compute(const Mean &mean, const Higher &higher, index first, index begin, index end) const
        {
            if (!components || begin == end)
                return 1;
#ifndef USE_NS_SA_NUT_REDUCED_ORDER
            if (kind == Physics::RANS::ModelKind::SA)
                return 1;
#endif
            real theta = 1;
            for (int j = 0; j < components; ++j)
            {
                const real value = mean(begin, first + j);
                DNDS_assert(std::isfinite(value) && value >= 0);
                for (index q = begin; q < end; ++q)
                {
                    if (higher(q, first + j) < 0)
                        theta = std::min(theta, value / (value - higher(q, first + j) + verySmallReal));
                }
            }
            return theta < 1 ? .9 * theta : 1;
        }
    };

    /// Scalar reconstruction bounds depend only on the selected model.
    /// Prescribed reference states remain relevant to mean repair, not this
    /// zero-floor polynomial correction.
    RANSReconstructionView ResolveRANSReconstructionPolicy(const EquationSystem &equation);
}
