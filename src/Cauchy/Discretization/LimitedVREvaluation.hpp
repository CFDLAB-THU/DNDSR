#pragma once

#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"

namespace DNDS::Cauchy
{
    /** @brief Affine evaluation of an accepted LVR target in one parent frame.
     * @details GradientPlan converts physical gradients to normalized linear
     * Taylor coefficients. Their zero-mean moment correction makes the mean
     * origin the cell barycenter, including on curved cells. Only the final
     * limited linear modes are used here; higher VR modes are not a second
     * sampling state. The caller retains all views for this numerical leaf.
     */
    template <class Means, class Coefficients>
    DNDS_DEVICE_CALLABLE FlowState EvaluateLVRLinearTarget(
        const FiniteVolumeStencilView &stencil, Means means, Coefficients coefficients,
        index iCell, const Geom::tPoint &point, int dimension, int modes) noexcept
    {
        FlowState value{};
        const Geom::tPoint delta = (point - stencil.metric.GetCellBary(iCell)) / stencil.metric.cells(iCell, 7);
        for (int j = 0; j < 5; ++j)
        {
            value[j] = means(iCell, j);
            for (int d = 0; d < dimension; ++d)
                value[j] += coefficients(iCell, j * modes + d) * delta[d];
        }
        return value;
    }

    namespace LimitedVRDetail
    {
        struct MeanValues
        {
            FlowState mean;
            DNDS_DEVICE_CALLABLE real operator()(index, int j) const noexcept { return mean[j]; }
        };

        /// One-cell bounded cache. No point-state field or query registry is
        /// allocated: a point's displacement is reused across its components.
        template <class Means, class Coefficients>
        struct CenterValues
        {
            const FiniteVolumeStencilView &stencil;
            Means means;
            Coefficients coefficients;
            index cell;
            int dimension, modes;
            mutable index cachedSlot = -1;
            mutable FlowState value{};
            DNDS_DEVICE_CALLABLE real operator()(index slot, int j) const noexcept
            {
                if (slot != cachedSlot)
                {
                    const index iFace = stencil.topology.CellFaces(cell)[slot];
                    const int side = stencil.topology.CellIsFaceBack(cell, iFace, rowsize(slot)) ? 0 : 1;
                    const auto point = stencil.metric.periodic.FacePoint(iFace, side, stencil.metric.GetFaceCent(iFace));
                    value = EvaluateLVRLinearTarget(stencil, means, coefficients, cell, point, dimension, modes);
                    cachedSlot = slot;
                }
                return value[j];
            }
            DNDS_DEVICE_CALLABLE void Invalidate() const noexcept { cachedSlot = -1; }
        };
    }

    /** @brief Mean-based flow5 PP on the selected-gradient LVR target.
     * @details This target is checked at face centers, independently of the
     * high-order final reconstruction's face/source-volume point set. Retain
     * Euler's common density/internal-energy beta and its power/safety factor.
     * Update actual coefficient storage, then verify the stored affine state;
     * any roundoff recovery is local to this cell and scales the locked flow5
     * target. Neither the mean nor a received interface packet is modified.
     */
    template <class Means, class Coefficients>
    DNDS_DEVICE_CALLABLE NumericalStatus LimitLVRTargetCell(
        const FiniteVolumeStencilView &stencil, Means means, Coefficients coefficients,
        index iCell, int dimension, int modes, const ThermalBounds &bounds, int betaPower, real betaCutoff) noexcept
    {
        FlowState mean{};
        for (int j = 0; j < 5; ++j)
            mean[j] = means(iCell, j);
        if (!ThermallyAdmissible(mean, bounds))
            return NumericalStatus{NumericalError::InvalidState, 2}.At(iCell);
        const index count = stencil.topology.CellFaces(iCell).size();
        const LimitedVRDetail::CenterValues<Means, Coefficients> points{stencil, means, coefficients, iCell, dimension, modes};
        for (index slot = 0; slot < count; ++slot)
            for (int j = 0; j < 5; ++j)
                if (!std::isfinite(points(slot, j)))
                    return NumericalStatus{NumericalError::InvalidState, 1}.At(iCell, slot);
        real beta = ReconstructionFraction(mean, LimitedVRDetail::MeanValues{mean}, points,
                                           0, count, bounds, betaPower, betaCutoff);
        if (!(std::isfinite(beta) && beta >= 0 && beta <= 1))
            return NumericalStatus{NumericalError::InvalidState, 3}.At(iCell);
        for (int j = 0; j < 5; ++j)
            for (int mode = 0; mode < modes; ++mode)
                coefficients(iCell, j * modes + mode) *= beta;
        const auto accepted = [&]()
        {
            points.Invalidate();
            for (index slot = 0; slot < count; ++slot)
            {
                FlowState value{};
                for (int j = 0; j < 5; ++j)
                    value[j] = points(slot, j);
                if (!ThermallyAdmissible(value, bounds))
                    return false;
            }
            return true;
        };
        for (int attempt = 0; !accepted() && attempt < 1000; ++attempt)
        {
            const real next = beta * .9 < betaCutoff ? 0 : beta * .9;
            const real factor = beta > 0 ? next / beta : 0;
            for (int j = 0; j < 5; ++j)
                for (int mode = 0; mode < modes; ++mode)
                    coefficients(iCell, j * modes + mode) *= factor;
            beta = next;
        }
        return accepted() ? NumericalStatus{} : NumericalStatus{NumericalError::InvalidState, 4}.At(iCell);
    }
}
