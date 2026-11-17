#pragma once

#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    template <int Components, class Fields>
    DNDS_DEVICE_CALLABLE void MeanPoint(const Fields &fields, index iCell, PerfectGasKernelPoint<Components> &point) noexcept
    {
        // Share the auxiliary/RHS mean loader. Gradients are explicitly absent
        // here; reconstructed point gradients are filled below, not borrowed
        // from a previous auxiliary calculation.
        point.Load(iCell, fields.mean, fields.mean, 0, false);
    }
    /// The sample's other center is already a periodic coordinate image in
    /// the anchor frame. Undo only the vector map, then add the native center.
    DNDS_DEVICE_CALLABLE inline Vector3 ParentPoint(const RHSGeometry &geometry, const FiniteVolumeDetail::SampleRecord &sample, bool other) noexcept
    {
        if (!other)
            return sample.ownPoint;
        Vector3 result{}, delta{};
        for (int d = 0; d < 3; ++d)
            delta[d] = sample.ownPoint[d] - sample.otherPoint[d];
        for (int d = 0; d < 3; ++d)
        {
            if (sample.changesFrame)
                for (int k = 0; k < 3; ++k)
                    result[d] += sample.rotation[k + 3 * d] * delta[k];
            else
                result[d] = delta[d];
            result[d] += geometry.cells(sample.other, 1 + d);
        }
        return result;
    }
    template <int Components, class PolynomialScalar = real, class Fields>
    DNDS_DEVICE_CALLABLE void ReconstructPoint(const RHSGeometry &geometry, const RHSKernelParameters &parameters,
                                               const RHSKernelControls &controls, const Fields &fields,
                                               const FiniteVolumeDetail::SampleRecord &sample, bool other,
                                               PerfectGasKernelPoint<Components> &point, index &fallbacks) noexcept
    {
        const index iCell = other ? sample.other : sample.cell;
        MeanPoint(fields, iCell, point);
        const int dimension = Fields::static_dimension ? Fields::static_dimension : controls.dimension;
        if (!controls.firstOrder && (!controls.direct || controls.linear))
        {
            using S = PolynomialScalar;
            std::array<S, Components> values{};
            std::array<std::array<S, 3>, Components> gradients{};
            for (int j = 0; j < Components; ++j)
                values[j] = DNDS::Math::ScalarCast<S>(point.values[j]);
            const auto position = ParentPoint(geometry, sample, other);
            const int modes = Fields::separate_roles || Fields::static_modes ? Fields::static_modes : controls.modes;
            const bool direct = Fields::separate_roles ? Fields::static_modes == 0 : controls.direct;
            // These operands can have different static extents (primitive
            // gradients versus modal coefficients). Select scalar values,
            // without decaying either typed view back to a dynamic policy.
            const auto coefficient = [&](int j)
            {
                if constexpr (Fields::separate_roles)
                {
                    if constexpr (Fields::static_modes == 0)
                        return fields.gradients(iCell, j);
                    else
                        return fields.coefficients(iCell, j);
                }
                else
                    return fields.reconstruction(iCell, j);
            };
            const auto viscous = [&](int j)
            {
                if constexpr (Fields::separate_roles)
                {
                    if constexpr (Fields::static_modes == 0)
                        return controls.distinctViscous ? fields.viscousGradients(iCell, j) : fields.gradients(iCell, j);
                    else
                        return controls.distinctViscous ? fields.viscousCoefficients(iCell, j) : fields.coefficients(iCell, j);
                }
                else
                    return controls.distinctViscous && !controls.primitiveGradient ? fields.viscousData(iCell, j) : fields.reconstruction(iCell, j);
            };
            if (direct)
            {
                for (int j = 0; j < Components; ++j)
                    for (int d = 0; d < dimension; ++d)
                    {
                        values[j] += DNDS::Math::ScalarCast<S>(coefficient(j * dimension + d)) *
                                     (DNDS::Math::ScalarCast<S>(position[d]) - DNDS::Math::ScalarCast<S>(geometry.cells(iCell, 4 + d)));
                        if (controls.viscous && !controls.primitiveGradient)
                            gradients[j][d] = DNDS::Math::ScalarCast<S>(viscous(j * dimension + d));
                    }
            }
            else
            {
                const int derivatives = controls.viscous && !controls.primitiveGradient ? dimension + 1 : 1;
                CFV::Math::TaylorPointEvaluationT<S> prepared;
                if (derivatives > geometry.basis.cachedDerivatives)
                    prepared = geometry.basis.PreparePoint<S>(iCell, position);
                // Consume one modal basis across all equations before advancing.
                // This keeps the original per-equation summation order while
                // avoiding a 4x19 thread-local table in the hot face kernel.
                for (int mode = 0; mode < modes; ++mode)
                {
                    S basis[4];
                    for (int d = 0; d < derivatives; ++d)
                        basis[d] = geometry.basis.Evaluate(iCell, sample.point, other ? 1 - sample.side : sample.side, prepared, d, mode);
                    for (int j = 0; j < Components; ++j)
                    {
                        values[j] += DNDS::Math::ScalarCast<S>(coefficient(j * modes + mode)) * basis[0];
                        for (int d = 1; d < derivatives; ++d)
                            gradients[j][d - 1] += DNDS::Math::ScalarCast<S>(viscous(j * modes + mode)) * basis[d];
                    }
                }
            }
            for (int j = 0; j < Components; ++j)
            {
                point.values[j] = DNDS::Math::ScalarCast<real>(values[j]);
                for (int d = 0; d < dimension; ++d)
                    point.gradients[j][d] = DNDS::Math::ScalarCast<real>(gradients[j][d]);
            }
        }
        if constexpr (Components > 5)
        {
            std::array<real, Components - 5> mean;
            for (int j = 0; j < Components - 5; ++j)
                mean[j] = fields.mean(iCell, 5 + j);
            RANSReconstructionView{parameters.physics.rans.kind, Components - 5}.RepairPoint(mean, SpanView<real>{point.values.data() + 5, Components - 5});
        }
        if (controls.primitiveGradient)
            for (int j = 0; j < 5; ++j)
                for (int d = 0; d < dimension; ++d)
                    if constexpr (Fields::separate_roles)
                        point.gradients[j][d] = fields.primitiveGradients(iCell, j * dimension + d);
                    else
                        point.gradients[j][d] = fields.viscousData(iCell, j * dimension + d);
        // PP POLICY: temporary point fallback only. Raw coefficient/gradient
        // arrays are unchanged. Perfect gas has no composition to project.
        if ((controls.positivity || controls.faceFallback) &&
            !(controls.positivity ? parameters.thermal.ClippedReferenceAdmissible(point.values) : parameters.thermal.Admissible(point.values)))
        {
            ++fallbacks;
            MeanPoint(fields, iCell, point);
        }
        if (other && sample.changesFrame)
            point.Rotate(sample.rotation);
    }
    DNDS_DEVICE_CALLABLE inline NumericalResult<Riemann::AcousticState> Acoustic(const PerfectGas &gas, const FlowState &flow) noexcept
    {
        const auto point = TryPerfectGasPoint(gas, flow, false);
        if (!point)
            return point.status;
        return Riemann::AcousticState{{flow, point.value.thermo.pressure}, point.value.thermo.soundSpeed};
    }
    template <int Components, class Fields>
    DNDS_DEVICE_CALLABLE NumericalResult<Riemann::AcousticState> PointAcoustic(const RHSKernelParameters &parameters,
                                                                               const RHSKernelControls &controls, const Fields &fields,
                                                                               const FiniteVolumeDetail::SampleRecord &sample, bool other,
                                                                               PerfectGasKernelPoint<Components> &point, index &fallbacks) noexcept
    {
        auto result = Acoustic(parameters.physics.gas, point.Flow());
        if (!result && (controls.positivity || controls.faceFallback) && Physics::AllowsMeanFallback(result.status))
        {
            ++fallbacks;
            MeanPoint(fields, other ? sample.other : sample.cell, point);
            // The fallback mean lives in its own cell axes; restore the same
            // anchor frame as the successful reconstructed point, even for a
            // periodic self-neighbor with an identical local cell index.
            if (other && sample.changesFrame)
                point.Rotate(sample.rotation);
            result = Acoustic(parameters.physics.gas, point.Flow());
        }
        return result;
    }
    template <class Fields>
    DNDS_DEVICE_CALLABLE inline NumericalResult<FlowState> BoundaryFlow(const RHSGeometry &geometry, const RHSKernelParameters &parameters,
                                                                        const RHSKernelControls &controls,
                                                                        const Fields &fields, const FiniteVolumeDetail::SampleRecord &sample,
                                                                        const FlowState &inside, bool exterior) noexcept
    {
        const auto &condition = geometry.conditions(sample.condition, 0);
        if (condition.kind == BoundaryKind::Interface)
        {
            if (!fields.hasIncoming || sample.boundaryPoint < 0 || sample.boundaryPoint >= fields.incoming.RowCount())
                return NumericalStatus{NumericalError::InvalidInput, 2};
            FlowState result;
            for (int j = 0; j < 5; ++j)
                result[j] = fields.incoming(sample.boundaryPoint, j);
            return result;
        }
        return TryPerfectGasBoundaryValue(condition, parameters.physics.gas, inside, sample.normal, sample.ownPoint,
                                          controls.time, controls.frame, exterior);
    }
    template <int Components, class Fields>
    DNDS_DEVICE_CALLABLE NumericalStatus ExteriorPoint(const RHSGeometry &geometry, const RHSKernelParameters &parameters,
                                                       const RHSKernelControls &controls,
                                                       const Fields &fields, const FiniteVolumeDetail::SampleRecord &sample,
                                                       PerfectGasKernelPoint<Components> &point) noexcept
    {
        const auto &condition = geometry.conditions(sample.condition, 0);
        if (condition.kind == BoundaryKind::Interface)
        {
            if (!fields.hasIncoming || sample.boundaryPoint < 0 || sample.boundaryPoint >= fields.incoming.RowCount())
                return {NumericalError::InvalidInput, 2};
            for (int j = 0; j < Components; ++j)
                point.values[j] = fields.incoming(sample.boundaryPoint, j);
            return {};
        }
        std::array<real, Components> mean;
        for (int j = 0; j < Components; ++j)
            mean[j] = fields.mean(sample.cell, j);
        return TryPerfectGasBoundaryState<Components>(condition, parameters.physics.gas, parameters.physics.rans.kind, point.values, mean,
                                                      geometry.hasWall ? geometry.wallMean(sample.cell, 0) : 0, sample.normal, sample.ownPoint, controls.time, controls.frame, true);
    }
}
