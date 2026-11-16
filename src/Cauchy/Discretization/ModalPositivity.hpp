#pragma once

#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"

namespace DNDS::Cauchy
{
    /** @brief Borrowed point-local modal PP geometry.
     * @details Faces use canonical incidences and CFV's parent-side basis bank;
     * volumes use the one source/PP publication. No query IDs, scatter map or
     * full-domain sampled-state allocation is part of this interface. The
     * enclosing operation retains every allocation and its source generation.
     */
    struct ModalPositivityGeometry
    {
        Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> stencil;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> volumeOffsets;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> volumePoints;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> volumeBasis;
        CFV::ReconstructionPointBasisView basis;
        int volumeCachedDerivatives = 0;
        index maximumCellPoints = 0;

        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, stencil, volumeOffsets, volumePoints, volumeBasis, basis);
        }
        DNDS_DEVICE_CALLABLE index PointNumber(index iCell) const
        {
            const auto &mesh = stencil(0, 0);
            index count = volumeOffsets(iCell + 1, 0) - volumeOffsets(iCell, 0);
            for (const auto iFace : mesh.topology.CellFaces(iCell))
                count += mesh.metric.FaceQuadratureEnd(iFace) - mesh.metric.FaceQuadratureBegin(iFace);
            return count;
        }
        struct PointEvaluation
        {
            CFV::Math::TaylorPointEvaluation prepared;
            index row = 0;
            // 0=direct, 1=face cache, 2=volume cache.
            int cache = 0;
        };
        /// Prepare once per point, never once per component or mode. Cached
        /// queries need neither polynomial coordinates nor a local basis array.
        DNDS_DEVICE_CALLABLE PointEvaluation PreparePoint(index iCell, index iPoint) const noexcept
        {
            const auto &mesh = stencil(0, 0);
            const auto faces = mesh.topology.CellFaces(iCell);
            index ordinal = iPoint;
            for (index slot = 0; slot < faces.size(); ++slot)
            {
                const index iFace = faces[slot];
                const index begin = mesh.metric.FaceQuadratureBegin(iFace);
                const index count = mesh.metric.FaceQuadratureEnd(iFace) - begin;
                if (ordinal < count)
                {
                    const int side = mesh.topology.CellIsFaceBack(iCell, iFace, rowsize(slot)) ? 0 : 1;
                    const index q = begin + ordinal;
                    if (basis.cachedDerivatives)
                        return {{}, 2 * q + side, 1};
                    const auto point = mesh.metric.periodic.FacePoint(iFace, side, mesh.metric.GetFacePoint(q));
                    return {basis.PreparePoint(iCell, {point[0], point[1], point[2]}), 0, 0};
                }
                ordinal -= count;
            }
            const index q = volumeOffsets(iCell, 0) + ordinal;
            DNDS_HD_assert(q < volumeOffsets(iCell + 1, 0));
            if (volumeCachedDerivatives)
                return {{}, q, 2};
            return {basis.PreparePoint(iCell, {volumePoints(q, 0), volumePoints(q, 1), volumePoints(q, 2)}), 0, 0};
        }
        DNDS_DEVICE_CALLABLE real BasisValue(index iCell, const PointEvaluation &point, int mode) const noexcept
        {
            if (point.cache == 1)
                return basis.cached(point.row, mode * basis.cachedDerivatives);
            if (point.cache == 2)
                return volumeBasis(point.row, mode * volumeCachedDerivatives);
            return basis.EvaluateAtPoint(iCell, point.prepared, 0, mode);
        }
        /// CPU callers can stage the small mode vector while GPU point lanes
        /// accumulate their statically bounded component registers directly.
        DNDS_DEVICE_CALLABLE void PointBasis(index iCell, index iPoint, SpanView<real> values) const noexcept
        {
            const auto point = PreparePoint(iCell, iPoint);
            for (int mode = 0; mode < basis.polynomial.size - 1; ++mode)
                values[mode] = BasisValue(iCell, point, mode);
        }
    };

    /// Frozen mathematical policy shared by CPU and device modal PP leaves.
    /// The mode endpoints are prepared from the canonical Taylor ordering.
    struct ModalPositivityPolicy
    {
        ThermalView bounds{};
        RANSReconstructionView rans{};
        index ransFirst = -1;
        int components = 0, modes = 0, maxOrder = 0, betaPower = 1;
        std::array<int, 4> modeEnd{};
        real betaCutoff = 0;
        bool meanBase = true, segment = false;
        ThermalBounds segmentBounds{};
        ReconstructionClippedReferencePolicy clippedPolicy = ReconstructionClippedReferencePolicy::FallbackToBase;
    };
    static_assert(std::is_trivially_copyable_v<ModalPositivityGeometry>);
    static_assert(std::is_trivially_copyable_v<ModalPositivityPolicy>);
    struct ModalPositivityStatistics
    {
        index limited = 0;
        real minimum = 1;
    };
    struct ModalPositivityCellResult
    {
        NumericalStatus status{};
        int degree = -1;
        real beta = 1;
        bool changed = false;
    };

    namespace ModalPositivityDetail
    {
        struct PointRows
        {
            SpanView<real> values;
            index components;
            DNDS_DEVICE_CALLABLE real &operator()(index q, index j) const { return values[q * components + j]; }
        };
        struct MeanRows
        {
            SpanView<const real> values;
            DNDS_DEVICE_CALLABLE real operator()(index, index j) const { return values[j]; }
        };
        DNDS_DEVICE_CALLABLE inline bool ReferenceGood(PointRows samples, index points, const ThermalView &bounds) noexcept
        {
            for (index q = 0; q < points; ++q)
            {
                const ThermalAdmissibilityDetail::PointRow<PointRows> point{samples, q};
                if (!bounds.RawThermallyAdmissible(point) || !bounds.ClippedReferenceAdmissible(point))
                    return false;
            }
            return true;
        }
        DNDS_DEVICE_CALLABLE inline bool StoredGood(PointRows samples, index points, const ThermalView &bounds) noexcept
        {
            for (index q = 0; q < points; ++q)
                if (!bounds.ClippedReferenceAdmissible(ThermalAdmissibilityDetail::PointRow<PointRows>{samples, q}))
                    return false;
            return true;
        }
        /// A point is evaluated once into cell-local scratch. Reuse that basis
        /// across all components; no per-component polynomial power rebuilding.
        template <class Output>
        DNDS_DEVICE_CALLABLE inline void SamplePoint(const ModalPositivityGeometry &geometry, index iCell, index q,
                                                     SpanView<const real> mean, SpanView<const real> coefficients, int modes, Output &&output) noexcept
        {
#ifdef __CUDA_ARCH__
            // The supported device profiles have 5/6/7 components. Constant
            // unroll bounds keep these accumulators in registers; do not hide
            // a per-point sampled-state field in local-memory array spills.
            DNDS_HD_assert(mean.size() <= 7);
            real values[7]{};
#    pragma unroll
            for (int j = 0; j < 7; ++j)
                if (j < mean.size())
                    values[j] = mean[j];
            const auto point = geometry.PreparePoint(iCell, q);
            for (int mode = 0; mode < modes; ++mode)
            {
                const real value = geometry.BasisValue(iCell, point, mode);
#    pragma unroll
                for (int j = 0; j < 7; ++j)
                    if (j < mean.size())
                        values[j] += coefficients[j * modes + mode] * value;
            }
#    pragma unroll
            for (int j = 0; j < 7; ++j)
                if (j < mean.size())
                    output(q, j) = values[j];
#else
            real basisValues[19];
            geometry.PointBasis(iCell, q, {basisValues, modes});
            for (index j = 0; j < mean.size(); ++j)
            {
                real value = mean[j];
                for (int mode = 0; mode < modes; ++mode)
                    value += coefficients[j * modes + mode] * basisValues[mode];
                output(q, j) = value;
            }
#endif
        }
        struct Sampler
        {
            const ModalPositivityGeometry &geometry;
            index cell, points;
            SpanView<const real> mean;
            int modes;
            DNDS_DEVICE_CALLABLE void operator()(SpanView<const real> coefficients, PointRows output) const noexcept
            {
                for (index q = 0; q < points; ++q)
                    SamplePoint(geometry, cell, q, mean, coefficients, modes, output);
            }
        };

        /** @brief Select the accepted base and compress a staged cell polynomial.
         * @details PP POLICY: raw polynomial species are retained for transport;
         * beta locks flow/species and does not impose a simplex condition. The
         * reference supports both raw-affine beta and clipped EOS/T evaluation.
         * RANS receives its independent scalar compression from the original
         * polynomial. Every caller must subsequently publish, resample actual
         * storage, and run the clipped-reference endpoint check/recovery below.
         *
         * Lower-base selection intentionally preserves Euler/Cauchy's existing
         * sequence: each rejected truncation becomes the next higher candidate.
         * It is not always a blend from the original highest-order polynomial.
         */
        template <class Sample>
        DNDS_DEVICE_CALLABLE ModalPositivityCellResult SelectAndCompress(
            const ModalPositivityPolicy &p, SpanView<const real> mean,
            SpanView<real> coefficients, SpanView<real> reference,
            PointRows higher, PointRows lower, index points, const Sample &sample) noexcept
        {
            ModalPositivityCellResult result;
            if (!p.bounds.Admissible(mean))
            {
                result.status = NumericalStatus{NumericalError::InvalidState, 1};
                return result;
            }
            for (index q = 0; q < points; ++q)
                for (int j = 0; j < p.components; ++j)
                    if (!std::isfinite(higher(q, j)))
                    {
                        result.status = NumericalStatus{NumericalError::InvalidState, 2, -1, q};
                        return result;
                    }
            const bool flowGood = ReferenceGood(higher, points, p.bounds);
            result.degree = flowGood ? p.maxOrder : -1;
            const real ransScale = p.rans.Compute(MeanRows{mean}, higher, p.ransFirst, 0, points);
            if (!(std::isfinite(ransScale) && ransScale >= 0 && ransScale <= 1))
            {
                result.status = NumericalStatus{NumericalError::InvalidState, 3};
                return result;
            }
            if (flowGood && ransScale == 1 && !p.segment)
                return result;
            if (!p.meanBase)
            {
                for (index j = 0; j < coefficients.size(); ++j)
                    reference[j] = coefficients[j];
                for (int order = p.maxOrder - 1; order >= 0 && result.degree < 0; --order)
                {
                    for (int j = 0; j < p.components; ++j)
                        if (p.bounds.CompressesReconstructionComponent(j))
                            for (int mode = p.modeEnd[order]; mode < p.modes; ++mode)
                                reference[j * p.modes + mode] = 0;
                    sample(reference, lower);
                    if (ReferenceGood(lower, points, p.bounds))
                        result.degree = order;
                    else
                    {
                        for (int j = 0; j < p.components; ++j)
                            if (p.bounds.CompressesReconstructionComponent(j))
                                for (int mode = 0; mode < p.modes; ++mode)
                                    coefficients[j * p.modes + mode] = reference[j * p.modes + mode];
                        result.changed = true;
                    }
                }
                // A mean base is guaranteed by the independently checked mean.
                if (result.degree < 0)
                {
                    result.status = NumericalStatus{NumericalError::InvalidState, 4};
                    return result;
                }
                sample(coefficients, higher);
            }
            if (ransScale < 1)
            {
                for (int j = 0; j < p.rans.components; ++j)
                    for (int mode = 0; mode < p.modes; ++mode)
                        coefficients[(p.ransFirst + j) * p.modes + mode] *= ransScale;
                result.changed = true;
            }
            if (result.degree == p.maxOrder)
                return result;
            if (p.segment)
            {
                const auto flow = p.bounds.Flow(mean);
                for (index q = 0; q < points; ++q)
                    result.beta = std::min(result.beta, AdmissibleFraction(flow,
                                                                           p.bounds.Flow(ThermalAdmissibilityDetail::PointRow<PointRows>{higher, q}), p.segmentBounds));
                if (result.beta < p.betaCutoff)
                    result.beta = 0;
            }
            else
                result.beta = p.meanBase ? p.bounds.ReconstructionFraction(mean, MeanRows{mean}, higher, 0, points,
                                                                           p.betaPower, p.betaCutoff, 1, p.clippedPolicy)
                                         : p.bounds.ReconstructionFraction(mean, lower, higher, 0, points,
                                                                           p.betaPower, p.betaCutoff, 1, p.clippedPolicy);
            if (!(std::isfinite(result.beta) && result.beta >= 0 && result.beta <= 1))
            {
                result.status = NumericalStatus{NumericalError::InvalidState, 5};
                return result;
            }
            if (result.beta < 1)
            {
                for (int j = 0; j < p.components; ++j)
                    if (p.bounds.CompressesReconstructionComponent(j))
                        for (int mode = 0; mode < p.modes; ++mode)
                        {
                            const index k = j * p.modes + mode;
                            const real base = p.meanBase ? 0 : reference[k];
                            coefficients[k] = base + result.beta * (coefficients[k] - base);
                        }
                result.changed = true;
            }
            return result;
        }
        /** @brief Last-resort recovery after sampling the stored coefficients.
         * @details The accepted invariant is the clipped-composition EOS/T bound,
         * not a species-simplex or raw-affine invariant. Species coefficients stay
         * unprojected. Only the locked flow/species modes are moved toward the
         * already accepted base; RANS remains independently limited. A zero beta
         * that still fails is a numerical error, never silent acceptance.
         */
        DNDS_DEVICE_CALLABLE inline NumericalStatus Recover(
            const ModalPositivityPolicy &p, ModalPositivityCellResult &result,
            SpanView<real> coefficients, SpanView<const real> reference) noexcept
        {
            const real previous = result.beta;
            if (!(previous > 0))
                return NumericalStatus{NumericalError::InvalidState, 6};
            result.beta = p.clippedPolicy == ReconstructionClippedReferencePolicy::FallbackToBase ? 0 : previous * .9;
            if (result.beta < p.betaCutoff)
                result.beta = 0;
            const real scale = result.beta / previous;
            for (int j = 0; j < p.components; ++j)
                if (p.bounds.CompressesReconstructionComponent(j))
                    for (int mode = 0; mode < p.modes; ++mode)
                    {
                        const index k = j * p.modes + mode;
                        const real base = p.meanBase ? 0 : reference[k];
                        coefficients[k] = base + scale * (coefficients[k] - base);
                    }
            result.changed = true;
            return {};
        }
    }
}
