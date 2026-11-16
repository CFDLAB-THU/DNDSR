#pragma once

#include "Cauchy/Physics/FlowState.hpp"
#include "Cauchy/Boundary/BoundaryInputs.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"

namespace DNDS::Cauchy
{
    /// Transient boundary-point arithmetic, or an explicitly requested output
    /// snapshot. Production operators derive it from one compact face range
    /// and the retained CFV publication; no persistent trace table is required.
    struct BoundaryTraceGeometry
    {
        /// Numerical transients use the global CFV quadrature row; centers
        /// use -1. BoundaryTraceSnapshot is an output-only record and uses
        /// the local face-quadrature ordinal instead. Numerical launchers take
        /// the compact stencil/ranges, never a snapshot Array of these records.
        index cell = 0, face = 0, point = 0, condition = 0, interfaceRow = 0;
        int side = 0;
        Vector3 normal{}, location{};
        /// Physical quadrature weight, including the configured axisymmetric
        /// metric. Output consumes this prepared value without rebuilding or
        /// downloading the complete volume/face metric publication.
        real weight = 0;
    };
    static_assert(std::is_trivially_copyable_v<BoundaryTraceGeometry>);

    /// Materialize one requested location. Centers are independently evaluated
    /// and use the center packet row, never the first quadrature packet row.
    DNDS_DEVICE_CALLABLE inline BoundaryTraceGeometry BoundaryTracePoint(
        const FiniteVolumeStencilView &stencil, const BoundaryFaceRange &range,
        index iBoundaryFace, index iLocalPoint, bool centers)
    {
        const index iPoint = stencil.metric.FaceQuadratureBegin(range.face) + iLocalPoint;
        const auto sample = stencil.CanonicalFaceSample(range.face, iPoint,
                                                        centers ? FaceEvaluationLocation::Center : FaceEvaluationLocation::Quadrature);
        BoundaryTraceGeometry result;
        result.cell = sample.cell;
        result.face = range.face;
        result.point = centers ? -1 : iPoint;
        result.condition = sample.condition;
        result.interfaceRow = centers ? iBoundaryFace : range.quadratureBegin + iLocalPoint;
        result.side = 0;
        result.normal = sample.normal;
        result.location = sample.ownPoint;
        result.weight = centers ? stencil.metric.GetFaceArea(range.face) : stencil.metric.GetFaceWeight(iPoint);
        return result;
    }
}
