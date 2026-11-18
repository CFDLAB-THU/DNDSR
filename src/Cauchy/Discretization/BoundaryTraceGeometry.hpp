#pragma once

#include "Cauchy/Physics/FlowState.hpp"
#include "Cauchy/Boundary/BoundaryInputs.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"

namespace DNDS::Cauchy
{
    /// Transient boundary-point arithmetic. Operators derive it from one
    /// compact face range and the retained CFV publication.
    struct BoundaryTraceGeometry
    {
        /// Global CFV quadrature row, or -1 for a face-center value.
        index cell = 0, face = 0, point = 0, condition = 0, interfaceRow = 0;
        int side = 0;
        Vector3 normal{}, location{};
        /// Physical quadrature weight, including the axisymmetric metric.
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
