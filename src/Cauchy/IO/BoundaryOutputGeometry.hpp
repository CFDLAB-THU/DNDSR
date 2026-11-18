#pragma once

#include "Cauchy/Physics/FlowState.hpp"
#include "DNDS/Runtime/Array/Array.hpp"

namespace DNDS::Cauchy
{
    /// A row of the extracted boundary mesh, expressed in the source FV order.
    struct BoundaryOutputRow
    {
        index face, cell, auxiliary, boundaryBegin, quadratureBegin, count;
        Vector3 normal;
        real area;
        int zone;
        bool centerNormal;
    };
    static_assert(std::is_trivially_copyable_v<BoundaryOutputRow>);

    namespace BoundaryOutputDetail
    {
        namespace R = Runtime;
        using Rows = R::ArrayView<const BoundaryOutputRow, R::AccessBoundsChecks, R::FixedRows<1>>;
        using Values = R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows>;
        using Momentum = R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<3>>;
        using Output = R::ArrayView<real, R::AccessBoundsChecks, R::TableRows>;

        /// Pack the integrated RHS flux and the signed tangential projection.
        /// The latter is projected at every numerical quadrature point before
        /// integration. Negative outward flux matches Euler's F and FT.
        DNDS_DEVICE_CALLABLE inline void GatherFace(BoundaryOutputRow row, Values flux, Momentum momentum,
                                                    Values quadrature, Output output, index iRow)
        {
            const index components = flux.Width();
            for (index j = 0; j < components + 3; ++j)
                output(iRow, j) = 0;
            if (row.area == 0)
                return; // Axisymmetric faces on the axis carry zero integrated flux.
            for (index j = 0; j < components; ++j)
                output(iRow, j) = flux(row.auxiliary, j) / row.area;
            Vector3 tangent{};
            for (index iPoint = 0; iPoint < row.count; ++iPoint)
            {
                Vector3 n = row.normal;
                if (!row.centerNormal)
                    for (int d = 0; d < 3; ++d)
                        n[d] = quadrature(row.quadratureBegin + iPoint, 3 + d);
                const auto weighted = momentum[row.boundaryBegin + iPoint];
                real normalMomentum = 0;
                for (int d = 0; d < 3; ++d)
                    normalMomentum += weighted[d] * n[d];
                for (int d = 0; d < 3; ++d)
                    tangent[d] -= weighted[d] - normalMomentum * n[d];
            }
            for (int d = 0; d < 3; ++d)
                output(iRow, components + d) = tangent[d] / row.area;
        }

#ifdef DNDS_USE_CUDA
        void GatherDeviceFaces(R::ExecutionContext, Rows, Values, Momentum, Values, Output);
#endif
    }
}
