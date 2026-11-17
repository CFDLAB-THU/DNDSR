#pragma once

#include "Cauchy/Boundary/ACMBoundary.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Cauchy/Implicit/CUDA/LinearBlockKernels.hpp"
#include "DNDS/Core/NumericalStatus.hpp"
#include "Physics/RANS/Point.hpp"

namespace DNDS::Cauchy::CUDA
{
    struct ACMJacobianInput
    {
        struct Means
        {
            LinearReadRows owned, ghost;
            DNDS_DEVICE_CALLABLE real operator()(index row, int j) const
            {
                return row < owned.RowCount() ? owned(row, j) : ghost(row - owned.RowCount(), j);
            }
        } input;
        Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> stencil;
        struct BoundaryCollection
        {
            Runtime::ArrayView<const ACM::BoundaryParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> rows;
            DNDS_DEVICE_CALLABLE const ACM::BoundaryParameters &operator[](index i) const { return rows(i, 0); }
        } conditions;
        ACM::Parameters physics;
        LinearReadRows tau, radius;
        decltype(WallDistanceView::cellMean) wallMean;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> lengths;
        Physics::RANS::PointParameters rans;
        MatrixWriteRows diagonal;
        Algebra::BlockMatrixArrayView<real> offDiagonal;
        real mass, residualScale;
        std::array<bool, 6> active;
    };
    using ACMJacobianStatusRows = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    /// One independent thread assembles one complete owned sparse row in its
    /// physical component frame. Every operand is resident through completion.
    void PrepareACMJacobian(Runtime::ExecutionContext, const ACMJacobianInput &, ACMJacobianStatusRows);
}
