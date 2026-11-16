#pragma once
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Boundary/BoundaryInputs.hpp"

namespace DNDS::Cauchy::CUDA
{
    struct ForceGeometry
    {
        Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> stencil;
        Runtime::ArrayView<const BoundaryFaceRange, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faces;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, stencil, faces);
        }
    };
    using ForceInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>>;
    using ForceOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>>;
    using ForceCounter = Runtime::ArrayView<unsigned int, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    index BoundaryForceScratchRows(index points);
    /// One fused fixed-tree reduction, including a last-block tail. Views and
    /// selected-zone metadata must be resident; output/counter/partials distinct.
    /// Setup guarantees compact face ranges select the matching momentum rows.
    /// Center-only consumers share their single center row. The wrapper checks shape/residency once, not per point.
    void AccumulateBoundaryForce(Runtime::ExecutionContext, ForceGeometry, ForceInput,
                                 SpanView<const int> zones, ForceOutput partial, ForceOutput result, ForceCounter);
}
