#pragma once

#include "Cauchy/Boundary/PerfectGasBoundary.hpp"
#include "Cauchy/Boundary/ACMBoundary.hpp"
#include "Cauchy/Discretization/ACMKernelPoint.hpp"
#include "Cauchy/Discretization/BoundaryTraceGeometry.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy::CUDA
{
    /// Setup-resolved small parameters, kept in one resident metadata row.
    /// Perfect gas has no species-energy arrays: thermal's spans are empty.
    /// Time/frame belong to the per-call request, so repeated boundary updates
    /// do not overwrite this immutable publication or require a metadata copy.
    struct BoundaryKernelParameters
    {
        PerfectGas gas;
        Physics::RANS::ModelKind rans = Physics::RANS::ModelKind::None;
        ThermalView thermal{};
        bool allowPointFallback = false;
        void Validate(int components) const;
    };
    static_assert(std::is_trivially_copyable_v<BoundaryKernelParameters>);
    using BoundaryInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using BoundaryOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using BoundaryRows = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using BoundaryStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    struct BoundaryTraceRequest
    {
        Runtime::ArrayView<const BoundaryKernelParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> parameters;
        Runtime::ArrayView<const PerfectGasBoundaryParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> conditions;
        Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> stencil;
        Runtime::ArrayView<const BoundaryFaceRange, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faces;
        BoundaryInput means, coefficients, incoming;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> wallMean;
        CFV::ReconstructionPointBasisView basis;
        BoundaryOutput output;
        BoundaryStatus status;
        RigidFrameParameters frame;
        real time = 0;
        int components = 5, modes = 0;
        bool centers = false, polynomial = false, exterior = false, hasIncoming = false, repairRANS = true;
        bool forcePointFallback = false;
        Runtime::ArrayView<const ACM::SpatialParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> acm;
        Runtime::ArrayView<const ACM::BoundaryParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> acmBC;
    };

    /** Host launchers receive borrowed, device-tagged array views. Operators
     * retain all owners/leases and order producers on context.stream. No host
     * wait, staging, allocation or boundary callback occurs here. Status has
     * one record per output row; inspect it before publishing failed outputs.
     * Reconstruction targets and interface packets use the same output table.
     */
    void FillBoundaryTraces(Runtime::ExecutionContext context, const BoundaryTraceRequest &request);
    /// Mean-based primitive [rho,u,v,w,T] targets for gradient preparation.
    /// Physical faces evaluate the selected center trace with a temporary left
    /// state; interface rows use the independently sampled center packet.
    /// Polynomial input is unused.
    void FillPrimitiveBoundary(Runtime::ExecutionContext context, const BoundaryTraceRequest &request);
    void CopyBoundaryRows(Runtime::ExecutionContext context, BoundaryInput source, BoundaryOutput output, BoundaryRows rows);
}
