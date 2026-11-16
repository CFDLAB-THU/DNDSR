#pragma once

#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Discretization/CUDA/ModalPositivityKernels.hpp"

namespace DNDS::Cauchy::CUDA
{
    using GradientInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using GradientOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using GradientMeans = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using GradientIndices = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using GradientStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    struct GradientLimiterGeometry
    {
        Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> stencil;
        index ownedCells = 0, allCells = 0;
    };
    /// Independent Barth limiting of physical gradients. Writes a disjoint
    /// output field; equation momentum is rotated on periodic neighbor reads.
    /// Core selections need no ghost view. Boundary values do not enter the
    /// Barth extrema, matching the current generic CPU operator.
    void LimitPhysicalGradients(Runtime::ExecutionContext, GradientLimiterGeometry,
                                GradientMeans, GradientInput source, GradientOutput destination,
                                GradientIndices selected, int dimension, bool barth, bool requireGhost = true);

    struct GradientPositivityParameters
    {
        ThermalView bounds{};
        RANSReconstructionView rans;
        int dimension = 0, betaPower = 1;
        real betaCutoff = 0;
        ReconstructionClippedReferencePolicy clippedPolicy = ReconstructionClippedReferencePolicy::FallbackToBase;
    };
    /** Cauchy physical-admissibility correction of already limited gradients.
     * PP POLICY -- common flow5 beta, independent RANS scalar correction, then
     * the selected whole-flow endpoint recovery. No mean increment is repaired.
     * Point samples are evaluated from bounded per-cell gradient scratch, so
     * the number of quadrature points imposes no local-memory array bound.
     * The producing CTA writes one status/count/minimum summary; a compact
     * finalizer merges summaries without rescanning cell fields.
     */
    void LimitPositivePhysicalGradients(Runtime::ExecutionContext, GradientLimiterGeometry,
                                        GradientInput means, GradientOutput gradients, GradientIndices selected,
                                        GradientPositivityParameters,
                                        ModalPositivitySummary partials);
    index GradientPositivitySummaryRows(index selectedCells);
    /// Mean [rho,u,v,w,T] preparation, with explicit point failure status.
    /// Input may be any selected-profile combined AoS row; output has width5.
    void FillPrimitiveMeans(Runtime::ExecutionContext, GradientInput input, GradientOutput output,
                            PerfectGas gas, GradientStatus status);
}
