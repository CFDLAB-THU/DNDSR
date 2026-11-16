#pragma once

#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    /// Common weighted point leaf and explicit CTA-local shared staging; the
    /// submitted operation retains geometry, packages and all changing fields.
    void LaunchSegmentedFluxFaces(Runtime::ExecutionContext, const RHSGeometryPublication &,
                                  Runtime::ArrayView<const RHSKernelParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>,
                                  RHSKernelControls, RHSFields, RHSFaceOutputs, RHSIndices, RHSFacePackageView);
}
