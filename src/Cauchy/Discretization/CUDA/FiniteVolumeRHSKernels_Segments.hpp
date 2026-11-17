#pragma once

#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    /// Validate setup-prepared packages once and return the launch grid. Shape
    /// instantiation TUs share this host contract; no numerical dispatch here.
    unsigned ValidateSegmentedFluxFaces(Runtime::ExecutionContext, const RHSGeometryPublication &,
                                        RHSKernelControls, RHSIndices, RHSFacePackageView);
}
