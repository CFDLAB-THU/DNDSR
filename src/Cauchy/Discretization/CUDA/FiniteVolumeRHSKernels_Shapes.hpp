#pragma once

#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    /// Host-only launch arguments borrowed for this wrapper invocation. The
    /// submitting operator owns all resources; this contains no retained work.
    struct RHSFaceLaunch
    {
        Runtime::ExecutionContext context;
        const RHSGeometryPublication &geometry;
        Runtime::ArrayView<const RHSKernelParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> parameters;
        RHSKernelControls controls;
        RHSFields fields;
        RHSFaceOutputs output;
        RHSIndices selected;
        RHSFacePackageView packages;
        unsigned segmentBlocks = 0;
    };
}
