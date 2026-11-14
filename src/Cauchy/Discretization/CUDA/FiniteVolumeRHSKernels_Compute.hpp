#pragma once

#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    /// Experimental arithmetic dispatch after the common public wrapper has
    /// validated shapes, residency, supported physics and all precision flags.
    /// Uniform physical views keep this isolated experiment's instantiation
    /// count bounded; native FP64 uses static input-layout specializations.
    void LaunchComputeFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                Runtime::ArrayView<const RHSKernelParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> parameters,
                                RHSKernelControls controls, RHSFields fields, RHSFaceOutputs output, RHSIndices selected);
}
