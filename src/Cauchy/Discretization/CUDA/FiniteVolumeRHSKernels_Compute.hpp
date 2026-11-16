#pragma once

#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    using RHSComputeParametersView = Runtime::ArrayView<const RHSKernelParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    /// Experimental arithmetic dispatch after the common public wrapper has
    /// validated shapes, residency, supported physics and all precision flags.
    /// Uniform physical views keep this isolated experiment's instantiation
    /// count bounded; native FP64 uses static input-layout specializations.
    void LaunchComputeFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                RHSComputeParametersView parameters,
                                RHSKernelControls controls, RHSFields fields, RHSFaceOutputs output, RHSIndices selected);

    /** @brief Internal precision/scenario launch leaves for parallel compilation.
     * @details Each definition TU instantiates one or two versions of the same
     * shared face arithmetic. The host-only dispatcher includes no numerical
     * kernel definition; splitting this file must not change device inlining,
     * launch geometry, supported arithmetic combinations or access lifetimes.
     * These leaves inherit the public wrapper's validation contract above.
     * @{
     */
    void LaunchFP32PolynomialFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                       RHSComputeParametersView parameters, RHSKernelControls controls,
                                       RHSFields fields, RHSFaceOutputs output, RHSIndices selected);
    void LaunchFP32DissipationFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                        RHSComputeParametersView parameters, RHSKernelControls controls,
                                        RHSFields fields, RHSFaceOutputs output, RHSIndices selected);
    void LaunchFP32FluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                             RHSComputeParametersView parameters, RHSKernelControls controls,
                             RHSFields fields, RHSFaceOutputs output, RHSIndices selected);
#ifdef DNDS_USE_CCCL_COMPUTE
    void LaunchFloatFloatPolynomialFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                             RHSComputeParametersView parameters, RHSKernelControls controls,
                                             RHSFields fields, RHSFaceOutputs output, RHSIndices selected);
    void LaunchFloatFloatDissipationFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                              RHSComputeParametersView parameters, RHSKernelControls controls,
                                              RHSFields fields, RHSFaceOutputs output, RHSIndices selected);
    void LaunchFloatFloatFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                   RHSComputeParametersView parameters, RHSKernelControls controls,
                                   RHSFields fields, RHSFaceOutputs output, RHSIndices selected);
#endif
    /// @}
}
