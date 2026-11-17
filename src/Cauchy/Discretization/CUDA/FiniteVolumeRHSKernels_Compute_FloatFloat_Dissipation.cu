#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Compute.hpp"
#ifdef DNDS_USE_CCCL_COMPUTE
#    include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Compute_Impl.cu.inc"

namespace DNDS::Cauchy::CUDA::Detail
{
    void LaunchFloatFloatDissipationFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                              RHSComputeParametersView parameters, RHSKernelControls controls,
                                              RHSFields fields, RHSFaceOutputs output, RHSIndices selected)
    {
        using Scalar = DNDS::Math::real_f32f32;
        if (controls.polynomialPrecision != DNDS::Math::ComputePrecision::FP64)
            LaunchFluxFaces<5, Scalar, real, Scalar>(context, geometry, parameters, controls, fields, output, selected);
        else
            LaunchFluxFaces<5, real, real, Scalar>(context, geometry, parameters, controls, fields, output, selected);
    }
}
#endif
