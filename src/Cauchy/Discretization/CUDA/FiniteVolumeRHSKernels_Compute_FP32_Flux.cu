#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Compute.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Compute_Impl.cu.inc"

namespace DNDS::Cauchy::CUDA::Detail
{
    void LaunchFP32FluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                             RHSComputeParametersView parameters, RHSKernelControls controls,
                             RHSFields fields, RHSFaceOutputs output, RHSIndices selected)
    {
        using Scalar = DNDS::Math::real_f32;
        if (controls.polynomialPrecision != DNDS::Math::ComputePrecision::FP64)
            LaunchFluxFaces<5, Scalar, Scalar>(context, geometry, parameters, controls, fields, output, selected);
        else
            LaunchFluxFaces<5, real, Scalar>(context, geometry, parameters, controls, fields, output, selected);
    }
}
