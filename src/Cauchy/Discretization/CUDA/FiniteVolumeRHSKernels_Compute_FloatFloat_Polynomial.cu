#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Compute.hpp"
#ifdef DNDS_USE_CCCL_COMPUTE
#    include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Launch.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    void LaunchFloatFloatPolynomialFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                             RHSComputeParametersView parameters, RHSKernelControls controls,
                                             RHSFields fields, RHSFaceOutputs output, RHSIndices selected)
    {
        LaunchFluxFaces<5, DNDS::Math::real_f32f32>(context, geometry, parameters, controls, fields, output, selected);
    }
}
#endif
