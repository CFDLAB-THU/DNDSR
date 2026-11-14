#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Compute.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Launch.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    namespace
    {
        template <class Scalar>
        void Dispatch(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                      RHSParametersView parameters, RHSKernelControls controls, RHSFields fields,
                      RHSFaceOutputs output, RHSIndices selected)
        {
            using P = DNDS::Math::ComputePrecision;
            const bool polynomial = controls.polynomialPrecision != P::FP64;
            if (controls.fluxPrecision != P::FP64)
            {
                if (polynomial)
                    LaunchFluxFaces<5, Scalar, Scalar>(context, geometry, parameters, controls, fields, output, selected);
                else
                    LaunchFluxFaces<5, real, Scalar>(context, geometry, parameters, controls, fields, output, selected);
            }
            else if (controls.dissipationPrecision != P::FP64)
            {
                if (polynomial)
                    LaunchFluxFaces<5, Scalar, real, Scalar>(context, geometry, parameters, controls, fields, output, selected);
                else
                    LaunchFluxFaces<5, real, real, Scalar>(context, geometry, parameters, controls, fields, output, selected);
            }
            else
                LaunchFluxFaces<5, Scalar>(context, geometry, parameters, controls, fields, output, selected);
        }
    }
    void LaunchComputeFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                RHSParametersView parameters, RHSKernelControls controls, RHSFields fields, RHSFaceOutputs output, RHSIndices selected)
    {
        using P = DNDS::Math::ComputePrecision;
        const auto precision = controls.polynomialPrecision != P::FP64 ? controls.polynomialPrecision : controls.fluxPrecision != P::FP64 ? controls.fluxPrecision
                                                                                                                                          : controls.dissipationPrecision;
        if (precision == P::FP32)
            Dispatch<DNDS::Math::real_f32>(context, geometry, parameters, controls, fields, output, selected);
#ifdef DNDS_USE_CCCL_COMPUTE
        else if (precision == P::FloatFloat)
            Dispatch<DNDS::Math::real_f32f32>(context, geometry, parameters, controls, fields, output, selected);
#endif
        else
            DNDS_check_throw_info(false, "CUDA RHS compute precision is unavailable");
    }
}
