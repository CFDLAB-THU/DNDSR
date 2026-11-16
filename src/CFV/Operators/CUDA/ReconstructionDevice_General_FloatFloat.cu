#include "CFV/Operators/CUDA/ReconstructionDevice_General.cuh"

namespace DNDS::CFV::Detail
{
    void RunGeneralFloatFloat(const GeneralSweepArguments &a)
    {
#ifdef DNDS_USE_CCCL_COMPUTE
        RunGeneralShapes<float, DNDS::Math::real_f32f32, DNDS::Math::real_f32f32>(a);
#else
        DNDS_check_throw_info(false, "float-float Jacobi requires DNDS_USE_CCCL_COMPUTE");
#endif
    }
}
