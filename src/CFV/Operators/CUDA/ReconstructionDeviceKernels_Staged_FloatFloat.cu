#include "CFV/Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh"

namespace DNDS::CFV::Detail
{
    ReconstructionSweepResources RunStagedFloatFloat(const StagedSweepArguments &args)
    {
#ifdef DNDS_USE_CCCL_COMPUTE
        return Staged::Comparison<real, DNDS::Math::real_f32f32, false, false, true>(args);
#else
        DNDS_check_throw_info(false, "float-float VR requires DNDS_USE_CCCL_COMPUTE");
        return {};
#endif
    }
}
