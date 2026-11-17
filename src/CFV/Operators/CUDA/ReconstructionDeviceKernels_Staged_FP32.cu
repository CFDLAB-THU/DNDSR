#include "CFV/Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh"

namespace DNDS::CFV::Detail
{
    ReconstructionSweepResources RunStagedFP32(const StagedSweepArguments &args, bool tensor)
    {
        return tensor ? Staged::Comparison<float, float, true, true, true>(args)
                      : Staged::Comparison<float, float, false, false, true>(args);
    }
    ReconstructionSweepResources RunQuantizedComparison(const StagedSweepArguments &original, int policy)
    {
        auto args = original;
        if (!args.options.threads)
            args.options.threads = 128; // Retain the non-pipelined comparison launch.
        if (policy == 0)
            return Staged::Comparison<float, real, false, false, false, float>(args);
        if (policy == 1)
            return Staged::Comparison<float, float, false, false, false>(args);
        if (policy == 3)
            return Staged::Comparison<float, real, true, true, false>(args);
#ifdef DNDS_USE_CCCL_COMPUTE
        if (policy == 2)
            return Staged::Comparison<float, DNDS::Math::real_f32f32, false, false, false>(args);
#endif
        DNDS_check_throw_info(false, "unavailable quantized VR comparison policy");
        return {};
    }
}
