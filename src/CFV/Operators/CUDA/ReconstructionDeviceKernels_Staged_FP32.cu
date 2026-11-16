#include "CFV/Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh"

namespace DNDS::CFV::Detail
{
    ReconstructionSweepResources RunStagedFP32(const StagedSweepArguments &args, bool tensor)
    {
        return tensor ? Staged::RunShapes<float, float, true, true>(args) : Staged::RunShapes<float, float, false>(args);
    }
}
