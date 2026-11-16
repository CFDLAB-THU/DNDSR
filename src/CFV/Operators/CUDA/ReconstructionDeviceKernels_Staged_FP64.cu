#include "CFV/Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh"

namespace DNDS::CFV::Detail
{
    ReconstructionSweepResources RunStagedFP64(const StagedSweepArguments &args, bool tensor)
    {
        return tensor ? Staged::RunShapes<real, real, true>(args) : Staged::RunShapes<real, real, false>(args);
    }
}
