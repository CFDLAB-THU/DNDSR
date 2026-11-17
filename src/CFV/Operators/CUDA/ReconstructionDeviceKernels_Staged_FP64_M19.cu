#include "CFV/Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh"

namespace DNDS::CFV::Detail
{
    ReconstructionSweepResources RunStagedModes19(const StagedSweepArguments &args)
    {
        return Staged::ProductionModes<19>(args);
    }
}
