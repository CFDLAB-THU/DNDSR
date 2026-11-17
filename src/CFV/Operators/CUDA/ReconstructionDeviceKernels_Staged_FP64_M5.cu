#include "CFV/Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh"

namespace DNDS::CFV::Detail
{
    ReconstructionSweepResources RunStagedModes5(const StagedSweepArguments &args)
    {
        return Staged::ProductionModes<5>(args);
    }
}
