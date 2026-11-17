#include "CFV/Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh"

namespace DNDS::CFV::Detail
{
    ReconstructionSweepResources RunStagedModes9(const StagedSweepArguments &args)
    {
        return Staged::ProductionModes<9>(args);
    }
}
