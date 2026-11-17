#include "CFV/Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh"

namespace DNDS::CFV::Detail
{
    ReconstructionSweepResources RunStagedModes2(const StagedSweepArguments &args)
    {
        return Staged::ProductionModes<2>(args);
    }
}
