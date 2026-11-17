#include "CFV/Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh"

namespace DNDS::CFV::Detail
{
    ReconstructionSweepResources RunStagedModes3(const StagedSweepArguments &args)
    {
        return Staged::ProductionModes<3>(args);
    }
}
