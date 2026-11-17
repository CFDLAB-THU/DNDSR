#include "CFV/Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh"

namespace DNDS::CFV::Detail
{
    ReconstructionSweepResources RunStagedModes2(const StagedSweepArguments &);
    ReconstructionSweepResources RunStagedModes3(const StagedSweepArguments &);
    ReconstructionSweepResources RunStagedModes5(const StagedSweepArguments &);
    ReconstructionSweepResources RunStagedModes9(const StagedSweepArguments &);
    ReconstructionSweepResources RunStagedModes19(const StagedSweepArguments &);

    ReconstructionSweepResources RunStagedFP64(const StagedSweepArguments &args, bool tensor)
    {
        if (tensor)
            return Staged::Comparison<real, real, true, false, true>(args);
        switch (args.plan.n)
        {
        case 2:
            return RunStagedModes2(args);
        case 3:
            return RunStagedModes3(args);
        case 5:
            return RunStagedModes5(args);
        case 9:
            return RunStagedModes9(args);
        case 19:
            return RunStagedModes19(args);
        default:
            DNDS_check_throw_info(false, "production VR pipeline needs P1/P2/P3 modal shape");
        }
        return {};
    }
    ReconstructionSweepResources RunMatrixShared(const StagedSweepArguments &args)
    {
        return Staged::Comparison<real, real, false, false, false>(args);
    }
}
