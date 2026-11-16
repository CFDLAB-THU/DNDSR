#include "CFV/Operators/CUDA/ReconstructionDevice_General.cuh"

namespace DNDS::CFV::Detail
{
    void RunGeneralFP32(const GeneralSweepArguments &a, bool full)
    {
        if (full)
            RunGeneralShapes<float, float, float>(a);
        else
            RunGeneralShapes<float, real, float>(a);
    }
}
