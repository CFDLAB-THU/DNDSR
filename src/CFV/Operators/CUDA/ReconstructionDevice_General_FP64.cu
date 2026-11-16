#include "CFV/Operators/CUDA/ReconstructionDevice_General.cuh"

namespace DNDS::CFV::Detail
{
    void RunGeneralFP64(const GeneralSweepArguments &a)
    {
        RunGeneralShapes<real, real, real>(a);
    }
    void RunGeneralLimited(const GeneralSweepArguments &a)
    {
        if (a.plan.quantized)
            RunGeneralShapes<float, real, float, true>(a);
        else
            RunGeneralShapes<real, real, real, true>(a);
    }
}
