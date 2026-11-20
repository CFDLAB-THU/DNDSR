#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "CFV/Operators/CUDA/ReconstructionDevice_General.cuh"

namespace DNDS::CFV::Detail
{
    namespace
    {
        void LaunchGeneral(const GeneralSweepArguments &a, bool limited)
        {
            ValidateSweepArguments(a, limited);
            DNDS_check_throw_info(!a.plan.quantized && !a.plan.packedRawB && !a.plan.packedInverse &&
                                      a.plan.n > 0 && a.mean.owned.Width() > 0 &&
                                      a.output.Width() == a.mean.owned.Width() * a.plan.n,
                                  "general FP64 reconstruction requires FP64 maps and matching field widths");
            const index bytes = a.output.Width() * index(sizeof(real));
            static thread_local int ordinal = -1;
            static thread_local size_t sharedLimit = 0;
            if (ordinal != a.context.device.ordinal)
            {
                cudaDeviceProp properties{};
                DNDS_check_throw_info(cudaGetDeviceProperties(&properties, a.context.device.ordinal) == cudaSuccess,
                                      "general FP64 reconstruction device query failed");
                sharedLimit = properties.sharedMemPerBlock;
                ordinal = a.context.device.ordinal;
            }
            DNDS_check_throw_info(bytes <= index(sharedLimit), "general FP64 reconstruction scratch exceeds shared memory");
            if (!a.selection.RowCount())
                return;
            GeneralJacobi<<<unsigned(std::min<index>(a.selection.RowCount(), 256)), 128, size_t(bytes), ::DNDS::Runtime::CUDAStream(a.context)>>>(
                RawReconstructionSweepView(a.plan), a.selection, a.vectors, a.mean, a.input, a.boundary,
                a.output, a.status, a.target, a.alpha, a.plan.raw, limited);
        }
    }
    void RunGeneralFP64(const GeneralSweepArguments &a) { LaunchGeneral(a, false); }
    void RunGeneralLimited(const GeneralSweepArguments &a) { LaunchGeneral(a, true); }
}
