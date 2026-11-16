#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    // Distinguish a native sm80+ image from a JIT of an older trap-only image.
    // Consumer Ampere does not have the datacenter FP64-MMA capability.
    __global__ void TensorImageMarker() {}
    bool FP64TensorCoreAvailable(Runtime::Device device)
    {
        if (device.space != Runtime::MemorySpace::CUDA || cudaSetDevice(device.ordinal) != cudaSuccess)
            return false;
        static thread_local int ordinal = -1;
        static thread_local bool available = false;
        if (ordinal == device.ordinal)
            return available;
        cudaDeviceProp properties{};
        cudaFuncAttributes attributes{};
        if (cudaGetDeviceProperties(&properties, device.ordinal) != cudaSuccess || cudaFuncGetAttributes(&attributes, TensorImageMarker) != cudaSuccess)
            return false;
        available = ((properties.major == 8 && properties.minor == 0) || (properties.major == 9 && properties.minor == 0)) &&
                    attributes.binaryVersion >= 80 && attributes.ptxVersion >= 80;
        ordinal = device.ordinal;
        return available;
    }
    void LaunchJacobiTensorCore(Runtime::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                                Runtime::PairReadView<real> mean, Runtime::PairReadView<real> input, Runtime::ArrayView<const real> boundary,
                                Runtime::ArrayView<real> output, Runtime::ArrayView<const int> status, bool wide)
    {
        ReconstructionSweepLaunch options;
        options.mapping = "warp";
        options.asynchronous = false;
        options.gridBlocks = wide ? 0 : 256;
        LaunchStagedJacobi(context, plan, selection, vectors, mean, input, boundary, output, status, options, true);
    }
}
