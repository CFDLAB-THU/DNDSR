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
    bool TF32TensorCoreAvailable(Runtime::Device device)
    {
        if (device.space != Runtime::MemorySpace::CUDA || cudaSetDevice(device.ordinal) != cudaSuccess)
            return false;
        cudaDeviceProp properties{};
        cudaFuncAttributes attributes{};
        return cudaGetDeviceProperties(&properties, device.ordinal) == cudaSuccess &&
               cudaFuncGetAttributes(&attributes, TensorImageMarker) == cudaSuccess && properties.major >= 8 &&
               attributes.binaryVersion >= 80 && attributes.ptxVersion >= 80;
    }
}
