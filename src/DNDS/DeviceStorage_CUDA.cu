#include "DeviceStorage_CUDA.cuh"

namespace DNDS
{
    // DNDS_DEVICE_STORAGE_CUDA_INST(int, )
    // DNDS_DEVICE_STORAGE_CUDA_INST(double, )
    //! how to resolve int-rowsize duplicate?
    DNDS_DEVICE_STORAGE_INST(rowsize, DeviceBackend::CUDA, )
    DNDS_DEVICE_STORAGE_INST(real, DeviceBackend::CUDA, )
    DNDS_DEVICE_STORAGE_INST(index, DeviceBackend::CUDA, )
}