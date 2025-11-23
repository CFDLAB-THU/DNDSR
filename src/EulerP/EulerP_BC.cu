#include "EulerP_BC.hpp"

#include "DNDS/DeviceStorage_CUDA.cuh"
#include "DNDS/DeviceStorage.hxx"

namespace DNDS::EulerP
{

}

namespace DNDS
{
    DNDS_DEVICE_STORAGE_BASE_DELETER_INST(EulerP::BC_DeviceView<DeviceBackend::CUDA>, )
    DNDS_DEVICE_STORAGE_INST(EulerP::BC_DeviceView<DeviceBackend::CUDA>, DeviceBackend::CUDA, )
}