#include "VRDefines.hpp"
#include "DNDS/DeviceStorage.hxx"

namespace DNDS
{
    DNDS_DEVICE_STORAGE_BASE_DELETER_INST(CFV::RecAtr, )
    DNDS_DEVICE_STORAGE_INST(CFV::RecAtr, DeviceBackend::Host, )
}