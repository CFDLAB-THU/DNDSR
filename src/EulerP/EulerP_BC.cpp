#include "EulerP_BC.hpp"
#include "DNDS/DeviceStorage.hxx" // we use this for device storage code

namespace DNDS::EulerP
{

}

namespace DNDS
{
    DNDS_DEVICE_STORAGE_BASE_DELETER_INST(EulerP::BC_DeviceView<DeviceBackend::Host>, )
    DNDS_DEVICE_STORAGE_INST(EulerP::BC_DeviceView<DeviceBackend::Host>, DeviceBackend::Host, )
}