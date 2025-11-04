#include "DeviceStorage.hpp"
#include "DeviceStorage.hxx"

namespace DNDS
{
    DeviceStorageBase::~DeviceStorageBase() = default;

    // DNDS_DEVICE_STORAGE_BASE_DELETER_INST(int, )
    // DNDS_DEVICE_STORAGE_BASE_DELETER_INST(double, )
    DNDS_DEVICE_STORAGE_BASE_DELETER_INST(rowsize, )
    DNDS_DEVICE_STORAGE_BASE_DELETER_INST(real, )
    DNDS_DEVICE_STORAGE_BASE_DELETER_INST(index, )

    // * explicit instantiation of device storage

    // DNDS_DEVICE_STORAGE_INST(int, DeviceBackend::Host, )
    // DNDS_DEVICE_STORAGE_INST(double, DeviceBackend::Host, )
    //! how to resolve int-rowsize duplicate?
    DNDS_DEVICE_STORAGE_INST(rowsize, DeviceBackend::Host, )
    DNDS_DEVICE_STORAGE_INST(real, DeviceBackend::Host, )
    DNDS_DEVICE_STORAGE_INST(index, DeviceBackend::Host, )
}