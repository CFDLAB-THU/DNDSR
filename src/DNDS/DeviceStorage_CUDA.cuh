#pragma once

#include "DeviceStorage.hpp"

#ifdef DNDS_USE_CUDA
#    include <thrust/host_vector.h>
#    include <thrust/device_vector.h>
#endif
namespace DNDS
{

#ifdef DNDS_USE_CUDA
    template <typename T>
    class DeviceStorage<T, DeviceBackend::CUDA> : public DeviceStorageBase
    {
        using self_type = DeviceStorage<T, DeviceBackend::CUDA>;
        thrust::device_vector<T> data;

    public:
        explicit DeviceStorage(size_t n) : data(n) {}

        void *raw_ptr() override
        {
            return reinterpret_cast<void *>(thrust::raw_pointer_cast(data.data()));
        }
        void copy_host_to_device(void *host_ptr, size_t n_bytes) override
        {
            DNDS_assert_info(n_bytes == bytes(), "bytes size mismatch");
            T *host_T_ptr = reinterpret_cast<T *>(host_ptr);
            thrust::copy(host_T_ptr, host_T_ptr + data.size(), data.begin());
        }
        void copy_device_to_host(void *host_ptr, size_t n_bytes) override
        {
            DNDS_assert_info(n_bytes == bytes(), "bytes size mismatch");
            T *host_T_ptr = reinterpret_cast<T *>(host_ptr);
            thrust::copy(data.begin(), data.end(), host_T_ptr);
        }
        t_supDeviceStorageBase clone() override
        {
            return {new self_type(*this), deviceStorageBase_deleter<T>}; // copy CTOR
        }
        [[nodiscard]] size_t bytes() const override
        {
            return data.size() * sizeof(T);
        }
        [[nodiscard]] DeviceBackend backend() const override
        {
            return DeviceBackend::CUDA;
        }
    };
#endif

    template <typename T>
    t_supDeviceStorageBase device_storage_factory<T, DeviceBackend::CUDA>::device_storage_create_unique(DeviceBackend backend, size_t n_elem)
    {
        // return std::make_unique<DeviceStorage<T, DeviceBackend::CUDA>>(n_elem);
        return {(new DeviceStorage<T, DeviceBackend::CUDA>(n_elem)), deviceStorageBase_deleter<T>};
    }
    template <typename T>
    t_sspDeviceStorageBase device_storage_factory<T, DeviceBackend::CUDA>::device_storage_create_shared(DeviceBackend backend, size_t n_elem)
    {
        // return std::make_shared<DeviceStorage<T, DeviceBackend::CUDA>>(n_elem);
        return {(new DeviceStorage<T, DeviceBackend::CUDA>(n_elem)), deviceStorageBase_deleter<T>};
    }

}