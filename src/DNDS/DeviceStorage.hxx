#pragma once
#include "DeviceStorage.hpp"

namespace DNDS
{
    // 1. partial specialize DeviceStorage
    template <typename T>
    class DeviceStorage<T, DeviceBackend::Host> : public DeviceStorageBase
    {
        using self_type = DeviceStorage<T, DeviceBackend::Host>;
        // std::vector<T> data;
        size_t _size = 0;
        T *data = nullptr;

    public:
        explicit DeviceStorage(size_t n) : //  data(n),
                                           _size(n)
        {
        }

        void *raw_ptr() override
        {
            // return reinterpret_cast<void *>(data.data());
            // ! point-back design:
            return reinterpret_cast<void *>(data);
        }
        void copy_host_to_device(void *host_ptr, size_t n_bytes) override
        {
            DNDS_assert_info(n_bytes == bytes(), "bytes size mismatch");
            T *host_T_ptr = reinterpret_cast<T *>(host_ptr);
            // std::copy(host_T_ptr, host_T_ptr + data.size(), data.begin());
            // ! point-back design:
            data = host_T_ptr;
        }
        void copy_device_to_host(void *host_ptr, size_t n_bytes) override
        {
            DNDS_assert_info(n_bytes == bytes(), "bytes size mismatch");
            T *host_T_ptr = reinterpret_cast<T *>(host_ptr);
            // std::copy(data.begin(), data.end(), host_T_ptr);
            // ! point-back design:
            // do nothing
        }
        t_supDeviceStorageBase clone() override
        {
            // return std::make_unique<self_type>(*this); // copy CTOR
            return {new self_type(*this), deviceStorageBase_deleter<T>};
        }
        [[nodiscard]] size_t bytes() const override
        {
            // return data.size() * sizeof(T);
            // ! point-back design:
            return _size * sizeof(T);
        }
        [[nodiscard]] DeviceBackend backend() const override
        {
            return DeviceBackend::Host;
        }
    };

    template <typename T>
    void deviceStorageBase_deleter(DeviceStorageBase *p)
    {
        delete p;
    }

    template <typename T>
    t_supDeviceStorageBase device_storage_factory<T, DeviceBackend::Host>::device_storage_create_unique(DeviceBackend backend, size_t n_elem)
    {
        // return std::make_unique<DeviceStorage<T, DeviceBackend::CUDA>>(n_elem);
        return {(new DeviceStorage<T, DeviceBackend::Host>(n_elem)), deviceStorageBase_deleter<T>};
    }
    template <typename T>
    t_sspDeviceStorageBase device_storage_factory<T, DeviceBackend::Host>::device_storage_create_shared(DeviceBackend backend, size_t n_elem)
    {
        // return std::make_shared<DeviceStorage<T, DeviceBackend::CUDA>>(n_elem);
        return {(new DeviceStorage<T, DeviceBackend::Host>(n_elem)), deviceStorageBase_deleter<T>};
    }
}