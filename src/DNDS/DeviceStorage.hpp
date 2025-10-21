#pragma once

#include "Defines.hpp"
#include <cstddef>
#include <memory>

#ifdef DNDS_USE_CUDA
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

namespace DNDS
{
    enum class DeviceBackend
    {
        Unknown = 0,
        Host = 1,
#ifdef DNDS_USE_CUDA
        CUDA = 2,
#endif
        Custom1 = 101,
    };

    class DeviceStorageBase
    {
    public:
        virtual void *raw_ptr() = 0;
        virtual void copy_host_to_device(void *host_ptr, size_t n_bytes) = 0;
        virtual void copy_device_to_host(void *host_ptr, size_t n_bytes) = 0;
        //! =0 is a definition and all virtual functions must be defined to have vtable
        //! never omit =0 or use {}
        virtual std::unique_ptr<DeviceStorageBase> clone() = 0;
        std::shared_ptr<DeviceStorageBase> shared_clone()
        {
            return std::shared_ptr<DeviceStorageBase>(clone().release());
        }
        [[nodiscard]] virtual size_t bytes() const = 0;
        [[nodiscard]] virtual DeviceBackend backend() const = 0;
        virtual ~DeviceStorageBase();
    };

    using t_supDeviceStorageBase = std::unique_ptr<DeviceStorageBase>;
    using t_sspDeviceStorageBase = std::shared_ptr<DeviceStorageBase>;

    template <typename T, DeviceBackend B>
    class DeviceStorage;

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
        std::unique_ptr<DeviceStorageBase> clone() override
        {
            return std::make_unique<self_type>(*this); // copy CTOR
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
        std::unique_ptr<DeviceStorageBase> clone() override
        {
            return std::make_unique<self_type>(*this); // copy CTOR
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
    t_supDeviceStorageBase device_storage_create(DeviceBackend backend, size_t n_elem)
    {
        switch (backend)
        {
        case DeviceBackend::Host:
            return std::make_unique<DeviceStorage<T, DeviceBackend::Host>>(n_elem);
#ifdef DNDS_USE_CUDA
        case DeviceBackend::CUDA:
            return std::make_unique<DeviceStorage<T, DeviceBackend::CUDA>>(n_elem);
#endif
        case DNDS::DeviceBackend::Custom1:
        {
            DNDS_assert_info(false, "not implemented");
            return nullptr;
        case DeviceBackend::Unknown:
        default:
            return nullptr;
        }
        }
    }

    template <typename T>
    t_sspDeviceStorageBase device_storage_create_shared(DeviceBackend backend, size_t n_elem)
    {
        switch (backend)
        {

        case DeviceBackend::Host:
            return std::make_shared<DeviceStorage<T, DeviceBackend::Host>>(n_elem);
#ifdef DNDS_USE_CUDA
        case DeviceBackend::CUDA:
            return std::make_shared<DeviceStorage<T, DeviceBackend::CUDA>>(n_elem);
#endif
        case DNDS::DeviceBackend::Custom1:
        {
            DNDS_assert_info(false, "not implemented");
            return nullptr;
        }
        case DeviceBackend::Unknown:
        default:
            return nullptr;
        }
    }

    template <typename T>
    struct host_device_vector : public std::vector<T>
    {
        using t_base = std::vector<T>;
        using t_base::t_base;

        t_supDeviceStorageBase deviceStorage;

        void to_device(DeviceBackend backend = DeviceBackend::Host)
        {
            if (!deviceStorage || deviceStorage->bytes() != this->size() * sizeof(T))
                deviceStorage = device_storage_create<T>(backend, this->size());
            deviceStorage->copy_host_to_device(this->data(), this->size() * sizeof(T));
        }

        void to_host()
        {
            DNDS_assert(deviceStorage);
            deviceStorage->copy_device_to_host(this->data(), this->size() * sizeof(T));
        }

        T *dataDevice()
        {
            return deviceStorage ? reinterpret_cast<T *>(deviceStorage->raw_ptr()) : nullptr;
        }

        host_device_vector &operator=(const host_device_vector<T> &R)
        {
            this->t_base::operator=(R);
            this->deviceStorage = R.deviceStorage ? R.deviceStorage->clone() : nullptr;
            return *this;
        }

        host_device_vector(const host_device_vector<T> &R) : t_base(R)
        {
            this->deviceStorage = R.deviceStorage ? R.deviceStorage->clone() : nullptr;
        }
    };
}