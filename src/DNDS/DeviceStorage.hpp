#pragma once

#include "DNDS/Errors.hpp"
#include "Defines.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

// #ifdef DNDS_USE_CUDA
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #endif

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

    // To extend backend:
    // specialize DeviceStorage in separate header (like DeviceStorage_CUDA.cuh)
    // specialize device_storage_factory in this header
    // define the factory functions of device_storage_factory in separate header (like DeviceStorage_CUDA.cuh)
    // extern declare explicit instantiations for basic types in this header
    // implement      explicit instantiations for basic types (like in DeviceStorage_CUDA.cu)
    // explicit instantiation for needed other types

    // To extend types:
    // extern declare explicit instantiations, like Geom/PeriodicInfo.hpp
    // implement      explicit instantiations of Host code,               like Geom/PeriodicInfo.cpp
    // implement      explicit instantiations of each needed device code, like Geom/PeriodicInfo.cu

    inline const char *device_backend_name(DeviceBackend B)
    {
        switch (B)
        {
        case DeviceBackend::Host:
            return "Host";
#ifdef DNDS_USE_CUDA
        case DeviceBackend::CUDA:
            return "CUDA";
#endif
        case DeviceBackend::Custom1:
            return "Custom1";
        case DeviceBackend::Unknown:
        default:
            return "Unknown";
        }
    }

    inline DeviceBackend device_backend_name_to_enum(std::string_view s)
    {
        if (s == "Host")
            return DeviceBackend::Host;
#ifdef DNDS_USE_CUDA
        if (s == "CUDA")
            return DeviceBackend::CUDA;
#endif
        return DeviceBackend::Unknown;
    }

    class DeviceStorageBase;
    void deviceStorageBase_deleter(DeviceStorageBase *p); // safe deleter to maintain cross-DLL safety

    using t_supDeviceStorageBase = std::unique_ptr<DeviceStorageBase, std::function<void(DeviceStorageBase *)>>;
    using t_sspDeviceStorageBase = std::shared_ptr<DeviceStorageBase>;

    inline t_supDeviceStorageBase null_supDeviceStorageBase()
    {
        return {nullptr, deviceStorageBase_deleter};
    }

    class DeviceStorageBase
    {
    public:
        virtual void *raw_ptr() = 0;
        virtual void copy_host_to_device(void *host_ptr, size_t n_bytes) = 0;
        virtual void copy_device_to_host(void *host_ptr, size_t n_bytes) = 0;
        //! =0 is a definition and all virtual functions must be defined to have vtable
        //! never omit =0 or use {}
        virtual t_supDeviceStorageBase clone() = 0;
        t_sspDeviceStorageBase shared_clone()
        {
            return t_sspDeviceStorageBase(clone().release());
        }
        [[nodiscard]] virtual size_t bytes() const = 0;
        [[nodiscard]] virtual DeviceBackend backend() const = 0;
        virtual ~DeviceStorageBase();
    };

    template <DeviceBackend B>
    class DeviceStorage;

    template <DeviceBackend B>
    struct device_storage_factory
    {
        static t_supDeviceStorageBase device_storage_create_unique(DeviceBackend backend, size_t n_bytes);
        static t_sspDeviceStorageBase device_storage_create_shared(DeviceBackend backend, size_t n_bytes);
    };

    template <>
    struct device_storage_factory<DeviceBackend::Host>
    {
        static t_supDeviceStorageBase device_storage_create_unique(DeviceBackend backend, size_t n_bytes);
        static t_sspDeviceStorageBase device_storage_create_shared(DeviceBackend backend, size_t n_bytes);
    };
/*
    #define DNDS_DEVICE_STORAGE_INST(T, B, ext)                                                                                               \
         ext template t_supDeviceStorageBase device_storage_factory<T, B>::device_storage_create_unique(DeviceBackend backend, size_t n_bytes); \
        ext template t_sspDeviceStorageBase device_storage_factory<T, B>::device_storage_create_shared(DeviceBackend backend, size_t n_bytes);
*/
#ifdef DNDS_USE_CUDA
    template <>
    struct device_storage_factory<DeviceBackend::CUDA>
    {
        static t_supDeviceStorageBase device_storage_create_unique(DeviceBackend backend, size_t n_bytes);
        static t_sspDeviceStorageBase device_storage_create_shared(DeviceBackend backend, size_t n_bytes);
    };

#endif

    inline t_supDeviceStorageBase device_storage_create(DeviceBackend backend, size_t n_bytes)
    {
        switch (backend)
        {
        case DeviceBackend::Host:
            // return std::make_unique<DeviceStorage<T, DeviceBackend::Host>>(n_elem);
            return device_storage_factory<DeviceBackend::Host>::device_storage_create_unique(backend, n_bytes);
#ifdef DNDS_USE_CUDA
        case DeviceBackend::CUDA:
            return device_storage_factory<DeviceBackend::CUDA>::device_storage_create_unique(backend, n_bytes);
#endif
        case DNDS::DeviceBackend::Custom1:
        {
            DNDS_assert_info(false, "not implemented");
            return null_supDeviceStorageBase();
        case DeviceBackend::Unknown:
        default:
            return null_supDeviceStorageBase();
        }
        }
    }

    inline t_sspDeviceStorageBase device_storage_create_shared(DeviceBackend backend, size_t n_bytes)
    {
        switch (backend)
        {

        case DeviceBackend::Host:
            return device_storage_factory<DeviceBackend::Host>::device_storage_create_shared(backend, n_bytes);
#ifdef DNDS_USE_CUDA
        case DeviceBackend::CUDA:
            return device_storage_factory<DeviceBackend::CUDA>::device_storage_create_shared(backend, n_bytes);
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

    template <DeviceBackend B, typename T, typename TSize = int64_t>
    class vector_DeviceView
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>,
                      "host_device_vector elements must be trivially_copyable and default_constructible");

        T *_data = nullptr;
        TSize _size = 0;

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(vector_DeviceView, vector_DeviceView)
        DNDS_DEVICE_CALLABLE vector_DeviceView(T *n_data, TSize n_size) : _data(n_data), _size(n_size) {}

        DNDS_DEVICE_CALLABLE T operator[](TSize i) const
        {
            DNDS_HD_assert(i >= 0 && i < _size);
            return _data[i];
        }
        DNDS_DEVICE_CALLABLE T &operator[](TSize i)
        {
            DNDS_HD_assert(i >= 0 && i < _size);
            return _data[i];
        }
        DNDS_DEVICE_CALLABLE TSize size() const { return _size; }
    };

    template <typename T>
    struct host_device_vector : public std::vector<T>
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>,
                      "host_device_vector elements must be trivially_copyable and default_constructible");
        using t_base = std::vector<T>;
        using t_base::t_base;

        t_supDeviceStorageBase deviceStorage = null_supDeviceStorageBase();

        DNDS_HOST host_device_vector<T> &operator=(const std::vector<T> &v)
        {
            this->t_base::operator=(v);
            return *this;
        }

        // DNDS_HOST explicit operator std::vector<T>() const
        // {
        //     std::vector<T> ret;
        //     ret.resize(this->size());
        //     std::copy(this->begin(), this->end(), ret.begin());
        //     return ret;
        // }

        void to_device(DeviceBackend backend = DeviceBackend::Host)
        {
            DNDS_assert_info(DeviceBackend::Unknown != backend, "cannot to_device to Unknown");
            if (!deviceStorage ||
                deviceStorage->bytes() != this->size() * sizeof(T) || // size change
                deviceStorage->backend() != backend)                  // backend change
                deviceStorage = device_storage_create(backend, this->size() * sizeof(T));
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

        template <DeviceBackend B, typename TSize = int64_t>
        using t_deviceView = vector_DeviceView<B, T, TSize>;
        template <DeviceBackend B, typename TSize = int64_t>
        t_deviceView<B, TSize> deviceView()
        {
            DNDS_assert_info(this->device() == B || (B == DeviceBackend::Host || B == DeviceBackend::Unknown),
                             "not on this device: " + std::string(device_backend_name(B)));
            if constexpr (B == DeviceBackend::Host || B == DeviceBackend::Unknown)
                return t_deviceView<B, TSize>(this->data(), this->size());
            else
                return t_deviceView<B, TSize>(this->dataDevice(), this->size());
        }

        const T *dataDevice() const
        {
            return deviceStorage ? reinterpret_cast<const T *>(deviceStorage->raw_ptr()) : nullptr;
        }

        host_device_vector &operator=(const host_device_vector<T> &R)
        {
            if (this == &R)
                return *this;
            this->t_base::operator=(R);
            this->deviceStorage = R.deviceStorage ? R.deviceStorage->clone() : null_supDeviceStorageBase();
            return *this;
        }

        host_device_vector(const host_device_vector<T> &R) : t_base(R)
        {
            this->deviceStorage = R.deviceStorage ? R.deviceStorage->clone() : null_supDeviceStorageBase();
        }

        DeviceBackend device()
        {
            return deviceStorage ? deviceStorage->backend() : DeviceBackend::Unknown;
        }
    };
}