#pragma once
/// @file DeviceStorage.hpp
/// @brief Device memory abstraction layer with backend-specific storage and factory creation.

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
    /// @brief Enumerates available device backends (Host, CUDA, etc.).
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

    /// @brief Abstract interface for device memory management (copy, query, lifetime).
    class DeviceStorageBase
    {
    public:
        virtual uint8_t *raw_ptr() = 0;
        virtual void copy_host_to_device(uint8_t *host_ptr, size_t n_bytes) = 0;
        virtual void copy_device_to_host(uint8_t *host_ptr, size_t n_bytes) = 0;
        virtual void copy_to_device(uint8_t *device_ptr_dst, size_t n_bytes) = 0;
        //! =0 is a definition and all virtual functions must be defined to have vtable
        //! never omit =0 or use {}
        [[nodiscard]] virtual size_t bytes() const = 0;
        [[nodiscard]] virtual DeviceBackend backend() const = 0;
        virtual ~DeviceStorageBase();
    };

    /// @brief Concrete device storage implementation, specialized per DeviceBackend.
    template <DeviceBackend B>
    class DeviceStorage;

    /// @brief Factory for creating DeviceStorage instances for a given backend.
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
        }
        case DeviceBackend::Unknown:
        default:
            return null_supDeviceStorageBase();
        }
    }
}