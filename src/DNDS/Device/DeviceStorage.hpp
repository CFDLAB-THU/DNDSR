#pragma once
/// @file DeviceStorage.hpp
/// @brief Device memory abstraction layer with backend-specific storage and factory creation.

#include "DNDS/Errors.hpp"
#include "DNDS/Defines.hpp"
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
    /**
     * @brief Enumerates the backends a @ref DNDS::DeviceStorage "DeviceStorage" / @ref DNDS::Array "Array" can live on.
     *
     * @details @ref Host is always available; @ref CUDA is compiled in when
     * @ref DNDS_USE_CUDA is defined. Additional slots (@ref Custom1, ...) are
     * placeholders for future backends (e.g., HIP, SYCL) that can be plugged
     * in by providing new factory specialisations.
     */
    enum class DeviceBackend
    {
        Unknown = 0, ///< Unset / sentinel.
        Host = 1,    ///< Plain CPU memory.
#ifdef DNDS_USE_CUDA
        CUDA = 2, ///< NVIDIA CUDA device memory.
#endif
        Custom1 = 101, ///< Reserved slot for a project-specific backend.
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

    /// @brief Canonical string name for a @ref DNDS::DeviceBackend "DeviceBackend" (used in log messages).
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

    /// @brief Inverse of #device_backend_name. Returns @ref Unknown for unrecognised names.
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
    /// @brief Stateless deleter for @ref DNDS::DeviceStorageBase "DeviceStorageBase" that works across shared-library
    /// boundaries where the vtable of `unique_ptr`'s default deleter would not.
    void deviceStorageBase_deleter(DeviceStorageBase *p); // safe deleter to maintain cross-DLL safety

    /// @brief Owning unique pointer to a @ref DNDS::DeviceStorageBase "DeviceStorageBase" with cross-DLL-safe deleter.
    using t_supDeviceStorageBase = std::unique_ptr<DeviceStorageBase, std::function<void(DeviceStorageBase *)>>;
    /// @brief Shared pointer equivalent of #t_supDeviceStorageBase.
    using t_sspDeviceStorageBase = std::shared_ptr<DeviceStorageBase>;

    /// @brief Null-value helper for #t_supDeviceStorageBase.
    inline t_supDeviceStorageBase null_supDeviceStorageBase()
    {
        return {nullptr, deviceStorageBase_deleter};
    }

    /**
     * @brief Abstract interface to a byte buffer owned by a specific backend.
     *
     * @details All DNDS device memory ultimately goes through this interface so
     * that the higher-level `host_device_vector<T>` can be backend-agnostic.
     * Concrete backends provide specialised @ref DNDS::DeviceStorage "DeviceStorage"<B> implementations;
     * creation funnels through #device_storage_factory.
     */
    class DeviceStorageBase
    {
    public:
        /// @brief Raw byte pointer to the underlying storage.
        virtual uint8_t *raw_ptr() = 0;
        /// @brief Copy `n_bytes` from `host_ptr` into this device buffer.
        virtual void copy_host_to_device(uint8_t *host_ptr, size_t n_bytes) = 0;
        /// @brief Copy `n_bytes` from this device buffer into `host_ptr`.
        virtual void copy_device_to_host(uint8_t *host_ptr, size_t n_bytes) = 0;
        /// @brief Device-to-device copy of `n_bytes` into `device_ptr_dst`.
        virtual void copy_to_device(uint8_t *device_ptr_dst, size_t n_bytes) = 0;
        //! =0 is a definition and all virtual functions must be defined to have vtable
        //! never omit =0 or use {}
        /// @brief Buffer size in bytes.
        [[nodiscard]] virtual size_t bytes() const = 0;
        /// @brief Which backend the buffer lives on.
        [[nodiscard]] virtual DeviceBackend backend() const = 0;
        virtual ~DeviceStorageBase();
    };

    /// @brief Compile-time-specialised storage class; one definition per @ref DNDS::DeviceBackend "DeviceBackend".
    template <DeviceBackend B>
    class DeviceStorage;

    /// @brief Factory functions for constructing @ref DNDS::DeviceStorageBase "DeviceStorageBase" instances of
    /// a specific backend. Specialised per backend so that the concrete type
    /// creation can live in backend-specific translation units.
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

    /// @brief Top-level factory: dispatches to the per-backend factory based on
    /// `backend`. Returns a null `unique_ptr` for @ref DNDS::DeviceBackend "DeviceBackend"::Unknown.
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