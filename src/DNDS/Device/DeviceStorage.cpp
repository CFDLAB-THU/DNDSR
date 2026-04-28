/// @file DeviceStorage.cpp
/// @brief Host-backend implementation of @ref DNDS::DeviceStorage "DeviceStorage" and the
/// #deviceStorageBase_deleter helper.

#include "DeviceStorage.hpp"
#include "DeviceStorage.hxx"

namespace DNDS
{
    DeviceStorageBase::~DeviceStorageBase() = default;

    void deviceStorageBase_deleter(DeviceStorageBase *p)
    {
        // Deleter callback passed to `std::shared_ptr`; the caller guarantees
        // `p` was allocated with `new`. Smart-pointer ownership is tracked
        // upstream.
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        delete p;
    }

    // DNDS_DEVICE_STORAGE_BASE_DELETER_INST(int, )
    // DNDS_DEVICE_STORAGE_BASE_DELETER_INST(double, )
    // DNDS_DEVICE_STORAGE_BASE_DELETER_INST(rowsize, )
    // DNDS_DEVICE_STORAGE_BASE_DELETER_INST(real, )
    // DNDS_DEVICE_STORAGE_BASE_DELETER_INST(index, )

    // * explicit instantiation of device storage

    // DNDS_DEVICE_STORAGE_INST(int, DeviceBackend::Host, )
    // DNDS_DEVICE_STORAGE_INST(double, DeviceBackend::Host, )
    //! how to resolve int-rowsize duplicate?
    // DNDS_DEVICE_STORAGE_INST(rowsize, DeviceBackend::Host, )
    // DNDS_DEVICE_STORAGE_INST(real, DeviceBackend::Host, )
    // DNDS_DEVICE_STORAGE_INST(index, DeviceBackend::Host, )

    // 1. specialize DeviceStorage
    template <>
    class DeviceStorage<DeviceBackend::Host> : public DeviceStorageBase
    {
        using self_type = DeviceStorage<DeviceBackend::Host>;
        // std::vector<T> data;
        size_t _size = 0;
        uint8_t *data = nullptr;

    public:
        explicit DeviceStorage(size_t n) : //  data(n),
                                           _size(n)
        {
        }

        uint8_t *raw_ptr() override
        {
            // return reinterpret_cast<void *>(data.data());
            // ! point-back design:
            return data;
        }
        void copy_host_to_device(uint8_t *host_ptr, size_t n_bytes) override
        {
            DNDS_assert_info(n_bytes == bytes(), "bytes size mismatch");
            auto *host_T_ptr = host_ptr;
            // std::copy(host_T_ptr, host_T_ptr + data.size(), data.begin());
            // ! point-back design:
            data = host_T_ptr;
        }
        void copy_device_to_host(uint8_t *host_ptr, size_t n_bytes) override
        {
            DNDS_assert_info(n_bytes == bytes(), "bytes size mismatch");
            auto *host_T_ptr = host_ptr;
            // std::copy(data.begin(), data.end(), host_T_ptr);
            // ! point-back design:
            // do nothing
        }
        void copy_to_device(uint8_t *device_ptr_dst, size_t n_bytes) override
        {
            DNDS_assert_info(n_bytes == bytes(), "bytes size mismatch");
            auto *device_T_ptr_dst = device_ptr_dst;
            // ! point-back design:
            // do nothing
        }

        [[nodiscard]] size_t bytes() const override
        {
            // return data.size() * sizeof(T);
            // ! point-back design:
            return _size;
        }
        [[nodiscard]] DeviceBackend backend() const override
        {
            return DeviceBackend::Host;
        }
    };

    t_supDeviceStorageBase device_storage_factory<DeviceBackend::Host>::device_storage_create_unique(DeviceBackend backend, size_t n_bytes)
    {
        // return std::make_unique<DeviceStorage<T, DeviceBackend::CUDA>>(n_elem);
        return {(new DeviceStorage<DeviceBackend::Host>(n_bytes)), deviceStorageBase_deleter};
    }
    t_sspDeviceStorageBase device_storage_factory<DeviceBackend::Host>::device_storage_create_shared(DeviceBackend backend, size_t n_bytes)
    {
        // return std::make_shared<DeviceStorage<T, DeviceBackend::CUDA>>(n_elem);
        return {(new DeviceStorage<DeviceBackend::Host>(n_bytes)), deviceStorageBase_deleter};
    }
}