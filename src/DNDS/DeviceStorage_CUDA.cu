#include "DeviceStorage_CUDA.cuh"

namespace DNDS
{
    // DNDS_DEVICE_STORAGE_CUDA_INST(int, )
    // DNDS_DEVICE_STORAGE_CUDA_INST(double, )
    //! how to resolve int-rowsize duplicate?
    // DNDS_DEVICE_STORAGE_INST(rowsize, DeviceBackend::CUDA, )
    // DNDS_DEVICE_STORAGE_INST(real, DeviceBackend::CUDA, )
    // DNDS_DEVICE_STORAGE_INST(index, DeviceBackend::CUDA, )
}

namespace DNDS
{

#ifdef DNDS_USE_CUDA
    template <>
    class DeviceStorage<DeviceBackend::CUDA> : public DeviceStorageBase
    {
        using self_type = DeviceStorage<DeviceBackend::CUDA>;
        thrust::device_vector<uint8_t> data;

    public:
        explicit DeviceStorage(size_t n) : data(n) {}

        void *raw_ptr() override
        {
            return reinterpret_cast<void *>(thrust::raw_pointer_cast(data.data()));
        }
        void copy_host_to_device(void *host_ptr, size_t n_bytes) override
        {
            DNDS_assert_info(n_bytes == bytes(), "bytes size mismatch");
            auto *host_T_ptr = reinterpret_cast<uint8_t *>(host_ptr);
            thrust::copy(host_T_ptr, host_T_ptr + data.size(), data.begin());
        }
        void copy_device_to_host(void *host_ptr, size_t n_bytes) override
        {
            DNDS_assert_info(n_bytes == bytes(), "bytes size mismatch");
            auto *host_T_ptr = reinterpret_cast<uint8_t *>(host_ptr);
            thrust::copy(data.begin(), data.end(), host_T_ptr);
        }
        t_supDeviceStorageBase clone() override
        {
            return {new self_type(*this), deviceStorageBase_deleter}; // copy CTOR
        }
        [[nodiscard]] size_t bytes() const override
        {
            return data.size();
        }
        [[nodiscard]] DeviceBackend backend() const override
        {
            return DeviceBackend::CUDA;
        }
    };
#endif

    t_supDeviceStorageBase device_storage_factory<DeviceBackend::CUDA>::device_storage_create_unique(DeviceBackend backend, size_t n_bytes)
    {
        // return std::make_unique<DeviceStorage<T, DeviceBackend::CUDA>>(n_elem);
        return {(new DeviceStorage<DeviceBackend::CUDA>(n_bytes)), deviceStorageBase_deleter};
    }
    t_sspDeviceStorageBase device_storage_factory<DeviceBackend::CUDA>::device_storage_create_shared(DeviceBackend backend, size_t n_bytes)
    {
        // return std::make_shared<DeviceStorage<T, DeviceBackend::CUDA>>(n_elem);
        return {(new DeviceStorage<DeviceBackend::CUDA>(n_bytes)), deviceStorageBase_deleter};
    }

}