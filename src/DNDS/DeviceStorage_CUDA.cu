#include "DeviceStorage_CUDA.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <string>
#include "CUDA_Utils.hpp"
#include "Errors.hpp" 

namespace DNDS
{
    // DNDS_DEVICE_STORAGE_CUDA_INST(int, )
    // DNDS_DEVICE_STORAGE_CUDA_INST(double, )
    //! how to resolve int-rowsize duplicate?
    // DNDS_DEVICE_STORAGE_INST(rowsize, DeviceBackend::CUDA, )
    // DNDS_DEVICE_STORAGE_INST(real, DeviceBackend::CUDA, )
    // DNDS_DEVICE_STORAGE_INST(index, DeviceBackend::CUDA, )
}
// #define DNDS_DEVICESTORAGE_CUDA_USE_THRUST
// #define DNDS_DEVICESTORAGE_CUDA_USE_THRUST_PTR
namespace DNDS
{

#ifdef DNDS_USE_CUDA
    template <>
    class DeviceStorage<DeviceBackend::CUDA> : public DeviceStorageBase
    {
        using self_type = DeviceStorage<DeviceBackend::CUDA>;
#    ifdef DNDS_DEVICESTORAGE_CUDA_USE_THRUST
        thrust::device_vector<uint8_t> data;

#    elif defined(DNDS_DEVICESTORAGE_CUDA_USE_THRUST_PTR)
        size_t n_;
        thrust::device_ptr<uint8_t> p_data = nullptr;
#    else
        size_t n_;
        uint8_t *p_data = nullptr;
#    endif
        int device_id = 0;

    public:
#    ifdef DNDS_DEVICESTORAGE_CUDA_USE_THRUST
        explicit DeviceStorage(size_t n) : data(n)
        {
            CUdevice d;
            DNDS_CUDA_DRIVER_CHECKED(cuCtxGetDevice(&d));
            device_id = d;
        }
#    elif defined(DNDS_DEVICESTORAGE_CUDA_USE_THRUST_PTR)
        explicit DeviceStorage(size_t n) : n_(n)
        {
            CUdevice d;
            DNDS_CUDA_DRIVER_CHECKED(cuCtxGetDevice(&d));
            device_id = d;

            uint8_t *p;
            // cudaSetDevice(0);
            cudaDeviceSynchronize();
            cudaFree(0);
            DNDS_CUDA_CHECKED(::cudaMalloc(&p, n));
            p_data = thrust::device_ptr<uint8_t>(p);
        }
        ~DeviceStorage() override
        {
            DNDS_CUDA_CHECKED(::cudaFree(p_data.get()));
            p_data = nullptr;
            n_ = 0;
        }
#    else
        explicit DeviceStorage(size_t n) : n_(n)
        {
            if (n_ == 0)
                return;
            //! seems directly using cudaMalloc could break the CUDA context
            //! we switch to only using driver API here
            //! we only use dummy_thrust_data here to align with thrust's default context
            static thrust::device_vector<uint8_t> dummy_thrust_data(1);
            // cudaFree(0);
            // cudaGetDevice(&device_id);
            // DNDS_CUDA_CHECKED(::cudaFree(0));
            // DNDS_CUDA_CHECKED(::cudaMalloc(&p_data, n_));
            CUdevice d;
            DNDS_CUDA_DRIVER_CHECKED(cuCtxGetDevice(&d));
            device_id = d;
            DNDS_check_throw_info(device_id >= 0, "Device id is " + std::to_string(d));
            CUdeviceptr dptr;
            DNDS_CUDA_DRIVER_CHECKED(cuMemAlloc(&dptr, n_));
            p_data = reinterpret_cast<uint8_t *>(dptr);
        }
        ~DeviceStorage() override
        {
            // DNDS_CUDA_CHECKED(::cudaFree(p_data));
            DNDS_CUDA_DRIVER_CHECKED(cuMemFree(reinterpret_cast<CUdeviceptr>(p_data)));
            p_data = nullptr;
            n_ = 0;
        }
#    endif

        void *raw_ptr() override
        {
#    ifdef DNDS_DEVICESTORAGE_CUDA_USE_THRUST
            return reinterpret_cast<void *>(thrust::raw_pointer_cast(data.data()));
#    elif defined(DNDS_DEVICESTORAGE_CUDA_USE_THRUST_PTR)
            return reinterpret_cast<void *>(p_data.get());
#    else
            // if (p_data)
            // {
            //     cudaPointerAttributes a;
            //     DNDS_CUDA_CHECKED(cudaPointerGetAttributes(&a, p_data));
            //     std::cout << "ptr " << (void *)p_data << ", " << a.device << std::endl;
            // }
            return reinterpret_cast<void *>(p_data);
            // return reinterpret_cast<void *>(thrust::raw_pointer_cast(data.data()));
#    endif
        }
        void copy_host_to_device(void *host_ptr, size_t n_bytes) override
        {
            // std::cout << "Host to device " << n_bytes << "\n " << getTraceString() << std::endl;
            DNDS_check_throw_info(n_bytes == bytes(), "bytes size mismatch");
            auto *host_T_ptr = reinterpret_cast<uint8_t *>(host_ptr);
#    ifdef DNDS_DEVICESTORAGE_CUDA_USE_THRUST
            thrust::copy(host_T_ptr, host_T_ptr + data.size(), data.begin());
#    elif defined(DNDS_DEVICESTORAGE_CUDA_USE_THRUST_PTR)
            DNDS_CUDA_CHECKED(::cudaMemcpy(p_data.get(), host_T_ptr, n_, ::cudaMemcpyHostToDevice));
#    else
            // thrust::copy(host_T_ptr, host_T_ptr + data.size(), data.begin());
            // DNDS_CUDA_CHECKED(::cudaMemcpy(p_data, host_T_ptr, n_, ::cudaMemcpyHostToDevice));
            DNDS_CUDA_DRIVER_CHECKED(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(p_data), host_ptr, n_));
#    endif
        }
        void copy_device_to_host(void *host_ptr, size_t n_bytes) override
        {
            DNDS_check_throw_info(n_bytes == bytes(), "bytes size mismatch");
            auto *host_T_ptr = reinterpret_cast<uint8_t *>(host_ptr);
#    ifdef DNDS_DEVICESTORAGE_CUDA_USE_THRUST
            thrust::copy(data.begin(), data.end(), host_T_ptr);
#    elif defined(DNDS_DEVICESTORAGE_CUDA_USE_THRUST_PTR)
            DNDS_CUDA_CHECKED(::cudaMemcpy(host_T_ptr, p_data.get(), n_bytes, ::cudaMemcpyDeviceToHost));
#    else
            // thrust::copy(data.begin(), data.end(), host_T_ptr);
            // DNDS_CUDA_CHECKED(::cudaMemcpy(host_T_ptr, p_data, n_bytes, ::cudaMemcpyDeviceToHost));
            DNDS_CUDA_DRIVER_CHECKED(cuMemcpyDtoH(host_ptr, reinterpret_cast<CUdeviceptr>(p_data), n_bytes));
#    endif
        }
        t_supDeviceStorageBase clone() override
        {
            return {new self_type(*this), deviceStorageBase_deleter}; // copy CTOR
        }
        [[nodiscard]] size_t bytes() const override
        {
#    ifdef DNDS_DEVICESTORAGE_CUDA_USE_THRUST
            return data.size();
#    elif defined(DNDS_DEVICESTORAGE_CUDA_USE_THRUST_PTR)
            return n_;
#    else
            return n_;
#    endif
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