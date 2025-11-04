#pragma once


#ifdef DNDS_USE_CUDA
#include <thrust/device_malloc_allocator.h>

namespace DNDS::CUDA
{
    template <typename T>
    struct DeviceObject
    {
        thrust::device_ptr<T> dev;
        DeviceObject(const T &host)
        {
            dev = thrust::device_malloc<T>(1);
            cudaMemcpy(dev.get(), &host, sizeof(T), cudaMemcpyHostToDevice);
        }
        ~DeviceObject() { thrust::device_free(dev); }
        T *get() { return dev.get(); }
    };

#    define DNDS_CUDA_DEVICE_VIEW_TMP_COPY(obj) \
        ::DNDS::CUDA::DeviceObject<std::remove_cv_t<std::remove_reference_t<decltype(obj)>>>(obj).get()
}

#endif