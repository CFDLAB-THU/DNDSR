#pragma once

#include "Defines.hpp"
#include "ArrayBasic.hpp"

#ifdef DNDS_USE_CUDA
#    include <thrust/device_malloc_allocator.h>
#    include <thrust/device_malloc.h>
#    include <thrust/device_free.h>
#    include <thrust/copy.h>
#    include <thrust/host_vector.h>
#    include <thrust/device_vector.h>
#    include <Eigen/Dense>

namespace DNDS::CUDA
{
    template <typename T>
    struct DeviceObject
    {
        thrust::device_ptr<T> dev;
        DeviceObject(const T &host)
        {
            dev = thrust::device_malloc<T>(1);
            // cudaMemcpy(dev.get(), &host, sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(thrust::raw_pointer_cast(dev), &host, sizeof(T), cudaMemcpyHostToDevice);
            // thrust::copy(&host, (&host) + 1, dev);
        }
        ~DeviceObject() { thrust::device_free(dev); }
        T *get() { return dev.get(); }
    };

#    define DNDS_CUDA_DEVICE_VIEW_TMP_COPY(obj) \
        ::DNDS::CUDA::DeviceObject<std::remove_cv_t<std::remove_reference_t<decltype(obj)>>>(obj).get()
}

#endif