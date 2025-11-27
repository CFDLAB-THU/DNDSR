#pragma once

#include "DNDS/Errors.hpp"
#include "Defines.hpp"
#include "ArrayBasic.hpp"
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>

#ifdef DNDS_USE_CUDA
#    include <thrust/device_malloc_allocator.h>
#    include <thrust/device_malloc.h>
#    include <thrust/device_free.h>
#    include <thrust/copy.h>
#    include <thrust/host_vector.h>
#    include <thrust/device_vector.h>
#    include <Eigen/Dense>
#    include <sstream>

#    define DNDS_CUDA_1D_TID_GLOBAL_INDEX ((index)blockIdx.x * (index)blockDim.x + (index)threadIdx.x)

#    define DNDS_CUDA_CHECKED(expr)                                          \
        do                                                                   \
        {                                                                    \
            cudaError_t _err = (expr);                                       \
            if (_err != cudaSuccess)                                         \
            {                                                                \
                std::stringstream ss;                                        \
                ss << "CUDA Error: " << cudaGetErrorString(_err)             \
                   << " (" << _err << ") at " << __FILE__ << ":" << __LINE__ \
                   << " in " << #expr << std::endl;                          \
                DNDS_check_throw_info(_err != cudaSuccess, ss.str());        \
            }                                                                \
        } while (0)

namespace DNDS::CUDA
{
    template <typename T>
    struct DeviceObject
    {
        static_assert(std::is_trivially_copyable_v<T>);
        thrust::device_ptr<T> dev;
        DeviceObject(const T &host)
        {

            dev = thrust::device_malloc<T>(1);
            // cudaMemcpy(dev.get(), &host, sizeof(T), cudaMemcpyHostToDevice);
            DNDS_CUDA_CHECKED(cudaMemcpy(thrust::raw_pointer_cast(dev), &host, sizeof(T), cudaMemcpyHostToDevice));
            // thrust::copy(&host, (&host) + 1, dev);
        }
        ~DeviceObject() { thrust::device_free(dev); }
        T *get() { return dev.get(); }
    };

#    define DNDS_CUDA_DEVICE_VIEW_COPY_OBJ(obj) \
        auto obj##_device_copy = ::DNDS::CUDA::DeviceObject<std::remove_cv_t<std::remove_reference_t<decltype(obj)>>>(obj);
#    define DNDS_CUDA_DEVICE_VIEW_TMP_COPY(obj) \
        ::DNDS::CUDA::DeviceObject<std::remove_cv_t<std::remove_reference_t<decltype(obj)>>>(obj).get()

    inline auto calckernelSizeSimple(index total_threads, uint32_t threadsPerBlock)
    {
        index result = index(total_threads + threadsPerBlock - 1) / index(threadsPerBlock);
        uint32_t blocksPerGrid = 0;
        if (result > 0 && result <= std::numeric_limits<uint32_t>::max())
            blocksPerGrid = result;
        else
            DNDS_assert_info(false, "too many blocks: " + std::to_string(result));
        return std::make_tuple(blocksPerGrid, threadsPerBlock);
    }

    class CudaEvent
    {
    public:
        cudaEvent_t ev;

        CudaEvent(unsigned flags = cudaEventDisableTiming)
        {
            if (cudaEventCreateWithFlags(&ev, flags) != cudaSuccess)
                throw std::runtime_error("Failed to create CUDA event");
        }

        ~CudaEvent() { DNDS_CUDA_CHECKED(cudaEventDestroy(ev)); }

        void record(cudaStream_t stream = 0)
        {
            if (cudaEventRecord(ev, stream) != cudaSuccess)
                throw std::runtime_error("Failed to record CUDA event");
        }

        void sync() const
        {
            if (cudaEventSynchronize(ev) != cudaSuccess)
                throw std::runtime_error("Failed to synchronize CUDA event");
        }

        cudaEvent_t get() { return ev; };
    };

    class CudaStream
    {
        cudaStream_t stream;
        std::unordered_set<ssp<CudaEvent>> waiting_events;

    public:
        CudaStream(unsigned flags = cudaStreamNonBlocking)
        {
            if (cudaStreamCreateWithFlags(&stream, flags) != cudaSuccess)
                throw std::runtime_error("Failed to create CUDA stream");
        }

        static CudaStream &DefaultStream();

        ~CudaStream() { DNDS_CUDA_CHECKED(cudaStreamDestroy(stream)); }

        cudaStream_t get() const { return stream; }

        /// Wait for this stream to finish, using a temporary event
        void wait() const
        {
            CudaEvent ev;
            ev.record(stream);
            ev.sync(); // blocks CPU until stream is done
        }

        void sync()
        {
            cudaStreamSynchronize(stream);
            waiting_events.clear();
        }

        void waitForEvent(const ssp<CudaEvent> &e)
        {
            cudaStreamWaitEvent(stream, e->get());
            waiting_events.insert(e);
        }

        void makeOtherStreamWait(CudaStream &s_other)
        {
            auto e = std::make_shared<CudaEvent>();
            e->record(stream);
            s_other.waitForEvent(e);
        }
    };
}

#endif