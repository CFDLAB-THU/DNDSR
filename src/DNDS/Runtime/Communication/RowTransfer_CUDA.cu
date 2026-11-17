#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Communication/DevicePacking.hpp"
#include <cuda_runtime.h>

namespace DNDS::Runtime
{
    namespace
    {
        template <class T>
        __device__ T Identity(Reduction op)
        {
            if (op == Reduction::Min)
                return std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
            if (op == Reduction::Max)
                return std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
            if constexpr (std::is_integral_v<T>)
                if (op == Reduction::BitAnd)
                    return T(~T(0));
            return T(0);
        }
        template <class T>
        __device__ T Combine(T a, T b, Reduction op)
        {
            if (op == Reduction::Sum)
            {
                if constexpr (std::is_integral_v<T>)
                    return T(std::make_unsigned_t<T>(a) + std::make_unsigned_t<T>(b));
                else
                    return a + b;
            }
            if (op == Reduction::Min)
            {
                if constexpr (std::is_floating_point_v<T>)
                    return fmin(a, b);
                else
                    return a < b ? a : b;
            }
            if (op == Reduction::Max)
            {
                if constexpr (std::is_floating_point_v<T>)
                    return fmax(a, b);
                else
                    return a > b ? a : b;
            }
            if constexpr (std::is_integral_v<T>)
            {
                if (op == Reduction::BitOr)
                    return a | b;
                if (op == Reduction::BitXor)
                    return a ^ b;
                if (op == Reduction::BitAnd)
                    return a & b;
            }
            return a;
        }
        // One active block per destination row, strided components per thread.
        // Stable contributor order avoids atomics and implements all operations.
        template <class T>
        __global__ void Reduce(const T *source, T *destination, const Contribution *groups, size_t count,
                               Reduction op, DestinationInit init)
        {
            for (size_t i = blockIdx.x; i < count; i += gridDim.x)
            {
                const auto group = groups[i];
                if (i && groups[i - 1].destinationElement == group.destinationElement)
                    continue;
                for (size_t j = threadIdx.x; j < group.count; j += blockDim.x)
                {
                    T value = init == DestinationInit::Identity ? Identity<T>(op) : destination[group.destinationElement + j];
                    for (size_t k = i; k < count && groups[k].destinationElement == group.destinationElement; ++k)
                        value = Combine(value, source[groups[k].sourceElement + j], op);
                    destination[group.destinationElement + j] = value;
                }
            }
        }
        template <class T>
        void Install(Reducer &reducer, Reduction op, DestinationInit init)
        {
            reducer.device = [op, init](const void *src, void *dst, const Contribution *groups, size_t count, ExecutionContext context)
            {
                if (!count)
                    return;
                int previous = 0;
                DNDS_check_throw_info(cudaGetDevice(&previous) == cudaSuccess, "CUDA current device query failed");
                DNDS_check_throw_info(cudaSetDevice(context.device.ordinal) == cudaSuccess, "CUDA reduction device selection failed");
                Reduce<<<unsigned(std::min(count, size_t(65535))), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(
                    static_cast<const T *>(src), static_cast<T *>(dst), groups, count, op, init);
                auto status = cudaGetLastError();
                auto restored = cudaSetDevice(previous);
                DNDS_check_throw_info(status == cudaSuccess && restored == cudaSuccess, "CUDA reduction launch failed");
            };
        }
    }
    void InstallBuiltinCUDA(Reducer &reducer, Reduction op, size_t bytes, bool integral, bool isSigned, DestinationInit init)
    {
        if (!integral)
        {
            if (bytes == sizeof(float))
                Install<float>(reducer, op, init);
            if (bytes == sizeof(double))
                Install<double>(reducer, op, init);
        }
        else
        {
#define DNDS_DEVICE_REDUCE(B, S, U)        \
    if (bytes == B)                        \
    {                                      \
        if (isSigned)                      \
            Install<S>(reducer, op, init); \
        else                               \
            Install<U>(reducer, op, init); \
    }
            DNDS_DEVICE_REDUCE(1, int8_t, uint8_t)
            DNDS_DEVICE_REDUCE(2, int16_t, uint16_t)
            DNDS_DEVICE_REDUCE(4, int32_t, uint32_t)
            DNDS_DEVICE_REDUCE(8, int64_t, uint64_t)
#undef DNDS_DEVICE_REDUCE
        }
    }
}

namespace DNDS::Runtime::Detail
{
    __global__ void PackRows(const uint8_t *source, uint8_t *destination, const DevicePackRow *rows, index rowCount,
                             const DevicePackField *fields, index fieldCount, index extent, bool unpack)
    {
        for (index iRow = blockIdx.x; iRow < rowCount; iRow += gridDim.x)
        {
            const auto row = rows[iRow];
            for (index iField = 0; iField < fieldCount; ++iField)
            {
                const auto field = fields[iField];
                for (index item = threadIdx.x; item < row.width * field.bytes; item += blockDim.x)
                {
                    const index component = item / field.bytes, byte = field.offset + item % field.bytes;
                    const index arrayByte = (row.base + component * row.stride) * extent + byte;
                    const index packedByte = (row.packed + component) * extent + byte;
                    if (unpack)
                        destination[arrayByte] = source[packedByte];
                    else
                        destination[packedByte] = source[arrayByte];
                }
            }
        }
    }
    void LaunchDevicePacking(ExecutionContext context, const void *source, void *destination, const DevicePackRow *rows,
                             index rowCount, const DevicePackField *fields, index fieldCount, index extent, bool unpack)
    {
        if (!rowCount)
            return;
        int previous = 0;
        DNDS_check_throw_info(context.device.space == MemorySpace::CUDA && cudaGetDevice(&previous) == cudaSuccess, "CUDA packing context unavailable");
        DNDS_check_throw_info(cudaSetDevice(context.device.ordinal) == cudaSuccess, "CUDA packing device selection failed");
        PackRows<<<unsigned(std::min<index>(rowCount, 65535)), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(
            static_cast<const uint8_t *>(source), static_cast<uint8_t *>(destination), rows, rowCount, fields, fieldCount, extent, unpack);
        const auto launched = cudaGetLastError(), restored = cudaSetDevice(previous);
        DNDS_check_throw_info(launched == cudaSuccess && restored == cudaSuccess, "CUDA packing launch failed");
    }
}
