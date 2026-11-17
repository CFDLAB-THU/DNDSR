#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace DNDS::Algebra::Detail
{
    template <class T>
    __global__ void PointwiseKernel(Pointwise operation, Runtime::ArrayView<T> output,
                                    Runtime::ArrayView<const T> x, Runtime::ArrayView<const T> y, T a, T b)
    {
        const auto count = output.rows * output.width;
        for (size_t item = size_t(blockIdx.x) * blockDim.x + threadIdx.x; item < count; item += size_t(gridDim.x) * blockDim.x)
        {
            const auto row = item / output.width, component = item % output.width;
            switch (operation)
            {
            case Pointwise::Fill:
                output(row, component) = a;
                break;
            case Pointwise::Copy:
                output(row, component) = x(row, component);
                break;
            case Pointwise::Scale:
                output(row, component) *= a;
                break;
            case Pointwise::Axpy:
                output(row, component) += a * x(row, component);
                break;
            case Pointwise::Combination:
            {
                const T left = x(row, component), right = y(row, component);
                output(row, component) = a * left + b * right;
                break;
            }
            }
        }
    }
    template <class T>
    __global__ void ReductionKernel(Reduction operation, Runtime::ArrayView<const T> x,
                                    Runtime::ArrayView<const T> y, Runtime::ArrayView<double> result)
    {
        // One deterministic row reduction per component. No atomics, device
        // MPI or payload-sized scratch; this can later gain a tiled backend.
        for (size_t component = size_t(blockIdx.x) * blockDim.x + threadIdx.x; component < result.width; component += size_t(gridDim.x) * blockDim.x)
        {
            double value = operation == Reduction::Min ? CUDART_INF : operation == Reduction::Max ? -CUDART_INF
                                                                                                  : 0;
            for (size_t row = 0; row < x.rows; ++row)
            {
                if (operation == Reduction::Dot)
                    for (size_t column = 0; column < x.width; ++column)
                        value += double(x(row, column)) * double(y(row, column));
                else
                {
                    const double item = x(row, component);
                    if (operation == Reduction::Min)
                        value = fmin(value, item);
                    else if (operation == Reduction::Max)
                        value = fmax(value, item);
                    else
                        value += operation == Reduction::Squares ? item * item : item;
                }
            }
            result(0, component) = value;
        }
    }
    template <class T>
    void Launch(Runtime::ExecutionContext context, Pointwise operation, Runtime::ArrayView<T> output,
                Runtime::ArrayView<const T> x, Runtime::ArrayView<const T> y, double a, double b)
    {
        DNDS_check_throw_info(context.device.space == Runtime::MemorySpace::CUDA, "CUDA algebra requires a CUDA context");
        Runtime::ValidateViewDevices(context.device, output, x, y);
        Runtime::CheckKernelArgumentBudget<decltype(operation), decltype(output), decltype(x), decltype(y), T, T>();
        const auto count = output.rows * output.width;
        if (!count)
            return;
        const unsigned blocks = unsigned(std::min<size_t>(256, (count + 255) / 256));
        PointwiseKernel<<<blocks, 256, 0, ::DNDS::Runtime::CUDAStream(context)>>>(operation, output, x, y, T(a), T(b));
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "DOF algebra pointwise CUDA launch failed");
    }
    template <class T>
    void LaunchReduction(Runtime::ExecutionContext context, Reduction operation, Runtime::ArrayView<const T> x,
                         Runtime::ArrayView<const T> y, Runtime::ArrayView<double> result)
    {
        DNDS_check_throw_info(context.device.space == Runtime::MemorySpace::CUDA, "CUDA reduction requires a CUDA context");
        Runtime::ValidateViewDevices(context.device, x, y, result);
        Runtime::CheckKernelArgumentBudget<decltype(operation), decltype(x), decltype(y), decltype(result)>();
        const unsigned blocks = unsigned(std::min<size_t>(256, (result.width + 127) / 128));
        ReductionKernel<<<blocks, 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(operation, x, y, result);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "DOF algebra reduction CUDA launch failed");
    }
    void ApplyCUDA(Runtime::ExecutionContext c, Pointwise op, Runtime::ArrayView<float> out, Runtime::ArrayView<const float> x, Runtime::ArrayView<const float> y, double a, double b) { Launch(c, op, out, x, y, a, b); }
    void ApplyCUDA(Runtime::ExecutionContext c, Pointwise op, Runtime::ArrayView<double> out, Runtime::ArrayView<const double> x, Runtime::ArrayView<const double> y, double a, double b) { Launch(c, op, out, x, y, a, b); }
    void ReduceCUDA(Runtime::ExecutionContext c, Reduction op, Runtime::ArrayView<const float> x, Runtime::ArrayView<const float> y, Runtime::ArrayView<double> result) { LaunchReduction(c, op, x, y, result); }
    void ReduceCUDA(Runtime::ExecutionContext c, Reduction op, Runtime::ArrayView<const double> x, Runtime::ArrayView<const double> y, Runtime::ArrayView<double> result) { LaunchReduction(c, op, x, y, result); }
}
