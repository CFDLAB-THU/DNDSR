#include "DNDS/Runtime/Execution/CUDAStream.hpp"
// Independent small dense-product oracle and timing for the shared FP64 adapter.
#include "DNDS/Algebra/CUDA/FP64Matrix.cuh"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace R = DNDS::Runtime;
using I = DNDS::index;
__global__ void ScalarProduct(R::ArrayView<const double> a, R::ArrayView<const double> b, R::ArrayView<double> c, I n, I components)
{
    // Match the tensor kernel's warp/cell mapping, so the comparison does not
    // credit MMA for merely processing four cells rather than one per block.
    const I lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (I iCell = I(blockIdx.x) * 4 + warp; iCell < c.rows; iCell += I(gridDim.x) * 4)
        for (I i = lane; i < c.width; i += 32)
        {
            const I mode = i / components, component = i % components;
            double value = 0;
            for (I k = 0; k < n; ++k)
                value += a(iCell, mode * n + k) * b(iCell, k * components + component);
            c(iCell, i) = value;
        }
}
__global__ void TensorProduct(R::ArrayView<const double> a, R::ArrayView<const double> b, R::ArrayView<double> c, I n, I components)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ DNDS::Algebra::CUDA::FP64Tile8 tiles[4];
    const I lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    auto &tile = tiles[warp];
    for (I iCell = I(blockIdx.x) * 4 + warp; iCell < c.rows; iCell += I(gridDim.x) * 4)
        for (I modeBegin = 0; modeBegin < n; modeBegin += 8)
            for (I componentBegin = 0; componentBegin < components; componentBegin += 8)
            {
                DNDS::Algebra::CUDA::FP64Accumulator8 product;
                for (I kBegin = 0; kBegin < n; kBegin += 4)
                {
                    const I mode = modeBegin + lane / 4, kA = kBegin + lane % 4;
                    const I component = componentBegin + lane % 8, kB = kBegin + lane / 8;
                    tile.a[lane] = mode < n && kA < n ? a(iCell, mode * n + kA) : 0;
                    tile.b[lane] = component < components && kB < n ? b(iCell, kB * components + component) : 0;
                    __syncwarp();
                    product.AddProduct(tile);
                    __syncwarp();
                }
                product.Store(tile);
                __syncwarp();
                for (I i = lane; i < 64; i += 32)
                {
                    const I mode = modeBegin + i / 8, component = componentBegin + i % 8;
                    if (mode < n && component < components)
                        c(iCell, mode * components + component) = tile.c[i];
                }
                __syncwarp();
            }
#else
    asm("trap;");
#endif
}
int main(int argc, char **argv)
{
    try
    {
        DNDS_check_throw_info(argc == 1 || argc == 2, "argument: optional row count");
        const I rows = argc == 2 ? std::stoll(argv[1]) : 65536;
        DNDS_check_throw_info(rows > 0, "row count must be positive");
        cudaDeviceProp properties;
        DNDS_check_throw_info(cudaGetDeviceProperties(&properties, 0) == cudaSuccess && properties.major == 8 && properties.minor == 0,
                              "this FP64 microbenchmark requires the validated sm80 device");
        R::EagerEngine engine;
        auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        for (I n : {I(2), I(5), I(9), I(19)})
            for (I components : {I(5), I(7), I(21), I(37)})
            {
                R::Array<double> hostA(R::RowLayout::Dense(rows, n * n)), hostB(R::RowLayout::Dense(rows, n * components));
                {
                    auto a = hostA.Write(), b = hostB.Write();
                    for (I iCell = 0; iCell < rows; ++iCell)
                    {
                        for (I j = 0; j < n * n; ++j)
                            a.View()(iCell, j) = .01 * ((iCell * 7 + j * 13) % 37 - 18);
                        for (I j = 0; j < n * components; ++j)
                            b.View()(iCell, j) = .02 * ((iCell * 3 + j * 17) % 41 - 20);
                    }
                }
                auto a = hostA.To(R::Device::CUDA(0)), b = hostB.To(R::Device::CUDA(0));
                R::Array<double> scalar(R::RowLayout::Dense(rows, n * components), R::Device::CUDA(0));
                R::Array<double> tensor(R::RowLayout::Dense(rows, n * components), R::Device::CUDA(0));
                auto submit = [&](bool useTensor)
                {
                    R::SubmitArrays(engine, stream, [=](auto context, auto a, auto b, auto c)
                                    {
                        if (useTensor)
                            TensorProduct<<<256,128,0,::DNDS::Runtime::CUDAStream(context)>>>(a, b, c, n, components);
                        else
                            ScalarProduct<<<256,128,0,::DNDS::Runtime::CUDAStream(context)>>>(a, b, c, n, components);
                        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "FP64 product launch failed"); }, R::Read(a), R::Read(b), R::Write(useTensor ? tensor : scalar));
                };
                for (int warm = 0; warm < 3; ++warm)
                {
                    submit(false);
                    submit(true);
                }
                engine.Synchronize();
                const auto reference = scalar.To(R::Device::Host()), result = tensor.To(R::Device::Host());
                const auto s = reference.Read(), t = result.Read(), ar = hostA.Read(), br = hostB.Read();
                const auto sv = s.View(), tv = t.View(), av = ar.View(), bv = br.View();
                double maximum = 0;
                for (I iCell = 0; iCell < rows; ++iCell)
                    for (I j = 0; j < n * components; ++j)
                        maximum = std::max(maximum, std::abs(sv(iCell, j) - tv(iCell, j)));
                // CPU long-double oracle independent of either GPU mapping.
                for (I iCell : {I(0), rows / 2, rows - 1})
                    for (I mode = 0; mode < n; ++mode)
                        for (I component = 0; component < components; ++component)
                        {
                            long double expected = 0;
                            for (I k = 0; k < n; ++k)
                                expected += static_cast<long double>(av(iCell, mode * n + k)) * bv(iCell, k * components + component);
                            DNDS_check_throw_info(std::abs(tv(iCell, mode * components + component) - double(expected)) < 1e-13,
                                                  "FP64 product CPU oracle mismatch");
                        }
                DNDS_check_throw_info(maximum < 1e-13, "FP64 product mapping mismatch");
                for (bool useTensor : {false, true})
                    for (int trial = 0; trial < 3; ++trial)
                    {
                        cudaEvent_t begin, end;
                        DNDS_check_throw_info(cudaEventCreate(&begin) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess, "timing event creation failed");
                        auto native = ::DNDS::Runtime::CUDAStream(stream.Context());
                        DNDS_check_throw_info(cudaEventRecord(begin, native) == cudaSuccess, "timing begin failed");
                        for (int repeat = 0; repeat < 20; ++repeat)
                            submit(useTensor);
                        DNDS_check_throw_info(cudaEventRecord(end, native) == cudaSuccess, "timing end failed");
                        engine.Synchronize();
                        float milliseconds = 0;
                        DNDS_check_throw_info(cudaEventElapsedTime(&milliseconds, begin, end) == cudaSuccess, "timing event failed");
                        cudaEventDestroy(begin);
                        cudaEventDestroy(end);
                        std::cout << "FP64_PRODUCT {\"rows\":" << rows << ",\"modes\":" << n << ",\"components\":" << components
                                  << ",\"kernel\":\"" << (useTensor ? "tensor" : "fp64") << "\",\"trial\":" << trial
                                  << ",\"seconds\":" << double(milliseconds) / 20000 << ",\"max_error\":" << maximum << "}" << std::endl;
                    }
            }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
