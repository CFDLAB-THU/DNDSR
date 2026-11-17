#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Implicit/CUDA/LinearDiagonalKernels.hpp"
#include "DNDS/Algebra/BatchedSVD.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "LinearInverseReference.hpp"
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <iostream>

namespace R = DNDS::Runtime;
namespace K = DNDS::Cauchy::CUDA;
using I = DNDS::index;
using DNDS::real;

namespace
{
    void Check(cudaError_t result)
    {
        DNDS_check_throw_info(result == cudaSuccess, cudaGetErrorString(result));
    }
    real Value(I row, int i, int j, I fallbackStride)
    {
        const real scale = 1 + .125 * i;
        return scale * (fallbackStride && row % fallbackStride == 0 ? 1
                        : i == j                                    ? 3 + .125 * i
                                                                    : .01 * (1 + 2 * i - j));
    }
}

/** Focused replay, not a solver convergence test. The input is immutable and
 * every algorithm overwrites its own output/scratch. Event timing includes
 * the actual sequence of provider/kernel launches, without a wait per row or
 * per iteration. Host output copies and independent Eigen checks occur after
 * timing. No MPI or solver orchestration is introduced in the numerical leaves.
 */
int main(int argc, char **argv)
{
    try
    {
        const int ordinal = argc > 1 ? std::stoi(argv[1]) : 0;
        const int n = argc > 2 ? std::stoi(argv[2]) : 5;
        const I rows = argc > 3 ? std::stoll(argv[3]) : 262144;
        const int iterations = argc > 4 ? std::stoi(argv[4]) : 10;
        const I fallbackStride = argc > 5 ? std::stoll(argv[5]) : 0;
        DNDS_check_throw_info(n >= 5 && n <= 7 && rows > 0 && iterations > 0 && fallbackStride >= 0,
                              "arguments: device width(5/6/7) rows iterations fallback_stride(0=healthy)");
        Check(cudaSetDevice(ordinal));
        R::EagerEngine engine;
        const auto lane = engine.CreateStream(std::make_shared<R::CUDALauncher>(ordinal));
        const auto context = lane.Context();
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        R::Array<real> host(R::RowLayout::Dense(rows, n * n));
        {
            auto write = host.Write();
            for (I iCell = 0; iCell < rows; ++iCell)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                        write.View()(iCell, i + n * j) = Value(iCell, i, j, fallbackStride);
        }
        const auto diagonal = host.To(context.device, context);
        R::Array<real> output(host.Layout(), context.device);
        R::Array<DNDS::NumericalStatus> status(R::RowLayout::Dense(rows, 1), context.device);
        R::Array<int> fallback(R::RowLayout::Dense(rows, 1), context.device), info(fallback.Layout(), context.device), truncated(fallback.Layout(), context.device);
        R::Array<I> counts(R::RowLayout::Dense(1, 1), context.device);
        R::Array<real> normalized(host.Layout(), context.device), left(host.Layout(), context.device), right(host.Layout(), context.device);
        R::Array<real> scales(R::RowLayout::Dense(rows, n), context.device), singular(scales.Layout(), context.device);
        std::shared_ptr<DNDS::Algebra::BatchedSVD> svd;
        {
            auto a = normalized.Write(context), s = singular.Write(context), u = left.Write(context), v = right.Write(context);
            svd = std::make_shared<DNDS::Algebra::BatchedSVD>(context, n, DNDS::Algebra::BatchedSVDViews{a.View(), s.View(), u.View(), v.View()});
        }
        R::Array<real> work(R::RowLayout::Dense(svd->WorkspaceElements(), 1), context.device);
        cudaEvent_t begin = nullptr, end = nullptr;
        Check(cudaEventCreate(&begin));
        Check(cudaEventCreate(&end));
        for (int algorithm = 0; algorithm < 2; ++algorithm)
        {
            std::array<real, 3> deviceMilliseconds{}, wallMilliseconds{};
            {
                const auto d = diagonal.Read(context);
                auto x = output.Write(context);
                auto state = status.Write(context);
                auto mask = fallback.Write(context);
                auto count = counts.Write(context);
                auto a = normalized.Write(context), u = left.Write(context), v = right.Write(context);
                auto s = singular.Write(context), scale = scales.Write(context), workspace = work.Write(context);
                auto flags = info.Write(context), modes = truncated.Write(context);
                const auto issue = [&]
                {
                    if (algorithm == 1)
                        K::InvertSmallBlocks(context, d.View<R::UniformTableRows>(), x.View<R::UniformTableRows>(),
                                             state.View<R::FixedRows<1>>(), mask.View<R::FixedRows<1>>(),
                                             count.View<R::FixedRows<1>>(), n);
                    else
                    {
                        K::PrepareDiagonalSVD(context, d.View<R::UniformTableRows>(), a.View<R::UniformTableRows>(), scale.View<R::UniformTableRows>(), state.View<R::FixedRows<1>>(), n);
                        svd->SubmitBorrowed(context, {a.View(), s.View(), u.View(), v.View()}, workspace.View(), flags.View());
                        K::CompleteDiagonalSVD(context, d.View<R::UniformTableRows>(), s.View<R::UniformTableRows>().ConstView(),
                                               u.View<R::UniformTableRows>().ConstView(), v.View<R::UniformTableRows>().ConstView(),
                                               scale.View<R::UniformTableRows>().ConstView(), flags.View<R::FixedRows<1>>().ConstView(),
                                               x.View<R::UniformTableRows>(), state.View<R::FixedRows<1>>(), modes.View<R::FixedRows<1>>(), n);
                    }
                };
                // Const projections borrow the same exclusive scopes; no new
                // lease or synchronization occurs between ordered kernels.
                for (int warm = 0; warm < 3; ++warm)
                    issue();
                Check(cudaStreamSynchronize(stream));
                for (int repeat = 0; repeat < 3; ++repeat)
                {
                    const auto started = std::chrono::steady_clock::now();
                    Check(cudaEventRecord(begin, stream));
                    for (int iteration = 0; iteration < iterations; ++iteration)
                        issue();
                    Check(cudaEventRecord(end, stream));
                    Check(cudaEventSynchronize(end));
                    float elapsed = 0;
                    Check(cudaEventElapsedTime(&elapsed, begin, end));
                    deviceMilliseconds[repeat] = elapsed / iterations;
                    wallMilliseconds[repeat] = std::chrono::duration<real, std::milli>(std::chrono::steady_clock::now() - started).count() / iterations;
                }
            }
            const auto answer = output.To(R::Device::Host());
            const auto rowStatus = status.To(R::Device::Host());
            const auto read = answer.Read();
            const auto states = rowStatus.Read();
            real maximumError = 0;
            for (I iCell = 0; iCell < rows; ++iCell)
            {
                DNDS_check_throw_info(bool(states.View()(iCell, 0)), "inverse replay reported a numerical failure");
                for (int j = 0; j < n * n; ++j)
                    DNDS_check_throw_info(std::isfinite(read.View()(iCell, j)), "nonfinite inverse replay output");
                if (iCell < 16 || iCell == rows - 1)
                {
                    Eigen::MatrixXd matrix(n, n);
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i)
                            matrix(i, j) = Value(iCell, i, j, fallbackStride);
                    const auto expected = CauchyTest::DiagonalInverseReference(matrix);
                    DNDS_check_throw_info(expected.status, "independent CPU inverse failed");
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i)
                            maximumError = std::max(maximumError, std::abs(read.View()(iCell, i + n * j) - expected.value(i, j)));
                }
            }
            DNDS_check_throw_info(maximumError < 2e-11, "inverse replay differs from CPU reference");
            std::cout << "SMALL_INVERSE " << (algorithm ? "guardedLU" : "batchedSVD")
                      << " width=" << n << " rows=" << rows << " fallback_stride=" << fallbackStride
                      << " max_abs_error=" << maximumError << " device_ms=";
            for (const auto time : deviceMilliseconds)
                std::cout << time << ',';
            std::cout << " wall_ms=";
            for (const auto time : wallMilliseconds)
                std::cout << time << ',';
            std::cout << '\n';
        }
        Check(cudaEventDestroy(end));
        Check(cudaEventDestroy(begin));
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
