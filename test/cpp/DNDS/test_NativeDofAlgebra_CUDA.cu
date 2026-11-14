#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Algebra/ArrayInputCache.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include <cuda_runtime.h>

namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;

TEST_CASE("CUDA operator input cache orders queued conversion before its consumer")
{
    constexpr DNDS::index rows = 33, width = 95;
    R::Array<DNDS::real> host(R::RowLayout::Dense(rows, width, width + 3));
    {
        auto write = host.Write();
        for (DNDS::index iRow = 0; iRow < rows; ++iRow)
            for (DNDS::index j = 0; j < width; ++j)
                write.View()(iRow, j) = .25 * iRow - .125 * j;
    }
    R::EagerEngine engine;
    const auto producer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto consumer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto source = host.To(R::Device::CUDA(0));
    R::Array<DNDS::real> output(R::RowLayout::Dense(rows, width), source.GetDevice());
    A::ArrayInputCache cache(source, "test/CUDA/tiledInput");
    const auto allocation = cache.Values().Storage().Identity();
    R::SubmitArrays(engine, producer, [](auto context, auto values)
                    { A::ApplyRowViews(context, A::PointwiseOperation::Scale, values.DynamicView(), {}, {}, 2); }, R::ReadWriteAs<R::UniformTableRows>(source));
    consumer.WaitFor(producer.Record());
    CHECK(cache.Refresh(engine, consumer, source));
    CHECK_FALSE(cache.Refresh(engine, consumer, source));
    R::SubmitArrays(engine, consumer, [](auto context, auto input, auto result)
                    { A::ApplyRowViews(context, A::PointwiseOperation::Copy, result.DynamicView(), input.DynamicView()); }, R::ReadAs<R::TiledColMajorTable<32>>(cache.Values()), R::WriteAs<R::TableRows>(output));
    engine.Synchronize();
    CHECK(cache.Values().Storage().Identity() == allocation);
    const auto actual = output.To(R::Device::Host());
    const auto read = actual.Read(), reference = host.Read();
    for (DNDS::index iRow = 0; iRow < rows; ++iRow)
        for (DNDS::index j = 0; j < width; ++j)
            CHECK(read.View()(iRow, j) == 2 * reference.View()(iRow, j));
}

TEST_CASE_TEMPLATE("Native CUDA algebra retains float/double fields, joins streams and excludes padding/ghosts", T, float, double)
{
    R::EagerEngine engine;
    auto producer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto consumer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    for (size_t width : {size_t(5), size_t(6), size_t(7), size_t(114)})
        for (size_t rows : {size_t(0), size_t(257)})
        {
            CAPTURE(width);
            CAPTURE(rows);
            A::DofSpace space(rows, 3);
            A::DofField<T> host(space, width, width + 3);
            {
                auto owned = host.Owned().Storage().Write(), ghost = host.Ghost().Storage().Write();
                std::fill_n(owned.template View<T>().data, host.Owned().Layout().Elements(), T(-919));
                std::fill_n(ghost.template View<T>().data, host.Ghost().Layout().Elements(), T(-727));
            }
            auto x = std::make_unique<A::DofField<T>>(space, host.Owned().To(R::Device::CUDA(0)), host.Ghost().To(R::Device::CUDA(0)));
            A::DofField<T> y(space, host.Owned().To(R::Device::CUDA(0)), host.Ghost().To(R::Device::CUDA(0)));
            A::Fill(engine, producer, *x, T(3));
            A::Scale(engine, producer, *x, T(2));
            consumer.WaitFor(producer.Record());
            A::Copy(engine, consumer, y, *x);
            A::Axpy(engine, consumer, y, T(-0.5), *x);
            A::LinearCombination(engine, consumer, y, T(2), y, T(1), *x);
            x.reset();
            auto dot = A::DotLocal(engine, consumer, y, y);
            auto norm = A::NormLocal(engine, consumer, y);
            const auto sum = A::ComponentSumLocal(engine, consumer, y).Get();
            CHECK(dot.Get() == doctest::Approx(double(rows * width) * 144));
            CHECK(norm.Get() == doctest::Approx(12 * std::sqrt(double(rows * width))));
            for (double value : sum)
                CHECK(value == double(rows) * 12);
            engine.Synchronize();
            const auto owned = y.Owned().To(R::Device::Host()), ghost = y.Ghost().To(R::Device::Host());
            const auto raw = owned.Storage().Read(), ghosts = ghost.Storage().Read();
            for (size_t row = 0; row < rows; ++row)
            {
                for (size_t component = 0; component < width; ++component)
                    CHECK(raw.template View<T>().data[row * (width + 3) + component] == T(12));
                for (size_t component = width; component < width + 3; ++component)
                    CHECK(raw.template View<T>().data[row * (width + 3) + component] == T(-919));
            }
            for (size_t item = 0; item < ghost.Layout().Elements(); ++item)
                CHECK(ghosts.template View<T>().data[item] == T(-727));
            {
                auto write = host.Owned().Write();
                for (size_t row = 0; row < rows; ++row)
                    for (size_t component = 0; component < width; ++component)
                        write.At(row, component) = T(-3.25 + double(row % 17) / 2 - double(component) / 4);
            }
            A::DofField<T> varied(space, host.Owned().To(R::Device::CUDA(0)), host.Ghost().To(R::Device::CUDA(0)));
            const auto minima = A::ComponentMinLocal(engine, consumer, varied).Get();
            const auto maxima = A::ComponentMaxLocal(engine, consumer, varied).Get();
            const auto sums = A::ComponentSumLocal(engine, consumer, varied).Get();
            double expectedSquares = 0;
            for (size_t component = 0; component < width; ++component)
            {
                const double base = -3.25 - double(component) / 4;
                const size_t full = rows / 17, tail = rows % 17;
                const double rowSum = double(full * 136) + double(tail) * double(tail ? tail - 1 : 0) / 2;
                CHECK(sums[component] == double(rows) * base + rowSum / 2);
                CHECK(minima[component] == (rows ? base : std::numeric_limits<double>::infinity()));
                CHECK(maxima[component] == (rows ? base + double(std::min<size_t>(rows - 1, 16)) / 2 : -std::numeric_limits<double>::infinity()));
                for (size_t row = 0; row < rows; ++row)
                {
                    const double value = base + double(row % 17) / 2;
                    expectedSquares += value * value;
                }
            }
            CHECK(A::DotLocal(engine, consumer, varied, varied).Get() == doctest::Approx(expectedSquares));
            engine.Synchronize();
            // Idle synchronous calls resolve storage placement too. They
            // finish before returning, while explicit stream overloads retain
            // their queueing contract. Neither touches ghosts or row padding.
            A::Scale(y, T(2));
            CHECK(A::DotLocal(y, y) == doctest::Approx(double(rows * width) * 576));
            const auto immediateSums = A::ComponentSumLocal(y);
            for (double value : immediateSums)
                CHECK(value == double(rows) * 24);
            A::Fill(y, T(3));
            const auto immediateMin = A::ComponentMinLocal(y);
            const auto immediateMax = A::ComponentMaxLocal(y);
            for (DNDS::index j = 0; j < DNDS::index(width); ++j)
            {
                CHECK(immediateMin[j] == (rows ? 3 : std::numeric_limits<double>::infinity()));
                CHECK(immediateMax[j] == (rows ? 3 : -std::numeric_limits<double>::infinity()));
            }
            auto cpu = engine.CreateCPUStream();
            CHECK_THROWS(A::Fill(engine, cpu, y, T(1)));
        }
    A::DofSpace fixedSpace(13, 2);
    A::DofField<T, 6> fixed(fixedSpace, 6, 9, R::Device::CUDA(0));
    A::Fill(engine, producer, fixed, T(-2));
    A::Scale(engine, producer, fixed, T(3));
    consumer.WaitFor(producer.Record());
    A::Axpy(engine, consumer, fixed, T(-0.5), fixed);
    CHECK(A::DotLocal(engine, consumer, fixed, fixed).Get() == 13 * 6 * 9);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
}
