#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <algorithm>
#include <cmath>

namespace R = DNDS::Runtime;
using I = DNDS::index;

TEST_CASE("Native flow-five AMG preserves an uncoupled zero transverse equation")
{
    constexpr I rows = 47, width = 5, entries = 3 * rows - 2;
    R::Array<int> offsets(R::RowLayout::Dense(rows + 1, 1)), columns(R::RowLayout::Dense(entries, 1));
    R::Array<double> values(R::RowLayout::Dense(entries * width * width, 1)), rhs(R::RowLayout::Dense(rows * width, 1));
    {
        auto o = offsets.Write(), c = columns.Write();
        auto a = values.Write(), b = rhs.Write();
        I entry = 0;
        for (I iRow = 0; iRow < rows; ++iRow)
        {
            o.View()(iRow, 0) = int(entry);
            for (I neighbor = std::max<I>(0, iRow - 1); neighbor <= std::min<I>(rows - 1, iRow + 1); ++neighbor)
            {
                c.View()(entry, 0) = int(neighbor);
                for (I i = 0; i < width; ++i)
                    for (I j = 0; j < width; ++j)
                        a.View()(entry * width * width + i * width + j, 0) = neighbor != iRow   ? (i == j ? -1 : 0)
                                                                             : i == j           ? 4
                                                                             : i == 3 || j == 3 ? 0
                                                                                                : -.1;
                ++entry;
            }
            for (I j = 0; j < width; ++j)
                b.View()(iRow * width + j, 0) = j == 3 ? 0 : 1 + .01 * (iRow + j);
        }
        o.View()(rows, 0) = int(entry);
        REQUIRE(entry == entries);
    }
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    DNDS::Algebra::AmgXSettings settings;
    SUBCASE("2D flow aggregate size") { settings.aggregateSize = 4; }
    SUBCASE("3D flow aggregate size") { settings.aggregateSize = 8; }
    DNDS::Algebra::AmgXPreconditioner provider(device, settings);
    const auto o = offsets.Read(), c = columns.Read();
    provider.SetPattern(offsets, columns, int(width));
    REQUIRE(provider.RowNumber() == rows * width);
    auto a = values.To(device), b = rhs.To(device);
    R::Array<double> x(R::RowLayout::Dense(rows * width, 1), device);
    {
        const auto ar = a.Read(context), br = b.Read(context);
        auto xw = x.Write(context);
        provider.UpdateValuesBorrowed(context, ar.View());
        provider.ApplyBorrowed(context, br.View(), xw.View());
    }
    const auto host = x.To(R::Device::Host());
    const auto xr = host.Read(), br = rhs.Read(), ar = values.Read();
    double initial = 0, residual = 0;
    for (I iRow = 0; iRow < rows; ++iRow)
    {
        CHECK(std::abs(xr.View()(iRow * width + 3, 0)) < 1e-14);
        for (I i = 0; i < width; ++i)
        {
            double defect = br.View()(iRow * width + i, 0);
            for (I k = o.View()(iRow, 0); k < o.View()(iRow + 1, 0); ++k)
                for (I j = 0; j < width; ++j)
                    defect -= ar.View()(k * width * width + i * width + j, 0) * xr.View()(c.View()(k, 0) * width + j, 0);
            initial += std::pow(br.View()(iRow * width + i, 0), 2);
            residual += defect * defect;
        }
    }
    CHECK(residual < .1 * initial);
}

TEST_CASE("Single-device AmgX applies a fixed linear local preconditioner and refreshes values")
{
    REQUIRE(DNDS::Algebra::AmgXPreconditioner::Available());
    // Each physical cell remains one native block row through aggregation,
    // restriction and smoothing, including the RANS widths six and seven.
    for (const I components : {1, 2, 3, 4, 5, 6, 7, 8, 9, 19})
    {
        INFO("components = ", components);
        constexpr I cells = 31;
        const I n = cells * components, nnz = 3 * cells - 2;
        R::Array<int> offsets(R::RowLayout::Dense(cells + 1, 1)), columns(R::RowLayout::Dense(nnz, 1));
        R::Array<double> hostValues(R::RowLayout::Dense(nnz * components * components, 1)), hostRhs(R::RowLayout::Dense(n, 1));
        {
            auto o = offsets.Write(), c = columns.Write();
            auto a = hostValues.Write(), r = hostRhs.Write();
            I entry = 0;
            for (I iCell = 0; iCell < cells; ++iCell)
            {
                o.View()(iCell, 0) = int(entry);
                for (I neighbor = std::max<I>(0, iCell - 1); neighbor <= std::min<I>(cells - 1, iCell + 1); ++neighbor)
                {
                    c.View()(entry, 0) = int(neighbor);
                    for (I i = 0; i < components; ++i)
                        for (I j = 0; j < components; ++j)
                            a.View()(entry * components * components + i * components + j, 0) =
                                neighbor == iCell ? (i == j ? 4 + .1 * std::max<I>(0, components - 5) : -.1) : (i == j ? -1 : 0);
                    ++entry;
                }
                for (I j = 0; j < components; ++j)
                    r.View()(iCell * components + j, 0) = 1 + .1 * (iCell * components + j);
            }
            o.View()(cells, 0) = int(entry);
            REQUIRE(entry == nnz);
        }
        const auto device = R::Device::CUDA(0);
        DNDS::Algebra::AmgXPreconditioner provider(device);
        {
            auto o = offsets.Read(), c = columns.Read();
            provider.SetPattern(offsets, columns, int(components));
        }
        auto values = hostValues.To(device), rhs = hostRhs.To(device);
        R::Array<double> solution(R::RowLayout::Dense(n, 1), device);
        auto apply = [&](bool update)
        {
            R::EagerEngine engine;
            auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            if (update)
                R::SubmitArrays(engine, stream, [&provider](auto context, auto a)
                                { provider.UpdateValuesBorrowed(context, a); }, R::Read(values));
            R::SubmitArrays(engine, stream, [&provider](auto context, auto r, auto z)
                            { provider.ApplyBorrowed(context, r, z); }, R::Read(rhs), R::Write(solution));
            engine.Close();
            return solution.To(R::Device::Host());
        };
        const auto first = apply(true);
        const auto firstRead = first.Read(), rRead = hostRhs.Read();
        const auto offsetRead = offsets.Read(), columnRead = columns.Read();
        double initial = 0, final = 0;
        for (I iRow = 0; iRow < n; ++iRow)
        {
            const auto v = firstRead.View();
            double residual = rRead.View()(iRow, 0);
            const I cell = iRow / components, i = iRow % components;
            for (I entry = offsetRead.View()(cell, 0); entry < offsetRead.View()(cell + 1, 0); ++entry)
            {
                const I column = columnRead.View()(entry, 0);
                for (I j = 0; j < components; ++j)
                {
                    const double value = column == cell ? (i == j ? 4 + .1 * std::max<I>(0, components - 5) : -.1) : (i == j ? -1 : 0);
                    residual -= value * v(column * components + j, 0);
                }
            }
            initial += std::pow(rRead.View()(iRow, 0), 2);
            final += residual * residual;
        }
        CHECK(final < 0.1 * initial);
        // Nonlinear setup/replacement and pMG keep multiple matrices alive.
        // AmgX Resources manages global pools: an independent resource per
        // instance used to invalidate the first matrix at creation/destruction
        // of the second. Different cycle settings must stay solver-local.
        {
            DNDS::Algebra::AmgXSettings settings;
            settings.cycles = 2;
            DNDS::Algebra::AmgXPreconditioner second(device, settings);
            second.SetPattern(offsets, columns, int(components));
            const auto valueRead = values.Read(R::ExecutionContext::CUDA(0));
            second.UpdateValuesBorrowed(R::ExecutionContext::CUDA(0), valueRead.View());
            const auto rhsRead = rhs.Read(R::ExecutionContext::CUDA(0));
            R::Array<double> other(R::RowLayout::Dense(n, 1), device);
            {
                auto write = other.Write(R::ExecutionContext::CUDA(0));
                second.ApplyBorrowed(R::ExecutionContext::CUDA(0), rhsRead.View(), write.View());
            }
            const auto hostOther = other.To(R::Device::Host());
            const auto readOther = hostOther.Read();
            for (I iRow = 0; iRow < n; ++iRow)
                CHECK(std::isfinite(readOther.View()(iRow, 0)));
        }
        // Repeated zero-start application must not retain the previous correction.
        // It must also survive retirement of the independent second solver.
        const auto again = apply(false);
        const auto againRead = again.Read();
        for (I iRow = 0; iRow < n; ++iRow)
            CHECK(againRead.View()(iRow, 0) == doctest::Approx(firstRead.View()(iRow, 0)).epsilon(1e-12));
        {
            auto a = hostValues.Write();
            for (I iRow = 0; iRow < nnz * components * components; ++iRow)
                a.View()(iRow, 0) *= 2;
        }
        values.CopyFrom(hostValues, R::ExecutionContext::CUDA(0)).Wait();
        const auto scaled = apply(true);
        const auto scaledRead = scaled.Read();
        for (I iRow = 0; iRow < n; ++iRow)
            CHECK(2 * scaledRead.View()(iRow, 0) == doctest::Approx(firstRead.View()(iRow, 0)).epsilon(1e-10));
        CHECK_THROWS(provider.ApplyBorrowed(R::ExecutionContext::Host(), {}, {}));
    }
}

TEST_CASE("Native flow and modal block inversion covers all launched thread scratch and dense entries")
{
    // The old grid-limit-sized scratch overflows beyond 256 block rows
    // (16 lanes/block); 5003 also exercises several full CTAs and a tail.
    // One uncoupled block per row makes one undamped Jacobi
    // sweep an exact solve, independently checked against a prescribed vector.
    for (I cells : {I(1), I(63), I(5003)})
        for (I n : {I(2), I(3), I(4), I(5), I(6), I(7), I(8), I(9), I(19)})
        {
            CAPTURE(n);
            CAPTURE(cells);
            R::Array<int> offsets(R::RowLayout::Dense(cells + 1, 1)), columns(R::RowLayout::Dense(cells, 1));
            R::Array<double> values(R::RowLayout::Dense(cells * n * n, 1)), rhs(R::RowLayout::Dense(cells * n, 1));
            const auto exact = [](I cell, I component)
            { return .5 + .0001 * cell + .03 * component; };
            {
                auto o = offsets.Write(), c = columns.Write();
                auto a = values.Write(), b = rhs.Write();
                for (I cell = 0; cell < cells; ++cell)
                {
                    o.View()(cell, 0) = c.View()(cell, 0) = int(cell);
                    for (I i = 0; i < n; ++i)
                    {
                        double sum = 0;
                        for (I j = 0; j < n; ++j)
                        {
                            const double value = i == j ? 4 + .1 * i : .01 * (i + 1) - .02 * (j + 1);
                            a.View()(cell * n * n + i * n + j, 0) = value;
                            sum += value * exact(cell, j);
                        }
                        b.View()(cell * n + i, 0) = sum;
                    }
                }
                o.View()(cells, 0) = int(cells);
            }
            DNDS::Algebra::AmgXSettings settings;
            settings.maxLevels = 1;
            settings.coarseSweeps = 1;
            settings.relaxation = 1;
            const auto device = R::Device::CUDA(0);
            const auto context = R::ExecutionContext::CUDA(0);
            DNDS::Algebra::AmgXPreconditioner provider(device, settings);
            const auto o = offsets.Read(), c = columns.Read();
            provider.SetPattern(offsets, columns, int(n));
            auto a = values.To(device), b = rhs.To(device);
            R::Array<double> x(rhs.Layout(), device);
            {
                const auto ar = a.Read(context), br = b.Read(context);
                auto xw = x.Write(context);
                provider.UpdateValuesBorrowed(context, ar.View());
                provider.ApplyBorrowed(context, br.View(), xw.View());
            }
            const auto host = x.To(R::Device::Host());
            const auto read = host.Read();
            for (I cell = 0; cell < cells; ++cell)
                for (I j = 0; j < n; ++j)
                    CHECK(read.View()(cell * n + j, 0) == doctest::Approx(exact(cell, j)).epsilon(2e-12));
        }
}

TEST_CASE("Native matrix attachment shares values with fixed-cycle upload equivalence and refresh")
{
    // One inline/native operator, one provider-private uploaded copy. Exercise
    // ordinary flow and the largest modal width without component padding.
    for (const I width : {I(5), I(19)})
        for (const bool separate : {false, true})
        {
            CAPTURE(width);
            CAPTURE(separate);
            constexpr I rows = 47;
            const I entries = separate ? 2 * rows - 2 : 3 * rows - 2;
            const I blocks = entries + (separate ? rows : 1);
            R::Array<int> offsets(R::RowLayout::Dense(rows + 1, 1)), columns(R::RowLayout::Dense(entries, 1));
            R::Array<double> host(R::RowLayout::Dense(blocks * width * width, 1));
            R::Array<double> hostRhs(R::RowLayout::Dense(rows * width, 1));
            {
                auto o = offsets.Write(), c = columns.Write();
                auto a = host.Write(), b = hostRhs.Write();
                I entry = 0;
                for (I iRow = 0; iRow < rows; ++iRow)
                {
                    o.View()(iRow, 0) = int(entry);
                    for (I column = std::max<I>(0, iRow - 1); column <= std::min<I>(rows - 1, iRow + 1); ++column)
                    {
                        const bool diagonal = iRow == column;
                        const I block = separate && diagonal ? entries + iRow : entry;
                        if (!separate || !diagonal)
                            c.View()(entry++, 0) = int(column);
                        for (I i = 0; i < width; ++i)
                            for (I j = 0; j < width; ++j)
                                a.View()(block * width * width + i * width + j, 0) =
                                    diagonal ? (i == j ? 6 + .1 * i : -.01) : (i == j ? -1 : 0);
                    }
                    for (I j = 0; j < width; ++j)
                        b.View()(iRow * width + j, 0) = 1 + .01 * (iRow + j);
                }
                o.View()(rows, 0) = int(entry);
                REQUIRE(entry == entries);
            }
            const auto device = R::Device::CUDA(0);
            const auto context = R::ExecutionContext::CUDA(0);
            auto values = host.To(device), rhs = hostRhs.To(device);
            R::Array<double> expected(hostRhs.Layout(), device), actual(hostRhs.Layout(), device);
            DNDS::Algebra::AmgXSettings settings;
            settings.geometricAggregation = separate;
            settings.aggregateSize = 4;
            settings.minCoarseRows = 3;
            DNDS::Algebra::AmgXPreconditioner upload(device, settings), attached(device, settings);
            upload.SetPattern(offsets, columns, int(width), separate);
            attached.SetPattern(offsets, columns, int(width), separate);
            if (separate)
            {
                R::Array<double> weights(R::RowLayout::Dense(entries, 1));
                {
                    auto a = weights.Write();
                    const auto o = offsets.Read(), c = columns.Read();
                    for (I row = 0; row < rows; ++row)
                        for (I j = o.View()(row, 0); j < o.View()(row + 1, 0); ++j)
                            a.View()(j, 0) = 1.0 + std::min<I>(row, c.View()(j, 0)) % 3;
                }
                const auto placed = weights.To(device);
                attached.SetGeometricWeights(placed);
                upload.SetGeometricWeights(placed);
            }
            {
                const auto read = values.Read(context);
                auto a = read.View();
                a.rows = entries * width * width;
                auto diagonal = read.View();
                diagonal.data += entries * width * width;
                diagonal.rows = rows * width * width;
                upload.UpdateValuesBorrowed(context, a, separate ? std::optional<decltype(diagonal)>(diagonal) : std::nullopt);
            }
            attached.AttachValues(context, values);
            if (width == 5 && !separate)
                CHECK_THROWS(attached.SetPattern(offsets, columns, int(width), separate));
            const auto apply = [&](auto &provider, auto &output)
            {
                const auto b = rhs.Read(context);
                auto x = output.Write(context);
                provider.ApplyBorrowed(context, b.View(), x.View());
            };
            apply(upload, expected);
            apply(attached, actual);
            const auto reference = expected.To(R::Device::Host()), result = actual.To(R::Device::Host());
            {
                const auto r = reference.Read(), x = result.Read();
                for (I i = 0; i < rows * width; ++i)
                    CHECK(x.View()(i, 0) == doctest::Approx(r.View()(i, 0)).epsilon(2e-12));
            }
            // Provider setup/application must not alter the caller's finest values.
            {
                const auto after = values.To(R::Device::Host());
                const auto a = after.Read(), h = host.Read();
                for (I i = 0; i < blocks * width * width; ++i)
                    CHECK(a.View()(i, 0) == h.View()(i, 0));
            }
            const auto memory = attached.MemorySnapshot(), publicMemory = upload.MemorySnapshot();
            REQUIRE(!memory.levels.empty());
            CHECK(memory.nativeStorage);
            CHECK(memory.levels.front().numericOwned == 0);
            CHECK(memory.levels.front().numericBorrowed == uint64_t(blocks * width * width * sizeof(double)));
            CHECK(memory.copies.matrixDeviceToDevice == 0);
            CHECK(publicMemory.copies.matrixDeviceToDevice == uint64_t((entries + (separate ? rows : 0)) * width * width * sizeof(double)));
            CHECK(publicMemory.levels.front().numericOwned >= publicMemory.copies.matrixDeviceToDevice);
            // Explicit between-solve mutation is allowed; unrefreshed application is not.
            {
                auto a = host.Write();
                for (I i = 0; i < blocks * width * width; ++i)
                    a.View()(i, 0) *= 2;
            }
            values.CopyFrom(host, context).Wait();
            {
                const auto b = rhs.Read(context);
                auto x = actual.Write(context);
                CHECK_THROWS(attached.ApplyBorrowed(context, b.View(), x.View()));
                x.CancelUnmodified();
            }
            attached.AttachValues(context, values);
            if (separate)
            {
                const auto refreshed = attached.MemorySnapshot();
                CHECK(refreshed.copies.numericRefreshes == 1);
                CHECK(refreshed.copies.patternHostToDevice == memory.copies.patternHostToDevice);
                CHECK(refreshed.levels.size() == memory.levels.size());
                CHECK(refreshed.levels.size() > 1);
            }
            apply(attached, actual);
            {
                const auto after = actual.To(R::Device::Host());
                const auto a = after.Read(), r = reference.Read();
                for (I i = 0; i < rows * width; ++i)
                    CHECK(2 * a.View()(i, 0) == doctest::Approx(r.View()(i, 0)).epsilon(2e-12));
            }
            // The provider retains the allocation even after the submitting
            // C++ handle is released. Public/private provider owners coexist.
            values = R::Array<double>{};
            apply(attached, actual);
        }
}
