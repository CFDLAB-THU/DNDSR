#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// Host-only test driver: device work is launched through the public C++
// wrappers. Compile the independent Eigen inverse oracle with the host
// compiler, rather than instantiating its dynamic LU in a CUDA translation unit.
#include "doctest.h"
#include "Cauchy/Implicit/CUDA/LinearBlockKernels.hpp"
#include "Cauchy/Implicit/CPU/LinearBlockKernels.hpp"
#include "DNDS/Runtime/Array/ArrayCopy.hpp"
#include <cuda_runtime.h>

namespace R = DNDS::Runtime;
namespace C = DNDS::Cauchy;
namespace A = DNDS::Algebra;
using I = DNDS::index;
using DNDS::real;

namespace
{
    template <class F>
    R::Array<real> Table(I rows, I width, F fill)
    {
        R::Array<real> result(R::RowLayout::Dense(rows, width));
        auto write = result.Write();
        const auto view = write.View<R::TableRows>();
        for (I iRow = 0; iRow < rows; ++iRow)
            for (I j = 0; j < width; ++j)
                view(iRow, j) = fill(iRow, j);
        return result;
    }
    void Compare(const R::Array<real> &device, const R::Array<real> &expected)
    {
        const auto host = device.To(R::Device::Host());
        const auto a = host.Read(), b = expected.Read();
        for (I iRow = 0; iRow < expected.RowCount(); ++iRow)
            for (I j = 0; j < expected.Layout().Width(); ++j)
                CHECK(a.View()(iRow, j) == doctest::Approx(b.View()(iRow, j)).epsilon(3e-12).scale(1));
    }
}

TEST_CASE("CUDA assembled actions and Jacobi preserve sparse block order and core halo separation")
{
    constexpr I rows = 259;
    std::vector<std::vector<I>> columns(rows);
    std::vector<I> core, dependent, order, position(rows), counts(rows);
    for (I iCell = 0; iCell < rows; ++iCell)
    {
        // Vary the row lengths, including empty rows. Ghost references occur
        // only in dependent rows; the core launch receives no ghost descriptor.
        if (iCell % 7)
            columns[iCell].push_back((iCell + 1) % rows);
        if (iCell % 5 == 0)
        {
            columns[iCell].push_back(rows + iCell % 2);
            dependent.push_back(iCell);
        }
        else
            core.push_back(iCell);
        counts[iCell] = I(columns[iCell].size());
    }
    order = core;
    order.insert(order.end(), dependent.begin(), dependent.end());
    for (I iRow = 0; iRow < rows; ++iRow)
        position[order[iRow]] = iRow;
    const C::LinearPatternStorage pattern(columns, order, position, core, dependent);
    const auto context = R::ExecutionContext::CUDA(0);
    for (bool tiled : {false, true})
        for (int components : {5, 6, 7})
            for (bool fullDiagonal : {false, true})
            {
                CAPTURE(components);
                CAPTURE(fullDiagonal);
                A::BlockMatrixStorage matrix(counts, components);
                {
                    auto write = matrix.Write();
                    const auto view = write.ArrayView();
                    for (I iCell = 0; iCell < rows; ++iCell)
                        for (I iBlock = 0; iBlock < counts[iCell]; ++iBlock)
                            for (int i = 0; i < components; ++i)
                                for (int j = 0; j < components; ++j)
                                    view.Element(iCell, iBlock, i, j) = .002 * (1 + iCell % 3 + iBlock + 2 * i - j);
                }
                auto diagonal = Table(rows, fullDiagonal ? components * components : components, [&](I iRow, I j)
                                      {
                if (!fullDiagonal) return 2 + .01 * (iRow % 3 + j);
                return j % components == j / components ? 2 + .01 * (iRow % 3 + j % components) : .001 * (1 + j); });
                // Inverse is prepared independently with the CPU dense solver;
                // this test targets row actions, not the diagonal inversion policy.
                auto inverse = diagonal.Clone();
                {
                    const auto read = diagonal.Read();
                    auto write = inverse.Write();
                    const auto d = read.View<R::UniformTableRows>();
                    const auto v = write.View<R::TableRows>();
                    for (I iCell = 0; iCell < rows; ++iCell)
                        if (fullDiagonal)
                        {
                            Eigen::Map<const Eigen::MatrixXd> block(d[iCell].data(), components, components);
                            Eigen::Map<Eigen::MatrixXd> result(v[iCell].data(), components, components);
                            result = block.inverse();
                        }
                        else
                            for (int j = 0; j < components; ++j)
                                v(iCell, j) = 1 / d(iCell, j);
                }
                auto owned = Table(rows, components, [](I iRow, I j)
                                   { return .1 + .003 * (iRow + j); });
                auto ghost = Table(2, components, [](I iRow, I j)
                                   { return -.2 + .013 * (iRow - j); });
                auto rhs = Table(rows, components, [](I iRow, I j)
                                 { return .7 - .002 * (iRow + 2 * j); });
                // A small discriminating set, not a Cartesian layout matrix.
                // The independent CPU oracle above retains original AoS data.
                const auto layout = [](A::MatrixTableLayout kind)
                { return A::MatrixTableOptions{kind, 32}; };
                const auto aos = layout(A::MatrixTableLayout::AoS), soa = layout(A::MatrixTableLayout::SoA), tiled32 = layout(A::MatrixTableLayout::AoSoA32);
                auto blockOptions = !tiled || components == 5 ? aos : components == 6 ? soa
                                                                                      : tiled32;
                // One pitched AoS case and one stronger base-alignment case,
                // folded into the existing discriminating fixtures.
                if (tiled && components == 5)
                    blockOptions.rowStridePaddingMultiple = 16;
                if (tiled && components == 7)
                    blockOptions.allocationAlignmentBytes = 256;
                const auto diagonalOptions = !tiled || components == 7 ? aos : components == 5 ? soa
                                                                                               : tiled32;
                auto inverseOptions = !tiled || components == 6 ? aos : components == 5 ? tiled32
                                                                                        : soa;
                if (tiled && components == 6)
                    inverseOptions.rowStridePaddingMultiple = 32;
                const bool rowMajorBlocks = tiled && components == 5;
                A::BlockMatrixStorage alternative(counts, components, R::Device::Host(), blockOptions, rowMajorBlocks);
                {
                    const auto read = matrix.Read();
                    auto write = alternative.Write();
                    for (I iCell = 0; iCell < rows; ++iCell)
                        for (I k = 0; k < counts[iCell]; ++k)
                            for (int j = 0; j < components; ++j)
                                for (int i = 0; i < components; ++i)
                                    write.ArrayView().Element(iCell, k, i, j) = read.ArrayView().Element(iCell, k, i, j);
                }
                const auto placedMatrix = alternative.Placed(context.device, context);
                CHECK(placedMatrix.Values().Layout() == alternative.Values().Layout());
                if (blockOptions.allocationAlignmentBytes)
                {
                    CHECK(alternative.Values().Storage().IsAligned(blockOptions.allocationAlignmentBytes));
                    CHECK(placedMatrix.Values().Storage().IsAligned(blockOptions.allocationAlignmentBytes));
                }
                if (tiled && components == 5)
                {
                    CHECK(placedMatrix.Values().Layout().Width() == I(components * components));
                    CHECK(placedMatrix.Values().Layout().Stride() == 32);
                }
                const auto matrixPlacement = [&](const auto &array, A::MatrixTableOptions options)
                {
                    R::Array<real> result(options.MakeLayout(array.RowCount(), array.Layout().Width()), R::Device::Host(), options.Allocation());
                    {
                        const auto read = array.Read();
                        auto write = result.Write();
                        for (I iCell = 0; iCell < array.RowCount(); ++iCell)
                            for (I j = 0; j < array.Layout().Width(); ++j)
                                write.View<R::UniformTableRows>()(iCell, fullDiagonal && rowMajorBlocks ? (j % components) * components + j / components : j) = read.View()(iCell, j);
                    }
                    R::Array<real> placed(result.Layout(), context.device, options.Allocation());
                    placed.CopyFrom(result, context).Wait();
                    return placed;
                };
                const auto dd = matrixPlacement(diagonal, diagonalOptions), di = matrixPlacement(inverse, inverseOptions);
                const C::LinearPatternStorage selectedPattern(columns, order, position, core, dependent,
                                                              tiled && components == 5 ? C::LinearIndexStorage::Int32 : C::LinearIndexStorage::Int64);
                CHECK(selectedPattern.CopyColumns() == columns);
                const auto placedPattern = selectedPattern.Placed(context.device, context);
                const auto place = [&](const auto &source)
                {
                    if (!tiled)
                        return source.To(context.device);
                    R::Array<real> converted(R::RowLayout::TiledColMajor(source.RowCount(), components, 32));
                    {
                        const auto read = source.Read();
                        auto write = converted.Write();
                        for (I iRow = 0; iRow < source.RowCount(); ++iRow)
                            for (I j = 0; j < components; ++j)
                                write.template View<R::UniformTableRows>()(iRow, j) = read.View()(iRow, j);
                    }
                    return converted.To(context.device);
                };
                const auto dx = place(owned), dg = place(ghost), db = place(rhs);
                for (bool componentGroups : {false, true})
                    for (int operation : {0, 1, 2}) // action, ordinary Jacobi, zero-start Jacobi
                    {
                        CAPTURE(componentGroups);
                        CAPTURE(operation);
                        auto expected = Table(rows, components, [](I, I)
                                              { return -900.; });
                        auto actual = expected.To(context.device), split = expected.To(context.device);
                        {
                            const auto p = pattern.Read();
                            const auto blocks = matrix.Read();
                            const auto d = (operation ? inverse : diagonal).Read();
                            const auto x = owned.Read(), g = ghost.Read(), b = rhs.Read();
                            auto output = expected.Write();
                            C::CPU::LinearBlockInput request{{blocks.View(), nullptr, components}, p.View(), p.View().order, d.View<R::UniformTableRows>(), components, fullDiagonal};
                            if (!operation)
                                C::CPU::ApplyLinearRows(request, x.View<R::UniformTableRows>(), g.View<R::UniformTableRows>(), output.View<R::UniformTableRows>());
                            else
                                C::CPU::JacobiLinearRows(request, b.View<R::UniformTableRows>(), x.View<R::UniformTableRows>(), g.View<R::UniformTableRows>(), output.View<R::UniformTableRows>(), operation == 2);
                        }
                        {
                            const auto p = placedPattern.Read(context);
                            const auto blocks = placedMatrix.Read(context);
                            const auto d = (operation ? di : dd).Read(context);
                            const auto x = dx.Read(context), g = dg.Read(context), b = db.Read(context);
                            auto output = actual.Write(context), splitOutput = split.Write(context);
                            C::CUDA::LinearBlockInput request{blocks.ArrayView(), p.View().columns, p.View().order,
                                                              d.View<R::UniformTableRows>(), rows, components, fullDiagonal, rowMajorBlocks};
                            if (!operation && components == 5 && !fullDiagonal)
                            {
                                const auto hostDiagonal = diagonal.Read();
                                auto invalid = request;
                                invalid.diagonal = hostDiagonal.View<R::UniformTableRows>();
                                CHECK_THROWS(C::CUDA::ApplyLinearRows(context, invalid, x.View<R::UniformTableRows>(),
                                                                      g.View<R::UniformTableRows>(), output.View<R::TableRows>()));
                            }
                            if (operation == 2)
                            {
                                request.blocks = {};
                                request.columns = {};
                            }
                            const auto launch = [&](auto out, auto remote)
                            {
                                if (!operation)
                                    C::CUDA::ApplyLinearRows(context, request, x.View<R::UniformTableRows>(), remote, out);
                                else
                                    C::CUDA::JacobiLinearRows(context, request, b.View<R::UniformTableRows>(),
                                                              operation == 2 ? C::CUDA::VectorReadRows{} : x.View<R::UniformTableRows>(), remote, out, operation == 2, componentGroups);
                            };
                            launch(output.View<R::TableRows>(), g.View<R::UniformTableRows>());
                            request.rows = p.View().core;
                            launch(splitOutput.View<R::TableRows>(), C::CUDA::VectorReadRows{});
                            request.rows = p.View().dependent;
                            launch(splitOutput.View<R::TableRows>(), g.View<R::UniformTableRows>());
                            REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
                        }
                        Compare(actual, expected);
                        Compare(split, expected);
                    }
            }
}
