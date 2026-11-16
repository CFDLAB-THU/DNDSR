#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Implicit/LinearProblem.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "InverseRefinementFixture.hpp"
#include "Solver/SmallBlockInverse.hpp"
#include "LinearInverseReference.hpp"
#include <numeric>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;
using DNDS::real;
using Field = A::DofField<real>;

TEST_CASE("Queued CUDA diagonal SVD and scalar inverses match native CPU actions")
{
    R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto context = stream.Context();
    for (I rows : {I(0), I(131)})
        for (int width : {5, 6, 7})
            for (auto mode : {C::DiagonalStorage::Entries, C::DiagonalStorage::Block})
                for (int refinement : {0, 2})
                    for (auto kernel : {C::CUDADiagonalInverse::BatchedSVD, C::CUDADiagonalInverse::GuardedLU})
                    {
                        if (mode == C::DiagonalStorage::Entries && refinement)
                            continue;
                        CAPTURE(refinement);
                        CAPTURE(rows);
                        CAPTURE(width);
                        CAPTURE(int(mode));
                        const A::DofSpace space(rows);
                        std::vector<std::vector<I>> columns(rows);
                        std::vector<I> order(rows);
                        std::iota(order.begin(), order.end(), I(0));
                        C::LinearProblemContext cpu(space, width, columns, order, mode);
                        C::LinearProblemContext gpu(space, width, columns, order, mode, C::OffDiagonalStorage::Assembled, context.device);
                        {
                            auto write = cpu.Diagonal().Write();
                            const auto diagonal = write.View<R::TableRows>();
                            for (I iCell = 0; iCell < rows; ++iCell)
                                if (mode == C::DiagonalStorage::Entries)
                                    for (int j = 0; j < width; ++j)
                                        diagonal(iCell, j) = 2 + .125 * (iCell % 3 + j);
                                else
                                    for (int j = 0; j < width; ++j)
                                        for (int i = 0; i < width; ++i)
                                        {
                                            // Exact powers of two make the singular
                                            // row-scaled block all ones (rank one).
                                            const real scale = real(1 << i);
                                            diagonal(iCell, i + width * j) = iCell % 3 == 0 ? scale
                                                                                            : scale * (i == j ? 3 + .125 * i : .01 * (1 + 2 * i - j));
                                        }
                        }
                        gpu.Diagonal().CopyFrom(cpu.Diagonal(), context).Wait();
                        gpu.PrepareDiagonalInverse(context, kernel);
                        cpu.InvertDiagonal(refinement);
                        gpu.SubmitInvertDiagonal(engine, stream, refinement);
                        engine.Synchronize();
                        gpu.ObserveDiagonalInverse();
                        const bool guarded = kernel == C::CUDADiagonalInverse::GuardedLU;
                        CHECK(gpu.LastInverseFallbacks() == (guarded ? cpu.LastInverseFallbacks() : 0));
                        CHECK(gpu.LastInverseSVDCells() == (guarded ? cpu.LastInverseFallbacks() : mode == C::DiagonalStorage::Block ? rows
                                                                                                                                     : 0));
                        Field rhs(space, width), expected(space, width);
                        Field deviceRHS(space, width, 0, context.device), actual(space, width, 0, context.device);
                        // Probe every matrix column, including filtered modes. This
                        // checks the pseudoinverse, not just one favorable RHS vector.
                        for (int column = 0; column < width; ++column)
                        {
                            {
                                auto write = rhs.Owned().Write();
                                for (I iCell = 0; iCell < rows; ++iCell)
                                    for (int j = 0; j < width; ++j)
                                        write.View()(iCell, j) = j == column ? 1 : 0;
                            }
                            deviceRHS.Owned().CopyFrom(rhs.Owned(), context).Wait();
                            cpu.JacobiZero(rhs, expected);
                            gpu.SubmitJacobiZero(engine, stream, deviceRHS, actual);
                            engine.Synchronize();
                            const auto host = actual.Owned().To(R::Device::Host());
                            const auto a = host.Read(), b = expected.Owned().Read();
                            for (I iCell = 0; iCell < rows; ++iCell)
                                for (int j = 0; j < width; ++j)
                                    CHECK(a.View()(iCell, j) == doctest::Approx(b.View()(iCell, j)).epsilon(2e-11).scale(1));
                        }
                    }
}

TEST_CASE("Optional CUDA inverse refinement preserves the independent SST rhoK action")
{
    const A::DofSpace space(1);
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto context = stream.Context();
    C::LinearProblemContext gpu(space, 7, {{}}, {0}, C::DiagonalStorage::Block,
                                C::OffDiagonalStorage::Assembled, context.device);
    R::Array<real> matrix(R::RowLayout::Dense(1, 49));
    Field rhs(space, 7), deviceRHS(space, 7, 0, context.device), output(space, 7, 0, context.device);
    const auto a = CauchyTest::InverseRefinementBlock();
    const auto b = CauchyTest::InverseRefinementRHS();
    {
        auto m = matrix.Write();
        auto r = rhs.Owned().Write();
        for (int i = 0; i < 7; ++i)
        {
            r.View()(0, i) = b[i];
            for (int j = 0; j < 7; ++j)
                m.View()(0, i + 7 * j) = a(i, j);
        }
    }
    gpu.Diagonal().CopyFrom(matrix, context).Wait();
    deviceRHS.Owned().CopyFrom(rhs.Owned(), context).Wait();
    for (auto kernel : {C::CUDADiagonalInverse::BatchedSVD, C::CUDADiagonalInverse::GuardedLU})
    {
        gpu.PrepareDiagonalInverse(context, kernel);
        gpu.SubmitInvertDiagonal(engine, stream, 2);
        engine.Synchronize();
        gpu.ObserveDiagonalInverse();
        gpu.SubmitJacobiZero(engine, stream, deviceRHS, output);
        engine.Synchronize();
        const auto host = output.Owned().To(R::Device::Host());
        const auto read = host.Read();
        CHECK(std::abs(read.View()(0, 5) - CauchyTest::InverseRefinementRhoK) < 1e-12);
    }
}

TEST_CASE("CUDA inverse observation rejects invalid diagonal before Jacobi publication")
{
    const A::DofSpace space(1);
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto context = stream.Context();
    C::LinearProblemContext problem(space, 5, {{}}, {0}, C::DiagonalStorage::Block,
                                    C::OffDiagonalStorage::Assembled, context.device);
    R::Array<real> invalid(R::RowLayout::Dense(1, 25));
    {
        auto write = invalid.Write();
        for (int j = 0; j < 25; ++j)
            write.View()(0, j) = 0;
    }
    problem.Diagonal().CopyFrom(invalid, context).Wait();
    problem.PrepareDiagonalInverse(context, C::CUDADiagonalInverse::GuardedLU);
    problem.SubmitInvertDiagonal(engine, stream);
    engine.Synchronize();
    CHECK_THROWS(problem.ObserveDiagonalInverse());
    Field rhs(space, 5, 0, context.device), result(space, 5, 0, context.device);
    CHECK_THROWS(problem.SubmitJacobiZero(engine, stream, rhs, result));
}

TEST_CASE("Fixed small LU condition guard matches Eigen on pivoted and near-singular blocks")
{
    const auto check = [](auto size)
    {
        constexpr int N = decltype(size)::value;
        using Matrix = Eigen::Matrix<real, N, N>;
        for (int sample = 0; sample < 160; ++sample)
        {
            Matrix a;
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i)
                    a(i, j) = i == j ? 1 : .6 * std::sin(.91 * sample + 1.13 * i - .67 * j);
            if (sample % 4 == 0)
            {
                // Both sides of the 1e-7 guard, separated from its rounding
                // boundary. All normalized diagonals remain nonzero.
                a.setIdentity();
                a(0, 1) = 1;
                a(1, 0) = 1 - (sample % 8 == 0 ? 1e-6 : 1e-9);
            }
            DNDS::Direct::SmallDenseMatrix<N> input, inverse;
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i)
                    input(i, j) = a(i, j);
            DNDS::Direct::SmallPartialPivLU<N> lu;
            REQUIRE(lu.Factor(input));
            const real expected = CauchyTest::PartialPivotReciprocalCondition(a), actual = lu.ConditionReciprocal();
            CHECK(actual == doctest::Approx(expected).epsilon(2e-6).scale(1e-12));
            const bool accepted = DNDS::Direct::TrySmallLUInverse<N, false>(input, inverse);
            CHECK(accepted == (expected >= DNDS::Direct::DiagonalLUConditionMinimum));
            if (accepted)
            {
                Matrix x;
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i < N; ++i)
                        x(i, j) = inverse(i, j);
                CHECK((a * x - Matrix::Identity()).norm() < 2e-8);
            }
        }
    };
    check(std::integral_constant<int, 5>{});
    check(std::integral_constant<int, 6>{});
    check(std::integral_constant<int, 7>{});
}

TEST_CASE("Guarded CUDA SVD fallback matches CPU for weak and discarded modes")
{
    R::EagerEngine engine;
    const auto lane = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto context = lane.Context();
    for (int n : {5, 6, 7})
    {
        const A::DofSpace space(4);
        C::LinearProblemContext cpu(space, n, {{}, {}, {}, {}}, {0, 1, 2, 3}, C::DiagonalStorage::Block);
        C::LinearProblemContext gpu(space, n, {{}, {}, {}, {}}, {0, 1, 2, 3}, C::DiagonalStorage::Block,
                                    C::OffDiagonalStorage::Assembled, context.device);
        {
            auto write = cpu.Diagonal().Write();
            for (I iCell = 0; iCell < 4; ++iCell)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                    {
                        real value = i == j ? 1 : 0;
                        if (i < 2 && j < 2)
                            value = i == j ? 1 : iCell == 0 ? .2
                                             : iCell == 1   ? 1 - 1e-9
                                                            : 1;
                        if (iCell == 3)
                            value = 1;
                        write.View()(iCell, i + n * j) = value * (1 << i);
                    }
        }
        cpu.InvertDiagonal();
        gpu.Diagonal().CopyFrom(cpu.Diagonal(), context).Wait();
        gpu.PrepareDiagonalInverse(context, C::CUDADiagonalInverse::GuardedLU);
        gpu.SubmitInvertDiagonal(engine, lane);
        engine.Synchronize();
        gpu.ObserveDiagonalInverse();
        CHECK(gpu.LastInverseFallbacks() == cpu.LastInverseFallbacks());
        CHECK(gpu.LastInverseFallbacks() == 3);
        Field rhs(space, n), expected(space, n), deviceRHS(space, n, 0, context.device), actual(space, n, 0, context.device);
        for (int column = 0; column < n; ++column)
        {
            {
                auto write = rhs.Owned().Write();
                for (I iCell = 0; iCell < 4; ++iCell)
                    for (int j = 0; j < n; ++j)
                        write.View()(iCell, j) = j == column ? 1 : 0;
            }
            deviceRHS.Owned().CopyFrom(rhs.Owned(), context).Wait();
            cpu.JacobiZero(rhs, expected);
            gpu.SubmitJacobiZero(engine, lane, deviceRHS, actual);
            engine.Synchronize();
            const auto host = actual.Owned().To(R::Device::Host());
            const auto a = host.Read(), b = expected.Owned().Read();
            for (I iCell = 0; iCell < 4; ++iCell)
                for (int j = 0; j < n; ++j)
                    CHECK(a.View()(iCell, j) == doctest::Approx(b.View()(iCell, j)).epsilon(iCell == 1 ? 3e-6 : 2e-11).scale(1));
        }
    }
}

TEST_CASE("Queued inverses preserve independent diagonal and inverse matrix layouts")
{
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto context = stream.Context();
    for (int width : {5, 6, 7})
        for (const auto diagonalLayout : {A::MatrixTableLayout::AoS, A::MatrixTableLayout::SoA, A::MatrixTableLayout::AoSoA32})
            for (const auto kernel : {C::CUDADiagonalInverse::BatchedSVD, C::CUDADiagonalInverse::GuardedLU})
            {
                CAPTURE(width);
                CAPTURE(int(diagonalLayout));
                CAPTURE(int(kernel));
                C::LinearMatrixOptions storage;
                storage.diagonal = {diagonalLayout, 32};
                storage.inverse.layout = diagonalLayout == A::MatrixTableLayout::AoS ? A::MatrixTableLayout::SoA : diagonalLayout == A::MatrixTableLayout::SoA ? A::MatrixTableLayout::AoSoA32
                                                                                                                                                               : A::MatrixTableLayout::AoS;
                storage.inverse.rowPaddingMultiple = 32;
                storage.rowMajorBlocks = diagonalLayout == A::MatrixTableLayout::SoA;
                const A::DofSpace space(5);
                const std::vector<std::vector<I>> columns(5);
                const std::vector<I> order{0, 1, 2, 3, 4};
                C::LinearProblemContext cpu(space, width, columns, order, C::DiagonalStorage::Block,
                                            C::OffDiagonalStorage::Assembled, R::Device::Host(), {}, storage);
                C::LinearProblemContext gpu(space, width, columns, order, C::DiagonalStorage::Block,
                                            C::OffDiagonalStorage::Assembled, context.device, {}, storage);
                {
                    auto write = cpu.Diagonal().Write();
                    const auto view = write.View<R::UniformTableRows>();
                    for (I iCell = 0; iCell < 5; ++iCell)
                        for (int j = 0; j < width; ++j)
                            for (int i = 0; i < width; ++i)
                                view(iCell, storage.rowMajorBlocks ? i * width + j : i + width * j) =
                                    (1 << i) * (iCell == 1 ? 1 : i == j ? 3
                                                                        : .03 * (1 + i - j));
                }
                gpu.Diagonal().CopyFrom(cpu.Diagonal(), context).Wait();
                cpu.InvertDiagonal();
                gpu.PrepareDiagonalInverse(context, kernel);
                gpu.SubmitInvertDiagonal(engine, stream);
                engine.Synchronize();
                gpu.ObserveDiagonalInverse();
                Field rhs(space, width), expected(space, width), deviceRHS(space, width, 0, context.device), actual(space, width, 0, context.device);
                for (int column = 0; column < width; ++column)
                {
                    {
                        auto write = rhs.Owned().Write();
                        for (I iCell = 0; iCell < 5; ++iCell)
                            for (int j = 0; j < width; ++j)
                                write.View()(iCell, j) = j == column ? 1 : 0;
                    }
                    deviceRHS.Owned().CopyFrom(rhs.Owned(), context).Wait();
                    cpu.JacobiZero(rhs, expected);
                    gpu.SubmitJacobiZero(engine, stream, deviceRHS, actual);
                    engine.Synchronize();
                    const auto host = actual.Owned().To(R::Device::Host());
                    const auto a = host.Read(), b = expected.Owned().Read();
                    for (I iCell = 0; iCell < 5; ++iCell)
                        for (int j = 0; j < width; ++j)
                            CHECK(a.View()(iCell, j) == doctest::Approx(b.View()(iCell, j)).epsilon(2e-11).scale(1));
                }
            }
}
