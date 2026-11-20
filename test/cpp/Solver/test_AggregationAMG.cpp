#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Solver/AggregationAMG.hpp"
#include <Eigen/Dense>

namespace R = DNDS::Runtime;
namespace S = DNDS::Solver;

TEST_CASE("A reused aggregation hierarchy matches a dense two-level block cycle")
{
    constexpr int n = 128, width = 2, half = n / 2, edges = 3 * n - 4;
    R::Array<int> offsets(R::RowLayout::Dense(n + 1, 1)), columns(R::RowLayout::Dense(edges, 1));
    R::Array<double> weights(R::RowLayout::Dense(edges, 1)), blocks(R::RowLayout::Dense(edges, 4, 6));
    R::Array<double> diagonal(R::RowLayout::Dense(n, 4, 5)), rhs(R::RowLayout::Dense(n, width));
    R::Array<double> correction(R::RowLayout::Dense(n, width, 3));
    Eigen::VectorXd f(n * width);
    Eigen::MatrixXd prolong = Eigen::MatrixXd::Zero(n * width, n);
    {
        auto o = offsets.Write(), c = columns.Write();
        auto w = weights.Write(), b = rhs.Write();
        int k = 0;
        for (int i = 0; i < n; ++i)
        {
            o.View()(i, 0) = k;
            // Two chains with strong rungs. The strongest neighbor typically
            // has neither the smallest column ID nor the first CSR slot.
            for (int j : {i - 1, i + 1, (i + half) % n})
                if (j >= 0 && j < n && (j / half == i / half || std::abs(i - j) == half))
                {
                    c.View()(k, 0) = j;
                    w.View()(k++, 0) = std::abs(i - j) == half ? 8. : 1.;
                }
            for (int j = 0; j < width; ++j)
            {
                prolong(width * i + j, width * (i % half) + j) = 1.;
                b.View()(i, j) = f[width * i + j] = 1 + .02 * i - .4 * j;
            }
        }
        o.View()(n, 0) = k;
        REQUIRE(k == edges);
    }
    S::AggregationAMGSettings settings;
    settings.aggregateSize = 2;
    settings.maxLevels = 2;
    settings.preSweeps = 1;
    settings.postSweeps = 2;
    settings.coarseSweeps = 3;
    S::AggregationAMG solver(offsets, columns, weights, width, settings);
    REQUIRE(solver.LevelCount() == 2);
    for (int epoch : {0, 1})
    {
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(n * width, n * width);
        {
            auto d = diagonal.Write(), a = blocks.Write();
            auto c = columns.Read(), o = offsets.Read();
            for (int i = 0; i < n; ++i)
                for (int p = 0; p < width; ++p)
                    for (int q = 0; q < width; ++q)
                    {
                        d.View()(i, p * width + q) = matrix(width * i + p, width * i + q) =
                            p == q ? 4 + p + .01 * i + epoch : .2 * (p - q);
                        for (int k = o.View()(i, 0); k < o.View()(i + 1, 0); ++k)
                        {
                            const int j = c.View()(k, 0);
                            a.View()(k, p * width + q) = matrix(width * i + p, width * j + q) =
                                p == q ? -.8 + .1 * epoch : .03 * (1 + p - q);
                        }
                    }
        }
        if (epoch)
            CHECK_THROWS_WITH_AS(solver.Apply(rhs, correction), doctest::Contains("changed since refresh"), std::exception);
        solver.Refresh(blocks, diagonal);
        solver.Apply(rhs, correction);

        const auto jacobi = [&](const Eigen::MatrixXd &A, const Eigen::VectorXd &b, Eigen::VectorXd &x, int sweeps)
        {
            for (int iter = 0; iter < sweeps; ++iter)
            {
                const Eigen::VectorXd r = b - A * x;
                for (int i = 0; i < x.size(); i += width)
                    x.segment<width>(i) += settings.relaxation * A.block<width, width>(i, i).inverse() * r.segment<width>(i);
            }
        };
        Eigen::VectorXd expected = Eigen::VectorXd::Zero(n * width);
        jacobi(matrix, f, expected, settings.preSweeps);
        Eigen::VectorXd coarse = Eigen::VectorXd::Zero(n);
        jacobi(prolong.transpose() * matrix * prolong, prolong.transpose() * (f - matrix * expected), coarse, settings.coarseSweeps);
        expected += prolong * coarse;
        jacobi(matrix, f, expected, settings.postSweeps);
        const auto result = correction.Read();
        double error = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < width; ++j)
                error = std::max(error, std::abs(result.View()(i, j) - expected[width * i + j]));
        CHECK(error < 2e-13);
    }
}

TEST_CASE("Block Jacobi preserves resolvable modes with a structural zero diagonal")
{
    R::Array<int> offsets(R::RowLayout::Dense(2, 1)), columns(R::RowLayout::Dense(0, 1));
    R::Array<double> weights(R::RowLayout::Dense(0, 1)), off(R::RowLayout::Dense(0, 4));
    R::Array<double> diagonal(R::RowLayout::Dense(1, 4)), rhs(R::RowLayout::Dense(1, 2)), x(rhs.Layout());
    {
        auto o = offsets.Write();
        o.View()(0, 0) = o.View()(1, 0) = 0;
        auto d = diagonal.Write(), f = rhs.Write();
        for (int k = 0; k < 4; ++k)
            d.View()(0, k) = k == 1 ? 2. : 0.;
        f.View()(0, 0) = 6.;
        f.View()(0, 1) = 0.;
    }
    S::AggregationAMGSettings settings;
    settings.coarseSweeps = 1;
    settings.relaxation = 1.;
    S::AggregationAMG solver(offsets, columns, weights, 2, settings);
    solver.Refresh(off, diagonal);
    solver.Apply(rhs, x);
    const auto read = x.Read();
    CHECK(read.View()(0, 0) == 0.);
    CHECK(read.View()(0, 1) == doctest::Approx(3.));
}
