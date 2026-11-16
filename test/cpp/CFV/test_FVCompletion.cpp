#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "CFV/Math/TaylorBasis.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "Solver/Linear.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include "RotatedSectorFixture.hpp"

namespace C = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Rotated periodic sector preserves vector linear fields across distributed ownership")
{
    auto mesh = CFVTest::Sector(mpi);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 2;
    settings.boundaryDerivativeWeights = {1, 0, 0, 0};
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto work = plan.BuildWorkspace(4, {0});
    auto coefficients = plan.BuildCoefficients(4);
    auto mean = metrics.BuildField(4);
    auto bc = plan.BuildBoundaryValues(4);
    auto gradient = metrics.BuildField(8);
    auto grad = C::GradientPlan::Prepare(*mesh, metrics);
    auto gradBC = grad.BuildBoundaryValues(4);
    {
        auto geometry = metrics.AcquireRead();
        auto u = mean.Owned().Write(), r = coefficients.Owned().Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto p = geometry.HostView().GetCellBary(iCell);
            u.View()(iCell, 0) = p[0], u.View()(iCell, 1) = p[1], u.View()(iCell, 2) = 0, u.View()(iCell, 3) = 3;
            const auto scale = geometry.HostView().cells(iCell, 7);
            r.View()(iCell, 0) = scale, r.View()(iCell, plan.NRecDOF() + 1) = scale;
        }
        auto b = bc.values.Write();
        for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto p = plan.GetBoundaryQuadrature(iPoint).point;
            b.View()(iPoint, 0) = p[0], b.View()(iPoint, 1) = p[1], b.View()(iPoint, 2) = 0, b.View()(iPoint, 3) = 3;
        }
        auto gb = gradBC.Write();
        for (I iPoint = 0; iPoint < grad.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto p = grad.GetBoundaryQuadrature(iPoint).point;
            gb.View()(iPoint, 0) = p[0], gb.View()(iPoint, 1) = p[1], gb.View()(iPoint, 2) = 0, gb.View()(iPoint, 3) = 3;
        }
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(coefficients).Begin().Wait();
    plan.DoReconstructionIter(mean, coefficients, work, bc);
    {
        auto geometry = metrics.AcquireRead();
        auto read = coefficients.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I component = 0; component < 4; ++component)
                for (I mode = 0; mode < plan.NRecDOF(); ++mode)
                    CHECK(read.View()(iCell, component * plan.NRecDOF() + mode) == doctest::Approx(
                                                                                       component < 2 && mode == component ? geometry.HostView().cells(iCell, 7) : 0)
                                                                                       .epsilon(1e-10)
                                                                                       .scale(1));
    }
    grad.Evaluate(mean, gradBC, gradient, {0});
    {
        auto read = gradient.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I component = 0; component < 4; ++component)
                for (I d = 0; d < 2; ++d)
                    CHECK(read.View()(iCell, component * 2 + d) == doctest::Approx(component == d ? 1. : 0.).epsilon(1e-10).scale(1));
    }
    // The vector field [x,y,0] has identity gradient. Across the 90-degree
    // periodic image it remains I only when the equation and derivative axes
    // are both transformed: R I R^T = I.
    metrics.BindHalo(gradient).Begin().Wait();
    auto limitedGradient = metrics.BuildField(8);
    grad.LimitWBAP3(gradient, limitedGradient, {0});
    {
        auto read = limitedGradient.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I component = 0; component < 4; ++component)
                for (I d = 0; d < 2; ++d)
                    CHECK(read.View()(iCell, component * 2 + d) == doctest::Approx(component == d ? 1. : 0.).epsilon(1e-10).scale(1));
    }
    metrics.BindHalo(coefficients).Begin().Wait();
    plan.DoLimiterWBAP_3Order(coefficients, work, 1);
    {
        auto geometry = metrics.AcquireRead();
        auto read = coefficients.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I component = 0; component < 4; ++component)
                for (I mode = 0; mode < plan.NRecDOF(); ++mode)
                    CHECK(read.View()(iCell, component * plan.NRecDOF() + mode) == doctest::Approx(
                                                                                       component < 2 && mode == component ? geometry.HostView().cells(iCell, 7) : 0)
                                                                                       .epsilon(1e-10)
                                                                                       .scale(1));
    }
    CHECK_THROWS(plan.BuildWorkspace(4, {2}));
}

TEST_CASE("Axisymmetric metrics and all gradient methods obey radial linear moments at the axis")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, true).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, 5, true);
    CHECK(metrics.Axisymmetric());
    double total = 0;
    {
        auto geometry = metrics.AcquireRead();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            total += geometry.HostView().GetCellVol(iCell);
    }
    double global = 0;
    MPI_Allreduce(&total, &global, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(global == doctest::Approx(.5).epsilon(1e-13));
    for (auto method : {C::GradientMethod::GreenGauss, C::GradientMethod::LeastSquares, C::GradientMethod::GGMP})
    {
        auto plan = C::GradientPlan::Prepare(*mesh, metrics, method);
        auto u = metrics.BuildField(1), gradient = metrics.BuildField(2);
        auto bc = plan.BuildBoundaryValues(1);
        {
            auto geometry = metrics.AcquireRead();
            auto write = u.Owned().Write(), b = bc.Write();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                const auto p = geometry.HostView().GetCellBary(iCell);
                write.View()(iCell, 0) = 2 + 3 * p[0] - .7 * p[1];
            }
            for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
            {
                const auto p = plan.GetBoundaryQuadrature(iPoint).point;
                b.View()(iPoint, 0) = 2 + 3 * p[0] - .7 * p[1];
            }
        }
        metrics.BindHalo(u).Begin().Wait();
        plan.Evaluate(u, bc, gradient);
        auto read = gradient.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            CHECK(read.View()(iCell, 0) == doctest::Approx(3).epsilon(1e-10));
            CHECK(read.View()(iCell, 1) == doctest::Approx(-.7).epsilon(1e-10));
        }
    }
    C::ReconstructionSettings settings;
    settings.boundaryDerivativeWeights = {1, 0, 0, 0};
    auto reconstruction = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto mean = metrics.BuildField(1);
    auto work = reconstruction.BuildWorkspace(1);
    auto coefficients = reconstruction.BuildCoefficients(1);
    auto boundary = reconstruction.BuildBoundaryValues(1);
    {
        auto geometry = metrics.AcquireRead();
        auto u = mean.Owned().Write(), r = coefficients.Owned().Write(), b = boundary.values.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto p = geometry.HostView().GetCellBary(iCell);
            u.View()(iCell, 0) = 2 + 3 * p[0] - .7 * p[1];
            r.View()(iCell, 0) = 3 * geometry.HostView().cells(iCell, 7), r.View()(iCell, 1) = -.7 * geometry.HostView().cells(iCell, 7);
        }
        for (I iPoint = 0; iPoint < reconstruction.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto p = reconstruction.GetBoundaryQuadrature(iPoint).point;
            b.View()(iPoint, 0) = 2 + 3 * p[0] - .7 * p[1];
        }
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(coefficients).Begin().Wait();
    reconstruction.DoReconstructionIter(mean, coefficients, work, boundary);
    auto read = coefficients.Owned().Read();
    auto geometry = metrics.AcquireRead();
    for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        for (I mode = 0; mode < reconstruction.NRecDOF(); ++mode)
            CHECK(read.View()(iCell, mode) == doctest::Approx(mode == 0 ? 3 * geometry.HostView().cells(iCell, 7) : mode == 1 ? -.7 * geometry.HostView().cells(iCell, 7)
                                                                                                                              : 0)
                                                  .epsilon(1e-9)
                                                  .scale(1));
}

TEST_CASE("Raw reconstruction operators preserve symmetry and agree with a preconditioned Jacobi step")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, true).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.jacobiRelax = .61;
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto work = plan.BuildWorkspace(1);
    auto coefficients = plan.BuildCoefficients(1);
    auto y = metrics.BuildField(plan.NRecDOF()), ax = metrics.BuildField(plan.NRecDOF()), ay = metrics.BuildField(plan.NRecDOF());
    auto mean = metrics.BuildField(1), rhs = metrics.BuildField(plan.NRecDOF()), residual = metrics.BuildField(plan.NRecDOF()), delta = metrics.BuildField(plan.NRecDOF());
    auto boundary = plan.BuildBoundaryValues(1);
    {
        auto xWrite = coefficients.Owned().Write(), yWrite = y.Owned().Write(), u = mean.Owned().Write(), b = boundary.values.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto id = mesh->OriginalEntityId(G::EntityKind::Cell, iCell);
            u.View()(iCell, 0) = std::cos(.2 * id);
            for (I mode = 0; mode < plan.NRecDOF(); ++mode)
            {
                xWrite.View()(iCell, mode) = std::sin(.7 * id + .4 * mode);
                yWrite.View()(iCell, mode) = std::cos(.3 * id - .2 * mode);
            }
        }
        for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
            b.View()(iPoint, 0) = std::sin(.2 * iPoint);
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(coefficients).Begin().Wait();
    metrics.BindHalo(y).Begin().Wait();
    plan.ApplyLinearOperator(coefficients, ax);
    plan.ApplyLinearOperator(y, ay);
    double local[4]{}, global[4]{};
    {
        auto x = coefficients.Owned().Read(), yr = y.Owned().Read(), a = ax.Owned().Read(), b = ay.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I mode = 0; mode < plan.NRecDOF(); ++mode)
            {
                local[0] += x.View()(iCell, mode) * b.View()(iCell, mode);
                local[1] += yr.View()(iCell, mode) * a.View()(iCell, mode);
                local[2] += x.View()(iCell, mode) * a.View()(iCell, mode);
                local[3] += yr.View()(iCell, mode) * b.View()(iCell, mode);
            }
    }
    MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(global[0] == doctest::Approx(global[1]).epsilon(1e-11).scale(1));
    CHECK(global[2] > 0);
    CHECK(global[3] > 0);
    plan.BuildRightHandSide(mean, boundary, rhs);
    plan.ComputeResidual(rhs, coefficients, residual);
    plan.ApplyLocalPreconditioner(residual, delta);
    plan.DoReconstructionIterTo(mean, coefficients, work, boundary);
    {
        auto old = coefficients.Owned().Read(), next = work.temporary.Owned().Read(), step = delta.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I mode = 0; mode < plan.NRecDOF(); ++mode)
                CHECK(next.View()(iCell, mode) == doctest::Approx(old.View()(iCell, mode) + settings.jacobiRelax * step.View()(iCell, mode)).epsilon(1e-10).scale(1));
    }
    CHECK_THROWS(plan.ApplyLinearOperator(coefficients, coefficients));
    CHECK_THROWS(plan.ApplyLocalPreconditioner(rhs, rhs));
    auto filteredSettings = settings;
    filteredSettings.svdTolerance = 1; // Every reciprocal is zero, but raw A/B must survive.
    auto filtered = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, filteredSettings);
    filtered.ApplyLinearOperator(coefficients, ay);
    filtered.ApplyLocalPreconditioner(rhs, delta);
    {
        auto original = ax.Owned().Read(), raw = ay.Owned().Read(), zero = delta.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I mode = 0; mode < plan.NRecDOF(); ++mode)
            {
                CHECK(raw.View()(iCell, mode) == original.View()(iCell, mode));
                CHECK(zero.View()(iCell, mode) == 0);
            }
    }
}

TEST_CASE("Cached owned AoS polynomial samples match strided output for common equation widths")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, true).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    for (int order : {1, 3})
    {
        C::ReconstructionSettings settings;
        settings.maxOrder = order;
        settings.cacheDiffBase = true;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        R::Array<I> cells(R::RowLayout::Dense(2 * metrics.NumCell(), 1));
        R::Array<double> points(R::RowLayout::Dense(cells.RowCount(), 3));
        {
            auto geometry = metrics.AcquireRead();
            auto ids = cells.Write();
            auto coordinates = points.Write();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                for (I offset = 0; offset < 2; ++offset)
                {
                    const I iPoint = 2 * iCell + offset;
                    ids.View()(iPoint, 0) = iCell;
                    const auto center = geometry.HostView().GetCellCent(iCell);
                    for (I d = 0; d < 3; ++d)
                        coordinates.View()(iPoint, d) = center[d] + (d < 2 ? .031 * (offset + 1) * (d + 1) : 0);
                }
        }
        C::PolynomialSamplingPlan sampler(plan, cells, points, false);
        for (I width : {5, 6, 7})
        {
            INFO("order=" << order << ", width=" << width);
            auto mean = metrics.BuildField(width), coefficients = metrics.BuildField(width * plan.NRecDOF());
            {
                auto values = mean.Owned().Write(), modes = coefficients.Owned().Write();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I component = 0; component < width; ++component)
                    {
                        values.View()(iCell, component) = 1.25 + .17 * component + .03 * iCell;
                        for (I mode = 0; mode < plan.NRecDOF(); ++mode)
                            modes.View()(iCell, component * plan.NRecDOF() + mode) = .021 * (mode + 1) - .013 * component;
                    }
            }
            // Output layout alone selects the generic kernel, keeping the same
            // cached basis, input arithmetic, point order and untouched halos.
            R::Array<double> aos(R::RowLayout::Dense(cells.RowCount(), width));
            R::Array<double> soa(R::RowLayout::ColMajor(cells.RowCount(), width));
            sampler.EvaluateValues(mean, coefficients, aos);
            sampler.Evaluate(mean, coefficients, soa);
            auto a = aos.Read(), b = soa.Read();
            double maximumDifference = 0;
            bool finite = true;
            for (I iPoint = 0; iPoint < cells.RowCount(); ++iPoint)
                for (I component = 0; component < width; ++component)
                {
                    finite = finite && std::isfinite(a.View()(iPoint, component)) && std::isfinite(b.View()(iPoint, component));
                    maximumDifference = std::max(maximumDifference, std::abs(a.View()(iPoint, component) - b.View()(iPoint, component)));
                }
            CHECK(finite);
            CHECK(maximumDifference < 1e-14);
        }
    }
}

TEST_CASE("Shared basis policies snapshot points and sample cubic values and derivatives")
{
    for (int cacheOrder : {-1, 0, 1, 3})
    {
        INFO("cache order=" << cacheOrder);
        auto mesh = G::OverlapTest::BuildNativeFixture(mpi, true).mesh;
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
        C::ReconstructionSettings settings;
        settings.maxOrder = 3;
        settings.cacheDiffBase = cacheOrder >= 0;
        settings.cacheDiffBaseOrder = std::max(0, cacheOrder);
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        auto mean = metrics.BuildField(1), coefficients = metrics.BuildField(plan.NRecDOF());
        R::Array<I> cells(R::RowLayout::Dense(metrics.NumCell(), 1));
        R::Array<double> points(R::RowLayout::Dense(metrics.NumCell(), 3)), result(R::RowLayout::Dense(metrics.NumCell(), 1));
        // f=x^3+2*y; the exact square-cell mean is cx^3+cx*h^2/4+2*cy.
        {
            auto geometry = metrics.AcquireRead();
            auto u = mean.Owned().Write(), r = coefficients.Owned().Write(), p = points.Write();
            auto c = cells.Write();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                const auto center = geometry.HostView().GetCellCent(iCell);
                const auto scale = geometry.HostView().cells(iCell, 7);
                u.View()(iCell, 0) = center[0] * center[0] * center[0] + center[0] * .01 / 4 + 2 * center[1];
                for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                    r.View()(iCell, mode) = 0;
                r.View()(iCell, 0) = 3 * center[0] * center[0] * scale;
                r.View()(iCell, 1) = 2 * scale;
                r.View()(iCell, 2) = 3 * center[0] * scale * scale;
                r.View()(iCell, 5) = scale * scale * scale;
                c.View()(iCell, 0) = iCell;
                for (int d = 0; d < 3; ++d)
                    p.View()(iCell, d) = center[d];
            }
        }
        C::PolynomialSamplingPlan sampler(plan, cells, points);
        const int expectedRows = cacheOrder < 0 ? 0 : cacheOrder == 0 ? 1
                                                  : cacheOrder == 1   ? 3
                                                                      : 10;
        CHECK(sampler.CachedDerivativeNumber() == expectedRows);
        if (metrics.NumCell() && settings.cacheDiffBase)
        {
            auto geometry = metrics.AcquireRead();
            auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
            const auto view = read.HostView();
            const auto iFace = view.CellFaces(0)[0];
            const int side = view.CellIsFaceBack(0, iFace, 0) ? 0 : 1;
            const auto basis = plan.GetCachedFaceBasis(iFace, side, 0);
            auto scope = plan.AcquireBasisRead();
            const auto cache = scope.HostView();
            const auto iPoint = cache.faceOffsets(iFace, 0);
            const auto matrix = cache.FaceMatrix(iPoint, side);
            CHECK(matrix.RowCount() == expectedRows);
            CHECK(matrix.Width() == plan.NRecDOF());
            const double scale = geometry.HostView().cells(0, 7);
            CHECK(basis.rows() == expectedRows);
            if (expectedRows >= 3)
            {
                CHECK(basis(1, 0) == doctest::Approx(1 / scale));
                CHECK(basis(2, 0) == 0);
            }
            if (expectedRows > 3)
                CHECK(basis(3, 2) == doctest::Approx(2 / (scale * scale)));
            for (int derivative = 0; derivative < expectedRows; ++derivative)
                for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                {
                    CHECK(cache.FaceDerivative(iPoint, side, derivative, mode) == basis(derivative, mode));
                    CHECK(matrix[derivative][mode] == basis(derivative, mode));
                    CHECK(matrix.DynamicView()(derivative, mode) == basis(derivative, mode));
                }
        }
        {
            auto c = cells.Write();
            auto p = points.Write();
            for (I iRow = 0; iRow < cells.RowCount(); ++iRow)
            {
                c.View()(iRow, 0) = -1;
                p.View()(iRow, 0) = 1e20;
            }
        }
        for (int derivative : {0, 1, 2, 3, 6})
        {
            sampler.Evaluate(mean, coefficients, result, derivative);
            auto read = result.Read();
            auto geometry = metrics.AcquireRead();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                const auto p = geometry.HostView().GetCellCent(iCell);
                const double expected = derivative == 0   ? p[0] * p[0] * p[0] + 2 * p[1]
                                        : derivative == 1 ? 3 * p[0] * p[0]
                                        : derivative == 2 ? 2
                                        : derivative == 3 ? 6 * p[0]
                                                          : 6;
                CHECK(read.View()(iCell, 0) == doctest::Approx(expected)
                                                   .epsilon(1e-11));
            }
        }
        {
            auto mutation = mesh->versions.BeginMutation(R::Change::Geometry);
            mutation.Commit();
        }
        CHECK_THROWS(sampler.Evaluate(mean, coefficients, result));
    }
}

TEST_CASE("Shared quadrature face banks preserve periodic parent frames and scattered outputs")
{
    auto mesh = CFVTest::Sector(mpi);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, 3);
    C::ReconstructionSettings settings;
    settings.maxOrder = 3;
    settings.intOrder = 3; // Quadrature degree is independent of polynomial degree.
    settings.cacheDiffBase = false;
    auto directPlan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    const I modes = directPlan.NRecDOF();
    auto mean = metrics.BuildField(2), coefficients = metrics.BuildField(2 * modes);
    for (auto *field : {&mean, &coefficients})
    {
        auto write = field->Owned().Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I component = 0; component < I(field->Width()); ++component)
                write.View()(iCell, component) = .013 * (component + 1) + .07 * (iCell + 1);
    }
    std::vector<std::array<I, 2>> queries;
    std::vector<G::tPoint> positions;
    {
        auto scope = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
        const auto topology = scope.HostView();
        auto read = metrics.AcquireRead();
        const auto geometry = read.HostView();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto faces = topology.CellFaces(iCell);
            for (I slot = 0; slot < faces.size(); ++slot)
            {
                const I iFace = faces[slot];
                const int side = topology.CellIsFaceBack(iCell, iFace, DNDS::rowsize(slot)) ? 0 : 1;
                for (I iPoint = geometry.FaceQuadratureBegin(iFace); iPoint < geometry.FaceQuadratureEnd(iFace); ++iPoint)
                {
                    queries.push_back({iCell, 2 * iPoint + side});
                    positions.push_back(C::Detail::FacePointInCell(topology, iFace, side, geometry.GetFacePoint(iPoint)));
                }
            }
        }
    }
    const I count = I(queries.size());
    R::Array<I> cells(R::RowLayout::Dense(count, 1)), rows(R::RowLayout::Dense(count, 1)), scatter(R::RowLayout::Dense(count, 1));
    R::Array<double> points(R::RowLayout::Dense(count, 3)), expected(R::RowLayout::Dense(count, 2));
    {
        auto c = cells.Write(), r = rows.Write(), s = scatter.Write();
        auto p = points.Write();
        for (I iPoint = 0; iPoint < count; ++iPoint)
        {
            c.View()(iPoint, 0) = queries[iPoint][0];
            r.View()(iPoint, 0) = queries[iPoint][1];
            s.View()(iPoint, 0) = 2 * (count - 1 - iPoint) + 1;
            for (int d = 0; d < 3; ++d)
                p.View()(iPoint, d) = positions[iPoint][d];
        }
    }
    // Two logical tables may refer to different ranges of one allocation.
    // Placement must not confuse them merely because identities/layouts match.
    {
        R::Array<double> bank(R::RowLayout::Dense(4, 1));
        auto first = R::Array<double>::FromStorage(R::RowLayout::Dense(2, 1), bank.Storage().Subspan(0, 2 * sizeof(double)));
        auto second = R::Array<double>::FromStorage(R::RowLayout::Dense(2, 1), bank.Storage().Subspan(2 * sizeof(double), 2 * sizeof(double)));
        C::SamplingPlacementMemo memo(R::Device::Host(), R::ExecutionContext::Host());
        const auto placedFirst = memo.Place(first), placedSecond = memo.Place(second);
        CHECK(placedFirst.Storage().SameRange(first.Storage()));
        CHECK(placedSecond.Storage().SameRange(second.Storage()));
        CHECK_FALSE(placedFirst.Storage().SameRange(placedSecond.Storage()));
    }
    C::PolynomialSamplingPlan direct(directPlan, cells, points);
    for (bool cache : {false, true})
    {
        settings.cacheDiffBase = cache;
        settings.cacheDiffBaseOrder = 1;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        auto shared = C::PolynomialSamplingPlan::FromSharedBasis(metrics, plan, cells, points,
                                                                 plan.SamplingFaceBasis(), rows, cache ? 3 : 0, true);
        if (cache)
            CHECK(shared.BasisStorage().Identity() == plan.SamplingFaceBasis().Identity());
        else
            CHECK(shared.BasisStorage().RowCount() == 0);
        for (int derivative : {0, 1, 3})
        {
            direct.Evaluate(mean, coefficients, expected, derivative);
            R::Array<double> result(R::RowLayout::Dense(2 * count + 1, 2));
            {
                auto write = result.Write();
                for (I iRow = 0; iRow < result.RowCount(); ++iRow)
                    for (int j = 0; j < 2; ++j)
                        write.View()(iRow, j) = -123;
            }
            R::EagerEngine worker(2);
            const auto stream = worker.CreateCPUStream();
            {
                // First-derivative consumers release unused direct metadata
                // when covered by cache; arbitrary higher derivatives still work.
                auto selected = derivative < 3 ? shared.WithDerivativeOrder(1) : shared;
                auto segment = selected.WithOutputRows(scatter, result.RowCount());
                segment.Evaluate(worker, stream, mean, coefficients, result, derivative);
            } // Queued access retains the segment's mapping and basis handles.
            worker.Close();
            auto output = result.Read(), reference = expected.Read();
            for (I iRow = 0; iRow < result.RowCount(); ++iRow)
                for (int j = 0; j < 2; ++j)
                {
                    const double wanted = iRow % 2 ? reference.View()(count - 1 - iRow / 2, j) : -123;
                    CHECK(output.View()(iRow, j) == doctest::Approx(wanted).epsilon(2e-11).scale(1));
                }
        }
        if (cache)
        {
            // The owner API deliberately returns a shared handle; any later
            // write through an alias invalidates its immutable sampling users.
            auto alias = plan.SamplingFaceBasis();
            {
                auto write = alias.Write();
                if (alias.RowCount())
                    write.View()(0, 0) += 1;
            }
            CHECK_THROWS(shared.Validate());
        }
        else
        {
            auto normalization = plan.SamplingMoments();
            {
                auto write = normalization.Write();
                if (normalization.RowCount())
                    write.View()(0, 0) += 1;
            }
            CHECK_THROWS(shared.Validate());
        }
    }
}

TEST_CASE("Bundled polynomial values and gradients preserve the independent sampler")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, true).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    for (bool commonTable : {true, false})
    {
        INFO("common AoS specialization=" << commonTable);
        C::ReconstructionSettings settings;
        settings.maxOrder = 3;
        settings.cacheDiffBaseOrder = commonTable ? 1 : 0;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        const auto layout = [&](I rows, I width)
        { return commonTable ? R::RowLayout::Dense(rows, width, width + 3) : R::RowLayout::ColMajor(rows, width); };
        auto mean = metrics.BuildField(layout(0, 6));
        auto coefficients = metrics.BuildField(layout(0, 6 * plan.NRecDOF()));
        R::Array<I> cells(R::RowLayout::Dense(metrics.NumCell(), 1));
        R::Array<double> points(R::RowLayout::Dense(metrics.NumCell(), 3));
        {
            auto geometry = metrics.AcquireRead();
            auto u = mean.Owned().Write(), r = coefficients.Owned().Write(), p = points.Write();
            auto c = cells.Write();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                const auto center = geometry.HostView().GetCellCent(iCell);
                const double scale = geometry.HostView().cells(iCell, 7);
                c.View()(iCell, 0) = iCell;
                for (int d = 0; d < 3; ++d)
                    p.View()(iCell, d) = center[d] + (d == 0 ? .013 : d == 1 ? -.007
                                                                             : 0);
                for (I component = 0; component < 6; ++component)
                {
                    const double factor = component + 1;
                    u.View()(iCell, component) = factor * (center[0] * center[0] * center[0] + center[0] * .01 / 4 + 2 * center[1]) + component;
                    for (I mode = 0; mode < plan.NRecDOF(); ++mode)
                        r.View()(iCell, component * plan.NRecDOF() + mode) = 0;
                    r.View()(iCell, component * plan.NRecDOF()) = factor * 3 * center[0] * center[0] * scale;
                    r.View()(iCell, component * plan.NRecDOF() + 1) = factor * 2 * scale;
                    r.View()(iCell, component * plan.NRecDOF() + 2) = factor * 3 * center[0] * scale * scale;
                    r.View()(iCell, component * plan.NRecDOF() + 5) = factor * scale * scale * scale;
                }
            }
        }
        C::PolynomialSamplingPlan sampler(plan, cells, points);
        std::array<R::Array<double>, 3> results{
            R::Array<double>(layout(metrics.NumCell(), 6)), R::Array<double>(layout(metrics.NumCell(), 6)), R::Array<double>(layout(metrics.NumCell(), 6))};
        C::PolynomialSampleOutputs outputs;
        outputs.values = &results[0];
        outputs.gradients[0] = &results[1];
        outputs.gradients[1] = &results[2];
        sampler.EvaluateValueGradient(mean, coefficients, outputs);
        R::Array<double> reference(R::RowLayout::Dense(metrics.NumCell(), 6));
        for (int derivative = 0; derivative < 3; ++derivative)
        {
            sampler.Evaluate(mean, coefficients, reference, derivative);
            const auto expected = reference.Read(), actual = results[derivative].Read(), p = points.Read();
            for (I iPoint = 0; iPoint < metrics.NumCell(); ++iPoint)
                for (I component = 0; component < 6; ++component)
                {
                    const double x = p.View()(iPoint, 0), y = p.View()(iPoint, 1), factor = component + 1;
                    const double analytic = derivative == 0 ? factor * (x * x * x + 2 * y) + component : derivative == 1 ? factor * 3 * x * x
                                                                                                                         : factor * 2;
                    CHECK(actual.View()(iPoint, component) == doctest::Approx(expected.View()(iPoint, component)).epsilon(1e-13).scale(1));
                    CHECK(actual.View()(iPoint, component) == doctest::Approx(analytic).epsilon(1e-11).scale(1));
                }
        }
        const auto version = results[0].Version();
        // A separate metadata handle may share the same payload/access state,
        // including the zero-byte array on an empty-owner rank.
        auto alias = R::Array<double>::FromStorage(results[0].Layout(), results[0].Storage());
        auto overlapping = outputs;
        overlapping.gradients[0] = &alias;
        CHECK_THROWS(sampler.EvaluateValueGradient(mean, coefficients, overlapping));
        CHECK(results[0].Version() == version);
        overlapping = outputs;
        overlapping.values = &mean.Owned();
        CHECK_THROWS(sampler.EvaluateValueGradient(mean, coefficients, overlapping));
        // Source-center matrix preparation requests both derivatives without
        // a value buffer. Compare exactly with each independent derivative;
        // include the generic column-major/cache fallback above as well.
        outputs.values = nullptr;
        sampler.EvaluateValueGradient(mean, coefficients, outputs);
        for (int derivative = 1; derivative < 3; ++derivative)
        {
            sampler.Evaluate(mean, coefficients, reference, derivative);
            const auto expected = reference.Read(), actual = results[derivative].Read();
            for (I iPoint = 0; iPoint < metrics.NumCell(); ++iPoint)
                for (I component = 0; component < 6; ++component)
                    CHECK(actual.View()(iPoint, component) == expected.View()(iPoint, component));
        }
        CHECK(results[0].Version() == version);
        // Partial requests remain supported through the reference path.
        outputs.gradients[1] = nullptr;
        sampler.EvaluateValueGradient(mean, coefficients, outputs);
    }
}

// Test-only adapter for the historical Krylov vector concept. Runtime handles
// are shallow owners; numerical assignment must explicitly copy values.
struct KrylovField
{
    std::optional<C::FVField> field;
    KrylovField() = default;
    KrylovField(KrylovField &&) = default;
    KrylovField(const KrylovField &other) { *this = other; }
    KrylovField &operator=(KrylovField &&) = default;
    KrylovField &operator=(const KrylovField &other)
    {
        if (this != &other)
        {
            if (!field)
                field.emplace(other.field->Space(), other.field->Width());
            DNDS::Algebra::Copy(*field, *other.field);
        }
        return *this;
    }
    void addTo(const KrylovField &other, double factor) { DNDS::Algebra::Axpy(*field, factor, *other.field); }
    void operator*=(double factor) { DNDS::Algebra::Scale(*field, factor); }
};

TEST_CASE("Existing GMRES consumes native FV operators and DNDS field algebra through an explicit vector adapter")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, true).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 1;
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    const auto initialize = [&](KrylovField &x)
    { x.field.emplace(metrics.BuildField(plan.NRecDOF())); };
    KrylovField exact, rhs, solution, check;
    initialize(exact);
    initialize(rhs);
    initialize(solution);
    initialize(check);
    {
        auto write = exact.field->Owned().Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I mode = 0; mode < plan.NRecDOF(); ++mode)
                write.View()(iCell, mode) = std::sin(.37 * mesh->OriginalEntityId(G::EntityKind::Cell, iCell) + .2 * mode);
    }
    const auto apply = [&](KrylovField &x, KrylovField &y)
    {
        metrics.BindHalo(*x.field).Begin().Wait();
        plan.ApplyLinearOperator(*x.field, *y.field);
    };
    const auto precondition = [&](KrylovField &x, KrylovField &y)
    { plan.ApplyLocalPreconditioner(*x.field, *y.field); };
    const auto dot = [](KrylovField &x, KrylovField &y)
    { return DNDS::Algebra::DotGlobal(*x.field, *y.field); };
    apply(exact, rhs);
    DNDS::Linear::GMRES_LeftPreconditioned<KrylovField> gmres(24, initialize);
    gmres.solve(apply, precondition, dot, rhs, solution, 20, [](int, double residual, double initial)
                { return residual < 1e-11 * initial; });
    apply(solution, check);
    DNDS::Algebra::Axpy(*check.field, -1., *rhs.field);
    const auto relative = std::sqrt(dot(check, check) / dot(rhs, rhs));
    CHECK(relative < 1e-9); // Independent final residual; return status alone is insufficient.
    DNDS::Algebra::Axpy(*solution.field, -1., *exact.field);
    CHECK(std::sqrt(dot(solution, solution) / dot(exact, exact)) < 1e-8);
}

int main(int argc, char **argv)
{
    // Native row kernels may create OMP/CPU workers even with controller MPI.
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    DNDS::MPI::Finalize();
    return result;
}
