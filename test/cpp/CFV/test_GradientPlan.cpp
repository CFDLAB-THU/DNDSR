#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/GradientPlan.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include "../Geom/InterfaceFixture.hpp"
#include "RotatedSectorFixture.hpp"

namespace C = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Standalone FV gradients recover all three spatial directions on a volume cell")
{
    auto mesh = InterfaceFixture::SingleCell(mpi, true);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    for (auto method : {C::GradientMethod::GreenGauss, C::GradientMethod::LeastSquares, C::GradientMethod::GGMP})
    {
        auto plan = C::GradientPlan::Prepare(*mesh, metrics, method);
        auto mean = metrics.BuildField(1), gradient = metrics.BuildField(3);
        auto boundary = plan.BuildBoundaryValues(1);
        const G::tPoint exact{2, -0.7, 0.3};
        {
            auto geometry = metrics.AcquireRead();
            auto u = mean.Owned().Write();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                u.View()(iCell, 0) = 1 + exact.dot(geometry.HostView().GetCellBary(iCell));
            auto b = boundary.Write();
            for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
                b.View()(iPoint, 0) = 1 + exact.dot(plan.GetBoundaryQuadrature(iPoint).point);
        }
        metrics.BindHalo(mean).Begin().Wait();
        plan.Evaluate(mean, boundary, gradient);
        auto read = gradient.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I d = 0; d < 3; ++d)
                CHECK(read.View()(iCell, d) == doctest::Approx(exact[d]).epsilon(1e-12));
    }
}

TEST_CASE("Standalone gradients recover linear fields with physical boundary samples")
{
    for (bool skew : {false, true})
    {
        auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true);
        auto mesh = fixture.mesh;
        if (skew)
            mesh->TransformCoords([](const G::tPoint &p) -> G::tPoint
                                  { return {p[0] + 0.23 * p[1], 0.7 * p[1], p[2]}; });
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
        for (auto method : {C::GradientMethod::GreenGauss, C::GradientMethod::LeastSquares, C::GradientMethod::GGMP})
        {
            if (skew && method == C::GradientMethod::GreenGauss)
                continue; // uncorrected GG is not linearly exact here
            auto plan = C::GradientPlan::Prepare(*mesh, metrics, method);
            auto mean = metrics.BuildField(2), gradient = metrics.BuildField(4);
            auto boundary = plan.BuildBoundaryValues(2);
            const auto function = [](const G::tPoint &p, I component)
            { return component == 0 ? 1 + 2 * p[0] - 0.7 * p[1] : -3 + 0.2 * p[0] + 4 * p[1]; };
            {
                auto geometry = metrics.AcquireRead();
                auto write = mean.Owned().Write();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I component = 0; component < 2; ++component)
                        write.View()(iCell, component) = function(geometry.HostView().GetCellBary(iCell), component);
                auto values = boundary.Write();
                for (I point = 0; point < plan.NumBoundaryQuadrature(); ++point)
                    for (I component = 0; component < 2; ++component)
                        values.View()(point, component) = function(plan.GetBoundaryQuadrature(point).point, component);
            }
            metrics.BindHalo(mean).Begin().Wait();
            plan.Evaluate(mean, boundary, gradient);
            const double expected[]{2, -0.7, 0.2, 4};
            {
                auto read = gradient.Owned().Read();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I component = 0; component < 4; ++component)
                        CHECK(read.View()(iCell, component) == doctest::Approx(expected[component]).epsilon(1e-11));
            }
            const auto reconstruction = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics);
            auto coefficients = metrics.BuildField(2 * reconstruction.NRecDOF());
            plan.ConvertToCoefficients(gradient, reconstruction, coefficients);
            auto recovered = metrics.BuildField(R::RowLayout::ColMajor(0, 4, {8, 1}));
            plan.ExtractLinearGradients(coefficients, reconstruction, recovered);
            {
                const auto read = recovered.Owned().Read();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I j = 0; j < 4; ++j)
                        CHECK(read.View<R::UniformTableRows>()(iCell, j) == doctest::Approx(expected[j]).epsilon(1e-11));
            }
            {
                auto geometry = metrics.AcquireRead();
                auto read = coefficients.Owned().Read();
                const I n = reconstruction.NRecDOF();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I component = 0; component < 2; ++component)
                        for (I mode = 0; mode < n; ++mode)
                            CHECK(read.View()(iCell, component * n + mode) == doctest::Approx(
                                                                                  mode < 2 ? expected[component * 2 + mode] * geometry.HostView().cells(iCell, 7) : 0)
                                                                                  .epsilon(1e-11));
            }
            const I modes = reconstruction.NRecDOF();
            {
                auto write = coefficients.Owned().Write();
                for (I cell = 0; cell < metrics.NumCell(); ++cell)
                    for (I j = 0; j < 2 * modes; ++j)
                        write.View()(cell, j) = 42;
            }
            R::EagerEngine engine;
            plan.SetLinearCoefficients(engine, engine.CreateCPUStream(), gradient, reconstruction, coefficients, 1, 1);
            engine.Close();
            {
                const auto geometry = metrics.AcquireRead();
                const auto read = coefficients.Owned().Read();
                for (I cell = 0; cell < metrics.NumCell(); ++cell)
                    for (I m = 0; m < modes; ++m)
                    {
                        CHECK(read.View()(cell, m) == 42);
                        CHECK(read.View()(cell, modes + m) == doctest::Approx(
                                                                  m < 2 ? expected[2 + m] * geometry.HostView().cells(cell, 7) : 0)
                                                                  .epsilon(1e-11));
                    }
            }
            R::Array<double> wrong(R::RowLayout::Dense(plan.NumBoundaryQuadrature(), 3));
            CHECK_THROWS(plan.Evaluate(mean, wrong, gradient));
        }
    }
}

TEST_CASE("Weighted P1 gradient matches independent least-squares fit on a stretched grid")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, false, 8).mesh;
    mesh->TransformCoords([](const G::tPoint &p) -> G::tPoint
                          { return {p[0] + .04 * p[1], .1 * p[1], p[2]}; });
    auto metric = C::FiniteVolumePlan::ConstructMetrics(mesh);
    const auto meshRead = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
    const auto meshView = meshRead.HostView();
    const auto metricRead = metric.AcquireRead();
    const auto geometry = metricRead.HostView();
    const auto polynomial = [](const G::tPoint &p)
    { return p[0] * p[0] + 2 * p[0] * p[1] + 3 * p[1] * p[1]; };
    auto mean = metric.BuildField(1), gradient = metric.BuildField(2);
    {
        auto write = mean.Owned().Write();
        for (I cell = 0; cell < metric.NumCell(); ++cell)
            write.View()(cell, 0) = polynomial(geometry.GetCellBary(cell));
    }
    metric.BindHalo(mean).Begin().Wait();
    for (const C::LeastSquaresWeights weights : {C::LeastSquaresWeights{}, {1, -1}})
    {
        auto plan = C::GradientPlan::Prepare(*mesh, metric, C::GradientMethod::LeastSquares, 1e-13, {}, weights);
        auto boundary = plan.BuildBoundaryValues(1);
        {
            auto write = boundary.Write();
            for (I row = 0; row < plan.NumBoundaryQuadrature(); ++row)
                write.View()(row, 0) = polynomial(plan.GetBoundaryQuadrature(row).point);
        }
        plan.Evaluate(mean, boundary, gradient);
        const auto result = gradient.Owned().Read();
        for (I cell = 0; cell < metric.NumCell(); ++cell)
        {
            const auto center = geometry.GetCellBary(cell);
            Eigen::Matrix2d normal = Eigen::Matrix2d::Zero();
            Eigen::Vector2d right = Eigen::Vector2d::Zero();
            const auto faces = meshView.CellFaces(cell);
            for (DNDS::rowsize slot = 0; slot < faces.size(); ++slot)
            {
                const I face = faces[slot], neighbor = meshView.CellFaceOther(cell, face, slot);
                const auto sample = neighbor == DNDS::UnInitIndex ? geometry.GetFaceCent(face) : geometry.GetCellBary(neighbor);
                const Eigen::Vector2d delta = (sample - center).head<2>();
                const double weight = std::pow(geometry.GetFaceArea(face), weights.areaPower) *
                                      std::pow(delta.norm(), weights.distancePower);
                normal += weight * delta * delta.transpose();
                right += weight * delta * (polynomial(sample) - polynomial(center));
            }
            const Eigen::Vector2d expected = normal.inverse() * right;
            for (int d = 0; d < 2; ++d)
                CHECK(result.View()(cell, d) == doctest::Approx(expected[d]).epsilon(2e-11));
        }
    }
}

TEST_CASE("Weighted P1 reproduces a rotation-equivariant affine vector across a periodic seam")
{
    auto mesh = CFVTest::Sector(mpi);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    auto plan = C::GradientPlan::Prepare(*mesh, metrics, C::GradientMethod::LeastSquares);
    auto mean = metrics.BuildField(3), gradient = metrics.BuildField(6);
    auto boundary = plan.BuildBoundaryValues(3);
    {
        const auto read = metrics.AcquireRead();
        const auto geometry = read.HostView();
        auto owned = mean.Owned().Write(), traces = boundary.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (int j = 0; j < 3; ++j)
                owned.View()(iCell, j) = geometry.GetCellBary(iCell)[j];
        for (I iFace = 0; iFace < plan.NumBoundaryQuadrature(); ++iFace)
            for (int j = 0; j < 3; ++j)
                traces.View()(iFace, j) = plan.GetBoundaryQuadrature(iFace).point[j];
    }
    metrics.BindHalo(mean).Begin().Wait();
    plan.Evaluate(mean, boundary, gradient, {0});
    const auto read = gradient.Owned().Read();
    for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        for (int j = 0; j < 3; ++j)
            for (int d = 0; d < 2; ++d)
                CHECK(std::abs(read.View()(iCell, j * 2 + d) - double(d == j)) < 1e-12);
}

TEST_CASE("Weighted P1 gradients use the curved triangular boundary sample locations")
{
    const bool active = mpi.rank == 0;
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, active ? 6 : 0);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, active ? 1 : 0);
    auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, active ? 3 : 0);
    std::vector<std::vector<I>> cones, faces;
    if (active)
    {
        cones.push_back({0, 1, 2, 3, 4, 5});
        faces = {{0, 1, 3}, {1, 2, 4}, {2, 0, 5}};
    }
    G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, InterfaceFixture::Rows(cones), InterfaceFixture::Rows(faces),
                             R::Array<double>(R::RowLayout::Dense(nodes.NumOwned(), 3)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(cells.NumOwned(), 1)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(boundaries.NumOwned(), 1))};
    if (active)
    {
        const std::vector<G::tPoint> points{{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {1, -.35, 0}, {1, 1, 0}, {0, 1, 0}};
        auto write = input.coordinates.Write();
        for (I row = 0; row < nodes.NumOwned(); ++row)
            for (int d = 0; d < 3; ++d)
                write.View()(row, d) = points[row][d];
        input.cellInfo.Write().View()(0, 0) = G::ElemInfo{G::Elem::Tri6, G::INTERNAL_ZONE};
        auto info = input.boundaryInfo.Write();
        for (I row = 0; row < boundaries.NumOwned(); ++row)
            info.View()(row, 0) = G::ElemInfo{G::Elem::Line3, int(7 + row)};
    }
    auto mesh = G::UnstructuredMesh::FromNative(mpi, G::ConstructNativeMesh(std::move(input)));
    mesh->PrepareNativeTopology();
    auto metric = C::FiniteVolumePlan::ConstructMetrics(mesh);
    auto plan = C::GradientPlan::Prepare(*mesh, metric, C::GradientMethod::LeastSquares);
    auto mean = metric.BuildField(1), result = metric.BuildField(2);
    auto values = plan.BuildBoundaryValues(1);
    const G::tPoint exact{2, -3, 0};
    if (active)
    {
        mean.Owned().Write().View()(0, 0) = 4 + exact.dot(metric.AcquireRead().HostView().GetCellBary(0));
        auto write = values.Write();
        bool curvedPoint = false;
        for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto point = plan.GetBoundaryQuadrature(iPoint).point;
            curvedPoint = curvedPoint || point[1] < -.1;
            write.View()(iPoint, 0) = 4 + exact.dot(point);
        }
        CHECK(curvedPoint);
    }
    metric.BindHalo(mean).Begin().Wait();
    plan.Evaluate(mean, values, result);
    if (active)
    {
        const auto read = result.Owned().Read();
        for (int d = 0; d < 2; ++d)
            CHECK(read.View()(0, d) == doctest::Approx(exact[d]).epsilon(1e-12));
    }
}

TEST_CASE("Gradient core phases exclude active MPI ghost writes across table layouts")
{
    // At np2 both ranks own cells; np4/8 additionally exercise an empty owner.
    // Three or more complete grid rows per active rank ensure a real core and
    // dependent region, instead of a vacuous empty-core split comparison.
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    const auto partition = metrics.CellWorkPartition(), shared = metrics.CellWorkPartition();
    CHECK(&partition.Core() == &shared.Core());
    CHECK(&partition.HaloDependent() == &shared.HaloDependent());
    if (metrics.NumCell())
    {
        REQUIRE_FALSE(partition.Core().empty());
        if (mpi.size > 1)
            REQUIRE_FALSE(partition.HaloDependent().empty());
    }
    const double expected[]{2, -.7, -.2, .3, .6, -.1, -.7, 1.2};
    const auto value = [&](const G::tPoint &point, I component)
    { return 1 + component + expected[2 * component] * point[0] + expected[2 * component + 1] * point[1]; };
    for (auto method : {C::GradientMethod::GreenGauss, C::GradientMethod::LeastSquares, C::GradientMethod::GGMP})
    {
        auto plan = C::GradientPlan::Prepare(*fixture.mesh, metrics, method);
        auto boundary = plan.BuildBoundaryValues(4);
        {
            auto write = boundary.Write();
            for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
                for (I component = 0; component < 4; ++component)
                    write.View()(iPoint, component) = value(plan.GetBoundaryQuadrature(iPoint).point, component);
        }
        for (int layout = 0; layout < 3; ++layout)
        {
            const auto shape = [&](I width)
            {
                return layout == 0 ? R::RowLayout::Dense(0, width, width + 2) : layout == 1 ? R::RowLayout::ColMajor(0, width, {16, 1})
                                                                                            : R::RowLayout::TiledColMajor(0, width, 16);
            };
            auto mean = metrics.BuildField(shape(4)), full = metrics.BuildField(shape(8)), split = metrics.BuildField(shape(8));
            {
                auto geometry = metrics.AcquireRead();
                auto write = mean.Owned().Write();
                const auto u = write.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I component = 0; component < 4; ++component)
                        u(iCell, component) = value(geometry.HostView().GetCellBary(iCell), component);
                auto output = split.Owned().Write();
                const auto out = output.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I j = 0; j < 8; ++j)
                        out(iCell, j) = 123456;
            }
            const auto version = mean.Owned().Version();
            auto halo = metrics.BindHalo(mean);
            halo.Begin().Wait();
            plan.Evaluate(mean, boundary, full, {1});
            auto pending = halo.Begin();
            CHECK_NOTHROW(plan.Evaluate(mean, boundary, split, {1}, C::ReconstructionRegion::Core));
            {
                auto read = split.Owned().Read();
                const auto out = read.View<R::UniformTableRows>();
                for (auto iCell : partition.Core())
                    for (I j = 0; j < 8; ++j)
                        CHECK(out(iCell, j) == doctest::Approx(expected[j]).epsilon(1e-11));
                for (auto iCell : partition.HaloDependent())
                    for (I j = 0; j < 8; ++j)
                        CHECK(out(iCell, j) == 123456);
            }
            if (metrics.Cells().NumGhost())
            {
                const auto before = split.Owned().Version();
                CHECK_THROWS(plan.Evaluate(mean, boundary, split, {1}, C::ReconstructionRegion::HaloDependent));
                CHECK_THROWS(plan.Evaluate(mean, boundary, split, {1}));
                CHECK(split.Owned().Version() == before);
            }
            pending.Wait();
            plan.Evaluate(mean, boundary, split, {1}, C::ReconstructionRegion::HaloDependent);
            {
                auto a = full.Owned().Read(), b = split.Owned().Read();
                const auto reference = a.View<R::UniformTableRows>(), actual = b.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I j = 0; j < 8; ++j)
                    {
                        CHECK(actual(iCell, j) == reference(iCell, j));
                        CHECK(actual(iCell, j) == doctest::Approx(expected[j]).epsilon(1e-11));
                    }
            }
            CHECK(mean.Owned().Version() == version);
            // The same split contract holds when arithmetic runs on a CPU
            // worker. Only the controller progresses MPI; a real pending halo
            // write must not conflict with the queued owned-only phase.
            auto queued = metrics.BuildField(shape(8));
            auto workspace = plan.BuildWorkspace(4, {1});
            for (const int workers : {0, 2})
            {
                R::EagerEngine engine(workers);
                const auto stream = engine.CreateCPUStream();
                auto exchange = halo.Begin();
                plan.Evaluate(engine, stream, mean, boundary, queued, workspace, C::ReconstructionRegion::Core);
                exchange.Wait();
                plan.Evaluate(engine, stream, mean, boundary, queued, workspace, C::ReconstructionRegion::HaloDependent);
                engine.Close();
                const auto reference = full.Owned().Read(), actual = queued.Owned().Read();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I j = 0; j < 8; ++j)
                        CHECK(actual.View<R::UniformTableRows>()(iCell, j) == reference.View<R::UniformTableRows>()(iCell, j));
            }
            CHECK_THROWS(plan.Evaluate(mean, boundary, split, {1}, static_cast<C::ReconstructionRegion>(100)));
            if (metrics.NumCell() && metrics.Cells().NumGhost())
            {
                // Core deliberately never acquires the input ghost read, so
                // explicit output/input alias validation must cover it too.
                const auto inputBytes = size_t(mean.Ghost().RowCount() * 4) * sizeof(double);
                const auto outputBytes = size_t(split.Owned().RowCount() * 8) * sizeof(double);
                auto buffer = R::Buffer::Allocate(std::max(inputBytes, outputBytes));
                auto aliasedInput = mean, aliasedOutput = split;
                aliasedInput.Ghost() = R::Array<double>::FromStorage(R::RowLayout::Dense(mean.Ghost().RowCount(), 4), buffer.Subspan(0, inputBytes));
                aliasedOutput.Owned() = R::Array<double>::FromStorage(R::RowLayout::Dense(split.Owned().RowCount(), 8), buffer.Subspan(0, outputBytes));
                CHECK_THROWS_WITH_AS(plan.Evaluate(aliasedInput, boundary, aliasedOutput, {1}, C::ReconstructionRegion::Core),
                                     doctest::Contains("gradient output overlaps"), std::exception);
            }
        }
    }
}

TEST_CASE("Physical-gradient WBAP is direct, characteristic-optional, and split-equivalent")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 12);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    auto plan = C::GradientPlan::Prepare(*fixture.mesh, metrics);
    const I dim = metrics.Dimension();
    auto input = metrics.BuildField(dim), direct = metrics.BuildField(dim),
         identity = metrics.BuildField(dim), split = metrics.BuildField(dim), inactive = metrics.BuildField(dim);
    {
        auto write = input.Owned().Write();
        const auto values = write.View<R::UniformTableRows>();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto global = metrics.Cells().LocalToGlobal(iCell);
            for (I d = 0; d < dim; ++d)
                values(iCell, d) = 0.25 + 0.07 * double((global * 11 + d * 17) % 23);
        }
    }
    metrics.BindHalo(input).Begin().Wait();

    const auto disabled = plan.BuildFaceTransform(0, 0);
    CHECK(disabled.count == 0);
    CHECK(disabled.forward.RowCount() == 0);
    CHECK(disabled.inverse.RowCount() == 0);
    plan.LimitWBAP3(input, direct, {}, std::cref(disabled));
    auto enabled = plan.BuildFaceTransform(0, 1);
    plan.LimitWBAP3(input, identity, {}, std::cref(enabled));

    // Independent scalar oracle: one face sample per neighbor, followed by the
    // central multiway WBAP aggregation. It deliberately does not call CFV's
    // WBAP helper or inspect GradientPlan's private prepared rows.
    {
        auto meshRead = fixture.mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
        const auto mesh = meshRead.HostView();
        auto owned = input.Owned().Read(), ghost = input.Ghost().Read();
        R::PairReadView<double, R::AccessBoundsChecks, R::UniformTableRows> values{
            owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
        auto actualRead = direct.Owned().Read(), transformedRead = identity.Owned().Read();
        const auto actual = actualRead.View<R::UniformTableRows>();
        const auto transformed = transformedRead.View<R::UniformTableRows>();
        const double epsilon = std::pow(DNDS::verySmallReal, .25);
        const auto pair = [epsilon](double center, double other)
        {
            if (other == 0)
                return 0.;
            const double ratio = center / (std::abs(other) + epsilon) * ((other > 0) - (other < 0));
            const double r2 = ratio * ratio;
            return center * (1 + r2 * ratio) / (1 + r2 * r2);
        };
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I d = 0; d < dim; ++d)
            {
                const double center = values(iCell, d);
                double up = 10, down = 10, valid = 1;
                const auto cellFaces = mesh.CellFaces(iCell);
                for (I slot = 0; slot < I(cellFaces.size()); ++slot)
                {
                    const I neighbor = mesh.CellFaceOther(iCell, cellFaces[slot], DNDS::rowsize(slot));
                    if (neighbor == DNDS::UnInitIndex)
                        continue;
                    const double limited = pair(center, values(neighbor, d));
                    const double ratio = center / (std::abs(limited) + epsilon) * ((limited > 0) - (limited < 0));
                    up += ratio * ratio * ratio;
                    down += ratio * ratio * ratio * ratio;
                    if (limited == 0)
                        valid = 0;
                }
                const double expected = center * valid * up / down;
                CHECK(actual(iCell, d) == doctest::Approx(expected).epsilon(1e-13));
                CHECK(transformed(iCell, d) == actual(iCell, d));
            }
    }

    R::Array<double> activation(R::RowLayout::Dense(metrics.NumCell(), 1));
    {
        auto write = activation.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            write.View()(iCell, 0) = -1;
    }
    plan.LimitWBAP3(input, inactive, {}, std::nullopt, C::ReconstructionRegion::All,
                    std::cref(activation), 0);
    {
        auto sourceRead = input.Owned().Read(), inactiveRead = inactive.Owned().Read();
        const auto source = sourceRead.View<R::UniformTableRows>();
        const auto actual = inactiveRead.View<R::UniformTableRows>();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I d = 0; d < dim; ++d)
                CHECK(actual(iCell, d) == source(iCell, d));
    }

    plan.LimitWBAP3(input, split, {}, std::nullopt, C::ReconstructionRegion::Core);
    plan.LimitWBAP3(input, split, {}, std::nullopt, C::ReconstructionRegion::HaloDependent);
    auto queued = metrics.BuildField(dim);
    auto work = plan.BuildWorkspace(1, {}, true);
    {
        R::EagerEngine engine(2);
        const auto stream = engine.CreateCPUStream();
        auto exchange = metrics.BindHalo(input).Begin();
        plan.LimitWBAP3(engine, stream, input, queued, work, std::nullopt, C::ReconstructionRegion::Core);
        exchange.Wait();
        plan.LimitWBAP3(engine, stream, input, queued, work, std::nullopt, C::ReconstructionRegion::HaloDependent);
        engine.Close();
    }
    {
        auto referenceRead = direct.Owned().Read(), splitRead = split.Owned().Read();
        const auto queuedRead = queued.Owned().Read();
        const auto reference = referenceRead.View<R::UniformTableRows>();
        const auto actual = splitRead.View<R::UniformTableRows>();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I d = 0; d < dim; ++d)
            {
                CHECK(actual(iCell, d) == reference(iCell, d));
                CHECK(queuedRead.View()(iCell, d) == reference(iCell, d));
            }
    }
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
