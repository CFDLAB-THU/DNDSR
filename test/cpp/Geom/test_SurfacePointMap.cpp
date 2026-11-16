#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Spatial/SurfacePointMap.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

namespace
{
    constexpr double warp = 0.25;
    G::tPoint QuadPoint(double a, double b, int rank = 0) { return {4. * rank + a, b, 1 + warp * a * b}; }
    G::tPoint QuadNormal(double a, double b) { return G::tPoint(-warp * b, -warp * a, 1).normalized(); }
    G::RigidPose Pose(int rank)
    {
        G::RigidPose pose;
        pose.rotation = Eigen::AngleAxisd(0.2 + 0.1 * rank, G::tPoint::UnitY()).toRotationMatrix();
        pose.translation = {0.3, -0.4, 0.2};
        return pose;
    }
    R::Array<double> MixedCoordinates(int rank)
    {
        R::Array<double> result(R::RowLayout::Ragged({12, 9, 6}));
        auto write = result.Write();
        const std::vector<std::vector<G::tPoint>> nodes{
            {QuadPoint(0, 0, rank), QuadPoint(1, 0, rank), QuadPoint(1, 1, rank), QuadPoint(0, 1, rank)},
            {{4. * rank, 2, 1}, {4. * rank + 1, 2, 1}, {4. * rank, 3, 1}},
            {{4. * rank, 4, 0}, {4. * rank + 1, 4, 0}}};
        for (I iElement = 0; iElement < 3; ++iElement)
            for (I iNode = 0; iNode < I(nodes[iElement].size()); ++iNode)
                for (int d = 0; d < 3; ++d)
                    write.View()(iElement, 3 * iNode + d) = nodes[iElement][iNode][d];
        return result;
    }
    std::vector<G::SurfaceElementRecord> Records(int rank)
    {
        return {{3 * rank, G::Elem::Quad4}, {3 * rank + 1, G::Elem::Tri3}, {3 * rank + 2, G::Elem::Line2}};
    }
}

TEST_CASE("Element geometry projects analytically known line triangle and warped quad points")
{
    auto coordinates = MixedCoordinates(0);
    auto read = coordinates.Read();
    const auto view = read.View();
    G::PointProjectionSettings settings;
    settings.maxGap = 0.03;
    const G::ElementGeometryView quad(G::Elem::Quad4, view[0].data(), 4);
    const G::ElementGeometryView triangle(G::Elem::Tri3, view[1].data(), 3);
    const G::ElementGeometryView line(G::Elem::Line2, view[2].data(), 2);
    const double a = 0.23, b = 0.67, gap = 0.02;
    const auto expected = QuadPoint(a, b);
    const auto normal = QuadNormal(a, b);
    const auto curved = quad.Project(expected + gap * normal, settings);
    REQUIRE(curved.Accepted());
    CHECK((curved.reference - G::tPoint(2 * a - 1, 2 * b - 1, 0)).norm() < 1e-11);
    CHECK((curved.point - expected).norm() < 1e-11);
    CHECK((curved.normal - normal).norm() < 1e-11);
    CHECK(curved.distance == doctest::Approx(gap).epsilon(1e-10));
    CHECK(curved.stationarity < 1e-11);
    const auto tri = triangle.Project({0.2, 2.3, 1.02}, settings);
    REQUIRE(tri.Accepted());
    CHECK((tri.reference - G::tPoint(0.2, 0.3, 0)).norm() < 1e-12);
    CHECK((tri.point - G::tPoint(0.2, 2.3, 1)).norm() < 1e-12);
    CHECK((tri.normal - G::tPoint::UnitZ()).norm() < 1e-12);
    const auto segment = line.Project({0.4, 4.02, 0}, settings);
    REQUIRE(segment.Accepted());
    CHECK((segment.point - G::tPoint(0.4, 4, 0)).norm() < 1e-12);
    CHECK(segment.normal.isZero()); // A 3D line has no unique surface normal.
    CHECK(line.Project({1.1, 4, 0}, settings).status == G::PointProjectionStatus::OutsideReference);
    CHECK(line.Project({0.5, 4.1, 0}, settings).status == G::PointProjectionStatus::GapExceeded);
    settings.maxIterations = 1;
    CHECK(quad.Project(expected + gap * normal, settings).status == G::PointProjectionStatus::NotConverged);
    G::tSmallCoords degenerate = G::tSmallCoords::Zero(3, 3);
    const G::ElementGeometryView bad(G::Elem::Tri3, degenerate.data(), 3);
    CHECK_THROWS(bad.Validate());
    CHECK(bad.Project(G::tPoint::Zero()).status == G::PointProjectionStatus::Degenerate);
    CHECK_THROWS(line.Project(G::tPoint::Constant(std::numeric_limits<double>::quiet_NaN())));
    CHECK_THROWS(G::ElementGeometryView(G::Elem::Line3, view[0].data(), 3));

    const auto pose = Pose(0);
    const auto bounds = quad.Bounds(pose, 1e-5);
    for (int i = 0; i <= 10; ++i)
        for (int j = 0; j <= 10; ++j)
            CHECK(bounds.Contains(pose.Apply(QuadPoint(i / 10., j / 10.))));
    CHECK(bounds.Intersects(quad.Bounds(pose)));
    G::SpatialBox distant;
    distant.Add(G::tPoint::Constant(100));
    CHECK_FALSE(bounds.Intersects(distant));
}

TEST_CASE("Surface snapshots freeze shared storage and local maps retain their geometry")
{
    auto coordinates = MixedCoordinates(0);
    auto surface = G::SurfacePatch::Create(Records(0), coordinates);
    CHECK_THROWS(coordinates.Write());
    R::Array<double> points(R::RowLayout::Dense(1, 3));
    R::Array<I> ids(R::RowLayout::Dense(1, 1));
    {
        auto p = points.Write();
        auto k = ids.Write();
        const G::tPoint query = QuadPoint(0.3, 0.4) + 0.02 * QuadNormal(0.3, 0.4);
        std::copy_n(query.data(), 3, p.View()[0].data());
        k.View()(0, 0) = 9;
    }
    G::SurfaceSearchSettings settings;
    settings.maxGap = 0.03;
    const auto map = G::SurfacePointMap::Match(surface, points, ids, settings);
    const auto nativeSurface = surface.WithBackend(G::BoundingBoxBackend::Native);
    const auto nativeMap = G::SurfacePointMap::Match(nativeSurface, points, ids, settings);
    CHECK(surface.Backend() == G::BoundingBoxBackend::CGAL);
    CHECK(nativeSurface.Backend() == G::BoundingBoxBackend::Native);
    CHECK((nativeMap.Get(0).surfacePoint - map.Get(0).surfacePoint).norm() < 1e-12);
    CHECK(nativeSurface.WithPose(Pose(0)).Backend() == G::BoundingBoxBackend::Native);
    CHECK_THROWS(map.ValidateFor(nativeSurface));
    surface = surface.WithPose(Pose(0));
    CHECK_THROWS(map.ValidateFor(surface));
    // The original snapshot (including coordinates and its read lease) stays
    // alive through the map, after the caller replaces its own handle.
    CHECK_THROWS(coordinates.Write());
    CHECK((map.Get(0).surfacePoint - QuadPoint(0.3, 0.4)).norm() < 1e-11);
}

TEST_CASE("Distributed mesh-independent surface location agrees across routing modes and transfers element data")
{
    const auto surface = G::SurfacePatch::Create(Records(mpi.rank), MixedCoordinates(mpi.rank), Pose(mpi.rank));
    const int owner = (mpi.rank + 1) % mpi.size;
    const std::array<G::tPoint, 3> expected{QuadPoint(0.23, 0.67, owner), G::tPoint(4. * owner + 0.2, 2.3, 1), G::tPoint(4. * owner + 0.4, 4, 0)};
    const std::array<G::tPoint, 3> offset{0.02 * QuadNormal(0.23, 0.67), G::tPoint(0, 0, 0.02), G::tPoint(0, 0.02, 0)};
    const std::array<I, 3> keys{7, 2, 9};
    R::Array<double> points(R::RowLayout::Dense(3, 3));
    R::Array<I> ids(R::RowLayout::Dense(3, 1));
    R::Array<I> payload(R::RowLayout::Dense(3, 2)), received(R::RowLayout::Dense(3, 2));
    {
        auto p = points.Write();
        auto k = ids.Write();
        auto data = payload.Write();
        for (I iQuery = 0; iQuery < 3; ++iQuery)
        {
            const auto query = Pose(owner).Apply(expected[iQuery] + offset[iQuery]);
            std::copy_n(query.data(), 3, p.View()[iQuery].data());
            k.View()(iQuery, 0) = keys[iQuery];
            data.View()(iQuery, 0) = 3 * mpi.rank + iQuery;
            data.View()(iQuery, 1) = mpi.rank;
        }
    }
    std::optional<G::DistributedSurfacePointMap> baseline;
    G::SurfaceSearchSettings settings;
    settings.maxGap = 0.03;
    settings.coarseBoxesPerRank = 3;
    auto unchecked = std::make_shared<R::CommunicationContext>(mpi.comm, false);
    // A caller's retained binding must not occupy the locator's private tag.
    const auto retainedTag = unchecked->Reserve(0);
    for (auto backend : {G::BoundingBoxBackend::Native, G::BoundingBoxBackend::CGAL})
        for (auto mode : {G::SurfaceRoutingMode::AllGatherSurface, G::SurfaceRoutingMode::CoarseBoxes})
        {
            const auto active = surface.WithBackend(backend);
            auto map = G::DistributedSurfacePointMap::Match(mpi, active, points, ids, mode, settings,
                                                            mode == G::SurfaceRoutingMode::CoarseBoxes ? unchecked : nullptr);
            REQUIRE(map.Size() == 3);
            for (I iQuery = 0; iQuery < 3; ++iQuery)
            {
                const auto match = map.Get(iQuery);
                CHECK(match.queryId == keys[iQuery]);
                CHECK(match.element == iQuery);
                CHECK(match.elementId == 3 * owner + iQuery);
                CHECK(match.geometryRank == owner);
                CHECK((match.surfacePoint - expected[iQuery]).norm() < 1e-11);
                CHECK((match.projectedPoint - Pose(owner).Apply(expected[iQuery])).norm() < 1e-11);
                CHECK(match.distance == doctest::Approx(0.02).epsilon(1e-9));
                if (iQuery == 0)
                    CHECK((match.normal - Pose(owner).rotation * QuadNormal(0.23, 0.67)).norm() < 1e-11);
                if (baseline)
                {
                    const auto other = baseline->Get(iQuery);
                    CHECK(match.elementId == other.elementId);
                    CHECK((match.reference - other.reference).norm() < 1e-12);
                    CHECK((match.projectedPoint - other.projectedPoint).norm() < 1e-12);
                    CHECK((match.normal - other.normal).norm() < 1e-12);
                }
            }
            const auto route = map.BuildElementDataRoute();
            CHECK(route.Context()->ChecksCollectivePreflight() == (mode != G::SurfaceRoutingMode::CoarseBoxes));
            auto binding = R::PayloadPlan(route, payload.Layout(), received.Layout(), R::ElementSchema::Arithmetic<I>()).Bind(payload, received, R::TransferTransport::Direct, {R::TransferMode::Transient, route.Context(), 0});
            binding.Begin().Wait();
            auto read = received.Read();
            for (I iQuery = 0; iQuery < 3; ++iQuery)
            {
                CHECK(read.View()(iQuery, 0) == 3 * owner + iQuery);
                CHECK(read.View()(iQuery, 1) == owner);
            }
            if (!baseline)
                baseline = map;
            CHECK_NOTHROW(map.ValidateFor(active));
            CHECK_THROWS(map.ValidateFor(active.WithPose({})));
        }
    // Backend selection is local execution policy, not a distributed geometry
    // setting. Different ranks can use different trees without changing donors.
    const auto mixed = surface.WithBackend(mpi.rank % 2 ? G::BoundingBoxBackend::Native : G::BoundingBoxBackend::CGAL);
    const auto mixedMap = G::DistributedSurfacePointMap::Match(mpi, mixed, points, ids, G::SurfaceRoutingMode::CoarseBoxes, settings);
    for (I iQuery = 0; iQuery < 3; ++iQuery)
    {
        CHECK(mixedMap.Get(iQuery).elementId == baseline->Get(iQuery).elementId);
        CHECK((mixedMap.Get(iQuery).projectedPoint - baseline->Get(iQuery).projectedPoint).norm() < 1e-12);
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
