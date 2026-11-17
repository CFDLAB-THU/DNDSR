#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Unstructured/NativeMeshBuilder.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include <filesystem>
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::ssp<G::UnstructuredMesh> Fixture(DNDS::MPIInfo mpi, bool curved = false, bool collectivePreflight = true)
{
    const bool ownsNodes = mpi.rank == mpi.size - 1;
    const bool ownsCell = mpi.rank == 0;
    const bool ownsBoundary = mpi.rank == (mpi.size > 1 ? 1 : 0);
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, ownsNodes ? (curved ? 7 : 3) : 0);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, ownsCell ? 1 : 0);
    auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, ownsBoundary ? 1 : 0);
    auto cone = [](bool owned, const std::vector<I> &values)
    {
        R::RowBuilder rows(owned ? 1 : 0, sizeof(I));
        if (owned)
        {
            rows.ResizeRow(0, values.size());
            auto row = rows.Row<I>(0);
            std::copy(values.begin(), values.end(), row.data);
        }
        return R::Array<I>::FromSharedStorage(rows.Commit());
    };
    R::Array<DNDS::real> points(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3));
    R::Array<G::ElemInfo> cellInfo(R::RowLayout::Dense(size_t(cells.NumOwned()), 1));
    R::Array<G::ElemInfo> bndInfo(R::RowLayout::Dense(size_t(boundaries.NumOwned()), 1));
    {
        auto write = points.Write();
        const std::vector<G::tPoint> coordinates = curved ? std::vector<G::tPoint>{{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {1, -0.5, 0}, {1, 1, 0}, {0, 1, 0}, {1, 0.5, 0}} : std::vector<G::tPoint>{{0, 0, 0}, {2, 0, 0}, {0, 1, 0}};
        for (size_t row = 0; row < points.RowCount(); ++row)
            for (size_t component = 0; component < 3; ++component)
                write.View()(row, component) = coordinates[row][component];
        if (ownsCell)
            cellInfo.Write().View()(0, 0) = G::ElemInfo{curved ? G::Elem::Tri6 : G::Elem::Tri3, 31};
        if (ownsBoundary)
            bndInfo.Write().View()(0, 0) = G::ElemInfo{curved ? G::Elem::Line3 : G::Elem::Line2, 24};
    }
    G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, cone(ownsCell, curved ? std::vector<I>{0, 1, 2, 3, 4, 5} : std::vector<I>{0, 1, 2}), cone(ownsBoundary, curved ? std::vector<I>{0, 1, 3} : std::vector<I>{0, 1}), points, cellInfo, bndInfo, {}};
    auto communication = std::make_shared<R::CommunicationContext>(mpi.comm, collectivePreflight);
    auto source = G::ConstructNativeMesh(input, {1, true, std::nullopt, communication});
    source = G::PrepareNativeMeshTopology(*source, mpi);
    return G::UnstructuredMesh::FromNative(mpi, source);
}

TEST_CASE("Disabled-preflight wall distance keeps the analytic remote-wall result")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = Fixture(mpi, false, false);
    G::UnstructuredMesh::WallDistOptions options;
    options.method = 0;
    mesh->BuildNodeWallDist([](auto)
                            { return true; }, options);
    auto requirements = G::MeshReadRequirements::Primary();
    requirements.wallDistance = true;
    auto read = mesh->AcquireRead(requirements);
    const auto view = read.HostView();
    for (I iNode = 0; iNode < view.NumNodeProc(); ++iNode)
    {
        const auto expected = G::tPoint{0, view.Coord(iNode)[1], 0};
        CHECK(view.NodeWallDist(iNode).isApprox(expected, 1e-12));
    }
    CHECK_FALSE(mesh->Entities(G::EntityKind::Node).ChecksCollectivePreflight());
}

TEST_CASE("Native wall distance uses remote boundary selection and retains geometry identity")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = Fixture(mpi);
    const auto namespaceBefore = mesh->CurrentNativeState()->MeshNamespace();
    const auto source = mesh->CurrentNativeState();
    const auto originalView = source->BorrowHostView();
    std::vector<G::tPoint> points;
    for (I node = 0; node < mesh->NumNodeProc(); ++node)
        points.emplace_back(originalView.Coord(node));
    for (int method : {0, 20, 1})
        for (int execution : {0, 1, 3})
        {
            G::UnstructuredMesh::WallDistOptions options;
            options.method = method;
            options.wallDistExecution = execution;
            options.subdivide_quad = 4;
            options.minWallDist = 1.5;
            auto geometry = mesh->versions.Stamp(R::Change::Geometry);
            mesh->BuildNodeWallDist([](auto zone)
                                    { return zone == 24; }, options);
            CHECK(geometry.Valid());
            CHECK(mesh->HasNativeState());
            CHECK(mesh->HasNodeWallDist());
            CHECK(mesh->CurrentNativeState()->MeshNamespace() == namespaceBefore);
            auto read = mesh->AcquireRead(mesh->CurrentNativeState()->Requirements());
            auto view = read.HostView();
            for (I node = 0; node < view.NumNodeProc(); ++node)
            {
                CHECK(view.Coord(node).isApprox(points[size_t(node)]));
                CHECK(view.Coord(node).data() == originalView.Coord(node).data());
                const auto p = view.Coord(node);
                const auto expected = p[1] > 0 ? G::tPoint{0, 1.5, 0} : G::tPoint::Zero().eval();
                CHECK(view.NodeWallDist(node).isApprox(expected, 1e-12));
                CHECK(originalView.Coord(node).isApprox(p));
            }
        }
    auto stamp = mesh->versions.Stamp();
    CHECK_THROWS(mesh->BuildNodeWallDist([&](auto) -> bool
                                         { if (mpi.rank == (mpi.size > 1 ? 1 : 0)) throw std::runtime_error("wall callback failure"); return true; }));
    CHECK(stamp.Valid());
    CHECK_THROWS(mesh->BuildNodeWallDist([&](auto)
                                         { auto rejected = mesh->versions.BeginMutation(R::Change::Values); rejected.Commit(); return true; }));
    CHECK(stamp.Valid());
    G::UnstructuredMesh::WallDistOptions bad;
    bad.method = mpi.rank == 0 ? -1 : 0;
    CHECK_THROWS(mesh->BuildNodeWallDist([](auto)
                                         { return true; }, bad));
    CHECK(stamp.Valid());
    G::UnstructuredMesh::WallDistOptions options;
    options.method = 1;
    options.subdivide_quad = 40;
    CHECK_THROWS(mesh->BuildNodeWallDist([](auto)
                                         { return true; }, options));
    CHECK(stamp.Valid());
    options.subdivide_quad = 4;
    mesh->BuildNodeWallDist([](auto)
                            { return false; }, options);
    for (I node = 0; node < mesh->NumNodeProc(); ++node)
        CHECK(mesh->NodeWallDist(node).norm() == std::pow(DNDS::veryLargeReal, 0.25));
}

TEST_CASE("Wall distance rejects a rank without native authority collectively")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    if (mpi.size == 1)
        return;
    // FromNative retains the preparation context's communicator, which need
    // not be MPI_COMM_WORLD. Reject missing authority on that SAME communicator
    // on every rank, and keep its owner alive while the empty facade uses it.
    auto source = Fixture(mpi);
    auto mesh = mpi.rank == 0
                    ? std::make_shared<G::UnstructuredMesh>(source->getMPI(), 2)
                    : source;
    auto stamp = mesh->versions.Stamp();
    CHECK_THROWS(mesh->BuildNodeWallDist([](auto)
                                         { return true; }));
    CHECK(stamp.Valid());
}

TEST_CASE("Wall vectors rebuilt after near-plane cleanup use the final shape and configured selector")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = Fixture(mpi);
    constexpr DNDS::real epsilon = 1e-4;
    mesh->ApplyAffineGeometryTransform(Eigen::Matrix3d::Identity(), G::tPoint{0, epsilon, 0});
    G::UnstructuredMesh::WallDistOptions options;
    options.method = 0;
    options.minWallDist = 0;
    const auto selected = [](auto zone)
    { return zone == 24; };
    mesh->BuildNodeWallDist(selected, options);
    auto before = mesh->CurrentNativeState();
    for (I node = 0; node < mesh->NumNodeProc(); ++node)
        if (mesh->Entities(G::EntityKind::Node).LocalToGlobal(node) == 2)
            CHECK(mesh->NodeWallDist(node).isApprox(G::tPoint{0, 1, 0}, 1e-12));
    mesh->RectifyNearPlaneCoordinates(2, 1e-3);
    CHECK_FALSE(mesh->HasNodeWallDist());
    CHECK_FALSE(mesh->CurrentNativeRequirements().wallDistance);
    CHECK_THROWS(mesh->NodeWallDist(0));
    mesh->BuildNodeWallDist(selected, options);
    CHECK(mesh->HasNodeWallDist());
    CHECK(mesh->CurrentNativeState()->MeshNamespace() == before->MeshNamespace());
    {
        auto read = mesh->AcquireRead(mesh->CurrentNativeRequirements());
        auto view = read.HostView();
        for (I node = 0; node < view.NumNodeProc(); ++node)
        {
            const auto global = mesh->Entities(G::EntityKind::Node).LocalToGlobal(node);
            const G::tPoint expected = global == 2 ? G::tPoint{0, 1 + epsilon, 0} : G::tPoint::Zero().eval();
            CHECK(view.NodeWallDist(node).isApprox(expected, 1e-12));
            CHECK(view.Coord(node)[1] == doctest::Approx(global == 2 ? 1 + epsilon : 0));
        }
    }
    const auto state = mesh->CurrentNativeState();
    mesh->RectifyNearPlaneCoordinates(2, 1e-3);
    mesh->RectifyBoundaryCoordinates([](auto)
                                     { return 0; });
    CHECK(mesh->CurrentNativeState() == state);
    CHECK(mesh->HasNodeWallDist());
}

TEST_CASE("Native curved wall preserves method conventions and remote midpoint zeros")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = Fixture(mpi, true);
    const auto nodes = mesh->Entities(G::EntityKind::Node);
    for (int method : {0, 20, 1})
    {
        G::UnstructuredMesh::WallDistOptions options;
        options.method = method;
        options.subdivide_quad = 1;
        options.minWallDist = 0;
        mesh->BuildNodeWallDist([](auto)
                                { return true; }, options);
        for (I node = 0; node < mesh->NumNodeProc(); ++node)
        {
            const auto global = nodes.LocalToGlobal(node);
            if (global == 0 || global == 1 || global == 3)
                CHECK(mesh->NodeWallDist(node).isZero(0));
            if (global == 6)
            {
                // Method0 extrudes the endpoint chord. Method1 splits at the
                // curved midpoint and retains the historical third point
                // endpoint1+(0,0,1). Closest point on its first patch is
                // (0.8,0,0.4), on edge endpoint0->third point.
                const G::tPoint expected = method == 1 ? G::tPoint{0.2, 0.5, -0.4} : G::tPoint{0, 0.5, 0};
                CHECK(mesh->NodeWallDist(node).isApprox(expected, 1e-12));
            }
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}

TEST_CASE("Wall distance rejects an unpublished mesh collectively")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = std::make_shared<G::UnstructuredMesh>(mpi, 2);
    const auto stamp = mesh->versions.Stamp();
    CHECK_THROWS(mesh->BuildNodeWallDist([](auto)
                                         { return true; }));
    CHECK(stamp.Valid());
    CHECK_FALSE(mesh->HasNativeState());
}
