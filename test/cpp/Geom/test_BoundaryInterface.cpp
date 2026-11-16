#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Interface/BoundaryInterface.hpp"
#include "NativeMeshFacadeFixture.hpp"
#include "InterfaceFixture.hpp"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Boundary interface projection preserves query IDs and deterministic seams through rigid motion")
{
    auto mesh = NativeFacadeFixture::Make(mpi).Publish(mpi);
    auto patch = G::BoundaryPatch::Create(mesh, {2, 1, 0});
    R::Array<double> points(R::RowLayout::Dense(3, 3));
    R::Array<I> ids(R::RowLayout::Dense(3, 1));
    const std::array<G::tPoint, 3> expected{{{.25, 0, 0}, {1, .7, 0}, {1, 0, 0}}};
    {
        auto write = points.Write();
        auto id = ids.Write();
        for (I i = 0; i < 3; ++i)
        {
            for (int d = 0; d < 3; ++d)
                write.View()(i, d) = expected[i][d];
            id.View()(i, 0) = 90 - 13 * i;
        }
    }
    if (mpi.size > 1)
    {
        CHECK_THROWS(G::InterfacePointMap::Match(patch, points, ids));
        return;
    }
    auto map = G::InterfacePointMap::Match(patch, points, ids);
    REQUIRE(map.Size() == 3);
    CHECK(map.Get(0).donorBoundary == 0);
    CHECK(map.Get(1).donorBoundary == 1);
    CHECK(map.Get(2).donorOriginalId == std::min(mesh->OriginalEntityId(G::EntityKind::Bnd, 0), mesh->OriginalEntityId(G::EntityKind::Bnd, 1)));
    for (I i = 0; i < 3; ++i)
    {
        CHECK(map.Get(i).queryId == 90 - 13 * i);
        CHECK((map.Get(i).donorPoint - expected[i]).norm() < 1e-12);
        CHECK(map.Get(i).distance < 1e-12);
    }
    for (double angle : {0.0, .3, 1.7})
    {
        G::RigidPose pose;
        pose.rotation = Eigen::AngleAxisd(angle, G::tPoint::UnitZ()).toRotationMatrix();
        pose.translation = {.5, -.3, .2};
        auto moved = patch.WithPose(pose);
        {
            auto write = points.Write();
            for (I i = 0; i < 3; ++i)
            {
                const auto p = pose.Apply(expected[i]);
                for (int d = 0; d < 3; ++d)
                    write.View()(i, d) = p[d];
            }
        }
        auto updated = G::InterfacePointMap::Match(moved, points, ids);
        auto rebuilt = G::InterfacePointMap::Match(G::BoundaryPatch::Create(mesh, {0, 1, 2}, pose), points, ids);
        for (I i = 0; i < 3; ++i)
        {
            CHECK(updated.Get(i).donorOriginalId == rebuilt.Get(i).donorOriginalId);
            CHECK((updated.Get(i).donorPoint - expected[i]).norm() < 1e-12);
            CHECK(updated.Get(i).donorOriginalId == map.Get(i).donorOriginalId);
        }
    }
    // Old immutable publication still describes the old pose.
    CHECK((map.Get(0).projectedPoint - expected[0]).norm() < 1e-12);
    {
        auto mutation = mesh->versions.BeginMutation(R::Change::Geometry);
        mutation.Commit();
    }
    CHECK_THROWS(map.Validate());
    CHECK_THROWS(patch.WithPose({}));
}

TEST_CASE("Interface search rejects uncovered points invalid controls duplicate IDs and unsupported distributed use")
{
    auto mesh = NativeFacadeFixture::Make(mpi).Publish(mpi);
    auto patch = G::BoundaryPatch::Create(mesh, {0});
    CHECK_THROWS(G::BoundaryPatch::Create(mesh, {0, 0}));
    CHECK_THROWS(G::BoundaryPatch::Create(mesh, {-1}));
    G::RigidPose badPose;
    badPose.rotation(0, 0) = 2;
    CHECK_THROWS(patch.WithPose(badPose));
    R::Array<double> points(R::RowLayout::Dense(2, 3));
    R::Array<I> ids(R::RowLayout::Dense(2, 1));
    {
        auto write = points.Write();
        auto id = ids.Write();
        for (I i = 0; i < 2; ++i)
        {
            write.View()(i, 0) = .4, write.View()(i, 1) = .01, write.View()(i, 2) = 0;
            id.View()(i, 0) = i;
        }
    }
    CHECK_THROWS(G::InterfacePointMap::Match(patch, points, ids));
    G::InterfaceSearchSettings setting;
    setting.maxGap = .02;
    if (mpi.size == 1)
    {
        auto map = G::InterfacePointMap::Match(patch, points, ids, setting);
        CHECK(map.Get(0).distance == doctest::Approx(.01));
        {
            auto write = ids.Write();
            write.View()(1, 0) = 0;
        }
        CHECK_THROWS(G::InterfacePointMap::Match(patch, points, ids, setting));
    }
    setting.maxGap = -1;
    CHECK_THROWS(G::InterfacePointMap::Match(patch, points, ids, setting));
}

TEST_CASE("Triangle and quadrilateral surface matching retains adjacent volume slots")
{
    for (bool hex : {false, true})
    {
        auto mesh = InterfaceFixture::SingleCell(mpi, hex);
        std::vector<I> selected(size_t(mesh->NumBnd()));
        std::iota(selected.begin(), selected.end(), I(0));
        auto patch = G::BoundaryPatch::Create(mesh, selected);
        R::Array<double> points(R::RowLayout::Dense(patch.Size(), 3));
        R::Array<I> ids(R::RowLayout::Dense(patch.Size(), 1));
        {
            auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
            const auto view = read.HostView();
            auto write = points.Write();
            auto id = ids.Write();
            for (I i = 0; i < patch.Size(); ++i)
            {
                const auto record = patch.Element(i);
                const auto nodes = view.FaceNodes(record.face);
                G::tPoint centroid = G::tPoint::Zero();
                for (I j = 0; j < nodes.size(); ++j)
                    centroid += view.Coord(nodes[j]);
                centroid /= double(nodes.size());
                for (int d = 0; d < 3; ++d)
                    write.View()(i, d) = centroid[d];
                id.View()(i, 0) = 100 + 3 * i;
                CHECK(view.CellFaces(record.cell)[record.cellFaceSlot] == record.face);
            }
        }
        if (mpi.size != 1)
        {
            CHECK_THROWS(G::InterfacePointMap::Match(patch, points, ids));
            continue;
        }
        auto map = G::InterfacePointMap::Match(patch, points, ids);
        for (I i = 0; i < patch.Size(); ++i)
        {
            CHECK(map.Get(i).donorBoundary == selected[i]);
            CHECK(map.Get(i).donorCell == 0);
            CHECK(map.Get(i).distance < 1e-12);
        }
    }
}

TEST_CASE("Boundary stencil preparation retains DSL chain endpoints and canonical ghost mappings")
{
    auto mesh = NativeFacadeFixture::Make(mpi).Publish(mpi);
    G::GhostSpec spec{{{G::EntityKind::Bnd, {G::Adj::Bnd2Node}, G::EntityKind::Node},
                       {G::EntityKind::Bnd, {G::Adj::Bnd2Node, G::Adj::Node2Bnd}, G::EntityKind::Bnd}}};
    auto requirements = G::MeshReadRequirements::CellFlux();
    requirements.inverse = true;
    auto plan = G::BoundaryStencilPlan::Prepare(*mesh, spec, requirements);
    auto adjacency = plan.Read(G::Adj::Bnd2Node);
    const auto expected = mesh->Entities(G::EntityKind::Node);
    const std::vector<std::vector<I>> nodes{{0, 1}, {1, 2}, {2, 0}};
    for (I i = 0; i < 3; ++i)
    {
        auto actual = adjacency.GlobalRow(i);
        REQUIRE(actual.size() == 2);
        for (I j = 0; j < 2; ++j)
            CHECK(actual[j] == expected.LocalToGlobal(nodes[i][j]));
    }
}

TEST_CASE("Relative sliding changes donor cells while preserving the previous publication")
{
    std::string path = __FILE__;
    for (int i = 0; i < 4; ++i)
        path = path.substr(0, path.find_last_of('/'));
    G::AutoAppendName2ID names;
    auto mesh = G::ReadNativeMeshFromCGNS(mpi, 2, path + "/data/mesh/Uniform_3x3_wall.cgns", {},
                                          [&](const std::string &name)
                                          { return names(name); });
    mesh->PrepareNativeTopology();
    if (mpi.size > 1)
        return; // distributed rejection is tested above
    std::vector<std::pair<G::tPoint, I>> centers;
    {
        auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
        const auto view = read.HostView();
        double bottom = std::numeric_limits<double>::infinity();
        for (I i = 0; i < mesh->NumBnd(); ++i)
        {
            const auto nodes = view.BoundaryNodes(i);
            const G::tPoint center = .5 * (view.Coord(nodes[0]) + view.Coord(nodes[1]));
            bottom = std::min(bottom, center[1]);
            centers.emplace_back(center, i);
        }
        centers.erase(std::remove_if(centers.begin(), centers.end(), [&](const auto &entry)
                                     { return std::abs(entry.first[1] - bottom) > 1e-12; }),
                      centers.end());
    }
    REQUIRE(centers.size() >= 2);
    std::sort(centers.begin(), centers.end(), [](const auto &a, const auto &b)
              { return a.first[0] < b.first[0]; });
    std::vector<I> selected;
    for (const auto &entry : centers)
        selected.push_back(entry.second);
    auto patch = G::BoundaryPatch::Create(mesh, selected);
    R::Array<double> points(R::RowLayout::Dense(1, 3));
    R::Array<I> ids(R::RowLayout::Dense(1, 1));
    {
        auto write = points.Write();
        auto id = ids.Write();
        for (int d = 0; d < 3; ++d)
            write.View()(0, d) = centers[0].first[d];
        id.View()(0, 0) = 733;
    }
    auto before = G::InterfacePointMap::Match(patch, points, ids);
    G::RigidPose shift;
    shift.translation = centers[0].first - centers[1].first;
    auto after = G::InterfacePointMap::Match(patch.WithPose(shift), points, ids);
    CHECK(before.Get(0).donorBoundary == centers[0].second);
    CHECK(after.Get(0).donorBoundary == centers[1].second);
    CHECK(before.Get(0).donorCell != after.Get(0).donorCell);
    CHECK(after.Get(0).queryId == 733);
    CHECK((after.Get(0).donorPoint - centers[1].first).norm() < 1e-12);
    CHECK((before.Get(0).donorPoint - centers[0].first).norm() < 1e-12);
    auto empty = G::BoundaryPatch::Create(mesh, {});
    CHECK_THROWS(G::InterfacePointMap::Match(empty, points, ids));
    R::Array<double> noPoints(R::RowLayout::Dense(0, 3));
    R::Array<I> noIds(R::RowLayout::Dense(0, 1));
    CHECK(G::InterfacePointMap::Match(empty, noPoints, noIds).Size() == 0);
}

TEST_CASE("Coincident disconnected surface interiors are rejected rather than silently selecting a donor")
{
    auto mesh = InterfaceFixture::SingleCell(mpi, true, true);
    if (mpi.size != 1)
        return;
    // Hex faces 0 and 6 have distinct node/entity IDs and coincident geometry.
    auto patch = G::BoundaryPatch::Create(mesh, {0, 6});
    R::Array<double> points(R::RowLayout::Dense(1, 3));
    R::Array<I> ids(R::RowLayout::Dense(1, 1));
    {
        auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
        const auto view = read.HostView();
        const auto nodes = view.FaceNodes(patch.Element(0).face);
        G::tPoint center = G::tPoint::Zero();
        for (const auto node : nodes)
            center += view.Coord(node);
        center /= double(nodes.size());
        auto write = points.Write();
        auto id = ids.Write();
        for (int d = 0; d < 3; ++d)
            write.View()(0, d) = center[d];
        id.View()(0, 0) = 19;
    }
    CHECK_THROWS_WITH(G::InterfacePointMap::Match(patch, points, ids), doctest::Contains("ambiguous"));
}

TEST_CASE("Rigid rotation through a full turn preserves matching and changes the physical donor side")
{
    auto mesh = InterfaceFixture::SingleCell(mpi, true);
    if (mpi.size != 1)
        return;
    auto patch = G::BoundaryPatch::Create(mesh, {0, 1, 2, 3, 4, 5});
    const G::tPoint query{.25, 0, .4}, center{.5, .5, .5};
    R::Array<double> points(R::RowLayout::Dense(1, 3));
    R::Array<I> ids(R::RowLayout::Dense(1, 1));
    {
        auto write = points.Write();
        auto id = ids.Write();
        for (int d = 0; d < 3; ++d)
            write.View()(0, d) = query[d];
        id.View()(0, 0) = 81;
    }
    const auto initial = G::InterfacePointMap::Match(patch, points, ids);
    const double pi = std::acos(-1.);
    I rotatedSide = -1;
    for (const double angle : {pi / 2, 2 * pi + pi / 2})
    {
        G::RigidPose pose;
        pose.rotation = Eigen::AngleAxisd(angle, G::tPoint::UnitZ()).toRotationMatrix();
        pose.translation = center - pose.rotation * center;
        const auto map = G::InterfacePointMap::Match(patch.WithPose(pose), points, ids);
        // Analytic inverse of a quarter-turn about the center, independent of
        // the implementation's inverse map/projection calculations.
        CHECK((map.Get(0).donorPoint - G::tPoint{0, .75, .4}).norm() < 1e-12);
        CHECK(map.Get(0).donorBoundary != initial.Get(0).donorBoundary);
        if (rotatedSide >= 0)
            CHECK(map.Get(0).donorBoundary == rotatedSide);
        rotatedSide = map.Get(0).donorBoundary;
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
