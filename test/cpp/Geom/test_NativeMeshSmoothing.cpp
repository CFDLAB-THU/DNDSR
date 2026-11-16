#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Geometry/NativeMeshSmoothing.hpp"
#include "Geom/Refinement/NativeMeshRefinement.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "NativePeriodicFixtures.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    R::Array<I> Rows(const std::vector<std::vector<I>> &rows)
    {
        R::RowBuilder builder(rows.size(), sizeof(I));
        for (size_t row = 0; row < rows.size(); ++row)
        {
            builder.ResizeRow(row, rows[row].size());
            for (size_t slot = 0; slot < rows[row].size(); ++slot)
                builder.Row<I>(row)[slot] = rows[row][slot];
        }
        return R::Array<I>::FromSharedStorage(builder.Commit());
    }
    std::shared_ptr<const G::NativeMeshState> Disk(const DNDS::MPIInfo &mpi, bool independentOwners, int dimension = 2, bool partialBoundary = false, bool preflight = true)
    {
        auto owner = [&](int kind, int logical)
        { return independentOwners ? (3 * logical + kind) % mpi.size : 0; };
        const int nNode = dimension == 2 ? 9 : 7;
        std::vector<I> nodeGlobal(size_t(nNode), I(-1));
        I count = 0;
        for (int rank = 0; rank < mpi.size; ++rank)
            for (int logical = 0; logical < nNode; ++logical)
                if (owner(1, logical) == rank)
                    nodeGlobal[size_t(logical)] = count++;
        std::vector<int> nodeLogical, cellLogical, boundaryLogical;
        for (int logical = 0; logical < nNode; ++logical)
            if (owner(1, logical) == mpi.rank)
                nodeLogical.push_back(logical);
        for (int logical = 0; logical < 8; ++logical)
        {
            if (owner(2, logical) == mpi.rank)
                cellLogical.push_back(logical);
            if (owner(3, logical) == mpi.rank)
                boundaryLogical.push_back(logical);
        }
        const auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, I(nodeLogical.size()));
        const auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(cellLogical.size()));
        const auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, I(boundaryLogical.size()));
        R::Array<DNDS::real> coords(R::RowLayout::Dense(nodeLogical.size(), 3));
        R::Array<I> ids(R::RowLayout::Dense(nodeLogical.size(), 1));
        R::Array<G::ElemInfo> info(R::RowLayout::Dense(cellLogical.size(), 1)), bndInfo(R::RowLayout::Dense(boundaryLogical.size(), 1));
        std::vector<std::vector<I>> cones, bndCones;
        auto surface = [&](int logical)
        {
            if (dimension == 2)
                return std::vector<I>{nodeGlobal[size_t(1 + logical)], nodeGlobal[size_t(1 + (logical + 1) % 8)]};
            int a = 1 + (logical & 1), b = 3 + ((logical >> 1) & 1), c = 5 + ((logical >> 2) & 1);
            if (__builtin_popcount(unsigned(logical)) % 2)
                std::swap(b, c);
            return std::vector<I>{nodeGlobal[size_t(a)], nodeGlobal[size_t(b)], nodeGlobal[size_t(c)]};
        };
        {
            auto coordinates = coords.Write();
            auto identities = ids.Write();
            for (size_t row = 0; row < nodeLogical.size(); ++row)
            {
                const int logical = nodeLogical[row];
                const double angle = 2 * DNDS::pi * (logical - 1) / 8;
                G::tPoint point = G::tPoint::Zero();
                if (dimension == 2 && logical)
                    point = G::tPoint{std::cos(angle), std::sin(angle), 0};
                if (dimension == 3 && logical)
                    point[(logical - 1) / 2] = logical % 2 ? 1 : -1;
                for (size_t component = 0; component < 3; ++component)
                    coordinates.View()(row, component) = point[Eigen::Index(component)];
                identities.View()(row, 0) = 1000 + logical;
            }
            auto metadata = info.Write();
            for (size_t row = 0; row < cellLogical.size(); ++row)
            {
                const int logical = cellLogical[row];
                auto face = surface(logical);
                face.insert(face.begin(), nodeGlobal[0]);
                cones.push_back(std::move(face));
                metadata.View()(row, 0) = {dimension == 2 ? G::Elem::Tri3 : G::Elem::Tet4, 0};
            }
            auto boundaryMetadata = bndInfo.Write();
            for (size_t row = 0; row < boundaryLogical.size(); ++row)
            {
                const int logical = boundaryLogical[row];
                bndCones.push_back(surface(logical));
                boundaryMetadata.View()(row, 0) = {dimension == 2 ? G::Elem::Line2 : G::Elem::Tri3, partialBoundary && logical >= 4 ? G::BC_ID_DEFAULT_FAR : G::BC_ID_DEFAULT_WALL};
            }
        }
        G::NativeMeshInput input{mpi, dimension, cells, nodes, boundaries, Rows(cones), Rows(bndCones), coords, info, bndInfo, {}};
        input.ids.emplace(G::EntityKind::Node, std::move(ids));
        G::NativeMeshBuildOptions options;
        if (!preflight)
            options.communication = std::make_shared<R::CommunicationContext>(mpi.comm, false);
        auto primary = G::ConstructNativeMesh(std::move(input), options);
        auto elevated = G::NativeMeshRefinement::Elevate(*primary, mpi);
        return G::PrepareNativeMeshTopology(*elevated.state, mpi);
    }
}

TEST_CASE("Native smoothing Hermite boundary oracle and detached local RBF")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    bool independentOwners = true;
    bool preflight = true;
    SUBCASE("independent family ownership") {}
    SUBCASE("empty peers") { independentOwners = false; }
    SUBCASE("independent ownership without optional preflight") { preflight = false; }
    auto source = Disk(mpi, independentOwners, 2, false, preflight);
    auto old = source->Snapshot(R::Device::Host()).AcquireRead();
    G::NativeSmoothingOptions options;
    options.MaxIncludedAngle = 50;
    options.RBFRadius = 4;
    auto workspace = G::NativeMeshSmoothing::PrepareBoundary(source, mpi, options, [](auto zone)
                                                             { return zone == G::BC_ID_DEFAULT_WALL; });
    CHECK(workspace->TotalMoved() == 8);
    auto displacement = workspace->Displacement().ReadCombined();
    auto roles = workspace->Constraints().ReadCombined();
    I counts[4]{};
    const double radius = std::cos(DNDS::pi / 8), increase = .5 * std::pow(std::sin(DNDS::pi / 8), 2);
    for (I node = 0; node < source->NumOwned(G::EntityKind::Node); ++node)
    {
        const auto role = G::NativeSmoothingConstraint(roles.View()(size_t(node), 0));
        ++counts[int(role)];
        const auto point = old.HostView().Coord(node);
        if (role == G::NativeSmoothingConstraint::Prescribed)
        {
            CHECK(point.norm() == doctest::Approx(radius));
            for (size_t component = 0; component < 3; ++component)
                CHECK(displacement.View()(size_t(node), component) == doctest::Approx(increase * point[Eigen::Index(component)] / radius).epsilon(1e-10));
        }
        if (source->OriginalEntityId(G::EntityKind::Node, node) >= 1001 && source->OriginalEntityId(G::EntityKind::Node, node) <= 1008)
            CHECK(role == G::NativeSmoothingConstraint::OriginalBoundary);
    }
    MPI_Allreduce(MPI_IN_PLACE, counts, 4, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
    CHECK(counts[0] == 9);
    CHECK(counts[1] == 8);
    CHECK(counts[2] == 8);
    CHECK(counts[3] == 0);
    auto alias = workspace->Displacement().owned;
    CHECK_THROWS(alias.Write());
    auto smoothed = G::NativeMeshSmoothing::LocalRBF(*workspace, mpi, options);
    CHECK(smoothed->MeshNamespace() == source->MeshNamespace());
    CHECK(smoothed->Entities(G::EntityKind::Node).ChecksCollectivePreflight() == preflight);
    auto current = smoothed->Snapshot(R::Device::Host()).AcquireRead();
    for (I node = 0; node < source->NumOwned(G::EntityKind::Node); ++node)
    {
        const auto before = old.HostView().Coord(node), after = current.HostView().Coord(node);
        CHECK(after.allFinite());
        const auto role = G::NativeSmoothingConstraint(roles.View()(size_t(node), 0));
        if (role == G::NativeSmoothingConstraint::Prescribed)
            CHECK(after.norm() == doctest::Approx(radius + increase));
        if (role == G::NativeSmoothingConstraint::OriginalBoundary)
            CHECK((after - before).norm() == 0);
        if (source->OriginalEntityId(G::EntityKind::Node, node) == 1000)
            CHECK(after.norm() < 1e-10);
        CHECK((source->BorrowHostView().Coord(node) - before).norm() == 0);
    }
    source.reset();
    workspace.reset();
    CHECK(current.HostView().NumCell() == old.HostView().NumCell());
}

TEST_CASE("Native smoothing no-op and collective callback/options failure retry")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto source = Disk(mpi, true);
    G::NativeSmoothingOptions options;
    auto none = G::NativeMeshSmoothing::PrepareBoundary(source, mpi, options, [](auto)
                                                        { return false; });
    CHECK(none->TotalMoved() == 0);
    CHECK(G::NativeMeshSmoothing::LocalRBF(*none, mpi, options) == source);
    CHECK_THROWS(G::NativeMeshSmoothing::PrepareBoundary(source, mpi, options, [&](auto)
                                                         { if (mpi.rank == 0) throw std::runtime_error("selector"); return true; }));
    if (mpi.rank == 0)
        options.nSearch = 0;
    CHECK_THROWS(G::NativeMeshSmoothing::LocalRBF(*none, mpi, options));
    options.nSearch = 30;
    auto retry = G::NativeMeshSmoothing::PrepareBoundary(source, mpi, options, [](auto zone)
                                                         { return zone == G::BC_ID_DEFAULT_WALL; });
    CHECK(retry->TotalMoved() == 8);
}

TEST_CASE("Native global RBF and elasticity preserve exact boundary constraints")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    bool empty = false, partialBoundary = false, preflight = true;
    SUBCASE("independent owners") {}
    SUBCASE("empty peers") { empty = true; }
    SUBCASE("unselected boundary stays fixed in elasticity") { partialBoundary = true; }
    SUBCASE("independent ownership without optional preflight") { preflight = false; }
    auto source = Disk(mpi, !empty, 2, partialBoundary, preflight);
    G::NativeSmoothingOptions options;
    options.MaxIncludedAngle = 50;
    options.RBFRadius = 1.2;
    options.nSearch = 16;
    options.nIter = 100;
    options.kernel = G::RBF::CPC2;
    auto problem = G::NativeMeshSmoothing::PrepareBoundary(source, mpi, options, [](auto zone)
                                                           { return zone == G::BC_ID_DEFAULT_WALL; });
    auto roles = problem->Constraints().ReadCombined();
    auto displacement = problem->Displacement().ReadCombined();
    if (partialBoundary)
    {
        I otherBoundary = 0;
        for (I node = 0; node < source->NumOwned(G::EntityKind::Node); ++node)
            otherBoundary += G::NativeSmoothingConstraint(roles.View()(size_t(node), 0)) == G::NativeSmoothingConstraint::OtherBoundary;
        MPI_Allreduce(MPI_IN_PLACE, &otherBoundary, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        CHECK(otherBoundary == 7);
    }
    for (int method = 0; method < 2; ++method)
    {
        auto result = method == 0 ? G::NativeMeshSmoothing::GlobalRBF(*problem, mpi, options) : G::NativeMeshSmoothing::Elasticity(*problem, mpi, options);
        const auto &before = source->BorrowHostView(), &after = result->BorrowHostView();
        CHECK(result->MeshNamespace() == source->MeshNamespace());
        CHECK(result->Entities(G::EntityKind::Node).ChecksCollectivePreflight() == preflight);
        for (I node = 0; node < source->NumOwned(G::EntityKind::Node); ++node)
        {
            const auto original = before.Coord(node), smoothed = after.Coord(node);
            CHECK(smoothed.allFinite());
            const auto role = G::NativeSmoothingConstraint(roles.View()(size_t(node), 0));
            if (role == G::NativeSmoothingConstraint::OriginalBoundary)
                CHECK((smoothed - original).norm() < 1e-9);
            if (method == 1 && role == G::NativeSmoothingConstraint::OtherBoundary)
                CHECK((smoothed - original).norm() < 1e-9);
            if (role == G::NativeSmoothingConstraint::Prescribed)
                for (size_t component = 0; component < 3; ++component)
                    CHECK(smoothed[Eigen::Index(component)] - original[Eigen::Index(component)] == doctest::Approx(displacement.View()(size_t(node), component)).epsilon(1e-8));
            if (!partialBoundary && source->OriginalEntityId(G::EntityKind::Node, node) == 1000)
                CHECK(smoothed.norm() < 1e-8);
        }
    }
}

TEST_CASE("Native 3D smoothing preserves independent octahedral Hermite geometry after redistribution")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto original = Disk(mpi, true, 3);
    G::ReorderInput placement;
    for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd, G::EntityKind::Face})
    {
        std::vector<DNDS::MPI_int> owners;
        for (I row = 0; row < original->NumOwned(kind); ++row)
            owners.push_back(DNDS::MPI_int((original->OriginalEntityId(kind, row) + I(kind) + 1) % mpi.size));
        placement.explicitMaps.push_back({kind, std::move(owners)});
    }
    auto source = G::NativeMeshTransition::Build(*original, mpi, placement).state;
    original.reset();
    G::NativeSmoothingOptions options;
    options.MaxIncludedAngle = 90;
    options.RBFRadius = 1.2;
    options.nSearch = 18;
    options.nIter = 100;
    options.kernel = G::RBF::CPC2;
    auto workspace = G::NativeMeshSmoothing::PrepareBoundary(source, mpi, options, [](auto zone)
                                                             { return zone == G::BC_ID_DEFAULT_WALL; });
    CHECK(workspace->TotalMoved() == 12);
    auto roles = workspace->Constraints().ReadCombined();
    auto displacement = workspace->Displacement().ReadCombined();
    const double increase = std::sqrt(2.) / 8;
    for (I node = 0; node < source->NumOwned(G::EntityKind::Node); ++node)
        if (G::NativeSmoothingConstraint(roles.View()(size_t(node), 0)) == G::NativeSmoothingConstraint::Prescribed)
        {
            const auto point = source->BorrowHostView().Coord(node);
            for (size_t component = 0; component < 3; ++component)
                CHECK(displacement.View()(size_t(node), component) == doctest::Approx(2 * increase * point[Eigen::Index(component)]));
        }
    for (int method = 0; method < 3; ++method)
    {
        auto result = method == 0 ? G::NativeMeshSmoothing::LocalRBF(*workspace, mpi, options) : method == 1 ? G::NativeMeshSmoothing::GlobalRBF(*workspace, mpi, options)
                                                                                                             : G::NativeMeshSmoothing::Elasticity(*workspace, mpi, options);
        for (I node = 0; node < result->NumOwned(G::EntityKind::Node); ++node)
        {
            const auto point = result->BorrowHostView().Coord(node);
            CHECK(point.allFinite());
            const auto role = G::NativeSmoothingConstraint(roles.View()(size_t(node), 0));
            if (role == G::NativeSmoothingConstraint::Prescribed)
                CHECK(point.norm() == doctest::Approx(std::sqrt(2.) * (.5 + increase)).epsilon(1e-8));
            if (role == G::NativeSmoothingConstraint::OriginalBoundary)
                CHECK(point.norm() == doctest::Approx(1));
            if (source->OriginalEntityId(G::EntityKind::Node, node) == 1000)
                CHECK(point.norm() < 1e-8);
        }
    }
}

TEST_CASE("Native periodic flat smoothing retains exact physical images in all three solvers")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto primary = G::ConstructNativeMesh(NativePeriodicFixtures::TwoCells(mpi));
    auto elevated = G::NativeMeshRefinement::Elevate(*primary, mpi);
    auto source = G::PrepareNativeMeshTopology(*elevated.state, mpi);
    G::NativeSmoothingOptions options;
    auto workspace = G::NativeMeshSmoothing::PrepareBoundary(source, mpi, options, [](auto zone)
                                                             { return zone == 12 || zone == 13; });
    CHECK(workspace->TotalMoved() == 4);
    for (int method = 0; method < 3; ++method)
    {
        auto result = method == 0 ? G::NativeMeshSmoothing::LocalRBF(*workspace, mpi, options) : method == 1 ? G::NativeMeshSmoothing::GlobalRBF(*workspace, mpi, options)
                                                                                                             : G::NativeMeshSmoothing::Elasticity(*workspace, mpi, options);
        const auto &before = source->BorrowHostView(), &after = result->BorrowHostView();
        CHECK(after.IsPeriodic());
        for (I node = 0; node < source->NumOwned(G::EntityKind::Node); ++node)
            CHECK((after.Coord(node) - before.Coord(node)).norm() < 1e-12);
        for (I cell = 0; cell < source->NumOwned(G::EntityKind::Cell); ++cell)
            for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(before.CellNodes(cell).size()); ++slot)
                CHECK((after.GetCoordNodeOnCell(cell, slot) - before.GetCoordNodeOnCell(cell, slot)).norm() < 1e-12);
    }
}

TEST_CASE("Native facade smoothing stages work, rejects readers and stale caches, then retries atomically")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    bool preflight = true;
    SUBCASE("recoverable checked failures") {}
    SUBCASE("unchecked successful publication") { preflight = false; }
    for (int method = 0; method < 3; ++method)
    {
        auto source = Disk(mpi, true, 2, false, preflight);
        auto mesh = G::UnstructuredMesh::FromNative(mpi, source);
        mesh->elevationInfo.MaxIncludedAngle = 50;
        mesh->elevationInfo.RBFRadius = 1.2;
        mesh->elevationInfo.nIter = 100;
        mesh->elevationInfo.kernel = G::RBF::CPC2;
        const auto beforePreparation = mesh->CurrentNativeState();
        mesh->ElevatedNodesGetBoundarySmooth([](auto zone)
                                             { return zone == G::BC_ID_DEFAULT_WALL; });
        CHECK(mesh->CurrentNativeState() == beforePreparation);
        const auto staged = mesh->CurrentNativeState();
        CHECK(staged->MeshNamespace() == source->MeshNamespace());
        auto solve = [&]
        {
            if (method == 0)
                mesh->ElevatedNodesSolveInternalSmooth();
            if (method == 1)
                mesh->ElevatedNodesSolveInternalSmoothV1();
            if (method == 2)
                mesh->ElevatedNodesSolveInternalSmoothV2();
        };
        std::optional<G::MeshReadAccess> retained;
        if (preflight && mpi.rank == 0)
            retained.emplace(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
        if (preflight)
            CHECK_THROWS(solve());
        CHECK(mesh->CurrentNativeState() == staged);
        retained.reset();
        solve();
        CHECK(mesh->CurrentNativeState() != staged);
        CHECK(mesh->CurrentNativeState()->MeshNamespace() == source->MeshNamespace());
        CHECK(mesh->Entities(G::EntityKind::Node).ChecksCollectivePreflight() == preflight);
        if (preflight)
            CHECK_THROWS(solve());
        mesh->ElevatedNodesGetBoundarySmooth([](auto)
                                             { return false; });
        const auto none = mesh->CurrentNativeState();
        solve();
        CHECK(mesh->CurrentNativeState() == none);
        mesh->ElevatedNodesGetBoundarySmooth([](auto zone)
                                             { return zone == G::BC_ID_DEFAULT_WALL; });
        mesh->TransformCoords([](const auto &point)
                              { return G::tPoint(point + G::tPoint{.1, 0, 0}); });
        if (preflight)
            CHECK_THROWS(solve());
        mesh->ElevatedNodesGetBoundarySmooth([](auto zone)
                                             { return zone == G::BC_ID_DEFAULT_WALL; });
        CHECK_NOTHROW(solve());
    }
}

TEST_CASE("Retired compact support smoother rejects explicitly without mutating native source")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto state = Disk(mpi, true);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, state);
    const auto stamp = mesh->versions.Stamp();
    CHECK_THROWS(mesh->ElevatedNodesSolveInternalSmoothV1Old());
    CHECK(stamp.Valid());
    CHECK(mesh->CurrentNativeState()->MeshNamespace() == state->MeshNamespace());
}

int main(int argc, char **argv)
{
    // Native row kernels may create OMP/CPU workers even with controller MPI.
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    int result = doctest::Context(argc, argv).run();
    DNDS::MPI::Finalize();
    return result;
}
