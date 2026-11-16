#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Unstructured/NativeMeshBuilder.hpp"
#include "Geom/Access/RelationAccess.hpp"
#include "Geom/Connectivity/NativeMeshClosure.hpp"
#include "Geom/IO/NativeMeshArchive.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Refinement/NativeMeshRefinement.hpp"
#include "NativePeriodicFixtures.hpp"
#include "Geom/Connectivity/TopologySubEntityQuery.hpp"
#include <random>
#include <numeric>
#include <limits>
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <filesystem>
#include <unistd.h>
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include <map>
#include <set>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    DNDS::MPIInfo MPI()
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        return mpi;
    }
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
    std::shared_ptr<const G::NativeMeshState> Fixture(const DNDS::MPIInfo &mpi, int dimension, bool duplicateBoundary = false, bool nonmanifold = false)
    {
        const I nodeCount = mpi.rank == 0 ? (dimension == 3 || nonmanifold ? 5 : 4) : 0;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, nodeCount);
        std::vector<std::vector<I>> cellRows, boundaryRows;
        if (nonmanifold)
        {
            if (mpi.rank == 0)
                cellRows = {{0, 1, 2}, {0, 1, 3}, {0, 1, 4}};
        }
        else
        {
            if (mpi.rank == 0)
                cellRows.push_back(dimension == 3 ? std::vector<I>{0, 1, 2, 3} : std::vector<I>{0, 1, 2});
            if (mpi.rank == (mpi.size > 1 ? 1 : 0))
                cellRows.push_back(dimension == 3 ? std::vector<I>{0, 2, 1, 4} : std::vector<I>{1, 3, 2});
            if (dimension == 2 && mpi.rank == mpi.size - 1)
            {
                boundaryRows = {{0, 1}, {1, 3}, {3, 2}, {2, 0}};
                if (duplicateBoundary)
                    boundaryRows.push_back({1, 0});
            }
        }
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(cellRows.size()));
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, I(boundaryRows.size()));
        G::NativeMeshInput input{mpi, dimension, cells, nodes, boundaries, Rows(cellRows), Rows(boundaryRows), R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodeCount), 3)), R::Array<G::ElemInfo>(R::RowLayout::Dense(cellRows.size(), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(boundaryRows.size(), 1)), {}};
        {
            auto coords = input.coordinates.Write();
            const DNDS::real points[5][3]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, -1, 0}};
            for (size_t row = 0; row < size_t(nodeCount); ++row)
                for (size_t component = 0; component < 3; ++component)
                    coords.View()(row, component) = points[row][component];
            if (dimension == 3 && nodeCount)
            {
                coords.View()(3, 0) = 0;
                coords.View()(3, 1) = 0;
                coords.View()(3, 2) = 1;
                coords.View()(4, 0) = 0;
                coords.View()(4, 1) = 0;
                coords.View()(4, 2) = -1;
            }
            auto info = input.cellInfo.Write();
            for (size_t row = 0; row < cellRows.size(); ++row)
                info.View()(row, 0) = G::ElemInfo{dimension == 3 ? G::Elem::Tet4 : G::Elem::Tri3, G::INTERNAL_ZONE};
            auto boundary = input.boundaryInfo.Write();
            for (size_t row = 0; row < boundaryRows.size(); ++row)
                boundary.View()(row, 0) = G::ElemInfo{G::Elem::Line2, G::t_index(2 + row)};
        }
        G::NativeMeshBuildOptions options;
        options.completeNodeStar = true;
        return G::ConstructNativeMesh(std::move(input), options);
    }
    std::shared_ptr<const G::NativeMeshState> Truncated(const DNDS::MPIInfo &mpi)
    {
        auto original = Fixture(mpi, 2);
        auto owned = G::NativeMeshAssembler::CaptureOwned(*original, mpi);
        const auto cells = owned.spaces.at(G::EntityKind::Cell);
        auto nodes = owned.spaces.at(G::EntityKind::Node);
        auto globalNodes = owned.relations[size_t(G::MeshRelation::CellNodes)]->Read();
        std::set<I> remote;
        for (I cell = 0; cell < cells.NumOwned(); ++cell)
            for (auto global : globalNodes.View()[size_t(cell)])
                if (nodes.Owner(global) != mpi.rank)
                    remote.insert(global);
        nodes = nodes.WithGhosts({remote.begin(), remote.end()});
        G::NativeMeshData data;
        data.dimension = 2;
        data.identity = owned.identity;
        data.coords = R::ArrayPair<DNDS::real>{*owned.coords, R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumGhost()), 3))};
        {
            auto write = data.coords->ghost.Write();
            const DNDS::real points[4][3]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
            for (I row = 0; row < nodes.NumGhost(); ++row)
                for (size_t component = 0; component < 3; ++component)
                    write.View()(size_t(row), component) = points[nodes.LocalToGlobal(nodes.NumOwned() + row)][component];
        }
        data.cellInfo = R::ArrayPair<G::ElemInfo>{*owned.cellInfo, R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
        data.bndInfo = R::ArrayPair<G::ElemInfo>{R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
        std::vector<std::vector<I>> local(size_t(cells.NumOwned())), empty(size_t(cells.NumOwned()));
        for (I cell = 0; cell < cells.NumOwned(); ++cell)
            for (auto global : globalNodes.View()[size_t(cell)])
                local[size_t(cell)].push_back(*nodes.GlobalToLocal(global));
        data.relations[size_t(G::MeshRelation::CellNodes)] = R::ArrayPair<I>{Rows(local), Rows({})};
        data.relations[size_t(G::MeshRelation::CellCells)] = R::ArrayPair<I>{Rows(empty), Rows({})};
        data.relations[size_t(G::MeshRelation::BoundaryNodes)] = R::ArrayPair<I>{Rows({}), Rows({})};
        data.relations[size_t(G::MeshRelation::BoundaryCells)] = R::ArrayPair<I>{R::Array<I>(R::RowLayout::Dense(0, 2)), R::Array<I>(R::RowLayout::Dense(0, 2))};
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
        std::map<G::EntityKind, G::EntitySpace> spaces{{G::EntityKind::Cell, cells}, {G::EntityKind::Node, nodes}, {G::EntityKind::Bnd, boundaries}};
        std::map<G::EntityKind, R::Array<I>> ids;
        for (const auto &[kind, space] : spaces)
        {
            R::Array<I> values(R::RowLayout::Dense(size_t(space.NumLocal()), 1));
            {
                auto write = values.Write();
                for (I row = 0; row < space.NumLocal(); ++row)
                    write.View()(size_t(row), 0) = space.LocalToGlobal(row);
            }
            ids.emplace(kind, std::move(values));
        }
        return G::NativeMeshState::FromNative(mpi, std::move(data), std::move(spaces), std::move(ids), G::MeshReadRequirements::Primary());
    }

    using Oracle = std::map<std::pair<int, I>, std::vector<I>>;
    Oracle PersistentTopology(const G::NativeMeshState &state, const DNDS::MPIInfo &mpi)
    {
        auto read = state.Snapshot(R::Device::Host()).AcquireRead();
        const auto view = read.HostView();
        std::vector<I> packed;
        for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Face, G::EntityKind::Edge})
            if (state.HasEntities(kind))
                for (I row = 0; row < state.NumOwned(kind); ++row)
                {
                    std::vector<I> values;
                    auto append = [&](G::MeshRowSpan<I> entries, G::EntityKind target)
                    {
                        values.push_back(I(entries.size()));
                        for (auto local : entries)
                        {
                            DNDS_check_throw_info(local == DNDS::UnInitIndex || local >= 0, "test oracle requires complete owned topology");
                            values.push_back(local == DNDS::UnInitIndex ? local : state.OriginalEntityId(target, local));
                        }
                    };
                    if (kind == G::EntityKind::Node)
                        for (int component = 0; component < 3; ++component)
                            values.push_back(I(view.Coord(row)[component]));
                    else if (kind == G::EntityKind::Cell)
                    {
                        values = {view.GetCellInfo(row).type, view.GetCellInfo(row).zone};
                        append(view.CellNodes(row), G::EntityKind::Node);
                        append(view.CellFaces(row), G::EntityKind::Face);
                        append(view.CellFaceCells(row), G::EntityKind::Cell);
                    }
                    else if (kind == G::EntityKind::Face)
                    {
                        values = {view.GetFaceInfo(row).type, view.GetFaceInfo(row).zone};
                        append(view.FaceNodes(row), G::EntityKind::Node);
                        append(view.FaceCells(row), G::EntityKind::Cell);
                    }
                    else
                    {
                        values = {view.GetEdgeInfo(row).type, view.GetEdgeInfo(row).zone};
                        append(view.EdgeNodes(row), G::EntityKind::Node);
                        append(view.EdgeCells(row), G::EntityKind::Cell);
                    }
                    packed.insert(packed.end(), {int(kind), state.OriginalEntityId(kind, row), I(values.size())});
                    packed.insert(packed.end(), values.begin(), values.end());
                }
        // This fixed tiny fixture oracle gathers records only in the test. The
        // production algorithms never gather global topology or allocate by IDs.
        const int local = int(packed.size());
        std::vector<int> counts(size_t(mpi.size)), offsets(size_t(mpi.size));
        MPI_Allgather(&local, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
        int total = 0;
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            offsets[size_t(peer)] = total;
            total += counts[size_t(peer)];
        }
        std::vector<I> global(size_t(total), 0);
        MPI_Allgatherv(packed.data(), local, DNDS::DNDS_MPI_INDEX, global.data(), counts.data(), offsets.data(), DNDS::DNDS_MPI_INDEX, mpi.comm);
        Oracle result;
        for (size_t position = 0; position < global.size();)
        {
            const auto kind = int(global[position++]);
            const auto id = global[position++];
            const auto length = size_t(global[position++]);
            auto inserted = result.emplace(std::make_pair(kind, id), std::vector<I>(global.begin() + position, global.begin() + position + length));
            CHECK(inserted.second);
            position += length;
        }
        return result;
    }

    std::set<I> NodeKey(const G::EntitySpace &nodes, G::MeshRowSpan<I> row)
    {
        std::set<I> result;
        for (auto local : row)
            result.insert(nodes.LocalToGlobal(local));
        return result;
    }
}

TEST_CASE("Native face completion matches independent topology with remote boundary ownership and empty ranks")
{
    const auto mpi = MPI();
    auto primary = Fixture(mpi, 2);
    const auto originalNamespace = primary->MeshNamespace();
    auto complete = G::PrepareNativeMeshTopology(*primary, mpi);
    CHECK(complete->NumGlobal(G::EntityKind::Face) == 5);
    CHECK(complete->MeshNamespace() == originalNamespace);
    CHECK(complete->NextEntityId(G::EntityKind::Face) == 5);
    auto read = complete->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = read.HostView();
    const auto nodes = complete->Entities(G::EntityKind::Node), cells = complete->Entities(G::EntityKind::Cell);
    std::map<std::set<I>, std::vector<I>> expected{{{0, 1}, {0}}, {{0, 2}, {0}}, {{1, 2}, {0, 1}}, {{1, 3}, {1}}, {{2, 3}, {1}}};
    std::map<std::set<I>, I> boundaries{{{0, 1}, 0}, {{1, 3}, 1}, {{2, 3}, 2}, {{0, 2}, 3}};
    const std::vector<std::vector<std::vector<I>>> orderedFaces{{{0, 1}, {1, 2}, {2, 0}}, {{1, 3}, {3, 2}, {2, 1}}};
    auto support = complete->Relation(G::Adj::Cell2Face.Inverted());
    CHECK(support.Kind() != G::Adj::Face2Cell);
    for (I face = 0; face < complete->NumOwned(G::EntityKind::Face); ++face)
    {
        const auto key = NodeKey(nodes, view.FaceNodes(face));
        REQUIRE(expected.count(key));
        CHECK(support.GlobalRow(face) == expected.at(key));
        std::vector<I> actual;
        for (auto local : view.FaceCells(face))
            if (local != DNDS::UnInitIndex)
                actual.push_back(cells.LocalToGlobal(local));
        std::sort(actual.begin(), actual.end());
        CHECK(actual == expected.at(key));
        const auto firstParent = cells.LocalToGlobal(view.FaceCells(face)[0]);
        std::vector<I> ordered;
        for (auto local : view.FaceNodes(face))
            ordered.push_back(nodes.LocalToGlobal(local));
        bool oriented = false;
        for (const auto &candidate : orderedFaces[size_t(firstParent)])
            if (std::set<I>(candidate.begin(), candidate.end()) == key)
                oriented = ordered == candidate;
        CHECK(oriented);
        if (boundaries.count(key))
        {
            CHECK(view.GetFaceZone(face) == 2 + boundaries.at(key));
            CHECK(complete->Entities(G::EntityKind::Bnd).LocalToGlobal(view.FaceBoundaries(face)[0]) == boundaries.at(key));
            CHECK(view.FaceCells(face)[1] == DNDS::UnInitIndex);
        }
        else
        {
            CHECK(view.GetFaceZone(face) == G::BC_ID_INTERNAL);
            CHECK(view.FaceBoundaries(face)[0] == DNDS::UnInitIndex);
        }
    }
    for (I cell = 0; cell < cells.NumOwned(); ++cell)
    {
        const I global = cells.LocalToGlobal(cell);
        for (size_t slot = 0; slot < view.CellFaces(cell).size(); ++slot)
        {
            const auto face = view.CellFaces(cell)[slot];
            const auto key = NodeKey(nodes, view.FaceNodes(face));
            CHECK(key == std::set<I>(orderedFaces[size_t(global)][slot].begin(), orderedFaces[size_t(global)][slot].end()));
            const auto neighbor = view.CellFaceCells(cell)[slot];
            if (key == std::set<I>{1, 2})
                CHECK(cells.LocalToGlobal(neighbor) == 1 - global);
            else
                CHECK(neighbor == DNDS::UnInitIndex);
        }
    }
    auto refreshed = G::PrepareNativeMeshTopology(*complete, mpi);
    CHECK(PersistentTopology(*refreshed, mpi) == PersistentTopology(*complete, mpi));
    CHECK(refreshed->IdHighWater() == complete->IdHighWater());
    CHECK_THROWS(G::PrepareNativeMeshTopology(*complete, mpi, true, false, 1,
                                              {{G::EntityKind::Face, complete->NextEntityId(G::EntityKind::Face) - 1}}));
    CHECK_NOTHROW(primary->Validate());
}

TEST_CASE("Native tetrahedron face and edge completion preserves exact incident cell sets")
{
    const auto mpi = MPI();
    auto primary = Fixture(mpi, 3);
    auto complete = G::PrepareNativeMeshTopology(*primary, mpi, true, true);
    CHECK(complete->NumGlobal(G::EntityKind::Face) == 7);
    CHECK(complete->NumGlobal(G::EntityKind::Edge) == 9);
    CHECK(complete->MeshNamespace() == primary->MeshNamespace());
    auto read = complete->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = read.HostView();
    const auto nodes = complete->Entities(G::EntityKind::Node), cells = complete->Entities(G::EntityKind::Cell);
    std::map<std::set<I>, std::vector<I>> expected{{{0, 1}, {0, 1}}, {{0, 2}, {0, 1}}, {{1, 2}, {0, 1}}, {{0, 3}, {0}}, {{1, 3}, {0}}, {{2, 3}, {0}}, {{0, 4}, {1}}, {{1, 4}, {1}}, {{2, 4}, {1}}};
    auto support = complete->Relation(G::Adj::Cell2Edge.Inverted());
    for (I edge = 0; edge < complete->NumOwned(G::EntityKind::Edge); ++edge)
    {
        const auto key = NodeKey(nodes, view.EdgeNodes(edge));
        REQUIRE(expected.count(key));
        CHECK(support.GlobalRow(edge) == expected.at(key));
        std::vector<I> actual;
        for (auto cell : view.EdgeCells(edge))
            actual.push_back(cells.LocalToGlobal(cell));
        CHECK(actual == expected.at(key));
    }
    for (I face = 0; face < complete->NumOwned(G::EntityKind::Face); ++face)
        CHECK(view.GetFaceZone(face) == (view.FaceCells(face)[1] == DNDS::UnInitIndex ? G::BC_ID_NULL : G::BC_ID_INTERNAL));
    auto edgesOnly = G::PrepareNativeMeshTopology(*primary, mpi, false, true);
    CHECK_FALSE(edgesOnly->HasEntities(G::EntityKind::Face));
    CHECK(edgesOnly->NumGlobal(G::EntityKind::Edge) == 9);
}

TEST_CASE("Native topology rejects nonmanifold faces and duplicate boundary assignments collectively")
{
    const auto mpi = MPI();
    SUBCASE("three cells share a face")
    {
        auto source = Fixture(mpi, 2, false, true);
        CHECK_THROWS(G::PrepareNativeMeshTopology(*source, mpi));
        CHECK_NOTHROW(source->Validate());
    }
    SUBCASE("two independently supplied boundaries name one face")
    {
        auto source = Fixture(mpi, 2, true);
        CHECK_THROWS(G::PrepareNativeMeshTopology(*source, mpi));
        CHECK_NOTHROW(source->Validate());
    }
}

TEST_CASE("Native topology repairs missing parent ghosts from authoritative owned node incidences")
{
    const auto mpi = MPI();
    auto source = Truncated(mpi);
    CHECK(source->NumGhost(G::EntityKind::Cell) == 0);
    auto closure = G::BuildNativeCellClosure(*source, mpi);
    auto parents = closure.cellNodes.Read();
    if (mpi.size > 1 && parents.NumOwned())
        CHECK(parents.Source().NumGhost() == 1);
    auto sourceRead = source->Snapshot(R::Device::Host()).AcquireRead();
    const auto sourceView = sourceRead.HostView();
    const auto nodeSpace = source->Entities(G::EntityKind::Node);
    auto ownedMetadata = closure.cellInfo.owned.Read();
    for (I row = 0; row < source->NumOwned(G::EntityKind::Cell); ++row)
    {
        const auto localNodes = sourceView.CellNodes(row);
        const auto globalNodes = parents.GlobalRow(row);
        REQUIRE(globalNodes.size() == localNodes.size());
        for (size_t slot = 0; slot < localNodes.size(); ++slot)
            CHECK(globalNodes[slot] == nodeSpace.LocalToGlobal(localNodes[slot]));
        CHECK(ownedMetadata.View()(size_t(row), 0).type == sourceView.GetCellInfo(row).type);
        CHECK(ownedMetadata.View()(size_t(row), 0).zone == sourceView.GetCellInfo(row).zone);
    }
    auto metadata = closure.cellInfo.ReadCombined();
    CHECK(metadata.View().RowCount() == size_t(parents.RowCount()));
    for (size_t row = 0; row < metadata.View().RowCount(); ++row)
        CHECK(metadata.View()(row, 0).type == G::Elem::Tri3);
    auto complete = G::PrepareNativeMeshTopology(*source, mpi);
    CHECK(complete->NumGlobal(G::EntityKind::Face) == 5);
    auto read = complete->Snapshot(R::Device::Host()).AcquireRead();
    int shared = 0;
    auto inverse = complete->Relation(G::Adj::Cell2Face.Inverted());
    for (I face = 0; face < complete->NumOwned(G::EntityKind::Face); ++face)
        if (NodeKey(complete->Entities(G::EntityKind::Node), read.HostView().FaceNodes(face)) == std::set<I>{1, 2})
        {
            ++shared;
            CHECK(inverse.GlobalRow(face) == std::vector<I>{0, 1});
        }
    MPI_Allreduce(MPI_IN_PLACE, &shared, 1, MPI_INT, MPI_SUM, mpi.comm);
    CHECK(shared == 1);
    CHECK(source->NumGhost(G::EntityKind::Cell) == 0);
}

TEST_CASE("Construction closure tree agrees across complete graph and node-star inputs")
{
    const auto mpi = MPI();
    auto source = Fixture(mpi, 2);
    const std::vector<std::vector<I>> expected{{0, 1, 2}, {1, 3, 2}};
    for (int mode = 0; mode < 3; ++mode)
    {
        auto owned = G::NativeMeshAssembler::CaptureOwned(*source, mpi);
        owned.cellCellsComplete = mode == 0;
        if (mode == 2)
            owned.relations[size_t(G::MeshRelation::NodeCells)].reset();
        auto closure = G::BuildNativeCellClosure(owned, mpi);
        auto rows = closure.cellNodes.Read();
        const I expectedRows = rows.NumOwned() ? 2 : 0;
        CHECK(rows.RowCount() == expectedRows);
        for (I iCell = 0; iCell < rows.RowCount(); ++iCell)
            CHECK(rows.GlobalRow(iCell) == expected.at(size_t(rows.Source().LocalToGlobal(iCell))));
        const auto &nodes = rows.Target();
        CHECK(nodes.NumGhost() == (rows.NumOwned() && mpi.rank != 0 ? 4 : 0));
        auto metadata = closure.cellInfo.ReadCombined();
        CHECK(metadata.View().RowCount() == expectedRows);
        for (I iCell = 0; iCell < expectedRows; ++iCell)
            CHECK(metadata.View()(iCell, 0).type == G::Elem::Tri3);
    }
}

TEST_CASE("Three tetrahedra retain a shared edge with three incident parents")
{
    const auto mpi = MPI();
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == 0 ? 5 : 0);
    const std::vector<std::vector<I>> all{{0, 1, 2, 3}, {0, 1, 3, 4}, {0, 1, 4, 2}};
    std::vector<std::vector<I>> local;
    for (size_t cell = 0; cell < all.size(); ++cell)
        if (int(cell) % mpi.size == mpi.rank)
            local.push_back(all[cell]);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(local.size())), boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
    G::NativeMeshInput input{mpi, 3, cells, nodes, boundaries, Rows(local), Rows({}), R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)), R::Array<G::ElemInfo>(R::RowLayout::Dense(local.size(), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1)), {}};
    {
        auto coords = input.coordinates.Write();
        const DNDS::real points[5][3]{{0, 0, -1}, {0, 0, 1}, {1, 0, 0}, {-0.5, 0.8660254037844386, 0}, {-0.5, -0.8660254037844386, 0}};
        for (I row = 0; row < nodes.NumOwned(); ++row)
            for (int c = 0; c < 3; ++c)
                coords.View()(size_t(row), size_t(c)) = points[row][c];
        auto info = input.cellInfo.Write();
        for (size_t row = 0; row < local.size(); ++row)
            info.View()(row, 0) = G::ElemInfo{G::Elem::Tet4, G::INTERNAL_ZONE};
    }
    auto source = G::ConstructNativeMesh(std::move(input));
    auto complete = G::PrepareNativeMeshTopology(*source, mpi, true, true);
    CHECK(complete->NumGlobal(G::EntityKind::Face) == 9);
    CHECK(complete->NumGlobal(G::EntityKind::Edge) == 10);
    auto read = complete->Snapshot(R::Device::Host()).AcquireRead();
    auto inverse = complete->Relation(G::Adj::Cell2Edge.Inverted());
    int shared = 0;
    for (I edge = 0; edge < complete->NumOwned(G::EntityKind::Edge); ++edge)
        if (NodeKey(complete->Entities(G::EntityKind::Node), read.HostView().EdgeNodes(edge)) == std::set<I>{0, 1})
        {
            ++shared;
            CHECK(inverse.GlobalRow(edge) == std::vector<I>{0, 1, 2});
            CHECK(read.HostView().EdgeCells(edge).size() == 3);
        }
    MPI_Allreduce(MPI_IN_PLACE, &shared, 1, MPI_INT, MPI_SUM, mpi.comm);
    CHECK(shared == 1);
}

TEST_CASE("Native topology rejects shared quadratic edge vertices with inconsistent midpoint identity")
{
    const auto mpi = MPI();
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == 0 ? 10 : 0);
    std::vector<std::vector<I>> rows;
    if (mpi.rank == 0)
        rows.push_back({0, 1, 2, 4, 5, 6});
    if (mpi.rank == (mpi.size > 1 ? 1 : 0))
        rows.push_back({1, 3, 2, 7, 8, 9});
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(rows.size())), boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
    G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, Rows(rows), Rows({}), R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)), R::Array<G::ElemInfo>(R::RowLayout::Dense(rows.size(), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1)), {}};
    {
        auto coords = input.coordinates.Write();
        const DNDS::real points[10][3]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0.5, 0, 0}, {0.5, 0.5, 0}, {0, 0.5, 0}, {1, 0.5, 0}, {0.5, 1, 0}, {0.5, 0.5, 0}};
        for (I row = 0; row < nodes.NumOwned(); ++row)
            for (int component = 0; component < 3; ++component)
                coords.View()(size_t(row), size_t(component)) = points[row][component];
        auto metadata = input.cellInfo.Write();
        for (size_t row = 0; row < rows.size(); ++row)
            metadata.View()(row, 0) = G::ElemInfo{G::Elem::Tri6, G::INTERNAL_ZONE};
    }
    auto primary = G::ConstructNativeMesh(input);
    // The shared vertices {1,2} refer to midpoint5 in cell0 and midpoint9 in
    // cell1. Distinct coordinates or a second generated face cannot repair the
    // contradictory topology; every participant must reject it.
    CHECK_THROWS(G::PrepareNativeMeshTopology(*primary, mpi));
    CHECK_NOTHROW(primary->Validate());
    input.cellNodes = input.cellNodes.Clone();
    {
        auto write = input.cellNodes.Write();
        if (mpi.rank == (mpi.size > 1 ? 1 : 0))
            write.View()(mpi.size > 1 ? 0 : 1, 5) = 5;
    }
    auto corrected = G::ConstructNativeMesh(input);
    auto complete = G::PrepareNativeMeshTopology(*corrected, mpi);
    CHECK(complete->NumGlobal(G::EntityKind::Face) == 5);
}

TEST_CASE("Generated topology keeps persistent IDs and oriented slots through archive and independent redistribution")
{
    const auto mpi = MPI();
    auto primary = Fixture(mpi, 3);
    auto completed = G::PrepareNativeMeshTopology(*primary, mpi, true, true);
    const auto expected = PersistentTopology(*completed, mpi);
    const auto name = completed->MeshNamespace();
    const auto marks = completed->IdHighWater();
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto file = (std::filesystem::temp_directory_path() / ("dndsr-topology-" + std::to_string(pid) + "-r" + std::to_string(mpi.rank) + ".json")).string();
    auto serializer = std::make_shared<DNDS::Serializer::SerializerJSON>();
    serializer->OpenFile(file, false);
    G::WriteNativeMeshArchive(*completed, mpi, serializer, "mesh");
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    auto current = G::ReadNativeMeshArchive(mpi, serializer, "mesh");
    serializer->CloseFile();
    CHECK(PersistentTopology(*current, mpi) == expected);
    for (int step = 1; step <= 2; ++step)
    {
        G::ReorderInput input;
        for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd, G::EntityKind::Face, G::EntityKind::Edge})
            input.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(current->NumOwned(kind)), (mpi.rank + step + int(kind)) % mpi.size)});
        current = G::NativeMeshTransition::Build(*current, mpi, input).state;
        CHECK(current->MeshNamespace() == name);
        CHECK(current->IdHighWater() == marks);
        CHECK(PersistentTopology(*current, mpi) == expected);
    }
    std::filesystem::remove(file);
}

namespace
{
    using TopologyKeys = std::map<std::vector<I>, I>;
    std::vector<I> RefreshOracleKey(const G::NativeMeshState &state, G::EntityKind kind, I row)
    {
        const auto &view = state.BorrowHostView();
        const auto nodes = kind == G::EntityKind::Face ? view.FaceNodes(row) : view.EdgeNodes(row);
        std::vector<std::pair<I, I>> images;
        for (size_t slot = 0; slot < nodes.size(); ++slot)
        {
            const I frame = !view.IsPeriodic() ? 0 : uint8_t(kind == G::EntityKind::Face ? view.FaceNodePeriodicBits(row)[slot] : view.EdgeNodePeriodicBits(row)[slot]);
            images.emplace_back(state.OriginalEntityId(G::EntityKind::Node, nodes[slot]), frame);
        }
        std::sort(images.begin(), images.end());
        std::vector<I> result;
        // Anchor on each occurrence of the smallest node ID. This independent
        // oracle allows orientation reversal and uniform periodic image shifts.
        for (const auto &[node, frame] : images)
        {
            if (node != images.front().first)
                break;
            auto normalized = images;
            for (auto &[id, bits] : normalized)
                bits ^= frame;
            std::sort(normalized.begin(), normalized.end());
            std::vector<I> key{int(kind), kind == G::EntityKind::Face ? view.GetFaceInfo(row).type : view.GetEdgeInfo(row).type};
            for (const auto &[id, bits] : normalized)
            {
                key.push_back(id);
                key.push_back(bits);
            }
            if (result.empty() || key < result)
                result = std::move(key);
        }
        return result;
    }
    TopologyKeys RefreshOracle(const G::NativeMeshState &state, const DNDS::MPIInfo &mpi)
    {
        std::vector<I> local;
        for (auto kind : {G::EntityKind::Face, G::EntityKind::Edge})
            if (state.HasEntities(kind))
                for (I row = 0; row < state.NumOwned(kind); ++row)
                {
                    const auto key = RefreshOracleKey(state, kind, row);
                    local.insert(local.end(), {I(key.size()), state.OriginalEntityId(kind, row)});
                    local.insert(local.end(), key.begin(), key.end());
                }
        const int count = int(local.size());
        std::vector<int> counts(size_t(mpi.size)), offsets(size_t(mpi.size) + 1);
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
        std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);
        std::vector<I> global(size_t(offsets.back()));
        MPI_Allgatherv(local.data(), count, DNDS::DNDS_MPI_INDEX, global.data(), counts.data(), offsets.data(), DNDS::DNDS_MPI_INDEX, mpi.comm);
        TopologyKeys result;
        for (size_t slot = 0; slot < global.size();)
        {
            const auto width = size_t(global[slot]);
            const auto id = global[slot + 1];
            REQUIRE(slot + 2 + width <= global.size());
            CHECK(result.emplace(std::vector<I>(global.begin() + slot + 2, global.begin() + slot + 2 + width), id).second);
            slot += width + 2;
        }
        return result;
    }
    G::AdjacencyDescriptor RefreshVariant(G::EntityKind kind, bool sorted = false)
    {
        G::AdjacencyDescriptor key{kind, kind};
        key.path = {kind, kind};
        key.selection = G::RelationSelection::Custom;
        key.order = sorted ? G::RelationOrder::SortedUnique : G::RelationOrder::Topological;
        key.periodic = G::RelationPeriodicPolicy::FrameMatched;
        key.semanticName = sorted ? "test.topology.refresh.sorted-adjacents" : "test.topology.refresh.ordered-self";
        key.semanticVersion = 1;
        return key;
    }
    std::shared_ptr<const G::NativeMeshState> RefreshPayloads(std::shared_ptr<const G::NativeMeshState> state,
                                                              const DNDS::MPIInfo &mpi)
    {
        const auto keys = RefreshOracle(*state, mpi);
        auto data = G::NativeMeshAssembler::CaptureOwned(*state, mpi);
        data.provenance.emplace();
        data.provenance->sourceNamespace = state->MeshNamespace();
        for (auto kind : {G::EntityKind::Face, G::EntityKind::Edge})
        {
            if (!state->HasEntities(kind))
                continue;
            std::vector<std::vector<I>> ids, ancestry, named;
            const I first = kind == G::EntityKind::Face ? I(1000000000000LL) : std::numeric_limits<I>::max() - 100000;
            I ordinal = 0, maximum = -1;
            std::map<std::vector<I>, I> expected;
            for (const auto &[key, unused] : keys)
                if (key[0] == int(kind))
                    expected.emplace(key, first + ordinal++ * 1009);
            for (I row = 0; row < state->NumOwned(kind); ++row)
            {
                const I id = expected.at(RefreshOracleKey(*state, kind, row));
                ids.push_back({id});
                ancestry.push_back({id, 42});
                const I global = state->Entities(kind).LocalToGlobal(row);
                named.push_back({global, DNDS::UnInitIndex, global});
                maximum = std::max(maximum, id);
            }
            MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, DNDS::DNDS_MPI_INDEX, MPI_MAX, mpi.comm);
            data.ids[kind] = Rows(ids);
            data.identity.nextIds[kind] = maximum + 101;
            data.provenance->sourceKinds[kind] = G::EntityKind::Cell;
            data.provenance->sourceIds.emplace(kind, Rows(ancestry));
            data.namedRelations.emplace(RefreshVariant(kind), Rows(named));
            std::vector<std::vector<I>> sorted(size_t(state->NumOwned(G::EntityKind::Node)));
            for (auto &row : sorted)
                for (I global = 0; global < state->NumGlobal(kind); ++global)
                    row.push_back(global);
            auto sortedKey = RefreshVariant(kind, true);
            sortedKey.from = G::EntityKind::Node;
            sortedKey.path = {G::EntityKind::Node, kind};
            data.namedRelations.emplace(sortedKey, Rows(sorted));
            const auto fixedSlot = kind == G::EntityKind::Face ? G::MeshRelation::FaceCells : G::MeshRelation::EdgeCells;
            data.namedRelations.emplace(kind == G::EntityKind::Face ? G::Adj::Face2Cell : G::Adj::Edge2Cell,
                                        *data.relations[size_t(fixedSlot)]);
        }
        return G::AssembleNativeMesh(mpi, std::move(data));
    }
    void CheckRefreshPayloads(const G::NativeMeshState &state)
    {
        REQUIRE(state.Provenance().has_value());
        for (auto kind : {G::EntityKind::Face, G::EntityKind::Edge})
        {
            if (!state.HasEntities(kind))
                continue;
            const auto read = state.Relation(RefreshVariant(kind));
            auto sortedKey = RefreshVariant(kind, true);
            sortedKey.from = G::EntityKind::Node;
            sortedKey.path = {G::EntityKind::Node, kind};
            const auto sorted = state.Relation(sortedKey);
            for (I row = 0; row < sorted.NumOwned(); ++row)
            {
                REQUIRE(sorted.RowSize(row) == size_t(state.NumGlobal(kind)));
                for (I slot = 0; slot < state.NumGlobal(kind); ++slot)
                    CHECK(sorted.ReadEntry(row, size_t(slot)).global == slot);
            }
            const auto ancestry = state.Provenance()->sourceIds.at(kind).Read();
            for (I row = 0; row < state.NumOwned(kind); ++row)
            {
                REQUIRE(read.RowSize(row) == 3);
                CHECK(read.ReadEntry(row, 1).absent);
                for (size_t slot : {size_t(0), size_t(2)})
                {
                    const auto target = state.Entities(kind).GlobalToLocal(read.ReadEntry(row, slot).global);
                    REQUIRE(target.has_value());
                    CHECK(state.OriginalEntityId(kind, *target) == state.OriginalEntityId(kind, row));
                }
                REQUIRE(ancestry.View().RowSize(size_t(row)) == 2);
                CHECK(ancestry.View()(size_t(row), 0) == state.OriginalEntityId(kind, row));
                CHECK(ancestry.View()(size_t(row), 1) == 42);
            }
        }
    }
}

TEST_CASE("Ordinary native topology refresh recovers every high ID after reorder and independent redistribution")
{
    const auto mpi = MPI();
    auto current = RefreshPayloads(G::PrepareNativeMeshTopology(*Fixture(mpi, 3), mpi, true, true), mpi);
    const auto expected = RefreshOracle(*current, mpi);
    CHECK(expected.size() == 16); // Seven distinct tetrahedron faces and nine edges.
    std::set<std::vector<I>> faces, edges;
    for (const auto &[key, id] : expected)
    {
        std::vector<I> nodes;
        for (size_t slot = 2; slot < key.size(); slot += 2)
            nodes.push_back(key[slot]);
        (key[0] == int(G::EntityKind::Face) ? faces : edges).insert(nodes);
    }
    CHECK(faces == std::set<std::vector<I>>{{0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4}, {1, 2, 3}, {1, 2, 4}});
    CHECK(edges == std::set<std::vector<I>>{{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}});
    const auto marks = current->IdHighWater();
    const auto name = current->MeshNamespace();
    for (int step = 0; step < 3; ++step)
    {
        if (step == 1)
            current = G::NativeMeshTransition::ReorderLocalCells(*current, mpi, 2).state;
        if (step == 2)
        {
            G::ReorderInput input;
            for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd, G::EntityKind::Face, G::EntityKind::Edge})
                input.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(current->NumOwned(kind)), (mpi.rank + 1 + int(kind)) % mpi.size)});
            current = G::NativeMeshTransition::Build(*current, mpi, input).state;
        }
        const auto retained = current;
        auto read = retained->Snapshot(R::Device::Host()).AcquireRead();
        current = G::PrepareNativeMeshTopology(*current, mpi, true, true);
        CHECK(current->MeshNamespace() == name);
        CHECK(current->IdHighWater() == marks);
        CHECK(RefreshOracle(*current, mpi) == expected);
        CheckRefreshPayloads(*current);
        CHECK_NOTHROW(read.HostView());
        CHECK(RefreshOracle(*retained, mpi) == expected);
    }
    auto facade = G::UnstructuredMesh::FromNative(mpi, current);
    facade->PrepareNativeTopology(true, true);
    CHECK(RefreshOracle(*facade->CurrentNativeState(), mpi) == expected);
    CHECK(facade->CurrentNativeState()->IdHighWater() == marks);
    CheckRefreshPayloads(*facade->CurrentNativeState());
    for (const auto &flags : {std::pair<bool, bool>{false, false}, {true, false}, {false, true}})
    {
        current = G::PrepareNativeMeshTopology(*current, mpi, flags.first, flags.second);
        CHECK(RefreshOracle(*current, mpi) == expected);
        CHECK(current->IdHighWater() == marks);
        CheckRefreshPayloads(*current);
    }
}

TEST_CASE("Native topology refresh matches exact periodic occurrence keys and preserves named ancestry")
{
    const auto mpi = MPI();
    for (bool repeated : {false, true})
    {
        CAPTURE(repeated);
        auto primary = G::ConstructNativeMesh(repeated ? NativePeriodicFixtures::QuadraticTorus(mpi) : NativePeriodicFixtures::TwoCells(mpi));
        auto current = RefreshPayloads(G::PrepareNativeMeshTopology(*primary, mpi), mpi);
        const auto expected = RefreshOracle(*current, mpi);
        CHECK(expected.size() == (repeated ? 2 : 6));
        const auto marks = current->IdHighWater();
        for (int step = 0; step < 2; ++step)
        {
            G::ReorderInput input;
            for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd, G::EntityKind::Face})
                input.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(current->NumOwned(kind)), (mpi.rank + step + 1 + int(kind)) % mpi.size)});
            current = G::NativeMeshTransition::Build(*current, mpi, input).state;
            current = G::PrepareNativeMeshTopology(*current, mpi);
            CHECK(current->IdHighWater() == marks);
            CHECK(RefreshOracle(*current, mpi) == expected);
            CheckRefreshPayloads(*current);
        }
    }
}

TEST_CASE("Native refresh allocates only unmatched keys and rejects uninferable retained associations collectively")
{
    const auto mpi = MPI();
    auto complete = RefreshPayloads(G::PrepareNativeMeshTopology(*Fixture(mpi, 3), mpi, true, true), mpi);
    const auto original = RefreshOracle(*complete, mpi);
    const I highWater = complete->NextEntityId(G::EntityKind::Face);
    auto staleData = G::NativeMeshAssembler::CaptureOwned(*complete, mpi);
    auto cone = staleData.relations[size_t(G::MeshRelation::FaceNodes)]->Clone();
    const auto faces = complete->Entities(G::EntityKind::Face);
    // Represent a stale generated cone left behind after primary topology was
    // changed. {0,3,4} is not any face of the two known tetrahedra. Typed array
    // validation cannot establish that full cell/subentity consistency.
    if (faces.Owner(0) == mpi.rank)
    {
        auto write = cone.Write();
        const size_t row = size_t(faces.OwnerLocalRow(0));
        for (size_t slot = 0; slot < 3; ++slot)
            write.View()(row, slot) = std::vector<I>{0, 3, 4}[slot];
    }
    staleData.relations[size_t(G::MeshRelation::FaceNodes)] = std::move(cone);
    auto stale = G::AssembleNativeMesh(mpi, staleData);
    // The new source face has no old application row; neither named rows nor
    // ancestry can be invented. Failure leaves both publications usable.
    CHECK_THROWS(G::PrepareNativeMeshTopology(*stale, mpi, true, false));
    CHECK_NOTHROW(stale->Validate());
    CHECK(RefreshOracle(*complete, mpi) == original);
    staleData.provenance->sourceIds.erase(G::EntityKind::Face);
    staleData.provenance->sourceKinds.erase(G::EntityKind::Face);
    staleData.namedRelations.erase(RefreshVariant(G::EntityKind::Face));
    auto sortedFace = RefreshVariant(G::EntityKind::Face, true);
    sortedFace.from = G::EntityKind::Node;
    sortedFace.path = {G::EntityKind::Node, G::EntityKind::Face};
    staleData.namedRelations.erase(sortedFace);
    auto noApplicationRows = G::AssembleNativeMesh(mpi, std::move(staleData));
    auto refreshed = G::PrepareNativeMeshTopology(*noApplicationRows, mpi, true, false);
    const auto actual = RefreshOracle(*refreshed, mpi);
    REQUIRE(actual.size() == original.size());
    size_t replaced = 0;
    for (const auto &[key, id] : actual)
    {
        const auto found = original.find(key);
        REQUIRE(found != original.end());
        if (id != found->second)
        {
            CHECK(key[0] == int(G::EntityKind::Face));
            CHECK(id == highWater);
            ++replaced;
        }
    }
    CHECK(replaced == 1);
    CHECK(refreshed->NextEntityId(G::EntityKind::Face) == highWater + 1);
    CHECK(refreshed->NextEntityId(G::EntityKind::Edge) == complete->NextEntityId(G::EntityKind::Edge));
    auto again = G::PrepareNativeMeshTopology(*refreshed, mpi, true, false);
    CHECK(RefreshOracle(*again, mpi) == actual);
    CHECK(again->IdHighWater() == refreshed->IdHighWater());
    const I floor = refreshed->NextEntityId(G::EntityKind::Face) + 23;
    auto fresh = G::PrepareNativeMeshTopology(*refreshed, mpi, true, false, 1, {{G::EntityKind::Face, floor}});
    const auto freshKeys = RefreshOracle(*fresh, mpi);
    for (const auto &[key, id] : freshKeys)
        if (key[0] == int(G::EntityKind::Face))
        {
            CHECK(id >= floor);
            CHECK(id < floor + fresh->NumGlobal(G::EntityKind::Face));
            CHECK(id != actual.at(key));
        }
    CHECK(fresh->NextEntityId(G::EntityKind::Face) == floor + fresh->NumGlobal(G::EntityKind::Face));
}

TEST_CASE("Periodic three dimensional refresh preserves repeated-node face and edge keys")
{
    const auto mpi = MPI();
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == mpi.size - 1 ? 4 : 0);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? 1 : 0);
    auto bnd = nodes.CreateSibling(G::EntityKind::Bnd, mpi.rank == mpi.size / 2 ? 2 : 0);
    G::NativeMeshInput input{mpi, 3, cells, nodes, bnd,
                             Rows(cells.NumOwned() ? std::vector<std::vector<I>>{{0, 0, 1, 1, 2, 2, 3, 3}} : std::vector<std::vector<I>>{}),
                             Rows(bnd.NumOwned() ? std::vector<std::vector<I>>{{0, 1, 3, 2}, {0, 2, 3, 1}} : std::vector<std::vector<I>>{}),
                             R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(bnd.NumOwned()), 1))};
    input.periodic = true;
    input.cellNodesPbi = NativePeriodicFixtures::Rows<G::NodePeriodicBits>(cells.NumOwned()
                                                                               ? std::vector<std::vector<G::NodePeriodicBits>>{{{0}, {1}, {1}, {0}, {0}, {1}, {1}, {0}}}
                                                                               : std::vector<std::vector<G::NodePeriodicBits>>{});
    input.boundaryNodesPbi = NativePeriodicFixtures::Rows<G::NodePeriodicBits>(bnd.NumOwned()
                                                                                   ? std::vector<std::vector<G::NodePeriodicBits>>{{{1}, {1}, {1}, {1}}, {{0}, {0}, {0}, {0}}}
                                                                                   : std::vector<std::vector<G::NodePeriodicBits>>{});
    {
        auto coords = input.coordinates.Write();
        for (I row = 0; row < nodes.NumOwned(); ++row)
        {
            coords.View()(size_t(row), 0) = 0;
            coords.View()(size_t(row), 1) = row % 2;
            coords.View()(size_t(row), 2) = row / 2;
        }
        auto info = input.cellInfo.Write();
        if (cells.NumOwned())
            info.View()(0, 0) = G::ElemInfo{G::Elem::Hex8, G::INTERNAL_ZONE};
        auto boundary = input.boundaryInfo.Write();
        if (bnd.NumOwned())
        {
            boundary.View()(0, 0) = G::ElemInfo{G::Elem::Quad4, G::BC_ID_PERIODIC_1};
            boundary.View()(1, 0) = G::ElemInfo{G::Elem::Quad4, G::BC_ID_PERIODIC_1_DONOR};
        }
    }
    auto primary = G::ConstructNativeMesh(std::move(input));
    auto current = RefreshPayloads(G::PrepareNativeMeshTopology(*primary, mpi, true, true), mpi);
    CHECK(current->NumGlobal(G::EntityKind::Face) == 5);
    CHECK(current->NumGlobal(G::EntityKind::Edge) == 8);
    const auto expected = RefreshOracle(*current, mpi);
    const auto marks = current->IdHighWater();
    for (int step = 0; step < 2; ++step)
    {
        G::ReorderInput placement;
        for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd, G::EntityKind::Face, G::EntityKind::Edge})
            placement.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(current->NumOwned(kind)), (mpi.rank + 1 + step + int(kind)) % mpi.size)});
        current = G::NativeMeshTransition::Build(*current, mpi, placement).state;
        current = G::PrepareNativeMeshTopology(*current, mpi, true, true);
        CHECK(current->IdHighWater() == marks);
        CHECK(RefreshOracle(*current, mpi) == expected);
        CheckRefreshPayloads(*current);
    }
}

TEST_CASE("Distinct full topology keys retain different IDs even under an exact rendezvous hash collision")
{
    const auto mpi = MPI();
    auto source = Fixture(mpi, 2);
    auto data = G::NativeMeshAssembler::CaptureOwned(*source, mpi);
    // These Line2 full keys [type,2,node,0,node,0] collide under the
    // rendezvous's 64-bit hash, not merely modulo the communicator size.
    const I idValues[]{0, 1, 2, 3329321212988115LL};
    auto hash = [](std::initializer_list<I> key)
    {
        uint64_t value = 1469598103934665603ULL;
        for (auto word : key)
            value = (value ^ uint64_t(word)) * 1099511628211ULL;
        return value;
    };
    REQUIRE(hash({G::Elem::Line2, 2, idValues[0], 0, idValues[1], 0}) ==
            hash({G::Elem::Line2, 2, idValues[2], 0, idValues[3], 0}));
    {
        auto write = data.ids.at(G::EntityKind::Node).Write();
        for (I row = 0; row < source->NumOwned(G::EntityKind::Node); ++row)
            write.View()(size_t(row), 0) = idValues[source->Entities(G::EntityKind::Node).LocalToGlobal(row)];
    }
    data.identity.nextIds[G::EntityKind::Node] = idValues[3] + 1;
    source = G::AssembleNativeMesh(mpi, std::move(data));
    auto current = RefreshPayloads(G::PrepareNativeMeshTopology(*source, mpi), mpi);
    const auto expected = RefreshOracle(*current, mpi);
    CHECK(expected.size() == 5);
    CHECK(expected.at({int(G::EntityKind::Face), G::Elem::Line2, idValues[0], 0, idValues[1], 0}) !=
          expected.at({int(G::EntityKind::Face), G::Elem::Line2, idValues[2], 0, idValues[3], 0}));
    current = G::PrepareNativeMeshTopology(*current, mpi);
    CHECK(RefreshOracle(*current, mpi) == expected);
    CheckRefreshPayloads(*current);
}

TEST_CASE("Intermediate-only named definitions reject added topology keys without inventing new paths")
{
    const auto mpi = MPI();
    auto complete = G::PrepareNativeMeshTopology(*Fixture(mpi, 3), mpi);
    const auto allKeys = RefreshOracle(*complete, mpi);
    auto data = G::NativeMeshAssembler::CaptureOwned(*complete, mpi);
    const auto oldSpace = complete->Entities(G::EntityKind::Face);
    const I removed = oldSpace.NumGlobal() - 1;
    const I newCount = oldSpace.NumOwned() - I(oldSpace.Owner(removed) == mpi.rank);
    auto truncate = [&](const R::Array<I> &array)
    {
        std::vector<std::vector<I>> rows;
        auto read = array.Read();
        for (I row = 0; row < newCount; ++row)
            rows.emplace_back(read.View()[size_t(row)].begin(), read.View()[size_t(row)].end());
        auto result = Rows(rows);
        if (!array.Layout().IsVariable())
            result = R::Array<I>::FromStorage(R::RowLayout::Dense(rows.size(), array.Layout().Width()), result.Storage());
        return result;
    };
    // Retain the first six exact keys. The stale primary-to-face slot for the
    // missing seventh key is still a valid index; only reconstruction can know
    // that one more physical face is required by the authoritative cell cones.
    for (auto slot : {G::MeshRelation::FaceNodes, G::MeshRelation::FaceCells, G::MeshRelation::FaceBoundaries})
        data.relations[size_t(slot)] = truncate(*data.relations[size_t(slot)]);
    data.ids[G::EntityKind::Face] = truncate(data.ids.at(G::EntityKind::Face));
    R::Array<G::ElemInfo> information(R::RowLayout::Dense(size_t(newCount), 1));
    {
        auto old = data.faceInfo->Read();
        auto write = information.Write();
        for (I row = 0; row < newCount; ++row)
            write.View()(size_t(row), 0) = old.View()(size_t(row), 0);
    }
    data.faceInfo = std::move(information);
    auto slots = data.relations[size_t(G::MeshRelation::CellFaces)]->Clone();
    {
        auto write = slots.Write();
        for (size_t row = 0; row < slots.RowCount(); ++row)
            for (size_t slot = 0; slot < write.View().RowSize(row); ++slot)
                if (write.View()(row, slot) == removed)
                    write.View()(row, slot) = 0;
    }
    data.relations[size_t(G::MeshRelation::CellFaces)] = std::move(slots);
    data.spaces.at(G::EntityKind::Face) = data.spaces.at(G::EntityKind::Node).CreateSibling(G::EntityKind::Face, newCount);
    for (auto item = data.namedRelations.begin(); item != data.namedRelations.end();)
    {
        const auto path = item->first.Path();
        if (std::find(path.begin(), path.end(), G::EntityKind::Face) != path.end())
            item = data.namedRelations.erase(item);
        else
            ++item;
    }
    auto composed = G::AdjacencyDescriptor::Composition(G::Adj::Cell2Face, G::Adj::Face2Cell, G::RelationSelection::Custom,
                                                        false, "test.refresh.intermediate-only", 1);
    composed.order = G::RelationOrder::Topological;
    data.namedRelations.emplace(composed, *data.relations[size_t(G::MeshRelation::CellFaceCells)]);
    auto partial = G::AssembleNativeMesh(mpi, data);
    REQUIRE(partial->NumGlobal(G::EntityKind::Face) == 6);
    CHECK_THROWS(G::PrepareNativeMeshTopology(*partial, mpi));
    CHECK_NOTHROW(partial->Validate());
    data.namedRelations.erase(composed);
    partial = G::AssembleNativeMesh(mpi, std::move(data));
    auto repaired = G::PrepareNativeMeshTopology(*partial, mpi);
    const auto keys = RefreshOracle(*repaired, mpi);
    CHECK(keys.size() == allKeys.size());
    I fresh = 0;
    for (const auto &[key, id] : keys)
    {
        REQUIRE(allKeys.count(key));
        if (id != allKeys.at(key))
        {
            CHECK(id == complete->NextEntityId(G::EntityKind::Face));
            ++fresh;
        }
    }
    CHECK(fresh == 1);
}

TEST_CASE("Topology ID reuse distinguishes higher order vertex roles despite equal full node image multisets")
{
    const auto mpi = MPI();
    auto primary = G::ConstructNativeMesh(NativePeriodicFixtures::QuadraticTorus(mpi));
    auto complete = G::PrepareNativeMeshTopology(*primary, mpi);
    const auto original = RefreshOracle(*complete, mpi);
    const auto space = complete->Entities(G::EntityKind::Face);
    auto data = G::NativeMeshAssembler::CaptureOwned(*complete, mpi);
    auto nodes = data.relations[size_t(G::MeshRelation::FaceNodes)]->Clone();
    auto bits = data.faceNodesPbi->Clone();
    if (space.Owner(0) == mpi.rank)
    {
        auto nodeWrite = nodes.Write();
        auto bitWrite = bits.Write();
        const auto row = size_t(space.OwnerLocalRow(0));
        // Swap an endpoint occurrence with the midside occurrence, retaining
        // the exact same full multiset of (persistent node ID,image) pairs.
        std::swap(nodeWrite.View()(row, 1), nodeWrite.View()(row, 2));
        std::swap(bitWrite.View()(row, 1), bitWrite.View()(row, 2));
    }
    data.relations[size_t(G::MeshRelation::FaceNodes)] = std::move(nodes);
    data.faceNodesPbi = std::move(bits);
    auto stale = G::AssembleNativeMesh(mpi, std::move(data));
    CHECK(RefreshOracle(*stale, mpi) == original); // A full-multiset-only matcher misses this.
    auto recovered = G::PrepareNativeMeshTopology(*stale, mpi);
    const auto actual = RefreshOracle(*recovered, mpi);
    REQUIRE(actual.size() == original.size());
    size_t fresh = 0;
    for (const auto &[key, id] : actual)
        if (id != original.at(key))
        {
            CHECK(id == complete->NextEntityId(G::EntityKind::Face));
            ++fresh;
        }
    CHECK(fresh == 1);
    CHECK(recovered->NextEntityId(G::EntityKind::Face) == complete->NextEntityId(G::EntityKind::Face) + 1);
}

namespace
{
    std::shared_ptr<const G::NativeMeshState> QuadraticHexPair(const DNDS::MPIInfo &mpi)
    {
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == mpi.size - 1 ? 12 : 0);
        const bool first = mpi.rank == 0, second = mpi.rank == (mpi.size > 1 ? 1 : 0);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(first) + I(second));
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
        std::vector<std::vector<I>> cones;
        if (first)
            cones.push_back({0, 1, 4, 3, 6, 7, 10, 9});
        if (second)
            cones.push_back({1, 2, 5, 4, 7, 8, 11, 10});
        G::NativeMeshInput input{mpi, 3, cells, nodes, boundaries, Rows(cones), Rows({}),
                                 R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
        {
            auto points = input.coordinates.Write();
            for (I row = 0; row < nodes.NumOwned(); ++row)
            {
                points.View()(size_t(row), 0) = row % 3;
                points.View()(size_t(row), 1) = row / 3 % 2;
                points.View()(size_t(row), 2) = row / 6;
            }
            auto info = input.cellInfo.Write();
            for (I row = 0; row < cells.NumOwned(); ++row)
                info.View()(size_t(row), 0) = G::ElemInfo{G::Elem::Hex8, G::INTERNAL_ZONE};
        }
        auto primary = G::ConstructNativeMesh(std::move(input));
        return G::NativeMeshRefinement::Elevate(*primary, mpi).state;
    }
}

TEST_CASE("Quadratic topology roles reject actual cracked shared faces and prevent stale generated ID reuse")
{
    const auto mpi = MPI();
    const auto primary = QuadraticHexPair(mpi);
    const auto complete = G::PrepareNativeMeshTopology(*primary, mpi, true, true);
    REQUIRE(complete->NumGlobal(G::EntityKind::Face) == 11);
    const auto expected = RefreshOracle(*complete, mpi);
    for (bool center : {false, true})
    {
        CAPTURE(center);
        auto data = G::NativeMeshAssembler::CaptureOwned(*primary, mpi);
        auto cone = data.relations[size_t(G::MeshRelation::CellNodes)]->Clone();
        const auto cells = primary->Entities(G::EntityKind::Cell);
        if (cells.Owner(1) == mpi.rank)
        {
            const I row = cells.OwnerLocalRow(1);
            G::Elem::Element element{G::Elem::Hex27};
            std::vector<I> localSlots(27);
            std::iota(localSlots.begin(), localSlots.end(), I(0));
            size_t found = 0;
            auto write = cone.Write();
            for (G::t_index face = 0; face < element.GetNumFaces(); ++face)
            {
                std::vector<I> slots(9);
                element.ExtractFaceNodes(face, localSlots, slots);
                bool shared = true;
                for (size_t corner = 0; corner < 4; ++corner)
                    shared = shared && primary->BorrowHostView().Coord(primary->BorrowHostView().CellNodes(row)[size_t(slots[corner])])[0] == 1;
                if (!shared)
                    continue;
                std::swap(write.View()(size_t(row), size_t(slots[4])), write.View()(size_t(row), size_t(slots[center ? 8 : 5])));
                ++found;
            }
            REQUIRE(found == 1);
        }
        data.relations[size_t(G::MeshRelation::CellNodes)] = cone;
        if (data.namedRelations.count(G::Adj::Cell2Node))
            data.namedRelations.at(G::Adj::Cell2Node) = cone;
        auto cracked = G::AssembleNativeMesh(mpi, std::move(data));
        CHECK_THROWS(G::PrepareNativeMeshTopology(*cracked, mpi, true, true));
        CHECK_NOTHROW(cracked->Validate());
        CHECK(RefreshOracle(*G::PrepareNativeMeshTopology(*primary, mpi, true, true), mpi) == expected);
        auto staleData = G::NativeMeshAssembler::CaptureOwned(*complete, mpi);
        auto faces = staleData.relations[size_t(G::MeshRelation::FaceNodes)]->Clone();
        const auto space = complete->Entities(G::EntityKind::Face);
        if (space.Owner(0) == mpi.rank)
        {
            auto write = faces.Write();
            const auto row = size_t(space.OwnerLocalRow(0));
            std::swap(write.View()(row, 4), write.View()(row, center ? 8 : 5));
        }
        staleData.relations[size_t(G::MeshRelation::FaceNodes)] = faces;
        if (staleData.namedRelations.count(G::Adj::Face2Node))
            staleData.namedRelations.at(G::Adj::Face2Node) = faces;
        auto stale = G::AssembleNativeMesh(mpi, std::move(staleData));
        CHECK(RefreshOracle(*stale, mpi) == expected); // Same corners and full node sets.
        auto repaired = G::PrepareNativeMeshTopology(*stale, mpi, true, false);
        size_t fresh = 0;
        for (const auto &[key, id] : RefreshOracle(*repaired, mpi))
            if (id != expected.at(key))
            {
                CHECK(key[0] == int(G::EntityKind::Face));
                CHECK(id == complete->NextEntityId(G::EntityKind::Face));
                ++fresh;
            }
        CHECK(fresh == 1);
    }
}

namespace
{
    // Frozen generic reference: enumerate the complete XOR orbit and construct
    // role vectors through Element extraction, without candidate-mask pruning
    // or the linear stack representation used by the production fast path.
    inline std::vector<I> ReferenceRoleKeyAllImages(G::Elem::Element element, const G::NativePeriodicNodes &pattern, bool periodic)
    {
        DNDS_check_throw_info(!pattern.empty(), "native topology identity requires node occurrences");
        std::vector<I> best;
        for (uint8_t mask = 0; mask < 8; ++mask)
        {
            if (!periodic && mask != 0)
                continue;
            auto image = pattern;
            for (auto &[node, frame] : image)
            {
                frame ^= mask;
            }
            std::sort(image.begin(), image.end());
            std::vector<I> key{element.type, I(image.size())};
            for (const auto &[node, frame] : image)
            {
                key.push_back(node);
                key.push_back(frame);
            }
            // Interpolation first compares element vertices, then
            // the complete higher-order nodes. Preserve that role
            // distinction: an endpoint cannot become a midside node.
            const auto count = element.GetNumVertices();
            auto vertices = G::NativePeriodicNodes(pattern.begin(), pattern.begin() + count);
            for (auto &[node, frame] : vertices)
                frame ^= mask;
            std::sort(vertices.begin(), vertices.end());
            key.push_back(count);
            for (const auto &[node, frame] : vertices)
            {
                key.push_back(node);
                key.push_back(frame);
            }
            // Generated 2-D faces have 1-D subfaces (the Element
            // edge API is for 3-D cells). Preserve which midpoint
            // belongs to each edge, and thus Quad9 center identity.
            std::vector<std::vector<I>> edges;
            if (element.GetDim() == 2)
                for (G::t_index slot = 0; slot < element.GetNumFaces(); ++slot)
                {
                    const auto sub = element.ObtainFace(slot);
                    G::NativePeriodicNodes occurrences(size_t(sub.GetNumNodes()));
                    element.ExtractFaceNodes(slot, pattern, occurrences);
                    for (auto &[node, frame] : occurrences)
                        frame ^= mask;
                    auto corners = G::NativePeriodicNodes(occurrences.begin(), occurrences.begin() + sub.GetNumVertices());
                    std::sort(corners.begin(), corners.end());
                    std::sort(occurrences.begin(), occurrences.end());
                    std::vector<I> edge{sub.type, sub.GetNumVertices()};
                    for (const auto &[node, frame] : corners)
                    {
                        edge.push_back(node);
                        edge.push_back(frame);
                    }
                    edge.push_back(I(occurrences.size()));
                    for (const auto &[node, frame] : occurrences)
                    {
                        edge.push_back(node);
                        edge.push_back(frame);
                    }
                    edges.push_back(std::move(edge));
                }
            std::sort(edges.begin(), edges.end());
            key.push_back(I(edges.size()));
            for (const auto &edge : edges)
            {
                key.push_back(I(edge.size()));
                key.insert(key.end(), edge.begin(), edge.end());
            }
            if (best.empty() || key < best)
                best = std::move(key);
        }
        return best;
    }

}

TEST_CASE("Canonical topology role keys equal the independent full-image reference for every supported subentity")
{
    std::mt19937_64 random(0x3856c0a9ULL);
    size_t checked = 0;
    for (auto type : {G::Elem::Line2, G::Elem::Line3, G::Elem::Tri3, G::Elem::Tri6, G::Elem::Quad4, G::Elem::Quad9})
    {
        const G::Elem::Element element{type};
        for (bool periodic : {false, true})
            for (int sample = 0; sample < 600; ++sample)
            {
                G::NativePeriodicNodes pattern(size_t(element.GetNumNodes()));
                for (size_t slot = 0; slot < pattern.size(); ++slot)
                {
                    // Repeated canonical IDs and distinct images intentionally
                    // include ambiguous minima, collisions and degenerate roles.
                    const I node = sample % 17 == 0  ? std::numeric_limits<I>::max() - 100 + I(random() % 7)
                                   : sample % 3 == 0 ? 7
                                                     : I(random() % (sample % 3 == 1 ? 3 : 1000));
                    pattern[slot] = {node, uint8_t(random() % 8)};
                }
                const auto expected = ReferenceRoleKeyAllImages(element, pattern, periodic);
                CHECK(G::detail::CanonicalRoleKey(element, pattern, periodic) == expected);
                ++checked;
                if (sample % 10 == 0)
                {
                    auto changed = pattern;
                    for (auto &entry : changed)
                        entry.second ^= uint8_t(sample % 8);
                    const auto changedExpected = ReferenceRoleKeyAllImages(element, changed, periodic);
                    CHECK(G::detail::CanonicalRoleKey(element, changed, periodic) == changedExpected);
                    if (periodic)
                        CHECK(changedExpected == expected);
                    std::shuffle(changed.begin(), changed.end(), random);
                    CHECK(G::detail::CanonicalRoleKey(element, changed, periodic) == ReferenceRoleKeyAllImages(element, changed, periodic));
                }
            }
    }
    CHECK(checked == 7200);
    const G::NativePeriodicNodes square{{10, 0}, {20, 1}, {30, 3}, {40, 2}};
    for (bool periodic : {false, true})
    {
        auto roleBroken = square;
        std::swap(roleBroken[1], roleBroken[2]); // Same vertex/image multiset, different cyclic edges.
        const auto key = G::detail::CanonicalRoleKey({G::Elem::Quad4}, square, periodic);
        CHECK(key == ReferenceRoleKeyAllImages({G::Elem::Quad4}, square, periodic));
        CHECK(key != G::detail::CanonicalRoleKey({G::Elem::Quad4}, roleBroken, periodic));
        auto reversed = square;
        std::reverse(reversed.begin(), reversed.end());
        CHECK(key == G::detail::CanonicalRoleKey({G::Elem::Quad4}, reversed, periodic));
    }
    CHECK_THROWS(G::detail::CanonicalRoleKey({G::Elem::Line2}, {}, true));
    CHECK_THROWS(G::detail::CanonicalRoleKey({G::Elem::Quad4}, {{0, 8}, {1, 0}, {2, 0}, {3, 0}}, true));
    CHECK_THROWS(G::detail::CanonicalRoleKey({G::Elem::Tri3}, {{0, 0}, {1, 9}, {2, 0}}, false));
    for (auto type : {G::Elem::Line3, G::Elem::Tri6, G::Elem::Quad9})
    {
        const G::Elem::Element element{type};
        G::NativePeriodicNodes roles(size_t(element.GetNumNodes()));
        for (size_t node = 0; node < roles.size(); ++node)
            roles[node] = {I(node + 1), 0};
        auto broken = roles;
        if (type == G::Elem::Line3)
            std::swap(broken[0], broken[2]); // Endpoint versus midside role.
        else if (type == G::Elem::Tri6)
            std::swap(broken[3], broken[4]); // Different edge midpoints.
        else
            std::swap(broken[4], broken[8]); // Edge midpoint versus center.
        for (bool periodic : {false, true})
        {
            const auto validKey = G::detail::CanonicalRoleKey(element, roles, periodic);
            const auto brokenKey = G::detail::CanonicalRoleKey(element, broken, periodic);
            CHECK(validKey == ReferenceRoleKeyAllImages(element, roles, periodic));
            CHECK(brokenKey == ReferenceRoleKeyAllImages(element, broken, periodic));
            CHECK(validKey != brokenKey);
        }
    }
}

int main(int argc, char **argv)
{
    // Native row kernels may create OMP/CPU workers even with controller MPI.
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    const auto result = doctest::Context(argc, argv).run();
    DNDS::MPI::Finalize();
    return result;
}
