#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Refinement/NativeMeshExtraction.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/IO/NativeMeshArchive.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
#include "Geom/Access/RelationAccess.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <filesystem>
#include <unistd.h>

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
            std::copy(rows[row].begin(), rows[row].end(), builder.Row<I>(row).data);
        }
        return R::Array<I>::FromSharedStorage(builder.Commit());
    }
    DNDS::real Coordinate(int dimension, I node, size_t component)
    {
        const DNDS::real tri[7][3]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {.5, 0, 0}, {.5, .5, 0}, {0, .5, 0}, {9, 9, 9}};
        const DNDS::real tet[11][3]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {.5, 0, 0}, {.5, .5, 0}, {0, .5, 0}, {0, 0, .5}, {.5, 0, .5}, {0, .5, .5}, {9, 9, 9}};
        return dimension == 2 ? tri[node][component] : tet[node][component];
    }
    std::shared_ptr<const G::NativeMeshState> Fixture(const DNDS::MPIInfo &mpi, int dimension, bool noBoundaries = false)
    {
        // Cell, boundary and node owners are independent; higher MPI sizes also
        // include ranks with no entities. Node order is known independently.
        const I nNode = mpi.rank == mpi.size - 1 ? (dimension == 2 ? 7 : 11) : 0;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, nNode);
        std::vector<std::vector<I>> cells, boundaries;
        if (mpi.rank == 0)
            cells = {dimension == 2 ? std::vector<I>{0, 1, 2, 3, 4, 5} : std::vector<I>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
        if (!noBoundaries && mpi.rank == mpi.size / 2)
            boundaries = dimension == 2 ? std::vector<std::vector<I>>{{1, 0, 3}, {2, 1, 4}} : std::vector<std::vector<I>>{{0, 1, 2, 4, 5, 6}, {0, 3, 1, 7, 8, 4}};
        auto cellSpace = nodes.CreateSibling(G::EntityKind::Cell, I(cells.size()));
        auto boundarySpace = nodes.CreateSibling(G::EntityKind::Bnd, I(boundaries.size()));
        G::NativeMeshInput input{mpi, dimension, cellSpace, nodes, boundarySpace, Rows(cells), Rows(boundaries),
                                 R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nNode), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(cells.size(), 1)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(boundaries.size(), 1))};
        {
            auto coords = input.coordinates.Write();
            for (I node = 0; node < nNode; ++node)
                for (size_t component = 0; component < 3; ++component)
                    coords.View()(size_t(node), component) = Coordinate(dimension, node, component);
            auto cellInfo = input.cellInfo.Write();
            for (size_t cell = 0; cell < cells.size(); ++cell)
                cellInfo.View()(cell, 0) = G::ElemInfo{dimension == 2 ? G::Elem::Tri6 : G::Elem::Tet10, G::INTERNAL_ZONE};
            auto boundaryInfo = input.boundaryInfo.Write();
            for (size_t boundary = 0; boundary < boundaries.size(); ++boundary)
                boundaryInfo.View()(boundary, 0) = G::ElemInfo{dimension == 2 ? G::Elem::Line3 : G::Elem::Tri6, G::t_index(12 + boundary)};
        }
        std::vector<std::vector<I>> nodeIds, bndIds;
        for (I node = 0; node < nNode; ++node)
            nodeIds.push_back({1000 + 7 * node});
        for (size_t row = 0; row < boundaries.size(); ++row)
            bndIds.push_back({2000 + 13 * I(row)});
        input.ids.emplace(G::EntityKind::Node, Rows(nodeIds));
        input.ids.emplace(G::EntityKind::Bnd, Rows(bndIds));
        return G::ConstructNativeMesh(std::move(input));
    }
    void CheckExtracted(const G::NativeMeshState &state, const std::string &sourceNamespace, int dimension, bool faces)
    {
        CHECK(state.BorrowHostView().Dimension() == dimension - 1);
        CHECK(state.MeshNamespace() != sourceNamespace);
        CHECK(state.NumGlobal(G::EntityKind::Cell) == 2);
        CHECK(state.NumGlobal(G::EntityKind::Node) == (dimension == 2 ? 5 : 9));
        CHECK(state.NumGlobal(G::EntityKind::Bnd) == 0);
        REQUIRE(state.Provenance());
        const auto &ancestry = *state.Provenance();
        CHECK(ancestry.sourceNamespace == sourceNamespace);
        CHECK(ancestry.sourceKinds.at(G::EntityKind::Cell) == G::EntityKind::Bnd);
        CHECK(ancestry.sourceKinds.at(G::EntityKind::Node) == G::EntityKind::Node);
        auto nodeIds = ancestry.sourceIds.at(G::EntityKind::Node).Read();
        auto cellIds = ancestry.sourceIds.at(G::EntityKind::Cell).Read();
        const auto &view = state.BorrowHostView();
        // Small test-only owner exchange makes the oracle independent of
        // current numbering after redistribution. Production never gathers it.
        auto mpi = MPI();
        std::vector<I> local(size_t(state.NumGlobal(G::EntityKind::Node)), -1), parents(local.size());
        for (I row = 0; row < state.NumOwned(G::EntityKind::Node); ++row)
        {
            const auto oldNode = (nodeIds.View()(size_t(row), 0) - 1000) / 7;
            local[size_t(state.Entities(G::EntityKind::Node).LocalToGlobal(row))] = oldNode;
            for (size_t component = 0; component < 3; ++component)
                CHECK(view.Coord(row)[component] == Coordinate(dimension, oldNode, component));
        }
        MPI_Allreduce(local.data(), parents.data(), int(local.size()), DNDS::DNDS_MPI_INDEX, MPI_MAX, mpi.comm);
        const std::vector<std::vector<I>> expected = dimension == 2 ? (faces ? std::vector<std::vector<I>>{{0, 1, 3}, {1, 2, 4}} : std::vector<std::vector<I>>{{1, 0, 3}, {2, 1, 4}}) : (faces ? std::vector<std::vector<I>>{{0, 2, 1, 6, 5, 4}, {0, 1, 3, 4, 8, 7}} : std::vector<std::vector<I>>{{0, 1, 2, 4, 5, 6}, {0, 3, 1, 7, 8, 4}});
        auto cones = state.Relation(G::Adj::Cell2Node);
        for (I cell = 0; cell < state.NumOwned(G::EntityKind::Cell); ++cell)
        {
            const auto original = (cellIds.View()(size_t(cell), 0) - 2000) / 13;
            REQUIRE(original >= 0);
            REQUIRE(original < 2);
            std::vector<I> actual;
            for (auto global : cones.GlobalRow(cell))
                actual.push_back(parents.at(size_t(global)));
            CHECK(actual == expected[size_t(original)]);
            CHECK(view.GetCellInfo(cell).zone == 12 + original);
        }
    }
}

TEST_CASE("Native boundary extraction compacts owner nodes and preserves ordered high-order geometry")
{
    const auto mpi = MPI();
    int dimension = 2;
    bool faces = false;
    SUBCASE("quadratic curve without faces") {}
    SUBCASE("quadratic curve with oriented faces") { faces = true; }
    SUBCASE("quadratic surface without faces") { dimension = 3; }
    SUBCASE("quadratic surface with oriented faces")
    {
        dimension = 3;
        faces = true;
    }
    auto source = Fixture(mpi, dimension);
    if (faces)
    {
        source = G::PrepareNativeMeshTopology(*source, mpi);
        auto read = source->Snapshot(R::Device::Host()).AcquireRead();
        const auto view = read.HostView();
        for (I iBnd = 0; iBnd < view.NumBnd(); ++iBnd)
        {
            const auto mapped = view.BoundaryFaces(iBnd);
            REQUIRE(mapped.size() == 1);
            REQUIRE(mapped[0] >= 0);
            REQUIRE(mapped[0] < view.NumFaceProc());
            // The fixture gives the cell/face to rank zero and boundaries
            // to another rank. Extraction must read this ghost face's
            // published orientation, not the boundary's input slot order.
            if (mpi.size > 1)
                CHECK(mapped[0] >= view.NumFace());
        }
    }
    const auto sourceNamespace = source->MeshNamespace();
    auto extracted = G::NativeMeshExtraction::ExtractBoundary(*source, mpi);
    CHECK(source->NumGlobal(G::EntityKind::Node) == (dimension == 2 ? 7 : 11));
    CHECK(extracted.state->NumOwned(G::EntityKind::Node) == (mpi.rank == mpi.size - 1 ? (dimension == 2 ? 5 : 9) : 0));
    source.reset();
    CheckExtracted(*extracted.state, sourceNamespace, dimension, faces);
    if (dimension == 2)
    {
        CHECK_THROWS(G::PrepareNativeMeshTopology(*extracted.state, mpi));
        CHECK_THROWS(G::NativeMeshExtraction::ExtractBoundary(*extracted.state, mpi));
    }
    G::ReorderInput reorder;
    for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd})
        reorder.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(extracted.state->NumOwned(kind)), (mpi.rank + 1) % mpi.size)});
    auto current = G::NativeMeshTransition::Build(*extracted.state, mpi, reorder).state;
    extracted = {};
    CheckExtracted(*current, sourceNamespace, dimension, faces);
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto file = (std::filesystem::temp_directory_path() / ("native-extraction-" + std::to_string(pid) + "-" + std::to_string(mpi.rank) + ".json")).string();
    auto serializer = std::make_shared<DNDS::Serializer::SerializerJSON>();
    serializer->OpenFile(file, false);
    G::WriteNativeMeshArchive(*current, mpi, serializer, "mesh");
    serializer->CloseFile();
    const auto name = current->MeshNamespace();
    current.reset();
    serializer->OpenFile(file, true);
    current = G::ReadNativeMeshArchive(mpi, serializer, "mesh");
    serializer->CloseFile();
    CHECK(current->MeshNamespace() == name);
    CheckExtracted(*current, sourceNamespace, dimension, faces);
    std::filesystem::remove(file);
}

TEST_CASE("Native boundary extraction handles an empty selection without retaining unused nodes")
{
    const auto mpi = MPI();
    auto source = Fixture(mpi, 2, true);
    auto result = G::NativeMeshExtraction::ExtractBoundary(*source, mpi);
    CHECK(result.state->NumGlobal(G::EntityKind::Node) == 0);
    CHECK(result.state->NumGlobal(G::EntityKind::Cell) == 0);
    CHECK(result.state->Provenance()->sourceIds.at(G::EntityKind::Node).RowCount() == 0);
    CHECK(source->NumGlobal(G::EntityKind::Node) == 7);
    auto invalidOptions = G::NativeMeshBuildOptions{};
    if (mpi.rank == 0)
        invalidOptions.cellLayers = 0;
    CHECK_THROWS(G::NativeMeshExtraction::ExtractBoundary(*source, mpi, invalidOptions));
    CHECK_NOTHROW(G::NativeMeshExtraction::ExtractBoundary(*source, mpi));
}

TEST_CASE("Native boundary extraction retains periodic tangential images and rejects inconsistent flags")
{
    const auto mpi = MPI();
    auto source = Fixture(mpi, 2);
    auto owned = G::NativeMeshAssembler::CaptureOwned(*source, mpi);
    owned.periodic = true;
    auto bits = [](const R::Array<I> &cones)
    {
        auto result = R::Array<G::NodePeriodicBits>(cones.Layout());
        auto write = result.Write();
        for (size_t row = 0; row < result.RowCount(); ++row)
            for (size_t slot = 0; slot < write.View().RowSize(row); ++slot)
                write.View()(row, slot) = G::NodePeriodicBits{0};
        return result;
    };
    owned.cellNodesPbi = bits(*owned.relations[size_t(G::MeshRelation::CellNodes)]);
    owned.bndNodesPbi = bits(*owned.relations[size_t(G::MeshRelation::BoundaryNodes)]);
    auto periodic = G::AssembleNativeMesh(mpi, std::move(owned));
    auto extracted = G::NativeMeshExtraction::ExtractBoundary(*periodic, mpi);
    CHECK(extracted.state->NumGlobal(G::EntityKind::Cell) == 2);
    CHECK(extracted.state->BorrowHostView().IsPeriodic());
    auto inconsistent = G::NativeMeshAssembler::CaptureOwned(*source, mpi);
    auto metadata = R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(source->NumOwned(G::EntityKind::Bnd)), 1));
    {
        auto read = inconsistent.bndInfo->Read();
        auto write = metadata.Write();
        for (size_t row = 0; row < metadata.RowCount(); ++row)
        {
            write.View()(row, 0) = read.View()(row, 0);
            if (row == 0)
                write.View()(row, 0).zone = G::BC_ID_PERIODIC_1;
        }
    }
    inconsistent.bndInfo = std::move(metadata);
    auto badZones = G::AssembleNativeMesh(mpi, std::move(inconsistent));
    CHECK_THROWS(G::NativeMeshExtraction::ExtractBoundary(*badZones, mpi));
    CHECK_NOTHROW(G::NativeMeshExtraction::ExtractBoundary(*source, mpi));
}

TEST_CASE("Native boundary facade retains source reads and publishes legacy parent caches explicitly")
{
    const auto mpi = MPI();
    auto source = G::UnstructuredMesh::FromNative(mpi, Fixture(mpi, 2));
    const auto state = source->CurrentNativeState();
    const auto name = state->MeshNamespace();
    auto read = source->AcquireRead();
    const auto sourceNodeCount = read.HostView().NumNodeProc();
    auto pure = source->ExtractBoundary();
    CHECK(source->node2bndNode.empty());
    CHECK(pure->node2parentNode.empty());
    CheckExtracted(*pure->CurrentNativeState(), name, 2, false);
    G::UnstructuredMesh wrong(mpi, 3), destination(mpi, 1);
    CHECK_THROWS(source->ConstructBndMesh(wrong));
    CHECK_THROWS(source->ConstructBndMesh(*source));
    auto populated = source->ExtractBoundary();
    const auto populatedState = populated->CurrentNativeState();
    std::optional<G::MeshReadAccess> populatedRead;
    if (mpi.rank == 0)
        populatedRead.emplace(populated->AcquireRead());
    const auto destinationStamp = populated->versions.Stamp();
    const auto sourceStamp = source->versions.Stamp();
    CHECK_THROWS(source->ConstructBndMesh(*populated));
    CHECK(destinationStamp.Valid());
    CHECK(sourceStamp.Valid());
    CHECK(populated->CurrentNativeState() == populatedState);
    populatedRead.reset();
    source->ConstructBndMesh(destination);
    CHECK(destination.HasNativeState());
    CHECK(source->CurrentNativeState() == state);
    CHECK(read.HostView().NumNodeProc() == sourceNodeCount);
    CHECK(source->node2bndNode.size() == size_t(sourceNodeCount));
    CHECK(destination.node2parentNode.size() == size_t(destination.NumNodeProc()));
    REQUIRE(destination.cell2parentCell.size() == size_t(source->NumBnd()));
    for (I row = 0; row < destination.NumCell(); ++row)
        CHECK(destination.cell2parentCell[size_t(row)] == row);
    for (I row = 0; row < destination.NumNodeProc(); ++row)
    {
        const auto parent = destination.node2parentNode[size_t(row)];
        REQUIRE(parent >= 0);
        REQUIRE(parent < sourceNodeCount);
        CHECK(source->node2bndNode[size_t(parent)] == row);
    }
    const auto firstState = destination.CurrentNativeState();
    const auto cache = source->node2bndNode;
    std::optional<G::MeshReadAccess> activeDestination;
    if (mpi.rank == 0)
        activeDestination.emplace(destination.AcquireRead());
    CHECK_THROWS(source->ConstructBndMesh(destination));
    CHECK(destination.CurrentNativeState() == firstState);
    CHECK(source->node2bndNode == cache);
    activeDestination.reset();
    source->ConstructBndMesh(destination);
    CHECK(destination.CurrentNativeState() != firstState);
    CHECK(read.HostView().NumNodeProc() == sourceNodeCount);
    CheckExtracted(*destination.CurrentNativeState(), name, 2, false);
}

TEST_CASE("Native boundary parent cache marks newly introduced ghosts absent from source closure")
{
    const auto mpi = MPI();
    if (mpi.size < 2)
        return;
    const bool active = mpi.rank < 2;
    auto nodeOwned = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == 0 ? 5 : 0);
    auto cells = nodeOwned.CreateSibling(G::EntityKind::Cell, active ? 1 : 0);
    auto boundaries = nodeOwned.CreateSibling(G::EntityKind::Bnd, active ? 1 : 0);
    auto nodes = nodeOwned.WithGhosts(mpi.rank == 1 ? std::vector<I>{1, 3, 4} : std::vector<I>{});
    G::NativeMeshData data;
    data.dimension = 2;
    auto pair = [](const std::vector<std::vector<I>> &rows)
    { return R::ArrayPair<I>{Rows(rows), Rows({})}; };
    data.relations[size_t(G::MeshRelation::CellNodes)] = pair(active ? std::vector<std::vector<I>>{{0, 1, 2}} : std::vector<std::vector<I>>{});
    data.relations[size_t(G::MeshRelation::BoundaryNodes)] = pair(active ? std::vector<std::vector<I>>{{0, 1}} : std::vector<std::vector<I>>{});
    data.relations[size_t(G::MeshRelation::CellCells)] = pair(active ? std::vector<std::vector<I>>{{}} : std::vector<std::vector<I>>{});
    data.relations[size_t(G::MeshRelation::BoundaryCells)] = pair(active ? std::vector<std::vector<I>>{{0, DNDS::UnInitIndex}} : std::vector<std::vector<I>>{});
    data.coords = R::ArrayPair<DNDS::real>{R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)), R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumGhost()), 3))};
    for (auto *part : {&data.coords->owned, &data.coords->ghost})
    {
        auto write = part->Write();
        const I start = part == &data.coords->owned ? 0 : nodes.NumOwned();
        for (size_t row = 0; row < part->RowCount(); ++row)
            for (size_t component = 0; component < 3; ++component)
                write.View()(row, component) = component == 0 ? DNDS::real(nodes.LocalToGlobal(start + I(row))) : 0.;
    }
    auto info = [&](G::Elem::ElemType type, G::t_index zone)
    {
        R::ArrayPair<G::ElemInfo> result{R::Array<G::ElemInfo>(R::RowLayout::Dense(active ? 1 : 0, 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
        if (active)
        {
            auto write = result.owned.Write();
            write.View()(0, 0) = G::ElemInfo{type, zone};
        }
        return result;
    };
    data.cellInfo = info(G::Elem::Tri3, G::INTERNAL_ZONE);
    data.bndInfo = info(G::Elem::Line2, 12);
    std::map<G::EntityKind, R::Array<I>> ids;
    for (const auto &[kind, space] : std::map<G::EntityKind, G::EntitySpace>{{G::EntityKind::Node, nodes}, {G::EntityKind::Cell, cells}, {G::EntityKind::Bnd, boundaries}})
    {
        std::vector<std::vector<I>> rows;
        for (I row = 0; row < space.NumLocal(); ++row)
            rows.push_back({1000 * (1 + I(kind)) + space.LocalToGlobal(row)});
        // Public FromNative requires a one-column table even on empty ranks.
        R::Array<I> values(R::RowLayout::Dense(rows.size(), 1));
        {
            auto write = values.Write();
            for (size_t row = 0; row < rows.size(); ++row)
                write.View()(row, 0) = rows[row][0];
        }
        ids.emplace(kind, std::move(values));
    }
    auto state = G::NativeMeshState::FromNative(mpi, std::move(data), {{G::EntityKind::Node, nodes}, {G::EntityKind::Cell, cells}, {G::EntityKind::Bnd, boundaries}}, std::move(ids), G::MeshReadRequirements::Primary());
    auto source = G::UnstructuredMesh::FromNative(mpi, std::move(state));
    G::UnstructuredMesh destination(mpi, 1);
    source->ConstructBndMesh(destination);
    if (mpi.rank == 1)
    {
        CHECK(destination.NumNodeGhost() == 3);
        REQUIRE(destination.node2parentNode.size() == 3);
        CHECK(destination.node2parentNode[0] == DNDS::UnInitIndex);
        CHECK(destination.node2parentNode[1] == 0);
        CHECK(destination.node2parentNode[2] == 1);
    }
}

TEST_CASE("Boundary facade rejects absent native authority collectively before mutating either mesh")
{
    const auto mpi = MPI();
    auto native = G::UnstructuredMesh::FromNative(mpi, Fixture(mpi, 2));
    auto empty = std::make_shared<G::UnstructuredMesh>(mpi, 2);
    G::UnstructuredMesh destination(mpi, 1);
    const auto sourceStamp = empty->versions.Stamp(), destinationStamp = destination.versions.Stamp();
    CHECK_THROWS(empty->ConstructBndMesh(destination));
    CHECK(sourceStamp.Valid());
    CHECK(destinationStamp.Valid());
    CHECK_FALSE(destination.HasNativeState());
    CHECK(empty->node2bndNode.empty());
    CHECK(destination.node2parentNode.empty());
    if (mpi.size > 1)
    {
        auto source = mpi.rank == 0 ? empty : native;
        CHECK_THROWS(source->ConstructBndMesh(destination));
        CHECK(sourceStamp.Valid());
        CHECK(destinationStamp.Valid());
        CHECK_FALSE(destination.HasNativeState());
        CHECK(source->node2bndNode.empty());
    }
    native->ConstructBndMesh(destination);
    CheckExtracted(*destination.CurrentNativeState(), native->CurrentNativeState()->MeshNamespace(), 2, false);
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
