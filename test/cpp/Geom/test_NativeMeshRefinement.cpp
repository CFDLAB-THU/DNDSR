#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Refinement/NativeMeshRefinement.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
#include "Geom/IO/NativeMeshArchive.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include <filesystem>
#include <chrono>
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
    std::shared_ptr<const G::NativeMeshState> Fixture(const DNDS::MPIInfo &mpi, const std::vector<G::Elem::ElemType> &types, bool withBoundary = false, bool emptyPeers = false, bool sparseIds = false)
    {
        const auto localTypes = emptyPeers && mpi.rank != 0 ? std::vector<G::Elem::ElemType>{} : types;
        I count = 0;
        for (auto type : localTypes)
            count += G::Elem::Element{type}.GetNumNodes();
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, count);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(localTypes.size()));
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, withBoundary ? I(localTypes.size()) : 0);
        R::Array<DNDS::real> coordinates(R::RowLayout::Dense(size_t(count), 3));
        R::Array<G::ElemInfo> info(R::RowLayout::Dense(localTypes.size(), 1)), boundaryInfo(R::RowLayout::Dense(withBoundary ? localTypes.size() : 0, 1));
        std::vector<std::vector<I>> cones;
        {
            auto coords = coordinates.Write();
            auto metadata = info.Write();
            I offset = 0;
            for (size_t cell = 0; cell < localTypes.size(); ++cell)
            {
                cones.emplace_back();
                G::Elem::DispatchElementType(localTypes[cell], [&](auto traits)
                                             {
                    using T = decltype(traits);
                    for (int node = 0; node < T::numNodes; ++node)
                    {
                        cones.back().push_back(nodes.LocalToGlobal(offset + node));
                        for (size_t component = 0; component < 3; ++component)
                            coords.View()(size_t(offset + node), component) = T::standardCoords[size_t(3 * node) + component] +
                                (component == 0 ? 5 * (mpi.rank + int(cell) * mpi.size) : 0);
                    }
                    offset += T::numNodes; });
                metadata.View()(cell, 0) = G::ElemInfo{localTypes[cell], G::t_index(100 + cell)};
            }
        }
        std::vector<std::vector<I>> boundaryCones;
        if (withBoundary)
        {
            auto write = boundaryInfo.Write();
            for (size_t cell = 0; cell < localTypes.size(); ++cell)
            {
                auto element = G::Elem::Element{localTypes[cell]};
                const auto face = element.ObtainFace(0);
                std::vector<I> nodes(size_t(face.GetNumNodes()));
                element.ExtractFaceNodes(0, cones[cell], nodes);
                boundaryCones.push_back(std::move(nodes));
                write.View()(cell, 0) = G::ElemInfo{face.type, G::BC_ID_DEFAULT_WALL};
            }
        }
        const int dimension = G::Elem::Element{types.front()}.GetDim();
        G::NativeMeshInput input{mpi, dimension, cells, nodes, boundaries, Rows(cones), Rows(boundaryCones), coordinates, info, boundaryInfo, {}};
        if (sparseIds)
            for (const auto &space : {nodes, cells, boundaries})
            {
                R::Array<I> ids(R::RowLayout::Dense(size_t(space.NumOwned()), 1));
                {
                    auto write = ids.Write();
                    for (I row = 0; row < space.NumOwned(); ++row)
                        write.View()(size_t(row), 0) = 1000000000000LL + 17 * space.LocalToGlobal(row) + int(space.Kind());
                }
                input.ids.emplace(space.Kind(), std::move(ids));
            }
        return G::ConstructNativeMesh(std::move(input), {1, true});
    }
    double Measure(const G::UnstructuredMeshView &view, I cell)
    {
        const auto row = view.CellNodes(cell);
        const auto type = view.GetCellInfo(cell).getElemType();
        const G::tPoint a = view.Coord(row[0]), b = view.Coord(row[1]), c = view.Coord(row[2]);
        if (type == G::Elem::Tet4)
        {
            const G::tPoint d = view.Coord(row[3]);
            return (b - a).dot((c - a).cross(d - a)) / 6;
        }
        if (type == G::Elem::Tri3)
            return (b - a).cross(c - a)[2] / 2;
        REQUIRE(type == G::Elem::Quad4);
        const G::tPoint d = view.Coord(row[3]);
        return ((b - a).cross(c - a)[2] + (c - a).cross(d - a)[2]) / 2;
    }
}

TEST_CASE("Owned elevation retains legacy original labels independently of current global row numbering")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (const bool sparse : {false, true})
    {
        auto source = Fixture(mpi, {G::Elem::Quad4}, true, false, sparse);
        auto owned = G::NativeMeshAssembler::CaptureOwned(*source, mpi);
        auto elevated = G::NativeMeshRefinement::ElevateOwned(owned, mpi, {1, true}, true);
        const auto first = source->IdHighWater().at(G::EntityKind::Node);
        const I additions = elevated.owned.spaces.at(G::EntityKind::Node).NumOwned() - source->NumOwned(G::EntityKind::Node);
        I offset = 0;
        MPI_Exscan(&additions, &offset, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        if (mpi.rank == 0)
            offset = 0;
        for (const auto kind : {G::EntityKind::Cell, G::EntityKind::Bnd, G::EntityKind::Node})
        {
            CHECK(elevated.owned.spaces.at(kind).NumGhost() == 0);
            auto actual = elevated.owned.ids.at(kind).Read();
            for (I row = 0; row < source->NumOwned(kind); ++row)
                CHECK(actual.View()(size_t(row), 0) == source->OriginalEntityId(kind, row));
            if (kind == G::EntityKind::Node)
                for (I row = 0; row < additions; ++row)
                    CHECK(actual.View()(size_t(source->NumOwned(kind) + row), 0) == first + offset + row);
        }
        CHECK(elevated.owned.identity.meshNamespace != source->MeshNamespace());
        auto bisected = G::NativeMeshRefinement::BisectOwned(elevated.owned, mpi, {1, true});
        for (const auto &[kind, space] : bisected.owned.spaces)
            CHECK(space.NumGhost() == 0);
        const auto recipe = G::DefaultNativeMeshGhostSpec(bisected.owned, 1, true);
        auto final = G::NativeMeshAssembler::Build(mpi, std::move(bisected.owned), recipe);
        CHECK(final->NumOwned(G::EntityKind::Cell) == 4);
        CHECK(final->NumOwned(G::EntityKind::Bnd) == 2);
    }
}

TEST_CASE("Native O2 bisection preserves independent exact child cones volume and durable parent IDs")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    std::vector<G::Elem::ElemType> types;
    SUBCASE("triangle") { types = {G::Elem::Tri6}; }
    SUBCASE("quadrilateral") { types = {G::Elem::Quad9}; }
    SUBCASE("mixed two dimensional") { types = {G::Elem::Tri6, G::Elem::Quad9}; }
    SUBCASE("tetrahedron") { types = {G::Elem::Tet10}; }
    auto source = Fixture(mpi, types);
    const auto sourceNamespace = source->MeshNamespace();
    const auto sourceNodes = source->Entities(G::EntityKind::Node);
    auto result = G::NativeMeshRefinement::Bisect(*source, mpi, {1, true});
    REQUIRE(result.state);
    {
        auto o2 = G::UnstructuredMesh::FromNative(mpi, source);
        CHECK_FALSE(o2->IsO1());
        CHECK(o2->IsO2());
        auto o1 = G::UnstructuredMesh::FromNative(mpi, result.state);
        CHECK(o1->IsO1());
        CHECK_FALSE(o1->IsO2());
    }
    CHECK(result.state->MeshNamespace() != sourceNamespace);
    CHECK_FALSE(result.state->Entities(G::EntityKind::Node).SameDomain(sourceNodes));
    CHECK(result.provenance.sourceNamespace == sourceNamespace);
    REQUIRE(result.state->Provenance());
    CHECK(result.state->Provenance()->sourceNamespace == sourceNamespace);
    for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd})
        CHECK(result.state->Provenance()->sourceKinds.at(kind) == kind);
    auto parentRead = result.provenance.sourceIds.at(G::EntityKind::Cell).Read();
    auto nodeParents = result.provenance.sourceIds.at(G::EntityKind::Node).Read();
    auto retained = result.state->Snapshot(R::Device::Host()).AcquireRead();
    auto view = retained.HostView();
    I child = 0, nodeOffset = 0;
    const I firstNode = sourceNodes.LocalToGlobal(0);
    for (size_t parent = 0; parent < types.size(); ++parent)
    {
        std::vector<std::vector<I>> expected;
        double expectedMeasure = 0;
        if (types[parent] == G::Elem::Tri6)
        {
            expected = {{0, 3, 5}, {3, 1, 4}, {5, 3, 4}, {5, 4, 2}};
            expectedMeasure = 0.5;
        }
        if (types[parent] == G::Elem::Quad9)
        {
            expected = {{0, 4, 8, 7}, {4, 1, 5, 8}, {7, 8, 6, 3}, {8, 5, 2, 6}};
            expectedMeasure = 4;
        }
        if (types[parent] == G::Elem::Tet10)
        {
            expected = {{0, 4, 6, 7}, {4, 1, 5, 8}, {6, 5, 2, 9}, {9, 7, 8, 3}, {4, 9, 6, 7}, {4, 8, 9, 7}, {4, 9, 8, 5}, {4, 6, 9, 5}};
            expectedMeasure = 1.0 / 6;
        }
        double total = 0;
        for (const auto &row : expected)
        {
            CHECK(parentRead.View()(size_t(child), 0) == I(mpi.rank * types.size() + parent));
            CHECK(view.GetCellInfo(child).zone == I(100 + parent));
            REQUIRE(view.CellNodes(child).size() == row.size());
            for (size_t slot = 0; slot < row.size(); ++slot)
                CHECK(nodeParents.View()(size_t(view.CellNodes(child)[slot]), 0) == firstNode + nodeOffset + row[slot]);
            const auto measure = Measure(view, child);
            CHECK(measure > 0);
            total += measure;
            ++child;
        }
        CHECK(total == doctest::Approx(expectedMeasure));
        nodeOffset += G::Elem::Element{types[parent]}.GetNumNodes();
    }
    CHECK(view.NumCell() == child);
    source.reset();
    CHECK(result.provenance.sourceNamespace == sourceNamespace);
    CHECK(parentRead.View().RowCount() == size_t(child));
}

TEST_CASE("Native bisection retains boundary labels ancestry and empty-rank participation")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto source = Fixture(mpi, {G::Elem::Tri6}, true, true);
    auto result = G::NativeMeshRefinement::Bisect(*source, mpi, {1, true});
    CHECK(result.state->NumOwned(G::EntityKind::Cell) == (mpi.rank == 0 ? 4 : 0));
    CHECK(result.state->NumOwned(G::EntityKind::Bnd) == (mpi.rank == 0 ? 2 : 0));
    auto read = result.state->Snapshot(R::Device::Host()).AcquireRead();
    auto view = read.HostView();
    if (mpi.rank == 0)
    {
        const std::vector<std::vector<I>> expected{{0, 3}, {3, 1}};
        auto nodeIds = result.provenance.sourceIds.at(G::EntityKind::Node).Read();
        auto parents = result.provenance.sourceIds.at(G::EntityKind::Bnd).Read();
        for (I boundary = 0; boundary < 2; ++boundary)
        {
            CHECK(parents.View()(size_t(boundary), 0) == 0);
            CHECK(view.GetBndInfo(boundary).zone == G::BC_ID_DEFAULT_WALL);
            REQUIRE(view.BoundaryNodes(boundary).size() == 2);
            for (size_t slot = 0; slot < 2; ++slot)
                CHECK(nodeIds.View()(size_t(view.BoundaryNodes(boundary)[slot]), 0) == expected[size_t(boundary)][slot]);
        }
    }
}

TEST_CASE("Native bisection rejects first-order input collectively without changing source")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto source = Fixture(mpi, {G::Elem::Tri3});
    const auto name = source->MeshNamespace();
    CHECK_THROWS(G::NativeMeshRefinement::Bisect(*source, mpi));
    CHECK(source->MeshNamespace() == name);
    CHECK(source->NumOwned(G::EntityKind::Cell) == 1);
}

TEST_CASE("Bisection sparse persistent ancestry survives source destruction redistribution and native restart")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto source = Fixture(mpi, {G::Elem::Tri6}, true, false, true);
    const auto sourceNamespace = source->MeshNamespace();
    auto derived = G::NativeMeshRefinement::Bisect(*source, mpi);
    std::map<G::EntityKind, std::map<I, I>> expected;
    G::ReorderInput input;
    for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd})
    {
        auto read = derived.state->Provenance()->sourceIds.at(kind).Read();
        std::vector<I> local;
        for (I row = 0; row < derived.state->NumOwned(kind); ++row)
        {
            local.push_back(derived.state->OriginalEntityId(kind, row));
            local.push_back(read.View()(size_t(row), 0));
        }
        std::vector<I> gathered(local.size() * size_t(mpi.size));
        MPI_Allgather(local.data(), int(local.size()), MPI_INT64_T, gathered.data(), int(local.size()), MPI_INT64_T, mpi.comm);
        for (size_t i = 0; i < gathered.size(); i += 2)
            expected[kind].emplace(gathered[i], gathered[i + 1]);
        input.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(derived.state->NumOwned(kind)), (mpi.rank + 1) % mpi.size)});
    }
    source.reset();
    auto moved = G::NativeMeshTransition::Build(*derived.state, mpi, input);
    auto verify = [&](const auto &state)
    {
        REQUIRE(state->Provenance());
        CHECK(state->Provenance()->sourceNamespace == sourceNamespace);
        for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd})
        {
            CHECK(state->Provenance()->sourceKinds.at(kind) == kind);
            auto read = state->Provenance()->sourceIds.at(kind).Read();
            CHECK(read.View().RowCount() == size_t(state->NumOwned(kind)));
            for (I row = 0; row < state->NumOwned(kind); ++row)
            {
                const auto id = state->OriginalEntityId(kind, row);
                CHECK(read.View()(size_t(row), 0) == expected.at(kind).at(id));
                CHECK(read.View()(size_t(row), 0) >= 1000000000000LL);
            }
        }
    };
    verify(moved.state);
    auto json = std::make_shared<DNDS::Serializer::SerializerJSON>();
    json->CreatePath("/");
    G::WriteNativeMeshArchive(*moved.state, mpi, json, "refined");
    verify(G::ReadNativeMeshArchive(mpi, json, "refined"));
    I stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    MPI_Bcast(&stamp, 1, MPI_INT64_T, 0, mpi.comm);
    const auto file = (std::filesystem::temp_directory_path() / ("native_refinement_" + std::to_string(stamp) + ".h5")).string();
    auto h5 = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
    h5->SetCollectiveRW(true, true);
    h5->OpenFile(file, false);
    G::WriteNativeMeshArchive(*moved.state, mpi, h5, "refined");
    h5->CloseFile();
    h5->OpenFile(file, true);
    verify(G::ReadNativeMeshArchive(mpi, h5, "refined", DNDS::Serializer::ArrayGlobalOffset_EvenSplit));
    h5->CloseFile();
    h5.reset();
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

TEST_CASE("Native bisection ancestry rejects malformed rank-local publication input")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto source = Fixture(mpi, {G::Elem::Tri6});
    auto derived = G::NativeMeshRefinement::Bisect(*source, mpi);
    auto owned = G::NativeMeshAssembler::CaptureOwned(*derived.state, mpi);
    SUBCASE("wrong row count")
    {
        if (mpi.rank == 0)
            owned.provenance->sourceIds.at(G::EntityKind::Cell) = R::Array<I>(R::RowLayout::Dense(5, 1));
    }
    SUBCASE("invalid source namespace")
    {
        if (mpi.rank == 0)
            owned.provenance->sourceNamespace = "invalid";
    }
    SUBCASE("negative source ID")
    {
        if (mpi.rank == 0)
        {
            R::Array<I> values(R::RowLayout::Dense(4, 1));
            {
                auto write = values.Write();
                for (size_t row = 0; row < 4; ++row)
                    write.View()(row, 0) = -1;
            }
            owned.provenance->sourceIds.at(G::EntityKind::Cell) = std::move(values);
        }
    }
    SUBCASE("missing source kind")
    {
        if (mpi.rank == 0)
            owned.provenance->sourceKinds.erase(G::EntityKind::Cell);
    }
    SUBCASE("wrong source kind key")
    {
        if (mpi.rank == 0)
        {
            owned.provenance->sourceKinds.erase(G::EntityKind::Cell);
            owned.provenance->sourceKinds.emplace(G::EntityKind::Face, G::EntityKind::Cell);
        }
    }
    SUBCASE("different valid source kind across ranks")
    {
        if (mpi.size == 1)
            return;
        if (mpi.rank == 0)
            owned.provenance->sourceKinds.at(G::EntityKind::Cell) = G::EntityKind::Bnd;
    }
    CHECK_THROWS(G::AssembleNativeMesh(mpi, std::move(owned)));
    CHECK(derived.state->NumOwned(G::EntityKind::Cell) == 4);
}

TEST_CASE("CSR zero and multiple source ancestry retains aliases and roundtrips redistribution with empty ranks")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto source = Fixture(mpi, {G::Elem::Tri6}, false, true, true);
    auto derived = G::NativeMeshRefinement::Bisect(*source, mpi);
    auto captured = G::NativeMeshAssembler::CaptureOwned(*derived.state, mpi);
    const std::vector<std::vector<I>> parentRows{{}, {9000000000001LL, 9000000000001LL, 9000000000007LL}, {9000000000013LL}, {}};
    auto ancestry = Rows(mpi.rank == 0 ? parentRows : std::vector<std::vector<I>>{});
    captured.provenance->sourceIds.at(G::EntityKind::Cell) = ancestry;
    captured.provenance->sourceKinds.at(G::EntityKind::Bnd) = G::EntityKind::Cell;
    // Exercise valid rank-local storage policy variation too: semantic ancestry
    // always archives ragged rows even when some families happen to be dense.
    if (mpi.rank == 0)
    {
        auto original = captured.provenance->sourceIds.at(G::EntityKind::Node);
        auto read = original.Read();
        std::vector<std::vector<I>> rows(original.RowCount());
        for (size_t i = 0; i < rows.size(); ++i)
            rows[i] = {read.View()(i, 0)};
        captured.provenance->sourceIds.at(G::EntityKind::Node) = Rows(rows);
    }
    auto state = G::AssembleNativeMesh(mpi, std::move(captured));
    auto retained = G::NativeMeshAssembler::CaptureOwned(*state, mpi);
    source.reset();
    derived.state.reset();
    state.reset();
    CHECK_THROWS(ancestry.Write());
    state = G::AssembleNativeMesh(mpi, std::move(retained));
    G::ReorderInput reorder;
    for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd})
        reorder.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(state->NumOwned(kind)), mpi.size - 1)});
    auto moved = G::NativeMeshTransition::Build(*state, mpi, reorder);
    auto verify = [&](const auto &publication)
    {
        REQUIRE(publication->Provenance());
        CHECK(publication->Provenance()->sourceKinds.at(G::EntityKind::Bnd) == G::EntityKind::Cell);
        auto read = publication->Provenance()->sourceIds.at(G::EntityKind::Cell).Read();
        for (I row = 0; row < publication->NumOwned(G::EntityKind::Cell); ++row)
        {
            const auto id = publication->OriginalEntityId(G::EntityKind::Cell, row);
            const auto expected = parentRows.at(size_t(id));
            std::vector<I> actual;
            for (auto value : read.View()[size_t(row)])
                actual.push_back(value);
            CHECK(actual == expected);
        }
    };
    verify(moved.state);
    auto json = std::make_shared<DNDS::Serializer::SerializerJSON>();
    json->CreatePath("/");
    G::WriteNativeMeshArchive(*moved.state, mpi, json, "csr");
    verify(G::ReadNativeMeshArchive(mpi, json, "csr"));
    I stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    MPI_Bcast(&stamp, 1, MPI_INT64_T, 0, mpi.comm);
    const auto file = (std::filesystem::temp_directory_path() / ("native_ancestry_csr_" + std::to_string(stamp) + ".h5")).string();
    auto h5 = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
    h5->SetCollectiveRW(true, true);
    h5->OpenFile(file, false);
    G::WriteNativeMeshArchive(*moved.state, mpi, h5, "csr");
    h5->CloseFile();
    h5->OpenFile(file, true);
    verify(G::ReadNativeMeshArchive(mpi, h5, "csr", DNDS::Serializer::ArrayGlobalOffset_EvenSplit));
    h5->CloseFile();
    h5.reset();
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

TEST_CASE("Native elevation deduplicates shared quadratic spans and preserves exact slots across MPI owners")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    int dimension = 2;
    SUBCASE("shared quadrilaterals") { dimension = 2; }
    SUBCASE("shared tetrahedra") { dimension = 3; }
    const I nodeCount = mpi.rank == 0 ? (dimension == 2 ? 6 : 5) : 0;
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, nodeCount);
    std::vector<std::vector<I>> cones, boundaries;
    if (mpi.rank == 0)
        cones.push_back(dimension == 2 ? std::vector<I>{0, 1, 4, 3} : std::vector<I>{0, 1, 2, 3});
    if (mpi.rank == (mpi.size > 1 ? 1 : 0))
        cones.push_back(dimension == 2 ? std::vector<I>{1, 2, 5, 4} : std::vector<I>{0, 2, 1, 4});
    if (dimension == 2 && mpi.rank == mpi.size - 1)
        boundaries = {{0, 1}, {1, 2}, {2, 5}, {5, 4}, {4, 3}, {3, 0}};
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(cones.size()));
    auto bnds = nodes.CreateSibling(G::EntityKind::Bnd, I(boundaries.size()));
    R::Array<DNDS::real> coordinates(R::RowLayout::Dense(size_t(nodeCount), 3));
    const std::vector<G::tPoint> points = dimension == 2 ? std::vector<G::tPoint>{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}} : std::vector<G::tPoint>{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1}};
    {
        auto write = coordinates.Write();
        for (I row = 0; row < nodeCount; ++row)
            for (size_t j = 0; j < 3; ++j)
                write.View()(size_t(row), j) = points[size_t(row)][j];
    }
    R::Array<G::ElemInfo> metadata(R::RowLayout::Dense(cones.size(), 1)), boundaryMetadata(R::RowLayout::Dense(boundaries.size(), 1));
    {
        auto write = metadata.Write();
        for (size_t row = 0; row < cones.size(); ++row)
            write.View()(row, 0) = G::ElemInfo{dimension == 2 ? G::Elem::Quad4 : G::Elem::Tet4, G::INTERNAL_ZONE};
    }
    {
        auto write = boundaryMetadata.Write();
        for (size_t row = 0; row < boundaries.size(); ++row)
            write.View()(row, 0) = G::ElemInfo{G::Elem::Line2, G::t_index(2 + row)};
    }
    G::NativeMeshInput input{mpi, dimension, cells, nodes, bnds, Rows(cones), Rows(boundaries), coordinates, metadata, boundaryMetadata, {}};
    for (const auto &space : {nodes, cells, bnds})
    {
        R::Array<I> ids(R::RowLayout::Dense(size_t(space.NumOwned()), 1));
        {
            auto write = ids.Write();
            for (I row = 0; row < space.NumOwned(); ++row)
                write.View()(size_t(row), 0) = 1000000000000LL + 17 * space.LocalToGlobal(row) + int(space.Kind());
        }
        input.ids.emplace(space.Kind(), std::move(ids));
    }
    auto source = G::ConstructNativeMesh(std::move(input), {1, true});
    auto elevated = G::NativeMeshRefinement::Elevate(*source, mpi, {1, true});
    {
        auto o1 = G::UnstructuredMesh::FromNative(mpi, source);
        CHECK(o1->IsO1());
        CHECK_FALSE(o1->IsO2());
        auto o2 = G::UnstructuredMesh::FromNative(mpi, elevated.state);
        CHECK_FALSE(o2->IsO1());
        CHECK(o2->IsO2());
    }
    CHECK(elevated.state->NumGlobal(G::EntityKind::Node) == (dimension == 2 ? 15 : 14));
    CHECK(elevated.state->MeshNamespace() != source->MeshNamespace());
    REQUIRE(elevated.state->Provenance());
    auto ancestry = elevated.state->Provenance()->sourceIds.at(G::EntityKind::Node).Read();
    auto access = elevated.state->Snapshot(R::Device::Host()).AcquireRead();
    auto view = access.HostView();
    for (I row = 0; row < elevated.state->NumOwned(G::EntityKind::Node); ++row)
    {
        G::tPoint mean = G::tPoint::Zero();
        const auto parents = ancestry.View()[size_t(row)];
        REQUIRE(!parents.empty());
        CHECK((parents.size() == 1 || parents.size() == 2 || (dimension == 2 && parents.size() == 4)));
        for (auto id : parents)
        {
            const I index = (id - 1000000000000LL - int(G::EntityKind::Node)) / 17;
            REQUIRE(index >= 0);
            REQUIRE(size_t(index) < points.size());
            mean += points[size_t(index)];
        }
        mean /= double(parents.size());
        CHECK(view.Coord(row).isApprox(mean));
    }
    const std::vector<std::pair<int, int>> spans = dimension == 2 ? std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {2, 3}, {3, 0}} : std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};
    for (I row = 0; row < view.NumCell(); ++row)
    {
        CHECK(view.GetCellInfo(row).getElemType() == (dimension == 2 ? G::Elem::Quad9 : G::Elem::Tet10));
        const auto output = view.CellNodes(row);
        const auto &original = cones[size_t(row)];
        for (size_t slot = 0; slot < original.size(); ++slot)
            CHECK(view.Coord(output[slot]).isApprox(points[size_t(original[slot])]));
        for (size_t span = 0; span < spans.size(); ++span)
        {
            const auto [a, b] = spans[span];
            CHECK(view.Coord(output[original.size() + span]).isApprox((points[size_t(original[size_t(a)])] + points[size_t(original[size_t(b)])]) / 2));
        }
        if (dimension == 2)
            CHECK(view.Coord(output[8]).isApprox((points[size_t(original[0])] + points[size_t(original[1])] + points[size_t(original[2])] + points[size_t(original[3])]) / 4));
    }
    for (I row = 0; row < view.NumBnd(); ++row)
    {
        CHECK(view.GetBndInfo(row).zone == G::t_index(2 + row));
        const auto output = view.BoundaryNodes(row);
        const auto &original = boundaries[size_t(row)];
        CHECK(view.Coord(output[0]).isApprox(points[size_t(original[0])]));
        CHECK(view.Coord(output[1]).isApprox(points[size_t(original[1])]));
        CHECK(view.Coord(output[2]).isApprox((points[size_t(original[0])] + points[size_t(original[1])]) / 2));
    }
    auto bisected = G::NativeMeshRefinement::Bisect(*elevated.state, mpi, {1, true});
    {
        auto o1 = G::UnstructuredMesh::FromNative(mpi, bisected.state);
        CHECK(o1->IsO1());
        CHECK_FALSE(o1->IsO2());
    }
    auto topology = G::PrepareNativeMeshTopology(*bisected.state, mpi, true, dimension == 3);
    CHECK(topology->NumGlobal(G::EntityKind::Cell) == (dimension == 2 ? 8 : 16));
    auto topologyRead = topology->Snapshot(R::Device::Host()).AcquireRead();
    double volume = 0;
    for (I cell = 0; cell < topology->NumOwned(G::EntityKind::Cell); ++cell)
        volume += Measure(topologyRead.HostView(), cell);
    MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(volume == doctest::Approx(dimension == 2 ? 2.0 : 1.0 / 3));
    source.reset();
    auto serializer = std::make_shared<DNDS::Serializer::SerializerJSON>();
    serializer->CreatePath("/");
    G::WriteNativeMeshArchive(*topology, mpi, serializer, "elevated");
    auto restarted = G::ReadNativeMeshArchive(mpi, serializer, "elevated");
    CHECK(restarted->NumGlobal(G::EntityKind::Node) == (dimension == 2 ? 15 : 14));
    REQUIRE(restarted->Provenance());
    CHECK(restarted->Provenance()->sourceNamespace == elevated.state->MeshNamespace());
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
