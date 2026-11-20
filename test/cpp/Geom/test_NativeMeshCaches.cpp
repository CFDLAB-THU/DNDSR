#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "LegacyMeshFixtureCodec.hpp"
#include "Geom/Unstructured/NativeMeshBuilder.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
#include "Geom/Transition/MeshTransition.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include "Solver/Direct.hpp"
#include "Geom/Transition/SerialAdjReordering.hpp"
#include "DNDS/Legacy/ArrayDerived/ArrayEigenVector.hpp"
#include "DNDS/Legacy/ArrayPair.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    std::shared_ptr<const G::NativeMeshState> Strip(const DNDS::MPIInfo &mpi, bool emptyPeers = false)
    {
        const bool active = !emptyPeers || mpi.rank == 0;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, active ? 8 : 0);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, active ? 3 : 0);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
        R::RowBuilder cones(size_t(cells.NumOwned()), sizeof(I));
        const I first = emptyPeers ? 0 : 8 * ((mpi.rank + 1) % mpi.size);
        for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
        {
            cones.ResizeRow(row, 4);
            const I values[]{first + I(row), first + I(row) + 1, first + I(row) + 5, first + I(row) + 4};
            std::copy(std::begin(values), std::end(values), cones.Row<I>(row).data);
        }
        R::Array<DNDS::real> coordinates(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3, 5));
        R::Array<G::ElemInfo> metadata(R::RowLayout::Dense(size_t(cells.NumOwned()), 1));
        {
            auto points = coordinates.Write();
            for (size_t row = 0; row < size_t(nodes.NumOwned()); ++row)
            {
                points.View()(row, 0) = DNDS::real(row % 4) + 5 * mpi.rank + DNDS::real(row) / 32;
                points.View()(row, 1) = DNDS::real(row / 4) + DNDS::real(row % 4) / 16;
                points.View()(row, 2) = 0.125 * mpi.rank + DNDS::real(row + 1) / 256;
                points.View().data[row * 5 + 3] = 1234;
                points.View().data[row * 5 + 4] = -5678;
            }
            auto info = metadata.Write();
            for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
                info.View()(row, 0) = G::ElemInfo{G::Elem::Quad4, 0};
        }
        G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, R::Array<I>::FromSharedStorage(cones.Commit()),
                                 R::Array<I>(), coordinates, metadata, R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
        return G::ConstructNativeMesh(std::move(input), {1, true});
    }

    // Independent numerical inputs: the graph below is hand-authored from
    // strip element faces. Only generic graph algorithms enter this oracle.
    std::pair<std::vector<I>, std::vector<I>> GenericOrdering(const G::tLocalMatStruct &graph, int code)
    {
        if (graph.empty() || code == 0)
            return {};
        if (code == -1)
        {
            std::vector<I> order, inverse(graph.size());
            for (size_t row = 0; row < graph.size(); row += 2)
                order.push_back(I(row));
            for (size_t row = 1; row < graph.size(); row += 2)
                order.push_back(I(row));
            for (size_t row = 0; row < order.size(); ++row)
                inverse[size_t(order[row])] = I(row);
            return {order, inverse};
        }
        if (code == 1)
            return G::ReorderSerialAdj_Metis(graph);
        if (code == 2)
            return G::ReorderSerialAdj_BoostMMD(graph);
        I oldWidth = 0, newWidth = 0;
        if (code == 3)
            return G::ReorderSerialAdj_BoostRCM(graph, oldWidth, newWidth);
        return G::ReorderSerialAdj_CorrectRCM(graph.begin(), graph.end(), oldWidth, newWidth);
    }

    void CheckHash(const std::shared_ptr<const G::NativeMeshState> &state, const DNDS::MPIInfo &mpi)
    {
        auto native = G::UnstructuredMesh::FromNative(mpi, state);
        DNDS::ArrayPair<DNDS::ArrayEigenVector<3>> reference;
        reference.InitPair("coordinate hash reference", mpi);
        auto read = native->AcquireRead();
        const auto view = read.HostView();
        reference.father->Resize(view.NumNode());
        reference.son->Resize(view.NumNodeGhost());
        for (I row = 0; row < view.NumNodeProc(); ++row)
            reference[row] = view.Coord(row);
        const auto expected = reference.hash();
        CHECK(native->CoordinateHash() == expected);
        auto retained = native->AcquireRead();
        CHECK(native->CoordinateHash() == expected);
    }
}

TEST_CASE("Coordinate compatibility hash matches legacy owned and ghost rows before and after redistribution")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto state = Strip(mpi);
    CheckHash(state, mpi);
    G::ReorderInput input;
    input.explicitMaps = {{G::EntityKind::Cell, std::vector<DNDS::MPI_int>(3, (mpi.rank + 1) % mpi.size)},
                          {G::EntityKind::Node, std::vector<DNDS::MPI_int>(8, (mpi.rank + 2) % mpi.size)}};
    auto moved = G::NativeMeshTransition::Build(*state, mpi, input);
    CheckHash(moved.state, mpi);
    CheckHash(Strip(mpi, true), mpi);
}

TEST_CASE("Native face graph and direct ordering match exact face slots and generic serial algorithms")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto primary = Strip(mpi);
    auto missingFaces = G::UnstructuredMesh::FromNative(mpi, primary);
    CHECK_THROWS(missingFaces->GetCell2CellFaceVLocal());
    auto state = G::PrepareNativeMeshTopology(*primary, mpi);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, state);
    auto decoded = G::TestFixtureCodec::DecodePrimaryReference(*mesh);
    const G::tLocalMatStruct expected{{1}, {2, 0}, {1}};
    CHECK(mesh->GetCell2CellFaceVLocal() == expected);
    CHECK(mesh->GetCell2CellFaceVLocal(true) == expected);
    CHECK(decoded->GetCell2CellFaceVLocal() == expected);
    const auto cells = mesh->Entities(G::EntityKind::Cell);
    const auto stamp = mesh->versions.Stamp();
    auto retained = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
    for (int ordering : {0, -1, 1, 2, 3, 4})
    {
        CAPTURE(ordering);
        DNDS::Direct::DirectPrecControl control;
        control.useDirectPrec = true;
        control.orderingCode = ordering;
        DNDS::Direct::SerialSymLUStructure actual(mpi, 3), reference(mpi, 3);
        mesh->ObtainLocalFactFillOrdering(actual, control);
        auto [new2old, old2new] = GenericOrdering(expected, ordering);
        reference.localFillOrderingNew2Old = std::move(new2old);
        reference.localFillOrderingOld2New = std::move(old2new);
        CHECK(actual.localFillOrderingNew2Old == reference.localFillOrderingNew2Old);
        CHECK(actual.localFillOrderingOld2New == reference.localFillOrderingOld2New);
        CHECK(mesh->LocalCellNeighbors() == expected);
        mesh->ObtainSymmetricSymbolicFactorization(actual, control);
        reference.ObtainSymmetricSymbolicFactorization(expected, {0, 3}, control.getILUCode());
        CHECK(actual.lowerTriStructure == reference.lowerTriStructure);
        CHECK(actual.upperTriStructure == reference.upperTriStructure);
        CHECK(actual.cell2cellFaceVLocal2FullRowPos == reference.cell2cellFaceVLocal2FullRowPos);
    }
    CHECK(stamp.Valid());
    CHECK(mesh->Entities(G::EntityKind::Cell).SameOrder(cells));
    CHECK(retained.HostView().NumCell() == 3);
    CHECK_THROWS(state->WithLocalCellSetup({0, 1, 3}, expected));
    CHECK_THROWS(state->WithLocalCellSetup({0, 3}, {{1}, {2, 0}}));
    CHECK_THROWS(state->WithLocalCellSetup({0, 3}, {{3}, {2, 0}, {1}}));
    CHECK_THROWS(state->WithLocalCellSetup({0, 3}, {{0}, {2, 0}, {1}}));

    // Native partition metadata preserves cross-part filtering. The ordering
    // oracle above invokes generic serial algorithms on the exact input graph.
    auto partitioned = G::UnstructuredMesh::FromNative(mpi, state->WithLocalPartitions({0, 1, 3}));
    const G::tLocalMatStruct filtered{{}, {2}, {1}};
    CHECK(partitioned->GetCell2CellFaceVLocal() == expected);
    CHECK(partitioned->GetCell2CellFaceVLocal(true) == filtered);
    DNDS::Direct::DirectPrecControl control;
    control.useDirectPrec = true;
    control.orderingCode = -1;
    DNDS::Direct::SerialSymLUStructure ordering(mpi, 3);
    partitioned->ObtainLocalFactFillOrdering(ordering, control);
    CHECK(ordering.localFillOrderingNew2Old == std::vector<I>{0, 2, 1});
    CHECK(partitioned->LocalCellNeighbors() == filtered);
    CHECK(partitioned->LocalPartStart(1) == 1);

    // A factor worker can own both ends of the strip. Its scan must never
    // visit the middle row or add edges through that other worker.
    auto striped = G::UnstructuredMesh::FromNative(mpi, state->WithLocalPartitionRanges(
                                                            {{{0, 1}, {2, 3}}, {{1, 2}, {}}}));
    const auto bothEnds = striped->LocalPartRanges(0);
    CHECK(bothEnds.RowAt(1) == 2);
    CHECK_FALSE(bothEnds.Contains(1));
    CHECK_THROWS(striped->LocalPartEnd(0));
    CHECK(striped->GetCell2CellFaceVLocal(true) == G::tLocalMatStruct(3));
    DNDS::Direct::SerialSymLUStructure stripedFactor(mpi, 3);
    control.orderingCode = 0;
    striped->ObtainLocalFactFillOrdering(stripedFactor, control);
    striped->ObtainSymmetricSymbolicFactorization(stripedFactor, control);
    CHECK(stripedFactor.localFillOrderingNew2Old == std::vector<I>{0, 2, 1});
    CHECK(stripedFactor.localPartStarts == std::vector<I>{0, 2, 3});

    if (mpi.size > 1)
    {
        // Move each strip's middle cell away from both endpoints. Every real
        // face neighbor is now a ghost, despite three owned rows on each rank.
        G::ReorderInput input;
        input.explicitMaps = {{G::EntityKind::Cell, {mpi.rank, (mpi.rank + 1) % mpi.size, mpi.rank}},
                              {G::EntityKind::Node, std::vector<DNDS::MPI_int>(8, mpi.rank)}};
        auto moved = G::NativeMeshTransition::Build(*state, mpi, input);
        auto current = G::UnstructuredMesh::FromNative(mpi, moved.state);
        CHECK(current->NumCell() == 3);
        CHECK(current->NumCellGhost() > 0);
        CHECK(current->GetCell2CellFaceVLocal() == G::tLocalMatStruct(3));
        DNDS::Direct::SerialSymLUStructure structure(mpi, 3);
        control.orderingCode = 0;
        current->ObtainLocalFactFillOrdering(structure, control);
        current->ObtainSymmetricSymbolicFactorization(structure, control);
        CHECK(current->LocalCellNeighbors() == G::tLocalMatStruct(3));
    }
}

TEST_CASE("Native direct setup keeps empty cell owners in RCM collectives")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto state = G::PrepareNativeMeshTopology(*Strip(mpi, true), mpi);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, state);
    auto decoded = G::TestFixtureCodec::DecodePrimaryReference(*mesh);
    const G::tLocalMatStruct expected = mpi.rank == 0 ? G::tLocalMatStruct{{1}, {2, 0}, {1}} : G::tLocalMatStruct{};
    CHECK(mesh->GetCell2CellFaceVLocal() == expected);
    for (int code : {0, 3, 4})
    {
        DNDS::Direct::DirectPrecControl control;
        control.useDirectPrec = true;
        control.orderingCode = code;
        DNDS::Direct::SerialSymLUStructure structure(mpi, mesh->NumCell()), reference(mpi, mesh->NumCell());
        mesh->ObtainLocalFactFillOrdering(structure, control);
        auto [new2old, old2new] = GenericOrdering(expected, code);
        reference.localFillOrderingNew2Old = std::move(new2old);
        reference.localFillOrderingOld2New = std::move(old2new);
        mesh->ObtainSymmetricSymbolicFactorization(structure, control);
        reference.ObtainSymmetricSymbolicFactorization(expected, {0, mesh->NumCell()}, control.getILUCode());
        CHECK(structure.localFillOrderingNew2Old == reference.localFillOrderingNew2Old);
        CHECK(mesh->LocalCellNeighbors() == expected);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
