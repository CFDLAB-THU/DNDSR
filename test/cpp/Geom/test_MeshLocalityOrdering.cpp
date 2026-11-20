#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Transition/MeshLocalityOrdering.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
#include "Geom/Transition/LocalCellOrdering.hpp"
#include "NativeMeshFacadeFixture.hpp"
#include "NativeMeshOverlapFixture.hpp"

using namespace DNDS::Geom;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("First-touch node and face permutations preserve distributed cell geometry")
{
    auto fixture = OverlapTest::BuildNativeFixture(mpi, true);
    auto order = MeshLocalityOrdering::Build(*fixture.mesh, EntityKind::Cell, {Adj::Cell2Node, Adj::Cell2Face});
    auto moved = NativeMeshTransition::Build(*fixture.mesh->CurrentNativeState(), mpi, order.Input());
    auto mesh = UnstructuredMesh::FromNative(mpi, moved.state);
    auto scope = mesh->AcquireRead(MeshReadRequirements::CellFlux());
    const auto view = scope.HostView();
    constexpr std::array<std::pair<I, I>, 4> corners{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
    for (I iCell = 0; iCell < view.NumCell(); ++iCell)
    {
        const I original = mesh->OriginalEntityId(EntityKind::Cell, iCell);
        const I x = original % 10, y = original / 10;
        for (I slot = 0; slot < 4; ++slot)
        {
            const tPoint expected{0.1 * (x + corners[slot].first), 0.1 * (y + corners[slot].second), 0};
            CHECK((view.Coord(view.CellNodes(iCell)[slot]) - expected).norm() < 1e-14);
            const auto face = view.CellFaces(iCell)[slot];
            const auto parents = view.FaceCells(face);
            CHECK((parents[0] == iCell || parents[1] == iCell));
        }
    }
}

TEST_CASE("Subregion cell ordering splits conservative core before grouped halo rows")
{
    auto fixture = OverlapTest::BuildNativeFixture(mpi, true);
    const auto work = EntityWorkPartition::CellNodeCell(*fixture.mesh);
    for (bool aggregate : {false, true})
        for (auto split : {CellCoreSplit::None, CellCoreSplit::CellNodeCell})
        {
            const auto order = BuildLocalCellOrdering(*fixture.mesh, {3, split, aggregate});
            CHECK(order.subregionStarts.size() == 4);
            CHECK(order.coreEnds.size() == 3);
            std::vector<I> sorted = order.oldToNew;
            std::sort(sorted.begin(), sorted.end());
            for (I iRow = 0; iRow < I(sorted.size()); ++iRow)
                CHECK(sorted[iRow] == iRow);
            if (split == CellCoreSplit::CellNodeCell)
            {
                for (auto old : work.Core())
                {
                    const auto row = order.oldToNew[old];
                    const auto region = std::upper_bound(order.subregionStarts.begin(), order.subregionStarts.end(), row) - order.subregionStarts.begin() - 1;
                    CHECK(row < order.coreEnds[region]);
                }
                for (auto old : work.HaloDependent())
                {
                    const auto row = order.oldToNew[old];
                    const auto region = std::upper_bound(order.subregionStarts.begin(), order.subregionStarts.end(), row) - order.subregionStarts.begin() - 1;
                    CHECK(row >= order.coreEnds[region]);
                }
            }
        }
    const LocalCellOrderingSettings settings{3, CellCoreSplit::CellNodeCell, true};
    const auto expected = BuildLocalCellOrdering(*fixture.mesh, settings);
    const auto moved = NativeMeshTransition::ReorderLocalCells(*fixture.mesh, settings);
    for (I old = 0; old < I(expected.oldToNew.size()); ++old)
        CHECK(moved.state->OriginalEntityId(EntityKind::Cell, expected.oldToNew[old]) ==
              fixture.mesh->OriginalEntityId(EntityKind::Cell, old));
}

TEST_CASE("Locality plans preserve first-touch order from cell or node anchors")
{
    auto input = NativeFacadeFixture::Make(mpi);
    // A cyclic triangle-node permutation preserves its physical orientation.
    // Face-slot order follows the same cyclic permutation, while entity rows
    // deliberately retain their original order.
    input.data.relations[size_t(MeshRelation::CellNodes)] = NativeFacadeFixture::Rows<I>({{2, 0, 1}});
    input.data.relations[size_t(MeshRelation::CellFaces)] = NativeFacadeFixture::Rows<I>({{2, 0, 1}});
    auto mesh = input.Publish(mpi);
    const auto cells = MeshLocalityOrdering::Build(*mesh, EntityKind::Cell, {Adj::Cell2Node, Adj::Cell2Face});
    CHECK(cells.Permutations().at(EntityKind::Node) == std::vector<I>{1, 2, 0});
    CHECK(cells.Permutations().at(EntityKind::Face) == std::vector<I>{1, 2, 0});
    const auto nodes = MeshLocalityOrdering::Build(*mesh, EntityKind::Node, {Adj::Node2Bnd});
    CHECK(nodes.Permutations().at(EntityKind::Bnd) == std::vector<I>{0, 2, 1});
    auto moved = NativeMeshTransition::Build(*mesh->CurrentNativeState(), mpi, cells.Input());
    auto reordered = UnstructuredMesh::FromNative(mpi, moved.state);
    CHECK(moved.transfers.size() == size_t(5)); // identities are available for unaffected families too
    const std::array<I, 3> oldRows{2, 0, 1};
    {
        auto scope = reordered->AcquireRead(NativeFacadeFixture::Requirements());
        const auto view = scope.HostView();
        for (I iRow = 0; iRow < 3; ++iRow)
        {
            CHECK(reordered->OriginalEntityId(EntityKind::Node, iRow) == mesh->OriginalEntityId(EntityKind::Node, oldRows[iRow]));
            CHECK(reordered->OriginalEntityId(EntityKind::Face, iRow) == mesh->OriginalEntityId(EntityKind::Face, oldRows[iRow]));
            CHECK(view.CellNodes(0)[iRow] == iRow);
            CHECK(view.CellFaces(0)[iRow] == iRow);
        }
    }
    auto boundaryMove = NativeMeshTransition::Build(*mesh->CurrentNativeState(), mpi, nodes.Input());
    auto boundaryMesh = UnstructuredMesh::FromNative(mpi, boundaryMove.state);
    {
        auto boundaryRead = boundaryMesh->AcquireRead(NativeFacadeFixture::Requirements());
        const std::array<I, 3> oldBoundaries{0, 2, 1};
        for (I iRow = 0; iRow < 3; ++iRow)
        {
            CHECK(boundaryMesh->OriginalEntityId(EntityKind::Bnd, iRow) == mesh->OriginalEntityId(EntityKind::Bnd, oldBoundaries[iRow]));
            CHECK(boundaryRead.HostView().BoundaryFaces(iRow)[0] == oldBoundaries[iRow]);
        }
    }
    // All family extents match, and the original source is still current.
    // Replaying its first-touch permutation on the reordered publication is stale.
    const auto saved = cells.Input();
    CHECK_THROWS(NativeMeshTransition::Build(*reordered->CurrentNativeState(), mpi, saved));
    auto invalid = cells.Input();
    if (mpi.rank == 0)
        invalid.localPermutations.at(EntityKind::Node)[0] = invalid.localPermutations.at(EntityKind::Node)[1];
    CHECK_THROWS(NativeMeshTransition::Build(*mesh->CurrentNativeState(), mpi, invalid));
    CHECK_THROWS(MeshLocalityOrdering::Build(*mesh, EntityKind::Node, {Adj::Cell2Node}));
    CHECK_THROWS(MeshLocalityOrdering::Build(*mesh, EntityKind::Cell, {Adj::Cell2Node, Adj::Cell2Node}));
    CHECK_NOTHROW(cells.Validate());
    {
        auto edit = mesh->versions.BeginMutation(DNDS::Runtime::Change::Geometry);
        edit.Commit();
    }
    CHECK_NOTHROW(cells.Validate());
    {
        auto edit = mesh->versions.BeginMutation(DNDS::Runtime::Change::Layout);
        edit.Commit();
    }
    CHECK_THROWS(cells.Permutations());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    MPI_Finalize();
    return result;
}
