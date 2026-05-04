/**
 * @file test_MeshReorder.cpp
 * @brief Unit tests for ReorderPlan, ReorderRegistry, and classification.
 *
 * Tests:
 * - AdjAction classification logic
 * - ReorderRegistry registration and lookup
 * - ReorderPlan::build + apply on synthetic data (no real mesh)
 * - Full mesh reorder via UnstructuredMesh (later phases)
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/ReorderPlan.hpp"
#include <numeric>

using namespace DNDS;
using namespace DNDS::Geom;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();
    MPI_Finalize();
    return res;
}

static MPIInfo worldMPI()
{
    MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

// =================================================================
// Test: classifyAdj logic
// =================================================================

TEST_CASE("classifyAdj basic classification")
{
    std::unordered_set<EntityKind> reordered;

    SUBCASE("empty reordered set")
    {
        CHECK(classifyAdj(Adj::Cell2Node, reordered) == AdjAction::SKIP);
        CHECK(classifyAdj(Adj::Cell2Cell, reordered) == AdjAction::SKIP);
    }

    SUBCASE("cell only reordered")
    {
        reordered = {EntityKind::Cell};
        CHECK(classifyAdj(Adj::Cell2Node, reordered) == AdjAction::RELOCATE);
        CHECK(classifyAdj(Adj::Cell2Face, reordered) == AdjAction::RELOCATE);
        CHECK(classifyAdj(Adj::Cell2Cell, reordered) == AdjAction::SELF);
        CHECK(classifyAdj(Adj::Bnd2Cell, reordered) == AdjAction::REMAP);
        CHECK(classifyAdj(Adj::Face2Cell, reordered) == AdjAction::REMAP);
        CHECK(classifyAdj(Adj::Node2Cell, reordered) == AdjAction::REMAP);
        CHECK(classifyAdj(Adj::Bnd2Node, reordered) == AdjAction::SKIP);
        CHECK(classifyAdj(Adj::Face2Node, reordered) == AdjAction::SKIP);
    }

    SUBCASE("node only reordered")
    {
        reordered = {EntityKind::Node};
        CHECK(classifyAdj(Adj::Cell2Node, reordered) == AdjAction::REMAP);
        CHECK(classifyAdj(Adj::Bnd2Node, reordered) == AdjAction::REMAP);
        CHECK(classifyAdj(Adj::Node2Cell, reordered) == AdjAction::RELOCATE);
        CHECK(classifyAdj(Adj::Node2Bnd, reordered) == AdjAction::RELOCATE);
        CHECK(classifyAdj(Adj::Cell2Cell, reordered) == AdjAction::SKIP);
    }

    SUBCASE("cell + node reordered")
    {
        reordered = {EntityKind::Cell, EntityKind::Node};
        CHECK(classifyAdj(Adj::Cell2Node, reordered) == AdjAction::RELOCATE_REMAP);
        CHECK(classifyAdj(Adj::Node2Cell, reordered) == AdjAction::RELOCATE_REMAP);
        CHECK(classifyAdj(Adj::Cell2Cell, reordered) == AdjAction::SELF);
        CHECK(classifyAdj(Adj::Bnd2Node, reordered) == AdjAction::REMAP);
        CHECK(classifyAdj(Adj::Bnd2Cell, reordered) == AdjAction::REMAP);
    }

    SUBCASE("cell + node + bnd reordered")
    {
        reordered = {EntityKind::Cell, EntityKind::Node, EntityKind::Bnd};
        CHECK(classifyAdj(Adj::Cell2Node, reordered) == AdjAction::RELOCATE_REMAP);
        CHECK(classifyAdj(Adj::Bnd2Node, reordered) == AdjAction::RELOCATE_REMAP);
        CHECK(classifyAdj(Adj::Bnd2Cell, reordered) == AdjAction::RELOCATE_REMAP);
        CHECK(classifyAdj(Adj::Node2Cell, reordered) == AdjAction::RELOCATE_REMAP);
        CHECK(classifyAdj(Adj::Node2Bnd, reordered) == AdjAction::RELOCATE_REMAP);
        CHECK(classifyAdj(Adj::Cell2Cell, reordered) == AdjAction::SELF);
    }
}

// =================================================================
// Test: ReorderRegistry basic operations
// =================================================================

TEST_CASE("ReorderRegistry register and query")
{
    auto mpi = worldMPI();
    ReorderRegistry reg;

    // Register a global mapping
    auto gm = make_ssp<GlobalOffsetsMapping>();
    gm->setMPIAlignBcast(mpi, 10);
    reg.registerGlobalMapping(EntityKind::Cell, gm);

    CHECK(reg.getGlobalMapping(EntityKind::Cell) == gm);
    CHECK(reg.getGlobalMapping(EntityKind::Node) == nullptr);

    // Register an adj
    bool remapCalled = false;
    bool relocateCalled = false;
    reg.registerAdj(
        Adj::Cell2Node,
        [&](const PermutationTransfer::LookupResult &)
        { remapCalled = true; },
        [&](const PermutationTransfer &, const MPIInfo &)
        { relocateCalled = true; },
        "cell2node");

    CHECK(reg.adjs.size() == 1);
    CHECK(reg.adjs[0].kind == Adj::Cell2Node);
    CHECK(reg.adjs[0].name == "cell2node");

    // Register a companion
    bool compCalled = false;
    reg.registerCompanion(
        EntityKind::Cell,
        [&](const PermutationTransfer &, const MPIInfo &)
        { compCalled = true; },
        "cellElemInfo");

    CHECK(reg.companions.size() == 1);
    CHECK(reg.companions[0].kind == EntityKind::Cell);
}

// =================================================================
// Test: ReorderPlan::apply with synthetic data
// =================================================================

TEST_CASE("ReorderPlan::apply cell-only local permutation")
{
    auto mpi = worldMPI();
    const DNDS::index nCell = 8;
    const DNDS::index nNode = 4;

    // Create synthetic cell2node: each cell references 2 nodes (global)
    ArrayAdjacencyPair<2> cell2node;
    cell2node.InitPair("cell2node", mpi);
    cell2node.father->Resize(nCell);
    cell2node.father->createGlobalMapping();

    DNDS::index cellOffset = (*cell2node.father->pLGlobalMapping)(mpi.rank, 0);

    // Create synthetic node array
    ArrayAdjacencyPair<1> nodeArr;
    nodeArr.InitPair("nodeArr", mpi);
    nodeArr.father->Resize(nNode);
    nodeArr.father->createGlobalMapping();

    DNDS::index nodeOffset = (*nodeArr.father->pLGlobalMapping)(mpi.rank, 0);

    // Fill cell2node: cell i references nodes (i%nNode) and ((i+1)%nNode)
    for (DNDS::index i = 0; i < nCell; i++)
    {
        cell2node(i, 0) = nodeOffset + (i % nNode);
        cell2node(i, 1) = nodeOffset + ((i + 1) % nNode);
    }

    // Create a companion array (cellElemInfo analog)
    ArrayAdjacencyPair<1> cellInfo;
    cellInfo.InitPair("cellInfo", mpi);
    cellInfo.father->Resize(nCell);
    for (DNDS::index i = 0; i < nCell; i++)
        cellInfo(i, 0) = 1000 + cellOffset + i; // tag = 1000 + global

    // Build registry
    ReorderRegistry reg;
    reg.registerGlobalMapping(EntityKind::Cell, cell2node.father->pLGlobalMapping);
    reg.registerGlobalMapping(EntityKind::Node, nodeArr.father->pLGlobalMapping);

    reg.registerAdj(
        Adj::Cell2Node,
        nullptr, // no remap needed (only Cell reordered, not Node)
        [&](const PermutationTransfer &t, const MPIInfo &m)
        { t.transferRows(cell2node, m); },
        "cell2node");

    reg.registerCompanion(
        EntityKind::Cell,
        [&](const PermutationTransfer &t, const MPIInfo &m)
        { t.transferRows(cellInfo, m); },
        "cellInfo");

    // Build plan: reverse cell permutation (all local)
    std::vector<MPI_int> cellPartition(nCell, mpi.rank);
    ReorderInput input;
    input.explicitMaps.push_back(EntityReorderMap{EntityKind::Cell, cellPartition});

    auto plan = ReorderPlan::build(input, reg, mpi);
    CHECK(plan.isLocalOnly);
    CHECK(plan.reorderedKinds.count(EntityKind::Cell));
    CHECK_FALSE(plan.reorderedKinds.count(EntityKind::Node));

    // Apply
    plan.apply(reg, mpi);

    // Since partition = all-self and ordering is preserved within rank,
    // the data should be unchanged (identity permutation via fromPartition).
    for (DNDS::index i = 0; i < nCell; i++)
    {
        CHECK(cell2node(i, 0) == nodeOffset + (i % nNode));
        CHECK(cell2node(i, 1) == nodeOffset + ((i + 1) % nNode));
        CHECK(cellInfo(i, 0) == 1000 + cellOffset + i);
    }
}

// =================================================================
// Test: ReorderPlan::apply with remap (node reorder, cells stay)
// =================================================================

TEST_CASE("ReorderPlan::apply node-only remap")
{
    auto mpi = worldMPI();
    const DNDS::index nCell = 4;
    const DNDS::index nNode = 6;

    // cell2node: Cell->Node (cell rows fixed, node entries need remapping)
    ArrayAdjacencyPair<2> cell2node;
    cell2node.InitPair("cell2node", mpi);
    cell2node.father->Resize(nCell);
    cell2node.father->createGlobalMapping();

    ArrayAdjacencyPair<1> nodeArr;
    nodeArr.InitPair("nodeArr", mpi);
    nodeArr.father->Resize(nNode);
    nodeArr.father->createGlobalMapping();

    DNDS::index nodeOffset = (*nodeArr.father->pLGlobalMapping)(mpi.rank, 0);

    // Fill: cell i refs nodes i and i+1
    for (DNDS::index i = 0; i < nCell; i++)
    {
        cell2node(i, 0) = nodeOffset + i;
        cell2node(i, 1) = nodeOffset + i + 1;
    }

    // Node companion: coords analog
    ArrayAdjacencyPair<1> coords;
    coords.InitPair("coords", mpi);
    coords.father->Resize(nNode);
    for (DNDS::index i = 0; i < nNode; i++)
        coords(i, 0) = 500 + nodeOffset + i; // value = 500 + globalNode

    // Build registry
    ReorderRegistry reg;
    reg.registerGlobalMapping(EntityKind::Cell, cell2node.father->pLGlobalMapping);
    reg.registerGlobalMapping(EntityKind::Node, nodeArr.father->pLGlobalMapping);

    reg.registerAdj(
        Adj::Cell2Node,
        [&](const PermutationTransfer::LookupResult &lookup)
        {
            for (DNDS::index i = 0; i < nCell; i++)
                for (rowsize j = 0; j < 2; j++)
                {
                    DNDS::index &v = cell2node(i, j);
                    if (v != UnInitIndex)
                        v = lookup.resolve(v);
                }
        },
        nullptr, // no relocate (Cell not reordered)
        "cell2node");

    reg.registerCompanion(
        EntityKind::Node,
        [&](const PermutationTransfer &t, const MPIInfo &m)
        { t.transferRows(coords, m); },
        "coords");

    // Reorder nodes: all stay local (identity partition)
    std::vector<MPI_int> nodePartition(nNode, mpi.rank);
    ReorderInput input;
    input.explicitMaps.push_back(EntityReorderMap{EntityKind::Node, nodePartition});

    auto plan = ReorderPlan::build(input, reg, mpi);
    CHECK(plan.isLocalOnly);
    CHECK(plan.reorderedKinds.count(EntityKind::Node));
    CHECK_FALSE(plan.reorderedKinds.count(EntityKind::Cell));

    // Apply
    plan.apply(reg, mpi);

    // With identity partition (all stay on same rank), new globals are
    // contiguous starting at newGlobalOffsets[mpi.rank].
    // For identity: newGlobalIndices[i] = nodeOffset + i (unchanged).
    // So remap should be identity, coords should be unchanged.
    for (DNDS::index i = 0; i < nCell; i++)
    {
        // Since it's identity partition, old globals map to same new globals
        auto &transfer = plan.transfers.at(EntityKind::Node);
        DNDS::index expectedNode0 = transfer.newGlobalIndices[i];
        DNDS::index expectedNode1 = transfer.newGlobalIndices[i + 1];
        CHECK(cell2node(i, 0) == expectedNode0);
        CHECK(cell2node(i, 1) == expectedNode1);
    }

    for (DNDS::index i = 0; i < nNode; i++)
        CHECK(coords(i, 0) == 500 + nodeOffset + i);
}
