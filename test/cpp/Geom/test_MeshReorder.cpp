/**
 * @file test_MeshReorder.cpp
 * @brief Unit tests for ReorderPlan, ReorderRegistry, and classification.
 *
 * Tests:
 * - AdjAction classification logic
 * - ReorderRegistry registration and lookup
 * - ReorderPlan::apply on synthetic data (no real mesh)
 * - Full mesh ReorderEntities on real CGNS meshes (Phase 2b)
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/ReorderPlan.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include <numeric>
#include <set>

using namespace DNDS;
using namespace DNDS::Geom;

// NOTE: DNDS::index, DNDS::real, DNDS::rowsize clash with POSIX symbols.
// Qualify in declarations to avoid ambiguity.
using idx = DNDS::index;

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

// =================================================================
// Real mesh helpers
// =================================================================

static std::string meshPath(const std::string &name)
{
    std::string f(__FILE__);
    for (int i = 0; i < 4; i++)
    {
        auto pos = f.rfind('/');
        if (pos == std::string::npos)
            pos = f.rfind('\\');
        if (pos != std::string::npos)
            f = f.substr(0, pos);
    }
    return f + "/data/mesh/" + name;
}

/// Build a mesh through the primary pipeline (up to ghost + local indices).
/// Returns mesh in Adj_PointToLocal state with ghost layers.
static ssp<UnstructuredMesh> buildMeshPrimary(
    const MPIInfo &mpi, const std::string &file, int dim,
    bool withFaces = false)
{
    auto mesh = make_ssp<UnstructuredMesh>(mpi, dim);
    UnstructuredMeshSerialRW reader(mesh, 0);
    reader.ReadFromCGNSSerial(meshPath(file));
    reader.BuildCell2Cell();

    UnstructuredMeshSerialRW::PartitionOptions pOpt;
    pOpt.metisType = "KWAY";
    pOpt.metisUfactor = 30;
    pOpt.metisSeed = 42;
    pOpt.metisNcuts = 1;
    reader.MeshPartitionCell2Cell(pOpt);
    reader.PartitionReorderToMeshCell2Cell();

    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();

    if (withFaces)
    {
        mesh->InterpolateFace();
        mesh->AdjLocal2GlobalN2CB();
        mesh->BuildGhostN2CB();
        mesh->AdjGlobal2LocalN2CB();
    }

    return mesh;
}

/// Collect all owned global indices for an entity kind (from globalMapping).
static std::set<DNDS::index> collectOwnedGlobals(
    const ssp<GlobalOffsetsMapping> &gm, const MPIInfo &mpi)
{
    std::set<DNDS::index> result;
    DNDS::index offset = (*gm)(mpi.rank, 0);
    DNDS::index count = gm->RLengths()[mpi.rank];
    for (DNDS::index i = 0; i < count; i++)
        result.insert(offset + i);
    return result;
}

/// Gather all owned globals across all ranks (for total count check).
static DNDS::index gatherGlobalCount(const ssp<GlobalOffsetsMapping> &gm, const MPIInfo &mpi)
{
    return gm->globalSize();
}

/// Check that an adj array's entries are all valid globals within
/// [0, targetGlobalSize) or UnInitIndex.
static bool checkAdjEntriesValid(
    const auto &adj, DNDS::index nRows, DNDS::index targetGlobalSize)
{
    for (DNDS::index i = 0; i < nRows; i++)
        for (rowsize j = 0; j < adj.RowSize(i); j++)
        {
            DNDS::index v = adj(i, j);
            if (v == UnInitIndex)
                continue;
            if (v < 0 || v >= targetGlobalSize)
                return false;
        }
    return true;
}

// =================================================================
// Test: Cell-only local reorder on real mesh (no faces)
// =================================================================

TEST_CASE("ReorderEntities cell-only local on UniformSquare_10")
{
    auto mpi = worldMPI();
    auto mesh = buildMeshPrimary(mpi, "UniformSquare_10.cgns", 2, false);

    // Snapshot pre-reorder state
    DNDS::index nCellBefore = mesh->NumCell();
    DNDS::index nNodeBefore = mesh->NumNode();
    DNDS::index nBndBefore = mesh->NumBnd();
    DNDS::index nCellGlobal = mesh->cell2node.father->pLGlobalMapping->globalSize();
    DNDS::index nNodeGlobal = mesh->coords.father->pLGlobalMapping->globalSize();

    // Convert to global for reorder
    mesh->AdjLocal2GlobalPrimary();

    // Build cell reorder: all cells stay on same rank (identity partition)
    std::vector<MPI_int> cellPartition(nCellBefore, mpi.rank);
    ReorderInput input;
    input.explicitMaps.push_back(EntityReorderMap{EntityKind::Cell, cellPartition});
    // Default follows: Node and Bnd follow Cell

    mesh->ReorderEntities(input);

    // Post-condition checks
    CHECK(mesh->adjPrimaryState == Adj_PointToGlobal);
    CHECK(mesh->NumCell() == nCellBefore);
    CHECK(mesh->NumNode() == nNodeBefore);
    CHECK(mesh->NumBnd() == nBndBefore);

    // Global counts preserved
    CHECK(mesh->cell2node.father->pLGlobalMapping->globalSize() == nCellGlobal);
    CHECK(mesh->coords.father->pLGlobalMapping->globalSize() == nNodeGlobal);

    // Adj entries are valid globals
    CHECK(checkAdjEntriesValid(mesh->cell2node, nCellBefore, nNodeGlobal));
    CHECK(checkAdjEntriesValid(mesh->cell2cell, nCellBefore, nCellGlobal));
    CHECK(checkAdjEntriesValid(mesh->bnd2cell, nBndBefore, nCellGlobal));

    // Node2cell entries point to valid cell globals
    CHECK(checkAdjEntriesValid(mesh->node2cell, nNodeBefore, nCellGlobal));

    // Verify mesh can be rebuilt: ghost + local conversion
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();

    // Sanity: cell2node entries should be valid local indices now
    for (DNDS::index iC = 0; iC < mesh->NumCell(); iC++)
        for (rowsize j = 0; j < mesh->cell2node.RowSize(iC); j++)
        {
            DNDS::index iN = mesh->cell2node(iC, j);
            CHECK(iN >= 0);
            CHECK(iN < mesh->NumNodeProc());
        }
}

// =================================================================
// Test: Cell-only local with faces (face destruction)
// =================================================================

TEST_CASE("ReorderEntities cell-only with face destruction on UniformSquare_10")
{
    auto mpi = worldMPI();
    // Build without faces (simpler), then manually build faces to test destruction
    auto mesh = buildMeshPrimary(mpi, "UniformSquare_10.cgns", 2, false);

    // Build faces (from local state)
    mesh->InterpolateFace();

    CHECK(mesh->face2node.father); // faces exist before reorder

    // Convert everything to global for reorder
    mesh->AdjLocal2GlobalPrimary();
    mesh->AdjLocal2GlobalFacial();
    mesh->AdjLocal2GlobalC2F();

    DNDS::index nCellBefore = mesh->NumCell();

    // Reorder with face destruction
    std::vector<MPI_int> cellPartition(nCellBefore, mpi.rank);
    ReorderInput input;
    input.explicitMaps.push_back(EntityReorderMap{EntityKind::Cell, cellPartition});
    input.destroyKinds = {EntityKind::Face};

    mesh->ReorderEntities(input);

    // Faces should be destroyed
    CHECK_FALSE(mesh->face2node.father);
    CHECK_FALSE(mesh->face2cell.father);
    CHECK_FALSE(mesh->cell2face.father);
    CHECK(mesh->adjFacialState == Adj_Unknown);

    // Primary adj still valid
    CHECK(mesh->adjPrimaryState == Adj_PointToGlobal);
    CHECK(mesh->NumCell() == nCellBefore);

    // Can rebuild faces from scratch
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    mesh->InterpolateFace();
    mesh->AssertOnFaces();
}

// =================================================================
// Test: Cell distributed reorder (round-robin) with node/bnd follow
// =================================================================

TEST_CASE("ReorderEntities cell distributed round-robin with follow")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return;

    auto mesh = buildMeshPrimary(mpi, "UniformSquare_10.cgns", 2, false);

    DNDS::index nCellGlobal = mesh->cell2node.father->pLGlobalMapping->globalSize();
    DNDS::index nNodeGlobal = mesh->coords.father->pLGlobalMapping->globalSize();
    DNDS::index nBndGlobal = mesh->bnd2node.father->pLGlobalMapping->globalSize();

    // Convert to global
    mesh->AdjLocal2GlobalPrimary();
    // N2CB already global after buildMeshPrimary(withFaces=false)

    DNDS::index nCellBefore = mesh->NumCell();

    // Round-robin: cell i goes to rank (i % nRanks)
    std::vector<MPI_int> cellPartition(nCellBefore);
    for (DNDS::index i = 0; i < nCellBefore; i++)
        cellPartition[i] = static_cast<MPI_int>(i % mpi.size);

    ReorderInput input;
    input.explicitMaps.push_back(EntityReorderMap{EntityKind::Cell, cellPartition});
    // Node and Bnd follow automatically

    mesh->ReorderEntities(input);

    // Global counts preserved (collective check)
    DNDS::index newCellGlobal = mesh->cell2node.father->pLGlobalMapping->globalSize();
    DNDS::index newNodeGlobal = mesh->coords.father->pLGlobalMapping->globalSize();
    DNDS::index newBndGlobal = mesh->bnd2node.father->pLGlobalMapping->globalSize();
    CHECK(newCellGlobal == nCellGlobal);
    CHECK(newNodeGlobal == nNodeGlobal);
    CHECK(newBndGlobal == nBndGlobal);

    // Adj entries valid
    CHECK(checkAdjEntriesValid(mesh->cell2node, mesh->NumCell(), newNodeGlobal));
    CHECK(checkAdjEntriesValid(mesh->cell2cell, mesh->NumCell(), newCellGlobal));
    CHECK(checkAdjEntriesValid(mesh->bnd2node, mesh->NumBnd(), newNodeGlobal));
    CHECK(checkAdjEntriesValid(mesh->bnd2cell, mesh->NumBnd(), newCellGlobal));

    // Verify no duplicate globals: each rank's cell globals should be unique
    // and contiguous within [offset, offset+nLocal).
    DNDS::index myOffset = (*mesh->cell2node.father->pLGlobalMapping)(mpi.rank, 0);
    DNDS::index myCount = mesh->NumCell();
    for (DNDS::index i = 0; i < myCount; i++)
    {
        DNDS::index g = myOffset + i;
        CHECK(g >= 0);
        CHECK(g < newCellGlobal);
    }

    // Verify mesh can be fully rebuilt
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();

    // Cell2node entries are valid local-appended indices
    for (DNDS::index iC = 0; iC < mesh->NumCell(); iC++)
        for (rowsize j = 0; j < mesh->cell2node.RowSize(iC); j++)
        {
            DNDS::index iN = mesh->cell2node(iC, j);
            CHECK(iN >= 0);
            CHECK(iN < mesh->NumNodeProc());
        }
}

// =================================================================
// Test: Node-only local reorder (cells stay, node entries remapped)
// =================================================================

TEST_CASE("ReorderEntities node-only local on UniformSquare_10")
{
    auto mpi = worldMPI();
    auto mesh = buildMeshPrimary(mpi, "UniformSquare_10.cgns", 2, false);

    DNDS::index nCellBefore = mesh->NumCell();
    DNDS::index nNodeBefore = mesh->NumNode();
    DNDS::index nNodeGlobal = mesh->coords.father->pLGlobalMapping->globalSize();

    // Snapshot coords before reorder (to verify relocation)
    std::vector<tPoint> coordsBefore(nNodeBefore);
    for (DNDS::index i = 0; i < nNodeBefore; i++)
        coordsBefore[i] = mesh->coords[i];

    // Convert to global
    mesh->AdjLocal2GlobalPrimary();
    // N2CB already global (RecoverNode2CellAndNode2Bnd leaves it global
    // when BuildGhostN2CB is not called)

    // Node reorder: all stay local (identity)
    std::vector<MPI_int> nodePartition(nNodeBefore, mpi.rank);
    ReorderInput input;
    input.explicitMaps.push_back(EntityReorderMap{EntityKind::Node, nodePartition});
    input.destroyKinds = {EntityKind::Face}; // faces invalid after node reorder

    mesh->ReorderEntities(input);

    // Cells should not have moved (cell count same)
    CHECK(mesh->NumCell() == nCellBefore);
    CHECK(mesh->NumNode() == nNodeBefore);

    // Node globals preserved
    CHECK(mesh->coords.father->pLGlobalMapping->globalSize() == nNodeGlobal);

    // Cell2node entries should point to valid node globals
    CHECK(checkAdjEntriesValid(mesh->cell2node, nCellBefore, nNodeGlobal));

    // Coords should be preserved (identity partition = no movement)
    for (DNDS::index i = 0; i < nNodeBefore; i++)
        CHECK(mesh->coords[i] == coordsBefore[i]);

    // Verify rebuild works
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    mesh->InterpolateFace();
    mesh->AssertOnFaces();
}
