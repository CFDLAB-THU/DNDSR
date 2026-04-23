/**
 * @file test_MeshConnectivity.cpp
 * @brief Unit tests for MeshConnectivity DSL operations: Inverse, Compose, ComposeFiltered.
 *        Also tests MeshConnectivity struct management (cone/support).
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "SyntheticMeshBuilders.hpp"
#include "Geom/MeshConnectivity.hpp"
#include "Geom/Mesh.hpp"
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_set>
#include <numeric>

using namespace DNDS;
using namespace DNDS::Geom;

static MPIInfo g_mpi;

// ===========================================================================
// Inverse tests
// ===========================================================================

// ===========================================================================
// Inverse Tests
// ===========================================================================

TEST_CASE("Inverse: 4-quad serial correctness")
{
    auto c2n = make4QuadCell2Node(g_mpi);
    c2n.father->createGlobalMapping();
    auto nodeGM = makeNodeGlobalMapping4Quad(g_mpi);

    DNDS::index nNodeLocal = nodeLocalCount4Quad(g_mpi);
    auto n2c = MeshConnectivity::Inverse(
        c2n, nNodeLocal, g_mpi,
        [&](DNDS::index i)
        { return cellLocal2Global4Quad(g_mpi, i); },
        [&](DNDS::index i)
        { return nodeLocal2Global4Quad(g_mpi, i); },
        nodeGM);

    // Gather global result for verification
    auto globalN2C = gatherInverseGlobal(
        n2c, nNodeLocal,
        [&](DNDS::index i)
        { return nodeLocal2Global4Quad(g_mpi, i); },
        9, g_mpi);

    // Expected node2cell (from the 4-quad layout):
    //   node 0: {0}
    //   node 1: {0, 1}
    //   node 2: {1}
    //   node 3: {0, 2}
    //   node 4: {0, 1, 2, 3}
    //   node 5: {1, 3}
    //   node 6: {2}
    //   node 7: {2, 3}
    //   node 8: {3}
    std::vector<std::set<DNDS::index>> expected = {
        {0},          // node 0
        {0, 1},       // node 1
        {1},          // node 2
        {0, 2},       // node 3
        {0, 1, 2, 3}, // node 4
        {1, 3},       // node 5
        {2},          // node 6
        {2, 3},       // node 7
        {3},          // node 8
    };

    for (DNDS::index n = 0; n < 9; n++)
    {
        CAPTURE(n);
        CHECK(globalN2C[n] == expected[n]);
    }
}

TEST_CASE("Inverse: round-trip covers original cell2node")
{
    // inverse(inverse(cell2node)) should give cell2cell-like structure where
    // each cell's row contains at least itself and all cells sharing any node.
    // Specifically: for each cell c, for each node n in c2n[c], all cells in
    // n2c[n] should appear in the re-inverted result for c.

    auto c2n = make4QuadCell2Node(g_mpi);
    c2n.father->createGlobalMapping();
    auto nodeGM = makeNodeGlobalMapping4Quad(g_mpi);

    DNDS::index nNodeLocal = nodeLocalCount4Quad(g_mpi);
    DNDS::index nCellLocal = c2n.father->Size();

    auto n2c = MeshConnectivity::Inverse(
        c2n, nNodeLocal, g_mpi,
        [&](DNDS::index i)
        { return cellLocal2Global4Quad(g_mpi, i); },
        [&](DNDS::index i)
        { return nodeLocal2Global4Quad(g_mpi, i); },
        nodeGM);

    // Now create a cell global mapping for the second inverse
    auto cellGM = std::make_shared<GlobalOffsetsMapping>();
    cellGM->setMPIAlignBcast(g_mpi, nCellLocal);

    // inverse(n2c) -> cell2node_roundtrip
    // n2c maps node->cells, so inverse gives cell->nodes
    auto c2n_rt = MeshConnectivity::Inverse(
        n2c, nCellLocal, g_mpi,
        [&](DNDS::index i)
        { return nodeLocal2Global4Quad(g_mpi, i); },
        [&](DNDS::index i)
        { return cellLocal2Global4Quad(g_mpi, i); },
        cellGM);

    // For each local cell, verify that the original c2n nodes are a subset
    // of the round-tripped nodes
    for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
    {
        std::set<DNDS::index> originalNodes;
        for (auto n : c2n.father->operator[](iCell))
            originalNodes.insert(n);

        std::set<DNDS::index> roundTripNodes;
        for (auto n : c2n_rt.father->operator[](iCell))
            roundTripNodes.insert(n);

        for (auto n : originalNodes)
        {
            CAPTURE(iCell);
            CAPTURE(n);
            CHECK(roundTripNodes.count(n) == 1);
        }
    }
}


// ===========================================================================
// ComposeFiltered tests
// ===========================================================================

// ===========================================================================
// Compose / ComposeFiltered Tests
// ===========================================================================

TEST_CASE("ComposeFiltered: 4-quad cell2cell via node-neighbor")
{
    auto c2n = make4QuadCell2Node(g_mpi);
    c2n.father->createGlobalMapping();
    auto nodeGM = makeNodeGlobalMapping4Quad(g_mpi);

    DNDS::index nNodeLocal = nodeLocalCount4Quad(g_mpi);
    DNDS::index nCellLocal = c2n.father->Size();

    // Step 1: Inverse to get node2cell
    auto n2c = MeshConnectivity::Inverse(
        c2n, nNodeLocal, g_mpi,
        [&](DNDS::index i)
        { return cellLocal2Global4Quad(g_mpi, i); },
        [&](DNDS::index i)
        { return nodeLocal2Global4Quad(g_mpi, i); },
        nodeGM);

    // Step 2: Ghost-pull n2c for off-rank nodes referenced by c2n
    // Build ghost list: all node globals in c2n that are off-rank
    std::unordered_set<DNDS::index> ghostNodeSet;
    for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
        for (auto iNode : c2n.father->operator[](iCell))
        {
            auto [ret, rank, val] = nodeGM->search(iNode);
            if (rank != g_mpi.rank)
                ghostNodeSet.insert(iNode);
        }
    std::vector<DNDS::index> ghostNodes(ghostNodeSet.begin(), ghostNodeSet.end());

    n2c.son = make_ssp<decltype(n2c.son)::element_type>(ObjName{"n2c.son"}, g_mpi);
    n2c.TransAttach();
    n2c.trans.createFatherGlobalMapping();
    n2c.trans.createGhostMapping(ghostNodes);
    n2c.trans.createMPITypes();
    n2c.trans.pullOnce();

    // Build bGlobal2Local map for n2c
    std::unordered_map<DNDS::index, DNDS::index> nodeG2L;
    for (DNDS::index i = 0; i < n2c.Size(); i++)
        nodeG2L[n2c.trans.pLGhostMapping->operator()(-1, i)] = i;

    // Step 3: ComposeFiltered with SharedCountPredicate{1, removeSelf=true}
    auto c2c = MeshConnectivity::ComposeFiltered(
        c2n, n2c, nCellLocal, nodeG2L,
        [&](DNDS::index i)
        { return cellLocal2Global4Quad(g_mpi, i); },
        SharedCountPredicate{.minShared = 1, .removeSelf = true});

    // Gather and verify globally
    // Expected cell2cell (node-neighbor, self removed):
    //   cell 0: {1, 2, 3}  (shares node 1 with cell 1, node 3 with cell 2, node 4 with all)
    //   cell 1: {0, 2, 3}
    //   cell 2: {0, 1, 3}
    //   cell 3: {0, 1, 2}
    auto cellGM = std::make_shared<GlobalOffsetsMapping>();
    cellGM->setMPIAlignBcast(g_mpi, nCellLocal);

    // Gather local c2c results
    std::vector<std::set<DNDS::index>> globalC2C(4);
    for (DNDS::index i = 0; i < nCellLocal; i++)
    {
        DNDS::index cG = cellLocal2Global4Quad(g_mpi, i);
        for (auto v : c2c.father->operator[](i))
            globalC2C[cG].insert(v);
    }

    // Allgather to merge
    for (DNDS::index c = 0; c < 4; c++)
    {
        std::vector<DNDS::index> localVec(globalC2C[c].begin(), globalC2C[c].end());
        int localSize = static_cast<int>(localVec.size());
        std::vector<int> sizes(g_mpi.size);
        MPI_Allgather(&localSize, 1, MPI_INT, sizes.data(), 1, MPI_INT, g_mpi.comm);
        std::vector<int> disps(g_mpi.size + 1, 0);
        for (int r = 0; r < g_mpi.size; r++)
            disps[r + 1] = disps[r] + sizes[r];
        std::vector<DNDS::index> allVec(disps[g_mpi.size]);
        MPI_Allgatherv(localVec.data(), localSize, DNDS_MPI_INDEX,
                       allVec.data(), sizes.data(), disps.data(), DNDS_MPI_INDEX,
                       g_mpi.comm);
        globalC2C[c].clear();
        for (auto v : allVec)
            globalC2C[c].insert(v);
    }

    // For 4-quad: all cells share at least node 4, so every cell is neighbor of every other
    for (DNDS::index c = 0; c < 4; c++)
    {
        CAPTURE(c);
        CHECK(globalC2C[c].size() == 3);
        for (DNDS::index other = 0; other < 4; other++)
            if (other != c)
                CHECK(globalC2C[c].count(other) == 1);
    }
}

TEST_CASE("ComposeFiltered: face-share filter (minShared=dim)")
{
    // For 2D quads, face-share means >= 2 shared nodes.
    // In the 4-quad grid:
    //   cell 0 shares edge with cell 1 (nodes 1,4) and cell 2 (nodes 3,4)
    //   cell 0 shares only node 4 with cell 3 -> NOT face-neighbor
    //   cell 3 shares edge with cell 1 (nodes 4,5) and cell 2 (nodes 4,7)
    //   cell 3 shares only node 4 with cell 0 -> NOT face-neighbor

    auto c2n = make4QuadCell2Node(g_mpi);
    c2n.father->createGlobalMapping();
    auto nodeGM = makeNodeGlobalMapping4Quad(g_mpi);

    DNDS::index nNodeLocal = nodeLocalCount4Quad(g_mpi);
    DNDS::index nCellLocal = c2n.father->Size();

    auto n2c = MeshConnectivity::Inverse(
        c2n, nNodeLocal, g_mpi,
        [&](DNDS::index i)
        { return cellLocal2Global4Quad(g_mpi, i); },
        [&](DNDS::index i)
        { return nodeLocal2Global4Quad(g_mpi, i); },
        nodeGM);

    // Ghost-pull n2c
    std::unordered_set<DNDS::index> ghostNodeSet;
    for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
        for (auto iNode : c2n.father->operator[](iCell))
        {
            auto [ret, rank, val] = nodeGM->search(iNode);
            if (rank != g_mpi.rank)
                ghostNodeSet.insert(iNode);
        }
    std::vector<DNDS::index> ghostNodes(ghostNodeSet.begin(), ghostNodeSet.end());

    n2c.son = make_ssp<decltype(n2c.son)::element_type>(ObjName{"n2c.son"}, g_mpi);
    n2c.TransAttach();
    n2c.trans.createFatherGlobalMapping();
    n2c.trans.createGhostMapping(ghostNodes);
    n2c.trans.createMPITypes();
    n2c.trans.pullOnce();

    std::unordered_map<DNDS::index, DNDS::index> nodeG2L;
    for (DNDS::index i = 0; i < n2c.Size(); i++)
        nodeG2L[n2c.trans.pLGhostMapping->operator()(-1, i)] = i;

    // ComposeFiltered with minShared=2 (face-share for 2D)
    auto c2c_face = MeshConnectivity::ComposeFiltered(
        c2n, n2c, nCellLocal, nodeG2L,
        [&](DNDS::index i)
        { return cellLocal2Global4Quad(g_mpi, i); },
        SharedCountPredicate{.minShared = 2, .removeSelf = true});

    // Gather globally
    std::vector<std::set<DNDS::index>> globalC2CFace(4);
    for (DNDS::index i = 0; i < nCellLocal; i++)
    {
        DNDS::index cG = cellLocal2Global4Quad(g_mpi, i);
        for (auto v : c2c_face.father->operator[](i))
            globalC2CFace[cG].insert(v);
    }
    for (DNDS::index c = 0; c < 4; c++)
    {
        std::vector<DNDS::index> localVec(globalC2CFace[c].begin(), globalC2CFace[c].end());
        int localSize = static_cast<int>(localVec.size());
        std::vector<int> sizes(g_mpi.size);
        MPI_Allgather(&localSize, 1, MPI_INT, sizes.data(), 1, MPI_INT, g_mpi.comm);
        std::vector<int> disps(g_mpi.size + 1, 0);
        for (int r = 0; r < g_mpi.size; r++)
            disps[r + 1] = disps[r] + sizes[r];
        std::vector<DNDS::index> allVec(disps[g_mpi.size]);
        MPI_Allgatherv(localVec.data(), localSize, DNDS_MPI_INDEX,
                       allVec.data(), sizes.data(), disps.data(), DNDS_MPI_INDEX,
                       g_mpi.comm);
        globalC2CFace[c].clear();
        for (auto v : allVec)
            globalC2CFace[c].insert(v);
    }

    // Expected face-neighbors:
    //   cell 0: {1, 2}       (not 3, only vertex-share)
    //   cell 1: {0, 3}       (not 2, only vertex-share)
    //   cell 2: {0, 3}       (not 1, only vertex-share)
    //   cell 3: {1, 2}       (not 0, only vertex-share)
    std::vector<std::set<DNDS::index>> expected = {
        {1, 2},
        {0, 3},
        {0, 3},
        {1, 2},
    };

    for (DNDS::index c = 0; c < 4; c++)
    {
        CAPTURE(c);
        CHECK(globalC2CFace[c] == expected[c]);
    }
}


// ===========================================================================
// Regression tests
// ===========================================================================

// ===========================================================================
// Regression: DSL vs legacy pipeline on real mesh
// ===========================================================================

/// Build a mesh through the legacy pipeline up to node2cell state,
/// then compare with DSL Inverse.
TEST_CASE("Regression: Inverse matches RecoverNode2CellAndNode2Bnd on UniformSquare_10")
{
    // Build mesh via legacy pipeline
    auto mesh = std::make_shared<UnstructuredMesh>(g_mpi, 2);
    UnstructuredMeshSerialRW reader(mesh, 0);
    reader.ReadFromCGNSSerial(meshPath("UniformSquare_10.cgns"));
    reader.Deduplicate1to1Periodic(1e-8);
    reader.BuildCell2Cell();

    UnstructuredMeshSerialRW::PartitionOptions pOpt;
    pOpt.metisType = "KWAY";
    pOpt.metisUfactor = 30;
    pOpt.metisSeed = 42;
    pOpt.metisNcuts = 1;
    reader.MeshPartitionCell2Cell(pOpt);
    reader.PartitionReorderToMeshCell2Cell();

    // Legacy: build node2cell
    mesh->RecoverNode2CellAndNode2Bnd();

    // Snapshot the legacy node2cell
    DNDS::index nNodeLocal = mesh->coords.father->Size();
    std::vector<std::set<DNDS::index>> legacyN2C(nNodeLocal);
    for (DNDS::index i = 0; i < nNodeLocal; i++)
        for (auto v : mesh->node2cell.father->operator[](i))
            legacyN2C[i].insert(v);

    // DSL: Inverse
    if (!mesh->coords.father->pLGlobalMapping)
        mesh->coords.father->createGlobalMapping();
    if (!mesh->cell2node.father->pLGlobalMapping)
        mesh->cell2node.father->createGlobalMapping();

    auto dslN2C = MeshConnectivity::Inverse(
        mesh->cell2node, nNodeLocal, g_mpi,
        [&](DNDS::index i)
        { return mesh->CellIndexLocal2Global_NoSon(i); },
        [&](DNDS::index i)
        { return mesh->NodeIndexLocal2Global_NoSon(i); },
        mesh->coords.father->pLGlobalMapping);

    // Compare: for each local node, the set of global cells must match
    for (DNDS::index i = 0; i < nNodeLocal; i++)
    {
        std::set<DNDS::index> dslSet;
        for (auto v : dslN2C.father->operator[](i))
            dslSet.insert(v);
        CAPTURE(i);
        CHECK(dslSet == legacyN2C[i]);
    }
}

TEST_CASE("Regression: ComposeFiltered matches RecoverCell2CellAndBnd2Cell on UniformSquare_10")
{
    // Build mesh via legacy pipeline
    auto mesh = std::make_shared<UnstructuredMesh>(g_mpi, 2);
    UnstructuredMeshSerialRW reader(mesh, 0);
    reader.ReadFromCGNSSerial(meshPath("UniformSquare_10.cgns"));
    reader.Deduplicate1to1Periodic(1e-8);
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

    // Snapshot legacy cell2cell (global indices, in Adj_PointToGlobal state)
    DNDS::index nCellLocal = mesh->cell2node.father->Size();
    std::vector<std::set<DNDS::index>> legacyC2C(nCellLocal);
    for (DNDS::index i = 0; i < nCellLocal; i++)
        for (auto v : mesh->cell2cell.father->operator[](i))
            legacyC2C[i].insert(v);

    // DSL approach: use Inverse to get node2cell, ghost-pull, then ComposeFiltered

    if (!mesh->coords.father->pLGlobalMapping)
        mesh->coords.father->createGlobalMapping();
    if (!mesh->cell2node.father->pLGlobalMapping)
        mesh->cell2node.father->createGlobalMapping();

    DNDS::index nNodeLocal = mesh->coords.father->Size();

    auto dslN2C = MeshConnectivity::Inverse(
        mesh->cell2node, nNodeLocal, g_mpi,
        [&](DNDS::index i)
        { return mesh->CellIndexLocal2Global_NoSon(i); },
        [&](DNDS::index i)
        { return mesh->NodeIndexLocal2Global_NoSon(i); },
        mesh->coords.father->pLGlobalMapping);

    // Ghost-pull dslN2C for off-rank nodes
    std::unordered_set<DNDS::index> ghostNodeSet;
    for (DNDS::index i = 0; i < nCellLocal; i++)
        for (auto iNode : mesh->cell2node.father->operator[](i))
        {
            auto [ret, rank, val] = mesh->coords.father->pLGlobalMapping->search(iNode);
            if (rank != g_mpi.rank)
                ghostNodeSet.insert(iNode);
        }
    std::vector<DNDS::index> ghostNodes(ghostNodeSet.begin(), ghostNodeSet.end());

    dslN2C.son = make_ssp<decltype(dslN2C.son)::element_type>(ObjName{"dslN2C.son"}, g_mpi);
    dslN2C.TransAttach();
    dslN2C.trans.createFatherGlobalMapping();
    dslN2C.trans.createGhostMapping(ghostNodes);
    dslN2C.trans.createMPITypes();
    dslN2C.trans.pullOnce();

    // Build bGlobal2Local map
    std::unordered_map<DNDS::index, DNDS::index> nodeG2L;
    for (DNDS::index i = 0; i < dslN2C.Size(); i++)
        nodeG2L[dslN2C.trans.pLGhostMapping->operator()(-1, i)] = i;

    // Verify all nodes in cell2node are in the map
    for (DNDS::index i = 0; i < nCellLocal; i++)
        for (auto iNode : mesh->cell2node.father->operator[](i))
            REQUIRE(nodeG2L.count(iNode));

    // ComposeFiltered with SharedCountPredicate{1, removeSelf=true} -> cell2cell (node-neighbor)
    auto dslC2C = MeshConnectivity::ComposeFiltered(
        mesh->cell2node, dslN2C, nCellLocal, nodeG2L,
        [&](DNDS::index i)
        { return mesh->CellIndexLocal2Global_NoSon(i); },
        SharedCountPredicate{.minShared = 1, .removeSelf = true});

    // Compare
    for (DNDS::index i = 0; i < nCellLocal; i++)
    {
        std::set<DNDS::index> dslSet;
        for (auto v : dslC2C.father->operator[](i))
            dslSet.insert(v);
        CAPTURE(i);
        CHECK(dslSet == legacyC2C[i]);
    }
}

// ===========================================================================
// MeshConnectivity struct management
// ===========================================================================

// ===========================================================================
// Cone / Support management tests
// ===========================================================================

TEST_CASE("MeshConnectivity: cone management")
{
    MeshConnectivity dag;
    dag.meshDim = 2;

    CHECK(!dag.hasCone(2, 0));
    auto &c = dag.addCone(2, 0);
    CHECK(dag.hasCone(2, 0));
    CHECK(!dag.hasCone(0, 2));
    CHECK(c.fromDepth == 2);
    CHECK(c.toDepth == 0);

    auto *found = dag.findCone(2, 0);
    CHECK(found == &c);
    CHECK(dag.findCone(0, 2) == nullptr);

    auto &c2 = dag.addCone(1, 0);
    CHECK(dag.hasCone(1, 0));
    CHECK(dag.cones.size() == 2);
}

TEST_CASE("MeshConnectivity: support management")
{
    MeshConnectivity dag;
    dag.meshDim = 2;

    CHECK(!dag.hasSupport(0, 2));
    auto &s = dag.addSupport(0, 2);
    CHECK(dag.hasSupport(0, 2));
    CHECK(!dag.hasSupport(2, 0));
    CHECK(s.fromDepth == 0);
    CHECK(s.toDepth == 2);

    CHECK(dag.findSupport(0, 2) == &s);
    CHECK(dag.findSupport(2, 0) == nullptr);
    CHECK(dag.supports.size() == 1);
}

TEST_CASE("ConeAdj: AdjVariant typed access")
{
    ConeAdj cone;
    cone.fromDepth = 2;
    cone.toDepth = 0;
    // Default-constructed ConeAdj has adj == nullptr (ssp<AdjVariant>)
    CHECK(!cone.adj);
    CHECK(!cone.initialized());
    CHECK(!cone.hasPbi());

    // Initialize with a tAdjPair via makeAdjVariant
    cone.adj = makeAdjVariant<tAdjPair>();
    CHECK(cone.adj);
    CHECK(std::holds_alternative<tAdjPair>(*cone.adj));
    CHECK(!cone.initialized()); // father is still null inside the pair

    // Initialize with a tAdj2Pair variant
    ConeAdj cone2;
    cone2.fromDepth = 1;
    cone2.toDepth = 2;
    cone2.adj = makeAdjVariant<tAdj2Pair>();
    CHECK(std::holds_alternative<tAdj2Pair>(*cone2.adj));
}

TEST_CASE("SupportAdj: no pbi member")
{
    SupportAdj sup;
    sup.fromDepth = 0;
    sup.toDepth = 2;
    CHECK(!sup.initialized());
    // SupportAdj has no pbi member -- compile-time check that it's absent
    // (this test just verifies the struct compiles correctly)
    CHECK(sup.fatherSize() == 0);
}

// ===========================================================================
// Fixed-width template tests
// ===========================================================================

TEST_CASE("Inverse<2>: fixed-width face2cell input")
{
    // Build a tiny mesh: 2 triangles sharing an edge.
    // face2cell: face0→{0}, face1→{0,1}, face2→{0}, face3→{1}, face4→{1}
    // (face1 is the shared internal face with 2 parents.)
    //
    // We construct face2cell as a tAdj2Pair (fixed-2) and call Inverse<2>
    // to produce cell2face (variable-width).

    DNDS::index nFaces = (g_mpi.rank == 0) ? 5 : 0;
    DNDS::index nCells = (g_mpi.rank == 0) ? 2 : 0;

    tAdj2Pair f2c;
    f2c.InitPair("f2c", g_mpi);
    f2c.father->Resize(nFaces);
    if (g_mpi.rank == 0)
    {
        // boundary face 0: cell 0 only
        f2c.father->operator()(0, 0) = 0;
        f2c.father->operator()(0, 1) = UnInitIndex;
        // internal face 1: cells 0 and 1
        f2c.father->operator()(1, 0) = 0;
        f2c.father->operator()(1, 1) = 1;
        // boundary faces 2, 3, 4
        f2c.father->operator()(2, 0) = 0;
        f2c.father->operator()(2, 1) = UnInitIndex;
        f2c.father->operator()(3, 0) = 1;
        f2c.father->operator()(3, 1) = UnInitIndex;
        f2c.father->operator()(4, 0) = 1;
        f2c.father->operator()(4, 1) = UnInitIndex;
    }

    auto cellGM = std::make_shared<GlobalOffsetsMapping>();
    cellGM->setMPIAlignBcast(g_mpi, nCells);

    // Call Inverse with fixed-width template param: Inverse<2>
    auto c2f = MeshConnectivity::Inverse<2>(
        f2c, nCells, g_mpi,
        [&](DNDS::index i) -> DNDS::index
        {
            // face local2global: identity (rank 0 has all faces)
            if (g_mpi.rank == 0) return i;
            return -1;
        },
        [&](DNDS::index i) -> DNDS::index
        {
            // cell local2global: identity
            if (g_mpi.rank == 0) return i;
            return -1;
        },
        cellGM);

    if (g_mpi.rank == 0)
    {
        // Verify: cell 0 should have faces {0, 1, 2}, cell 1 should have faces {1, 3, 4}
        CHECK(c2f.father->Size() == 2);

        std::set<DNDS::index> c0faces, c1faces;
        for (auto f : c2f.father->operator[](0))
            c0faces.insert(f);
        for (auto f : c2f.father->operator[](1))
            c1faces.insert(f);

        CHECK(c0faces == std::set<DNDS::index>{0, 1, 2});
        CHECK(c1faces == std::set<DNDS::index>{1, 3, 4});
    }
}

TEST_CASE("narrowAdjToFixed: variable-to-fixed-2 conversion")
{
    // Build a variable-width adjacency with rows of sizes 1 and 2.
    tAdjPair source;
    source.InitPair("source", g_mpi);
    source.father->Resize(3);
    source.father->ResizeRow(0, 2);
    source.father->operator()(0, 0) = 10;
    source.father->operator()(0, 1) = 20;
    source.father->ResizeRow(1, 1);
    source.father->operator()(1, 0) = 30;
    source.father->ResizeRow(2, 2);
    source.father->operator()(2, 0) = 40;
    source.father->operator()(2, 1) = 50;

    tAdj2Pair target;
    target.InitPair("target", g_mpi);
    target.father->Resize(3);

    narrowAdjToFixed<2>(source, target, 3);

    CHECK(target.father->operator()(0, 0) == 10);
    CHECK(target.father->operator()(0, 1) == 20);
    CHECK(target.father->operator()(1, 0) == 30);
    CHECK(target.father->operator()(1, 1) == UnInitIndex);
    CHECK(target.father->operator()(2, 0) == 40);
    CHECK(target.father->operator()(2, 1) == 50);
}

// ===========================================================================
// Periodic ComposeFiltered test
// ===========================================================================

TEST_CASE("Periodic 2x2x2: ComposeFiltered cell2cellFace is WRONG without pbi filter")
{
    // This test uses MPI-collective operations (Inverse, ghost comm),
    // so all ranks must participate. Only rank 0 has data.
    DNDS::index nCells = (g_mpi.rank == 0) ? 8 : 0;
    DNDS::index nNodes = (g_mpi.rank == 0) ? 8 : 0;

    tAdjPair c2n;
    c2n.InitPair("p2x2x2_c2n_compose", g_mpi);
    c2n.father->Resize(nCells);

    if (g_mpi.rank == 0)
    {
        auto pm = makePeriodic2x2x2Mesh(g_mpi);
        // Copy data from the full mesh
        for (DNDS::index i = 0; i < 8; i++)
        {
            c2n.father->ResizeRow(i, 8);
            for (DNDS::rowsize j = 0; j < 8; j++)
                c2n.father->operator()(i, j) = pm.cell2node.father->operator()(i, j);
        }
    }

    c2n.father->createGlobalMapping();
    auto nodeGM = std::make_shared<GlobalOffsetsMapping>();
    nodeGM->setMPIAlignBcast(g_mpi, nNodes);

    auto n2c = MeshConnectivity::Inverse(
        c2n, nNodes, g_mpi,
        [](DNDS::index i) { return i; },
        [](DNDS::index i) { return i; },
        nodeGM);

    n2c.son = make_ssp<decltype(n2c.son)::element_type>(ObjName{"n2c.son"}, g_mpi);
    n2c.TransAttach();
    n2c.trans.createFatherGlobalMapping();
    std::vector<DNDS::index> emptyGhost;
    n2c.trans.createGhostMapping(emptyGhost);
    n2c.trans.createMPITypes();
    n2c.trans.pullOnce();

    std::unordered_map<DNDS::index, DNDS::index> nodeG2L;
    for (DNDS::index i = 0; i < n2c.Size(); i++)
        nodeG2L[n2c.trans.pLGhostMapping->operator()(-1, i)] = i;

    auto c2cFace = MeshConnectivity::ComposeFiltered(
        c2n, n2c, nCells, nodeG2L,
        [](DNDS::index i) { return i; },
        SharedCountPredicate{.minShared = 3, .removeSelf = true});

    // Only rank 0 has cells to verify
    if (g_mpi.rank == 0)
    {
        // Without pbi filter: each cell has 7 neighbors (WRONG, should be 3)
        for (DNDS::index i = 0; i < 8; i++)
        {
            CAPTURE(i);
            CHECK(c2cFace.father->RowSize(i) == 7);
        }
    }
}

// ===========================================================================
// ComposeFiltered with matchExtra: pbi containment check for bnd2cell
// ===========================================================================

/// Build a pbi containment matchExtra for ComposeFiltered.
/// For bnd2cell: checks that every (node, pbi) pair in the A-entity (bnd)
/// appears in the C-entity (cell). This is stricter than uniform XOR —
/// it requires exact pbi matching per node, not just consistent shift.
///
/// @param a2nodePbi  A-entity → node pbi (bnd2nodePbi).
/// @param c2node     C-entity → nodes (cell2node, with ghost for lookup).
/// @param c2nodePbi  C-entity → node pbi (cell2nodePbi, with ghost).
/// @param cGlobal2Local  Maps global C-index to local-appended index.
static std::function<bool(DNDS::index, DNDS::index, const std::vector<DNDS::index> &)>
makePbiContainmentMatchExtra(
    const tAdjPair &a2node,
    const tPbiPair &a2nodePbi,
    const tAdjPair &c2node,
    const tPbiPair &c2nodePbi,
    const std::unordered_map<DNDS::index, DNDS::index> &cGlobal2Local)
{
    return [&](DNDS::index aLocal, DNDS::index cGlobal,
               const std::vector<DNDS::index> & /*sharedNodes*/) -> bool
    {
        auto itC = cGlobal2Local.find(cGlobal);
        if (itC == cGlobal2Local.end())
            return false;
        DNDS::index cLocal = itC->second;

        // For every node in A's row, check that the same (node, pbi) pair
        // exists in C's row.
        for (DNDS::rowsize ia = 0; ia < a2node.father->RowSize(aLocal); ia++)
        {
            DNDS::index nodeA = a2node.father->operator()(aLocal, ia);
            uint8_t pbiA = uint8_t(a2nodePbi.father->operator()(aLocal, ia));
            bool found = false;
            auto cRow = c2node[cLocal];
            for (DNDS::rowsize ic = 0; ic < cRow.size(); ic++)
            {
                if (cRow[ic] == nodeA && uint8_t(c2nodePbi(cLocal, ic)) == pbiA)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
                return false;
        }
        return true;
    };
}

/// Verify that cell2cellFace on periodic meshes should be derived from
/// Interpolate (face→cell), not from ComposeFiltered. The compose-based
/// approach cannot correctly distinguish face-neighbors from non-face-neighbors
/// when all cells share all nodes (as on 2×2 doubly-periodic or 2×2×2
/// triply-periodic meshes).
///
/// The correct cell2cellFace derivation is tested in
/// "Periodic 2x2: cell2face from Interpolate" and
/// "Periodic 2x2x2: face interpolation" tests above.


// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    MPI_Finalize();
    return res;
}
