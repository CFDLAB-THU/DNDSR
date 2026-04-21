/**
 * @file test_MeshConnectivity.cpp
 * @brief Unit tests for MeshConnectivity DSL operations: Inverse, Compose, ComposeFiltered, Interpolate.
 *
 * Tests:
 *   - Inverse: hand-crafted 4-quad mesh, verify node2cell correctness
 *   - Inverse: MPI partitioned mesh (UniformSquare_10), verify globally-complete node2cell
 *   - Inverse: round-trip inverse(inverse(cell2node)) covers original cell2node
 *   - Compose: cell2node + node2cell -> cell2cell (node-neighbor)
 *   - ComposeFiltered: with SharedCountPredicate, verify against known cell2cell
 *   - ComposeFiltered: bnd2node + node2cell -> bnd2cell with face-share filter
 *   - Regression: DSL Inverse matches RecoverNode2CellAndNode2Bnd on real mesh
 *   - Regression: DSL ComposeFiltered matches RecoverCell2CellAndBnd2Cell on real mesh
 *   - Interpolate: 4-quad mesh face extraction (2D)
 *   - Interpolate: 2-tri mesh face extraction (2D)
 *   - Interpolate: single tet face extraction (3D)
 *   - Interpolate: two tets sharing a face (3D)
 *   - Interpolate: single hex face extraction (3D)
 *   - Interpolate: edge extraction from single tet (3D)
 *   - Interpolate: regression vs legacy InterpolateFace on real mesh
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

/// Build a small hand-crafted 4-quad mesh on a single rank.
///
/// Layout (nodes 0-8, cells 0-3):
///
///     6---7---8
///     |   |   |
///     | 2 | 3 |
///     |   |   |
///     3---4---5
///     |   |   |
///     | 0 | 1 |
///     |   |   |
///     0---1---2
///
/// cell2node:
///   cell 0: [0, 1, 4, 3]
///   cell 1: [1, 2, 5, 4]
///   cell 2: [3, 4, 7, 6]
///   cell 3: [4, 5, 8, 7]
///
static tAdjPair make4QuadCell2Node(const MPIInfo &mpi)
{
    tAdjPair c2n;
    c2n.InitPair("test_c2n", mpi);

    if (mpi.size == 1)
    {
        c2n.father->Resize(4);
        int data[4][4] = {
            {0, 1, 4, 3},
            {1, 2, 5, 4},
            {3, 4, 7, 6},
            {4, 5, 8, 7}};
        for (DNDS::index i = 0; i < 4; i++)
        {
            c2n.father->ResizeRow(i, 4);
            for (DNDS::rowsize j = 0; j < 4; j++)
                c2n.father->operator()(i, j) = data[i][j];
        }
    }
    else if (mpi.size == 2)
    {
        // Partition: rank 0 owns cells 0,1; rank 1 owns cells 2,3
        c2n.father->Resize(2);
        if (mpi.rank == 0)
        {
            int data[2][4] = {
                {0, 1, 4, 3},
                {1, 2, 5, 4}};
            for (DNDS::index i = 0; i < 2; i++)
            {
                c2n.father->ResizeRow(i, 4);
                for (DNDS::rowsize j = 0; j < 4; j++)
                    c2n.father->operator()(i, j) = data[i][j];
            }
        }
        else
        {
            int data[2][4] = {
                {3, 4, 7, 6},
                {4, 5, 8, 7}};
            for (DNDS::index i = 0; i < 2; i++)
            {
                c2n.father->ResizeRow(i, 4);
                for (DNDS::rowsize j = 0; j < 4; j++)
                    c2n.father->operator()(i, j) = data[i][j];
            }
        }
    }
    else
    {
        // For np>2, only rank 0 and 1 have cells; others have 0
        if (mpi.rank == 0)
        {
            c2n.father->Resize(2);
            int data[2][4] = {
                {0, 1, 4, 3},
                {1, 2, 5, 4}};
            for (DNDS::index i = 0; i < 2; i++)
            {
                c2n.father->ResizeRow(i, 4);
                for (DNDS::rowsize j = 0; j < 4; j++)
                    c2n.father->operator()(i, j) = data[i][j];
            }
        }
        else if (mpi.rank == 1)
        {
            c2n.father->Resize(2);
            int data[2][4] = {
                {3, 4, 7, 6},
                {4, 5, 8, 7}};
            for (DNDS::index i = 0; i < 2; i++)
            {
                c2n.father->ResizeRow(i, 4);
                for (DNDS::rowsize j = 0; j < 4; j++)
                    c2n.father->operator()(i, j) = data[i][j];
            }
        }
        else
        {
            c2n.father->Resize(0);
        }
    }
    return c2n;
}

/// Create a node distribution for the 9-node grid.
/// For np=1: rank 0 owns all 9.
/// For np=2: rank 0 owns nodes 0-4, rank 1 owns nodes 5-8.
/// For np>2: rank 0 owns 0-4, rank 1 owns 5-8, others own 0.
static DNDS::index nodeLocalCount4Quad(const MPIInfo &mpi)
{
    if (mpi.size == 1)
        return 9;
    if (mpi.rank == 0)
        return 5; // nodes 0-4
    if (mpi.rank == 1)
        return 4; // nodes 5-8
    return 0;
}

static DNDS::index nodeLocal2Global4Quad(const MPIInfo &mpi, DNDS::index local)
{
    if (mpi.size == 1)
        return local;
    if (mpi.rank == 0)
        return local;
    if (mpi.rank == 1)
        return local + 5;
    return -1; // should not be called
}

static ssp<GlobalOffsetsMapping> makeNodeGlobalMapping4Quad(const MPIInfo &mpi)
{
    auto gm = std::make_shared<GlobalOffsetsMapping>();
    gm->setMPIAlignBcast(mpi, nodeLocalCount4Quad(mpi));
    return gm;
}

static DNDS::index cellLocal2Global4Quad(const MPIInfo &mpi, DNDS::index local)
{
    if (mpi.size == 1)
        return local;
    // np>=2: rank 0 owns cells 0-1, rank 1 owns cells 2-3
    if (mpi.rank == 0)
        return local;
    if (mpi.rank == 1)
        return local + 2;
    return -1;
}

/// Collect all (to-global -> set-of-from-globals) across all ranks for verification.
static std::vector<std::set<DNDS::index>> gatherInverseGlobal(
    const tAdjPair &support, DNDS::index nToLocal,
    const std::function<DNDS::index(DNDS::index)> &toLocal2Global,
    DNDS::index nToGlobal, const MPIInfo &mpi)
{
    // Gather local data
    // Each rank packs: [toGlobal, count, from0, from1, ...]
    std::vector<DNDS::index> localPacked;
    for (DNDS::index i = 0; i < nToLocal; i++)
    {
        DNDS::index toG = toLocal2Global(i);
        auto row = support.father->operator[](i);
        localPacked.push_back(toG);
        localPacked.push_back(row.size());
        for (auto v : row)
            localPacked.push_back(v);
    }

    // Gather sizes
    int localSize = static_cast<int>(localPacked.size());
    std::vector<int> sizes(mpi.size);
    MPI_Allgather(&localSize, 1, MPI_INT, sizes.data(), 1, MPI_INT, mpi.comm);

    std::vector<int> disps(mpi.size + 1, 0);
    for (int r = 0; r < mpi.size; r++)
        disps[r + 1] = disps[r] + sizes[r];

    std::vector<DNDS::index> allPacked(disps[mpi.size]);
    MPI_Allgatherv(localPacked.data(), localSize, DNDS_MPI_INDEX,
                   allPacked.data(), sizes.data(), disps.data(), DNDS_MPI_INDEX,
                   mpi.comm);

    // Unpack into global result
    std::vector<std::set<DNDS::index>> result(nToGlobal);
    DNDS::index pos = 0;
    while (pos < static_cast<DNDS::index>(allPacked.size()))
    {
        DNDS::index toG = allPacked[pos++];
        DNDS::index count = allPacked[pos++];
        for (DNDS::index k = 0; k < count; k++)
            result[toG].insert(allPacked[pos++]);
    }
    return result;
}

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
    // Default-constructed variant holds tAdjPair (index 0)
    CHECK(std::holds_alternative<tAdjPair>(cone.adj));
    CHECK(!cone.initialized());
    CHECK(!cone.hasPbi());

    // Initialize with a tAdj2Pair variant
    ConeAdj cone2;
    cone2.fromDepth = 1;
    cone2.toDepth = 2;
    cone2.adj = tAdj2Pair{};
    CHECK(std::holds_alternative<tAdj2Pair>(cone2.adj));
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
// Interpolate Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: build SubEntityQuery from Element topology API
// ---------------------------------------------------------------------------

/// Build a SubEntityQuery that extracts faces (dim-1 sub-entities)
/// from parent elements described by parentElemInfo.
static SubEntityQuery makeFaceQuery(const tElemInfoArrayPair &parentElemInfo)
{
    SubEntityQuery q;
    q.numSubEntities = [&parentElemInfo](DNDS::index iParent) -> int
    {
        auto e = Elem::Element{parentElemInfo[iParent]->getElemType()};
        return e.GetNumFaces();
    };
    q.describe = [&parentElemInfo](DNDS::index iParent, int iSub) -> SubEntityDesc
    {
        auto eParent = Elem::Element{parentElemInfo[iParent]->getElemType()};
        auto eFace = eParent.ObtainFace(iSub);
        return SubEntityDesc{eFace.GetNumVertices(), eFace.GetNumNodes(),
                             static_cast<t_index>(eFace.type)};
    };
    q.extractNodes = [&parentElemInfo](DNDS::index iParent, int iSub,
                                        const std::function<DNDS::index(int)> &parentNodes,
                                        DNDS::index *out)
    {
        auto eParent = Elem::Element{parentElemInfo[iParent]->getElemType()};
        auto eFace = eParent.ObtainFace(iSub);
        // Build a temporary vector of parent nodes to pass to ExtractFaceNodes
        std::vector<DNDS::index> pNodes(eParent.GetNumNodes());
        for (int i = 0; i < eParent.GetNumNodes(); i++)
            pNodes[i] = parentNodes(i);
        std::vector<DNDS::index> fNodes(eFace.GetNumNodes());
        eParent.ExtractFaceNodes(iSub, pNodes, fNodes);
        for (int i = 0; i < eFace.GetNumNodes(); i++)
            out[i] = fNodes[i];
    };
    return q;
}

/// Build a SubEntityQuery that extracts edges (1D sub-entities)
/// from 3D parent elements described by parentElemInfo.
static SubEntityQuery makeEdgeQuery(const tElemInfoArrayPair &parentElemInfo)
{
    SubEntityQuery q;
    q.numSubEntities = [&parentElemInfo](DNDS::index iParent) -> int
    {
        auto e = Elem::Element{parentElemInfo[iParent]->getElemType()};
        return e.GetNumEdges();
    };
    q.describe = [&parentElemInfo](DNDS::index iParent, int iSub) -> SubEntityDesc
    {
        auto eParent = Elem::Element{parentElemInfo[iParent]->getElemType()};
        auto eEdge = eParent.ObtainEdge(iSub);
        return SubEntityDesc{eEdge.GetNumVertices(), eEdge.GetNumNodes(),
                             static_cast<t_index>(eEdge.type)};
    };
    q.extractNodes = [&parentElemInfo](DNDS::index iParent, int iSub,
                                        const std::function<DNDS::index(int)> &parentNodes,
                                        DNDS::index *out)
    {
        auto eParent = Elem::Element{parentElemInfo[iParent]->getElemType()};
        auto eEdge = eParent.ObtainEdge(iSub);
        std::vector<DNDS::index> pNodes(eParent.GetNumNodes());
        for (int i = 0; i < eParent.GetNumNodes(); i++)
            pNodes[i] = parentNodes(i);
        std::vector<DNDS::index> eNodes(eEdge.GetNumNodes());
        eParent.ExtractEdgeNodes(iSub, pNodes, eNodes);
        for (int i = 0; i < eEdge.GetNumNodes(); i++)
            out[i] = eNodes[i];
    };
    return q;
}

/// Helper: build cell2node and cellElemInfo for a hand-crafted mesh on rank 0 only.
/// cells: vector of (ElemType, vector<node indices>).
/// Returns (cell2node, cellElemInfo, nNodes).
struct HandCraftedMesh
{
    tAdjPair cell2node;
    tElemInfoArrayPair cellElemInfo;
    DNDS::index nNodes;
};

static HandCraftedMesh makeHandCraftedMesh(
    const MPIInfo &mpi,
    const std::vector<std::pair<Elem::ElemType, std::vector<DNDS::index>>> &cells,
    DNDS::index nNodes)
{
    HandCraftedMesh m;
    m.nNodes = nNodes;

    m.cell2node.InitPair("test_c2n", mpi);
    m.cellElemInfo.InitPair("test_cei", mpi);

    DNDS::index nCells = static_cast<DNDS::index>(cells.size());
    m.cell2node.father->Resize(nCells);
    m.cellElemInfo.father->Resize(nCells);

    for (DNDS::index i = 0; i < nCells; i++)
    {
        auto &[et, nodes] = cells[i];
        m.cell2node.father->ResizeRow(i, nodes.size());
        for (DNDS::rowsize j = 0; j < static_cast<DNDS::rowsize>(nodes.size()); j++)
            m.cell2node.father->operator()(i, j) = nodes[j];

        ElemInfo ei;
        ei.setElemType(et);
        ei.zone = 0;
        m.cellElemInfo(i, 0) = ei;
    }

    return m;
}

/// Collect sorted vertex set from an entity's node row in entity2node.
static std::set<DNDS::index> getEntityVertexSet(
    const tAdjPair &entity2node,
    const std::vector<ElemInfo> &entityElemInfo,
    DNDS::index iEntity)
{
    auto eEnt = Elem::Element{entityElemInfo[iEntity].getElemType()};
    int nVerts = eEnt.GetNumVertices();
    std::set<DNDS::index> verts;
    for (int j = 0; j < nVerts; j++)
        verts.insert(entity2node.father->operator()(iEntity, j));
    return verts;
}

// ---------------------------------------------------------------------------
// Interpolate: 4-quad mesh (2D) — face extraction
// ---------------------------------------------------------------------------

TEST_CASE("Interpolate: 4-quad faces (2D)")
{
    // Only run on rank 0 (serial test, other ranks have 0 cells)
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    //     6---7---8
    //     |   |   |
    //     | 2 | 3 |
    //     |   |   |
    //     3---4---5
    //     |   |   |
    //     | 0 | 1 |
    //     |   |   |
    //     0---1---2
    //
    // Quad4 face table: {0,1}, {1,2}, {2,3}, {3,0}

    auto m = makeHandCraftedMesh(g_mpi,
        {
            {Elem::Quad4, {0, 1, 4, 3}},
            {Elem::Quad4, {1, 2, 5, 4}},
            {Elem::Quad4, {3, 4, 7, 6}},
            {Elem::Quad4, {4, 5, 8, 7}},
        }, 9);

    auto query = makeFaceQuery(m.cellElemInfo);
    auto res = MeshConnectivity::Interpolate(m.cell2node, query, 4, 9, g_mpi);

    // 4 quads × 4 edges = 16 half-edges, with 4 internal shared edges.
    // Unique edges: 16 - 4 = 12
    CHECK(res.nEntities == 12);

    // Each cell should reference exactly 4 entities
    for (DNDS::index i = 0; i < 4; i++)
    {
        CAPTURE(i);
        CHECK(res.parent2entity.father->RowSize(i) == 4);
    }

    // All entities should be Line2
    for (DNDS::index i = 0; i < res.nEntities; i++)
    {
        CAPTURE(i);
        CHECK(res.entityElemInfo[i].getElemType() == Elem::Line2);
    }

    // Internal edges: count entities with both parents set (second != UnInitIndex)
    int nInternal = 0;
    for (DNDS::index i = 0; i < res.nEntities; i++)
        if (res.entity2parent.father->operator()(i, 1) != DNDS::UnInitIndex)
            nInternal++;
    CHECK(nInternal == 4);

    // Boundary edges: 12 - 4 = 8
    CHECK(res.nEntities - nInternal == 8);

    // Verify no duplicate vertex sets
    std::set<std::set<DNDS::index>> allVertSets;
    for (DNDS::index i = 0; i < res.nEntities; i++)
    {
        auto vs = getEntityVertexSet(res.entity2node, res.entityElemInfo, i);
        CHECK(allVertSets.count(vs) == 0);
        allVertSets.insert(vs);
    }

    // Verify entity2node has 2 nodes per entity (Line2)
    for (DNDS::index i = 0; i < res.nEntities; i++)
    {
        CAPTURE(i);
        CHECK(res.entity2node.father->RowSize(i) == 2);
    }

    // Verify parent2entity → entity2parent consistency
    for (DNDS::index iCell = 0; iCell < 4; iCell++)
    {
        for (DNDS::rowsize j = 0; j < res.parent2entity.father->RowSize(iCell); j++)
        {
            DNDS::index iEnt = res.parent2entity.father->operator()(iCell, j);
            CAPTURE(iCell);
            CAPTURE(j);
            CAPTURE(iEnt);
            auto p0 = res.entity2parent.father->operator()(iEnt, 0);
            auto p1 = res.entity2parent.father->operator()(iEnt, 1);
            CHECK((p0 == iCell || p1 == iCell));
        }
    }
}

// ---------------------------------------------------------------------------
// Interpolate: 2-tri mesh (2D) — face extraction
// ---------------------------------------------------------------------------

TEST_CASE("Interpolate: 2-tri faces (2D)")
{
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    //     2---3
    //     |\ 1|
    //     | \ |
    //     |0 \|
    //     0---1
    //
    // Tri3 face table: {0,1}, {1,2}, {2,0}
    // Cell 0: nodes {0,1,2} → faces: {0,1}, {1,2}, {2,0}
    // Cell 1: nodes {1,3,2} → faces: {1,3}, {3,2}, {2,1}
    // Shared edge: {1,2}
    // Total unique: 3 + 3 - 1 = 5

    auto m = makeHandCraftedMesh(g_mpi,
        {
            {Elem::Tri3, {0, 1, 2}},
            {Elem::Tri3, {1, 3, 2}},
        }, 4);

    auto query = makeFaceQuery(m.cellElemInfo);
    auto res = MeshConnectivity::Interpolate(m.cell2node, query, 2, 4, g_mpi);

    CHECK(res.nEntities == 5);

    // One internal edge (shared)
    int nInternal = 0;
    for (DNDS::index i = 0; i < res.nEntities; i++)
        if (res.entity2parent.father->operator()(i, 1) != DNDS::UnInitIndex)
            nInternal++;
    CHECK(nInternal == 1);

    // The shared edge should have vertex set {1, 2}
    for (DNDS::index i = 0; i < res.nEntities; i++)
    {
        if (res.entity2parent.father->operator()(i, 1) != DNDS::UnInitIndex)
        {
            auto vs = getEntityVertexSet(res.entity2node, res.entityElemInfo, i);
            CHECK(vs == std::set<DNDS::index>{1, 2});
        }
    }
}

// ---------------------------------------------------------------------------
// Interpolate: single tet (3D) — face extraction
// ---------------------------------------------------------------------------

TEST_CASE("Interpolate: single tet faces (3D)")
{
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    // Tet4: nodes {0,1,2,3}, 4 Tri3 faces
    // face 0: {0,2,1}, face 1: {0,1,3}, face 2: {1,2,3}, face 3: {2,0,3}
    auto m = makeHandCraftedMesh(g_mpi,
        {
            {Elem::Tet4, {0, 1, 2, 3}},
        }, 4);

    auto query = makeFaceQuery(m.cellElemInfo);
    auto res = MeshConnectivity::Interpolate(m.cell2node, query, 1, 4, g_mpi);

    CHECK(res.nEntities == 4);

    // All faces Tri3
    for (DNDS::index i = 0; i < 4; i++)
    {
        CHECK(res.entityElemInfo[i].getElemType() == Elem::Tri3);
        CHECK(res.entity2node.father->RowSize(i) == 3);
    }

    // All boundary (single parent)
    for (DNDS::index i = 0; i < 4; i++)
        CHECK(res.entity2parent.father->operator()(i, 1) == DNDS::UnInitIndex);

    // Verify face vertex sets match Tet4 topology
    std::set<std::set<DNDS::index>> expectedFaces = {
        {0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}};
    std::set<std::set<DNDS::index>> actualFaces;
    for (DNDS::index i = 0; i < 4; i++)
        actualFaces.insert(getEntityVertexSet(res.entity2node, res.entityElemInfo, i));
    CHECK(actualFaces == expectedFaces);
}

// ---------------------------------------------------------------------------
// Interpolate: two tets sharing a face (3D)
// ---------------------------------------------------------------------------

TEST_CASE("Interpolate: two tets sharing a face (3D)")
{
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    // Tet A: {0,1,2,3}  Tet B: {1,2,3,4}
    // Shared face: vertices {1,2,3}
    // Total unique faces: 4 + 4 - 1 = 7
    auto m = makeHandCraftedMesh(g_mpi,
        {
            {Elem::Tet4, {0, 1, 2, 3}},
            {Elem::Tet4, {1, 2, 3, 4}},
        }, 5);

    auto query = makeFaceQuery(m.cellElemInfo);
    auto res = MeshConnectivity::Interpolate(m.cell2node, query, 2, 5, g_mpi);

    CHECK(res.nEntities == 7);

    int nInternal = 0;
    for (DNDS::index i = 0; i < res.nEntities; i++)
        if (res.entity2parent.father->operator()(i, 1) != DNDS::UnInitIndex)
            nInternal++;
    CHECK(nInternal == 1);

    // Shared face is {1,2,3}
    for (DNDS::index i = 0; i < res.nEntities; i++)
    {
        if (res.entity2parent.father->operator()(i, 1) != DNDS::UnInitIndex)
        {
            auto vs = getEntityVertexSet(res.entity2node, res.entityElemInfo, i);
            CHECK(vs == std::set<DNDS::index>{1, 2, 3});
        }
    }
}

// ---------------------------------------------------------------------------
// Interpolate: single hex (3D) — face extraction
// ---------------------------------------------------------------------------

TEST_CASE("Interpolate: single hex faces (3D)")
{
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    // Hex8: 6 Quad4 faces
    auto m = makeHandCraftedMesh(g_mpi,
        {
            {Elem::Hex8, {0, 1, 2, 3, 4, 5, 6, 7}},
        }, 8);

    auto query = makeFaceQuery(m.cellElemInfo);
    auto res = MeshConnectivity::Interpolate(m.cell2node, query, 1, 8, g_mpi);

    CHECK(res.nEntities == 6);

    for (DNDS::index i = 0; i < 6; i++)
    {
        CHECK(res.entityElemInfo[i].getElemType() == Elem::Quad4);
        CHECK(res.entity2node.father->RowSize(i) == 4);
        CHECK(res.entity2parent.father->operator()(i, 1) == DNDS::UnInitIndex);
    }

    // Verify face vertex sets
    std::set<std::set<DNDS::index>> expectedFaces = {
        {0, 1, 2, 3}, {0, 1, 4, 5}, {1, 2, 5, 6},
        {2, 3, 6, 7}, {0, 3, 4, 7}, {4, 5, 6, 7}};
    std::set<std::set<DNDS::index>> actualFaces;
    for (DNDS::index i = 0; i < 6; i++)
        actualFaces.insert(getEntityVertexSet(res.entity2node, res.entityElemInfo, i));
    CHECK(actualFaces == expectedFaces);
}

// ---------------------------------------------------------------------------
// Interpolate: edge extraction from single tet (3D)
// ---------------------------------------------------------------------------

TEST_CASE("Interpolate: single tet edges (3D)")
{
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    // Tet4: 6 Line2 edges
    auto m = makeHandCraftedMesh(g_mpi,
        {
            {Elem::Tet4, {0, 1, 2, 3}},
        }, 4);

    auto query = makeEdgeQuery(m.cellElemInfo);
    auto res = MeshConnectivity::Interpolate(m.cell2node, query, 1, 4, g_mpi);

    CHECK(res.nEntities == 6);

    for (DNDS::index i = 0; i < 6; i++)
    {
        CHECK(res.entityElemInfo[i].getElemType() == Elem::Line2);
        CHECK(res.entity2node.father->RowSize(i) == 2);
    }

    // Tet4 edges: {0,1},{1,2},{2,0},{0,3},{1,3},{2,3}
    std::set<std::set<DNDS::index>> expectedEdges = {
        {0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
    std::set<std::set<DNDS::index>> actualEdges;
    for (DNDS::index i = 0; i < 6; i++)
    {
        std::set<DNDS::index> vs;
        for (DNDS::rowsize j = 0; j < 2; j++)
            vs.insert(res.entity2node.father->operator()(i, j));
        actualEdges.insert(vs);
    }
    CHECK(actualEdges == expectedEdges);
}

// ---------------------------------------------------------------------------
// Interpolate: edge deduplication across two tets (3D)
// ---------------------------------------------------------------------------

TEST_CASE("Interpolate: two-tet shared edges (3D)")
{
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    // Tet A: {0,1,2,3}, Tet B: {1,2,3,4}
    // Tet A edges: {0,1},{1,2},{2,0},{0,3},{1,3},{2,3}
    // Tet B edges: {1,2},{2,3},{3,1},{1,4},{2,4},{3,4}
    // Shared: {1,2},{1,3},{2,3} -> 3 shared
    // Unique: 6 + 6 - 3 = 9
    auto m = makeHandCraftedMesh(g_mpi,
        {
            {Elem::Tet4, {0, 1, 2, 3}},
            {Elem::Tet4, {1, 2, 3, 4}},
        }, 5);

    auto query = makeEdgeQuery(m.cellElemInfo);
    auto res = MeshConnectivity::Interpolate(m.cell2node, query, 2, 5, g_mpi);

    CHECK(res.nEntities == 9);

    int nShared = 0;
    for (DNDS::index i = 0; i < res.nEntities; i++)
        if (res.entity2parent.father->operator()(i, 1) != DNDS::UnInitIndex)
            nShared++;
    CHECK(nShared == 3);

    // Shared edges: {1,2}, {1,3}, {2,3}
    std::set<std::set<DNDS::index>> expectedShared = {{1, 2}, {1, 3}, {2, 3}};
    std::set<std::set<DNDS::index>> actualShared;
    for (DNDS::index i = 0; i < res.nEntities; i++)
    {
        if (res.entity2parent.father->operator()(i, 1) != DNDS::UnInitIndex)
        {
            std::set<DNDS::index> vs;
            for (DNDS::rowsize j = 0; j < 2; j++)
                vs.insert(res.entity2node.father->operator()(i, j));
            actualShared.insert(vs);
        }
    }
    CHECK(actualShared == expectedShared);
}

// ---------------------------------------------------------------------------
// Interpolate: mixed 2D mesh (tri + quad)
// ---------------------------------------------------------------------------

TEST_CASE("Interpolate: mixed tri+quad faces (2D)")
{
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    //     3---4---5
    //     |\ 1| 2 |
    //     | \ |   |
    //     |0 \|   |
    //     0---1---2
    //
    // Cell 0: Tri3  {0,1,3}  → 3 Line2 faces
    // Cell 1: Tri3  {1,4,3}  → 3 Line2 faces
    // Cell 2: Quad4 {1,2,5,4} → 4 Line2 faces
    // Shared edges: {0,1}? No. {1,3} between cell 0 & 1, {1,4} between cell 1 & 2
    // Cell 0 faces: {0,1}, {1,3}, {3,0}
    // Cell 1 faces: {1,4}, {4,3}, {3,1}
    // Cell 2 faces: {1,2}, {2,5}, {5,4}, {4,1}
    // Shared: {1,3} (cells 0&1), {1,4} (cells 1&2)
    // Total: 3 + 3 + 4 - 2 = 8

    auto m = makeHandCraftedMesh(g_mpi,
        {
            {Elem::Tri3, {0, 1, 3}},
            {Elem::Tri3, {1, 4, 3}},
            {Elem::Quad4, {1, 2, 5, 4}},
        }, 6);

    auto query = makeFaceQuery(m.cellElemInfo);
    auto res = MeshConnectivity::Interpolate(m.cell2node, query, 3, 6, g_mpi);

    CHECK(res.nEntities == 8);

    int nInternal = 0;
    for (DNDS::index i = 0; i < res.nEntities; i++)
        if (res.entity2parent.father->operator()(i, 1) != DNDS::UnInitIndex)
            nInternal++;
    CHECK(nInternal == 2);

    // All should be Line2
    for (DNDS::index i = 0; i < res.nEntities; i++)
        CHECK(res.entityElemInfo[i].getElemType() == Elem::Line2);
}

// ===========================================================================
// Regression: Interpolate vs legacy InterpolateFace
// ===========================================================================

TEST_CASE("Regression: Interpolate matches InterpolateFace on UniformSquare_10")
{
    // Build mesh via legacy pipeline all the way through InterpolateFace
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
    mesh->BuildGhostPrimary();

    mesh->AdjGlobal2LocalPrimary();
    mesh->AdjGlobal2LocalN2CB();

    // Now InterpolateFace requires Adj_PointToLocal state
    mesh->InterpolateFace();

    // Snapshot after InterpolateFace: indices are local
    DNDS::index nCellLocal = mesh->cell2node.father->Size();
    DNDS::index nCellGhost = mesh->cell2node.son ? mesh->cell2node.son->Size() : 0;
    DNDS::index nCellAll = nCellLocal + nCellGhost;
    DNDS::index nNodeAll = mesh->coords.father->Size() +
                           (mesh->coords.son ? mesh->coords.son->Size() : 0);

    // DSL: run Interpolate on all cells (local + ghost), using local node indices
    auto query = makeFaceQuery(mesh->cellElemInfo);
    auto res = MeshConnectivity::Interpolate(
        mesh->cell2node, query, nCellAll, nNodeAll, g_mpi);

    // Compare: for each local cell, the set of face vertex sets from DSL should
    // match the legacy cell2face → face2node vertex sets.
    // Note: the DSL operates on ALL cells (local+ghost) without ownership filtering,
    // so DSL may produce more faces than legacy (which discards ghost-only faces).
    // But for each local cell, all its faces must exist in both and match.

    for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
    {
        CAPTURE(iCell);
        auto eCell = Elem::Element{mesh->cellElemInfo[iCell]->getElemType()};
        int nFaces = eCell.GetNumFaces();

        // Legacy: collect face vertex sets from cell2face
        std::set<std::set<DNDS::index>> legacyFaceSets;
        for (int j = 0; j < nFaces; j++)
        {
            DNDS::index iFace = mesh->cell2face[iCell][j];
            auto eFace = eCell.ObtainFace(j);
            int nVerts = eFace.GetNumVertices();
            std::set<DNDS::index> vs;
            for (int k = 0; k < nVerts; k++)
                vs.insert(mesh->face2node[iFace][k]);
            legacyFaceSets.insert(vs);
        }

        // DSL: collect face vertex sets from parent2entity
        std::set<std::set<DNDS::index>> dslFaceSets;
        for (DNDS::rowsize j = 0; j < res.parent2entity.father->RowSize(iCell); j++)
        {
            DNDS::index iEnt = res.parent2entity.father->operator()(iCell, j);
            auto eEnt = Elem::Element{res.entityElemInfo[iEnt].getElemType()};
            int nVerts = eEnt.GetNumVertices();
            std::set<DNDS::index> vs;
            for (int k = 0; k < nVerts; k++)
                vs.insert(res.entity2node.father->operator()(iEnt, k));
            dslFaceSets.insert(vs);
        }

        CHECK(dslFaceSets == legacyFaceSets);
    }
}

// ===========================================================================
// Periodic Interpolate: 2×2 doubly-periodic quad mesh
// ===========================================================================

/// Build a 2×2 quad mesh on [0,2]×[0,2], doubly-periodic (X period 2, Y period 2).
///
/// After periodic deduplication, 9 original nodes collapse to 4:
///
///   Original layout:         After dedup (4 nodes):
///     6---7---8                 0=(0,0)  1=(1,0)
///     | 2 | 3 |                 2=(0,1)  3=(1,1)
///     3---4---5
///     | 0 | 1 |    Right col → left col + P1
///     0---1---2    Top row → bottom row + P2
///                  Corner 8 → 0 + P1|P2
///
/// cell2node (deduped):
///   cell 0: {0,1,3,2}  pbi: {0,0,0,0}
///   cell 1: {1,0,2,3}  pbi: {0,P1,P1,0}
///   cell 2: {2,3,1,0}  pbi: {0,0,P2,P2}
///   cell 3: {3,2,0,1}  pbi: {0,P1,P1|P2,P2}
///
/// All 4 cells share all 4 nodes! Without the collaborating check,
/// face dedup would incorrectly merge edges across the periodic corner.
/// Expected: 8 unique faces (4 cells × 4 edges / 2), all internal.
///
struct Periodic2x2Mesh
{
    tAdjPair cell2node;
    tElemInfoArrayPair cellElemInfo;
    tPbiPair cell2nodePbi;
    DNDS::index nNodes = 4;
    DNDS::index nCells = 4;
};

static Periodic2x2Mesh makePeriodic2x2Mesh(const MPIInfo &mpi)
{
    Periodic2x2Mesh m;
    m.cell2node.InitPair("p2x2_c2n", mpi);
    m.cellElemInfo.InitPair("p2x2_cei", mpi);
    m.cell2nodePbi.InitPair("p2x2_pbi", mpi);

    m.cell2node.father->Resize(4);
    m.cellElemInfo.father->Resize(4);
    m.cell2nodePbi.father->Resize(4);

    // Quad4 node order: BL, BR, TR, TL (CCW)
    DNDS::index c2n[4][4] = {
        {0, 1, 3, 2}, // cell 0
        {1, 0, 2, 3}, // cell 1 (right col wraps)
        {2, 3, 1, 0}, // cell 2 (top row wraps)
        {3, 2, 0, 1}, // cell 3 (corner wraps)
    };
    uint8_t pbi[4][4] = {
        {0, 0, 0, 0},                   // cell 0: all main
        {0, 0x01, 0x01, 0},             // cell 1: nodes 1,2 have P1
        {0, 0, 0x02, 0x02},             // cell 2: nodes 2,3 have P2
        {0, 0x01, 0x01 | 0x02, 0x02},   // cell 3: mixed
    };

    ElemInfo qi;
    qi.setElemType(Elem::Quad4);
    qi.zone = 0;

    for (DNDS::index i = 0; i < 4; i++)
    {
        m.cell2node.father->ResizeRow(i, 4);
        m.cell2nodePbi.father->ResizeRow(i, 4);
        m.cellElemInfo(i, 0) = qi;
        for (DNDS::rowsize j = 0; j < 4; j++)
        {
            m.cell2node.father->operator()(i, j) = c2n[i][j];
            m.cell2nodePbi.father->operator()(i, j) = NodePeriodicBits{pbi[i][j]};
        }
    }
    return m;
}

TEST_CASE("Interpolate: 2x2 periodic quad mesh — collaborating check required")
{
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    auto pm = makePeriodic2x2Mesh(g_mpi);

    // --- Without collaborating check: expect failure ---
    {
        auto faceQueryNoPbi = makeFaceQuery(pm.cellElemInfo);
        auto resNoPbi = MeshConnectivity::Interpolate(
            pm.cell2node, faceQueryNoPbi, 4, 4, g_mpi);

        // Without the collaborating check, cells sharing all 4 nodes through
        // different periodic images cause false face merges (3+ parents).
        CHECK(resNoPbi.duplicateOverflow == true);
    }

    // --- With collaborating check: expect correct result ---
    SubEntityQuery faceQueryPbi = makeFaceQuery(pm.cellElemInfo);
    faceQueryPbi.matchExtra =
        [&pm](DNDS::index iParent, int iSub,
              DNDS::index /*iCandEntity*/,
              DNDS::index candidateParent, int candidateSub) -> bool
    {
        auto eParentA = Elem::Element{pm.cellElemInfo[iParent]->getElemType()};
        auto eFace = eParentA.ObtainFace(iSub);
        int nFN = eFace.GetNumNodes();

        std::vector<DNDS::index> nodesA(nFN), nodesB(nFN);
        eParentA.ExtractFaceNodes(iSub, pm.cell2node[iParent], nodesA);
        auto eParentB = Elem::Element{pm.cellElemInfo[candidateParent]->getElemType()};
        eParentB.ExtractFaceNodes(candidateSub, pm.cell2node[candidateParent], nodesB);

        std::vector<NodePeriodicBits> pbiA(nFN), pbiB(nFN);
        eParentA.ExtractFaceNodes(iSub, pm.cell2nodePbi[iParent], pbiA);
        eParentB.ExtractFaceNodes(candidateSub, pm.cell2nodePbi[candidateParent], pbiB);

        using P = std::pair<DNDS::index, NodePeriodicBits>;
        auto cmp = [](const P &L, const P &R)
        { return L.first == R.first ? uint8_t(L.second) < uint8_t(R.second)
                                    : L.first < R.first; };

        std::vector<P> pA(nFN), pB(nFN);
        for (int i = 0; i < nFN; i++)
        {
            pA[i] = {nodesA[i], pbiA[i]};
            pB[i] = {nodesB[i], pbiB[i]};
        }
        std::sort(pA.begin(), pA.end(), cmp);
        std::sort(pB.begin(), pB.end(), cmp);

        auto v0 = pA[0].second ^ pB[0].second;
        for (int i = 1; i < nFN; i++)
            if ((pA[i].second ^ pB[i].second) != v0)
                return false;
        return true;
    };

    auto resPbi = MeshConnectivity::Interpolate(
        pm.cell2node, faceQueryPbi, 4, 4, g_mpi);

    // With collaborating check: expect exactly 8 unique faces
    CHECK(resPbi.nEntities == 8);

    // All faces should be internal (both parents set)
    int nInternal = 0;
    for (DNDS::index i = 0; i < resPbi.nEntities; i++)
        if (resPbi.entity2parent.father->operator()(i, 1) != DNDS::UnInitIndex)
            nInternal++;
    CHECK(nInternal == 8);

    // Each cell should reference exactly 4 faces
    for (DNDS::index i = 0; i < 4; i++)
    {
        CAPTURE(i);
        CHECK(resPbi.parent2entity.father->RowSize(i) == 4);
    }

    // Verify parent2entity → entity2parent consistency
    for (DNDS::index iCell = 0; iCell < 4; iCell++)
    {
        for (DNDS::rowsize j = 0; j < resPbi.parent2entity.father->RowSize(iCell); j++)
        {
            DNDS::index iEnt = resPbi.parent2entity.father->operator()(iCell, j);
            auto p0 = resPbi.entity2parent.father->operator()(iEnt, 0);
            auto p1 = resPbi.entity2parent.father->operator()(iEnt, 1);
            CAPTURE(iCell); CAPTURE(j); CAPTURE(iEnt);
            CHECK((p0 == iCell || p1 == iCell));
        }
    }

    // Verify no face connects a cell to itself
    for (DNDS::index i = 0; i < resPbi.nEntities; i++)
    {
        auto p0 = resPbi.entity2parent.father->operator()(i, 0);
        auto p1 = resPbi.entity2parent.father->operator()(i, 1);
        CAPTURE(i);
        CHECK(p0 != p1);
    }

    // Verify each cell-pair shares exactly the right number of faces:
    // Cell 0-1: 2 faces, Cell 0-2: 2 faces, Cell 1-3: 2 faces, Cell 2-3: 2 faces
    // Cell 0-3: 0 faces (diagonal, no shared face), Cell 1-2: 0 faces (diagonal)
    std::map<std::set<DNDS::index>, int> pairCount;
    for (DNDS::index i = 0; i < resPbi.nEntities; i++)
    {
        auto p0 = resPbi.entity2parent.father->operator()(i, 0);
        auto p1 = resPbi.entity2parent.father->operator()(i, 1);
        pairCount[{p0, p1}]++;
    }
    CHECK(pairCount[{0, 1}] == 2);
    CHECK(pairCount[{0, 2}] == 2);
    CHECK(pairCount[{1, 3}] == 2);
    CHECK(pairCount[{2, 3}] == 2);
    CHECK(pairCount.count({0, 3}) == 0);
    CHECK(pairCount.count({1, 2}) == 0);
}

// ===========================================================================
// 2×2×2 triply-periodic hex mesh
// ===========================================================================

/// Build a 2×2×2 Hex8 mesh on [0,2]³, triply-periodic (P1=X, P2=Y, P3=Z).
///
/// After periodic dedup: 27 nodes → 8 nodes, 8 cells.
/// All 8 cells reference all 8 nodes (with varying pbi).
///
/// Expected topology (3-torus T³):
///   V=8, E=24, F=24, C=8 → Euler χ = 8-24+24-8 = 0.
///
struct Periodic2x2x2Mesh
{
    tAdjPair cell2node;
    tElemInfoArrayPair cellElemInfo;
    tPbiPair cell2nodePbi;
    DNDS::index nNodes = 8;
    DNDS::index nCells = 8;
};

static Periodic2x2x2Mesh makePeriodic2x2x2Mesh(const MPIInfo &mpi)
{
    Periodic2x2x2Mesh m;
    m.cell2node.InitPair("p2x2x2_c2n", mpi);
    m.cellElemInfo.InitPair("p2x2x2_cei", mpi);
    m.cell2nodePbi.InitPair("p2x2x2_pbi", mpi);

    m.cell2node.father->Resize(8);
    m.cellElemInfo.father->Resize(8);
    m.cell2nodePbi.father->Resize(8);

    // Deduped node index: d(i,j,k) = i + 2*j + 4*k for i,j,k ∈ {0,1}
    // Cell index: ci + 2*cj + 4*ck for ci,cj,ck ∈ {0,1}
    // Hex8 local ordering: BL-front-bot, BR-front-bot, BR-back-bot, BL-back-bot,
    //                       BL-front-top, BR-front-top, BR-back-top, BL-back-top
    DNDS::index c2n[8][8] = {
        {0, 1, 3, 2, 4, 5, 7, 6}, // cell 0 (0,0,0)
        {1, 0, 2, 3, 5, 4, 6, 7}, // cell 1 (1,0,0)
        {2, 3, 1, 0, 6, 7, 5, 4}, // cell 2 (0,1,0)
        {3, 2, 0, 1, 7, 6, 4, 5}, // cell 3 (1,1,0)
        {4, 5, 7, 6, 0, 1, 3, 2}, // cell 4 (0,0,1)
        {5, 4, 6, 7, 1, 0, 2, 3}, // cell 5 (1,0,1)
        {6, 7, 5, 4, 2, 3, 1, 0}, // cell 6 (0,1,1)
        {7, 6, 4, 5, 3, 2, 0, 1}, // cell 7 (1,1,1)
    };
    uint8_t pbi[8][8] = {
        {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // cell 0: all main
        {0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00}, // cell 1: X-wrap
        {0x00, 0x00, 0x02, 0x02, 0x00, 0x00, 0x02, 0x02}, // cell 2: Y-wrap
        {0x00, 0x01, 0x03, 0x02, 0x00, 0x01, 0x03, 0x02}, // cell 3: XY-wrap
        {0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04}, // cell 4: Z-wrap
        {0x00, 0x01, 0x01, 0x00, 0x04, 0x05, 0x05, 0x04}, // cell 5: XZ-wrap
        {0x00, 0x00, 0x02, 0x02, 0x04, 0x04, 0x06, 0x06}, // cell 6: YZ-wrap
        {0x00, 0x01, 0x03, 0x02, 0x04, 0x05, 0x07, 0x06}, // cell 7: XYZ-wrap
    };

    ElemInfo hi;
    hi.setElemType(Elem::Hex8);
    hi.zone = 0;

    for (DNDS::index i = 0; i < 8; i++)
    {
        m.cell2node.father->ResizeRow(i, 8);
        m.cell2nodePbi.father->ResizeRow(i, 8);
        m.cellElemInfo(i, 0) = hi;
        for (DNDS::rowsize j = 0; j < 8; j++)
        {
            m.cell2node.father->operator()(i, j) = c2n[i][j];
            m.cell2nodePbi.father->operator()(i, j) = NodePeriodicBits{pbi[i][j]};
        }
    }
    return m;
}

/// Helper: build the periodic matchExtra callback for a given mesh.
static std::function<bool(DNDS::index, int, DNDS::index, DNDS::index, int)>
makePeriodicMatchExtra(const tElemInfoArrayPair &cellElemInfo,
                       const tAdjPair &cell2node,
                       const tPbiPair &cell2nodePbi,
                       bool forEdges)
{
    return [&cellElemInfo, &cell2node, &cell2nodePbi, forEdges](
               DNDS::index iParent, int iSub,
               DNDS::index, DNDS::index candidateParent, int candidateSub) -> bool
    {
        auto eParentA = Elem::Element{cellElemInfo[iParent]->getElemType()};
        auto eParentB = Elem::Element{cellElemInfo[candidateParent]->getElemType()};

        int nFN;
        std::vector<DNDS::index> nodesA, nodesB;
        std::vector<NodePeriodicBits> pbiA, pbiB;

        if (forEdges)
        {
            auto eEdge = eParentA.ObtainEdge(iSub);
            nFN = eEdge.GetNumNodes();
            nodesA.resize(nFN); nodesB.resize(nFN);
            pbiA.resize(nFN); pbiB.resize(nFN);
            eParentA.ExtractEdgeNodes(iSub, cell2node[iParent], nodesA);
            eParentA.ExtractEdgeNodes(iSub, cell2nodePbi[iParent], pbiA);
            eParentB.ExtractEdgeNodes(candidateSub, cell2node[candidateParent], nodesB);
            eParentB.ExtractEdgeNodes(candidateSub, cell2nodePbi[candidateParent], pbiB);
        }
        else
        {
            auto eFace = eParentA.ObtainFace(iSub);
            nFN = eFace.GetNumNodes();
            nodesA.resize(nFN); nodesB.resize(nFN);
            pbiA.resize(nFN); pbiB.resize(nFN);
            eParentA.ExtractFaceNodes(iSub, cell2node[iParent], nodesA);
            eParentA.ExtractFaceNodes(iSub, cell2nodePbi[iParent], pbiA);
            eParentB.ExtractFaceNodes(candidateSub, cell2node[candidateParent], nodesB);
            eParentB.ExtractFaceNodes(candidateSub, cell2nodePbi[candidateParent], pbiB);
        }

        using P = std::pair<DNDS::index, NodePeriodicBits>;
        auto cmp = [](const P &L, const P &R)
        { return L.first == R.first ? uint8_t(L.second) < uint8_t(R.second)
                                    : L.first < R.first; };
        std::vector<P> pA(nFN), pB(nFN);
        for (int i = 0; i < nFN; i++)
        {
            pA[i] = {nodesA[i], pbiA[i]};
            pB[i] = {nodesB[i], pbiB[i]};
        }
        std::sort(pA.begin(), pA.end(), cmp);
        std::sort(pB.begin(), pB.end(), cmp);

        auto v0 = pA[0].second ^ pB[0].second;
        for (int i = 1; i < nFN; i++)
            if ((pA[i].second ^ pB[i].second) != v0)
                return false;
        return true;
    };
}

TEST_CASE("Periodic 2x2x2: face interpolation (3D, collaborating check)")
{
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    auto pm = makePeriodic2x2x2Mesh(g_mpi);

    // Without collaborating check: expect overflow
    {
        auto q = makeFaceQuery(pm.cellElemInfo);
        auto res = MeshConnectivity::Interpolate(pm.cell2node, q, 8, 8, g_mpi);
        CHECK(res.duplicateOverflow == true);
    }

    // With collaborating check: expect 24 faces, all internal
    {
        auto q = makeFaceQuery(pm.cellElemInfo);
        q.matchExtra = makePeriodicMatchExtra(pm.cellElemInfo, pm.cell2node, pm.cell2nodePbi, false);
        auto res = MeshConnectivity::Interpolate(pm.cell2node, q, 8, 8, g_mpi);

        CHECK(res.duplicateOverflow == false);
        CHECK(res.nEntities == 24);

        // All faces internal (both parents set)
        int nInternal = 0;
        for (DNDS::index i = 0; i < res.nEntities; i++)
            if (res.entity2parent.father->operator()(i, 1) != DNDS::UnInitIndex)
                nInternal++;
        CHECK(nInternal == 24);

        // Each cell references 6 faces
        for (DNDS::index i = 0; i < 8; i++)
        {
            CAPTURE(i);
            CHECK(res.parent2entity.father->RowSize(i) == 6);
        }

        // All faces are Quad4
        for (DNDS::index i = 0; i < res.nEntities; i++)
            CHECK(res.entityElemInfo[i].getElemType() == Elem::Quad4);

        // No face connects a cell to itself
        for (DNDS::index i = 0; i < res.nEntities; i++)
        {
            auto p0 = res.entity2parent.father->operator()(i, 0);
            auto p1 = res.entity2parent.father->operator()(i, 1);
            CHECK(p0 != p1);
        }

        // Each cell has exactly 3 face-neighbors (X±, Y±, Z± — but periodic
        // means X- wraps to X+, so each cell is face-adjacent to 6 cells?
        // No: 2×2×2 periodic hex, each cell has 6 faces, each face is shared
        // with exactly 1 other cell. Each cell has at most 6 neighbors, but
        // some neighbors repeat (e.g., cell 0's +X neighbor is cell 1,
        // cell 0's -X neighbor is also cell 1 via periodicity).
        // Actually: cell 0 at (0,0,0), +X neighbor = cell 1 at (1,0,0),
        // -X neighbor = cell 1 via X-periodic wrap. So cell 0 and cell 1
        // share 2 faces (both X-faces). Similarly Y→cell2, Z→cell4.
        // So each cell has 3 distinct face-neighbors, each sharing 2 faces.
        std::map<DNDS::index, std::set<DNDS::index>> cellFN;
        for (DNDS::index i = 0; i < res.nEntities; i++)
        {
            auto p0 = res.entity2parent.father->operator()(i, 0);
            auto p1 = res.entity2parent.father->operator()(i, 1);
            cellFN[p0].insert(p1);
            cellFN[p1].insert(p0);
        }
        for (DNDS::index i = 0; i < 8; i++)
        {
            CAPTURE(i);
            CHECK(cellFN[i].size() == 3);
        }
    }
}

TEST_CASE("Periodic 2x2x2: edge interpolation (3D, collaborating check)")
{
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    auto pm = makePeriodic2x2x2Mesh(g_mpi);

    // Without collaborating check: expect overflow
    {
        auto q = makeEdgeQuery(pm.cellElemInfo);
        auto res = MeshConnectivity::Interpolate(pm.cell2node, q, 8, 8, g_mpi);
        CHECK(res.duplicateOverflow == true);
    }

    // With collaborating check: expect 24 edges, all shared by 4 cells
    {
        auto q = makeEdgeQuery(pm.cellElemInfo);
        q.matchExtra = makePeriodicMatchExtra(pm.cellElemInfo, pm.cell2node, pm.cell2nodePbi, true);
        auto res = MeshConnectivity::Interpolate(pm.cell2node, q, 8, 8, g_mpi);

        CHECK(res.duplicateOverflow == false);
        CHECK(res.nEntities == 24);

        // All edges are Line2
        for (DNDS::index i = 0; i < res.nEntities; i++)
            CHECK(res.entityElemInfo[i].getElemType() == Elem::Line2);

        // Each cell references 12 edges
        for (DNDS::index i = 0; i < 8; i++)
        {
            CAPTURE(i);
            CHECK(res.parent2entity.father->RowSize(i) == 12);
        }

        // entity2parent only stores 2 parents, but edges are shared by 4 cells.
        // After Interpolate, entity2parent has the first 2 cells that
        // encountered each edge. The other 2 cells reference the same entity
        // via parent2entity but aren't recorded in entity2parent.
        // So we count how many cells reference each edge via parent2entity.
        std::vector<int> edgeRefCount(res.nEntities, 0);
        for (DNDS::index iCell = 0; iCell < 8; iCell++)
            for (DNDS::rowsize j = 0; j < res.parent2entity.father->RowSize(iCell); j++)
                edgeRefCount[res.parent2entity.father->operator()(iCell, j)]++;

        // Each edge should be referenced by exactly 4 cells
        for (DNDS::index i = 0; i < res.nEntities; i++)
        {
            CAPTURE(i);
            CHECK(edgeRefCount[i] == 4);
        }

        // Euler characteristic: V - E + F = 8 - 24 + 24 = 8
        // (For the cell complex: V - E + F - C = 8 - 24 + 24 - 8 = 0 for T³)
    }
}

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
