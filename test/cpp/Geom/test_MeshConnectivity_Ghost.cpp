/**
 * @file test_MeshConnectivity_Ghost.cpp
 * @brief Unit tests for ghost chain types, compilation, and BFS evaluation.
 *
 * Tests:
 *   - GhostChain: defaultPrimary compiles into correct tree structure
 *   - GhostChain: prefix merging (shared trie paths)
 *   - GhostChain: invalid chain detection (empty, mismatch)
 *   - GhostChain: checkAvailable (missing adjacency detection)
 *   - GhostChain: dump (diagnostic output)
 *   - GhostChain: face ghost chain
 *   - AdjKind: equality and hash (direct vs intra-level)
 *   - entityDepth: 2D and 3D
 *   - adjKindName: formatted names
 *   - evaluateGhostTree: single hop cell2cell
 *   - evaluateGhostTree: two-hop cell2cell2node with manual ghost
 *   - evaluateGhostTree: 2-ring cell chain
 *   - evaluateGhostTree: union of multiple chains
 *   - evaluateGhostTree: np=1 produces no ghosts
 *   - evaluateGhostTree: performance benchmark (NxN grids with correctness)
 *
 * Running:
 *   cmake --build build -t geom_test_mesh_connectivity_ghost -j8
 *   ctest --test-dir build -R geom_mesh_connectivity_ghost --output-on-failure
 *
 * Running benchmark only:
 *   mpirun -np 2 ./build/test/cpp/geom_test_mesh_connectivity_ghost -tc="*performance*"
 *   mpirun -np 4 ./build/test/cpp/geom_test_mesh_connectivity_ghost -tc="*performance*"
 *
 * The benchmark generates synthetic NxN quad grids (N=100,500,1000),
 * verifies exact ghost cell counts against analytical expectations,
 * and prints timing for 1-hop (cell2cell) and 2-hop (cell2cell + cell2node)
 * ghost evaluation.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "SyntheticMeshBuilders.hpp"
#include "Geom/Mesh/MeshConnectivity.hpp"
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_set>

using namespace DNDS;
using namespace DNDS::Geom;

static MPIInfo g_mpi;

// ---------------------------------------------------------------------------
// Helpers: 4-quad cell2cell and cell2node with mapping (ghost-test specific)
// ---------------------------------------------------------------------------

static tAdjPair make4QuadCell2Cell(const MPIInfo &mpi)
{
    tAdjPair c2c;
    c2c.InitPair("test_c2c", mpi);

    DNDS::index nLocal = 0;
    if (mpi.size == 1)
        nLocal = 4;
    else if (mpi.rank < 2)
        nLocal = 2;

    c2c.father->Resize(nLocal);

    DNDS::index allNeighbors[4][3] = {
        {1, 2, 3},
        {0, 2, 3},
        {0, 1, 3},
        {0, 1, 2},
    };

    DNDS::index cellOffset = (mpi.size > 1) ? mpi.rank * 2 : 0;
    for (DNDS::index i = 0; i < nLocal; i++)
    {
        DNDS::index globalCell = cellOffset + i;
        c2c.father->ResizeRow(i, 3);
        for (DNDS::rowsize j = 0; j < 3; j++)
            c2c.father->operator()(i, j) = allNeighbors[globalCell][j];
    }

    c2c.father->pLGlobalMapping = std::make_shared<GlobalOffsetsMapping>();
    c2c.father->pLGlobalMapping->setMPIAlignBcast(mpi, nLocal);

    return c2c;
}

/// Build partitioned cell2node with pLGlobalMapping.
static tAdjPair make4QuadCell2NodeWithMapping(const MPIInfo &mpi)
{
    tAdjPair c2n = make4QuadCell2Node(mpi);

    DNDS::index nLocal = 0;
    if (mpi.size == 1)
        nLocal = 4;
    else if (mpi.rank < 2)
        nLocal = 2;

    c2n.father->pLGlobalMapping = std::make_shared<GlobalOffsetsMapping>();
    c2n.father->pLGlobalMapping->setMPIAlignBcast(mpi, nLocal);

    return c2n;
}

// ===========================================================================
// Ghost chain compilation tests
// ===========================================================================

TEST_CASE("GhostChain: defaultPrimary compiles")
{
    auto spec = GhostSpec::defaultPrimary();
    auto tree = CompiledGhostTree::compile(spec);

    CHECK(tree.roots.size() == 2);

    const GhostTreeNode *cellRoot = nullptr;
    const GhostTreeNode *bndRoot = nullptr;
    for (auto &r : tree.roots)
    {
        if (r.kind == EntityKind::Cell)
            cellRoot = &r;
        if (r.kind == EntityKind::Bnd)
            bndRoot = &r;
    }
    REQUIRE(cellRoot);
    REQUIRE(bndRoot);

    CHECK(!cellRoot->collect);
    CHECK(cellRoot->level == 0);

    REQUIRE(cellRoot->children.size() == 1);
    auto &c2cNode = cellRoot->children[0];
    CHECK(c2cNode.kind == EntityKind::Cell);
    CHECK(c2cNode.hop == Adj::Cell2Cell);
    CHECK(c2cNode.collect);
    CHECK(c2cNode.level == 1);

    REQUIRE(c2cNode.children.size() == 1);
    auto &c2nNode = c2cNode.children[0];
    CHECK(c2nNode.kind == EntityKind::Node);
    CHECK(c2nNode.hop == Adj::Cell2Node);
    CHECK(c2nNode.collect);
    CHECK(c2nNode.level == 2);
    CHECK(c2nNode.children.empty());

    CHECK(!bndRoot->collect);
    CHECK(bndRoot->level == 0);

    REQUIRE(bndRoot->children.size() == 1);
    auto &b2nNode = bndRoot->children[0];
    CHECK(b2nNode.kind == EntityKind::Node);
    CHECK(b2nNode.hop == Adj::Bnd2Node);
    CHECK(b2nNode.collect); // chain 4 marks intermediate Node as collect
    CHECK(b2nNode.level == 1);

    REQUIRE(b2nNode.children.size() == 1);
    auto &n2bNode = b2nNode.children[0];
    CHECK(n2bNode.kind == EntityKind::Bnd);
    CHECK(n2bNode.hop == Adj::Node2Bnd);
    CHECK(n2bNode.collect);
    CHECK(n2bNode.level == 2);

    REQUIRE(n2bNode.children.size() == 1);
    auto &b2n2Node = n2bNode.children[0];
    CHECK(b2n2Node.kind == EntityKind::Node);
    CHECK(b2n2Node.hop == Adj::Bnd2Node);
    CHECK(b2n2Node.collect);
    CHECK(b2n2Node.level == 3);
    CHECK(b2n2Node.children.empty());

    CHECK(tree.maxLevel == 3);

    auto kinds = tree.collectedKinds();
    CHECK(kinds.count(EntityKind::Cell) == 1);
    CHECK(kinds.count(EntityKind::Node) == 1);
    CHECK(kinds.count(EntityKind::Bnd) == 1);
    CHECK(kinds.count(EntityKind::Face) == 0);

    auto adjs = tree.requiredAdjs();
    CHECK(adjs.count(Adj::Cell2Cell) == 1);
    CHECK(adjs.count(Adj::Cell2Node) == 1);
    CHECK(adjs.count(Adj::Bnd2Node) == 1);
    CHECK(adjs.count(Adj::Node2Bnd) == 1);
    CHECK(adjs.size() == 4);
}

TEST_CASE("GhostChain: prefix merging")
{
    GhostSpec spec{{
        {EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell},
        {EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Cell}, EntityKind::Cell},
    }};
    auto tree = CompiledGhostTree::compile(spec);

    REQUIRE(tree.roots.size() == 1);
    auto &root = tree.roots[0];
    CHECK(root.kind == EntityKind::Cell);
    REQUIRE(root.children.size() == 1);

    auto &ring1 = root.children[0];
    CHECK(ring1.kind == EntityKind::Cell);
    CHECK(ring1.collect);
    CHECK(ring1.level == 1);
    REQUIRE(ring1.children.size() == 1);

    auto &ring2 = ring1.children[0];
    CHECK(ring2.kind == EntityKind::Cell);
    CHECK(ring2.collect);
    CHECK(ring2.level == 2);
    CHECK(ring2.children.empty());

    CHECK(tree.maxLevel == 2);
}

TEST_CASE("GhostChain: invalid chain detection")
{
    CHECK_THROWS_AS(
        CompiledGhostTree::compile(GhostSpec{{
            {EntityKind::Cell, {}, EntityKind::Cell},
        }}),
        std::runtime_error);

    CHECK_THROWS_AS(
        CompiledGhostTree::compile(GhostSpec{{
            {EntityKind::Node, {Adj::Cell2Node}, EntityKind::Node},
        }}),
        std::runtime_error);

    CHECK_THROWS_AS(
        CompiledGhostTree::compile(GhostSpec{{
            {EntityKind::Cell, {Adj::Cell2Node}, EntityKind::Cell},
        }}),
        std::runtime_error);

    CHECK_THROWS_AS(
        CompiledGhostTree::compile(GhostSpec{{
            {EntityKind::Bnd, {Adj::Bnd2Node, Adj::Cell2Node}, EntityKind::Node},
        }}),
        std::runtime_error);
}

TEST_CASE("GhostChain: checkAvailable")
{
    MeshConnectivity dag;
    dag.meshDim = 2;

    tAdjPair c2c, c2n;
    dag.registerAdj(Adj::Cell2Cell, c2c);
    dag.registerAdj(Adj::Cell2Node, c2n);

    auto spec = GhostSpec::defaultPrimary();
    auto tree = CompiledGhostTree::compile(spec);

    auto missing = tree.checkAvailable(dag);
    CHECK(missing.size() == 2);

    std::unordered_set<AdjKind, AdjKindHash> missingSet(missing.begin(), missing.end());
    CHECK(missingSet.count(Adj::Bnd2Node) == 1);
    CHECK(missingSet.count(Adj::Node2Bnd) == 1);
}

TEST_CASE("GhostChain: dump")
{
    auto spec = GhostSpec::defaultPrimary();
    auto tree = CompiledGhostTree::compile(spec);
    std::string d = tree.dump();

    CHECK(d.find("Cell") != std::string::npos);
    CHECK(d.find("Bnd") != std::string::npos);
    CHECK(d.find("Node") != std::string::npos);
    CHECK(d.find("COLLECT") != std::string::npos);
}

TEST_CASE("GhostChain: face ghost chain")
{
    GhostSpec spec{{
        {EntityKind::Cell, {Adj::Cell2Face}, EntityKind::Face},
    }};
    auto tree = CompiledGhostTree::compile(spec);

    REQUIRE(tree.roots.size() == 1);
    auto &root = tree.roots[0];
    CHECK(root.kind == EntityKind::Cell);
    REQUIRE(root.children.size() == 1);

    auto &faceNode = root.children[0];
    CHECK(faceNode.kind == EntityKind::Face);
    CHECK(faceNode.hop == Adj::Cell2Face);
    CHECK(faceNode.collect);
    CHECK(faceNode.level == 1);

    auto adjs = tree.requiredAdjs();
    CHECK(adjs.count(Adj::Cell2Face) == 1);
    CHECK(adjs.size() == 1);
}

TEST_CASE("GhostChain: AdjKind equality and hash")
{
    AdjKind a(EntityKind::Cell, EntityKind::Node);
    AdjKind b(EntityKind::Cell, EntityKind::Node, EntityKind::Face);
    CHECK(a == b); // direct: via ignored

    AdjKind c(EntityKind::Cell, EntityKind::Cell, EntityKind::Node);
    AdjKind d(EntityKind::Cell, EntityKind::Cell, EntityKind::Face);
    CHECK(c != d); // intra-level: via matters

    AdjKind e(EntityKind::Cell, EntityKind::Cell, EntityKind::Node);
    CHECK(c == e);

    AdjKindHash h;
    CHECK(h(a) == h(b));
    CHECK(h(c) == h(e));
    (void)h(d);
}

TEST_CASE("GhostChain: entityDepth")
{
    CHECK(entityDepth(EntityKind::Cell, 3) == 3);
    CHECK(entityDepth(EntityKind::Face, 3) == 2);
    CHECK(entityDepth(EntityKind::Edge, 3) == 1);
    CHECK(entityDepth(EntityKind::Node, 3) == 0);
    CHECK(entityDepth(EntityKind::Bnd, 3) == 2);

    CHECK(entityDepth(EntityKind::Cell, 2) == 2);
    CHECK(entityDepth(EntityKind::Face, 2) == 1);
    CHECK(entityDepth(EntityKind::Edge, 2) == 1);
    CHECK(entityDepth(EntityKind::Node, 2) == 0);
    CHECK(entityDepth(EntityKind::Bnd, 2) == 1);
}

TEST_CASE("GhostChain: adjKindName")
{
    CHECK(adjKindName(Adj::Cell2Node) == "Cell2Node");
    CHECK(adjKindName(Adj::Node2Cell) == "Node2Cell");
    CHECK(adjKindName(Adj::Cell2Cell) == "Cell2Cell(Node)");
    CHECK(adjKindName(Adj::Cell2CellFace) == "Cell2Cell(Face)");
    CHECK(adjKindName(Adj::Bnd2Node) == "Bnd2Node");
}

// ===========================================================================
// evaluateGhostTree standalone tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Synthetic NxN-per-rank tiled quad grid for benchmarking
// ---------------------------------------------------------------------------

/// Generate a tiled NxN quad grid: each rank owns one NxN tile.
/// The global mesh is N rows × (np*N) columns of quad cells.
///
///   Rank 0 tile     Rank 1 tile     Rank 2 tile ...
///   [0,N)×[0,N)     [0,N)×[N,2N)    [0,N)×[2N,3N)
///
/// Cell (row r, col c) has global index r * totalCols + c,
/// where totalCols = np * N.
///
/// Node (row r, col c) has global index r * (totalCols + 1) + c.
///
/// Each rank owns cells with col in [rank*N, (rank+1)*N).
/// Ghost cells: 1-ring node-neighbors across tile boundaries.
///
/// NOTE: Node global indices are row-major across the full (N+1)×(np*N+1)
/// grid, so they are NOT rank-contiguous.  The node GlobalOffsetsMapping
/// from setMPIAlignBcast assigns contiguous ownership ranges
/// (rank 0 owns [0, nNodeLocal_0), rank 1 owns [nNodeLocal_0, ...)).
/// This abstract partition does NOT align with geometric locality — a
/// rank's "owned" range may include node indices physically located on
/// other ranks' tiles.  The test is self-consistent: both the evaluator
/// and the analytical expected-value functions use the same nodeGM, so
/// ghost counts are exact within this abstract partition.
struct SyntheticTiledGrid
{
    bool periodic{false};
    DNDS::index N, totalCols, totalRows;
    DNDS::index nCellLocal, nNodeLocal;
    DNDS::index colStart, colEnd, rowStart, rowEnd;
    DNDS::index rankCol, rankRow, ranksPerRow, ranksPerCol;
    std::vector<DNDS::index> ownerNodeOffsets; // pre-computed, no MPI
    tAdjPair cell2cell, cell2node;
    ssp<GlobalOffsetsMapping> cellGM, nodeGM;

    DNDS::index cellGlobal(DNDS::index rank, DNDS::index localRow, DNDS::index localCol) const
    {
        return rank * N * N + localRow * N + localCol;
    }
    DNDS::index cellOwnerRank(DNDS::index gr, DNDS::index gc) const
    {
        DNDS::index rc = std::min(gc / N, ranksPerRow - 1);
        DNDS::index rr = std::min(gr / N, ranksPerCol - 1);
        return rr * ranksPerRow + rc;
    }
    /// Node global index via per-rank local matrix.  Node at (r,c) belongs to
    /// cellOwnerRank(r,c).  Its local (row,col) within that rank and its
    /// rank's offset give the 1-D global index.  Periodic nodes wrap.
    DNDS::index nodeGlobal(DNDS::index r, DNDS::index c) const
    {
        DNDS::index wr = r, wc = c;
        if (periodic)
        {
            wr = ((r % totalRows) + totalRows) % totalRows;
            wc = ((c % totalCols) + totalCols) % totalCols;
        }
        DNDS::index owner = cellOwnerRank(wr, wc);
        DNDS::index ownerRC = (owner % ranksPerRow) * N;
        DNDS::index ownerRR = (owner / ranksPerRow) * N;
        DNDS::index ownerNCW = periodic ? N
                                        : N + ((owner % ranksPerRow == ranksPerRow - 1) ? 1 : 0);
        DNDS::index ownerNRH = periodic ? N
                                        : N + ((owner / ranksPerRow == ranksPerCol - 1) ? 1 : 0);
        DNDS::index localCol = wc - ownerRC;
        DNDS::index localRow = wr - ownerRR;
        return ownerNodeOffsets[owner] + localRow * ownerNCW + localCol;
    }

    void build(DNDS::index tileN, const MPIInfo &mpi)
    {
        N = tileN;

        ranksPerRow = DNDS::index(std::sqrt(double(mpi.size)));
        ranksPerCol = mpi.size / ranksPerRow;
        while (ranksPerRow * ranksPerCol != mpi.size && ranksPerRow > 1)
            ranksPerRow--, ranksPerCol = mpi.size / ranksPerRow;
        if (ranksPerRow * ranksPerCol != mpi.size)
            ranksPerRow = mpi.size, ranksPerCol = 1;

        rankCol = mpi.rank % ranksPerRow;
        rankRow = mpi.rank / ranksPerRow;

        totalCols = ranksPerRow * N;
        totalRows = ranksPerCol * N;
        colStart = rankCol * N;
        colEnd = (rankCol + 1) * N;
        rowStart = rankRow * N;
        rowEnd = (rankRow + 1) * N;
        nCellLocal = N * N;

        bool lastInRow = (rankCol == ranksPerRow - 1);
        bool lastInCol = (rankRow == ranksPerCol - 1);

        // Node partition: pre-compute offsets from known shapes.
        ownerNodeOffsets.resize(mpi.size);
        {
            DNDS::index off = 0;
            for (DNDS::index ri = 0; ri < mpi.size; ri++)
            {
                ownerNodeOffsets[ri] = off;
                DNDS::index rc = ri % ranksPerRow;
                DNDS::index rr = ri / ranksPerRow;
                DNDS::index ncw = N + (periodic ? 0 : (rc == ranksPerRow - 1 ? 1 : 0));
                DNDS::index nrh = N + (periodic ? 0 : (rr == ranksPerCol - 1 ? 1 : 0));
                off += nrh * ncw;
            }
        }
        nNodeLocal = (periodic ? N * N
                               : (N + (lastInRow ? 1 : 0)) * (N + (lastInCol ? 1 : 0)));

        cellGM = std::make_shared<GlobalOffsetsMapping>();
        cellGM->setMPIAlignBcast(mpi, nCellLocal);

        nodeGM = std::make_shared<GlobalOffsetsMapping>();
        nodeGM->setMPIAlignBcast(mpi, nNodeLocal);

        cell2node.InitPair("tiled_c2n", mpi);
        cell2node.father->Resize(nCellLocal);
        for (DNDS::index iLocal = 0; iLocal < nCellLocal; iLocal++)
        {
            DNDS::index r = rowStart + iLocal / N;
            DNDS::index c = colStart + iLocal % N;
            cell2node.father->ResizeRow(iLocal, 4);
            cell2node.father->operator()(iLocal, 0) = nodeGlobal(r, c);
            cell2node.father->operator()(iLocal, 1) = nodeGlobal(r, c + 1);
            cell2node.father->operator()(iLocal, 2) = nodeGlobal(r + 1, c + 1);
            cell2node.father->operator()(iLocal, 3) = nodeGlobal(r + 1, c);
        }
        cell2node.father->pLGlobalMapping = cellGM;

        cell2cell.InitPair("tiled_c2c", mpi);
        cell2cell.father->Resize(nCellLocal);
        for (DNDS::index iLocal = 0; iLocal < nCellLocal; iLocal++)
        {
            DNDS::index r = rowStart + iLocal / N;
            DNDS::index c = colStart + iLocal % N;
            DNDS::index myGlobal = cellGlobal(mpi.rank, iLocal / N, iLocal % N);

            std::set<DNDS::index> neighbors;
            for (DNDS::index dr = -1; dr <= 1; dr++)
            {
                for (DNDS::index dc = -1; dc <= 1; dc++)
                {
                    if (dr == 0 && dc == 0)
                        continue;
                    DNDS::index nr = r + dr;
                    DNDS::index nc = c + dc;
                    if (periodic)
                    {
                        nr = ((nr % totalRows) + totalRows) % totalRows;
                        nc = ((nc % totalCols) + totalCols) % totalCols;
                    }
                    else if (nr < 0 || nr >= totalRows || nc < 0 || nc >= totalCols)
                        continue;
                    DNDS::index owner = cellOwnerRank(nr, nc);
                    DNDS::index nbr = nr - (owner / ranksPerRow) * N;
                    DNDS::index nbc = nc - (owner % ranksPerRow) * N;
                    DNDS::index nbGlobal = cellGlobal(owner, nbr, nbc);
                    if (nbGlobal != myGlobal)
                        neighbors.insert(nbGlobal);
                }
            }
            std::vector<DNDS::index> nbVec(neighbors.begin(), neighbors.end());
            cell2cell.father->ResizeRow(iLocal, nbVec.size());
            for (DNDS::rowsize j = 0; j < static_cast<DNDS::rowsize>(nbVec.size()); j++)
                cell2cell.father->operator()(iLocal, j) = nbVec[j];
        }
        cell2cell.father->pLGlobalMapping = cellGM;
    }

    DNDS::index expectedGhostNodesN(DNDS::index nLayers) const
    {
        DNDS::index dx = std::min(N + 2 * nLayers + 1, totalRows);
        DNDS::index dy = std::min(N + 2 * nLayers + 1, totalCols);
        DNDS::index total = (periodic ? N * N : nNodeLocal);
        return dx * dy - total;
    }

    std::set<DNDS::index> expectedGhostCells(const MPIInfo &mpi) const
    {
        DNDS::index cellOffset = cellGM->operator()(mpi.rank, 0);
        std::set<DNDS::index> ghosts;
        for (DNDS::index iLocal = 0; iLocal < nCellLocal; iLocal++)
            for (DNDS::rowsize j = 0; j < cell2cell.father->operator[](iLocal).size(); j++)
            {
                DNDS::index nb = cell2cell.father->operator()(iLocal, j);
                if (nb < cellOffset || nb >= cellOffset + nCellLocal)
                    ghosts.insert(nb);
            }
        return ghosts;
    }

    std::set<DNDS::index> expectedGhostNodesFromOwnedCells(const MPIInfo &mpi) const
    {
        DNDS::index nodeOffset = nodeGM->operator()(mpi.rank, 0);
        DNDS::index nodeEnd = nodeOffset + nNodeLocal;
        std::set<DNDS::index> ghosts;
        for (DNDS::index iLocal = 0; iLocal < nCellLocal; iLocal++)
            for (DNDS::rowsize j = 0; j < 4; j++)
            {
                DNDS::index n = cell2node.father->operator()(iLocal, j);
                if (n < nodeOffset || n >= nodeEnd)
                    ghosts.insert(n);
            }
        return ghosts;
    }

    /// Analytical n-layer cell ghost count.
    DNDS::index expectedGhostCellsN(DNDS::index nLayers) const
    {
        if (periodic)
        {
            DNDS::index dx = std::min(N + 2 * nLayers, totalCols);
            DNDS::index dy = std::min(N + 2 * nLayers, totalRows);
            return dx * dy - N * N;
        }
        DNDS::index rMin = std::max(DNDS::index(0), rowStart - nLayers * N);
        DNDS::index rMax = std::min(totalRows, rowEnd + nLayers * N);
        DNDS::index cMin = std::max(DNDS::index(0), colStart - nLayers * N);
        DNDS::index cMax = std::min(totalCols, colEnd + nLayers * N);
        return (rMax - rMin) * (cMax - cMin) - N * N;
    }
};
TEST_CASE("evaluateGhostTree: performance benchmark")
{
    if (g_mpi.size < 2)
        return;

    std::vector<DNDS::index> sizes = {32, 100};
    if (g_mpi.size <= 4)
        sizes.push_back(500);

    for (auto N : sizes)
    {
        SyntheticTiledGrid grid;
        grid.build(N, g_mpi);

        MeshConnectivity dag;
        dag.meshDim = 2;
        dag.registerAdj(Adj::Cell2Cell, grid.cell2cell);
        dag.registerAdj(Adj::Cell2Node, grid.cell2node);
        dag.registerGlobalMapping(EntityKind::Cell, grid.cellGM);
        dag.registerGlobalMapping(EntityKind::Node, grid.nodeGM);

        // --- 1-hop: cell ghost ---
        GhostSpec spec1{{{EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell}}};
        auto tree1 = CompiledGhostTree::compile(spec1);

        auto t0 = MPI_Wtime();
        auto result1 = dag.evaluateGhostTree(tree1, g_mpi);
        auto t1 = MPI_Wtime();

        // Correctness: exact ghost cell count.
        auto expectedCells = grid.expectedGhostCells(g_mpi);
        auto &ghostCells = result1.ghostIndices[EntityKind::Cell];
        std::set<DNDS::index> actualCells(ghostCells.begin(), ghostCells.end());
        CHECK(actualCells.size() == expectedCells.size());
        CHECK(actualCells == expectedCells);

        // --- 2-hop: cell2cell + cell2node (owned cells only, no ghost c2n) ---
        GhostSpec spec2{{
            {EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell},
            {EntityKind::Cell, {Adj::Cell2Node}, EntityKind::Node},
        }};
        auto tree2 = CompiledGhostTree::compile(spec2);

        auto t2 = MPI_Wtime();
        auto result2 = dag.evaluateGhostTree(tree2, g_mpi);
        auto t3 = MPI_Wtime();

        // Correctness: exact ghost node count (from owned cells only).
        auto expectedNodes = grid.expectedGhostNodesFromOwnedCells(g_mpi);
        auto &ghostNodes = result2.ghostIndices[EntityKind::Node];
        std::set<DNDS::index> actualNodes(ghostNodes.begin(), ghostNodes.end());
        CHECK(actualNodes.size() == expectedNodes.size());
        CHECK(actualNodes == expectedNodes);

        if (g_mpi.rank == 0)
        {
            fmt::print("  N={}(per rank): cells/rank={}, ghost_cells={} (expected={}), "
                       "ghost_nodes={} (expected={}), "
                       "1-hop={:.4f}s, 2-hop={:.4f}s\n",
                       N, grid.nCellLocal,
                       ghostCells.size(), expectedCells.size(),
                       ghostNodes.size(), expectedNodes.size(),
                       t1 - t0, t3 - t2);
        }

        CHECK(result1.hasGhosts(EntityKind::Cell));
    }
}

TEST_CASE("evaluateGhostTree: single hop cell2cell")
{
    if (g_mpi.size < 2)
        return;

    auto c2c = make4QuadCell2Cell(g_mpi);

    MeshConnectivity dag;
    dag.meshDim = 2;
    dag.registerAdj(Adj::Cell2Cell, c2c);
    dag.registerGlobalMapping(EntityKind::Cell, c2c.father->pLGlobalMapping);

    GhostSpec spec{{
        {EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell},
    }};
    auto tree = CompiledGhostTree::compile(spec);
    auto missing = tree.checkAvailable(dag);
    REQUIRE(missing.empty());

    auto result = dag.evaluateGhostTree(tree, g_mpi);

    // hasGhosts is collective — true on all ranks if any rank has ghosts.
    CHECK(result.hasGhosts(EntityKind::Cell));
    CHECK(!result.hasGhosts(EntityKind::Node)); // no node chain in this spec
    auto &ghostCells = result.ghostIndices[EntityKind::Cell];

    if (g_mpi.rank == 0)
    {
        // Rank 0 owns cells {0,1}. All 4 cells share node 4, so neighbors = {2,3}.
        // Exact: 2 ghost cells, no more, no less.
        CHECK(ghostCells.size() == 2);
        CHECK(result.totalGhosts() == 2);
        CHECK(ghostCells[0] == 2);
        CHECK(ghostCells[1] == 3);
    }
    else if (g_mpi.rank == 1)
    {
        CHECK(ghostCells.size() == 2);
        CHECK(result.totalGhosts() == 2);
        CHECK(ghostCells[0] == 0);
        CHECK(ghostCells[1] == 1);
    }
    else
    {
        // Ranks 2+: no cells, no ghosts on this rank.
        CHECK(ghostCells.empty());
        CHECK(result.totalGhosts() == 0);
    }
}

TEST_CASE("evaluateGhostTree: two-hop cell2cell2node with manual ghost")
{
    if (g_mpi.size < 2)
        return;

    auto c2c = make4QuadCell2Cell(g_mpi);
    auto c2n = make4QuadCell2NodeWithMapping(g_mpi);

    // Node global mapping: rank 0 owns 0-4, rank 1 owns 5-8.
    DNDS::index nNodesLocal = 0;
    if (g_mpi.size == 1)
        nNodesLocal = 9;
    else if (g_mpi.rank == 0)
        nNodesLocal = 5;
    else if (g_mpi.rank == 1)
        nNodesLocal = 4;

    auto nodeGM = std::make_shared<GlobalOffsetsMapping>();
    nodeGM->setMPIAlignBcast(g_mpi, nNodesLocal);

    // Manually ghost cell2node for ghost cells.
    // All ranks must participate in MPI collectives.
    {
        c2n.TransAttach();
        c2n.trans.createFatherGlobalMapping();

        std::vector<DNDS::index> ghostIndices;
        if (g_mpi.rank == 0)
            ghostIndices = {2, 3};
        else if (g_mpi.rank == 1)
            ghostIndices = {0, 1};
        // ranks >= 2: empty ghost set

        c2n.trans.createGhostMapping(ghostIndices);
        c2n.trans.createMPITypes();
        c2n.trans.pullOnce();
    }

    MeshConnectivity dag;
    dag.meshDim = 2;
    dag.registerAdj(Adj::Cell2Cell, c2c);
    dag.registerAdj(Adj::Cell2Node, c2n);
    dag.registerGlobalMapping(EntityKind::Cell, c2c.father->pLGlobalMapping);
    dag.registerGlobalMapping(EntityKind::Node, nodeGM);

    GhostSpec spec{{
        {EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell},
        {EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Node}, EntityKind::Node},
    }};
    auto tree = CompiledGhostTree::compile(spec);
    auto result = dag.evaluateGhostTree(tree, g_mpi);

    // Collective: all ranks agree these kinds have ghosts.
    CHECK(result.hasGhosts(EntityKind::Cell));
    CHECK(result.hasGhosts(EntityKind::Node));

    auto &ghostNodes = result.ghostIndices[EntityKind::Node];

    if (g_mpi.rank == 0)
    {
        // Rank 0 owns nodes 0-4.
        // Ghost cells 2,3 have nodes {3,4,7,6} and {4,5,8,7}.
        // Owned cells 0,1 have nodes {0,1,4,3} and {1,2,5,4}.
        // cell2cell2node = all nodes of owned+ghost cells = {0..8}.
        // Non-owned: {5,6,7,8}. Exact count: 4 ghost nodes.
        CHECK(ghostNodes.size() == 4);
        std::set<DNDS::index> ghostNodeSet(ghostNodes.begin(), ghostNodes.end());
        CHECK(ghostNodeSet == std::set<DNDS::index>{5, 6, 7, 8});
        // Total: 2 ghost cells + 4 ghost nodes = 6.
        CHECK(result.totalGhosts() == 6);
    }
    else if (g_mpi.rank == 1)
    {
        // Rank 1 owns nodes 5-8.
        // cell2cell2node = all nodes of owned+ghost cells = {0..8}.
        // Non-owned: {0,1,2,3,4}. Exact count: 5 ghost nodes.
        CHECK(ghostNodes.size() == 5);
        std::set<DNDS::index> ghostNodeSet(ghostNodes.begin(), ghostNodes.end());
        CHECK(ghostNodeSet == std::set<DNDS::index>{0, 1, 2, 3, 4});
        CHECK(result.totalGhosts() == 7); // 2 ghost cells + 5 ghost nodes
    }
    else
    {
        // Ranks 2+: no cells, no ghosts of any kind.
        CHECK(result.ghostIndices[EntityKind::Node].empty());
        CHECK(result.totalGhosts() == 0);
    }
}

TEST_CASE("evaluateGhostTree: 2-ring cell chain")
{
    if (g_mpi.size < 2)
        return;

    auto c2c = make4QuadCell2Cell(g_mpi);

    // Ghost cell2cell for 2-ring traversal. All ranks must participate.
    {
        c2c.TransAttach();
        c2c.trans.createFatherGlobalMapping();
        std::vector<DNDS::index> ghostIndices;
        if (g_mpi.rank == 0)
            ghostIndices = {2, 3};
        else if (g_mpi.rank == 1)
            ghostIndices = {0, 1};
        c2c.trans.createGhostMapping(ghostIndices);
        c2c.trans.createMPITypes();
        c2c.trans.pullOnce();
    }

    MeshConnectivity dag;
    dag.meshDim = 2;
    dag.registerAdj(Adj::Cell2Cell, c2c);
    dag.registerGlobalMapping(EntityKind::Cell, c2c.father->pLGlobalMapping);

    GhostSpec spec1{{{EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell}}};
    auto result1 = dag.evaluateGhostTree(CompiledGhostTree::compile(spec1), g_mpi);

    GhostSpec spec2{{{EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Cell}, EntityKind::Cell}}};
    auto result2 = dag.evaluateGhostTree(CompiledGhostTree::compile(spec2), g_mpi);

    if (g_mpi.rank < 2)
    {
        // On this mesh, 1-ring already covers all cells. 2-ring is identical.
        auto &ghost1 = result1.ghostIndices[EntityKind::Cell];
        auto &ghost2 = result2.ghostIndices[EntityKind::Cell];
        CHECK(ghost1.size() == 2);
        CHECK(ghost2.size() == 2);
        CHECK(ghost1 == ghost2);
    }
}

TEST_CASE("evaluateGhostTree: union of multiple chains")
{
    if (g_mpi.size < 2)
        return;

    auto c2c = make4QuadCell2Cell(g_mpi);
    auto c2n = make4QuadCell2NodeWithMapping(g_mpi);

    DNDS::index nNodesLocal = 0;
    if (g_mpi.size == 1)
        nNodesLocal = 9;
    else if (g_mpi.rank == 0)
        nNodesLocal = 5;
    else if (g_mpi.rank == 1)
        nNodesLocal = 4;
    auto nodeGM = std::make_shared<GlobalOffsetsMapping>();
    nodeGM->setMPIAlignBcast(g_mpi, nNodesLocal);

    // Ghost cell2node. All ranks participate.
    {
        c2n.TransAttach();
        c2n.trans.createFatherGlobalMapping();
        std::vector<DNDS::index> ghostIndices;
        if (g_mpi.rank == 0)
            ghostIndices = {2, 3};
        else if (g_mpi.rank == 1)
            ghostIndices = {0, 1};
        c2n.trans.createGhostMapping(ghostIndices);
        c2n.trans.createMPITypes();
        c2n.trans.pullOnce();
    }

    MeshConnectivity dag;
    dag.meshDim = 2;
    dag.registerAdj(Adj::Cell2Cell, c2c);
    dag.registerAdj(Adj::Cell2Node, c2n);
    dag.registerGlobalMapping(EntityKind::Cell, c2c.father->pLGlobalMapping);
    dag.registerGlobalMapping(EntityKind::Node, nodeGM);

    // Chain 2 alone.
    GhostSpec spec2{{{EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Node}, EntityKind::Node}}};
    auto result2 = dag.evaluateGhostTree(CompiledGhostTree::compile(spec2), g_mpi);

    // Union of chain 1 (cell2node) + chain 2 (cell2cell2node).
    GhostSpec specUnion{{
        {EntityKind::Cell, {Adj::Cell2Node}, EntityKind::Node},
        {EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Node}, EntityKind::Node},
    }};
    auto resultUnion = dag.evaluateGhostTree(CompiledGhostTree::compile(specUnion), g_mpi);

    if (g_mpi.rank == 0)
    {
        // Chain 2 alone: cell2cell→cell2node on all 4 cells → nodes {0..8},
        // non-owned [0,5) → ghost = {5,6,7,8}.
        auto &ghost2 = result2.ghostIndices[EntityKind::Node];
        std::set<DNDS::index> set2(ghost2.begin(), ghost2.end());
        CHECK(set2 == std::set<DNDS::index>{5, 6, 7, 8});

        // Union of chain1 (owned cell2node → ghost {5}) + chain2 (→ ghost {5,6,7,8}).
        // Exact: {5,6,7,8}.
        auto &ghostU = resultUnion.ghostIndices[EntityKind::Node];
        std::set<DNDS::index> setU(ghostU.begin(), ghostU.end());
        CHECK(setU == std::set<DNDS::index>{5, 6, 7, 8});
    }
    else if (g_mpi.rank == 1)
    {
        // Chain 2 alone: rank 1 owns nodes [5,9). Ghost = {0,1,2,3,4}.
        auto &ghost2 = result2.ghostIndices[EntityKind::Node];
        std::set<DNDS::index> set2(ghost2.begin(), ghost2.end());
        CHECK(set2 == std::set<DNDS::index>{0, 1, 2, 3, 4});

        // Union: chain1 on rank 1 owned cells {2,3} → nodes {3,4,5,6,7,8},
        // ghost from chain1 = {3,4} (outside [5,9)).
        // Union = {3,4} ∪ {0,1,2,3,4} = {0,1,2,3,4}.
        auto &ghostU = resultUnion.ghostIndices[EntityKind::Node];
        std::set<DNDS::index> setU(ghostU.begin(), ghostU.end());
        CHECK(setU == std::set<DNDS::index>{0, 1, 2, 3, 4});
    }
    else
    {
        // Ranks 2+: no cells, no ghost nodes.
        CHECK(resultUnion.ghostIndices[EntityKind::Node].empty());
    }
}

TEST_CASE("evaluateGhostTree: np=1 produces no ghosts")
{
    if (g_mpi.size != 1)
        return;

    auto c2c = make4QuadCell2Cell(g_mpi);

    MeshConnectivity dag;
    dag.meshDim = 2;
    dag.registerAdj(Adj::Cell2Cell, c2c);
    dag.registerGlobalMapping(EntityKind::Cell, c2c.father->pLGlobalMapping);

    GhostSpec spec{{{EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell}}};
    auto result = dag.evaluateGhostTree(CompiledGhostTree::compile(spec), g_mpi);

    CHECK(!result.hasGhosts(EntityKind::Cell));
    CHECK(result.totalGhosts() == 0);
}

// ===========================================================================
// Doubly-periodic tiled grid test
// ===========================================================================

/// Generate a doubly-periodic NxN-per-rank tiled grid.
///
/// Tiled rectangular grid with optional periodic wrapping in both directions.
/// 2D rank decomposition per sqrt(np); falls back to 1×np when not factorable.

TEST_CASE("evaluateGhostTree: doubly-periodic tiled grid")
{
    if (g_mpi.size < 2)
        return;

    DNDS::index N = 32;
    SyntheticTiledGrid grid;
    grid.periodic = true;
    grid.build(N, g_mpi);

    MeshConnectivity dag;
    dag.meshDim = 2;
    dag.registerAdj(Adj::Cell2Cell, grid.cell2cell);
    dag.registerAdj(Adj::Cell2Node, grid.cell2node);
    dag.registerGlobalMapping(EntityKind::Cell, grid.cellGM);
    dag.registerGlobalMapping(EntityKind::Node, grid.nodeGM);

    // --- Cell + node ghost via defaultPrimary-style unified spec ---
    for (int nL = 1; nL <= 4; nL++)
    {
        std::vector<AdjKind> cellHops(nL, Adj::Cell2Cell);
        std::vector<AdjKind> nodeHops(nL, Adj::Cell2Cell);
        nodeHops.push_back(Adj::Cell2Node);

        GhostSpec spec{{
            {EntityKind::Cell, cellHops, EntityKind::Cell},
            {EntityKind::Cell, nodeHops, EntityKind::Node},
        }};
        auto result = dag.evaluateGhostTree(
            CompiledGhostTree::compile(spec), g_mpi);

        auto &gcL = result.ghostIndices[EntityKind::Cell];
        auto &gnL = result.ghostIndices[EntityKind::Node];
        CHECK(gcL.size() == grid.expectedGhostCellsN(nL));
        CHECK(gnL.size() == grid.expectedGhostNodesN(nL));

        const char *pos = (grid.rankCol == 0 || grid.rankCol == grid.ranksPerRow - 1 ||
                           grid.rankRow == 0 || grid.rankRow == grid.ranksPerCol - 1)
                              ? "boundary"
                              : "inner";
        fmt::print("  [{}] {} Periodic N={}: nLayers={}: cell_ghost={} (expected={}), "
                   "node_ghost={} (expected={})\n",
                   g_mpi.rank, pos, N, nL,
                   gcL.size(), grid.expectedGhostCellsN(nL),
                   gnL.size(), grid.expectedGhostNodesN(nL));
    }

    // --- n-layer cell ghost ---
    for (int nL = 2; nL <= 4; nL++)
    {
        std::vector<AdjKind> hops(nL, Adj::Cell2Cell);
        GhostSpec specML{{{EntityKind::Cell, hops, EntityKind::Cell}}};
        auto resultML = dag.evaluateGhostTree(
            CompiledGhostTree::compile(specML), g_mpi);
        auto &gcL = resultML.ghostIndices[EntityKind::Cell];
        CHECK(gcL.size() == grid.expectedGhostCellsN(nL));

        fmt::print("  [{}] Periodic N={}: {}-layer cell ghost: size={}\n",
                   g_mpi.rank, N, nL, gcL.size());
    }
}

TEST_CASE("evaluateGhostTree: small periodic tiled grid")
{
    if (g_mpi.size != 4 && g_mpi.size != 8)
        return;

    DNDS::index N = 2;
    SyntheticTiledGrid grid;
    grid.periodic = true;
    grid.build(N, g_mpi);

    MeshConnectivity dag;
    dag.meshDim = 2;
    dag.registerAdj(Adj::Cell2Cell, grid.cell2cell);
    dag.registerAdj(Adj::Cell2Node, grid.cell2node);
    dag.registerGlobalMapping(EntityKind::Cell, grid.cellGM);
    dag.registerGlobalMapping(EntityKind::Node, grid.nodeGM);

    // 1-hop cell ghost.
    GhostSpec spec1{{{EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell}}};
    auto result1 = dag.evaluateGhostTree(CompiledGhostTree::compile(spec1), g_mpi);
    auto &ghostCells = result1.ghostIndices[EntityKind::Cell];
    DNDS::index expected1 = grid.expectedGhostCellsN(1);
    CHECK(ghostCells.size() == expected1);

    auto expectedCells = grid.expectedGhostCells(g_mpi);
    CHECK(ghostCells.size() == expectedCells.size());

    // 2-hop cell ghost: check via brute-force expected set.
    {
        GhostSpec spec2{{{EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Cell}, EntityKind::Cell}}};
        auto result2 = dag.evaluateGhostTree(CompiledGhostTree::compile(spec2), g_mpi);
        auto &gc2 = result2.ghostIndices[EntityKind::Cell];
        CHECK(gc2.size() == grid.expectedGhostCellsN(2));
    }

    // Node ghost.
    GhostSpec specNodes{{
        {EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Node}, EntityKind::Node},
    }};
    auto resultNodes = dag.evaluateGhostTree(CompiledGhostTree::compile(specNodes), g_mpi);
    auto &ghostNodes = resultNodes.ghostIndices[EntityKind::Node];
    CHECK(ghostNodes.size() == grid.expectedGhostNodesN(1));

    fmt::print("  [{}] np={} Periodic N=2: 1-hop cells={} (expected={}), "
               "nodes={} (expected={})\n",
               g_mpi.rank, g_mpi.size,
               ghostCells.size(), expected1,
               ghostNodes.size(), grid.expectedGhostNodesN(1));
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
