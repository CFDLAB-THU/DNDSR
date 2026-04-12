/**
 * @file test_MeshIndexConversion.cpp
 * @brief Doctest-based unit tests for UnstructuredMesh index conversion and
 *        adjacency Global2Local / Local2Global methods.
 *
 * Reads a real CGNS mesh (UniformSquare_10.cgns -- 100 quad cells, non-periodic),
 * partitions it with Metis, builds ghost layers through the normal
 * UnstructuredMeshSerialRW pipeline, then exercises:
 *
 *   1. Per-entity index conversions (Global2Local, Local2Global, _NoSon)
 *   2. Adjacency state-machine round-trips (Primary, ForBnd)
 *   3. Round-trip stability under repeated conversions
 *   4. UnInitIndex pass-through
 *   5. Negative encoding for not-found globals
 *   6. Local index range validity after AdjGlobal2LocalPrimary
 *
 * All meshes are pre-built in main() and destroyed together after tests.
 * This avoids an OpenMPI resource exhaustion observed when building/destroying
 * 20+ meshes in a single process at np=8 (each mesh creates dozens of MPI
 * persistent requests via ArrayTransformer; OpenMPI's shared-memory BTL
 * appears to leak internal state across request lifecycles).
 *
 *   - g_ro:       shared read-only mesh for 19 non-mutating tests
 *   - g_mut[0..2]: one per mutating adjacency test
 *
 * Run under mpirun with 1, 2, 4, and 8 ranks.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "Geom/Mesh.hpp"
#include <string>
#include <vector>

using namespace DNDS;
using namespace DNDS::Geom;

// NOTE: DNDS::index, DNDS::real, DNDS::rowsize clash with POSIX symbols
// from <strings.h> (pulled in by doctest/MPI).  Every declaration of these
// types must use the fully-qualified DNDS:: prefix.  See AGENTS.md.

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static MPIInfo g_mpi;
static ssp<UnstructuredMesh> g_ro;     ///< Read-only mesh shared across most tests.
static ssp<UnstructuredMesh> g_mut[3]; ///< One per mutating test.
static int g_mutNext = 0;

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

static ssp<UnstructuredMesh> buildMesh(int meshId)
{
    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] START" << std::endl;

    auto mesh = std::make_shared<UnstructuredMesh>(g_mpi, 2);
    UnstructuredMeshSerialRW reader(mesh, 0);

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] ReadFromCGNSSerial..." << std::endl;
    reader.ReadFromCGNSSerial(meshPath("UniformSquare_10.cgns"));

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] Deduplicate1to1Periodic..." << std::endl;
    reader.Deduplicate1to1Periodic(1e-8);

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] BuildCell2Cell..." << std::endl;
    reader.BuildCell2Cell();

    UnstructuredMeshSerialRW::PartitionOptions pOpt;
    pOpt.metisType = "KWAY";
    pOpt.metisUfactor = 30;
    pOpt.metisSeed = 42;
    pOpt.metisNcuts = 1;

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] MeshPartitionCell2Cell..." << std::endl;
    reader.MeshPartitionCell2Cell(pOpt);

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] PartitionReorderToMeshCell2Cell..." << std::endl;
    reader.PartitionReorderToMeshCell2Cell();

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] RecoverNode2CellAndNode2Bnd..." << std::endl;
    mesh->RecoverNode2CellAndNode2Bnd();

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] RecoverCell2CellAndBnd2Cell..." << std::endl;
    mesh->RecoverCell2CellAndBnd2Cell();

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] BuildGhostPrimary..." << std::endl;
    mesh->BuildGhostPrimary();

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] AdjGlobal2LocalPrimary..." << std::endl;
    mesh->AdjGlobal2LocalPrimary();

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] DONE" << std::endl;

    return mesh;
}

/// Return next pre-built mutable mesh (one per mutating test).
static ssp<UnstructuredMesh> nextMut()
{
    DNDS_assert(g_mutNext < 3);
    return g_mut[g_mutNext++];
}

static std::vector<std::vector<DNDS::index>> snapshotAdj(
    const tAdjPair &adj, DNDS::index nRows)
{
    std::vector<std::vector<DNDS::index>> out(nRows);
    for (DNDS::index i = 0; i < nRows; i++)
    {
        out[i].resize(adj.RowSize(i));
        for (DNDS::rowsize j = 0; j < adj.RowSize(i); j++)
            out[i][j] = adj(i, j);
    }
    return out;
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();

    // Pre-build: 1 read-only + 3 mutable = 4 meshes total.
    // All built here so MPI collectives are synchronized across ranks.
    g_ro = buildMesh(0);
    for (int i = 0; i < 3; i++)
        g_mut[i] = buildMesh(i + 1);

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    // Destroy all meshes together before MPI_Finalize.
    g_ro.reset();
    for (auto &m : g_mut)
        m.reset();
    MPI_Finalize();
    return res;
}

// ===========================================================================
// Mesh sanity checks  (read-only)
// ===========================================================================
TEST_CASE("Mesh setup: every rank has cells and nodes")
{
    auto m = g_ro;
    CHECK(m->NumCell() >= 1);
    CHECK(m->NumNode() >= 1);
    CHECK(m->adjPrimaryState == Adj_PointToLocal);
}

TEST_CASE("Mesh setup: multi-rank produces ghosts")
{
    if (g_mpi.size < 2)
        return;
    auto m = g_ro;
    CHECK(m->NumCellGhost() > 0);
    CHECK(m->NumNodeGhost() > 0);
}

// ===========================================================================
// Per-entity index conversions -- Node  (read-only)
// ===========================================================================
TEST_CASE("NodeIndex: Global2Local round-trip on father nodes")
{
    auto m = g_ro;
    for (DNDS::index iNode = 0; iNode < m->NumNode(); iNode++)
    {
        DNDS::index g = m->NodeIndexLocal2Global(iNode);
        CHECK(g >= 0);
        CHECK(m->NodeIndexGlobal2Local(g) == iNode);
    }
}

TEST_CASE("NodeIndex: Global2Local round-trip on ghost nodes")
{
    if (g_mpi.size < 2)
        return;
    auto m = g_ro;
    for (DNDS::index i = m->NumNode(); i < m->NumNode() + m->NumNodeGhost(); i++)
    {
        DNDS::index g = m->NodeIndexLocal2Global(i);
        CHECK(g >= 0);
        CHECK(m->NodeIndexGlobal2Local(g) == i);
    }
}

TEST_CASE("NodeIndex: _NoSon round-trip on father nodes")
{
    auto m = g_ro;
    for (DNDS::index iNode = 0; iNode < m->NumNode(); iNode++)
    {
        DNDS::index g = m->NodeIndexLocal2Global_NoSon(iNode);
        CHECK(g >= 0);
        CHECK(m->NodeIndexGlobal2Local_NoSon(g) == iNode);
    }
}

TEST_CASE("NodeIndex: _NoSon returns negative for non-local global")
{
    if (g_mpi.size < 2)
        return;
    auto m = g_ro;
    MPI_int other = (g_mpi.rank + 1) % g_mpi.size;
    DNDS::index g = (*m->coords.father->pLGlobalMapping)(other, 0);
    CHECK(m->NodeIndexGlobal2Local_NoSon(g) < 0);
}

TEST_CASE("NodeIndex: Global2Local returns negative for unknown global")
{
    auto m = g_ro;
    DNDS::index bogus = m->coords.father->globalSize() + 999;
    DNDS::index result = m->NodeIndexGlobal2Local(bogus);
    CHECK(result < 0);
    CHECK(result == -1 - bogus);
}

// ===========================================================================
// Per-entity index conversions -- Cell  (read-only)
// ===========================================================================
TEST_CASE("CellIndex: Global2Local round-trip on father cells")
{
    auto m = g_ro;
    for (DNDS::index iCell = 0; iCell < m->NumCell(); iCell++)
    {
        DNDS::index g = m->CellIndexLocal2Global(iCell);
        CHECK(g >= 0);
        CHECK(m->CellIndexGlobal2Local(g) == iCell);
    }
}

TEST_CASE("CellIndex: Global2Local round-trip on ghost cells")
{
    if (g_mpi.size < 2)
        return;
    auto m = g_ro;
    CHECK(m->NumCellGhost() > 0);
    for (DNDS::index i = m->NumCell(); i < m->NumCell() + m->NumCellGhost(); i++)
    {
        DNDS::index g = m->CellIndexLocal2Global(i);
        CHECK(g >= 0);
        CHECK(m->CellIndexGlobal2Local(g) == i);
    }
}

TEST_CASE("CellIndex: _NoSon round-trip on father cells")
{
    auto m = g_ro;
    for (DNDS::index iCell = 0; iCell < m->NumCell(); iCell++)
    {
        DNDS::index g = m->CellIndexLocal2Global_NoSon(iCell);
        CHECK(g >= 0);
        CHECK(m->CellIndexGlobal2Local_NoSon(g) == iCell);
    }
}

TEST_CASE("CellIndex: _NoSon returns negative for non-local global")
{
    if (g_mpi.size < 2)
        return;
    auto m = g_ro;
    MPI_int other = (g_mpi.rank + 1) % g_mpi.size;
    DNDS::index g = (*m->cell2node.father->pLGlobalMapping)(other, 0);
    CHECK(m->CellIndexGlobal2Local_NoSon(g) < 0);
}

// ===========================================================================
// Per-entity index conversions -- Bnd  (read-only)
// ===========================================================================
TEST_CASE("BndIndex: Global2Local round-trip on father bnds")
{
    auto m = g_ro;
    for (DNDS::index iBnd = 0; iBnd < m->NumBnd(); iBnd++)
    {
        DNDS::index g = m->BndIndexLocal2Global(iBnd);
        CHECK(g >= 0);
        CHECK(m->BndIndexGlobal2Local(g) == iBnd);
    }
}

TEST_CASE("BndIndex: _NoSon round-trip on father bnds")
{
    auto m = g_ro;
    for (DNDS::index iBnd = 0; iBnd < m->NumBnd(); iBnd++)
    {
        DNDS::index g = m->BndIndexLocal2Global_NoSon(iBnd);
        CHECK(g >= 0);
        CHECK(m->BndIndexGlobal2Local_NoSon(g) == iBnd);
    }
}

// ===========================================================================
// UnInitIndex pass-through and special encodings  (read-only)
// ===========================================================================
TEST_CASE("UnInitIndex pass-through for all 12 conversion methods")
{
    auto m = g_ro;

    CHECK(m->NodeIndexGlobal2Local(UnInitIndex) == UnInitIndex);
    CHECK(m->CellIndexGlobal2Local(UnInitIndex) == UnInitIndex);
    CHECK(m->BndIndexGlobal2Local(UnInitIndex) == UnInitIndex);

    CHECK(m->NodeIndexLocal2Global(UnInitIndex) == UnInitIndex);
    CHECK(m->CellIndexLocal2Global(UnInitIndex) == UnInitIndex);
    CHECK(m->BndIndexLocal2Global(UnInitIndex) == UnInitIndex);

    CHECK(m->NodeIndexLocal2Global_NoSon(UnInitIndex) == UnInitIndex);
    CHECK(m->CellIndexLocal2Global_NoSon(UnInitIndex) == UnInitIndex);
    CHECK(m->BndIndexLocal2Global_NoSon(UnInitIndex) == UnInitIndex);

    CHECK(m->NodeIndexGlobal2Local_NoSon(UnInitIndex) == UnInitIndex);
    CHECK(m->CellIndexGlobal2Local_NoSon(UnInitIndex) == UnInitIndex);
    CHECK(m->BndIndexGlobal2Local_NoSon(UnInitIndex) == UnInitIndex);
}

TEST_CASE("Local2Global: negative local index decodes via -1-x encoding")
{
    auto m = g_ro;
    DNDS::index fakeGlobal = 42;
    DNDS::index neg = -1 - fakeGlobal;

    CHECK(m->NodeIndexLocal2Global(neg) == fakeGlobal);
    CHECK(m->CellIndexLocal2Global(neg) == fakeGlobal);
    CHECK(m->BndIndexLocal2Global(neg) == fakeGlobal);
}

// ===========================================================================
// Local index range validity  (read-only)
// ===========================================================================
TEST_CASE("AdjPrimary: cell2node local indices are in valid range")
{
    auto m = g_ro;
    REQUIRE(m->adjPrimaryState == Adj_PointToLocal);
    DNDS::index totalNodes = m->NumNode() + m->NumNodeGhost();
    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        for (DNDS::rowsize j = 0; j < m->cell2node.RowSize(iC); j++)
        {
            DNDS::index iNode = m->cell2node(iC, j);
            CHECK(iNode >= 0);
            CHECK(iNode < totalNodes);
        }
}

TEST_CASE("AdjPrimary: cell2cell local entries are valid or not-found")
{
    auto m = g_ro;
    REQUIRE(m->adjPrimaryState == Adj_PointToLocal);
    DNDS::index totalCells = m->NumCell() + m->NumCellGhost();
    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        for (DNDS::rowsize j = 0; j < m->cell2cell.RowSize(iC); j++)
        {
            DNDS::index n = m->cell2cell(iC, j);
            if (n >= 0)
                CHECK(n < totalCells);
        }
}

TEST_CASE("AdjPrimary: bnd2cell owner cell is a local father cell")
{
    auto m = g_ro;
    REQUIRE(m->adjPrimaryState == Adj_PointToLocal);
    for (DNDS::index ib = 0; ib < m->NumBnd(); ib++)
    {
        DNDS::index owner = m->bnd2cell(ib, 0);
        CHECK(owner >= 0);
        CHECK(owner < m->NumCell());
    }
}

TEST_CASE("AdjPrimary: bnd2node local indices are in valid range")
{
    auto m = g_ro;
    REQUIRE(m->adjPrimaryState == Adj_PointToLocal);
    DNDS::index totalNodes = m->NumNode() + m->NumNodeGhost();
    for (DNDS::index ib = 0; ib < m->NumBnd(); ib++)
        for (DNDS::rowsize j = 0; j < m->bnd2node.RowSize(ib); j++)
        {
            DNDS::index iNode = m->bnd2node(ib, j);
            CHECK(iNode >= 0);
            CHECK(iNode < totalNodes);
        }
}

// ===========================================================================
// Adjacency state machine -- each gets its own pre-built mesh via nextMut()
// ===========================================================================
TEST_CASE("AdjPrimary: Local2Global then Global2Local is identity")
{
    auto m = nextMut();
    REQUIRE(m->adjPrimaryState == Adj_PointToLocal);

    auto c2nSnap = snapshotAdj(m->cell2node, m->NumCell());
    auto c2cSnap = snapshotAdj(m->cell2cell, m->NumCell());

    m->AdjLocal2GlobalPrimary();
    CHECK(m->adjPrimaryState == Adj_PointToGlobal);

    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        for (DNDS::rowsize j = 0; j < m->cell2node.RowSize(iC); j++)
        {
            DNDS::index gNode = m->cell2node(iC, j);
            CHECK(gNode >= 0);
            // globalSize() is now non-collective (cached), safe to call in loops
            CHECK(gNode < m->coords.father->globalSize());
        }

    m->AdjGlobal2LocalPrimary();
    CHECK(m->adjPrimaryState == Adj_PointToLocal);

    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
    {
        REQUIRE(static_cast<DNDS::rowsize>(c2nSnap[iC].size()) == m->cell2node.RowSize(iC));
        for (DNDS::rowsize j = 0; j < m->cell2node.RowSize(iC); j++)
            CHECK(m->cell2node(iC, j) == c2nSnap[iC][j]);
        REQUIRE(static_cast<DNDS::rowsize>(c2cSnap[iC].size()) == m->cell2cell.RowSize(iC));
        for (DNDS::rowsize j = 0; j < m->cell2cell.RowSize(iC); j++)
            CHECK(m->cell2cell(iC, j) == c2cSnap[iC][j]);
    }
}

TEST_CASE("AdjPrimary: three consecutive round-trips are stable")
{
    auto m = nextMut();
    auto snap = snapshotAdj(m->cell2node, m->NumCell());

    for (int trip = 0; trip < 3; trip++)
    {
        m->AdjLocal2GlobalPrimary();
        CHECK(m->adjPrimaryState == Adj_PointToGlobal);
        m->AdjGlobal2LocalPrimary();
        CHECK(m->adjPrimaryState == Adj_PointToLocal);
    }

    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        for (DNDS::rowsize j = 0; j < m->cell2node.RowSize(iC); j++)
            CHECK(m->cell2node(iC, j) == snap[iC][j]);
}

TEST_CASE("AdjPrimaryForBnd: round-trip on cell2node only")
{
    auto m = nextMut();
    REQUIRE(m->adjPrimaryState == Adj_PointToLocal);

    auto snap = snapshotAdj(m->cell2node, m->NumCell());

    m->AdjLocal2GlobalPrimaryForBnd();
    CHECK(m->adjPrimaryState == Adj_PointToGlobal);

    m->AdjGlobal2LocalPrimaryForBnd();
    CHECK(m->adjPrimaryState == Adj_PointToLocal);

    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        for (DNDS::rowsize j = 0; j < m->cell2node.RowSize(iC); j++)
            CHECK(m->cell2node(iC, j) == snap[iC][j]);
}
