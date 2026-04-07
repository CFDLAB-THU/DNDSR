/**
 * @file test_MeshPipeline.cpp
 * @brief Doctest tests for the full mesh pipeline covering refactored code
 *        from Phases 1-3 and the complete Adj state machine.
 *
 * Tests cover:
 *   - Pipeline state tracking: every state variable checked at each step
 *   - InterpolateFace (Phase 2b): face topology, face2cell, bnd2face
 *   - ReorderLocalCells (Phase 2c): cell permutation, partition starts
 *   - All 5 Adj conversion round-trips as perfect inverses:
 *       Primary, Facial, C2F, N2CB, C2CFace
 *   - ConstructBndMesh + AdjForBnd round-trip
 *   - BuildVTKConnectivity: VTK output arrays
 *   - Global conservation: cell, bnd, face counts
 *
 * Pre-built meshes in main() to avoid OpenMPI resource exhaustion at np=8.
 *   g_full:     read-only, full pipeline (faces, N2CB, C2CFace, VTK)
 *   g_reord:    read-only, with ReorderLocalCells(2)
 *   g_mut[0..5]: one per mutating round-trip test
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "Geom/Mesh.hpp"
#include <string>
#include <vector>

using namespace DNDS;
using namespace DNDS::Geom;

// NOTE: DNDS::index, DNDS::real, DNDS::rowsize clash with POSIX symbols.
// Every declaration must use the fully-qualified DNDS:: prefix.

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static MPIInfo g_mpi;
static ssp<UnstructuredMesh> g_full;
static ssp<UnstructuredMesh> g_reord;

static constexpr int N_MUT = 6;
static ssp<UnstructuredMesh> g_mut[N_MUT];
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

/// Build a mesh through the full pipeline including faces.
static ssp<UnstructuredMesh> buildFullMesh(int meshId, int nReorderParts = 1)
{
    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] START" << std::endl;

    auto mesh = std::make_shared<UnstructuredMesh>(g_mpi, 2);
    UnstructuredMeshSerialRW reader(mesh, 0);

    // --- Serial read and partition ---
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

    // --- Topology recovery (global state) ---
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();

    // --- Localize primary and N2CB ---
    mesh->AdjGlobal2LocalPrimary();
    mesh->AdjGlobal2LocalN2CB();

    // --- Optional reorder ---
    if (nReorderParts > 1)
        mesh->ReorderLocalCells(nReorderParts);

    // --- Face interpolation ---
    mesh->InterpolateFace();
    mesh->AssertOnFaces();

    // --- N2CB ghost expansion ---
    mesh->AdjLocal2GlobalN2CB();
    mesh->BuildGhostN2CB();
    mesh->AdjGlobal2LocalN2CB();
    mesh->AssertOnN2CB();

    // --- Cell2CellFace ---
    mesh->BuildCell2CellFace();
    mesh->AdjGlobal2LocalC2CFace();

    // --- VTK connectivity ---
    mesh->BuildVTKConnectivity();

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << "] DONE" << std::endl;

    return mesh;
}

/// Return next pre-built mutable mesh (one per mutating test).
static ssp<UnstructuredMesh> nextMut()
{
    DNDS_assert(g_mutNext < N_MUT);
    return g_mut[g_mutNext++];
}

/// Snapshot a variable-row adjacency array into a vector-of-vectors.
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

/// Snapshot a fixed-2-column adjacency (face2cell, bnd2cell).
static std::vector<std::pair<DNDS::index, DNDS::index>> snapshotAdj2(
    const tAdj2Pair &adj, DNDS::index nRows)
{
    std::vector<std::pair<DNDS::index, DNDS::index>> out(nRows);
    for (DNDS::index i = 0; i < nRows; i++)
        out[i] = {adj(i, 0), adj(i, 1)};
    return out;
}

/// Snapshot a single-column adjacency (bnd2face, face2bnd).
static std::vector<DNDS::index> snapshotAdj1(
    const tAdj1Pair &adj, DNDS::index nRows)
{
    std::vector<DNDS::index> out(nRows);
    for (DNDS::index i = 0; i < nRows; i++)
        out[i] = adj(i, 0);
    return out;
}

/// Compare a variable-row adjacency against a snapshot.
static void checkAdjMatchesSnapshot(
    const tAdjPair &adj, DNDS::index nRows,
    const std::vector<std::vector<DNDS::index>> &snap)
{
    for (DNDS::index i = 0; i < nRows; i++)
    {
        REQUIRE(adj.RowSize(i) == static_cast<DNDS::rowsize>(snap[i].size()));
        for (DNDS::rowsize j = 0; j < adj.RowSize(i); j++)
            CHECK(adj(i, j) == snap[i][j]);
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();

    g_full = buildFullMesh(0);
    g_reord = buildFullMesh(1, 2);
    for (int i = 0; i < N_MUT; i++)
        g_mut[i] = buildFullMesh(i + 2);

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    g_full.reset();
    g_reord.reset();
    for (auto &m : g_mut)
        m.reset();
    MPI_Finalize();
    return res;
}

// ===========================================================================
// Pipeline state tracking (read-only on g_full)
// ===========================================================================

TEST_CASE("Pipeline state: all five states are Local after full pipeline")
{
    auto m = g_full;
    CHECK(m->adjPrimaryState == Adj_PointToLocal);
    CHECK(m->adjFacialState == Adj_PointToLocal);
    CHECK(m->adjC2FState == Adj_PointToLocal);
    CHECK(m->adjN2CBState == Adj_PointToLocal);
    CHECK(m->adjC2CFaceState == Adj_PointToLocal);
}

TEST_CASE("Pipeline state: reordered mesh also has all states Local")
{
    auto m = g_reord;
    CHECK(m->adjPrimaryState == Adj_PointToLocal);
    CHECK(m->adjFacialState == Adj_PointToLocal);
    CHECK(m->adjC2FState == Adj_PointToLocal);
    CHECK(m->adjN2CBState == Adj_PointToLocal);
    CHECK(m->adjC2CFaceState == Adj_PointToLocal);
}

// ===========================================================================
// Global count conservation
// ===========================================================================

TEST_CASE("Pipeline: global cell count sums to 100")
{
    auto m = g_full;
    DNDS::index localCells = m->NumCell();
    DNDS::index globalCells = 0;
    MPI_Allreduce(&localCells, &globalCells, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    CHECK(globalCells == 100);
}

TEST_CASE("Pipeline: global bnd count sums to 40")
{
    auto m = g_full;
    DNDS::index localBnds = m->NumBnd();
    DNDS::index globalBnds = 0;
    MPI_Allreduce(&localBnds, &globalBnds, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    CHECK(globalBnds == 40);
}

TEST_CASE("Pipeline: global face count is conserved after reorder")
{
    DNDS::index nFaceFull = 0, nFaceReord = 0;
    {
        DNDS::index loc = g_full->NumFace();
        MPI_Allreduce(&loc, &nFaceFull, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    }
    {
        DNDS::index loc = g_reord->NumFace();
        MPI_Allreduce(&loc, &nFaceReord, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    }
    CHECK(nFaceFull > 0);
    CHECK(nFaceFull == nFaceReord);
}

// ===========================================================================
// InterpolateFace: face topology validation (Phase 2b)
// ===========================================================================

TEST_CASE("InterpolateFace: face count is positive")
{
    CHECK(g_full->NumFace() > 0);
}

TEST_CASE("InterpolateFace: face2cell has exactly 2 entries per face")
{
    for (DNDS::index iF = 0; iF < g_full->NumFace(); iF++)
        CHECK(g_full->face2cell.RowSize(iF) == 2);
}

TEST_CASE("InterpolateFace: face2cell owner is a valid local cell")
{
    auto m = g_full;
    DNDS::index totalCells = m->NumCell() + m->NumCellGhost();
    for (DNDS::index iF = 0; iF < m->NumFace(); iF++)
    {
        DNDS::index owner = m->face2cell(iF, 0);
        CHECK(owner >= 0);
        CHECK(owner < totalCells);
    }
}

TEST_CASE("InterpolateFace: face2node indices in valid range")
{
    auto m = g_full;
    DNDS::index totalNodes = m->NumNode() + m->NumNodeGhost();
    for (DNDS::index iF = 0; iF < m->NumFace(); iF++)
        for (DNDS::rowsize j = 0; j < m->face2node.RowSize(iF); j++)
        {
            DNDS::index iN = m->face2node(iF, j);
            CHECK(iN >= 0);
            CHECK(iN < totalNodes);
        }
}

TEST_CASE("InterpolateFace: cell2face row sizes match cell2node")
{
    auto m = g_full;
    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        CHECK(m->cell2face.RowSize(iC) == m->cell2node.RowSize(iC));
}

TEST_CASE("InterpolateFace: cell2face entries are valid face indices")
{
    auto m = g_full;
    DNDS::index totalFaces = m->NumFace() + m->NumFaceGhost();
    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        for (DNDS::rowsize j = 0; j < m->cell2face.RowSize(iC); j++)
        {
            DNDS::index iF = m->cell2face(iC, j);
            CHECK(iF >= 0);
            CHECK(iF < totalFaces);
        }
}

TEST_CASE("InterpolateFace: bnd2face maps to valid faces")
{
    auto m = g_full;
    for (DNDS::index ib = 0; ib < m->NumBnd(); ib++)
    {
        DNDS::index iFace = m->bnd2face(ib, 0);
        CHECK(iFace >= 0);
        CHECK(iFace < m->NumFace() + m->NumFaceGhost());
    }
}

TEST_CASE("InterpolateFace: face element types are valid")
{
    for (DNDS::index iF = 0; iF < g_full->NumFace(); iF++)
        CHECK(g_full->GetFaceElement(iF).type != Elem::ElemType::UnknownElem);
}

// ===========================================================================
// N2CB adjacency
// ===========================================================================

TEST_CASE("N2CB: every local node has at least one adjacent cell")
{
    auto m = g_full;
    for (DNDS::index iNode = 0; iNode < m->NumNode(); iNode++)
    {
        bool hasCell = false;
        for (DNDS::rowsize j = 0; j < m->node2cell.RowSize(iNode); j++)
            if (m->node2cell(iNode, j) >= 0)
                hasCell = true;
        CHECK(hasCell);
    }
}

TEST_CASE("N2CB: node2cell entries are valid local indices")
{
    auto m = g_full;
    DNDS::index totalCells = m->NumCell() + m->NumCellGhost();
    for (DNDS::index iNode = 0; iNode < m->NumNodeProc(); iNode++)
        for (DNDS::rowsize j = 0; j < m->node2cell.RowSize(iNode); j++)
        {
            DNDS::index iC = m->node2cell(iNode, j);
            if (iC >= 0)
                CHECK(iC < totalCells);
        }
}

// ===========================================================================
// BuildCell2CellFace
// ===========================================================================

TEST_CASE("BuildCell2CellFace: row sizes match cell2face")
{
    auto m = g_full;
    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        CHECK(m->cell2cellFace.RowSize(iC) == m->cell2face.RowSize(iC));
}

TEST_CASE("BuildCell2CellFace: entries are valid cells or negative")
{
    auto m = g_full;
    DNDS::index totalCells = m->NumCell() + m->NumCellGhost();
    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        for (DNDS::rowsize j = 0; j < m->cell2cellFace.RowSize(iC); j++)
        {
            DNDS::index nb = m->cell2cellFace(iC, j);
            if (nb >= 0)
                CHECK(nb < totalCells);
        }
}

// ===========================================================================
// ReorderLocalCells (Phase 2c)
// ===========================================================================

TEST_CASE("ReorderLocalCells: cell count preserved")
{
    DNDS::index g1 = 0, g2 = 0;
    {
        DNDS::index loc = g_full->NumCell();
        MPI_Allreduce(&loc, &g1, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    }
    {
        DNDS::index loc = g_reord->NumCell();
        MPI_Allreduce(&loc, &g2, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    }
    CHECK(g1 == g2);
}

TEST_CASE("ReorderLocalCells: partition starts are valid")
{
    auto m = g_reord;
    int nParts = m->NLocalParts();
    CHECK(nParts == 2);
    CHECK(m->LocalPartStart(0) == 0);
    CHECK(m->LocalPartEnd(nParts - 1) == m->NumCell());
    for (int ip = 0; ip < nParts; ip++)
        CHECK(m->LocalPartStart(ip) <= m->LocalPartEnd(ip));
}

TEST_CASE("ReorderLocalCells: cell2node still valid")
{
    auto m = g_reord;
    DNDS::index totalNodes = m->NumNode() + m->NumNodeGhost();
    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        for (DNDS::rowsize j = 0; j < m->cell2node.RowSize(iC); j++)
        {
            DNDS::index iN = m->cell2node(iC, j);
            CHECK(iN >= 0);
            CHECK(iN < totalNodes);
        }
}

TEST_CASE("ReorderLocalCells: cellElemInfo non-degenerate")
{
    for (DNDS::index iC = 0; iC < g_reord->NumCell(); iC++)
        CHECK(g_reord->GetCellElement(iC).type != Elem::ElemType::UnknownElem);
}

// ===========================================================================
// ConstructBndMesh
// ===========================================================================

TEST_CASE("ConstructBndMesh: correct dimensions and cell count")
{
    auto m = g_full;
    UnstructuredMesh bMesh(g_mpi, m->dim - 1);
    m->ConstructBndMesh(bMesh);

    CHECK(bMesh.dim == m->dim - 1);
    CHECK(bMesh.NumCell() == m->NumBnd());
    CHECK(bMesh.NumNode() > 0);
    CHECK(bMesh.adjPrimaryState == Adj_PointToLocal);
}

TEST_CASE("ConstructBndMesh: node2parentNode populated correctly")
{
    auto m = g_full;
    UnstructuredMesh bMesh(g_mpi, m->dim - 1);
    m->ConstructBndMesh(bMesh);

    CHECK(bMesh.node2parentNode.size() ==
          static_cast<size_t>(bMesh.NumNode() + bMesh.NumNodeGhost()));
    for (DNDS::index i = 0; i < bMesh.NumNode(); i++)
    {
        DNDS::index parentNode = bMesh.node2parentNode[i];
        CHECK(parentNode >= 0);
        CHECK(parentNode < m->NumNode() + m->NumNodeGhost());
    }
}

TEST_CASE("ConstructBndMesh: cell2parentCell maps to valid boundaries")
{
    auto m = g_full;
    UnstructuredMesh bMesh(g_mpi, m->dim - 1);
    m->ConstructBndMesh(bMesh);

    CHECK(bMesh.cell2parentCell.size() == static_cast<size_t>(bMesh.NumCell()));
    for (DNDS::index iC = 0; iC < bMesh.NumCell(); iC++)
    {
        DNDS::index parentBnd = bMesh.cell2parentCell[iC];
        CHECK(parentBnd >= 0);
        CHECK(parentBnd < m->NumBnd());
    }
}

// ===========================================================================
// BuildVTKConnectivity
// ===========================================================================

TEST_CASE("BuildVTKConnectivity: arrays populated")
{
    auto m = g_full;
    CHECK(m->vtkCell2node.size() > 0);
    CHECK(m->vtkCell2nodeOffsets.size() == static_cast<size_t>(m->NumCell() + 1));
    CHECK(m->vtkCellType.size() == static_cast<size_t>(m->NumCell()));
}

TEST_CASE("BuildVTKConnectivity: offsets monotonically increasing")
{
    auto m = g_full;
    for (size_t i = 1; i < m->vtkCell2nodeOffsets.size(); i++)
        CHECK(m->vtkCell2nodeOffsets[i] >= m->vtkCell2nodeOffsets[i - 1]);
}

TEST_CASE("BuildVTKConnectivity: cell types non-zero")
{
    for (auto ct : g_full->vtkCellType)
        CHECK(ct != 0);
}

// ===========================================================================
// Adj round-trip tests — each uses a dedicated mutable mesh via nextMut()
// ===========================================================================

TEST_CASE("AdjFacial round-trip: Local2Global then Global2Local is identity on face2node and face2cell")
{
    auto m = nextMut();
    REQUIRE(m->adjFacialState == Adj_PointToLocal);

    auto f2nSnap = snapshotAdj(m->face2node, m->NumFace());
    auto f2cSnap = snapshotAdj2(m->face2cell, m->NumFace());

    m->AdjLocal2GlobalFacial();
    CHECK(m->adjFacialState == Adj_PointToGlobal);

    // In global state, face2node entries should be global node indices
    for (DNDS::index iF = 0; iF < m->NumFace(); iF++)
        for (DNDS::rowsize j = 0; j < m->face2node.RowSize(iF); j++)
            CHECK(m->face2node(iF, j) >= 0);

    m->AdjGlobal2LocalFacial();
    CHECK(m->adjFacialState == Adj_PointToLocal);

    // face2node and face2cell must be perfectly restored
    checkAdjMatchesSnapshot(m->face2node, m->NumFace(), f2nSnap);
    for (DNDS::index iF = 0; iF < m->NumFace(); iF++)
    {
        CHECK(m->face2cell(iF, 0) == f2cSnap[iF].first);
        CHECK(m->face2cell(iF, 1) == f2cSnap[iF].second);
    }
    // NOTE: face2bnd is intentionally NOT round-tripped by AdjGlobal2LocalFacial
    // (Local2Global applies BndIndexGlobal2Local encoding but Global2Local skips it)
}

TEST_CASE("AdjC2F round-trip: Local2Global then Global2Local is identity")
{
    auto m = nextMut();
    REQUIRE(m->adjC2FState == Adj_PointToLocal);

    auto c2fSnap = snapshotAdj(m->cell2face, m->NumCell());
    auto b2fSnap = snapshotAdj1(m->bnd2face, m->NumBnd());

    m->AdjLocal2GlobalC2F();
    CHECK(m->adjC2FState == Adj_PointToGlobal);

    m->AdjGlobal2LocalC2F();
    CHECK(m->adjC2FState == Adj_PointToLocal);

    checkAdjMatchesSnapshot(m->cell2face, m->NumCell(), c2fSnap);
    for (DNDS::index ib = 0; ib < m->NumBnd(); ib++)
        CHECK(m->bnd2face(ib, 0) == b2fSnap[ib]);
}

TEST_CASE("AdjN2CB round-trip: Local2Global then Global2Local is identity")
{
    auto m = nextMut();
    REQUIRE(m->adjN2CBState == Adj_PointToLocal);

    auto n2cSnap = snapshotAdj(m->node2cell, m->NumNodeProc());
    auto n2bSnap = snapshotAdj(m->node2bnd, m->NumNodeProc());

    m->AdjLocal2GlobalN2CB();
    CHECK(m->adjN2CBState == Adj_PointToGlobal);

    m->AdjGlobal2LocalN2CB();
    CHECK(m->adjN2CBState == Adj_PointToLocal);

    checkAdjMatchesSnapshot(m->node2cell, m->NumNodeProc(), n2cSnap);
    checkAdjMatchesSnapshot(m->node2bnd, m->NumNodeProc(), n2bSnap);
}

TEST_CASE("AdjC2CFace round-trip: Local2Global then Global2Local is identity")
{
    auto m = nextMut();
    REQUIRE(m->adjC2CFaceState == Adj_PointToLocal);

    auto c2cfSnap = snapshotAdj(m->cell2cellFace, m->NumCell());

    m->AdjLocal2GlobalC2CFace();
    CHECK(m->adjC2CFaceState == Adj_PointToGlobal);

    m->AdjGlobal2LocalC2CFace();
    CHECK(m->adjC2CFaceState == Adj_PointToLocal);

    checkAdjMatchesSnapshot(m->cell2cellFace, m->NumCell(), c2cfSnap);
}

TEST_CASE("AdjPrimary round-trip: serial-out pattern (Local->Global->Local)")
{
    auto m = nextMut();
    REQUIRE(m->adjPrimaryState == Adj_PointToLocal);

    auto c2nSnap = snapshotAdj(m->cell2node, m->NumCell());
    auto c2cSnap = snapshotAdj(m->cell2cell, m->NumCell());
    auto b2nSnap = snapshotAdj(m->bnd2node, m->NumBnd());
    auto b2cSnap = snapshotAdj2(m->bnd2cell, m->NumBnd());

    // Simulate the serial-out pattern from the Euler solver
    m->AdjLocal2GlobalPrimary();
    CHECK(m->adjPrimaryState == Adj_PointToGlobal);

    // Global entries for cell2node should be in [0, globalSize)
    DNDS::index globalNodeSize = m->coords.father->globalSize();
    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        for (DNDS::rowsize j = 0; j < m->cell2node.RowSize(iC); j++)
        {
            DNDS::index gN = m->cell2node(iC, j);
            CHECK(gN >= 0);
            CHECK(gN < globalNodeSize);
        }

    m->AdjGlobal2LocalPrimary();
    CHECK(m->adjPrimaryState == Adj_PointToLocal);

    checkAdjMatchesSnapshot(m->cell2node, m->NumCell(), c2nSnap);
    checkAdjMatchesSnapshot(m->cell2cell, m->NumCell(), c2cSnap);
    checkAdjMatchesSnapshot(m->bnd2node, m->NumBnd(), b2nSnap);
    for (DNDS::index ib = 0; ib < m->NumBnd(); ib++)
    {
        CHECK(m->bnd2cell(ib, 0) == b2cSnap[ib].first);
        CHECK(m->bnd2cell(ib, 1) == b2cSnap[ib].second);
    }
}

TEST_CASE("AdjForBnd round-trip: ConstructBndMesh + ForBnd Local->Global->Local")
{
    auto m = nextMut();
    UnstructuredMesh bMesh(g_mpi, m->dim - 1);
    m->ConstructBndMesh(bMesh);
    REQUIRE(bMesh.adjPrimaryState == Adj_PointToLocal);

    auto c2nSnap = snapshotAdj(bMesh.cell2node, bMesh.NumCell());

    bMesh.AdjLocal2GlobalPrimaryForBnd();
    CHECK(bMesh.adjPrimaryState == Adj_PointToGlobal);

    // Global indices should be valid
    DNDS::index globalNodeSize = bMesh.coords.father->globalSize();
    for (DNDS::index iC = 0; iC < bMesh.NumCell(); iC++)
        for (DNDS::rowsize j = 0; j < bMesh.cell2node.RowSize(iC); j++)
        {
            DNDS::index gN = bMesh.cell2node(iC, j);
            CHECK(gN >= 0);
            CHECK(gN < globalNodeSize);
        }

    bMesh.AdjGlobal2LocalPrimaryForBnd();
    CHECK(bMesh.adjPrimaryState == Adj_PointToLocal);

    checkAdjMatchesSnapshot(bMesh.cell2node, bMesh.NumCell(), c2nSnap);
}
