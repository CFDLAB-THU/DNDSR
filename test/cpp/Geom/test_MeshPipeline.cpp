/**
 * @file test_MeshPipeline.cpp
 * @brief Parameterized doctest tests for the full mesh pipeline.
 *
 * Each mesh configuration (file, dimension, periodic settings, expected counts)
 * is tested through the complete pipeline:
 *   - Pipeline state tracking: every state variable checked
 *   - InterpolateFace: face topology, face2cell, bnd2face
 *   - ReorderLocalCells: cell permutation, partition starts
 *   - N2CB, C2F, Facial, C2CFace adjacency validation
 *   - ConstructBndMesh, BuildVTKConnectivity
 *   - Global count conservation
 *   - All 5 Adj round-trip inverses (on one mesh config)
 *
 * Mesh configurations:
 *   [0] UniformSquare_10  -- 2D, 100 quad cells, non-periodic
 *   [1] IV10_10           -- 2D, 100 quad cells, periodic (isentropic vortex)
 *   [2] NACA0012_H2       -- 2D, 20816 unstructured quad cells, non-periodic (airfoil)
 *   [3] IV10U_10          -- 2D, 322 unstructured tri cells, periodic (isentropic vortex)
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "Geom/Mesh.hpp"
#include <string>
#include <vector>
#include <array>

using namespace DNDS;
using namespace DNDS::Geom;

// NOTE: DNDS::index, DNDS::real, DNDS::rowsize clash with POSIX symbols.

// ---------------------------------------------------------------------------
// Mesh configuration
// ---------------------------------------------------------------------------
struct MeshConfig
{
    const char *name;         ///< Human-readable label
    const char *file;         ///< Filename relative to data/mesh/
    int dim;                  ///< Spatial dimension
    bool periodic;            ///< Has periodic boundaries
    tPoint translation1;      ///< Periodic translation axis 1
    tPoint translation2;      ///< Periodic translation axis 2
    tPoint translation3;      ///< Periodic translation axis 3
    DNDS::index expectedCells; ///< Expected global cell count (-1 = skip check)
    DNDS::index expectedBnds;  ///< Expected global bnd count (-1 = skip check)
};

static const MeshConfig g_configs[] = {
    {"UniformSquare_10", "UniformSquare_10.cgns", 2, false,
     {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 100, 40},
    {"IV10_10", "IV10_10.cgns", 2, true,
     {10, 0, 0}, {0, 10, 0}, {0, 0, 10}, 100, -1},
    {"NACA0012_H2", "NACA0012_H2.cgns", 2, false,
     {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 20816, 484},
    {"IV10U_10", "IV10U_10.cgns", 2, true,
     {10, 0, 0}, {0, 10, 0}, {0, 0, 10}, 322, -1},
};
static constexpr int N_CONFIGS = sizeof(g_configs) / sizeof(g_configs[0]);

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static MPIInfo g_mpi;

/// One full-pipeline mesh per config.
static ssp<UnstructuredMesh> g_full[N_CONFIGS];

/// One reordered mesh (only for config 0).
static ssp<UnstructuredMesh> g_reord;

/// Mutable meshes for round-trip tests (built from config 0 only).
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

static ssp<UnstructuredMesh> buildFullMesh(
    int meshId, const MeshConfig &cfg, int nReorderParts = 1)
{
    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << " " << cfg.name << "] START" << std::endl;

    auto mesh = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    UnstructuredMeshSerialRW reader(mesh, 0);

    if (cfg.periodic)
    {
        tPoint zero{0, 0, 0};
        mesh->SetPeriodicGeometry(
            cfg.translation1, zero, zero,
            cfg.translation2, zero, zero,
            cfg.translation3, zero, zero);
    }

    reader.ReadFromCGNSSerial(meshPath(cfg.file));
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

    if (nReorderParts > 1)
        mesh->ReorderLocalCells(nReorderParts);

    mesh->InterpolateFace();
    mesh->AssertOnFaces();

    mesh->AdjLocal2GlobalN2CB();
    mesh->BuildGhostN2CB();
    mesh->AdjGlobal2LocalN2CB();
    mesh->AssertOnN2CB();

    mesh->BuildCell2CellFace();
    mesh->AdjGlobal2LocalC2CFace();

    if (cfg.periodic)
        mesh->RecreatePeriodicNodes();
    mesh->BuildVTKConnectivity();

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << " " << cfg.name << "] DONE" << std::endl;

    return mesh;
}

static ssp<UnstructuredMesh> nextMut()
{
    DNDS_assert(g_mutNext < N_MUT);
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

static std::vector<std::pair<DNDS::index, DNDS::index>> snapshotAdj2(
    const tAdj2Pair &adj, DNDS::index nRows)
{
    std::vector<std::pair<DNDS::index, DNDS::index>> out(nRows);
    for (DNDS::index i = 0; i < nRows; i++)
        out[i] = {adj(i, 0), adj(i, 1)};
    return out;
}

static std::vector<DNDS::index> snapshotAdj1(
    const tAdj1Pair &adj, DNDS::index nRows)
{
    std::vector<DNDS::index> out(nRows);
    for (DNDS::index i = 0; i < nRows; i++)
        out[i] = adj(i, 0);
    return out;
}

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

    // Build one full-pipeline mesh per config.
    for (int ic = 0; ic < N_CONFIGS; ic++)
        g_full[ic] = buildFullMesh(ic, g_configs[ic]);

    // Build reordered mesh from config 0.
    g_reord = buildFullMesh(N_CONFIGS, g_configs[0], 2);

    // Build mutable meshes from config 0 for round-trip tests.
    for (int i = 0; i < N_MUT; i++)
        g_mut[i] = buildFullMesh(N_CONFIGS + 1 + i, g_configs[0]);

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    for (auto &m : g_full)
        m.reset();
    g_reord.reset();
    for (auto &m : g_mut)
        m.reset();
    MPI_Finalize();
    return res;
}

// ===========================================================================
// Parameterized tests over all mesh configs (read-only)
// ===========================================================================

#define FOR_EACH_MESH_CONFIG(body)                    \
    for (int _ci = 0; _ci < N_CONFIGS; _ci++)        \
    {                                                 \
        CAPTURE(_ci);                                 \
        const auto &cfg = g_configs[_ci];             \
        auto m = g_full[_ci];                         \
        SUBCASE(cfg.name) { body }                    \
    }

// --- Pipeline state ---

TEST_CASE("Pipeline: all five states are Local")
{
    FOR_EACH_MESH_CONFIG({
        CHECK(m->adjPrimaryState == Adj_PointToLocal);
        CHECK(m->adjFacialState == Adj_PointToLocal);
        CHECK(m->adjC2FState == Adj_PointToLocal);
        CHECK(m->adjN2CBState == Adj_PointToLocal);
        CHECK(m->adjC2CFaceState == Adj_PointToLocal);
    })
}

// --- Global count conservation ---

TEST_CASE("Pipeline: global cell count matches expected")
{
    FOR_EACH_MESH_CONFIG({
        DNDS::index localCells = m->NumCell();
        DNDS::index globalCells = 0;
        MPI_Allreduce(&localCells, &globalCells, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        CHECK(globalCells > 0);
        if (cfg.expectedCells > 0)
            CHECK(globalCells == cfg.expectedCells);
    })
}

TEST_CASE("Pipeline: global bnd count matches expected")
{
    FOR_EACH_MESH_CONFIG({
        DNDS::index localBnds = m->NumBnd();
        DNDS::index globalBnds = 0;
        MPI_Allreduce(&localBnds, &globalBnds, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        if (cfg.expectedBnds > 0)
            CHECK(globalBnds == cfg.expectedBnds);
        else if (!cfg.periodic)
            CHECK(globalBnds > 0); // non-periodic meshes must have boundaries
    })
}

// --- InterpolateFace topology ---

TEST_CASE("InterpolateFace: face count is positive")
{
    FOR_EACH_MESH_CONFIG({
        CHECK(m->NumFace() > 0);
    })
}

TEST_CASE("InterpolateFace: face2cell has exactly 2 entries per face")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iF = 0; iF < m->NumFace(); iF++)
            CHECK(m->face2cell.RowSize(iF) == 2);
    })
}

TEST_CASE("InterpolateFace: face2cell owner is a valid local cell")
{
    FOR_EACH_MESH_CONFIG({
        DNDS::index totalCells = m->NumCell() + m->NumCellGhost();
        for (DNDS::index iF = 0; iF < m->NumFace(); iF++)
        {
            DNDS::index owner = m->face2cell(iF, 0);
            CHECK(owner >= 0);
            CHECK(owner < totalCells);
        }
    })
}

TEST_CASE("InterpolateFace: face2node indices in valid range")
{
    FOR_EACH_MESH_CONFIG({
        DNDS::index totalNodes = m->NumNode() + m->NumNodeGhost();
        for (DNDS::index iF = 0; iF < m->NumFace(); iF++)
            for (DNDS::rowsize j = 0; j < m->face2node.RowSize(iF); j++)
            {
                DNDS::index iN = m->face2node(iF, j);
                CHECK(iN >= 0);
                CHECK(iN < totalNodes);
            }
    })
}

TEST_CASE("InterpolateFace: cell2face row sizes match cell2node")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
            CHECK(m->cell2face.RowSize(iC) == m->cell2node.RowSize(iC));
    })
}

TEST_CASE("InterpolateFace: cell2face entries are valid face indices")
{
    FOR_EACH_MESH_CONFIG({
        DNDS::index totalFaces = m->NumFace() + m->NumFaceGhost();
        for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
            for (DNDS::rowsize j = 0; j < m->cell2face.RowSize(iC); j++)
            {
                DNDS::index iF = m->cell2face(iC, j);
                CHECK(iF >= 0);
                CHECK(iF < totalFaces);
            }
    })
}

TEST_CASE("InterpolateFace: bnd2face maps to valid faces")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index ib = 0; ib < m->NumBnd(); ib++)
        {
            DNDS::index iFace = m->bnd2face(ib, 0);
            // Periodic boundaries that became internal faces may have bnd2face == -1
            if (iFace >= 0)
                CHECK(iFace < m->NumFace() + m->NumFaceGhost());
        }
    })
}

TEST_CASE("InterpolateFace: face element types are valid")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iF = 0; iF < m->NumFace(); iF++)
            CHECK(m->GetFaceElement(iF).type != Elem::ElemType::UnknownElem);
    })
}

// --- N2CB ---

TEST_CASE("N2CB: every local node has at least one adjacent cell")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iNode = 0; iNode < m->NumNode(); iNode++)
        {
            bool hasCell = false;
            for (DNDS::rowsize j = 0; j < m->node2cell.RowSize(iNode); j++)
                if (m->node2cell(iNode, j) >= 0)
                    hasCell = true;
            CHECK(hasCell);
        }
    })
}

TEST_CASE("N2CB: node2cell entries are valid local indices")
{
    FOR_EACH_MESH_CONFIG({
        DNDS::index totalCells = m->NumCell() + m->NumCellGhost();
        for (DNDS::index iNode = 0; iNode < m->NumNodeProc(); iNode++)
            for (DNDS::rowsize j = 0; j < m->node2cell.RowSize(iNode); j++)
            {
                DNDS::index iC = m->node2cell(iNode, j);
                if (iC >= 0)
                    CHECK(iC < totalCells);
            }
    })
}

// --- BuildCell2CellFace ---

TEST_CASE("BuildCell2CellFace: row sizes match cell2face")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
            CHECK(m->cell2cellFace.RowSize(iC) == m->cell2face.RowSize(iC));
    })
}

TEST_CASE("BuildCell2CellFace: entries are valid cells or negative")
{
    FOR_EACH_MESH_CONFIG({
        DNDS::index totalCells = m->NumCell() + m->NumCellGhost();
        for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
            for (DNDS::rowsize j = 0; j < m->cell2cellFace.RowSize(iC); j++)
            {
                DNDS::index nb = m->cell2cellFace(iC, j);
                if (nb >= 0)
                    CHECK(nb < totalCells);
            }
    })
}

// --- ConstructBndMesh ---

TEST_CASE("ConstructBndMesh: correct dimensions and state")
{
    FOR_EACH_MESH_CONFIG({
        if (cfg.periodic)
            continue; // periodic meshes may have no real boundaries after deduplication
        UnstructuredMesh bMesh(g_mpi, m->dim - 1);
        m->ConstructBndMesh(bMesh);
        CHECK(bMesh.dim == m->dim - 1);
        CHECK(bMesh.NumCell() == m->NumBnd());
        CHECK(bMesh.adjPrimaryState == Adj_PointToLocal);
    })
}

TEST_CASE("ConstructBndMesh: node2parentNode valid")
{
    FOR_EACH_MESH_CONFIG({
        if (cfg.periodic)
            continue;
        UnstructuredMesh bMesh(g_mpi, m->dim - 1);
        m->ConstructBndMesh(bMesh);
        CHECK(bMesh.node2parentNode.size() ==
              static_cast<size_t>(bMesh.NumNode() + bMesh.NumNodeGhost()));
        for (DNDS::index i = 0; i < bMesh.NumNode(); i++)
        {
            DNDS::index pn = bMesh.node2parentNode[i];
            CHECK(pn >= 0);
            CHECK(pn < m->NumNode() + m->NumNodeGhost());
        }
    })
}

// --- BuildVTKConnectivity ---

TEST_CASE("BuildVTKConnectivity: arrays populated correctly")
{
    FOR_EACH_MESH_CONFIG({
        CHECK(m->vtkCell2node.size() > 0);
        CHECK(m->vtkCell2nodeOffsets.size() == static_cast<size_t>(m->NumCell() + 1));
        CHECK(m->vtkCellType.size() == static_cast<size_t>(m->NumCell()));
        for (size_t i = 1; i < m->vtkCell2nodeOffsets.size(); i++)
            CHECK(m->vtkCell2nodeOffsets[i] >= m->vtkCell2nodeOffsets[i - 1]);
        for (auto ct : m->vtkCellType)
            CHECK(ct != 0);
    })
}

// ===========================================================================
// ReorderLocalCells (config 0 only)
// ===========================================================================

TEST_CASE("ReorderLocalCells: all states preserved")
{
    auto m = g_reord;
    CHECK(m->adjPrimaryState == Adj_PointToLocal);
    CHECK(m->adjFacialState == Adj_PointToLocal);
    CHECK(m->adjC2FState == Adj_PointToLocal);
    CHECK(m->adjN2CBState == Adj_PointToLocal);
    CHECK(m->adjC2CFaceState == Adj_PointToLocal);
}

TEST_CASE("ReorderLocalCells: cell count preserved")
{
    DNDS::index g1 = 0, g2 = 0;
    {
        DNDS::index loc = g_full[0]->NumCell();
        MPI_Allreduce(&loc, &g1, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    }
    {
        DNDS::index loc = g_reord->NumCell();
        MPI_Allreduce(&loc, &g2, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    }
    CHECK(g1 == g2);
}

TEST_CASE("ReorderLocalCells: partition starts valid")
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

TEST_CASE("ReorderLocalCells: face count preserved")
{
    DNDS::index g1 = 0, g2 = 0;
    {
        DNDS::index loc = g_full[0]->NumFace();
        MPI_Allreduce(&loc, &g1, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    }
    {
        DNDS::index loc = g_reord->NumFace();
        MPI_Allreduce(&loc, &g2, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    }
    CHECK(g1 == g2);
}

// ===========================================================================
// Adj round-trip tests (config 0 mutable meshes)
// ===========================================================================

TEST_CASE("AdjFacial round-trip: face2node and face2cell identity")
{
    auto m = nextMut();
    REQUIRE(m->adjFacialState == Adj_PointToLocal);

    auto f2nSnap = snapshotAdj(m->face2node, m->NumFace());
    auto f2cSnap = snapshotAdj2(m->face2cell, m->NumFace());

    m->AdjLocal2GlobalFacial();
    CHECK(m->adjFacialState == Adj_PointToGlobal);

    for (DNDS::index iF = 0; iF < m->NumFace(); iF++)
        for (DNDS::rowsize j = 0; j < m->face2node.RowSize(iF); j++)
            CHECK(m->face2node(iF, j) >= 0);

    m->AdjGlobal2LocalFacial();
    CHECK(m->adjFacialState == Adj_PointToLocal);

    checkAdjMatchesSnapshot(m->face2node, m->NumFace(), f2nSnap);
    for (DNDS::index iF = 0; iF < m->NumFace(); iF++)
    {
        CHECK(m->face2cell(iF, 0) == f2cSnap[iF].first);
        CHECK(m->face2cell(iF, 1) == f2cSnap[iF].second);
    }
    // NOTE: face2bnd intentionally not round-tripped (asymmetric by design)
}

TEST_CASE("AdjC2F round-trip: cell2face and bnd2face identity")
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

TEST_CASE("AdjN2CB round-trip: node2cell and node2bnd identity")
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

TEST_CASE("AdjC2CFace round-trip: cell2cellFace identity")
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

TEST_CASE("AdjPrimary round-trip: serial-out pattern")
{
    auto m = nextMut();
    REQUIRE(m->adjPrimaryState == Adj_PointToLocal);

    auto c2nSnap = snapshotAdj(m->cell2node, m->NumCell());
    auto c2cSnap = snapshotAdj(m->cell2cell, m->NumCell());
    auto b2nSnap = snapshotAdj(m->bnd2node, m->NumBnd());
    auto b2cSnap = snapshotAdj2(m->bnd2cell, m->NumBnd());

    m->AdjLocal2GlobalPrimary();
    CHECK(m->adjPrimaryState == Adj_PointToGlobal);

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

TEST_CASE("AdjForBnd round-trip: ConstructBndMesh + ForBnd")
{
    auto m = nextMut();
    UnstructuredMesh bMesh(g_mpi, m->dim - 1);
    m->ConstructBndMesh(bMesh);
    REQUIRE(bMesh.adjPrimaryState == Adj_PointToLocal);

    auto c2nSnap = snapshotAdj(bMesh.cell2node, bMesh.NumCell());

    bMesh.AdjLocal2GlobalPrimaryForBnd();
    CHECK(bMesh.adjPrimaryState == Adj_PointToGlobal);

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
