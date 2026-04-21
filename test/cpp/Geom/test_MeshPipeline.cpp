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
#include "Geom/BoundaryCondition.hpp"
#include <string>
#include <vector>
#include <array>
#include <cmath>

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
    {"Ball2", "Ball2.cgns", 3, false,
     {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 958994, 16402}, // 3D mixed-element mesh (Ball2 is np=8 only)
};
static constexpr int N_CONFIGS = sizeof(g_configs) / sizeof(g_configs[0]);

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static MPIInfo g_mpi;

/// One full-pipeline mesh per config.
static ssp<UnstructuredMesh> g_full[N_CONFIGS];

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

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    for (auto &m : g_full)
        m.reset();
    MPI_Finalize();
    return res;
}

// ===========================================================================
// Parameterized tests over all mesh configs (read-only)
// ===========================================================================

#define FOR_EACH_MESH_CONFIG(body)                                        \
    for (int _ci = 0; _ci < N_CONFIGS; _ci++)                            \
    {                                                                     \
        /* Ball2 (index 4) is only tested with np=8 */                   \
        if (_ci == 4 && g_mpi.size != 8) continue;                       \
        CAPTURE(_ci);                                                     \
        const auto &cfg = g_configs[_ci];                                \
        auto m = g_full[_ci];                                             \
        SUBCASE(cfg.name) { body }                                        \
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

TEST_CASE("InterpolateFace: cell2face row sizes match expected face count")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
        {
            auto elem = m->GetCellElement(iC);
            int nFaceExpected = elem.GetNumFaces();
            CHECK(m->cell2face.RowSize(iC) == nFaceExpected);
        }
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

// --- InterpolateFace: DSL vs Legacy comparison ---

/// Build a mesh through the pipeline up to (and including) InterpolateFace,
/// using the specified face interpolation method.
static ssp<UnstructuredMesh> buildMeshUpToFace(
    const MeshConfig &cfg, bool useLegacy)
{
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

    if (useLegacy)
        mesh->InterpolateFaceLegacy();
    else
        mesh->InterpolateFace();

    return mesh;
}

/// Collect sorted face vertex sets for all local cells.
static std::vector<std::set<std::set<DNDS::index>>>
collectCellFaceVertexSets(const ssp<UnstructuredMesh> &m)
{
    DNDS::index nCells = m->cell2node.father->Size();
    std::vector<std::set<std::set<DNDS::index>>> result(nCells);
    for (DNDS::index iCell = 0; iCell < nCells; iCell++)
    {
        auto eCell = Elem::Element{m->cellElemInfo[iCell]->getElemType()};
        for (DNDS::rowsize j = 0; j < m->cell2face.RowSize(iCell); j++)
        {
            DNDS::index iFace = m->cell2face(iCell, j);
            auto eFace = eCell.ObtainFace(j);
            int nVerts = eFace.GetNumVertices();
            std::set<DNDS::index> vs;
            for (int k = 0; k < nVerts; k++)
                vs.insert(m->face2node[iFace][k]);
            result[iCell].insert(vs);
        }
    }
    return result;
}

TEST_CASE("InterpolateFace: DSL matches Legacy on all mesh configs")
{
    for (int ci = 0; ci < N_CONFIGS; ci++)
    {
        if (ci == 4 && g_mpi.size != 8)
            continue;
        CAPTURE(ci);
        const auto &cfg = g_configs[ci];
        SUBCASE(cfg.name)
        {
            auto mDSL = buildMeshUpToFace(cfg, false);
            auto mLeg = buildMeshUpToFace(cfg, true);

            // Both should have the same number of local faces
            CHECK(mDSL->NumFace() == mLeg->NumFace());
            CHECK(mDSL->NumFaceGhost() == mLeg->NumFaceGhost());

            // For each local cell, the face vertex sets must match
            auto dslSets = collectCellFaceVertexSets(mDSL);
            auto legSets = collectCellFaceVertexSets(mLeg);
            DNDS::index nCells = mDSL->cell2node.father->Size();
            CHECK(nCells == static_cast<DNDS::index>(legSets.size()));

            for (DNDS::index iCell = 0; iCell < nCells; iCell++)
            {
                CAPTURE(iCell);
                CHECK(dslSets[iCell] == legSets[iCell]);
            }

            // face2cell: for each local face, the set of (face vertex set, cell pair)
            // should match. We compare by face vertex set → cell global indices.
            // (Face ordering may differ, so we compare via vertex sets.)
            std::map<std::set<DNDS::index>, std::pair<DNDS::index, DNDS::index>> dslF2C, legF2C;
            for (DNDS::index iF = 0; iF < mDSL->NumFace(); iF++)
            {
                auto eFace = Elem::Element{mDSL->faceElemInfo(iF, 0).getElemType()};
                std::set<DNDS::index> vs;
                for (int k = 0; k < eFace.GetNumVertices(); k++)
                    vs.insert(mDSL->face2node[iF][k]);
                dslF2C[vs] = {mDSL->face2cell(iF, 0), mDSL->face2cell(iF, 1)};
            }
            for (DNDS::index iF = 0; iF < mLeg->NumFace(); iF++)
            {
                auto eFace = Elem::Element{mLeg->faceElemInfo(iF, 0).getElemType()};
                std::set<DNDS::index> vs;
                for (int k = 0; k < eFace.GetNumVertices(); k++)
                    vs.insert(mLeg->face2node[iF][k]);
                legF2C[vs] = {mLeg->face2cell(iF, 0), mLeg->face2cell(iF, 1)};
            }
            CHECK(dslF2C.size() == legF2C.size());

            for (auto &[vs, dslPair] : dslF2C)
            {
                auto it = legF2C.find(vs);
                REQUIRE(it != legF2C.end());
                // Cell pairs should be same set (order of L/R might differ)
                std::set<DNDS::index> dslCells{dslPair.first, dslPair.second};
                std::set<DNDS::index> legCells{it->second.first, it->second.second};
                CHECK(dslCells == legCells);
            }

            // Boundary zone assignment: same face vertex set should have same zone
            std::map<std::set<DNDS::index>, t_index> dslZones, legZones;
            for (DNDS::index iF = 0; iF < mDSL->NumFace(); iF++)
            {
                auto eFace = Elem::Element{mDSL->faceElemInfo(iF, 0).getElemType()};
                std::set<DNDS::index> vs;
                for (int k = 0; k < eFace.GetNumVertices(); k++)
                    vs.insert(mDSL->face2node[iF][k]);
                dslZones[vs] = mDSL->faceElemInfo(iF, 0).zone;
            }
            for (DNDS::index iF = 0; iF < mLeg->NumFace(); iF++)
            {
                auto eFace = Elem::Element{mLeg->faceElemInfo(iF, 0).getElemType()};
                std::set<DNDS::index> vs;
                for (int k = 0; k < eFace.GetNumVertices(); k++)
                    vs.insert(mLeg->face2node[iF][k]);
                legZones[vs] = mLeg->faceElemInfo(iF, 0).zone;
            }
            for (auto &[vs, dslZone] : dslZones)
            {
                auto it = legZones.find(vs);
                REQUIRE(it != legZones.end());
                CHECK(dslZone == it->second);
            }
        }
    }
}

// --- Periodic face invariants ---

using idx_pbi_t = std::pair<DNDS::index, uint8_t>;

TEST_CASE("Periodic: face pbi has uniform XOR between owner and neighbor (collaborating check)")
{
    for (int _ci = 0; _ci < N_CONFIGS; _ci++)
    {
        if (_ci == 4 && g_mpi.size != 8) continue;
        const auto &cfg = g_configs[_ci];
        if (!cfg.periodic) continue;
        auto m = g_full[_ci];
        CAPTURE(_ci);
        SUBCASE(cfg.name)
        {
            for (DNDS::index iFace = 0; iFace < m->NumFace(); iFace++)
            {
                DNDS::index iCellL = m->face2cell(iFace, 0);
                DNDS::index iCellR = m->face2cell(iFace, 1);
                if (iCellR == DNDS::UnInitIndex)
                    continue;

                auto eCellL = Elem::Element{m->cellElemInfo[iCellL]->getElemType()};
                auto eCellR = Elem::Element{m->cellElemInfo[iCellR]->getElemType()};

                int slotL = -1, slotR = -1;
                for (DNDS::rowsize j = 0; j < m->cell2face.RowSize(iCellL); j++)
                    if (m->cell2face(iCellL, j) == iFace) { slotL = j; break; }
                for (DNDS::rowsize j = 0; j < m->cell2face.RowSize(iCellR); j++)
                    if (m->cell2face(iCellR, j) == iFace) { slotR = j; break; }
                REQUIRE(slotL >= 0);
                REQUIRE(slotR >= 0);

                auto eFaceL = eCellL.ObtainFace(slotL);
                int nFN = eFaceL.GetNumNodes();

                std::vector<DNDS::index> nodesL(nFN), nodesR(nFN);
                std::vector<NodePeriodicBits> pbiL(nFN), pbiR(nFN);
                eCellL.ExtractFaceNodes(slotL, m->cell2node[iCellL], nodesL);
                eCellL.ExtractFaceNodes(slotL, m->cell2nodePbi[iCellL], pbiL);
                eCellR.ExtractFaceNodes(slotR, m->cell2node[iCellR], nodesR);
                eCellR.ExtractFaceNodes(slotR, m->cell2nodePbi[iCellR], pbiR);

                std::vector<idx_pbi_t> pairsL(nFN), pairsR(nFN);
                for (int k = 0; k < nFN; k++)
                {
                    pairsL[k] = {nodesL[k], uint8_t(pbiL[k])};
                    pairsR[k] = {nodesR[k], uint8_t(pbiR[k])};
                }
                std::sort(pairsL.begin(), pairsL.end());
                std::sort(pairsR.begin(), pairsR.end());

                for (int k = 0; k < nFN; k++)
                {
                    CAPTURE(iFace); CAPTURE(k);
                    CHECK(pairsL[k].first == pairsR[k].first);
                }

                uint8_t v0 = pairsL[0].second ^ pairsR[0].second;
                for (int k = 1; k < nFN; k++)
                {
                    uint8_t vk = pairsL[k].second ^ pairsR[k].second;
                    CAPTURE(iFace); CAPTURE(k); CAPTURE(v0); CAPTURE(vk);
                    CHECK(vk == v0);
                }
            }
        }
    }
}

TEST_CASE("Periodic: face2nodePbi is consistent with owner cell's cell2nodePbi")
{
    for (int _ci = 0; _ci < N_CONFIGS; _ci++)
    {
        if (_ci == 4 && g_mpi.size != 8) continue;
        const auto &cfg = g_configs[_ci];
        if (!cfg.periodic) continue;
        auto m = g_full[_ci];
        CAPTURE(_ci);
        SUBCASE(cfg.name)
        {
            for (DNDS::index iFace = 0; iFace < m->NumFace(); iFace++)
            {
                DNDS::index iCellOwner = m->face2cell(iFace, 0);
                auto eCell = Elem::Element{m->cellElemInfo[iCellOwner]->getElemType()};

                int iSlot = -1;
                for (DNDS::rowsize j = 0; j < m->cell2face.RowSize(iCellOwner); j++)
                    if (m->cell2face(iCellOwner, j) == iFace) { iSlot = j; break; }
                REQUIRE(iSlot >= 0);

                auto eFace = eCell.ObtainFace(iSlot);
                int nFN = eFace.GetNumNodes();

                std::vector<DNDS::index> cellFaceNodes(nFN);
                std::vector<NodePeriodicBits> cellFacePbi(nFN);
                eCell.ExtractFaceNodes(iSlot, m->cell2node[iCellOwner], cellFaceNodes);
                eCell.ExtractFaceNodes(iSlot, m->cell2nodePbi[iCellOwner], cellFacePbi);

                std::set<idx_pbi_t> fromCell, fromFace;
                for (int k = 0; k < nFN; k++)
                {
                    fromCell.insert({cellFaceNodes[k], uint8_t(cellFacePbi[k])});
                    fromFace.insert({m->face2node(iFace, k), uint8_t(m->face2nodePbi(iFace, k))});
                }
                CAPTURE(iFace);
                CHECK(fromCell == fromFace);
            }
        }
    }
}

TEST_CASE("Periodic: RecreatePeriodicNodes produces consistent counts DSL vs Legacy")
{
    for (int ci = 0; ci < N_CONFIGS; ci++)
    {
        if (ci == 4 && g_mpi.size != 8)
            continue;
        const auto &cfg = g_configs[ci];
        if (!cfg.periodic)
            continue;
        CAPTURE(ci);
        SUBCASE(cfg.name)
        {
            auto mDSL = buildMeshUpToFace(cfg, false);
            auto mLeg = buildMeshUpToFace(cfg, true);

            mDSL->AdjLocal2GlobalN2CB();
            mDSL->BuildGhostN2CB();
            mDSL->AdjGlobal2LocalN2CB();
            mDSL->RecreatePeriodicNodes();

            mLeg->AdjLocal2GlobalN2CB();
            mLeg->BuildGhostN2CB();
            mLeg->AdjGlobal2LocalN2CB();
            mLeg->RecreatePeriodicNodes();

            CHECK(mDSL->coordsPeriodicRecreated.father->Size() ==
                  mLeg->coordsPeriodicRecreated.father->Size());

            auto dslGlobal = mDSL->coordsPeriodicRecreated.father->globalSize();
            auto legGlobal = mLeg->coordsPeriodicRecreated.father->globalSize();
            CHECK(dslGlobal == legGlobal);
        }
    }
}

TEST_CASE("Periodic: face2nodePbi matches DSL vs Legacy")
{
    for (int ci = 0; ci < N_CONFIGS; ci++)
    {
        if (ci == 4 && g_mpi.size != 8)
            continue;
        const auto &cfg = g_configs[ci];
        if (!cfg.periodic)
            continue;
        CAPTURE(ci);
        SUBCASE(cfg.name)
        {
            auto mDSL = buildMeshUpToFace(cfg, false);
            auto mLeg = buildMeshUpToFace(cfg, true);

            REQUIRE(mDSL->NumFace() == mLeg->NumFace());

            // Map face vertex set → sorted (node, pbi) pairs for each path
            using FaceKey = std::set<DNDS::index>;
            using NodePbiSet = std::set<std::pair<DNDS::index, uint8_t>>;
            std::map<FaceKey, NodePbiSet> dslMap, legMap;

            for (DNDS::index iF = 0; iF < mDSL->NumFace(); iF++)
            {
                auto eF = Elem::Element{mDSL->faceElemInfo(iF, 0).getElemType()};
                FaceKey vs;
                for (int k = 0; k < eF.GetNumVertices(); k++)
                    vs.insert(mDSL->face2node(iF, k));
                NodePbiSet nps;
                for (int k = 0; k < eF.GetNumNodes(); k++)
                    nps.insert({mDSL->face2node(iF, k), uint8_t(mDSL->face2nodePbi(iF, k))});
                dslMap[vs] = nps;
            }
            for (DNDS::index iF = 0; iF < mLeg->NumFace(); iF++)
            {
                auto eF = Elem::Element{mLeg->faceElemInfo(iF, 0).getElemType()};
                FaceKey vs;
                for (int k = 0; k < eF.GetNumVertices(); k++)
                    vs.insert(mLeg->face2node(iF, k));
                NodePbiSet nps;
                for (int k = 0; k < eF.GetNumNodes(); k++)
                    nps.insert({mLeg->face2node(iF, k), uint8_t(mLeg->face2nodePbi(iF, k))});
                legMap[vs] = nps;
            }

            CHECK(dslMap.size() == legMap.size());
            for (auto &[vs, dslNps] : dslMap)
            {
                auto it = legMap.find(vs);
                REQUIRE(it != legMap.end());
                CHECK(dslNps == it->second);
            }
        }
    }
}

// --- bnd2cell DSL vs Legacy ---

/// Build mesh through RecoverCell2CellAndBnd2Cell only (not InterpolateFace).
static ssp<UnstructuredMesh> buildMeshUpToBnd2Cell(
    const MeshConfig &cfg, bool useLegacy)
{
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

    if (useLegacy)
    {
        mesh->RecoverNode2CellAndNode2BndLegacy();
        mesh->RecoverCell2CellAndBnd2CellLegacy();
    }
    else
    {
        mesh->RecoverNode2CellAndNode2Bnd();
        mesh->RecoverCell2CellAndBnd2Cell();
    }

    return mesh;
}

TEST_CASE("bnd2cell: DSL matches Legacy on all mesh configs")
{
    for (int ci = 0; ci < N_CONFIGS; ci++)
    {
        if (ci == 4 && g_mpi.size != 8)
            continue;
        const auto &cfg = g_configs[ci];
        CAPTURE(ci);
        SUBCASE(cfg.name)
        {
            auto mDSL = buildMeshUpToBnd2Cell(cfg, false);
            auto mLeg = buildMeshUpToBnd2Cell(cfg, true);

            REQUIRE(mDSL->NumBnd() == mLeg->NumBnd());

            // Compare bnd2cell: for each bnd, the cell pair (as a set) must match.
            for (DNDS::index iBnd = 0; iBnd < mDSL->NumBnd(); iBnd++)
            {
                CAPTURE(iBnd);
                std::set<DNDS::index> dslCells{mDSL->bnd2cell(iBnd, 0), mDSL->bnd2cell(iBnd, 1)};
                std::set<DNDS::index> legCells{mLeg->bnd2cell(iBnd, 0), mLeg->bnd2cell(iBnd, 1)};
                CHECK(dslCells == legCells);

                // For periodic boundaries with 2 cells, the ordering matters
                // (bnd2cell(i,0) is donor-side). Check exact match.
                if (mDSL->bnd2cell(iBnd, 1) != DNDS::UnInitIndex)
                {
                    CHECK(mDSL->bnd2cell(iBnd, 0) == mLeg->bnd2cell(iBnd, 0));
                    CHECK(mDSL->bnd2cell(iBnd, 1) == mLeg->bnd2cell(iBnd, 1));
                }
            }

            // Also compare cell2cell: for each cell, the neighbor set must match.
            REQUIRE(mDSL->NumCell() == mLeg->NumCell());
            for (DNDS::index iCell = 0; iCell < mDSL->NumCell(); iCell++)
            {
                CAPTURE(iCell);
                std::set<DNDS::index> dslNeighbors, legNeighbors;
                for (DNDS::rowsize j = 0; j < mDSL->cell2cell.father->RowSize(iCell); j++)
                    dslNeighbors.insert(mDSL->cell2cell.father->operator()(iCell, j));
                for (DNDS::rowsize j = 0; j < mLeg->cell2cell.father->RowSize(iCell); j++)
                    legNeighbors.insert(mLeg->cell2cell.father->operator()(iCell, j));
                CHECK(dslNeighbors == legNeighbors);
            }
        }
    }
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
    FOR_EACH_MESH_CONFIG({
        m->ReorderLocalCells(2);
        CHECK(m->adjPrimaryState == Adj_PointToLocal);
        CHECK(m->adjFacialState == Adj_PointToLocal);
        CHECK(m->adjC2FState == Adj_PointToLocal);
        CHECK(m->adjN2CBState == Adj_PointToLocal);
        CHECK(m->adjC2CFaceState == Adj_PointToLocal);
    })
}

TEST_CASE("ReorderLocalCells: cell count preserved")
{
    FOR_EACH_MESH_CONFIG(
        DNDS::index g1 = 0;
        DNDS::index g2 = 0;
        DNDS::index loc1 = m->NumCell();
        MPI_Allreduce(&loc1, &g1, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        m->ReorderLocalCells(2);
        DNDS::index loc2 = m->NumCell();
        MPI_Allreduce(&loc2, &g2, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        CHECK(g1 == g2);
    )
}

TEST_CASE("ReorderLocalCells: partition starts valid")
{
    FOR_EACH_MESH_CONFIG({
        m->ReorderLocalCells(2);
        int nParts = m->NLocalParts();
        CHECK(nParts == 2);
        CHECK(m->LocalPartStart(0) == 0);
        CHECK(m->LocalPartEnd(nParts - 1) == m->NumCell());
        for (int ip = 0; ip < nParts; ip++)
            CHECK(m->LocalPartStart(ip) <= m->LocalPartEnd(ip));
    })
}

TEST_CASE("ReorderLocalCells: cell2node still valid")
{
    FOR_EACH_MESH_CONFIG({
        m->ReorderLocalCells(2);
        DNDS::index totalNodes = m->NumNode() + m->NumNodeGhost();
        for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
            for (DNDS::rowsize j = 0; j < m->cell2node.RowSize(iC); j++)
            {
                DNDS::index iN = m->cell2node(iC, j);
                CHECK(iN >= 0);
                CHECK(iN < totalNodes);
            }
    })
}

TEST_CASE("ReorderLocalCells: face count preserved")
{
    FOR_EACH_MESH_CONFIG(
        DNDS::index g1 = 0;
        DNDS::index g2 = 0;
        DNDS::index loc1 = m->NumFace();
        MPI_Allreduce(&loc1, &g1, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        m->ReorderLocalCells(2);
        DNDS::index loc2 = m->NumFace();
        MPI_Allreduce(&loc2, &g2, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        CHECK(g1 == g2);
    )
}

// ===========================================================================
// Adj round-trip tests (all mesh configs)
// ===========================================================================

TEST_CASE("AdjFacial round-trip: face2node and face2cell identity")
{
    FOR_EACH_MESH_CONFIG(
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
    )
}

TEST_CASE("AdjC2F round-trip: cell2face and bnd2face identity")
{
    FOR_EACH_MESH_CONFIG(
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
    )
}

TEST_CASE("AdjN2CB round-trip: node2cell and node2bnd identity")
{
    FOR_EACH_MESH_CONFIG(
        REQUIRE(m->adjN2CBState == Adj_PointToLocal);

        auto n2cSnap = snapshotAdj(m->node2cell, m->NumNodeProc());
        auto n2bSnap = snapshotAdj(m->node2bnd, m->NumNodeProc());

        m->AdjLocal2GlobalN2CB();
        CHECK(m->adjN2CBState == Adj_PointToGlobal);

        m->AdjGlobal2LocalN2CB();
        CHECK(m->adjN2CBState == Adj_PointToLocal);

        checkAdjMatchesSnapshot(m->node2cell, m->NumNodeProc(), n2cSnap);
        checkAdjMatchesSnapshot(m->node2bnd, m->NumNodeProc(), n2bSnap);
    )
}

TEST_CASE("AdjC2CFace round-trip: cell2cellFace identity")
{
    FOR_EACH_MESH_CONFIG(
        REQUIRE(m->adjC2CFaceState == Adj_PointToLocal);

        auto c2cfSnap = snapshotAdj(m->cell2cellFace, m->NumCell());

        m->AdjLocal2GlobalC2CFace();
        CHECK(m->adjC2CFaceState == Adj_PointToGlobal);

        m->AdjGlobal2LocalC2CFace();
        CHECK(m->adjC2CFaceState == Adj_PointToLocal);

        checkAdjMatchesSnapshot(m->cell2cellFace, m->NumCell(), c2cfSnap);
    )
}

TEST_CASE("AdjPrimary round-trip: serial-out pattern")
{
    FOR_EACH_MESH_CONFIG(
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
    )
}

TEST_CASE("AdjForBnd round-trip: ConstructBndMesh + ForBnd")
{
    FOR_EACH_MESH_CONFIG(
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
    )
}

// ===========================================================================
// Ball2: 3D mixed-element mesh (np=8 only)
// ===========================================================================

TEST_CASE("Ball2: mixed 3D element types")
{
    // Ball2 (config 4) is only tested with np=8
    if (g_mpi.size != 8)
        return;

    auto m = g_full[4]; // Ball2 is index 4
    DNDS_assert(m != nullptr);

    // Count element types locally
    std::map<Elem::ElemType, int> typeCounts;
    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
    {
        auto elem = m->GetCellElement(iC);
        typeCounts[elem.type]++;
    }

    // Gather all unique element types from all ranks
    std::vector<int> localTypes;
    for (const auto& [type, count] : typeCounts)
        localTypes.push_back(static_cast<int>(type));

    int nLocalTypes = localTypes.size();
    std::vector<int> allRankSizes(g_mpi.size);
    MPI_Gather(&nLocalTypes, 1, MPI_INT, allRankSizes.data(), 1, MPI_INT, 0, g_mpi.comm);

    std::vector<int> allTypes;
    std::vector<int> displs;
    int totalTypes = 0;

    if (g_mpi.rank == 0)
    {
        displs.resize(g_mpi.size);
        for (int i = 0; i < g_mpi.size; i++)
        {
            displs[i] = totalTypes;
            totalTypes += allRankSizes[i];
        }
        allTypes.resize(totalTypes);
    }

    MPI_Gatherv(localTypes.data(), nLocalTypes, MPI_INT,
                allTypes.data(), allRankSizes.data(), displs.data(), MPI_INT, 0, g_mpi.comm);

    // Build set of unique types (on rank 0)
    std::set<Elem::ElemType> uniqueTypes;
    if (g_mpi.rank == 0)
    {
        for (int typeInt : allTypes)
            uniqueTypes.insert(static_cast<Elem::ElemType>(typeInt));
    }

    // Broadcast unique types to all ranks
    int nUnique = uniqueTypes.size();
    MPI_Bcast(&nUnique, 1, MPI_INT, 0, g_mpi.comm);
    std::vector<int> uniqueTypesVec;
    if (g_mpi.rank == 0)
    {
        for (auto type : uniqueTypes)
            uniqueTypesVec.push_back(static_cast<int>(type));
    }
    uniqueTypesVec.resize(nUnique);
    MPI_Bcast(uniqueTypesVec.data(), nUnique, MPI_INT, 0, g_mpi.comm);

    // All ranks iterate over the same types in the same order
    std::map<Elem::ElemType, int> globalCounts;
    for (int typeInt : uniqueTypesVec)
    {
        Elem::ElemType type = static_cast<Elem::ElemType>(typeInt);
        int localCount = typeCounts.count(type) ? typeCounts[type] : 0;
        int globalCount;
        MPI_Reduce(&localCount, &globalCount, 1, MPI_INT, MPI_SUM, 0, g_mpi.comm);
        if (g_mpi.rank == 0)
            globalCounts[type] = globalCount;
    }

    if (g_mpi.rank == 0)
    {
        std::cout << "\nBall2 element type counts:" << std::endl;
        for (const auto& [type, count] : globalCounts)
        {
            std::cout << "  Type " << static_cast<int>(type) << ": " << count << std::endl;
        }

        // Verify expected total
        int total = 0;
        for (const auto& [type, count] : globalCounts)
            total += count;
        CHECK(total == 958994);
    }
}

// ===========================================================================
// Elevation/Bisection tests
// ===========================================================================

TEST_CASE("Elevation: O2 mesh cell count equals O1 cell count")
{
    // Only test with config 0 (UniformSquare_10) for now
    const auto& cfg = g_configs[0];

    // Build fresh O1 mesh
    auto mO1 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    UnstructuredMeshSerialRW reader(mO1, 0);

    if (cfg.periodic)
    {
        tPoint zero{0, 0, 0};
        mO1->SetPeriodicGeometry(cfg.translation1, zero, zero, cfg.translation2, zero, zero, cfg.translation3, zero, zero);
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

    mO1->RecoverNode2CellAndNode2Bnd();
    mO1->RecoverCell2CellAndBnd2Cell();
    mO1->BuildGhostPrimary();
    mO1->AdjGlobal2LocalPrimary();

    // Build O2 mesh via elevation
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);

    // Set up MPI state for mO2
    mO2->RecoverNode2CellAndNode2Bnd();
    mO2->RecoverCell2CellAndBnd2Cell();
    mO2->BuildGhostPrimary();
    mO2->AdjGlobal2LocalPrimary();

    DNDS::index nCellO1 = mO1->NumCellGlobal();
    DNDS::index nCellO2 = mO2->NumCellGlobal();

    // Elevation preserves cell count
    CHECK(nCellO2 == nCellO1);

    // All cells should be O2 type
    for (DNDS::index iC = 0; iC < mO2->NumCell(); iC++)
    {
        auto elem = mO2->GetCellElement(iC);
        CHECK(elem.GetOrder() == 2);
    }
}

TEST_CASE("Elevation: O2 mesh has more nodes than O1")
{
    const auto& cfg = g_configs[0];

    // Build fresh O1 mesh
    auto mO1 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    UnstructuredMeshSerialRW reader(mO1, 0);

    if (cfg.periodic)
    {
        tPoint zero{0, 0, 0};
        mO1->SetPeriodicGeometry(cfg.translation1, zero, zero, cfg.translation2, zero, zero, cfg.translation3, zero, zero);
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

    mO1->RecoverNode2CellAndNode2Bnd();
    mO1->RecoverCell2CellAndBnd2Cell();
    mO1->BuildGhostPrimary();
    mO1->AdjGlobal2LocalPrimary();

    // Build O2 mesh via elevation
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);

    // Set up MPI state for mO2
    mO2->RecoverNode2CellAndNode2Bnd();
    mO2->RecoverCell2CellAndBnd2Cell();
    mO2->BuildGhostPrimary();
    mO2->AdjGlobal2LocalPrimary();

    DNDS::index nNodeO1 = mO1->coords.father->globalSize();
    DNDS::index nNodeO2 = mO2->coords.father->globalSize();

    // Elevation adds nodes
    CHECK(nNodeO2 > nNodeO1);
}

TEST_CASE("Bisection: O1 mesh has more cells than O2")
{
    const auto& cfg = g_configs[0];

    // Build fresh O1 mesh
    auto mO1 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    UnstructuredMeshSerialRW reader(mO1, 0);

    if (cfg.periodic)
    {
        tPoint zero{0, 0, 0};
        mO1->SetPeriodicGeometry(cfg.translation1, zero, zero, cfg.translation2, zero, zero, cfg.translation3, zero, zero);
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

    mO1->RecoverNode2CellAndNode2Bnd();
    mO1->RecoverCell2CellAndBnd2Cell();
    mO1->BuildGhostPrimary();
    mO1->AdjGlobal2LocalPrimary();

    // Build O2 mesh
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);
    mO2->RecoverNode2CellAndNode2Bnd();
    mO2->RecoverCell2CellAndBnd2Cell();
    mO2->BuildGhostPrimary();
    mO2->AdjGlobal2LocalPrimary();

    // Build O1 mesh via bisection
    mO2->AdjLocal2GlobalPrimary();
    auto mO1B = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO1B->BuildBisectO1FormO2(*mO2);

    DNDS::index nCellO2 = mO2->NumCellGlobal();
    DNDS::index nCellO1B = mO1B->NumCellGlobal();

    // Bisection increases cell count
    CHECK(nCellO1B > nCellO2);

    // All cells should be O1 (linear) element types
    for (DNDS::index iC = 0; iC < mO1B->NumCell(); iC++)
    {
        auto elem = mO1B->GetCellElement(iC);
        CHECK(elem.GetOrder() == 1);
    }
}

TEST_CASE("Bisection: global node count is preserved from O2 mesh")
{
    const auto& cfg = g_configs[0];

    // Build fresh O1 mesh
    auto mO1 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    UnstructuredMeshSerialRW reader(mO1, 0);

    if (cfg.periodic)
    {
        tPoint zero{0, 0, 0};
        mO1->SetPeriodicGeometry(cfg.translation1, zero, zero, cfg.translation2, zero, zero, cfg.translation3, zero, zero);
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

    mO1->RecoverNode2CellAndNode2Bnd();
    mO1->RecoverCell2CellAndBnd2Cell();
    mO1->BuildGhostPrimary();
    mO1->AdjGlobal2LocalPrimary();

    // Build O2 mesh
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);
    mO2->RecoverNode2CellAndNode2Bnd();
    mO2->RecoverCell2CellAndBnd2Cell();
    mO2->BuildGhostPrimary();
    mO2->AdjGlobal2LocalPrimary();

    // Build O1 mesh via bisection
    mO2->AdjLocal2GlobalPrimary();
    auto mO1B = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO1B->BuildBisectO1FormO2(*mO2);

    // No new nodes created during bisection
    DNDS::index nNodeO2 = mO2->coords.father->globalSize();
    DNDS::index nNodeO1B = mO1B->coords.father->globalSize();
    CHECK(nNodeO1B == nNodeO2);
}

TEST_CASE("Elevation+Bisection: exact cell count progression")
{
    // UniformSquare_10: 100 Quad4 cells
    // After elevation: 100 Quad9 cells (same count, higher order)
    // After bisection: 400 Quad4 cells (4 sub-cells per Quad9)
    const auto& cfg = g_configs[0];

    // Build fresh O1 mesh
    auto mO1 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    UnstructuredMeshSerialRW reader(mO1, 0);

    if (cfg.periodic)
    {
        tPoint zero{0, 0, 0};
        mO1->SetPeriodicGeometry(cfg.translation1, zero, zero, cfg.translation2, zero, zero, cfg.translation3, zero, zero);
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

    mO1->RecoverNode2CellAndNode2Bnd();
    mO1->RecoverCell2CellAndBnd2Cell();
    mO1->BuildGhostPrimary();
    mO1->AdjGlobal2LocalPrimary();

    // Build O2 mesh
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);
    mO2->RecoverNode2CellAndNode2Bnd();
    mO2->RecoverCell2CellAndBnd2Cell();
    mO2->BuildGhostPrimary();
    mO2->AdjGlobal2LocalPrimary();

    // Build O1 mesh via bisection
    mO2->AdjLocal2GlobalPrimary();
    auto mO1B = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO1B->BuildBisectO1FormO2(*mO2);

    DNDS::index nCellOrig = mO1->NumCellGlobal();
    DNDS::index nCellElevated = mO2->NumCellGlobal();
    DNDS::index nCellBisected = mO1B->NumCellGlobal();

    CHECK(nCellOrig == 100);
    CHECK(nCellElevated == 100);
    CHECK(nCellBisected == 400);
}

// ===========================================================================
// Elevation + Boundary/Internal Smooth tests (NACA0012_H2, config 2)
// ===========================================================================

/**
 * \brief Build an O2 mesh with full face interpolation, ready for smoothing.
 *
 * Sequence: read O1 -> partition -> ghost -> elevate -> rebuild topology
 * -> face interpolation -> N2CB ghost.
 */
static ssp<UnstructuredMesh> buildO2MeshWithFaces(
    const MeshConfig &cfg, ssp<UnstructuredMesh> &mO1Out)
{
    // Build O1 mesh
    auto mO1 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    UnstructuredMeshSerialRW reader(mO1, 0);

    if (cfg.periodic)
    {
        tPoint zero{0, 0, 0};
        mO1->SetPeriodicGeometry(
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

    mO1->RecoverNode2CellAndNode2Bnd();
    mO1->RecoverCell2CellAndBnd2Cell();
    mO1->BuildGhostPrimary();
    mO1->AdjGlobal2LocalPrimary();

    // Build O2 mesh via elevation
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);

    // Rebuild full topology for O2
    mO2->RecoverNode2CellAndNode2Bnd();
    mO2->RecoverCell2CellAndBnd2Cell();
    mO2->BuildGhostPrimary();
    mO2->AdjGlobal2LocalPrimary();
    mO2->AdjGlobal2LocalN2CB();

    mO2->InterpolateFace();
    mO2->AssertOnFaces();

    mO2->AdjLocal2GlobalN2CB();
    mO2->BuildGhostN2CB();
    mO2->AdjGlobal2LocalN2CB();

    mO2->BuildCell2CellFace();
    mO2->AdjGlobal2LocalC2CFace();

    mO1Out = mO1;
    return mO2;
}

TEST_CASE("BoundarySmooth: NACA0012 moves boundary O2 nodes")
{
    // NACA0012_H2 is config 2 — non-periodic, curved airfoil boundary
    const auto &cfg = g_configs[2];

    ssp<UnstructuredMesh> mO1;
    auto mO2 = buildO2MeshWithFaces(cfg, mO1);

    // Snapshot pre-smooth coordinates
    std::vector<tPoint> preSmooth(mO2->NumNode());
    for (DNDS::index i = 0; i < mO2->NumNode(); i++)
        preSmooth[i] = mO2->coords[i];

    // Run boundary smooth — all external BCs are smoothed
    mO2->ElevatedNodesGetBoundarySmooth(
        [](Geom::t_index bndId)
        { return Geom::FaceIDIsExternalBC(bndId); });

    // Check that nTotalMoved > 0 (some O2 boundary nodes moved)
    CHECK(mO2->nTotalMoved > 0);

    // Check that coordsElevDisp was initialized
    CHECK(mO2->coordsElevDisp.father != nullptr);

    // All coordinates must be finite (no NaN/Inf)
    for (DNDS::index i = 0; i < mO2->NumNode(); i++)
    {
        for (int d = 0; d < 3; d++)
            CHECK(std::isfinite(mO2->coords[i](d)));
    }
}

TEST_CASE("BoundarySmooth: no NaN in displacement field")
{
    const auto &cfg = g_configs[2]; // NACA0012_H2

    ssp<UnstructuredMesh> mO1;
    auto mO2 = buildO2MeshWithFaces(cfg, mO1);

    mO2->ElevatedNodesGetBoundarySmooth(
        [](Geom::t_index bndId)
        { return Geom::FaceIDIsExternalBC(bndId); });

    // coordsElevDisp should have no NaN for moved nodes
    DNDS::index nMovedLocal = 0;
    for (DNDS::index i = 0; i < mO2->coordsElevDisp.father->Size(); i++)
    {
        auto disp = mO2->coordsElevDisp[i];
        // Entries with disp(0) != largeReal were set by the smooth
        if (disp(0) != UnInitReal && std::abs(disp(0)) < 1e30)
        {
            for (int d = 0; d < 3; d++)
                CHECK(std::isfinite(disp(d)));
            nMovedLocal++;
        }
    }

    DNDS::index nMovedGlobal = 0;
    MPI_Allreduce(&nMovedLocal, &nMovedGlobal, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    CHECK(nMovedGlobal > 0);
}

TEST_CASE("InternalSmooth V2: coordinates remain finite after solve")
{
    const auto &cfg = g_configs[2]; // NACA0012_H2

    ssp<UnstructuredMesh> mO1;
    auto mO2 = buildO2MeshWithFaces(cfg, mO1);

    // Set up elevation info for fast convergence (small problem)
    mO2->elevationInfo.nIter = 10;
    mO2->elevationInfo.nSearch = 20;
    mO2->elevationInfo.RBFRadius = 5.0;

    // Run boundary smooth
    mO2->ElevatedNodesGetBoundarySmooth(
        [](Geom::t_index bndId)
        { return Geom::FaceIDIsExternalBC(bndId); });

    if (mO2->nTotalMoved == 0)
        return; // nothing to smooth

    // Save pre-smooth coords
    std::vector<tPoint> preSmoothCoords(mO2->NumNode());
    for (DNDS::index i = 0; i < mO2->NumNode(); i++)
        preSmoothCoords[i] = mO2->coords[i];

    // Run internal smooth V2
    mO2->ElevatedNodesSolveInternalSmoothV2();

    // All coordinates must be finite
    for (DNDS::index i = 0; i < mO2->NumNode(); i++)
        for (int d = 0; d < 3; d++)
            CHECK(std::isfinite(mO2->coords[i](d)));

    // At least some coordinates should have changed
    DNDS::index nChangedLocal = 0;
    for (DNDS::index i = 0; i < mO2->NumNode(); i++)
        if ((mO2->coords[i] - preSmoothCoords[i]).norm() > 1e-15)
            nChangedLocal++;

    DNDS::index nChangedGlobal = 0;
    MPI_Allreduce(&nChangedLocal, &nChangedGlobal, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    CHECK(nChangedGlobal > 0);
}

TEST_CASE("Elevation: O2 coordinates have no NaN")
{
    // Test on all non-periodic configs (0, 2)
    for (int ci : {0, 2})
    {
        const auto &cfg = g_configs[ci];
        SUBCASE(cfg.name)
        {
            auto mO1 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
            UnstructuredMeshSerialRW reader(mO1, 0);

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

            mO1->RecoverNode2CellAndNode2Bnd();
            mO1->RecoverCell2CellAndBnd2Cell();
            mO1->BuildGhostPrimary();
            mO1->AdjGlobal2LocalPrimary();

            auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
            mO2->BuildO2FromO1Elevation(*mO1);

            // All O2 coordinates must be finite
            for (DNDS::index i = 0; i < mO2->coords.father->Size(); i++)
                for (int d = 0; d < 3; d++)
                    CHECK(std::isfinite(mO2->coords[i](d)));
        }
    }
}
