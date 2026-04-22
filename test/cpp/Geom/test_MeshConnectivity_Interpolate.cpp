/**
 * @file test_MeshConnectivity_Interpolate.cpp
 * @brief Unit tests for MeshConnectivity Interpolate (local) and InterpolateGlobal (distributed).
 *
 * Tests cover face and edge extraction from 2D/3D meshes, periodic dedup with
 * pbi collaborating check, and distributed interpolation with ownership resolution.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "SyntheticMeshBuilders.hpp"
#include "Geom/MeshConnectivity.hpp"
#include "Geom/Mesh.hpp"

using namespace DNDS;
using namespace DNDS::Geom;

static MPIInfo g_mpi;

// ===========================================================================
// Local Interpolate tests
// ===========================================================================

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

    // Internal edges: count entities with 2+ parents (variable-width entity2parent)
    int nInternal = 0;
    for (DNDS::index i = 0; i < res.nEntities; i++)
        if (res.entity2parent.father->RowSize(i) >= 2)
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
            bool found = false;
            for (DNDS::rowsize p = 0; p < res.entity2parent.father->RowSize(iEnt); p++)
                if (res.entity2parent.father->operator()(iEnt, p) == iCell)
                    found = true;
            CHECK(found);
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
        if (res.entity2parent.father->RowSize(i) >= 2)
            nInternal++;
    CHECK(nInternal == 1);

    // The shared edge should have vertex set {1, 2}
    for (DNDS::index i = 0; i < res.nEntities; i++)
    {
        if (res.entity2parent.father->RowSize(i) >= 2)
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
        CHECK(res.entity2parent.father->RowSize(i) < 2);

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
        if (res.entity2parent.father->RowSize(i) >= 2)
            nInternal++;
    CHECK(nInternal == 1);

    // Shared face is {1,2,3}
    for (DNDS::index i = 0; i < res.nEntities; i++)
    {
        if (res.entity2parent.father->RowSize(i) >= 2)
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
        CHECK(res.entity2parent.father->RowSize(i) < 2);
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
        if (res.entity2parent.father->RowSize(i) >= 2)
            nShared++;
    CHECK(nShared == 3);

    // Shared edges: {1,2}, {1,3}, {2,3}
    std::set<std::set<DNDS::index>> expectedShared = {{1, 2}, {1, 3}, {2, 3}};
    std::set<std::set<DNDS::index>> actualShared;
    for (DNDS::index i = 0; i < res.nEntities; i++)
    {
        if (res.entity2parent.father->RowSize(i) >= 2)
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
        if (res.entity2parent.father->RowSize(i) >= 2)
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

// ===========================================================================
// Periodic Interpolate tests
// ===========================================================================

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
        // With variable-width entity2parent, the extra parents are appended
        // instead of overflowing. Check that at least one entity has >2 parents.
        {
            bool hasMultiParent = false;
            for (DNDS::index i = 0; i < resNoPbi.nEntities; i++)
                if (resNoPbi.entity2parent.father->RowSize(i) > 2)
                    hasMultiParent = true;
            CHECK(hasMultiParent == true);
        }
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
        if (resPbi.entity2parent.father->RowSize(i) >= 2)
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
            CAPTURE(iCell); CAPTURE(j); CAPTURE(iEnt);
            bool found = false;
            for (DNDS::rowsize p = 0; p < resPbi.entity2parent.father->RowSize(iEnt); p++)
                if (resPbi.entity2parent.father->operator()(iEnt, p) == iCell)
                    found = true;
            CHECK(found);
        }
    }

    // Verify no face connects a cell to itself
    for (DNDS::index i = 0; i < resPbi.nEntities; i++)
    {
        CAPTURE(i);
        for (DNDS::rowsize p = 0; p < resPbi.entity2parent.father->RowSize(i); p++)
            for (DNDS::rowsize q = p + 1; q < resPbi.entity2parent.father->RowSize(i); q++)
                CHECK(resPbi.entity2parent.father->operator()(i, p) !=
                      resPbi.entity2parent.father->operator()(i, q));
    }

    // Verify each cell-pair shares exactly the right number of faces:
    // Cell 0-1: 2 faces, Cell 0-2: 2 faces, Cell 1-3: 2 faces, Cell 2-3: 2 faces
    // Cell 0-3: 0 faces (diagonal, no shared face), Cell 1-2: 0 faces (diagonal)
    std::map<std::set<DNDS::index>, int> pairCount;
    for (DNDS::index i = 0; i < resPbi.nEntities; i++)
    {
        REQUIRE(resPbi.entity2parent.father->RowSize(i) >= 2);
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

TEST_CASE("Periodic 2x2x2: face interpolation (3D, collaborating check)")
{
    if (g_mpi.rank != 0 && g_mpi.size > 1)
        return;

    auto pm = makePeriodic2x2x2Mesh(g_mpi);

    // Without collaborating check: expect multi-parent entities from false merges
    {
        auto q = makeFaceQuery(pm.cellElemInfo);
        auto res = MeshConnectivity::Interpolate(pm.cell2node, q, 8, 8, g_mpi);
        // With variable-width entity2parent, extra parents are appended.
        // Check that at least one entity has >2 parents.
        bool hasMultiParent = false;
        for (DNDS::index i = 0; i < res.nEntities; i++)
            if (res.entity2parent.father->RowSize(i) > 2)
                hasMultiParent = true;
        CHECK(hasMultiParent == true);
    }

    // With collaborating check: expect 24 faces, all internal
    {
        auto q = makeFaceQuery(pm.cellElemInfo);
        q.matchExtra = makePeriodicMatchExtra(pm.cellElemInfo, pm.cell2node, pm.cell2nodePbi, false);
        auto res = MeshConnectivity::Interpolate(pm.cell2node, q, 8, 8, g_mpi);

        CHECK(res.nEntities == 24);

        // All faces internal (both parents set)
        int nInternal = 0;
        for (DNDS::index i = 0; i < res.nEntities; i++)
            if (res.entity2parent.father->RowSize(i) >= 2)
                nInternal++;
        CHECK(nInternal == 24);

        // No entity should have >2 parents (collaborating check prevents false merges)
        for (DNDS::index i = 0; i < res.nEntities; i++)
            CHECK(res.entity2parent.father->RowSize(i) <= 2);

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
            REQUIRE(res.entity2parent.father->RowSize(i) >= 2);
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
            REQUIRE(res.entity2parent.father->RowSize(i) >= 2);
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

    // Without collaborating check: expect multi-parent entities from false merges
    {
        auto q = makeEdgeQuery(pm.cellElemInfo);
        auto res = MeshConnectivity::Interpolate(pm.cell2node, q, 8, 8, g_mpi);
        // With variable-width entity2parent, extra parents are appended.
        // Check that at least one entity has >2 parents (false merges without pbi check).
        bool hasMultiParent = false;
        for (DNDS::index i = 0; i < res.nEntities; i++)
            if (res.entity2parent.father->RowSize(i) > 2)
                hasMultiParent = true;
        CHECK(hasMultiParent == true);
    }

    // With collaborating check: expect 24 edges, all shared by 4 cells
    {
        auto q = makeEdgeQuery(pm.cellElemInfo);
        q.matchExtra = makePeriodicMatchExtra(pm.cellElemInfo, pm.cell2node, pm.cell2nodePbi, true);
        auto res = MeshConnectivity::Interpolate(pm.cell2node, q, 8, 8, g_mpi);

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

        // With variable-width entity2parent, edges shared by 4 cells now store
        // all 4 parents. Verify via entity2parent RowSize.
        for (DNDS::index i = 0; i < res.nEntities; i++)
        {
            CAPTURE(i);
            CHECK(res.entity2parent.father->RowSize(i) == 4);
        }

        // Also verify via parent2entity reference counting.
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

// ===========================================================================
// InterpolateGlobal distributed tests
// ===========================================================================

TEST_CASE("InterpolateGlobal: 4x4x4 distributed hex faces (non-periodic)")
{
    DistributedHex3D mesh;
    mesh.build(4, false, g_mpi);
    DNDS::index N = 4;
    DNDS::index np = g_mpi.size;
    DNDS::index nCellAll = mesh.cell2node.Size();
    DNDS::index nLocalCells = mesh.nCellLocal;

    auto faceQuery = makeHex8FaceQueryPbi(mesh.cellElemInfo);

    // Ownership: min parent rank wins.
    OwnershipResolverMulti resolver =
        [&](const std::vector<DNDS::index> &parents,
            const std::vector<DNDS::MPI_int> &parentRanks,
            DNDS::index nLocal) -> OwnershipDecision
    {
        DNDS::MPI_int minRank = *std::min_element(parentRanks.begin(), parentRanks.end());
        bool anyLocal = false;
        for (auto p : parents)
            if (p < nLocal)
                anyLocal = true;
        if (!anyLocal)
            return {false, {}};
        if (minRank != g_mpi.rank)
            return {false, {}};
        std::vector<DNDS::MPI_int> peers;
        for (size_t i = 0; i < parents.size(); i++)
            if (parents[i] >= nLocal && parentRanks[i] != g_mpi.rank)
                peers.push_back(parentRanks[i]);
        std::sort(peers.begin(), peers.end());
        peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
        return {true, std::move(peers)};
    };

    auto result = MeshConnectivity::InterpolateGlobal(
        mesh.cell2node, tPbiPair{},
        *mesh.cellGhostMapping, *mesh.cellGM,
        *mesh.nodeGhostMapping,
        faceQuery, nLocalCells, nCellAll, mesh.nNodeTotal,
        resolver, g_mpi);

    // Check global face count: sum of owned faces across all ranks.
    DNDS::index localOwned = result.nOwnedEntities;
    DNDS::index globalOwned = 0;
    MPI_Allreduce(&localOwned, &globalOwned, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);

    // Expected: 3 * (np*N) * N * N + 3 * N * N = ... let me compute:
    // X-normal faces: (np*N + 1) * N * N
    // Y-normal faces: np * N * (N + 1) * N
    // Z-normal faces: np * N * N * (N + 1)
    DNDS::index expectedFaces = (np * N + 1) * N * N +
                                np * N * (N + 1) * N +
                                np * N * N * (N + 1);
    CHECK(globalOwned == expectedFaces);

    // Verify entity2node: all face nodes are Quad4 (4 nodes each).
    for (DNDS::index i = 0; i < result.nOwnedEntities; i++)
    {
        CHECK(result.entity2node.father->RowSize(i) == 4);
        CHECK(result.entityElemInfo.father->operator()(i, 0).getElemType() == Elem::Quad4);
    }

    // Verify entity2parent: boundary faces have 1 parent, internal have 2.
    DNDS::index nBndFaces = 0, nIntFaces = 0;
    for (DNDS::index i = 0; i < result.nOwnedEntities; i++)
    {
        auto pRow = result.entity2parent.father->operator[](i);
        if (pRow.size() == 1)
            nBndFaces++;
        else if (pRow.size() == 2)
            nIntFaces++;
        else
            FAIL("face has " << pRow.size() << " parents");
    }

    DNDS::index globalBnd = 0, globalInt = 0;
    MPI_Allreduce(&nBndFaces, &globalBnd, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    MPI_Allreduce(&nIntFaces, &globalInt, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    // Boundary faces: 2 * N * N (X-boundaries) + 2 * np*N * N (Y-boundaries) + 2 * np*N * N (Z-boundaries)
    DNDS::index expectedBnd = 2 * N * N + 2 * np * N * N + 2 * np * N * N;
    CHECK(globalBnd == expectedBnd);
    CHECK(globalInt == expectedFaces - expectedBnd);

    // Verify parent2entity: each local cell should have 6 faces.
    for (DNDS::index iCell = 0; iCell < nLocalCells; iCell++)
    {
        CAPTURE(iCell);
        CHECK(result.parent2entity.father->RowSize(iCell) == 6);
        // All entries should be valid global face IDs (not UnInitIndex).
        for (rowsize j = 0; j < 6; j++)
        {
            DNDS::index gFace = result.parent2entity.father->operator()(iCell, j);
            CAPTURE(j);
            CHECK(gFace != UnInitIndex);
            CHECK(gFace >= 0);
            CHECK(gFace < globalOwned);
        }
    }
}

TEST_CASE("InterpolateGlobal: 4x4x4 distributed hex faces (X-periodic)")
{
    if (g_mpi.size < 2)
        return; // Periodic X needs at least 2 ranks to be meaningful.

    DistributedHex3D mesh;
    mesh.build(4, true, g_mpi);
    DNDS::index N = 4;
    DNDS::index np = g_mpi.size;
    DNDS::index nCellAll = mesh.cell2node.Size();
    DNDS::index nLocalCells = mesh.nCellLocal;

    auto faceQuery = makeHex8FaceQueryPbi(mesh.cellElemInfo);
    // No pbi for now (X-periodic without pbi = just wrapping cell connectivity).
    // For a proper periodic test we'd need cell2nodePbi, but the key test is
    // that face dedup works across the periodic boundary.

    OwnershipResolverMulti resolver =
        [&](const std::vector<DNDS::index> &parents,
            const std::vector<DNDS::MPI_int> &parentRanks,
            DNDS::index nLocal) -> OwnershipDecision
    {
        DNDS::MPI_int minRank = *std::min_element(parentRanks.begin(), parentRanks.end());
        bool anyLocal = false;
        for (auto p : parents)
            if (p < nLocal)
                anyLocal = true;
        if (!anyLocal)
            return {false, {}};
        if (minRank != g_mpi.rank)
            return {false, {}};
        std::vector<DNDS::MPI_int> peers;
        for (size_t i = 0; i < parents.size(); i++)
            if (parents[i] >= nLocal && parentRanks[i] != g_mpi.rank)
                peers.push_back(parentRanks[i]);
        std::sort(peers.begin(), peers.end());
        peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
        return {true, std::move(peers)};
    };

    auto result = MeshConnectivity::InterpolateGlobal(
        mesh.cell2node, tPbiPair{},
        *mesh.cellGhostMapping, *mesh.cellGM,
        *mesh.nodeGhostMapping,
        faceQuery, nLocalCells, nCellAll, mesh.nNodeTotal,
        resolver, g_mpi);

    DNDS::index localOwned = result.nOwnedEntities;
    DNDS::index globalOwned = 0;
    MPI_Allreduce(&localOwned, &globalOwned, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);

    // X-periodic: no X-boundary faces, all X-faces are internal.
    // X-normal faces: np*N * N * N (no +1 boundary)
    // Y-normal faces: np*N * (N+1) * N
    // Z-normal faces: np*N * N * (N+1)
    // Y and Z boundaries still exist.
    DNDS::index expectedFaces = np * N * N * N +
                                np * N * (N + 1) * N +
                                np * N * N * (N + 1);
    CHECK(globalOwned == expectedFaces);

    // All faces should have exactly 2 parents (no boundary in X, Y/Z still have boundary).
    // Actually Y and Z boundaries have 1 parent.
    DNDS::index nBndFaces = 0;
    for (DNDS::index i = 0; i < result.nOwnedEntities; i++)
    {
        auto pRow = result.entity2parent.father->operator[](i);
        if (pRow.size() == 1)
            nBndFaces++;
    }
    DNDS::index globalBnd = 0;
    MPI_Allreduce(&nBndFaces, &globalBnd, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    // Only Y and Z boundaries remain: 2 * np*N * N (Y) + 2 * np*N * N (Z)
    DNDS::index expectedBnd = 2 * np * N * N + 2 * np * N * N;
    CHECK(globalBnd == expectedBnd);

    // Each local cell still has 6 faces.
    for (DNDS::index iCell = 0; iCell < nLocalCells; iCell++)
    {
        CAPTURE(iCell);
        CHECK(result.parent2entity.father->RowSize(iCell) == 6);
        for (rowsize j = 0; j < 6; j++)
        {
            DNDS::index gFace = result.parent2entity.father->operator()(iCell, j);
            CHECK(gFace != UnInitIndex);
        }
    }
}

TEST_CASE("InterpolateGlobal: 4x4x4 distributed hex edges (non-periodic)")
{
    DistributedHex3D mesh;
    mesh.build(4, false, g_mpi);
    DNDS::index N = 4;
    DNDS::index np = g_mpi.size;
    DNDS::index nCellAll = mesh.cell2node.Size();
    DNDS::index nLocalCells = mesh.nCellLocal;

    auto edgeQuery = makeHex8EdgeQueryPbi(mesh.cellElemInfo);

    OwnershipResolverMulti resolver =
        [&](const std::vector<DNDS::index> &parents,
            const std::vector<DNDS::MPI_int> &parentRanks,
            DNDS::index nLocal) -> OwnershipDecision
    {
        DNDS::MPI_int minRank = *std::min_element(parentRanks.begin(), parentRanks.end());
        bool anyLocal = false;
        for (auto p : parents)
            if (p < nLocal)
                anyLocal = true;
        if (!anyLocal)
            return {false, {}};
        if (minRank != g_mpi.rank)
            return {false, {}};
        std::vector<DNDS::MPI_int> peers;
        for (size_t i = 0; i < parents.size(); i++)
            if (parents[i] >= nLocal && parentRanks[i] != g_mpi.rank)
                peers.push_back(parentRanks[i]);
        std::sort(peers.begin(), peers.end());
        peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
        return {true, std::move(peers)};
    };

    auto result = MeshConnectivity::InterpolateGlobal(
        mesh.cell2node, tPbiPair{},
        *mesh.cellGhostMapping, *mesh.cellGM,
        *mesh.nodeGhostMapping,
        edgeQuery, nLocalCells, nCellAll, mesh.nNodeTotal,
        resolver, g_mpi);

    // Check global edge count.
    DNDS::index localOwned = result.nOwnedEntities;
    DNDS::index globalOwned = 0;
    MPI_Allreduce(&localOwned, &globalOwned, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);

    // Expected edges for non-periodic NxNxN stacked np times in X:
    // X-aligned: np*N * (N+1) * (N+1)
    // Y-aligned: (np*N+1) * N * (N+1)
    // Z-aligned: (np*N+1) * (N+1) * N
    DNDS::index expectedEdges = np * N * (N + 1) * (N + 1) +
                                (np * N + 1) * N * (N + 1) +
                                (np * N + 1) * (N + 1) * N;
    CHECK(globalOwned == expectedEdges);

    // All edges should be Line2 (2 nodes each).
    for (DNDS::index i = 0; i < result.nOwnedEntities; i++)
    {
        CHECK(result.entity2node.father->RowSize(i) == 2);
        CHECK(result.entityElemInfo.father->operator()(i, 0).getElemType() == Elem::Line2);
    }

    // Edge parent counts: internal edges have 4 parents, face-boundary 2, edge-boundary 1.
    // Just verify all edges have >= 1 parent and <= 4 parents.
    for (DNDS::index i = 0; i < result.nOwnedEntities; i++)
    {
        auto pRow = result.entity2parent.father->operator[](i);
        CAPTURE(i);
        CHECK(pRow.size() >= 1);
        CHECK(pRow.size() <= 4);
    }

    // Each local cell should have 12 edges.
    for (DNDS::index iCell = 0; iCell < nLocalCells; iCell++)
    {
        CAPTURE(iCell);
        CHECK(result.parent2entity.father->RowSize(iCell) == 12);
        for (rowsize j = 0; j < 12; j++)
        {
            DNDS::index gEdge = result.parent2entity.father->operator()(iCell, j);
            CAPTURE(j);
            CHECK(gEdge != UnInitIndex);
            CHECK(gEdge >= 0);
            CHECK(gEdge < globalOwned);
        }
    }
}

TEST_CASE("InterpolateGlobal: 4x4x4 distributed hex edges (X-periodic)")
{
    if (g_mpi.size < 2)
        return;

    DistributedHex3D mesh;
    mesh.build(4, true, g_mpi);
    DNDS::index N = 4;
    DNDS::index np = g_mpi.size;
    DNDS::index nCellAll = mesh.cell2node.Size();
    DNDS::index nLocalCells = mesh.nCellLocal;

    auto edgeQuery = makeHex8EdgeQueryPbi(mesh.cellElemInfo);

    OwnershipResolverMulti resolver =
        [&](const std::vector<DNDS::index> &parents,
            const std::vector<DNDS::MPI_int> &parentRanks,
            DNDS::index nLocal) -> OwnershipDecision
    {
        DNDS::MPI_int minRank = *std::min_element(parentRanks.begin(), parentRanks.end());
        bool anyLocal = false;
        for (auto p : parents)
            if (p < nLocal)
                anyLocal = true;
        if (!anyLocal)
            return {false, {}};
        if (minRank != g_mpi.rank)
            return {false, {}};
        std::vector<DNDS::MPI_int> peers;
        for (size_t i = 0; i < parents.size(); i++)
            if (parents[i] >= nLocal && parentRanks[i] != g_mpi.rank)
                peers.push_back(parentRanks[i]);
        std::sort(peers.begin(), peers.end());
        peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
        return {true, std::move(peers)};
    };

    auto result = MeshConnectivity::InterpolateGlobal(
        mesh.cell2node, tPbiPair{},
        *mesh.cellGhostMapping, *mesh.cellGM,
        *mesh.nodeGhostMapping,
        edgeQuery, nLocalCells, nCellAll, mesh.nNodeTotal,
        resolver, g_mpi);

    DNDS::index localOwned = result.nOwnedEntities;
    DNDS::index globalOwned = 0;
    MPI_Allreduce(&localOwned, &globalOwned, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);

    // X-periodic: X-aligned edges wrap, no X-boundary.
    // X-aligned: np*N * (N+1) * (N+1)  (same as non-periodic, but no +1 boundary -- wait)
    // Actually for non-periodic, X-aligned edges span from node X=0 to X=np*N,
    // so there are np*N edge segments per Y-Z grid line, times (N+1)^2 grid lines.
    // For X-periodic, same count: np*N edges per Y-Z grid line.
    // Y-aligned: np*N * N * (N+1) (no +1 in X dimension)
    // Z-aligned: np*N * (N+1) * N (no +1 in X dimension)
    DNDS::index expectedEdges = np * N * (N + 1) * (N + 1) +
                                np * N * N * (N + 1) +
                                np * N * (N + 1) * N;
    CHECK(globalOwned == expectedEdges);

    // All edges should be Line2.
    for (DNDS::index i = 0; i < result.nOwnedEntities; i++)
        CHECK(result.entity2node.father->RowSize(i) == 2);

    // X-periodic removes X-boundary edges. Internal edges in the bulk have 4 parents.
    // Y/Z boundary edges still have 1 or 2 parents.
    for (DNDS::index i = 0; i < result.nOwnedEntities; i++)
    {
        auto pRow = result.entity2parent.father->operator[](i);
        CHECK(pRow.size() >= 1);
        CHECK(pRow.size() <= 4);
    }

    // Each local cell has 12 edges.
    for (DNDS::index iCell = 0; iCell < nLocalCells; iCell++)
    {
        CAPTURE(iCell);
        CHECK(result.parent2entity.father->RowSize(iCell) == 12);
        for (rowsize j = 0; j < 12; j++)
        {
            DNDS::index gEdge = result.parent2entity.father->operator()(iCell, j);
            CHECK(gEdge != UnInitIndex);
        }
    }
}

TEST_CASE("InterpolateGlobal: 4x4x4 distributed hex faces (triply-periodic)")
{
    if (g_mpi.size < 2)
        return;

    // Triply-periodic: X wraps across ranks, Y and Z wrap within each rank.
    // After periodic dedup: N^3 unique nodes per rank.
    // Node at (ix, iy, iz) on rank p: local index = ix*N*N + iy*N + iz.
    // Global index = rank_offset + local_index.
    // Cell (ix, iy, iz) has nodes at (ix+di, iy+dj, iz+dk) for di,dj,dk in {0,1},
    // with wrapping: Y mod N, Z mod N, X mod (np*N) across ranks.
    // Pbi: P1 set if X wraps, P2 if Y wraps, P3 if Z wraps.

    const DNDS::index N = 4;
    const DNDS::index np = g_mpi.size;
    const DNDS::index nCellLocal = N * N * N;
    const DNDS::index nNodeLocal = N * N * N; // triply-periodic: N^3 unique nodes

    auto cellGM = std::make_shared<GlobalOffsetsMapping>();
    cellGM->setMPIAlignBcast(g_mpi, nCellLocal);
    auto nodeGM = std::make_shared<GlobalOffsetsMapping>();
    nodeGM->setMPIAlignBcast(g_mpi, nNodeLocal);

    DNDS::index cellOff = (*cellGM)(g_mpi.rank, 0);
    DNDS::index nodeOff = (*nodeGM)(g_mpi.rank, 0);

    // Helper: global node index for physical (gx, gy, gz) with triply-periodic wrapping.
    auto nodeGlobal = [&](DNDS::index gx, DNDS::index gy, DNDS::index gz) -> DNDS::index
    {
        DNDS::index totalNx = np * N;
        gx = ((gx % totalNx) + totalNx) % totalNx;
        gy = ((gy % N) + N) % N;
        gz = ((gz % N) + N) % N;
        DNDS::MPI_int ownerRank = static_cast<DNDS::MPI_int>(gx / N);
        DNDS::index localIx = gx - ownerRank * N;
        return (*nodeGM)(ownerRank, 0) + localIx * N * N + gy * N + gz;
    };

    // Helper: pbi for a cell-to-node reference at (gx+di, gy+dj, gz+dk).
    auto nodePbi = [&](DNDS::index gx, DNDS::index gy, DNDS::index gz,
                       int di, int dj, int dk) -> NodePeriodicBits
    {
        NodePeriodicBits pbi{};
        DNDS::index totalNx = np * N;
        if ((gx + di) >= totalNx || (gx + di) < 0)
            pbi.setP1True();
        if ((gy + dj) >= N || (gy + dj) < 0)
            pbi.setP2True();
        if ((gz + dk) >= N || (gz + dk) < 0)
            pbi.setP3True();
        return pbi;
    };

    // Build cell2node with pbi.
    tAdjPair cell2node;
    cell2node.InitPair("c2n", g_mpi);
    cell2node.father->Resize(nCellLocal, 8);
    tPbiPair cell2nodePbi;
    cell2nodePbi.InitPair("c2nPbi", g_mpi);
    cell2nodePbi.father->Resize(nCellLocal, 8);
    tElemInfoArrayPair cellElemInfo;
    cellElemInfo.InitPair("cInfo", g_mpi);
    cellElemInfo.father->Resize(nCellLocal);

    // Hex8 node order: (0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)
    const int di8[8] = {0, 1, 1, 0, 0, 1, 1, 0};
    const int dj8[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    const int dk8[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    for (DNDS::index ix = 0; ix < N; ix++)
        for (DNDS::index iy = 0; iy < N; iy++)
            for (DNDS::index iz = 0; iz < N; iz++)
            {
                DNDS::index iCell = ix * N * N + iy * N + iz;
                DNDS::index gx = g_mpi.rank * N + ix;
                for (int k = 0; k < 8; k++)
                {
                    cell2node.father->operator()(iCell, k) =
                        nodeGlobal(gx + di8[k], iy + dj8[k], iz + dk8[k]);
                    cell2nodePbi.father->operator()(iCell, k) =
                        nodePbi(gx, iy, iz, di8[k], dj8[k], dk8[k]);
                }
                cellElemInfo.father->operator()(iCell, 0) = ElemInfo{
                    static_cast<t_index>(Elem::Hex8), INTERNAL_ZONE};
            }

    // Ghost cells: 1-layer from left and right neighbor ranks in X.
    DNDS::MPI_int leftRank = (g_mpi.rank - 1 + g_mpi.size) % g_mpi.size;
    DNDS::MPI_int rightRank = (g_mpi.rank + 1) % g_mpi.size;

    struct GhostCell { DNDS::index global; std::array<DNDS::index, 8> nodes; std::array<NodePeriodicBits, 8> pbi; };
    std::vector<GhostCell> ghostCells;

    auto addGhostLayer = [&](DNDS::MPI_int srcRank, DNDS::index srcIx, DNDS::index physX)
    {
        for (DNDS::index iy = 0; iy < N; iy++)
            for (DNDS::index iz = 0; iz < N; iz++)
            {
                GhostCell gc;
                gc.global = (*cellGM)(srcRank, 0) + srcIx * N * N + iy * N + iz;
                for (int k = 0; k < 8; k++)
                {
                    gc.nodes[k] = nodeGlobal(physX + di8[k], iy + dj8[k], iz + dk8[k]);
                    gc.pbi[k] = nodePbi(physX, iy, iz, di8[k], dj8[k], dk8[k]);
                }
                ghostCells.push_back(gc);
            }
    };
    addGhostLayer(leftRank, N - 1, g_mpi.rank * N - 1);
    addGhostLayer(rightRank, 0, (g_mpi.rank + 1) * N);

    // Sort ghost cells by global index.
    std::sort(ghostCells.begin(), ghostCells.end(),
              [](const GhostCell &a, const GhostCell &b) { return a.global < b.global; });

    // Collect all node globals.
    std::set<DNDS::index> allNodeGlobals;
    for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
        for (int k = 0; k < 8; k++)
            allNodeGlobals.insert(cell2node.father->operator()(iCell, k));
    for (auto &gc : ghostCells)
        for (auto ng : gc.nodes)
            allNodeGlobals.insert(ng);

    // Ghost node mapping.
    std::vector<DNDS::index> ghostNodeGlobals;
    for (auto ng : allNodeGlobals)
    {
        DNDS::MPI_int r; DNDS::index v;
        if (nodeGM->search(ng, r, v) && r != g_mpi.rank)
            ghostNodeGlobals.push_back(ng);
    }
    tAdjPair dummyNode;
    dummyNode.InitPair("dn", g_mpi);
    dummyNode.father->Resize(nNodeLocal);
    dummyNode.TransAttach();
    dummyNode.trans.createFatherGlobalMapping();
    dummyNode.trans.createGhostMapping(ghostNodeGlobals);
    dummyNode.trans.createMPITypes();
    auto nodeGhostMapping = dummyNode.trans.pLGhostMapping;
    DNDS::index nNodeTotal = nNodeLocal + static_cast<DNDS::index>(ghostNodeGlobals.size());

    // Build ghost cell son.
    std::vector<DNDS::index> ghostGlobals;
    for (auto &gc : ghostCells)
        ghostGlobals.push_back(gc.global);

    cell2node.son = std::make_shared<tAdj::element_type>(ObjName{"c2n.son"}, g_mpi);
    cell2node.son->Resize(static_cast<DNDS::index>(ghostCells.size()), 8);
    cell2nodePbi.son = std::make_shared<decltype(cell2nodePbi.son)::element_type>(
        ObjName{"c2nPbi.son"}, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), g_mpi);
    cell2nodePbi.son->Resize(static_cast<DNDS::index>(ghostCells.size()), 8);
    cellElemInfo.son = std::make_shared<tElemInfoArray::element_type>(ObjName{"cInfo.son"}, g_mpi);
    cellElemInfo.son->Resize(static_cast<DNDS::index>(ghostCells.size()));

    cell2node.TransAttach();
    cell2node.trans.createFatherGlobalMapping();
    cell2node.trans.createGhostMapping(ghostGlobals);
    cell2node.trans.createMPITypes();
    cell2nodePbi.TransAttach();
    cell2nodePbi.trans.BorrowGGIndexing(cell2node.trans);
    cell2nodePbi.trans.createMPITypes();
    cellElemInfo.TransAttach();
    cellElemInfo.trans.BorrowGGIndexing(cell2node.trans);
    cellElemInfo.trans.createMPITypes();

    // Populate son AFTER createMPITypes.
    for (DNDS::index ig = 0; ig < static_cast<DNDS::index>(ghostCells.size()); ig++)
    {
        for (int k = 0; k < 8; k++)
        {
            cell2node.son->operator()(ig, k) = ghostCells[ig].nodes[k];
            cell2nodePbi.son->operator()(ig, k) = ghostCells[ig].pbi[k];
        }
        cellElemInfo.son->operator()(ig, 0) = ElemInfo{
            static_cast<t_index>(Elem::Hex8), INTERNAL_ZONE};
    }

    auto cellGhostMapping = cell2node.trans.pLGhostMapping;

    // Convert cell2node from global to local-appended node indices.
    DNDS::index nCellAll = cell2node.Size();
    for (DNDS::index iCell = 0; iCell < nCellAll; iCell++)
        for (rowsize k = 0; k < 8; k++)
        {
            DNDS::index &ng = cell2node(iCell, k);
            auto [found, r, la] = nodeGhostMapping->search_indexAppend(ng);
            DNDS_assert(found);
            ng = la;
        }

    // Build face query with pbi collaborating check.
    auto faceQuery = makeHex8FaceQueryPbi(cellElemInfo);
    faceQuery.matchExtra = [&](DNDS::index iParent, int iSub,
                               DNDS::index, DNDS::index candidateParent, int candidateSub) -> bool
    {
        auto eA = Elem::Element{cellElemInfo[iParent]->getElemType()};
        auto eB = Elem::Element{cellElemInfo[candidateParent]->getElemType()};
        auto fA = eA.ObtainFace(iSub);
        int nFN = fA.GetNumNodes();
        std::vector<DNDS::index> nodesA(nFN), nodesB(nFN);
        eA.ExtractFaceNodes(iSub, cell2node[iParent], nodesA);
        eB.ExtractFaceNodes(candidateSub, cell2node[candidateParent], nodesB);
        std::vector<NodePeriodicBits> pbiA(nFN), pbiB(nFN);
        eA.ExtractFaceNodes(iSub, cell2nodePbi[iParent], pbiA);
        eB.ExtractFaceNodes(candidateSub, cell2nodePbi[candidateParent], pbiB);
        using P = std::pair<DNDS::index, uint8_t>;
        auto cmp = [](const P &l, const P &r)
        { return l.first == r.first ? l.second < r.second : l.first < r.first; };
        std::vector<P> pa(nFN), pb(nFN);
        for (int i = 0; i < nFN; i++) { pa[i] = {nodesA[i], uint8_t(pbiA[i])}; pb[i] = {nodesB[i], uint8_t(pbiB[i])}; }
        std::sort(pa.begin(), pa.end(), cmp);
        std::sort(pb.begin(), pb.end(), cmp);
        uint8_t v0 = pa[0].second ^ pb[0].second;
        for (int i = 1; i < nFN; i++)
            if ((pa[i].second ^ pb[i].second) != v0)
                return false;
        return true;
    };
    faceQuery.extractPbi = [&](DNDS::index iP, int iSub,
                                const std::function<NodePeriodicBits(int)> &pPbi,
                                NodePeriodicBits *out)
    {
        auto e = Elem::Element{cellElemInfo[iP]->getElemType()};
        auto f = e.ObtainFace(iSub);
        std::vector<NodePeriodicBits> pp(e.GetNumNodes());
        for (int i = 0; i < e.GetNumNodes(); i++) pp[i] = pPbi(i);
        std::vector<NodePeriodicBits> fp(f.GetNumNodes());
        e.ExtractFaceNodes(iSub, pp, fp);
        for (int i = 0; i < f.GetNumNodes(); i++) out[i] = fp[i];
    };

    OwnershipResolverMulti resolver =
        [&](const std::vector<DNDS::index> &parents,
            const std::vector<DNDS::MPI_int> &parentRanks,
            DNDS::index nLocal) -> OwnershipDecision
    {
        DNDS::MPI_int minRank = *std::min_element(parentRanks.begin(), parentRanks.end());
        bool anyLocal = false;
        for (auto p : parents)
            if (p < nLocal) anyLocal = true;
        if (!anyLocal) return {false, {}};
        if (minRank != g_mpi.rank) return {false, {}};
        std::vector<DNDS::MPI_int> peers;
        for (size_t i = 0; i < parents.size(); i++)
            if (parents[i] >= nLocal && parentRanks[i] != g_mpi.rank)
                peers.push_back(parentRanks[i]);
        std::sort(peers.begin(), peers.end());
        peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
        return {true, std::move(peers)};
    };

    auto result = MeshConnectivity::InterpolateGlobal(
        cell2node, cell2nodePbi,
        *cellGhostMapping, *cellGM,
        *nodeGhostMapping,
        faceQuery, nCellLocal, nCellAll, nNodeTotal,
        resolver, g_mpi);

    DNDS::index localOwned = result.nOwnedEntities;
    DNDS::index globalOwned = 0;
    MPI_Allreduce(&localOwned, &globalOwned, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);

    // Triply-periodic: 3 * np*N^3 faces (N^2 faces per direction per cell-layer,
    // np*N layers in X, N in Y, N in Z).
    DNDS::index expectedFaces = 3 * np * N * N * N;
    CHECK(globalOwned == expectedFaces);

    // All faces are internal (2 parents) in triply-periodic.
    DNDS::index nBnd = 0;
    for (DNDS::index i = 0; i < result.nOwnedEntities; i++)
        if (result.entity2parent.father->RowSize(i) != 2)
            nBnd++;
    DNDS::index globalNBnd = 0;
    MPI_Allreduce(&nBnd, &globalNBnd, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    CHECK(globalNBnd == 0); // no boundary faces

    // Each cell has 6 faces.
    for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
    {
        CAPTURE(iCell);
        CHECK(result.parent2entity.father->RowSize(iCell) == 6);
    }

    // --- Verify face2nodePbi VALUES ---
    // For each owned face, the stored pbi should match the pbi extracted from
    // the first parent cell's perspective (entity2parent[iFace][0]).
    {
        DNDS::index nFail = 0;
        for (DNDS::index iFace = 0; iFace < result.nOwnedEntities; iFace++)
        {
            // Get the first parent (global A) and resolve to local-appended
            DNDS::index parentGlobal = result.entity2parent.father->operator()(iFace, 0);
            auto [pFound, pRank, parentLocal] = cellGhostMapping->search_indexAppend(parentGlobal);
            REQUIRE(pFound);

            // Find which sub-entity slot of the parent maps to this face.
            // We need the local parent2entity, but we only have global B IDs.
            // Instead, recompute the face nodes and pbi from the parent, then
            // match by comparing global node sets.

            // Get face's global node set
            auto faceNodeRow = result.entity2node.father->operator[](iFace);
            std::set<DNDS::index> faceNodeGlobals;
            for (auto gn : faceNodeRow)
                faceNodeGlobals.insert(gn);

            auto eParent = Elem::Element{cellElemInfo[parentLocal]->getElemType()};
            int nFaces = eParent.GetNumFaces();
            bool matched = false;
            for (int iSub = 0; iSub < nFaces && !matched; iSub++)
            {
                auto eFace = eParent.ObtainFace(iSub);
                int nFN = eFace.GetNumNodes();

                // Extract nodes (local-appended) and convert to global
                std::vector<DNDS::index> subNodes(nFN);
                std::vector<DNDS::index> parentNodes(8);
                for (int k = 0; k < 8; k++)
                    parentNodes[k] = cell2node(parentLocal, k);
                eParent.ExtractFaceNodes(iSub, parentNodes, subNodes);

                std::set<DNDS::index> subNodeGlobals;
                for (auto la : subNodes)
                    subNodeGlobals.insert(nodeGhostMapping->operator()(-1, la));

                if (subNodeGlobals != faceNodeGlobals)
                    continue;

                // Match found. Extract pbi from cell2nodePbi for this sub-entity.
                std::vector<NodePeriodicBits> parentPbiVec(8);
                for (int k = 0; k < 8; k++)
                    parentPbiVec[k] = cell2nodePbi(parentLocal, k);
                std::vector<NodePeriodicBits> expectedPbi(nFN);
                eParent.ExtractFaceNodes(iSub, parentPbiVec, expectedPbi);

                // Compare with stored entity2nodePbi
                REQUIRE(result.entity2nodePbi.father->RowSize(iFace) == nFN);
                for (int j = 0; j < nFN; j++)
                {
                    auto stored = result.entity2nodePbi.father->operator()(iFace, j);
                    if (!(stored == expectedPbi[j]))
                        nFail++;
                }
                matched = true;
            }
            REQUIRE(matched);
        }
        DNDS::index globalNFail = 0;
        MPI_Allreduce(&nFail, &globalNFail, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        CHECK(globalNFail == 0);
    }

    // --- Verify parent2entityPbi VALUES (faces) ---
    // For each local cell and each face slot, the parent2entityPbi should be the
    // uniform XOR between the cell's face-node pbi and the face's stored pbi.
    // We verify: for each local cell iCell, sub j → face iFace:
    //   cell's face-pbi (from cell2nodePbi) XOR face's stored pbi (entity2nodePbi)
    //   should be uniform and equal to parent2entityPbi[iCell][j].
    REQUIRE(bool(result.parent2entityPbi.father));
    {
        DNDS::index nFail = 0;
        for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
        {
            auto eCell = Elem::Element{cellElemInfo[iCell]->getElemType()};
            for (rowsize j = 0; j < result.parent2entity.father->RowSize(iCell); j++)
            {
                DNDS::index gFace = result.parent2entity.father->operator()(iCell, j);
                if (gFace == UnInitIndex)
                    continue;
                NodePeriodicBits storedRelPbi = result.parent2entityPbi.father->operator()(iCell, j);

                // Find the face in owned entities to get its stored pbi.
                // gFace is a global face ID. We need to check if it's owned by this rank.
                auto faceFatherGM = result.entity2node.trans.pLGlobalMapping;
                DNDS::index myFaceStart = (*faceFatherGM)(g_mpi.rank, 0);
                DNDS::index myFaceEnd = myFaceStart + result.nOwnedEntities;
                if (gFace < myFaceStart || gFace >= myFaceEnd)
                    continue; // face owned by another rank, skip (we don't have entity2nodePbi for it)

                DNDS::index iFaceLocal = gFace - myFaceStart;

                // Extract cell's face pbi
                auto eFace = eCell.ObtainFace(j);
                int nFN = eFace.GetNumNodes();
                std::vector<NodePeriodicBits> cellFacePbi(nFN);
                std::vector<DNDS::index> cellFaceNodes(nFN);
                {
                    std::vector<NodePeriodicBits> parentPbiVec(8);
                    std::vector<DNDS::index> parentNodes(8);
                    for (int k = 0; k < 8; k++)
                    {
                        parentPbiVec[k] = cell2nodePbi(iCell, k);
                        parentNodes[k] = cell2node(iCell, k);
                    }
                    eCell.ExtractFaceNodes(j, parentPbiVec, cellFacePbi);
                    eCell.ExtractFaceNodes(j, parentNodes, cellFaceNodes);
                }

                // Face stored pbi
                auto faceNodeRow = result.entity2node.father->operator[](iFaceLocal);

                // Match nodes by identity (convert cell nodes to global)
                using NP = std::pair<DNDS::index, uint8_t>;
                auto cmp = [](const NP &a, const NP &b)
                { return a.first == b.first ? a.second < b.second : a.first < b.first; };
                std::vector<NP> cellNP(nFN), faceNP(nFN);
                for (int k = 0; k < nFN; k++)
                {
                    cellNP[k] = {nodeGhostMapping->operator()(-1, cellFaceNodes[k]),
                                 uint8_t(cellFacePbi[k])};
                    faceNP[k] = {faceNodeRow[k],
                                 uint8_t(result.entity2nodePbi.father->operator()(iFaceLocal, k))};
                }
                std::sort(cellNP.begin(), cellNP.end(), cmp);
                std::sort(faceNP.begin(), faceNP.end(), cmp);

                // Compute expected uniform XOR
                uint8_t expectedXor = cellNP[0].second ^ faceNP[0].second;
                bool uniform = true;
                for (int k = 1; k < nFN; k++)
                    if ((cellNP[k].second ^ faceNP[k].second) != expectedXor)
                        uniform = false;

                if (!uniform || !(NodePeriodicBits{expectedXor} == storedRelPbi))
                    nFail++;
            }
        }
        DNDS::index globalNFail = 0;
        MPI_Allreduce(&nFail, &globalNFail, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        CHECK(globalNFail == 0);
    }

    // Similarly test edges.
    auto edgeQuery = makeHex8EdgeQueryPbi(cellElemInfo);
    edgeQuery.matchExtra = [&](DNDS::index iParent, int iSub,
                               DNDS::index, DNDS::index candidateParent, int candidateSub) -> bool
    {
        auto eA = Elem::Element{cellElemInfo[iParent]->getElemType()};
        auto eB = Elem::Element{cellElemInfo[candidateParent]->getElemType()};
        auto edA = eA.ObtainEdge(iSub);
        int nEN = edA.GetNumNodes();
        std::vector<DNDS::index> nodesA(nEN), nodesB(nEN);
        eA.ExtractEdgeNodes(iSub, cell2node[iParent], nodesA);
        eB.ExtractEdgeNodes(candidateSub, cell2node[candidateParent], nodesB);
        std::vector<NodePeriodicBits> pbiA(nEN), pbiB(nEN);
        eA.ExtractEdgeNodes(iSub, cell2nodePbi[iParent], pbiA);
        eB.ExtractEdgeNodes(candidateSub, cell2nodePbi[candidateParent], pbiB);
        using P = std::pair<DNDS::index, uint8_t>;
        auto cmp = [](const P &l, const P &r)
        { return l.first == r.first ? l.second < r.second : l.first < r.first; };
        std::vector<P> pa(nEN), pb(nEN);
        for (int i = 0; i < nEN; i++) { pa[i] = {nodesA[i], uint8_t(pbiA[i])}; pb[i] = {nodesB[i], uint8_t(pbiB[i])}; }
        std::sort(pa.begin(), pa.end(), cmp);
        std::sort(pb.begin(), pb.end(), cmp);
        uint8_t v0 = pa[0].second ^ pb[0].second;
        for (int i = 1; i < nEN; i++)
            if ((pa[i].second ^ pb[i].second) != v0)
                return false;
        return true;
    };
    edgeQuery.extractPbi = [&](DNDS::index iP, int iSub,
                                const std::function<NodePeriodicBits(int)> &pPbi,
                                NodePeriodicBits *out)
    {
        auto e = Elem::Element{cellElemInfo[iP]->getElemType()};
        auto ed = e.ObtainEdge(iSub);
        std::vector<NodePeriodicBits> pp(e.GetNumNodes());
        for (int i = 0; i < e.GetNumNodes(); i++) pp[i] = pPbi(i);
        std::vector<NodePeriodicBits> ep(ed.GetNumNodes());
        e.ExtractEdgeNodes(iSub, pp, ep);
        for (int i = 0; i < ed.GetNumNodes(); i++) out[i] = ep[i];
    };

    auto edgeResult = MeshConnectivity::InterpolateGlobal(
        cell2node, cell2nodePbi,
        *cellGhostMapping, *cellGM,
        *nodeGhostMapping,
        edgeQuery, nCellLocal, nCellAll, nNodeTotal,
        resolver, g_mpi);

    DNDS::index edgeLocalOwned = edgeResult.nOwnedEntities;
    DNDS::index edgeGlobalOwned = 0;
    MPI_Allreduce(&edgeLocalOwned, &edgeGlobalOwned, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);

    // Triply-periodic: 3 * np*N^3 edges (same as faces for a hex mesh).
    DNDS::index expectedEdges = 3 * np * N * N * N;
    CHECK(edgeGlobalOwned == expectedEdges);

    // All edges are internal (4 parents) in triply-periodic hex.
    for (DNDS::index i = 0; i < edgeResult.nOwnedEntities; i++)
    {
        CAPTURE(i);
        CHECK(edgeResult.entity2parent.father->RowSize(i) == 4);
    }

    // Each cell has 12 edges.
    for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
    {
        CAPTURE(iCell);
        CHECK(edgeResult.parent2entity.father->RowSize(iCell) == 12);
    }

    // --- Verify edge2nodePbi VALUES ---
    // Same approach: for each owned edge, find the first parent cell, extract
    // the expected edge pbi, and compare with stored entity2nodePbi.
    {
        DNDS::index nFail = 0;
        for (DNDS::index iEdge = 0; iEdge < edgeResult.nOwnedEntities; iEdge++)
        {
            DNDS::index parentGlobal = edgeResult.entity2parent.father->operator()(iEdge, 0);
            auto [pFound, pRank, parentLocal] = cellGhostMapping->search_indexAppend(parentGlobal);
            REQUIRE(pFound);

            auto edgeNodeRow = edgeResult.entity2node.father->operator[](iEdge);
            std::set<DNDS::index> edgeNodeGlobals;
            for (auto gn : edgeNodeRow)
                edgeNodeGlobals.insert(gn);

            auto eParent = Elem::Element{cellElemInfo[parentLocal]->getElemType()};
            int nEdges = eParent.GetNumEdges();
            bool matched = false;
            for (int iSub = 0; iSub < nEdges && !matched; iSub++)
            {
                auto eEdge = eParent.ObtainEdge(iSub);
                int nEN = eEdge.GetNumNodes();

                std::vector<DNDS::index> subNodes(nEN);
                std::vector<DNDS::index> parentNodes(8);
                for (int k = 0; k < 8; k++)
                    parentNodes[k] = cell2node(parentLocal, k);
                eParent.ExtractEdgeNodes(iSub, parentNodes, subNodes);

                std::set<DNDS::index> subNodeGlobals;
                for (auto la : subNodes)
                    subNodeGlobals.insert(nodeGhostMapping->operator()(-1, la));

                if (subNodeGlobals != edgeNodeGlobals)
                    continue;

                std::vector<NodePeriodicBits> parentPbiVec(8);
                for (int k = 0; k < 8; k++)
                    parentPbiVec[k] = cell2nodePbi(parentLocal, k);
                std::vector<NodePeriodicBits> expectedPbi(nEN);
                eParent.ExtractEdgeNodes(iSub, parentPbiVec, expectedPbi);

                REQUIRE(edgeResult.entity2nodePbi.father->RowSize(iEdge) == nEN);
                for (int j = 0; j < nEN; j++)
                {
                    auto stored = edgeResult.entity2nodePbi.father->operator()(iEdge, j);
                    if (!(stored == expectedPbi[j]))
                        nFail++;
                }
                matched = true;
            }
            REQUIRE(matched);
        }
        DNDS::index globalNFail = 0;
        MPI_Allreduce(&nFail, &globalNFail, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        CHECK(globalNFail == 0);
    }

    // --- Verify parent2entityPbi for faces is at most 1-bit ---
    // A face can cross at most one periodic boundary, so the relative pbi
    // should have at most one bit set (P1, P2, P3, or 0).
    {
        DNDS::index nMultiBit = 0;
        for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
            for (rowsize j = 0; j < result.parent2entity.father->RowSize(iCell); j++)
            {
                uint8_t v = uint8_t(result.parent2entityPbi.father->operator()(iCell, j));
                int bits = __builtin_popcount(v);
                if (bits > 1)
                    nMultiBit++;
            }
        DNDS::index globalMultiBit = 0;
        MPI_Allreduce(&nMultiBit, &globalMultiBit, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        CHECK(globalMultiBit == 0);
    }

    // --- Coordinate-based verification for faces and edges ---
    // Build synthetic node coordinates: node at local index ix*N*N+iy*N+iz
    // has physical coords (rank*N + ix, iy, iz).
    // Periodicity: P1 → +np*N in X, P2 → +N in Y, P3 → +N in Z.
    using tPoint = Eigen::Vector3d;
    auto makeCoord = [&](DNDS::index nodeGlobalIdx) -> tPoint
    {
        // Find owning rank and local index
        DNDS::MPI_int r; DNDS::index v;
        nodeGM->search(nodeGlobalIdx, r, v);
        DNDS::index localIdx = nodeGlobalIdx - (*nodeGM)(r, 0);
        DNDS::index ix = localIdx / (N * N);
        DNDS::index iy = (localIdx / N) % N;
        DNDS::index iz = localIdx % N;
        return tPoint(double(r * N + ix), double(iy), double(iz));
    };
    auto applyPbi = [&](const tPoint &c, NodePeriodicBits pbi) -> tPoint
    {
        tPoint ret = c;
        if (pbi.getP1()) ret(0) += double(np * N);
        if (pbi.getP2()) ret(1) += double(N);
        if (pbi.getP3()) ret(2) += double(N);
        return ret;
    };

    // Face center verification
    {
        DNDS::index nFail = 0;
        for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
        {
            auto eCell = Elem::Element{cellElemInfo[iCell]->getElemType()};
            for (rowsize j = 0; j < result.parent2entity.father->RowSize(iCell); j++)
            {
                DNDS::index gFace = result.parent2entity.father->operator()(iCell, j);
                if (gFace == UnInitIndex) continue;
                auto faceFatherGM = result.entity2node.trans.pLGlobalMapping;
                DNDS::index myFaceStart = (*faceFatherGM)(g_mpi.rank, 0);
                DNDS::index myFaceEnd = myFaceStart + result.nOwnedEntities;
                if (gFace < myFaceStart || gFace >= myFaceEnd) continue;
                DNDS::index iFaceLocal = gFace - myFaceStart;

                NodePeriodicBits relPbi = result.parent2entityPbi.father->operator()(iCell, j);

                // Compute face center from CELL's perspective
                auto eFace = eCell.ObtainFace(j);
                int nFN = eFace.GetNumNodes();
                std::vector<DNDS::index> cellFaceNodes(nFN);
                std::vector<NodePeriodicBits> cellFacePbiVec(nFN);
                {
                    std::vector<DNDS::index> pN(8);
                    std::vector<NodePeriodicBits> pP(8);
                    for (int k = 0; k < 8; k++)
                    {
                        pN[k] = cell2node(iCell, k);
                        pP[k] = cell2nodePbi(iCell, k);
                    }
                    eCell.ExtractFaceNodes(j, pN, cellFaceNodes);
                    eCell.ExtractFaceNodes(j, pP, cellFacePbiVec);
                }
                tPoint centerCell = tPoint::Zero();
                for (int k = 0; k < nFN; k++)
                {
                    DNDS::index ng = nodeGhostMapping->operator()(-1, cellFaceNodes[k]);
                    centerCell += applyPbi(makeCoord(ng), cellFacePbiVec[k]);
                }
                centerCell /= double(nFN);

                // Compute face center from ENTITY's perspective, then transform via relPbi
                auto faceNodeRow = result.entity2node.father->operator[](iFaceLocal);
                tPoint centerEntity = tPoint::Zero();
                for (rowsize k = 0; k < static_cast<rowsize>(faceNodeRow.size()); k++)
                {
                    NodePeriodicBits ePbi = result.entity2nodePbi.father->operator()(iFaceLocal, k);
                    centerEntity += applyPbi(makeCoord(faceNodeRow[k]), ePbi);
                }
                centerEntity /= double(faceNodeRow.size());
                // Transform entity center to cell perspective
                tPoint centerConverted = applyPbi(centerEntity, relPbi);

                if ((centerCell - centerConverted).norm() > 1e-12)
                    nFail++;
            }
        }
        DNDS::index globalNFail = 0;
        MPI_Allreduce(&nFail, &globalNFail, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        CHECK(globalNFail == 0);
    }

    // Edge center verification
    {
        DNDS::index nFail = 0;
        for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
        {
            auto eCell = Elem::Element{cellElemInfo[iCell]->getElemType()};
            for (rowsize j = 0; j < edgeResult.parent2entity.father->RowSize(iCell); j++)
            {
                DNDS::index gEdge = edgeResult.parent2entity.father->operator()(iCell, j);
                if (gEdge == UnInitIndex) continue;
                auto edgeFatherGM = edgeResult.entity2node.trans.pLGlobalMapping;
                DNDS::index myEdgeStart = (*edgeFatherGM)(g_mpi.rank, 0);
                DNDS::index myEdgeEnd = myEdgeStart + edgeResult.nOwnedEntities;
                if (gEdge < myEdgeStart || gEdge >= myEdgeEnd) continue;
                DNDS::index iEdgeLocal = gEdge - myEdgeStart;

                NodePeriodicBits relPbi = edgeResult.parent2entityPbi.father->operator()(iCell, j);

                // Compute edge center from CELL's perspective
                auto eEdge = eCell.ObtainEdge(j);
                int nEN = eEdge.GetNumNodes();
                std::vector<DNDS::index> cellEdgeNodes(nEN);
                std::vector<NodePeriodicBits> cellEdgePbiVec(nEN);
                {
                    std::vector<DNDS::index> pN(8);
                    std::vector<NodePeriodicBits> pP(8);
                    for (int k = 0; k < 8; k++)
                    {
                        pN[k] = cell2node(iCell, k);
                        pP[k] = cell2nodePbi(iCell, k);
                    }
                    eCell.ExtractEdgeNodes(j, pN, cellEdgeNodes);
                    eCell.ExtractEdgeNodes(j, pP, cellEdgePbiVec);
                }
                tPoint centerCell = tPoint::Zero();
                for (int k = 0; k < nEN; k++)
                {
                    DNDS::index ng = nodeGhostMapping->operator()(-1, cellEdgeNodes[k]);
                    centerCell += applyPbi(makeCoord(ng), cellEdgePbiVec[k]);
                }
                centerCell /= double(nEN);

                // Compute edge center from ENTITY's perspective, then transform via relPbi
                auto edgeNodeRow = edgeResult.entity2node.father->operator[](iEdgeLocal);
                tPoint centerEntity = tPoint::Zero();
                for (rowsize k = 0; k < static_cast<rowsize>(edgeNodeRow.size()); k++)
                {
                    NodePeriodicBits ePbi = edgeResult.entity2nodePbi.father->operator()(iEdgeLocal, k);
                    centerEntity += applyPbi(makeCoord(edgeNodeRow[k]), ePbi);
                }
                centerEntity /= double(edgeNodeRow.size());
                tPoint centerConverted = applyPbi(centerEntity, relPbi);

                if ((centerCell - centerConverted).norm() > 1e-12)
                    nFail++;
            }
        }
        DNDS::index globalNFail = 0;
        MPI_Allreduce(&nFail, &globalNFail, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
        CHECK(globalNFail == 0);
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
