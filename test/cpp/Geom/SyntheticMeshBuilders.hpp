#pragma once
/// @file SyntheticMeshBuilders.hpp
/// @brief Shared synthetic mesh builders for MeshConnectivity unit tests.

#include "Geom/Mesh/MeshConnectivity.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <numeric>
#include <fmt/core.h>

using namespace DNDS;
using namespace DNDS::Geom;

// ===========================================================================
// Path helper
// ===========================================================================

static inline std::string meshPath(const std::string &name)
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

// ===========================================================================
// 4-quad mesh builder + helpers
// ===========================================================================

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
static inline tAdjPair make4QuadCell2Node(const MPIInfo &mpi)
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
static inline DNDS::index nodeLocalCount4Quad(const MPIInfo &mpi)
{
    if (mpi.size == 1)
        return 9;
    if (mpi.rank == 0)
        return 5; // nodes 0-4
    if (mpi.rank == 1)
        return 4; // nodes 5-8
    return 0;
}

static inline DNDS::index nodeLocal2Global4Quad(const MPIInfo &mpi, DNDS::index local)
{
    if (mpi.size == 1)
        return local;
    if (mpi.rank == 0)
        return local;
    if (mpi.rank == 1)
        return local + 5;
    return -1; // should not be called
}

static inline ssp<GlobalOffsetsMapping> makeNodeGlobalMapping4Quad(const MPIInfo &mpi)
{
    auto gm = std::make_shared<GlobalOffsetsMapping>();
    gm->setMPIAlignBcast(mpi, nodeLocalCount4Quad(mpi));
    return gm;
}

static inline DNDS::index cellLocal2Global4Quad(const MPIInfo &mpi, DNDS::index local)
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

// ===========================================================================
// MPI verification helper
// ===========================================================================

/// Collect all (to-global -> set-of-from-globals) across all ranks for verification.
static inline std::vector<std::set<DNDS::index>> gatherInverseGlobal(
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
// SubEntityQuery factories
// ===========================================================================

// ===========================================================================
// Interpolate Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: build SubEntityQuery from Element topology API
// ---------------------------------------------------------------------------

/// Build a SubEntityQuery that extracts faces (dim-1 sub-entities)
/// from parent elements described by parentElemInfo.
static inline SubEntityQuery makeFaceQuery(const tElemInfoArrayPair &parentElemInfo)
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
static inline SubEntityQuery makeEdgeQuery(const tElemInfoArrayPair &parentElemInfo)
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

// ===========================================================================
// Hand-crafted mesh helper
// ===========================================================================

/// Helper: build cell2node and cellElemInfo for a hand-crafted mesh on rank 0 only.
/// cells: vector of (ElemType, vector<node indices>).
/// Returns (cell2node, cellElemInfo, nNodes).
struct HandCraftedMesh
{
    tAdjPair cell2node;
    tElemInfoArrayPair cellElemInfo;
    DNDS::index nNodes;
};

static inline HandCraftedMesh makeHandCraftedMesh(
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
static inline std::set<DNDS::index> getEntityVertexSet(
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

// ===========================================================================
// Periodic 2x2 mesh
// ===========================================================================

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

static inline Periodic2x2Mesh makePeriodic2x2Mesh(const MPIInfo &mpi)
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

// ===========================================================================
// Periodic 2x2x2 mesh
// ===========================================================================

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

static inline Periodic2x2x2Mesh makePeriodic2x2x2Mesh(const MPIInfo &mpi)
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

// ===========================================================================
// Periodic match-extra callback factories
// ===========================================================================

static inline std::function<bool(DNDS::index, int, DNDS::index, DNDS::index, int)>
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

// ===========================================================================
// Distributed 3D hex mesh
// ===========================================================================


/// Build a distributed NxNxN hex mesh. Each rank owns one NxNxN block.
/// Blocks are stacked in the X direction: rank p has X in [p*N, (p+1)*N].
/// Node ownership: rank p owns nodes with X in [p*N, (p+1)*N), except
/// the last rank also owns X = np*N. Shared boundary nodes at X = p*N
/// are owned by rank p.
///
/// Global cell index: rank-contiguous. Rank p's cells are [p*N^3, (p+1)*N^3).
/// Cell (ix, iy, iz) on rank p → global index = p*N^3 + ix*N*N + iy*N + iz.
///
/// Global node index: rank-contiguous. Rank p owns nodes with
///   ix in [0, N) (or [0, N] for last rank), iy in [0, N], iz in [0, N].
/// Local node (ix, iy, iz) → local index = ix*(N+1)*(N+1) + iy*(N+1) + iz.
/// Global index = rank-offset + local index.
struct DistributedHex3D
{
    DNDS::index N;
    DNDS::MPI_int np;
    DNDS::index nCellLocal;
    DNDS::index nNodeOwned;   // nodes owned by this rank
    DNDS::index nNodeGhost;   // ghost nodes from neighbor ranks
    DNDS::index nNodeTotal;   // owned + ghost

    tAdjPair cell2node;       // father = local cells, son = ghost cells (1-layer)
    tElemInfoArrayPair cellElemInfo;
    tPbiPair cell2nodePbi;    // only if periodic

    ssp<GlobalOffsetsMapping> cellGM, nodeGM;
    ssp<OffsetAscendIndexMapping> cellGhostMapping, nodeGhostMapping;

    bool periodic{false};

    // Local cell (ix, iy, iz) → local cell index
    DNDS::index cellLocal(DNDS::index ix, DNDS::index iy, DNDS::index iz) const
    {
        return ix * N * N + iy * N + iz;
    }

    // Local node (ix, iy, iz) on this rank → local node index (within owned)
    DNDS::index nodeLocal(DNDS::index ix, DNDS::index iy, DNDS::index iz) const
    {
        return ix * (N + 1) * (N + 1) + iy * (N + 1) + iz;
    }

    void build(DNDS::index tileN, bool isPeriodic, const MPIInfo &mpi)
    {
        N = tileN;
        np = mpi.size;
        periodic = isPeriodic;

        // Node ownership: each rank owns N slices in X (except last: N+1 if not periodic).
        // For periodic: each rank owns exactly N*(N+1)*(N+1) nodes.
        // For non-periodic: last rank owns (N+1)*(N+1)*(N+1), others own N*(N+1)*(N+1).
        DNDS::index nodeXSlices = N;
        if (!periodic && mpi.rank == mpi.size - 1)
            nodeXSlices = N + 1;
        nNodeOwned = nodeXSlices * (N + 1) * (N + 1);
        nCellLocal = N * N * N;

        cellGM = std::make_shared<GlobalOffsetsMapping>();
        cellGM->setMPIAlignBcast(mpi, nCellLocal);
        nodeGM = std::make_shared<GlobalOffsetsMapping>();
        nodeGM->setMPIAlignBcast(mpi, nNodeOwned);

        DNDS::index cellOffset = (*cellGM)(mpi.rank, 0);
        DNDS::index nodeOffset = (*nodeGM)(mpi.rank, 0);

        // Helper: get global node index for a node at physical position
        // (gx, gy, gz) where gx = rank*N + ix.
        // The owning rank of gx is gx / N (or np-1 for periodic wrapping).
        auto nodeGlobalPhys = [&](DNDS::index gx, DNDS::index gy, DNDS::index gz) -> DNDS::index
        {
            DNDS::index totalNx = np * N;
            if (periodic)
                gx = ((gx % totalNx) + totalNx) % totalNx;
            // Find owning rank
            DNDS::MPI_int ownerRank;
            DNDS::index localIx;
            if (!periodic)
            {
                if (gx >= totalNx)
                {
                    ownerRank = mpi.size - 1;
                    localIx = gx - (mpi.size - 1) * N;
                }
                else
                {
                    ownerRank = static_cast<DNDS::MPI_int>(gx / N);
                    localIx = gx - ownerRank * N;
                }
            }
            else
            {
                ownerRank = static_cast<DNDS::MPI_int>(gx / N);
                if (ownerRank >= mpi.size)
                    ownerRank = mpi.size - 1;
                localIx = gx - ownerRank * N;
            }
            DNDS::index localIdx = localIx * (N + 1) * (N + 1) + gy * (N + 1) + gz;
            return (*nodeGM)(ownerRank, 0) + localIdx;
        };

        // Build cell2node for local cells (8 nodes per hex).
        // Also collect ghost cell and node info.
        cell2node.InitPair("c2n", mpi);
        cell2node.father->Resize(nCellLocal, 8);
        cellElemInfo.InitPair("cInfo", mpi);
        cellElemInfo.father->Resize(nCellLocal);

        // Ghost cells: 1-layer in X direction from neighbor ranks.
        // Left neighbor (rank-1): cells at ix = N-1 on rank-1 that share X-face
        // Right neighbor (rank+1): cells at ix = 0 on rank+1
        std::vector<DNDS::index> ghostCellGlobals;
        std::vector<std::vector<DNDS::index>> ghostCellNodes; // node globals per ghost cell
        DNDS::index leftRank = mpi.rank - 1;
        DNDS::index rightRank = mpi.rank + 1;
        if (periodic)
        {
            leftRank = (mpi.rank - 1 + mpi.size) % mpi.size;
            rightRank = (mpi.rank + 1) % mpi.size;
        }

        auto addGhostCellsFromRank = [&](DNDS::MPI_int srcRank, DNDS::index srcIx,
                                         DNDS::index physXBase)
        {
            for (DNDS::index iy = 0; iy < N; iy++)
                for (DNDS::index iz = 0; iz < N; iz++)
                {
                    DNDS::index gcGlobal = (*cellGM)(srcRank, 0) +
                                           srcIx * N * N + iy * N + iz;
                    ghostCellGlobals.push_back(gcGlobal);

                    std::vector<DNDS::index> nodes(8);
                    DNDS::index gx = physXBase;
                    // Hex8 node ordering: (0,0,0), (1,0,0), (1,1,0), (0,1,0),
                    //                     (0,0,1), (1,0,1), (1,1,1), (0,1,1)
                    nodes[0] = nodeGlobalPhys(gx, iy, iz);
                    nodes[1] = nodeGlobalPhys(gx + 1, iy, iz);
                    nodes[2] = nodeGlobalPhys(gx + 1, iy + 1, iz);
                    nodes[3] = nodeGlobalPhys(gx, iy + 1, iz);
                    nodes[4] = nodeGlobalPhys(gx, iy, iz + 1);
                    nodes[5] = nodeGlobalPhys(gx + 1, iy, iz + 1);
                    nodes[6] = nodeGlobalPhys(gx + 1, iy + 1, iz + 1);
                    nodes[7] = nodeGlobalPhys(gx, iy + 1, iz + 1);
                    ghostCellNodes.push_back(std::move(nodes));
                }
        };

        // Left ghost (X = rank*N - 1)
        if (periodic || mpi.rank > 0)
        {
            DNDS::index physX = mpi.rank * N - 1;
            if (periodic && mpi.rank == 0)
                physX = np * N - 1;
            addGhostCellsFromRank(static_cast<DNDS::MPI_int>(leftRank), N - 1, physX);
        }
        // Right ghost (X = (rank+1)*N)
        if (periodic || mpi.rank < mpi.size - 1)
        {
            DNDS::index physX = (mpi.rank + 1) * N;
            addGhostCellsFromRank(static_cast<DNDS::MPI_int>(rightRank), 0, physX);
        }

        // Collect all node globals (from local cells + ghost cells)
        std::set<DNDS::index> allNodeGlobals;
        for (DNDS::index ix = 0; ix < N; ix++)
            for (DNDS::index iy = 0; iy < N; iy++)
                for (DNDS::index iz = 0; iz < N; iz++)
                {
                    DNDS::index gx = mpi.rank * N + ix;
                    auto n = std::array<DNDS::index, 8>{
                        nodeGlobalPhys(gx, iy, iz),
                        nodeGlobalPhys(gx + 1, iy, iz),
                        nodeGlobalPhys(gx + 1, iy + 1, iz),
                        nodeGlobalPhys(gx, iy + 1, iz),
                        nodeGlobalPhys(gx, iy, iz + 1),
                        nodeGlobalPhys(gx + 1, iy, iz + 1),
                        nodeGlobalPhys(gx + 1, iy + 1, iz + 1),
                        nodeGlobalPhys(gx, iy + 1, iz + 1),
                    };
                    for (auto ng : n)
                        allNodeGlobals.insert(ng);
                    DNDS::index iCell = cellLocal(ix, iy, iz);
                    for (int k = 0; k < 8; k++)
                        cell2node.father->operator()(iCell, k) = n[k];
                    cellElemInfo.father->operator()(iCell, 0) = ElemInfo{
                        static_cast<t_index>(Elem::Hex8), INTERNAL_ZONE};
                }

        for (auto &gc : ghostCellNodes)
            for (auto ng : gc)
                allNodeGlobals.insert(ng);

        // Sort ghost cells by global index to match createGhostMapping's ordering.
        {
            std::vector<size_t> sortOrder(ghostCellGlobals.size());
            std::iota(sortOrder.begin(), sortOrder.end(), 0);
            std::sort(sortOrder.begin(), sortOrder.end(),
                      [&](size_t a, size_t b)
                      { return ghostCellGlobals[a] < ghostCellGlobals[b]; });
            std::vector<DNDS::index> sortedGlobals(ghostCellGlobals.size());
            std::vector<std::vector<DNDS::index>> sortedNodes(ghostCellNodes.size());
            for (size_t i = 0; i < sortOrder.size(); i++)
            {
                sortedGlobals[i] = ghostCellGlobals[sortOrder[i]];
                sortedNodes[i] = ghostCellNodes[sortOrder[i]];
            }
            ghostCellGlobals = std::move(sortedGlobals);
            ghostCellNodes = std::move(sortedNodes);
        }

        // Build ghost cell son + ghost node ghost mapping.
        cell2node.son = std::make_shared<tAdj::element_type>(ObjName{"c2n.son"}, mpi);
        cell2node.son->Resize(static_cast<DNDS::index>(ghostCellGlobals.size()), 8);
        cellElemInfo.son = std::make_shared<tElemInfoArray::element_type>(ObjName{"cInfo.son"}, mpi);
        cellElemInfo.son->Resize(static_cast<DNDS::index>(ghostCellGlobals.size()));

        // Build cell ghost mapping
        cell2node.TransAttach();
        cell2node.trans.createFatherGlobalMapping();
        cell2node.trans.createGhostMapping(ghostCellGlobals);
        cell2node.trans.createMPITypes();
        // Don't pullOnce — populate son manually AFTER createMPITypes
        // (createMPITypes resizes the son).

        cellElemInfo.TransAttach();
        cellElemInfo.trans.BorrowGGIndexing(cell2node.trans);
        cellElemInfo.trans.createMPITypes();

        // Now populate the son arrays.
        for (DNDS::index ig = 0; ig < static_cast<DNDS::index>(ghostCellGlobals.size()); ig++)
        {
            for (int k = 0; k < 8; k++)
                cell2node.son->operator()(ig, k) = ghostCellNodes[ig][k];
            cellElemInfo.son->operator()(ig, 0) = ElemInfo{
                static_cast<t_index>(Elem::Hex8), INTERNAL_ZONE};
        }

        cellGhostMapping = cell2node.trans.pLGhostMapping;

        // Build node ghost mapping
        std::vector<DNDS::index> ghostNodeGlobals;
        for (auto ng : allNodeGlobals)
        {
            DNDS::MPI_int r;
            DNDS::index v;
            if (nodeGM->search(ng, r, v) && r != mpi.rank)
                ghostNodeGlobals.push_back(ng);
        }
        // Create a dummy node array pair for ghost mapping setup.
        tAdjPair dummyNode;
        dummyNode.InitPair("dummyNode", mpi);
        dummyNode.father->Resize(nNodeOwned);
        dummyNode.TransAttach();
        dummyNode.trans.createFatherGlobalMapping();
        dummyNode.trans.createGhostMapping(ghostNodeGlobals);
        dummyNode.trans.createMPITypes();
        nodeGhostMapping = dummyNode.trans.pLGhostMapping;
        nNodeGhost = static_cast<DNDS::index>(ghostNodeGlobals.size());
        nNodeTotal = nNodeOwned + nNodeGhost;

        // Convert cell2node from global to local-appended node indices.
        for (DNDS::index iCell = 0; iCell < nCellLocal; iCell++)
            for (rowsize k = 0; k < 8; k++)
            {
                DNDS::index &ng = cell2node.father->operator()(iCell, k);
                auto [found, r, la] = nodeGhostMapping->search_indexAppend(ng);
                DNDS_assert(found);
                ng = la;
            }
        for (DNDS::index ig = 0; ig < static_cast<DNDS::index>(ghostCellGlobals.size()); ig++)
            for (int k = 0; k < 8; k++)
            {
                DNDS::index &ng = cell2node.son->operator()(ig, k);
                auto [found, r, la] = nodeGhostMapping->search_indexAppend(ng);
                DNDS_assert_info(found, fmt::format("ghost cell {} (global {}) node {} global {} not found "
                    "(rank {}, nNodeOwned {}, nNodeGhost {}, expected from ghostCellNodes {})",
                    ig, ghostCellGlobals[ig], k, ng, mpi.rank, nNodeOwned, nNodeGhost,
                    ghostCellNodes[ig][k]));
                ng = la;
            }
    }
};

// ===========================================================================
// SubEntityQueryPbi factories for Hex8
// ===========================================================================

// Helper: build SubEntityQueryPbi for face extraction from Hex8
static inline SubEntityQueryPbi makeHex8FaceQueryPbi(const tElemInfoArrayPair &cellElemInfo)
{
    SubEntityQueryPbi q;
    q.numSubEntities = [&](DNDS::index iP) -> int
    {
        return Elem::Element{cellElemInfo[iP]->getElemType()}.GetNumFaces();
    };
    q.describe = [&](DNDS::index iP, int iSub) -> SubEntityDesc
    {
        auto e = Elem::Element{cellElemInfo[iP]->getElemType()};
        auto f = e.ObtainFace(iSub);
        return SubEntityDesc{f.GetNumVertices(), f.GetNumNodes(), static_cast<t_index>(f.type)};
    };
    q.extractNodes = [&](DNDS::index iP, int iSub,
                          const std::function<DNDS::index(int)> &pN,
                          DNDS::index *out)
    {
        auto e = Elem::Element{cellElemInfo[iP]->getElemType()};
        auto f = e.ObtainFace(iSub);
        std::vector<DNDS::index> pNodes(e.GetNumNodes());
        for (int i = 0; i < e.GetNumNodes(); i++)
            pNodes[i] = pN(i);
        std::vector<DNDS::index> fNodes(f.GetNumNodes());
        e.ExtractFaceNodes(iSub, pNodes, fNodes);
        for (int i = 0; i < f.GetNumNodes(); i++)
            out[i] = fNodes[i];
    };
    return q;
}


// Helper: build SubEntityQueryPbi for edge extraction from Hex8
static inline SubEntityQueryPbi makeHex8EdgeQueryPbi(const tElemInfoArrayPair &cellElemInfo)
{
    SubEntityQueryPbi q;
    q.numSubEntities = [&](DNDS::index iP) -> int
    {
        return Elem::Element{cellElemInfo[iP]->getElemType()}.GetNumEdges();
    };
    q.describe = [&](DNDS::index iP, int iSub) -> SubEntityDesc
    {
        auto e = Elem::Element{cellElemInfo[iP]->getElemType()};
        auto ed = e.ObtainEdge(iSub);
        return SubEntityDesc{ed.GetNumVertices(), ed.GetNumNodes(), static_cast<t_index>(ed.type)};
    };
    q.extractNodes = [&](DNDS::index iP, int iSub,
                          const std::function<DNDS::index(int)> &pN,
                          DNDS::index *out)
    {
        auto e = Elem::Element{cellElemInfo[iP]->getElemType()};
        auto ed = e.ObtainEdge(iSub);
        std::vector<DNDS::index> pNodes(e.GetNumNodes());
        for (int i = 0; i < e.GetNumNodes(); i++)
            pNodes[i] = pN(i);
        std::vector<DNDS::index> eNodes(ed.GetNumNodes());
        e.ExtractEdgeNodes(iSub, pNodes, eNodes);
        for (int i = 0; i < ed.GetNumNodes(); i++)
            out[i] = eNodes[i];
    };
    return q;
}

