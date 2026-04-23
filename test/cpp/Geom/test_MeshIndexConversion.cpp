/**
 * @file test_MeshIndexConversion.cpp
 * @brief Parameterized doctest-based unit tests for UnstructuredMesh index 
 *        conversion and adjacency Global2Local / Local2Global methods.
 *
 * Tests run against 4 mesh configurations (same as test_MeshPipeline):
 *   [0] UniformSquare_10  -- 2D, 100 quad cells, non-periodic
 *   [1] IV10_10           -- 2D, 100 quad cells, periodic (isentropic vortex)
 *   [2] NACA0012_H2       -- 2D, 20816 unstructured quad cells, non-periodic
 *   [3] IV10U_10          -- 2D, 322 unstructured tri cells, periodic
 *
 * Each mesh is partitioned with Metis, ghost layers built, then exercises:
 *   1. Per-entity index conversions (Global2Local, Local2Global, _NoSon)
 *   2. Adjacency state-machine round-trips (Primary, ForBnd)
 *   3. Round-trip stability under repeated conversions
 *   4. UnInitIndex pass-through
 *   5. Negative encoding for not-found globals
 *   6. Local index range validity after AdjGlobal2LocalPrimary
 *
 * All meshes are pre-built in main() and destroyed together after tests.
 * This avoids an OpenMPI resource exhaustion observed when building/destroying
 * 20+ meshes in a single process at np=8.
 *
 * Run under mpirun with 1, 2, 4, and 8 ranks.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "Geom/Mesh/Mesh.hpp"
#include <string>
#include <vector>

using namespace DNDS;
using namespace DNDS::Geom;

// NOTE: DNDS::index, DNDS::real, DNDS::rowsize clash with POSIX symbols
// from <strings.h> (pulled in by doctest/MPI).  Every declaration of these
// types must use the fully-qualified DNDS:: prefix.  See AGENTS.md.

// ---------------------------------------------------------------------------
// Mesh configuration (shared with test_MeshPipeline)
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
};

static const MeshConfig g_configs[] = {
    {"UniformSquare_10", "UniformSquare_10.cgns", 2, false,
     {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {"IV10_10", "IV10_10.cgns", 2, true,
     {10, 0, 0}, {0, 10, 0}, {0, 0, 10}},
    {"NACA0012_H2", "NACA0012_H2.cgns", 2, false,
     {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {"IV10U_10", "IV10U_10.cgns", 2, true,
     {10, 0, 0}, {0, 10, 0}, {0, 0, 10}},
    {"Ball2", "Ball2.cgns", 3, false,
     {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},  // 3D mixed-element mesh (Ball2 is np=8 only)
};
static constexpr int N_CONFIGS = sizeof(g_configs) / sizeof(g_configs[0]);

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static MPIInfo g_mpi;

/// One mesh per config for read-only tests.
static ssp<UnstructuredMesh> g_meshes[N_CONFIGS];

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

static ssp<UnstructuredMesh> buildMesh(int meshId, const MeshConfig &cfg)
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

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << " " << cfg.name << "] ReadFromCGNSSerial..." << std::endl;
    reader.ReadFromCGNSSerial(meshPath(cfg.file));

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << " " << cfg.name << "] Deduplicate1to1Periodic..." << std::endl;
    reader.Deduplicate1to1Periodic(1e-8);

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << " " << cfg.name << "] BuildCell2Cell..." << std::endl;
    reader.BuildCell2Cell();

    UnstructuredMeshSerialRW::PartitionOptions pOpt;
    pOpt.metisType = "KWAY";
    pOpt.metisUfactor = 30;
    pOpt.metisSeed = 42;
    pOpt.metisNcuts = 1;

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << " " << cfg.name << "] MeshPartitionCell2Cell..." << std::endl;
    reader.MeshPartitionCell2Cell(pOpt);

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << " " << cfg.name << "] PartitionReorderToMeshCell2Cell..." << std::endl;
    reader.PartitionReorderToMeshCell2Cell();

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << " " << cfg.name << "] RecoverNode2CellAndNode2Bnd..." << std::endl;
    mesh->RecoverNode2CellAndNode2Bnd();

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << " " << cfg.name << "] RecoverCell2CellAndBnd2Cell..." << std::endl;
    mesh->RecoverCell2CellAndBnd2Cell();

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << " " << cfg.name << "] BuildGhostPrimary..." << std::endl;
    mesh->BuildGhostPrimary();

    if (g_mpi.rank == 0)
        std::cout << "[buildMesh " << meshId << " " << cfg.name << "] AdjGlobal2LocalPrimary..." << std::endl;
    mesh->AdjGlobal2LocalPrimary();

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

// ---------------------------------------------------------------------------
// Parameterized test macro
// ---------------------------------------------------------------------------

#define FOR_EACH_MESH_CONFIG(body)                                        \
    for (int _ci = 0; _ci < N_CONFIGS; _ci++)                            \
    {                                                                     \
        /* Ball2 (index 4) is only tested with np=8 */                   \
        if (_ci == 4 && g_mpi.size != 8) continue;                       \
        CAPTURE(_ci);                                                     \
        const auto &cfg = g_configs[_ci];                                \
        auto m = g_meshes[_ci];                                           \
        SUBCASE(cfg.name) { body }                                        \
    }

// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();

    // Pre-build all mesh configs for read-only tests.
    for (int i = 0; i < N_CONFIGS; i++)
    {
        // Skip Ball2 (index 4) for np < 8 to avoid timeout
        if (i == 4 && g_mpi.size != 8)
            continue;
        g_meshes[i] = buildMesh(i, g_configs[i]);
    }

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    // Destroy all meshes together before MPI_Finalize.
    for (auto &m : g_meshes)
        m.reset();
    MPI_Finalize();
    return res;
}

// ===========================================================================
// Mesh sanity checks (parameterized over all configs)
// ===========================================================================
TEST_CASE("Mesh setup: every rank has cells and nodes")
{
    FOR_EACH_MESH_CONFIG({
        CHECK(m->NumCell() >= 1);
        CHECK(m->NumNode() >= 1);
        CHECK(m->adjPrimaryState == Adj_PointToLocal);
    })
}

TEST_CASE("Mesh setup: multi-rank produces ghosts")
{
    if (g_mpi.size < 2)
        return;
    FOR_EACH_MESH_CONFIG({
        CHECK(m->NumCellGhost() > 0);
        CHECK(m->NumNodeGhost() > 0);
    })
}

TEST_CASE("Mesh setup: periodic state matches configuration")
{
    FOR_EACH_MESH_CONFIG({
        // cfg.periodic indicates if the mesh was configured as periodic
        // After Deduplicate1to1Periodic, isPeriodic should match
        CHECK(m->isPeriodic == cfg.periodic);
    })
}

TEST_CASE("Ball2: mixed 3D element types")
{
    // This test only runs for Ball2 (index 4) with np=8
    if (g_mpi.size != 8)
        return;

    auto m = g_meshes[4]; // Ball2 is index 4
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
// Per-entity index conversions -- Node (parameterized)
// ===========================================================================
TEST_CASE("NodeIndex: Global2Local round-trip on father nodes")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iNode = 0; iNode < m->NumNode(); iNode++)
        {
            DNDS::index g = m->NodeIndexLocal2Global(iNode);
            CHECK(g >= 0);
            CHECK(m->NodeIndexGlobal2Local(g) == iNode);
        }
    })
}

TEST_CASE("NodeIndex: Global2Local round-trip on ghost nodes")
{
    if (g_mpi.size < 2)
        return;
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index i = m->NumNode(); i < m->NumNode() + m->NumNodeGhost(); i++)
        {
            DNDS::index g = m->NodeIndexLocal2Global(i);
            CHECK(g >= 0);
            CHECK(m->NodeIndexGlobal2Local(g) == i);
        }
    })
}

TEST_CASE("NodeIndex: _NoSon round-trip on father nodes")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iNode = 0; iNode < m->NumNode(); iNode++)
        {
            DNDS::index g = m->NodeIndexLocal2Global_NoSon(iNode);
            CHECK(g >= 0);
            CHECK(m->NodeIndexGlobal2Local_NoSon(g) == iNode);
        }
    })
}

TEST_CASE("NodeIndex: _NoSon returns negative for non-local global")
{
    if (g_mpi.size < 2)
        return;
    FOR_EACH_MESH_CONFIG({
        MPI_int other = (g_mpi.rank + 1) % g_mpi.size;
        DNDS::index g = (*m->coords.father->pLGlobalMapping)(other, 0);
        CHECK(m->NodeIndexGlobal2Local_NoSon(g) < 0);
    })
}

TEST_CASE("NodeIndex: Global2Local returns negative for unknown global")
{
    FOR_EACH_MESH_CONFIG({
        DNDS::index bogus = m->coords.father->globalSize() + 999;
        DNDS::index result = m->NodeIndexGlobal2Local(bogus);
        CHECK(result < 0);
        CHECK(result == -1 - bogus);
    })
}

// ===========================================================================
// Per-entity index conversions -- Cell (parameterized)
// ===========================================================================
TEST_CASE("CellIndex: Global2Local round-trip on father cells")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iCell = 0; iCell < m->NumCell(); iCell++)
        {
            DNDS::index g = m->CellIndexLocal2Global(iCell);
            CHECK(g >= 0);
            CHECK(m->CellIndexGlobal2Local(g) == iCell);
        }
    })
}

TEST_CASE("CellIndex: Global2Local round-trip on ghost cells")
{
    if (g_mpi.size < 2)
        return;
    FOR_EACH_MESH_CONFIG({
        CHECK(m->NumCellGhost() > 0);
        for (DNDS::index i = m->NumCell(); i < m->NumCell() + m->NumCellGhost(); i++)
        {
            DNDS::index g = m->CellIndexLocal2Global(i);
            CHECK(g >= 0);
            CHECK(m->CellIndexGlobal2Local(g) == i);
        }
    })
}

TEST_CASE("CellIndex: _NoSon round-trip on father cells")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iCell = 0; iCell < m->NumCell(); iCell++)
        {
            DNDS::index g = m->CellIndexLocal2Global_NoSon(iCell);
            CHECK(g >= 0);
            CHECK(m->CellIndexGlobal2Local_NoSon(g) == iCell);
        }
    })
}

TEST_CASE("CellIndex: _NoSon returns negative for non-local global")
{
    if (g_mpi.size < 2)
        return;
    FOR_EACH_MESH_CONFIG({
        MPI_int other = (g_mpi.rank + 1) % g_mpi.size;
        DNDS::index g = (*m->cell2node.father->pLGlobalMapping)(other, 0);
        CHECK(m->CellIndexGlobal2Local_NoSon(g) < 0);
    })
}

// ===========================================================================
// Per-entity index conversions -- Bnd (parameterized)
// ===========================================================================
TEST_CASE("BndIndex: Global2Local round-trip on father bnds")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iBnd = 0; iBnd < m->NumBnd(); iBnd++)
        {
            DNDS::index g = m->BndIndexLocal2Global(iBnd);
            CHECK(g >= 0);
            CHECK(m->BndIndexGlobal2Local(g) == iBnd);
        }
    })
}

TEST_CASE("BndIndex: _NoSon round-trip on father bnds")
{
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iBnd = 0; iBnd < m->NumBnd(); iBnd++)
        {
            DNDS::index g = m->BndIndexLocal2Global_NoSon(iBnd);
            CHECK(g >= 0);
            CHECK(m->BndIndexGlobal2Local_NoSon(g) == iBnd);
        }
    })
}

// ===========================================================================
// UnInitIndex pass-through and special encodings (parameterized)
// ===========================================================================
TEST_CASE("UnInitIndex pass-through for all 12 conversion methods")
{
    FOR_EACH_MESH_CONFIG({
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
    })
}

TEST_CASE("Local2Global: negative local index decodes via -1-x encoding")
{
    FOR_EACH_MESH_CONFIG({
        DNDS::index fakeGlobal = 42;
        DNDS::index neg = -1 - fakeGlobal;

        CHECK(m->NodeIndexLocal2Global(neg) == fakeGlobal);
        CHECK(m->CellIndexLocal2Global(neg) == fakeGlobal);
        CHECK(m->BndIndexLocal2Global(neg) == fakeGlobal);
    })
}

// ===========================================================================
// Local index range validity (parameterized)
// ===========================================================================
TEST_CASE("AdjPrimary: cell2node local indices are in valid range")
{
    FOR_EACH_MESH_CONFIG({
        REQUIRE(m->adjPrimaryState == Adj_PointToLocal);
        DNDS::index totalNodes = m->NumNode() + m->NumNodeGhost();
        for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
            for (DNDS::rowsize j = 0; j < m->cell2node.RowSize(iC); j++)
            {
                DNDS::index iNode = m->cell2node(iC, j);
                CHECK(iNode >= 0);
                CHECK(iNode < totalNodes);
            }
    })
}

TEST_CASE("AdjPrimary: cell2cell local entries are valid or not-found")
{
    FOR_EACH_MESH_CONFIG({
        REQUIRE(m->adjPrimaryState == Adj_PointToLocal);
        DNDS::index totalCells = m->NumCell() + m->NumCellGhost();
        for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
            for (DNDS::rowsize j = 0; j < m->cell2cell.RowSize(iC); j++)
            {
                DNDS::index n = m->cell2cell(iC, j);
                if (n >= 0)
                    CHECK(n < totalCells);
            }
    })
}

TEST_CASE("AdjPrimary: bnd2cell owner cell is a local father cell")
{
    FOR_EACH_MESH_CONFIG({
        REQUIRE(m->adjPrimaryState == Adj_PointToLocal);
        for (DNDS::index ib = 0; ib < m->NumBnd(); ib++)
        {
            DNDS::index owner = m->bnd2cell(ib, 0);
            CHECK(owner >= 0);
            CHECK(owner < m->NumCell());
        }
    })
}

TEST_CASE("AdjPrimary: bnd2node local indices are in valid range")
{
    FOR_EACH_MESH_CONFIG({
        REQUIRE(m->adjPrimaryState == Adj_PointToLocal);
        DNDS::index totalNodes = m->NumNode() + m->NumNodeGhost();
        for (DNDS::index ib = 0; ib < m->NumBnd(); ib++)
            for (DNDS::rowsize j = 0; j < m->bnd2node.RowSize(ib); j++)
            {
                DNDS::index iNode = m->bnd2node(ib, j);
                CHECK(iNode >= 0);
                CHECK(iNode < totalNodes);
            }
    })
}

// ===========================================================================
// Adjacency state machine -- parameterized over all configs
// ===========================================================================
TEST_CASE("AdjPrimary: Local2Global then Global2Local is identity")
{
    FOR_EACH_MESH_CONFIG({
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
    })
}

TEST_CASE("AdjPrimary: three consecutive round-trips are stable")
{
    FOR_EACH_MESH_CONFIG({
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
    })
}

TEST_CASE("AdjPrimaryForBnd: round-trip on cell2node only")
{
    FOR_EACH_MESH_CONFIG({
        REQUIRE(m->adjPrimaryState == Adj_PointToLocal);

        auto snap = snapshotAdj(m->cell2node, m->NumCell());

        m->AdjLocal2GlobalPrimaryForBnd();
        CHECK(m->adjPrimaryState == Adj_PointToGlobal);

        m->AdjGlobal2LocalPrimaryForBnd();
        CHECK(m->adjPrimaryState == Adj_PointToLocal);

        for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
            for (DNDS::rowsize j = 0; j < m->cell2node.RowSize(iC); j++)
                CHECK(m->cell2node(iC, j) == snap[iC][j]);
    })
}
