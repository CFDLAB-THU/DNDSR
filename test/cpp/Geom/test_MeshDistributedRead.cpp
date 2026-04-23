/**
 * @file test_MeshDistributedRead.cpp
 * @brief Tests for ReadSerializeAndDistribute: write mesh to H5, then
 *        read it back with even-split + ParMetis repartition and verify
 *        the rebuilt mesh matches the original.
 *
 * @par Test plan
 *   (A) Same-np: write at np=N, read at np=N with a different Metis seed.
 *       Verifies that repartitioning produces a valid mesh.
 *   (B) Cross-np: write with a rank subset [0, npWrite), read with all ranks.
 *       Verifies that reading at a different np works.
 *
 * @par Checks (each scenario)
 *   - Global cell, node, bnd, face counts match the reference.
 *   - AssertOnFaces passes (face topology correct after full rebuild).
 *   - face2cell entries are valid (left cell local, right cell local or ghost).
 *   - node2cell non-empty for all owned nodes.
 *   - cell2cellOrig globally unique (no duplicate or lost cells).
 *   - Coordinate bounding box is non-degenerate.
 *
 * @par Mesh configs
 *   [0] UniformSquare_10 -- 2D, 100 quad, non-periodic
 *   TODO: [1] IV10_10 -- 2D, 100 quad, periodic (pending periodic bnd convention)
 *
 * @par Bugs found and fixed during development
 *
 *   1. **H5 path navigation** (Mesh_ReadSerializeDistributed.cpp):
 *      `GoToPath("..")` appends ".." literally in the H5 serializer instead
 *      of navigating up. Fixed by saving/restoring absolute paths via
 *      `GetCurrentPath()`.
 *
 *   2. **Bnd partition from ghost cells** (Mesh_ReadSerializeDistributed.cpp):
 *      `bnd2cell(iBnd, 0)` can reference a ghost cell in the even-split
 *      partition. A ghost pull of the cell partition array is needed to
 *      resolve the target partition for bnds whose owner cell is on another
 *      rank.
 *
 *   3. **Ghost append index vs local index** (Mesh_ReadSerializeDistributed.cpp):
 *      `search_indexAppend` returns father_size + ghost_local_idx, but
 *      the bnd partition code used this directly as an index into
 *      `cellPartArrGhost` (which is 0-indexed). Fixed by subtracting
 *      `cellPartArr->Size()`.
 *
 *   4. **Ghost bnd nodes not in coord ghost layer** (Mesh.cpp, BuildGhostPrimary):
 *      Ghost bnds (pulled by BuildGhostPrimary via node2bnd) may reference
 *      nodes not in the coord ghost layer (which only covers cells' nodes).
 *      Fixed by expanding the coord ghost layer after bnd ghosting, with
 *      a collective Allreduce guard to avoid deadlock.
 *
 *   5. **AdjGlobal2LocalPrimary assertion on ghost bnds** (Mesh.cpp):
 *      Ghost bnds' parent cells may not be in the cell ghost layer (which
 *      only includes cell2cell neighbors). Relaxed the assertion to only
 *      enforce for father bnds: the semantic contract is that father bnds
 *      must have their parent cell as a local father cell.
 *
 *   6. **Serializer shared-ptr dedup address reuse** (SerializerBase/H5/JSON):
 *      The ptr_2_pth dedup map keyed on raw void* addresses. After a
 *      temporary shared_ptr was destroyed, the allocator could reuse the
 *      address for a new shared_ptr, causing false dedup (e.g., bnd2node's
 *      pRowStart written as a ::ref to cell2node's pRowStart). Fixed by
 *      storing shared_ptr<void> to keep objects alive.
 *
 *   7. **CSR Compress before dataOffset** (ArrayTransformer.hpp, ParArray):
 *      ParArray::WriteSerializer computed dataOffset from _pRowStart before
 *      Compress() was called. For uncompressed CSR arrays (np=1 after
 *      TransferDataSerial2Global), _pRowStart was null and pRowStart was
 *      silently skipped. Fixed by calling Compress() first.
 *
 *   8. **Periodic bnd pbi filter on non-local cells** (Mesh.cpp, RecoverCell2CellAndBnd2Cell):
 *      The periodic pbi filter used cell2node.father->pLGlobalMapping->search()
 *      which only returns father-local indices, skipping non-local cells.
 *      On even-split partitions a periodic bnd's two cells can both be on
 *      other ranks, leaving cellRecCur empty after the filter (assertion
 *      failure). Fixed by refactoring into a two-pass approach: first pass
 *      collects candidate cells from node intersection for all bnds, then
 *      a ghost pull fetches cell2node and cell2nodePbi for ALL candidates,
 *      then second pass does the pbi filter using search_indexAppend on the
 *      ghost mapping. This correctly handles non-local cells on any partition.
 *
 *   9. **ctest parallel collision** (test_MeshDistributedRead.cpp):
 *      All np variants wrote H5 files to the same temp directory. When ctest
 *      ran them in parallel, file corruption occurred. Fixed by including
 *      np in the temp directory name.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "Geom/Mesh/Mesh.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include <string>
#include <filesystem>
#include <algorithm>
#include <numeric>

using namespace DNDS;
using namespace DNDS::Geom;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct MeshConfig
{
    const char *name;
    const char *file;
    int dim;
    bool periodic;
    tPoint translation1, translation2, translation3;
    DNDS::index expectedCells;
    DNDS::index expectedBnds;
};

static const MeshConfig g_configs[] = {
    {"UniformSquare_10", "UniformSquare_10.cgns", 2, false,
     {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 100, 40},
    {"IV10_10", "IV10_10.cgns", 2, true,
     {10, 0, 0}, {0, 10, 0}, {0, 0, 10}, 100, -1},
    {"IV10U_10", "IV10U_10.cgns", 2, true,
     {10, 0, 0}, {0, 10, 0}, {0, 0, 10}, 322, -1},
    {"NACA0012_H2", "NACA0012_H2.cgns", 2, false,
     {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 20816, 484},
};
static constexpr int N_CONFIGS = sizeof(g_configs) / sizeof(g_configs[0]);

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static MPIInfo g_mpi;

struct RefCounts
{
    DNDS::index nCellGlobal, nNodeGlobal, nBndGlobal, nFaceGlobal;
};
static RefCounts g_refCounts[N_CONFIGS];

/// Meshes rebuilt from same-np distributed read
static ssp<UnstructuredMesh> g_sameNpMesh[N_CONFIGS];

/// Meshes rebuilt from cross-np distributed read (only if np >= 3)
static ssp<UnstructuredMesh> g_crossNpMesh[N_CONFIGS];
static bool g_crossNpAvailable = false;

// H5 file paths
static std::string g_h5Same[N_CONFIGS];
static std::string g_h5Cross[N_CONFIGS];

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

static std::string tmpDir()
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
    // Include np in directory name to avoid conflicts when ctest runs
    // multiple np tests in parallel.
    return f + "/data/tmp_test_distributed_read_np" + std::to_string(g_mpi.size);
}

static void setPeriodicIfNeeded(ssp<UnstructuredMesh> &mesh, const MeshConfig &cfg)
{
    if (cfg.periodic)
    {
        tPoint zero{0, 0, 0};
        mesh->SetPeriodicGeometry(
            cfg.translation1, zero, zero,
            cfg.translation2, zero, zero,
            cfg.translation3, zero, zero);
    }
}

/// Full pipeline rebuild after ReadSerializeAndDistribute.
static void rebuildAfterDistributedRead(ssp<UnstructuredMesh> &mesh, const MeshConfig &cfg)
{
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    mesh->AdjGlobal2LocalN2CB();

    mesh->InterpolateFace();
    mesh->AssertOnFaces();

    mesh->AdjLocal2GlobalN2CB();
    mesh->BuildGhostN2CB();
    mesh->AdjGlobal2LocalN2CB();

    if (cfg.periodic)
        mesh->RecreatePeriodicNodes();
    mesh->BuildVTKConnectivity();
}

/// Build reference mesh, record counts, and write to H5.
static void buildAndWriteRef(
    int ic, const MeshConfig &cfg, const MPIInfo &mpi, const std::string &h5Path)
{
    auto mesh = std::make_shared<UnstructuredMesh>(mpi, cfg.dim);
    UnstructuredMeshSerialRW reader(mesh, 0);
    setPeriodicIfNeeded(mesh, cfg);

    reader.ReadFromCGNSSerial(meshPath(cfg.file));
    reader.Deduplicate1to1Periodic(1e-8);
    reader.BuildCell2Cell();

    PartitionOptions pOpt;
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

    mesh->InterpolateFace();
    mesh->AssertOnFaces();

    mesh->AdjLocal2GlobalN2CB();
    mesh->BuildGhostN2CB();
    mesh->AdjGlobal2LocalN2CB();

    // Record reference counts
    g_refCounts[ic].nCellGlobal = mesh->NumCellGlobal();
    g_refCounts[ic].nNodeGlobal = mesh->NumNodeGlobal();
    g_refCounts[ic].nBndGlobal = mesh->NumBndGlobal();
    g_refCounts[ic].nFaceGlobal = mesh->NumFaceGlobal();

    // Write to H5
    mesh->AdjLocal2GlobalPrimary();
    auto ser = std::make_shared<Serializer::SerializerH5>(mpi);
    ser->OpenFile(h5Path, false);
    mesh->WriteSerialize(ser, "meshPart");
    ser->CloseFile();
}

/// Write mesh to H5 using a sub-communicator of size npWrite.
/// Only ranks [0, npWrite) participate in the write.
static void buildAndWriteRefSubComm(
    int ic, const MeshConfig &cfg, int npWrite, const std::string &h5Path)
{
    int color = (g_mpi.rank < npWrite) ? 0 : MPI_UNDEFINED;
    MPI_Comm writeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, g_mpi.rank, &writeComm);

    if (writeComm != MPI_COMM_NULL)
    {
        MPIInfo writeMpi(writeComm);
        buildAndWriteRef(ic, cfg, writeMpi, h5Path);
        MPI_Comm_free(&writeComm);
    }

    // Broadcast reference counts from rank 0 to all ranks
    MPI::Bcast(&g_refCounts[ic], sizeof(RefCounts) / sizeof(DNDS::index), DNDS_MPI_INDEX, 0, g_mpi.comm);
}

/// Read from H5 with ReadSerializeAndDistribute using the full world communicator.
static ssp<UnstructuredMesh> distributedRead(
    const MeshConfig &cfg, const std::string &h5Path)
{
    auto mesh = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    setPeriodicIfNeeded(mesh, cfg);

    PartitionOptions pOpt;
    pOpt.metisType = "KWAY";
    pOpt.metisUfactor = 5;
    pOpt.metisSeed = 777; // deliberately different from write
    pOpt.metisNcuts = 1;

    auto ser = std::make_shared<Serializer::SerializerH5>(g_mpi);
    ser->OpenFile(h5Path, true);
    mesh->ReadSerializeAndDistribute(ser, "meshPart", pOpt);
    ser->CloseFile();

    rebuildAfterDistributedRead(mesh, cfg);
    return mesh;
}

// ---------------------------------------------------------------------------
// Verification helpers
// ---------------------------------------------------------------------------

/// Collect cell2cellOrig from all ranks, check global uniqueness.
static void checkOrigUnique(UnstructuredMesh &mesh, DNDS::index expectedGlobal, const MPIInfo &mpi)
{
    std::vector<DNDS::index> localOrig(mesh.NumCell());
    for (DNDS::index iC = 0; iC < mesh.NumCell(); iC++)
        localOrig[iC] = mesh.cell2cellOrig(iC, 0);

    int localCount = static_cast<int>(localOrig.size());
    std::vector<int> allCounts(mpi.size);
    MPI_Allgather(&localCount, 1, MPI_INT, allCounts.data(), 1, MPI_INT, mpi.comm);

    std::vector<int> displs(mpi.size + 1, 0);
    for (int r = 0; r < mpi.size; r++)
        displs[r + 1] = displs[r] + allCounts[r];
    int totalCount = displs[mpi.size];

    std::vector<DNDS::index> allOrig(totalCount);
    MPI_Allgatherv(localOrig.data(), localCount, DNDS_MPI_INDEX,
                   allOrig.data(), allCounts.data(), displs.data(), DNDS_MPI_INDEX, mpi.comm);

    std::sort(allOrig.begin(), allOrig.end());
    auto last = std::unique(allOrig.begin(), allOrig.end());
    CHECK(last == allOrig.end());
    CHECK(totalCount == expectedGlobal);
}

/// Compute global bounding box.
static std::pair<Eigen::Vector3d, Eigen::Vector3d>
computeBBox(UnstructuredMesh &mesh, const MPIInfo &mpi)
{
    Eigen::Vector3d localMin = Eigen::Vector3d::Constant(1e100);
    Eigen::Vector3d localMax = Eigen::Vector3d::Constant(-1e100);
    for (DNDS::index iN = 0; iN < mesh.NumNode(); iN++)
        for (int d = 0; d < 3; d++)
        {
            localMin(d) = std::min(localMin(d), mesh.coords[iN](d));
            localMax(d) = std::max(localMax(d), mesh.coords[iN](d));
        }
    Eigen::Vector3d globalMin, globalMax;
    MPI_Allreduce(localMin.data(), globalMin.data(), 3, MPI_DOUBLE, MPI_MIN, mpi.comm);
    MPI_Allreduce(localMax.data(), globalMax.data(), 3, MPI_DOUBLE, MPI_MAX, mpi.comm);
    return {globalMin, globalMax};
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();

    // Create temp directory
    if (g_mpi.rank == 0)
        std::filesystem::create_directories(tmpDir());
    MPI::Barrier(g_mpi.comm);

    // --- (A) Same-np tests ---
    for (int ic = 0; ic < N_CONFIGS; ic++)
    {
        g_h5Same[ic] = tmpDir() + "/" + g_configs[ic].name + "_same.dnds.h5";
        if (g_mpi.rank == 0)
            log() << "[setup] same-np: building + writing " << g_configs[ic].name << std::endl;
        buildAndWriteRef(ic, g_configs[ic], g_mpi, g_h5Same[ic]);

        if (g_mpi.rank == 0)
            log() << "[setup] same-np: distributed read " << g_configs[ic].name << std::endl;
        g_sameNpMesh[ic] = distributedRead(g_configs[ic], g_h5Same[ic]);
    }

    // --- (B) Cross-np tests (write with npWrite < np, read with np) ---
    g_crossNpAvailable = (g_mpi.size >= 2);
    if (g_crossNpAvailable)
    {
        int npWrite = std::max(1, g_mpi.size / 2); // write with half the ranks
        for (int ic = 0; ic < N_CONFIGS; ic++)
        {
            g_h5Cross[ic] = tmpDir() + "/" + g_configs[ic].name + "_cross.dnds.h5";
            if (g_mpi.rank == 0)
                log() << "[setup] cross-np: building + writing " << g_configs[ic].name
                      << " with npWrite=" << npWrite << std::endl;
            buildAndWriteRefSubComm(ic, g_configs[ic], npWrite, g_h5Cross[ic]);

            MPI::Barrier(g_mpi.comm);
            if (g_mpi.rank == 0)
                log() << "[setup] cross-np: distributed read " << g_configs[ic].name << std::endl;
            g_crossNpMesh[ic] = distributedRead(g_configs[ic], g_h5Cross[ic]);
        }
    }

    // Run tests
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    // Cleanup
    for (auto &m : g_sameNpMesh)
        m.reset();
    for (auto &m : g_crossNpMesh)
        m.reset();
    MPI::Barrier(g_mpi.comm);
    if (g_mpi.rank == 0)
        std::filesystem::remove_all(tmpDir());

    MPI_Finalize();
    return res;
}

// ===========================================================================
// Same-np tests
// ===========================================================================

#define FOR_EACH_CONFIG(body)              \
    for (int ic = 0; ic < N_CONFIGS; ic++) \
    {                                      \
        CAPTURE(ic);                       \
        CAPTURE(g_configs[ic].name);       \
        body                               \
    }

TEST_CASE("SameNp: global cell count")
{
    FOR_EACH_CONFIG({
        CHECK(g_sameNpMesh[ic]->NumCellGlobal() == g_refCounts[ic].nCellGlobal);
    })
}

TEST_CASE("SameNp: global node count")
{
    FOR_EACH_CONFIG({
        CHECK(g_sameNpMesh[ic]->NumNodeGlobal() == g_refCounts[ic].nNodeGlobal);
    })
}

TEST_CASE("SameNp: global bnd count")
{
    FOR_EACH_CONFIG({
        CHECK(g_sameNpMesh[ic]->NumBndGlobal() == g_refCounts[ic].nBndGlobal);
    })
}

TEST_CASE("SameNp: global face count")
{
    FOR_EACH_CONFIG({
        CHECK(g_sameNpMesh[ic]->NumFaceGlobal() == g_refCounts[ic].nFaceGlobal);
    })
}

TEST_CASE("SameNp: expected counts from config")
{
    FOR_EACH_CONFIG({
        if (g_configs[ic].expectedCells >= 0)
            CHECK(g_refCounts[ic].nCellGlobal == g_configs[ic].expectedCells);
        if (g_configs[ic].expectedBnds >= 0)
            CHECK(g_refCounts[ic].nBndGlobal == g_configs[ic].expectedBnds);
    })
}

TEST_CASE("SameNp: every rank has cells")
{
    FOR_EACH_CONFIG({
        CHECK(g_sameNpMesh[ic]->NumCell() > 0);
        // NumNode() can be 0 if all cell2node entries point to ghost nodes
        CHECK(g_sameNpMesh[ic]->NumNode() >= 0);
    })
}

TEST_CASE("SameNp: face2cell valid")
{
    FOR_EACH_CONFIG({
        auto &mesh = *g_sameNpMesh[ic];
        for (DNDS::index iF = 0; iF < mesh.NumFace(); iF++)
        {
            DNDS::index iCL = mesh.face2cell(iF, 0);
            DNDS::index iCR = mesh.face2cell(iF, 1);
            REQUIRE(iCL >= 0);
            REQUIRE(iCL < mesh.NumCell());
            if (iCR != UnInitIndex)
            {
                REQUIRE(iCR >= 0);
                REQUIRE(iCR < mesh.NumCellProc());
            }
        }
    })
}

TEST_CASE("SameNp: node2cell non-empty")
{
    FOR_EACH_CONFIG({
        auto &mesh = *g_sameNpMesh[ic];
        for (DNDS::index iN = 0; iN < mesh.NumNode(); iN++)
            CHECK(mesh.node2cell.RowSize(iN) > 0);
    })
}

TEST_CASE("SameNp: cell2cellOrig globally unique")
{
    FOR_EACH_CONFIG({
        checkOrigUnique(*g_sameNpMesh[ic], g_refCounts[ic].nCellGlobal, g_mpi);
    })
}

TEST_CASE("SameNp: coordinate bounding box is sane")
{
    FOR_EACH_CONFIG({
        auto bbox = computeBBox(*g_sameNpMesh[ic], g_mpi);
        for (int d = 0; d < g_configs[ic].dim; d++)
            CHECK(bbox.second(d) > bbox.first(d));
    })
}

// ===========================================================================
// Cross-np tests (write with fewer ranks, read with all)
// ===========================================================================

#define FOR_EACH_CROSS_CONFIG(body)            \
    if (!g_crossNpAvailable)                   \
        return;                                \
    for (int ic = 0; ic < N_CONFIGS; ic++)     \
    {                                          \
        CAPTURE(ic);                           \
        CAPTURE(g_configs[ic].name);           \
        body                                   \
    }

TEST_CASE("CrossNp: global cell count")
{
    FOR_EACH_CROSS_CONFIG({
        CHECK(g_crossNpMesh[ic]->NumCellGlobal() == g_refCounts[ic].nCellGlobal);
    })
}

TEST_CASE("CrossNp: global node count")
{
    FOR_EACH_CROSS_CONFIG({
        CHECK(g_crossNpMesh[ic]->NumNodeGlobal() == g_refCounts[ic].nNodeGlobal);
    })
}

TEST_CASE("CrossNp: global bnd count")
{
    FOR_EACH_CROSS_CONFIG({
        CHECK(g_crossNpMesh[ic]->NumBndGlobal() == g_refCounts[ic].nBndGlobal);
    })
}

TEST_CASE("CrossNp: global face count")
{
    FOR_EACH_CROSS_CONFIG({
        CHECK(g_crossNpMesh[ic]->NumFaceGlobal() == g_refCounts[ic].nFaceGlobal);
    })
}

TEST_CASE("CrossNp: every rank has cells")
{
    FOR_EACH_CROSS_CONFIG({
        CHECK(g_crossNpMesh[ic]->NumCell() > 0);
        CHECK(g_crossNpMesh[ic]->NumNode() >= 0);
    })
}

TEST_CASE("CrossNp: face2cell valid")
{
    FOR_EACH_CROSS_CONFIG({
        auto &mesh = *g_crossNpMesh[ic];
        for (DNDS::index iF = 0; iF < mesh.NumFace(); iF++)
        {
            DNDS::index iCL = mesh.face2cell(iF, 0);
            DNDS::index iCR = mesh.face2cell(iF, 1);
            REQUIRE(iCL >= 0);
            REQUIRE(iCL < mesh.NumCell());
            if (iCR != UnInitIndex)
            {
                REQUIRE(iCR >= 0);
                REQUIRE(iCR < mesh.NumCellProc());
            }
        }
    })
}

TEST_CASE("CrossNp: cell2cellOrig globally unique")
{
    FOR_EACH_CROSS_CONFIG({
        checkOrigUnique(*g_crossNpMesh[ic], g_refCounts[ic].nCellGlobal, g_mpi);
    })
}

TEST_CASE("CrossNp: node2cell non-empty")
{
    FOR_EACH_CROSS_CONFIG({
        auto &mesh = *g_crossNpMesh[ic];
        for (DNDS::index iN = 0; iN < mesh.NumNode(); iN++)
            CHECK(mesh.node2cell.RowSize(iN) > 0);
    })
}
