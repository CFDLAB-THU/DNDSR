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
 *   2. Explicit detached adjacency state round-trips (Primary, ForBnd)
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

#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/Unstructured/NativeMeshPreparation.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include <string>
#include <vector>
#include <numeric>

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
    const char *name;    ///< Human-readable label
    const char *file;    ///< Filename relative to data/mesh/
    int dim;             ///< Spatial dimension
    bool periodic;       ///< Has periodic boundaries
    tPoint translation1; ///< Periodic translation axis 1
    tPoint translation2; ///< Periodic translation axis 2
    tPoint translation3; ///< Periodic translation axis 3
};

static const MeshConfig g_configs[] = {
    {"UniformSquare_10", "UniformSquare_10.cgns", 2, false, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {"IV10_10", "IV10_10.cgns", 2, true, {10, 0, 0}, {0, 10, 0}, {0, 0, 10}},
    {"NACA0012_H2", "NACA0012_H2.cgns", 2, false, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {"IV10U_10", "IV10U_10.cgns", 2, true, {10, 0, 0}, {0, 10, 0}, {0, 0, 10}},
    {"Ball2", "Ball2.cgns", 3, false, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, // 3D mixed-element mesh (Ball2 is np=8 only)
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
    NativePeriodicImportOptions periodic;
    periodic.tolerance = 1e-8;
    if (cfg.periodic)
    {
        periodic.frames.translation[1].map() = cfg.translation1;
        periodic.frames.translation[2].map() = cfg.translation2;
        periodic.frames.translation[3].map() = cfg.translation3;
    }
    AutoAppendName2ID names;
    auto owned = ReadNativeCGNSOwned(g_mpi, meshPath(cfg.file), cfg.dim, [&](const std::string &name)
                                     { return names(name); }, 0, {1, true}, periodic, NativeSerialDistribution::ReaderOwned);
    NativePartitionOptions options;
    options.metisType = "KWAY";
    options.metisUfactor = 30;
    options.metisSeed = 42;
    options.metisNcuts = 1;
    owned = NativeMeshPreparation::PartitionOwned(owned, g_mpi, options).owned;
    const auto recipe = DefaultNativeMeshGhostSpec(owned, 1, true);
    auto state = NativeMeshAssembler::Build(g_mpi, std::move(owned), recipe);
    return UnstructuredMesh::FromNative(g_mpi, std::move(state));
}

static MeshAdjacencyRead readAdj(UnstructuredMesh &mesh, AdjacencyDescriptor key)
{
    return MeshAdjacencyRead::Current(mesh, key, MeshReadRequirements::Primary());
}

static std::vector<std::vector<DNDS::index>> snapshotAdj(const MeshAdjacencyRead &adj)
{
    std::vector<std::vector<DNDS::index>> out(size_t(adj.RowCount()));
    for (DNDS::index row = 0; row < adj.RowCount(); ++row)
        for (size_t slot = 0; slot < adj.RowSize(row); ++slot)
            out[size_t(row)].push_back(adj.Raw(row, slot));
    return out;
}

static DNDS::index firstGlobalOnRank(const EntitySpace &space, int rank)
{
    const auto local = space.NumOwned();
    std::vector<DNDS::index> counts(size_t(g_mpi.size));
    MPI_Allgather(&local, 1, DNDS_MPI_INDEX, counts.data(), 1, DNDS_MPI_INDEX, g_mpi.comm);
    return std::accumulate(counts.begin(), counts.begin() + rank, DNDS::index(0));
}

static void roundTrip(UnstructuredMesh &mesh, AdjacencyDescriptor key, int repeats)
{
    auto original = readAdj(mesh, key);
    REQUIRE(original.State() == Adj_PointToLocal);
    const auto snapshot = snapshotAdj(original);
    auto round = MeshAdjacency::Convert(original, Adj_PointToLocal);
    for (int trip = 0; trip < repeats; ++trip)
    {
        auto global = MeshAdjacency::Convert(round.Read(), Adj_PointToGlobal);
        auto globalRead = global.Read();
        CHECK(globalRead.State() == Adj_PointToGlobal);
        for (DNDS::index row = 0; row < original.RowCount(); ++row)
            for (size_t slot = 0; slot < original.RowSize(row); ++slot)
            {
                const auto entry = original.ReadEntry(row, slot);
                CHECK(globalRead.Raw(row, slot) == (entry.absent ? UnInitIndex : entry.global));
            }
        round = MeshAdjacency::Convert(globalRead, Adj_PointToLocal);
        CHECK(round.Read().State() == Adj_PointToLocal);
        CHECK(snapshotAdj(round.Read()) == snapshot);
    }
    // Detached setup conversion must never change the live compute state.
    CHECK(readAdj(mesh, key).State() == Adj_PointToLocal);
    CHECK(snapshotAdj(readAdj(mesh, key)) == snapshot);
}

// ---------------------------------------------------------------------------
// Parameterized test macro
// ---------------------------------------------------------------------------

static ssp<UnstructuredMesh> FullMesh(int id)
{
    if (!g_meshes[id])
        g_meshes[id] = buildMesh(id, g_configs[id]);
    return g_meshes[id];
}

#define FOR_EACH_MESH_CONFIG(body)                     \
    for (int _ci = 0; _ci < N_CONFIGS; _ci++)          \
    {                                                  \
        /* Ball2 (index 4) is only tested with np=8 */ \
        if (_ci == 4 && g_mpi.size != 8)               \
            continue;                                  \
        CAPTURE(_ci);                                  \
        const auto &cfg = g_configs[_ci];              \
        SUBCASE(cfg.name)                              \
        {                                              \
            auto m = FullMesh(_ci);                    \
            body                                       \
        }                                              \
    }

// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    g_mpi.setWorld();

    // Build fixtures lazily after doctest selects a test/subcase.
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    // Destroy all meshes together before MPI_Finalize.
    for (auto &m : g_meshes)
        m.reset();
    DNDS::MPI::Finalize();
    return res;
}

// ===========================================================================
// Mesh sanity checks (parameterized over all configs)
// ===========================================================================
TEST_CASE("Mesh setup: every rank has cells and nodes"){
    FOR_EACH_MESH_CONFIG({
        CHECK(m->NumCell() >= 1);
        CHECK(m->NumNodeProc() >= 1);
        CHECK(m->NumNode() >= 1);
        CHECK(m->NumNode() <= m->NumNodeProc());
        CHECK(m->Entities(EntityKind::Node).NumOwned() == m->NumNode());
        CHECK(m->Entities(EntityKind::Node).NumLocal() == m->NumNodeProc());
        CHECK(readAdj(*m, Adj::Cell2Node).State() == Adj_PointToLocal);
    })}

TEST_CASE("Mesh setup: multi-rank produces ghosts")
{
    if (g_mpi.size < 2)
        return;
    FOR_EACH_MESH_CONFIG({
        CHECK(m->NumCellGhost() > 0);
        CHECK(m->NumNodeGhost() > 0);
    })
}

TEST_CASE("Mesh setup: periodic state matches configuration"){
    FOR_EACH_MESH_CONFIG({
        // cfg.periodic indicates if the mesh was configured as periodic
        // After Deduplicate1to1Periodic, isPeriodic should match
        CHECK(m->AcquireRead().HostView().IsPeriodic() == cfg.periodic);
    })}

// ===========================================================================
// Per-entity index conversions -- Node (parameterized)
// ===========================================================================
TEST_CASE("NodeIndex: Global2Local round-trip on father nodes"){
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iNode = 0; iNode < m->NumNode(); iNode++)
        {
            DNDS::index g = m->NodeIndexLocal2Global(iNode);
            CHECK(g == m->Entities(EntityKind::Node).LocalToGlobal(iNode));
            CHECK(g >= 0);
            CHECK(m->NodeIndexGlobal2Local(g) == iNode);
        }
    })}

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

TEST_CASE("NodeIndex: _NoSon round-trip on father nodes"){
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iNode = 0; iNode < m->NumNode(); iNode++)
        {
            DNDS::index g = m->NodeIndexLocal2Global_NoSon(iNode);
            CHECK(g >= 0);
            CHECK(m->NodeIndexGlobal2Local_NoSon(g) == iNode);
        }
    })}

TEST_CASE("NodeIndex: _NoSon returns negative for non-local global")
{
    if (g_mpi.size < 2)
        return;
    FOR_EACH_MESH_CONFIG({
        MPI_int other = (g_mpi.rank + 1) % g_mpi.size;
        DNDS::index g = firstGlobalOnRank(m->Entities(EntityKind::Node), other);
        CHECK(m->NodeIndexGlobal2Local_NoSon(g) < 0);
    })
}

TEST_CASE("NodeIndex: Global2Local returns negative for unknown global"){
    FOR_EACH_MESH_CONFIG({
        DNDS::index bogus = m->Entities(EntityKind::Node).NumGlobal() + 999;
        DNDS::index result = m->NodeIndexGlobal2Local(bogus);
        CHECK(result < 0);
        CHECK(result == -1 - bogus);
    })}

// ===========================================================================
// Per-entity index conversions -- Cell (parameterized)
// ===========================================================================
TEST_CASE("CellIndex: Global2Local round-trip on father cells"){
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iCell = 0; iCell < m->NumCell(); iCell++)
        {
            DNDS::index g = m->CellIndexLocal2Global(iCell);
            CHECK(g == m->Entities(EntityKind::Cell).LocalToGlobal(iCell));
            CHECK(g >= 0);
            CHECK(m->CellIndexGlobal2Local(g) == iCell);
        }
    })}

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

TEST_CASE("CellIndex: _NoSon round-trip on father cells"){
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iCell = 0; iCell < m->NumCell(); iCell++)
        {
            DNDS::index g = m->CellIndexLocal2Global_NoSon(iCell);
            CHECK(g >= 0);
            CHECK(m->CellIndexGlobal2Local_NoSon(g) == iCell);
        }
    })}

TEST_CASE("CellIndex: _NoSon returns negative for non-local global")
{
    if (g_mpi.size < 2)
        return;
    FOR_EACH_MESH_CONFIG({
        MPI_int other = (g_mpi.rank + 1) % g_mpi.size;
        DNDS::index g = firstGlobalOnRank(m->Entities(EntityKind::Cell), other);
        CHECK(m->CellIndexGlobal2Local_NoSon(g) < 0);
    })
}

// ===========================================================================
// Per-entity index conversions -- Bnd (parameterized)
// ===========================================================================
TEST_CASE("BndIndex: Global2Local round-trip on father bnds"){
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iBnd = 0; iBnd < m->NumBnd(); iBnd++)
        {
            DNDS::index g = m->BndIndexLocal2Global(iBnd);
            CHECK(g == m->Entities(EntityKind::Bnd).LocalToGlobal(iBnd));
            CHECK(g >= 0);
            CHECK(m->BndIndexGlobal2Local(g) == iBnd);
        }
    })}

TEST_CASE("BndIndex: _NoSon round-trip on father bnds"){
    FOR_EACH_MESH_CONFIG({
        for (DNDS::index iBnd = 0; iBnd < m->NumBnd(); iBnd++)
        {
            DNDS::index g = m->BndIndexLocal2Global_NoSon(iBnd);
            CHECK(g >= 0);
            CHECK(m->BndIndexGlobal2Local_NoSon(g) == iBnd);
        }
    })}

// ===========================================================================
// UnInitIndex pass-through and special encodings (parameterized)
// ===========================================================================
TEST_CASE("UnInitIndex pass-through for all 12 conversion methods"){
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
    })}

TEST_CASE("Local2Global: negative local index decodes via -1-x encoding"){
    FOR_EACH_MESH_CONFIG({
        DNDS::index fakeGlobal = 42;
        DNDS::index neg = -1 - fakeGlobal;

        CHECK(m->NodeIndexLocal2Global(neg) == fakeGlobal);
        CHECK(m->CellIndexLocal2Global(neg) == fakeGlobal);
        CHECK(m->BndIndexLocal2Global(neg) == fakeGlobal);
    })}

// ===========================================================================
// Local index range validity (parameterized)
// ===========================================================================
TEST_CASE("AdjPrimary: cell2node local indices are in valid range"){
    FOR_EACH_MESH_CONFIG({
        auto access = m->AcquireRead(MeshReadRequirements::Primary());
        const auto view = access.HostView();
        REQUIRE(view.RelationState(MeshRelation::CellNodes) == Adj_PointToLocal);
        const DNDS::index totalNodes = view.NumNodeProc();
        for (DNDS::index iC = 0; iC < view.NumCell(); ++iC)
            for (const DNDS::index iNode : view.CellNodes(iC))
            {
                CHECK(iNode >= 0);
                CHECK(iNode < totalNodes);
            }
    })}

TEST_CASE("AdjPrimary: cell2cell local entries are valid or not-found"){
    FOR_EACH_MESH_CONFIG({
        auto access = m->AcquireRead(MeshReadRequirements::Primary());
        const auto view = access.HostView();
        REQUIRE(view.RelationState(MeshRelation::CellCells) == Adj_PointToLocal);
        const DNDS::index totalCells = view.NumCellProc();
        for (DNDS::index iC = 0; iC < view.NumCell(); ++iC)
            for (const DNDS::index other : view.CellCells(iC))
                if (other >= 0)
                    CHECK(other < totalCells);
    })}

TEST_CASE("AdjPrimary: bnd2cell owner cell is a local father cell"){
    FOR_EACH_MESH_CONFIG({
        auto access = m->AcquireRead(MeshReadRequirements::Primary());
        const auto view = access.HostView();
        REQUIRE(view.RelationState(MeshRelation::BoundaryCells) == Adj_PointToLocal);
        for (DNDS::index boundary = 0; boundary < view.NumBnd(); ++boundary)
        {
            const DNDS::index owner = view.BoundaryCells(boundary)[0];
            CHECK(owner >= 0);
            CHECK(owner < view.NumCell());
        }
    })}

TEST_CASE("AdjPrimary: bnd2node local indices are in valid range"){
    FOR_EACH_MESH_CONFIG({
        auto access = m->AcquireRead(MeshReadRequirements::Primary());
        const auto view = access.HostView();
        REQUIRE(view.RelationState(MeshRelation::BoundaryNodes) == Adj_PointToLocal);
        const DNDS::index totalNodes = view.NumNodeProc();
        for (DNDS::index boundary = 0; boundary < view.NumBnd(); ++boundary)
            for (const DNDS::index node : view.BoundaryNodes(boundary))
            {
                CHECK(node >= 0);
                CHECK(node < totalNodes);
            }
    })}

// ===========================================================================
// Adjacency state machine -- parameterized over all configs
// ===========================================================================
TEST_CASE("AdjPrimary: Local2Global then Global2Local is identity"){
    FOR_EACH_MESH_CONFIG({
        roundTrip(*m, Adj::Cell2Node, 1);
        roundTrip(*m, Adj::Cell2Cell, 1);
        roundTrip(*m, Adj::Bnd2Node, 1);
        roundTrip(*m, Adj::Bnd2Cell, 1);
    })}

TEST_CASE("AdjPrimary: three consecutive round-trips are stable"){
    FOR_EACH_MESH_CONFIG({ roundTrip(*m, Adj::Cell2Node, 3); })}

TEST_CASE("AdjPrimaryForBnd: detached cell2node conversion leaves all other published relations unchanged")
{
    FOR_EACH_MESH_CONFIG({
        const auto c2c = snapshotAdj(readAdj(*m, Adj::Cell2Cell));
        const auto b2n = snapshotAdj(readAdj(*m, Adj::Bnd2Node));
        const auto b2c = snapshotAdj(readAdj(*m, Adj::Bnd2Cell));
        roundTrip(*m, Adj::Cell2Node, 1);
        CHECK(snapshotAdj(readAdj(*m, Adj::Cell2Cell)) == c2c);
        CHECK(snapshotAdj(readAdj(*m, Adj::Bnd2Node)) == b2n);
        CHECK(snapshotAdj(readAdj(*m, Adj::Bnd2Cell)) == b2c);
    })
}
