/**
 * @file test_IndexMapping.cpp
 * @brief Doctest-based unit tests for DNDS::GlobalOffsetsMapping and
 *        DNDS::OffsetAscendIndexMapping.
 *
 * Run under mpirun with 1, 2, and 4 ranks.  Tests offset computation,
 * local-to-global and global-to-local mapping, and search operations
 * for both uniform and non-uniform distributions.
 *
 * @see @ref dnds_unit_tests for the full test-suite overview.
 * @test GlobalOffsetsMapping (uniform, non-uniform, search),
 *       OffsetAscendIndexMapping (pull-based, search_indexAppend,
 *       empty ghost set).
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/IndexMapping.hpp"
#include "DNDS/MPI.hpp"
#include <numeric>

using namespace DNDS;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();
    MPI_Finalize();
    return res;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static MPIInfo worldMPI()
{
    MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

// ===================================================================
// GlobalOffsetsMapping
// ===================================================================

TEST_CASE("GlobalOffsetsMapping uniform distribution")
{
    MPIInfo mpi = worldMPI();
    const DNDS::index nLocal = 100;

    GlobalOffsetsMapping gm;
    gm.setMPIAlignBcast(mpi, nLocal);

    SUBCASE("globalSize equals nLocal * size")
    {
        CHECK(gm.globalSize() == nLocal * mpi.size);
    }

    SUBCASE("RLengths all equal nLocal")
    {
        auto &lens = gm.RLengths();
        REQUIRE(static_cast<int>(lens.size()) == mpi.size);
        for (int r = 0; r < mpi.size; r++)
            CHECK(lens[r] == nLocal);
    }

    SUBCASE("ROffsets are multiples of nLocal")
    {
        auto &offs = gm.ROffsets();
        REQUIRE(static_cast<int>(offs.size()) == mpi.size + 1);
        for (int r = 0; r <= mpi.size; r++)
            CHECK(offs[r] == static_cast<DNDS::index>(r) * nLocal);
    }

    SUBCASE("operator() returns correct global index")
    {
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            CHECK(gm(r, 0) == static_cast<DNDS::index>(r) * nLocal);
            CHECK(gm(r, 99) == static_cast<DNDS::index>(r) * nLocal + 99);
        }
    }
}

// -------------------------------------------------------------------

TEST_CASE("GlobalOffsetsMapping non-uniform distribution")
{
    MPIInfo mpi = worldMPI();

    // Rank r owns (r+1)*10 elements
    DNDS::index myLength = static_cast<DNDS::index>(mpi.rank + 1) * 10;

    GlobalOffsetsMapping gm;
    gm.setMPIAlignBcast(mpi, myLength);

    // Expected total: sum_{r=0}^{size-1} (r+1)*10 = 10 * size*(size+1)/2
    DNDS::index expectedTotal = 10 * static_cast<DNDS::index>(mpi.size) * (mpi.size + 1) / 2;

    SUBCASE("globalSize matches expected sum")
    {
        CHECK(gm.globalSize() == expectedTotal);
    }

    SUBCASE("offsets are cumulative sums of lengths")
    {
        auto &offs = gm.ROffsets();
        auto &lens = gm.RLengths();
        CHECK(offs[0] == 0);
        for (int r = 0; r < mpi.size; r++)
        {
            CHECK(lens[r] == static_cast<DNDS::index>(r + 1) * 10);
            CHECK(offs[r + 1] == offs[r] + lens[r]);
        }
    }

    SUBCASE("search recovers rank and local index for each rank's first element")
    {
        auto &offs = gm.ROffsets();
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            auto [found, rank, loc] = gm.search(offs[r]);
            CHECK(found);
            CHECK(rank == r);
            CHECK(loc == 0);
        }
    }

    SUBCASE("search recovers rank and local index for each rank's last element")
    {
        auto &offs = gm.ROffsets();
        auto &lens = gm.RLengths();
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            DNDS::index lastGlobal = offs[r] + lens[r] - 1;
            auto [found, rank, loc] = gm.search(lastGlobal);
            CHECK(found);
            CHECK(rank == r);
            CHECK(loc == lens[r] - 1);
        }
    }
}

// -------------------------------------------------------------------

TEST_CASE("GlobalOffsetsMapping search")
{
    MPIInfo mpi = worldMPI();
    const DNDS::index nLocal = 25;

    GlobalOffsetsMapping gm;
    gm.setMPIAlignBcast(mpi, nLocal);

    SUBCASE("search for first global index")
    {
        auto [found, rank, loc] = gm.search(0);
        CHECK(found);
        CHECK(rank == 0);
        CHECK(loc == 0);
    }

    SUBCASE("search for last global index")
    {
        DNDS::index lastGlobal = gm.globalSize() - 1;
        auto [found, rank, loc] = gm.search(lastGlobal);
        CHECK(found);
        CHECK(rank == mpi.size - 1);
        CHECK(loc == nLocal - 1);
    }

    SUBCASE("search for a middle global index")
    {
        // Pick the middle element of rank 0's range
        DNDS::index mid = nLocal / 2;
        auto [found, rank, loc] = gm.search(mid);
        CHECK(found);
        CHECK(rank == 0);
        CHECK(loc == mid);
    }

    SUBCASE("search across rank boundary")
    {
        if (mpi.size > 1)
        {
            // First element of rank 1
            auto [found, rank, loc] = gm.search(nLocal);
            CHECK(found);
            CHECK(rank == 1);
            CHECK(loc == 0);
        }
    }

    SUBCASE("search out-of-range returns false")
    {
        auto [found, rank, loc] = gm.search(gm.globalSize());
        CHECK_FALSE(found);
    }

    SUBCASE("search for negative index returns false")
    {
        auto [found, rank, loc] = gm.search(-1);
        CHECK_FALSE(found);
    }

    SUBCASE("search with reference outputs matches tuple outputs")
    {
        for (DNDS::index g = 0; g < gm.globalSize(); g += nLocal)
        {
            auto [found1, rank1, loc1] = gm.search(g);
            MPI_int rank2{-1};
            DNDS::index loc2{-1};
            bool found2 = gm.search(g, rank2, loc2);
            CHECK(found1 == found2);
            CHECK(rank1 == rank2);
            CHECK(loc1 == loc2);
        }
    }
}

// ===================================================================
// OffsetAscendIndexMapping — pull-based construction
// ===================================================================

TEST_CASE("OffsetAscendIndexMapping pull-based construction")
{
    MPIInfo mpi = worldMPI();
    const DNDS::index nLocal = 50;

    GlobalOffsetsMapping gm;
    gm.setMPIAlignBcast(mpi, nLocal);

    DNDS::index myOffset = gm(mpi.rank, 0);

    // Build a pull set: request the first 3 elements from every other rank
    std::vector<DNDS::index> pullIdx;
    for (MPI_int r = 0; r < mpi.size; r++)
    {
        if (r == mpi.rank)
            continue;
        DNDS::index base = gm(r, 0);
        for (DNDS::index i = 0; i < 3; i++)
            pullIdx.push_back(base + i);
    }

    OffsetAscendIndexMapping mapping(myOffset, nLocal, std::move(pullIdx), gm, mpi);

    SUBCASE("searchInMain finds local indices")
    {
        for (DNDS::index i = 0; i < nLocal; i++)
        {
            DNDS::index val = -1;
            bool found = mapping.searchInMain(myOffset + i, val);
            CHECK(found);
            CHECK(val == i);
        }
    }

    SUBCASE("searchInMain rejects out-of-range")
    {
        DNDS::index val = -1;
        if (mpi.size > 1)
        {
            MPI_int other = (mpi.rank + 1) % mpi.size;
            bool found = mapping.searchInMain(gm(other, 0), val);
            CHECK_FALSE(found);
        }
    }

    SUBCASE("searchInGhost finds pulled indices per rank")
    {
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            if (r == mpi.rank)
                continue;
            DNDS::index base = gm(r, 0);
            for (DNDS::index i = 0; i < 3; i++)
            {
                DNDS::index val = -1;
                bool found = mapping.searchInGhost(base + i, r, val);
                CHECK(found);
                CHECK(val == i);
            }
        }
    }

    SUBCASE("searchInAllGhost finds pulled indices")
    {
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            if (r == mpi.rank)
                continue;
            DNDS::index base = gm(r, 0);
            for (DNDS::index i = 0; i < 3; i++)
            {
                MPI_int foundRank = -1;
                DNDS::index val = -1;
                bool found = mapping.searchInAllGhost(base + i, foundRank, val);
                CHECK(found);
                CHECK(foundRank == r);
                CHECK(val >= 0);
            }
        }
    }

    SUBCASE("search dispatches to main (rank==-1) and ghost correctly")
    {
        // Search for a local element
        {
            auto [found, rank, val] = mapping.search(myOffset + 5);
            CHECK(found);
            CHECK(rank == -1);
            CHECK(val == 5);
        }

        // Search for a ghost element
        if (mpi.size > 1)
        {
            MPI_int other = (mpi.rank + 1) % mpi.size;
            DNDS::index ghostGlobal = gm(other, 0);
            auto [found, rank, val] = mapping.search(ghostGlobal);
            CHECK(found);
            CHECK(rank != -1); // ghost, not main
        }
    }

    SUBCASE("operator() reverse mapping for main data")
    {
        for (DNDS::index i = 0; i < nLocal; i++)
        {
            DNDS::index global = mapping(-1, i);
            CHECK(global == myOffset + i);
        }
    }

    SUBCASE("operator() reverse mapping for ghost data")
    {
        // Recover ghost global indices via operator()(rank, val) where
        // val is relative to ghostStart[rank]
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            if (r == mpi.rank)
                continue;
            DNDS::index base = gm(r, 0);
            for (DNDS::index i = 0; i < 3; i++)
            {
                DNDS::index global = mapping(r, i);
                CHECK(global == base + i);
            }
        }
    }
}

// ===================================================================
// OffsetAscendIndexMapping search_indexAppend
// ===================================================================

TEST_CASE("OffsetAscendIndexMapping search_indexAppend")
{
    MPIInfo mpi = worldMPI();
    const DNDS::index nLocal = 50;

    GlobalOffsetsMapping gm;
    gm.setMPIAlignBcast(mpi, nLocal);

    DNDS::index myOffset = gm(mpi.rank, 0);

    // Pull first 4 elements from every other rank
    std::vector<DNDS::index> pullIdx;
    for (MPI_int r = 0; r < mpi.size; r++)
    {
        if (r == mpi.rank)
            continue;
        DNDS::index base = gm(r, 0);
        for (DNDS::index i = 0; i < 4; i++)
            pullIdx.push_back(base + i);
    }

    OffsetAscendIndexMapping mapping(myOffset, nLocal, std::move(pullIdx), gm, mpi);

    SUBCASE("search_indexAppend returns local val for main data")
    {
        auto [found, rank, val] = mapping.search_indexAppend(myOffset + 10);
        CHECK(found);
        CHECK(rank == -1);
        CHECK(val == 10);
    }

    SUBCASE("search_indexAppend offsets ghost val by mainSize")
    {
        if (mpi.size > 1)
        {
            MPI_int other = (mpi.rank + 1) % mpi.size;
            DNDS::index ghostGlobal = gm(other, 0);

            // First, get the raw ghost val from search()
            auto [found1, rank1, rawVal] = mapping.search(ghostGlobal);
            REQUIRE(found1);

            // Now get the appended val
            auto [found2, rank2, appendVal] = mapping.search_indexAppend(ghostGlobal);
            REQUIRE(found2);
            CHECK(rank2 == rank1);
            CHECK(appendVal == rawVal + nLocal);
        }
    }

    SUBCASE("search_indexAppend with reference outputs matches tuple outputs")
    {
        if (mpi.size > 1)
        {
            MPI_int other = (mpi.rank + 1) % mpi.size;
            DNDS::index ghostGlobal = gm(other, 1);

            auto [found1, rank1, val1] = mapping.search_indexAppend(ghostGlobal);

            MPI_int rank2{-1};
            DNDS::index val2{-1};
            bool found2 = mapping.search_indexAppend(ghostGlobal, rank2, val2);

            CHECK(found1 == found2);
            CHECK(rank1 == rank2);
            CHECK(val1 == val2);
        }
    }
}

// ===================================================================
// OffsetAscendIndexMapping — empty ghost set
// ===================================================================

TEST_CASE("OffsetAscendIndexMapping empty ghost set")
{
    MPIInfo mpi = worldMPI();
    const DNDS::index nLocal = 50;

    GlobalOffsetsMapping gm;
    gm.setMPIAlignBcast(mpi, nLocal);

    DNDS::index myOffset = gm(mpi.rank, 0);

    // Pull nothing
    std::vector<DNDS::index> emptyPull;
    OffsetAscendIndexMapping mapping(myOffset, nLocal, std::move(emptyPull), gm, mpi);

    SUBCASE("searchInMain still works")
    {
        for (DNDS::index i = 0; i < nLocal; i++)
        {
            DNDS::index val = -1;
            bool found = mapping.searchInMain(myOffset + i, val);
            CHECK(found);
            CHECK(val == i);
        }
    }

    SUBCASE("searchInGhost returns false for any query")
    {
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            DNDS::index val = -1;
            bool found = mapping.searchInGhost(gm(r, 0), r, val);
            CHECK_FALSE(found);
        }
    }

    SUBCASE("searchInAllGhost returns false")
    {
        if (mpi.size > 1)
        {
            MPI_int other = (mpi.rank + 1) % mpi.size;
            MPI_int foundRank = -1;
            DNDS::index val = -1;
            bool found = mapping.searchInAllGhost(gm(other, 0), foundRank, val);
            CHECK_FALSE(found);
        }
    }

    SUBCASE("search falls back to main only")
    {
        auto [found, rank, val] = mapping.search(myOffset);
        CHECK(found);
        CHECK(rank == -1);
        CHECK(val == 0);

        if (mpi.size > 1)
        {
            MPI_int other = (mpi.rank + 1) % mpi.size;
            auto [found2, rank2, val2] = mapping.search(gm(other, 0));
            CHECK_FALSE(found2);
        }
    }

    SUBCASE("operator() still works for main data")
    {
        for (DNDS::index i = 0; i < nLocal; i++)
            CHECK(mapping(-1, i) == myOffset + i);
    }
}

// ===================================================================
// Parametric tests over local sizes
// ===================================================================

template <DNDS::index N>
struct SizeTag
{
    static constexpr DNDS::index n = N;
};

TYPE_TO_STRING(SizeTag<10>);
TYPE_TO_STRING(SizeTag<50>);
TYPE_TO_STRING(SizeTag<200>);
TYPE_TO_STRING(SizeTag<1000>);

TEST_CASE_TEMPLATE("Parametric GlobalOffsetsMapping", Tag,
                    SizeTag<10>, SizeTag<50>, SizeTag<200>, SizeTag<1000>)
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = Tag::n;

    GlobalOffsetsMapping gm;
    gm.setMPIAlignBcast(mpi, nLocal);

    SUBCASE("globalSize equals nLocal * size")
    {
        CHECK(gm.globalSize() == nLocal * mpi.size);
    }

    SUBCASE("search first element of each rank")
    {
        auto &offs = gm.ROffsets();
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            auto [found, rank, loc] = gm.search(offs[r]);
            CHECK(found);
            CHECK(rank == r);
            CHECK(loc == 0);
        }
    }

    SUBCASE("operator() returns correct global index")
    {
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            CHECK(gm(r, 0) == static_cast<DNDS::index>(r) * nLocal);
            CHECK(gm(r, nLocal - 1) == static_cast<DNDS::index>(r) * nLocal + nLocal - 1);
        }
    }
}

TEST_CASE_TEMPLATE("Parametric OffsetAscendIndexMapping", Tag,
                    SizeTag<10>, SizeTag<50>, SizeTag<200>, SizeTag<1000>)
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = Tag::n;

    GlobalOffsetsMapping gm;
    gm.setMPIAlignBcast(mpi, nLocal);

    DNDS::index myOffset = gm(mpi.rank, 0);

    // Pull the first 3 elements from every other rank.
    std::vector<DNDS::index> pullIdx;
    for (MPI_int r = 0; r < mpi.size; r++)
    {
        if (r == mpi.rank)
            continue;
        DNDS::index base = gm(r, 0);
        for (DNDS::index i = 0; i < 3; i++)
            pullIdx.push_back(base + i);
    }

    OffsetAscendIndexMapping mapping(myOffset, nLocal, std::move(pullIdx), gm, mpi);

    SUBCASE("searchInMain finds local indices")
    {
        for (DNDS::index i = 0; i < nLocal; i++)
        {
            DNDS::index val = -1;
            bool found = mapping.searchInMain(myOffset + i, val);
            CHECK(found);
            CHECK(val == i);
        }
    }

    SUBCASE("searchInGhost finds pulled ghost indices")
    {
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            if (r == mpi.rank)
                continue;
            DNDS::index base = gm(r, 0);
            for (DNDS::index i = 0; i < 3; i++)
            {
                DNDS::index val = -1;
                bool found = mapping.searchInGhost(base + i, r, val);
                CHECK(found);
                CHECK(val == i);
            }
        }
    }

    SUBCASE("operator() reverse mapping for main data")
    {
        for (DNDS::index i = 0; i < nLocal; i++)
        {
            DNDS::index global = mapping(-1, i);
            CHECK(global == myOffset + i);
        }
    }

    SUBCASE("operator() reverse mapping for ghost data")
    {
        for (MPI_int r = 0; r < mpi.size; r++)
        {
            if (r == mpi.rank)
                continue;
            DNDS::index base = gm(r, 0);
            for (DNDS::index i = 0; i < 3; i++)
            {
                DNDS::index global = mapping(r, i);
                CHECK(global == base + i);
            }
        }
    }
}
