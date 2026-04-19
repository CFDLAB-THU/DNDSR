/**
 * @file test_ArrayTransformer.cpp
 * @brief Comprehensive doctest-based unit tests for DNDS::ArrayTransformer
 *        ghost/halo communication, run under mpirun with 1, 2, and 4 ranks.
 *
 * Tests pull-based and push-based ghost exchange for TABLE_StaticFixed,
 * TABLE_Fixed, CSR, and compound element types.  Also covers persistent
 * communication and BorrowGGIndexing.
 *
 * @see @ref dnds_unit_tests for the full test-suite overview.
 * @test ParArray basics, pull (StaticFixed, Fixed, CSR, std::array),
 *       persistent pull, BorrowGGIndexing, push.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Array.hpp"
#include "DNDS/ArrayTransformer.hpp"
#include <cstdint>
#include <memory>
#include <vector>
#include <cstdlib>
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

/// Build a vector of global indices owned by other ranks (first `nPer` from
/// each non-local rank).  Indices are relative to the GlobalOffsetsMapping.
static std::vector<DNDS::index> pullFirstNFromOthers(
    const MPIInfo &mpi,
    const GlobalOffsetsMapping &gm,
    DNDS::index nPerRank)
{
    std::vector<DNDS::index> idx;
    for (MPI_int r = 0; r < mpi.size; r++)
    {
        if (r == mpi.rank)
            continue;
        DNDS::index base = gm(r, 0);
        for (DNDS::index i = 0; i < nPerRank; i++)
            idx.push_back(base + i);
    }
    return idx;
}

// ---------------------------------------------------------------------------
TEST_CASE("ParArray basics")
{
    MPIInfo mpi = worldMPI();

    ParArray<DNDS::real, 3> a;
    a.setMPI(mpi);
    a.Resize(10);

    // Fill: a(i,j) = rank*1000 + i*10 + j
    for (DNDS::index i = 0; i < a.Size(); i++)
        for (DNDS::rowsize j = 0; j < 3; j++)
            a(i, j) = mpi.rank * 1000.0 + i * 10.0 + j;

    a.createGlobalMapping();

    SUBCASE("globalSize is consistent")
    {
        DNDS::index gs = a.globalSize();
        CHECK(gs == 10 * mpi.size);
    }

    SUBCASE("AssertConsistent does not throw")
    {
        CHECK_NOTHROW(a.AssertConsistent());
    }

    SUBCASE("local data unchanged after mapping")
    {
        for (DNDS::index i = 0; i < a.Size(); i++)
            for (DNDS::rowsize j = 0; j < 3; j++)
                CHECK(a(i, j) == doctest::Approx(mpi.rank * 1000.0 + i * 10.0 + j));
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayTransformer pull - TABLE_StaticFixed")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = 100;
    constexpr DNDS::rowsize cols = 3;

    auto father = std::make_shared<ParArray<DNDS::real, cols>>(mpi);
    father->Resize(nLocal);

    // Fill with a value encoding the global index
    father->createGlobalMapping();
    DNDS::index gOff = (*father->pLGlobalMapping)(mpi.rank, 0);
    for (DNDS::index i = 0; i < nLocal; i++)
        for (DNDS::rowsize j = 0; j < cols; j++)
            (*father)(i, j) = static_cast<DNDS::real>((gOff + i) * 10 + j);

    // Ghost mapping: pull first 5 from every other rank
    auto son = std::make_shared<ParArray<DNDS::real, cols>>(mpi);
    ArrayTransformer<DNDS::real, cols> trans;
    trans.setFatherSon(father, son);
    trans.createFatherGlobalMapping();
    auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, 5);
    trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
    trans.createMPITypes();
    trans.pullOnce();

    CHECK(son->Size() == static_cast<DNDS::index>(pullIdx.size()));

    // Verify each ghost value
    for (DNDS::index g = 0; g < son->Size(); g++)
    {
        // Recover which global index this ghost corresponds to.
        // Ghost array is sorted by rank then ascending index.
        // We verify via the encoded value.
        for (DNDS::rowsize j = 0; j < cols; j++)
        {
            DNDS::real val = (*son)(g, j);
            // val == globalIdx*10 + j  =>  globalIdx = (val - j) / 10
            DNDS::index gIdx = static_cast<DNDS::index>((val - j) / 10.0 + 0.5);
            CHECK(val == doctest::Approx(static_cast<DNDS::real>(gIdx * 10 + j)));
        }
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayTransformer pull - TABLE_Fixed (DynamicSize)")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = 50;
    constexpr DNDS::rowsize dynCols = 5;

    auto father = std::make_shared<ParArray<DNDS::real, DynamicSize>>(mpi);
    father->Resize(nLocal, dynCols);

    father->createGlobalMapping();
    DNDS::index gOff = (*father->pLGlobalMapping)(mpi.rank, 0);
    for (DNDS::index i = 0; i < nLocal; i++)
        for (DNDS::rowsize j = 0; j < dynCols; j++)
            (*father)(i, j) = (gOff + i) * 0.1 + j * 0.001;

    // Pull everything: complete replication
    DNDS::index gSize = father->globalSize();
    std::vector<DNDS::index> pullAll(gSize);
    std::iota(pullAll.begin(), pullAll.end(), DNDS::index(0));

    auto son = std::make_shared<ParArray<DNDS::real, DynamicSize>>(mpi);
    ArrayTransformer<DNDS::real, DynamicSize> trans;
    trans.setFatherSon(father, son);
    trans.createFatherGlobalMapping();
    trans.createGhostMapping(std::vector<DNDS::index>(pullAll));
    trans.createMPITypes();
    trans.pullOnce();

    CHECK(son->Size() == gSize);

    for (DNDS::index g = 0; g < son->Size(); g++)
        for (DNDS::rowsize j = 0; j < dynCols; j++)
        {
            // ghost element g has global index determined by the mapping
            // We can reconstruct the expected value from the ghost index mapping
            DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
            DNDS::real expected = ghostGlobal * 0.1 + j * 0.001;
            CHECK((*son)(g, j) == doctest::Approx(expected));
        }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayTransformer pull - CSR layout")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = 30;

    auto father = std::make_shared<ParArray<DNDS::real, NonUniformSize, NonUniformSize>>(mpi);

    // Build with varying row sizes: row i has (i % 4 + 1) columns
    father->Resize(nLocal, [](DNDS::index i) -> DNDS::rowsize
                   { return static_cast<DNDS::rowsize>(i % 4 + 1); });

    father->createGlobalMapping();
    DNDS::index gOff = (*father->pLGlobalMapping)(mpi.rank, 0);
    for (DNDS::index i = 0; i < nLocal; i++)
        for (DNDS::rowsize j = 0; j < father->RowSize(i); j++)
            (*father)(i, j) = (gOff + i) * 100.0 + j;

    // Pull a subset: first 3 elements from every other rank
    auto son = std::make_shared<ParArray<DNDS::real, NonUniformSize, NonUniformSize>>(mpi);
    ArrayTransformer<DNDS::real, NonUniformSize, NonUniformSize> trans;
    trans.setFatherSon(father, son);
    trans.createFatherGlobalMapping();
    auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, 3);
    trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
    trans.createMPITypes();
    trans.pullOnce();

    CHECK(son->Size() == static_cast<DNDS::index>(pullIdx.size()));

    // Verify ghost data and row sizes
    for (DNDS::index g = 0; g < son->Size(); g++)
    {
        DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
        // Derive the original local rank and local index
        MPI_int srcRank = -1;
        DNDS::index srcLoc = -1;
        bool found = trans.pLGlobalMapping->search(ghostGlobal, srcRank, srcLoc);
        CHECK(found);

        DNDS::rowsize expectedRowSize = static_cast<DNDS::rowsize>(srcLoc % 4 + 1);
        CHECK(son->RowSize(g) == expectedRowSize);

        for (DNDS::rowsize j = 0; j < son->RowSize(g); j++)
            CHECK((*son)(g, j) == doctest::Approx(ghostGlobal * 100.0 + j));
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayTransformer pull - std::array elements")
{
    MPIInfo mpi = worldMPI();
    using Arr9 = std::array<DNDS::real, 9>;
    constexpr DNDS::index nLocal = 20;

    auto father = std::make_shared<ParArray<Arr9, DynamicSize>>(mpi);
    father->Resize(nLocal, 1);

    father->createGlobalMapping();
    DNDS::index gOff = (*father->pLGlobalMapping)(mpi.rank, 0);
    for (DNDS::index i = 0; i < nLocal; i++)
    {
        Arr9 val{};
        for (int k = 0; k < 9; k++)
            val[k] = static_cast<DNDS::real>(gOff + i) + k * 0.01;
        (*father)(i, 0) = val;
    }

    auto son = std::make_shared<ParArray<Arr9, DynamicSize>>(mpi);
    ArrayTransformer<Arr9, DynamicSize> trans;
    trans.setFatherSon(father, son);
    trans.createFatherGlobalMapping();

    // Pull first 4 from each other rank
    auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, 4);
    trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
    trans.createMPITypes();
    trans.pullOnce();

    CHECK(son->Size() == static_cast<DNDS::index>(pullIdx.size()));

    for (DNDS::index g = 0; g < son->Size(); g++)
    {
        DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
        Arr9 got = (*son)(g, 0);
        for (int k = 0; k < 9; k++)
        {
            DNDS::real expected = static_cast<DNDS::real>(ghostGlobal) + k * 0.01;
            CHECK(got[k] == doctest::Approx(expected).epsilon(1e-14));
        }
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayTransformer persistent pull")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = 40;
    constexpr DNDS::rowsize dynCols = 4;

    auto father = std::make_shared<ParArray<DNDS::real, DynamicSize>>(mpi);
    father->Resize(nLocal, dynCols);

    father->createGlobalMapping();
    DNDS::index gOff = (*father->pLGlobalMapping)(mpi.rank, 0);

    auto fillFather = [&](DNDS::real offset)
    {
        for (DNDS::index i = 0; i < nLocal; i++)
            for (DNDS::rowsize j = 0; j < dynCols; j++)
                (*father)(i, j) = (gOff + i) + j * 0.01 + offset;
    };

    fillFather(0.0);

    auto son = std::make_shared<ParArray<DNDS::real, DynamicSize>>(mpi);
    ArrayTransformer<DNDS::real, DynamicSize> trans;
    trans.setFatherSon(father, son);
    trans.createFatherGlobalMapping();

    auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, 5);
    trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
    trans.createMPITypes();

    // First persistent pull round
    trans.initPersistentPull();
    trans.startPersistentPull();
    trans.waitPersistentPull();

    CHECK(son->Size() == static_cast<DNDS::index>(pullIdx.size()));

    for (DNDS::index g = 0; g < son->Size(); g++)
    {
        DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
        for (DNDS::rowsize j = 0; j < dynCols; j++)
            CHECK((*son)(g, j) == doctest::Approx(ghostGlobal + j * 0.01 + 0.0));
    }

    // Modify father data and pull again
    fillFather(1000.0);
    trans.startPersistentPull();
    trans.waitPersistentPull();

    for (DNDS::index g = 0; g < son->Size(); g++)
    {
        DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
        for (DNDS::rowsize j = 0; j < dynCols; j++)
            CHECK((*son)(g, j) == doctest::Approx(ghostGlobal + j * 0.01 + 1000.0));
    }

    trans.clearPersistentPull();
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayTransformer BorrowGGIndexing")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = 60;
    constexpr DNDS::rowsize cols = 2;

    // First array: full ghost setup
    auto father1 = std::make_shared<ParArray<DNDS::real, cols>>(mpi);
    father1->Resize(nLocal);

    father1->createGlobalMapping();
    DNDS::index gOff = (*father1->pLGlobalMapping)(mpi.rank, 0);
    for (DNDS::index i = 0; i < nLocal; i++)
        for (DNDS::rowsize j = 0; j < cols; j++)
            (*father1)(i, j) = (gOff + i) * 10.0 + j;

    auto son1 = std::make_shared<ParArray<DNDS::real, cols>>(mpi);
    ArrayTransformer<DNDS::real, cols> trans1;
    trans1.setFatherSon(father1, son1);
    trans1.createFatherGlobalMapping();

    auto pullIdx = pullFirstNFromOthers(mpi, *trans1.pLGlobalMapping, 8);
    trans1.createGhostMapping(std::vector<DNDS::index>(pullIdx));
    trans1.createMPITypes();
    trans1.pullOnce();

    // Second array: same distribution, borrow indexing
    auto father2 = std::make_shared<ParArray<DNDS::real, cols>>(mpi);
    father2->Resize(nLocal);
    father2->createGlobalMapping();
    for (DNDS::index i = 0; i < nLocal; i++)
        for (DNDS::rowsize j = 0; j < cols; j++)
            (*father2)(i, j) = (gOff + i) * -1.0 + j * 0.5;

    auto son2 = std::make_shared<ParArray<DNDS::real, cols>>(mpi);
    ArrayTransformer<DNDS::real, cols> trans2;
    trans2.setFatherSon(father2, son2);
    trans2.BorrowGGIndexing(trans1);
    trans2.createMPITypes();
    trans2.pullOnce();

    CHECK(son2->Size() == son1->Size());

    for (DNDS::index g = 0; g < son2->Size(); g++)
    {
        DNDS::index ghostGlobal = trans2.pLGhostMapping->ghostIndex[g];
        for (DNDS::rowsize j = 0; j < cols; j++)
            CHECK((*son2)(g, j) == doctest::Approx(ghostGlobal * -1.0 + j * 0.5));
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayTransformer push")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = 50;
    constexpr DNDS::rowsize cols = 3;

    // Father: fill with zeros
    auto father = std::make_shared<ParArray<DNDS::real, cols>>(mpi);
    father->Resize(nLocal);
    for (DNDS::index i = 0; i < nLocal; i++)
        for (DNDS::rowsize j = 0; j < cols; j++)
            (*father)(i, j) = 0.0;

    father->createGlobalMapping();
    DNDS::index gOff = (*father->pLGlobalMapping)(mpi.rank, 0);

    auto son = std::make_shared<ParArray<DNDS::real, cols>>(mpi);
    ArrayTransformer<DNDS::real, cols> trans;
    trans.setFatherSon(father, son);
    trans.createFatherGlobalMapping();

    // Pull first 5 from every other rank
    auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, 5);
    trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
    trans.createMPITypes();
    trans.pullOnce(); // son now has ghost data (all zeros)

    // Write recognisable values into son (ghost data)
    for (DNDS::index g = 0; g < son->Size(); g++)
        for (DNDS::rowsize j = 0; j < cols; j++)
            (*son)(g, j) = 9999.0 + g + j * 0.1;

    // Push ghost values back to father
    trans.pushOnce();

    // Verify: the father's first 5 elements should have been overwritten
    // by the pushes from other ranks (each other rank pushed into our first 5).
    // With nproc > 1, elements 0..4 of father will get the last push value
    // (MPI push order is implementation-defined for overlapping pushes, but
    // with correct ghost mapping each ghost maps to a unique father element).
    // Our pullIdx requests indices from *other* ranks, so pushOnce sends
    // son data back to the rank that owns those father elements.
    // Therefore OUR father[0..4] will receive pushes from every other rank.

    if (mpi.size > 1)
    {
        // At least the first 5 elements of father should be non-zero now
        // (overwritten by push from remote ranks).
        bool anyNonZero = false;
        for (DNDS::index i = 0; i < 5; i++)
            for (DNDS::rowsize j = 0; j < cols; j++)
                if ((*father)(i, j) != 0.0)
                    anyNonZero = true;
        CHECK(anyNonZero);
    }

    // For a more precise check: with 2 ranks, rank 0 pulls first 5 from
    // rank 1 and vice versa.  After push, rank 0's elements 0..4 receive
    // the values that rank 1 wrote into its son, and vice versa.
    if (mpi.size == 2)
    {
        MPI_int other = 1 - mpi.rank;
        // Our first 5 elements were pushed from the other rank.
        // The other rank's son had ghost entries for our first 5.
        // Each entry is: 9999.0 + g + j*0.1 where g is the ghost index on *that* rank.
        // The ghost index ordering is deterministic (sorted), so g == 0..4
        // for the 5 elements pulled from us.
        for (DNDS::index i = 0; i < 5; i++)
            for (DNDS::rowsize j = 0; j < cols; j++)
            {
                DNDS::real expected = 9999.0 + i + j * 0.1;
                CHECK((*father)(i, j) == doctest::Approx(expected));
            }
    }
}

// ===================================================================
// Parametric: ArrayTransformer pull across types, layouts, and row sizes
// ===================================================================
// Full cross-product:
//   Types:   real, index, uint16_t, int32_t
//   Layouts: StaticFixed, Dynamic (each with RS = 1, 3, 7), CSR
//   = 4 types x (2 layouts x 3 RS + 1 CSR) = 28 cases

struct LayoutStaticFixed {};
struct LayoutDynamic {};
struct LayoutCSR {};

template <class T, class Layout, DNDS::rowsize RS>
struct TransTag
{
    using type = T;
    using layout = Layout;
    static constexpr DNDS::rowsize rs = RS;
};

#define TRANS_TAG_STR(T, L, RS) TYPE_TO_STRING(TransTag<T, L, RS>)

// real
TRANS_TAG_STR(DNDS::real, LayoutStaticFixed, 1);
TRANS_TAG_STR(DNDS::real, LayoutStaticFixed, 3);
TRANS_TAG_STR(DNDS::real, LayoutStaticFixed, 7);
TRANS_TAG_STR(DNDS::real, LayoutDynamic, 1);
TRANS_TAG_STR(DNDS::real, LayoutDynamic, 3);
TRANS_TAG_STR(DNDS::real, LayoutDynamic, 7);
TRANS_TAG_STR(DNDS::real, LayoutCSR, 0);
// index
TRANS_TAG_STR(DNDS::index, LayoutStaticFixed, 1);
TRANS_TAG_STR(DNDS::index, LayoutStaticFixed, 3);
TRANS_TAG_STR(DNDS::index, LayoutStaticFixed, 7);
TRANS_TAG_STR(DNDS::index, LayoutDynamic, 1);
TRANS_TAG_STR(DNDS::index, LayoutDynamic, 3);
TRANS_TAG_STR(DNDS::index, LayoutDynamic, 7);
TRANS_TAG_STR(DNDS::index, LayoutCSR, 0);
// uint16_t
TRANS_TAG_STR(uint16_t, LayoutStaticFixed, 1);
TRANS_TAG_STR(uint16_t, LayoutStaticFixed, 3);
TRANS_TAG_STR(uint16_t, LayoutStaticFixed, 7);
TRANS_TAG_STR(uint16_t, LayoutDynamic, 1);
TRANS_TAG_STR(uint16_t, LayoutDynamic, 3);
TRANS_TAG_STR(uint16_t, LayoutDynamic, 7);
TRANS_TAG_STR(uint16_t, LayoutCSR, 0);
// int32_t
TRANS_TAG_STR(int32_t, LayoutStaticFixed, 1);
TRANS_TAG_STR(int32_t, LayoutStaticFixed, 3);
TRANS_TAG_STR(int32_t, LayoutStaticFixed, 7);
TRANS_TAG_STR(int32_t, LayoutDynamic, 1);
TRANS_TAG_STR(int32_t, LayoutDynamic, 3);
TRANS_TAG_STR(int32_t, LayoutDynamic, 7);
TRANS_TAG_STR(int32_t, LayoutCSR, 0);

#undef TRANS_TAG_STR

#define TRANS_ALL_TAGS                                      \
    TransTag<DNDS::real, LayoutStaticFixed, 1>,             \
    TransTag<DNDS::real, LayoutStaticFixed, 3>,             \
    TransTag<DNDS::real, LayoutStaticFixed, 7>,             \
    TransTag<DNDS::real, LayoutDynamic, 1>,                 \
    TransTag<DNDS::real, LayoutDynamic, 3>,                 \
    TransTag<DNDS::real, LayoutDynamic, 7>,                 \
    TransTag<DNDS::real, LayoutCSR, 0>,                     \
    TransTag<DNDS::index, LayoutStaticFixed, 1>,            \
    TransTag<DNDS::index, LayoutStaticFixed, 3>,            \
    TransTag<DNDS::index, LayoutStaticFixed, 7>,            \
    TransTag<DNDS::index, LayoutDynamic, 1>,                \
    TransTag<DNDS::index, LayoutDynamic, 3>,                \
    TransTag<DNDS::index, LayoutDynamic, 7>,                \
    TransTag<DNDS::index, LayoutCSR, 0>,                    \
    TransTag<uint16_t, LayoutStaticFixed, 1>,               \
    TransTag<uint16_t, LayoutStaticFixed, 3>,               \
    TransTag<uint16_t, LayoutStaticFixed, 7>,               \
    TransTag<uint16_t, LayoutDynamic, 1>,                   \
    TransTag<uint16_t, LayoutDynamic, 3>,                   \
    TransTag<uint16_t, LayoutDynamic, 7>,                   \
    TransTag<uint16_t, LayoutCSR, 0>,                       \
    TransTag<int32_t, LayoutStaticFixed, 1>,                \
    TransTag<int32_t, LayoutStaticFixed, 3>,                \
    TransTag<int32_t, LayoutStaticFixed, 7>,                \
    TransTag<int32_t, LayoutDynamic, 1>,                    \
    TransTag<int32_t, LayoutDynamic, 3>,                    \
    TransTag<int32_t, LayoutDynamic, 7>,                    \
    TransTag<int32_t, LayoutCSR, 0>

TEST_CASE_TEMPLATE("ArrayTransformer pull", Tag, TRANS_ALL_TAGS)
{
    using T = typename Tag::type;
    using L = typename Tag::layout;
    constexpr DNDS::rowsize RS = Tag::rs;

    MPIInfo mpi = worldMPI();

    for (DNDS::index nLocal : {10, 50, 200})
    {
    CAPTURE(nLocal);
    constexpr DNDS::index nGhostPerRank = 5;

    if constexpr (std::is_same_v<L, LayoutStaticFixed>)
    {
        auto father = std::make_shared<ParArray<T, RS>>(mpi);
        father->Resize(nLocal);

        father->createGlobalMapping();
        DNDS::index gOff = (*father->pLGlobalMapping)(mpi.rank, 0);
        for (DNDS::index i = 0; i < nLocal; i++)
            for (DNDS::rowsize j = 0; j < RS; j++)
                (*father)(i, j) = static_cast<T>((gOff + i) * 100 + j);

        auto son = std::make_shared<ParArray<T, RS>>(mpi);
        ArrayTransformer<T, RS> trans;
        trans.setFatherSon(father, son);
        trans.createFatherGlobalMapping();
        auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, nGhostPerRank);
        trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
        trans.createMPITypes();
        trans.pullOnce();

        CHECK(son->Size() == static_cast<DNDS::index>(pullIdx.size()));

        for (DNDS::index g = 0; g < son->Size(); g++)
        {
            DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
            for (DNDS::rowsize j = 0; j < RS; j++)
                CHECK((*son)(g, j) == static_cast<T>(ghostGlobal * 100 + j));
        }
    }
    else if constexpr (std::is_same_v<L, LayoutDynamic>)
    {
        auto father = std::make_shared<ParArray<T, DynamicSize>>(mpi);
        father->Resize(nLocal, RS);

        father->createGlobalMapping();
        DNDS::index gOff = (*father->pLGlobalMapping)(mpi.rank, 0);
        for (DNDS::index i = 0; i < nLocal; i++)
            for (DNDS::rowsize j = 0; j < RS; j++)
                (*father)(i, j) = static_cast<T>((gOff + i) * 100 + j);

        auto son = std::make_shared<ParArray<T, DynamicSize>>(mpi);
        ArrayTransformer<T, DynamicSize> trans;
        trans.setFatherSon(father, son);
        trans.createFatherGlobalMapping();
        auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, nGhostPerRank);
        trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
        trans.createMPITypes();
        trans.pullOnce();

        CHECK(son->Size() == static_cast<DNDS::index>(pullIdx.size()));

        for (DNDS::index g = 0; g < son->Size(); g++)
        {
            DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
            for (DNDS::rowsize j = 0; j < RS; j++)
                CHECK((*son)(g, j) == static_cast<T>(ghostGlobal * 100 + j));
        }
    }
    else // LayoutCSR
    {
        auto father = std::make_shared<ParArray<T, NonUniformSize, NonUniformSize>>(mpi);

        father->Resize(nLocal, [](DNDS::index i) -> DNDS::rowsize
                       { return static_cast<DNDS::rowsize>(i % 4 + 1); });

        father->createGlobalMapping();
        DNDS::index gOff = (*father->pLGlobalMapping)(mpi.rank, 0);
        for (DNDS::index i = 0; i < nLocal; i++)
            for (DNDS::rowsize j = 0; j < father->RowSize(i); j++)
                (*father)(i, j) = static_cast<T>((gOff + i) * 100 + j);

        auto son = std::make_shared<ParArray<T, NonUniformSize, NonUniformSize>>(mpi);
        ArrayTransformer<T, NonUniformSize, NonUniformSize> trans;
        trans.setFatherSon(father, son);
        trans.createFatherGlobalMapping();
        auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, nGhostPerRank);
        trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
        trans.createMPITypes();
        trans.pullOnce();

        CHECK(son->Size() == static_cast<DNDS::index>(pullIdx.size()));

        for (DNDS::index g = 0; g < son->Size(); g++)
        {
            DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
            MPI_int srcRank = -1;
            DNDS::index srcLoc = -1;
            bool found = trans.pLGlobalMapping->search(ghostGlobal, srcRank, srcLoc);
            CHECK(found);

            DNDS::rowsize expectedRowSize = static_cast<DNDS::rowsize>(srcLoc % 4 + 1);
            CHECK(son->RowSize(g) == expectedRowSize);

            for (DNDS::rowsize j = 0; j < son->RowSize(g); j++)
                CHECK((*son)(g, j) == static_cast<T>(ghostGlobal * 100 + j));
        }
    }
    } // for nLocal
}
