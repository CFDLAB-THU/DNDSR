/**
 * @file test_PermutationTransfer.cpp
 * @brief Unit tests for DNDS::PermutationTransfer.
 *
 * Tests local permutation, distributed partition transfer, and lookup
 * resolution under MPI with 1, 2, 4, and 8 ranks.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/PermutationTransfer.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include <numeric>
#include <algorithm>

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

static MPIInfo worldMPI()
{
    MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

// =================================================================
// Test: fromLocalPermutation — reverse permutation
// =================================================================

TEST_CASE("PermutationTransfer::fromLocalPermutation reverse")
{
    auto mpi = worldMPI();
    const DNDS::index nLocal = 10;

    // Create a simple array: each rank owns 10 entries with values = global index
    ArrayAdjacencyPair<1> arr;
    arr.InitPair("test_arr", mpi);
    arr.father->Resize(nLocal);
    arr.father->createGlobalMapping();

    DNDS::index myOffset = (*arr.father->pLGlobalMapping)(mpi.rank, 0);
    for (DNDS::index i = 0; i < nLocal; i++)
        arr(i, 0) = myOffset + i; // value = own global index

    // Build reverse permutation: new[i] = old[N-1-i]
    std::vector<DNDS::index> old2new(nLocal);
    for (DNDS::index i = 0; i < nLocal; i++)
        old2new[i] = nLocal - 1 - i;

    auto pt = PermutationTransfer::fromLocalPermutation(old2new, arr.father->pLGlobalMapping, mpi);

    CHECK(pt.isLocalOnly);
    CHECK(pt.size() == nLocal);
    CHECK(pt.localOld2New.size() == static_cast<size_t>(nLocal));

    // Verify new global indices
    for (DNDS::index i = 0; i < nLocal; i++)
        CHECK(pt.newGlobalIndices[i] == myOffset + old2new[i]);

    // Transfer rows
    pt.transferRows(arr, mpi);

    // After transfer: arr(newSlot, 0) should contain the old global of
    // the entity that was moved there.
    // old slot i had value (myOffset + i), moved to new slot old2new[i].
    // So new slot j should have value (myOffset + reverseOf(j)).
    // reverseOf(j) = N-1-j (since old2new[i] = N-1-i => i = N-1-j when j = old2new[i])
    for (DNDS::index j = 0; j < nLocal; j++)
    {
        DNDS::index expectedOldSlot = nLocal - 1 - j;
        DNDS::index expectedValue = myOffset + expectedOldSlot;
        CHECK(arr(j, 0) == expectedValue);
    }
}

// =================================================================
// Test: fromLocalPermutation — identity (no change)
// =================================================================

TEST_CASE("PermutationTransfer::fromLocalPermutation identity")
{
    auto mpi = worldMPI();
    const DNDS::index nLocal = 5;

    ArrayAdjacencyPair<1> arr;
    arr.InitPair("test_arr", mpi);
    arr.father->Resize(nLocal);
    arr.father->createGlobalMapping();

    DNDS::index myOffset = (*arr.father->pLGlobalMapping)(mpi.rank, 0);
    for (DNDS::index i = 0; i < nLocal; i++)
        arr(i, 0) = 100 + myOffset + i;

    // Identity permutation
    std::vector<DNDS::index> old2new(nLocal);
    std::iota(old2new.begin(), old2new.end(), DNDS::index{0});

    auto pt = PermutationTransfer::fromLocalPermutation(old2new, arr.father->pLGlobalMapping, mpi);
    CHECK(pt.isLocalOnly);

    pt.transferRows(arr, mpi);

    for (DNDS::index i = 0; i < nLocal; i++)
        CHECK(arr(i, 0) == 100 + myOffset + i);
}

// =================================================================
// Test: fromPartition — all stay on same rank (local-only detected)
// =================================================================

TEST_CASE("PermutationTransfer::fromPartition all-local")
{
    auto mpi = worldMPI();
    const DNDS::index nLocal = 8;

    ArrayAdjacencyPair<1> arr;
    arr.InitPair("test_arr", mpi);
    arr.father->Resize(nLocal);
    arr.father->createGlobalMapping();

    DNDS::index myOffset = (*arr.father->pLGlobalMapping)(mpi.rank, 0);
    for (DNDS::index i = 0; i < nLocal; i++)
        arr(i, 0) = myOffset + i;

    // Partition: all entities stay on current rank
    std::vector<MPI_int> partition(nLocal, mpi.rank);

    auto pt = PermutationTransfer::fromPartition(partition, arr.father->pLGlobalMapping, mpi);

    CHECK(pt.isLocalOnly);
    CHECK(pt.size() == nLocal);

    // New globals should be contiguous starting at newGlobalOffsets[mpi.rank]
    DNDS::index newOffset = pt.newGlobalOffsets[mpi.rank];
    for (DNDS::index i = 0; i < nLocal; i++)
        CHECK(pt.newGlobalIndices[i] == newOffset + i);

    // Transfer should be identity (since partition = all-self, ordering preserved)
    pt.transferRows(arr, mpi);
    for (DNDS::index i = 0; i < nLocal; i++)
        CHECK(arr(i, 0) == myOffset + i);
}

// =================================================================
// Test: fromPartition — round-robin redistribution
// =================================================================

TEST_CASE("PermutationTransfer::fromPartition round-robin")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return; // skip on 1 rank

    const DNDS::index nLocal = 6;

    ArrayAdjacencyPair<1> arr;
    arr.InitPair("test_arr", mpi);
    arr.father->Resize(nLocal);
    arr.father->createGlobalMapping();

    DNDS::index myOffset = (*arr.father->pLGlobalMapping)(mpi.rank, 0);
    for (DNDS::index i = 0; i < nLocal; i++)
        arr(i, 0) = myOffset + i; // value = old global

    // Round-robin: entity i goes to rank (i % nRanks)
    std::vector<MPI_int> partition(nLocal);
    for (DNDS::index i = 0; i < nLocal; i++)
        partition[i] = static_cast<MPI_int>(i % mpi.size);

    auto pt = PermutationTransfer::fromPartition(partition, arr.father->pLGlobalMapping, mpi);

    CHECK_FALSE(pt.isLocalOnly);

    // Transfer rows
    pt.transferRows(arr, mpi);

    // After transfer, this rank should have received entities from all ranks
    // whose slot i satisfies (i % nRanks == mpi.rank).
    // Count expected: each rank sends nLocal/nRanks entities to this rank
    // (approximately — depends on nLocal and nRanks).
    DNDS::index expectedCount = 0;
    for (DNDS::index i = 0; i < nLocal * mpi.size; i++)
        if (i % mpi.size == mpi.rank)
            expectedCount++; // but we only count from all ranks

    // More precise: each rank sends (number of slots where i%size == mpi.rank) entities
    DNDS::index myReceiveCount = 0;
    for (DNDS::index i = 0; i < nLocal; i++)
        if (partition[i] == mpi.rank)
            myReceiveCount++; // from self
    // From other ranks: they also have nLocal entities and send some here
    // Total receive = sum over all ranks of (their slots targeting me)
    // Each rank has nLocal entities; slot i of rank r targets rank (i % size).
    // So from rank r, I receive count of {i : i%size == mpi.rank, 0 <= i < nLocal}

    DNDS::index totalReceive = 0;
    for (int r = 0; r < mpi.size; r++)
    {
        for (DNDS::index i = 0; i < nLocal; i++)
            if (i % mpi.size == mpi.rank)
                totalReceive++;
    }

    CHECK(arr.father->Size() == totalReceive);

    // Verify all received values are valid old globals (in range [0, nLocal*nRanks))
    DNDS::index globalTotal = nLocal * mpi.size;
    for (DNDS::index i = 0; i < arr.father->Size(); i++)
    {
        DNDS::index val = arr(i, 0);
        CHECK(val >= 0);
        CHECK(val < globalTotal);
    }
}

// =================================================================
// Test: buildLookup — resolve old globals to new globals
// =================================================================

TEST_CASE("PermutationTransfer::buildLookup resolve")
{
    auto mpi = worldMPI();
    const DNDS::index nLocal = 4;

    ArrayAdjacencyPair<1> arr;
    arr.InitPair("test_arr", mpi);
    arr.father->Resize(nLocal);
    arr.father->createGlobalMapping();

    DNDS::index myOffset = (*arr.father->pLGlobalMapping)(mpi.rank, 0);

    // All-local partition (identity-like)
    std::vector<MPI_int> partition(nLocal, mpi.rank);
    auto pt = PermutationTransfer::fromPartition(partition, arr.father->pLGlobalMapping, mpi);

    // Pull set: request globals from other ranks (first 2 from each neighbor)
    std::vector<DNDS::index> pullSet;
    for (int r = 0; r < mpi.size; r++)
    {
        if (r == mpi.rank)
            continue;
        DNDS::index rOffset = (*arr.father->pLGlobalMapping)(r, 0);
        for (DNDS::index i = 0; i < std::min(nLocal, DNDS::index{2}); i++)
            pullSet.push_back(rOffset + i);
    }
    std::sort(pullSet.begin(), pullSet.end());

    auto lookup = pt.buildLookup(pullSet, mpi);

    // Resolve own globals: should map to new globals
    for (DNDS::index i = 0; i < nLocal; i++)
    {
        DNDS::index oldGlobal = myOffset + i;
        DNDS::index newGlobal = lookup.resolve(oldGlobal);
        CHECK(newGlobal == pt.newGlobalIndices[i]);
    }

    // Resolve pulled globals from other ranks
    for (auto oldGlobal : pullSet)
    {
        DNDS::index newGlobal = lookup.resolve(oldGlobal);
        // The new global should be valid (in range [0, total))
        CHECK(newGlobal >= 0);
        CHECK(newGlobal < pt.newGlobalOffsets.back());
    }

    // UnInitIndex passthrough
    CHECK(lookup.resolve(UnInitIndex) == UnInitIndex);
}

// =================================================================
// Test: fromPartition + buildLookup — distributed with cross-rank resolve
// =================================================================

TEST_CASE("PermutationTransfer distributed lookup cross-rank")
{
    auto mpi = worldMPI();
    if (mpi.size < 2)
        return;

    const DNDS::index nLocal = 4;

    ArrayAdjacencyPair<1> arr;
    arr.InitPair("test_arr", mpi);
    arr.father->Resize(nLocal);
    arr.father->createGlobalMapping();

    DNDS::index myOffset = (*arr.father->pLGlobalMapping)(mpi.rank, 0);

    // Send all entities to the next rank (ring shift)
    MPI_int targetRank = (mpi.rank + 1) % mpi.size;
    std::vector<MPI_int> partition(nLocal, targetRank);

    auto pt = PermutationTransfer::fromPartition(partition, arr.father->pLGlobalMapping, mpi);
    CHECK_FALSE(pt.isLocalOnly);

    // Build lookup: pull the previous rank's old globals
    MPI_int sourceRank = (mpi.rank - 1 + mpi.size) % mpi.size;
    DNDS::index sourceOffset = (*arr.father->pLGlobalMapping)(sourceRank, 0);
    std::vector<DNDS::index> pullSet;
    for (DNDS::index i = 0; i < nLocal; i++)
        pullSet.push_back(sourceOffset + i);
    std::sort(pullSet.begin(), pullSet.end());

    auto lookup = pt.buildLookup(pullSet, mpi);

    // Verify: my old globals should resolve to new globals on targetRank
    for (DNDS::index i = 0; i < nLocal; i++)
    {
        DNDS::index oldGlobal = myOffset + i;
        DNDS::index newGlobal = lookup.resolve(oldGlobal);
        // New global should be in [newGlobalOffsets[target], newGlobalOffsets[target+1])
        CHECK(newGlobal >= pt.newGlobalOffsets[targetRank]);
        CHECK(newGlobal < pt.newGlobalOffsets[targetRank + 1]);
    }

    // Verify: previous rank's old globals should also resolve
    for (auto oldGlobal : pullSet)
    {
        DNDS::index newGlobal = lookup.resolve(oldGlobal);
        // sourceRank sent to (sourceRank+1)%size == mpi.rank
        CHECK(newGlobal >= pt.newGlobalOffsets[mpi.rank]);
        CHECK(newGlobal < pt.newGlobalOffsets[mpi.rank + 1]);
    }
}

// =================================================================
// Test: transferRows with CSR (variable-row-size) array
// =================================================================

TEST_CASE("PermutationTransfer::transferRows CSR local permutation")
{
    auto mpi = worldMPI();
    const DNDS::index nLocal = 6;

    // Create a CSR array where row i has (i+1) entries
    ArrayAdjacencyPair<NonUniformSize> arr;
    arr.InitPair("test_csr", mpi);
    arr.father->Resize(nLocal);
    for (DNDS::index i = 0; i < nLocal; i++)
        arr.father->ResizeRow(i, static_cast<rowsize>(i + 1));
    arr.father->Compress();
    arr.father->createGlobalMapping();

    // Fill: row i, entry j = i * 100 + j
    for (DNDS::index i = 0; i < nLocal; i++)
        for (rowsize j = 0; j < arr.father->RowSize(i); j++)
            arr(i, j) = i * 100 + j;

    // Reverse permutation
    std::vector<DNDS::index> old2new(nLocal);
    for (DNDS::index i = 0; i < nLocal; i++)
        old2new[i] = nLocal - 1 - i;

    auto pt = PermutationTransfer::fromLocalPermutation(old2new, arr.father->pLGlobalMapping, mpi);
    pt.transferRows(arr, mpi);

    // Verify: new slot j should contain old slot (N-1-j)'s data
    for (DNDS::index j = 0; j < nLocal; j++)
    {
        DNDS::index oldSlot = nLocal - 1 - j;
        rowsize expectedRowSize = static_cast<rowsize>(oldSlot + 1);
        CHECK(arr.father->RowSize(j) == expectedRowSize);
        for (rowsize k = 0; k < expectedRowSize; k++)
            CHECK(arr(j, k) == oldSlot * 100 + k);
    }
}
