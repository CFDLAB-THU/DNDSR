/**
 * @file test_ArrayDerived.cpp
 * @brief Comprehensive doctest-based unit tests for DNDS ArrayDerived types:
 *        ArrayAdjacency, ArrayEigenVector, ArrayEigenMatrix,
 *        ArrayEigenMatrixBatch, and ArrayEigenUniMatrixBatch.
 *
 * Run under mpirun with 1, 2, and 4 ranks.  Tests construction, resize,
 * compress, element access, ghost communication via ArrayTransformer, and
 * clone independence for each derived array type.
 *
 * @see @ref dnds_unit_tests for the full test-suite overview.
 * @test ArrayAdjacency (basics, ghost comm, clone, fixed-size),
 *       ArrayEigenVector (static, dynamic, ghost comm),
 *       ArrayEigenMatrix (static, dynamic, NonUniform, ghost comm),
 *       ArrayEigenMatrixBatch (basics, ghost comm),
 *       ArrayEigenUniMatrixBatch (static, dynamic).
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/ArrayPair.hpp"
#include "DNDS/ArrayTransformer.hpp"
#include <memory>
#include <numeric>
#include <vector>

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

/// Build a vector of global indices: first `nPer` elements from each
/// non-local rank, for use as ghost pull indices.
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

// ===========================================================================
// ArrayAdjacency basics
// ===========================================================================
TEST_CASE("ArrayAdjacency basics")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 20;

    ArrayAdjacency<NonUniformSize> adj(mpi);
    adj.Resize(N);

    // Assign each row a varying size and fill with known values
    for (DNDS::index i = 0; i < N; i++)
        adj.ResizeRow(i, static_cast<DNDS::rowsize>(i % 3 + 1));

    for (DNDS::index i = 0; i < N; i++)
    {
        DNDS::index *rp = adj.rowPtr(i);
        for (DNDS::rowsize j = 0; j < adj.RowSize(i); j++)
            rp[j] = i * 100 + j;
    }

    adj.Compress();

    SUBCASE("verify all values after Compress")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            CHECK(adj.RowSize(i) == static_cast<DNDS::rowsize>(i % 3 + 1));
            for (DNDS::rowsize j = 0; j < adj.RowSize(i); j++)
            {
                DNDS::index expected = i * 100 + j;
                CHECK(adj.rowPtr(i)[j] == expected);
            }
        }
    }

    SUBCASE("operator[] returns AdjacencyRow with correct data")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            auto row = adj[i];
            CHECK(row.size() == static_cast<DNDS::rowsize>(i % 3 + 1));
            for (DNDS::rowsize j = 0; j < row.size(); j++)
                CHECK(row[j] == i * 100 + j);
        }
    }

    SUBCASE("rowPtr returns valid pointer")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            DNDS::index *ptr = adj.rowPtr(i);
            REQUIRE(ptr != nullptr);
            CHECK(ptr[0] == i * 100);
        }
    }
}

// ===========================================================================
// ArrayAdjacency ghost communication
// ===========================================================================
TEST_CASE("ArrayAdjacency ghost communication")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = 20;
    constexpr DNDS::index nGhostPer = 3;

    auto father = std::make_shared<ArrayAdjacency<NonUniformSize>>(mpi);
    father->Resize(nLocal, [](DNDS::index i) -> DNDS::rowsize
                   { return static_cast<DNDS::rowsize>(i % 3 + 1); });

    father->createGlobalMapping();
    DNDS::index gOff = (*father->pLGlobalMapping)(mpi.rank, 0);

    // Fill: each element encodes its global index
    for (DNDS::index i = 0; i < nLocal; i++)
        for (DNDS::rowsize j = 0; j < father->RowSize(i); j++)
            father->operator()(i, j) = (gOff + i) * 100 + j;

    auto son = std::make_shared<ArrayAdjacency<NonUniformSize>>(mpi);
    ArrayTransformer<DNDS::index, NonUniformSize> trans;
    trans.setFatherSon(father, son);
    trans.createFatherGlobalMapping();

    auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, nGhostPer);
    trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
    trans.createMPITypes();
    trans.pullOnce();

    CHECK(son->Size() == static_cast<DNDS::index>(pullIdx.size()));

    // Verify ghost adjacency data
    for (DNDS::index g = 0; g < son->Size(); g++)
    {
        DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
        MPI_int srcRank = -1;
        DNDS::index srcLoc = -1;
        bool found = trans.pLGlobalMapping->search(ghostGlobal, srcRank, srcLoc);
        CHECK(found);

        DNDS::rowsize expectedRowSize = static_cast<DNDS::rowsize>(srcLoc % 3 + 1);
        CHECK(son->RowSize(g) == expectedRowSize);

        for (DNDS::rowsize j = 0; j < son->RowSize(g); j++)
            CHECK(son->operator()(g, j) == ghostGlobal * 100 + j);
    }
}

// ===========================================================================
// ArrayEigenVector basics
// ===========================================================================
TEST_CASE("ArrayEigenVector basics")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 30;
    constexpr DNDS::rowsize vecSize = 5;

    ArrayEigenVector<DynamicSize> vec(mpi);
    vec.Resize(N, vecSize);

    CHECK(vec.Size() == N);
    CHECK(vec.RowSize() == vecSize);

    // Fill each row with a known pattern: v(j) = i * 10 + j
    for (DNDS::index i = 0; i < N; i++)
    {
        auto v = vec[i];
        for (DNDS::rowsize j = 0; j < vecSize; j++)
            v(j) = static_cast<DNDS::real>(i * 10 + j);
    }

    SUBCASE("operator[] returns Eigen::Map with correct values")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            auto v = vec[i];
            CHECK(v.size() == vecSize);
            for (DNDS::rowsize j = 0; j < vecSize; j++)
                CHECK(v(j) == doctest::Approx(static_cast<DNDS::real>(i * 10 + j)));
        }
    }

    SUBCASE("Size and RowSize")
    {
        CHECK(vec.Size() == N);
        CHECK(vec.RowSize() == vecSize);
    }
}

// ===========================================================================
// ArrayEigenVector ghost communication
// ===========================================================================
TEST_CASE("ArrayEigenVector ghost communication")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = 25;
    constexpr DNDS::rowsize vecSize = 4;
    constexpr DNDS::index nGhostPer = 5;

    auto father = std::make_shared<ArrayEigenVector<DynamicSize>>(mpi);
    father->Resize(nLocal, vecSize);
    father->createGlobalMapping();
    DNDS::index gOff = (*father->pLGlobalMapping)(mpi.rank, 0);

    // Fill with rank-specific vectors: v(j) = (gOff + i)*10 + j
    for (DNDS::index i = 0; i < nLocal; i++)
    {
        auto v = father->operator[](i);
        for (DNDS::rowsize j = 0; j < vecSize; j++)
            v(j) = static_cast<DNDS::real>((gOff + i) * 10 + j);
    }

    auto son = std::make_shared<ArrayEigenVector<DynamicSize>>(mpi);
    ArrayTransformer<DNDS::real, DynamicSize> trans;
    trans.setFatherSon(father, son);
    trans.createFatherGlobalMapping();

    auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, nGhostPer);
    trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
    trans.createMPITypes();
    trans.pullOnce();

    CHECK(son->Size() == static_cast<DNDS::index>(pullIdx.size()));

    for (DNDS::index g = 0; g < son->Size(); g++)
    {
        DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
        auto v = son->operator[](g);
        for (DNDS::rowsize j = 0; j < vecSize; j++)
            CHECK(v(j) == doctest::Approx(static_cast<DNDS::real>(ghostGlobal * 10 + j)));
    }
}

// ===========================================================================
// ArrayEigenMatrix basics (static sizes)
// ===========================================================================
TEST_CASE("ArrayEigenMatrix basics")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 20;

    ArrayEigenMatrix<3, 4> mat(mpi);
    mat.Resize(N, 3, 4);

    CHECK(mat.Size() == N);
    CHECK(mat.MatRowSize() == 3);
    CHECK(mat.MatColSize() == 4);

    // Fill each matrix: Identity-like + scaled offset
    for (DNDS::index i = 0; i < N; i++)
    {
        auto m = mat[i];
        m.setZero();
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
                m(r, c) = static_cast<DNDS::real>(i * 100 + r * 10 + c);
    }

    SUBCASE("verify via operator[] Eigen::Map")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            auto m = mat[i];
            CHECK(m.rows() == 3);
            CHECK(m.cols() == 4);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++)
                    CHECK(m(r, c) == doctest::Approx(static_cast<DNDS::real>(i * 100 + r * 10 + c)));
        }
    }

    SUBCASE("MatRowSize and MatColSize")
    {
        CHECK(mat.MatRowSize() == 3);
        CHECK(mat.MatColSize() == 4);
    }
}

// ===========================================================================
// ArrayEigenMatrix dynamic sizes
// ===========================================================================
TEST_CASE("ArrayEigenMatrix dynamic sizes")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 15;
    constexpr DNDS::rowsize dynRows = 4;
    constexpr DNDS::rowsize dynCols = 3;

    ArrayEigenMatrix<DynamicSize, DynamicSize> mat(mpi);
    mat.Resize(N, dynRows, dynCols);

    CHECK(mat.Size() == N);
    CHECK(mat.MatRowSize() == dynRows);
    CHECK(mat.MatColSize() == dynCols);

    // Fill and verify
    for (DNDS::index i = 0; i < N; i++)
    {
        auto m = mat[i];
        for (int r = 0; r < dynRows; r++)
            for (int c = 0; c < dynCols; c++)
                m(r, c) = static_cast<DNDS::real>(i * 1000 + r * 10 + c);
    }

    for (DNDS::index i = 0; i < N; i++)
    {
        auto m = mat[i];
        CHECK(m.rows() == dynRows);
        CHECK(m.cols() == dynCols);
        for (int r = 0; r < dynRows; r++)
            for (int c = 0; c < dynCols; c++)
                CHECK(m(r, c) == doctest::Approx(static_cast<DNDS::real>(i * 1000 + r * 10 + c)));
    }
}

// ===========================================================================
// ArrayEigenMatrix ghost communication
// ===========================================================================
TEST_CASE("ArrayEigenMatrix ghost communication")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = 20;
    constexpr DNDS::rowsize matR = 3;
    constexpr DNDS::rowsize matC = 4;
    constexpr DNDS::index nGhostPer = 4;

    // Use ArrayPair which properly manages father/son/transformer
    ArrayEigenMatrixPair<matR, matC> pair;
    pair.InitPair("matGhost::pair", mpi);
    pair.father->Resize(nLocal, matR, matC);
    pair.father->createGlobalMapping();
    DNDS::index gOff = (*pair.father->pLGlobalMapping)(mpi.rank, 0);

    for (DNDS::index i = 0; i < nLocal; i++)
    {
        auto m = pair.father->operator[](i);
        for (int r = 0; r < matR; r++)
            for (int c = 0; c < matC; c++)
                m(r, c) = static_cast<DNDS::real>((gOff + i) * 1000 + r * 10 + c);
    }

    pair.TransAttach();
    pair.trans.createFatherGlobalMapping();

    auto pullIdx = pullFirstNFromOthers(mpi, *pair.trans.pLGlobalMapping, nGhostPer);
    pair.trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
    pair.trans.createMPITypes();
    pair.trans.pullOnce();

    CHECK(pair.son->Size() == static_cast<DNDS::index>(pullIdx.size()));

    for (DNDS::index g = 0; g < pair.son->Size(); g++)
    {
        DNDS::index ghostGlobal = pair.trans.pLGhostMapping->ghostIndex[g];
        auto m = pair.son->operator[](g);
        for (int r = 0; r < matR; r++)
            for (int c = 0; c < matC; c++)
                CHECK(m(r, c) == doctest::Approx(
                                     static_cast<DNDS::real>(ghostGlobal * 1000 + r * 10 + c)));
    }
}

// ===========================================================================
// ArrayEigenMatrixBatch basics
// ===========================================================================
TEST_CASE("ArrayEigenMatrixBatch basics")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 10;

    ArrayEigenMatrixBatch batch(mpi);
    batch.Resize(N);

    // For each row, create a batch of (i%3 + 1) matrices of varying sizes
    for (DNDS::index i = 0; i < N; i++)
    {
        int nMats = static_cast<int>(i % 3 + 1);
        std::vector<Eigen::MatrixXd> matrices;
        matrices.reserve(nMats);
        for (int k = 0; k < nMats; k++)
        {
            int rows = k + 1;
            int cols = k + 2;
            Eigen::MatrixXd m(rows, cols);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    m(r, c) = static_cast<DNDS::real>(i * 10000 + k * 100 + r * 10 + c);
            matrices.push_back(m);
        }
        batch.InitializeWriteRow(i, matrices);
    }

    batch.Compress();

    SUBCASE("verify BatchSize")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            auto expectedBatchSize = static_cast<DNDS::index>(i % 3 + 1);
            CHECK(batch.BatchSize(i) == static_cast<DNDS::index>(expectedBatchSize));
        }
    }

    SUBCASE("read back via operator()(i, j)")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            int nMats = static_cast<int>(i % 3 + 1);
            for (int k = 0; k < nMats; k++)
            {
                auto m = batch(i, k);
                int rows = k + 1;
                int cols = k + 2;
                CHECK(m.rows() == rows);
                CHECK(m.cols() == cols);
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        CHECK(m(r, c) == doctest::Approx(
                                             static_cast<DNDS::real>(i * 10000 + k * 100 + r * 10 + c)));
            }
        }
    }
}

// ===========================================================================
// ArrayEigenUniMatrixBatch basics
// ===========================================================================
TEST_CASE("ArrayEigenUniMatrixBatch basics")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 15;
    constexpr int matRows = 3;
    constexpr int matCols = 4;

    ArrayEigenUniMatrixBatch<Eigen::Dynamic, Eigen::Dynamic> ubatch(mpi);
    ubatch.Resize(N, matRows, matCols);

    CHECK(ubatch.Size() == N);
    CHECK(ubatch.Rows() == matRows);
    CHECK(ubatch.Cols() == matCols);
    CHECK(ubatch.MSize() == matRows * matCols);

    // Assign each row a batch size of (i % 4 + 1)
    for (DNDS::index i = 0; i < N; i++)
        ubatch.ResizeBatch(i, static_cast<DNDS::rowsize>(i % 4 + 1));

    // Fill matrices
    for (DNDS::index i = 0; i < N; i++)
    {
        DNDS::rowsize bs = ubatch.BatchSize(i);
        for (DNDS::rowsize j = 0; j < bs; j++)
        {
            auto m = ubatch(i, j);
            for (int r = 0; r < matRows; r++)
                for (int c = 0; c < matCols; c++)
                    m(r, c) = static_cast<DNDS::real>(i * 10000 + j * 100 + r * 10 + c);
        }
    }

    ubatch.Compress();

    SUBCASE("verify BatchSize after Compress")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            DNDS::rowsize expected = static_cast<DNDS::rowsize>(i % 4 + 1);
            CHECK(ubatch.BatchSize(i) == expected);
        }
    }

    SUBCASE("verify matrix values after Compress")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            DNDS::rowsize bs = ubatch.BatchSize(i);
            for (DNDS::rowsize j = 0; j < bs; j++)
            {
                auto m = ubatch(i, j);
                CHECK(m.rows() == matRows);
                CHECK(m.cols() == matCols);
                for (int r = 0; r < matRows; r++)
                    for (int c = 0; c < matCols; c++)
                        CHECK(m(r, c) == doctest::Approx(
                                             static_cast<DNDS::real>(i * 10000 + j * 100 + r * 10 + c)));
            }
        }
    }
}

// ===========================================================================
// ArrayEigenUniMatrixBatch static sizes
// ===========================================================================
TEST_CASE("ArrayEigenUniMatrixBatch static sizes")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 10;

    ArrayEigenUniMatrixBatch<2, 3> ubatch(mpi);
    ubatch.Resize(N);

    CHECK(ubatch.Rows() == 2);
    CHECK(ubatch.Cols() == 3);
    CHECK(ubatch.MSize() == 6);

    for (DNDS::index i = 0; i < N; i++)
        ubatch.ResizeBatch(i, static_cast<DNDS::rowsize>(i % 3 + 1));

    for (DNDS::index i = 0; i < N; i++)
    {
        DNDS::rowsize bs = ubatch.BatchSize(i);
        for (DNDS::rowsize j = 0; j < bs; j++)
        {
            auto m = ubatch(i, j);
            m.setConstant(static_cast<DNDS::real>(i * 100 + j));
        }
    }

    ubatch.Compress();

    for (DNDS::index i = 0; i < N; i++)
    {
        DNDS::rowsize bs = ubatch.BatchSize(i);
        CHECK(bs == static_cast<DNDS::rowsize>(i % 3 + 1));
        for (DNDS::rowsize j = 0; j < bs; j++)
        {
            auto m = ubatch(i, j);
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 3; c++)
                    CHECK(m(r, c) == doctest::Approx(static_cast<DNDS::real>(i * 100 + j)));
        }
    }
}

// ===========================================================================
// ArrayAdjacency clone
// ===========================================================================
TEST_CASE("ArrayAdjacency clone")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 10;

    ArrayAdjacency<NonUniformSize> adj(mpi);
    adj.Resize(N, [](DNDS::index i) -> DNDS::rowsize
               { return static_cast<DNDS::rowsize>(i % 3 + 1); });

    for (DNDS::index i = 0; i < N; i++)
        for (DNDS::rowsize j = 0; j < adj.RowSize(i); j++)
            adj.rowPtr(i)[j] = i * 100 + j;

    ArrayAdjacency<NonUniformSize> adj2(mpi);
    adj2.clone(adj);

    for (DNDS::index i = 0; i < N; i++)
    {
        CHECK(adj2.RowSize(i) == adj.RowSize(i));
        for (DNDS::rowsize j = 0; j < adj.RowSize(i); j++)
            CHECK(adj2.rowPtr(i)[j] == adj.rowPtr(i)[j]);
    }

    // Modifying original does not affect clone
    adj.rowPtr(0)[0] = -999;
    CHECK(adj2.rowPtr(0)[0] != -999);
}

// ===========================================================================
// ArrayEigenMatrix NonUniformSize rows
// ===========================================================================
TEST_CASE("ArrayEigenMatrix NonUniformSize rows")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 12;

    ArrayEigenMatrix<NonUniformSize, NonUniformSize> mat(mpi);
    // Initial size with some default row/col dimensions
    mat.Resize(N, 2, 3);

    // Resize individual matrices to different dimensions
    for (DNDS::index i = 0; i < N; i++)
    {
        DNDS::rowsize nr = static_cast<DNDS::rowsize>(i % 3 + 1);
        DNDS::rowsize nc = static_cast<DNDS::rowsize>(i % 2 + 2);
        mat.ResizeMat(i, nr, nc);
    }

    // Fill
    for (DNDS::index i = 0; i < N; i++)
    {
        auto m = mat[i];
        for (int r = 0; r < m.rows(); r++)
            for (int c = 0; c < m.cols(); c++)
                m(r, c) = static_cast<DNDS::real>(i * 1000 + r * 10 + c);
    }

    mat.Compress();

    // Verify
    for (DNDS::index i = 0; i < N; i++)
    {
        DNDS::rowsize nr = static_cast<DNDS::rowsize>(i % 3 + 1);
        DNDS::rowsize nc = static_cast<DNDS::rowsize>(i % 2 + 2);
        CHECK(mat.MatRowSize(i) == nr);
        CHECK(mat.MatColSize(i) == nc);

        auto m = mat[i];
        for (int r = 0; r < nr; r++)
            for (int c = 0; c < nc; c++)
                CHECK(m(r, c) == doctest::Approx(static_cast<DNDS::real>(i * 1000 + r * 10 + c)));
    }
}

// ===========================================================================
// ArrayEigenVector with static size
// ===========================================================================
TEST_CASE("ArrayEigenVector static size")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 20;

    ArrayEigenVector<5> vec(mpi);
    vec.Resize(N);

    CHECK(vec.Size() == N);
    CHECK(vec.RowSize() == 5);

    for (DNDS::index i = 0; i < N; i++)
    {
        auto v = vec[i];
        v.setLinSpaced(5, static_cast<DNDS::real>(i), static_cast<DNDS::real>(i + 4));
    }

    for (DNDS::index i = 0; i < N; i++)
    {
        auto v = vec[i];
        CHECK(v.size() == 5);
        // setLinSpaced(5, i, i+4) produces {i, i+1, i+2, i+3, i+4}
        for (int j = 0; j < 5; j++)
            CHECK(v(j) == doctest::Approx(static_cast<DNDS::real>(i + j)));
    }
}

// ===========================================================================
// ArrayEigenMatrixBatch ghost communication
// ===========================================================================
TEST_CASE("ArrayEigenMatrixBatch ghost communication")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index nLocal = 10;
    constexpr DNDS::index nGhostPer = 2;

    // Use ArrayPair for proper ghost communication
    ArrayEigenMatrixBatchPair pair;
    pair.InitPair("batchGhost::pair", mpi);
    pair.father->Resize(nLocal);

    // Uniform batches: each row has 2 matrices of size 2x2
    for (DNDS::index i = 0; i < nLocal; i++)
    {
        std::vector<Eigen::MatrixXd> matrices;
        for (int k = 0; k < 2; k++)
        {
            Eigen::MatrixXd m(2, 2);
            m.setZero();
            matrices.push_back(m);
        }
        pair.father->InitializeWriteRow(i, matrices);
    }
    pair.father->Compress();
    pair.father->createGlobalMapping();

    DNDS::index gOff = (*pair.father->pLGlobalMapping)(mpi.rank, 0);

    // Fill with global-index-based values
    for (DNDS::index i = 0; i < nLocal; i++)
    {
        for (int k = 0; k < 2; k++)
        {
            auto m = pair.father->operator()(i, k);
            m.setConstant(static_cast<DNDS::real>((gOff + i) * 1000 + k));
        }
    }

    pair.TransAttach();
    pair.trans.createFatherGlobalMapping();

    auto pullIdx = pullFirstNFromOthers(mpi, *pair.trans.pLGlobalMapping, nGhostPer);
    pair.trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
    pair.trans.createMPITypes();
    pair.trans.pullOnce();

    CHECK(pair.son->Size() == static_cast<DNDS::index>(pullIdx.size()));

    // Verify ghost batch data
    for (DNDS::index g = 0; g < pair.son->Size(); g++)
    {
        DNDS::index ghostGlobal = pair.trans.pLGhostMapping->ghostIndex[g];
        CHECK(pair.son->BatchSize(g) == 2);
        for (int k = 0; k < 2; k++)
        {
            auto m = pair.son->operator()(g, k);
            CHECK(m.rows() == 2);
            CHECK(m.cols() == 2);
            DNDS::real expected = static_cast<DNDS::real>(ghostGlobal * 1000 + k);
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 2; c++)
                    CHECK(m(r, c) == doctest::Approx(expected));
        }
    }
}

// ===========================================================================
// ArrayAdjacency fixed-size variant
// ===========================================================================
TEST_CASE("ArrayAdjacency fixed-size variant")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 25;

    // ArrayAdjacency<3> has TABLE_StaticFixed layout with 3 columns per row
    ArrayAdjacency<3> adj(mpi);
    adj.Resize(N);

    CHECK(adj.Size() == N);
    CHECK(adj.RowSize() == 3);

    for (DNDS::index i = 0; i < N; i++)
    {
        auto row = adj[i];
        for (DNDS::rowsize j = 0; j < 3; j++)
            row[j] = i * 10 + j;
    }

    for (DNDS::index i = 0; i < N; i++)
    {
        auto row = adj[i];
        CHECK(row.size() == 3);
        for (DNDS::rowsize j = 0; j < 3; j++)
            CHECK(row[j] == i * 10 + j);
    }
}

// ===========================================================================
// Parametric: ArrayAdjacency across row sizes
// ===========================================================================
// Axes: RS = {2, 5, 8, NonUniformSize}

template <DNDS::rowsize RS>
struct AdjTag
{
    static constexpr DNDS::rowsize rs = RS;
};

TYPE_TO_STRING(AdjTag<2>);
TYPE_TO_STRING(AdjTag<5>);
TYPE_TO_STRING(AdjTag<8>);
TYPE_TO_STRING(AdjTag<NonUniformSize>);

TEST_CASE_TEMPLATE("ArrayAdjacency parametric", Tag,
                    AdjTag<2>, AdjTag<5>, AdjTag<8>, AdjTag<NonUniformSize>)
{
    constexpr DNDS::rowsize RS = Tag::rs;
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 25;

    ArrayAdjacency<RS> adj(mpi);

    if constexpr (RS == NonUniformSize)
    {
        adj.Resize(N);
        for (DNDS::index i = 0; i < N; i++)
            adj.ResizeRow(i, static_cast<DNDS::rowsize>(i % 5 + 1));
        adj.Compress();
    }
    else
    {
        adj.Resize(N);
    }

    // Fill
    for (DNDS::index i = 0; i < N; i++)
        for (DNDS::rowsize j = 0; j < adj.RowSize(i); j++)
            adj(i, j) = i * 100 + j;

    SUBCASE("data round-trip")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            if constexpr (RS == NonUniformSize)
                CHECK(adj.RowSize(i) == static_cast<DNDS::rowsize>(i % 5 + 1));
            else
                CHECK(adj.RowSize(i) == RS);
            for (DNDS::rowsize j = 0; j < adj.RowSize(i); j++)
                CHECK(adj(i, j) == i * 100 + j);
        }
    }

    SUBCASE("ghost pull")
    {
        if (mpi.size < 2)
            return;

        adj.createGlobalMapping();
        DNDS::index gOff = (*adj.pLGlobalMapping)(mpi.rank, 0);

        // Re-fill with global-index encoding
        for (DNDS::index i = 0; i < N; i++)
            for (DNDS::rowsize j = 0; j < adj.RowSize(i); j++)
                adj(i, j) = (gOff + i) * 100 + j;

        auto father = std::make_shared<ArrayAdjacency<RS>>(adj);
        auto son = std::make_shared<ArrayAdjacency<RS>>(mpi);
        using TTransformer = typename ArrayTransformerType<ArrayAdjacency<RS>>::Type;
        TTransformer trans;
        trans.setFatherSon(father, son);
        trans.createFatherGlobalMapping();
        auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, 3);
        trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
        trans.createMPITypes();
        trans.pullOnce();

        CHECK(son->Size() == static_cast<DNDS::index>(pullIdx.size()));
        for (DNDS::index g = 0; g < son->Size(); g++)
        {
            DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
            for (DNDS::rowsize j = 0; j < son->RowSize(g); j++)
                CHECK(son->operator()(g, j) == ghostGlobal * 100 + j);
        }
    }
}

// ===========================================================================
// Parametric: ArrayEigenVector across vector sizes
// ===========================================================================
// Axes: vec_size = {1, 3, 7, DynamicSize}

template <DNDS::rowsize VS>
struct VecTag
{
    static constexpr DNDS::rowsize vs = VS;
};

TYPE_TO_STRING(VecTag<1>);
TYPE_TO_STRING(VecTag<3>);
TYPE_TO_STRING(VecTag<7>);
TYPE_TO_STRING(VecTag<DynamicSize>);

TEST_CASE_TEMPLATE("ArrayEigenVector parametric", Tag,
                    VecTag<1>, VecTag<3>, VecTag<7>, VecTag<DynamicSize>)
{
    constexpr DNDS::rowsize VS = Tag::vs;
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 30;
    constexpr DNDS::rowsize actualSize = (VS == DynamicSize) ? 5 : VS;

    ArrayEigenVector<VS> vec(mpi);
    if constexpr (VS == DynamicSize)
        vec.Resize(N, actualSize);
    else
        vec.Resize(N);

    CHECK(vec.Size() == N);
    CHECK(vec.RowSize() == actualSize);

    // Fill
    for (DNDS::index i = 0; i < N; i++)
    {
        auto v = vec[i];
        for (DNDS::rowsize j = 0; j < actualSize; j++)
            v(j) = static_cast<DNDS::real>(i * 100 + j);
    }

    SUBCASE("data round-trip")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            auto v = vec[i];
            CHECK(v.size() == actualSize);
            for (DNDS::rowsize j = 0; j < actualSize; j++)
                CHECK(v(j) == doctest::Approx(static_cast<DNDS::real>(i * 100 + j)));
        }
    }

    SUBCASE("ghost pull")
    {
        if (mpi.size < 2)
            return;

        vec.createGlobalMapping();
        DNDS::index gOff = (*vec.pLGlobalMapping)(mpi.rank, 0);
        for (DNDS::index i = 0; i < N; i++)
        {
            auto v = vec[i];
            for (DNDS::rowsize j = 0; j < actualSize; j++)
                v(j) = static_cast<DNDS::real>((gOff + i) * 100 + j);
        }

        auto father = std::make_shared<ArrayEigenVector<VS>>(vec);
        auto son = std::make_shared<ArrayEigenVector<VS>>(mpi);
        using TTransformer = typename ArrayTransformerType<ArrayEigenVector<VS>>::Type;
        TTransformer trans;
        trans.setFatherSon(father, son);
        trans.createFatherGlobalMapping();
        auto pullIdx = pullFirstNFromOthers(mpi, *trans.pLGlobalMapping, 4);
        trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
        trans.createMPITypes();
        trans.pullOnce();

        CHECK(son->Size() == static_cast<DNDS::index>(pullIdx.size()));
        for (DNDS::index g = 0; g < son->Size(); g++)
        {
            DNDS::index ghostGlobal = trans.pLGhostMapping->ghostIndex[g];
            auto v = son->operator[](g);
            for (DNDS::rowsize j = 0; j < actualSize; j++)
                CHECK(v(j) == doctest::Approx(static_cast<DNDS::real>(ghostGlobal * 100 + j)));
        }
    }
}

// ===========================================================================
// Parametric: ArrayEigenMatrix across matrix dimensions
// ===========================================================================
// Axes: (ni, nj) = {(2,3), (4,5), (1,7), (DynamicSize,DynamicSize)}

template <DNDS::rowsize NI, DNDS::rowsize NJ>
struct MatTag
{
    static constexpr DNDS::rowsize ni = NI;
    static constexpr DNDS::rowsize nj = NJ;
};

TYPE_TO_STRING(MatTag<2, 3>);
TYPE_TO_STRING(MatTag<4, 5>);
TYPE_TO_STRING(MatTag<1, 7>);
TYPE_TO_STRING(MatTag<DynamicSize, DynamicSize>);

TEST_CASE_TEMPLATE("ArrayEigenMatrix parametric", Tag,
                    MatTag<2, 3>, MatTag<4, 5>, MatTag<1, 7>,
                    MatTag<DynamicSize, DynamicSize>)
{
    constexpr DNDS::rowsize NI = Tag::ni;
    constexpr DNDS::rowsize NJ = Tag::nj;
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 20;
    constexpr DNDS::rowsize actualNI = (NI == DynamicSize) ? 3 : NI;
    constexpr DNDS::rowsize actualNJ = (NJ == DynamicSize) ? 4 : NJ;

    ArrayEigenMatrix<NI, NJ> mat(mpi);
    mat.Resize(N, actualNI, actualNJ);

    CHECK(mat.Size() == N);
    CHECK(mat.MatRowSize() == actualNI);
    CHECK(mat.MatColSize() == actualNJ);

    // Fill
    for (DNDS::index i = 0; i < N; i++)
    {
        auto m = mat[i];
        for (int r = 0; r < actualNI; r++)
            for (int c = 0; c < actualNJ; c++)
                m(r, c) = static_cast<DNDS::real>(i * 1000 + r * 10 + c);
    }

    SUBCASE("data round-trip")
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            auto m = mat[i];
            CHECK(m.rows() == actualNI);
            CHECK(m.cols() == actualNJ);
            for (int r = 0; r < actualNI; r++)
                for (int c = 0; c < actualNJ; c++)
                    CHECK(m(r, c) == doctest::Approx(static_cast<DNDS::real>(i * 1000 + r * 10 + c)));
        }
    }

    SUBCASE("ghost pull")
    {
        if (mpi.size < 2)
            return;

        ArrayEigenMatrixPair<NI, NJ> pair;
        pair.InitPair("matParam::pair", mpi);
        pair.father->Resize(N, actualNI, actualNJ);
        pair.son->Resize(0, actualNI, actualNJ);

        pair.father->createGlobalMapping();
        DNDS::index gOff = (*pair.father->pLGlobalMapping)(mpi.rank, 0);
        for (DNDS::index i = 0; i < N; i++)
        {
            auto m = pair.father->operator[](i);
            for (int r = 0; r < actualNI; r++)
                for (int c = 0; c < actualNJ; c++)
                    m(r, c) = static_cast<DNDS::real>((gOff + i) * 1000 + r * 10 + c);
        }

        pair.TransAttach();
        pair.trans.createFatherGlobalMapping();

        auto pullIdx = pullFirstNFromOthers(mpi, *pair.trans.pLGlobalMapping, 3);
        pair.trans.createGhostMapping(std::vector<DNDS::index>(pullIdx));
        pair.trans.createMPITypes();
        pair.trans.pullOnce();

        CHECK(pair.son->Size() == static_cast<DNDS::index>(pullIdx.size()));
        for (DNDS::index g = 0; g < pair.son->Size(); g++)
        {
            DNDS::index ghostGlobal = pair.trans.pLGhostMapping->ghostIndex[g];
            auto m = pair.son->operator[](g);
            for (int r = 0; r < actualNI; r++)
                for (int c = 0; c < actualNJ; c++)
                    CHECK(m(r, c) == doctest::Approx(
                                         static_cast<DNDS::real>(ghostGlobal * 1000 + r * 10 + c)));
        }
    }
}
