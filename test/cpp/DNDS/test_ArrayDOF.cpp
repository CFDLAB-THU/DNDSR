/**
 * @file test_ArrayDOF.cpp
 * @brief Comprehensive doctest-based unit tests for DNDS::ArrayDof
 *        (degree-of-freedom array) vector-space operations.
 *
 * Run under mpirun with 1, 2, and 4 ranks.  All MPI-global reductions
 * (norm2, dot, min, max, sum, componentWiseNorm1) are tested with
 * rank-aware expected values that scale correctly at any process count.
 *
 * @see @ref dnds_unit_tests for the full test-suite overview.
 * @test setConstant, operator+= (scalar/array/matrix), operator-=,
 *       operator*= (scalar/element-wise/matrix), operator/=, addTo,
 *       norm2, dot, min, max, sum, componentWiseNorm1, operator= (copy),
 *       clone, scalar-array multiply, identity dot(x,x)==norm2(x)^2.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/ArrayDOF.hpp"
#include <cmath>

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

/// Build an ArrayDof<M,N> with father-only storage (no ghost/son data).
/// The son is allocated with zero rows so that the pair is valid for
/// vector-space operations which only touch father data.
template <int M, int N>
static ArrayDof<M, N> makeDof(const MPIInfo &mpi, DNDS::index nCells,
                               DNDS::rowsize mRow = M, DNDS::rowsize nCol = N)
{
    ArrayDof<M, N> dof;
    DNDS_MAKE_SSP(dof.father, mpi);
    DNDS_MAKE_SSP(dof.son, mpi);
    dof.father->Resize(nCells, mRow, nCol);
    dof.son->Resize(0, mRow, nCol);
    dof.TransAttach();
    return dof;
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF setup and setConstant")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 100;

    auto dof = makeDof<5, 1>(mpi, N);

    SUBCASE("setConstant with scalar")
    {
        dof.setConstant(2.0);
        for (DNDS::index i = 0; i < dof.father->Size(); i++)
        {
            auto mat = dof[i];
            for (int r = 0; r < mat.rows(); r++)
                CHECK(mat(r, 0) == doctest::Approx(2.0));
        }
    }

    SUBCASE("setConstant with Eigen matrix")
    {
        Eigen::Matrix<real, 5, 1> v;
        v << 1.0, 2.0, 3.0, 4.0, 5.0;
        dof.setConstant(v);

        for (DNDS::index i = 0; i < dof.father->Size(); i++)
        {
            auto mat = dof[i];
            CHECK(mat(0, 0) == doctest::Approx(1.0));
            CHECK(mat(1, 0) == doctest::Approx(2.0));
            CHECK(mat(2, 0) == doctest::Approx(3.0));
            CHECK(mat(3, 0) == doctest::Approx(4.0));
            CHECK(mat(4, 0) == doctest::Approx(5.0));
        }
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF operator+= scalar and array")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 80;

    auto dof1 = makeDof<5, 1>(mpi, N);
    dof1.setConstant(1.0);

    SUBCASE("+= scalar")
    {
        dof1 += 2.0;

        for (DNDS::index i = 0; i < dof1.father->Size(); i++)
        {
            auto mat = dof1[i];
            for (int r = 0; r < mat.rows(); r++)
                CHECK(mat(r, 0) == doctest::Approx(3.0));
        }
    }

    SUBCASE("+= another ArrayDof")
    {
        dof1 += 2.0; // dof1 is now 3.0

        auto dof2 = makeDof<5, 1>(mpi, N);
        dof2.setConstant(10.0);

        dof1 += dof2;

        for (DNDS::index i = 0; i < dof1.father->Size(); i++)
        {
            auto mat = dof1[i];
            for (int r = 0; r < mat.rows(); r++)
                CHECK(mat(r, 0) == doctest::Approx(13.0));
        }
    }

    SUBCASE("+= Eigen matrix")
    {
        Eigen::Matrix<real, 5, 1> v;
        v << 10.0, 20.0, 30.0, 40.0, 50.0;
        dof1 += v;

        for (DNDS::index i = 0; i < dof1.father->Size(); i++)
        {
            auto mat = dof1[i];
            CHECK(mat(0, 0) == doctest::Approx(11.0));
            CHECK(mat(1, 0) == doctest::Approx(21.0));
            CHECK(mat(2, 0) == doctest::Approx(31.0));
            CHECK(mat(3, 0) == doctest::Approx(41.0));
            CHECK(mat(4, 0) == doctest::Approx(51.0));
        }
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF operator-= and operator*=")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 60;

    auto dof1 = makeDof<5, 1>(mpi, N);
    auto dof2 = makeDof<5, 1>(mpi, N);
    dof1.setConstant(10.0);
    dof2.setConstant(3.0);

    SUBCASE("-= array then *= scalar")
    {
        dof1 -= dof2;

        for (DNDS::index i = 0; i < dof1.father->Size(); i++)
        {
            auto mat = dof1[i];
            for (int r = 0; r < mat.rows(); r++)
                CHECK(mat(r, 0) == doctest::Approx(7.0));
        }

        dof1 *= 2.0;

        for (DNDS::index i = 0; i < dof1.father->Size(); i++)
        {
            auto mat = dof1[i];
            for (int r = 0; r < mat.rows(); r++)
                CHECK(mat(r, 0) == doctest::Approx(14.0));
        }
    }

    SUBCASE("*= by 1.0 is identity")
    {
        dof1 *= 1.0;

        for (DNDS::index i = 0; i < dof1.father->Size(); i++)
        {
            auto mat = dof1[i];
            for (int r = 0; r < mat.rows(); r++)
                CHECK(mat(r, 0) == doctest::Approx(10.0));
        }
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF addTo")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 50;

    auto dof1 = makeDof<5, 1>(mpi, N);
    auto dof2 = makeDof<5, 1>(mpi, N);
    dof1.setConstant(1.0);
    dof2.setConstant(5.0);

    // dof1 += dof2 * 3.0  =>  1.0 + 5.0*3.0 = 16.0
    dof1.addTo(dof2, 3.0);

    for (DNDS::index i = 0; i < dof1.father->Size(); i++)
    {
        auto mat = dof1[i];
        for (int r = 0; r < mat.rows(); r++)
            CHECK(mat(r, 0) == doctest::Approx(16.0));
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF norm2")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 100;

    SUBCASE("uniform value")
    {
        // ArrayDof<1,1>: one scalar per cell.
        // All set to 1.0. Squared norm = N * size, so norm2 = sqrt(N*size).
        auto dof = makeDof<1, 1>(mpi, N);
        dof.setConstant(1.0);

        real n2 = dof.norm2();
        real expected = std::sqrt(static_cast<real>(N) * mpi.size);
        CHECK(n2 == doctest::Approx(expected));
    }

    SUBCASE("multi-component uniform")
    {
        // ArrayDof<5,1>: 5 components per cell, all 1.0.
        // Each cell contributes 5*1^2 = 5 to the squared norm.
        // Global: 5 * N * size  =>  norm2 = sqrt(5*N*size).
        auto dof = makeDof<5, 1>(mpi, N);
        dof.setConstant(1.0);

        real n2 = dof.norm2();
        real expected = std::sqrt(5.0 * N * mpi.size);
        CHECK(n2 == doctest::Approx(expected));
    }

    SUBCASE("non-uniform values")
    {
        // ArrayDof<1,1>, each cell = rank + 1.
        auto dof = makeDof<1, 1>(mpi, N);
        real val = static_cast<real>(mpi.rank + 1);
        dof.setConstant(val);

        // Sum of squares across all ranks:
        // Each rank r contributes N * (r+1)^2.
        real globalSqSum = 0;
        for (MPI_int r = 0; r < mpi.size; r++)
            globalSqSum += N * static_cast<real>((r + 1) * (r + 1));

        real n2 = dof.norm2();
        CHECK(n2 == doctest::Approx(std::sqrt(globalSqSum)));
    }

    SUBCASE("norm2 difference overload")
    {
        auto dof1 = makeDof<1, 1>(mpi, N);
        auto dof2 = makeDof<1, 1>(mpi, N);
        dof1.setConstant(5.0);
        dof2.setConstant(2.0);

        // norm2(dof1, dof2) = sqrt(sum((5-2)^2)) = 3 * sqrt(N*size)
        real n2 = dof1.norm2(dof2);
        real expected = 3.0 * std::sqrt(static_cast<real>(N) * mpi.size);
        CHECK(n2 == doctest::Approx(expected));
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF dot")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 100;

    SUBCASE("scalar dof")
    {
        auto dof1 = makeDof<1, 1>(mpi, N);
        auto dof2 = makeDof<1, 1>(mpi, N);
        dof1.setConstant(2.0);
        dof2.setConstant(3.0);

        // dot = sum_i (2*3) = 6 * N * size
        real d = dof1.dot(dof2);
        real expected = 6.0 * N * mpi.size;
        CHECK(d == doctest::Approx(expected));
    }

    SUBCASE("multi-component dof")
    {
        auto dof1 = makeDof<5, 1>(mpi, N);
        auto dof2 = makeDof<5, 1>(mpi, N);
        dof1.setConstant(2.0);
        dof2.setConstant(3.0);

        // Each cell contributes 5 * (2*3) = 30.
        // Global: 30 * N * size.
        real d = dof1.dot(dof2);
        real expected = 30.0 * N * mpi.size;
        CHECK(d == doctest::Approx(expected));
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF min and max")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 50;

    auto dof = makeDof<3, 1>(mpi, N);
    dof.setConstant(0.0);

    // Rank 0 writes a very negative value into its first cell.
    if (mpi.rank == 0)
        dof[0].setConstant(-100.0);

    // Last rank writes a very large value into its last cell.
    if (mpi.rank == mpi.size - 1)
        dof[N - 1].setConstant(999.0);

    real gmin = dof.min();
    real gmax = dof.max();

    CHECK(gmin == doctest::Approx(-100.0));
    CHECK(gmax == doctest::Approx(999.0));
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF sum")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 100;

    // ArrayDof<1,1>: one scalar per cell, all set to 1.0.
    auto dof = makeDof<1, 1>(mpi, N);
    dof.setConstant(1.0);

    real s = dof.sum();
    real expected = static_cast<real>(N) * mpi.size;
    CHECK(s == doctest::Approx(expected));
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF componentWiseNorm1")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 100;

    auto dof = makeDof<3, 1>(mpi, N);

    // Set per-component values: comp0 = 1, comp1 = -2, comp2 = 3.
    Eigen::Matrix<real, 3, 1> v;
    v << 1.0, -2.0, 3.0;
    dof.setConstant(v);

    auto norm1 = dof.componentWiseNorm1();

    // L1 norm per component across all ranks: |val| * N * size.
    real totalCells = static_cast<real>(N) * mpi.size;
    CHECK(norm1(0, 0) == doctest::Approx(1.0 * totalCells));
    CHECK(norm1(1, 0) == doctest::Approx(2.0 * totalCells));
    CHECK(norm1(2, 0) == doctest::Approx(3.0 * totalCells));
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF componentWiseNorm1 difference")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 80;

    auto dof1 = makeDof<3, 1>(mpi, N);
    auto dof2 = makeDof<3, 1>(mpi, N);

    Eigen::Matrix<real, 3, 1> v1, v2;
    v1 << 5.0, 10.0, -3.0;
    v2 << 2.0, 7.0, 1.0;
    dof1.setConstant(v1);
    dof2.setConstant(v2);

    // componentWiseNorm1(dof2) = sum |dof1 - dof2| per component.
    // diffs: |3|, |3|, |4|
    auto norm1 = dof1.componentWiseNorm1(dof2);

    real totalCells = static_cast<real>(N) * mpi.size;
    CHECK(norm1(0, 0) == doctest::Approx(3.0 * totalCells));
    CHECK(norm1(1, 0) == doctest::Approx(3.0 * totalCells));
    CHECK(norm1(2, 0) == doctest::Approx(4.0 * totalCells));
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF element-wise multiply and divide")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 50;

    auto dof1 = makeDof<3, 1>(mpi, N);
    auto dof2 = makeDof<3, 1>(mpi, N);

    Eigen::Matrix<real, 3, 1> v1, v2;
    v1 << 6.0, 8.0, 10.0;
    v2 << 2.0, 4.0, 5.0;
    dof1.setConstant(v1);
    dof2.setConstant(v2);

    SUBCASE("*= array (element-wise)")
    {
        dof1 *= dof2;

        for (DNDS::index i = 0; i < dof1.father->Size(); i++)
        {
            auto mat = dof1[i];
            CHECK(mat(0, 0) == doctest::Approx(12.0));
            CHECK(mat(1, 0) == doctest::Approx(32.0));
            CHECK(mat(2, 0) == doctest::Approx(50.0));
        }
    }

    SUBCASE("/= array (element-wise)")
    {
        dof1 /= dof2;

        for (DNDS::index i = 0; i < dof1.father->Size(); i++)
        {
            auto mat = dof1[i];
            CHECK(mat(0, 0) == doctest::Approx(3.0));
            CHECK(mat(1, 0) == doctest::Approx(2.0));
            CHECK(mat(2, 0) == doctest::Approx(2.0));
        }
    }

    SUBCASE("*= Eigen matrix (element-wise)")
    {
        Eigen::Matrix<real, 3, 1> scale;
        scale << 0.5, 0.25, 0.1;
        dof1 *= scale;

        for (DNDS::index i = 0; i < dof1.father->Size(); i++)
        {
            auto mat = dof1[i];
            CHECK(mat(0, 0) == doctest::Approx(3.0));
            CHECK(mat(1, 0) == doctest::Approx(2.0));
            CHECK(mat(2, 0) == doctest::Approx(1.0));
        }
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF operator= (copy values)")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 40;

    auto dof1 = makeDof<5, 1>(mpi, N);
    auto dof2 = makeDof<5, 1>(mpi, N);

    Eigen::Matrix<real, 5, 1> v;
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    dof1.setConstant(v);
    dof2.setConstant(0.0);

    // Copy values from dof1 into dof2.
    dof2 = dof1;

    // Verify dof2 now has dof1's values.
    for (DNDS::index i = 0; i < dof2.father->Size(); i++)
    {
        auto mat = dof2[i];
        CHECK(mat(0, 0) == doctest::Approx(1.0));
        CHECK(mat(1, 0) == doctest::Approx(2.0));
        CHECK(mat(2, 0) == doctest::Approx(3.0));
        CHECK(mat(3, 0) == doctest::Approx(4.0));
        CHECK(mat(4, 0) == doctest::Approx(5.0));
    }

    // Modify dof1, verify dof2 is unaffected (deep copy of raw data).
    dof1.setConstant(-999.0);

    for (DNDS::index i = 0; i < dof2.father->Size(); i++)
    {
        auto mat = dof2[i];
        CHECK(mat(0, 0) == doctest::Approx(1.0));
        CHECK(mat(4, 0) == doctest::Approx(5.0));
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF clone")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 30;

    auto dof1 = makeDof<3, 1>(mpi, N);
    Eigen::Matrix<real, 3, 1> v;
    v << 7.0, 8.0, 9.0;
    dof1.setConstant(v);

    // clone creates a fully independent copy (new father/son allocations).
    ArrayDof<3, 1> dof2;
    dof2.clone(dof1);

    for (DNDS::index i = 0; i < dof2.father->Size(); i++)
    {
        auto mat = dof2[i];
        CHECK(mat(0, 0) == doctest::Approx(7.0));
        CHECK(mat(1, 0) == doctest::Approx(8.0));
        CHECK(mat(2, 0) == doctest::Approx(9.0));
    }

    // Modify original, verify clone unchanged.
    dof1.setConstant(0.0);
    CHECK(dof2[0](0, 0) == doctest::Approx(7.0));
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF scalar array multiply")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 50;

    auto dof = makeDof<3, 1>(mpi, N);
    Eigen::Matrix<real, 3, 1> v;
    v << 2.0, 4.0, 6.0;
    dof.setConstant(v);

    // Build a scalar ArrayDof<1,1> with per-cell scaling factors.
    auto scale = makeDof<1, 1>(mpi, N);
    scale.setConstant(3.0);

    // dof *= scale multiplies each cell's vector by the scalar.
    dof *= scale;

    for (DNDS::index i = 0; i < dof.father->Size(); i++)
    {
        auto mat = dof[i];
        CHECK(mat(0, 0) == doctest::Approx(6.0));
        CHECK(mat(1, 0) == doctest::Approx(12.0));
        CHECK(mat(2, 0) == doctest::Approx(18.0));
    }
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF norm2 zero array")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 100;

    auto dof = makeDof<5, 1>(mpi, N);
    dof.setConstant(0.0);

    CHECK(dof.norm2() == doctest::Approx(0.0));
}

// ---------------------------------------------------------------------------
TEST_CASE("ArrayDOF dot self equals norm2 squared")
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 100;

    auto dof = makeDof<5, 1>(mpi, N);
    Eigen::Matrix<real, 5, 1> v;
    v << 1.0, -2.0, 3.0, -4.0, 5.0;
    dof.setConstant(v);

    real n2 = dof.norm2();
    real d = dof.dot(dof);

    CHECK(d == doctest::Approx(n2 * n2));
}

// ---------------------------------------------------------------------------
// Parametric tests over nVars dimension
// ---------------------------------------------------------------------------

template <int M>
struct DofTag
{
    static constexpr int m = M;
};

TYPE_TO_STRING(DofTag<1>);
TYPE_TO_STRING(DofTag<3>);
TYPE_TO_STRING(DofTag<5>);
TYPE_TO_STRING(DofTag<DNDS::DynamicSize>);

TEST_CASE_TEMPLATE("ArrayDOF parametric over nVars", T,
                    DofTag<1>, DofTag<3>, DofTag<5>, DofTag<DNDS::DynamicSize>)
{
    MPIInfo mpi = worldMPI();
    constexpr DNDS::index N = 64;
    constexpr int M = T::m;

    // For DynamicSize, rows must be provided at runtime.  Pick 4 as the
    // runtime row count so it differs from every fixed tag.
    constexpr DNDS::rowsize rtRows = 4;
    const int effectiveRows = (M == DNDS::DynamicSize) ? rtRows : M;

    // Helper lambda to build the dof with the right overload.
    auto buildDof = [&]()
    {
        if constexpr (M == DNDS::DynamicSize)
            return makeDof<DNDS::DynamicSize, 1>(mpi, N, rtRows, 1);
        else
            return makeDof<M, 1>(mpi, N);
    };

    SUBCASE("setConstant scalar")
    {
        auto dof = buildDof();
        dof.setConstant(3.0);

        for (DNDS::index i = 0; i < dof.father->Size(); i++)
        {
            auto mat = dof[i];
            CHECK(mat.rows() == effectiveRows);
            for (int r = 0; r < mat.rows(); r++)
                CHECK(mat(r, 0) == doctest::Approx(3.0));
        }
    }

    SUBCASE("+= scalar")
    {
        auto dof = buildDof();
        dof.setConstant(1.0);
        dof += 4.0;

        for (DNDS::index i = 0; i < dof.father->Size(); i++)
        {
            auto mat = dof[i];
            for (int r = 0; r < mat.rows(); r++)
                CHECK(mat(r, 0) == doctest::Approx(5.0));
        }
    }

    SUBCASE("norm2")
    {
        auto dof = buildDof();
        dof.setConstant(1.0);

        real n2 = dof.norm2();
        // Each cell contributes effectiveRows * 1^2.
        real expected = std::sqrt(static_cast<real>(effectiveRows) * N * mpi.size);
        CHECK(n2 == doctest::Approx(expected));
    }

    SUBCASE("dot")
    {
        auto dof1 = buildDof();
        auto dof2 = buildDof();
        dof1.setConstant(2.0);
        dof2.setConstant(3.0);

        real d = dof1.dot(dof2);
        // Each cell: effectiveRows * (2*3) = effectiveRows * 6.
        real expected = static_cast<real>(effectiveRows) * 6.0 * N * mpi.size;
        CHECK(d == doctest::Approx(expected));
    }

    SUBCASE("clone")
    {
        auto dof1 = buildDof();
        dof1.setConstant(7.0);

        ArrayDof<M, 1> dof2;
        dof2.clone(dof1);

        for (DNDS::index i = 0; i < dof2.father->Size(); i++)
        {
            auto mat = dof2[i];
            CHECK(mat.rows() == effectiveRows);
            for (int r = 0; r < mat.rows(); r++)
                CHECK(mat(r, 0) == doctest::Approx(7.0));
        }

        // Modify original, verify clone unchanged.
        dof1.setConstant(0.0);
        CHECK(dof2[0](0, 0) == doctest::Approx(7.0));
    }
}
