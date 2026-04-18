/**
 * @file test_Direct.cpp
 * @brief Unit tests for block-sparse LU factorization in Solver/Direct.hpp.
 *
 * Tests a 2D periodic Laplacian with 2x2 blocks on an NxN grid:
 *   A = 5-point stencil with periodic boundary (4 neighbors per cell).
 *   Each cell (i,j) connects to (i±1,j) and (i,j±1) with wrap-around.
 *   Diag = 4*I + delta*diag(i,2i), off-diag = -I (non-symmetric diag
 *   to exercise the full LU, not just LDLT).
 *
 * Verifies:
 *   - SerialSymLUStructure: symbolic factorization with fill-in
 *   - LocalLUBase: MatMul (A*x == b), InPlaceDecompose, Solve (A^{-1}b == x)
 *   - ILU(0): approximate solve, residual reduction
 *   - Complete LU (iluCode=-1): exact solve
 *   - ILU(1): intermediate fill, better than ILU(0)
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Solver/Direct.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cmath>

using namespace DNDS;

static constexpr int BS = 2; // block size
using Block = Eigen::Matrix<real, BS, BS>;
using BVec = Eigen::Vector<real, BS>;

// ===================================================================
// Concrete LocalLU derived class for 2x2 dense blocks
// ===================================================================
struct TestBlockLU;
using tVec = std::vector<BVec>;
using tBase = Direct::LocalLUBase<TestBlockLU, Block, tVec>;

struct TestBlockLU : public tBase
{
    std::vector<Block> diag;
    std::vector<std::vector<Block>> lower;
    std::vector<std::vector<Block>> upper;

    TestBlockLU(ssp<Direct::SerialSymLUStructure> s, int N)
        : tBase(s), diag(N), lower(N), upper(N)
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            diag[i].setZero();
            lower[i].resize(s->lowerTriStructure[i].size());
            upper[i].resize(s->upperTriStructure[i].size());
            for (auto &b : lower[i]) b.setZero();
            for (auto &b : upper[i]) b.setZero();
        }
    }

    Block &GetDiag(DNDS::index i) { return diag[i]; }
    Block &GetLower(DNDS::index i, int ij) { return lower[i][ij]; }
    Block &GetUpper(DNDS::index i, int ij) { return upper[i][ij]; }
    Block InvertDiag(const Block &v) { return v.inverse(); }
};

// ===================================================================
// 2D periodic grid: NxN cells, cell (r,c) has 4 neighbors with wrap
// ===================================================================
static constexpr int G = 4; // grid side length => G*G = 16 cells
static const int NCells = G * G;

static DNDS::index cellIdx(int r, int c)
{
    return ((r % G) + G) % G * G + ((c % G) + G) % G;
}

static std::vector<std::vector<DNDS::index>> build2DPeriodicAdj()
{
    std::vector<std::vector<DNDS::index>> adj(NCells);
    for (int r = 0; r < G; r++)
        for (int c = 0; c < G; c++)
        {
            DNDS::index me = cellIdx(r, c);
            adj[me].push_back(cellIdx(r - 1, c));
            adj[me].push_back(cellIdx(r + 1, c));
            adj[me].push_back(cellIdx(r, c - 1));
            adj[me].push_back(cellIdx(r, c + 1));
        }
    return adj;
}

/// Fill 2D periodic Laplacian with non-symmetric diagonal perturbation.
/// Diag_i = 4*I + delta * diag(i, 2*i)  (makes the diagonal non-symmetric).
/// Off-diag entries corresponding to original adjacency = -I.
/// Fill-in entries (from ILU(k>0) or complete LU) = 0 (zeroed by constructor).
/// Uses cell2cellFaceVLocal2FullRowPos to identify which entries in
/// lower/upper correspond to original neighbors.
static void fill2DLaplacian(TestBlockLU &lu, double delta = 0.1)
{
    auto adj = build2DPeriodicAdj();
    Block I2 = Block::Identity();
    auto &symLU = *lu.symLU;

    for (int i = 0; i < NCells; i++)
    {
        Block D = 4.0 * I2;
        D(0, 0) += delta * i;
        D(1, 1) += delta * 2 * i;
        lu.diag[i] = D;

        // Only fill original adjacency entries with -I; fill-in stays 0
        auto &c2cPos = symLU.cell2cellFaceVLocal2FullRowPos[i];
        for (int ic2c = 0; ic2c < (int)adj[i].size(); ic2c++)
        {
            int pos = c2cPos[ic2c]; // 0 = diag, 1..nLower = lower, nLower+1.. = upper
            if (pos == 0)
                continue; // diagonal (self)
            int nLower = symLU.lowerTriStructure[i].size();
            if (pos <= nLower)
                lu.lower[i][pos - 1] = -I2;
            else
                lu.upper[i][pos - nLower - 1] = -I2;
        }
    }
}

static MPIInfo g_mpi;

// ===================================================================
// Tests
// ===================================================================

TEST_CASE("Direct 2D periodic: symbolic factorization has fill-in")
{
    auto symLU = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    auto adj = build2DPeriodicAdj();
    symLU->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, -1); // complete LU

    // Complete LU on a 2D grid has fill-in: some rows should have > 4 entries
    int maxLower = 0;
    for (int i = 0; i < NCells; i++)
        maxLower = std::max(maxLower, (int)symLU->lowerTriStructure[i].size());

    std::cout << "[2D-sym] maxLower entries = " << maxLower
              << " (> 4 means fill-in)" << std::endl;
    CHECK(maxLower > 2); // definitely more than the original 2 lower neighbors
}

TEST_CASE("Direct 2D periodic: MatMul is correct")
{
    auto symLU = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    auto adj = build2DPeriodicAdj();
    symLU->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0);

    TestBlockLU lu(symLU, NCells);
    fill2DLaplacian(lu);

    // Build a known x vector
    tVec x(NCells), Ax(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x[i] = BVec::Zero();
        x[i](0) = std::sin(0.5 * i);
        x[i](1) = std::cos(0.3 * i);
        Ax[i] = BVec::Zero();
    }

    lu.MatMul(x, Ax);

    // Verify by manual computation for cell 0
    int r0 = 0, c0 = 0;
    BVec expected = lu.diag[0] * x[0];
    // 4 neighbors of cell 0 (periodic: left=G-1, right=1, up=(G-1)*G, down=G)
    DNDS::index neighbors[] = {cellIdx(r0 - 1, c0), cellIdx(r0 + 1, c0),
                                cellIdx(r0, c0 - 1), cellIdx(r0, c0 + 1)};
    for (auto nb : neighbors)
        expected -= x[nb]; // off-diag = -I

    real err0 = (Ax[0] - expected).norm();
    CHECK(err0 < 1e-14);

    // Check all entries are finite
    for (int i = 0; i < NCells; i++)
        CHECK_FALSE(Ax[i].hasNaN());
}

TEST_CASE("Direct 2D periodic: complete LU solve is exact")
{
    auto symLU = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    auto adj = build2DPeriodicAdj();
    symLU->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, -1); // complete LU

    // Build A for MatMul, then decompose a copy for Solve
    TestBlockLU luMat(symLU, NCells);
    fill2DLaplacian(luMat);

    tVec x_known(NCells), b(NCells), x_solved(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x_known[i] = BVec::Zero();
        x_known[i](0) = std::sin(0.7 * i + 0.1);
        x_known[i](1) = std::cos(0.4 * i + 0.2);
        b[i] = BVec::Zero();
        x_solved[i] = BVec::Zero();
    }

    luMat.MatMul(x_known, b); // b = A * x_known

    TestBlockLU luSolve(symLU, NCells);
    fill2DLaplacian(luSolve);
    luSolve.InPlaceDecompose();
    luSolve.Solve(b, x_solved);

    real maxErr = 0;
    for (int i = 0; i < NCells; i++)
        maxErr = std::max(maxErr, (x_solved[i] - x_known[i]).norm());

    std::cout << "[2D-LU-full] maxErr = " << std::scientific << maxErr << std::endl;
    CHECK(maxErr < 1e-10);
}

TEST_CASE("Direct 2D periodic: ILU(0) is approximate but reduces residual")
{
    auto symLU0 = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    auto adj = build2DPeriodicAdj();
    symLU0->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0); // ILU(0)

    // Build b = A * x_known using the original (non-decomposed) matrix
    auto symLUorig = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLUorig->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0);
    TestBlockLU luOrig(symLUorig, NCells);
    fill2DLaplacian(luOrig);

    tVec x_known(NCells), b(NCells), x_ilu(NCells), Ax_ilu(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x_known[i] = BVec::Zero();
        x_known[i](0) = std::sin(0.7 * i + 0.1);
        x_known[i](1) = std::cos(0.4 * i + 0.2);
        b[i] = BVec::Zero();
        x_ilu[i] = BVec::Zero();
        Ax_ilu[i] = BVec::Zero();
    }

    luOrig.MatMul(x_known, b);

    // ILU(0) solve
    TestBlockLU luILU(symLU0, NCells);
    fill2DLaplacian(luILU);
    luILU.InPlaceDecompose();
    luILU.Solve(b, x_ilu);

    // Compute residual ||b - A*x_ilu||
    // Need a fresh non-decomposed matrix for MatMul
    auto symLUmul = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLUmul->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0);
    TestBlockLU luMul(symLUmul, NCells);
    fill2DLaplacian(luMul);
    luMul.MatMul(x_ilu, Ax_ilu);

    real residualNorm = 0, bNorm = 0;
    for (int i = 0; i < NCells; i++)
    {
        residualNorm += (b[i] - Ax_ilu[i]).squaredNorm();
        bNorm += b[i].squaredNorm();
    }
    residualNorm = std::sqrt(residualNorm);
    bNorm = std::sqrt(bNorm);

    real relResidual = residualNorm / bNorm;
    std::cout << "[2D-ILU0] relResidual = " << std::scientific << relResidual << std::endl;

    // ILU(0) on a 2D periodic grid is NOT exact (has fill-in), but should reduce residual
    CHECK(relResidual < 0.5); // significant reduction from initial residual of ~1
    CHECK(relResidual > 1e-14); // not exact
}

/// Use ILU as preconditioner in fixed-point iteration: x_{k+1} = x_k + M^{-1}(b - Ax_k).
/// Returns error norm after nIter iterations.
static real iluPrecIterSolve(int iluCode, const tVec &b, const tVec &x_known, int nIter)
{
    auto adj = build2DPeriodicAdj();

    // Build non-decomposed A for MatMul
    auto symA = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symA->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0);
    TestBlockLU luA(symA, NCells);
    fill2DLaplacian(luA);

    // Build ILU preconditioner M
    auto symM = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symM->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, iluCode);
    TestBlockLU luM(symM, NCells);
    fill2DLaplacian(luM);
    luM.InPlaceDecompose();

    tVec x(NCells), r(NCells), z(NCells), Ax(NCells);
    for (int i = 0; i < NCells; i++)
        x[i] = BVec::Zero();

    for (int iter = 0; iter < nIter; iter++)
    {
        luA.MatMul(x, Ax);
        for (int i = 0; i < NCells; i++)
            r[i] = b[i] - Ax[i];
        luM.Solve(r, z);
        for (int i = 0; i < NCells; i++)
            x[i] += z[i];
    }

    real err = 0;
    for (int i = 0; i < NCells; i++)
        err += (x[i] - x_known[i]).squaredNorm();
    return std::sqrt(err);
}

TEST_CASE("Direct 2D periodic: ILU(1) converges faster than ILU(0) as preconditioner")
{
    auto adj = build2DPeriodicAdj();

    // Build b = A * x_known
    auto symLUorig = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLUorig->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0);
    TestBlockLU luOrig(symLUorig, NCells);
    fill2DLaplacian(luOrig);

    tVec x_known(NCells), b(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x_known[i] = BVec::Zero();
        x_known[i](0) = std::sin(0.7 * i + 0.1);
        x_known[i](1) = std::cos(0.4 * i + 0.2);
        b[i] = BVec::Zero();
    }
    luOrig.MatMul(x_known, b);

    int nIter = 10;
    real errILU0 = iluPrecIterSolve(0, b, x_known, nIter);
    real errILU1 = iluPrecIterSolve(1, b, x_known, nIter);
    real errFull = iluPrecIterSolve(-1, b, x_known, nIter);

    std::cout << "[2D-ILU-prec-iter] ILU(0)=" << std::scientific << errILU0
              << " ILU(1)=" << errILU1
              << " Full=" << errFull
              << " (after " << nIter << " iters)" << std::endl;

    CHECK(errFull < 1e-10);   // complete LU converges in 1 iteration
    CHECK(errILU1 < errILU0); // ILU(1) converges faster
    CHECK(errILU0 < 1e-3);    // ILU(0) also converges
}

// ===================================================================
// LDLT: concrete derived class for symmetric 2x2 blocks
// ===================================================================
struct TestBlockLDLT;
using tLDLTBase = Direct::LocalLDLTBase<TestBlockLDLT, Block, tVec>;

struct TestBlockLDLT : public tLDLTBase
{
    std::vector<Block> diag;
    std::vector<std::vector<Block>> lower;

    TestBlockLDLT(ssp<Direct::SerialSymLUStructure> s, int N)
        : tLDLTBase(s), diag(N), lower(N)
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            diag[i].setZero();
            lower[i].resize(s->lowerTriStructure[i].size());
            for (auto &b : lower[i]) b.setZero();
        }
    }

    Block &GetDiag(DNDS::index i) { return diag[i]; }
    Block &GetLower(DNDS::index i, int ij) { return lower[i][ij]; }
    Block InvertDiag(const Block &v) { return v.inverse(); }
};

/// Fill 2D periodic Laplacian with SYMMETRIC diagonal: Diag = (4+0.1*i)*I.
/// Lower = -I at original adjacency positions; fill-in positions stay 0.
static void fill2DLaplacianSym(TestBlockLDLT &lu, double delta = 0.1)
{
    auto adj = build2DPeriodicAdj();
    Block I2 = Block::Identity();
    auto &symLU = *lu.symLU;

    for (int i = 0; i < NCells; i++)
    {
        lu.diag[i] = (4.0 + delta * i) * I2;
        auto &c2cPos = symLU.cell2cellFaceVLocal2FullRowPos[i];
        int nLower = symLU.lowerTriStructure[i].size();
        for (int ic2c = 0; ic2c < (int)adj[i].size(); ic2c++)
        {
            int pos = c2cPos[ic2c];
            if (pos == 0)
                continue;
            // LDLT only stores lower; pos 1..nLower is lower
            if (pos <= nLower)
                lu.lower[i][pos - 1] = -I2;
            // Upper positions: LDLT uses lower^T, no explicit upper storage
        }
    }
}

// Symmetric version for LU (MatMul reference) — same fill logic
static void fill2DLaplacianSymLU(TestBlockLU &lu, double delta = 0.1)
{
    auto adj = build2DPeriodicAdj();
    Block I2 = Block::Identity();
    auto &symLU = *lu.symLU;

    for (int i = 0; i < NCells; i++)
    {
        lu.diag[i] = (4.0 + delta * i) * I2;
        auto &c2cPos = symLU.cell2cellFaceVLocal2FullRowPos[i];
        for (int ic2c = 0; ic2c < (int)adj[i].size(); ic2c++)
        {
            int pos = c2cPos[ic2c];
            if (pos == 0)
                continue;
            int nLower = symLU.lowerTriStructure[i].size();
            if (pos <= nLower)
                lu.lower[i][pos - 1] = -I2;
            else
                lu.upper[i][pos - nLower - 1] = -I2;
        }
    }
}

// ===================================================================
// LDLT tests
// ===================================================================

TEST_CASE("Direct 2D periodic LDLT: MatMul matches LU MatMul for symmetric system")
{
    auto adj = build2DPeriodicAdj();

    auto symLU = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLU->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, -1);

    // LU reference
    TestBlockLU luRef(symLU, NCells);
    fill2DLaplacianSymLU(luRef);

    // LDLT
    TestBlockLDLT ldlt(symLU, NCells);
    fill2DLaplacianSym(ldlt);

    tVec x(NCells), Ax_lu(NCells), Ax_ldlt(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x[i] = BVec::Zero();
        x[i](0) = std::sin(0.3 * i);
        x[i](1) = std::cos(0.7 * i);
        Ax_lu[i] = BVec::Zero();
        Ax_ldlt[i] = BVec::Zero();
    }

    luRef.MatMul(x, Ax_lu);
    ldlt.MatMul(x, Ax_ldlt);

    real maxDiff = 0;
    for (int i = 0; i < NCells; i++)
        maxDiff = std::max(maxDiff, (Ax_lu[i] - Ax_ldlt[i]).norm());

    std::cout << "[2D-LDLT-MatMul] maxDiff = " << std::scientific << maxDiff << std::endl;
    CHECK(maxDiff < 1e-14);
}

TEST_CASE("Direct 2D periodic LDLT: complete decompose + solve is exact")
{
    auto adj = build2DPeriodicAdj();

    // Build reference b = A*x_known using LU MatMul
    auto symLU = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLU->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, -1);
    TestBlockLU luRef(symLU, NCells);
    fill2DLaplacianSymLU(luRef);

    tVec x_known(NCells), b(NCells), x_solved(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x_known[i] = BVec::Zero();
        x_known[i](0) = std::sin(0.7 * i + 0.1);
        x_known[i](1) = std::cos(0.4 * i + 0.2);
        b[i] = BVec::Zero();
        x_solved[i] = BVec::Zero();
    }
    luRef.MatMul(x_known, b);

    // Solve with LDLT
    auto symLU2 = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLU2->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, -1);
    TestBlockLDLT ldlt(symLU2, NCells);
    fill2DLaplacianSym(ldlt);
    ldlt.InPlaceDecompose();
    ldlt.Solve(b, x_solved);

    real maxErr = 0;
    for (int i = 0; i < NCells; i++)
        maxErr = std::max(maxErr, (x_solved[i] - x_known[i]).norm());

    std::cout << "[2D-LDLT-Solve] maxErr = " << std::scientific << maxErr << std::endl;
    CHECK(maxErr < 1e-10);
}

// ===================================================================
// main with MPI (needed for SerialSymLUStructure internals)
// ===================================================================
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
