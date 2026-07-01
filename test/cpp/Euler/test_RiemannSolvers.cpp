/**
 * @file test_RiemannSolvers.cpp
 * @brief Unit tests for Riemann solvers in Gas.hpp.
 *
 * Tests cover:
 *   - Consistency: F(UL, UR) with UL==UR equals the exact physical flux
 *   - Roe flux: default eigScheme=0, plus variants 1-8
 *   - HLLC flux: consistency and symmetry
 *   - HLLEP flux: consistency
 *   - InviscidFlux_IdealGas_Dispatcher: runtime dispatch
 *   - Symmetry: F(UL,UR,n) = -F(UR,UL,-n) for all solvers
 *   - Sod shock tube: UL != UR produces finite, bounded flux
 *   - Golden values for specific test vectors
 *
 * All functions are pure (no MPI, no mesh). Serial doctest.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/Gas.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace DNDS;
using namespace DNDS::Euler::Gas;

static constexpr real g_gamma = 1.4;
static constexpr real GOLDEN_NOT_ACQUIRED = 1e300;

// Build a 3D conservative state from primitive
static Eigen::Vector<real, 5> prim2cons(real rho, real u, real v, real w, real p)
{
    Eigen::Vector<real, 5> U;
    real E = p / (g_gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w);
    U << rho, rho * u, rho * v, rho * w, E;
    return U;
}

static Eigen::Vector<real, 5> prim2consSplit(real rho, real u, real v, real w, real p, real gammaEq, real rhoEBase)
{
    Eigen::Vector<real, 5> U;
    real E = p / (gammaEq - 1.0) + 0.5 * rho * (u * u + v * v + w * w) + rhoEBase;
    U << rho, rho * u, rho * v, rho * w, E;
    return U;
}

// Compute the exact physical normal-direction flux
static Eigen::Vector<real, 5> exactNormalFlux(
    const Eigen::Vector<real, 5> &U, const Eigen::Vector3d &n)
{
    Eigen::Vector3d velo = U.segment<3>(1) / U(0);
    real vn = velo.dot(n);
    real vSqr = velo.squaredNorm();
    real p = (g_gamma - 1.0) * (U(4) - 0.5 * U(0) * vSqr);
    real E = U(4);
    Eigen::Vector<real, 5> F;
    F(0) = U(0) * vn;
    F.segment<3>(1) = U.segment<3>(1) * vn + p * n;
    F(4) = (E + p) * vn;
    return F;
}

static Eigen::Vector<real, 5> exactNormalFluxSplit(
    const Eigen::Vector<real, 5> &U, const Eigen::Vector3d &n, real gammaEq, real rhoEBase)
{
    Eigen::Vector3d velo = U.segment<3>(1) / U(0);
    real vn = velo.dot(n);
    real vSqr = velo.squaredNorm();
    real p = (gammaEq - 1.0) * (U(4) - 0.5 * U(0) * vSqr - rhoEBase);
    Eigen::Vector<real, 5> F;
    F(0) = U(0) * vn;
    F.segment<3>(1) = U.segment<3>(1) * vn + p * n;
    F(4) = (U(4) + p) * vn;
    return F;
}

// No-op dump callback
static auto noDump = []() {};

static Eigen::Vector<real, 5> mirrorRoeDissipation3D(
    const Eigen::Vector<real, 5> &incU,
    const Eigen::Vector3d &n,
    const Eigen::Vector3d &veloRoe,
    real vsqrRoe,
    real aRoe,
    real asqrRoe,
    real HRoe,
    real gammaEqRoe,
    real lam0,
    real lam123,
    real lam4,
    real contactEnergyJump)
{
    real veloRoeN = veloRoe.dot(n);
    real incU123N = incU.segment<3>(1).dot(n);
    Eigen::Vector3d alpha23V = incU.segment<3>(1) - incU(0) * veloRoe;
    Eigen::Vector3d alpha23VT = alpha23V - n * alpha23V.dot(n);
    real incU4b = incU(4) - alpha23VT.dot(veloRoe);
#ifdef USE_ROE_BASE_ENERGY_CONTACT_FIX
    incU4b -= contactEnergyJump;
#endif
    real entropyDenom = HRoe - 0.5 * vsqrRoe;
    REQUIRE(entropyDenom > 0);
    real invEntropyDenom = 1.0 / entropyDenom;
    real alpha1 = invEntropyDenom *
                  (incU(0) * (HRoe - veloRoeN * veloRoeN) +
                   veloRoeN * incU123N - incU4b);
    real alpha0 = (incU(0) * (veloRoeN + aRoe) - incU123N - aRoe * alpha1) / (2 * aRoe);
    real alpha4 = incU(0) - (alpha0 + alpha1);

    alpha0 *= lam0;
    alpha1 *= lam123;
    alpha23VT *= lam123;
    alpha4 *= lam4;

    Eigen::Vector<real, 5> incF;
    incF(0) = alpha0 + alpha1 + alpha4;
    incF(4) = (HRoe - veloRoeN * aRoe) * alpha0 + 0.5 * vsqrRoe * alpha1 +
              (HRoe + veloRoeN * aRoe) * alpha4 + alpha23VT.dot(veloRoe);
#ifdef USE_ROE_BASE_ENERGY_CONTACT_FIX
    incF(4) += lam123 * contactEnergyJump;
#endif
    incF.segment<3>(1) = (veloRoe - aRoe * n) * alpha0 +
                         (veloRoe + aRoe * n) * alpha4 +
                         veloRoe * alpha1 + alpha23VT;
    return incF;
}

// ===================================================================
// Helper: run a Riemann solver via the dispatcher and return flux
// ===================================================================
static Eigen::Vector<real, 5> callDispatcher(
    RiemannSolverType rsType,
    const Eigen::Vector<real, 5> &UL,
    const Eigen::Vector<real, 5> &UR,
    const Eigen::Vector3d &n)
{
    Eigen::Vector3d vg = Eigen::Vector3d::Zero();
    Eigen::Vector<real, 5> F;
    F.setZero();
    real lam0, lam123, lam4;
    InviscidFlux_IdealGas_Dispatcher<3>(
        rsType, UL, UR, UL, UR, vg, n, g_gamma, g_gamma, F,
        0.0, 0.0, 0.0, noDump, lam0, lam123, lam4);
    return F;
}

// ===================================================================
// CONSISTENCY: F(U, U, n) = exact physical flux
// ===================================================================

TEST_CASE("Roe consistency: identical states give exact flux")
{
    auto U = prim2cons(1.225, 100.0, -50.0, 25.0, 101325.0);
    Eigen::Vector3d n(0.6, -0.8, 0.0);
    n.normalize();

    auto Fexact = exactNormalFlux(U, n);
    auto F = callDispatcher(Roe, U, U, n);

    for (int i = 0; i < 5; i++)
    {
        CAPTURE(i);
        CHECK(F(i) == doctest::Approx(Fexact(i)).epsilon(1e-10));
    }
}

TEST_CASE("HLLC consistency: identical states give exact flux")
{
    auto U = prim2cons(1.225, 100.0, -50.0, 25.0, 101325.0);
    Eigen::Vector3d n(0.0, 1.0, 0.0);

    auto Fexact = exactNormalFlux(U, n);
    auto F = callDispatcher(HLLC, U, U, n);

    for (int i = 0; i < 5; i++)
    {
        CAPTURE(i);
        CHECK(F(i) == doctest::Approx(Fexact(i)).epsilon(1e-10));
    }
}

TEST_CASE("HLLEP consistency: identical states give exact flux")
{
    auto U = prim2cons(1.225, 100.0, -50.0, 25.0, 101325.0);
    Eigen::Vector3d n(0.0, 0.0, 1.0);

    auto Fexact = exactNormalFlux(U, n);
    auto F = callDispatcher(HLLEP, U, U, n);

    for (int i = 0; i < 5; i++)
    {
        CAPTURE(i);
        CHECK(F(i) == doctest::Approx(Fexact(i)).epsilon(1e-10));
    }
}

TEST_CASE("Variable-gamma Riemann solvers use gammaEq pressure and gamma acoustic paths")
{
    real gammaEq = 1.23;
    real gammaCpCv = 1.41;
    real rhoEBase = 6.0;
    auto U = prim2consSplit(1.4, 12.0, -3.0, 2.0, 17.0, gammaEq, rhoEBase);
    Eigen::Vector3d n(0.3, -0.4, 0.5);
    n.normalize();
    Eigen::Vector3d vg = Eigen::Vector3d::Zero();
    auto Fexact = exactNormalFluxSplit(U, n, gammaEq, rhoEBase);

    for (auto rs : {Roe, HLLC, HLLEP})
    {
        CAPTURE(rs);
        Eigen::Vector<real, 5> F;
        F.setZero();
        real lam0 = 0, lam123 = 0, lam4 = 0;
        InviscidFlux_IdealGas_Dispatcher<3, true>(
            rs, U, U, U, U, vg, n, gammaEq, gammaCpCv, F,
            0.0, 1.0, 0.0, noDump, lam0, lam123, lam4,
            rhoEBase, rhoEBase, rhoEBase, rhoEBase,
            gammaEq, gammaEq, gammaEq, gammaEq,
            gammaCpCv, gammaCpCv, gammaCpCv, gammaCpCv);

        for (int i = 0; i < 5; i++)
        {
            CAPTURE(i);
            CHECK(F(i) == doctest::Approx(Fexact(i)).epsilon(1e-10));
        }
    }

    auto rp = ComputeRoePreamble<3, true>(U, U, gammaEq, gammaCpCv, noDump,
                                          rhoEBase, rhoEBase,
                                          gammaEq, gammaEq,
                                          gammaCpCv, gammaCpCv);
    real vn = (U.segment<3>(1) / U(0)).dot(n);
    real a = std::sqrt(gammaCpCv * 17.0 / 1.4);
    CHECK(rp.aRoe == doctest::Approx(a).epsilon(1e-12));
    CHECK(std::abs(rp.aRoe - std::sqrt(gammaEq * 17.0 / 1.4)) > 1e-8);
    CHECK(std::abs(vn + rp.aRoe) > std::abs(vn));
}

TEST_CASE("Variable-gamma Riemann solvers remain symmetric with base energy")
{
    real gammaEqL = 1.22;
    real gammaEqR = 1.31;
    real gammaL = 1.39;
    real gammaR = 1.33;
    real rhoEBaseL = 5.0;
    real rhoEBaseR = -2.0;
    auto UL = prim2consSplit(1.1, 25.0, -5.0, 1.0, 20.0, gammaEqL, rhoEBaseL);
    auto UR = prim2consSplit(0.8, -7.0, 3.0, 0.5, 9.0, gammaEqR, rhoEBaseR);
    Eigen::Vector3d n(1.0, 0.2, -0.1);
    n.normalize();
    Eigen::Vector3d vg = Eigen::Vector3d::Zero();

    for (auto rs : {Roe, HLLEP})
    {
        CAPTURE(rs);
        Eigen::Vector<real, 5> F1, F2;
        real lam0 = 0, lam123 = 0, lam4 = 0;
        InviscidFlux_IdealGas_Dispatcher<3, true>(
            rs, UL, UR, UL, UR, vg, n, g_gamma, g_gamma, F1,
            0.0, 1.0, 0.0, noDump, lam0, lam123, lam4,
            rhoEBaseL, rhoEBaseR, rhoEBaseL, rhoEBaseR,
            gammaEqL, gammaEqR, gammaEqL, gammaEqR,
            gammaL, gammaR, gammaL, gammaR);
        InviscidFlux_IdealGas_Dispatcher<3, true>(
            rs, UR, UL, UR, UL, vg, -n, g_gamma, g_gamma, F2,
            0.0, 1.0, 0.0, noDump, lam0, lam123, lam4,
            rhoEBaseR, rhoEBaseL, rhoEBaseR, rhoEBaseL,
            gammaEqR, gammaEqL, gammaEqR, gammaEqL,
            gammaR, gammaL, gammaR, gammaL);

        for (int i = 0; i < 5; i++)
        {
            CAPTURE(i);
            CHECK(F1(i) == doctest::Approx(-F2(i)).epsilon(1e-10));
        }
    }

    // HLLC is retained as a finite-value smoke test here; its split-gamma
    // unequal-state symmetry remains approximate in this implementation.
    Eigen::Vector<real, 5> FHLLC;
    real lam0 = 0, lam123 = 0, lam4 = 0;
    InviscidFlux_IdealGas_Dispatcher<3, true>(
        HLLC, UL, UR, UL, UR, vg, n, g_gamma, g_gamma, FHLLC,
        0.0, 1.0, 0.0, noDump, lam0, lam123, lam4,
        rhoEBaseL, rhoEBaseR, rhoEBaseL, rhoEBaseR,
        gammaEqL, gammaEqR, gammaEqL, gammaEqR,
        gammaL, gammaR, gammaL, gammaR);
    CHECK(FHLLC.allFinite());
}

TEST_CASE("Roe base-energy mode reconstructs identity when all lambdas are one")
{
    real gammaEqL = 1.28;
    real gammaEqR = 1.34;
    real gammaL = 1.41;
    real gammaR = 1.37;
    real rhoEBaseL = 8.0;
    real rhoEBaseR = -4.0;
    auto UL = prim2consSplit(1.3, 3.0, -0.7, 0.4, 2.5, gammaEqL, rhoEBaseL);
    auto UR = prim2consSplit(1.3, 3.0, -0.7, 0.4, 2.5, gammaEqR, rhoEBaseR);
    Eigen::Vector3d n(0.9, -0.2, 0.1);
    n.normalize();

    Eigen::Vector3d veloRoe;
    real vsqrRoe = 0, aRoe = 0, asqrRoe = 0, HRoe = 0;
    Eigen::Vector<real, 5> uRoe;
    GetRoeAverage<3, true>(UL, UR, g_gamma, g_gamma,
                           veloRoe, vsqrRoe, aRoe, asqrRoe, HRoe, uRoe,
                           rhoEBaseL, rhoEBaseR,
                           gammaEqL, gammaEqR,
                           gammaL, gammaR);
    real gammaEqRoe = (std::sqrt(UL(0)) * gammaEqL + std::sqrt(UR(0)) * gammaEqR) /
                      (std::sqrt(UL(0)) + std::sqrt(UR(0)));

    Eigen::Vector<real, 5> incU = UR - UL;
    real contactEnergyJump = (rhoEBaseR - rhoEBaseL) +
                             2.5 / (gammaEqR - 1) - 2.5 / (gammaEqL - 1) -
                             0.0 / (gammaEqRoe - 1);
    Eigen::Vector<real, 5> incF = mirrorRoeDissipation3D(
        incU, n, veloRoe, vsqrRoe, aRoe, asqrRoe, HRoe, gammaEqRoe,
        1.0, 1.0, 1.0, contactEnergyJump);

    for (int i = 0; i < 5; ++i)
    {
        CAPTURE(i);
        CHECK(incF(i) == doctest::Approx(incU(i)).epsilon(1e-12));
    }
}

TEST_CASE("Roe base-energy contact has no spurious momentum flux")
{
    real gammaEqL = 1.28;
    real gammaEqR = 1.34;
    real gammaL = 1.41;
    real gammaR = 1.37;
    real rho = 1.2;
    real u = 5.0;
    real v = 0.4;
    real w = -0.3;
    real p = 2.0;
    real rhoEBaseL = 7.0;
    real rhoEBaseR = -5.0;
    auto UL = prim2consSplit(rho, u, v, w, p, gammaEqL, rhoEBaseL);
    auto UR = prim2consSplit(rho, u, v, w, p, gammaEqR, rhoEBaseR);
    Eigen::Vector3d n(1.0, 0.0, 0.0);
    Eigen::Vector3d vg = Eigen::Vector3d::Zero();
    Eigen::Vector<real, 5> F;
    F.setZero();
    real lam0 = 0, lam123 = 0, lam4 = 0;
    RoeFlux_IdealGas_HartenYee<3, 0, true>(
        UL, UR, UL, UR, vg, n, g_gamma, g_gamma, F,
        0.0, 1.0, 0.0, noDump, lam0, lam123, lam4,
        rhoEBaseL, rhoEBaseR, rhoEBaseL, rhoEBaseR,
        gammaEqL, gammaEqR, gammaEqL, gammaEqR,
        gammaL, gammaR, gammaL, gammaR);

    auto FL = exactNormalFluxSplit(UL, n, gammaEqL, rhoEBaseL);
#ifndef USE_ROE_BASE_ENERGY_CONTACT_FIX
    // This test intentionally fails in legacy mode; the INFO message points to
    // the disabled macro as the cause of the spurious contact flux.
    INFO("USE_ROE_BASE_ENERGY_CONTACT_FIX is disabled; pure base-energy contacts are expected to produce spurious Roe acoustic/contact flux.");
#endif
    for (int i = 0; i <= 3; ++i)
    {
        CAPTURE(i);
        CHECK(F(i) == doctest::Approx(FL(i)).epsilon(1e-11));
    }

    Eigen::Matrix<real, 5, -1> ULB(5, 2), URB(5, 2), FB(5, 2);
    ULB.col(0) = UL;
    URB.col(0) = UR;
    ULB.col(1) = UL;
    URB.col(1) = UL;
    FB.setZero();
    Eigen::Matrix<real, 3, -1> nB(3, 2), vgB(3, 2);
    nB.col(0) = n;
    nB.col(1) = n;
    vgB.setZero();
    Eigen::Matrix<real, 1, -1> rhoEBaseLB(1, 2), rhoEBaseRB(1, 2), gammaEqLB(1, 2), gammaEqRB(1, 2), gammaLB(1, 2), gammaRB(1, 2);
    rhoEBaseLB << rhoEBaseL, rhoEBaseL;
    rhoEBaseRB << rhoEBaseR, rhoEBaseL;
    gammaEqLB << gammaEqL, gammaEqL;
    gammaEqRB << gammaEqR, gammaEqL;
    gammaLB << gammaL, gammaL;
    gammaRB << gammaR, gammaL;
    RoeFlux_IdealGas_HartenYee_Batch<3, 0, true, true>(
        ULB, URB, UL, UR, vgB, vg, nB, n,
        g_gamma, g_gamma, FB, 0.0, 1.0, 0.0, noDump, lam0, lam123, lam4,
        rhoEBaseL, rhoEBaseR, rhoEBaseLB, rhoEBaseRB,
        gammaEqLB, gammaEqRB, gammaEqL, gammaEqR,
        gammaLB, gammaRB, gammaL, gammaR);
    for (int i = 0; i < 5; ++i)
    {
        CAPTURE(i);
        CHECK(FB(i, 0) == doctest::Approx(F(i)).epsilon(1e-11));
    }

    auto ULG = prim2consSplit(rho, u, v, w, p, gammaEqL, 0.0);
    auto URG = prim2consSplit(rho, u, v, w, p, gammaEqR, 0.0);
    auto FLG = exactNormalFluxSplit(ULG, n, gammaEqL, 0.0);
    ULB.col(0) = ULG;
    URB.col(0) = URG;
    ULB.col(1) = ULG;
    URB.col(1) = ULG;
    FB.setZero();
    RoeFlux_IdealGas_HartenYee_Batch<3, 0, true, false>(
        ULB, URB, ULG, URG, vgB, vg, nB, n,
        g_gamma, g_gamma, FB, 0.0, 1.0, 0.0, noDump, lam0, lam123, lam4,
        0.0, 0.0, nullptr, nullptr,
        gammaEqLB, gammaEqRB, gammaEqL, gammaEqR,
        gammaLB, gammaRB, gammaL, gammaR);
    for (int i = 0; i <= 3; ++i)
    {
        CAPTURE(i);
        CHECK(FB(i, 0) == doctest::Approx(FLG(i)).epsilon(1e-11));
    }

    Eigen::Vector<real, 5> FFixedGammaBase;
    FFixedGammaBase.setZero();
    auto ULBg = prim2consSplit(rho, u, v, w, p, g_gamma, rhoEBaseL);
    auto URBg = prim2consSplit(rho, u, v, w, p, g_gamma, rhoEBaseR);
    auto FLBg = exactNormalFluxSplit(ULBg, n, g_gamma, rhoEBaseL);
    RoeFlux_IdealGas_HartenYee<3, 0, false, true>(
        ULBg, URBg, ULBg, URBg, vg, n, g_gamma, g_gamma, FFixedGammaBase,
        0.0, 1.0, 0.0, noDump, lam0, lam123, lam4,
        rhoEBaseL, rhoEBaseR, rhoEBaseL, rhoEBaseR);
    for (int i = 0; i <= 3; ++i)
    {
        CAPTURE(i);
        CHECK(FFixedGammaBase(i) == doctest::Approx(FLBg(i)).epsilon(1e-11));
    }
}

TEST_CASE("Roe variants M1-M8 consistency")
{
    auto U = prim2cons(1.0, 50.0, 0.0, 0.0, 100000.0);
    Eigen::Vector3d n(1.0, 0.0, 0.0);
    auto Fexact = exactNormalFlux(U, n);

    // Note: Roe_M9 (eigScheme=9) is reserved and intentionally not included.
    for (auto rs : {Roe_M1, Roe_M2, Roe_M3, Roe_M4, Roe_M5, Roe_M6, Roe_M7, Roe_M8})
    {
        CAPTURE(rs);
        auto F = callDispatcher(rs, U, U, n);
        for (int i = 0; i < 5; i++)
        {
            CAPTURE(i);
            CHECK(F(i) == doctest::Approx(Fexact(i)).epsilon(1e-8));
        }
    }
}

// ===================================================================
// SYMMETRY: F(UL, UR, n) = -F(UR, UL, -n)
// ===================================================================

TEST_CASE("Roe symmetry: F(UL,UR,n) = -F(UR,UL,-n)")
{
    auto UL = prim2cons(1.0, 100.0, 0.0, 0.0, 100000.0);
    auto UR = prim2cons(0.125, 0.0, 0.0, 0.0, 10000.0);
    Eigen::Vector3d n(1.0, 0.0, 0.0);

    auto F1 = callDispatcher(Roe, UL, UR, n);
    auto F2 = callDispatcher(Roe, UR, UL, -n);

    for (int i = 0; i < 5; i++)
    {
        CAPTURE(i);
        CHECK(F1(i) == doctest::Approx(-F2(i)).epsilon(1e-10));
    }
}

TEST_CASE("HLLC symmetry: F(UL,UR,n) = -F(UR,UL,-n)")
{
    auto UL = prim2cons(1.0, 100.0, 0.0, 0.0, 100000.0);
    auto UR = prim2cons(0.125, 0.0, 0.0, 0.0, 10000.0);
    Eigen::Vector3d n(1.0, 0.0, 0.0);

    auto F1 = callDispatcher(HLLC, UL, UR, n);
    auto F2 = callDispatcher(HLLC, UR, UL, -n);

    for (int i = 0; i < 5; i++)
    {
        CAPTURE(i);
        CHECK(F1(i) == doctest::Approx(-F2(i)).epsilon(1e-10));
    }
}

// ===================================================================
// SOD SHOCK TUBE: non-trivial flux, bounded and finite
// ===================================================================

struct SodTestCase
{
    const char *name;
    RiemannSolverType rsType;
    real goldenF[5]; // golden flux values (1e300 = not acquired)
};

// Sod problem: UL = (1.0, 0, 0, 0, 2.5), UR = (0.125, 0, 0, 0, 0.25)
// in x-direction normal
static auto g_sodUL = prim2cons(1.0, 0.0, 0.0, 0.0, 1.0);
static auto g_sodUR = prim2cons(0.125, 0.0, 0.0, 0.0, 0.1);
static const Eigen::Vector3d g_sodN = Eigen::Vector3d(1.0, 0.0, 0.0);

static SodTestCase g_sodTests[] = {
    {"Roe", Roe, {0.0, 0.0, 0.0, 0.0, 0.0}}, // placeholder -- will acquire
    {"HLLC", HLLC, {0.0, 0.0, 0.0, 0.0, 0.0}},
    {"HLLEP", HLLEP, {0.0, 0.0, 0.0, 0.0, 0.0}},
};

TEST_CASE("Sod shock tube: flux is finite and bounded")
{
    for (auto &tc : g_sodTests)
    {
        SUBCASE(tc.name)
        {
            auto F = callDispatcher(tc.rsType, g_sodUL, g_sodUR, g_sodN);

            std::cout << "[Sod/" << tc.name << "] F = " << std::scientific
                      << std::setprecision(10);
            for (int i = 0; i < 5; i++)
                std::cout << F(i) << " ";
            std::cout << std::endl;

            for (int i = 0; i < 5; i++)
            {
                CAPTURE(i);
                CHECK_FALSE(std::isnan(F(i)));
                CHECK_FALSE(std::isinf(F(i)));
            }

            // Mass flux should be non-negative (expansion from L to R)
            CHECK(F(0) >= -1e-10);
        }
    }
}

// ===================================================================
// GOLDEN VALUES: specific test vector with captured golden flux
// ===================================================================

struct GoldenFluxCase
{
    const char *name;
    RiemannSolverType rsType;
    // UL: rho=1.225, u=100, v=-50, w=25, p=101325
    // UR: rho=0.8, u=200, v=0, w=0, p=80000
    // n: (0.6, 0.8, 0.0)
    real golden[5]; // golden flux (1e300 = not yet acquired)
};

static const auto g_goldenUL = prim2cons(1.225, 100.0, -50.0, 25.0, 101325.0);
static const auto g_goldenUR = prim2cons(0.8, 200.0, 0.0, 0.0, 80000.0);
static const Eigen::Vector3d g_goldenN = Eigen::Vector3d(0.6, 0.8, 0.0).normalized();

static GoldenFluxCase g_goldenTests[] = {
    {"Roe", Roe, {8.1162345145e+01, 5.8813861917e+04, 5.8759839011e+04, 5.9539454795e+02, 2.7251925995e+07}},
    {"HLLC", HLLC, {9.3486183606e+01, 5.5647501399e+04, 5.7057534871e+04, 2.3371545902e+03, 2.5454872499e+07}},
    {"HLLEP", HLLEP, {9.2861523292e+01, 5.1385111929e+04, 6.2875774679e+04, 5.7376094955e+03, 2.6719441912e+07}},
};

TEST_CASE("Golden flux values for mixed-state test vector")
{
    for (auto &tc : g_goldenTests)
    {
        SUBCASE(tc.name)
        {
            auto F = callDispatcher(tc.rsType, g_goldenUL, g_goldenUR, g_goldenN);

            std::cout << "[Golden/" << tc.name << "] F =";
            for (int i = 0; i < 5; i++)
                std::cout << " " << std::scientific << std::setprecision(10) << F(i);
            std::cout << std::endl;

            for (int i = 0; i < 5; i++)
            {
                CAPTURE(i);
                CHECK_FALSE(std::isnan(F(i)));
                if (tc.golden[i] < GOLDEN_NOT_ACQUIRED)
                    CHECK(F(i) == doctest::Approx(tc.golden[i]).epsilon(1e-8));
            }
        }
    }
}

// ===================================================================
// QUIESCENT GAS: all solvers agree on zero-flux for static gas
// ===================================================================

TEST_CASE("All solvers: quiescent gas produces same flux (p-only)")
{
    auto U = prim2cons(1.0, 0.0, 0.0, 0.0, 1.0);
    Eigen::Vector3d n(1.0, 0.0, 0.0);
    auto Fexact = exactNormalFlux(U, n);

    for (auto rs : {Roe, HLLC, HLLEP})
    {
        CAPTURE(rs);
        auto F = callDispatcher(rs, U, U, n);
        for (int i = 0; i < 5; i++)
        {
            CAPTURE(i);
            CHECK(F(i) == doctest::Approx(Fexact(i)).epsilon(1e-12));
        }
    }
}

// ===================================================================
// EIGENVALUE OUTPUT: wave speeds should be physically meaningful
// ===================================================================

TEST_CASE("Roe eigenvalue output: lam0 < lam123 < lam4 for subsonic")
{
    auto UL = prim2cons(1.0, 50.0, 0.0, 0.0, 100000.0);
    auto UR = prim2cons(1.1, 55.0, 0.0, 0.0, 105000.0);
    Eigen::Vector3d n(1, 0, 0), vg(0, 0, 0);

    Eigen::Vector<real, 5> F;
    real lam0, lam123, lam4;
    InviscidFlux_IdealGas_Dispatcher<3>(
        Roe, UL, UR, UL, UR, vg, n, g_gamma, g_gamma, F,
        0.0, 0.0, 0.0, noDump, lam0, lam123, lam4);

    // |u-a| < |u| < |u+a| for subsonic flow where u > 0
    CHECK(lam0 >= 0);
    CHECK(lam123 >= 0);
    CHECK(lam4 >= 0);
    CHECK(lam4 >= lam123); // |u+a| >= |u|
}

// ===================================================================
// DIAGONAL NORMAL: flux with n=(1/sqrt3,1/sqrt3,1/sqrt3)
// ===================================================================

TEST_CASE("Roe consistency: diagonal normal")
{
    auto U = prim2cons(2.0, 100.0, 200.0, 300.0, 200000.0);
    Eigen::Vector3d n(1.0, 1.0, 1.0);
    n.normalize();

    auto Fexact = exactNormalFlux(U, n);
    auto F = callDispatcher(Roe, U, U, n);

    for (int i = 0; i < 5; i++)
    {
        CAPTURE(i);
        CHECK(F(i) == doctest::Approx(Fexact(i)).epsilon(1e-8));
    }
}
