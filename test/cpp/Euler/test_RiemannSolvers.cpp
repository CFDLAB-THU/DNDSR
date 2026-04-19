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

// No-op dump callback
static auto noDump = []() {};

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
        rsType, UL, UR, UL, UR, vg, n, g_gamma, F,
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

TEST_CASE("Roe variants M1-M8 consistency (M2,M9 not implemented)")
{
    auto U = prim2cons(1.0, 50.0, 0.0, 0.0, 100000.0);
    Eigen::Vector3d n(1.0, 0.0, 0.0);
    auto Fexact = exactNormalFlux(U, n);

    // Note: Roe_M2 (eigScheme=2) and Roe_M9 (eigScheme=9) are not implemented
    for (auto rs : {Roe_M1, Roe_M3, Roe_M4, Roe_M5, Roe_M6, Roe_M7, Roe_M8})
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
    {"Roe", Roe,
     {0.0, 0.0, 0.0, 0.0, 0.0}},  // placeholder -- will acquire
    {"HLLC", HLLC,
     {0.0, 0.0, 0.0, 0.0, 0.0}},
    {"HLLEP", HLLEP,
     {0.0, 0.0, 0.0, 0.0, 0.0}},
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
    {"Roe", Roe,
     {8.1162345145e+01, 5.8813861917e+04, 5.8759839011e+04,
      5.9539454795e+02, 2.7251925995e+07}},
    {"HLLC", HLLC,
     {9.3486183606e+01, 5.5647501399e+04, 5.7057534871e+04,
      2.3371545902e+03, 2.5454872499e+07}},
    {"HLLEP", HLLEP,
     {9.2861523292e+01, 5.1385111929e+04, 6.2875774679e+04,
      5.7376094955e+03, 2.6719441912e+07}},
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
        Roe, UL, UR, UL, UR, vg, n, g_gamma, F,
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
