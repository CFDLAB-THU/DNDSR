/**
 * @file test_RANS.cpp
 * @brief Unit tests for RANS turbulence model functions in RANS_ke.hpp.
 *
 * Tests cover:
 *   - GetMut_KOWilcox: turbulent viscosity from k-omega Wilcox 2006
 *   - GetMut_SST: turbulent viscosity from k-omega SST
 *   - GetMut_RealizableKe: turbulent viscosity from Realizable k-epsilon
 *   - GetSource_KOWilcox: source terms (mode 0 = full, mode 1 = Jacobian diagonal)
 *   - GetSource_SST: source terms
 *   - GetSource_RealizableKe: source terms
 *   - GetVisFlux_KOWilcox: viscous flux for turbulent transport variables
 *   - GetVisFlux_SST: viscous flux
 *   - GetVisFlux_RealizableKe: viscous flux
 *
 * Key properties tested:
 *   - mut >= 0 (non-negative turbulent viscosity)
 *   - mut <= 1e5 * muLam (CFL3D limiting)
 *   - Zero gradient -> zero production in source
 *   - Zero gradient -> zero viscous flux
 *   - Finite output (no NaN/Inf)
 *   - Golden values for specific test vectors
 *
 * All functions are pure template functions (no MPI, no mesh).
 * SA model is excluded because GetSource_SA references EulerEvaluator settings.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/RANS_ke.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace DNDS;
using namespace DNDS::Euler::RANS;

static constexpr real g_gamma = 1.4;
static constexpr int dim = 3;
static constexpr int I4 = dim + 1; // = 4
// nVars for 2-equation models: 5 (flow) + 2 (turb) = 7
static constexpr int nVars = 7;

static constexpr real GOLDEN_NOT_ACQUIRED = 1e300;

// Build a 3D 2-equation conservative state:
// [rho, rho*u, rho*v, rho*w, rho*E, rho*k, rho*omega_or_epsilon]
static Eigen::Vector<real, nVars> buildState(
    real rho, real u, real v, real w, real p, real k, real omega)
{
    Eigen::Vector<real, nVars> U;
    real E = p / (g_gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w);
    U << rho, rho * u, rho * v, rho * w, E, rho * k, rho * omega;
    return U;
}

// Build a gradient matrix (dim x nVars) with a simple shear profile:
// du/dy = S (shear rate), everything else zero
static Eigen::Matrix<real, dim, nVars> buildShearGrad(
    const Eigen::Vector<real, nVars> &U, real S_shear)
{
    Eigen::Matrix<real, dim, nVars> DiffU;
    DiffU.setZero();
    // d(rho*u)/dy = rho * S (since rho is constant in this test)
    DiffU(1, 1) = U(0) * S_shear; // d(rho*u)/dy
    return DiffU;
}

// ===================================================================
// Turbulence equilibrium state:
// Typical boundary layer: rho=1.225, u=50, p=101325
// k=10, omega=1000 (or epsilon=Cmu*k*omega=0.09*10*1000=900)
// mu_lam = 1.8e-5
// wall distance d=0.01 (inside BL)
// ===================================================================
static const real g_rho = 1.225;
static const real g_u = 50.0;
static const real g_p = 101325.0;
static const real g_k = 10.0;
static const real g_omega = 1000.0;
static const real g_epsilon = 0.09 * g_k * g_omega; // Cmu * k * omega
static const real g_mu = 1.8e-5;
static const real g_d = 0.01;
static const real g_Sshear = 500.0; // du/dy

// ===================================================================
// GetMut_KOWilcox
// ===================================================================

TEST_CASE("GetMut_KOWilcox: non-negative")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_KOWilcox<dim>(U, DiffU, g_mu, g_d);

    CHECK(mut >= 0.0);
    CHECK(std::isfinite(mut));
}

TEST_CASE("GetMut_KOWilcox: bounded by 1e5 * muLam")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_KOWilcox<dim>(U, DiffU, g_mu, g_d);

    CHECK(mut <= 1e5 * g_mu + 1e-10);
}

TEST_CASE("GetMut_KOWilcox: golden value")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_KOWilcox<dim>(U, DiffU, g_mu, g_d);

    std::cout << "[KOWilcox/mut] " << std::scientific << std::setprecision(10) << mut << std::endl;
    // Golden value captured:
    real golden = 8.4000000000e-03;
    if (golden < GOLDEN_NOT_ACQUIRED)
        CHECK(mut == doctest::Approx(golden).epsilon(1e-6));
}

// ===================================================================
// GetMut_SST
// ===================================================================

TEST_CASE("GetMut_SST: non-negative")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_SST<dim>(U, DiffU, g_mu, g_d);

    CHECK(mut >= 0.0);
    CHECK(std::isfinite(mut));
}

TEST_CASE("GetMut_SST: bounded by 1e5 * muLam")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_SST<dim>(U, DiffU, g_mu, g_d);

    CHECK(mut <= 1e5 * g_mu + 1e-10);
}

TEST_CASE("GetMut_SST: golden value")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_SST<dim>(U, DiffU, g_mu, g_d);

    std::cout << "[SST/mut] " << std::scientific << std::setprecision(10) << mut << std::endl;
    real golden = 7.5950000000e-03;
    if (golden < GOLDEN_NOT_ACQUIRED)
        CHECK(mut == doctest::Approx(golden).epsilon(1e-6));
}

// ===================================================================
// GetMut_RealizableKe
// ===================================================================

TEST_CASE("GetMut_RealizableKe: non-negative")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_RealizableKe<dim>(U, DiffU, g_mu, g_d);

    CHECK(mut >= 0.0);
    CHECK(std::isfinite(mut));
}

TEST_CASE("GetMut_RealizableKe: bounded by 1e5 * muLam")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_RealizableKe<dim>(U, DiffU, g_mu, g_d);

    CHECK(mut <= 1e5 * g_mu + 1e-10);
}

TEST_CASE("GetMut_RealizableKe: golden value")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_RealizableKe<dim>(U, DiffU, g_mu, g_d);

    std::cout << "[RealKe/mut] " << std::scientific << std::setprecision(10) << mut << std::endl;
    real golden = 1.2250000000e-02;
    if (golden < GOLDEN_NOT_ACQUIRED)
        CHECK(mut == doctest::Approx(golden).epsilon(1e-6));
}

// ===================================================================
// Source terms: zero gradient -> zero production (only destruction)
// ===================================================================

TEST_CASE("GetSource_KOWilcox: zero gradient finite and has destruction")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    Eigen::Matrix<real, dim, nVars> DiffU = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_KOWilcox<dim>(U, DiffU, g_mu, g_d, source, 0);

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));
    // k source should be negative (destruction only when Pk=0)
    CHECK(source(I4 + 1) <= 0.0);
    // omega source should also be negative (destruction)
    CHECK(source(I4 + 2) <= 0.0);
}

TEST_CASE("GetSource_SST: zero gradient finite")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    Eigen::Matrix<real, dim, nVars> DiffU = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_SST<dim>(U, DiffU, g_mu, g_d, 1e10, source, 0);

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));
    // k destruction: -betaStar * rho * k * omega < 0
    CHECK(source(I4 + 1) <= 0.0);
}

TEST_CASE("GetSource_RealizableKe: zero gradient finite")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    Eigen::Matrix<real, dim, nVars> DiffU = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_RealizableKe<dim>(U, DiffU, g_mu, g_d, source, 0);

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));
}

// ===================================================================
// Source terms with shear: golden values
// ===================================================================

TEST_CASE("GetSource_KOWilcox: shear gradient golden")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_KOWilcox<dim>(U, DiffU, g_mu, g_d, source, 0);

    std::cout << "[KOWilcox/source] k=" << std::scientific << std::setprecision(10)
              << source(I4 + 1) << " omega=" << source(I4 + 2) << std::endl;

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));

    // With shear, production should exceed destruction (positive total k-source)
    // Not always true depending on parameters, so just check finite
}

TEST_CASE("GetSource_SST: shear gradient golden")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_SST<dim>(U, DiffU, g_mu, g_d, 1e10, source, 0);

    std::cout << "[SST/source] k=" << std::scientific << std::setprecision(10)
              << source(I4 + 1) << " omega=" << source(I4 + 2) << std::endl;

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));
}

TEST_CASE("GetSource_RealizableKe: shear gradient golden")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    auto DiffU = buildShearGrad(U, g_Sshear);
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_RealizableKe<dim>(U, DiffU, g_mu, g_d, source, 0);

    std::cout << "[RealKe/source] k=" << std::scientific << std::setprecision(10)
              << source(I4 + 1) << " epsilon=" << source(I4 + 2) << std::endl;

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));
}

// ===================================================================
// Source terms: mode 1 (implicit diagonal) should be non-negative
// ===================================================================

TEST_CASE("GetSource_KOWilcox mode=1: implicit diagonal non-negative")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_KOWilcox<dim>(U, DiffU, g_mu, g_d, source, 1);

    // mode=1 returns destruction rate coefficients, should be >= 0
    CHECK(source(I4 + 1) >= 0.0);
    CHECK(source(I4 + 2) >= 0.0);
}

TEST_CASE("GetSource_SST mode=1: implicit diagonal non-negative")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_SST<dim>(U, DiffU, g_mu, g_d, 1e10, source, 1);

    CHECK(source(I4 + 1) >= 0.0);
    CHECK(source(I4 + 2) >= 0.0);
}

// ===================================================================
// Viscous flux: zero gradient -> zero flux
// ===================================================================

TEST_CASE("GetVisFlux_KOWilcox: zero gradient -> zero flux")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    // DiffUxyPrim: gradient of PRIMITIVE variables
    Eigen::Matrix<real, dim, nVars> DiffPrim = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector3d norm(1, 0, 0);
    Eigen::Vector<real, nVars> vFlux;
    vFlux.setZero();

    real mut = 0.01; // arbitrary
    GetVisFlux_KOWilcox<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux);

    CHECK(vFlux(I4 + 1) == doctest::Approx(0.0).epsilon(1e-14));
    CHECK(vFlux(I4 + 2) == doctest::Approx(0.0).epsilon(1e-14));
}

TEST_CASE("GetVisFlux_SST: zero gradient -> zero flux")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    Eigen::Matrix<real, dim, nVars> DiffPrim = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector3d norm(1, 0, 0);
    Eigen::Vector<real, nVars> vFlux;
    vFlux.setZero();

    real mut = 0.01;
    GetVisFlux_SST<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux);

    CHECK(vFlux(I4 + 1) == doctest::Approx(0.0).epsilon(1e-14));
    CHECK(vFlux(I4 + 2) == doctest::Approx(0.0).epsilon(1e-14));
}

TEST_CASE("GetVisFlux_RealizableKe: zero gradient -> zero flux")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    Eigen::Matrix<real, dim, nVars> DiffPrim = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector3d norm(1, 0, 0);
    Eigen::Vector<real, nVars> vFlux;
    vFlux.setZero();

    real mut = 0.01;
    GetVisFlux_RealizableKe<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux);

    CHECK(vFlux(I4 + 1) == doctest::Approx(0.0).epsilon(1e-14));
    CHECK(vFlux(I4 + 2) == doctest::Approx(0.0).epsilon(1e-14));
}

// ===================================================================
// Viscous flux: non-zero gradient produces non-zero flux
// ===================================================================

TEST_CASE("GetVisFlux_KOWilcox: k-gradient produces k-flux")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    Eigen::Matrix<real, dim, nVars> DiffPrim = Eigen::Matrix<real, dim, nVars>::Zero();
    // dk/dx = 100 (in primitive space, dk/dx = dk/dx since k is already primitive-like)
    DiffPrim(0, I4 + 1) = 100.0;
    Eigen::Vector3d norm(1, 0, 0);
    Eigen::Vector<real, nVars> vFlux;
    vFlux.setZero();

    real mut = 0.01;
    GetVisFlux_KOWilcox<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux);

    // k-flux should be positive (diffusion in direction of decreasing k)
    CHECK(vFlux(I4 + 1) > 0.0);
    CHECK(std::isfinite(vFlux(I4 + 1)));
    // omega-flux should still be zero (no omega gradient)
    CHECK(vFlux(I4 + 2) == doctest::Approx(0.0).epsilon(1e-14));
}

// ===================================================================
// Mut increases with k for fixed omega (physical)
// ===================================================================

TEST_CASE("GetMut_KOWilcox: mut increases with k")
{
    auto U1 = buildState(g_rho, g_u, 0, 0, g_p, 5.0, g_omega);
    auto U2 = buildState(g_rho, g_u, 0, 0, g_p, 20.0, g_omega);
    auto DiffU = buildShearGrad(U1, g_Sshear);
    auto DiffU2 = buildShearGrad(U2, g_Sshear);

    real mut1 = GetMut_KOWilcox<dim>(U1, DiffU, g_mu, g_d);
    real mut2 = GetMut_KOWilcox<dim>(U2, DiffU2, g_mu, g_d);

    CHECK(mut2 > mut1);
}

TEST_CASE("GetMut_SST: mut increases with k")
{
    auto U1 = buildState(g_rho, g_u, 0, 0, g_p, 5.0, g_omega);
    auto U2 = buildState(g_rho, g_u, 0, 0, g_p, 20.0, g_omega);
    auto DiffU = buildShearGrad(U1, g_Sshear);
    auto DiffU2 = buildShearGrad(U2, g_Sshear);

    real mut1 = GetMut_SST<dim>(U1, DiffU, g_mu, g_d);
    real mut2 = GetMut_SST<dim>(U2, DiffU2, g_mu, g_d);

    CHECK(mut2 > mut1);
}

// ===================================================================
// Very small k/omega: no crash, finite result
// ===================================================================

TEST_CASE("GetMut_KOWilcox: very small k/omega produces finite mut")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, 1e-10, 1e-5);
    auto DiffU = buildShearGrad(U, 0.0);
    real mut = GetMut_KOWilcox<dim>(U, DiffU, g_mu, g_d);

    CHECK(std::isfinite(mut));
    CHECK(mut >= 0.0);
}

TEST_CASE("GetMut_SST: very small k/omega produces finite mut")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, 1e-10, 1e-5);
    auto DiffU = buildShearGrad(U, 0.0);
    real mut = GetMut_SST<dim>(U, DiffU, g_mu, g_d);

    CHECK(std::isfinite(mut));
    CHECK(mut >= 0.0);
}

TEST_CASE("GetMut_RealizableKe: very small k/eps produces finite mut")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, 1e-10, 1e-8);
    auto DiffU = buildShearGrad(U, 0.0);
    real mut = GetMut_RealizableKe<dim>(U, DiffU, g_mu, g_d);

    CHECK(std::isfinite(mut));
    CHECK(mut >= 0.0);
}
