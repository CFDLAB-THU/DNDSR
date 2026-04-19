/**
 * @file test_GasThermo.cpp
 * @brief Unit tests for ideal-gas thermodynamics and eigenvector routines in Gas.hpp.
 *
 * Tests cover:
 *   - IdealGasThermal: pressure, speed-of-sound, enthalpy from conservative
 *   - Conservative2Primitive / Primitive2Conservative round-trip (2D and 3D)
 *   - PrimitiveGetP0T0: stagnation quantities
 *   - EulerGasRightEigenVector / LeftEigenVector: L*R = I orthogonality (3D)
 *   - IdealGas_EulerGas{Right,Left}EigenVector convenience wrappers
 *   - IdealGasUIncrement: velocity/pressure increments from conservative increments
 *   - GasInviscidFlux / GasInviscidFlux_XY: inviscid flux computation
 *   - GradientCons2Prim_IdealGas: gradient transformation
 *   - GetRoeAverage: Roe-averaged state properties
 *   - IdealGasGetCompressionRatioPressure: positivity limiter
 *
 * All functions are pure (no MPI, no mesh). Serial doctest.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/Gas.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using namespace DNDS;
using namespace DNDS::Euler::Gas;

// Standard air at rest: rho=1.0, p=1/gamma (so a=1), gamma=1.4
static constexpr real g_gamma = 1.4;

// Build a 3D conservative state from primitive [rho, u, v, w, p]
static Eigen::Vector<real, 5> primToCons3D(real rho, real u, real v, real w, real p)
{
    Eigen::Vector<real, 5> U;
    real E = p / (g_gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w);
    U << rho, rho * u, rho * v, rho * w, E;
    return U;
}

// Build a 2D conservative state from primitive [rho, u, v, p]
[[maybe_unused]] static Eigen::Vector<real, 4> primToCons2D(real rho, real u, real v, real p)
{
    Eigen::Vector<real, 4> U;
    real E = p / (g_gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    U << rho, rho * u, rho * v, E;
    return U;
}

// ===================================================================
// IdealGasThermal
// ===================================================================

TEST_CASE("IdealGasThermal: standard quiescent air")
{
    // rho=1.0, p=1.0/1.4, v=0 => a^2 = gamma*p/rho = 1.0
    real rho = 1.0, p_expected = 1.0 / g_gamma;
    real E = p_expected / (g_gamma - 1.0); // rho*E = p/(gamma-1)
    real vSqr = 0.0;

    real p, asqr, H;
    IdealGasThermal(E, rho, vSqr, g_gamma, p, asqr, H);

    CHECK(p == doctest::Approx(p_expected).epsilon(1e-14));
    CHECK(asqr == doctest::Approx(1.0).epsilon(1e-14));
    CHECK(H == doctest::Approx((E + p) / rho).epsilon(1e-14));
}

TEST_CASE("IdealGasThermal: Mach 2 flow")
{
    real rho = 1.0, u = 2.0, p_expected = 1.0;
    real vSqr = u * u;
    real E = p_expected / (g_gamma - 1.0) + rho * 0.5 * vSqr;

    real p, asqr, H;
    IdealGasThermal(E, rho, vSqr, g_gamma, p, asqr, H);

    CHECK(p == doctest::Approx(p_expected).epsilon(1e-14));
    CHECK(asqr == doctest::Approx(g_gamma * p_expected / rho).epsilon(1e-14));
    real H_expected = (E + p_expected) / rho;
    CHECK(H == doctest::Approx(H_expected).epsilon(1e-14));
}

// ===================================================================
// Conservative <-> Primitive round-trip
// ===================================================================

TEST_CASE("Cons2Prim and Prim2Cons round-trip: 3D")
{
    Eigen::Vector<real, 5> prim0;
    prim0 << 1.225, 100.0, 50.0, -30.0, 101325.0; // air at ~Mach 0.3

    Eigen::Vector<real, 5> U, prim1;
    IdealGasThermalPrimitive2Conservative<3>(prim0, U, g_gamma);
    IdealGasThermalConservative2Primitive<3>(U, prim1, g_gamma);

    for (int i = 0; i < 5; i++)
    {
        CAPTURE(i);
        CHECK(prim1(i) == doctest::Approx(prim0(i)).epsilon(1e-12));
    }
}

TEST_CASE("Cons2Prim and Prim2Cons round-trip: 2D")
{
    Eigen::Vector<real, 4> prim0;
    prim0 << 0.5, 300.0, -100.0, 50000.0;

    Eigen::Vector<real, 4> U, prim1;
    IdealGasThermalPrimitive2Conservative<2>(prim0, U, g_gamma);
    IdealGasThermalConservative2Primitive<2>(U, prim1, g_gamma);

    for (int i = 0; i < 4; i++)
    {
        CAPTURE(i);
        CHECK(prim1(i) == doctest::Approx(prim0(i)).epsilon(1e-12));
    }
}

TEST_CASE("Prim2Cons: known state verification")
{
    // rho=2, u=3, v=0, w=0, p=10, gamma=1.4
    // E = p/(gamma-1) + 0.5*rho*vSqr = 10/0.4 + 0.5*2*9 = 25 + 9 = 34
    Eigen::Vector<real, 5> prim;
    prim << 2.0, 3.0, 0.0, 0.0, 10.0;
    Eigen::Vector<real, 5> U;
    IdealGasThermalPrimitive2Conservative<3>(prim, U, g_gamma);

    CHECK(U(0) == doctest::Approx(2.0));       // rho
    CHECK(U(1) == doctest::Approx(6.0));       // rho*u
    CHECK(U(2) == doctest::Approx(0.0));       // rho*v
    CHECK(U(3) == doctest::Approx(0.0));       // rho*w
    CHECK(U(4) == doctest::Approx(34.0));      // rho*E
}

// ===================================================================
// PrimitiveGetP0T0: stagnation quantities
// ===================================================================

TEST_CASE("PrimitiveGetP0T0: quiescent gas")
{
    // At rest: p0 = p, T0 = T
    Eigen::Vector<real, 5> prim;
    prim << 1.0, 0.0, 0.0, 0.0, 100000.0;
    real Rgas = 287.0;

    auto [p0, T0] = IdealGasThermalPrimitiveGetP0T0<3>(prim, g_gamma, Rgas);
    real T = prim(4) / (prim(0) * Rgas);

    CHECK(p0 == doctest::Approx(100000.0).epsilon(1e-10));
    CHECK(T0 == doctest::Approx(T).epsilon(1e-10));
}

TEST_CASE("PrimitiveGetP0T0: p0 > p for moving gas")
{
    Eigen::Vector<real, 5> prim;
    prim << 1.225, 300.0, 0.0, 0.0, 101325.0;
    real Rgas = 287.0;

    auto [p0, T0] = IdealGasThermalPrimitiveGetP0T0<3>(prim, g_gamma, Rgas);
    real T = prim(4) / (prim(0) * Rgas);

    CHECK(p0 > prim(4));
    CHECK(T0 > T);
}

// ===================================================================
// Eigenvectors: L * R = I
// ===================================================================

TEST_CASE("EulerGas eigenvectors: L*R = I for 3D")
{
    Eigen::Vector3d velo(1.5, -0.3, 0.7);
    real Vsqr = velo.squaredNorm();
    // Compute p, a, H from a state
    real rho = 1.0, p = 1.0;
    real E = p / (g_gamma - 1.0) + 0.5 * rho * Vsqr;
    real a = std::sqrt(g_gamma * p / rho);
    real H = (E + p) / rho;

    Eigen::Matrix<real, 5, 5> R, L;
    R.setZero();
    L.setZero();
    EulerGasRightEigenVector<3>(velo, Vsqr, H, a, R);
    EulerGasLeftEigenVector<3>(velo, Vsqr, H, a, g_gamma, L);

    Eigen::Matrix<real, 5, 5> LR = L * R;
    Eigen::Matrix<real, 5, 5> I = Eigen::Matrix<real, 5, 5>::Identity();

    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
        {
            CAPTURE(i);
            CAPTURE(j);
            CHECK(LR(i, j) == doctest::Approx(I(i, j)).epsilon(1e-12));
        }
}

TEST_CASE("EulerGas eigenvectors: L*R = I for non-trivial velocity")
{
    Eigen::Vector3d velo(300.0, -150.0, 75.0);
    real Vsqr = velo.squaredNorm();
    real rho = 1.225, p = 101325.0;
    real E = p / (g_gamma - 1.0) + 0.5 * rho * Vsqr;
    real a = std::sqrt(g_gamma * p / rho);
    real H = (E + p) / rho;

    Eigen::Matrix<real, 5, 5> R, L;
    R.setZero();
    L.setZero();
    EulerGasRightEigenVector<3>(velo, Vsqr, H, a, R);
    EulerGasLeftEigenVector<3>(velo, Vsqr, H, a, g_gamma, L);

    Eigen::Matrix<real, 5, 5> LR = L * R;
    Eigen::Matrix<real, 5, 5> I = Eigen::Matrix<real, 5, 5>::Identity();

    real maxErr = (LR - I).cwiseAbs().maxCoeff();
    CHECK(maxErr < 1e-10);
}

TEST_CASE("IdealGas convenience eigenvector wrappers produce L*R=I")
{
    auto U = primToCons3D(1.225, 100.0, -50.0, 25.0, 101325.0);

    auto R = IdealGas_EulerGasRightEigenVector<3>(U, g_gamma);
    auto L = IdealGas_EulerGasLeftEigenVector<3>(U, g_gamma);

    auto LR = L * R;
    real maxErr = (LR - Eigen::Matrix<real, 5, 5>::Identity()).cwiseAbs().maxCoeff();
    CHECK(maxErr < 1e-10);
}

// ===================================================================
// GasInviscidFlux: x-direction flux
// ===================================================================

TEST_CASE("GasInviscidFlux: x-direction, quiescent gas")
{
    // At rest: flux = [0, p, 0, 0, 0]
    auto U = primToCons3D(1.0, 0.0, 0.0, 0.0, 1.0);
    Eigen::Vector3d velo(0, 0, 0), vg(0, 0, 0);
    real p = 1.0;

    Eigen::Vector<real, 5> F;
    GasInviscidFlux<3>(U, velo, vg, p, F);

    CHECK(F(0) == doctest::Approx(0.0).epsilon(1e-14));
    CHECK(F(1) == doctest::Approx(p).epsilon(1e-14));   // momentum flux = p
    CHECK(F(2) == doctest::Approx(0.0).epsilon(1e-14));
    CHECK(F(3) == doctest::Approx(0.0).epsilon(1e-14));
    CHECK(F(4) == doctest::Approx(0.0).epsilon(1e-14));
}

TEST_CASE("GasInviscidFlux: x-direction, moving gas")
{
    real rho = 2.0, u = 3.0, p = 10.0;
    auto U = primToCons3D(rho, u, 0.0, 0.0, p);
    Eigen::Vector3d velo(u, 0, 0), vg(0, 0, 0);

    Eigen::Vector<real, 5> F;
    GasInviscidFlux<3>(U, velo, vg, p, F);

    // F = [rho*u, rho*u^2+p, 0, 0, (E+p)*u]
    real E = U(4);
    CHECK(F(0) == doctest::Approx(rho * u).epsilon(1e-12));
    CHECK(F(1) == doctest::Approx(rho * u * u + p).epsilon(1e-12));
    CHECK(F(2) == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(F(3) == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(F(4) == doctest::Approx((E + p) * u).epsilon(1e-12));
}

// ===================================================================
// GasInviscidFlux_XY: normal-direction flux
// ===================================================================

TEST_CASE("GasInviscidFlux_XY: n=(1,0,0) equals GasInviscidFlux")
{
    auto U = primToCons3D(1.225, 100.0, -50.0, 25.0, 101325.0);
    Eigen::Vector3d velo = U.segment<3>(1) / U(0);
    Eigen::Vector3d vg(0, 0, 0);
    real p_val;
    {
        real asqr, H;
        IdealGasThermal(U(4), U(0), velo.squaredNorm(), g_gamma, p_val, asqr, H);
    }

    Eigen::Vector<real, 5> Fx, Fn;
    GasInviscidFlux<3>(U, velo, vg, p_val, Fx);
    Eigen::Vector3d nx(1, 0, 0);
    GasInviscidFlux_XY<3>(U, velo, vg, nx, p_val, Fn);

    for (int i = 0; i < 5; i++)
    {
        CAPTURE(i);
        CHECK(Fn(i) == doctest::Approx(Fx(i)).epsilon(1e-10));
    }
}

// ===================================================================
// IdealGasUIncrement
// ===================================================================

TEST_CASE("IdealGasUIncrement: zero increment gives zero output")
{
    auto U = primToCons3D(1.0, 100.0, -50.0, 25.0, 101325.0);
    Eigen::Vector<real, 5> dU = Eigen::Vector<real, 5>::Zero();
    Eigen::Vector3d velo = U.segment<3>(1) / U(0);
    Eigen::Vector3d dVelo;
    real dp;

    IdealGasUIncrement<3>(U, dU, velo, g_gamma, dVelo, dp);

    CHECK(dVelo.norm() == doctest::Approx(0.0).epsilon(1e-14));
    CHECK(dp == doctest::Approx(0.0).epsilon(1e-10));
}

TEST_CASE("IdealGasUIncrement: finite-difference verification")
{
    // IdealGasUIncrement is a linearization (exact for infinitesimal dU).
    // Use a very small perturbation to reduce second-order error.
    auto U0 = primToCons3D(1.225, 100.0, -50.0, 25.0, 101325.0);
    Eigen::Vector3d velo0 = U0.segment<3>(1) / U0(0);

    real eps = 1e-7;
    Eigen::Vector<real, 5> prim0;
    IdealGasThermalConservative2Primitive<3>(U0, prim0, g_gamma);

    Eigen::Vector<real, 5> prim1;
    prim1 << prim0(0) + 0.01 * eps, prim0(1) + 0.5 * eps,
             prim0(2) - 0.3 * eps, prim0(3) + 0.1 * eps,
             prim0(4) + 100.0 * eps;
    Eigen::Vector<real, 5> U1;
    IdealGasThermalPrimitive2Conservative<3>(prim1, U1, g_gamma);

    Eigen::Vector<real, 5> dU = U1 - U0;
    Eigen::Vector3d dVelo;
    real dp;
    IdealGasUIncrement<3>(U0, dU, velo0, g_gamma, dVelo, dp);

    Eigen::Vector3d dVeloExpected = prim1.segment<3>(1) - prim0.segment<3>(1);
    real dpExpected = prim1(4) - prim0(4);

    for (int i = 0; i < 3; i++)
    {
        CAPTURE(i);
        CHECK(dVelo(i) == doctest::Approx(dVeloExpected(i)).epsilon(1e-4));
    }
    CHECK(dp == doctest::Approx(dpExpected).epsilon(1e-3));
}

// ===================================================================
// GetRoeAverage
// ===================================================================

TEST_CASE("GetRoeAverage: identical states give same state")
{
    auto U = primToCons3D(1.225, 100.0, -50.0, 25.0, 101325.0);

    Eigen::Vector3d veloRoe;
    real vsqrRoe, aRoe, asqrRoe, HRoe;
    Eigen::Vector<real, 5> URoe;

    GetRoeAverage<3>(U, U, g_gamma, veloRoe, vsqrRoe, aRoe, asqrRoe, HRoe, URoe);

    Eigen::Vector3d velo = U.segment<3>(1) / U(0);
    real p_val, asqr_val, H_val;
    IdealGasThermal(U(4), U(0), velo.squaredNorm(), g_gamma, p_val, asqr_val, H_val);

    for (int i = 0; i < 3; i++)
    {
        CAPTURE(i);
        CHECK(veloRoe(i) == doctest::Approx(velo(i)).epsilon(1e-10));
    }
    CHECK(HRoe == doctest::Approx(H_val).epsilon(1e-10));
    CHECK(aRoe == doctest::Approx(std::sqrt(asqr_val)).epsilon(1e-10));
}

TEST_CASE("GetRoeAverage: density is geometric mean")
{
    auto UL = primToCons3D(1.0, 100.0, 0.0, 0.0, 100000.0);
    auto UR = primToCons3D(4.0, 100.0, 0.0, 0.0, 100000.0);

    Eigen::Vector3d veloRoe;
    real vsqrRoe, aRoe, asqrRoe, HRoe;
    Eigen::Vector<real, 5> URoe;

    GetRoeAverage<3>(UL, UR, g_gamma, veloRoe, vsqrRoe, aRoe, asqrRoe, HRoe, URoe);

    // Roe-averaged rho = sqrt(rhoL * rhoR) = sqrt(1*4) = 2
    CHECK(URoe(0) == doctest::Approx(2.0).epsilon(1e-10));
}

// ===================================================================
// GradientCons2Prim_IdealGas
// ===================================================================

TEST_CASE("GradientCons2Prim: zero gradient produces zero")
{
    auto U = primToCons3D(1.225, 100.0, -50.0, 25.0, 101325.0);
    Eigen::Matrix<real, 3, 5> GradU = Eigen::Matrix<real, 3, 5>::Zero();
    Eigen::Matrix<real, 3, 5> GradPrim;

    GradientCons2Prim_IdealGas<3>(U, GradU, GradPrim, g_gamma);

    CHECK(GradPrim.cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("GradientCons2Prim: finite-difference verification")
{
    // Base state
    Eigen::Vector<real, 5> prim0;
    prim0 << 1.225, 100.0, -50.0, 25.0, 101325.0;
    Eigen::Vector<real, 5> U0;
    IdealGasThermalPrimitive2Conservative<3>(prim0, U0, g_gamma);

    // Perturbed state (small perturbation in x-direction)
    real dx = 1e-6;
    Eigen::Vector<real, 5> prim1;
    prim1 << 1.225 + 0.01 * dx, 100.0 + 0.5 * dx, -50.0 - 0.3 * dx, 25.0 + 0.1 * dx, 101325.0 + 100.0 * dx;
    Eigen::Vector<real, 5> U1;
    IdealGasThermalPrimitive2Conservative<3>(prim1, U1, g_gamma);

    // Conservative gradient: dU/dx in x-direction only
    Eigen::Matrix<real, 3, 5> GradU = Eigen::Matrix<real, 3, 5>::Zero();
    GradU.row(0) = (U1 - U0).transpose() / dx;

    Eigen::Matrix<real, 3, 5> GradPrim;
    GradientCons2Prim_IdealGas<3>(U0, GradU, GradPrim, g_gamma);

    // Expected primitive gradient
    Eigen::RowVector<real, 5> dPrimdx_expected = (prim1 - prim0).transpose() / dx;

    for (int j = 0; j < 5; j++)
    {
        CAPTURE(j);
        CHECK(GradPrim(0, j) == doctest::Approx(dPrimdx_expected(j)).epsilon(1e-4));
    }
}

// ===================================================================
// IdealGasGetCompressionRatioPressure
// ===================================================================

TEST_CASE("CompressionRatio: zero increment gives alpha=0 (no compression needed)")
{
    // Zero increment means nothing to limit; the quadratic solver
    // has c1=c2=0, yielding alpha=0 by convention.
    auto U = primToCons3D(1.225, 100.0, -50.0, 25.0, 101325.0);
    Eigen::Vector<real, 5> dU = Eigen::Vector<real, 5>::Zero();

    real alpha = IdealGasGetCompressionRatioPressure<3, 0, 5>(U, dU, 0.0);
    CHECK(alpha == doctest::Approx(0.0).epsilon(1e-14));
}

TEST_CASE("CompressionRatio: alpha in [0,1]")
{
    auto U = primToCons3D(1.0, 0.0, 0.0, 0.0, 1.0);
    // Large decrement that would make pressure negative
    Eigen::Vector<real, 5> dU;
    dU << 0, 0, 0, 0, -100.0; // massive energy reduction

    real alpha = IdealGasGetCompressionRatioPressure<3, 0, 5>(U, dU, 0.0);
    CHECK(alpha >= 0.0);
    CHECK(alpha <= 1.0);
    // With alpha, the state should maintain non-negative internal energy
    auto Unew = U + alpha * dU;
    real vSqr = (Unew.segment<3>(1) / Unew(0)).squaredNorm();
    real eInternal = Unew(4) - 0.5 * Unew(0) * vSqr;
    CHECK(eInternal >= -1e-10);
}

// ===================================================================
// ViscousFlux_IdealGas: uniform flow produces zero viscous flux
// ===================================================================

TEST_CASE("ViscousFlux: zero gradient produces zero flux")
{
    auto U = primToCons3D(1.225, 100.0, -50.0, 25.0, 101325.0);
    Eigen::Matrix<real, 3, 5> GradPrim = Eigen::Matrix<real, 3, 5>::Zero();
    Eigen::Vector3d norm(1, 0, 0);
    real mu = 1.8e-5, k = 0.025, Cp = 1005.0;

    Eigen::Vector<real, 5> Flux;
    ViscousFlux_IdealGas<3>(U, GradPrim, norm, false, g_gamma, mu, 0.0, false, k, Cp, Flux);

    for (int i = 0; i < 5; i++)
    {
        CAPTURE(i);
        CHECK(Flux(i) == doctest::Approx(0.0).epsilon(1e-10));
    }
}
