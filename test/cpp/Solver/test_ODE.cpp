/**
 * @file test_ODE.cpp
 * @brief Unit tests for ODE time integrators in Solver/ODE.hpp.
 *
 * Tests the harmonic oscillator du/dt = A*u, A = [[0,-w],[w,0]],
 * u(0) = [1,0], exact: u(t) = [cos(wt), sin(wt)].
 * This oscillatory (non-dissipative) problem reveals phase/amplitude
 * errors that decay tests hide.
 *
 * Verified convergence orders:
 *   - ExplicitSSPRK3: 3rd
 *   - ImplicitEuler: 1st
 *   - SDIRK4 sc=0: 4th
 *   - ESDIRK4 sc=1 (DITR U2R2 form): 4th
 *   - ESDIRK3 sc=2: 3rd
 *   - Trapezoid sc=3: 2nd
 *   - ESDIRK2 sc=4: 2nd
 *   - VBDF k=1: 1st
 *   - VBDF k=2: 2nd
 *   - DITR U2R2 (Hermite3 mask=0): 4th (single-step, FSAL)
 *   - DITR U2R1 (Hermite3 mask=1): 3rd (L-stable)
 *
 * All tests are serial (no MPI).
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Solver/ODE.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>

using namespace DNDS;

// ===================================================================
// 2-component vector satisfying TDATA/TDTAU interface
// ===================================================================
struct Vec2
{
    double x = 0, y = 0;
    Vec2() = default;
    Vec2(double a, double b) : x(a), y(b) {}
    Vec2 &operator=(const Vec2 &) = default;
    void operator+=(const Vec2 &o) { x += o.x; y += o.y; }
    void operator*=(double s) { x *= s; y *= s; }
    void operator*=(const Vec2 &o) { x *= o.x; y *= o.y; } // element-wise (SSPRK3 dTau)
    void setConstant(double s) { x = s; y = s; }
    void addTo(const Vec2 &o, double s) { x += o.x * s; y += o.y * s; }
    double norm() const { return std::sqrt(x * x + y * y); }
};

static auto v2init = [](Vec2 &v) { v = Vec2(0, 0); };

// ===================================================================
// Harmonic oscillator: du/dt = [[0,-w],[w,0]] * u, w=2*pi
// Exact: u(t) = [cos(wt), sin(wt)]
// Jacobian J = [[0,-w],[w,0]]
// ===================================================================
static const double g_omega = 2.0 * pi;

using tODE = ODE::ImplicitDualTimeStep<Vec2, Vec2>;

// frhs: rhs = A*u
static tODE::Frhs g_frhs = [](Vec2 &rhs, Vec2 &u, Vec2 &, int, real, int)
{
    rhs.x = -g_omega * u.y;
    rhs.y = g_omega * u.x;
};

// fdt: dTau = very large
static tODE::Fdt g_fdt = [](Vec2 &, Vec2 &dTau, real, int)
{
    dTau = Vec2(1e100, 1e100);
};

// fsolve: solve (1/dTau + 1/dt - alpha*J) * xinc = rhs
// J = [[0,-w],[w,0]], so -alpha*J = [[0,alpha*w],[-alpha*w,0]]
// Diagonal of (1/dTau + 1/dt) * I - alpha * J:
//   [[1/dTau+1/dt, alpha*w],[-alpha*w, 1/dTau+1/dt]]
// With dTau~inf: [[1/dt, alpha*w],[-alpha*w, 1/dt]]
static tODE::Fsolve g_fsolve = [](Vec2 &, Vec2 &rhs, Vec2 &, Vec2 &dTau,
                                   real dt, real alpha, Vec2 &xinc, int, real, int)
{
    double d = 1.0 / dTau.x + 1.0 / dt;
    double aw = alpha * g_omega;
    double det = d * d + aw * aw;
    xinc.x = (d * rhs.x - aw * rhs.y) / det;
    xinc.y = (aw * rhs.x + d * rhs.y) / det;
};

static tODE::Fincrement g_fincrement = [](Vec2 &u, Vec2 &xinc, real s, int)
{
    u.x += s * xinc.x;
    u.y += s * xinc.y;
};

// ===================================================================
// Integration helper
// ===================================================================
static const int g_maxNewton = 50;
static const double g_newtonTol = 1e-13;

static double integrateOscillator(ODE::ImplicitDualTimeStep<Vec2, Vec2> &ode,
                                  double T, int N)
{
    double dt = T / N;
    Vec2 u(1.0, 0.0), xinc;

    // Stop when residual norm < threshold OR max iterations reached
    auto fstop = [&](int iter, Vec2 &res, int) -> bool
    {
        double resNorm = std::sqrt(res.x * res.x + res.y * res.y);
        return resNorm < g_newtonTol || iter >= g_maxNewton;
    };

    for (int step = 0; step < N; step++)
        ode.Step(u, xinc, g_frhs, g_fdt, g_fsolve, g_maxNewton, fstop, g_fincrement, dt);

    // Error: distance from exact [cos(wT), sin(wT)]
    double ex = std::cos(g_omega * T);
    double ey = std::sin(g_omega * T);
    return std::sqrt((u.x - ex) * (u.x - ex) + (u.y - ey) * (u.y - ey));
}

struct ConvergenceResult
{
    double err_coarse, err_fine, order;
};

static ConvergenceResult measureOrder(
    ODE::ImplicitDualTimeStep<Vec2, Vec2> &ode,
    double T, int N_base)
{
    double e1 = integrateOscillator(ode, T, N_base);
    double e2 = integrateOscillator(ode, T, 2 * N_base);
    return {e1, e2, std::log2(e1 / e2)};
}

// ESDIRK: fresh instances per run (internal FSAL state)
template <int sc>
static ConvergenceResult measureOrderESDIRK(double T, int N_base)
{
    auto run = [&](int N) {
        ODE::ImplicitSDIRK4DualTimeStep<Vec2, Vec2> ode(1, v2init, v2init, sc);
        return integrateOscillator(ode, T, N);
    };
    double e1 = run(N_base);
    double e2 = run(2 * N_base);
    return {e1, e2, std::log2(e1 / e2)};
}

// DITR (Hermite3): fresh instances per run
static ConvergenceResult measureOrderDITR(int mask, double alpha,
                                          double T, int N_base)
{
    auto run = [&](int N) {
        ODE::ImplicitHermite3SimpleJacobianDualStep<Vec2, Vec2> ode(
            1, v2init, v2init, alpha, /*curSolveMethod=*/0, /*nStartIter=*/0,
            /*thetaM1n=*/0.9146, /*thetaM2n=*/0.0, mask, /*nMGn=*/0);
        return integrateOscillator(ode, T, N);
    };
    double e1 = run(N_base);
    double e2 = run(2 * N_base);
    return {e1, e2, std::log2(e1 / e2)};
}

// ===================================================================
// Tests
// ===================================================================

TEST_CASE("SSPRK3: 3rd-order on oscillator")
{
    ODE::ExplicitSSPRK3TimeStepAsImplicitDualTimeStep<Vec2, Vec2> ode(1, v2init, v2init);
    auto res = measureOrder(ode, 1.0, 100);
    std::cout << "[SSPRK3] err1=" << std::scientific << res.err_coarse
              << " err2=" << res.err_fine << " order=" << std::fixed << res.order << std::endl;
    CHECK(res.order > 2.8);
    CHECK(res.order < 3.5);
}

TEST_CASE("ImplicitEuler: 1st-order on oscillator")
{
    ODE::ImplicitEulerDualTimeStep<Vec2, Vec2> ode(1, v2init, v2init);
    auto res = measureOrder(ode, 1.0, 200);
    std::cout << "[ImplicitEuler] err1=" << std::scientific << res.err_coarse
              << " err2=" << res.err_fine << " order=" << std::fixed << res.order << std::endl;
    CHECK(res.order > 0.9);
    CHECK(res.order < 1.3);
}

TEST_CASE("SDIRK4 (sc=0): 4th-order on oscillator")
{
    ODE::ImplicitSDIRK4DualTimeStep<Vec2, Vec2> ode(1, v2init, v2init, 0);
    auto res = measureOrder(ode, 1.0, 20);
    std::cout << "[SDIRK4-sc0] err1=" << std::scientific << res.err_coarse
              << " err2=" << res.err_fine << " order=" << std::fixed << res.order << std::endl;
    CHECK(res.order > 3.5);
    CHECK(res.order < 4.5);
}

TEST_CASE("ESDIRK4 (sc=1): 4th-order on oscillator")
{
    auto res = measureOrderESDIRK<1>(1.0, 20);
    std::cout << "[ESDIRK4-sc1] err1=" << std::scientific << res.err_coarse
              << " err2=" << res.err_fine << " order=" << std::fixed << res.order << std::endl;
    CHECK(res.order > 3.5);
    CHECK(res.order < 5.0);
}

TEST_CASE("ESDIRK3 (sc=2): 3rd-order on oscillator")
{
    auto res = measureOrderESDIRK<2>(1.0, 40);
    std::cout << "[ESDIRK3-sc2] err1=" << std::scientific << res.err_coarse
              << " err2=" << res.err_fine << " order=" << std::fixed << res.order << std::endl;
    CHECK(res.order > 2.7);
    CHECK(res.order < 3.5);
}

TEST_CASE("Trapezoid (sc=3): 2nd-order on oscillator")
{
    auto res = measureOrderESDIRK<3>(1.0, 40);
    std::cout << "[Trapezoid-sc3] err1=" << std::scientific << res.err_coarse
              << " err2=" << res.err_fine << " order=" << std::fixed << res.order << std::endl;
    CHECK(res.order > 1.8);
    CHECK(res.order < 2.5);
}

TEST_CASE("ESDIRK2 (sc=4): 2nd-order on oscillator")
{
    auto res = measureOrderESDIRK<4>(1.0, 40);
    std::cout << "[ESDIRK2-sc4] err1=" << std::scientific << res.err_coarse
              << " err2=" << res.err_fine << " order=" << std::fixed << res.order << std::endl;
    CHECK(res.order > 1.8);
    CHECK(res.order < 2.5);
}

TEST_CASE("VBDF k=1: 1st-order on oscillator")
{
    ODE::ImplicitVBDFDualTimeStep<Vec2, Vec2> ode(1, v2init, v2init, 1);
    auto res = measureOrder(ode, 1.0, 200);
    std::cout << "[VBDF1] err1=" << std::scientific << res.err_coarse
              << " err2=" << res.err_fine << " order=" << std::fixed << res.order << std::endl;
    CHECK(res.order > 0.9);
    CHECK(res.order < 1.3);
}

TEST_CASE("VBDF k=2: 2nd-order on oscillator")
{
    auto run = [](int N) {
        ODE::ImplicitVBDFDualTimeStep<Vec2, Vec2> ode(1, v2init, v2init, 2);
        return integrateOscillator(ode, 1.0, N);
    };
    double e1 = run(100);
    double e2 = run(200);
    double order = std::log2(e1 / e2);
    std::cout << "[VBDF2] err1=" << std::scientific << e1
              << " err2=" << e2 << " order=" << std::fixed << order << std::endl;
    CHECK(order > 1.8);
    CHECK(order < 2.5);
}

// ===================================================================
// DITR (Hermite3) methods
// ===================================================================

// DITR methods have coupled mid-point + endpoint Newton iterations.
// Need enough iterations for both sub-solves to converge on nonlinear
// (oscillatory) problems.
TEST_CASE("DITR U2R2 (mask=0, alpha=0.5): 4th-order on oscillator")
{
    // U2R2 with alpha=0.5 is Lobatto IIIA, 4th order
    auto res = measureOrderDITR(0, 0.5, 1.0, 20);
    std::cout << "[DITR-U2R2] err1=" << std::scientific << res.err_coarse
              << " err2=" << res.err_fine << " order=" << std::fixed << res.order << std::endl;
    CHECK(res.order > 3.5);
    CHECK(res.order < 5.0);
}

TEST_CASE("DITR U2R2 (mask=0, alpha=0.55): 3rd-order on oscillator")
{
    // U2R2 with alpha != 0.5 is 3rd order
    auto res = measureOrderDITR(0, 0.55, 1.0, 40);
    std::cout << "[DITR-U2R2-0.55] err1=" << std::scientific << res.err_coarse
              << " err2=" << res.err_fine << " order=" << std::fixed << res.order << std::endl;
    CHECK(res.order > 2.7);
    CHECK(res.order < 4.0);
}

TEST_CASE("DITR U2R1 (mask=1): 3rd-order L-stable on oscillator")
{
    auto res = measureOrderDITR(1, 0.55, 1.0, 40);
    std::cout << "[DITR-U2R1] err1=" << std::scientific << res.err_coarse
              << " err2=" << res.err_fine << " order=" << std::fixed << res.order << std::endl;
    CHECK(res.order > 2.7);
    CHECK(res.order < 3.5);
}

// ===================================================================
// Golden value: SSPRK3 on oscillator at T=0.5, N=200
// ===================================================================

TEST_CASE("SSPRK3: golden value on oscillator")
{
    ODE::ExplicitSSPRK3TimeStepAsImplicitDualTimeStep<Vec2, Vec2> ode(1, v2init, v2init);
    double err = integrateOscillator(ode, 0.5, 200);
    std::cout << "[SSPRK3/golden] err=" << std::scientific << std::setprecision(10) << err << std::endl;
    CHECK(err < 1e-6); // very small at 200 steps for T=0.5
}
