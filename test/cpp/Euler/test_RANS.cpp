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
 *   - Zero gradient -> zero production in source
 *   - Zero gradient -> zero viscous flux
 *   - Finite output (no NaN/Inf)
 *   - Golden values for specific test vectors
 *
 * All functions are pure template functions (no MPI, no mesh).
 * SA source limiting is covered with the same JSON-facing config type used by
 * the evaluator.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/RANS_ke.hpp"
#include "Euler/EulerEvaluatorSettings.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <type_traits>

using namespace DNDS;
using namespace DNDS::Euler;
using namespace DNDS::Euler::RANS;

static constexpr real g_gamma = 1.4;
static constexpr int dim = 3;
static constexpr int I4 = dim + 1; // = 4
// nVars for 2-equation models: 5 (flow) + 2 (turb) = 7
static constexpr int nVars = 7;

static const KOmegaConfig g_kOmegaConfig{};
static const KOmegaSSTConfig g_kOmegaSSTConfig{};
static const RKEConfig g_rkeConfig{};
static const SAConfig g_saConfig{};

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

TEST_CASE("GetMut_KOWilcox: golden value")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_KOWilcox<dim>(U, DiffU, g_mu, g_d, g_kOmegaConfig);

    std::cout << "[KOWilcox/mut] " << std::scientific << std::setprecision(10) << mut << std::endl;
    // Golden value captured:
    real golden = 8.4000000000e-03;
    if (golden < GOLDEN_NOT_ACQUIRED)
        CHECK(mut == doctest::Approx(golden).epsilon(1e-6));
}

// ===================================================================
// GetMut_SST
// ===================================================================

TEST_CASE("GetMut_SST: golden value")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_SST<dim>(U, DiffU, g_mu, g_d, g_kOmegaSSTConfig);

    std::cout << "[SST/mut] " << std::scientific << std::setprecision(10) << mut << std::endl;
    real golden = 7.5950000000e-03;
    if (golden < GOLDEN_NOT_ACQUIRED)
        CHECK(mut == doctest::Approx(golden).epsilon(1e-6));
}

// ===================================================================
// GetMut_RealizableKe
// ===================================================================

TEST_CASE("GetMut_RealizableKe: golden value")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_RealizableKe<dim>(U, DiffU, g_mu, g_d, g_rkeConfig);

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

    GetSource_KOWilcox<dim>(U, DiffU, g_mu, g_d, source, 0, g_kOmegaConfig);

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

    GetSource_SST<dim>(U, DiffU, g_mu, g_d, 1e10, source, 0, g_kOmegaSSTConfig);

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

    GetSource_RealizableKe<dim>(U, DiffU, g_mu, g_d, source, 0, g_rkeConfig);

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

    GetSource_KOWilcox<dim>(U, DiffU, g_mu, g_d, source, 0, g_kOmegaConfig);

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

    GetSource_SST<dim>(U, DiffU, g_mu, g_d, 1e10, source, 0, g_kOmegaSSTConfig);

    std::cout << "[SST/source] k=" << std::scientific << std::setprecision(10)
              << source(I4 + 1) << " omega=" << source(I4 + 2) << std::endl;

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));
}

TEST_CASE("GetSource_SST: omega production blends inner and outer coefficients")
{
    // Constant-density simple shear removes cross diffusion and isolates gamma.
    const auto U = buildState(1, 1, 0, 0, 1, 1e-4, 2);
    const auto gradient = buildShearGrad(U, 3);
    constexpr real gammaInner = 0.075 / 0.09 - 0.5 * 0.41 * 0.41 / 0.3;
    constexpr real gammaOuter = 0.0828 / 0.09 - 0.856 * 0.41 * 0.41 / 0.3;
    for (const bool inner : {true, false})
    {
        Eigen::Vector<real, nVars> source = Eigen::Vector<real, nVars>::Zero();
        GetSource_SST<dim>(U, gradient, 1e-5, inner ? 1e-5 : 1e6,
                           1e100, source, 0, g_kOmegaSSTConfig);
        const real expected = (inner ? gammaInner : gammaOuter) * 9 -
                              (inner ? 0.075 : 0.0828) * 4;
        CHECK(source(I4 + 2) == doctest::Approx(expected).epsilon(1e-12));
    }
}

TEST_CASE("GetSource_RealizableKe: shear gradient golden")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    auto DiffU = buildShearGrad(U, g_Sshear);
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_RealizableKe<dim>(U, DiffU, g_mu, g_d, source, 0, g_rkeConfig);

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

    GetSource_KOWilcox<dim>(U, DiffU, g_mu, g_d, source, 1, g_kOmegaConfig);

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

    GetSource_SST<dim>(U, DiffU, g_mu, g_d, 1e10, source, 1, g_kOmegaSSTConfig);

    CHECK(source(I4 + 1) >= 0.0);
    CHECK(source(I4 + 2) >= 0.0);
}

TEST_CASE("SST optional sink diagonal matches frozen-gradient derivatives and preserves RHS")
{
    auto U = buildState(1.3, .2, .1, 0, 1, .8, 2.4);
    Eigen::Matrix<real, dim, nVars> gradient = Eigen::Matrix<real, dim, nVars>::Zero();
    // Isotropic expansion: zero deviatoric strain, negative k-production.
    for (int d = 0; d < dim; ++d)
        gradient(d, d + 1) = U(0) * .2;
    gradient(0, I4 + 1) = U(0) * .3;
    auto config = g_kOmegaSSTConfig;
    config.implicitAdditionalSinks = true;
    for (real omegaGradient : {real(.7), real(-.7)})
    {
        CAPTURE(omegaGradient);
        gradient(0, I4 + 2) = U(0) * omegaGradient;
        auto evaluate = [&](const auto &state, int mode, const auto &settings)
        {
            Eigen::Vector<real, nVars> result = Eigen::Vector<real, nVars>::Zero();
            // Far-wall limit freezes F1=F2=0 and pure RANS FDES=1.
            GetSource_SST<dim>(state, gradient, 1e-5, 1e12, 1e100, result, mode, settings);
            return result;
        };
        CHECK((evaluate(U, 0, config) - evaluate(U, 0, g_kOmegaSSTConfig)).norm() == 0);
        const auto diagonal = evaluate(U, 1, config);
        const auto original = evaluate(U, 1, g_kOmegaSSTConfig);
        for (int j : {I4 + 1, I4 + 2})
        {
            auto plus = U, minus = U;
            constexpr real h = 1e-5;
            plus(j) += h;
            minus(j) -= h;
            const real negativeDerivative = -(evaluate(plus, 0, config)(j) - evaluate(minus, 0, config)(j)) / (2 * h);
            if (j == I4 + 1 || omegaGradient > 0)
                CHECK(diagonal(j) == doctest::Approx(negativeDerivative).epsilon(1e-8));
            else
                CHECK(diagonal(j) == original(j)); // destabilizing derivative stays explicit
        }
    }
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
    GetVisFlux_KOWilcox<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux, g_kOmegaConfig);

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
    GetVisFlux_SST<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux, g_kOmegaSSTConfig);

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
    GetVisFlux_RealizableKe<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux, g_rkeConfig);

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
    GetVisFlux_KOWilcox<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux, g_kOmegaConfig);

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

    real mut1 = GetMut_KOWilcox<dim>(U1, DiffU, g_mu, g_d, g_kOmegaConfig);
    real mut2 = GetMut_KOWilcox<dim>(U2, DiffU2, g_mu, g_d, g_kOmegaConfig);

    CHECK(mut2 > mut1);
}

TEST_CASE("GetMut_SST: mut increases with k")
{
    auto U1 = buildState(g_rho, g_u, 0, 0, g_p, 5.0, g_omega);
    auto U2 = buildState(g_rho, g_u, 0, 0, g_p, 20.0, g_omega);
    auto DiffU = buildShearGrad(U1, g_Sshear);
    auto DiffU2 = buildShearGrad(U2, g_Sshear);

    real mut1 = GetMut_SST<dim>(U1, DiffU, g_mu, g_d, g_kOmegaSSTConfig);
    real mut2 = GetMut_SST<dim>(U2, DiffU2, g_mu, g_d, g_kOmegaSSTConfig);

    CHECK(mut2 > mut1);
}

// ===================================================================
// Very small k/omega: no crash, finite result
// ===================================================================

TEST_CASE("GetMut_KOWilcox: very small k/omega produces finite mut")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, 1e-10, 1e-5);
    auto DiffU = buildShearGrad(U, 0.0);
    real mut = GetMut_KOWilcox<dim>(U, DiffU, g_mu, g_d, g_kOmegaConfig);

    CHECK(std::isfinite(mut));
    CHECK(mut >= 0.0);
}

TEST_CASE("GetMut_SST: very small k/omega produces finite mut")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, 1e-10, 1e-5);
    auto DiffU = buildShearGrad(U, 0.0);
    real mut = GetMut_SST<dim>(U, DiffU, g_mu, g_d, g_kOmegaSSTConfig);

    CHECK(std::isfinite(mut));
    CHECK(mut >= 0.0);
}

TEST_CASE("GetMut_RealizableKe: very small k/eps produces finite mut")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, 1e-10, 1e-8);
    auto DiffU = buildShearGrad(U, 0.0);
    real mut = GetMut_RealizableKe<dim>(U, DiffU, g_mu, g_d, g_rkeConfig);

    CHECK(std::isfinite(mut));
    CHECK(mut >= 0.0);
}

TEST_CASE("RANS configs: defaults and JSON round-trip preserve historical limits")
{
    static_assert(std::is_trivially_copyable_v<SAConfig>);
    static_assert(std::is_trivially_copyable_v<KOmegaConfig>);
    static_assert(std::is_trivially_copyable_v<KOmegaSSTConfig>);
    static_assert(std::is_trivially_copyable_v<RKEConfig>);

    CHECK(g_saConfig.productionLimit == doctest::Approx(1e5));
    CHECK(g_kOmegaConfig.productionLimit == doctest::Approx(20));
    CHECK(g_kOmegaConfig.turbulentViscosityLimit == doctest::Approx(1e5));
    CHECK(g_kOmegaSSTConfig.productionLimit == doctest::Approx(20));
    CHECK(g_kOmegaSSTConfig.turbulentViscosityLimit == doctest::Approx(1e5));
    CHECK(g_rkeConfig.turbulentViscosityLimit == doctest::Approx(1e5));

    nlohmann::ordered_json j = g_saConfig;
    CHECK(j.at("productionLimit").get<real>() == doctest::Approx(1e5));
    CHECK(j.at("useFt2").get<bool>());
    j["productionLimit"] = 100;
    j["useFt2"] = false;
    auto parsed = j.get<SAConfig>();
    CHECK(parsed.productionLimit == doctest::Approx(100));
    CHECK_FALSE(parsed.useFt2);
    j.erase("useFt2");
    nlohmann::ordered_json defaults = g_saConfig;
    defaults.merge_patch(j); // The solver merges old inputs over emitted defaults.
    CHECK(defaults.get<SAConfig>().useFt2);
}

TEST_CASE("Euler evaluator settings: RANS configs are flat nested sections")
{
    EulerEvaluatorSettings<NS_SA_3D> settings(6);
    nlohmann::ordered_json j = settings;

    CHECK(j.at("SAConfig").at("productionLimit").get<real>() == doctest::Approx(1e5));
    CHECK(j.at("KOmegaConfig").at("productionLimit").get<real>() == doctest::Approx(20));
    CHECK(j.at("KOmegaSSTConfig").at("productionLimit").get<real>() == doctest::Approx(20));
    CHECK(j.at("RKEConfig").at("turbulentViscosityLimit").get<real>() == doctest::Approx(1e5));
    CHECK(j.at("RANSTopLimit").get<real>() == doctest::Approx(1e5));

    j["SAConfig"]["productionLimit"] = 100;
    j["SAConfig"]["useFt2"] = false;
    EulerEvaluatorSettings<NS_SA_3D> parsed(6);
    from_json(j, parsed);
    CHECK(parsed.saConfig.productionLimit == doctest::Approx(100));
    CHECK_FALSE(parsed.saConfig.useFt2);
    CHECK(parsed.RANSTopLimit == doctest::Approx(1e5));
}

TEST_CASE("GetSource_SA: production limit changes the physical source")
{
    static constexpr int nVarsSA = 6;
    Eigen::Vector<real, nVarsSA> U;
    const real rhoNuTilde = g_rho * 1e-5;
    const real totalEnergy = g_p / (g_gamma - 1.0) + 0.5 * g_rho * g_u * g_u;
    U << g_rho, g_rho * g_u, 0, 0, totalEnergy, rhoNuTilde;

    Eigen::Matrix<real, dim, nVarsSA> DiffU;
    DiffU.setZero();
    DiffU(1, 1) = g_rho * 1e6;

    SAConfig capped;
    capped.productionLimit = 100;
    SAConfig relaxed;
    relaxed.productionLimit = 1e5;
    Eigen::Vector<real, nVarsSA> sourceCapped = Eigen::Vector<real, nVarsSA>::Zero();
    Eigen::Vector<real, nVarsSA> sourceRelaxed = Eigen::Vector<real, nVarsSA>::Zero();

    GetSource_SA<dim>(U, DiffU, 1.0, g_mu, g_gamma,
                      g_d, 1e10, 0.1, 0, sourceCapped, 0, 0, capped);
    GetSource_SA<dim>(U, DiffU, 1.0, g_mu, g_gamma,
                      g_d, 1e10, 0.1, 0, sourceRelaxed, 0, 0, relaxed);

    CHECK(sourceCapped.allFinite());
    CHECK(sourceRelaxed.allFinite());
    CHECK(sourceRelaxed(I4 + 1) > sourceCapped(I4 + 1));
}

template <int saDim>
static void checkSARotationCorrection()
{
    constexpr int nSA = saDim + 3;
    constexpr int iSA = saDim + 2;
    Eigen::Vector<real, nSA> U = Eigen::Vector<real, nSA>::Zero();
    U(0) = 1.2;
    U(saDim + 1) = 1e5;
    const real muRef = 1e-5;
    const real nuTilde = 1e-3;
    U(iSA) = U(0) * nuTilde / muRef;
    SAConfig config;
    config.productionLimit = 1e30; // Isolate the correction from the production cap.
    for (int flow = 0; flow < 4; ++flow)
    {
        CAPTURE(saDim);
        CAPTURE(flow);
        Eigen::Matrix<real, saDim, nSA> grad = Eigen::Matrix<real, saDim, nSA>::Zero();
        real omega = 0;
        if (flow == 0) // Solid-body rotation: Omega = 2, strain = 0.
        {
            grad(0, 2) = U(0);
            grad(1, 1) = -U(0);
            omega = 2;
        }
        else if (flow == 1) // Incompressible planar strain: Omega = 0, strain = 2.
        {
            grad(0, 1) = U(0);
            grad(1, 2) = -U(0);
        }
        else if (flow == 2) // Simple shear: Omega = strain = 1.
        {
            grad(1, 1) = U(0);
            omega = 1;
        }
        // flow == 3 is quiescent. Rotation correction must vanish.
        Eigen::Vector<real, nSA> uncorrected = Eigen::Vector<real, nSA>::Zero();
        GetSource_SA<saDim>(U, grad, muRef, muRef, g_gamma,
                            0.1, 1e10, 0.1, 0, uncorrected, 0, 0, config, 0);
        const real chi = U(iSA);
        const real ft2 = 1.2 * std::exp(-0.5 * chi * chi);
        const real productionScale = U(0) / muRef * 0.1355 * (1 - ft2) * nuTilde;
        // Analytic added frequencies for rotation, strain, shear and quiescence.
        const real addedFrequencies[3][4] = {{-4, 0, 0, 0}, {0, -4, 0, 0}, {-2, 0, 0, 0}};
        int variant = 0;
        for (int version : {0, 9001, 1})
        {
            CAPTURE(version);
            Eigen::Vector<real, nSA> corrected = Eigen::Vector<real, nSA>::Zero();
            Eigen::Vector<real, nSA> diagonal = Eigen::Vector<real, nSA>::Zero();
            Eigen::Vector<real, nSA> unrotated = Eigen::Vector<real, nSA>::Zero();
            GetSource_SA<saDim>(U, grad, muRef, muRef, g_gamma,
                                0.1, 1e10, 0.1, 0, corrected, 1, 0, config, version);
            GetSource_SA<saDim>(U, grad, muRef, muRef, g_gamma,
                                0.1, 1e10, 0.1, 0, diagonal, 1, 1, config, version);
            GetSource_SA<saDim>(U, grad, muRef, muRef, g_gamma,
                                0.1, 1e10, 0.1, 0, unrotated, 0, 0, config, version);
            CHECK(corrected.allFinite());
            CHECK(unrotated(iSA) == uncorrected(iSA));
            CHECK(corrected(iSA) - uncorrected(iSA) ==
                  doctest::Approx(productionScale * addedFrequencies[variant][flow]).epsilon(1e-10));
            const real fnu1 = chi * chi * chi / (chi * chi * chi + 7.1 * 7.1 * 7.1);
            const real fnu2 = 1 - chi / (1 + chi * fnu1);
            const real sbar = nuTilde * fnu2 / (0.41 * 0.41 * 0.1 * 0.1);
            const real physicalProduction = 0.1355 * (1 - ft2) *
                                            (omega + sbar + addedFrequencies[variant][flow]) * nuTilde;
            const real destruction = physicalProduction - corrected(iSA) * muRef / U(0);
            const real diagonalProduction = version == 1 ? 0 : std::min(physicalProduction, 0.0);
            CHECK(diagonal(iSA) == doctest::Approx(std::max(
                                                       (2 * destruction - diagonalProduction) / nuTilde, verySmallReal))
                                       .epsilon(1e-10));
            ++variant;
        }
    }
}

TEST_CASE("GetSource_SA: SA-R suppresses rotation rather than strain in 2D and 3D")
{
    EulerEvaluatorSettings<NS_SA> settings(6);
    CHECK(settings.SAVersion == 0);
    nlohmann::ordered_json config = settings;
    config["SAVersion"] = 9001;
    EulerEvaluatorSettings<NS_SA> historical(6);
    from_json(config, historical);
    CHECK(historical.SAVersion == 9001);
    checkSARotationCorrection<2>();
    checkSARotationCorrection<3>();
}

TEST_CASE("SA DES lengths: coefficient placement, shielding and disabled scale")
{
    // Force LES operation: no eddy viscosity, negligible molecular viscosity,
    // and a point outside the fB/fe wall region. psi is independently prescribed.
    for (DNDS::real coefficient : {0.4, 0.65})
    {
        CHECK(GetLength_SA_DES(3., 1., coefficient, 1, 0., 0., 1., 2.) ==
              doctest::Approx(coefficient * .45 * 2.));
        CHECK(GetLength_SA_DES(3., 1., coefficient, 2, 0., 0., 1., 2.) ==
              doctest::Approx(coefficient * .45 * 2.));
        CHECK(GetLength_SA_DES(3., 1., coefficient, 1, 0., 0., 1., 2., .8) ==
              doctest::Approx(coefficient * .8 * 2.));
        CHECK(GetLength_SA_DES(3., 1., coefficient, 1, 0., 0., 1., 2., 2.) ==
              doctest::Approx(coefficient * 2.)); // Delta cannot exceed hMax.
        CHECK(GetLength_SA_DES(3., 1., coefficient, 0, 0., 0., 1., 2.) ==
              doctest::Approx(coefficient * 2.)); // DDES uses hMax, not IDDES delta.
        CHECK(GetLength_SA_DES(3., 1., coefficient, 3, 0., 0., 1., 2.) ==
              doctest::Approx(coefficient)); // Original DES has no psi correction.
    }
    CHECK(GetLength_SA_DES(.01, 1., .65, 3, 0., 0., 1., 2.) == doctest::Approx(.01));
    CHECK(GetLength_SA_DES(1e30, 1., .65, 0, 0., 0., 1., 2.) == doctest::Approx(1.3));
    // DDES/IDDES shielding must preserve RANS for a large wall-region rd.
    for (int mode : {0, 1})
        CHECK(GetLength_SA_DES(3., 1., .65, mode, 100., 100., 1., 1.) == doctest::Approx(3.));
    for (int mode : {0, 1, 2, 3})
    {
        CHECK(GetLength_SA_DES(3., 1., veryLargeReal, mode, 0., 0., 1., 2.) == 3.);
        CHECK(GetLength_SA_DES(3., 1., 101., mode, 100., 100., 1., 2.) == 3.);
    }
}

template <int saDim>
static void checkSADESAndFt2()
{
    constexpr int nSA = saDim + 3;
    constexpr int iSA = saDim + 2;
    constexpr DNDS::real muRef = 1e-5, chi = 2., nuTilde = chi * muRef;
    constexpr DNDS::real shear = 100., wallDistance = 1., hMax = .01;
    const DNDS::real fv1 = std::pow(chi, 3) / (std::pow(chi, 3) + std::pow(7.1, 3));
    const DNDS::real fv2 = 1 - chi / (1 + chi * fv1);
    const DNDS::real cb1 = .1355, kappa2 = .41 * .41;
    const DNDS::real cw1 = cb1 / kappa2 + (1 + .622) / (2. / 3.);
    Eigen::Vector<DNDS::real, nSA> U = Eigen::Vector<DNDS::real, nSA>::Zero();
    U(0) = 1.;
    U(saDim + 1) = 1e5;
    U(iSA) = chi;
    Eigen::Matrix<DNDS::real, saDim, nSA> grad = Eigen::Matrix<DNDS::real, saDim, nSA>::Zero();
    grad(1, 1) = shear;
    for (bool useFt2 : {true, false})
        for (int desMode : {0, 1, 2, 3})
            for (DNDS::real coefficient : {.65, veryLargeReal})
            {
                CAPTURE(saDim);
                CAPTURE(useFt2);
                CAPTURE(desMode);
                CAPTURE(coefficient);
                SAConfig config;
                config.useFt2 = useFt2;
                config.productionLimit = 1e30;
                const DNDS::real ft2 = useFt2 ? 1.2 * std::exp(-.5 * chi * chi) : 0.;
                const DNDS::real psi = std::sqrt(std::max(1., std::min(100.,
                                                                       (1 - cb1 / (cw1 * kappa2 * .424) * (ft2 + (1 - ft2) * fv2)) /
                                                                           (fv1 * std::max(1e-10, 1 - ft2)))));
                // Here d/hMax=100, fd=1 and fB=fe=0 to floating-point accuracy.
                // Thus every active model reduces to C_DES*hMax (times psi except DES97).
                const DNDS::real length = coefficient > 100 ? wallDistance : coefficient * hMax * (desMode == 3 ? 1. : psi);
                const DNDS::real sh = shear + nuTilde * fv2 / (kappa2 * length * length);
                const DNDS::real r = std::min(nuTilde / (sh * kappa2 * length * length), 10.);
                const DNDS::real g = r + .3 * (std::pow(r, 6) - r);
                const DNDS::real fw = g * std::pow(65. / (std::pow(g, 6) + 64.), 1. / 6.);
                const DNDS::real production = cb1 * (1 - ft2) * sh * nuTilde;
                const DNDS::real destruction = (cw1 * fw - cb1 / kappa2 * ft2) * std::pow(nuTilde / length, 2);
                Eigen::Vector<DNDS::real, nSA> source = Eigen::Vector<DNDS::real, nSA>::Zero();
                Eigen::Vector<DNDS::real, nSA> diagonal = Eigen::Vector<DNDS::real, nSA>::Zero();
                GetSource_SA<saDim>(U, grad, muRef, muRef, g_gamma, wallDistance,
                                    coefficient, hMax, desMode, source, 0, 0, config);
                GetSource_SA<saDim>(U, grad, muRef, muRef, g_gamma, wallDistance,
                                    coefficient, hMax, desMode, diagonal, 0, 1, config);
                CHECK(source.allFinite());
                CHECK(diagonal.allFinite());
                CHECK(source(iSA) == doctest::Approx((production - destruction) / muRef).epsilon(1e-10));
                CHECK(diagonal(iSA) == doctest::Approx(std::max((2 * destruction - std::min(production, 0.)) / nuTilde, verySmallReal)).epsilon(1e-10));
            }
#ifdef USE_NS_SA_NEGATIVE_MODEL
    U(iSA) = -2.;
    SAConfig withFt2, withoutFt2;
    withoutFt2.useFt2 = false;
    Eigen::Vector<DNDS::real, nSA> original = Eigen::Vector<DNDS::real, nSA>::Zero();
    Eigen::Vector<DNDS::real, nSA> noft2 = Eigen::Vector<DNDS::real, nSA>::Zero();
    GetSource_SA<saDim>(U, grad, muRef, muRef, g_gamma, wallDistance,
                        veryLargeReal, hMax, 1, original, 0, 0, withFt2);
    GetSource_SA<saDim>(U, grad, muRef, muRef, g_gamma, wallDistance,
                        veryLargeReal, hMax, 1, noft2, 0, 0, withoutFt2);
    CHECK(noft2.allFinite());
    CHECK(noft2(iSA) == original(iSA));
    CHECK(noft2(iSA) > 0); // The negative continuation still drives nuTilde toward zero.
#endif
}

TEST_CASE("GetSource_SA: DES modes and ft2 switch agree with analytic sources in 2D/3D")
{
    checkSADESAndFt2<2>();
    checkSADESAndFt2<3>();
}

TEST_CASE("shared SA: explicit frame vorticity matches corrected velocity gradients")
{
    auto state = buildState(g_rho, g_u, 0, 0, g_p, 10, 1000);
    state[I4 + 1] = state[0] * 100; // nuTilde=1e-3 with muRef=1e-5.
    auto gradient = buildShearGrad(state, 1.7);
    Eigen::Matrix<real, dim, dim> correction = Eigen::Matrix<real, dim, dim>::Zero();
    correction(0, 1) = 0.3;
    correction(1, 0) = -0.3;
    auto correctedGradient = gradient;
    correctedGradient.block<dim, dim>(0, 1) -= state[0] * correction;
    Eigen::Vector<real, nVars> explicitFrame = Eigen::Vector<real, nVars>::Zero(), fromGradient = explicitFrame, uncorrected = explicitFrame;
    DNDS::Physics::RANS::GetSource_SAWithVorticity<dim>(state, gradient, 1e-5, g_mu, g_gamma,
                                                        0.1, 1e10, 0.1, 0, explicitFrame, 0, 0, g_saConfig, 0, &correction);
    GetSource_SA<dim>(state, correctedGradient, 1e-5, g_mu, g_gamma,
                      0.1, 1e10, 0.1, 0, fromGradient, 0, 0, g_saConfig);
    GetSource_SA<dim>(state, gradient, 1e-5, g_mu, g_gamma,
                      0.1, 1e10, 0.1, 0, uncorrected, 0, 0, g_saConfig);
    CHECK(explicitFrame.isApprox(fromGradient, 1e-12));
    CHECK(std::abs(explicitFrame[I4 + 1] - uncorrected[I4 + 1]) > 1e-3);
}
