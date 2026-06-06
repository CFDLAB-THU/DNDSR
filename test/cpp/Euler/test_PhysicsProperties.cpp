#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/Chemistry/ChemicalSource.hpp"
#include "Euler/Euler.hpp"
#include "Euler/Physics/PhysicsProperties.hpp"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace DNDS;
using namespace DNDS::Euler;
using namespace DNDS::Euler::Chemistry;

static std::string mechFile()
{
    const char *env = std::getenv("DNDS_MECH_PATH");
    return env ? std::string(env) + "/h2o2.yaml" : "h2o2.yaml";
}

template <EulerModel model>
static typename EulerEvaluatorSettings<model>::IdealGasProperty makeIdealGasProperty()
{
    typename EulerEvaluatorSettings<model>::IdealGasProperty igProp;
    igProp.gamma = 1.4;
    igProp.Rgas = 287.0;
    igProp.U0 = 379.0;
    igProp.rho0 = 1.0;
    igProp.T0 = 1.0;
    igProp.muGas = 1e-200;
    igProp.prGas = 0.72;
    return igProp;
}

// Build IdealGasProperty with non-default reference scales for scaling tests.
static auto makeScaledIdealGas()
{
    EulerEvaluatorSettings<NS_EX>::IdealGasProperty igProp;
    igProp.gamma = 1.4;
    igProp.Rgas = 287.0;
    igProp.U0 = 340.0;
    igProp.rho0 = 1.225;
    igProp.T0 = 288.15;
    igProp.L0 = 2.0;
    igProp.muGas = 1.789e-5;
    igProp.prGas = 0.72;
    return igProp;
}

static auto makeReactiveFixture()
{
    auto pool = std::make_shared<std::vector<ChemicalSource>>();
    pool->emplace_back(mechFile(), "", 379.0, 1.0);

    auto phys = std::make_unique<PhysicsProperties<NS_EX>>(makeIdealGasProperty<NS_EX>());
    phys->setChemicalSourcePool(pool);

    struct Fixture
    {
        std::shared_ptr<std::vector<ChemicalSource>> pool;
        std::unique_ptr<PhysicsProperties<NS_EX>> phys;
    };
    return Fixture{std::move(pool), std::move(phys)};
}

static auto makeScaledReactiveFixture()
{
    auto pool = std::make_shared<std::vector<ChemicalSource>>();
    pool->emplace_back(mechFile(), "", 340.0, 1.225);

    auto phys = std::make_unique<PhysicsProperties<NS_EX>>(makeScaledIdealGas());
    phys->setChemicalSourcePool(pool);

    struct Fixture
    {
        std::shared_ptr<std::vector<ChemicalSource>> pool;
        std::unique_ptr<PhysicsProperties<NS_EX>> phys;
    };
    return Fixture{std::move(pool), std::move(phys)};
}

static PhysicsProperties<NS_EX>::StateConversionOptions strictReactiveOptions()
{
    PhysicsProperties<NS_EX>::StateConversionOptions options;
    options.temperatureUVTolerance = 1e-13;
    options.gammaTolerance = 1e-12;
    options.gammaMaxIterations = 40;
    options.totalToStaticTolerance = 1e-13;
    options.totalToStaticMaxIterations = 60;
    return options;
}

// ============================================================================
// Non-reactive (single-species) tests
// ============================================================================

TEST_CASE("PhysicsProperties non-reactive state conversions")
{
    using Phys = PhysicsProperties<NS>;
    using TU = typename Phys::TU;

    auto igProp = makeIdealGasProperty<NS>();
    Phys phys(igProp);

    TU prim;
    prim << 1.2, 0.22, -0.07, 0.03, 0.72;

    TU cons;
    TU primOut;
    phys.primToConservative(prim, cons);
    phys.conservativeToPrimitive(cons, primOut);
    for (int i = 0; i < prim.size(); ++i)
    {
        CAPTURE(i);
        CHECK(primOut(i) == doctest::Approx(prim(i)).epsilon(1e-12));
    }

    TU primRhoT;
    TU primRhoTOut;
    primRhoT << 1.2, 0.22, -0.07, 0.03, 300.0;
    phys.primRhoTToConservative(primRhoT, cons);
    phys.conservativeToPrimRhoT(cons, primRhoTOut);
    for (int i = 0; i < primRhoT.size(); ++i)
    {
        CAPTURE(i);
        CHECK(primRhoTOut(i) == doctest::Approx(primRhoT(i)).epsilon(1e-12));
    }

    TU primTP;
    TU primTPOut;
    primTP << 300.0, 0.22, -0.07, 0.03, 0.72;
    phys.primTPToConservative(primTP, cons);
    phys.conservativeToPrimTP(cons, primTPOut);
    for (int i = 0; i < primTP.size(); ++i)
    {
        CAPTURE(i);
        CHECK(primTPOut(i) == doctest::Approx(primTP(i)).epsilon(1e-12));
    }
}

TEST_CASE("PhysicsProperties non-reactive conservativeThermal returns closure and acoustic gamma")
{
    using Phys = PhysicsProperties<NS>;
    using TU = typename Phys::TU;

    auto igProp = makeIdealGasProperty<NS>();
    Phys phys(igProp);

    TU prim;
    prim << 1.4, 0.18, -0.05, 0.03, 0.67;
    TU cons;
    phys.primToConservative(prim, cons);

    auto [T, p, asqr, H, gammaEq, gamma] = phys.conservativeThermal(cons);

    CHECK(gammaEq == doctest::Approx(igProp.gamma).epsilon(1e-14));
    CHECK(gamma == doctest::Approx(igProp.gamma).epsilon(1e-14));
    CHECK(p == doctest::Approx(prim(4)).epsilon(1e-12));
    CHECK(asqr == doctest::Approx(igProp.gamma * prim(4) / prim(0)).epsilon(1e-12));
    CHECK(H == doctest::Approx((cons(4) + prim(4)) / cons(0)).epsilon(1e-12));
}

TEST_CASE("PhysicsProperties non-reactive total-to-static conversion is closed form")
{
    using Phys = PhysicsProperties<NS>;
    using TU = typename Phys::TU;

    auto igProp = makeIdealGasProperty<NS>();
    Phys phys(igProp);

    TU primStatic;
    primStatic << 0.0, 0.24, -0.08, 0.04, 0.0;

    real pTotal = 0.95;
    real TTotal = 320.0;
    real vSqr = primStatic(Eigen::seq(Eigen::fix<1>, Eigen::fix<3>)).squaredNorm();
    real Cp = phys.toCode(igProp.CpGas());
    real Rgas = phys.toCode(igProp.Rgas);

    real TStaticExpected = TTotal - 0.5 * vSqr / Cp;
    real pStaticExpected = pTotal * std::pow(TStaticExpected / TTotal, igProp.gamma / (igProp.gamma - 1));
    real rhoExpected = pStaticExpected / (Rgas * TStaticExpected);

    phys.totalToStaticPrimitive(pTotal, TTotal, primStatic);

    CHECK(primStatic(0) == doctest::Approx(rhoExpected).epsilon(1e-12));
    CHECK(primStatic(4) == doctest::Approx(pStaticExpected).epsilon(1e-12));

    TU cons;
    phys.primToConservative(primStatic, cons);
    CHECK(phys.temperature(cons) == doctest::Approx(TStaticExpected).epsilon(1e-12));
}

TEST_CASE("PhysicsProperties non-reactive static-to-total conversion is closed form")
{
    using Phys = PhysicsProperties<NS>;
    using TU = typename Phys::TU;

    auto igProp = makeIdealGasProperty<NS>();
    Phys phys(igProp);

    TU prim;
    prim << 1.225, 0.30, 0.0, 0.0, 0.90;

    auto [pTotal, TTotal] = phys.primitiveStaticToTotalPT(prim);
    real Rgas = phys.toCode(igProp.Rgas);
    real TStatic = prim(4) / (prim(0) * Rgas);
    real asqr = igProp.gamma * prim(4) / prim(0);
    real Msqr = prim(Eigen::seq(Eigen::fix<1>, Eigen::fix<3>)).squaredNorm() / asqr;
    real factor = 1 + 0.5 * (igProp.gamma - 1) * Msqr;

    CHECK(pTotal == doctest::Approx(prim(4) * std::pow(factor, igProp.gamma / (igProp.gamma - 1))).epsilon(1e-12));
    CHECK(TTotal == doctest::Approx(TStatic * factor).epsilon(1e-12));
}

TEST_CASE("PhysicsProperties non-reactive cons code-phys round-trip with non-default scales")
{
    using Phys = PhysicsProperties<NS_EX>;
    using TU = typename Phys::TU;

    auto igProp = makeScaledIdealGas();
    Phys phys(igProp);
    phys.setRANS(RANS_SA); // 1 RANS var (nuTilde)
    int nR = phys.nRANSVars();
    REQUIRE(nR == 1);

    int nVars = 5 + nR; // rho, rhoU, rhoV, rhoW, rhoE, nuTilde
    TU cons(nVars);
    cons << 1.0, 0.2, -0.1, 0.05, 0.8, 0.01;

    real rho0 = igProp.rho0;
    real U0 = igProp.U0;

    // Test individual physical values
    TU physCons;
    phys.consCodeToPhys(cons, physCons);
    CHECK(physCons(0) == doctest::Approx(cons(0) * rho0).epsilon(1e-12));
    CHECK(physCons(1) == doctest::Approx(cons(1) * rho0 * U0).epsilon(1e-12));
    CHECK(physCons(4) == doctest::Approx(cons(4) * rho0 * U0 * U0).epsilon(1e-12));
    // nuTilde: cons scaling = rho0 * 1 = rho0
    CHECK(physCons(5) == doctest::Approx(cons(5) * rho0).epsilon(1e-12));

    // Round-trip
    TU consRT;
    phys.consPhysToCode(physCons, consRT);
    for (int i = 0; i < nVars; ++i)
    {
        CAPTURE(i);
        CHECK(consRT(i) == doctest::Approx(cons(i)).epsilon(2e-12));
    }
}

TEST_CASE("PhysicsProperties non-reactive prim code-phys round-trip with RANS and non-default scales")
{
    using Phys = PhysicsProperties<NS_EX>;
    using TU = typename Phys::TU;

    auto igProp = makeScaledIdealGas();
    Phys phys(igProp);
    phys.setRANS(RANS_KOWilcox); // 2 RANS vars (k, omega)
    int nR = phys.nRANSVars();
    REQUIRE(nR == 2);

    real rho0 = igProp.rho0;
    real U0 = igProp.U0;
    real L0 = igProp.L0;

    int nVars = 5 + nR; // PrimRhoP: [rho, u, v, w, p, k, omega]
    TU prim(nVars);
    prim << 1.2, 0.22, -0.07, 0.03, 0.72, 0.04, 300.0;

    // primCodeToPhys / primPhysToCode (PrimRhoP)
    {
        TU physPrim;
        phys.primCodeToPhys(prim, physPrim);
        CHECK(physPrim(0) == doctest::Approx(prim(0) * rho0).epsilon(1e-12));
        CHECK(physPrim(4) == doctest::Approx(prim(4) * rho0 * U0 * U0).epsilon(1e-12));
        // k: prim scale = U0²
        CHECK(physPrim(5) == doctest::Approx(prim(5) * U0 * U0).epsilon(1e-12));
        // omega: prim scale = U0/L0
        CHECK(physPrim(6) == doctest::Approx(prim(6) * U0 / L0).epsilon(1e-12));

        TU primRT;
        phys.primPhysToCode(physPrim, primRT);
        for (int i = 0; i < nVars; ++i)
        {
            CAPTURE(i);
            CHECK(primRT(i) == doctest::Approx(prim(i)).epsilon(2e-12));
        }
    }

    // primRhoTCodeToPhys / primRhoTPhysToCode
    {
        TU primRhoT(nVars);
        primRhoT << 1.2, 0.22, -0.07, 0.03, 300.0, 0.04, 500.0;

        TU physPrimRT;
        phys.primRhoTCodeToPhys(primRhoT, physPrimRT);
        // Position 4 is temperature: phys = code * T0 (T0=288.15)
        CHECK(physPrimRT(4) == doctest::Approx(primRhoT(4) * igProp.T0).epsilon(1e-12));
        // RANS: k scaled by U0²
        CHECK(physPrimRT(5) == doctest::Approx(primRhoT(5) * U0 * U0).epsilon(1e-12));

        TU primRTRT;
        phys.primRhoTPhysToCode(physPrimRT, primRTRT);
        for (int i = 0; i < nVars; ++i)
        {
            CAPTURE(i);
            CHECK(primRTRT(i) == doctest::Approx(primRhoT(i)).epsilon(1e-12));
        }
    }

    // primTPCodeToPhys / primTPPhysToCode
    {
        TU primTP(nVars);
        primTP << 300.0, 0.22, -0.07, 0.03, 0.72, 0.04, 500.0;

        TU physPrimTP;
        phys.primTPCodeToPhys(primTP, physPrimTP);
        // Position 0 is temperature
        CHECK(physPrimTP(0) == doctest::Approx(primTP(0) * igProp.T0).epsilon(1e-12));
        CHECK(physPrimTP(5) == doctest::Approx(primTP(5) * U0 * U0).epsilon(1e-12));

        TU primTPRT;
        phys.primTPPhysToCode(physPrimTP, primTPRT);
        for (int i = 0; i < nVars; ++i)
        {
            CAPTURE(i);
            CHECK(primTPRT(i) == doctest::Approx(primTP(i)).epsilon(1e-12));
        }
    }
}

// ============================================================================
// Multi-species (Cantera) tests
// ============================================================================

TEST_CASE("PhysicsProperties reactive state conversions")
{
    auto fx = makeReactiveFixture();
    auto &phys = *fx.phys;
    auto &chem = (*fx.pool)[0];
    auto options = strictReactiveOptions();
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    REQUIRE(Ns == 10);

    using TU = typename PhysicsProperties<NS_EX>::TU;
    TU prim(5 + Ns1);
    TU cons(5 + Ns1);
    TU primOut(5 + Ns1);
    prim.setZero();
    prim(0) = 0.85;
    prim(1) = 0.18;
    prim(2) = -0.04;
    prim(3) = 0.02;
    prim(4) = 0.55;
    prim(5 + 0) = 0.028; // H2
    prim(5 + 3) = 0.222; // O2; N2 is the dependent last species

    phys.primToConservative(prim, cons, options);
    phys.conservativeToPrimitive(cons, primOut, options);
    for (int i = 0; i < prim.size(); ++i)
    {
        CAPTURE(i);
        CHECK(primOut(i) == doctest::Approx(prim(i)).epsilon(1e-11));
    }

    TU primTP(5 + Ns1);
    TU primTPOut(5 + Ns1);
    primTP = prim;
    primTP(0) = 1200.0;
    primTP(4) = 0.70;
    phys.primTPToConservative(primTP, cons, options);
    phys.conservativeToPrimTP(cons, primTPOut, options);
    for (int i = 0; i < primTP.size(); ++i)
    {
        CAPTURE(i);
        CHECK(primTPOut(i) == doctest::Approx(primTP(i)).epsilon(1e-11));
    }

    TU primRhoT(5 + Ns1);
    TU primRhoTOut(5 + Ns1);
    primRhoT = prim;
    primRhoT(0) = 0.92;
    primRhoT(4) = 1050.0;
    phys.primRhoTToConservative(primRhoT, cons, options);
    phys.conservativeToPrimRhoT(cons, primRhoTOut, options);
    for (int i = 0; i < primRhoT.size(); ++i)
    {
        CAPTURE(i);
        CHECK(primRhoTOut(i) == doctest::Approx(primRhoT(i)).epsilon(1e-11));
    }
}

TEST_CASE("PhysicsProperties reactive conservativeThermal separates gammaEq from acoustic gamma")
{
    auto fx = makeReactiveFixture();
    auto &phys = *fx.phys;
    auto &chem = (*fx.pool)[0];
    auto options = strictReactiveOptions();
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    REQUIRE(Ns == 10);

    using TU = typename PhysicsProperties<NS_EX>::TU;
    TU primTP(5 + Ns1);
    TU cons(5 + Ns1);
    primTP.setZero();
    primTP(0) = 1400.0;
    primTP(1) = 0.16;
    primTP(2) = -0.03;
    primTP(3) = 0.01;
    primTP(4) = 0.62;
    primTP(5 + 0) = 0.030;
    primTP(5 + 3) = 0.220;
    phys.primTPToConservative(primTP, cons, options);
    auto [T, p, asqr, H, gammaEq, gamma] = phys.conservativeThermal(cons);

    real rho = cons(0);
    real vSqr = (cons(Eigen::seq(Eigen::fix<1>, Eigen::fix<3>)) / rho).squaredNorm();
    real rhoEBase = phys.mixtureBaseInternalRhoE(cons);
    real sensibleRhoE = cons(4) - 0.5 * rho * vSqr - rhoEBase;
    real gammaCantera = phys.gamma(T, cons);

    CHECK(T == doctest::Approx(primTP(0)).epsilon(1e-11));
    CHECK(p == doctest::Approx(primTP(4)).epsilon(1e-11));
    CHECK(gamma == doctest::Approx(gammaCantera).epsilon(1e-12));
    CHECK(gammaEq == doctest::Approx(1.0 + p / sensibleRhoE).epsilon(1e-12));
    CHECK(asqr == doctest::Approx(gammaCantera * p / rho).epsilon(1e-12));
    CHECK(H == doctest::Approx((cons(4) + p - rhoEBase) / rho).epsilon(1e-12));
    CHECK(std::abs(gammaEq - gamma) > 1e-4);
}

TEST_CASE("PhysicsProperties reactive transport uses Cantera mixture coefficients in code units")
{
    auto fx = makeReactiveFixture();
    auto &phys = *fx.phys;
    auto &chem = (*fx.pool)[0];
    auto options = strictReactiveOptions();
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    REQUIRE(Ns == 10);

    using TU = typename PhysicsProperties<NS_EX>::TU;
    TU primTP(5 + Ns1);
    TU cons(5 + Ns1);
    primTP.setZero();
    primTP(0) = 1300.0;
    primTP(1) = 0.08;
    primTP(2) = -0.02;
    primTP(3) = 0.01;
    primTP(4) = 0.66;
    primTP(5 + 0) = 0.026;
    primTP(5 + 3) = 0.224;
    phys.primTPToConservative(primTP, cons, options);

    auto [T, p, asqr, H, gammaEq, gamma] = phys.conservativeThermal(cons);
    std::vector<double> Ybuf(Ns);
    SpeciesBufferView Ywrite{Ybuf.data(), Ns};
    chem.massFractions(cons(0), cons.data() + 5, Ns1, Ywrite);
    ConstSpeciesBufferView Y{Ybuf.data(), Ns};

    real muCode = phys.mixtureViscosity(T, p, cons);
    real kCode = phys.mixtureConductivity(T, p, cons);
    real DCode = phys.speciesDiffusivityK(T, p, cons, 0);
    double TPhys = phys.toPhysT(T);
    double pPhys = phys.toPhysP(p);

    CHECK(muCode == doctest::Approx(chem.viscosity(TPhys, pPhys, Y) / phys.mu0()).epsilon(1e-12));
    CHECK(kCode == doctest::Approx(chem.thermalConductivity(TPhys, pPhys, Y) / phys.k0()).epsilon(1e-12));

    std::vector<double> Dbuf(Ns);
    SpeciesBufferView Dv{Dbuf.data(), Ns};
    chem.speciesDiffusivity(TPhys, pPhys, Y, Dv);
    CHECK(DCode == doctest::Approx(Dbuf[0] / phys.D0()).epsilon(1e-12));
    CHECK(std::abs(kCode - phys.Cp(T, cons) * muCode / phys.Pr()) > 1e-12);
}

TEST_CASE("PhysicsProperties reactive total-to-static conversion iterates mixture thermodynamics")
{
    auto fx = makeReactiveFixture();
    auto &phys = *fx.phys;
    auto &chem = (*fx.pool)[0];
    auto options = strictReactiveOptions();
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    REQUIRE(Ns == 10);

    using TU = typename PhysicsProperties<NS_EX>::TU;
    TU primStatic(5 + Ns1);
    primStatic.setZero();
    primStatic(1) = 150.0 / 379.0;
    primStatic(2) = -30.0 / 379.0;
    primStatic(3) = 15.0 / 379.0;
    primStatic(5 + 0) = 0.028; // H2
    primStatic(5 + 3) = 0.222; // O2; N2 is the dependent last species

    real pTotal = 101325.0 / (379.0 * 379.0);
    real TTotal = 1200.0;
    phys.totalToStaticPrimitive(pTotal, TTotal, primStatic, options);

    TU cons(5 + Ns1);
    phys.primToConservative(primStatic, cons, options);
    real TStatic = phys.temperature(cons, primStatic(4), options.temperatureUVTolerance);
    real Rgas = phys.Rgas(cons);
    real vSqr = primStatic(Eigen::seq(Eigen::fix<1>, Eigen::fix<3>)).squaredNorm();

    std::vector<double> Ybuf(Ns);
    SpeciesBufferView Ywrite{Ybuf.data(), Ns};
    chem.massFractions(1.0, primStatic.data() + 5, Ns1, Ywrite);
    ConstSpeciesBufferView Y{Ybuf.data(), Ns};

    real pStatic = primStatic(4);
    double hTotal = chem.mixtureEnthalpy(TTotal, Y, pTotal * 379.0 * 379.0);
    double hStatic = chem.mixtureEnthalpy(TStatic, Y, pStatic * 379.0 * 379.0);
    double sTotal = chem.mixtureEntropy(TTotal, Y, pTotal * 379.0 * 379.0);
    double sStatic = chem.mixtureEntropy(TStatic, Y, pStatic * 379.0 * 379.0);
    real rhoExpected = pStatic / (Rgas * TStatic);
    double hResidual = hStatic + 0.5 * vSqr * 379.0 * 379.0 - hTotal;
    double sResidual = sStatic - sTotal;

    CHECK(hResidual == doctest::Approx(0.0).scale(std::abs(hTotal)).epsilon(1e-11));
    CHECK(sResidual == doctest::Approx(0.0).scale(std::abs(sTotal)).epsilon(1e-11));
    CHECK(primStatic(0) == doctest::Approx(rhoExpected).epsilon(1e-11));
    CHECK(TStatic < TTotal);
    CHECK(primStatic(4) < pTotal);
}

TEST_CASE("PhysicsProperties reactive static-to-total inverts total-to-static")
{
    auto fx = makeReactiveFixture();
    auto &phys = *fx.phys;
    auto &chem = (*fx.pool)[0];
    auto options = strictReactiveOptions();
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    REQUIRE(Ns == 10);

    using TU = typename PhysicsProperties<NS_EX>::TU;
    TU primStatic(5 + Ns1);
    primStatic.setZero();
    primStatic(1) = 120.0 / 379.0;
    primStatic(2) = 25.0 / 379.0;
    primStatic(3) = -10.0 / 379.0;
    primStatic(5 + 0) = 0.028;
    primStatic(5 + 3) = 0.222;

    real pTotalIn = 101325.0 / (379.0 * 379.0);
    real TTotalIn = 1250.0;
    phys.totalToStaticPrimitive(pTotalIn, TTotalIn, primStatic, options);

    auto [pTotalOut, TTotalOut] = phys.primitiveStaticToTotalPT(primStatic, options);
    CHECK(pTotalOut == doctest::Approx(pTotalIn).epsilon(2e-11));
    CHECK(TTotalOut == doctest::Approx(TTotalIn).epsilon(2e-11));
}

TEST_CASE("PhysicsProperties reactive total-to-static default options converge or throw")
{
    auto fx = makeReactiveFixture();
    auto &phys = *fx.phys;
    auto &chem = (*fx.pool)[0];
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    REQUIRE(Ns == 10);

    using TU = typename PhysicsProperties<NS_EX>::TU;
    TU primStatic(5 + Ns1);
    primStatic.setZero();
    primStatic(1) = 150.0 / 379.0;
    primStatic(2) = -30.0 / 379.0;
    primStatic(3) = 15.0 / 379.0;
    primStatic(5 + 0) = 0.028;
    primStatic(5 + 3) = 0.222;

    real pTotal = 101325.0 / (379.0 * 379.0);
    real TTotal = 1200.0;

    TU primDefault = primStatic;
    phys.totalToStaticPrimitive(pTotal, TTotal, primDefault);
    CHECK(primDefault(0) > 0);
    CHECK(primDefault(4) > 0);

    auto tooFewIterations = strictReactiveOptions();
    tooFewIterations.totalToStaticMaxIterations = 1;
    TU primFail = primStatic;
    CHECK_THROWS_AS(phys.totalToStaticPrimitive(pTotal, TTotal, primFail, tooFewIterations), std::runtime_error);
}

// ============================================================================
// Phys-code scale conversion tests (new public API)
// ============================================================================

TEST_CASE("PhysicsProperties cons code-phys round-trip multi-species with non-default scales")
{
    auto fx = makeScaledReactiveFixture();
    auto &phys = *fx.phys;
    auto &chem = (*fx.pool)[0];
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    REQUIRE(Ns >= 10);

    phys.setRANS(RANS_KOWilcox);
    int nR = phys.nRANSVars();
    REQUIRE(nR == 2);

    int I4 = 4; // dim+1 for 3D
    int nVars = I4 + 1 + nR + Ns1;

    using TU = typename PhysicsProperties<NS_EX>::TU;
    TU cons(nVars);
    cons.setZero();
    cons(0) = 0.8;
    cons(1) = 0.15;
    cons(2) = -0.05;
    cons(3) = 0.02;
    cons(4) = 0.65;
    cons(5) = 0.03;  // rho_k
    cons(6) = 120.0; // rho_omega
    int Isp = nVars - Ns1;
    cons(Isp + 0) = 0.024; // rhoY_H2
    cons(Isp + 3) = 0.178; // rhoY_O2

    real rho0 = makeScaledIdealGas().rho0;
    real U0 = makeScaledIdealGas().U0;
    real L0 = makeScaledIdealGas().L0;

    TU physCons;
    phys.consCodeToPhys(cons, physCons);

    // Physical value checks
    CHECK(physCons(0) == doctest::Approx(cons(0) * rho0).epsilon(1e-12));
    for (int j = 1; j <= 3; ++j)
        CHECK(physCons(j) == doctest::Approx(cons(j) * rho0 * U0).epsilon(1e-12));
    CHECK(physCons(4) == doctest::Approx(cons(4) * rho0 * U0 * U0).epsilon(1e-12));
    // RANS conservative: rho_k_phys = rho_k_code * rho0 * U0²
    CHECK(physCons(5) == doctest::Approx(cons(5) * rho0 * U0 * U0).epsilon(1e-12));
    // rho_omega_phys = rho_omega_code * rho0 * U0 / L0
    CHECK(physCons(6) == doctest::Approx(cons(6) * rho0 * U0 / L0).epsilon(1e-12));
    // Species: rhoY_phys = rhoY_code * rho0
    CHECK(physCons(Isp) == doctest::Approx(cons(Isp) * rho0).epsilon(1e-12));

    // Round-trip
    TU consRT;
    phys.consPhysToCode(physCons, consRT);
    for (int i = 0; i < nVars; ++i)
    {
        CAPTURE(i);
        CHECK(consRT(i) == doctest::Approx(cons(i)).epsilon(2e-12));
    }
}

TEST_CASE("PhysicsProperties prim code-phys round-trip multi-species with RANS and non-default scales")
{
    auto fx = makeScaledReactiveFixture();
    auto &phys = *fx.phys;
    auto &chem = (*fx.pool)[0];
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    REQUIRE(Ns >= 10);

    phys.setRANS(RANS_RKE);
    int nR = phys.nRANSVars();
    REQUIRE(nR == 2);

    using TU = typename PhysicsProperties<NS_EX>::TU;

    int I4 = 4;
    int Isp = I4 + 1 + nR;

    real rho0 = makeScaledIdealGas().rho0;
    real U0 = makeScaledIdealGas().U0;
    real T0 = makeScaledIdealGas().T0;
    real L0 = makeScaledIdealGas().L0;

    // PrimRhoP: [rho, u, v, w, p, k, epsilon, Y_0...]
    {
        int nVars = Isp + Ns1;
        TU prim(nVars);
        prim.setZero();
        prim(0) = 0.9;
        prim(1) = 0.14;
        prim(2) = -0.03;
        prim(3) = 0.01;
        prim(4) = 0.58;
        prim(5) = 0.03;  // k
        prim(6) = 200.0; // epsilon
        prim(Isp + 0) = 0.026;
        prim(Isp + 3) = 0.210;

        TU physPrim;
        phys.primCodeToPhys(prim, physPrim);
        CHECK(physPrim(5) == doctest::Approx(prim(5) * U0 * U0).epsilon(1e-12));           // k: U0²
        CHECK(physPrim(6) == doctest::Approx(prim(6) * U0 * U0 * U0 / L0).epsilon(1e-12)); // epsilon: U0³/L0

        TU primRT;
        phys.primPhysToCode(physPrim, primRT);
        for (int i = 0; i < nVars; ++i)
        {
            CAPTURE(i);
            CHECK(primRT(i) == doctest::Approx(prim(i)).epsilon(2e-12));
        }
    }

    // PrimRhoT: [rho, u, v, w, T, k, epsilon, Y_0...]
    {
        int nVars = Isp + Ns1;
        TU primRhoT(nVars);
        primRhoT.setZero();
        primRhoT(0) = 0.9;
        primRhoT(1) = 0.14;
        primRhoT(2) = -0.03;
        primRhoT(3) = 0.01;
        primRhoT(4) = 4.0; // T_code = T_phys / T0 = 1152.6/288.15 ≈ 4.0
        primRhoT(5) = 0.03;
        primRhoT(6) = 200.0;
        primRhoT(Isp + 0) = 0.026;
        primRhoT(Isp + 3) = 0.210;

        TU physPrimRT;
        phys.primRhoTCodeToPhys(primRhoT, physPrimRT);
        CHECK(physPrimRT(4) == doctest::Approx(primRhoT(4) * T0).epsilon(1e-12));
        CHECK(physPrimRT(5) == doctest::Approx(primRhoT(5) * U0 * U0).epsilon(1e-12));

        TU primRTRT;
        phys.primRhoTPhysToCode(physPrimRT, primRTRT);
        for (int i = 0; i < nVars; ++i)
        {
            CAPTURE(i);
            CHECK(primRTRT(i) == doctest::Approx(primRhoT(i)).epsilon(1e-12));
        }
    }

    // PrimTP: [T, u, v, w, p, k, epsilon, Y_0...]
    {
        int nVars = Isp + Ns1;
        TU primTP(nVars);
        primTP.setZero();
        primTP(0) = 4.0; // T_code
        primTP(1) = 0.14;
        primTP(2) = -0.03;
        primTP(3) = 0.01;
        primTP(4) = 0.58;
        primTP(5) = 0.03;
        primTP(6) = 200.0;
        primTP(Isp + 0) = 0.026;
        primTP(Isp + 3) = 0.210;

        TU physPrimTP;
        phys.primTPCodeToPhys(primTP, physPrimTP);
        CHECK(physPrimTP(0) == doctest::Approx(primTP(0) * T0).epsilon(1e-12));
        for (int j = 1; j <= 3; ++j)
            CHECK(physPrimTP(j) == doctest::Approx(primTP(j) * U0).epsilon(1e-12));
        CHECK(physPrimTP(4) == doctest::Approx(primTP(4) * rho0 * U0 * U0).epsilon(1e-12));
        CHECK(physPrimTP(5) == doctest::Approx(primTP(5) * U0 * U0).epsilon(1e-12));
        CHECK(physPrimTP(6) == doctest::Approx(primTP(6) * U0 * U0 * U0 / L0).epsilon(1e-12));

        TU primTPRT;
        phys.primTPPhysToCode(physPrimTP, primTPRT);
        for (int i = 0; i < nVars; ++i)
        {
            CAPTURE(i);
            CHECK(primTPRT(i) == doctest::Approx(primTP(i)).epsilon(1e-12));
        }
    }
}

// ============================================================================
// RANS scaling tests
// ============================================================================

TEST_CASE("PhysicsProperties RANS prim scale factors for all models")
{
    using Phys = PhysicsProperties<NS_EX>;

    auto igProp = makeScaledIdealGas();
    Phys phys(igProp);

    SUBCASE("SA nuTilde scale")
    {
        phys.setRANS(RANS_SA);
        CHECK(phys.nRANSVars() == 1);
        // nuTilde is non-dimensional: scale = 1.0
        CHECK(phys.ransPrimScaleCodeToPhys(0) == doctest::Approx(1.0).epsilon(1e-14));
        CHECK(phys.ransPrimScalePhysToCode(0) == doctest::Approx(1.0).epsilon(1e-14));
    }

    SUBCASE("KOWilcox k-omega scale")
    {
        phys.setRANS(RANS_KOWilcox);
        CHECK(phys.nRANSVars() == 2);
        real U0 = igProp.U0;
        real L0 = igProp.L0;
        // k: prim scale = U0²
        CHECK(phys.ransPrimScaleCodeToPhys(0) == doctest::Approx(U0 * U0).epsilon(1e-12));
        // omega: prim scale = U0/L0 (= 1/t0)
        CHECK(phys.ransPrimScaleCodeToPhys(1) == doctest::Approx(U0 / L0).epsilon(1e-12));
    }

    SUBCASE("KOSST (omega variant)")
    {
        phys.setRANS(RANS_KOSST);
        CHECK(phys.nRANSVars() == 2);
        real U0 = igProp.U0;
        real L0 = igProp.L0;
        CHECK(phys.ransPrimScaleCodeToPhys(0) == doctest::Approx(U0 * U0).epsilon(1e-12));
        CHECK(phys.ransPrimScaleCodeToPhys(1) == doctest::Approx(U0 / L0).epsilon(1e-12));
    }

    SUBCASE("RKE k-epsilon scale")
    {
        phys.setRANS(RANS_RKE);
        CHECK(phys.nRANSVars() == 2);
        real U0 = igProp.U0;
        real L0 = igProp.L0;
        // k: prim scale = U0²
        CHECK(phys.ransPrimScaleCodeToPhys(0) == doctest::Approx(U0 * U0).epsilon(1e-12));
        // epsilon: prim scale = U0³/L0
        CHECK(phys.ransPrimScaleCodeToPhys(1) == doctest::Approx(U0 * U0 * U0 / L0).epsilon(1e-12));
    }
}

TEST_CASE("PhysicsProperties RANS cons scale factors")
{
    using Phys = PhysicsProperties<NS_EX>;

    auto igProp = makeScaledIdealGas();
    Phys phys(igProp);
    real rho0 = igProp.rho0;
    real U0 = igProp.U0;
    real L0 = igProp.L0;

    phys.setRANS(RANS_KOWilcox);
    CHECK(phys.ransConsScaleCodeToPhys(0) == doctest::Approx(rho0 * U0 * U0).epsilon(1e-12)); // rho_k: rho0*U0²
    CHECK(phys.ransConsScaleCodeToPhys(1) == doctest::Approx(rho0 * U0 / L0).epsilon(1e-12)); // rho_omega: rho0*U0/L0
}

TEST_CASE("PhysicsProperties scaleRansPrim round-trip")
{
    using Phys = PhysicsProperties<NS_EX>;

    auto igProp = makeScaledIdealGas();
    Phys phys(igProp);
    phys.setRANS(RANS_RKE);
    CHECK(phys.nRANSVars() == 2);

    constexpr int dim = 3;
    int I4 = dim + 1;
    int nVars = I4 + 1 + 2; // fluid + RANS (k, epsilon), no species
    typename Phys::TU vec(nVars);
    vec << 1.0, 0.2, -0.1, 0.05, 0.7, 0.05, 200.0;

    typename Phys::TU original = vec;
    phys.scaleRansPrimCodeToPhys(vec);
    for (int j = 5; j < 7; ++j)
        CHECK(vec(j) != doctest::Approx(original(j)).epsilon(1e-12)); // scaled
    for (int j = 0; j < 5; ++j)
        CHECK(vec(j) == doctest::Approx(original(j)).epsilon(1e-12)); // fluid untouched

    phys.scaleRansPrimPhysToCode(vec);
    for (int i = 0; i < nVars; ++i)
        CHECK(vec(i) == doctest::Approx(original(i)).epsilon(2e-12));
}

TEST_CASE("PhysicsProperties scaleRansCons round-trip")
{
    using Phys = PhysicsProperties<NS_EX>;

    auto igProp = makeScaledIdealGas();
    Phys phys(igProp);
    phys.setRANS(RANS_SA);
    CHECK(phys.nRANSVars() == 1);

    constexpr int dim = 3;
    int I4 = dim + 1;
    int nVars = I4 + 1 + 1; // fluid + nuTilde
    typename Phys::TU vec(nVars);
    vec << 1.0, 0.2, -0.1, 0.05, 0.7, 0.02;

    typename Phys::TU original = vec;
    phys.scaleRansConsCodeToPhys(vec);
    // Cons scaling: nuTilde → * rho0
    real rho0 = igProp.rho0;
    CHECK(vec(5) == doctest::Approx(original(5) * rho0).epsilon(1e-12));

    phys.scaleRansConsPhysToCode(vec);
    for (int i = 0; i < nVars; ++i)
        CHECK(vec(i) == doctest::Approx(original(i)).epsilon(2e-12));
}

// ============================================================================
// StateValueOrigin round-trip tests
// ============================================================================

TEST_CASE("PhysicsProperties StateValueOrigin round-trip single-species")
{
    using Phys = PhysicsProperties<NS_EX>;
    using TU = typename Phys::TU;

    auto igProp = makeScaledIdealGas();
    Phys phys(igProp);
    phys.setRANS(RANS_KOWilcox);
    int nR = phys.nRANSVars();
    REQUIRE(nR == 2);

    constexpr int dim = 3;
    int I4 = dim + 1;
    int nVars = I4 + 1 + nR; // no species

    TU cons(nVars);
    cons << 1.0, 0.2, -0.1, 0.05, 0.7, 0.04, 300.0;

    // Test round-trip for each physically meaningful origin
    std::vector<StateValueOrigin> origins = {
        StateValueOrigin::Cons,
        StateValueOrigin::ConsSensible,
        StateValueOrigin::PrimRhoP,
        StateValueOrigin::PrimRhoT,
        StateValueOrigin::PrimTP,
        StateValueOrigin::ConsPhy,
        StateValueOrigin::ConsSensiblePhy,
        StateValueOrigin::PrimRhoPPhy,
        StateValueOrigin::PrimRhoTPhy,
        StateValueOrigin::PrimTPPhy,
    };

    for (auto origin : origins)
    {
        CAPTURE(StateValueOriginName(origin));

        TU state;
        phys.conservativeToStateValueOrigin(cons, state, origin);

        TU consRT;
        phys.stateValueOriginToConservative(state, consRT, origin);

        for (int i = 0; i < nVars; ++i)
        {
            CAPTURE(i);
            CHECK(consRT(i) == doctest::Approx(cons(i)).epsilon(2e-12));
        }
    }
}

TEST_CASE("PhysicsProperties StateValueOrigin round-trip multi-species Cantera")
{
    auto fx = makeScaledReactiveFixture();
    auto &phys = *fx.phys;
    auto &chem = (*fx.pool)[0];
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    REQUIRE(Ns >= 10);

    phys.setRANS(RANS_SA);
    int nR = phys.nRANSVars();
    REQUIRE(nR == 1);

    constexpr int dim = 3;
    int I4 = dim + 1;
    int Isp = I4 + 1 + nR;
    int nVars = Isp + Ns1;

    typename PhysicsProperties<NS_EX>::TU cons(nVars);
    cons.setZero();
    cons(0) = 0.85;
    cons(1) = 0.14;
    cons(2) = -0.04;
    cons(3) = 0.01;
    cons(4) = 0.62;
    cons(5) = 0.02; // nuTilde
    cons(Isp + 0) = 0.025;
    cons(Isp + 3) = 0.185;

    std::vector<StateValueOrigin> origins = {
        StateValueOrigin::Cons,
        StateValueOrigin::ConsSensible,
        StateValueOrigin::PrimRhoP,
        StateValueOrigin::PrimRhoT,
        StateValueOrigin::PrimTP,
        StateValueOrigin::ConsPhy,
        StateValueOrigin::ConsSensiblePhy,
        StateValueOrigin::PrimRhoPPhy,
        StateValueOrigin::PrimRhoTPhy,
        StateValueOrigin::PrimTPPhy,
    };

    for (auto origin : origins)
    {
        CAPTURE(StateValueOriginName(origin));

        typename PhysicsProperties<NS_EX>::TU state;
        phys.conservativeToStateValueOrigin(cons, state, origin);

        typename PhysicsProperties<NS_EX>::TU consRT;
        phys.stateValueOriginToConservative(state, consRT, origin);

        for (int i = 0; i < nVars; ++i)
        {
            CAPTURE(i);
            CHECK(consRT(i) == doctest::Approx(cons(i)).epsilon(1e-8));
        }
    }
}

// ============================================================================
// Parametric scale-config and linearity tests
// ============================================================================

struct ScaleConfig
{
    real U0, rho0, T0, L0;
    std::string label;
};

static std::vector<ScaleConfig> scaleConfigs()
{
    return {
        {379.0, 1.0, 1.0, 1.0, "U0=379"},
        {340.0, 1.225, 288.15, 2.0, "aero"},
        {1200.0, 0.3, 500.0, 0.05, "small L0"},
        {50.0, 1000.0, 280.0, 10.0, "large rho0"},
    };
}

static PhysicsProperties<NS_EX> makePhysForScale(const ScaleConfig &sc)
{
    EulerEvaluatorSettings<NS_EX>::IdealGasProperty igProp;
    igProp.gamma = 1.4;
    igProp.Rgas = 287.0;
    igProp.U0 = sc.U0;
    igProp.rho0 = sc.rho0;
    igProp.T0 = sc.T0;
    igProp.L0 = sc.L0;
    igProp.muGas = 1e-200;
    igProp.prGas = 0.72;
    return PhysicsProperties<NS_EX>(igProp);
}

TEST_CASE("PhysicsProperties cons code-phys linearity and range")
{
    using Phys = PhysicsProperties<NS_EX>;
    using TU = typename Phys::TU;

    for (auto &sc : scaleConfigs())
    {
        CAPTURE(sc.label);
        Phys phys = makePhysForScale(sc);
        phys.setRANS(RANS_KOWilcox);
        REQUIRE(phys.nRANSVars() == 2);

        constexpr int dim = 3;
        int I4 = dim + 1;
        int nVars = I4 + 1 + 2; // fluid + k + omega

        SUBCASE("2x code -> 2x phys linearity")
        {
            TU cons(nVars);
            cons << 0.5, 0.1, -0.05, 0.03, 0.4, 0.02, 100.0;

            TU phys1, phys2;
            phys.consCodeToPhys(cons, phys1);
            TU cons2 = cons * 2;
            phys.consCodeToPhys(cons2, phys2);
            for (int i = 0; i < nVars; ++i)
                CHECK(phys2(i) == doctest::Approx(2.0 * phys1(i)).epsilon(1e-12));
        }

        SUBCASE("small values")
        {
            TU cons(nVars);
            cons << 1e-4, 1e-5, -5e-6, 2e-6, 5e-5, 1e-6, 1e-2;
            TU physCons;
            phys.consCodeToPhys(cons, physCons);
            TU consRT;
            phys.consPhysToCode(physCons, consRT);
            for (int i = 0; i < nVars; ++i)
                CHECK(consRT(i) == doctest::Approx(cons(i)).epsilon(1e-10));
        }

        SUBCASE("large values")
        {
            TU cons(nVars);
            cons << 10.0, 5.0, -3.0, 2.0, 8.0, 2.0, 5000.0;
            TU physCons;
            phys.consCodeToPhys(cons, physCons);
            TU consRT;
            phys.consPhysToCode(physCons, consRT);
            for (int i = 0; i < nVars; ++i)
                CHECK(consRT(i) == doctest::Approx(cons(i)).epsilon(1e-7));
        }

        SUBCASE("zero RANS entries survive round-trip")
        {
            TU cons(nVars);
            cons << 1.0, 0.2, -0.1, 0.05, 0.7, 0.0, 0.0;
            TU physCons;
            phys.consCodeToPhys(cons, physCons);
            CHECK(physCons(5) == doctest::Approx(0.0).epsilon(1e-14));
            CHECK(physCons(6) == doctest::Approx(0.0).epsilon(1e-14));
            TU consRT;
            phys.consPhysToCode(physCons, consRT);
            for (int i = 0; i < nVars; ++i)
                CHECK(consRT(i) == doctest::Approx(cons(i)).epsilon(1e-12));
        }
    }
}

TEST_CASE("PhysicsProperties prim code-phys linearity and range")
{
    using Phys = PhysicsProperties<NS_EX>;
    using TU = typename Phys::TU;

    for (auto &sc : scaleConfigs())
    {
        CAPTURE(sc.label);
        Phys phys = makePhysForScale(sc);
        phys.setRANS(RANS_RKE);
        REQUIRE(phys.nRANSVars() == 2);

        constexpr int dim = 3;
        int I4 = dim + 1;
        int nVars = I4 + 1 + 2; // fluid + k + epsilon

        SUBCASE("2x prim -> 2x phys linearity (PrimRhoP)")
        {
            TU prim(nVars);
            prim << 0.6, 0.12, -0.04, 0.02, 0.35, 0.015, 150.0;

            TU phys1, phys2;
            phys.primCodeToPhys(prim, phys1);
            TU prim2 = prim * 2;
            phys.primCodeToPhys(prim2, phys2);
            for (int i = 0; i < nVars; ++i)
                CHECK(phys2(i) == doctest::Approx(2.0 * phys1(i)).epsilon(1e-12));
        }

        SUBCASE("2x prim -> 2x phys linearity (PrimRhoT)")
        {
            TU primRhoT(nVars);
            primRhoT << 0.6, 0.12, -0.04, 0.02, 4.0, 0.015, 150.0;

            TU phys1, phys2;
            phys.primRhoTCodeToPhys(primRhoT, phys1);
            TU prim2 = primRhoT * 2;
            phys.primRhoTCodeToPhys(prim2, phys2);
            for (int i = 0; i < nVars; ++i)
                CHECK(phys2(i) == doctest::Approx(2.0 * phys1(i)).epsilon(1e-12));
        }

        SUBCASE("small PrimRhoP")
        {
            TU prim(nVars);
            prim << 1e-4, 1e-5, -3e-6, 2e-6, 1e-4, 1e-6, 1e-2;
            TU physPrim;
            phys.primCodeToPhys(prim, physPrim);
            TU primRT;
            phys.primPhysToCode(physPrim, primRT);
            for (int i = 0; i < nVars; ++i)
                CHECK(primRT(i) == doctest::Approx(prim(i)).epsilon(1e-10));
        }

        SUBCASE("large PrimRhoP")
        {
            TU prim(nVars);
            prim << 5.0, 3.0, -2.0, 1.5, 4.0, 1.0, 3000.0;
            TU physPrim;
            phys.primCodeToPhys(prim, physPrim);
            TU primRT;
            phys.primPhysToCode(physPrim, primRT);
            for (int i = 0; i < nVars; ++i)
                CHECK(primRT(i) == doctest::Approx(prim(i)).epsilon(1e-7));
        }
    }
}

TEST_CASE("PhysicsProperties RANS scale linearity with multiple scales")
{
    using Phys = PhysicsProperties<NS_EX>;

    for (auto &sc : scaleConfigs())
    {
        CAPTURE(sc.label);
        Phys phys = makePhysForScale(sc);

        SUBCASE("KOWilcox: inverse identity")
        {
            phys.setRANS(RANS_KOWilcox);
            CHECK(phys.ransPrimScaleCodeToPhys(0) * phys.ransPrimScalePhysToCode(0) ==
                  doctest::Approx(1.0).epsilon(2e-12));
            CHECK(phys.ransPrimScaleCodeToPhys(1) * phys.ransPrimScalePhysToCode(1) ==
                  doctest::Approx(1.0).epsilon(2e-12));
            CHECK(phys.ransConsScaleCodeToPhys(0) * phys.ransConsScalePhysToCode(0) ==
                  doctest::Approx(1.0).epsilon(2e-12));
            CHECK(phys.ransConsScaleCodeToPhys(1) * phys.ransConsScalePhysToCode(1) ==
                  doctest::Approx(1.0).epsilon(2e-12));
        }

        SUBCASE("RKE: code->phys->code identity")
        {
            phys.setRANS(RANS_RKE);
            CHECK(phys.ransPrimScaleCodeToPhys(0) * phys.ransPrimScalePhysToCode(0) ==
                  doctest::Approx(1.0).epsilon(2e-12));
            CHECK(phys.ransPrimScaleCodeToPhys(1) * phys.ransPrimScalePhysToCode(1) ==
                  doctest::Approx(1.0).epsilon(2e-12));
        }

        SUBCASE("SA: nRANSVars==1, nuTilde scale==1")
        {
            phys.setRANS(RANS_SA);
            CHECK(phys.nRANSVars() == 1);
            CHECK(phys.ransPrimScaleCodeToPhys(0) == doctest::Approx(1.0).epsilon(1e-14));
        }
    }
}

TEST_CASE("PhysicsProperties default-RANS nRANSVars=0 cons and prim helpers")
{
    using Phys = PhysicsProperties<NS_EX>;
    using TU = typename Phys::TU;

    for (auto &sc : scaleConfigs())
    {
        CAPTURE(sc.label);
        Phys phys = makePhysForScale(sc);
        CHECK(phys.nRANSVars() == 0);

        constexpr int dim = 3;
        int I4 = dim + 1;
        int nVars = I4 + 1; // only fluid, no RANS, no species

        // cons round-trip
        {
            TU cons(nVars);
            cons << 1.0, 0.2, -0.1, 0.05, 0.7;
            TU physCons;
            phys.consCodeToPhys(cons, physCons);
            for (int j = 1; j <= dim; ++j)
                CHECK(physCons(j) == doctest::Approx(cons(j) * sc.rho0 * sc.U0).epsilon(1e-12));
            TU consRT;
            phys.consPhysToCode(physCons, consRT);
            for (int i = 0; i < nVars; ++i)
                CHECK(consRT(i) == doctest::Approx(cons(i)).epsilon(1e-12));
        }

        // PrimRhoP round-trip
        {
            TU prim(nVars);
            prim << 1.2, 0.22, -0.07, 0.03, 0.72;
            TU physPrim;
            phys.primCodeToPhys(prim, physPrim);
            for (int j = 1; j <= dim; ++j)
                CHECK(physPrim(j) == doctest::Approx(prim(j) * sc.U0).epsilon(1e-12));
            CHECK(physPrim(4) == doctest::Approx(prim(4) * sc.rho0 * sc.U0 * sc.U0).epsilon(1e-12));
            TU primRT;
            phys.primPhysToCode(physPrim, primRT);
            for (int i = 0; i < nVars; ++i)
                CHECK(primRT(i) == doctest::Approx(prim(i)).epsilon(1e-12));
        }
    }
}

TEST_CASE("PhysicsProperties reference derived scales")
{
    using Phys = PhysicsProperties<NS_EX>;
    auto sc = scaleConfigs()[1]; // aero: U0=340, rho0=1.225, T0=288.15, L0=2.0
    Phys phys = makePhysForScale(sc);

    real rho0 = sc.rho0, U0 = sc.U0, T0 = sc.T0, L0 = sc.L0;

    CHECK(phys.p0() == doctest::Approx(rho0 * U0 * U0).epsilon(1e-14));
    CHECK(phys.mu0() == doctest::Approx(rho0 * U0 * L0).epsilon(1e-12));
    CHECK(phys.k0() == doctest::Approx(rho0 * U0 * U0 * U0 * L0 / T0).epsilon(1e-12));
    CHECK(phys.D0() == doctest::Approx(U0 * L0).epsilon(1e-14));
    CHECK(phys.t0() == doctest::Approx(L0 / U0).epsilon(1e-14));

    CHECK(phys.toPhysT(1.0) == doctest::Approx(T0).epsilon(1e-13));
    CHECK(phys.toCodeT(phys.toPhysT(1.0)) == doctest::Approx(1.0).epsilon(1e-13));
    {
        real pPhys = phys.toPhysP(1.0);
        CHECK(pPhys == doctest::Approx(rho0 * U0 * U0).epsilon(1e-12));
        CHECK(phys.toPhysP(2.0) == doctest::Approx(2.0 * pPhys).epsilon(1e-12)); // linearity
    }
}

TEST_CASE("StateValue JSON round-trip")
{
    using TSV = StateValue;

    SUBCASE("round-trip primTP with valid data")
    {
        TSV sv;
        sv.originType = StateValueOrigin::PrimTP;
        sv.primTP.resize(7);
        sv.primTP << 300.0, 0.2, -0.1, 0.05, 0.8, 0.03, 200.0;
        sv.keepOnlyOrigin();

        nlohmann::ordered_json j = sv;
        TSV svBack = j.get<TSV>();
        CHECK(svBack.originType == sv.originType);
        REQUIRE(svBack.primTP.size() == sv.primTP.size());
        for (int i = 0; i < sv.primTP.size(); ++i)
            CHECK(svBack.primTP(i) == doctest::Approx(sv.primTP(i)).epsilon(1e-14));
    }

    SUBCASE("default StateValue round-trips as empty")
    {
        TSV sv;
        nlohmann::ordered_json j = sv;
        TSV svBack = j.get<TSV>();
        CHECK(svBack.originType == StateValueOrigin::None);
        CHECK(svBack.originVector().size() == 0);
    }

    SUBCASE("round-trip consSensiblePhy")
    {
        TSV sv;
        sv.originType = StateValueOrigin::ConsSensiblePhy;
        sv.consSensible_phy.resize(5);
        sv.consSensible_phy << 1.0, 0.2, -0.1, 0.05, 0.7;
        sv.keepOnlyOrigin();

        nlohmann::ordered_json j = sv;
        TSV svBack = j.get<TSV>();
        CHECK(svBack.originType == sv.originType);
        REQUIRE(svBack.consSensible_phy.size() == sv.consSensible_phy.size());
        for (int i = 0; i < sv.consSensible_phy.size(); ++i)
            CHECK(svBack.consSensible_phy(i) == doctest::Approx(sv.consSensible_phy(i)).epsilon(1e-14));
    }

    SUBCASE("round-trip all canonical origin types")
    {
        std::vector<std::pair<StateValueOrigin, std::string>> origins = {
            {StateValueOrigin::Cons, "cons"},
            {StateValueOrigin::ConsSensible, "consSensible"},
            {StateValueOrigin::PrimRhoP, "primRhoP"},
            {StateValueOrigin::PrimRhoT, "primRhoT"},
            {StateValueOrigin::PrimTP, "primTP"},
            {StateValueOrigin::ConsPhy, "cons_phy"},
            {StateValueOrigin::ConsSensiblePhy, "consSensible_phy"},
            {StateValueOrigin::PrimRhoPPhy, "primRhoP_phy"},
            {StateValueOrigin::PrimRhoTPhy, "primRhoT_phy"},
            {StateValueOrigin::PrimTPPhy, "primTP_phy"},
        };
        for (auto &[origin, name] : origins)
        {
            CAPTURE(name);
            int nVars = 5;
            TSV sv;
            sv.originType = origin;
            sv.originVectorMutable(origin).resize(nVars);
            sv.originVectorMutable(origin).setZero();
            sv.originVectorMutable(origin)(0) = 1.0;
            sv.originVectorMutable(origin)(4) = 0.7;
            sv.keepOnlyOrigin();

            nlohmann::ordered_json j = sv;
            CHECK(j["type"].get<std::string>() == name);
            CHECK(j["state"].size() == static_cast<size_t>(nVars));

            TSV svBack = j.get<TSV>();
            CHECK(svBack.originType == sv.originType);
            REQUIRE(svBack.originVector().size() == nVars);
            for (int i = 0; i < nVars; ++i)
                CHECK(svBack.originVector()(i) == doctest::Approx(sv.originVector()(i)).epsilon(1e-14));
        }
    }

    SUBCASE("invalid type name deserializes to empty StateValue")
    {
        auto j = nlohmann::ordered_json::parse("{\"type\":\"invalid\",\"state\":[1,2,3]}");
        TSV sv = j.get<TSV>();
        CHECK(sv.originType == StateValueOrigin::None);
        CHECK(sv.originVector().size() == 0);
    }
}
