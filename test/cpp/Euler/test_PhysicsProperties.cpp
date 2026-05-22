#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/Chemistry/ChemicalSource.hpp"
#include "Euler/Euler.hpp"
#include "Euler/Physics/PhysicsProperties.hpp"

#include <cmath>
#include <cstdlib>
#include <memory>
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
    igProp.recomputeDerived();
    return igProp;
}

static auto makeReactiveFixture()
{
    auto pool = std::make_shared<std::vector<ChemicalSource>>();
    pool->emplace_back(mechFile());

    auto phys = std::make_unique<PhysicsProperties<NS_EX>>(makeIdealGasProperty<NS_EX>());
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
    phys.template primToConservative<3>(prim, cons);
    phys.template conservativeToPrimitive<3>(cons, primOut);
    for (int i = 0; i < prim.size(); ++i)
    {
        CAPTURE(i);
        CHECK(primOut(i) == doctest::Approx(prim(i)).epsilon(1e-12));
    }

    TU primRhoT;
    TU primRhoTOut;
    primRhoT << 1.2, 0.22, -0.07, 0.03, 300.0;
    phys.template primRhoTToConservative<3>(primRhoT, cons);
    phys.template conservativeToPrimRhoT<3>(cons, primRhoTOut);
    for (int i = 0; i < primRhoT.size(); ++i)
    {
        CAPTURE(i);
        CHECK(primRhoTOut(i) == doctest::Approx(primRhoT(i)).epsilon(1e-12));
    }

    TU primTP;
    TU primTPOut;
    primTP << 300.0, 0.22, -0.07, 0.03, 0.72;
    phys.template primTPToConservative<3>(primTP, cons);
    phys.template conservativeToPrimTP<3>(cons, primTPOut);
    for (int i = 0; i < primTP.size(); ++i)
    {
        CAPTURE(i);
        CHECK(primTPOut(i) == doctest::Approx(primTP(i)).epsilon(1e-12));
    }
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
    real Cp = phys.toCode(igProp.CpGas);
    real Rgas = phys.toCode(igProp.Rgas);

    real TStaticExpected = TTotal - 0.5 * vSqr / Cp;
    real pStaticExpected = pTotal * std::pow(TStaticExpected / TTotal, igProp.gamma / (igProp.gamma - 1));
    real rhoExpected = pStaticExpected / (Rgas * TStaticExpected);

    phys.template totalToStaticPrimitive<3>(pTotal, TTotal, primStatic);

    CHECK(primStatic(0) == doctest::Approx(rhoExpected).epsilon(1e-12));
    CHECK(primStatic(4) == doctest::Approx(pStaticExpected).epsilon(1e-12));

    TU cons;
    phys.template primToConservative<3>(primStatic, cons);
    CHECK(phys.template temperature<3>(cons) == doctest::Approx(TStaticExpected).epsilon(1e-12));
}

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

    phys.template primToConservative<3>(prim, cons, options);
    phys.template conservativeToPrimitive<3>(cons, primOut, options);
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
    phys.template primTPToConservative<3>(primTP, cons, options);
    phys.template conservativeToPrimTP<3>(cons, primTPOut, options);
    for (int i = 0; i < primTP.size(); ++i)
    {
        CAPTURE(i);
        CHECK(primTPOut(i) == doctest::Approx(primTP(i)).epsilon(1e-11));
    }
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
    phys.template totalToStaticPrimitive<3>(pTotal, TTotal, primStatic, options);

    TU cons(5 + Ns1);
    phys.template primToConservative<3>(primStatic, cons, options);
    real TStatic = phys.template temperature<3>(cons, primStatic(4), options.temperatureUVTolerance);
    real Rgas = phys.Rgas(cons);
    real vSqr = primStatic(Eigen::seq(Eigen::fix<1>, Eigen::fix<3>)).squaredNorm();

    auto Yv = chem.massFractions(1.0, primStatic.data() + 5, Ns1);
    std::vector<double> Ybuf(Ns);
    for (int k = 0; k < Ns; ++k)
        Ybuf[k] = Yv[k];
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
