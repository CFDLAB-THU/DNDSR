#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/Chemistry/ChemicalSource.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

using namespace DNDS::Euler::Chemistry;

static std::string mechFile()
{
    const char *env = std::getenv("DNDS_MECH_PATH");
    if (env)
        return std::string(env) + "/h2o2.yaml";
    return "h2o2.yaml"; // Cantera data dirs or CWD
}

TEST_CASE("ChemicalSource::productionRates — RHS signs at active T")
{
    ChemicalSource chem(mechFile());
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    auto names = chem.speciesNames();
    auto MW = chem.molecularWeights();

    std::vector<double> Y(Ns, 0.0);
    Y[0] = 0.028;
    Y[3] = 0.222;
    Y[9] = 0.75;
    double ysum = 0;
    for (auto y : Y)
        ysum += y;
    for (auto &y : Y)
        y /= ysum;

    double Rmix = chem.mixtureR(ConstSpeciesBufferView{Y.data(), Ns});

    for (double T : {416.0, 870.0, 1200.0, 1400.0, 1800.0, 2500.0})
    {
        double p = 1.0 * Rmix * T;
        ConstSpeciesBufferView Yv{Y.data(), Ns};
        std::vector<double> w(Ns);
        SpeciesBufferView ov{w.data(), Ns};
        chem.productionRates(T, p, Yv, ov);

        double maxAbsW = 0;
        for (int k = 0; k < Ns; ++k)
            maxAbsW = std::max(maxAbsW, std::abs(w[k]));

        INFO("T=" << T << "K p=" << p / 1e5 << "bar max|omega|=" << maxAbsW);

        if (maxAbsW > 1e-10)
        {
            double rhs_H2 = w[0] * MW[0];
            CHECK(rhs_H2 < 0);
        }

        if (T >= 1400)
        {
            double rhs_H = w[1] * MW[1];
            double rhs_O = w[2] * MW[2];
            INFO("T=" << T << " H:" << rhs_H << " O:" << rhs_O);
            CHECK(rhs_H > 0);
        }
    }
}

TEST_CASE("ChemicalSource::productionRatesAndJacobian — Jacobian sign convention")
{
    ChemicalSource chem(mechFile());
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    int nVars = 5 + Ns1;
    auto names = chem.speciesNames();
    auto MW = chem.molecularWeights();

    std::vector<double> Y(Ns, 0.0);
    Y[0] = 0.028;
    Y[3] = 0.222;
    Y[9] = 0.75;
    double ysum = 0;
    for (auto y : Y)
        ysum += y;
    for (auto &y : Y)
        y /= ysum;
    double Rmix = chem.mixtureR(ConstSpeciesBufferView{Y.data(), Ns});

    double T = 1800, p = 1.0 * Rmix * T, rho = 1.0;
    ConstSpeciesBufferView Yv{Y.data(), Ns};

    std::vector<double> w(Ns), jbuf(Ns * nVars, 0.0);
    SpeciesBufferView ov{w.data(), Ns};
    JacobianBufferView Jv{jbuf.data(), Ns, nVars, Ns};
    chem.productionRatesAndJacobian(T, p, rho, Yv, ov, Jv);

    int Isp = 5;

    SUBCASE("consumer species (H2) diagonal is negative")
    {
        double dW0_dU0 = Jv(0, Isp + 0) * MW[0]; // d(RHS_H2)/d(rhoY_H2)
        MESSAGE("d(RHS_H2)/d(rhoY_H2) = ", dW0_dU0, " 1/s");
        // H2 is consumed → more H2 → more consumption → RHS more negative → derivative negative
        CHECK(dW0_dU0 < 0);
    }

    SUBCASE("JSource sign matches body-force convention: JSource = -dR/dU")
    {
        // body-force: jac(I4,Seq123) -= massForce, where dR/dU = +massForce → JSource = -dR/dU
        // For chemistry: a consumer species has dR/dU < 0, so JSource_chem = -dR/dU > 0
        // The code does jac -= Mk*Jv → JSource = -Mk*Jv = -dR/dU.  CHECK that sign.
        double dR_dU_H2 = Jv(0, Isp + 0) * MW[0]; // dR/dU = Mk*Jv(k, Isp+k)
        // JSource = -dR/dU.  For dR/dU < 0, JSource > 0 → adds to diagonal → stabilising.
        double JSource_H2 = -dR_dU_H2;
        MESSAGE("dR/dU(H2) = ", dR_dU_H2, "  JSource = ", JSource_H2);
        CHECK(dR_dU_H2 < 0);   // consumer → negative derivative
        CHECK(JSource_H2 > 0); // JSource = -dR/dU → positive → added to diag → larger diagonal
    }

    SUBCASE("fuel self-coupling sign")
    {
        // H2 + O2 → products: dω_H2/dC_H2 < 0 (more H2 → faster consumption)
        double dw0_dc0 = Jv(0, Isp + 0); // = dω_H2/d(ρY_H2) * 1/M_H2???
        // Actually Jv(0, Isp+0) = dω_H2/dC_H2 * 1/M_H2  (from ChemicalSource.cpp line 158)
        // So dω_H2/dC_H2 = Jv(0, Isp+0) * M_H2
        double dw0_dc0_raw = Jv(0, Isp + 0) * MW[0];
        MESSAGE("d(omega_H2)/d(Y_H2) (via C) = ", Jv(0, Isp + 0));
        CHECK(dw0_dc0_raw < 0);
    }
}

TEST_CASE("ChemicalSource::mixtureR — gas constant correctness")
{
    ChemicalSource chem(mechFile());
    int Ns = chem.nSpecies();
    auto MW = chem.molecularWeights();

    // Pure N2
    {
        std::vector<double> Y(Ns, 0.0);
        Y[9] = 1.0; // N2 at index 9
        double R = chem.mixtureR(ConstSpeciesBufferView{Y.data(), Ns});
        double R_N2_expected = 8314.462618 / MW[9];
        MESSAGE("R_N2 = ", R, " expected ", R_N2_expected);
        CHECK(std::abs(R - R_N2_expected) / R_N2_expected < 1e-6);
    }

    // Fuel-air mixture
    {
        std::vector<double> Y(Ns, 0.0);
        Y[0] = 0.028;
        Y[3] = 0.222;
        Y[9] = 0.75;
        double ysum = 0;
        for (auto y : Y)
            ysum += y;
        for (auto &y : Y)
            y /= ysum;
        double R = chem.mixtureR(ConstSpeciesBufferView{Y.data(), Ns});
        double expected = 0.028 * 8314.462618 / MW[0] + 0.222 * 8314.462618 / MW[3] + 0.750 * 8314.462618 / MW[9];
        MESSAGE("R_mix = ", R, " expected ", expected);
        CHECK(std::abs(R - expected) / expected < 1e-4);
    }
}
