#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/Chemistry/ChemicalSource.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace DNDS::Euler::Chemistry;

static std::string mechFile()
{
    const char *env = std::getenv("DNDS_MECH_PATH");
    return env ? std::string(env) + "/h2o2.yaml" : "h2o2.yaml";
}

TEST_CASE("0D implicit Euler — single cell chemical ODE")
{
    auto chem = std::make_shared<ChemicalSource>(mechFile());
    int Ns = chem->nSpecies();
    int Ns1 = Ns - 1;
    int nVars = 5 + Ns1; // 14
    REQUIRE(Ns == 10);

    auto MW = chem->molecularWeights();

    // Species order: H2, H, O, O2, OH, H2O, HO2, H2O2, AR, N2
    double Y0_H2 = 0.028, Y0_O2 = 0.222, Y0_N2 = 0.75;
    double rho0 = 1.0, rhoE0 = 6.0; // ρE=6 → T ≈ 1204K

    // Conservative state: ρ, ρu, ρv, ρw, ρE, then species 0..Ns1-1
    Eigen::VectorXd U = Eigen::VectorXd::Zero(nVars);
    U[0] = rho0;
    U[4] = rhoE0;
    int Isp = 5;
    U[Isp + 0] = rho0 * Y0_H2;
    U[Isp + 1] = 0; // H
    U[Isp + 2] = 0; // O
    U[Isp + 3] = rho0 * Y0_O2;
    U[Isp + 4] = 0; // OH
    U[Isp + 5] = 0; // H2O
    U[Isp + 6] = 0; // HO2
    U[Isp + 7] = 0; // H2O2
    U[Isp + 8] = 0; // AR

    // Helper: mass fractions from U
    auto getY = [&](const Eigen::VectorXd &U, std::vector<double> &Y)
    {
        Y.resize(Ns);
        double rhoInv = 1.0 / std::max(U[0], 1e-60);
        double sum = 0;
        for (int k = 0; k < Ns1; k++)
        {
            Y[k] = U[Isp + k] * rhoInv;
            sum += Y[k];
        }
        Y[Ns1] = 1.0 - sum;
    };

    // Helper: T from U using Cantera (manual Newton)
    auto getT = [&](const Eigen::VectorXd &U) -> double
    {
        double rho = U[0];
        double rhoInv = 1.0 / rho;
        double uInternal = U[4] * rhoInv;         // v=0
        double uPhys = uInternal * 379.0 * 379.0; // U0=379
        double vPhys = rhoInv / 1.0;              // rho0=1

        std::vector<double> Y;
        getY(U, Y);
        ConstSpeciesBufferView Yv{Y.data(), Ns};

        return chem->temperatureFromUV(uPhys, vPhys, Yv, 1200.0);
    };

    // Initial T
    double T_init = getT(U);
    printf("[ODE0D] T_init=%.1fK\n", T_init);

    // ---- Time integration (implicit Euler + Newton) ----
    double dt = 1e-6; // microsecond steps for stiff chemistry
    int nSteps = 100;

    double T = T_init;
    for (int step = 0; step < nSteps; step++)
    {
        // p from ideal gas: p = ρ·Rmix·T
        std::vector<double> Y;
        getY(U, Y);
        ConstSpeciesBufferView Yv{Y.data(), Ns};
        double Rmix = chem->mixtureR(Yv); // physical R [J/kg/K]
        double p = U[0] * Rmix * T;

        // Newton for implicit Euler
        Eigen::VectorXd Uk = U;
        double Tk = T;
        for (int iter = 0; iter < 15; iter++)
        {
            // RHS and Jacobian at current Newton iterate
            std::vector<double> Yk;
            getY(Uk, Yk);
            ConstSpeciesBufferView Ykv{Yk.data(), Ns};
            double Rmix_k = chem->mixtureR(Ykv);
            double pk = Uk[0] * Rmix_k * Tk;

            // Production rates
            std::vector<double> omega(Ns);
            SpeciesBufferView omegav{omega.data(), Ns};
            chem->productionRates(Tk, pk, Ykv, omegav);

            // RHS: ret[Isp+k] = M_k * ω_k
            Eigen::VectorXd ret = Eigen::VectorXd::Zero(nVars);
            for (int k = 0; k < Ns1; k++)
                ret[Isp + k] = omega[k] * MW[k];

            // Energy source: -Σ h_k * M_k * ω_k
            // We need h_k from Cantera.  Approximate: cv_mix * dT = Σ h_k * M_k * ω_k * dt
            // More precisely: for the EOS, ρE = ρ·u + 0.5·ρ·v².  u changes by Σ h_k * Y_k * ω_k...
            // Actually, in the CFD solver the energy equation handles this implicitly.
            // For 0-D, the chemistry changes composition but NOT ρE directly.
            // The temperature change comes from recomputing EOS with new composition.
            // So we DON'T add energy source in ret[4].
            // The Newton iteration with J = dω/d(ρE) handles the temperature feedback.

            // Jacobian
            std::vector<double> jbuf(Ns * nVars, 0.0);
            JacobianBufferView Jv{jbuf.data(), Ns, nVars, Ns};
            chem->productionRatesAndJacobian(Tk, pk, Uk[0], Ykv, omegav, Jv);

            Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(nVars, nVars);
            for (int k = 0; k < Ns1; k++)
                for (int j = 0; j < nVars; j++)
                    jac(Isp + k, j) = MW[k] * Jv(k, j);

            // Newton: F = Uk - U - dt * ret,  J = I - dt * jac
            Eigen::VectorXd F = Uk - U - dt * ret;
            Eigen::MatrixXd Jnewt = Eigen::MatrixXd::Identity(nVars, nVars) - dt * jac;

            // Enforce ρ and ρE identically (no source for these in 0D)
            Jnewt.row(0) = Eigen::VectorXd::Unit(nVars, 0);
            Jnewt.row(4) = Eigen::VectorXd::Unit(nVars, 4);
            F[0] = 0;
            F[4] = 0;

            Eigen::PartialPivLU<Eigen::MatrixXd> lu(Jnewt);
            Eigen::VectorXd dU = lu.solve(-F);

            double norm = dU.lpNorm<Eigen::Infinity>();

            // Clamp species positivity
            Uk += dU * 0.5; // damped Newton
            for (int k = Isp; k < Isp + Ns1; k++)
                if (Uk[k] < 0)
                    Uk[k] = 1e-20;

            // Recompute T after update
            Tk = getT(Uk);

            if (norm < 1e-12)
                break;
        }

        U = Uk;
        T = Tk;

        if (step % 10 == 0)
        {
            std::vector<double> Yf;
            getY(U, Yf);
            printf("[ODE0D] step=%d T=%.1fK Y_H2=%.4e Y_H=%.4e Y_O2=%.4e Y_H2O=%.4e\n",
                   step, T, Yf[0], Yf[1], Yf[3], Yf[5]);
        }
    }

    double Y_H2_end = U[Isp + 0] / U[0];
    double Y_H2O_end = U[Isp + 5] / U[0];
    printf("[ODE0D] Final: T=%.1fK Y_H2=%.6f Y_H2O=%.6f\n", T, Y_H2_end, Y_H2O_end);

    CHECK(Y_H2_end < 0.028);
}
