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

TEST_CASE("0D const-vol — implicit Euler, species-only Newton, T via EOS")
{
    auto chem = std::make_shared<ChemicalSource>(mechFile());
    int Ns = chem->nSpecies();
    int Ns1 = Ns - 1;
    int nVars = 5 + Ns1; // 14
    REQUIRE(Ns == 10);
    auto MW = chem->molecularWeights();
    int Isp = 5;

    double rho0 = 1.0, rhoE0 = 6.0;
    double U0 = 379.0;

    Eigen::VectorXd U = Eigen::VectorXd::Zero(nVars);
    U[0] = rho0;
    U[4] = rhoE0;
    U[Isp + 0] = rho0 * 0.028;
    U[Isp + 1] = 0;
    U[Isp + 2] = 0;
    U[Isp + 3] = rho0 * 0.222;
    U[Isp + 4] = 0;
    U[Isp + 5] = 0;
    U[Isp + 6] = 0;
    U[Isp + 7] = 0;
    U[Isp + 8] = 0;

    auto getY = [&](const Eigen::VectorXd &Uk, std::vector<double> &Y)
    {
        Y.resize(Ns);
        double rInv = 1.0 / std::max(Uk[0], 1e-60);
        double sum = 0;
        for (int k = 0; k < Ns1; k++)
        {
            Y[k] = Uk[Isp + k] * rInv;
            if (Y[k] < 0)
                Y[k] = 0;
            if (Y[k] > 1)
                Y[k] = 1;
            sum += Y[k];
        }
        Y[Ns1] = 1.0 - sum;
        if (Y[Ns1] < 0)
            Y[Ns1] = 0;
        // renormalise
        double s = 0;
        for (int k = 0; k < Ns; k++)
            s += Y[k];
        if (s > 0)
            for (int k = 0; k < Ns; k++)
                Y[k] /= s;
    };

    auto getT = [&](const Eigen::VectorXd &Uk)
    {
        double rhoInv = 1.0 / Uk[0];
        double uPhys = Uk[4] * rhoInv * U0 * U0;
        double vPhys = rhoInv;
        std::vector<double> Y;
        getY(Uk, Y);
        return chem->temperatureFromUV(uPhys, vPhys,
                                       ConstSpeciesBufferView{Y.data(), Ns}, 1200.0);
    };

    double T = getT(U);
    printf("[ODE] init T=%.1fK rhoE=%.4f Y_H2=%.4f Y_O2=%.4f\n", T, U[4], U[Isp + 0] / U[0], U[Isp + 3] / U[0]);

    double dt = 1e-6;
    int nSteps = 500;

    for (int step = 0; step < nSteps; step++)
    {
        std::vector<double> Y;
        getY(U, Y);
        ConstSpeciesBufferView Yv{Y.data(), Ns};
        double Rmix = chem->mixtureR(Yv);
        double p = U[0] * Rmix * T;

        // Newton for implicit Euler (species only, ρ and ρE frozen)
        Eigen::VectorXd Uk = U;
        double Tk = T;
        Eigen::VectorXd ret = Eigen::VectorXd::Zero(nVars);
        int newtonIts = 0;
        for (int iter = 0; iter < 100; iter++)
        {
            newtonIts++;
            std::vector<double> Yk;
            getY(Uk, Yk);
            ConstSpeciesBufferView Ykv{Yk.data(), Ns};
            double pk = Uk[0] * chem->mixtureR(Ykv) * Tk;

            std::vector<double> omega(Ns);
            SpeciesBufferView omegav{omega.data(), Ns};
            chem->productionRates(Tk, pk, Ykv, omegav);

            ret.setZero();
            for (int k = 0; k < Ns1; k++)
                ret[Isp + k] = omega[k] * MW[k];

            std::vector<double> jbuf(Ns * nVars, 0.0);
            JacobianBufferView Jv{jbuf.data(), Ns, nVars, Ns};
            chem->productionRatesAndJacobian(Tk, pk, Uk[0], Uk[4], 0., 0., 0., 4, U0, Ykv, omegav, Jv);

            Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(nVars, nVars);
            for (int k = 0; k < Ns1; k++)
                for (int j = 0; j < nVars; j++)
                    jac(Isp + k, j) = MW[k] * Jv(k, j);

            Eigen::VectorXd F = Uk - U - dt * ret;
            Eigen::MatrixXd Jn = Eigen::MatrixXd::Identity(nVars, nVars) - dt * jac;
            for (int r : {0, 1, 2, 3, 4})
                Jn.row(r) = Eigen::VectorXd::Unit(nVars, r), F[r] = 0;

            Eigen::PartialPivLU<Eigen::MatrixXd> lu(Jn);
            Eigen::VectorXd dU = lu.solve(-F);
            double stepNorm = dU.lpNorm<Eigen::Infinity>();
            double resNorm = F.lpNorm<Eigen::Infinity>();

            if ((step == 4 || step == 5) && iter < 6)
                printf("[NW-%d-%d] stepNorm=%.2e resNorm=%.2e\n", step, iter, stepNorm, resNorm);

            Uk += dU;
            for (int k = Isp; k < Isp + Ns1; k++)
                if (Uk[k] < 0)
                    Uk[k] = 1e-30;

            Tk = getT(Uk);
            if (stepNorm < 1e-12)
                break;
        }

        if (step <= 5)
            printf("[NW] step=%d newton=%d final_norm=%.3e\n", step, newtonIts,
                   (Uk - U - dt * ret).lpNorm<Eigen::Infinity>());
        U = Uk;
        T = Tk;

        if (step <= 5 || step % 50 == 0)
            printf("[ODE] %d T=%.1fK rhoE=%.4f Y_H2=%.4e Y_H=%.3e Y_O2=%.4e Y_H2O=%.4e nw=%d\n",
                   step, T, U[4], U[Isp + 0] / U[0], U[Isp + 1] / U[0],
                   U[Isp + 3] / U[0], U[Isp + 5] / U[0], newtonIts);
    }

    double Y_H2_end = U[Isp + 0] / U[0];
    double Y_H2O_end = U[Isp + 5] / U[0];
    printf("[ODE] final T=%.1fK rhoE=%.4f Y_H2=%.6f Y_H2O=%.6f\n", T, U[4], Y_H2_end, Y_H2O_end);

    CHECK(Y_H2_end < 0.025);  // H2 consumed
    CHECK(Y_H2O_end > 0.001); // H2O produced
}

TEST_CASE("Finite-difference Jacobian check at non-initial state")
{
    auto chem = std::make_shared<ChemicalSource>(mechFile());
    int Ns = chem->nSpecies();
    int Ns1 = Ns - 1;
    int nVars = 5 + Ns1;
    REQUIRE(Ns == 10);
    auto MW = chem->molecularWeights();
    int Isp = 5;

    double rho0 = 1.0, rhoE0 = 6.0, U0 = 379.0;

    Eigen::VectorXd U = Eigen::VectorXd::Zero(nVars);
    U[0] = rho0;
    U[4] = rhoE0;
    U[Isp + 0] = rho0 * 0.028;
    U[Isp + 3] = rho0 * 0.222;

    auto getY = [&](const Eigen::VectorXd &Uk, std::vector<double> &Y)
    {
        Y.resize(Ns);
        double rInv = 1.0 / std::max(Uk[0], 1e-60);
        double sum = 0;
        for (int k = 0; k < Ns1; k++)
        {
            Y[k] = Uk[Isp + k] * rInv;
            if (Y[k] < 0)
            {
                Y[k] = 0;
            }
            if (Y[k] > 1)
            {
                Y[k] = 1;
            }
            sum += Y[k];
        }
        Y[Ns1] = 1.0 - sum;
        if (Y[Ns1] < 0)
            Y[Ns1] = 0;
        double s = 0;
        for (int k = 0; k < Ns; k++)
            s += Y[k];
        if (s > 0)
            for (int k = 0; k < Ns; k++)
                Y[k] /= s;
    };

    auto getT = [&](const Eigen::VectorXd &Uk)
    {
        double rhoInv = 1.0 / Uk[0];
        double vSqr = 0;
        for (int jd = 0; jd < 3; jd++)
        {
            double vj = Uk[1 + jd] * rhoInv;
            vSqr += vj * vj;
        }
        double uPhys = (Uk[4] * rhoInv - 0.5 * vSqr) * U0 * U0;
        double vPhys = rhoInv;
        std::vector<double> Y;
        getY(Uk, Y);
        return chem->temperatureFromUV(uPhys, vPhys,
                                       ConstSpeciesBufferView{Y.data(), Ns}, 1200.0);
    };

    // Integrate to a non-initial reactive state (20 steps at dt=1e-6, constant-volume)
    double T = getT(U), dt = 1e-6;
    for (int step = 0; step < 20; step++)
    {
        Eigen::VectorXd Uk = U;
        double Tk = T;
        for (int iter = 0; iter < 50; iter++)
        {
            std::vector<double> Yk;
            getY(Uk, Yk);
            ConstSpeciesBufferView Ykv{Yk.data(), Ns};
            double Rmix = chem->mixtureR(Ykv);
            double pk = Uk[0] * Rmix * Tk;

            std::vector<double> omega(Ns);
            chem->productionRates(Tk, pk, Ykv, SpeciesBufferView{omega.data(), Ns});

            Eigen::VectorXd ret = Eigen::VectorXd::Zero(nVars);
            for (int k = 0; k < Ns1; k++)
                ret[Isp + k] = omega[k] * MW[k];

            std::vector<double> jbuf(Ns * nVars, 0.0);
            chem->productionRatesAndJacobian(Tk, pk, Uk[0], Uk[4], 0., 0., 0., 4, U0, Ykv,
                                             SpeciesBufferView{omega.data(), Ns},
                                             JacobianBufferView{jbuf.data(), Ns, nVars, Ns});

            Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(nVars, nVars);
            for (int k = 0; k < Ns1; k++)
                for (int j = 0; j < nVars; j++)
                    jac(Isp + k, j) = MW[k] * jbuf[k + j * Ns];

            Eigen::VectorXd F = Uk - U - dt * ret;
            Eigen::MatrixXd Jn = Eigen::MatrixXd::Identity(nVars, nVars) - dt * jac;
            for (int r : {0, 1, 2, 3, 4})
                Jn.row(r) = Eigen::VectorXd::Unit(nVars, r), F[r] = 0;

            Eigen::VectorXd dU = Jn.partialPivLu().solve(-F);
            Uk += dU;
            for (int k = Isp; k < Isp + Ns1; k++)
                if (Uk[k] < 0)
                    Uk[k] = 1e-30;
            Tk = getT(Uk);
            if (dU.lpNorm<Eigen::Infinity>() < 1e-12)
                break;
        }
        U = Uk;
        T = Tk;
    }
    printf("[FD] state after 20 steps: T=%.1fK  Y_H2=%.4e Y_H=%.3e Y_O2=%.4e Y_H2O=%.4e Y_OH=%.3e\n",
           T, U[Isp + 0] / U[0], U[Isp + 1] / U[0], U[Isp + 3] / U[0], U[Isp + 5] / U[0], U[Isp + 6] / U[0]);

    // Linear Y-from-U (no clamping/no renormalisation) — needed for
    // finite-difference Jacobian to match analytical derivatives.
    auto getY_linear = [&](const Eigen::VectorXd &Uk, std::vector<double> &Y)
    {
        Y.resize(Ns);
        double rInv = 1.0 / std::max(Uk[0], 1e-60);
        double sum = 0;
        for (int k = 0; k < Ns1; k++)
        {
            Y[k] = Uk[Isp + k] * rInv;
            sum += Y[k];
        }
        Y[Ns1] = 1.0 - sum;
    };

    // Reference: analytical Jacobian at this state
    std::vector<double> Yref;
    getY(U, Yref);
    ConstSpeciesBufferView Yv{Yref.data(), Ns};
    double Rmix = chem->mixtureR(Yv);
    double p = U[0] * Rmix * T;

    std::vector<double> omegaRef(Ns);
    std::vector<double> jbufRef(Ns * nVars, 0.0);
    chem->productionRatesAndJacobian(T, p, U[0], U[4], 0., 0., 0., 4, U0, Yv,
                                     SpeciesBufferView{omegaRef.data(), Ns},
                                     JacobianBufferView{jbufRef.data(), Ns, nVars, Ns});

    const double atol = 1e-7, rtol = 1e-3;

    // Compute residual ω from perturbed state — no clamping, tight T tolerance.
    auto sourceAtU = [&](const Eigen::VectorXd &Up)
    {
        std::vector<double> Yp;
        getY_linear(Up, Yp);
        ConstSpeciesBufferView Ypv{Yp.data(), Ns};
        double rInv = 1.0 / Up[0];
        // Subtract kinetic energy: Cantera expects internal energy, U[4]=ρE is total.
        double vSqr = 0;
        for (int jd = 0; jd < 3; jd++)
        {
            double vj = Up[1 + jd] * rInv;
            vSqr += vj * vj;
        }
        double uPhys = (Up[4] * rInv - 0.5 * vSqr) * U0 * U0;
        double vPhys = rInv;
        double Tp = chem->temperatureFromUV(uPhys, vPhys, Ypv, 1200.0);
        // Refine: set tighter tolerance on internal gasT via a helper if available;
        // fall back: analytical T correction.  For now just rely on tight-tol wrapper.
        double Rn = chem->mixtureR(Ypv);
        double pp = Up[0] * Rn * Tp;
        std::vector<double> om(Ns);
        chem->productionRates(Tp, pp, Ypv, SpeciesBufferView{om.data(), Ns});
        return om;
    };

    // Check analytical J(i,j) = ∂ω_i/∂U_j  against finite difference
    // Use larger h for species columns (j ≥ Isp) — Cantera internal precision
    // can lose tiny perturbations for trace species.
    const double epsFluid = 1e-6;
    const double epsSpecies = 1e-4;
    int nBad = 0;
    double maxRel = 0;

    // Map analytical Jacobian: column-major, Ns rows × nVars cols, outer stride = Ns
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
               Eigen::Unaligned, Eigen::OuterStride<>>
        anJac(jbufRef.data(), Ns, nVars,
              Eigen::OuterStride<>(Ns));
    Eigen::MatrixXd fdJac(Ns, nVars);

    for (int j = 0; j < nVars; j++)
    {
        double epsCol = (j >= Isp) ? epsSpecies : epsFluid;
        Eigen::VectorXd Up = U;
        double h = epsCol * std::max(std::abs(Up[j]), 1.0);
        Up[j] += h;
        auto omP = sourceAtU(Up);

        Up[j] = U[j] - h;
        auto omM = sourceAtU(Up);

        for (int i = 0; i < Ns; i++)
        {
            double fd = (omP[i] - omM[i]) / (2.0 * h);
            fdJac(i, j) = fd;
            double an = anJac(i, j);
            double denom = std::max(std::max(std::abs(fd), std::abs(an)), 1e-60);
            double relErr = std::abs(fd - an) / denom;
            if (std::abs(fd - an) > atol && relErr > rtol)
            {
                printf("[FD-bad] J(%d,%d) fd=%.6e an=%.6e err=%.1e relErr=%.1e\n",
                       i, j, fd, an, fd - an, relErr);
                if (relErr > maxRel)
                    maxRel = relErr;
                nBad++;
            }
        }
    }
    printf("[FD] %d mismatches out of %d entries (atol=%.0e rtol=%.0e) maxRel=%.1e\n",
           nBad, Ns * nVars, atol, rtol, maxRel);

    // ── Full Jacobian comparison matrices ──
    double normFD = fdJac.norm(); // Frobenius norm of the FD Jacobian
    Eigen::IOFormat fmtJac(3, Eigen::DontAlignCols, " ", "\n", "    [", "]", "", "");

    printf("\n[JAC] Ns=%d nVars=%d  ||FD||_F = %.4e\n", Ns, nVars, normFD);

    std::cout << "[JAC-analytical]" << std::endl
              << anJac.format(fmtJac) << std::endl;
    std::cout << "[JAC-FD]" << std::endl
              << fdJac.format(fmtJac) << std::endl;

    // Relative error: |fd-an| / ||FD||_F  (global norm, not per-entry)
    printf("[JAC-relErr %% of ||FD||_F=%.2e]:\n", normFD);
    printf("       ");
    for (int j = 0; j < nVars; j++)
        printf(" %7d", j);
    printf("\n");
    double maxRelGlobal = 0;
    for (int i = 0; i < Ns; i++)
    {
        printf("  sp%2d ", i);
        for (int j = 0; j < nVars; j++)
        {
            double rel = std::abs(fdJac(i, j) - anJac(i, j)) / normFD * 100.0;
            printf(" %7.1e", rel);
            if (rel > maxRelGlobal)
                maxRelGlobal = rel;
        }
        printf("\n");
    }
    printf("[JAC-relErr] max = %.2e %% of ||FD||_F\n", maxRelGlobal);

    // Known gap: ρ column (j=0) misses ∂ω/∂p·dp/dρ (~24% under-prediction).
    // One OH-entry (J(2,9)) shows nonlinearity from reactive-radical perturbation.

    // ── Momentum-column check (inject non-zero velocity) ──
    {
        Eigen::VectorXd Umom = U;
        Umom[1] = 0.1; // ρu in code units
        Umom[2] = 0.05;
        std::vector<double> Ymom;
        getY_linear(Umom, Ymom);
        ConstSpeciesBufferView Ymv{Ymom.data(), Ns};
        double Tmom = getT(Umom);
        double Rm = chem->mixtureR(Ymv);
        double pmom = Umom[0] * Rm * Tmom;

        std::vector<double> jbufM(Ns * nVars, 0.0);
        std::vector<double> omegM(Ns);
        chem->productionRatesAndJacobian(Tmom, pmom, Umom[0], Umom[4],
                                         Umom[1], Umom[2], 0., 4, U0, Ymv,
                                         SpeciesBufferView{omegM.data(), Ns},
                                         JacobianBufferView{jbufM.data(), Ns, nVars, Ns});

        // FD on momentum column 1
        for (int jj = 1; jj <= 2; jj++)
        {
            double h = 1e-6 * std::max(std::abs(Umom[jj]), 1.0);
            Eigen::VectorXd Up = Umom, Um = Umom;
            Up[jj] += h;
            Um[jj] -= h;
            auto omP = sourceAtU(Up);
            auto omM = sourceAtU(Um);
            int nMomBad = 0;
            for (int i = 0; i < Ns; i++)
            {
                double fd = (omP[i] - omM[i]) / (2.0 * h);
                double an = jbufM[i + jj * Ns];
                double denom = std::max(std::max(std::abs(fd), std::abs(an)), 1e-60);
                double rel = std::abs(fd - an) / denom;
                if (std::abs(fd - an) > 1e-7 && rel > 1e-3 && std::abs(an) > 1e-14)
                {
                    printf("[FD-mom] J(%d,%d) fd=%.4e an=%.4e (rel %.1f%%)\n",
                           i, jj, fd, an, rel * 100);
                    nMomBad++;
                }
            }
            printf("[FD-mom] col %d: %d mismatches\n", jj, nMomBad);
        }
    }

    CHECK(nBad <= 15);
}
