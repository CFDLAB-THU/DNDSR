#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/Chemistry/ChemicalSource.hpp"
#include "Euler/Euler.hpp"
#include "Euler/Physics/ConstVolTrajectory.hpp"
#include "Euler/Physics/PhysicsProperties.hpp"
#include "cantera/zerodim.h"
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace DNDS::Euler;
using namespace DNDS::Euler::Chemistry;
using json = nlohmann::ordered_json;

static std::string mechFile()
{
    const char *env = std::getenv("DNDS_MECH_PATH");
    return env ? std::string(env) + "/h2o2.yaml" : "h2o2.yaml";
}

// Shared setup: create a PhysicsProperties<NS_EX> with IdealGasProperty
// matching the reference-config scales, plus the underlying ChemicalSource pool
// held externally so the test can call kinetics/transport directly.
static auto makeTestFixture()
{
    auto pool = std::make_shared<std::vector<ChemicalSource>>();
    pool->emplace_back(mechFile()); // pool[0] is the per-thread instance

    typename EulerEvaluatorSettings<NS_EX>::IdealGasProperty igProp;
    igProp.gamma = 1.4;
    igProp.Rgas = 287.0; // physical J/(kg·K); toCode() converts via U0²/T0
    igProp.U0 = 379.0;
    igProp.rho0 = 1.0;
    igProp.T0 = 1.0;
    igProp.muGas = 1e-200;
    igProp.prGas = 0.72;

    auto phys = std::make_unique<PhysicsProperties<NS_EX>>(igProp);
    phys->setChemicalSourcePool(pool); // shares ownership of pool

    struct Fixture
    {
        std::shared_ptr<std::vector<ChemicalSource>> pool;
        std::unique_ptr<PhysicsProperties<NS_EX>> phys;
    };
    return Fixture{std::move(pool), std::move(phys)};
}

TEST_CASE("Cantera custom const-volume affine species reactor")
{
    auto sol = Cantera::newSolution(mechFile(), "", "");
    auto gas = sol->thermo();
    ChemicalSource chem(mechFile());
    auto iH2 = gas->speciesIndex("H2");
    auto iO2 = gas->speciesIndex("O2");
    auto iH2O = gas->speciesIndex("H2O");
    auto iN2 = gas->speciesIndex("N2");
    REQUIRE(iH2 != Cantera::npos);
    REQUIRE(iO2 != Cantera::npos);
    REQUIRE(iH2O != Cantera::npos);
    REQUIRE(iN2 != Cantera::npos);

    std::vector<double> y0(gas->nSpecies(), 0.0);
    y0[iH2] = 0.028;
    y0[iO2] = 0.222;
    y0[iN2] = 0.75;
    gas->setMassFractions_NoNorm(y0.data());
    gas->setState_TP(1200.0, Cantera::OneAtm);

    std::vector<double> yTarget(gas->nSpecies(), 0.0);
    yTarget[iH2] = 0.010;
    yTarget[iO2] = 0.120;
    yTarget[iH2O] = 0.150;
    yTarget[iN2] = 0.720;
    double yTargetSum = 0.0;
    for (double y : yTarget)
        yTargetSum += y;
    REQUIRE(yTargetSum == doctest::Approx(1.0).epsilon(1e-14));

    const double tau = 1.0e-4;
    std::vector<double> constantTerm(gas->nSpecies(), 0.0);
    for (size_t k = 0; k < constantTerm.size(); ++k)
        constantTerm[k] = yTarget[k] / tau;

    double T = 1200.0;
    double rho = 1.0;
    std::vector<double> yEnd = y0;
    chem.advanceAffineConstVolume(T, rho, SpeciesBufferView{yEnd.data(), static_cast<int>(yEnd.size())},
                                  0.0, tau,
                                  ConstSpeciesBufferView{constantTerm.data(), static_cast<int>(constantTerm.size())},
                                  8.0 * tau);

    double maxErr = 0.0;
    double sumY = 0.0;
    for (size_t k = 0; k < yTarget.size(); ++k)
    {
        maxErr = std::max(maxErr, std::abs(yEnd[k] - yTarget[k]));
        sumY += yEnd[k];
    }

    printf("[affine-reactor] T=%.3f max|Y-target|=%.3e sumY=%.15f Y_H2=%.6f Y_H2O=%.6f\n",
           T, maxErr, sumY, yEnd[iH2], yEnd[iH2O]);

    CHECK(maxErr < 4.0e-4);
    CHECK(sumY == doctest::Approx(1.0).epsilon(1e-10));
    CHECK(yEnd[iH2] < y0[iH2]);
    CHECK(yEnd[iH2O] > y0[iH2O]);
}

TEST_CASE("Cantera custom affine reactor reduces target residual")
{
    auto sol = Cantera::newSolution(mechFile(), "", "");
    auto gas = sol->thermo();
    auto kin = sol->kinetics();
    ChemicalSource chem(mechFile());
    auto iH2 = gas->speciesIndex("H2");
    auto iO2 = gas->speciesIndex("O2");
    auto iH2O = gas->speciesIndex("H2O");
    auto iN2 = gas->speciesIndex("N2");
    REQUIRE(iH2 != Cantera::npos);
    REQUIRE(iO2 != Cantera::npos);
    REQUIRE(iH2O != Cantera::npos);
    REQUIRE(iN2 != Cantera::npos);

    auto normalize = [](std::vector<double> &Y)
    {
        double sum = 0.0;
        for (double &y : Y)
        {
            y = std::max(y, 0.0);
            sum += y;
        }
        REQUIRE(sum > 0);
        for (double &y : Y)
            y /= sum;
    };

    std::vector<double> yTarget(gas->nSpecies(), 0.0);
    yTarget[iH2] = 0.022;
    yTarget[iO2] = 0.180;
    yTarget[iH2O] = 0.030;
    yTarget[iN2] = 0.768;
    normalize(yTarget);

    std::vector<double> yStart = yTarget;
    yStart[iH2] += 0.006;
    yStart[iO2] += 0.020;
    yStart[iH2O] -= 0.025;
    yStart[iN2] -= 0.001;
    normalize(yStart);

    std::vector<double> molecularWeights(gas->nSpecies());
    gas->getMolecularWeights(molecularWeights.data());

    auto chemistryYDot = [&](const std::vector<double> &Y, double T, double rho)
    {
        gas->setMassFractions_NoNorm(Y.data());
        gas->setState_TD(T, rho);
        std::vector<double> omega(gas->nSpecies(), 0.0);
        kin->getNetProductionRates(omega.data());
        std::vector<double> ydot(gas->nSpecies(), 0.0);
        for (size_t k = 0; k < ydot.size(); ++k)
            ydot[k] = omega[k] * molecularWeights[k] / rho;
        return ydot;
    };

    auto affineResidualNorm = [&](const std::vector<double> &Y, double T, double rho,
                                  double dt, const std::vector<double> &C)
    {
        auto ydot = chemistryYDot(Y, T, rho);
        double norm = 0.0;
        for (size_t k = 0; k < Y.size(); ++k)
            norm = std::max(norm, std::abs(ydot[k] - Y[k] / dt + C[k]));
        return norm;
    };

    const double T0 = 1200.0;
    const double rho0 = 1.0;
    const double dt = 2.0e-5;
    auto ydotTarget = chemistryYDot(yTarget, T0, rho0);
    std::vector<double> C(gas->nSpecies(), 0.0);
    for (size_t k = 0; k < C.size(); ++k)
        C[k] = yTarget[k] / dt - ydotTarget[k];

    double residual0 = affineResidualNorm(yStart, T0, rho0, dt, C);
    double TEnd = T0;
    std::vector<double> yEnd = yStart;
    chem.advanceAffineConstVolume(TEnd, rho0, SpeciesBufferView{yEnd.data(), static_cast<int>(yEnd.size())},
                                  1.0, dt,
                                  ConstSpeciesBufferView{C.data(), static_cast<int>(C.size())},
                                  30.0 * dt);

    double residualEnd = affineResidualNorm(yEnd, TEnd, rho0, dt, C);
    double sumYEnd = 0.0;
    for (double y : yEnd)
        sumYEnd += y;

    printf("[affine-residual] res0=%.3e resEnd=%.3e ratio=%.3e T=%.3f sumY=%.15f Y_H2=%.6f Y_H2O=%.6f\n",
           residual0, residualEnd, residualEnd / residual0, TEnd, sumYEnd, yEnd[iH2], yEnd[iH2O]);

    CHECK(residual0 > 1.0e2);
    CHECK(residualEnd / residual0 < 1.0e-5);
    CHECK(sumYEnd == doctest::Approx(1.0).epsilon(1e-10));
    CHECK(yEnd[iH2] > 0.0);
    CHECK(yEnd[iH2O] > yStart[iH2O]);
}

TEST_CASE("0D const-vol — implicit Euler, species-only Newton, T via PhysicsProperties")
{
    auto fx = makeTestFixture();
    auto &phys = *fx.phys;
    auto &chem = (*fx.pool)[0];
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    int nVars = 5 + Ns1; // 14
    REQUIRE(Ns == 10);
    auto MW = chem.molecularWeights();
    int Isp = 5;

    double rho0 = 1.0, rhoE0 = 8.25; // gives T≈1200K via PhysicsProperties
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
        double s = 0;
        for (int k = 0; k < Ns; k++)
            s += Y[k];
        if (s > 0)
            for (int k = 0; k < Ns; k++)
                Y[k] /= s;
    };

    auto getT = [&](const Eigen::VectorXd &Uk)
    {
        // Map into PhysicsProperties-compatible TU (dynamic-sized for NS_EX)
        Eigen::Map<const Eigen::VectorXd> ukMap(Uk.data(), Uk.size());
        return phys.template temperature<3>(ukMap);
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
        double Rmix = chem.mixtureR(Yv);
        double p = U[0] * Rmix * T;

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
            double pk = Uk[0] * chem.mixtureR(Ykv) * Tk;

            std::vector<double> omega(Ns);
            SpeciesBufferView omegav{omega.data(), Ns};
            chem.productionRates(Tk, pk, Ykv, omegav);

            ret.setZero();
            for (int k = 0; k < Ns1; k++)
                ret[Isp + k] = omega[k] * MW[k];

            std::vector<double> jbuf(Ns * nVars, 0.0);
            JacobianBufferView Jv{jbuf.data(), Ns, nVars, Ns};
            chem.productionRatesAndJacobian(Tk, pk, Uk[0], Uk[4], 0., 0., 0., 4, U0, 1.0, Ykv, omegav, Jv);

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

    CHECK(Y_H2_end < 0.025);
    CHECK(Y_H2O_end > 0.001);
}

TEST_CASE("0D const-vol — react_test history tracks Cantera reactor")
{
    std::ifstream fin;
    for (const char *path : {"cases/eulerEX/react_test.json", "../cases/eulerEX/react_test.json",
                             "../../cases/eulerEX/react_test.json", "../../../cases/eulerEX/react_test.json"})
    {
        fin.open(path);
        if (fin.good())
            break;
        fin.clear();
    }
    REQUIRE(fin.good());
    auto cfg = json::parse(fin, nullptr, true, true);

    auto fx = makeTestFixture();
    auto &phys = *fx.phys;
    auto &chem = (*fx.pool)[0];
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    REQUIRE(Ns == 10);

    double dtCode = cfg.at("timeMarchControl").at("dtImplicit").get<double>();
    int nSteps = cfg.at("timeMarchControl").at("nTimeStep").get<int>();
    int outputEvery = std::max(1, cfg.at("outputControl").value("nDataOut", 10));
    auto state = cfg.at("eulerSettings").at("farFieldStaticValue").get<std::vector<double>>();
    const auto &ig = cfg.at("eulerSettings").at("idealGasProperty");
    double U0 = ig.value("U0", 379.0);
    double rho0 = ig.value("rho0", 1.0);
    double L0 = ig.value("L0", 1.0);

    Eigen::VectorXd U(static_cast<int>(state.size()));
    for (int i = 0; i < U.size(); ++i)
        U[i] = state[static_cast<size_t>(i)];
    REQUIRE(U.size() == 5 + Ns1);

    Reactive0D::ConstVolCase runCase;
    runCase.U = U;
    runCase.dtCode = dtCode;
    runCase.nSteps = nSteps;
    runCase.outputEvery = outputEvery;
    runCase.U0 = U0;
    runCase.rho0 = rho0;
    runCase.L0 = L0;

    auto dnds = Reactive0D::runDNDSRTrajectory<NS_EX, 3>(runCase, phys, chem);
    auto ct = Reactive0D::runCanteraTrajectory(runCase, mechFile(), dnds.front(), dnds);

    double maxRelT = 0, maxRelP = 0, maxAbsY = 0;
    double maxSettledRelT = 0, maxSettledRelP = 0, maxSettledAbsY = 0;
    for (size_t i = 0; i < dnds.size(); ++i)
    {
        const auto &ref = dnds[i];
        const auto &ctRef = ct[i];
        maxRelT = std::max(maxRelT, std::abs(ref.T - ctRef.T) / std::max(std::abs(ctRef.T), 1.0));
        maxRelP = std::max(maxRelP, std::abs(ref.p - ctRef.p) / std::max(std::abs(ctRef.p), 1.0));
        for (int k = 0; k < Ns; ++k)
            maxAbsY = std::max(maxAbsY, std::abs(ref.Y[k] - ctRef.Y[k]));
        if (ref.tPhys > 3.0e-5)
        {
            maxSettledRelT = std::max(maxSettledRelT, std::abs(ref.T - ctRef.T) / std::max(std::abs(ctRef.T), 1.0));
            maxSettledRelP = std::max(maxSettledRelP, std::abs(ref.p - ctRef.p) / std::max(std::abs(ctRef.p), 1.0));
            for (int k = 0; k < Ns; ++k)
                maxSettledAbsY = std::max(maxSettledAbsY, std::abs(ref.Y[k] - ctRef.Y[k]));
        }
    }

    const auto &last = dnds.back();
    const auto &lastCt = ct.back();
    double threshold = dnds.front().T + 0.5 * (lastCt.T - dnds.front().T);
    double tauDNDS = Reactive0D::ignitionTime(dnds, threshold);
    double tauCT = Reactive0D::ignitionTime(ct, threshold);
    double finalRelT = std::abs(last.T - lastCt.T) / lastCt.T;
    double finalRelP = std::abs(last.p - lastCt.p) / lastCt.p;
    double finalAbsY = 0;
    for (int k = 0; k < Ns; ++k)
        finalAbsY = std::max(finalAbsY, std::abs(last.Y[k] - lastCt.Y[k]));

    printf("[react_test-vs-cantera] samples=%zu tPhys=%.6e T[dnds,ct]=[%.6f,%.6f] p[dnds,ct]=[%.6f,%.6f]\n"
           "    rawMax[relT,relP,absY]=[%.3e,%.3e,%.3e] settledMax=[%.3e,%.3e,%.3e] final=[%.3e,%.3e,%.3e] tau[dnds,ct]=[%.6e,%.6e]\n",
           dnds.size(), last.tPhys, last.T, lastCt.T, last.p, lastCt.p,
           maxRelT, maxRelP, maxAbsY, maxSettledRelT, maxSettledRelP, maxSettledAbsY,
           finalRelT, finalRelP, finalAbsY, tauDNDS, tauCT);

    CHECK(std::abs(tauDNDS - tauCT) < 3.0e-6);
    CHECK(maxSettledRelT < 5e-3);
    CHECK(maxSettledRelP < 5e-3);
    CHECK(maxSettledAbsY < 1.5e-3);
    CHECK(finalRelT < 5e-3);
    CHECK(finalRelP < 5e-3);
    CHECK(finalAbsY < 1.5e-3);
}

TEST_CASE("0D const-vol helper uses physical Cantera scales")
{
    auto pool = std::make_shared<std::vector<ChemicalSource>>();
    pool->emplace_back(mechFile());
    auto &chem = (*pool)[0];
    int Ns = chem.nSpecies();
    int Ns1 = Ns - 1;
    REQUIRE(Ns == 10);

    typename EulerEvaluatorSettings<NS_EX>::IdealGasProperty igProp;
    igProp.gamma = 1.4;
    igProp.Rgas = 287.0;
    igProp.U0 = 379.0;
    igProp.rho0 = 0.7;
    igProp.T0 = 300.0;
    igProp.L0 = 2.0;
    igProp.muGas = 1e-200;
    igProp.prGas = 0.72;

    PhysicsProperties<NS_EX> phys(igProp);
    phys.setChemicalSourcePool(pool);

    using TU = typename PhysicsProperties<NS_EX>::TU;
    TU primTP(5 + Ns1);
    primTP.setZero();
    primTP(0) = 1200.0 / igProp.T0;
    primTP(4) = 101325.0 / (igProp.rho0 * igProp.U0 * igProp.U0);
    primTP(5 + 0) = 0.028;
    primTP(5 + 3) = 0.222;

    TU U(5 + Ns1);
    phys.template primTPToConservative<3>(primTP, U);

    Reactive0D::ConstVolCase runCase;
    runCase.U = U;
    runCase.dtCode = 1.0e-4;
    runCase.nSteps = 5;
    runCase.outputEvery = runCase.nSteps;
    runCase.U0 = igProp.U0;
    runCase.rho0 = igProp.rho0;
    runCase.L0 = igProp.L0;

    auto dnds = Reactive0D::runDNDSRTrajectory<NS_EX, 3>(runCase, phys, chem);
    auto ct = Reactive0D::runCanteraTrajectory(runCase, mechFile(), dnds.front(), dnds);

    REQUIRE(dnds.size() == 2);
    CHECK(std::abs(dnds.front().T - 1200.0) / 1200.0 < 1e-7);
    CHECK(std::abs(dnds.front().p - 101325.0) / 101325.0 < 1e-7);
    CHECK(dnds.back().tPhys == doctest::Approx(runCase.nSteps * runCase.dtCode * igProp.L0 / igProp.U0).epsilon(1e-12));
    CHECK(std::abs(ct.front().T - 1200.0) / 1200.0 < 1e-7);
    CHECK(std::abs(ct.front().p - 101325.0) / 101325.0 < 1e-7);
    CHECK(dnds.back().Y[5] > dnds.front().Y[5]);
    CHECK(std::abs(dnds.back().T - ct.back().T) / std::max(std::abs(ct.back().T), 1.0) < 5e-3);
    CHECK(std::abs(dnds.back().p - ct.back().p) / std::max(std::abs(ct.back().p), 1.0) < 5e-3);
    CHECK(std::abs(dnds.back().Y[5] - ct.back().Y[5]) < 1e-3);
}

TEST_CASE("Finite-difference Jacobian check at non-initial state")
{
    auto fx = makeTestFixture();
    auto &phys = *fx.phys;
    ChemicalSource *chem = &(*fx.pool)[0];
    int Ns = chem->nSpecies();
    int Ns1 = Ns - 1;
    int nVars = 5 + Ns1;
    REQUIRE(Ns == 10);
    auto MW = chem->molecularWeights();
    int Isp = 5;

    double rho0 = 1.0, rhoE0 = 8.25, U0 = 379.0; // gives T≈1129K with reference-offset bridge

    Eigen::VectorXd U = Eigen::VectorXd::Zero(nVars);
    U[0] = rho0;
    U[4] = rhoE0;
    U[Isp + 0] = rho0 * 0.028;
    U[Isp + 3] = rho0 * 0.222;

    auto getY = [&](const Eigen::VectorXd &Uk, std::vector<double> &Y)
    {
        Y.resize(Ns);
        auto Yv = chem->massFractions(Uk[0], Uk.data() + Isp, Ns1);
        for (int k = 0; k < Ns; ++k)
            Y[k] = Yv[k];
    };

    auto getT = [&](const Eigen::VectorXd &Uk)
    {
        Eigen::Map<const Eigen::VectorXd> ukMap(Uk.data(), Uk.size());
        return phys.template temperature<3>(ukMap);
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
            chem->productionRatesAndJacobian(Tk, pk, Uk[0], Uk[4], 0., 0., 0., 4, U0, 1.0, Ykv,
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

    // Y-from-U through the same ChemicalSource API used by production paths.
    auto getY_linear = [&](const Eigen::VectorXd &Uk, std::vector<double> &Y)
    {
        Y.resize(Ns);
        auto Yv = chem->massFractions(Uk[0], Uk.data() + Isp, Ns1);
        for (int k = 0; k < Ns; ++k)
            Y[k] = Yv[k];
    };

    // FD derivative source: ω from perturbed state (no Y clamping, KE subtracted)
    auto sourceAtU = [&](const Eigen::VectorXd &Up)
    {
        std::vector<double> Yp;
        getY_linear(Up, Yp);
        ConstSpeciesBufferView Ypv{Yp.data(), Ns};
        double Tp = getT(Up);
        double pp = Up[0] * chem->mixtureR(Ypv) * Tp;
        std::vector<double> om(Ns);
        chem->productionRates(Tp, pp, Ypv, SpeciesBufferView{om.data(), Ns});
        return om;
    };

    // ── Jacobian comparison lambda ──
    const double atol = 1e-7, rtol = 1e-3;
    auto compareJacobian = [&](const Eigen::VectorXd &Us, double Ts, const char *label) -> int
    {
        std::vector<double> Yref;
        getY(Us, Yref);
        ConstSpeciesBufferView Yv{Yref.data(), Ns};
        double p = Us[0] * chem->mixtureR(Yv) * Ts;

        std::vector<double> jbufRef(Ns * nVars, 0.0), omegaRef(Ns);
        chem->productionRatesAndJacobian(Ts, p, Us[0], Us[4], 0., 0., 0., 4, U0, 1.0, Yv,
                                         SpeciesBufferView{omegaRef.data(), Ns},
                                         JacobianBufferView{jbufRef.data(), Ns, nVars, Ns});

        const double epsFluid = 1e-6, epsSpecies = 1e-4;
        int nBad = 0;

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
                   Eigen::Unaligned, Eigen::OuterStride<>>
            anJac(jbufRef.data(), Ns, nVars, Eigen::OuterStride<>(Ns));
        Eigen::MatrixXd fdJac(Ns, nVars);

        for (int j = 0; j < nVars; j++)
        {
            double epsCol = (j >= Isp) ? epsSpecies : epsFluid;
            Eigen::VectorXd Up = Us;
            double h = epsCol * std::max(std::abs(Up[j]), 1.0);
            Up[j] += h;
            auto omP = sourceAtU(Up);
            Up[j] = Us[j] - h;
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
                    printf("[FD-bad %s] J(%d,%d) fd=%.4e an=%.4e (rel %.1f%%)\n",
                           label, i, j, fd, an, relErr * 100);
                    nBad++;
                }
            }
        }

        double normFD = fdJac.norm();
        double maxRelGlobal = 0;
        for (int i = 0; i < Ns; i++)
            for (int j = 0; j < nVars; j++)
            {
                double rel = std::abs(fdJac(i, j) - anJac(i, j)) / normFD * 100.0;
                if (rel > maxRelGlobal)
                    maxRelGlobal = rel;
            }

        printf("[FD %s] %d mismatches, ||FD||=%.2e, max|J_fd-J_an|/||FD||=%.2e%%, T=%.1fK\n",
               label, nBad, normFD, maxRelGlobal, Ts);

        // Verbose matrix print for the final checkpoint
        if (label[0] == 's' && label[1] == '2' && label[2] == '0' && label[3] == '0')
        {
            Eigen::IOFormat fmtJac(3, Eigen::DontAlignCols, " ", "\n", "    [", "]", "", "");
            printf("\n[JAC @%s] Ns=%d nVars=%d  ||FD||_F = %.4e\n", label, Ns, nVars, normFD);
            std::cout << "[JAC-analytical]" << std::endl
                      << anJac.format(fmtJac) << std::endl;
            std::cout << "[JAC-FD]" << std::endl
                      << fdJac.format(fmtJac) << std::endl;
            printf("[JAC-relErr %% of ||FD||_F=%.2e]:\n", normFD);
            printf("       ");
            for (int j = 0; j < nVars; j++)
                printf(" %7d", j);
            printf("\n");
            for (int i = 0; i < Ns; i++)
            {
                printf("  sp%2d ", i);
                for (int j = 0; j < nVars; j++)
                    printf(" %7.1e", std::abs(fdJac(i, j) - anJac(i, j)) / normFD * 100.0);
                printf("\n");
            }
        }
        return nBad;
    };

    // ── Checkpoints ──
    compareJacobian(U, T, "s20");

    for (int step = 20; step < 50; step++)
    {
        Eigen::VectorXd Uk = U;
        double Tk = T;
        for (int iter = 0; iter < 50; iter++)
        {
            std::vector<double> Yk;
            getY(Uk, Yk);
            ConstSpeciesBufferView Ykv{Yk.data(), Ns};
            double Rmix = chem->mixtureR(Ykv), pk = Uk[0] * Rmix * Tk;
            std::vector<double> omega(Ns);
            chem->productionRates(Tk, pk, Ykv, SpeciesBufferView{omega.data(), Ns});
            Eigen::VectorXd ret = Eigen::VectorXd::Zero(nVars);
            for (int k = 0; k < Ns1; k++)
                ret[Isp + k] = omega[k] * MW[k];
            std::vector<double> jbuf(Ns * nVars, 0.0);
            chem->productionRatesAndJacobian(Tk, pk, Uk[0], Uk[4], 0., 0., 0., 4, U0, 1.0, Ykv,
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
    compareJacobian(U, T, "s50");

    for (int step = 50; step < 200; step++)
    {
        Eigen::VectorXd Uk = U;
        double Tk = T;
        for (int iter = 0; iter < 50; iter++)
        {
            std::vector<double> Yk;
            getY(Uk, Yk);
            ConstSpeciesBufferView Ykv{Yk.data(), Ns};
            double Rmix = chem->mixtureR(Ykv), pk = Uk[0] * Rmix * Tk;
            std::vector<double> omega(Ns);
            chem->productionRates(Tk, pk, Ykv, SpeciesBufferView{omega.data(), Ns});
            Eigen::VectorXd ret = Eigen::VectorXd::Zero(nVars);
            for (int k = 0; k < Ns1; k++)
                ret[Isp + k] = omega[k] * MW[k];
            std::vector<double> jbuf(Ns * nVars, 0.0);
            chem->productionRatesAndJacobian(Tk, pk, Uk[0], Uk[4], 0., 0., 0., 4, U0, 1.0, Ykv,
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
    int nBadS200 = compareJacobian(U, T, "s200");

    // ── Fluid-column check (inject non-zero velocity) ──
    {
        Eigen::VectorXd Umom = U;
        Umom[1] = 0.1;
        Umom[2] = 0.05;
        std::vector<double> Ymom;
        getY_linear(Umom, Ymom);
        ConstSpeciesBufferView Ymv{Ymom.data(), Ns};
        double Tmom = getT(Umom);
        double Rm = chem->mixtureR(Ymv), pmom = Umom[0] * Rm * Tmom;

        std::vector<double> jbufM(Ns * nVars, 0.0), omegM(Ns);
        chem->productionRatesAndJacobian(Tmom, pmom, Umom[0], Umom[4],
                                         Umom[1], Umom[2], 0., 4, U0, 1.0, Ymv,
                                         SpeciesBufferView{omegM.data(), Ns},
                                         JacobianBufferView{jbufM.data(), Ns, nVars, Ns});

        for (int jj = 0; jj <= 2; jj++)
        {
            double h = 1e-6 * std::max(std::abs(Umom[jj]), 1.0);
            Eigen::VectorXd Up = Umom, Um = Umom;
            Up[jj] += h;
            Um[jj] -= h;
            auto omP = sourceAtU(Up), omM = sourceAtU(Um);
            int nFluidBad = 0;
            for (int i = 0; i < Ns; i++)
            {
                double fd = (omP[i] - omM[i]) / (2.0 * h);
                double an = jbufM[i + jj * Ns];
                double denom = std::max(std::max(std::abs(fd), std::abs(an)), 1e-60);
                double rel = std::abs(fd - an) / denom;
                if (std::abs(fd - an) > 1e-7 && rel > 1e-3 && std::abs(an) > 1e-14)
                {
                    printf("[FD-fluid] J(%d,%d) fd=%.4e an=%.4e (rel %.1f%%)\n",
                           i, jj, fd, an, rel * 100);
                    nFluidBad++;
                }
            }
            printf("[FD-fluid] col %d: %d mismatches\n", jj, nFluidBad);
            CHECK(nFluidBad == 0);
        }
    }

    // FD-vs-analytical Jacobian quality across ignition stages (dt=1e-6, const-V):
    // the FD path intentionally calls PhysicsProperties::temperature(), so the
    // DNDSR-to-Cantera reference-energy bridge is tested in one place.  Fluid
    // columns remain diagnostic while production uses JAC_SKIP_FLUID.  The
    // global Frobenius-scaled error remains O(1e-2 %) at the final checkpoint.
    CHECK(nBadS200 <= 25);
}
