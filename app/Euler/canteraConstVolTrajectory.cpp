#include "Euler/Chemistry/ChemicalSource.hpp"
#include "Euler/Euler.hpp"
#include "Euler/Physics/PhysicsProperties.hpp"

#include "cantera/zerodim.h"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace DNDS::Euler;
using namespace DNDS::Euler::Chemistry;
using json = nlohmann::ordered_json;

namespace
{
    struct CaseData
    {
        std::string mechanismFile;
        Eigen::VectorXd U;
        double dtCode = 0;
        int nSteps = 0;
        int outputEvery = 1;
        double gamma = 1.4;
        double Rgas = 287.0;
        double U0 = 1.0;
        double rho0 = 1.0;
        double T0 = 1.0;
        double L0 = 1.0;
    };

    struct StateSample
    {
        double tCode = 0;
        double tPhys = 0;
        double T = 0;
        double p = 0;
        std::vector<double> Y;
    };

    std::string resolveMechanism(const std::string &mechanismFile)
    {
        std::filesystem::path mech(mechanismFile);
        if (mech.is_absolute() || std::filesystem::exists(mech))
            return mech.string();
        if (const char *env = std::getenv("DNDS_MECH_PATH"))
        {
            auto candidate = std::filesystem::path(env) / mech;
            if (std::filesystem::exists(candidate))
                return candidate.string();
        }
        if (const char *env = std::getenv("CANTERA_DATA"))
        {
            auto candidate = std::filesystem::path(env) / mech;
            if (std::filesystem::exists(candidate))
                return candidate.string();
        }
#ifdef DNDS_CANTERA_DATA_DIR
        auto candidate = std::filesystem::path(DNDS_CANTERA_DATA_DIR) / mech;
        if (std::filesystem::exists(candidate))
            return candidate.string();
#endif
        return mechanismFile;
    }

    CaseData readCase(const std::string &caseFile)
    {
        std::ifstream fin(caseFile);
        if (!fin)
            throw std::runtime_error("failed to open case file: " + caseFile);
        auto cfg = json::parse(fin, nullptr, true, true);

        CaseData c;
        c.dtCode = cfg.at("timeMarchControl").at("dtImplicit").get<double>();
        c.nSteps = cfg.at("timeMarchControl").at("nTimeStep").get<int>();
        c.outputEvery = std::max(1, cfg.at("outputControl").value("nDataOut", 1));

        const auto &euler = cfg.at("eulerSettings");
        c.mechanismFile = euler.at("reactiveFlow").at("mechanismFile").get<std::string>();
        auto state = euler.at("farFieldStaticValue").get<std::vector<double>>();
        c.U.resize(static_cast<int>(state.size()));
        for (int i = 0; i < c.U.size(); ++i)
            c.U[i] = state[static_cast<size_t>(i)];

        const auto &ig = euler.at("idealGasProperty");
        c.gamma = ig.value("gamma", c.gamma);
        c.Rgas = ig.value("Rgas", c.Rgas);
        c.U0 = ig.value("U0", c.U0);
        c.rho0 = ig.value("rho0", c.rho0);
        c.T0 = ig.value("T0", c.T0);
        c.L0 = ig.value("L0", c.L0);
        return c;
    }

    PhysicsProperties<NS_EX> makePhysics(const CaseData &c, std::shared_ptr<std::vector<ChemicalSource>> pool)
    {
        typename EulerEvaluatorSettings<NS_EX>::IdealGasProperty ig;
        ig.gamma = c.gamma;
        ig.Rgas = c.Rgas;
        ig.U0 = c.U0;
        ig.rho0 = c.rho0;
        ig.T0 = c.T0;
        ig.L0 = c.L0;
        ig.recomputeDerived();
        PhysicsProperties<NS_EX> phys(ig);
        phys.setChemicalSourcePool(std::move(pool));
        return phys;
    }

    std::vector<double> massFractions(const ChemicalSource &chem, const Eigen::VectorXd &U)
    {
        int Ns = chem.nSpecies();
        int Ns1 = Ns - 1;
        int Isp = static_cast<int>(U.size()) - Ns1;
        auto Yv = chem.massFractions(U[0], U.data() + Isp, Ns1);
        std::vector<double> Y(Ns);
        for (int k = 0; k < Ns; ++k)
            Y[k] = Yv[k];
        return Y;
    }

    StateSample sampleDNDSR(const CaseData &c, PhysicsProperties<NS_EX> &phys, ChemicalSource &chem,
                            const Eigen::VectorXd &U, int step)
    {
        auto Y = massFractions(chem, U);
        double T = phys.template temperature<3>(U);
        double p = U[0] * c.rho0 * chem.mixtureR(ConstSpeciesBufferView{Y.data(), static_cast<int>(Y.size())}) * T;
        return {step * c.dtCode, step * c.dtCode * c.L0 / c.U0, T, p, std::move(Y)};
    }

    void stepDNDSRImplicit(const CaseData &c, PhysicsProperties<NS_EX> &phys, ChemicalSource &chem, Eigen::VectorXd &U)
    {
        int Ns = chem.nSpecies();
        int Ns1 = Ns - 1;
        int nVars = static_cast<int>(U.size());
        int Isp = nVars - Ns1;
        int I4 = Isp - 1;
        auto MW = chem.molecularWeights();
        double invS0 = c.L0 / (c.rho0 * c.U0);

        Eigen::VectorXd Uk = U;
        for (int iter = 0; iter < 80; ++iter)
        {
            auto Y = massFractions(chem, Uk);
            ConstSpeciesBufferView Yv{Y.data(), Ns};
            double T = phys.template temperature<3>(Uk);
            double p = Uk[0] * c.rho0 * chem.mixtureR(Yv) * T;

            std::vector<double> omega(Ns);
            chem.productionRates(std::max(T, 200.0), p, Yv, SpeciesBufferView{omega.data(), Ns});

            Eigen::VectorXd ret = Eigen::VectorXd::Zero(nVars);
            for (int k = 0; k < Ns1; ++k)
                ret[Isp + k] = omega[k] * MW[k] * invS0;

            std::vector<double> jbuf(Ns * nVars, 0.0);
            chem.productionRatesAndJacobian(std::max(T, 200.0), p, Uk[0], Uk[I4], 0.0, 0.0, 0.0, I4,
                                            c.U0, c.rho0, Yv, SpeciesBufferView{omega.data(), Ns},
                                            JacobianBufferView{jbuf.data(), Ns, nVars, Ns});

            Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(nVars, nVars);
            for (int k = 0; k < Ns1; ++k)
                for (int j = 0; j < nVars; ++j)
                    jac(Isp + k, j) = MW[k] * jbuf[k + j * Ns] * invS0;

            Eigen::VectorXd F = Uk - U - c.dtCode * ret;
            Eigen::MatrixXd Jn = Eigen::MatrixXd::Identity(nVars, nVars) - c.dtCode * jac;
            for (int r = 0; r <= I4; ++r)
            {
                Jn.row(r) = Eigen::VectorXd::Unit(nVars, r);
                F[r] = 0;
            }

            Eigen::VectorXd dU = Jn.partialPivLu().solve(-F);
            Uk += dU;
            for (int k = Isp; k < nVars; ++k)
                Uk[k] = std::max(Uk[k], 1e-30);
            if (dU.lpNorm<Eigen::Infinity>() < 1e-12)
                break;
        }
        U = Uk;
    }

    std::vector<StateSample> runDNDSRTrajectory(const CaseData &c, PhysicsProperties<NS_EX> &phys,
                                                ChemicalSource &chem)
    {
        std::vector<StateSample> out;
        Eigen::VectorXd U = c.U;
        out.push_back(sampleDNDSR(c, phys, chem, U, 0));
        for (int step = 1; step <= c.nSteps; ++step)
        {
            stepDNDSRImplicit(c, phys, chem, U);
            if (step % c.outputEvery == 0 || step == c.nSteps)
                out.push_back(sampleDNDSR(c, phys, chem, U, step));
        }
        return out;
    }

    std::vector<StateSample> runCanteraTrajectory(const CaseData &c, const std::string &mechPath,
                                                  const StateSample &initial, const std::vector<StateSample> &times)
    {
        auto sol = Cantera::newSolution(mechPath, "", "none");
        auto gas = sol->thermo();
        gas->setMassFractions_NoNorm(initial.Y.data());
        gas->setState_TD(initial.T, c.rho0 * c.U[0]);
        auto reactor = Cantera::newReactorBase("IdealGasReactor", sol);
        Cantera::ReactorNet net(reactor);

        std::vector<StateSample> out;
        out.reserve(times.size());
        for (const auto &target : times)
        {
            if (target.tPhys > 0)
                net.advance(target.tPhys);
            auto &th = *reactor->phase()->thermo();
            std::vector<double> Y(th.nSpecies());
            th.getMassFractions(Y.data());
            out.push_back({target.tCode, target.tPhys, th.temperature(), th.pressure(), std::move(Y)});
        }
        return out;
    }

    void writeCsv(const std::string &path, const std::vector<std::string> &species,
                  const std::vector<StateSample> &dnds, const std::vector<StateSample> &ct)
    {
        std::ofstream fout(path);
        fout << "t_code,t_phys,T_dnds,T_cantera,p_dnds,p_cantera";
        for (auto &s : species)
            fout << ",Y_" << s << "_dnds,Y_" << s << "_cantera";
        fout << "\n";
        fout << std::setprecision(16);
        for (size_t i = 0; i < dnds.size(); ++i)
        {
            fout << dnds[i].tCode << ',' << dnds[i].tPhys << ',' << dnds[i].T << ',' << ct[i].T << ','
                 << dnds[i].p << ',' << ct[i].p;
            for (size_t k = 0; k < species.size(); ++k)
                fout << ',' << dnds[i].Y[k] << ',' << ct[i].Y[k];
            fout << "\n";
        }
    }

    double ignitionTime(const std::vector<StateSample> &hist, double threshold)
    {
        for (size_t i = 1; i < hist.size(); ++i)
        {
            if (hist[i - 1].T <= threshold && hist[i].T >= threshold)
            {
                double a = (threshold - hist[i - 1].T) / std::max(hist[i].T - hist[i - 1].T, 1e-300);
                return hist[i - 1].tPhys + a * (hist[i].tPhys - hist[i - 1].tPhys);
            }
        }
        return hist.back().tPhys;
    }
}

int main(int argc, char **argv)
{
    std::string caseFile = argc > 1 ? argv[1] : "../cases/eulerEX/react_test.json";
    std::string outFile = argc > 2 ? argv[2] : "cantera_constvol_trajectory.csv";

    try
    {
        CaseData c = readCase(caseFile);
        std::string mechPath = resolveMechanism(c.mechanismFile);
        auto pool = std::make_shared<std::vector<ChemicalSource>>();
        pool->emplace_back(mechPath);
        auto phys = makePhysics(c, pool);
        auto &chem = (*pool)[0];

        auto dnds = runDNDSRTrajectory(c, phys, chem);
        auto cantera = runCanteraTrajectory(c, mechPath, dnds.front(), dnds);
        writeCsv(outFile, chem.speciesNames(), dnds, cantera);

        double maxRelT = 0, maxRelP = 0, maxAbsY = 0;
        double maxSettledRelT = 0, maxSettledRelP = 0, maxSettledAbsY = 0;
        for (size_t i = 0; i < dnds.size(); ++i)
        {
            maxRelT = std::max(maxRelT, std::abs(dnds[i].T - cantera[i].T) / std::max(std::abs(cantera[i].T), 1.0));
            maxRelP = std::max(maxRelP, std::abs(dnds[i].p - cantera[i].p) / std::max(std::abs(cantera[i].p), 1.0));
            for (size_t k = 0; k < dnds[i].Y.size(); ++k)
                maxAbsY = std::max(maxAbsY, std::abs(dnds[i].Y[k] - cantera[i].Y[k]));
            if (dnds[i].tPhys > 3.0e-5)
            {
                maxSettledRelT = std::max(maxSettledRelT,
                                          std::abs(dnds[i].T - cantera[i].T) / std::max(std::abs(cantera[i].T), 1.0));
                maxSettledRelP = std::max(maxSettledRelP,
                                          std::abs(dnds[i].p - cantera[i].p) / std::max(std::abs(cantera[i].p), 1.0));
                for (size_t k = 0; k < dnds[i].Y.size(); ++k)
                    maxSettledAbsY = std::max(maxSettledAbsY, std::abs(dnds[i].Y[k] - cantera[i].Y[k]));
            }
        }
        const auto &a = dnds.back();
        const auto &b = cantera.back();
        double threshold = dnds.front().T + 0.5 * (b.T - dnds.front().T);
        double tauDNDS = ignitionTime(dnds, threshold);
        double tauCantera = ignitionTime(cantera, threshold);
        double finalRelT = std::abs(a.T - b.T) / std::max(std::abs(b.T), 1.0);
        double finalRelP = std::abs(a.p - b.p) / std::max(std::abs(b.p), 1.0);
        double finalAbsY = 0;
        for (size_t k = 0; k < a.Y.size(); ++k)
            finalAbsY = std::max(finalAbsY, std::abs(a.Y[k] - b.Y[k]));
        std::cout << std::scientific << std::setprecision(9);
        std::cout << "wrote " << dnds.size() << " samples to " << outFile << '\n';
        std::cout << "final t_phys=" << a.tPhys << " T[dnds,cantera]=[" << a.T << ',' << b.T
                  << "] p=[" << a.p << ',' << b.p << "]\n";
        std::cout << "raw max rel T=" << maxRelT << " max rel p=" << maxRelP << " max abs Y=" << maxAbsY << '\n';
        std::cout << "settled max rel T=" << maxSettledRelT << " max rel p=" << maxSettledRelP
                  << " max abs Y=" << maxSettledAbsY << '\n';
        std::cout << "final rel T=" << finalRelT << " final rel p=" << finalRelP
                  << " final abs Y=" << finalAbsY << '\n';
        std::cout << "midpoint ignition t_phys[dnds,cantera]=[" << tauDNDS << ',' << tauCantera
                  << "] delta=" << std::abs(tauDNDS - tauCantera) << '\n';
        Cantera::appdelete();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        Cantera::appdelete();
        return 1;
    }
}
