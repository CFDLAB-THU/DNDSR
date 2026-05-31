#include "Euler/Chemistry/ChemicalSource.hpp"
#include "Euler/Euler.hpp"
#include "Euler/Physics/ConstVolTrajectory.hpp"
#include "Euler/Physics/PhysicsProperties.hpp"

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
    struct CaseData : Reactive0D::ConstVolCase
    {
        std::string mechanismFile;
        double gamma = 1.4;
        double Rgas = 287.0;
        double T0 = 1.0;
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
        PhysicsProperties<NS_EX> phys(ig);
        phys.setChemicalSourcePool(std::move(pool));
        return phys;
    }

    void writeCsv(const std::string &path, const std::vector<std::string> &species,
                  const std::vector<Reactive0D::StateSample> &dnds,
                  const std::vector<Reactive0D::StateSample> &ct)
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

        auto dnds = Reactive0D::runDNDSRTrajectory<NS_EX, 3>(c, phys, chem);
        auto cantera = Reactive0D::runCanteraTrajectory(c, mechPath, dnds.front(), dnds);
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
        double tauDNDS = Reactive0D::ignitionTime(dnds, threshold);
        double tauCantera = Reactive0D::ignitionTime(cantera, threshold);
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
