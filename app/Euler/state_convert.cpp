#include "Euler/Euler.hpp"
#include "Euler/Gas.hpp"
#include "Euler/Physics/PhysicsProperties.hpp"
#include "Euler/Chemistry/ChemicalSource.hpp"

#include <argparse.hpp>
#include <nlohmann/json.hpp>
#include <fmt/format.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <memory>

using namespace DNDS::Euler;
using namespace DNDS::Euler::Chemistry;
using json = nlohmann::json;
using DNDS::real;

namespace
{

    struct Cfg
    {
        real gamma = 1.4;
        real Rgas = 287.0;
        real U0 = 1.0;
        real rho0 = 1.0;
        real T0 = 1.0;
    };

    EulerModel parseModel(const std::string &s)
    {
        if (s == "NS")
            return NS;
        if (s == "NS_3D")
            return NS_3D;
        if (s == "NS_EX")
            return NS_EX;
        if (s == "NS_EX_3D")
            return NS_EX_3D;
        if (s == "NS_SA")
            return NS_SA;
        if (s == "NS_SA_3D")
            return NS_SA_3D;
        if (s == "NS_2EQ")
            return NS_2EQ;
        if (s == "NS_2EQ_3D")
            return NS_2EQ_3D;
        throw std::runtime_error("Unknown model: " + s);
    }

    Cfg parseConfig(const std::string &s)
    {
        Cfg c;
        if (s.empty())
            return c;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            auto eq = item.find('=');
            if (eq == std::string::npos)
                continue;
            auto key = item.substr(0, eq);
            auto val = item.substr(eq + 1);
            if (key == "gamma")
                c.gamma = std::stod(val);
            else if (key == "Rgas")
                c.Rgas = std::stod(val);
            else if (key == "U0")
                c.U0 = std::stod(val);
            else if (key == "rho0")
                c.rho0 = std::stod(val);
            else if (key == "T0")
                c.T0 = std::stod(val);
        }
        return c;
    }

    Eigen::VectorXd parseState(const std::string &s)
    {
        auto v = json::parse(s).get<std::vector<double>>();
        Eigen::VectorXd U(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            U[i] = v[i];
        return U;
    }

    template <int dim>
    real computeTemperatureNonReactive(const Eigen::VectorXd &U, real gamma, real R_code)
    {
        real rho = U[0];
        real rhoInv = 1.0 / std::max(rho, 1e-60);
        real vel2 = 0;
        for (int j = 1; j <= dim; ++j)
            vel2 += U[j] * U[j];
        vel2 *= rhoInv * rhoInv;
        int I4 = dim + 1;
        real e_sensible = U[I4] * rhoInv - 0.5 * vel2;
        real p = (gamma - 1.0) * rho * e_sensible;
        return p / (rho * R_code);
    }

    real computeRgasCode(real RgasPhys, real U0, real T0)
    {
        return RgasPhys * T0 / (U0 * U0);
    }

    void printVec(const Eigen::VectorXd &v, const std::string &prefix, int nPerLine = 4)
    {
        std::cout << prefix;
        for (int i = 0; i < (int)v.size(); ++i)
        {
            if (i > 0 && i % nPerLine == 0)
                std::cout << fmt::format("\n{:{}}", "", prefix.size());
            std::cout << fmt::format("{:12.6f}", v[i]);
        }
        std::cout << "\n";
    }

} // anonymous namespace

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("state_convert", "1.0");
    program.add_argument("--model")
        .required()
        .help("Euler model: NS, NS_3D, NS_EX, NS_EX_3D, NS_SA, NS_SA_3D, NS_2EQ, NS_2EQ_3D");
    program.add_argument("--nVars")
        .scan<'i', int>()
        .default_value(0)
        .help("Number of conservative variables (for dynamic models)");
    program.add_argument("--from")
        .required()
        .help("Input format: cons-total, cons-sensible, prim, prim-RT, prim-TP");
    program.add_argument("--scaling")
        .default_value(std::string("code"))
        .help("Input scaling: code, phys");
    program.add_argument("--config")
        .default_value(std::string(""))
        .help("Key=value pairs: gamma=1.4,Rgas=287,U0=379,rho0=1,T0=1");
    program.add_argument("--mechanism")
        .default_value(std::string(""))
        .help("Mechanism YAML path (for reactive models)");
    program.add_argument("--state")
        .required()
        .help("State as JSON array: [1.0,0,0,0,6.0,0.028,...]");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n"
                  << program;
        return 1;
    }

    auto model = parseModel(program.get<std::string>("--model"));
    int nVarsArg = program.get<int>("--nVars");
    auto fromStr = program.get<std::string>("--from");
    auto scalingStr = program.get<std::string>("--scaling");
    auto configStr = program.get<std::string>("--config");
    auto mechStr = program.get<std::string>("--mechanism");
    auto stateStr = program.get<std::string>("--state");

    int dim = getDim_Fixed(model);
    int nVarsFixed = getnVarsFixed(model);
    int nVars = (nVarsFixed > 0) ? nVarsFixed : nVarsArg;
    if (nVars <= 0)
    {
        std::cerr << "Error: --nVars required for dynamic models\n";
        return 1;
    }

    auto cfg = parseConfig(configStr);
    real R_code = computeRgasCode(cfg.Rgas, cfg.U0, cfg.T0);
    bool isReactive = !mechStr.empty();
    bool inputPhys = (scalingStr == "phys");

    Eigen::VectorXd inputState = parseState(stateStr);
    if ((int)inputState.size() != nVars)
    {
        std::cerr << "Error: --state has " << inputState.size()
                  << " elements, expected " << nVars << "\n";
        return 1;
    }

    // --- Build PhysicsProperties for reactive case ---
    std::shared_ptr<std::vector<ChemicalSource>> pool;
    std::unique_ptr<PhysicsProperties<NS_EX>> phys;
    int nSpecies = 0;
    std::vector<std::string> speciesNames;

    if (isReactive)
    {
        if (model != NS_EX && model != NS_EX_3D)
        {
            std::cerr << "Error: --mechanism only valid for NS_EX / NS_EX_3D\n";
            return 1;
        }
        pool = std::make_shared<std::vector<ChemicalSource>>();
        pool->emplace_back(mechStr);
        nSpecies = (*pool)[0].nSpecies();
        speciesNames = (*pool)[0].speciesNames();

        typename EulerEvaluatorSettings<NS_EX>::IdealGasProperty igProp;
        igProp.gamma = cfg.gamma;
        igProp.Rgas = cfg.Rgas;
        igProp.U0 = cfg.U0;
        igProp.rho0 = cfg.rho0;
        igProp.T0 = cfg.T0;
        igProp.muGas = 1e-200;
        igProp.prGas = 0.72;

        phys = std::make_unique<PhysicsProperties<NS_EX>>(igProp);
        phys->setChemicalSourcePool(pool);
    }

    // --- Scaling conversion (phys → code) ---
    if (inputPhys)
    {
        auto U = inputState;
        if (fromStr == "prim")
        {
            U[0] /= cfg.rho0;
            for (int j = 1; j <= dim; ++j)
                U[j] /= cfg.U0;
            U[dim + 1] /= (cfg.rho0 * cfg.U0 * cfg.U0);
        }
        else
        {
            U[0] /= cfg.rho0;
            for (int j = 1; j <= dim; ++j)
                U[j] /= (cfg.rho0 * cfg.U0);
            U[dim + 1] /= (cfg.rho0 * cfg.U0 * cfg.U0);
        }
        inputState = U;
    }

    std::cout << fmt::format("=== Input: {}, {} units ===\n", fromStr,
                             inputPhys ? "physical" : "code");
    std::cout << fmt::format("  model={}  dim={}  nVars={}  gamma={:.6f}  Rgas={:.1f} J/(kg.K)  U0={:.1f}  rho0={:.1f}  T0={:.1f}\n",
                             program.get<std::string>("--model"), dim, nVars,
                             cfg.gamma, cfg.Rgas, cfg.U0, cfg.rho0, cfg.T0);
    std::cout << "\n";
    printVec(inputState, "  input = [", 4);

    // --- Dispatch by dim for gas conversion functions ---
    real rhoH_form = 0;
    Eigen::VectorXd consTotal(nVars);
    if (fromStr == "cons-total")
    {
        consTotal = inputState;
    }
    else if (fromStr == "cons-sensible")
    {
        rhoH_form = phys ? phys->mixtureFormationRhoE(inputState)
                         : real(0.0);
        consTotal = inputState;
        consTotal[dim + 1] += rhoH_form;
    }
    else // prim, prim-RT, prim-TP
    {
        auto fullPrim = inputState;

        // Compute mixture gas constant code (needed for prim-TP)
        auto computeRmixCode = [&](const Eigen::VectorXd &pvec)
        {
            if (!isReactive)
                return R_code;
            int Isp = dim + 2;
            int Ns1 = nSpecies - 1;
            double Ru = 8314.46261815324; // J/(kmol·K)
            double invR0 = cfg.T0 / (cfg.U0 * cfg.U0);
            double Rmix = 0;
            for (int k = 0; k < Ns1; ++k)
                Rmix += pvec[Isp + k] * (Ru / (*pool)[0].molecularWeights()[k]);
            real lastY = 1.0;
            for (int k = 0; k < Ns1; ++k)
                lastY -= pvec[Isp + k];
            Rmix += lastY * (Ru / (*pool)[0].molecularWeights()[Ns1]);
            return Rmix * invR0; // code-scaled
        };

        if (fromStr == "prim-TP")
        {
            real T_raw = fullPrim[0];
            real T_in = inputPhys ? T_raw / cfg.T0 : T_raw; // code T
            real p_code = fullPrim[dim + 1];
            real Rm = computeRmixCode(fullPrim);
            fullPrim[0] = p_code / std::max(Rm * T_in, 1e-60); // ρ = p/(RT)
        }
        else if (fromStr == "prim-RT")
        {
            real RT_code = fullPrim[dim + 1];
            fullPrim[dim + 1] = fullPrim[0] * RT_code; // p = ρ·RT
        }

        Eigen::VectorXd prim = fullPrim;
        int Ns1 = nSpecies;
        if (isReactive)
        {
            Ns1 = nSpecies - 1;
            int Isp = dim + 2;
            double invU0sq = 1.0 / (cfg.U0 * cfg.U0);
            auto hfView = (*pool)[0].mixtureFormationRhoESpecies(invU0sq);
            rhoH_form = 0;
            for (int k = 0; k < Ns1; ++k)
                rhoH_form += prim[Isp + k] * prim[0] * hfView[k];
            real lastY = 1.0;
            for (int k = 0; k < Ns1; ++k)
                lastY -= prim[Isp + k];
            rhoH_form += lastY * prim[0] * hfView[Ns1];
        }

        if (dim == 3)
            Gas::IdealGasThermalPrimitive2Conservative<3>(prim, consTotal, cfg.gamma, rhoH_form);
        else
            Gas::IdealGasThermalPrimitive2Conservative<2>(prim, consTotal, cfg.gamma, rhoH_form);
    }

    // --- Compute temperature, gamma, rhoH_form from consTotal ---
    real T_code;
    real gammaEq;
    real rhoH_form_cons;
    real Rmix_code;

    if (isReactive)
    {
        T_code = (dim == 3) ? phys->temperature<3>(consTotal)
                            : phys->temperature<2>(consTotal);
        gammaEq = (dim == 3) ? phys->gammaEq<3>(T_code, consTotal)
                             : phys->gammaEq<2>(T_code, consTotal);
        rhoH_form_cons = phys->mixtureFormationRhoE(consTotal);
        Rmix_code = phys->Rgas(consTotal);
    }
    else
    {
        T_code = (dim == 3) ? computeTemperatureNonReactive<3>(consTotal, cfg.gamma, R_code)
                            : computeTemperatureNonReactive<2>(consTotal, cfg.gamma, R_code);
        gammaEq = cfg.gamma;
        rhoH_form_cons = 0;
        Rmix_code = R_code;
    }

    // --- Convert consTotal to prim --- (use cfg.gamma for exact round-trip)
    Eigen::VectorXd primCode(nVars);
    if (dim == 3)
    {
        Gas::IdealGasThermalConservative2Primitive<3>(consTotal, primCode, cfg.gamma, rhoH_form_cons);
    }
    else
    {
        Gas::IdealGasThermalConservative2Primitive<2>(consTotal, primCode, cfg.gamma, rhoH_form_cons);
    }

    // --- Build consSensible ---
    Eigen::VectorXd consSensible = consTotal;
    consSensible[dim + 1] -= rhoH_form_cons;

    // --- Build variable names ---
    std::vector<std::string> consNames;
    consNames.push_back("rho");
    if (dim >= 1)
        consNames.push_back("rhoU");
    if (dim >= 2)
        consNames.push_back("rhoV");
    if (dim >= 3)
        consNames.push_back("rhoW");
    consNames.push_back("rhoE");
    for (int k = 0; k < nVars - (dim + 2); ++k)
        consNames.push_back("rhoY_" + std::to_string(k));

    std::vector<std::string> primNames;
    primNames.push_back("rho");
    primNames.push_back("u");
    primNames.push_back("v");
    if (dim >= 3)
        primNames.push_back("w");
    primNames.push_back("p");
    for (int k = 0; k < nVars - (dim + 2); ++k)
        primNames.push_back("Y_" + std::to_string(k));

    // --- Print all representations ---
    std::cout << "\n--- Conservative (total rhoE, code) ---\n";
    for (int i = 0; i < nVars; ++i)
    {
        const std::string nm = (i < (int)consNames.size()) ? consNames[i] : fmt::format("[{}]", i);
        std::string sp;
        if (i >= dim + 2 && !speciesNames.empty() && i - (dim + 2) < (int)speciesNames.size())
            sp = fmt::format("  ({})", speciesNames[i - (dim + 2)]);
        std::cout << fmt::format("  {:10s} = {:12.6f}{}\n", nm, consTotal[i], sp);
    }

    std::cout << "\n--- Conservative (sensible rhoE, code) ---\n";
    for (int i = 0; i < nVars; ++i)
    {
        const std::string nm = (i < (int)consNames.size()) ? consNames[i] : fmt::format("[{}]", i);
        std::string sp;
        if (i >= dim + 2 && !speciesNames.empty() && i - (dim + 2) < (int)speciesNames.size())
            sp = fmt::format("  ({})", speciesNames[i - (dim + 2)]);
        std::cout << fmt::format("  {:10s} = {:12.6f}{}\n", nm, consSensible[i], sp);
    }

    std::cout << "\n--- Primitive (code) ---\n";
    for (int i = 0; i < nVars; ++i)
    {
        const std::string nm = (i < (int)primNames.size()) ? primNames[i] : fmt::format("[{}]", i);
        std::string sp;
        if (i >= dim + 2 && !speciesNames.empty() && i - (dim + 2) < (int)speciesNames.size())
            sp = fmt::format("  ({})", speciesNames[i - (dim + 2)]);
        std::cout << fmt::format("  {:10s} = {:12.6f}{}\n", nm, primCode[i], sp);
    }

    real T_phys = T_code * cfg.T0;
    real p_phys = primCode[dim + 1] * cfg.rho0 * cfg.U0 * cfg.U0;

    std::cout << "\n--- Derived ---\n";
    std::cout << fmt::format("  T          = {:12.6f} K (code)\n", T_code);
    std::cout << fmt::format("  T_phys     = {:12.6f} K (physical)\n", T_phys);
    std::cout << fmt::format("  p_phys     = {:12.6f} Pa\n", p_phys);
    std::cout << fmt::format("  gamma_cfg  = {:12.6f} (input config)\n", cfg.gamma);
    std::cout << fmt::format("  gamma_eq   = {:12.6f} (from Cantera EOS)\n", gammaEq);
    if (isReactive && std::abs(gammaEq - cfg.gamma) > 1e-4)
    {
        real vel2 = 0;
        for (int j = 1; j <= dim; ++j)
            vel2 += consTotal[j] * consTotal[j];
        vel2 /= (consTotal[0] * consTotal[0]);
        real e_sensible = consTotal[dim + 1] / consTotal[0] - 0.5 * vel2 - rhoH_form_cons / consTotal[0];
        std::cout << fmt::format("  p_eos      = {:12.6f} (code, via gamma_eq; differs from input gamma_cfg)\n",
                                 gammaEq * consTotal[0] * e_sensible);
    }
    std::cout << fmt::format("  Rmix_phys  = {:12.6f} J/(kg.K)\n",
                             Rmix_code * cfg.U0 * cfg.U0 / cfg.T0);
    std::cout << fmt::format("  rhoH_form  = {:12.6f} (code)\n", rhoH_form_cons);

    if (!speciesNames.empty())
    {
        int Ns1 = nSpecies - 1;
        int Isp = dim + 2;
        std::cout << fmt::format("\n--- Species ({} transported + 1 derived) ---\n", Ns1);
        for (int k = 0; k < Ns1; ++k)
            std::cout << fmt::format("  {:10s} Y={:.6f}  rhoY={:.6f}\n",
                                     speciesNames[k], primCode[Isp + k], consTotal[Isp + k]);
        real Y_derived = 1.0;
        real rhoY_derived = consTotal[0];
        for (int k = 0; k < Ns1; ++k)
        {
            Y_derived -= primCode[Isp + k];
            rhoY_derived -= consTotal[Isp + k];
        }
        std::cout << fmt::format("  {:10s} Y={:.6f}  rhoY={:.6f} (derived)\n",
                                 speciesNames[Ns1], Y_derived, rhoY_derived);
    }

    if (inputPhys)
    {
        std::cout << "\n--- Physical units ---\n";
        Eigen::VectorXd consPhys = consTotal;
        consPhys[0] *= cfg.rho0;
        for (int j = 1; j <= dim; ++j)
            consPhys[j] *= (cfg.rho0 * cfg.U0);
        consPhys[dim + 1] *= (cfg.rho0 * cfg.U0 * cfg.U0);
        for (int k = dim + 2; k < nVars; ++k)
            consPhys[k] *= cfg.rho0;

        printVec(consPhys, "  cons-total = [", 4);

        Eigen::VectorXd primPhys = primCode;
        primPhys[0] *= cfg.rho0;
        for (int j = 1; j <= dim; ++j)
            primPhys[j] *= cfg.U0;
        primPhys[dim + 1] *= (cfg.rho0 * cfg.U0 * cfg.U0);

        printVec(primPhys, "  prim       = [", 4);
    }

    return 0;
}
