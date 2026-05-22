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
        real L0 = 1.0;
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
            else if (key == "L0")
                c.L0 = std::stod(val);
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
            std::cout << fmt::format("{:12.4g}", v[i]);
        }
        std::cout << "\n";
    }

    void printJsonVec(const Eigen::VectorXd &v, const std::string &label)
    {
        json j;
        for (int i = 0; i < (int)v.size(); ++i)
            j.push_back(v[i]);
        std::cout << fmt::format("  // {}: {}\n", label, j.dump());
    }

    void printSection(const Eigen::VectorXd &v,
                      const std::vector<std::string> &names,
                      int nVars, int dim,
                      const std::string &title, const std::string &jsonLabel,
                      const Cfg &cfg, bool isPrim)
    {
        std::string unitRho = "kg/m^3", unitVel = "kg/(m^2 s)", unitPress = "J/m^3";
        if (isPrim)
            unitVel = "m/s", unitPress = "Pa";
        std::cout << fmt::format("\n--- {} ---\n", title);
        std::cout << fmt::format("  {:12s} {:>12s} {:>12s}  {}\n", "var", "code", "phys", "unit");
        std::cout << fmt::format("  {:12s} {:>12s} {:>12s}  {}\n",
                                 std::string(12, '-'), std::string(12, '-'), std::string(12, '-'), std::string(20, '-'));
        int I4 = dim + 1;
        for (int i = 0; i < nVars; ++i)
        {
            const std::string nm = (i < (int)names.size()) ? names[i] : fmt::format("[{}]", i);
            real phys;
            std::string unit;
            if (i == 0)
            {
                phys = v[i] * cfg.rho0;
                unit = unitRho;
            }
            else if (i >= 1 && i <= dim)
            {
                phys = v[i] * (isPrim ? cfg.U0 : cfg.rho0 * cfg.U0);
                unit = unitVel;
            }
            else if (i == I4)
            {
                phys = v[i] * cfg.rho0 * cfg.U0 * cfg.U0;
                unit = unitPress;
            }
            else
            {
                if (isPrim)
                {
                    phys = v[i];
                    unit = "-";
                }
                else
                {
                    phys = v[i] * cfg.rho0;
                    unit = unitRho;
                }
            }
            std::cout << fmt::format("  {:12s} {:12.4g} {:12.4g}  {}\n", nm, v[i], phys, unit);
        }
        printJsonVec(v, jsonLabel);
        Eigen::VectorXd vPhys = v;
        vPhys[0] = v[0] * cfg.rho0;
        for (int j = 1; j <= dim; ++j)
            vPhys[j] = v[j] * (isPrim ? cfg.U0 : cfg.rho0 * cfg.U0);
        vPhys[I4] = v[I4] * cfg.rho0 * cfg.U0 * cfg.U0;
        if (!isPrim)
            for (int k = I4 + 1; k < (int)vPhys.size(); ++k)
                vPhys[k] = v[k] * cfg.rho0;
        printJsonVec(vPhys, jsonLabel + "-phys");
    }

} // anonymous namespace

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("eulerState", "1.0");
    program.add_argument("--model")
        .required()
        .help("Euler model: NS, NS_3D, NS_EX, NS_EX_3D, NS_SA, NS_SA_3D, NS_2EQ, NS_2EQ_3D");
    program.add_argument("--nVars")
        .scan<'i', int>()
        .default_value(0)
        .help("Number of conservative variables (for dynamic models)");
    program.add_argument("--from")
        .required()
        .help("Input format: cons-total, cons-sensible, prim, prim-rhoT, prim-TP");
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

    using TU = typename PhysicsProperties<NS_EX>::TU; // VectorFMTSafe<real,Dynamic> assignable from VectorXd

    // --- Scaling conversion (phys → code) --- using PhysicsProperties API
    if (inputPhys)
    {
        if (isReactive)
        {
            TU u(inputState.size());
            u = inputState;
            TU o(u.size());
            auto callSc = [&](auto fn)
            { fn(u, o); inputState = o; };
            bool isPrim = (fromStr == "prim" || fromStr == "prim-rhoT" || fromStr == "prim-TP");
            if (dim == 3)
            {
                if (!isPrim)
                    callSc([&](auto &a, auto &b)
                           { phys->consPhysToCode<3>(a, b); });
                else if (fromStr == "prim-rhoT")
                    callSc([&](auto &a, auto &b)
                           { phys->primRhoTPhysToCode<3>(a, b); });
                else if (fromStr == "prim-TP")
                    callSc([&](auto &a, auto &b)
                           { phys->primTPPhysToCode<3>(a, b); });
                else
                    callSc([&](auto &a, auto &b)
                           { phys->primPhysToCode<3>(a, b); });
            }
            else
            {
                if (!isPrim)
                    callSc([&](auto &a, auto &b)
                           { phys->consPhysToCode<2>(a, b); });
                else if (fromStr == "prim-rhoT")
                    callSc([&](auto &a, auto &b)
                           { phys->primRhoTPhysToCode<2>(a, b); });
                else if (fromStr == "prim-TP")
                    callSc([&](auto &a, auto &b)
                           { phys->primTPPhysToCode<2>(a, b); });
                else
                    callSc([&](auto &a, auto &b)
                           { phys->primPhysToCode<2>(a, b); });
            }
        }
        else
        {
            // Non-reactive phys→code: use local scaling
            auto U = inputState;
            bool isPrim = (fromStr == "prim" || fromStr == "prim-rhoT" || fromStr == "prim-TP");
            if (isPrim)
            {
                if (fromStr == "prim-TP")
                    U[0] /= cfg.T0;
                else
                    U[0] /= cfg.rho0;
                for (int j = 1; j <= dim; ++j)
                    U[j] /= cfg.U0;
                if (fromStr == "prim-rhoT")
                    U[dim + 1] /= cfg.T0;
                else
                    U[dim + 1] /= (cfg.rho0 * cfg.U0 * cfg.U0);
            }
            else
            {
                U[0] /= cfg.rho0;
                for (int j = 1; j <= dim; ++j)
                    U[j] /= (cfg.rho0 * cfg.U0);
                U[dim + 1] /= (cfg.rho0 * cfg.U0 * cfg.U0);
                for (int k = dim + 2; k < nVars; ++k)
                    U[k] /= cfg.rho0;
            }
            inputState = U;
        }
    }

    std::cout << fmt::format("=== Input: {}, {} units ===\n", fromStr,
                             inputPhys ? "physical" : "code");
    std::cout << fmt::format("  model={}  dim={}  nVars={}  gamma={:.4g}  Rgas_cfg={:.4g}  U0={:.4g}  rho0={:.4g}  T0={:.4g}\n",
                             program.get<std::string>("--model"), dim, nVars,
                             cfg.gamma, R_code, cfg.U0, cfg.rho0, cfg.T0);
    std::cout << "\n";
    printVec(inputState, "  input = [", 4);

    // --- State conversion --- using PhysicsProperties API
    TU consTotal(nVars);
    if (fromStr == "cons-total")
    {
        consTotal = inputState;
    }
    else if (fromStr == "cons-sensible")
    {
        TU u(nVars);
        u = inputState;
        if (isReactive)
        {
            if (dim == 3)
                phys->consSensibleToTotal<3>(u, consTotal);
            else
                phys->consSensibleToTotal<2>(u, consTotal);
        }
        else
        {
            consTotal = u;
        }
    }
    else if (fromStr == "prim" && isReactive)
    {
        TU u(nVars);
        u = inputState;
        if (dim == 3)
            phys->primToConservative<3>(u, consTotal);
        else
            phys->primToConservative<2>(u, consTotal);
    }
    else if (fromStr == "prim-rhoT" && isReactive)
    {
        TU u(nVars);
        u = inputState;
        if (dim == 3)
            phys->primRhoTToConservative<3>(u, consTotal);
        else
            phys->primRhoTToConservative<2>(u, consTotal);
    }
    else if (fromStr == "prim-TP" && isReactive)
    {
        TU u(nVars);
        u = inputState;
        if (dim == 3)
            phys->primTPToConservative<3>(u, consTotal);
        else
            phys->primTPToConservative<2>(u, consTotal);
    }
    else
    {
        // Non-reactive primitive-family inputs: use cfg.gamma directly.
        TU prim(nVars);
        prim = inputState;
        if (fromStr == "prim-rhoT")
        {
            real rho = prim[0];
            real T = prim[dim + 1];
            prim[dim + 1] = rho * R_code * T;
        }
        else if (fromStr == "prim-TP")
        {
            real T = prim[0];
            real p = prim[dim + 1];
            prim[0] = p / std::max(R_code * T, real(1e-60));
        }
        if (dim == 3)
            Gas::IdealGasThermalPrimitive2Conservative<3>(prim, consTotal, cfg.gamma, 0);
        else
            Gas::IdealGasThermalPrimitive2Conservative<2>(prim, consTotal, cfg.gamma, 0);
    }

    // --- Compute temperature, gamma, rhoH_form from consTotal ---
    real T_code, gammaEq, rhoH_form_cons, Rmix_code;
    if (isReactive)
    {
        T_code = (dim == 3) ? phys->temperature<3>(consTotal) : phys->temperature<2>(consTotal);
        gammaEq = (dim == 3) ? phys->gammaEq<3>(T_code, consTotal) : phys->gammaEq<2>(T_code, consTotal);
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

    // --- Convert consTotal to prim ---
    TU primCode(nVars);
    if (isReactive)
    {
        if (dim == 3)
            phys->conservativeToPrimitive<3>(consTotal, primCode);
        else
            phys->conservativeToPrimitive<2>(consTotal, primCode);
    }
    else
    {
        if (dim == 3)
            Gas::IdealGasThermalConservative2Primitive<3>(consTotal, primCode, gammaEq, 0);
        else
            Gas::IdealGasThermalConservative2Primitive<2>(consTotal, primCode, gammaEq, 0);
    }

    // --- Build consSensible ---
    TU consSensible(nVars);
    if (isReactive)
    {
        if (dim == 3)
            phys->consTotalToSensible<3>(consTotal, consSensible);
        else
            phys->consTotalToSensible<2>(consTotal, consSensible);
    }
    else
    {
        consSensible = consTotal; // no formation
    }

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
    int Nsp = nVars - (dim + 2);
    for (int k = 0; k < Nsp; ++k)
    {
        std::string nm = "rhoY_" + std::to_string(k);
        if (k < (int)speciesNames.size())
            nm = "rho" + speciesNames[k];
        consNames.push_back(nm);
    }

    std::vector<std::string> primNames;
    primNames.push_back("rho");
    primNames.push_back("u");
    primNames.push_back("v");
    if (dim >= 3)
        primNames.push_back("w");
    primNames.push_back("p");
    for (int k = 0; k < Nsp; ++k)
    {
        std::string nm = "Y_" + std::to_string(k);
        if (k < (int)speciesNames.size())
            nm = speciesNames[k];
        primNames.push_back(nm);
    }

    // --- Scales ---
    {
        real p0 = cfg.rho0 * cfg.U0 * cfg.U0;
        real t0 = cfg.L0 > 0 ? cfg.L0 / cfg.U0 : 0;
        real R0 = cfg.U0 * cfg.U0 / std::max(cfg.T0, 1e-60);
        real mu0 = cfg.rho0 * cfg.U0 * cfg.L0;
        real k0 = cfg.rho0 * cfg.U0 * cfg.U0 * cfg.U0 * cfg.L0 / std::max(cfg.T0, 1e-60);
        real D0 = cfg.U0 * cfg.L0;
        real S0 = cfg.rho0 * cfg.U0 / std::max(cfg.L0, 1e-60);
        std::cout << "\n--- Reference Scales ---\n";
        std::cout << fmt::format("  p0   = {:12.4g} Pa         (rho0 * U0^2)\n", p0);
        std::cout << fmt::format("  rho0 = {:12.4g} kg/m^3\n", cfg.rho0);
        std::cout << fmt::format("  U0   = {:12.4g} m/s\n", cfg.U0);
        std::cout << fmt::format("  T0   = {:12.4g} K\n", cfg.T0);
        std::cout << fmt::format("  L0   = {:12.4g} m\n", cfg.L0);
        std::cout << fmt::format("  t0   = {:12.4g} s          (L0 / U0)\n", t0);
        std::cout << fmt::format("  R0   = {:12.4g} J/(kg K)   (U0^2 / T0)\n", R0);
        std::cout << fmt::format("  mu0  = {:12.4g} Pa s       (rho0 * U0 * L0)\n", mu0);
        std::cout << fmt::format("  k0   = {:12.4g} W/(m K)    (rho0 * U0^3 * L0 / T0)\n", k0);
        std::cout << fmt::format("  D0   = {:12.4g} m^2/s      (U0 * L0)\n", D0);
        std::cout << fmt::format("  S0   = {:12.4g} kg/(m^3 s) (rho0 * U0 / L0)\n", S0);
    }

    // --- Print all representations ---
    printSection(consTotal, consNames, nVars, dim,
                 "Conservative (total rhoE, code)", "cons-total", cfg, false);
    printSection(consSensible, consNames, nVars, dim,
                 "Conservative (sensible rhoE, code)", "cons-sensible", cfg, false);
    printSection(primCode, primNames, nVars, dim,
                 "Primitive (code)", "prim", cfg, true);

    real T_phys = T_code * cfg.T0;
    real p_phys = primCode[dim + 1] * cfg.rho0 * cfg.U0 * cfg.U0;

    std::cout << "\n--- Derived ---\n";
    std::cout << fmt::format("  T          = {:12.4g} K (code)\n", T_code);
    std::cout << fmt::format("  T_phys     = {:12.4g} K (physical)\n", T_phys);
    std::cout << fmt::format("  p_phys     = {:12.4g} Pa\n", p_phys);
    std::cout << fmt::format("  gamma_cfg  = {:12.4g} (input config)\n", cfg.gamma);
    std::cout << fmt::format("  gamma_eq   = {:12.4g} (from Cantera EOS)\n", gammaEq);
    if (isReactive && std::abs(gammaEq - cfg.gamma) > 1e-4)
    {
        real vel2 = 0;
        for (int j = 1; j <= dim; ++j)
            vel2 += consTotal[j] * consTotal[j];
        vel2 /= (consTotal[0] * consTotal[0]);
        real e_sensible = consTotal[dim + 1] / consTotal[0] - 0.5 * vel2 - rhoH_form_cons / consTotal[0];
        std::cout << fmt::format("  p_eos      = {:12.4g} (code, via gamma_eq)\n",
                                 (gammaEq - 1.0) * consTotal[0] * e_sensible);
    }
    std::cout << fmt::format("  Rmix_phys  = {:12.4g} J/(kg.K)\n",
                             Rmix_code * cfg.U0 * cfg.U0 / cfg.T0);
    std::cout << fmt::format("  rhoH_form  = {:12.4g} (code)\n", rhoH_form_cons);

    if (!speciesNames.empty())
    {
        int Ns1 = nSpecies - 1;
        int Isp = dim + 2;
        std::cout << fmt::format("\n--- Species ({} transported + 1 derived) ---\n", Ns1);
        for (int k = 0; k < Ns1; ++k)
            std::cout << fmt::format("  {:10s} Y={:12.4g}  rhoY={:12.4g}\n",
                                     speciesNames[k], primCode[Isp + k], consTotal[Isp + k]);
        real Y_derived = 1.0;
        real rhoY_derived = consTotal[0];
        for (int k = 0; k < Ns1; ++k)
        {
            Y_derived -= primCode[Isp + k];
            rhoY_derived -= consTotal[Isp + k];
        }
        std::cout << fmt::format("  {:10s} Y={:12.4g}  rhoY={:12.4g} (derived)\n",
                                 speciesNames[Ns1], Y_derived, rhoY_derived);
    }

    if (isReactive)
    {
        int Ns = nSpecies;
        std::vector<double> Ybuf(Ns);
        for (int k = 0; k < Ns; ++k)
        {
            if (k < Ns - 1)
                Ybuf[k] = primCode[dim + 2 + k];
            else
            {
                real sumY = 0;
                for (int j = 0; j < Ns - 1; ++j)
                    sumY += primCode[dim + 2 + j];
                Ybuf[k] = 1.0 - sumY;
            }
        }
        ConstSpeciesBufferView Yv{Ybuf.data(), Ns};
        auto &chem = (*pool)[0];
        double u_ct = chem.mixtureIntEnergy(T_phys, Yv, p_phys);
        double h_ct = chem.mixtureEnthalpy(T_phys, Yv, p_phys);
        double cv_ct = chem.mixtureCv(T_phys, Yv, p_phys);
        double cp_ct = chem.mixtureCp(T_phys, Yv, p_phys);
        double a_ct = chem.speedOfSound(T_phys, Yv, p_phys);
        double R_ct = chem.mixtureR(Yv);

        std::cout << "\n--- Cantera state at T_phys ---\n";
        std::cout << fmt::format("  intEnergy_mass = {:12.4g} J/kg\n", u_ct);
        std::cout << fmt::format("  enthalpy_mass  = {:12.4g} J/kg\n", h_ct);
        std::cout << fmt::format("  cv_mass        = {:12.4g} J/(kg K)\n", cv_ct);
        std::cout << fmt::format("  cp_mass        = {:12.4g} J/(kg K)\n", cp_ct);
        std::cout << fmt::format("  gamma (cp/cv)  = {:12.4g}\n", cp_ct / std::max(cv_ct, 1e-30));
        std::cout << fmt::format("  gamma (eq)     = {:12.4g}  (from DNDSR state)\n", gammaEq);
        std::cout << fmt::format("  speed_of_sound = {:12.4g} m/s\n", a_ct);
        std::cout << fmt::format("  mixture_R      = {:12.4g} J/(kg K)\n", R_ct);

        // Code-unit conversions
        std::cout << fmt::format("  intEnergy_code = {:12.4g}\n", u_ct / (cfg.U0 * cfg.U0));
        std::cout << fmt::format("  enthalpy_code  = {:12.4g}\n", h_ct / (cfg.U0 * cfg.U0));
        std::cout << fmt::format("  cv_code        = {:12.4g}\n", cv_ct / (cfg.U0 * cfg.U0 / cfg.T0));
        std::cout << fmt::format("  cp_code        = {:12.4g}\n", cp_ct / (cfg.U0 * cfg.U0 / cfg.T0));

        // Verify energy consistency: u_sent should match Cantera's intEnergy_mass
        real velSqr = 0;
        for (int j = 1; j <= dim; ++j)
            velSqr += consTotal[j] * consTotal[j];
        velSqr /= (consTotal[0] * consTotal[0]);
        real uInternal = consTotal[dim + 1] / consTotal[0] - 0.5 * velSqr;
        real uPhysFromInput = uInternal * cfg.U0 * cfg.U0;
        std::cout << fmt::format("  u_phys(input)  = {:12.4g} J/kg  (from consTotal, pre-conversion)\n", uPhysFromInput);
        std::cout << fmt::format("  u_phys(sent)   = {:12.4g} J/kg  (after pVRef + sensible offset)\n",
                                 uPhysFromInput - chem.pVAtReference(Yv) - chem.sensibleInternalEnergyAtReference(Yv));
        std::cout << fmt::format("  diff           = {:12.4g} J/kg  (sent - cantera)\n",
                                 (uPhysFromInput - chem.pVAtReference(Yv) - chem.sensibleInternalEnergyAtReference(Yv)) - u_ct);
    }

    return 0;
}
