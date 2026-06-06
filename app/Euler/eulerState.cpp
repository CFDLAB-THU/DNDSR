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
#include <cmath>

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
        if (s == "NS_2D")
            return NS_2D;
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

    bool validPositive(real v)
    {
        return std::isfinite(v) && v > 0;
    }

    std::string canonicalFrom(std::string from, bool &inputPhys)
    {
        if (from == "cons_phy")
            inputPhys = true, from = "cons";
        else if (from == "consSensible_phy")
            inputPhys = true, from = "consSensible";
        else if (from == "primRhoP_phy")
            inputPhys = true, from = "primRhoP";
        else if (from == "primRhoT_phy")
            inputPhys = true, from = "primRhoT";
        else if (from == "primTP_phy")
            inputPhys = true, from = "primTP";

        if (from == "cons-total")
            return "cons";
        if (from == "cons-sensible")
            return "consSensible";
        if (from == "prim")
            return "primRhoP";
        if (from == "prim-rhoT")
            return "primRhoT";
        if (from == "prim-TP")
            return "primTP";
        return from;
    }

    bool validFrom(const std::string &from)
    {
        return from == "cons" || from == "consSensible" ||
               from == "primRhoP" || from == "primRhoT" || from == "primTP";
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
        printJsonVec(vPhys, jsonLabel + "_phy");
    }

    template <EulerModel model>
    int run_main(int nVars, Eigen::VectorXd inputState, bool isReactive, bool inputPhys,
                 std::string mechStr,
                 std::string fromStr,
                 Cfg cfg)
    {
        constexpr int dim = EulerModelTraits<model>::dim;
        std::shared_ptr<std::vector<ChemicalSource>> pool;
        std::unique_ptr<PhysicsProperties<model>> phys;
        int nSpecies = 0;
        std::vector<std::string> speciesNames;
        typename EulerEvaluatorSettings<model>::IdealGasProperty igProp;
        igProp.gamma = cfg.gamma;
        igProp.Rgas = cfg.Rgas;
        igProp.U0 = cfg.U0;
        igProp.rho0 = cfg.rho0;
        igProp.T0 = cfg.T0;
        igProp.L0 = cfg.L0;
        igProp.muGas = 1e-200;
        igProp.prGas = 0.72;
        phys = std::make_unique<PhysicsProperties<model>>(igProp);

        if (isReactive)
        {
            if (model != NS_EX && model != NS_EX_3D)
            {
                std::cerr << "Error: --mechanism only valid for NS_EX / NS_EX_3D\n";
                return 1;
            }
            pool = std::make_shared<std::vector<ChemicalSource>>();
            pool->emplace_back(mechStr, "", cfg.U0, cfg.rho0);
            nSpecies = (*pool)[0].nSpecies();
            speciesNames = (*pool)[0].speciesNames();
            int expectedNVars = dim + 2 + nSpecies - 1;
            if (nVars != expectedNVars)
            {
                std::cerr << "Error: reactive --state has nVars=" << nVars
                          << ", but mechanism has " << nSpecies
                          << " species; expected nVars=" << expectedNVars << "\n";
                return 1;
            }
            phys->setChemicalSourcePool(pool);
        }

        using TU = typename PhysicsProperties<model>::TU; // VectorFMTSafe<real,Dynamic> assignable from VectorXd

        // --- Scaling conversion (phys → code) --- using PhysicsProperties API
        if (inputPhys)
        {
            TU u(inputState.size());
            u = inputState;
            TU o(u.size());
            auto callSc = [&](auto fn)
            { fn(u, o); inputState = o; };
            bool isPrim = (fromStr == "primRhoP" || fromStr == "primRhoT" || fromStr == "primTP");
            if (!isPrim)
                callSc([&](auto &a, auto &b)
                       { phys->consPhysToCode(a, b); });
            else if (fromStr == "primRhoT")
                callSc([&](auto &a, auto &b)
                       { phys->primRhoTPhysToCode(a, b); });
            else if (fromStr == "primTP")
                callSc([&](auto &a, auto &b)
                       { phys->primTPPhysToCode(a, b); });
            else if (fromStr == "primRhoP")
                callSc([&](auto &a, auto &b)
                       { phys->primPhysToCode(a, b); });
            else
                DNDS_assert_info(false, fmt::format("fromStr invalid {}", fromStr));
        }

        std::cout << "\n";
        printJsonVec(inputState, " code-scaled input = ");

        if (fromStr == "primRhoP" && (inputState[0] <= 0 || inputState[dim + 1] <= 0))
        {
            std::cerr << "Error: primRhoP input requires positive rho and p\n";
            return 1;
        }
        if (fromStr == "primRhoT" && (inputState[0] <= 0 || inputState[dim + 1] <= 0))
        {
            std::cerr << "Error: primRhoT input requires positive rho and T\n";
            return 1;
        }
        if (fromStr == "primTP" && (inputState[0] <= 0 || inputState[dim + 1] <= 0))
        {
            std::cerr << "Error: primTP input requires positive T and p\n";
            return 1;
        }

        // --- State conversion --- using PhysicsProperties API
        TU consTotal(nVars);
        if (fromStr == "cons")
        {
            consTotal = inputState;
        }
        else if (fromStr == "consSensible")
        {
            TU u(nVars);
            u = inputState;
            phys->consSensibleToTotal(u, consTotal);
        }
        else if (fromStr == "primRhoP")
        {
            TU u(nVars);
            u = inputState;
            phys->primToConservative(u, consTotal);
        }
        else if (fromStr == "primRhoT")
        {
            TU u(nVars);
            u = inputState;
            phys->primRhoTToConservative(u, consTotal);
        }
        else if (fromStr == "primTP")
        {
            TU u(nVars);
            u = inputState;
            phys->primTPToConservative(u, consTotal);
        }
        else
            DNDS_assert_info(false, fmt::format("fromStr invalid {}", fromStr));

        printJsonVec(consTotal, "consTotal = ");

        // --- Compute temperature, gamma, rhoE_base from consTotal ---
        real T_code = phys->temperature(consTotal);
        real gammaEq = phys->gammaEq(T_code, consTotal);
        real rhoE_base_cons = phys->mixtureBaseInternalRhoE(consTotal);
        real Rmix_code = phys->Rgas(consTotal);

        // --- Convert consTotal to prim ---
        TU primCode(nVars);
        phys->conservativeToPrimitive(consTotal, primCode);

        // --- Build consSensible ---
        TU consSensible(nVars);
        phys->consTotalToSensible(consTotal, consSensible);

        // --- Build variable names ---
        std::vector<std::string> consNames;
        consNames.emplace_back("rho");
        if (dim >= 1)
            consNames.emplace_back("rhoU");
        if (dim >= 2)
            consNames.emplace_back("rhoV");
        if (dim >= 3)
            consNames.emplace_back("rhoW");
        consNames.emplace_back("rhoE");
        int Nsp = nVars - (dim + 2);
        for (int k = 0; k < Nsp; ++k)
        {
            std::string nm = "rhoY_" + std::to_string(k);
            if (k < (int)speciesNames.size())
                nm = "rho" + speciesNames[k];
            consNames.emplace_back(nm);
        }

        std::vector<std::string> primNames;
        primNames.emplace_back("rho");
        primNames.emplace_back("u");
        primNames.emplace_back("v");
        if (dim >= 3)
            primNames.emplace_back("w");
        primNames.emplace_back("p");
        for (int k = 0; k < Nsp; ++k)
        {
            std::string nm = "Y_" + std::to_string(k);
            if (k < (int)speciesNames.size())
                nm = speciesNames[k];
            primNames.push_back(nm);
        }

        // --- Scales ---
        {
            real p0 = phys->p0();
            real t0 = phys->t0();
            real R0 = phys->R0();
            real mu0 = phys->mu0();
            real k0 = phys->k0();
            real D0 = phys->D0();
            real S0 = phys->S0();
            real rhoU0 = phys->rhoU0();
            real rhoE0 = p0;
            real rhoEFlux0 = phys->rhoEFlux0();
            std::cout << "\n--- Reference Scales ---\n";
            std::cout << fmt::format("  rho0      = {:12.4g} kg/m^3\n", cfg.rho0);
            std::cout << fmt::format("  U0        = {:12.4g} m/s\n", cfg.U0);
            std::cout << fmt::format("  T0        = {:12.4g} K\n", cfg.T0);
            std::cout << fmt::format("  L0        = {:12.4g} m\n", cfg.L0);
            std::cout << "\n--- Derived Scales ---\n";
            std::cout << fmt::format("  t0        = {:12.4g} s              (L0 / U0)               time\n", t0);
            std::cout << fmt::format("  p0        = {:12.4g} Pa             (rho0 * U0^2)           pressure\n", p0);
            std::cout << fmt::format("  R0        = {:12.4g} J/(kg K)       (U0^2 / T0)             gas constant\n", R0);
            std::cout << fmt::format("  mu0       = {:12.4g} Pa s           (rho0 * U0 * L0)        dynamic viscosity\n", mu0);
            std::cout << fmt::format("  k0        = {:12.4g} W/(m K)        (rho0 * U0^3 * L0 / T0) thermal conductivity\n", k0);
            std::cout << fmt::format("  D0        = {:12.4g} m^2/s          (U0 * L0)               diffusivity\n", D0);
            std::cout << fmt::format("  S0        = {:12.4g} kg/(m^3 s)     (rho0 * U0 / L0)        volumetric source rate\n", S0);
            std::cout << "\n--- Conservative Variable Scales ---\n";
            std::cout << fmt::format("  rhoU0     = {:12.4g} kg/(m^2 s)     (rho0 * U0)             momentum density, mass flux/area\n", rhoU0);
            std::cout << fmt::format("  rhoE0     = {:12.4g} Pa             (rho0 * U0^2)           total energy density\n", rhoE0);
            std::cout << "\n--- Flux Scales (per unit face area) ---\n";
            std::cout << fmt::format("  rhoFlux0  = {:12.4g} kg/(m^2 s)     (rho0 * U0)             mass flux per unit area\n", rhoU0);
            std::cout << fmt::format("  rhoUFlux0 = {:12.4g} Pa             (rho0 * U0^2)           momentum flux per unit area\n", p0);
            std::cout << fmt::format("  rhoEFlux0 = {:12.4g} kg/s^3         (rho0 * U0^3)           energy flux per unit area\n", rhoEFlux0);
        }

        // --- Print all representations ---
        printSection(consTotal, consNames, nVars, dim,
                     "Conservative (total rhoE, code)", "cons", cfg, false);
        printSection(consSensible, consNames, nVars, dim,
                     "Conservative (sensible rhoE, code)", "consSensible", cfg, false);
        printSection(primCode, primNames, nVars, dim,
                     "Primitive rho/u/p/Y (code)", "primRhoP", cfg, true);

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
            real e_sensible = consTotal[dim + 1] / consTotal[0] - 0.5 * vel2 - rhoE_base_cons / consTotal[0];
            std::cout << fmt::format("  p_eos      = {:12.4g} (code, via gamma_eq)\n",
                                     (gammaEq - 1.0) * consTotal[0] * e_sensible);
        }
        std::cout << fmt::format("  Rmix_phys  = {:12.4g} J/(kg.K)\n",
                                 Rmix_code * cfg.U0 * cfg.U0 / cfg.T0);
        std::cout << fmt::format("  rhoE_base  = {:12.4g} (code)\n", rhoE_base_cons);

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
            int Ns1 = Ns - 1;
            int Isp = dim + 2;
            auto &chem = (*pool)[0];
            SpeciesBufferView YvSanitized{Ybuf.data(), Ns};
            chem.massFractions(1.0, primCode.data() + Isp, Ns1, YvSanitized);
            ConstSpeciesBufferView Yv{Ybuf.data(), Ns};
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
            std::cout << fmt::format("  u_phys(sent)   = {:12.4g} J/kg  (direct Cantera UV input)\n",
                                     uPhysFromInput);
            std::cout << fmt::format("  diff           = {:12.4g} J/kg  (sent - cantera)\n",
                                     uPhysFromInput - u_ct);
        }

        {
            real rho_code = primCode[0];
            real p_code = primCode[dim + 1];
            real a_code = std::sqrt(gammaEq * p_code / rho_code);
            real a_phys = a_code * cfg.U0;
            real vel2 = 0;
            for (int j = 1; j <= dim; ++j)
                vel2 += primCode[j] * primCode[j];
            real M = std::sqrt(vel2) / a_code;
            std::cout << "\n--- Acoustic ---\n";
            std::cout << fmt::format("  speed_of_sound = {:12.4g} (code)  {:12.4g} m/s (phys)\n", a_code, a_phys);
            std::cout << fmt::format("  Mach_number    = {:12.4g}\n", M);
        }
        return 0;
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
        .help("Input format: cons, consSensible, primRhoP, primRhoT, primTP, or *_phy variants");
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
    if (!validPositive(cfg.T0) || !validPositive(cfg.rho0) || !validPositive(cfg.U0) || !validPositive(cfg.L0))
    {
        std::cerr << "Error: T0, rho0, U0, and L0 must be finite and > 0\n";
        return 1;
    }
    if (!std::isfinite(cfg.gamma) || cfg.gamma <= 1 || !validPositive(cfg.Rgas))
    {
        std::cerr << "Error: gamma must be finite and > 1, and Rgas must be finite and > 0\n";
        return 1;
    }
    real R_code = computeRgasCode(cfg.Rgas, cfg.U0, cfg.T0);
    bool isReactive = !mechStr.empty();
    bool inputPhys = (scalingStr == "phys");
    fromStr = canonicalFrom(fromStr, inputPhys);
    if (scalingStr != "code" && scalingStr != "phys")
    {
        std::cerr << "Error: --scaling must be code or phys\n";
        return 1;
    }
    if (!validFrom(fromStr))
    {
        std::cerr << "Error: --from must be one of cons, consSensible, primRhoP, primRhoT, primTP, or *_phy variants\n";
        return 1;
    }

    Eigen::VectorXd inputState = parseState(stateStr);
    if ((int)inputState.size() != nVars)
    {
        std::cerr << "Error: --state has " << inputState.size()
                  << " elements, expected " << nVars << "\n";
        return 1;
    }

    std::cout << fmt::format("=== Input: {}, {} units ===\n", fromStr,
                             inputPhys ? "physical" : "code");
    std::cout << fmt::format("  model={}  dim={}  nVars={}  gamma={:.4g}  Rgas_cfg={:.4g}  U0={:.4g}  rho0={:.4g}  T0={:.4g}\n",
                             program.get<std::string>("--model"), dim, nVars,
                             cfg.gamma, R_code, cfg.U0, cfg.rho0, cfg.T0);

#define RUN_MAIN_CALL(model)                                  \
    run_main<model>(nVars, inputState, isReactive, inputPhys, \
                    mechStr,                                  \
                    fromStr,                                  \
                    cfg)

#define SWITCH_MODEL(model) \
    case model:             \
        return RUN_MAIN_CALL(model);

    switch (model)
    {
        SWITCH_MODEL(NS)
        SWITCH_MODEL(NS_2D)
        SWITCH_MODEL(NS_3D)
        SWITCH_MODEL(NS_SA)
        SWITCH_MODEL(NS_SA_3D)
        SWITCH_MODEL(NS_2EQ)
        SWITCH_MODEL(NS_2EQ_3D)
        SWITCH_MODEL(NS_EX)
        SWITCH_MODEL(NS_EX_3D)
    default:
        DNDS_assert(false);
    }
    return -1;
}
