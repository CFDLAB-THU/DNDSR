#include "Cauchy/Physics/PhysicsResources.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Input/StateInput.hpp"
#include <filesystem>
#include <cstdlib>

namespace DNDS::Cauchy
{
    PhysicsResources PhysicsResources::Resolve(const EquationSettings &settings, index workerNumber)
    {
        const auto scales = ResolveReferenceScales(settings);
        if (settings.system == "incompressibleACM" || settings.eos == "perfectGas")
        {
            DNDS_check_throw_info(settings.mixture.mechanismFile.empty() && settings.mixture.phaseName.empty(), "perfect gas has unused mixture input");
            return {EquationSystem::Resolve(settings), {}};
        }
        DNDS_check_throw_info(settings.system == "compressible" && settings.eos == "idealGasMixture" &&
                                  (settings.source == "none" || settings.source == "chemical") &&
                                  (settings.transport == "inviscid" || settings.transport == "NewtonianFourier") &&
                                  settings.viscosity == 0 && settings.viscosityModel == "constant" &&
                                  settings.viscosityReferenceTemperature == 1 && settings.sutherlandConstant == 0 && workerNumber > 0 && !settings.mixture.mechanismFile.empty(),
                              "invalid or unavailable mixture equation setup");
        const auto &input = settings.mixture;
        auto file = std::filesystem::path(input.mechanismFile);
        if (file.is_relative() && !std::filesystem::exists(file))
            if (const auto *directory = std::getenv("DNDS_MECH_PATH"))
                if (const auto candidate = std::filesystem::path(directory) / file; std::filesystem::exists(candidate))
                    file = candidate;
        Physics::Chemistry::ChemicalSource prototype(file.string(), input.phaseName, scales.U0, scales.rho0);
        auto equation = EquationSystem::FromIdealGasMixture(Physics::Thermodynamics::IdealGasMixture(prototype, scales), settings.transport == "NewtonianFourier");
        return {equation.WithRANS(settings.rans), Physics::Chemistry::ChemicalWorkerPool(prototype, workerNumber)};
    }
}
