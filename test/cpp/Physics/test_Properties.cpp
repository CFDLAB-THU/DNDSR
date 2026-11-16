#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Physics/Thermodynamics/IdealGasMixture.hpp"
#include "Physics/RANS/RANSModels.hpp"
#include "Euler/Physics/PhysicsProperties.hpp"
#include "Euler/RANS_ke.hpp"
#include <future>
#include <optional>
#include <type_traits>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace P = DNDS::Physics;
namespace C = P::Chemistry;
using Mixture = P::Thermodynamics::IdealGasMixture;
static_assert(std::is_same_v<C::ChemicalSource, DNDS::Euler::Chemistry::ChemicalSource>);
static_assert(std::is_same_v<P::RANS::SAConfig, DNDS::Euler::RANS::SAConfig>);

static std::string Mechanism()
{
    const char *path = std::getenv("DNDS_MECH_PATH");
    return path ? std::string(path) + "/h2o2.yaml" : "h2o2.yaml";
}
static std::vector<double> Fractions(int species)
{
    std::vector<double> result(species, 0);
    result[0] = 0.026;
    result[3] = 0.224;
    result.back() = 0.75;
    return result;
}

TEST_CASE("Loaded model description detects changed kinetics with unchanged species")
{
    C::ChemicalSource source(Mechanism(), "", 340, 1.225);
    const auto original = source.ModelDescription();
    auto changed = original;
    const auto start = changed.find("A: ");
    REQUIRE(start != std::string::npos);
    size_t length = 0;
    const auto rate = std::stod(changed.substr(start + 3), &length);
    REQUIRE(rate > 0);
    changed.replace(start + 3, length, std::to_string(2 * rate));
    const auto path = std::filesystem::temp_directory_path() / ("cauchy-model-" + std::to_string(getpid()) + ".yaml");
    struct Cleanup
    {
        std::filesystem::path path;
        ~Cleanup()
        {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{path};
    {
        std::ofstream file(path);
        file << changed;
        REQUIRE(file.good());
    }
    C::ChemicalSource other(path.string(), "", 340, 1.225);
    CHECK(other.speciesNames() == source.speciesNames());
    CHECK(other.ModelDescription() != original);
}

TEST_CASE("shared mixture: fused SI outputs, scaled UV and legacy Euler properties agree")
{
    const P::ReferenceScales scales{1.225, 340, 2, 288.15};
    auto legacyPool = std::make_shared<std::vector<C::ChemicalSource>>();
    legacyPool->emplace_back(Mechanism(), "", scales.U0, scales.rho0);
    auto &source = legacyPool->front();
    const auto modelDescription = source.ModelDescription();
    CHECK(modelDescription == source.clone()->ModelDescription());
    const int species = source.nSpecies();
    REQUIRE(species == 10);
    auto fractions = Fractions(species);
    const C::ConstSpeciesBufferView Y{fractions.data(), species};
    const double temperature = 1300, pressure = 170000;
    P::Thermodynamics::Properties thermo;
    P::Transport::Properties transport;
    std::vector<double> diffusion(species), enthalpy(species);
    source.EvaluateProperties(temperature, pressure, Y, thermo, &transport,
                              {diffusion.data(), species}, {enthalpy.data(), species});
    CHECK(source.ModelDescription() == modelDescription); // Mutable TPY is not restart identity.
    CHECK(thermo.pressure == pressure);
    CHECK(thermo.internalEnergy == doctest::Approx(source.mixtureIntEnergy(temperature, Y, pressure)).epsilon(1e-13));
    CHECK(thermo.soundSpeed == doctest::Approx(source.speedOfSound(temperature, Y, pressure)).epsilon(1e-13));
    CHECK(thermo.cp == doctest::Approx(source.mixtureCp(temperature, Y, pressure)).epsilon(1e-13));
    CHECK(thermo.enthalpy == doctest::Approx(thermo.internalEnergy + pressure / thermo.density).epsilon(1e-13));
    CHECK(transport.viscosity == source.viscosity(temperature, pressure, Y));

    DNDS::Euler::EulerEvaluatorSettings<DNDS::Euler::NS_EX_3D>::IdealGasProperty settings;
    settings.rho0 = scales.rho0;
    settings.U0 = scales.U0;
    settings.L0 = scales.L0;
    settings.T0 = scales.T0;
    DNDS::Euler::PhysicsProperties<DNDS::Euler::NS_EX_3D> legacy(settings);
    legacy.setChemicalSourcePool(legacyPool);
    DNDS::Euler::PhysicsProperties<DNDS::Euler::NS_EX_3D>::TU prim(5 + species - 1), state(prim.size());
    prim.setZero();
    prim[0] = temperature / scales.T0;
    prim[1] = 0.08;
    prim[2] = -0.02;
    prim[3] = 0.01;
    prim[4] = pressure / scales.Pressure();
    for (int k = 0; k < species - 1; ++k)
        prim[5 + k] = fractions[k];
    legacy.primTPToConservative(prim, state);
    const double rho = state[0];
    const double internalEnergy = state[4] / rho - 0.5 * state.segment<3>(1).squaredNorm() / (rho * rho);
    C::ChemicalWorkerPool pool(source, 1);
    Mixture material(source, scales);
    auto work = material.Borrow(pool.Acquire(), {true, true, true});
    auto rhoYInput = work.SpeciesInput();
    for (int k = 0; k < rhoYInput.nSpecies; ++k)
        rhoYInput[k] = state[5 + k];
    auto point = work.FromRhoE(rho, internalEnergy, rhoYInput, 800 / scales.T0);
    // The numerical adapter preserves the established UV solution and reports
    // malformed requests as values rather than an exception-based fallback.
    P::Thermodynamics::MixturePoint checkedPoint;
    const auto status = work.TryFromConservative(rho, state[4], {state[1], state[2], state[3]},
                                                 rhoYInput, 800 / scales.T0, {true, true, true}, checkedPoint);
    REQUIRE(bool(status));
    CHECK(checkedPoint.thermo.temperature == doctest::Approx(point.thermo.temperature).epsilon(2e-11));
    const auto invalid = work.TryFromConservative(rho, state[4], {state[1], state[2], state[3]},
                                                  {}, 0, {}, checkedPoint);
    CHECK(invalid.code == DNDS::NumericalError::InvalidInput);
    CHECK_FALSE(P::AllowsMeanFallback(invalid));
    CHECK(point.thermo.temperature == doctest::Approx(prim[0]).epsilon(2e-11));
    CHECK(point.thermo.temperature == doctest::Approx(legacy.temperature(state, prim[0])).epsilon(2e-11));
    CHECK(point.thermo.pressure == doctest::Approx(prim[4]).epsilon(2e-11));
    CHECK(point.thermo.soundSpeed == doctest::Approx(thermo.soundSpeed / scales.U0).epsilon(2e-11));
    CHECK(point.thermo.cp == doctest::Approx(legacy.Cp(prim[0], state)).epsilon(2e-11));
    CHECK(point.transport.viscosity == doctest::Approx(legacy.mixtureViscosity(prim[0], prim[4], state)).epsilon(2e-11));
    CHECK(point.transport.conductivity == doctest::Approx(legacy.mixtureConductivity(prim[0], prim[4], state)).epsilon(2e-11));
    for (int k = 0; k < species; ++k)
    {
        CHECK(point.fractions[k] == doctest::Approx(fractions[k]).epsilon(1e-13));
        CHECK(point.diffusivity[k] == doctest::Approx(diffusion[k] / scales.Diffusivity()).epsilon(2e-11));
        CHECK(point.speciesEnthalpy[k] == doctest::Approx(enthalpy[k] / scales.SpecificEnergy()).epsilon(2e-11));
    }
    CHECK(material.BaseInternalEnergy(rho, {state.data() + 5, species - 1}) ==
          doctest::Approx(legacy.mixtureBaseInternalRhoE(state)).epsilon(1e-13));
    CHECK(work.FromRhoP(rho, prim[4], {state.data() + 5, species - 1}).thermo.temperature ==
          doctest::Approx(prim[0]).epsilon(1e-13));
    // Formation-energy mixtures can have negative full e at an admissible T.
    std::vector<double> wet(species - 1, 0);
    wet[5] = 0.7 * rho;
    auto cold = work.FromRhoT(rho, 400 / scales.T0, {wet.data(), species - 1});
    REQUIRE(cold.thermo.internalEnergy < 0);
    auto recovered = work.FromRhoE(rho, cold.thermo.internalEnergy, {wet.data(), species - 1}, 1100 / scales.T0);
    CHECK(recovered.thermo.temperature == doctest::Approx(400 / scales.T0).epsilon(2e-11));
}

TEST_CASE("shared mixture workers: independent concurrent contexts and retained lifetime")
{
    C::ChemicalSource prototype(Mechanism(), "", 340, 1.225);
    Mixture material(prototype, {1.225, 340, 2, 288.15});
    C::ChemicalWorkerPool pool(prototype, 2);
    {
        auto team = pool.AcquireUpTo(16);
        REQUIRE(team.size() == 2);
        CHECK(team[0].Slot() != team[1].Slot());
        CHECK_THROWS(pool.AcquireUpTo(1));
        team.pop_back();
        auto remaining = pool.AcquireUpTo(16);
        REQUIRE(remaining.size() == 1);
        CHECK(remaining.front().Slot() != team.front().Slot());
    }
    auto first = pool.Acquire(), second = pool.Acquire();
    CHECK(first.Slot() != second.Slot());
    CHECK_THROWS(pool.Acquire());
    std::promise<void> start;
    auto ready = start.get_future().share();
    auto run = [material, ready](C::ChemicalWorkerPool::Lease worker, int id)
    {
        auto work = material.Borrow(std::move(worker), {true, true, true});
        const int species = material.SpeciesNumber();
        auto Y = Fractions(species);
        Y[0] += id * 0.01;
        Y.back() -= id * 0.01;
        const double rho = 0.8 + id;
        std::vector<double> rhoY(species - 1);
        for (int k = 0; k < species - 1; ++k)
            rhoY[k] = rho * Y[k];
        ready.wait();
        double maxError = 0;
        for (int iPoint = 0; iPoint < 64; ++iPoint)
        {
            const double T = (700 + 11 * iPoint + 130 * id) / material.Scales().T0;
            auto direct = work.FromRhoT(rho, T, {rhoY.data(), species - 1});
            const double pressure = direct.thermo.pressure;
            auto solved = work.FromRhoE(rho, direct.thermo.internalEnergy, {rhoY.data(), species - 1}, T * 0.8);
            maxError = std::max(maxError, std::abs(solved.thermo.temperature / T - 1));
            maxError = std::max(maxError, std::abs(solved.thermo.pressure / pressure - 1));
        }
        return maxError;
    };
    auto one = std::async(std::launch::async, run, std::move(first), 0);
    auto two = std::async(std::launch::async, run, std::move(second), 1);
    start.set_value();
    CHECK(one.get() < 2e-11);
    CHECK(two.get() < 2e-11);
    // Both moved leases returned their slots exactly once.
    auto again = pool.Acquire(), also = pool.Acquire();
    CHECK(again.Slot() != also.Slot());
    std::optional<Mixture::PointEvaluator> retained;
    {
        C::ChemicalSource local(Mechanism(), "", 1, 1);
        Mixture temporary(local, {});
        C::ChemicalWorkerPool temporaryPool(local, 1);
        retained.emplace(temporary.Borrow(temporaryPool.Acquire()));
    }
    auto Y = Fractions(material.SpeciesNumber());
    Y.pop_back();
    CHECK(retained->FromRhoT(1, 900, {Y.data(), int(Y.size())}).thermo.temperature == 900);
}

TEST_CASE("shared mixture: workers reject a different loaded material even with identical filenames")
{
    C::ChemicalSource first(Mechanism(), "", 1, 1), independent(Mechanism(), "", 1, 1);
    Mixture material(first, {});
    C::ChemicalWorkerPool wrong(independent, 1);
    CHECK_THROWS(material.Borrow(wrong.Acquire()));
    // Rejection releases its consumed lease rather than exhausting the pool.
    CHECK(bool(wrong.Acquire()));
}
