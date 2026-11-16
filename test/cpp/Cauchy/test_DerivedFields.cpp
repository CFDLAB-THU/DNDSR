#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/State/DerivedFields.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include <cstdlib>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using Q = C::DerivedQuantity;

TEST_CASE("Derived point refresh fills selected properties with lazy provenance and stale temperature guesses")
{
    C::DomainSettings settings;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .003;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    for (DNDS::index rows : {0, 35})
        for (int layout = 0; layout < 3; ++layout)
        {
            INFO("rows=" << rows << ", layout=" << layout);
            const auto prototype = layout == 0   ? R::RowLayout::Dense(0, 5, 7)
                                   : layout == 1 ? R::RowLayout::ColMajor(0, 5, {16, 1})
                                                 : R::RowLayout::TiledColMajor(0, 5, 32);
            A::DofSpace space(size_t(rows), 0, "derived fixture");
            auto state = C::EquationDofs::Coupled(equation.Schema(), A::DofField<DNDS::real>(space, prototype));
            {
                auto write = state.WriteOwned();
                const auto u = write.Block("flow");
                for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                {
                    const double rho = 1 + .01 * iRow, velocity = .3 + .001 * iRow, pressure = 1 + .02 * iRow;
                    u(iRow, 0) = rho;
                    u(iRow, 1) = rho * velocity;
                    u(iRow, 2) = -.1 * rho;
                    u(iRow, 3) = .02 * rho;
                    u(iRow, 4) = pressure / .4 + .5 * rho * (velocity * velocity + .01 + .0004);
                }
            }
            C::DerivedFields derived(space, {Q::Temperature, Q::Pressure, Q::MolecularViscosity});
            const auto sourceVersion = state.CoupledField().Owned().Version();
            CHECK_FALSE(derived.Current(state, equation));
            CHECK(C::RefreshDerivedFields(state, equation, derived));
            auto changedLaw = settings.equation;
            changedLaw.viscosityModel = "constantKinematic";
            CHECK_FALSE(derived.Current(state, C::EquationSystem::Resolve(changedLaw)));
            CHECK(state.ComponentNumber() == 5);
            CHECK(state.CoupledField().Owned().Version() == sourceVersion);
            CHECK(derived.Values().Layout().Width() == 3);
            {
                auto read = derived.ReadCurrent(state, equation);
                const auto temperature = read.Field(Q::Temperature), pressure = read.Field(Q::Pressure), viscosity = read.Field(Q::MolecularViscosity);
                for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                {
                    const double rho = 1 + .01 * iRow, p = 1 + .02 * iRow;
                    CHECK(temperature(iRow, 0) == doctest::Approx(p / (rho * equation.Gas().gasConstant)));
                    CHECK(pressure(iRow, 0) == doctest::Approx(p));
                    CHECK(viscosity(iRow, 0) == .003);
                }
            }
            const auto version = derived.Values().Version();
            CHECK_FALSE(C::RefreshDerivedFields(state, equation, derived, true));
            CHECK(derived.Values().Version() == version);
            {
                auto write = state.WriteOwned();
                const auto u = write.Block("flow");
                for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                    u(iRow, 4) += .25;
            }
            CHECK_FALSE(derived.Current(state, equation));
            CHECK_THROWS(derived.ReadCurrent(state, equation));
            {
                auto guess = derived.ReadTemperatureGuess(state, equation);
                const auto t = guess.Field(Q::Temperature);
                for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                    CHECK(t(iRow, 0) == doctest::Approx((1 + .02 * iRow) / ((1 + .01 * iRow) * equation.Gas().gasConstant)));
            }
            auto alias = derived;
            CHECK(C::RefreshDerivedFields(state, equation, derived, true));
            CHECK(alias.Current(state, equation));
            {
                auto read = alias.ReadCurrent(state, equation);
                const auto p = read.Field(Q::Pressure);
                for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                    CHECK(p(iRow, 0) == doctest::Approx(1 + .02 * iRow + .1));
            }
            C::DerivedFields disabled(space, {});
            CHECK_FALSE(disabled.Enabled());
            CHECK_FALSE(C::RefreshDerivedFields(state, equation, disabled));
            CHECK(disabled.Values().RowCount() == 0);
        }
}

#ifdef DNDS_USE_CANTERA
TEST_CASE("Mixture derived refresh consumes compatible stale temperatures across coupled and separate layouts")
{
    namespace P = DNDS::Physics;
    const char *path = std::getenv("DNDS_MECH_PATH");
    P::Chemistry::ChemicalSource prototype(path ? std::string(path) + "/h2o2.yaml" : "h2o2.yaml", "", 340, 1.225);
    const P::ReferenceScales scales{1.225, 340, 2, 288.15};
    P::Thermodynamics::IdealGasMixture material(prototype, scales);
    const auto equation = C::EquationSystem::FromIdealGasMixture(material, true);
    P::Chemistry::ChemicalWorkerPool workers(prototype, 2);
    const int speciesNumber = material.SpeciesNumber();
    REQUIRE(speciesNumber == 10);
    A::DofSpace space(35, 2, "mixture means and separate ghost publication");
    for (bool separate : {false, true})
    {
        INFO("separate=" << separate);
        auto state = [&]
        {
            if (!separate)
                return C::EquationDofs::Coupled(equation.Schema(),
                                                A::DofField<double>(space, R::RowLayout::TiledColMajor(0, 5 + speciesNumber - 1, 32)));
            A::DofBundle fields(space);
            fields.Add("flow", A::DofField<double>(space, R::RowLayout::Dense(0, 5, 7)));
            fields.Add("species", A::DofField<double>(space, R::RowLayout::ColMajor(0, speciesNumber - 1, {16, 1})));
            return C::EquationDofs::Separate(equation.Schema(), std::move(fields));
        }();
        std::vector<C::DerivedPointValues> expected(space.OwnedRows());
        const auto fill = [&](double temperatureIncrement)
        {
            auto writer = state.WriteOwned();
            const auto flow = writer.Block("flow"), species = writer.Block("species");
            std::vector<double> Y(speciesNumber), diffusion(speciesNumber);
            for (DNDS::index iCell = 0; iCell < DNDS::index(space.OwnedRows()); ++iCell)
            {
                const double rho = .8 + .01 * iCell, temperature = 900 + 3 * iCell + temperatureIncrement;
                std::fill(Y.begin(), Y.end(), 0);
                Y[0] = .026 + .0001 * iCell + .00001 * temperatureIncrement;
                Y[3] = .224 - .00005 * iCell - .000005 * temperatureIncrement;
                Y[5] = .02 + .00007 * iCell;
                Y.back() = 1 - Y[0] - Y[3] - Y[5];
                P::Chemistry::ConstSpeciesBufferView fractions{Y.data(), speciesNumber};
                const double Rmix = prototype.mixtureR(fractions);
                const double pressure = rho * scales.rho0 * Rmix * temperature;
                const double internalEnergy = prototype.mixtureIntEnergy(temperature, fractions, pressure) / scales.SpecificEnergy();
                flow(iCell, 0) = rho;
                flow(iCell, 1) = .2 * rho;
                flow(iCell, 2) = -.04 * rho;
                flow(iCell, 3) = .03 * rho;
                flow(iCell, 4) = rho * (internalEnergy + .5 * (.04 + .0016 + .0009));
                for (int k = 0; k < speciesNumber - 1; ++k)
                    species(iCell, k) = rho * Y[k];
                const double cv = prototype.mixtureCv(temperature, fractions, pressure) / scales.HeatCapacity();
                const double mu = prototype.viscosity(temperature, pressure, fractions) / scales.Viscosity();
                const double conductivity = prototype.thermalConductivity(temperature, pressure, fractions) / scales.Conductivity();
                prototype.speciesDiffusivity(temperature, pressure, fractions, {diffusion.data(), speciesNumber});
                double bound = std::max((4. / 3) * mu / rho, conductivity / (rho * cv));
                for (double value : diffusion)
                    bound = std::max(bound, value / scales.Diffusivity());
                expected[iCell] = {temperature / scales.T0, pressure / scales.Pressure(),
                                   prototype.speedOfSound(temperature, fractions, pressure) / scales.U0, mu, conductivity,
                                   prototype.mixtureCp(temperature, fractions, pressure) / scales.HeatCapacity(), cv,
                                   Rmix / scales.HeatCapacity(), bound,
                                   (flow(iCell, 4) + pressure / scales.Pressure()) / rho,
                                   Rmix / scales.HeatCapacity() / cv};
            }
        };
        const auto check = [&](const C::DerivedFields &fields)
        {
            const auto read = fields.ReadCurrent(state, equation);
            for (int quantity = 0; quantity < C::DerivedQuantityNumber; ++quantity)
            {
                const auto field = read.Field(Q(quantity));
                for (DNDS::index iCell = 0; iCell < DNDS::index(space.OwnedRows()); ++iCell)
                    CHECK(field(iCell, 0) == doctest::Approx(expected[iCell][quantity]).epsilon(3e-11));
            }
        };
        fill(0);
        C::DerivedRequest request;
        request.enabled.fill(true);
        C::DerivedFields fields(space, request);
        std::vector<uint64_t> versions;
        for (DNDS::index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            versions.push_back(state.PhysicalField(iField).Owned().Version());
        {
            auto activeGhost = state.PhysicalField(0).Ghost().Write();
            CHECK(C::RefreshDerivedFields(state, equation, fields, workers.Acquire()));
            activeGhost.CancelUnmodified();
        }
        check(fields);
        // Queued refresh reserves a bounded context team; both CPU launcher
        // modes must reproduce the same independently evaluated properties.
        for (const size_t engineWorkers : {size_t(0), size_t(1)})
        {
            R::EagerEngine engine(engineWorkers);
            const auto stream = engine.CreateCPUStream();
            C::DerivedFields parallelFields(space, request);
            C::RefreshDerivedFields(engine, stream, state, equation, parallelFields, workers, false);
            engine.Synchronize();
            check(parallelFields);
        }
        for (DNDS::index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            CHECK(state.PhysicalField(iField).Owned().Version() == versions[iField]);
        CHECK_FALSE(C::RefreshDerivedFields(state, equation, fields, workers.Acquire(), true));
        fill(75);
        CHECK_FALSE(fields.Current(state, equation));
        {
            const auto stale = fields.ReadTemperatureGuess(state, equation);
            const auto temperature = stale.Field(Q::Temperature);
            CHECK(temperature(17, 0) == doctest::Approx((900 + 3 * 17) / scales.T0).epsilon(3e-11));
        }
        CHECK(C::RefreshDerivedFields(state, equation, fields, workers.Acquire(), true));
        check(fields);
        C::DerivedFields temperatureOnly(space, {Q::Temperature});
        CHECK(C::RefreshDerivedFields(state, equation, temperatureOnly, workers.Acquire()));
        CHECK(temperatureOnly.Values().Layout().Width() == 1);
        {
            const auto read = temperatureOnly.ReadCurrent(state, equation);
            CHECK(read.Field(Q::Temperature)(17, 0) == doctest::Approx(expected[17][int(Q::Temperature)]).epsilon(3e-11));
        }
        auto otherScales = scales;
        otherScales.T0 *= 2;
        const auto other = C::EquationSystem::FromIdealGasMixture(P::Thermodynamics::IdealGasMixture(prototype, otherScales), true);
        CHECK_FALSE(fields.Current(state, other));
        CHECK_THROWS(fields.ReadTemperatureGuess(state, other)); // Same loaded mechanism, incompatible code-temperature units.
        CHECK_THROWS(equation.Gas());                            // No silent perfect-gas fallback for a mixture.
    }
}
#endif
