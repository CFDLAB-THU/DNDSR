#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Input/StateInput.hpp"
#include "Cauchy/Input/InitialField.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/PhysicsResources.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Euler/Physics/PhysicsProperties.hpp"
#include <cstdlib>

namespace C = DNDS::Cauchy;
namespace P = DNDS::Physics;
namespace E = DNDS::Euler;

TEST_CASE("Perfect gas molecular laws preserve Euler scaling and all property consumers")
{
    const P::ReferenceScales scales{1.7, 230, 3.2, 310};
    for (const auto &entry : {std::pair<const char *, int>{"constant", 0}, {"sutherland", 1}, {"constantKinematic", 2}})
    {
        C::EquationSettings settings;
        settings.transport = "NewtonianFourier";
        settings.gasConstant = 287 / scales.HeatCapacity();
        settings.viscosity = 1.8e-5 / scales.Viscosity();
        settings.viscosityModel = entry.first;
        if (entry.second == 1)
        {
            settings.viscosityReferenceTemperature = 273.15 / scales.T0;
            settings.sutherlandConstant = 110.4 / scales.T0;
        }
        const auto equation = C::EquationSystem::Resolve(settings);
        E::EulerEvaluatorSettings<E::NS_EX_3D>::IdealGasProperty gas;
        gas.rho0 = scales.rho0;
        gas.U0 = scales.U0;
        gas.L0 = scales.L0;
        gas.T0 = scales.T0;
        gas.Rgas = 287;
        gas.muGas = 1.8e-5;
        gas.muModel = entry.second;
        gas.TRef = 273.15;
        gas.CSutherland = 110.4;
        E::PhysicsProperties<E::NS_EX_3D> legacy(gas);
        auto evaluator = equation.MakePointEvaluator({}, {true, false, false, false, false});
        for (double rho : {.3, 1.7})
            for (double temperature : {220., 273.15, 900.})
            {
                const double codeT = temperature / scales.T0;
                const C::FlowState primitive{rho, .2, -.1, 0, rho * settings.gasConstant * codeT};
                const auto state = equation.Gas().Conservative(primitive);
                const auto point = evaluator.Evaluate(state);
                double expected = settings.viscosity;
                if (entry.second == 1)
                    expected *= std::pow(temperature / 273.15, 1.5) * (273.15 + 110.4) / (temperature + 110.4);
                if (entry.second == 2)
                    expected *= rho;
                E::PhysicsProperties<E::NS_EX_3D>::TU old(5);
                for (int j = 0; j < 5; ++j)
                    old[j] = state[j];
                CHECK(point.transport.viscosity / expected == doctest::Approx(1).epsilon(3e-14));
                CHECK(legacy.mixtureViscosity(codeT, primitive[4], old) / expected == doctest::Approx(1).epsilon(3e-14));
                CHECK(point.transport.conductivity / (expected * point.thermo.cp / settings.prandtl) == doctest::Approx(1).epsilon(3e-14));
                CHECK(equation.DiffusionScale(state) / point.diffusionBound == doctest::Approx(1).epsilon(3e-14));
                const auto thermalOnly = evaluator.Evaluate(state, {}, 0, {});
                CHECK(thermalOnly.thermo.pressure == point.thermo.pressure);
                CHECK(thermalOnly.transport.viscosity == 0);
                C::FlowGradient gradient{};
                gradient[1][1] = 2;
                gradient[4][1] = 3;
                const auto flux = equation.Gas().ViscousFlux(equation.Gas().Primitive(state), gradient, {0, 1, 0});
                CHECK(flux[1] / (2 * expected) == doctest::Approx(1).epsilon(3e-14));
                CHECK(flux[4] / (primitive[1] * 2 * expected + 3 * point.transport.conductivity) == doctest::Approx(1).epsilon(3e-14));
            }
        if (entry.second == 1)
        {
            auto changed = settings;
            changed.sutherlandConstant *= 2;
            CHECK(equation.PropertyKey().parameters != C::EquationSystem::Resolve(changed).PropertyKey().parameters);
            changed.viscosityReferenceTemperature = 0;
            CHECK_THROWS(C::EquationSystem::Resolve(changed));
        }
    }
}

TEST_CASE("Tagged physical state conversions preserve Euler variants and distinct RANS scales")
{
    const P::ReferenceScales scales{1.7, 230, 3.2, 310};
    for (const auto *model : {"none", "SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
    {
        C::EquationSettings settings;
        settings.transport = "NewtonianFourier";
        settings.viscosity = .001;
        settings.gasConstant = 287. / scales.HeatCapacity();
        settings.rans.model = model;
        const auto equation = C::EquationSystem::Resolve(settings);
        E::EulerEvaluatorSettings<E::NS_EX_3D>::IdealGasProperty gas;
        gas.rho0 = scales.rho0;
        gas.U0 = scales.U0;
        gas.L0 = scales.L0;
        gas.T0 = scales.T0;
        gas.gamma = settings.gamma;
        gas.Rgas = 287;
        E::PhysicsProperties<E::NS_EX_3D> legacy(gas);
        const auto rm = std::string(model) == "none" ? E::RANS_None : std::string(model) == "SA"         ? E::RANS_SA
                                                                  : std::string(model) == "KOmegaWilcox" ? E::RANS_KOWilcox
                                                                  : std::string(model) == "KOmegaSST"    ? E::RANS_KOSST
                                                                                                         : E::RANS_RKE;
        legacy.setRANS(rm);
        const int nr = equation.HasRANS() ? equation.RANS().ComponentNumber() : 0;
        E::PhysicsProperties<E::NS_EX_3D>::TU primitive(5 + nr), expected;
        primitive.head<5>() << .43, .12, -.03, .02, .8;
        if (nr)
            primitive[5] = .017;
        if (nr == 2)
            primitive[6] = .23;
        legacy.primToConservative(primitive, expected);
        for (const auto *type : {"cons", "consSensible", "primRhoP", "primRhoT", "primTP", "cons_phy", "consSensible_phy", "primRhoP_phy", "primRhoT_phy", "primTP_phy"})
        {
            CAPTURE(model);
            CAPTURE(type);
            E::PhysicsProperties<E::NS_EX_3D>::TU input;
            legacy.conservativeToStateValueOrigin(expected, input, E::StateValueOriginFromName(type));
            const C::Json tagged{{"type", type}, {"state", std::vector<double>(input.data(), input.data() + input.size())}};
            const auto actual = C::ResolvePhysicalState(tagged, equation, scales);
            C::PhysicalStateConverter converter(equation, scales);
            const auto descriptor = P::StateInput::Parse(tagged);
            const auto represented = converter.Represent(actual.conservative, descriptor.form, descriptor.physical);
            for (DNDS::index j = 0; j < expected.size(); ++j)
            {
                CHECK(actual.conservative[j] == doctest::Approx(expected[j]).epsilon(2e-12));
                CHECK(represented.values[j] == doctest::Approx(input[j]).epsilon(2e-12).scale(1));
            }
            CHECK(actual.primitive[0] == doctest::Approx(.43));
            if (std::string(type) == "consSensible")
                CHECK(C::ResolvePhysicalState(tagged["state"], equation, scales).conservative == actual.conservative);
        }
    }
}

TEST_CASE("Initial boundary and reference adapters share conversion without density fallback")
{
    C::DomainSettings settings;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .001;
    settings.equation.rans.model = "KOmegaSST";
    settings.numerics.faceFallback = false;
    settings.numerics.viscousGradient = "polynomial";
    settings.equation.referenceScales = {{"rho0", 2}, {"U0", 3}, {"L0", 4}, {"T0", 5}};
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    settings.initialState = settings.referenceState = C::Json{{"type", "primRhoP_phy"}, {"state", {.6, .9, 0, 0, 18, .18, .3}}};
    settings.referenceState["state"][0] = .9; // Deliberately distinct from initial rho and rho0.
    settings.boundaries["101"].kind = "farfield";
    settings.boundaries["101"].state = settings.initialState;
    const auto normalized = C::ResolvePhysicalStateInputs(settings, equation, {});
    CHECK(normalized.initialConservative[0] == doctest::Approx(.3));
    CHECK(normalized.referenceConservative[0] == doctest::Approx(.45));
    CHECK(normalized.settings.initialPrimitive == normalized.settings.boundaries.at("101").primitive);
    CHECK(normalized.settings.initialRANS == normalized.settings.boundaries.at("101").ransConservative);
    CHECK(normalized.settings.numerics.ransState.referenceConservative.at("rho") == doctest::Approx(.45));
    const auto &reference = normalized.settings.numerics.ransState.referenceConservative;
    CHECK(reference.at("rhoK") / reference.at("rho") == doctest::Approx(.02).epsilon(1e-13));
    CHECK(reference.at("rhoOmega") / reference.at("rho") == doctest::Approx(.4).epsilon(1e-13));
    const auto parsed = C::SimulationConfig::Resolve({{"defaults", C::Json(settings)}}, 1);
    CHECK(parsed.Domains()[0].settings.referenceState == settings.referenceState);
    auto bad = settings;
    bad.referenceState["state"][0] = 0;
    CHECK_THROWS(C::ResolvePhysicalStateInputs(bad, equation, {}));
    bad = settings;
    bad.equation.referenceScales.erase("rho0");
    CHECK_THROWS(C::ResolveReferenceScales(bad.equation));
    bad = settings;
    bad.initialPrimitive[0] = .4;
    CHECK_THROWS(C::ResolvePhysicalStateInputs(bad, equation, {}));
}

#ifdef DNDS_USE_CANTERA
TEST_CASE("Mixture state inputs match Euler total and sensible energies with nonunit scales")
{
    const P::ReferenceScales scales{1.225, 340, 2, 288.15};
    auto pool = std::make_shared<std::vector<P::Chemistry::ChemicalSource>>();
    const std::string path = (std::getenv("DNDS_MECH_PATH") ? std::string(std::getenv("DNDS_MECH_PATH")) + "/" : "") + "h2o2.yaml";
    pool->emplace_back(path, "", scales.U0, scales.rho0);
    auto &source = pool->front();
    P::Chemistry::ChemicalWorkerPool workers(source, 1);
    const auto equation = C::EquationSystem::FromIdealGasMixture(P::Thermodynamics::IdealGasMixture(source, scales), false);
    E::EulerEvaluatorSettings<E::NS_EX_3D>::IdealGasProperty gas;
    gas.rho0 = scales.rho0;
    gas.U0 = scales.U0;
    gas.L0 = scales.L0;
    gas.T0 = scales.T0;
    E::PhysicsProperties<E::NS_EX_3D> legacy(gas);
    legacy.setRANS(E::RANS_None);
    legacy.setChemicalSourcePool(pool);
    E::PhysicsProperties<E::NS_EX_3D>::TU primitive(5 + source.nSpecies() - 1), expected;
    primitive.setZero();
    primitive.head<5>() << .8, .2, -.04, .03, 1400 / scales.T0;
    primitive[5] = .027;
    primitive[8] = .22;
    primitive[10] = .02;
    legacy.primRhoTToConservative(primitive, expected);
    for (const auto *type : {"cons", "consSensible", "primRhoP", "primRhoT", "primTP", "cons_phy", "consSensible_phy", "primRhoP_phy", "primRhoT_phy", "primTP_phy"})
    {
        CAPTURE(type);
        E::PhysicsProperties<E::NS_EX_3D>::TU input;
        legacy.conservativeToStateValueOrigin(expected, input, E::StateValueOriginFromName(type));
        const auto actual = C::ResolvePhysicalState({{"type", type}, {"state", std::vector<double>(input.data(), input.data() + input.size())}}, equation, scales, workers.Acquire());
        C::PhysicalStateConverter converter(equation, scales, workers.Acquire());
        const auto descriptor = P::StateInput::Parse({{"type", type}, {"state", std::vector<double>(input.data(), input.data() + input.size())}});
        const auto represented = converter.Represent(actual.conservative, descriptor.form, descriptor.physical);
        for (DNDS::index j = 0; j < expected.size(); ++j)
        {
            CHECK(actual.conservative[j] == doctest::Approx(expected[j]).epsilon(3e-11).scale(1));
            CHECK(represented.values[j] == doctest::Approx(input[j]).epsilon(3e-11).scale(1));
        }
        CHECK(actual.temperature == doctest::Approx(1400 / scales.T0).epsilon(3e-11));
    }
    C::InitialFieldSettings field;
    field.exprtkInitializers.push_back({{"inRegion := 1; UExprtk[4] := 1400 + 100*x[0]; 0;"}, "primRhoT_phy"});
    C::InitialFieldEvaluator initializer(field, equation, scales, workers.Acquire());
    const std::vector<double> background(expected.data(), expected.data() + expected.size());
    for (const double x : {0., .25, 1.})
    {
        primitive[4] = (1400 + 100 * x) / scales.T0;
        legacy.primRhoTToConservative(primitive, expected);
        const auto actual = initializer.Evaluate({x, 0, 0}, 0, 0, background);
        for (DNDS::index j = 0; j < expected.size(); ++j)
            CHECK(actual[j] == doctest::Approx(expected[j]).epsilon(3e-11).scale(1));
    }
}
#endif

TEST_CASE("Spatial state overrides preserve precedence plane sign and physical expression seeds")
{
    C::InitialFieldSettings field;
    const auto state = [](double rho)
    { return C::Json{{"type", "primRhoP"}, {"state", {rho, .2, 0, 0, 1}}}; };
    field.boxInitializers.push_back({0, 1, -1, 1, -1, 1, state(2)});
    field.boxInitializers.push_back({.25, .75, -1, 1, -1, 1, state(3)});
    field.planeInitializers.push_back({1, 0, 0, -.5, state(4)});
    field.exprtkInitializers.push_back({{"if (x[0] > 0.6) { inRegion := 1; UExprtk[0] := UExprtk[0] + 2; };", "0;"}, "primRhoP_phy"});
    const P::ReferenceScales scales{2, 3, 4, 5};
    const auto equation = C::EquationSystem::Resolve(C::EquationSettings{});
    const auto background = C::ResolvePhysicalState(state(1), equation, scales).conservative;
    C::InitialFieldEvaluator evaluator(field, equation, scales);
    for (const auto entry : {std::pair<double, double>{-.1, 1}, {0, 1}, {.1, 2}, {.3, 3}, {.5, 3}, {.6, 4}, {.7, 5}, {-.1, 1}})
    {
        const auto actual = evaluator.Evaluate({entry.first, 0, 0}, 0, 0, background);
        CHECK(actual[0] == doctest::Approx(entry.second));
        CHECK(actual[1] == doctest::Approx(.2 * entry.second));
        CHECK(actual[4] == doctest::Approx(2.5 + .02 * entry.second));
    }
    // The descriptor can be valid while a selected program produces an
    // invalid physical state. Do not silently fill or clamp density.
    field.exprtkInitializers = {{{"inRegion := 1; UExprtk[0] := 0; 0;"}, "primRhoP"}};
    C::InitialFieldEvaluator bad(field, equation, scales);
    CHECK_THROWS(bad.Evaluate({.2, 0, 0}, 0, 0, background));
}
