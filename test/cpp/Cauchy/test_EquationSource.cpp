#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Operators/Source/ChemicalSource.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/ViscousFlux.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/State/EquationTimeStepper.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/Input/StateInput.hpp"
#include "Euler/SourceTermContributor.hpp"
#include "Physics/Chemistry/SpeciesAdmissibility.hpp"
#include <Eigen/LU>
#include <cstdlib>
#include <numeric>
#include <iomanip>

namespace C = DNDS::Cauchy;
namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;
namespace P = DNDS::Physics;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
namespace
{
    std::string Mechanism()
    {
        const auto *directory = std::getenv("DNDS_MECH_PATH");
        return directory ? std::string(directory) + "/h2o2.yaml" : "h2o2.yaml";
    }
    struct Fixture
    {
        const P::ReferenceScales scales{1.225, 340, 2, 288.15};
        P::Chemistry::ChemicalSource prototype{Mechanism(), "", scales.U0, scales.rho0};
        P::Thermodynamics::IdealGasMixture material{prototype, scales};
        C::EquationSystem equation;
        P::Chemistry::ChemicalWorkerPool workers{prototype, 1};
        const int species = prototype.nSpecies(), components = species + 4;
        explicit Fixture(bool viscous = false) : equation(C::EquationSystem::FromIdealGasMixture(material, viscous)) {}
        std::vector<double> Fractions(int sample) const
        {
            std::vector<double> fractions(species, 1e-6);
            fractions[0] = .027 + sample * 1e-5;
            fractions[3] = .22 - sample * 2e-5;
            fractions[5] = .02 + sample * 1e-5;
            fractions.back() = 0;
            fractions.back() = 1 - std::accumulate(fractions.begin(), fractions.end(), 0.);
            return fractions;
        }
        Eigen::VectorXd Point(int sample, double temperature = 1400)
        {
            const auto fractions = Fractions(sample);
            const P::Chemistry::ConstSpeciesBufferView Y{fractions.data(), species};
            const double rho = .8 + sample * .003;
            const double pressure = rho * scales.rho0 * prototype.mixtureR(Y) * temperature;
            Eigen::VectorXd u(components);
            u[0] = rho;
            u[1] = rho * .2;
            u[2] = rho * -.04;
            u[3] = rho * .03;
            u[4] = rho * prototype.mixtureIntEnergy(temperature, Y, pressure) / scales.SpecificEnergy() + .5 * u.segment<3>(1).squaredNorm() / rho;
            for (int k = 0; k < species - 1; ++k)
                u[5 + k] = rho * fractions[k];
            return u;
        }
        C::EquationDofs State(const A::DofSpace &space, int layout)
        {
            if (layout == 0)
                return C::EquationDofs::Coupled(equation.Schema(), A::DofField<double>(space, R::RowLayout::Dense(0, components, components + 3)));
            A::DofBundle fields(space);
            if (layout == 1)
            {
                fields.Add("flow", A::DofField<double>(space, R::RowLayout::ColMajor(0, 5, {16, 1})));
                fields.Add("species", A::DofField<double>(space, R::RowLayout::TiledColMajor(0, species - 1, 32)));
                return C::EquationDofs::Separate(equation.Schema(), std::move(fields));
            }
            // Physical species-first storage must not change logical ordering.
            fields.Add("packed", A::DofField<double>(space, R::RowLayout::TiledColMajor(0, components, 32)));
            return C::EquationDofs(equation.Schema(), std::move(fields), {{"flow", "packed", species - 1}, {"species", "packed", 0}});
        }
        void Fill(C::EquationDofs &state)
        {
            auto output = state.WriteOwned();
            const auto flow = output.Block("flow"), speciesValues = output.Block("species");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                const auto point = Point(int(iCell) + mpi.rank, 1300 + iCell * 2);
                for (int j = 0; j < 5; ++j)
                    flow(iCell, j) = point[j];
                for (int k = 0; k < species - 1; ++k)
                    speciesValues(iCell, k) = point[5 + k];
            }
        }
    };
}

TEST_CASE("Total mixture inlet preserves prescribed composition stagnation enthalpy and entropy")
{
    Fixture f;
    const auto fractions = f.Fractions(0);
    const P::Chemistry::ConstSpeciesBufferView Y{fractions.data(), f.species};
    const double totalT = 1400, totalP = 200000;
    C::DomainSettings settings;
    settings.equation.referenceScales = {{"rho0", f.scales.rho0}, {"U0", f.scales.U0}, {"L0", f.scales.L0}, {"T0", f.scales.T0}};
    std::vector<double> total{totalP, totalT, -1, 0, 0};
    total.insert(total.end(), fractions.begin(), fractions.end() - 1);
    settings.boundaries["7"].kind = "totalConditionInlet";
    settings.boundaries["7"].state = {{"type", "totalPT_phy"}, {"state", total}};
    const auto prepared = C::ResolvePhysicalStateInputs(settings, f.equation, f.workers);
    const C::BoundarySystem boundaries(prepared.settings.boundaries, f.equation);
    const C::BoundaryPointOperator boundary(f.equation);
    auto evaluator = f.equation.MakePointEvaluator(f.workers.Acquire(), {});
    for (const double speed : {.3, 100.})
    {
        const auto input = f.Point(0);
        std::vector<double> point(input.data(), input.data() + input.size());
        point[1] = point[0] * speed;
        point[2] = point[3] = 0;
        REQUIRE(boundary.TryEvaluate(point, boundaries.At(7), 0, {1, 0, 0}, {}, evaluator, {}, 0,
                                     C::BoundaryPointStateKind::ExteriorState));
        C::FlowState flow;
        std::copy_n(point.begin(), 5, flow.begin());
        const auto properties = evaluator.Evaluate(flow, {point.data() + 5, f.species - 1});
        const double t = properties.thermo.temperature * f.scales.T0;
        const double p = properties.thermo.pressure * f.scales.Pressure();
        const double kinetic = .5 * DNDS::sqr(point[1] / point[0]) * f.scales.SpecificEnergy();
        const double h0 = f.prototype.mixtureEnthalpy(totalT, Y, totalP);
        CHECK(f.prototype.mixtureEnthalpy(t, Y, p) + kinetic == doctest::Approx(h0).epsilon(2e-9));
        CHECK(f.prototype.mixtureEntropy(t, Y, p) == doctest::Approx(f.prototype.mixtureEntropy(totalT, Y, totalP)).epsilon(2e-9));
        CHECK(point[1] < 0);
        for (int k = 0; k < f.species - 1; ++k)
            CHECK(point[5 + k] / point[0] == doctest::Approx(fractions[k]).epsilon(1e-12));
        if (speed < 1)
            CHECK(point[1] / point[0] == doctest::Approx(-speed));
        else
            CHECK(std::abs(point[1] / point[0]) < speed);
    }
}

TEST_CASE("Shared transported species repair preserves Euler constants and strided neighbors")
{
    for (const std::array<double, 3> values : {std::array<double, 3>{-.01, .1, .2}, std::array<double, 3>{-.01, 2, 3}})
    {
        std::array<double, 8> storage{9, values[0], 8, values[1], 7, values[2], 6, 5};
        const auto before = storage;
        auto expected = values;
        for (auto &value : expected)
            value = std::max(value, 1e-30);
        double sum = 0;
        for (auto value : expected)
            sum += value;
        if (sum > 1 - 1e-14)
            for (auto &value : expected)
                value *= (1 - 1e-14) / sum;
        const R::RowSpan<double> row{storage.data() + 1, 3, 2};
        CHECK(P::Chemistry::RepairSpeciesDensities(1, row, 3));
        for (int k = 0; k < 3; ++k)
            CHECK(storage[2 * k + 1] == expected[k]);
        for (int k : {0, 2, 4, 6, 7})
            CHECK(storage[k] == before[k]);
    }
}

TEST_CASE("Constant-volume source preserves frozen blocks and matches the SI reactor at nonunit scales")
{
    Fixture fixture(true);
    P::RANS::ModelSettings model;
    model.model = "SA";
    const auto equation = fixture.equation.WithRANS(model);
    const I rows = mpi.rank == 0 ? 2 : 0;
    const A::DofSpace space(size_t(rows), 2, "constant-volume source with frozen RANS");
    A::DofBundle fields(space);
    fields.Add("packed", A::DofField<double>(space, R::RowLayout::TiledColMajor(0, 6, 32)));
    fields.Add("species", A::DofField<double>(space, R::RowLayout::ColMajor(0, fixture.species - 1, {16, 1})));
    C::EquationDofs state(equation.Schema(), std::move(fields), {{"flow", "packed", 1}, {"rans", "packed", 0}, {"species", "species", 0}});
    fixture.Fill(state);
    {
        auto write = state.WriteOwned();
        for (I iCell = 0; iCell < rows; ++iCell)
            write.Block("rans")(iCell, 0) = .013 + .001 * iCell;
    }
    auto result = state.Clone();
    auto parallelResult = state.Clone();
    constexpr double dt = .0003;
    const P::Chemistry::ReactorStepSettings controls;
    auto ghostWrite = state.PhysicalField(0).Ghost().Write();
    C::AdvanceChemicalSource(equation, state, result, fixture.workers.Acquire(), dt, controls, .7);
    P::Chemistry::ChemicalWorkerPool parallelWorkers(fixture.prototype, 2);
    C::AdvanceChemicalSource(equation, state, parallelResult, parallelWorkers, dt, controls, .7);
    ghostWrite.CancelUnmodified();
    {
        const auto read = state.ReadOwned(), output = result.ReadOwned(), parallel = parallelResult.ReadOwned();
        for (I iCell = 0; iCell < rows; ++iCell)
        {
            const auto point = fixture.Point(int(iCell) + mpi.rank, 1300 + iCell * 2);
            auto fractions = fixture.Fractions(int(iCell) + mpi.rank);
            const P::Chemistry::ConstSpeciesBufferView Y{fractions.data(), fixture.species};
            const double internal = (point[4] / point[0] - .5 * point.segment<3>(1).squaredNorm() / (point[0] * point[0])) * fixture.scales.SpecificEnergy();
            double temperature = fixture.prototype.temperatureFromUV(internal, 1 / (point[0] * fixture.scales.rho0), Y);
            // Independent SI conversion, not the new model's code-unit adapter.
            fixture.prototype.advanceConstVolume(temperature, point[0] * fixture.scales.rho0, {fractions.data(), fixture.species}, .7,
                                                 dt * fixture.scales.L0 / fixture.scales.U0,
                                                 controls.rtol, controls.atol, controls.maxOrder, controls.maxSteps);
            double change = 0;
            for (int k = 0; k < fixture.species - 1; ++k)
            {
                CHECK(output.Block("species")(iCell, k) == doctest::Approx(point[0] * fractions[k]).epsilon(2e-9).scale(1e-13));
                CHECK(parallel.Block("species")(iCell, k) == doctest::Approx(output.Block("species")(iCell, k)).epsilon(2e-11).scale(1e-13));
                change += std::abs(output.Block("species")(iCell, k) - read.Block("species")(iCell, k));
            }
            CHECK(change > 1e-8);
            for (const std::string name : {"flow", "rans"})
                for (I j = 0; j < read.Block(name).ComponentNumber(); ++j)
                    CHECK(output.Block(name)(iCell, j) == read.Block(name)(iCell, j));
        }
    }
    C::AdvanceChemicalSource(equation, state, result, fixture.workers.Acquire(), dt, controls, 0);
    {
        const auto read = state.ReadOwned(), output = result.ReadOwned();
        for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            for (I iCell = 0; iCell < rows; ++iCell)
                for (I j = 0; j < read.Block(iBlock).ComponentNumber(); ++j)
                    CHECK(read.Block(iBlock)(iCell, j) == output.Block(iBlock)(iCell, j));
    }
    CHECK_THROWS(C::AdvanceChemicalSource(equation, state, state, fixture.workers.Acquire(), dt, controls));
}
TEST_CASE("Equation primitive and mixture boundary producers preserve full energy composition and moving axes")
{
    Fixture fixture;
    auto evaluator = fixture.equation.MakePointEvaluator(fixture.workers.Acquire(), {});
    const auto expected = fixture.Point(5, 2100);
    const auto fractions = fixture.Fractions(5);
    const double pressure = expected[0] * fixture.scales.rho0 * fixture.prototype.mixtureR({fractions.data(), fixture.species}) * 2100 / fixture.scales.Pressure();
    const C::FlowState primitive{expected[0], .2, -.04, .03, pressure};
    const auto fromPressure = evaluator.FromPrimitive(primitive, {expected.data() + 5, fixture.species - 1});
    const auto fromTemperature = evaluator.FromPrimitive(primitive, {expected.data() + 5, fixture.species - 1}, 2100 / fixture.scales.T0);
    for (int j = 0; j < 5; ++j)
    {
        CHECK(fromPressure[j] == doctest::Approx(expected[j]).epsilon(2e-12));
        CHECK(fromTemperature[j] == doctest::Approx(expected[j]).epsilon(2e-12));
    }
    C::BoundarySettings input;
    input.kind = "farfield";
    input.primitive = primitive;
    for (int k = 0; k < fixture.species; ++k)
        input.massFractions[fixture.material.SpeciesNames()[k]] = fractions[k];
    C::BoundarySystem boundaries({{"7", input}}, fixture.equation);
    const auto &condition = boundaries.At(7);
    condition.ValidateEquationProducer();
    C::RigidMotionSettings motion;
    motion.angularVelocity = {0, 0, .3};
    motion.translationVelocity = {.07, -.02, .01};
    const auto frame = C::RigidMotion(motion, 3).At(.4);
    const C::Vector3 normal{.6, .8, 0}, location{.2, -.3, .1};
    const auto grid = frame.GridVelocity(location);
    for (const double mach : {-2., -.2, .2, 2.})
    {
        const auto initial = fixture.Point(0);
        std::vector<double> rhoY(size_t(fixture.species - 1));
        for (int k = 0; k < fixture.species - 1; ++k)
            rhoY[k] = initial[5 + k];
        C::FlowState inside;
        for (int j = 0; j < 5; ++j)
            inside[j] = initial[j];
        const auto properties = evaluator.Evaluate(inside, {rhoY.data(), fixture.species - 1});
        const double speed = mach * properties.thermo.soundSpeed;
        // Retain the independent oracle's thermal energy while changing all
        // momenta to force the chosen characteristic branch in moving axes.
        inside[4] -= .5 * (inside[1] * inside[1] + inside[2] * inside[2] + inside[3] * inside[3]) / inside[0];
        for (int d = 0; d < 3; ++d)
        {
            const double velocity = grid[d] + speed * normal[d];
            inside[1 + d] = inside[0] * velocity;
            inside[4] += .5 * inside[0] * velocity * velocity;
        }
        const auto value = condition.Exterior(evaluator, inside, {rhoY.data(), fixture.species - 1}, normal, location, .4,
                                              DNDS::OptionalRef<const C::RigidFrame>(frame));
        const auto point = evaluator.Evaluate(value, {rhoY.data(), fixture.species - 1});
        const bool inflow = mach < 0;
        const double rho = inflow ? primitive[0] : inside[0];
        const auto prescribedVelocity = frame.LocalVector({primitive[1], primitive[2], primitive[3]});
        CHECK(value[0] == rho);
        CHECK(point.thermo.pressure == doctest::Approx(mach < -1 || (mach > 0 && mach < 1) ? pressure : properties.thermo.pressure).epsilon(3e-12));
        for (int d = 0; d < 3; ++d)
            CHECK(value[1 + d] / rho == doctest::Approx(inflow ? prescribedVelocity[d] : inside[1 + d] / inside[0]).epsilon(3e-12));
        for (int k = 0; k < fixture.species - 1; ++k)
            CHECK(rhoY[k] / rho == doctest::Approx(inflow ? fractions[k] : initial[5 + k] / initial[0]).epsilon(3e-12));
    }
    input.kind = "noSlipWall";
    input.massFractions.clear();
    input.thermal = "isothermal";
    input.temperature = 600 / fixture.scales.T0;
    input.velocity = {.01, -.02, .03};
    C::BoundarySystem walls({{"7", input}}, fixture.equation);
    auto rhoY = std::vector<double>(expected.data() + 5, expected.data() + fixture.components);
    const auto before = rhoY;
    const auto face = walls.At(7).FaceValue(evaluator, fromPressure, {rhoY.data(), fixture.species - 1}, normal, location, .4,
                                            DNDS::OptionalRef<const C::RigidFrame>(frame));
    const auto faceProperties = evaluator.Evaluate(face, {rhoY.data(), fixture.species - 1});
    CHECK(faceProperties.thermo.temperature == doctest::Approx(input.temperature).epsilon(3e-12));
    CHECK(faceProperties.thermo.pressure == doctest::Approx(pressure).epsilon(3e-12));
    const auto wallVelocity = walls.At(7).WallVelocityAt(
        location, DNDS::OptionalRef<const C::RigidFrame>(frame));
    for (int d = 0; d < 3; ++d)
        CHECK(face[1 + d] / face[0] == doctest::Approx(wallVelocity[d]).epsilon(3e-12));
    for (int k = 0; k < fixture.species - 1; ++k)
        CHECK(rhoY[k] / face[0] == doctest::Approx(before[k] / expected[0]).epsilon(3e-12));
    rhoY = before;
    const auto exterior = walls.At(7).Exterior(evaluator, fromPressure, {rhoY.data(), fixture.species - 1}, normal, location, .4,
                                               DNDS::OptionalRef<const C::RigidFrame>(frame));
    const auto exteriorProperties = evaluator.Evaluate(exterior, {rhoY.data(), fixture.species - 1});
    // Isothermal exterior uses prescribed temperature and interior pressure,
    // with velocity mirrored about the wall. Density and rhoY change together;
    // it is mass fraction, not conservative species density, that is retained.
    CHECK(exteriorProperties.thermo.temperature == doctest::Approx(input.temperature).epsilon(3e-12));
    CHECK(exteriorProperties.thermo.pressure == doctest::Approx(pressure).epsilon(3e-12));
    for (int k = 0; k < fixture.species - 1; ++k)
        CHECK(rhoY[k] / exterior[0] == doctest::Approx(before[k] / fromPressure[0]).epsilon(3e-12));
    CHECK_THROWS(fixture.equation.IndependentMassFractions({{"missing-species", .1}}));
    CHECK_THROWS(fixture.equation.IndependentMassFractions({{fixture.material.SpeciesNames().back(), .2}}));
}
TEST_CASE("Mixture thermal references enforce a requested temperature through actual conservative energy")
{
    Fixture fixture;
    constexpr double temperature = 900;
    const double codeTemperature = temperature / fixture.scales.T0;
    const C::ThermalPolicy policy(fixture.equation, 1e-12, codeTemperature, 1e-12, 1e-12, fixture.workers.Acquire());
    const auto bounds = policy.View();
    for (int sample : {0, 5})
    {
        const auto cold = fixture.Point(sample, temperature), hot = fixture.Point(sample, 1400);
        C::FlowState flow;
        for (int j = 0; j < 5; ++j)
            flow[j] = cold[j];
        CHECK(bounds.Reference(cold, 1) == doctest::Approx(C::PositivityDetail::InternalEnergy(flow)).epsilon(1e-12));
        CHECK(bounds.Reference(hot, 0) == doctest::Approx(fixture.material.BaseInternalEnergy(hot[0], {hot.data() + 5, fixture.species - 1})).epsilon(1e-14));
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(fixture.components);
        delta[0] = -.2 * hot[0];
        delta[4] = -3 * (hot[4] - cold[4]);
        delta[5] = .005;
        const double alpha = bounds.IncrementFraction(hot, delta, .5, 2, 1);
        CHECK(alpha > 0);
        CHECK(alpha < 1);
        delta *= alpha;
        const double fix = bounds.FixFraction(hot, delta);
        CHECK(std::isfinite(fix));
        const Eigen::VectorXd candidate = hot + fix * delta;
        CHECK(bounds.Admissible(candidate));
        auto evaluator = fixture.equation.MakePointEvaluator(fixture.workers.Acquire(), {});
        for (int j = 0; j < 5; ++j)
            flow[j] = candidate[j];
        const auto point = evaluator.Evaluate(flow, {candidate.data() + 5, fixture.species - 1});
        CHECK(point.thermo.temperature >= codeTemperature);

        // Final mean repair owns the stored composition: repair the exact
        // target first, then line-search that target without another species
        // projection. On this simplex segment the raw and clipped references
        // are the same invariant seen by the EOS/T preparation.
        auto repairedTarget = hot;
        repairedTarget[5] = -0.25 * hot[0];
        repairedTarget[6] = 1.75 * hot[0];
        P::Chemistry::RepairSpeciesDensities(
            repairedTarget[0], repairedTarget.segment(5, fixture.species - 1),
            fixture.species - 1);
        for (int c = 0; c < 2; ++c)
            CHECK(bounds.Reference(repairedTarget, c) ==
                  doctest::Approx(bounds.ClippedReference(repairedTarget, c)).epsilon(1e-13));
        const double kinetic = .5 * repairedTarget.segment<3>(1).squaredNorm() / repairedTarget[0];
        double required = -std::numeric_limits<double>::infinity();
        for (int c = 0; c < 2; ++c)
            required = std::max(required, bounds.Reference(repairedTarget, c) + bounds.floors[c]);
        repairedTarget[4] = kinetic + required - std::max(1e-12, std::abs(required) * 1e-4);
        REQUIRE_FALSE(bounds.RawThermallyAdmissible(repairedTarget));
        const auto repairedIncrement = repairedTarget - hot;
        const double repairedFraction = bounds.FixFraction(hot, repairedIncrement);
        CHECK(repairedFraction > 0);
        CHECK(repairedFraction < 1);
        CHECK(bounds.Admissible(hot + repairedFraction * repairedIncrement));
    }
    auto one = fixture.Point(0), two = fixture.Point(5);
    const Eigen::VectorXd middle = .3 * one + .7 * two;
    for (int c = 0; c < 2; ++c)
    {
        CHECK(bounds.Reference(middle, c) == doctest::Approx(.3 * bounds.Reference(one, c) + .7 * bounds.Reference(two, c)).epsilon(1e-14));
        CHECK(bounds.ClippedReference(one, c) == doctest::Approx(bounds.Reference(one, c)).epsilon(1e-14));
    }

    // A nonconstant polynomial base is selected by the locked conservative
    // flow/species thermal condition, not by a raw pointwise simplex test.
    // Property evaluation may repair this temporary composition; the mean
    // remains separately simplex-valid.
    Eigen::MatrixXd base(2, fixture.components), higher(2, fixture.components);
    base.row(0) = base.row(1) = one.transpose();
    higher = base;
    higher(0, 5) = -one[0] * 1e-12;
    const C::ThermalAdmissibilityDetail::PointRow<Eigen::MatrixXd> speciesOnly{higher, 0};
    CHECK(bounds.ClippedReference(speciesOnly, 0) != doctest::Approx(bounds.Reference(speciesOnly, 0)).epsilon(1e-14));
    CHECK_FALSE(bounds.Admissible(speciesOnly));
    CHECK(bounds.Admissible(speciesOnly, false));
    CHECK(bounds.ReconstructionFraction(one, base, higher, 0, 2, 1, 0) == 1);
    {
        C::FlowState rawFlow;
        for (int j = 0; j < 5; ++j)
            rawFlow[j] = higher(0, j);
        std::vector<double> rawSpecies(size_t(fixture.species - 1));
        for (int k = 0; k < fixture.species - 1; ++k)
            rawSpecies[size_t(k)] = higher(0, 5 + k);
        const auto before = rawSpecies;
        auto evaluator = fixture.equation.MakePointEvaluator(fixture.workers.Acquire(), {});
        const auto properties = evaluator.Evaluate(
            rawFlow, {rawSpecies.data(), fixture.species - 1});
        CHECK(std::isfinite(properties.thermo.temperature));
        CHECK(rawSpecies == before);
        CHECK(higher(0, 5) < 0);
    }

    // The analytic reconstruction beta uses raw affine energy, while the
    // physical endpoint uses the clipped composition seen by the T solver.
    // Find an out-of-simplex direction whose clipped reference is stricter;
    // compatibility drops to the base, while the optional policy recovers an
    // intermediate common flow/species beta without modifying the raw point.
    Eigen::VectorXd clippedFailure;
    bool found = false;
    for (int k = 0; k < fixture.species - 1 && !found; ++k)
        for (double multiple : {-2., 3.})
        {
            auto trial = one;
            trial[5 + k] = multiple * one[0];
            double rawRequired = -std::numeric_limits<double>::infinity();
            double clippedRequired = -std::numeric_limits<double>::infinity();
            for (int c = 0; c < 2; ++c)
            {
                rawRequired = std::max(rawRequired, bounds.floors[c] + bounds.Reference(trial, c));
                clippedRequired = std::max(clippedRequired, bounds.floors[c] + bounds.ClippedReference(trial, c));
            }
            if (clippedRequired > rawRequired + 1e-10 * std::max(1., std::abs(rawRequired)))
            {
                const double kinetic = .5 * trial.segment<3>(1).squaredNorm() / trial[0];
                trial[4] = kinetic + .5 * (rawRequired + clippedRequired);
                clippedFailure = std::move(trial);
                found = true;
                break;
            }
        }
    REQUIRE(found);
    CHECK(bounds.RawThermallyAdmissible(clippedFailure));
    CHECK_FALSE(bounds.ClippedReferenceAdmissible(clippedFailure));
    Eigen::MatrixXd clippedHigh(1, fixture.components), clippedBase(1, fixture.components);
    clippedHigh.row(0) = clippedFailure.transpose();
    clippedBase.row(0) = one.transpose();
    const double fallback = bounds.ReconstructionFraction(
        one, clippedBase, clippedHigh, 0, 1, 1, 0, 1,
        C::ReconstructionClippedReferencePolicy::FallbackToBase);
    const double searched = bounds.ReconstructionFraction(
        one, clippedBase, clippedHigh, 0, 1, 1, 0, 1,
        C::ReconstructionClippedReferencePolicy::Backtracking);
    CHECK(fallback == 0);
    CHECK(searched > 0);
    CHECK(searched < 1);
    const Eigen::VectorXd searchedPoint = one + searched * (clippedFailure - one);
    CHECK(bounds.ClippedReferenceAdmissible(searchedPoint));
}
TEST_CASE("Attached RANS leaves mixture thermal bounds and chemical Jacobian columns unchanged")
{
    Fixture fixture(true);
    P::RANS::ModelSettings model;
    model.model = "KOmegaSST";
    const auto equation = fixture.equation.WithRANS(model);
    CHECK(equation.Schema().blocks[1].name == "rans");
    CHECK(equation.Schema().blocks[2].name == "species");
    const I rows = mpi.rank == 0 ? 2 : 0;
    A::DofSpace space(size_t(rows), 0, "RANS plus chemical point source");
    auto base = fixture.State(space, 1), baseSource = fixture.State(space, 1);
    fixture.Fill(base);
    A::DofBundle fields(space);
    fields.Add("flow", A::DofField<double>(space, R::RowLayout::ColMajor(0, 5, {16, 1})));
    fields.Add("rans", A::DofField<double>(space, R::RowLayout::Dense(0, 2)));
    fields.Add("species", A::DofField<double>(space, R::RowLayout::TiledColMajor(0, fixture.species - 1, 32)));
    auto state = C::EquationDofs::Separate(equation.Schema(), std::move(fields));
    fixture.Fill(state);
    {
        auto write = state.WriteOwned();
        const auto rans = write.Block("rans");
        for (I iCell = 0; iCell < rows; ++iCell)
        {
            rans(iCell, 0) = .03;
            rans(iCell, 1) = 2.;
        }
    }
    auto source = state.Clone();
    const I width = state.ComponentNumber();
    R::Array<double> jac(R::RowLayout::Dense(rows, width * width));
    R::Array<double> baseJac(R::RowLayout::Dense(rows, fixture.components * fixture.components));
    C::EvaluateChemicalSource(fixture.equation, base, baseSource, fixture.workers.Acquire(), std::ref(baseJac));
    C::EvaluateChemicalSource(equation, state, source, fixture.workers.Acquire(), std::ref(jac));
    C::ThermalPolicy bounds(equation, 1e-10, .5, 0., 1e-10, fixture.workers.Acquire());
    C::ThermalPolicy baseBounds(fixture.equation, 1e-10, .5, 0., 1e-10, fixture.workers.Acquire());
    const auto input = state.ReadOwned(), output = source.ReadOwned(), baseInput = base.ReadOwned(), baseOutput = baseSource.ReadOwned();
    const auto inputLayout = input.ComponentLayout(), outputLayout = output.ComponentLayout(),
               baseInputLayout = baseInput.ComponentLayout(), baseOutputLayout = baseOutput.ComponentLayout();
    const auto u = inputLayout.View(), s = outputLayout.View(), ub = baseInputLayout.View(), sb = baseOutputLayout.View();
    const auto matrixRead = jac.Read(), baseMatrixRead = baseJac.Read();
    const auto m = matrixRead.View(), mb = baseMatrixRead.View();
    const auto mapped = [](I j)
    { return j < 5 ? j : j + 2; };
    for (I iCell = 0; iCell < rows; ++iCell)
    {
        for (int constraint = 0; constraint < 2; ++constraint)
            CHECK(bounds.View().Sensible(C::ThermalAdmissibilityDetail::PointRow<decltype(u)>{u, iCell}, constraint) ==
                  doctest::Approx(baseBounds.View().Sensible(C::ThermalAdmissibilityDetail::PointRow<decltype(ub)>{ub, iCell}, constraint)).epsilon(1e-13));
        for (I j = 0; j < fixture.components; ++j)
        {
            CHECK(s(iCell, mapped(j)) == doctest::Approx(sb(iCell, j)).epsilon(1e-11));
            for (I k = 0; k < fixture.components; ++k)
                CHECK(m(iCell, mapped(j) + width * mapped(k)) == doctest::Approx(mb(iCell, j + fixture.components * k)).epsilon(1e-8).scale(1e-10));
        }
        for (I j : {I(5), I(6)})
        {
            CHECK(s(iCell, j) == 0);
            for (I k = 0; k < width; ++k)
            {
                CHECK(m(iCell, j + width * k) == 0);
                CHECK(m(iCell, k + width * j) == 0);
            }
        }
    }
}

TEST_CASE("Mixture source point matches the Euler source and all conservative derivative columns")
{
    Fixture fixture;
    auto evaluate = fixture.equation.MakePointEvaluator(fixture.workers.Acquire(), {false, false, false, true, true});
    const auto u = fixture.Point(2);
    const auto acousticOnly = evaluate.Evaluate({u[0], u[1], u[2], u[3], u[4]}, {u.data() + 5, fixture.species - 1}, 0, {});
    CHECK(acousticOnly.chemicalSource.nSpecies == 0);
    CHECK(acousticOnly.sourceJacobian.data == nullptr);
    auto point = evaluate.Evaluate({u[0], u[1], u[2], u[3], u[4]}, {u.data() + 5, fixture.species - 1}, 900 / fixture.scales.T0);
    Eigen::MatrixXd derivative(fixture.species - 1, fixture.components);
    for (int k = 0; k < derivative.rows(); ++k)
        for (int j = 0; j < derivative.cols(); ++j)
            derivative(k, j) = point.sourceJacobian(k, j);

    using Traits = DNDS::Euler::EulerModelTraits<DNDS::Euler::NS_EX_3D>;
    auto legacyPool = std::make_shared<std::vector<P::Chemistry::ChemicalSource>>();
    legacyPool->push_back(std::move(*fixture.prototype.clone()));
    DNDS::Euler::EulerEvaluatorSettings<DNDS::Euler::NS_EX_3D>::IdealGasProperty settings;
    settings.rho0 = fixture.scales.rho0;
    settings.U0 = fixture.scales.U0;
    settings.L0 = fixture.scales.L0;
    settings.T0 = fixture.scales.T0;
    // Preserve two untouched RANS slots between energy and species in Euler.
    const int nVars = fixture.components + 2;
    DNDS::Euler::ChemicalContributor<DNDS::Euler::NS_EX_3D> legacy(legacyPool, settings, .7, nVars);
    Traits::TU oldU(nVars), source(nVars);
    oldU.head<5>() = u.head<5>();
    oldU.segment<2>(5).setConstant(.12);
    oldU.tail(fixture.species - 1) = u.tail(fixture.species - 1);
    Traits::TJacobianU jacobian(nVars, nVars);
    Traits::TDiffU gradient(3, nVars);
    gradient.setZero();
    DNDS::Euler::SourceCellAux aux;
    aux.T = point.thermo.temperature;
    aux.p = point.thermo.pressure;
    aux.pPhys = aux.p * fixture.scales.Pressure();
    aux.reactiveSplitCoupledScale = .6;
    source.setZero();
    jacobian.setZero();
    legacy.evaluate(source, jacobian, oldU, gradient, {}, aux, 0, 0, 2);
    CHECK(source.head<7>().isZero(0));
    CHECK(jacobian.topRows(7).isZero(0));
    CHECK(jacobian.middleCols(5, 2).isZero(0));
    for (int k = 0; k < fixture.species - 1; ++k)
    {
        CHECK(source[7 + k] == doctest::Approx(.42 * point.chemicalSource[k]).epsilon(5e-10).scale(1e-9));
        for (int j = 0; j < fixture.components; ++j)
            CHECK(jacobian(7 + k, j < 5 ? j : j + 2) == doctest::Approx(-.42 * derivative(k, j)).epsilon(5e-10).scale(1e-9));
    }
    // An independent conservative perturbation includes UV/composition chain
    // rules, not merely a fixed-temperature kinetics derivative.
    for (int j = 0; j < fixture.components; ++j)
    {
        CAPTURE(j);
        const double h = 1e-6 * std::max(std::abs(u[j]), 1e-3);
        auto sample = [&](double sign)
        {
            Eigen::VectorXd shifted = u;
            shifted[j] += sign * h;
            const auto value = evaluate.Evaluate({shifted[0], shifted[1], shifted[2], shifted[3], shifted[4]}, {shifted.data() + 5, fixture.species - 1});
            Eigen::VectorXd result(fixture.species - 1);
            for (int k = 0; k < result.size(); ++k)
                result[k] = value.chemicalSource[k];
            return result;
        };
        const Eigen::VectorXd finiteDifference = (sample(1) - sample(-1)) / (2 * h);
        CHECK((finiteDifference - derivative.col(j)).lpNorm<Eigen::Infinity>() <= 3e-5 * std::max(1., derivative.col(j).lpNorm<Eigen::Infinity>()));
    }
}

TEST_CASE("Mixture physical ALE flux derivatives include composition energy and all conservative columns")
{
    Fixture fixture;
    auto evaluator = fixture.equation.MakePointEvaluator(fixture.workers.Acquire(), {false, false, true});
    const auto u = fixture.Point(2);
    const C::FlowState flow{u[0], u[1], u[2], u[3], u[4]};
    const C::Vector3 normal{.6, 0, .8}, grid{.12, -.04, .07};
    for (int j = 0; j < fixture.components; ++j)
    {
        const auto point = evaluator.Evaluate(flow, {u.data() + 5, fixture.species - 1});
        C::FlowState direction{};
        if (j < 5)
            direction[j] = 1;
        struct SpeciesDirection
        {
            int column;
            double operator[](int k) const { return column == 5 + k ? 1 : 0; }
        } speciesDirection{j};
        const auto primitive = C::PreparePrimitiveDerivative(flow, direction, point, speciesDirection);
        const auto derivative = C::PrepareCompressibleFluxDerivative(flow, direction, point.thermo.pressure, primitive.pressure, normal, grid);
        double speed = 0, deltaSpeed = 0;
        for (int d = 0; d < 3; ++d)
        {
            speed += (flow[1 + d] / flow[0] - grid[d]) * normal[d];
            deltaSpeed += primitive.rhoVelocityTemperature[1 + d] * normal[d];
        }
        Eigen::VectorXd expected(fixture.components);
        for (int i = 0; i < 5; ++i)
            expected[i] = derivative[i];
        for (int k = 0; k < fixture.species - 1; ++k)
            expected[5 + k] = C::PrepareAdvectedScalarFluxDerivative(u[5 + k], j == 5 + k ? 1 : 0, speed, deltaSpeed);
        const double h = 2e-6 * std::max(std::abs(u[j]), 1e-3);
        const auto sample = [&](double sign)
        {
            Eigen::VectorXd shifted = u;
            shifted[j] += sign * h;
            const C::FlowState state{shifted[0], shifted[1], shifted[2], shifted[3], shifted[4]};
            const auto properties = evaluator.Evaluate(state, {shifted.data() + 5, fixture.species - 1});
            const auto flux = C::Riemann::PhysicalFlux({state, properties.thermo.pressure}, normal, grid);
            Eigen::VectorXd result(fixture.components);
            for (int i = 0; i < 5; ++i)
                result[i] = flux[i];
            double q = 0;
            for (int d = 0; d < 3; ++d)
                q += (state[1 + d] / state[0] - grid[d]) * normal[d];
            for (int k = 0; k < fixture.species - 1; ++k)
                result[5 + k] = shifted[5 + k] * q;
            return result;
        };
        const Eigen::VectorXd actual = (sample(1) - sample(-1)) / (2 * h);
        CAPTURE(j);
        CHECK((actual - expected).lpNorm<Eigen::Infinity>() < 2e-6 * std::max(1., expected.lpNorm<Eigen::Infinity>()));
    }
}
TEST_CASE("Owned chemical sources honor separate layouts, stale temperature guesses and ghost independence")
{
    Fixture fixture;
    const I rows = mpi.size > 1 && mpi.rank == mpi.size - 1 ? 0 : 35;
    A::DofSpace space(size_t(rows), 2, "local chemical sources");
    std::vector<double> canonicalMatrix(size_t(rows) * fixture.components * fixture.components);
    for (int layout = 0; layout < 3; ++layout)
    {
        CAPTURE(layout);
        auto state = fixture.State(space, layout), source = fixture.State(space, (layout + 1) % 3);
        fixture.Fill(state);
        C::DerivedFields oldTemperature(space, {C::DerivedQuantity::Temperature});
        C::RefreshDerivedFields(state, fixture.equation, oldTemperature, fixture.workers.Acquire());
        // Deliberately invalidate freshness without changing values.
        {
            auto write = state.WriteOwned();
        }
        CHECK_FALSE(oldTemperature.Current(state, fixture.equation));
        auto activeGhost = state.PhysicalField(0).Ghost().Write();
        R::Array<double> jacobian(R::RowLayout::ColMajor(rows, fixture.components * fixture.components, {16, 1}));
        C::EvaluateChemicalSource(fixture.equation, state, source, fixture.workers.Acquire(), std::ref(jacobian), .7, std::cref(oldTemperature));
        activeGhost.CancelUnmodified();
        const auto read = source.ReadOwned();
        const auto flow = read.Block("flow").owned, speciesValues = read.Block("species").owned;
        const auto matrixRead = jacobian.Read();
        const auto matrix = matrixRead.View<R::UniformTableRows>();
        const auto guessRead = oldTemperature.ReadTemperatureGuess(state, fixture.equation);
        const auto oldT = guessRead.Field(C::DerivedQuantity::Temperature);
        double relativeJacobianError = 0, observedWorst = 0, expectedWorst = 0, layoutError = 0;
        for (I iCell = 0; iCell < rows; ++iCell)
        {
            std::vector<double> fractions(fixture.species);
            const double temperature = 1300 + iCell * 2;
            const auto point = fixture.Point(int(iCell) + mpi.rank, temperature);
            P::Chemistry::RepairMassFractions(point[0], {point.data() + 5, fixture.species - 1}, {fractions.data(), fixture.species});
            const P::Chemistry::ConstSpeciesBufferView Y{fractions.data(), fixture.species};
            double speedSquared = 0;
            for (int d = 0; d < 3; ++d)
                speedSquared += DNDS::sqr(point[1 + d] / point[0]);
            // Compare derivatives at the same recovered conservative state,
            // rather than the analytic generator's rounded temperature. The
            // independent point test above also checks all columns by FD.
            const double recoveredT = fixture.prototype.temperatureFromUV((point[4] / point[0] - .5 * speedSquared) * fixture.scales.SpecificEnergy(),
                                                                          1 / (point[0] * fixture.scales.rho0), Y, oldT(iCell, 0) * fixture.scales.T0);
            CHECK(recoveredT == doctest::Approx(temperature).epsilon(1e-11));
            P::Thermodynamics::Properties thermo;
            const double pressure = point[0] * fixture.scales.rho0 * fixture.prototype.mixtureR(Y) * recoveredT;
            fixture.prototype.EvaluateProperties(recoveredT, pressure, Y, thermo);
            const double sourceT = thermo.temperature / fixture.scales.T0 * fixture.scales.T0;
            const double sourceP = thermo.pressure / fixture.scales.Pressure() * fixture.scales.Pressure();
            std::vector<double> rates(fixture.species), rawJacobian(fixture.species * fixture.components);
            P::Chemistry::JacobianBufferView expected{rawJacobian.data(), fixture.species, fixture.components, fixture.species};
            fixture.prototype.productionRatesAndJacobian(sourceT, sourceP, point[0], point[4], point[1], point[2], point[3], 4, Y, {rates.data(), fixture.species}, expected);
            for (int j = 0; j < 5; ++j)
            {
                CHECK(flow(iCell, j) == 0);
                for (int column = 0; column < fixture.components; ++column)
                    CHECK(matrix(iCell, j + fixture.components * column) == 0);
            }
            for (int k = 0; k < fixture.species - 1; ++k)
            {
                const double scaling = .7 * fixture.prototype.molecularWeights()[k] / fixture.scales.SourceRate();
                CHECK(speciesValues(iCell, k) == doctest::Approx(scaling * rates[k]).epsilon(5e-10).scale(1e-9));
                for (int j = 0; j < fixture.components; ++j)
                {
                    const double observed = matrix(iCell, 5 + k + fixture.components * j), reference = scaling * expected(k, j);
                    const double error = std::abs(observed - reference) / (1e-9 + std::max(std::abs(observed), std::abs(reference)));
                    if (error > relativeJacobianError)
                    {
                        relativeJacobianError = error;
                        observedWorst = observed;
                        expectedWorst = reference;
                    }
                }
            }
            for (int j = 0; j < fixture.components * fixture.components; ++j)
            {
                const auto offset = size_t(iCell) * fixture.components * fixture.components + j;
                if (layout == 0)
                    canonicalMatrix[offset] = matrix(iCell, j);
                else
                    layoutError = std::max(layoutError, std::abs(matrix(iCell, j) - canonicalMatrix[offset]));
            }
        }
        INFO(std::setprecision(17) << "worst observed=" << observedWorst << ", expected=" << expectedWorst);
        CAPTURE(relativeJacobianError);
        // Cantera's numerical rate derivatives vary under roundoff in the
        // independent UV/property recovery. The full conservative FD test
        // above supplies a second physical oracle. Layouts themselves must
        // produce exactly the same complete block, including zero flow rows.
        CHECK(relativeJacobianError < 1e-6);
        CHECK(layoutError == 0);
    }
}

TEST_CASE("Mixture primitive and conservative gradients include composition energy and all kinetic terms")
{
    namespace T = P::Thermodynamics;
    Fixture fixture;
    for (const double temperature : {400., 1400.})
    {
        CAPTURE(temperature);
        const auto u = fixture.Point(0, temperature);
        const C::FlowState flow{u[0], u[1], u[2], u[3], u[4]};
        const auto velocity = T::ConservativeVelocity<3>(flow);
        {
            auto materialPoint = fixture.material.Borrow(fixture.workers.Acquire());
            const auto rhoT = materialPoint.FromRhoT(flow[0], temperature / fixture.scales.T0, {u.data() + 5, fixture.species - 1});
            CHECK(T::ConservativeTotalEnergyDensity<3>(flow[0], velocity, rhoT.thermo.internalEnergy) == doctest::Approx(flow[4]).epsilon(2e-12));
            const auto rhoP = materialPoint.FromRhoP(flow[0], rhoT.thermo.pressure, {u.data() + 5, fixture.species - 1});
            CHECK(rhoP.thermo.temperature == doctest::Approx(temperature / fixture.scales.T0).epsilon(2e-12));
            CHECK(T::ConservativeSpecificInternalEnergy<3>(flow) == doctest::Approx(rhoP.thermo.internalEnergy).epsilon(2e-12));
        }
        if (temperature == 400)
            CHECK(T::ConservativeSpecificInternalEnergy<3>(flow) < 0);
        auto evaluate = fixture.equation.MakePointEvaluator(fixture.workers.Acquire(), {false, false, true});
        const auto point = evaluate.Evaluate(flow, {u.data() + 5, fixture.species - 1});
        const C::FlowState direction{.002, -.001, .003, -.004, .007};
        std::vector<double> speciesDerivative(size_t(2 * (fixture.species - 1)), 777);
        for (int k = 0; k < fixture.species - 1; ++k)
            speciesDerivative[2 * k] = .0001 * (k - 4);
        const R::RowSpan<const double> dSpecies{speciesDerivative.data(), fixture.species - 1, 2};
        const auto derivative = C::PreparePrimitiveDerivative(flow, direction, point, dSpecies);
        constexpr double step = 1e-4;
        const auto sample = [&](double sign)
        {
            Eigen::VectorXd shifted = u;
            for (int j = 0; j < 5; ++j)
                shifted[j] += sign * step * direction[j];
            for (int k = 0; k < fixture.species - 1; ++k)
                shifted[5 + k] += sign * step * dSpecies[k];
            const auto property = evaluate.Evaluate({shifted[0], shifted[1], shifted[2], shifted[3], shifted[4]},
                                                    {shifted.data() + 5, fixture.species - 1}, temperature / fixture.scales.T0);
            return std::array<double, 6>{shifted[0], shifted[1] / shifted[0], shifted[2] / shifted[0], shifted[3] / shifted[0],
                                         property.thermo.temperature, property.thermo.pressure};
        };
        const auto plus = sample(1), minus = sample(-1);
        for (int j = 0; j < 5; ++j)
            CHECK(derivative.rhoVelocityTemperature[j] == doctest::Approx((plus[j] - minus[j]) / (2 * step)).epsilon(3e-6).scale(1e-6));
        CHECK(derivative.pressure == doctest::Approx((plus[5] - minus[5]) / (2 * step)).epsilon(3e-6).scale(1e-6));
    }
}

TEST_CASE("Mixture viscous transport matches Euler heat and species enthalpy flux with effective coefficients")
{
    Fixture fixture(true);
    const auto u = fixture.Point(1);
    const C::FlowState flow{u[0], u[1], u[2], u[3], u[4]};
    auto evaluator = fixture.equation.MakePointEvaluator(fixture.workers.Acquire(), {true, true, true});
    const auto point = evaluator.Evaluate(flow, {u.data() + 5, fixture.species - 1});
    C::FlowGradient gradient;
    Eigen::MatrixXd speciesGradient(fixture.species - 1, 3);
    Eigen::Matrix<double, 3, Eigen::Dynamic> oldGradient(3, fixture.components);
    for (int d = 0; d < 3; ++d)
    {
        for (int j = 0; j < 5; ++j)
            oldGradient(d, j) = gradient[j][d] = .002 * (j + 1) * (d + 1);
        for (int k = 0; k < fixture.species - 1; ++k)
            oldGradient(d, 5 + k) = speciesGradient(k, d) = .0001 * (k - 4) * (d + 1);
    }
    auto legacyPool = std::make_shared<std::vector<P::Chemistry::ChemicalSource>>();
    legacyPool->push_back(std::move(*fixture.prototype.clone()));
    DNDS::Euler::EulerEvaluatorSettings<DNDS::Euler::NS_EX_3D>::IdealGasProperty settings;
    settings.rho0 = fixture.scales.rho0;
    settings.U0 = fixture.scales.U0;
    settings.L0 = fixture.scales.L0;
    settings.T0 = fixture.scales.T0;
    DNDS::Euler::PhysicsProperties<DNDS::Euler::NS_EX_3D> legacy(settings);
    legacy.setChemicalSourcePool(legacyPool);
    const auto thermal = legacy.conservativeThermal(u, point.thermo.temperature, 1e-12);
    Eigen::Matrix<double, 3, Eigen::Dynamic> primitiveGradient;
    DNDS::Euler::Gas::GradientCons2Prim_IdealGas<3>(u, oldGradient, primitiveGradient, thermal.gammaEq, legacy.mixtureBaseInternalRhoESpecies());
    C::ViscousFluxScratch scratch(fixture.species);
    const C::Vector3 normal{2. / 3, -1. / 3, 2. / 3};
    for (bool wall : {false, true})
    {
        CAPTURE(wall);
        C::ViscousFluxOptions options;
        options.adiabatic = options.impermeable = wall;
        options.turbulentViscosity = .3 * point.transport.viscosity;
        options.turbulentConductivity = .2 * point.transport.conductivity;
        options.turbulentSpeciesDiffusivity = 1e-7;
        std::vector<double> output(size_t(2 * (fixture.species - 1)), 777);
        R::RowSpan<double> speciesFlux{output.data(), fixture.species - 1, 2};
        const P::Chemistry::ConstSpeciesBufferView rawRhoY{u.data() + 5, fixture.species - 1};
        const auto flux = C::PrepareViscousFlux(flow, gradient, point, rawRhoY, speciesGradient, normal, speciesFlux, scratch, options);
        // Legacy flow slots overwrite, but species diffusion is additive;
        // its production caller zeroes those slots before invoking the helper.
        Eigen::VectorXd expected = Eigen::VectorXd::Zero(fixture.components);
        DNDS::Euler::PhysicsProperties<DNDS::Euler::NS_EX_3D>::MixtureAveragedDiffusionBuffers buffers;
        const Eigen::Vector3d oldNormal(normal[0], normal[1], normal[2]);
        const double totalViscosity = point.transport.viscosity + options.turbulentViscosity;
        legacy.viscousFluxIdealGas(thermal.T, thermal.p, u, oldGradient, primitiveGradient, oldNormal, wall, wall,
                                   thermal.gammaEq, thermal.gamma, totalViscosity, options.turbulentViscosity / totalViscosity, false,
                                   point.transport.conductivity + options.turbulentConductivity, point.thermo.cp,
                                   options.turbulentSpeciesDiffusivity, buffers, expected);
        for (int j = 0; j < 5; ++j)
            CHECK(flux[j] == doctest::Approx(expected[j]).epsilon(3e-10).scale(1e-20));
        for (int k = 0; k < fixture.species - 1; ++k)
        {
            CHECK(speciesFlux[k] == doctest::Approx(expected[5 + k]).epsilon(3e-10).scale(1e-20));
            CHECK(output[2 * k + 1] == 777);
            if (wall)
                CHECK(speciesFlux[k] == 0);
        }
        if (!wall)
        {
            // Independently include the dependent species to test total mass
            // balance, rather than defining its flux as minus the output sum.
            double sumRaw = 0, sumGradient = 0, sumFractions = 0;
            for (int k = 0; k < fixture.species - 1; ++k)
            {
                const double Y = u[5 + k] / u[0];
                double normalDerivative = 0;
                for (int d = 0; d < 3; ++d)
                    normalDerivative += (speciesGradient(k, d) - Y * gradient[0][d]) / u[0] * normal[d];
                sumRaw -= u[0] * (point.diffusivity[k] + options.turbulentSpeciesDiffusivity) * normalDerivative;
                sumGradient += normalDerivative;
                sumFractions += Y;
            }
            const double lastRaw = u[0] * (point.diffusivity[fixture.species - 1] + options.turbulentSpeciesDiffusivity) * sumGradient;
            sumRaw += lastRaw;
            const double lastFlux = -(lastRaw - (1 - sumFractions) * sumRaw);
            double total = lastFlux, scale = std::abs(lastFlux);
            for (int k = 0; k < fixture.species - 1; ++k)
            {
                total += speciesFlux[k];
                scale += std::abs(speciesFlux[k]);
            }
            CHECK(scale > 0);
            CHECK(std::abs(total) < 1e-12 * scale);
        }
    }
}

TEST_CASE("Chemical source rejects output reservations and Jacobian aliases before changing values")
{
    Fixture fixture;
    A::DofSpace space(2, 0, "chemical reservation failures");
    auto state = C::EquationDofs::Coupled(fixture.equation.Schema(), A::DofField<double>(space,
                                                                                         R::RowLayout::Dense(0, fixture.components, fixture.components * fixture.components)));
    auto source = fixture.State(space, 1);
    fixture.Fill(state);
    C::EquationDofOperations::Fill(source, 17);
    R::Array<double> jacobian(R::RowLayout::Dense(2, fixture.components * fixture.components));
    const auto jacobianVersion = jacobian.Version(), sourceVersion = source.PhysicalField(0).Owned().Version();
    {
        const auto occupied = source.PhysicalField(1).Owned().Read();
        CHECK_THROWS(C::EvaluateChemicalSource(fixture.equation, state, source, fixture.workers.Acquire(), std::ref(jacobian)));
    }
    CHECK(jacobian.Version() == jacobianVersion);
    CHECK(source.PhysicalField(0).Owned().Version() == sourceVersion);
    auto aliasedJacobian = R::Array<double>::FromStorage(R::RowLayout::Dense(2, fixture.components * fixture.components), state.PhysicalField(0).Owned().Storage());
    CHECK_THROWS(C::EvaluateChemicalSource(fixture.equation, state, source, fixture.workers.Acquire(), std::ref(aliasedJacobian)));
    const auto read = source.ReadOwned();
    CHECK(read.Block("flow")(0, 0) == 17);
    CHECK(read.Block("species")(1, 3) == 17);
}

TEST_CASE("Actual conservative chemical sources advance a separate-field implicit reactor toward Cantera")
{
    Fixture fixture;
    auto communication = std::make_shared<R::CommunicationContext>(mpi.comm);
    const A::DofSpace space(communication, 1, 0, "homogeneous chemical stage");
    const auto route = R::RowTransferPlan::Known(communication, 1, 0,
                                                 std::vector<std::vector<I>>(mpi.size), std::vector<std::vector<I>>(mpi.size));
    const auto initial = fixture.Point(0);
    constexpr double totalTime = 1e-5;
    const auto set = [&](C::EquationDofs &state, const Eigen::VectorXd &u)
    {
        auto write = state.WriteOwned();
        const auto flow = write.Block("flow"), species = write.Block("species");
        for (int j = 0; j < 5; ++j)
            flow(0, j) = u[j];
        for (int k = 0; k < fixture.species - 1; ++k)
            species(0, k) = u[5 + k];
    };
    const auto get = [&](const C::EquationDofs &state)
    {
        auto read = state.ReadOwned();
        const auto flow = read.Block("flow").owned, species = read.Block("species").owned;
        Eigen::VectorXd result(fixture.components);
        for (int j = 0; j < 5; ++j)
            result[j] = flow(0, j);
        for (int k = 0; k < fixture.species - 1; ++k)
            result[5 + k] = species(0, k);
        return result;
    };
    const auto run = [&](int steps)
    {
        auto accepted = fixture.State(space, 1), trial = fixture.State(space, 2), rates = fixture.State(space, 0);
        auto linearRHS = C::EquationDofOperations::NewVector(accepted);
        set(accepted, initial);
        R::Array<double> jacobian(R::RowLayout::Dense(1, fixture.components * fixture.components));
        auto problem = std::make_shared<C::LinearProblemContext>(space, fixture.components, std::vector<std::vector<I>>(1), std::vector<I>{0}, C::DiagonalStorage::Block);
        C::LinearSolveSettings controls;
        controls.method = "gmres";
        controls.preconditioner = "ilu";
        controls.subspace = 4;
        controls.maxIterations = 8;
        controls.relativeTolerance = 1e-12;
        controls.absoluteTolerance = 1e-15;
        C::DofLinearSolver linear(problem, accepted, [&](auto &field)
                                  { return R::PayloadPlan(route, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::For<double>()).Bind(field.Owned(), field.Ghost()); }, controls);
        C::EquationTimeStepper stepper(accepted, "bdf1");
        const auto evaluate = [&](auto &state, auto &rate, double, int)
        { C::EvaluateChemicalSource(fixture.equation, state, rate, fixture.workers.Acquire()); };
        const auto solve = [&](C::EquationDofs &state, const C::EquationDofs &source, double mass, double scale, double, int)
        {
            const auto right = get(source);
            for (int iteration = 0; iteration < 15; ++iteration)
            {
                C::EvaluateChemicalSource(fixture.equation, state, rates, fixture.workers.Acquire(), std::ref(jacobian));
                const auto u = get(state), rate = get(rates);
                Eigen::MatrixXd matrix = mass * Eigen::MatrixXd::Identity(fixture.components, fixture.components);
                {
                    const auto read = jacobian.Read();
                    const auto view = read.View<R::UniformTableRows>();
                    for (int j = 0; j < fixture.components; ++j)
                        for (int i = 0; i < fixture.components; ++i)
                            matrix(i, j) -= scale * view(0, i + fixture.components * j);
                }
                const Eigen::VectorXd b = right - mass * u + scale * rate;
                const Eigen::VectorXd expected = matrix.partialPivLu().solve(b);
                {
                    auto write = problem->Diagonal().Write();
                    const auto view = write.View();
                    for (int j = 0; j < fixture.components; ++j)
                        for (int i = 0; i < fixture.components; ++i)
                            view(0, i + fixture.components * j) = matrix(i, j);
                }
                set(linearRHS, b);
                const auto report = linear.Solve(linearRHS);
                CHECK(report.converged);
                if (!report.usable)
                    return false;
                const Eigen::VectorXd delta = get(linear.Solution());
                CHECK((delta - expected).lpNorm<Eigen::Infinity>() < 1e-12 * std::max(1., expected.lpNorm<Eigen::Infinity>()));
                if (!delta.allFinite())
                    return false;
                C::EquationDofOperations::Axpy(state, 1, linear.Solution());
                if (delta.lpNorm<Eigen::Infinity>() < 1e-13 * std::max(1., u.lpNorm<Eigen::Infinity>()))
                    return true;
            }
            return false;
        };
        const double dt = totalTime / steps;
        for (int step = 0; step < steps; ++step)
        {
            stepper.BeginStep(accepted, trial, step * dt, dt, evaluate);
            for (int stage = 0; stage < stepper.GroupNumber(); ++stage)
            {
                const auto group = stepper.PrepareGroup(trial);
                if (group.implicit)
                {
                    const auto equation = stepper.PrepareEquation(group, trial, 0);
                    const auto &c = equation.coefficients;
                    REQUIRE(solve(equation.state, equation.resOther,
                                  c.inverseDt, c.rhsScale, c.time, group.stage));
                }
                stepper.CompleteGroup(group, trial, evaluate);
            }
            C::EquationDofOperations::Copy(accepted, trial);
            stepper.Commit();
        }
        return get(accepted);
    };
    auto fractions = fixture.Fractions(0);
    double referenceTemperature = 1400;
    fixture.prototype.advanceConstVolume(referenceTemperature, initial[0] * fixture.scales.rho0,
                                         {fractions.data(), fixture.species}, 1, totalTime * fixture.scales.Time(), 1e-11, 1e-20);
    Eigen::VectorXd expected = initial;
    for (int k = 0; k < fixture.species - 1; ++k)
        expected[5 + k] = initial[0] * fractions[k];
    const auto coarse = run(4), fine = run(8);
    const double coarseError = (coarse - expected).lpNorm<Eigen::Infinity>(), fineError = (fine - expected).lpNorm<Eigen::Infinity>();
    CAPTURE(coarseError);
    CAPTURE(fineError);
    CAPTURE(referenceTemperature);
    CHECK((fine.head<5>() - initial.head<5>()).isZero(1e-13));
    CHECK((fine - initial).lpNorm<Eigen::Infinity>() > 1e-9);
    CHECK(coarseError > 1e-12);
    CHECK(fineError < .8 * coarseError);
    CHECK(fineError < 1e-5);
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
