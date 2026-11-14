#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Physics/PerfectGasViscousFlux.hpp"
#include "Cauchy/Physics/PerfectGasCompute.hpp"
#include "Cauchy/Physics/RiemannPreparation.hpp"
#include "Cauchy/Riemann/RiemannFluxMath.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"
#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Cauchy/Input/FlowProfile.hpp"
#include "Cauchy/Physics/Positivity.hpp"
#include "Cauchy/Physics/ImplicitPositivity.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Physics/ViscousFlux.hpp"
#include "Cauchy/Discretization/ViscousFlux.hpp"
#include "Euler/Gas.hpp"
#include <limits>
#include <vector>

namespace C = DNDS::Cauchy;
TEST_CASE("QCR stress and energy work match an independent commutator and legacy Euler")
{
    C::EquationSettings settings;
    settings.transport = "NewtonianFourier";
    settings.viscosity = .03;
    settings.gasConstant = 2.5;
    const auto equation = C::EquationSystem::Resolve(settings);
    auto evaluator = equation.MakePointEvaluator();
    const auto state = equation.Gas().Conservative({1.2, .4, -.3, .2, 2.1});
    const Eigen::Vector3d velocity{.4, -.3, .2}, normal = Eigen::Vector3d{1, -2, 3}.normalized();
    Eigen::Matrix3d derivative;
    derivative << .2, -.3, .1, .6, -.4, .25, -.15, .35, .1; // (component,direction)
    C::FlowGradient gradient{};
    Eigen::Matrix<double, 3, 5> primitive = Eigen::Matrix<double, 3, 5>::Zero();
    for (int d = 0; d < 3; ++d)
    {
        for (int i = 0; i < 3; ++i)
        {
            gradient[1 + i][d] = state[0] * derivative(i, d);
            primitive(d, 1 + i) = derivative(i, d);
        }
        gradient[4][d] = state[0] * velocity.dot(derivative.col(d)); // Constant T/rho.
    }
    const Eigen::Matrix3d rotation = (derivative - derivative.transpose()) / (derivative.norm() + DNDS::verySmallReal);
    C::ViscousFluxScratch scratch(0);
    const auto point = evaluator.Evaluate(state);
    Eigen::Matrix<double, 5, 1> oldState, oldFlux;
    for (int j = 0; j < 5; ++j)
        oldState[j] = state[j];
    const auto noSpeciesGradient = [](int, int)
    { return 0.; };
    std::vector<double> noSpecies;
    for (const double turbulent : {0., .04})
    {
        const double mu = settings.viscosity + turbulent, ratio = turbulent / (mu + DNDS::verySmallReal);
        const Eigen::Matrix3d linear = mu * (derivative + derivative.transpose() - (2. / 3.) * derivative.trace() * Eigen::Matrix3d::Identity());
        const Eigen::Matrix3d stress = linear - .3 * ratio * (rotation * linear - linear * rotation);
        const Eigen::Vector3d traction = stress * normal;
        C::ViscousFluxOptions options;
        options.adiabatic = true;
        options.turbulentViscosity = turbulent;
        options.useQCR = true;
        const auto result = C::PrepareViscousFlux(state, gradient, point, noSpecies, noSpeciesGradient,
                                                  {normal[0], normal[1], normal[2]}, noSpecies, scratch, options);
        DNDS::Euler::Gas::ViscousFlux_IdealGas<3>(oldState, primitive, normal, true, settings.gamma, settings.gamma,
                                                  mu, ratio, true, 0, point.thermo.cp, oldFlux);
        for (int d = 0; d < 3; ++d)
        {
            CHECK(result[1 + d] == doctest::Approx(traction[d]).epsilon(2e-13));
            CHECK(result[1 + d] == doctest::Approx(oldFlux[1 + d]).epsilon(2e-13));
        }
        CHECK(result[4] == doctest::Approx(velocity.dot(traction)).epsilon(2e-13));
        CHECK(result[4] == doctest::Approx(oldFlux[4]).epsilon(2e-13));
        if (turbulent > 0)
            CHECK((stress - linear).norm() > 1e-4);
    }
}

TEST_CASE("Equation thermal view preserves the established perfect-gas increment and quadrature rules")
{
    C::EquationSettings settings;
    settings.gasConstant = 2.5;
    const auto equation = C::EquationSystem::Resolve(settings);
    const auto bounds = equation.PrepareThermalBounds(1e-8, .2, 1e-7);
    const C::ThermalPolicy policy(equation, bounds.densityFloor, .2, 0, 1e-7);
    const auto thermal = policy.View();
    const auto mean = equation.Gas().Conservative({1.2, .4, -.3, .2, 2.1});
    for (const C::FlowState delta : {C::FlowState{-.5, .1, -.2, .3, -40}, C::FlowState{-2, 0, 0, 0, -2}, C::FlowState{.1, 0, 0, 0, 1}})
    {
        CHECK(thermal.IncrementFraction(mean, delta, .5, 2, 1) == doctest::Approx(C::FlowIncrementFraction(mean, delta, bounds, .5, 2, 1)).epsilon(1e-14));
        const auto fixed = C::FixFlowIncrement(mean, delta, bounds);
        const auto fraction = thermal.FixFraction(mean, delta);
        for (int j = 0; j < 5; ++j)
            CHECK(fraction * delta[j] == doctest::Approx(fixed[j]).epsilon(1e-14).scale(1));
    }
    struct Points
    {
        std::array<C::FlowState, 2> values;
        double operator()(DNDS::index q, DNDS::index j) const { return values[q][j]; }
    } base{{mean, mean}}, high = base;
    high.values[0][4] -= 20;
    high.values[1][0] -= 2;
    CHECK(thermal.ReconstructionFraction(mean, base, high, 0, 2, 1, 1e-3) ==
          doctest::Approx(C::ReconstructionFraction(mean, base, high, 0, 2, bounds, 1, 1e-3)).epsilon(1e-14));
}
TEST_CASE("Polynomial viscous flux reproduces Euler conservative gradient and face-jump formulas")
{
    C::EquationSettings settings;
    settings.transport = "NewtonianFourier";
    settings.viscosity = .03;
    settings.gasConstant = 2.5;
    const auto equation = C::EquationSystem::Resolve(settings);
    const auto &gas = equation.Gas();
    const auto left = gas.Conservative({1.2, .4, -.3, .2, 2.1});
    const auto right = gas.Conservative({.8, -.1, .2, .3, 1.7});
    C::FlowGradient l{}, r{};
    C::FlowState vl = left, vr = right;
    for (int j = 0; j < 5; ++j)
    {
        vl[j] += .01 * (j + 1);
        vr[j] -= .005 * (j + 1);
        for (int d = 0; d < 3; ++d)
        {
            l[j][d] = .02 * (j + 1) * (d + 1);
            r[j][d] = -.01 * (j + 2) * (d + 1);
        }
    }
    const Eigen::Vector3d normal = Eigen::Vector3d{1, -2, 3}.normalized();
    for (double inverseDistance : {0., .3, 4.})
        for (bool adiabatic : {false, true})
        {
            Eigen::Matrix<double, 5, 1> mean, legacyFlux;
            Eigen::Matrix<double, 3, 5> gradient, primitive;
            for (int j = 0; j < 5; ++j)
            {
                mean[j] = .5 * (left[j] + right[j]);
                for (int d = 0; d < 3; ++d)
                    gradient(d, j) = .5 * (l[j][d] + r[j][d]) + normal[d] * (vr[j] - vl[j]) * inverseDistance;
            }
            const Eigen::VectorXd base;
            DNDS::Euler::Gas::GradientCons2Prim_IdealGas<3>(mean, gradient, primitive, gas.gamma, base);
            const double cp = gas.gamma * gas.gasConstant / (gas.gamma - 1);
            DNDS::Euler::Gas::ViscousFlux_IdealGas<3>(mean, primitive, normal, adiabatic, gas.gamma, gas.gamma, gas.viscosity, 0, false, gas.viscosity * cp / gas.prandtl, cp, legacyFlux);
            const auto flux = C::PolynomialViscousFlux(equation, left, right, vl, vr, l, r, {normal[0], normal[1], normal[2]}, inverseDistance, adiabatic);
            for (int j = 0; j < 5; ++j)
                CHECK(flux[j] == doctest::Approx(legacyFlux[j]).epsilon(5e-14));
        }
    // A separate chain-rule oracle from centered differences of physical point
    // states exercises density, all momenta and a non-unit gas constant.
    const auto gradient = equation.PreparePrimitiveGradient(left, l);
    for (int d = 0; d < 3; ++d)
    {
        auto minus = left, plus = left;
        const double h = 1e-5;
        for (int j = 0; j < 5; ++j)
        {
            minus[j] -= h * l[j][d];
            plus[j] += h * l[j][d];
        }
        const auto a = gas.Primitive(minus), b = gas.Primitive(plus);
        const C::FlowState pa{a.density, a.velocity[0], a.velocity[1], a.velocity[2], a.temperature};
        const C::FlowState pb{b.density, b.velocity[0], b.velocity[1], b.velocity[2], b.temperature};
        for (int j = 0; j < 5; ++j)
            CHECK(gradient[j][d] == doctest::Approx((pb[j] - pa[j]) / (2 * h)).epsilon(2e-8));
    }
}
namespace
{
    struct PointValues
    {
        const std::vector<C::FlowState> *points;
        double operator()(DNDS::index iPoint, int j) const { return points->at(iPoint)[j]; }
    };
    double LegacyQuadratureBeta(const C::FlowState &mean, const std::vector<C::FlowState> &base,
                                const std::vector<C::FlowState> &high, const C::ThermalBounds &bounds, int power, double cutoff)
    {
        using Vector = Eigen::Matrix<double, 5, 1>;
        Eigen::Matrix<double, Eigen::Dynamic, 5> value(base.size(), 5), increment(base.size(), 5);
        bool good = true;
        for (DNDS::index iPoint = 0; iPoint < DNDS::index(base.size()); ++iPoint)
        {
            good = good && C::ThermallyAdmissible(high[iPoint], bounds);
            for (int j = 0; j < 5; ++j)
            {
                value(iPoint, j) = base[iPoint][j];
                increment(iPoint, j) = high[iPoint][j] - base[iPoint][j];
            }
        }
        if (good)
            return 1;
        double density = 1;
        for (Eigen::Index iPoint = 0; iPoint < value.rows(); ++iPoint)
            if (increment(iPoint, 0) < 0)
                density = std::min(density, (value(iPoint, 0) - bounds.densityFloor) / (-increment(iPoint, 0) + DNDS::verySmallReal));
        increment *= density;
        double energy = 1;
        for (int constraint = 0; constraint < 2; ++constraint)
        {
            if (constraint == 0 && !std::isfinite(bounds.energyDensityFloor))
                continue;
            const double floor = constraint == 0 ? bounds.energyDensityFloor : 0;
            const auto internal = [](const Vector &u)
            { return u[4] - .5 * u.segment<3>(1).squaredNorm() / u[0]; };
            Vector center;
            for (int j = 0; j < 5; ++j)
                center[j] = mean[j];
            const double refCenter = constraint == 0 ? 0 : mean[0] * bounds.specificEnergyAtMinimumTemperature;
            if (internal(center) - refCenter <= 2 * floor)
            {
                energy = 0;
                continue;
            }
            for (Eigen::Index iPoint = 0; iPoint < value.rows(); ++iPoint)
            {
                const Vector u = value.row(iPoint).transpose(), delta = increment.row(iPoint).transpose(), next = u + delta;
                const double refOld = constraint == 0 ? 0 : u[0] * bounds.specificEnergyAtMinimumTemperature;
                const double refNew = constraint == 0 ? 0 : next[0] * bounds.specificEnergyAtMinimumTemperature;
                if (internal(next) - refNew < 2 * floor)
                    energy = std::min(energy, DNDS::Euler::Gas::IdealGasGetCompressionRatioPressure<3, 1, 5>(u, delta, floor, refOld, refNew));
            }
        }
        double beta = density * energy;
        if (beta < 1)
            beta *= std::pow(beta, power) * (1 - 1e-5);
        return beta < cutoff ? 0 : beta;
    }
}
TEST_CASE("Quadrature reconstruction compression preserves Euler common fractions and energy references")
{
    const C::PerfectGas gas{1.4, 1, 0, .72};
    const auto mean = gas.Conservative({1, .5, -.2, .3, 1});
    const C::ThermalBounds bounds{.01, .05, .02};
    const std::vector<C::FlowState> base{mean, gas.Conservative({.8, .2, .1, -.2, .7})};
    auto high = base;
    high[0][0] = -.2;
    high[1][1] = 3;
    high[1][2] = -2;
    for (int power : {0, 1, 3})
    {
        const auto beta = C::ReconstructionFraction(mean, PointValues{&base}, PointValues{&high}, 0, 2, bounds, power);
        CHECK(beta == doctest::Approx(LegacyQuadratureBeta(mean, base, high, bounds, power, 1e-3)).epsilon(3e-14));
        CHECK(beta >= 0);
        CHECK(beta < 1);
        for (int iPoint = 0; iPoint < 2; ++iPoint)
        {
            auto point = base[iPoint];
            for (int j = 0; j < 5; ++j)
                point[j] += beta * (high[iPoint][j] - point[j]);
            CHECK(C::ThermallyAdmissible(point, bounds));
        }
    }
    CHECK(C::ReconstructionFraction(mean, PointValues{&base}, PointValues{&base}, 0, 2, bounds) == 1);
    CHECK(C::ReconstructionFraction(mean, PointValues{&base}, PointValues{&high}, 0, 2, bounds, 1, 1) == 0);
    const C::ThermalBounds shifted{.01, -3, -std::numeric_limits<double>::infinity()};
    const C::FlowState shiftedMean{1, .2, -.1, .3, -2.7};
    const std::vector<C::FlowState> shiftedBase{shiftedMean}, shiftedHigh{{1, .2, -.1, .3, -3.5}};
    const auto beta = C::ReconstructionFraction(shiftedMean, PointValues{&shiftedBase}, PointValues{&shiftedHigh}, 0, 1, shifted);
    CHECK(beta == doctest::Approx(LegacyQuadratureBeta(shiftedMean, shiftedBase, shiftedHigh, shifted, 1, 1e-3)).epsilon(3e-14));
    auto point = shiftedMean;
    for (int j = 0; j < 5; ++j)
        point[j] += beta * (shiftedHigh[0][j] - point[j]);
    CHECK(C::ThermallyAdmissible(point, shifted));
    const C::ThermalBounds edgeBounds{.01, 1, -std::numeric_limits<double>::infinity()};
    const C::FlowState edge{1, 0, 0, 0, 1};
    const std::vector<C::FlowState> edgeBase{edge}, edgeHigh{{1, 1, 0, 0, 1}};
    CHECK(C::ReconstructionFraction(edge, PointValues{&edgeBase}, PointValues{&edgeHigh}, 0, 1, edgeBounds) == 0);
    CHECK(C::ReconstructionFraction(edge, PointValues{&edgeBase}, PointValues{&edgeBase}, 0, 1, edgeBounds) == 1);
    CHECK(C::ReconstructionFraction(mean, PointValues{&base}, PointValues{&high}, 1, 1, bounds) == 1);
}
TEST_CASE("Implicit energy compression preserves Euler scheme-one mathematics with explicit energy references")
{
    const C::PerfectGas gas{1.4, 1, 0, .72};
    const auto state = gas.Conservative({1.2, 4, -.7, .3, .8});
    const std::array<C::FlowState, 4> increments{{{-.3, 1, -2, .4, -4}, {.1, 8, 2, -.2, -2}, {0, 0, 0, 0, -.2}, {-.9, -3, 0, 0, -10}}};
    for (const auto &increment : increments)
        for (double baseOld : {0., -.5, .2})
        {
            const double baseNew = baseOld + .01;
            Eigen::Matrix<double, 5, 1> u, delta;
            for (int j = 0; j < 5; ++j)
            {
                u[j] = state[j];
                delta[j] = increment[j];
            }
            const double expected = DNDS::Euler::Gas::IdealGasGetCompressionRatioPressure<3, 1, 5>(u, delta, 1e-12, baseOld, baseNew);
            CHECK(C::InternalEnergyCompression(state, increment, 1e-12, baseOld, baseNew) == doctest::Approx(expected).epsilon(2e-14));
        }
}
TEST_CASE("Implicit flow compression enforces density and both temperature and energy constraints")
{
    const C::ThermalBounds bounds{.1, 2., .4};
    const C::FlowState state{1, 2, -.3, .1, 6};
    for (const C::FlowState increment : {C::FlowState{-2, 3, 0, 0, -4}, C::FlowState{1, 8, 1, -.4, 0}, C::FlowState{-.7, -1, 0, 0, -5}, C::FlowState{0, 0, 0, 0, -10}})
        for (double relax : {1., .5, .1})
        {
            const double alpha = C::FlowIncrementFraction(state, increment, bounds, relax);
            CHECK(alpha >= 0);
            CHECK(alpha < 1);
            auto updated = state;
            for (int j = 0; j < 5; ++j)
                updated[j] += alpha * increment[j];
            CHECK(C::ThermallyAdmissible(updated, bounds));
            const auto fixed = C::FixFlowIncrement(state, increment, bounds);
            updated = state;
            for (int j = 0; j < 5; ++j)
                updated[j] += fixed[j];
            CHECK(C::ThermallyAdmissible(updated, bounds));
        }
    const C::FlowState safe{.01, .01, 0, 0, .1};
    CHECK(C::FlowIncrementFraction(state, safe, bounds) == 1);
    CHECK(C::FixFlowIncrement(state, safe, bounds) == safe);
    // A shifted reference energy remains valid; temperature PP is never a
    // hardcoded positivity check on the stored total/internal energy itself.
    const C::ThermalBounds shifted{.1, -10, -std::numeric_limits<double>::infinity()};
    const C::FlowState shiftedState{1, 3, 0, 0, -2}, shiftedIncrement{0, 6, 0, 0, 0};
    const auto fixed = C::FixFlowIncrement(shiftedState, shiftedIncrement, shifted);
    CHECK(C::ThermallyAdmissible(C::PositivityDetail::Add(shiftedState, fixed), shifted));
}
TEST_CASE("Equation-owned physical flux Jacobian differentiates all three momentum components")
{
    for (double gamma : {1.4, 1.67})
    {
        C::EquationSettings settings;
        settings.gamma = gamma;
        const auto equation = C::EquationSystem::Resolve(settings);
        for (const C::Vector3 normal : {C::Vector3{1, 0, 0}, C::Vector3{.6, 0, -.8}, C::Vector3{0, -1, 0}})
        {
            const auto state = equation.Gas().Conservative({1.3, .7, -.2, .4, 1.1});
            const auto matrix = equation.PrepareFluxJacobian(state, normal);
            for (int column = 0; column < 5; ++column)
            {
                auto plus = state, minus = state;
                const double step = 1e-6 * std::max(1., std::abs(state[column]));
                plus[column] += step;
                minus[column] -= step;
                const auto left = C::Riemann::PhysicalFlux(equation.PrepareFluxState(plus).flow, normal);
                const auto right = C::Riemann::PhysicalFlux(equation.PrepareFluxState(minus).flow, normal);
                for (int row = 0; row < 5; ++row)
                    CHECK(matrix[column * 5 + row] == doctest::Approx((left[row] - right[row]) / (2 * step)).epsilon(2e-8).scale(1));
            }
        }
    }
}

TEST_CASE("Physical boundary Jacobians match conservative exterior-state derivatives")
{
    for (double gamma : {1.4, 1.67})
        for (const std::string kind : {"extrapolate", "farfield", "slipWall", "noSlipWall", "analytic", "doubleMachBottom"})
        {
            const C::PerfectGas gas{gamma, 1, .02, .72};
            C::BoundarySettings settings;
            settings.kind = kind;
            if (kind == "noSlipWall")
            {
                settings.thermal = "isothermal";
                settings.temperature = .8;
            }
            settings.primitive = {2, -.3, .2, .1, 1.4};
            settings.velocity = {.04, -.02, .03};
            C::BoundarySystem boundaries({{"7", settings}}, gas);
            const auto &boundary = boundaries.At(7);
            for (const C::Vector3 normal : {C::Vector3{1, 0, 0}, C::Vector3{.6, .8, 0}})
                for (double speed : {-4., -.2, .2, 4.})
                    for (double x : {.1, .5})
                    {
                        const auto inside = gas.Conservative({1.3, speed * normal[0] - .1 * normal[1], speed * normal[1] + .1 * normal[0], .15, 1.1});
                        const C::Vector3 point{x, .2, .3};
                        const auto matrix = boundary.ExteriorJacobian(gas, inside, normal, point, .2);
                        for (int column = 0; column < 5; ++column)
                        {
                            auto plus = inside, minus = inside;
                            const double h = 1e-6 * std::max(1., std::abs(inside[column]));
                            plus[column] += h;
                            minus[column] -= h;
                            const auto a = boundary.Exterior(gas, plus, normal, point, .2);
                            const auto b = boundary.Exterior(gas, minus, normal, point, .2);
                            for (int row = 0; row < 5; ++row)
                                CHECK(matrix[column * 5 + row] == doctest::Approx((a[row] - b[row]) / (2 * h)).epsilon(3e-8).scale(1));
                        }
                    }
        }
}
TEST_CASE("Thermal PP uses an internal-energy temperature bound including shifted energy references")
{
    const C::ThermalBounds bounds{.1, -10, -std::numeric_limits<double>::infinity()};
    const C::FlowState base{1, 3, 0, 0, -2}; // internal=-6.5, above shifted Tmin energy=-10
    CHECK(C::ThermallyAdmissible(base, bounds));
    CHECK(C::AdmissibleFraction(C::FlowState{1, 0, 0, 0, 2}, C::FlowState{-1, 0, 0, 0, 2}, bounds) == doctest::Approx(.45).epsilon(2e-9));
    const C::FlowState target{1, 7, 0, 0, -2};
    CHECK_FALSE(C::ThermallyAdmissible(target, bounds));
    const double exact = (4 - 3.) / (7 - 3.); // kinetic <=8, so |momentum|<=4
    const auto fraction = C::AdmissibleFraction(base, target, bounds);
    CHECK(fraction == doctest::Approx(exact).epsilon(2e-9));
    C::FlowState state;
    for (int j = 0; j < 5; ++j)
        state[j] = (1 - fraction) * base[j] + fraction * target[j];
    CHECK(C::ThermallyAdmissible(state, bounds));
    struct NonlinearBound
    {
        double densityFloor = .1;
        double MinimumInternalEnergyDensity(const C::FlowState &u) const { return u[0] * u[0]; }
    } nonlinear;
    const C::FlowState a{1, 0, 0, 0, 2}, b{3, 0, 0, 0, 2};
    CHECK(C::AdmissibleFraction(a, b, nonlinear) == doctest::Approx((std::sqrt(2.) - 1) / 2).epsilon(2e-9));
    const C::PerfectGas gas{1.4, 2, 0, .72};
    const C::ThermalBounds temperature{.1, 2 * .5 / .4, 0};
    CHECK_FALSE(C::ThermallyAdmissible(gas.Conservative({1, 100, 0, 0, .5}), temperature)); // T=.25 <.5, yet p>0
    CHECK(C::ThermallyAdmissible(gas.Conservative({1, 100, 0, 0, 2}), temperature));
    for (int sample = 1; sample < 40; ++sample)
    {
        const double momentum = 4 + sample * .2;
        const C::FlowState point{1, momentum, 0, 0, -2};
        const double expected = 1 / (momentum - 3);
        CHECK(C::AdmissibleFraction(base, point, bounds) == doctest::Approx(expected).epsilon(2e-9));
    }
    struct MixtureBound
    {
        double densityFloor = .1;
        double MinimumInternalEnergyDensity(const std::array<double, 6> &u) const
        {
            return -10 * u[0] + 2 * u[5];
        } // shifted species energy, affine at Tmin
    } mixture;
    CHECK(C::ThermallyAdmissible(std::array<double, 6>{1, 3, 0, 0, -2, .25}, mixture));
    CHECK_FALSE(C::ThermallyAdmissible(std::array<double, 6>{1, 3, 0, 0, -2, 2}, mixture));
    const C::ThermalBounds tinyTemperature{1e-12, 2.5e-10, 0};
    C::FlowState nearlyCold{1, 1, 0, 0, .5 + 2.5e-10 + 8 * std::numeric_limits<double>::epsilon()};
    CHECK(C::ThermallyAdmissible(nearlyCold, tinyTemperature));
    CHECK_FALSE(C::ThermallyInterior(nearlyCold, tinyTemperature));
    auto tooFast = nearlyCold;
    tooFast[1] += .1;
    CHECK(C::AdmissibleFraction(nearlyCold, tooFast, tinyTemperature) == 0);
}
TEST_CASE("Cold mean limiter characteristics retain a reliable inverse without altering physical sound speed")
{
    const auto equation = C::EquationSystem::Resolve(C::EquationSettings{});
    const auto state = equation.Gas().Conservative({1, 1, .2, .1, 1e-10});
    const auto physical = equation.PrepareFluxState(state);
    const auto basis = equation.PrepareCharacteristics(state, state, {1, 0, 0});
    const Eigen::Map<const Eigen::Matrix<double, 5, 5>> l(basis.forward.data()), r(basis.inverse.data());
    CHECK((l * r).isApprox(Eigen::Matrix<double, 5, 5>::Identity(), 1e-7));
    CHECK(basis.componentwiseFallback);
    CHECK(physical.soundSpeed * physical.soundSpeed == doctest::Approx(1.4 * equation.Gas().Pressure(state)).epsilon(1e-12));
}
TEST_CASE("Perfect-gas characteristic matrices match independent Euler analytic matrices in rotated frames")
{
    const auto equation = C::EquationSystem::Resolve(C::EquationSettings{});
    const auto left = equation.Gas().Conservative({.7, .4, -.2, .1, 1.3});
    const auto right = equation.Gas().Conservative({1.6, -.3, .1, .2, .8});
    const double rl = std::sqrt(left[0]), rr = std::sqrt(right[0]);
    Eigen::Vector3d velocity;
    for (int d = 0; d < 3; ++d)
        velocity[d] = (left[d + 1] / rl + right[d + 1] / rr) / (rl + rr);
    const double h = ((left[4] + 1.3) / rl + (right[4] + .8) / rr) / (rl + rr);
    const double a = std::sqrt(.4 * (h - .5 * velocity.squaredNorm()));
    for (Eigen::Vector3d normal : {Eigen::Vector3d::UnitX().eval(), Eigen::Vector3d::UnitY().eval(), Eigen::Vector3d{1, -2, 3}.normalized().eval()})
    {
        Eigen::Matrix3d basis;
        basis.col(0) = normal;
        basis.col(1) = normal.unitOrthogonal();
        basis.col(2) = normal.cross(basis.col(1));
        const Eigen::Vector3d localVelocity = basis.transpose() * velocity;
        Eigen::Matrix<double, 5, 5> eulerL, eulerR, rotation = Eigen::Matrix<double, 5, 5>::Identity();
        DNDS::Euler::Gas::EulerGasLeftEigenVector<3>(localVelocity, velocity.squaredNorm(), h, a, 1.4, eulerL);
        DNDS::Euler::Gas::EulerGasRightEigenVector<3>(localVelocity, velocity.squaredNorm(), h, a, eulerR);
        rotation.block<3, 3>(1, 1) = basis;
        eulerR = (rotation * eulerR).eval();
        eulerL = (eulerL * rotation.transpose()).eval();
        const auto actual = equation.PrepareCharacteristics(left, right, {normal[0], normal[1], normal[2]});
        REQUIRE_FALSE(actual.componentwiseFallback);
        const Eigen::Map<const Eigen::Matrix<double, 5, 5>> l(actual.forward.data()), r(actual.inverse.data());
        constexpr int permutation[]{1, 2, 3, 4, 0};
        for (int iWave = 0; iWave < 5; ++iWave)
        {
            CHECK(l.row(iWave).isApprox(eulerL.row(permutation[iWave]), 2e-13));
            CHECK(r.col(iWave).isApprox(eulerR.col(permutation[iWave]), 2e-13));
        }
    }
}
TEST_CASE("Cached generic characteristics preserve caloric energy offsets and the frozen EOS Jacobian")
{
    const auto equation = C::EquationSystem::Resolve(C::EquationSettings{});
    // Independent variable-caloric ideal EOS: e(T)=e0+cv0*T+cv1*T*T/2.
    // Nonzero e0 distinguishes the generic contact wave from kinetic energy.
    const double gasConstant = .7, cv0 = 1.8, cv1 = .2;
    const Eigen::Vector3d velocity{.4, -.2, .1}, normal = Eigen::Vector3d{1, -2, 3}.normalized();
    const double kinetic = .5 * velocity.squaredNorm(), rho = 1.2, temperature = 2.3;
    for (double energyOffset : {-4., 3.})
    {
        const double internal = energyOffset + cv0 * temperature + .5 * cv1 * temperature * temperature;
        const double pressure = rho * gasConstant * temperature, beta = gasConstant / (cv0 + cv1 * temperature);
        const double a2 = (1 + beta) * pressure / rho;
        const C::FlowState state{rho, rho * velocity[0], rho * velocity[1], rho * velocity[2], rho * (internal + kinetic)};
        const C::CharacteristicThermodynamics thermal{pressure, std::sqrt(a2), internal + kinetic + pressure / rho, beta};
        const auto actual = equation.PrepareCharacteristics(state, state, thermal, thermal, {normal[0], normal[1], normal[2]});
        REQUIRE_FALSE(actual.componentwiseFallback);
        const Eigen::Map<const Eigen::Matrix<double, 5, 5>> l(actual.forward.data()), r(actual.inverse.data());
        CHECK((l * r).isApprox(Eigen::Matrix<double, 5, 5>::Identity(), 1e-12));
        CHECK((r * l).isApprox(Eigen::Matrix<double, 5, 5>::Identity(), 1e-12));
        CHECK(std::abs(r(4, 0) - kinetic) > .1);
        // Assemble the normal flux derivative directly from dp=rho*R*dT+R*T*drho.
        Eigen::Matrix<double, 5, 5> jacobian;
        const double un = velocity.dot(normal);
        for (int j = 0; j < 5; ++j)
        {
            const double drho = j == 0, dEnergy = j == 4;
            const Eigen::Vector3d dMomentum{double(j == 1), double(j == 2), double(j == 3)};
            const Eigen::Vector3d dVelocity = (dMomentum - velocity * drho) / rho;
            const double dInternal = (dEnergy - velocity.dot(dMomentum) + (kinetic - internal) * drho) / rho;
            const double dp = gasConstant * temperature * drho + rho * beta * dInternal;
            const double dun = dVelocity.dot(normal);
            jacobian(0, j) = normal.dot(dMomentum);
            jacobian.block<3, 1>(1, j) = dMomentum * un + rho * velocity * dun + dp * normal;
            jacobian(4, j) = (dEnergy + dp) * un + (state[4] + pressure) * dun;
        }
        Eigen::Matrix<double, 5, 1> eigenvalues;
        eigenvalues << un, un, un, un + std::sqrt(a2), un - std::sqrt(a2);
        CHECK((jacobian * r).isApprox(r * eigenvalues.asDiagonal(), 2e-12));
        auto invalid = thermal;
        invalid.pressureInternalEnergyDerivative = 0;
        CHECK_THROWS(equation.PrepareCharacteristics(state, state, thermal, invalid, {normal[0], normal[1], normal[2]}));
    }
}
TEST_CASE("Cauchy prescribed DMR shock and boundary stage times match canonical conservative data")
{
    C::FlowProfileSettings profile;
    profile.kind = "doubleMachReflection";
    const C::PerfectGas gas;
    const auto ahead = C::ProfileValue(profile, {1, 0, 0, 0, 1}, gas, {1, .5, 0}, 0);
    const auto behind = C::ProfileValue(profile, {1, 0, 0, 0, 1}, gas, {1, .5, 0}, .1);
    const C::FlowState unshocked{1.4, 0, 0, 0, 2.5};
    for (int j = 0; j < 5; ++j)
        CHECK(ahead[j] == doctest::Approx(unshocked[j]).epsilon(2e-14));
    const C::FlowState shocked{8, 57.157676649772960, -33, 0, 563.5};
    for (int j = 0; j < 5; ++j)
        CHECK(behind[j] == doctest::Approx(shocked[j]).epsilon(2e-14));
    C::BoundarySettings settings;
    settings.kind = "doubleMachBottom";
    settings.profile = profile;
    C::BoundarySystem boundaries({{"1", settings}}, gas);
    const auto &condition = boundaries.At(1);
    const auto prescribed = condition.Exterior(gas, ahead, {0, -1, 0}, {.1, 0, 0}, .05);
    for (int j = 0; j < 5; ++j)
        CHECK(prescribed[j] == doctest::Approx(shocked[j]).epsilon(2e-14));
    const auto wall = condition.Exterior(gas, shocked, {0, -1, 0}, {.5, 0, 0}, .05);
    CHECK(wall[2] == -shocked[2]);
    CHECK(wall[4] == doctest::Approx(shocked[4]));
}
TEST_CASE("Cauchy beta-five vortex oracle advects periodically and preserves isentropy")
{
    C::FlowProfileSettings profile;
    profile.kind = "isentropicVortex";
    const C::PerfectGas gas;
    const auto a = C::ProfileValue(profile, {1, 1, 1, 0, 1}, gas, {5, 5, 0}, 0);
    const auto b = C::ProfileValue(profile, {1, 1, 1, 0, 1}, gas, {5, 5, 0}, 10);
    CHECK(a[0] == doctest::Approx(.49380732389534654));
    CHECK(gas.Pressure(a) == doctest::Approx(std::pow(a[0], gas.gamma)));
    CHECK(a == b);
}
TEST_CASE("Cauchy farfield respects supersonic directions and subsonic incoming component count")
{
    C::PerfectGas gas;
    C::BoundarySettings settings;
    settings.kind = "farfield";
    settings.primitive = {2, -3, .1, .2, 4};
    C::BoundarySystem boundaries({{"1", settings}}, gas);
    const auto &bc = boundaries.At(1);
    const C::Vector3 normal{1, 0, 0};
    for (double speed : {-4., -.2, .2, 4.})
    {
        const auto inside = gas.Conservative({1, speed, .5, -.1, 1});
        const auto value = gas.Primitive(bc.Exterior(gas, inside, normal));
        if (speed > 2)
            CHECK(bc.Exterior(gas, inside, normal) == inside);
        if (speed < -2)
            CHECK(value.density == 2);
        if (speed == .2)
        {
            CHECK(value.pressure == doctest::Approx(4));
            CHECK(value.density == 1);
        }
        if (speed == -.2)
        {
            CHECK(value.pressure == doctest::Approx(1));
            CHECK(value.density == 2);
            CHECK(value.velocity[0] == -3);
        }
    }
}
static C::Riemann::AcousticState Prepare(const C::PerfectGas &gas, const C::FlowState &state)
{
    const auto primitive = gas.Primitive(state);
    return {{state, primitive.pressure}, primitive.soundSpeed};
}
TEST_CASE("Flow5 retains transverse momentum on planar geometry and exact physical flux")
{
    const C::PerfectGas gas{1.4, 2., 0., 0.72};
    const C::FlowState primitive{2, 3, -4, 5, 7};
    const auto state = gas.Conservative(primitive);
    REQUIRE(gas.Admissible(state));
    CHECK(state[4] == doctest::Approx(67.5)); // 7/.4 + .5*2*(9+16+25)
    const auto back = gas.Primitive(state);
    CHECK(back.density == 2);
    CHECK(back.pressure == doctest::Approx(7));
    CHECK(back.temperature == doctest::Approx(1.75));
    CHECK(back.velocity == C::Vector3{3, -4, 5});
    const auto flux = C::Riemann::PhysicalFlux(Prepare(gas, state).flow, {1, 0, 0});
    const C::FlowState exact{6, 25, -24, 30, 223.5};
    for (int j = 0; j < 5; ++j)
        CHECK(flux[j] == doctest::Approx(exact[j]));
    CHECK_FALSE(gas.Admissible({0, 0, 0, 0, 1}));
    CHECK_FALSE(gas.Admissible({1, 3, 0, 0, 4}));
    CHECK_FALSE(gas.Admissible({1, 0, 0, 0, std::numeric_limits<double>::infinity()}));
}
TEST_CASE("Inviscid kernels agree with legacy flux and conserve under oriented face reversal")
{
    const C::PerfectGas gas;
    const C::Vector3 normal{1. / 3, 2. / 3, 2. / 3}, reverse{-1. / 3, -2. / 3, -2. / 3};
    for (int sample = 0; sample < 40; ++sample)
    {
        const auto left = gas.Conservative({0.5 + .1 * sample, .3 * sample - 2, .7, -1.1, 2. + .2 * sample});
        const auto right = gas.Conservative({1.4, -0.2, .3 - .1 * sample, 1.3, 1.1});
        Eigen::Matrix<double, 5, 1> state, legacy;
        Eigen::Vector3d velocity, n;
        for (int j = 0; j < 5; ++j)
            state[j] = left[j];
        for (int d = 0; d < 3; ++d)
        {
            velocity[d] = left[1 + d] / left[0];
            n[d] = normal[d];
        }
        DNDS::Euler::Gas::GasInviscidFlux_XY<3>(state, velocity, Eigen::Vector3d::Zero(), n, gas.Pressure(left), legacy);
        const auto leftInput = Prepare(gas, left), rightInput = Prepare(gas, right);
        const auto physical = C::Riemann::PhysicalFlux(leftInput.flow, normal);
        const auto same = C::Riemann::Rusanov(leftInput, leftInput, normal);
        const auto numerical = C::Riemann::Rusanov(leftInput, rightInput, normal), flipped = C::Riemann::Rusanov(rightInput, leftInput, reverse);
        for (int j = 0; j < 5; ++j)
        {
            CHECK(physical[j] == doctest::Approx(legacy[j]).epsilon(2e-14));
            CHECK(same[j] == doctest::Approx(physical[j]).epsilon(2e-14));
            CHECK(numerical[j] == doctest::Approx(-flipped[j]).epsilon(2e-14));
        }
    }
}
TEST_CASE("Newtonian Fourier kernels reproduce analytic shear heat flux and legacy three-dimensional stress")
{
    const C::PerfectGas gas{1.4, 2, .03, .72};
    const auto state = gas.Conservative({1.2, 0, 0, 0, 2.4});
    C::FlowGradient gradient{};
    gradient[1][1] = 4;
    auto flux = gas.ViscousFlux(gas.Primitive(state), gradient, {0, 1, 0});
    CHECK(flux[1] == doctest::Approx(.12));
    CHECK(flux[2] == 0);
    CHECK(flux[4] == 0);
    gradient = {};
    gradient[4][0] = 3;
    flux = gas.ViscousFlux(gas.Primitive(state), gradient, {1, 0, 0});
    const double cp = gas.gamma * gas.gasConstant / (gas.gamma - 1), k = gas.viscosity * cp / gas.prandtl;
    CHECK(flux[4] == doctest::Approx(k * 3));
    gradient = {};
    gradient[1][1] = 2;
    gradient[2][0] = -2; // Rigid rotation: strain/stress identically zero.
    for (auto value : gas.ViscousFlux(gas.Primitive(state), gradient, {1, 0, 0}))
        CHECK(value == 0);

    const auto moving = gas.Conservative({1.2, .4, -.3, .2, 2.4});
    const auto primitive = gas.Primitive(moving);
    Eigen::Matrix<double, 5, 1> legacyState, legacyFlux;
    Eigen::Matrix<double, 3, 5> legacyGradient;
    Eigen::Vector3d normal(1. / 3, 2. / 3, -2. / 3);
    for (int j = 0; j < 5; ++j)
    {
        legacyState[j] = moving[j];
        for (int d = 0; d < 3; ++d)
            gradient[j][d] = .13 * (j + 1) - .2 * (d + 1);
    }
    for (int d = 0; d < 3; ++d)
    {
        for (int j = 0; j < 4; ++j)
            legacyGradient(d, j) = gradient[j][d];
        legacyGradient(d, 4) = gas.gasConstant * (moving[0] * gradient[4][d] + primitive.temperature * gradient[0][d]);
    }
    DNDS::Euler::Gas::ViscousFlux_IdealGas<3>(legacyState, legacyGradient, normal, false, gas.gamma, gas.gamma, gas.viscosity, 0, false, k, cp, legacyFlux);
    const auto result = gas.ViscousFlux(primitive, gradient, {normal[0], normal[1], normal[2]});
    for (int j = 0; j < 5; ++j)
        CHECK(result[j] == doctest::Approx(legacyFlux[j]).epsilon(3e-14));
}
TEST_CASE("Boundary values and exterior states implement slip no-slip and thermal wall contracts")
{
    const C::PerfectGas gas{1.4, 2, .03, .72};
    const auto state = gas.Conservative({2, 3, -4, 5, 7});
    const C::Vector3 normal{0, 1, 0};
    C::BoundaryCondition wall;
    wall.kind = C::BoundaryKind::SlipWall;
    const auto slipFace = gas.Primitive(wall.FaceValue(gas, state, normal));
    const auto slipOutside = gas.Primitive(wall.Exterior(gas, state, normal));
    CHECK(slipFace.velocity == C::Vector3{3, 0, 5});
    CHECK(slipOutside.velocity == C::Vector3{3, 4, 5});
    CHECK(slipFace.pressure == doctest::Approx(7));
    wall.kind = C::BoundaryKind::NoSlipWall;
    wall.velocity = {.5, 0, -.2};
    wall.isothermal = true;
    wall.temperature = .8;
    const auto face = gas.Primitive(wall.FaceValue(gas, state, normal));
    CHECK(face.velocity == wall.velocity);
    CHECK(face.temperature == doctest::Approx(.8));
    CHECK(face.pressure == doctest::Approx(7));
    CHECK(face.density == doctest::Approx(7 / (2 * .8)));
    const auto outside = gas.Primitive(wall.Exterior(gas, state, normal));
    CHECK(outside.velocity == C::Vector3{-2, 4, -5.4});
    CHECK(outside.density == doctest::Approx(7 / (2 * .8)));
    CHECK(outside.temperature == doctest::Approx(.8));
    C::BoundarySettings config;
    C::BoundarySystem system({{"7", config}}, gas);
    CHECK_NOTHROW(system.ValidateUsed({7}));
    CHECK_THROWS(system.ValidateUsed({}));
    CHECK_THROWS(system.At(8));
}

TEST_CASE("Farfield sonic branches and approximate wall Jacobians follow Euler conventions")
{
    C::EquationSettings settings;
    settings.gamma = 2;
    settings.gasConstant = 1;
    const auto equation = C::EquationSystem::Resolve(settings);
    const auto &gas = equation.Gas();
    auto evaluator = equation.MakePointEvaluator();
    C::BoundarySettings far;
    far.kind = "farfield";
    far.primitive = {2, -.3, .1, .2, 3};
    const auto boundary = C::BoundarySystem({{"7", far}}, equation).At(7);
    for (double speed : {-3., -2., 0., 2., 3.})
    {
        // rho=1,p=2,gamma=2 gives exactly sound speed two.
        const auto inside = gas.Conservative({1, speed, 0, 0, 2});
        const auto expected = speed > 2 ? inside : speed > 0 ? gas.Conservative({1, speed, 0, 0, 3})
                                               : speed > -2  ? gas.Conservative({2, -.3, .1, .2, 2})
                                                             : gas.Conservative(far.primitive);
        const auto actual = boundary.Exterior(gas, inside, {1, 0, 0});
        const auto generic = boundary.Exterior(evaluator, inside, {}, {1, 0, 0});
        for (int j = 0; j < 5; ++j)
        {
            CHECK(actual[j] == doctest::Approx(expected[j]).epsilon(1e-13));
            CHECK(generic[j] == doctest::Approx(expected[j]).epsilon(1e-13));
        }
        CHECK(boundary.ApproximateExteriorJacobian(gas, inside, {1, 0, 0}) == std::array<double, 25>{});
    }
    C::BoundarySettings wall;
    wall.kind = "noSlipWall";
    wall.thermal = "isothermal";
    wall.temperature = .8;
    const auto condition = C::BoundarySystem({{"7", wall}}, equation).At(7);
    const auto inside = gas.Conservative({1, .3, -.2, .1, 2});
    const auto outside = condition.Exterior(evaluator, inside, {}, {1, 0, 0});
    const auto expected = gas.Conservative({2 / .8, -.3, .2, -.1, 2});
    for (int j = 0; j < 5; ++j)
        CHECK(outside[j] == doctest::Approx(expected[j]).epsilon(1e-13));
    const auto approximate = condition.ApproximateExteriorJacobian(gas, inside, {1, 0, 0});
    for (int j = 0; j < 5; ++j)
        for (int k = 0; k < 5; ++k)
            CHECK(approximate[j * 5 + k] == (j == k ? (j >= 1 && j <= 3 ? -1 : 1) : 0));
}

namespace
{
    template <class Scalar>
    void CheckTypedPerfectGas(double tolerance)
    {
        C::PerfectGas gas{1.4, 2., .03, .72};
        gas.viscosityModel = DNDS::Physics::Transport::ViscosityModel::Sutherland;
        gas.viscosityReferenceTemperature = .7;
        gas.sutherlandConstant = .2;
        const auto typedGas = C::CastPerfectGas<Scalar>(gas);
        const auto state = gas.Conservative({1.2, .4, -.3, .2, 2.1});
        C::FlowStateT<Scalar> typedState{};
        C::FlowGradient gradient{};
        C::FlowGradientT<Scalar> typedGradient{};
        for (int j = 0; j < 5; ++j)
        {
            typedState[j] = Scalar(state[j]);
            for (int d = 0; d < 3; ++d)
            {
                gradient[j][d] = .03 * (j + 1) - .02 * (d + 1);
                typedGradient[j][d] = Scalar(gradient[j][d]);
            }
        }
        const auto point = C::TryPerfectGasPoint(gas, state, true);
        const auto typedPoint = C::TryPerfectGasPoint(typedGas, typedState, true);
        REQUIRE(bool(point));
        REQUIRE(bool(typedPoint));
        const C::Vector3 normal{1. / 3, 2. / 3, -2. / 3};
        const C::Vector3T<Scalar> typedNormal{Scalar(normal[0]), Scalar(normal[1]), Scalar(normal[2])};
        const auto expected = C::PreparePerfectGasViscousFlux(state, gradient, point.value, normal, 0., 0., false, false);
        const auto actual = C::PreparePerfectGasViscousFlux(typedState, typedGradient, typedPoint.value, typedNormal, Scalar(0), Scalar(0), false, false);
        CHECK(DNDS::Math::ScalarCast<double>(typedPoint.value.thermo.soundSpeed) == doctest::Approx(point.value.thermo.soundSpeed).epsilon(tolerance));
        CHECK(DNDS::Math::ScalarCast<double>(typedPoint.value.transport.viscosity) == doctest::Approx(point.value.transport.viscosity).epsilon(tolerance));
        CHECK(DNDS::Math::ScalarCast<double>(typedPoint.value.diffusionBound) == doctest::Approx(point.value.diffusionBound).epsilon(tolerance));
        for (int j = 0; j < 5; ++j)
            CHECK(DNDS::Math::ScalarCast<double>(actual[j]) == doctest::Approx(expected[j]).epsilon(tolerance).scale(1));
        const C::Riemann::AcousticState acoustic{{state, point.value.thermo.pressure}, point.value.thermo.soundSpeed};
        const C::Riemann::AcousticStateT<Scalar> typedAcoustic{{typedState, typedPoint.value.thermo.pressure}, typedPoint.value.thermo.soundSpeed};
        const auto prepared = C::PreparePerfectGasRiemann(gas, C::Riemann::Solver::Roe_M9, acoustic, acoustic, acoustic, acoustic, normal);
        const auto typedPrepared = C::PreparePerfectGasRiemann(typedGas, C::Riemann::Solver::Roe_M9, typedAcoustic, typedAcoustic, typedAcoustic, typedAcoustic, typedNormal);
        const auto flux = C::Riemann::Math::Evaluate(C::Riemann::Solver::Roe_M9, prepared.Input(acoustic, acoustic, normal), C::Riemann::FluxSettings{});
        const auto typedFlux = C::Riemann::Math::Evaluate(C::Riemann::Solver::Roe_M9, typedPrepared.Input(typedAcoustic, typedAcoustic, typedNormal), C::Riemann::FluxSettingsT<Scalar>{});
        for (int j = 0; j < 5; ++j)
            CHECK(DNDS::Math::ScalarCast<double>(typedFlux.flux[j]) == doctest::Approx(flux.flux[j]).epsilon(tolerance).scale(1));
        const auto right = gas.Conservative({1.1, .5, -.4, .25, 1.9});
        const auto fullConvection = C::TryPerfectGasRiemannCompute<Scalar>(gas, C::Riemann::Solver::Roe_M9, state, right, state, right, normal);
        const auto referenceConvection = C::TryPerfectGasRiemannCompute<double>(gas, C::Riemann::Solver::Roe_M9, state, right, state, right, normal);
        REQUIRE(bool(fullConvection));
        REQUIRE(bool(referenceConvection));
        C::FlowState faceState;
        C::FlowGradient faceGradient;
        for (int j = 0; j < 5; ++j)
        {
            faceState[j] = .5 * (state[j] + right[j]);
            for (int d = 0; d < 3; ++d)
                faceGradient[j][d] = gradient[j][d] + normal[d] * (right[j] - state[j]) * .7;
        }
        const auto facePoint = C::TryPerfectGasPoint(gas, faceState, true);
        REQUIRE(bool(facePoint));
        const auto referenceDiffusion = C::PreparePerfectGasViscousFlux(faceState, faceGradient, facePoint.value, normal, 0., 0., false, false);
        const auto fullDiffusion = C::TryPerfectGasPolynomialViscousCompute<Scalar>(gas, state, right, gradient, gradient, normal, .7, false);
        REQUIRE(bool(fullDiffusion));
        for (int j = 0; j < 5; ++j)
        {
            CHECK(fullConvection.value.flux[j] == doctest::Approx(referenceConvection.value.flux[j]).epsilon(tolerance).scale(1));
            CHECK(fullDiffusion.value[j] == doctest::Approx(referenceDiffusion[j]).epsilon(tolerance).scale(1));
        }
    }
}
TEST_CASE("Perfect gas transport and flux preparation preserve typed local arithmetic")
{
    CheckTypedPerfectGas<DNDS::Math::real_f32>(4e-6);
#ifdef DNDS_USE_CCCL_COMPUTE
    CheckTypedPerfectGas<DNDS::Math::real_f32f32>(2e-12);
#endif
}

namespace
{
    template <class Scalar>
    void CheckComputeShockAndContact(DNDS::real tolerance)
    {
        const C::PerfectGas gas;
        // A large pressure jump and a low-pressure stationary contact probe
        // arithmetic cancellation outside the smooth TGV pilot. No repairs.
        for (const auto rightPrimitive : {C::FlowState{.125, 0, 0, 0, .1}, C::FlowState{.02, 0, 0, 0, 1e-6}})
        {
            const bool contact = rightPrimitive[4] < 1e-5;
            const auto left = gas.Conservative({1, 0, 0, 0, contact ? 1e-6 : 1000.});
            const auto right = gas.Conservative(rightPrimitive);
            for (const auto solver : {C::Riemann::Solver::Roe_M1, C::Riemann::Solver::Rusanov})
            {
                const auto expected = C::TryPerfectGasRiemannCompute<double>(gas, solver, left, right, left, right, {1, 0, 0});
                const auto actual = C::TryPerfectGasRiemannCompute<Scalar>(gas, solver, left, right, left, right, {1, 0, 0});
                const auto reverse = C::TryPerfectGasRiemannCompute<Scalar>(gas, solver, right, left, right, left, {-1, 0, 0});
                REQUIRE(bool(expected));
                REQUIRE(bool(actual));
                REQUIRE(bool(reverse));
                for (int j = 0; j < 5; ++j)
                {
                    CHECK(std::isfinite(actual.value.flux[j]));
                    CHECK(std::abs(actual.value.flux[j] - expected.value.flux[j]) / (1 + std::abs(expected.value.flux[j])) <= tolerance);
                    CHECK(std::abs(actual.value.flux[j] + reverse.value.flux[j]) / (1 + std::abs(actual.value.flux[j])) <= tolerance);
                }
            }
        }
    }
}
TEST_CASE("Reduced perfect gas flux preserves shock and low pressure contact bounds")
{
    CheckComputeShockAndContact<DNDS::Math::real_f32>(5e-5);
#ifdef DNDS_USE_CCCL_COMPUTE
    CheckComputeShockAndContact<DNDS::Math::real_f32f32>(2e-10);
#endif
}
