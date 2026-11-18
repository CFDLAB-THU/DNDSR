#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "Cauchy/Physics/RANSPoint.hpp"
#include "Cauchy/Boundary/CPU/RANSWallKernels.hpp"
#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;

namespace
{
    C::EquationSystem Equation(const char *model)
    {
        C::EquationSettings settings;
        settings.transport = "NewtonianFourier";
        settings.viscosity = .002;
        DNDS::Physics::RANS::ModelSettings rans;
        rans.model = model;
        return C::EquationSystem::Resolve(settings).WithRANS(rans);
    }
    R::RowLayout Layout(I width, int layout)
    {
        if (layout == 1)
            return R::RowLayout::ColMajor(0, width, {16, 1});
        if (layout == 2)
            return R::RowLayout::TiledColMajor(0, width, 32);
        return R::RowLayout::Dense(0, width, width + 2);
    }
}

TEST_CASE("Generic wall Jacobian follows the centralized RANS exterior-state law")
{
    const auto equation = Equation("SA");
    C::BoundaryPointOperator pointOperator(equation);
    C::BoundaryCondition wall;
    wall.kind = C::BoundaryKind::NoSlipWall;
    std::vector<double> inside{1.2, .31, -.17, .08, 3.4, .025};
    const C::Vector3 normal{.6, .8, 0}, location{.2, -.1, .3};
    const C::RigidFrame frame{};
    Eigen::MatrixXd derivative(inside.size(), inside.size());
    pointOperator.ApproximateExteriorJacobian(
        derivative, wall, inside, normal, location, frame, .4);
    CHECK(derivative(5, 5) == -1);
    for (int j = 0; j < 5; ++j)
        CHECK(derivative(5, j) == 0);
    for (int j = 0; j < 6; ++j)
        CHECK(derivative(j, 5) == (j == 5 ? -1 : 0));

    auto evaluator = equation.MakePointEvaluator();
    const auto exterior = [&](double perturbation)
    {
        auto point = inside;
        point[5] += perturbation;
        pointOperator.Evaluate(point, wall, -1, normal, location, evaluator,
                               frame, .4,
                               C::BoundaryPointStateKind::ExteriorState);
        return point[5];
    };
    constexpr double h = 1e-7;
    CHECK((exterior(h) - exterior(-h)) / (2 * h) ==
          doctest::Approx(derivative(5, 5)).epsilon(1e-9));

    wall.kind = C::BoundaryKind::SlipWall;
    pointOperator.ApproximateExteriorJacobian(
        derivative, wall, inside, normal, location, frame, .4);
    CHECK(derivative(5, 5) == 1);
    wall.kind = C::BoundaryKind::Analytic;
    pointOperator.ApproximateExteriorJacobian(
        derivative, wall, inside, normal, location, frame, .4);
    CHECK(derivative.cwiseAbs().maxCoeff() == 0);
}

TEST_CASE("RANS model and point gathering follow logical blocks across split and coupled layouts")
{
    for (const auto *name : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
    {
        const auto equation = Equation(name);
        const auto &model = equation.RANS();
        const I n = model.ComponentNumber();
        REQUIRE(equation.Schema().blocks.size() == 2);
        CHECK(equation.Schema().blocks[1].name == "rans");
        CHECK(equation.Schema().ResolveComponents({model.ComponentNames()[0]}) == std::vector<I>{5});
        auto policy = C::PrepareRANSStatePolicy(equation, {.01, 3});
        for (int layout = 0; layout < 3; ++layout)
            for (bool split : {false, true})
            {
                const A::DofSpace space(35, 0, "RANS point layouts");
                A::DofBundle fields(space);
                std::vector<C::EquationDofBinding> bindings;
                if (split)
                {
                    fields.Add("turb", A::DofField<double>(space, Layout(n, layout)));
                    fields.Add("fluid", A::DofField<double>(space, Layout(5, layout)));
                    bindings = {{"flow", "fluid", 0}, {"rans", "turb", 0}};
                }
                else
                {
                    fields.Add("coupled", A::DofField<double>(space, Layout(5 + n, layout)));
                    // RANS physically precedes flow, contrary to logical order.
                    bindings = {{"flow", "coupled", n}, {"rans", "coupled", 0}};
                }
                C::EquationDofs state(equation.Schema(), std::move(fields), std::move(bindings));
                state.RequireSchema(equation.Schema());
                {
                    auto write = state.WriteOwned();
                    auto flow = write.Block("flow"), scalars = write.Block("rans");
                    for (I iCell = 0; iCell < 35; ++iCell)
                    {
                        for (I j = 0; j < 5; ++j)
                            flow(iCell, j) = 2 + .01 * iCell + .1 * j;
                        for (I j = 0; j < n; ++j)
                            scalars(iCell, j) = .02 + .001 * iCell + .3 * j;
                    }
                }
                auto read = state.ReadOwned();
                const auto flow = read.Block("flow"), scalars = read.Block("rans");
                Eigen::Matrix<double, 5, 3> flowGradient;
                Eigen::Matrix<double, 2, 3> scalarGradient;
                for (int j = 0; j < 5; ++j)
                    for (int d = 0; d < 3; ++d)
                        flowGradient(j, d) = .01 * (1 + j - 2 * d);
                scalarGradient << .1, -.2, .3, -.1, .05, .08;
                for (I iCell = 0; iCell < 35; ++iCell)
                {
                    const auto point = C::PrepareRANSPoint(policy.components, flow[iCell], scalars[iCell], flowGradient, scalarGradient,
                                                           .002, 3.5, 1.4, .1, .03);
                    for (int j = 0; j < 5; ++j)
                    {
                        CHECK(point.state[j] == 2 + .01 * iCell + .1 * j);
                        for (int d = 0; d < 3; ++d)
                            CHECK(point.gradient(d, j) == flowGradient(j, d));
                    }
                    for (int j = 0; j < n; ++j)
                    {
                        CHECK(point.state[5 + j] == .02 + .001 * iCell + .3 * j);
                        for (int d = 0; d < 3; ++d)
                            CHECK(point.gradient(d, 5 + j) == scalarGradient(j, d));
                    }
                    if (n == 1)
                    {
                        CHECK(point.state[6] == 0);
                        CHECK(point.gradient.col(6).isZero());
                    }
                    CHECK(std::isfinite(model.Source(point)[0]));
                    CHECK(std::isfinite(model.Transport(point).viscosity));
                }
            }
    }
    auto base = C::EquationSystem::Resolve(C::EquationSettings{});
    DNDS::Physics::RANS::ModelSettings rans;
    rans.model = "SA";
    CHECK_THROWS(base.WithRANS(rans)); // Inviscid transport cannot host RANS.
    CHECK_FALSE(base.HasRANS());
    CHECK_THROWS(Equation("SA").WithRANS(rans));
}

TEST_CASE("RANS scalar repairs retain the Euler componentwise and exponential rules")
{
    for (const auto *name : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
    {
        const auto equation = Equation(name);
        const auto policy = C::PrepareRANSStatePolicy(equation, {.2, 4.}, 10., .02);
        const bool sa = policy.components == 1;
        const std::array<double, 2> mean{.2, 1.};
        std::array<double, 2> sample{-.1, 2.};
        C::RANSReconstructionView{policy.kind, policy.components}.RepairPoint(mean, sample);
#ifdef USE_NS_SA_NUT_REDUCED_ORDER
        CHECK(sample[0] == mean[0]);
#else
        CHECK(sample[0] == (sa ? -.1 : mean[0]));
#endif
        CHECK(sample[1] == 2.); // No global contraction of other components.
        std::array<double, 2> delta{-3., -2.};
        policy.RepairIncrement(mean, delta);
        for (int j = 0; j < policy.components; ++j)
        {
#ifdef USE_NS_SA_ALLOW_NEGATIVE_MEAN
            if (sa)
            {
                CHECK(delta[j] == -3.);
                continue;
            }
#endif
            const double raw = mean[j] * std::exp((j ? -2. : -3.) / (mean[j] + 1e-6));
            const double expected = sa ? std::max(1e-6, raw) : raw;
            CHECK(mean[j] + delta[j] == doctest::Approx(expected).epsilon(1e-11));
        }
        std::array<double, 2> next{100., .001};
        policy.RepairMean(2., next);
        CHECK(next[0] == (sa ? 20. : 100.));
        const bool omega = equation.RANS().Kind() == DNDS::Physics::RANS::ModelKind::KOmegaWilcox ||
                           equation.RANS().Kind() == DNDS::Physics::RANS::ModelKind::KOmegaSST;
        CHECK(next[1] == (omega ? .08 : .001));
    }
}

TEST_CASE("RANS reconstruction scales about the mean with a zero scalar floor")
{
    struct Samples
    {
        std::array<std::array<double, 3>, 2> values;
        double operator()(I q, I j) const { return values[q][j]; }
    };
    const Samples mean{std::array{std::array{9., .2, .4}, std::array{9., .2, .4}}};
    const Samples higher{std::array{std::array{9., -.2, .5}, std::array{9., .3, -.4}}};
    for (const auto *name : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
    {
        const auto policy = C::ResolveRANSReconstructionPolicy(Equation(name));
        double expected = .45; // .9 * min(.2/(.2+.2), .4/(.4+.4))
#ifndef USE_NS_SA_NUT_REDUCED_ORDER
        if (policy.kind == DNDS::Physics::RANS::ModelKind::SA)
            expected = 1;
#endif
        CHECK(policy.Compute(mean, higher, 1, 0, 2) == doctest::Approx(expected).epsilon(1e-14));
        CHECK(policy.Compute(mean, mean, 1, 0, 2) == 1);
        const Samples tiny{std::array{std::array{9., 1e-30, 1e-30}, std::array{9., 1e-30, 1e-30}}};
        CHECK(policy.Compute(tiny, tiny, 1, 0, 2) == 1);
    }
}

TEST_CASE("Named RANS initialization and numerical reference resolve against the model")
{
    C::RANSStateSettings settings;
    settings.saUpperLimit = 12;
    const auto sa = Equation("SA");
    CHECK(C::ResolveRANSValues(sa, {{"rhoNuTilde", .02}})[0] == .02);
    CHECK(C::ResolveRANSStatePolicy(sa, settings).saUpperLimit == 12);
    CHECK_THROWS(C::ResolveRANSValues(sa, {{"rhoK", .02}}));
    for (const auto *name : {"KOmegaWilcox", "KOmegaSST"})
    {
        const auto equation = Equation(name);
        settings.referenceConservative.clear();
        CHECK_THROWS(C::ResolveRANSStatePolicy(equation, settings));
        settings.referenceConservative = {{"rhoK", .1}, {"rhoOmega", 4}};
        CHECK(C::ResolveRANSStatePolicy(equation, settings).omegaLowerLimit == .04);
        settings.referenceConservative["rho"] = .25;
        CHECK(C::ResolveRANSReconstructionPolicy(equation).components == 2);
        CHECK(C::ResolveRANSStatePolicy(equation, settings).omegaLowerLimit == .04);
        const auto roundTrip = C::Json(settings).get<C::RANSStateSettings>();
        CHECK(roundTrip.referenceConservative == settings.referenceConservative);
        settings.referenceConservative["rho"] = 0;
        CHECK_THROWS(C::ResolveRANSStatePolicy(equation, settings));
    }
}

TEST_CASE("RANS no-slip targets retain conservative density and viscosity conventions")
{
    DNDS::Physics::RANS::Point mean;
    mean.state[0] = 2;
    mean.state[5] = .06;
    mean.wallDistance = .02;
    for (double mu : {.001, .004}) // Mean-temperature and wall-temperature producers.
    {
        mean.molecularViscosity = mu;
        for (const auto *name : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
        {
            const auto equation = Equation(name);
            const auto wall = C::CPU::RANSWallTarget(equation.RANS(), mean, 1.5);
            CHECK(wall[0] == 0);
            const double expected = std::string(name) == "SA" ? 0 : std::string(name) == "RealizableKE" ? 2 * mu / 2 * (.06 / 2) / (.02 * .02) * 1.5
                                                                                                        : 800 * mu / (.02 * .02);
            CHECK(wall[1] == doctest::Approx(expected));
        }
    }
}
