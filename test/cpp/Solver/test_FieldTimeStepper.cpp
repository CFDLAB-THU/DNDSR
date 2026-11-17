#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Solver/FieldTimeStepper.hpp"
#include "Solver/FieldDITR.hpp"
#include "Solver/TimeIntegration/DITRMethod.hpp"
#include <Eigen/LU>

namespace A = DNDS::Algebra;
namespace O = DNDS::ODE;
namespace R = DNDS::Runtime;
using Field = A::DofField<DNDS::real>;
namespace
{
    struct CountTimeAllocations : O::FieldTimeOperations
    {
        inline static int allocations = 0;
        static Field NewVector(const Field &prototype)
        {
            ++allocations;
            return O::FieldTimeOperations::NewVector(prototype);
        }
    };
    double Value(const Field &field)
    {
        const auto read = field.Owned().Read();
        return read.View<R::UniformTableRows>()(0, 0);
    }
    auto Evaluate()
    {
        return [](Field &state, Field &rate, double time, int)
        {
            const double value = Value(state);
            auto write = rate.Owned().Write();
            write.View()(0, 0) = -2 * value + 2 * std::cos(time) - std::sin(time);
        };
    }
    auto Solve()
    {
        return [](Field &state, const Field &source, double mass, double scale, double time, int)
        {
            const double value = (Value(source) + scale * (2 * std::cos(time) - std::sin(time))) / (mass + 2 * scale);
            auto write = state.Owned().Write();
            write.View()(0, 0) = value;
            return true;
        };
    }
    // Test-owned solve policy: the time method supplies equations, never the
    // nonlinear loop. Exact scalar solves retain the independent order oracle.
    template <class Stepper, class SolveStage>
    bool SolveGroup(Stepper &stepper, const typename Stepper::StageGroup &group,
                    Field &trial, const SolveStage &solve)
    {
        if (!group.implicit)
            return true;
        REQUIRE(group.coupledStages == 1);
        const auto equation = stepper.PrepareEquation(group, trial, 0);
        const auto &c = equation.coefficients;
        return solve(equation.state, equation.resOther, c.inverseDt,
                     c.rhsScale, c.time, group.stage);
    }
    template <class Stepper, class SolveStage>
    bool PrepareTrial(Stepper &stepper, const Field &initial, Field &trial,
                      double time, double dt, const SolveStage &solve,
                      bool steady = false)
    {
        stepper.BeginStep(initial, trial, time, dt, Evaluate(), steady);
        for (int stage = 0; stage < stepper.GroupNumber(); ++stage)
        {
            const auto group = stepper.PrepareGroup(trial);
            if (!SolveGroup(stepper, group, trial, solve))
                return false;
            stepper.CompleteGroup(group, trial, Evaluate());
        }
        return true;
    }
    double Error(const std::string &method, int steps)
    {
        const A::DofSpace space(1);
        Field state(space, 1), trial(space, 1);
        A::Fill(state, 1.);
        O::FieldTimeStepper stepper(state, method);
        const double dt = 1. / steps;
        for (int i = 0; i < steps; ++i)
        {
            const double oldValue = Value(state);
            REQUIRE(PrepareTrial(stepper, state, trial, i * dt, dt, Solve()));
            CHECK(Value(state) == oldValue);
            A::Copy(state, trial);
            stepper.Commit();
        }
        return std::abs(Value(state) - std::cos(1.));
    }
}
TEST_CASE("Time methods own only their relevant history and stage buffers")
{
    Field state(A::DofSpace(1), 1), trial(state.Space(), 1);
    A::Fill(state, 1.);
    using Stepper = O::StateTimeStepper<Field, CountTimeAllocations>;
    for (const std::string method : {"bdf1", "bdf2", "trapz", "esdirk4", "ditr"})
    {
        CountTimeAllocations::allocations = 0;
        Stepper stepper(state, method);
        const bool bdf = method == "bdf1" || method == "bdf2";
        const int expected = bdf ? 3 : method == "ditr" ? 6
                                                        : 2 + stepper.GroupNumber();
        CHECK(CountTimeAllocations::allocations == expected);
        CHECK_FALSE(stepper.HasHistory());
        if (!bdf)
            CHECK_THROWS(stepper.RestoreHistory(state, .1));
        if (method != "ditr")
        {
            REQUIRE(PrepareTrial(stepper, state, trial, 0, .1, Solve()));
            stepper.Commit();
            CHECK(stepper.HasHistory() == bdf);
            CHECK(CountTimeAllocations::allocations == expected);
        }
    }
}
TEST_CASE("DITR transformed algebraic rate equations have independent midpoint and endpoint rows")
{
    constexpr double a = .55, theta = .9146;
    for (auto variant : {O::DITRVariant::U2R1, O::DITRVariant::U2R2})
    {
        const O::DITRCoefficients c(a, variant);
        Eigen::Matrix2d matrix;
        matrix << theta * c.integration[1], c.interpolation[3] + theta * c.integration[2],
            c.integration[1], c.integration[2];
        CHECK(matrix.determinant() == doctest::Approx((variant == O::DITRVariant::U2R1 ? 1. : a) / 6));
        const auto norm = [](const Eigen::Matrix2d &m)
        { return m.cwiseAbs().rowwise().sum().maxCoeff(); };
        CHECK(c.AlgebraicReciprocalCondition(theta) == doctest::Approx(1 / (norm(matrix) * norm(matrix.inverse()))));
    }
}
TEST_CASE("Public DITR method preserves Euler one-correction sequence and I plus H stopping")
{
    const A::DofSpace space(1);
    using Method = O::DITRMethod<Field, O::FieldTimeOperations>;
    for (const auto variant : {O::DITRVariant::U2R2, O::DITRVariant::U2R1})
    {
        CAPTURE(int(variant));
        Field initial(space, 1), endpoint(space, 1), stop(space, 1), rate(space, 1), defect(space, 1);
        A::Fill(initial, 1.);
        Method method(initial, .5, .9146, variant);
        std::vector<int> visits;
        const auto f = [](double u, double t)
        { return -2 * u + t + 1; };
        const auto evaluate = [&](Field &u, Field &out, double time, int coupledStage)
        {
            visits.push_back(coupledStage);
            A::Fill(out, f(Value(u), time));
        };
        method.Begin(initial, endpoint, .3, .2, evaluate);
        CHECK(visits == std::vector<int>{1}); // Initial RHS reused for both first guesses.
        CHECK_THROWS(method.PrepareCoupledStage(endpoint, 1));
        // Independent a=1/2 polynomial coefficients, not the production builder.
        const std::array<double, 4> c = variant == O::DITRVariant::U2R2
                                            ? std::array<double, 4>{.5, .5, .125, -.125}
                                            : std::array<double, 4>{.25, .75, 0, -.25};
        const double w0 = 1. / 6, w1 = 2. / 3, w2 = 1. / 6;
        double middle = 1, end = 1, fm = f(1, .3), fe = fm;
        const double f0 = fm;
        for (int iteration = 0; iteration < 2; ++iteration)
        {
            for (int coupledStage = 0; coupledStage < 2; ++coupledStage)
            {
                const auto equation = method.PrepareCoupledStage(endpoint, coupledStage);
                const double expectedSource = coupledStage == 0
                                                  ? 5 * (c[0] + .9146) + 5 * (c[1] - .9146) * end + (c[2] + .9146 * w0) * f0 + (c[3] + .9146 * w2) * fe
                                                  : 5 + w0 * f0 + w1 * fm;
                const double alpha = coupledStage == 0 ? .9146 * w1 : w2;
                const double oldValue = coupledStage == 0 ? middle : end;
                const double oldRate = coupledStage == 0 ? fm : fe;
                CHECK(equation.coupledStage == coupledStage);
                CHECK(equation.coefficients.inverseDt == 5);
                CHECK(equation.coefficients.rhsScale == doctest::Approx(alpha));
                CHECK(equation.coefficients.time == doctest::Approx(coupledStage == 0 ? .4 : .5));
                CHECK(Value(equation.resOther) == doctest::Approx(expectedSource));
                A::Fill(rate, oldRate);
                equation.coefficients.AssembleResidual<O::FieldTimeOperations>(defect, equation.resOther, equation.state, rate);
                const double expectedDefect = expectedSource + alpha * oldRate - 5 * oldValue;
                CHECK(Value(defect) == doctest::Approx(expectedDefect));
                const double next = oldValue + .6 * expectedDefect / (5 + 2 * alpha);
                A::Fill(equation.state, next); // Exactly one relaxed correction, not a converged solve.
                CHECK(Value(equation.resOther) == doctest::Approx(expectedSource));
                if (coupledStage == 0)
                {
                    middle = next;
                    fm = f(middle, .4);
                }
                else
                {
                    end = next;
                    fe = f(end, .5);
                }
                method.CompleteCoupledStage(endpoint, evaluate);
            }
            method.AssembleStoppingResidual(endpoint, stop);
            const double h = 5 * (c[0] + c[1] * end - middle) + c[2] * f0 + c[3] * fe;
            const double i = 5 * (1 - end) + w0 * f0 + w1 * fm + w2 * fe;
            CHECK(Value(stop) == doctest::Approx(i + h));
            CHECK(Value(endpoint) == doctest::Approx(end));
            CHECK(Value(initial) == 1);
        }
        CHECK(visits == std::vector<int>{1, 0, 1, 0, 1});
        const double result = Value(endpoint);
        method.Discard();
        method.Begin(initial, endpoint, .3, .2, evaluate);
        CHECK(Value(endpoint) == 1);
        CHECK(result != 1);
        method.Discard();
        method.IntermediateState() = initial;
        CHECK_THROWS(method.Begin(initial, endpoint, .3, .2, evaluate));
        CHECK(Value(initial) == 1);
    }
}

TEST_CASE("Public normalized BDF sources exclude the current unknown and use accepted step ratios")
{
    const A::DofSpace space(1);
    Field initial(space, 1), previous(space, 1), source(space, 1), current(space, 1), rate(space, 1), residual(space, 1);
    A::Fill(initial, 3.);
    A::Fill(previous, 2.);
    A::Fill(current, 4.);
    A::Fill(rate, 7.);
    const O::BDFCoefficients bdf(2, 1, .5, .25);
    bdf.AssembleResOther<O::FieldTimeOperations>(source, initial, &previous);
    CHECK(bdf.equation.inverseDt == 2);
    CHECK(bdf.equation.rhsScale == doctest::Approx(.6));
    CHECK(Value(source) == doctest::Approx(7.6));
    bdf.equation.AssembleResidual<O::FieldTimeOperations>(residual, source, current, rate);
    CHECK(Value(residual) == doctest::Approx(3.8));
    A::Fill(current, 6.);
    bdf.equation.AssembleResidual<O::FieldTimeOperations>(residual, source, current, rate);
    CHECK(Value(residual) == doctest::Approx(-.2));
    CHECK(Value(source) == doctest::Approx(7.6));
    const O::BDFCoefficients steady(1, 1, 0, 0, true);
    steady.AssembleResOther<O::FieldTimeOperations>(source, initial);
    CHECK(Value(source) == 0);
    steady.equation.AssembleResidual<O::FieldTimeOperations>(residual, source, current, rate);
    CHECK(Value(residual) == 7);
}

TEST_CASE("Native implicit time methods converge with independent nonautonomous exact stages")
{
    const std::vector<std::pair<std::string, double>> methods{{"bdf1", .8}, {"bdf2", 1.8}, {"trapz", 1.9}, {"esdirk2", 1.8}, {"esdirk3", 2.8}, {"esdirk4", 3.5}};
    for (const auto &[method, minimumOrder] : methods)
    {
        const double coarse = Error(method, 8), fine = Error(method, 16);
        CAPTURE(method);
        CAPTURE(coarse);
        CAPTURE(fine);
        CHECK(coarse > 0);
        CHECK(fine > 0);
        CHECK(std::log2(coarse / fine) >= minimumOrder);
    }
}
TEST_CASE("Time group tickets reject foreign sessions and reused scratch generations")
{
    const A::DofSpace space(1);
    Field state(space, 1), trial(space, 1), otherTrial(space, 1);
    A::Fill(state, 1.);
    O::FieldTimeStepper stepper(state, "bdf1"), other(state, "bdf1");
    stepper.BeginStep(state, trial, 0, .1);
    const auto stale = stepper.PrepareGroup(trial);
    other.BeginStep(state, otherTrial, 0, .1);
    const auto foreign = other.PrepareGroup(otherTrial);
    CHECK_THROWS(stepper.ValidateGroup(foreign, trial));
    stepper.Discard();
    CHECK_THROWS(stepper.ValidateGroup(stale, trial));
    stepper.BeginStep(state, trial, 0, .2);
    const auto current = stepper.PrepareGroup(trial);
    CHECK_THROWS(stepper.ValidateGroup(stale, trial));
    REQUIRE(SolveGroup(stepper, current, trial, Solve()));
    stepper.CompleteGroup(current, trial);
    CHECK_THROWS(stepper.ValidateGroup(current, trial));
}

TEST_CASE("External controller interleaves coherent stages without changing native time equations")
{
    const A::DofSpace space(1);
    for (const std::string method : {"bdf2", "trapz", "esdirk2", "esdirk3", "esdirk4"})
    {
        CAPTURE(method);
        Field state(space, 1), trial(space, 1), reference(space, 1), expected(space, 1);
        A::Fill(state, 1.);
        A::Fill(reference, 1.);
        O::FieldTimeStepper external(state, method), serial(reference, method);
        CHECK_THROWS(external.ActiveGroup());
        double time = 0;
        for (const double dt : {.1, .05})
        {
            REQUIRE(PrepareTrial(serial, reference, expected, time, dt, Solve()));
            external.BeginStep(state, trial, time, dt);
            const auto generation = state.Owned().Version();
            for (int stage = 0; stage < external.GroupNumber(); ++stage)
            {
                CHECK_FALSE(external.HasPendingTrial());
                const auto equation = external.PrepareGroup(trial);
                CHECK(equation.stage == stage);
                CHECK(equation.times[0] >= time);
                CHECK(equation.times[0] <= time + dt);
                CHECK(Value(trial) == Value(state));
                // A real controller prepares other domains and exchanges their
                // traces here, before any local stage solver changes its input.
                if (equation.implicit)
                    REQUIRE(SolveGroup(external, equation, trial, Solve()));
                external.CompleteGroup(equation, trial, Evaluate());
                if (equation.implicit)
                    CHECK_THROWS(external.ValidateGroup(equation, trial));
            }
            CHECK(external.HasPendingTrial());
            CHECK(state.Owned().Version() == generation);
            CHECK(Value(trial) == Value(expected));
            A::Copy(state, trial);
            A::Copy(reference, expected);
            external.Commit();
            serial.Commit();
            CHECK(external.PreviousDt() == (method == "bdf1" || method == "bdf2" ? dt : 0));
            time += dt;
        }
    }
}
TEST_CASE("Serialized BDF and DIRK stages restore their trial source and completed rate prefix")
{
    const A::DofSpace space(1);
    for (const std::string method : {"bdf2", "esdirk4"})
    {
        CAPTURE(method);
        Field initial(space, 1), trial(space, 1), restoredTrial(space, 1);
        A::Fill(initial, .8);
        O::FieldTimeStepper original(initial, method), restored(initial, method);
        original.BeginStep(initial, trial, .2, .1);
        std::optional<O::FieldTimeStepper::StageGroup> active;
        while (!active)
        {
            auto equation = original.PrepareGroup(trial);
            if (equation.implicit)
                active.emplace(equation);
            else
                original.CompleteGroup(equation, trial, Evaluate());
        }
        A::Fill(trial, .73); // a nonlinear iterate, not the stage initial guess
        const auto checkpoint = original.CaptureActive(trial);
        restored.RestoreActive(checkpoint, restoredTrial);
        CHECK(Value(restoredTrial) == Value(trial));

        const auto complete = [&](auto &stepper, auto &state, auto equation)
        {
            REQUIRE(SolveGroup(stepper, equation, state, Solve()));
            stepper.CompleteGroup(equation, state, Evaluate());
            for (int stage = equation.stage + 1; stage < stepper.GroupNumber(); ++stage)
            {
                const auto next = stepper.PrepareGroup(state);
                if (next.implicit)
                    REQUIRE(SolveGroup(stepper, next, state, Solve()));
                stepper.CompleteGroup(next, state, Evaluate());
            }
            REQUIRE(stepper.HasPendingTrial());
        };
        complete(original, trial, *active);
        complete(restored, restoredTrial, restored.ActiveGroup());
        CHECK(Value(restoredTrial) == Value(trial));
    }
}
TEST_CASE("Native BDF1 steady limit has zero physical mass and history source at a frozen time")
{
    const A::DofSpace space(2);
    Field initial(space, 2), trial(space, 2);
    A::Fill(initial, 3.);
    O::FieldTimeStepper stepper(initial, "bdf1");
    stepper.RestoreHistory(initial, .125);
    Eigen::Matrix2d matrix;
    matrix << -2, 1, .5, -3;
    const Eigen::Vector2d forcing(1, 2), exact = matrix.fullPivLu().solve(-forcing);
    const auto version = initial.Owned().Version();
    int calls = 0;
    const auto solve = [&](Field &state, const Field &source, double mass, double scale, double time, int stage)
    {
        ++calls;
        CHECK(mass == 0);
        CHECK(scale == 1);
        CHECK(time == 2.75);
        CHECK(stage == 0);
        const auto read = source.Owned().Read();
        auto write = state.Owned().Write();
        for (DNDS::index iRow = 0; iRow < 2; ++iRow)
            for (int j = 0; j < 2; ++j)
            {
                CHECK(read.View()(iRow, j) == 0);
                write.View()(iRow, j) = exact[j];
            }
        return calls != 1; // Rejected preparation must preserve accepted history.
    };
    CHECK_FALSE(PrepareTrial(stepper, initial, trial, 2.75, 0, solve, true));
    CHECK(stepper.HasHistory());
    CHECK(stepper.PreviousDt() == .125);
    CHECK(initial.Owned().Version() == version);
    CHECK_THROWS(stepper.Commit());
    stepper.Discard();
    REQUIRE(PrepareTrial(stepper, initial, trial, 2.75, 0, solve, true));
    CHECK(initial.Owned().Version() == version);
    stepper.Commit();
    CHECK_FALSE(stepper.HasHistory());
    CHECK(stepper.PreviousDt() == 0);
    const auto read = trial.Owned().Read();
    for (DNDS::index iRow = 0; iRow < 2; ++iRow)
        for (int j = 0; j < 2; ++j)
            CHECK(read.View()(iRow, j) == doctest::Approx(exact[j]));
    O::FieldTimeStepper wrong(initial, "esdirk4");
    CHECK_THROWS(PrepareTrial(wrong, initial, trial, 2.75, 0, solve, true));
    CHECK_THROWS(PrepareTrial(stepper, initial, initial, 2.75, 0, solve, true));
}
TEST_CASE("Native time target comparison excludes only relative roundoff remainders")
{
    CHECK(O::FieldTimeStepper::ReachedTimeTarget(300 - 2e-13, 300));
    CHECK(O::FieldTimeStepper::ReachedTimeTarget(300, 300));
    CHECK_FALSE(O::FieldTimeStepper::ReachedTimeTarget(299.9, 300));
    CHECK_FALSE(O::FieldTimeStepper::ReachedTimeTarget(0, 1e-20));
    CHECK_FALSE(O::FieldTimeStepper::ReachedTimeTarget(1, 1.00001));
    for (const auto [steps, dt] : {std::pair<int, double>{800, .00125}, {2500, .1}})
    {
        double time = 0;
        for (int iStep = 0; iStep < steps - 1; ++iStep)
            time += dt;
        CHECK_FALSE(O::FieldTimeStepper::ReachedTimeTarget(time, steps * dt));
        time += dt;
        CHECK(O::FieldTimeStepper::ReachedTimeTarget(time, steps * dt));
    }
}
TEST_CASE("Native BDF history is committed explicitly and retries use the last accepted step ratio")
{
    const A::DofSpace space(1);
    Field state(space, 1), trial(space, 1);
    A::Fill(state, 1.);
    O::FieldTimeStepper stepper(state, "bdf2");
    CHECK_FALSE(stepper.HasHistory());
    CHECK_THROWS(stepper.PreviousState());
    CHECK_THROWS(stepper.Commit());
    CHECK(PrepareTrial(stepper, state, trial, 0, .1, Solve()));
    const double first = Value(trial);
    stepper.Discard();
    CHECK_FALSE(stepper.HasHistory());
    CHECK_THROWS(stepper.Commit());
    CHECK(PrepareTrial(stepper, state, trial, 0, .1, Solve()));
    CHECK(Value(trial) == first);
    A::Copy(state, trial);
    stepper.Commit();
    CHECK(stepper.HasHistory());
    CHECK(stepper.PreviousDt() == .1);
    CHECK(Value(stepper.PreviousState()) == 1);
    // Rejected .2 trial must not replace accepted .1 history. The next .05
    // trial has r=.5, independent of the failed attempt's r=2.
    CHECK_FALSE(PrepareTrial(stepper, state, trial, .1, .2, [](auto &, const auto &, auto, auto, auto, auto)
                             { return false; }));
    CHECK(stepper.PreviousDt() == .1);
    CHECK(Value(stepper.PreviousState()) == 1);
    double observedMass = 0, observedScale = 0, observedSource = 0;
    CHECK(PrepareTrial(stepper, state, trial, .1, .05, [&](Field &x, const Field &source, double mass, double scale, double time, int stage)
                       {
        observedMass = mass; observedScale = scale; observedSource = Value(source);
        return Solve()(x, source, mass, scale, time, stage); }));
    CHECK(observedMass == 20);
    CHECK(observedScale == .75);
    CHECK(observedSource == doctest::Approx((1.125 * first - .125) * 20));
    O::FieldTimeStepper restored(state, "bdf2");
    restored.RestoreHistory(stepper.PreviousState(), stepper.PreviousDt());
    Field other(space, 1);
    CHECK(PrepareTrial(restored, state, other, .1, .05, Solve()));
    CHECK(Value(other) == Value(trial));
    stepper.ResetHistory();
    CHECK_FALSE(stepper.HasHistory());
    CHECK_THROWS(PrepareTrial(stepper, state, state, .1, .05, Solve()));
    CHECK_THROWS(O::FieldTimeStepper(state, "unknown"));
}
namespace
{
    // Test-side controller. The public method deliberately contains no loop or
    // convergence decision. Each visit is one linear correction using its
    // cached physical RHS, including the initial-time first-iteration guesses.
    struct DITRResult
    {
        bool converged;
        int iterations;
    };
    DITRResult SolveDITR(O::FieldDITR &method, Field &endpoint, int limit = 100, double tolerance = 2e-13)
    {
        auto defect = O::FieldTimeOperations::NewVector(endpoint);
        for (int iteration = 1; iteration <= limit; ++iteration)
        {
            for (int coupledStage = 0; coupledStage < 2; ++coupledStage)
            {
                const auto equation = method.PrepareCoupledStage(endpoint, coupledStage);
                const auto &c = equation.coefficients;
                c.AssembleResidual<O::FieldTimeOperations>(defect, equation.resOther, equation.state, method.CurrentRate());
                const double next = Value(equation.state) + Value(defect) / (c.inverseDt + 2 * c.rhsScale);
                A::Fill(equation.state, next);
                method.CompleteCoupledStage(endpoint, Evaluate());
            }
            method.AssembleStoppingResidual(endpoint, defect);
            if (std::abs(Value(defect)) < tolerance)
                return {true, iteration};
        }
        return {false, limit};
    }
    void AdvanceDITROnce(O::FieldTimeStepper &stepper,
                         const O::FieldTimeStepper::StageGroup &group,
                         Field &endpoint, Field &defect)
    {
        for (int coupledStage = 0; coupledStage < 2; ++coupledStage)
        {
            const auto equation = stepper.PrepareEquation(group, endpoint, coupledStage);
            const auto &c = equation.coefficients;
            const auto cached = stepper.CachedRate(group);
            REQUIRE(cached);
            c.AssembleResidual<O::FieldTimeOperations>(defect, equation.resOther, equation.state, *cached);
            A::Fill(equation.state, Value(equation.state) + Value(defect) / (c.inverseDt + 2 * c.rhsScale));
            stepper.CompleteEquation(group, endpoint, Evaluate());
        }
        stepper.AssembleStoppingResidual(group, endpoint, defect);
    }
    DITRResult SolveDITR(O::FieldTimeStepper &stepper,
                         const O::FieldTimeStepper::StageGroup &group,
                         Field &endpoint, int limit = 100, double tolerance = 2e-13)
    {
        auto defect = O::FieldTimeOperations::NewVector(endpoint);
        for (int iteration = 1; iteration <= limit; ++iteration)
        {
            AdvanceDITROnce(stepper, group, endpoint, defect);
            if (std::abs(Value(defect)) < tolerance)
                return {true, iteration};
        }
        return {false, limit};
    }
}
TEST_CASE("Public DITR converges to the independent original two-equation dense oracle")
{
    for (const auto variant : {O::DITRVariant::U2R2, O::DITRVariant::U2R1})
    {
        const A::DofSpace space(1);
        Field state(space, R::RowLayout::TiledColMajor(0, 1, 8)), trial(space, 1);
        A::Fill(state, .7);
        O::FieldDITR method(state, .55, .9146, variant);
        const double t = .3, h = .2, a = .55;
        // Fixed coefficients independent of the production coefficient builder.
        const bool u2r1 = variant == O::DITRVariant::U2R1;
        const double c0 = u2r1 ? .2025 : .42525, c1 = u2r1 ? .7975 : .57475;
        const double c2 = u2r1 ? 0 : .111375, c3 = u2r1 ? -.2475 : -.136125;
        const double w0 = 13. / 66, w1 = 200. / 297, w2 = 7. / 54;
        const auto g = [](double x)
        { return 2 * std::cos(x) - std::sin(x); };
        Eigen::Matrix2d matrix;
        matrix << 1 / h, -c1 / h + 2 * c3, 2 * w1, 1 / h + 2 * w2;
        Eigen::Vector2d forcing;
        forcing << c0 * .7 / h + c2 * (-2 * .7 + g(t)) + c3 * g(t + h),
            .7 / h + w0 * (-2 * .7 + g(t)) + w1 * g(t + a * h) + w2 * g(t + h);
        const Eigen::Vector2d exact = matrix.fullPivLu().solve(forcing);
        method.Begin(state, trial, t, h, Evaluate());
        const auto report = SolveDITR(method, trial);
        REQUIRE(report.converged);
        CHECK(report.iterations > 0);
        CHECK(Value(trial) == doctest::Approx(exact[1]).epsilon(2e-12));
        CHECK(Value(method.IntermediateState()) == doctest::Approx(exact[0]).epsilon(2e-12));
        const Eigen::Vector2d actual(Value(method.IntermediateState()), Value(trial));
        CHECK((matrix * actual - forcing).lpNorm<Eigen::Infinity>() < 1e-11);
        CHECK(Value(state) == .7);
        const double first = Value(trial);
        method.Discard();
        method.Begin(state, trial, t, h, Evaluate());
        REQUIRE(SolveDITR(method, trial).converged);
        CHECK(Value(trial) == first);
    }
}
TEST_CASE("Public DITR retains nonautonomous temporal order and controller-owned capped retry")
{
    const auto error = [](int steps, double fraction)
    {
        const A::DofSpace space(1);
        Field state(space, 1), trial(space, 1);
        A::Fill(state, 1.);
        O::FieldDITR method(state, fraction);
        for (int step = 0; step < steps; ++step)
        {
            method.Begin(state, trial, double(step) / steps, 1. / steps, Evaluate());
            REQUIRE(SolveDITR(method, trial).converged);
            A::Copy(state, trial);
        }
        return std::abs(Value(state) - std::cos(1.));
    };
    for (double fraction : {.55, .5})
    {
        const double coarse = error(32, fraction), fine = error(64, fraction);
        CAPTURE(fraction);
        CAPTURE(coarse);
        CAPTURE(fine);
        CHECK(coarse > 0);
        CHECK(fine > 0);
        CHECK(std::log2(coarse / fine) > (fraction == .5 ? 3.7 : 2.9));
    }
    const A::DofSpace space(1);
    Field state(space, 1), trial(space, 1);
    A::Fill(state, 1.);
    O::FieldDITR method(state);
    method.Begin(state, trial, 0, .2, Evaluate());
    const auto cap = SolveDITR(method, trial, 5, 2e-10);
    CHECK_FALSE(cap.converged);
    CHECK(cap.iterations == 5);
    CHECK(Value(state) == 1);
    const double capped = Value(trial);
    // The controller may publish a capped finite state or discard it; the
    // method makes neither decision and retains no accepted-step history.
    method.Discard();
    method.Begin(state, trial, 0, .2, Evaluate());
    CHECK_FALSE(SolveDITR(method, trial, 5, 2e-10).converged);
    CHECK(Value(trial) == capped);
    method.Discard();
    method.Begin(state, trial, 0, .001, Evaluate());
    CHECK(SolveDITR(method, trial, 5, 2e-10).converged);
    CHECK(Value(trial) == doctest::Approx(std::cos(.001)).epsilon(2e-10));
}
TEST_CASE("DITR rejects rebound intermediate and stateful rate callbacks without touching accepted state")
{
    const A::DofSpace space(1);
    Field state(space, 1), trial(space, 1);
    A::Fill(state, 1.);
    O::FieldDITR method(state);
    const auto generation = state.Owned().Version();
    const auto bad = [](Field &input, Field &output, double, int)
    { A::Scale(input, 2.); A::Copy(output, input); };
    CHECK_THROWS(method.Begin(state, trial, 0, .1, bad));
    CHECK(state.Owned().Version() == generation);
    method.Discard();
    method.Begin(state, trial, 0, .1, Evaluate());
    CHECK(SolveDITR(method, trial).converged);
    method.IntermediateState() = state;
    CHECK_THROWS(method.Begin(state, trial, 0, .1, Evaluate()));
    CHECK(Value(state) == 1);
    CHECK_THROWS(O::DITRCoefficients(0));
    CHECK_THROWS(O::DITRCoefficients(1));
}
TEST_CASE("Time facade reserves DITR publication while the external controller owns convergence")
{
    const A::DofSpace space(1);
    Field state(space, 1), trial(space, 1);
    A::Fill(state, 1.);
    O::FieldTimeStepper stepper(state, "ditr");
    CHECK_THROWS(stepper.BeginStep(state, trial, 0, .1));
    stepper.BeginStep(state, trial, 0, .1, Evaluate());
    const auto group = stepper.PrepareGroup(trial);
    CHECK(group.coupledStages == 2);
    REQUIRE(SolveDITR(stepper, group, trial).converged);
    stepper.CompleteGroup(group, trial);
    CHECK(stepper.HasPendingTrial());
    CHECK_FALSE(stepper.HasHistory());
    const double expected = Value(trial);
    stepper.Discard();
    CHECK_FALSE(stepper.HasPendingTrial());
    CHECK_THROWS(stepper.Commit());
    stepper.BeginStep(state, trial, 0, .1, Evaluate());
    const auto retry = stepper.PrepareGroup(trial);
    REQUIRE(SolveDITR(stepper, retry, trial).converged);
    CHECK(Value(trial) == expected);
    auto fresh = stepper.Fresh();
    Field freshTrial(space, 1);
    fresh->BeginStep(state, freshTrial, 0, .1, Evaluate());
    const auto freshGroup = fresh->PrepareGroup(freshTrial);
    CHECK(fresh->GroupState(freshGroup, freshTrial, 0).Owned().SharedStorage() !=
          stepper.GroupState(retry, trial, 0).Owned().SharedStorage());
    REQUIRE(SolveDITR(*fresh, freshGroup, freshTrial).converged);
    CHECK(Value(freshTrial) == expected);
    fresh->CompleteGroup(freshGroup, freshTrial);
    A::Copy(state, freshTrial);
    fresh->Commit();
    CHECK_FALSE(fresh->HasHistory()); // DITR is a one-step method, not BDF history.
    CHECK_FALSE(fresh->HasPendingTrial());
    fresh->BeginStep(state, freshTrial, .1, .1, Evaluate());
    const auto rejected = fresh->PrepareGroup(freshTrial);
    CHECK_FALSE(SolveDITR(*fresh, rejected, freshTrial, 1).converged);
    CHECK_FALSE(fresh->HasPendingTrial());
    fresh->Discard();
    CHECK(Value(state) == expected);
}
TEST_CASE("Time facade restores DITR coupled state at a completed iteration boundary")
{
    const A::DofSpace space(1);
    Field initial(space, 1), trial(space, 1), restoredTrial(space, 1);
    A::Fill(initial, .7);
    O::FieldTimeStepper original(initial, "ditr"), restored(initial, "ditr");
    original.BeginStep(initial, trial, .3, .2, Evaluate());
    const auto group = original.PrepareGroup(trial);
    auto defect = O::FieldTimeOperations::NewVector(trial);
    AdvanceDITROnce(original, group, trial, defect);
    const auto checkpoint = original.CaptureActive(trial);
    auto differentMethod = checkpoint;
    REQUIRE(differentMethod.ditrSettings);
    differentMethod.ditrSettings->fraction = .5;
    A::Fill(restoredTrial, -.33);
    CHECK_THROWS_WITH(restored.RestoreActive(differentMethod, restoredTrial),
                      doctest::Contains("DITR coefficients differ"));
    CHECK(Value(restoredTrial) == -.33);
    restored.RestoreActive(checkpoint, restoredTrial);
    CHECK(Value(restoredTrial) == Value(trial));
    const auto resumed = restored.ActiveGroup();
    REQUIRE(SolveDITR(original, group, trial).converged);
    REQUIRE(SolveDITR(restored, resumed, restoredTrial).converged);
    original.CompleteGroup(group, trial);
    restored.CompleteGroup(resumed, restoredTrial);
    CHECK(Value(restoredTrial) == Value(trial));
}
