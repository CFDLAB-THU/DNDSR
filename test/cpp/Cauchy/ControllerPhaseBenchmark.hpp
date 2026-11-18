#pragma once

// Private benchmark workload, not an additional solver implementation. Reset
// trial/guess data outside nonlinear timers and check actual numerical work.
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Physics/ImplicitPositivity.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Solver/FieldTimeStepper.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace CauchyBenchmark
{
    template <class Measure>
    DNDS::Cauchy::Json ControllerPhases(DNDS::Cauchy::FiniteVolume &fv, DNDS::Cauchy::EquationDofs &frozenState,
                                        DNDS::Cauchy::FiniteVolume::ReconstructionState &reconstruction,
                                        DNDS::Cauchy::FiniteVolume::Workspace &workspace, DNDS::Runtime::EagerEngine &engine,
                                        const DNDS::Runtime::ExecutionStream &stream, DNDS::Cauchy::DomainSettings settings,
                                        bool checked, int repetitions, Measure &measure)
    {
        namespace C = DNDS::Cauchy;
        namespace R = DNDS::Runtime;
        namespace A = DNDS::Algebra;
        using I = DNDS::index;
        const DNDS::MPIInfo mpi(fv.Metrics().Communicator());
        const auto &metrics = fv.Metrics();
        auto &frozen = frozenState.CoupledField();
        const I rows = metrics.NumCell();
        I globalRows = rows;
        MPI_Allreduce(MPI_IN_PLACE, &globalRows, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
        const auto bounds = fv.ThermalConstraints();
        auto increment = metrics.BuildField(5), integrated = metrics.BuildField(5), corrected = metrics.BuildField(5);
        auto integratedState = frozenState.RebindCoupled(integrated);
        auto correctedState = frozenState.RebindCoupled(corrected);
        const C::ThermalPolicy thermalPolicy(fv.Equation(), settings.numerics.densityFloor,
                                             settings.numerics.temperatureFloor,
                                             settings.numerics.sensibleEnergyFloor,
                                             settings.numerics.pressureFloor);
        R::Array<double> fractions(R::RowLayout::Dense(rows, 1));
        I expectedActive = 0;
        double expectedAlpha = 1;
        std::vector<double> expectedFractions(rows, 1);
        {
            const auto geometry = metrics.AcquireRead();
            const auto metric = geometry.HostView();
            const auto read = frozen.Owned().Read();
            auto delta = increment.Owned().Write(), rhs = integrated.Owned().Write();
            const auto u = read.View<R::UniformTableRows>();
            const auto d = delta.View<R::UniformTableRows>(), r = rhs.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < rows; ++iCell)
            {
                double momentum2 = 0;
                for (int j = 1; j < 4; ++j)
                    momentum2 += u(iCell, j) * u(iCell, j);
                const double energy = u(iCell, 4) - .5 * momentum2 / (u(iCell, 0) + DNDS::verySmallReal);
                const bool active = iCell % 4 == 0;
                for (int j = 0; j < 5; ++j)
                {
                    d(iCell, j) = active && j == 4 ? -2 * energy : 0;
                    r(iCell, j) = d(iCell, j) * metric.GetCellVol(iCell);
                }
                if (!active)
                    continue;
                ++expectedActive;
                double fraction = 1;
                // Independent affine-energy oracle: density and momentum are
                // constant. Retain Euler's two explicit 1e-5 safety factors.
                for (int constraint = 0; constraint < 2; ++constraint)
                {
                    if (!constraint && !std::isfinite(bounds.energyDensityFloor))
                        continue;
                    const double base = constraint ? u(iCell, 0) * bounds.specificEnergyAtMinimumTemperature : 0;
                    const double floor = std::max(constraint ? 0. : bounds.energyDensityFloor,
                                                  DNDS::smallReal * std::max(energy - base, 0.));
                    fraction = std::min(fraction, (energy - base - floor) / (2 * energy) * (1 - 1e-5));
                }
                expectedAlpha = std::min(expectedAlpha, fraction * (1 - 1e-5));
                expectedFractions[iCell] = fraction * fraction * (1 - 1e-5);
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &expectedActive, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &expectedAlpha, 1, MPI_DOUBLE, MPI_MIN, mpi.comm);
        C::Json output;
        output["pp_fraction_loop"] = measure([&]
                                             {
            const auto input = frozen.Owned().Read(), delta = increment.Owned().Read();
            auto write = fractions.Write();
            const auto u = input.View<R::UniformTableRows>(), d = delta.View<R::UniformTableRows>();
            const auto a = write.View<R::FixedRows<1>>();
            for (I iCell = 0; iCell < rows; ++iCell)
            {
                C::FlowState mean{}, change{};
                for (int j = 0; j < 5; ++j) { mean[j] = u(iCell, j); change[j] = d(iCell, j); }
                a(iCell, 0) = C::FlowIncrementFraction(mean, change, bounds, 1, 2);
            } }, repetitions);
        double fractionError = 0;
        {
            const auto read = fractions.Read();
            const auto values = read.View<R::FixedRows<1>>();
            for (I iCell = 0; iCell < rows; ++iCell)
                fractionError = std::max(fractionError, std::abs(values(iCell, 0) - expectedFractions[iCell]));
        }
        MPI_Allreduce(MPI_IN_PLACE, &fractionError, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(fractionError < 2e-12, "PP fraction benchmark differs from its independent affine-energy oracle");
        output["pp_fraction_max_error"] = fractionError;
        output["pp_final_increment"] = measure([&]
                                               {
            const auto input = frozen.Owned().Read(), delta = increment.Owned().Read(), alpha = fractions.Read();
            auto write = corrected.Owned().Write();
            const auto u = input.View<R::UniformTableRows>(), d = delta.View<R::UniformTableRows>(), a = alpha.View<R::UniformTableRows>();
            const auto result = write.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < rows; ++iCell)
            {
                C::FlowState mean{}, change{};
                for (int j = 0; j < 5; ++j) { mean[j] = u(iCell, j); change[j] = a(iCell, 0) * d(iCell, j); }
                change = C::FixFlowIncrement(mean, change, bounds);
                for (int j = 0; j < 5; ++j) result(iCell, j) = mean[j] + change[j];
            } }, repetitions);
        DNDS_check_throw_info(fv.Admissible(correctedState), "benchmark PP final update is not admissible");
        {
            const auto original = frozen.Owned().Read(), changes = increment.Owned().Read(), result = corrected.Owned().Read();
            const auto u = original.View<R::UniformTableRows>(), d = changes.View<R::UniformTableRows>(), y = result.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < rows; ++iCell)
                for (int j = 0; j < 5; ++j)
                    DNDS_check_throw_info(std::abs(y(iCell, j) - (u(iCell, j) + expectedFractions[iCell] * d(iCell, j))) < 2e-12,
                                          "PP final update benchmark differs from its affine-energy oracle");
        }
        C::PhysicalStepLimit limit;
        output["pp_physical_step"] = measure([&]
                                             { limit = fv.LimitPhysicalStep(frozenState, integratedState, thermalPolicy.View(), 1, 1, 1); }, repetitions);
        DNDS_check_throw_info(limit.limitedCells == expectedActive && std::abs(limit.dt - expectedAlpha) < 2e-12,
                              "physical PP benchmark differs from its affine-energy oracle");
        output["pp_active_cells"] = expectedActive;
        output["pp_physical_dt"] = limit.dt;
        const auto &guess = fv.ReconstructionGuess(reconstruction);
        auto saved = guess.Clone(), synthetic = guess.Clone();
        auto &syntheticArray = synthetic.CoupledField().Owned();
        {
            const auto mean = frozen.Owned().Read();
            auto write = syntheticArray.Write();
            const auto u = mean.View<R::UniformTableRows>();
            const auto r = write.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < rows; ++iCell)
            {
                for (I j = 0; j < r.Width(); ++j)
                    r(iCell, j) = 0;
                // Energy-only perturbation keeps quadrature density positive.
                // A density-only trial reaches the density floor first; its
                // fixed momentum then makes Euler's energy bound tiny and
                // legitimately selects the zero-polynomial cutoff instead.
                double momentum2 = 0;
                for (int j = 1; j < 4; ++j)
                    momentum2 += u(iCell, j) * u(iCell, j);
                const double energy = u(iCell, 4) - .5 * momentum2 / (u(iCell, 0) + DNDS::verySmallReal);
                r(iCell, 4 * (r.Width() / 5)) = 4 * energy;
            }
        }
        fv.RestoreReconstructionGuess(reconstruction, synthetic);
        output["pp_quadrature"] = measure([&]
                                          { fv.LimitReconstructionPP(frozenState, reconstruction, workspace); }, repetitions);
        const auto [limited, beta] = fv.PositivityReconstructionStats(workspace);
        DNDS_check_throw_info(limited == globalRows && beta > 0 && beta < 1,
                              "quadrature PP benchmark did not exercise active limiting: cells=" + std::to_string(limited) + "/" + std::to_string(globalRows) + " beta=" + std::to_string(beta));
        output["pp_quadrature_cells"] = limited;
        output["pp_quadrature_beta"] = beta;
        fv.RestoreReconstructionGuess(reconstruction, saved);
        const auto local = [mpi, checked](const std::function<void()> &run)
        {
            if (checked)
                DNDS::MPI::LocalStage(mpi, run);
            else
                run();
        };
        for (const auto *method : {"bdf2", "esdirk4"})
        {
            auto initial = metrics.BuildField(5), trial = metrics.BuildField(5);
            A::Copy(initial, frozen);
            DNDS::ODE::FieldTimeStepper driver(initial, method, local);
            I evaluations = 0, solves = 0;
            const auto evaluate = [&](auto &, auto &rate, double, int)
            { ++evaluations; A::Fill(rate, 0.); };
            const auto solve = [&](auto &value, const auto &source, double mass, double, double, int)
            {
                ++solves;
                A::Copy(value, source);
                A::Scale(value, 1 / mass);
                return true;
            };
            const auto advance = [&]
            {
                driver.BeginStep(initial, trial, 0, .001, evaluate);
                for (int stage = 0; stage < driver.GroupNumber(); ++stage)
                {
                    const auto group = driver.PrepareGroup(trial);
                    if (group.implicit)
                    {
                        const auto equation = driver.PrepareEquation(group, trial, 0);
                        const auto &c = equation.coefficients;
                        DNDS_check_throw_info(
                            solve(equation.state, equation.resOther, c.inverseDt,
                                  c.rhsScale, c.time, group.stage),
                            "ODE benchmark stage failed");
                    }
                    driver.CompleteGroup(group, trial, evaluate);
                }
                A::Copy(initial, trial);
                driver.Commit();
            };
            advance(); // Warm history/tableau and allocator caches outside timing.
            evaluations = solves = 0;
            output[std::string("ode_") + method] = measure(advance, repetitions);
            const bool dirk = std::string(method) == "esdirk4";
            DNDS_check_throw_info(evaluations == repetitions * (dirk ? 6 : 0) && solves == repetitions * (dirk ? 5 : 1), "ODE benchmark performed the wrong number of stages");
            const auto expected = frozen.Owned().Read(), actual = initial.Owned().Read();
            const auto x = expected.View<R::UniformTableRows>(), y = actual.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < rows; ++iCell)
                for (int j = 0; j < 5; ++j)
                    DNDS_check_throw_info(std::abs(x(iCell, j) - y(iCell, j)) < 2e-12, "ODE benchmark changed its equilibrium solution");
        }
        // One real nonlinear correction, reset to identical inputs each time.
        // Source/reset setup is excluded; all internal Solve agreements remain.
        settings.implicit.collectiveChecks = checked;
        settings.implicit.linear.method = "sgs";
        settings.implicit.linear.iterations = 5;
        settings.implicit.linear.relativeTolerance = settings.implicit.linear.absoluteTolerance = 0;
        settings.implicit.positivity.rhsFPPMode = "dTau";
        auto trial = metrics.BuildField(5), source = metrics.BuildField(5);
        A::Copy(trial, frozen);
        auto trialState = C::EquationDofs::Coupled(fv.Equation().Schema(), trial);
        auto sourceState = C::EquationDofs::Coupled(fv.Equation().Schema(), source);
        auto work = fv.BuildWorkspace(trialState);
        auto trialReconstruction = fv.BuildReconstructionState(trialState);
        C::NonlinearSolver solver(
            fv, trialState, trialReconstruction, work, engine, stream, settings,
            A::ComponentSelection(trialState.ComponentNumber(),
                                  trialState.Schema().ResolveComponents(settings.implicit.ResolveResidualComponents(settings.equation.system == "incompressibleACM"))));
        solver.EvaluateEquation(trialState, sourceState, 1e5, 1, 0);
        C::NonlinearProfile accumulated;
        for (int repetition = 0; repetition < repetitions; ++repetition)
        {
            A::Copy(trial, frozen);
            fv.ResetReconstruction(*trialReconstruction, *work);
            {
                auto write = trial.Owned().Write();
                const auto value = write.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < rows; ++iCell)
                    value(iCell, 4) += .001 * value(iCell, 0);
            }
            MPI_Barrier(mpi.comm);
            auto profile = std::make_shared<C::NonlinearProfile>();
            const auto report = solver.Solve(trialState, sourceState, 1e5, 1, 0, 0, {1, 0, profile});
            DNDS_check_throw_info(report.iterations == 1 && report.linearIterations == 5 && profile->residualEvaluations == 2 &&
                                      report.residual[4] < report.initialResidual[4],
                                  "nonlinear benchmark did not perform the declared correction or reduce its defect");
            double sections = 0;
            for (int phase = 0; phase < C::NonlinearProfile::Total; ++phase)
                sections += profile->seconds[phase];
            DNDS_check_throw_info(profile->seconds[C::NonlinearProfile::Total] >= sections, "nonlinear benchmark sections overlap");
            for (int phase = 0; phase < C::NonlinearProfile::PhaseNumber; ++phase)
                accumulated.seconds[phase] += profile->seconds[phase] / repetitions;
        }
        std::array<double, C::NonlinearProfile::PhaseNumber + 1> maximum{}, average{};
        double remainder = accumulated.seconds[C::NonlinearProfile::Total];
        for (int phase = 0; phase < C::NonlinearProfile::PhaseNumber; ++phase)
        {
            maximum[phase] = accumulated.seconds[phase];
            if (phase < C::NonlinearProfile::Total)
                remainder -= accumulated.seconds[phase];
        }
        maximum.back() = remainder;
        average = maximum;
        MPI_Allreduce(MPI_IN_PLACE, maximum.data(), int(maximum.size()), MPI_DOUBLE, MPI_MAX, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, average.data(), int(average.size()), MPI_DOUBLE, MPI_SUM, mpi.comm);
        const char *names[]{"entry", "residual", "convergence", "pseudo_time_pp", "jacobian", "linear_solve", "correction", "total", "other_controller"};
        for (int phase = 0; phase <= C::NonlinearProfile::PhaseNumber; ++phase)
        {
            output["nonlinear_max"][names[phase]] = maximum[phase];
            output["nonlinear_mean"][names[phase]] = average[phase] / mpi.size;
        }
        output["nonlinear_residual_evaluations_per_call"] = 2;
        output["nonlinear_corrections_per_call"] = 1;
        output["nonlinear_linear_iterations_per_call"] = 5;
        return output;
    }
}
