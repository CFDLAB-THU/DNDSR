#pragma once

// Fixed-work cost evidence, not an alternative solver or an acceleration test.
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <cstring>

namespace CauchyBenchmark
{
    template <class Measure>
    DNDS::Cauchy::Json PolynomialPhases(DNDS::ssp<DNDS::Geom::UnstructuredMesh> mesh, DNDS::Cauchy::FiniteVolume &fv,
                                        const DNDS::CFV::FVField &frozen, DNDS::Runtime::EagerEngine &engine,
                                        const DNDS::Runtime::ExecutionStream &stream, DNDS::Cauchy::DomainSettings settings,
                                        bool checked, int repetitions, Measure &measure)
    {
        namespace C = DNDS::Cauchy;
        namespace R = DNDS::Runtime;
        namespace A = DNDS::Algebra;
        using I = DNDS::index;
        const DNDS::MPIInfo mpi(fv.Metrics().Communicator());
        const auto &metrics = fv.Metrics();
        constexpr double mass = 1e5;
        C::Json output = C::Json::array();
        for (int variant = 0; variant < 4; ++variant)
        {
            settings.implicit.collectiveChecks = checked;
            settings.implicit.minIterations = 0;
            settings.implicit.relativeTolerance = settings.implicit.absoluteTolerance = 0;
            settings.implicit.linear.method = "sgs";
            settings.implicit.linear.iterations = 5;
            settings.implicit.linear.relativeTolerance = settings.implicit.linear.absoluteTolerance = 0;
            settings.implicit.multigrid.levels = variant == 0 ? 0 : variant == 1 ? 1
                                                                                 : 2;
            settings.implicit.multigrid.cycles = 1;
            settings.implicit.multigrid.preSmooth = settings.implicit.multigrid.postSmooth = 1;
            settings.implicit.multigrid.coarseSmooth = 3;
            settings.implicit.multigrid.updateComponents = variant == 3 ? std::vector<I>{0, 4} : std::vector<I>{};
            auto trial = metrics.BuildField(5), source = metrics.BuildField(5), value = metrics.BuildField(5), defect = metrics.BuildField(5);
            auto trialState = C::EquationDofs::Coupled(fv.Equation().Schema(), trial);
            auto sourceState = C::EquationDofs::Coupled(fv.Equation().Schema(), source);
            auto valueState = C::EquationDofs::Coupled(fv.Equation().Schema(), value);
            std::shared_ptr<C::FiniteVolume::Workspace> workspace;
            std::shared_ptr<C::FiniteVolume::ReconstructionState> reconstruction;
            std::optional<C::NonlinearSolver> smoother;
            std::optional<C::PolynomialMultigrid> multigrid;
            const double preparation = measure([&]
                                               {
                workspace = fv.BuildWorkspace(trialState);
                reconstruction = fv.BuildReconstructionState(trialState);
                smoother.emplace(
                    fv, trialState, reconstruction, workspace, engine, stream, settings,
                    A::ComponentSelection(trialState.ComponentNumber(),
                                          trialState.Schema().ResolveComponents(settings.implicit.ResolveResidualComponents(settings.equation.system == "incompressibleACM"))));
                if (variant) multigrid.emplace(mesh, fv, trialState, *smoother, engine, stream, settings); });
            // The target changes density/energy only, so subset and full runs
            // solve the same finest equation from the same initial means.
            A::Copy(trial, frozen);
            {
                auto geometry = metrics.AcquireRead();
                const auto metric = geometry.HostView();
                auto write = trial.Owned().Write();
                const auto u = write.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < u.RowCount(); ++iCell)
                {
                    const auto point = metric.GetCellBary(iCell);
                    const double factor = 1 + .001 * (1 + .3 * std::sin(2 * std::acos(-1.) * (point[0] + point[1])));
                    for (int j : {0, 4})
                        u(iCell, j) *= factor;
                }
            }
            smoother->EvaluateEquation(trialState, sourceState, mass, 1, 0);
            const auto sourceVersion = source.Owned().Version();
            const auto norm = [&]
            {
                // Cold guesses make both independent residual evaluations
                // comparable. This evaluation/reset is outside Solve timing.
                fv.ResetReconstruction(*reconstruction, *workspace);
                smoother->EvaluateEquation(trialState, valueState, mass, 1, 0);
                A::LinearCombination(defect, 1., source, -1., value);
                return A::NormGlobal(defect);
            };
            C::Json row{{"variant", variant}, {"levels", settings.implicit.multigrid.levels}, {"selected", variant == 3}, {"solver_prepare_max_seconds", preparation}, {"calls", C::Json::array()}};
            for (int repetition = 0; repetition < repetitions; ++repetition)
            {
                A::Copy(trial, frozen);
                const double initial = norm();
                fv.ResetReconstruction(*reconstruction, *workspace);
                C::NonlinearIterationReport report;
                const double elapsed = measure([&]
                                               { report = multigrid ? multigrid->Solve(sourceState, mass, 1, 0) : smoother->Solve(trialState, sourceState, mass, 1, 0, 0, {5}); });
                const double final = norm();
                DNDS_check_throw_info(std::isfinite(initial) && initial > 0 && std::isfinite(final) && final < .1 * initial,
                                      "fixed-work pMG benchmark did not reduce its original full finest defect");
                DNDS_check_throw_info(report.iterations > 0 && report.iterations <= (variant > 1 ? 7 : 5) &&
                                          report.linearIterations > 0 && report.linearIterations <= 5 * report.iterations &&
                                          report.multigridCycles == (variant ? 1 : 0) &&
                                          report.levelCorrections.size() == (variant ? size_t(settings.implicit.multigrid.levels + 1) : size_t(0)),
                                      "fixed-work pMG benchmark reports unexpected work");
                DNDS::MPI::LocalStage(mpi, [&]
                                      {
                    DNDS_check_throw_info(source.Owned().Version() == sourceVersion, "pMG benchmark changed its source");
                    if (variant != 3) return;
                    const auto reference = frozen.Owned().Read(), actual = trial.Owned().Read();
                    const auto x = reference.View<R::UniformTableRows>(), y = actual.View<R::UniformTableRows>();
                    for (I iCell = 0; iCell < x.RowCount(); ++iCell)
                        for (int j : {1, 2, 3})
                            DNDS_check_throw_info(std::memcmp(&x(iCell, j), &y(iCell, j), sizeof(double)) == 0,
                                                  "pMG benchmark changed a frozen component"); });
                DNDS_check_throw_info(fv.Admissible(trialState), "pMG benchmark produced an inadmissible mean");
                row["calls"].push_back({{"repetition", repetition}, {"max_rank_seconds", elapsed}, {"nonlinear_corrections", report.iterations}, {"linear_iterations", report.linearIterations}, {"level_corrections", report.levelCorrections}, {"cycles", report.multigridCycles}, {"initial_full_defect", initial}, {"final_full_defect", final}, {"reported_initial_residual", report.initialResidual}, {"reported_residual", report.residual}, {"converged", report.converged}, {"frozen_components_preserved", true}});
            }
            output.push_back(std::move(row));
        }
        return output;
    }
}
