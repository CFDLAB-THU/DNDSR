#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Implicit/ResidualConvergence.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "CartesianFixture.hpp"
#include <cstring>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
static DNDS::MPIInfo mpi;
namespace
{
    C::EquationDofs Bind(const C::FiniteVolume &fv, const DNDS::CFV::FVField &field)
    {
        return C::EquationDofs::Coupled(fv.Equation().Schema(), field);
    }
    C::NonlinearSolver MakeSolver(const C::FiniteVolume &fv, C::EquationDofs &state,
                                  R::EagerEngine &engine, const R::ExecutionStream &stream,
                                  const C::DomainSettings &settings)
    {
        return {fv, state, fv.BuildReconstructionState(state), fv.BuildWorkspace(state), engine, stream, settings,
                A::ComponentSelection(state.ComponentNumber(),
                                      state.Schema().ResolveComponents(settings.implicit.ResolveResidualComponents(settings.equation.system == "incompressibleACM")))};
    }
    C::DomainSettings Controls()
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 160;
        settings.numerics.reconstructionTolerance = 1e-12;
        settings.implicit.minIterations = 0;
        settings.implicit.relativeTolerance = 1e-9;
        settings.implicit.absoluteTolerance = 1e-11;
        settings.implicit.multigrid.levels = 2;
        settings.implicit.multigrid.coarseSmooth = 3;
        return settings;
    }
    void Fill(C::FiniteVolume const &fv, C::EquationSystem const &equation, DNDS::CFV::FVField &state, double perturbation)
    {
        auto write = state.Owned().Write();
        const auto values = write.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
        {
            const auto global = fv.Metrics().Cells().LocalToGlobal(iCell);
            const auto mean = equation.Gas().Conservative({1 + .05 * std::sin(global * .7), .2 + .02 * std::cos(global), -.1, .03, 1.1});
            for (int j = 0; j < 5; ++j)
                values(iCell, j) = mean[j] * (1 + perturbation * std::cos(global + j));
        }
    }
    void Fill(C::FiniteVolume const &fv, C::EquationSystem const &equation,
              C::EquationDofs &state, double perturbation)
    {
        auto write = state.WriteOwned();
        const auto values = write.Block("flow");
        for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
        {
            const auto global = fv.Metrics().Cells().LocalToGlobal(iCell);
            const auto mean = equation.Gas().Conservative(
                {1 + .05 * std::sin(global * .7), .2 + .02 * std::cos(global), -.1, .03, 1.1});
            for (int j = 0; j < 5; ++j)
                values(iCell, j) = mean[j] * (1 + perturbation * std::cos(global + j));
        }
    }
}
TEST_CASE("Bottom pMG uses mean convection and jump-only face viscosity in both RHS profiles")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
    auto fine = Controls();
    fine.numerics.viscousGradient = "polynomial";
    fine.numerics.faceFallback = false;
    fine.numerics.characteristicLimiter = false;
    const auto bottom = C::PolynomialMultigrid::CoarseSettings(fine, 1);
    CHECK(bottom.numerics.reconstruction == "gradient");
    CHECK(bottom.numerics.firstOrderConvection);
    CHECK(bottom.numerics.RHSIntegrationOrder() == 1);
    CHECK_FALSE(C::PolynomialMultigrid::CoarseSettings(fine, 0).numerics.firstOrderConvection);
    for (bool logical : {false, true})
    {
        std::vector<double> inviscid;
        for (bool viscous : {false, true})
        {
            std::vector<double> results[2];
            for (int variant = 0; variant < 2; ++variant)
            {
                auto settings = bottom;
                settings.numerics.positivity = false;
                settings.equation.transport = viscous ? "NewtonianFourier" : "inviscid";
                settings.equation.viscosity = viscous ? .01 : 0;
                if (variant)
                    settings.numerics.reconstruction = "constant";
                const auto equation = C::EquationSystem::Resolve(settings.equation);
                C::FiniteVolume fv(mesh, settings, equation);
                {
                    const auto read = fv.Metrics().AcquireRead();
                    const auto metric = read.HostView();
                    for (DNDS::index iFace = 0; iFace < fv.Metrics().NumFaceProc(); ++iFace)
                        CHECK(metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace) == 1);
                }
                auto state = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
                Fill(fv, equation, state, 0);
                R::EagerEngine engine;
                const auto stream = engine.CreateCPUStream();
                auto input = Bind(fv, state), output = Bind(fv, residual);
                auto reconstruction = fv.BuildReconstructionState(input);
                auto workspace = logical
                                     ? fv.BuildWorkspace(input)
                                     : fv.BuildWorkspace(
                                           input, std::nullopt,
                                           C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic});
                fv.Evaluate(input, output, *reconstruction, *workspace, engine, stream);
                const auto read = residual.Owned().Read();
                const auto values = read.View<R::UniformTableRows>();
                for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        results[variant].push_back(values(iCell, j));
            }
            double difference = 0;
            for (DNDS::index i = 0; i < DNDS::index(results[0].size()); ++i)
                difference = std::max(difference, std::abs(results[0][i] - results[1][i]));
            MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            CAPTURE(logical);
            CAPTURE(viscous);
            CAPTURE(difference);
            CHECK(difference < 1e-14); // Face result cannot depend on the reconstructed polynomial.
            if (!viscous)
                inviscid = results[0];
            else
            {
                double jumpEffect = 0;
                for (DNDS::index i = 0; i < DNDS::index(results[0].size()); ++i)
                    jumpEffect = std::max(jumpEffect, std::abs(results[0][i] - inviscid[i]));
                CHECK(jumpEffect > 1e-6); // Equal paths must not silently omit viscosity.
            }
        }
    }
}
TEST_CASE("Each coarse smoother inherits defaults and resolves independent partial controls")
{
    auto settings = Controls();
    settings.implicit.linear.method = "ilu";
    settings.implicit.linear.iterations = 2;
    settings.implicit.ilu.fill = 2;
    settings.implicit.multigrid.coarseLevelSettings["1"].solver = {
        {"linear", {{"method", "sgs"}, {"iterations", 3}}},
        {"sgs", {{"exchangeBetweenSweeps", true}}},
        {"pseudoCFL", 5},
        {"maxPseudoCFL", 5}};
    settings.implicit.multigrid.coarseLevelSettings["2"].solver = {{"ilu", {{"fill", 1}}}};
    const auto intermediate = C::PolynomialMultigrid::CoarseSettings(settings, 0);
    const auto bottom = C::PolynomialMultigrid::CoarseSettings(settings, 1);
    CHECK(intermediate.implicit.linear.method == "sgs");
    CHECK(intermediate.implicit.linear.iterations == 3);
    CHECK(intermediate.implicit.sgs.exchangeBetweenSweeps);
    CHECK(intermediate.implicit.pseudoCFL == 5);
    CHECK(intermediate.implicit.linear.relativeTolerance == settings.implicit.linear.relativeTolerance);
    CHECK(bottom.implicit.linear.method == "ilu");
    CHECK(bottom.implicit.linear.iterations == 2);
    CHECK(bottom.implicit.ilu.fill == 1);
    CHECK(settings.implicit.ilu.fill == 2);
    CHECK(bottom.implicit.multigrid.levels == 0);
    CHECK_FALSE(settings.implicit.multigrid.coarseRHSFPP);
    CHECK(intermediate.implicit.positivity.rhsFPPMode == "none");
    CHECK(bottom.implicit.positivity.rhsFPPMode == "none");
    CHECK(settings.implicit.positivity.rhsFPPMode == "rhs");
    CHECK(bottom.numerics.positivity == settings.numerics.positivity);
    CHECK(bottom.implicit.positivity.incrementRelax == settings.implicit.positivity.incrementRelax);
    settings.implicit.multigrid.coarseRHSFPP = true;
    CHECK(C::PolynomialMultigrid::CoarseSettings(settings, 0).implicit.positivity.rhsFPPMode == "rhs");
    settings.implicit.multigrid.coarseLevelSettings["2"].solver["positivity"] = {{"rhsFPPMode", "dTau"}};
    CHECK(C::PolynomialMultigrid::CoarseSettings(settings, 1).implicit.positivity.rhsFPPMode == "dTau");
    settings.implicit.multigrid.coarseRHSFPP = false;
    CHECK(C::PolynomialMultigrid::CoarseSettings(settings, 1).implicit.positivity.rhsFPPMode == "none");
    C::Json input;
    input["defaults"]["implicit"]["multigrid"]["coarseLevelSettings"]["1"] = {{"solver", {{"linear", {{"method", "sgs"}}}}}};
    const auto config = C::SimulationConfig::Resolve(input, mpi.size);
    CHECK_FALSE(config.Domains().front().settings.implicit.multigrid.coarseRHSFPP);
    input["defaults"]["implicit"]["multigrid"]["coarseRHSFPP"] = true;
    CHECK(C::SimulationConfig::Resolve(input, mpi.size).Domains().front().settings.implicit.multigrid.coarseRHSFPP);
    const auto resolved = config.Domains().front().settings.implicit.ResolveCoarseLevel(1);
    CHECK(resolved.linear.method == "sgs");
    CHECK(resolved.linear.iterations == C::ImplicitSettings{}.linear.iterations);
    settings.implicit.multigrid.coarseLevelSettings["2"].solver["linear"]["iterations"] = 0;
    CHECK_THROWS(settings.implicit.Validate());
}
TEST_CASE("Finest FAS acceptance shares selected-component L1 L2 convergence with the stage solver")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int order : {1, 2})
        for (bool checked : {false, true})
        {
            auto settings = Controls();
            settings.implicit.collectiveChecks = checked;
            settings.implicit.normOrder = order;
            settings.implicit.residualComponents = {"rho"};
            settings.implicit.residualReference = "runningMaximum";
            const auto equation = C::EquationSystem::Resolve(settings.equation);
            C::FiniteVolume fine(mesh, settings, equation);
            auto state = fine.Metrics().BuildField(5), source = fine.Metrics().BuildField(5);
            auto input = Bind(fine, state), right = Bind(fine, source);
            fine.Initialize(input, {}, {1, 0, 0, 0, 1});
            const auto mean = equation.Gas().Conservative({1, 0, 0, 0, 1});
            {
                auto write = source.Owned().Write();
                const auto values = write.View<R::UniformTableRows>();
                for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
                {
                    for (int j = 0; j < 5; ++j)
                        values(iCell, j) = 20 * mean[j];
                    values(iCell, 4) += .1 * (fine.Metrics().Cells().LocalToGlobal(iCell) + 1);
                }
            }
            R::EagerEngine engine;
            const auto stream = engine.CreateCPUStream();
            auto smoother = MakeSolver(fine, input, engine, stream, settings);
            C::PolynomialMultigrid solver(mesh, fine, input, smoother, engine, stream, settings);
            const auto version = state.Owned().Version();
            const auto report = solver.Solve(right, 20, 1, 0);
            CHECK(report.converged);
            CHECK(report.multigridCycles == 0);
            CHECK(report.initialResidual == report.referenceResidual);
            CHECK(report.residual[4] == doctest::Approx(order == 1 ? .85 : std::sqrt(.935)).epsilon(1e-12));
            CHECK(report.residualTargetRatio < 1);
            CHECK(state.Owned().Version() == version);
        }
}
TEST_CASE("Coarse FAS rate reuse and fail-fast halos preserve distributed viscous cycle results")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 5);
    std::vector<double> reference;
    for (int variant = 0; variant < 4; ++variant)
    {
        const bool reuse = variant != 0;
        auto settings = Controls();
        if (variant == 2)
            settings.execution.transferPreflight = "abortOnFailure";
        settings.implicit.multigrid.reuseCoarseRate = reuse;
        settings.implicit.multigrid.scratchPool = variant == 3;
        settings.numerics.reconstructionIterations = 2;
        settings.numerics.reconstructionTolerance = 0;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .01;
        settings.numerics.viscousGradient = "polynomial";
        settings.implicit.linear.method = "ilu";
        settings.implicit.linear.iterations = 2;
        settings.implicit.ilu.fill = 1;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fine(mesh, settings, equation);
        auto state = fine.Metrics().BuildField(5), source = fine.Metrics().BuildField(5);
        Fill(fine, equation, state, .01);
        // A finite time-equation source retains the positive background. Zero
        // source would instead drive this mass-dominated equation toward vacuum
        // and can fail admissibility independently of the reuse being tested.
        A::Copy(source, state);
        A::Scale(source, 20.);
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();
        auto input = Bind(fine, state), right = Bind(fine, source);
        auto smoother = MakeSolver(fine, input, engine, stream, settings);
        auto solver = [&]
        {
            auto original = smoother.RetainForMultigrid();
            C::PolynomialMultigrid hierarchy(mesh, fine, input, original, engine, stream, settings);
            auto moved = std::move(original);
            return hierarchy;
        }(); // Both caller facades expire before cached-rate cycle execution.
        for (int cycle = 0; cycle < 2; ++cycle)
            solver.ApplyCycle(right, 20, .5, .1);
        const auto [scratchBytes, peakBorrows] = solver.ScratchPoolStatistics();
        if (variant == 3)
        {
            CHECK(scratchBytes == 4 * (state.Owned().Storage().Bytes() + state.Ghost().Storage().Bytes()));
            CHECK(peakBorrows == 4);
        }
        else
            CHECK(scratchBytes == 0);
        const auto read = state.Owned().Read();
        const auto values = read.View<R::UniformTableRows>();
        DNDS::index entry = 0;
        for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
            for (DNDS::index j = 0; j < 5; ++j, ++entry)
                if (reuse)
                    CHECK(values(iCell, j) == doctest::Approx(reference[entry]).epsilon(2e-12));
                else
                    reference.push_back(values(iCell, j));
    }
}

TEST_CASE("FAS consumes the measured finest VR rate once for field and logical equations")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 5);
    for (bool logical : {false, true})
    {
        std::vector<double> reference;
        for (int mode : {0, 1, 2})
        {
            const bool manual = mode == 1;
            auto settings = Controls();
            settings.implicit.multigrid.scratchPool = mode == 2;
            settings.numerics.faceFallback = false;
            settings.numerics.characteristicLimiter = false;
            settings.numerics.reconstructionIterations = 1;
            settings.numerics.reconstructionTolerance = 0;
            settings.implicit.minIterations = 1;
            settings.implicit.absoluteTolerance = settings.implicit.relativeTolerance = 0;
            settings.implicit.multigrid.cycles = 1;
            const auto equation = C::EquationSystem::Resolve(settings.equation);
            C::FiniteVolume fine(mesh, settings, equation);
            auto state = fine.Metrics().BuildField(5), source = fine.Metrics().BuildField(5);
            Fill(fine, equation, state, .01);
            A::Copy(source, state);
            A::Scale(source, 20.);
            R::EagerEngine engine;
            const auto stream = engine.CreateCPUStream();
            auto input = Bind(fine, state), forcing = Bind(fine, source);
            const auto reconstruction = fine.BuildReconstructionState(input);
            const auto workspace = logical
                                       ? fine.BuildWorkspace(input)
                                       : fine.BuildWorkspace(
                                             input, std::nullopt,
                                             C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic});
            C::NonlinearSolver smoother(
                fine, input, reconstruction, workspace, engine, stream, settings,
                A::ComponentSelection(input.ComponentNumber(),
                                      input.Schema().ResolveComponents(settings.implicit.ResolveResidualComponents(settings.equation.system == "incompressibleACM"))));
            C::PolynomialMultigrid solver(mesh, fine, input, smoother, engine, stream, settings);
            if (manual)
            {
                auto value = C::EquationDofOperations::NewVector(input), rate = C::EquationDofOperations::NewVector(input);
                smoother.EvaluateEquation(input, value, 20, .5, .1, std::ref(rate));
                solver.ApplyCycle(forcing, 20, .5, .1, 0, 0, std::cref(rate));
            }
            else
                CHECK(solver.Solve(forcing, 20, .5, .1).multigridCycles == 1);
            const auto read = state.Owned().Read();
            const auto values = read.View<R::UniformTableRows>();
            DNDS::index entry = 0;
            for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j, ++entry)
                    if (mode != 0)
                        CHECK(values(iCell, j) == doctest::Approx(reference[entry]).epsilon(2e-13));
                    else
                        reference.push_back(values(iCell, j));
        }
    }
}

TEST_CASE("Polynomial FAS recovers the original P3 manufactured equation across layouts and distributed halos")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        auto settings = Controls();
        settings.implicit.multigrid.levels = scenario == 0 ? 1 : 2;
        if (scenario == 2)
        {
            // Maintained Euler ILU/4ILU/4ILU schedule: one finest
            // correction, four intermediate and four bottom corrections.
            settings.implicit.multigrid.coarseLevelSettings["1"].preSmooth = 4;
            settings.implicit.multigrid.coarseLevelSettings["2"].preSmooth = 4;
            settings.implicit.multigrid.postSmooth = 0;
            settings.implicit.multigrid.coarseSmooth = 4;
            settings.implicit.multigrid.coarseLevelSettings["1"].solver = {
                {"linear", {{"method", "ilu"}, {"iterations", 2}, {"relativeTolerance", 0}, {"absoluteTolerance", 0}}},
                {"ilu", {{"fill", 1}}},
                {"diagonal", "block"}};
            settings.implicit.multigrid.coarseLevelSettings["2"].solver = {
                {"linear", {{"method", "sgs"}, {"iterations", 3}}}, {"pseudoCFL", 5}, {"maxPseudoCFL", 5}};
        }
        settings.numerics.positivity = scenario == 2;
        // Retain one explicit opt-in coarse-PP execution alongside the new
        // Euler-default coarse policy exercised by the other FAS fixtures.
        settings.implicit.multigrid.coarseRHSFPP = scenario == 2;
        settings.execution.overlap = scenario != 1;
        settings.implicit.collectiveChecks = scenario != 1;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fine(mesh, settings, equation);
        const auto layout = scenario == 0 ? R::RowLayout::Dense(0, 5, 8) : scenario == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                         : R::RowLayout::TiledColMajor(0, 5, 32);
        auto state = fine.Metrics().BuildField(layout), exact = fine.Metrics().BuildField(layout), source = fine.Metrics().BuildField(layout);
        R::EagerEngine engine(scenario == 2 ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        auto input = Bind(fine, state), forcing = Bind(fine, source);
        Fill(fine, equation, state, 0);
        auto smoother = MakeSolver(fine, input, engine, stream, settings);
        A::Copy(exact, state);
        smoother.EvaluateEquation(input, forcing, 30, .5, .1);
        const auto generation = source.Owned().Version();
        Fill(fine, equation, state, .01);
        C::PolynomialMultigrid solver(mesh, fine, input, smoother, engine, stream, settings);
        const auto report = solver.Solve(forcing, 30, .5, .1);
        CAPTURE(scenario);
        CAPTURE(report.multigridCycles);
        CAPTURE(report.residual[0]);
        CHECK(report.converged);
        CHECK(report.multigridCycles > 0);
        CHECK(report.iterations == report.multigridCycles);
        CHECK(report.levelCorrections.front() == report.multigridCycles *
                                                     (settings.implicit.multigrid.preSmooth + settings.implicit.multigrid.postSmooth));
        REQUIRE(report.levelCorrections.size() == size_t(settings.implicit.multigrid.levels + 1));
        CHECK(report.levelCorrections.back() > 0);
        if (scenario == 2)
        {
            CHECK(report.levelCorrections[1] == 4 * report.multigridCycles);
            CHECK(report.levelCorrections[2] == 4 * report.multigridCycles);
        }
        CHECK(report.linearIterations > 0);
        CHECK(source.Owned().Version() == generation);
        const auto solutionRead = state.Owned().Read(), exactRead = exact.Owned().Read();
        const auto actual = solutionRead.View<R::UniformTableRows>(), expected = exactRead.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < actual.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(actual(iCell, j) == doctest::Approx(expected(iCell, j)).epsilon(5e-9));
    }
}
TEST_CASE("Polynomial FAS fixed points and preflight errors preserve fine means")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    auto settings = Controls();
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fine(mesh, settings, equation);
    auto state = fine.Metrics().BuildField(5), source = fine.Metrics().BuildField(5), wrong = fine.Metrics().BuildField(4);
    auto input = Bind(fine, state), forcing = Bind(fine, source);
    fine.Initialize(input, {}, {1, .2, -.1, 0, 1});
    A::Copy(source, state);
    A::Scale(source, 10.);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    auto reconstruction = fine.BuildReconstructionState(input);
    auto workspace = fine.BuildWorkspace(input);
    C::NonlinearSolver smoother(
        fine, input, reconstruction, workspace, engine, stream, settings,
        A::ComponentSelection(input.ComponentNumber(),
                              input.Schema().ResolveComponents(settings.implicit.ResolveResidualComponents(settings.equation.system == "incompressibleACM"))));
    C::PolynomialMultigrid solver(mesh, fine, input, smoother, engine, stream, settings);
    const auto generation = state.Owned().Version();
    CHECK_THROWS(solver.Solve(input, 10, .5, 0));
    CHECK_THROWS(Bind(fine, wrong));
    const auto report = solver.Solve(forcing, 10, .5, 0);
    CHECK(report.converged);
    CHECK(report.multigridCycles == 0);
    CHECK(state.Owned().Version() == generation);
    A::Fill(source, -1e6);
    CHECK_THROWS(solver.Solve(forcing, 10, .5, 0));
    CHECK(state.Owned().IsValid());
    CHECK(state.Owned().Version() == generation);
    // A callback failure invalidates the generic cycle. The physical wrapper
    // must rebuild its cycle scratch before the next trial, not silently reuse
    // the partially updated hierarchy. Fine means survived the rejected delta.
    A::Copy(source, state);
    A::Scale(source, 10.);
    Fill(fine, equation, state, .01);
    fine.ResetReconstruction(*reconstruction, *workspace);
    CHECK(solver.Solve(forcing, 10, .5, 0).converged);
}
TEST_CASE("Selected pMG recovers nonconstant principal equations while preserving frozen means and full residual reporting")
{
    for (int scenario = 0; scenario < 4; ++scenario)
    {
        const auto mesh = CauchyTest::PeriodicQuads(mpi, scenario == 3 ? 1 : 4);
        auto settings = Controls();
        settings.implicit.multigrid.updateComponents = {4, 0};
        settings.numerics.positivity = scenario == 2;
        settings.execution.overlap = scenario != 1;
        settings.implicit.collectiveChecks = scenario != 1;
        settings.implicit.diagonal = scenario == 0 ? "entries" : "block";
        settings.implicit.roeDissipation = scenario != 0;
        settings.implicit.jacobianAction = scenario == 2 ? "semiMatrixFree" : "assembled";
        settings.implicit.residualComponents = scenario == 3 ? std::vector<std::string>{} : std::vector<std::string>{"rho", "rhoE"};
        if (scenario == 1)
        {
            settings.implicit.linear.method = "gmres";
            settings.implicit.linear.preconditioner = "ilu";
        }
        if (scenario == 3)
            settings.implicit.multigrid.cycles = 8;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fine(mesh, settings, equation);
        const auto layout = scenario == 0 ? R::RowLayout::Dense(0, 5, 8) : scenario == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                         : R::RowLayout::TiledColMajor(0, 5, 8);
        auto state = fine.Metrics().BuildField(layout), exact = fine.Metrics().BuildField(layout), source = fine.Metrics().BuildField(layout);
        auto value = fine.Metrics().BuildField(layout);
        R::EagerEngine engine(scenario == 2 ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        auto input = Bind(fine, state), forcing = Bind(fine, source), measured = Bind(fine, value);
        Fill(fine, equation, state, 0);
        auto reconstruction = fine.BuildReconstructionState(input);
        auto workspace = fine.BuildWorkspace(input);
        C::NonlinearSolver smoother(
            fine, input, reconstruction, workspace, engine, stream, settings,
            A::ComponentSelection(input.ComponentNumber(),
                                  input.Schema().ResolveComponents(settings.implicit.ResolveResidualComponents(settings.equation.system == "incompressibleACM"))));
        A::Copy(exact, state);
        smoother.EvaluateEquation(input, forcing, 30, .5, .1);
        {
            auto write = state.Owned().Write();
            const auto u = write.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < u.RowCount(); ++iCell)
                for (int j : {0, 4})
                    u(iCell, j) *= 1 + .01 * std::cos(fine.Metrics().Cells().LocalToGlobal(iCell) + j);
        }
        if (scenario == 3)
        {
            // This frozen momentum equation is deliberately incompatible.
            // A subset solve may reduce its own equations, but must not claim
            // convergence of the configured full finest equation.
            auto write = source.Owned().Write();
            for (DNDS::index iCell = 0; iCell < DNDS::index(source.Space().OwnedRows()); ++iCell)
                write.View<R::UniformTableRows>()(iCell, 2) += .1;
        }
        const auto sourceVersion = source.Owned().Version();
        const auto measure = [&]
        {
            smoother.EvaluateEquation(input, measured, 30, .5, .1);
            std::array<double, 6> norms{};
            {
                const auto read = value.Owned().Read(), bRead = source.Owned().Read();
                for (DNDS::index iCell = 0; iCell < DNDS::index(state.Space().OwnedRows()); ++iCell)
                {
                    const auto volume = fine.Metrics().GetCellVol(iCell);
                    norms[5] += volume;
                    for (int j = 0; j < 5; ++j)
                    {
                        const auto error = bRead.View<R::UniformTableRows>()(iCell, j) - read.View<R::UniformTableRows>()(iCell, j);
                        norms[j] += volume * error * error;
                    }
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, norms.data(), 6, MPI_DOUBLE, MPI_SUM, mpi.comm);
            C::FlowState result{};
            for (int j = 0; j < 5; ++j)
                result[j] = std::sqrt(norms[j]);
            return result;
        };
        // Compare the initial equation with the same cold VR guess on both
        // evaluations; otherwise finite VR stopping changes the operator by
        // a few tolerance units before any mean correction has occurred.
        fine.ResetReconstruction(*reconstruction, *workspace);
        const auto initialNorm = measure();
        fine.ResetReconstruction(*reconstruction, *workspace);
        C::PolynomialMultigrid solver(mesh, fine, input, smoother, engine, stream, settings);
        const auto report = solver.Solve(forcing, 30, .5, .1);
        CAPTURE(scenario);
        CAPTURE(report.multigridCycles);
        CAPTURE(report.residual);
        CHECK(report.multigridCycles > 0);
        CHECK(report.converged == (scenario != 3));
        CHECK(source.Owned().Version() == sourceVersion);
        for (int j = 0; j < 5; ++j)
            CHECK(std::abs(report.initialResidual[j] - initialNorm[j]) < 1e-12);
        if (scenario == 3)
        {
            CHECK(report.residual[2] == doctest::Approx(.1).epsilon(1e-10));
            auto subsetControls = settings.implicit;
            subsetControls.residualComponents = {"rho", "rhoE"};
            C::BasicResidualConvergence<std::vector<DNDS::real>> subset(
                subsetControls, equation.Schema(), std::vector<DNDS::real>(5));
            subset.Observe(report.initialResidual);
            subset.Observe(report.residual);
            CHECK(subset.Accepted());
        }
        {
            const auto read = state.Owned().Read(), reference = exact.Owned().Read();
            const auto u = read.View<R::UniformTableRows>(), expected = reference.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < u.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    if (j == 0 || j == 4)
                    {
                        if (scenario != 3)
                            CHECK(u(iCell, j) == doctest::Approx(expected(iCell, j)).epsilon(1e-8));
                    }
                    else
                        CHECK(std::memcmp(&u(iCell, j), &expected(iCell, j), sizeof(double)) == 0);
        }
        const auto finalNorm = measure();
        // Re-evaluation advances an already converged VR warm guess. Do not
        // demand relative precision below its absolute roundoff/tolerance
        // floor; the independent initial norms above are nontrivial, and the
        // incompatible frozen momentum equation below remains O(0.1).
        for (int j = 0; j < 5; ++j)
            CHECK(std::abs(report.residual[j] - finalNorm[j]) < 1e-12);
    }
}
TEST_CASE("Cauchy pMG physical discard and reprepare retain the fine reconstruction transaction")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (const std::string method : {"esdirk2", "ditr"})
    {
        auto settings = Controls();
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::TimeSettings time;
        time.method = method;
        C::Domain reference("reference", mesh, settings, time), retry("retry", mesh, settings, time);
        Fill(reference.Discretization(), equation, reference.Dofs().CoupledField(), 0);
        Fill(retry.Discretization(), equation, retry.Dofs().CoupledField(), 0);
        REQUIRE(reference.Advance(.001).accepted);
        REQUIRE(retry.PrepareStep(.001).accepted);
        retry.DiscardStep();
        REQUIRE(retry.PrepareStep(.001).accepted);
        {
            std::optional<R::Array<double>::ReadLease> retained;
            if (mpi.rank == 0)
                retained.emplace(retry.Dofs().CoupledField().Owned().Read());
            CHECK_THROWS(retry.CommitStep());
        }
        retry.CommitStep();
        CHECK(retry.Time() == .001);
        CHECK(retry.Epoch().acceptedStep == 1);
        const auto first = reference.Dofs().CoupledField().Owned().Read(), second = retry.Dofs().CoupledField().Owned().Read();
        const auto a = first.View<R::UniformTableRows>(), b = second.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < a.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(a(iCell, j) == b(iCell, j));
        REQUIRE_FALSE(retry.LastImplicitStages().empty());
        for (const auto &stage : retry.LastImplicitStages())
        {
            CHECK(stage.converged);
            CHECK(stage.multigridCycles > 0);
        }
        if (method == "ditr")
            CHECK(retry.LastCoupledTargetRatio() <= 1);
    }
}
TEST_CASE("Logical polynomial FAS recovers nonconstant RANS and reacting mixture equations without packing")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    const int scenarios =
#ifdef DNDS_USE_CANTERA
        2;
#else
        1;
#endif
    for (int scenario = 0; scenario < scenarios; ++scenario)
    {
        CAPTURE(scenario);
        auto settings = Controls();
        settings.equation.rans.model = "SA";
        settings.equation.rans.viscosityReference = .01;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = scenario ? 0 : .001;
        if (scenario)
        {
            settings.equation.eos = "idealGasMixture";
            settings.equation.source = "chemical";
            settings.equation.mixture.mechanismFile = "h2o2.yaml";
            settings.equation.mixture.rho0 = 2;
            settings.equation.mixture.U0 = 200;
            settings.equation.mixture.L0 = .02;
            settings.equation.mixture.T0 = 300;
        }
        settings.numerics.faceFallback = false;
        settings.numerics.characteristicLimiter = false;
        settings.numerics.limiter = "WBAP";
        settings.numerics.viscousGradient = "polynomial";
        settings.implicit.iterations = 40;
        settings.implicit.multigrid.cycles = 40;
        settings.implicit.multigrid.coarseSmooth = 2;
        settings.implicit.pseudoCFL = settings.implicit.maxPseudoCFL = 100;
        settings.implicit.relativeTolerance = 1e-8;
        settings.implicit.absoluteTolerance = 1e-8;
        settings.implicit.residualComponents = {};
        settings.implicit.collectiveChecks = scenario == 0;
        if (!scenario)
        {
            settings.implicit.multigrid.updateComponents = {0, 4};
            settings.implicit.residualComponents = {"rho", "rhoE"};
        }
        settings.implicit.linear.method = "sgs";
        settings.implicit.linear.iterations = 3;
        const auto setup = C::PhysicsResources::Resolve(settings.equation);
        C::FiniteVolume fine(mesh, settings, setup.equation);
        A::DofBundle fields(fine.Metrics().Space());
        // Deliberately permute the physical allocation order and couple SA
        // before flow in its parent. Width=5 is never the parent's row pitch.
        fields.Add("turbulentFlow", fine.Metrics().BuildField(R::RowLayout::TiledColMajor(0, 6, 4)));
        std::vector<C::EquationDofBinding> bindings{{"flow", "turbulentFlow", 1}, {"rans", "turbulentFlow", 0}};
        if (scenario)
        {
            fields.Add("composition", fine.Metrics().BuildField(R::RowLayout::ColMajor(0, setup.equation.Mixture().SpeciesNumber() - 1, {8, 1})));
            bindings.push_back({"species", "composition", 0});
        }
        C::EquationDofs state(setup.equation.Schema(), std::move(fields), std::move(bindings));
        std::map<std::string, double> fractions;
        if (scenario)
        {
            const auto &names = setup.equation.Mixture().SpeciesNames();
            for (DNDS::index k = 0; k + 1 < DNDS::index(names.size()); ++k)
                fractions[names[k]] = 1e-6;
            fractions["H2"] = .03;
            fractions["O2"] = .23;
        }
        fine.InitializeConstant(state, {1.2, .1, -.05, .02, 1}, fractions, scenario ? 4 : 0, setup.workers, {{"rhoNuTilde", .02}});
        {
            auto evaluator = setup.equation.MakePointEvaluator(setup.workers ? setup.workers->Acquire() : DNDS::Physics::Chemistry::ChemicalWorkerPool::Lease{}, {});
            const auto composition = scenario ? setup.equation.IndependentMassFractions(fractions) : std::vector<double>{};
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow"), rans = write.Block("rans");
            const auto species = scenario ? write.Block("species") : C::EquationBlockArrayView<double>{};
            for (DNDS::index iCell = 0; iCell < fine.Metrics().NumCell(); ++iCell)
            {
                const auto x = fine.Metrics().GetCellCent(iCell);
                const double wave = std::sin(6.283185307179586 * x[0]) * std::cos(6.283185307179586 * x[1]);
                const double rho = 1.2 + .02 * wave;
                auto rhoY = evaluator.SpeciesInput();
                for (int k = 0; k < rhoY.nSpecies; ++k)
                    species(iCell, k) = rhoY[k] = rho * composition[k];
                const auto value = evaluator.FromPrimitive({rho, .1 + .01 * wave, -.05, .02, 1 + .01 * wave}, rhoY, scenario ? 4 + .01 * wave : 0);
                for (int j = 0; j < 5; ++j)
                    flow(iCell, j) = value[j];
                rans(iCell, 0) = .02 * (1 + .02 * wave);
            }
        }
        auto exact = C::EquationDofOperations::NewVector(state), source = C::EquationDofOperations::NewVector(state);
        C::EquationDofOperations::Copy(exact, state);
        auto reconstruction = fine.BuildReconstructionState(state);
        auto work = fine.BuildWorkspace(state, setup.workers);
        R::EagerEngine engine(scenario ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        C::NonlinearSolver smoother(fine, state, reconstruction, work, engine, stream, settings,
                                    A::ComponentSelection(state.ComponentNumber(), state.Schema().ResolveComponents(settings.implicit.ResolveResidualComponents(settings.equation.system == "incompressibleACM"))), setup.workers);
        constexpr double mass = 10000, scale = .5;
        smoother.EvaluateEquation(state, source, mass, scale, .1);
        const C::EquationDofOperations::Binding sourceBinding(source);
        {
            auto write = state.WriteOwned();
            const A::ComponentSelection selected(state.ComponentNumber(), settings.implicit.multigrid.updateComponents);
            DNDS::index first = 0;
            for (DNDS::index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            {
                const auto values = write.Block(iBlock);
                for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
                    for (DNDS::index j = 0; j < values.ComponentNumber(); ++j)
                        if (selected.Includes(first + j))
                            values(iCell, j) *= 1 + 1e-4 * std::cos(double(fine.Metrics().Cells().LocalToGlobal(iCell)) + j);
                first += values.ComponentNumber();
            }
        }
        auto solver = [&]
        {
            auto original = smoother.RetainForMultigrid();
            C::PolynomialMultigrid hierarchy(mesh, fine, state, original, engine, stream, settings, setup.workers);
            auto moved = std::move(original);
            return hierarchy;
        }(); // Equation callbacks must not borrow either expired facade.
        const auto report = solver.Solve(source, mass, scale, .1);
        CHECK(report.converged);
        CHECK(report.multigridCycles > 0);
        CHECK(report.iterations == report.multigridCycles);
        CHECK(report.levelCorrections == std::vector<DNDS::index>{2 * report.multigridCycles, 2 * report.multigridCycles, 2 * report.multigridCycles});
        CHECK_NOTHROW(sourceBinding.Check(source, true));
        {
            const auto a = state.ReadOwned(), b = exact.ReadOwned();
            const auto al = a.ComponentLayout(), bl = b.ComponentLayout();
            const auto actual = al.View(), expected = bl.View();
            for (DNDS::index iCell = 0; iCell < actual.RowCount(); ++iCell)
                for (DNDS::index j = 0; j < actual.ComponentNumber(); ++j)
                {
                    CHECK(actual(iCell, j) == doctest::Approx(expected(iCell, j)).epsilon(5e-8).scale(1e-10));
                    if (!scenario && j != 0 && j != 4)
                        CHECK(std::memcmp(&actual(iCell, j), &expected(iCell, j), sizeof(double)) == 0);
                }
        }
        auto value = C::EquationDofOperations::NewVector(state);
        smoother.EvaluateEquation(state, value, mass, scale, .1);
        std::vector<double> norm(size_t(state.ComponentNumber()) + 1, 0);
        {
            const auto a = source.ReadOwned(), b = value.ReadOwned();
            const auto al = a.ComponentLayout(), bl = b.ComponentLayout();
            const auto desired = al.View(), actual = bl.View();
            for (DNDS::index iCell = 0; iCell < desired.RowCount(); ++iCell)
            {
                const double volume = fine.Metrics().GetCellVol(iCell);
                norm.back() += volume;
                for (DNDS::index j = 0; j < desired.ComponentNumber(); ++j)
                {
                    const double defect = desired(iCell, j) - actual(iCell, j);
                    norm[j] += volume * defect * defect;
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, norm.data(), int(norm.size()), MPI_DOUBLE, MPI_SUM, mpi.comm);
        for (DNDS::index j = 0; j < state.ComponentNumber(); ++j)
            CHECK(std::abs(report.residual[j] - std::sqrt(norm[j])) < 1e-7);
    }
}
TEST_CASE("Equation Domain performs one spatial FAS update per DITR coupled stage and retries exactly")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    auto settings = Controls();
    settings.equation.rans.model = "SA";
    settings.equation.rans.viscosityReference = .01;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .001;
    settings.initialRANS = {{"rhoNuTilde", .02}};
    settings.numerics.ransState.referenceConservative = {{"rho", 1}, {"rhoNuTilde", .02}};
    settings.numerics.faceFallback = false;
    settings.numerics.characteristicLimiter = false;
    settings.numerics.limiter = "WBAP";
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.positivity = true;
    settings.numerics.reconstructionIterations = 4;
    settings.numerics.reconstructionTolerance = 0;
    settings.execution.fieldGrouping = "equationBlocks";
    settings.execution.layout = "aosoa32";
    settings.execution.hostThreads = 2;
    settings.implicit.linear.method = "sgs";
    settings.implicit.linear.iterations = 2;
    settings.implicit.absoluteTolerance = 1e-7;
    settings.implicit.relativeTolerance = 1e-7;
    settings.implicit.residualComponents = {"rho", "rhoNuTilde"};
    C::TimeSettings time;
    time.method = "ditr";
    time.ditr.variant = "U2R1";
    time.ditr.absoluteTolerance = 1e-7;
    time.ditr.relativeTolerance = 1e-7;
    C::Domain reference("reference", mesh, settings, time), retry("retry", mesh, settings, time);
    const auto equation = C::PhysicsResources::Resolve(settings.equation).equation;
    Fill(reference.Discretization(), equation, reference.Dofs(), 0);
    Fill(retry.Discretization(), equation, retry.Dofs(), 0);
    REQUIRE(reference.Advance(.001).accepted);
    REQUIRE(retry.PrepareStep(.001).accepted);
    retry.DiscardStep();
    REQUIRE(retry.Advance(.001).accepted);
    CHECK(retry.Time() == .001);
    REQUIRE(retry.LastImplicitStages().size() == 1);
    const auto &report = retry.LastImplicitStages().front();
    CHECK(report.converged);
    CHECK(report.multigridCycles == 2 * report.iterations);
    CHECK(report.levelCorrections == std::vector<DNDS::index>{2 * report.multigridCycles, 2 * report.multigridCycles, 3 * report.multigridCycles});
    const auto a = retry.Dofs().ReadOwned(), b = reference.Dofs().ReadOwned();
    const auto al = a.ComponentLayout(), bl = b.ComponentLayout();
    const auto actual = al.View(), expected = bl.View();
    for (DNDS::index iCell = 0; iCell < actual.RowCount(); ++iCell)
        for (DNDS::index j = 0; j < actual.ComponentNumber(); ++j)
            CHECK(actual(iCell, j) == expected(iCell, j));
}
int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
