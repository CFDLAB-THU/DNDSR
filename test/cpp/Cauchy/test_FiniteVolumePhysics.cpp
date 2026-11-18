#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/State/EquationTimeStepper.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Input/StateInput.hpp"
#include "Cauchy/Operators/Source/ChemicalSource.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "DNDS/Runtime/Storage/AllocationDiagnostics.hpp"
#include "../CFV/RotatedSectorFixture.hpp"
#include <tuple>
#include <cstdlib>
#include <numeric>
#include <limits>
#include <atomic>

namespace C = DNDS::Cauchy;
namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;
namespace P = DNDS::Physics;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
static std::atomic<bool> countReductions{false};
static std::atomic<int> reductions{0};
extern "C" int MPI_Allreduce(const void *send, void *receive, int count, MPI_Datatype type, MPI_Op operation, MPI_Comm comm)
{
    if (countReductions)
        ++reductions;
    return PMPI_Allreduce(send, receive, count, type, operation, comm);
}
namespace
{
    void Compare(const C::EquationDofs &a, const C::EquationDofs &b, double tolerance)
    {
        auto x = a.ReadOwned(), y = b.ReadOwned();
        for (I iBlock = 0; iBlock < a.BlockNumber(); ++iBlock)
        {
            const auto u = x.Block(iBlock), v = y.Block(iBlock);
            for (I iCell = 0; iCell < I(a.Space().OwnedRows()); ++iCell)
                for (I j = 0; j < u.ComponentNumber(); ++j)
                    if (tolerance == 0)
                        CHECK(u(iCell, j) == v(iCell, j));
                    else
                        CHECK(u(iCell, j) == doctest::Approx(v(iCell, j)).epsilon(tolerance).scale(1));
        }
    }
    void ConservativeIntegral(const C::EquationDofs &residual)
    {
        auto read = residual.ReadOwned();
        for (I iBlock = 0; iBlock < residual.BlockNumber(); ++iBlock)
        {
            const auto u = read.Block(iBlock);
            for (I j = 0; j < u.ComponentNumber(); ++j)
            {
                double integral = 0, absolute = 0;
                for (I iCell = 0; iCell < u.owned.RowCount(); ++iCell)
                {
                    integral += u(iCell, j);
                    absolute += std::abs(u(iCell, j));
                }
                MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
                MPI_Allreduce(MPI_IN_PLACE, &absolute, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
                CHECK(std::abs(integral) < 1e-10 * (1 + absolute));
            }
        }
    }
    void CompareAuxiliary(const C::AuxiliaryValues &a, const C::AuxiliaryValues &b)
    {
        const auto arrays = [](const auto &v)
        {
            return std::array<const R::Array<double> *, 7>{v.meanThermodynamics ? &*v.meanThermodynamics : nullptr,
                                                           v.faceSpectralRadius ? &*v.faceSpectralRadius : nullptr, v.faceWaveJump ? &*v.faceWaveJump : nullptr,
                                                           v.facePressureMinimum ? &*v.facePressureMinimum : nullptr, v.cellSpectralSum ? &v.cellSpectralSum->Owned() : nullptr,
                                                           v.cellWaveJump ? &v.cellWaveJump->Owned() : nullptr, v.cellPressureMinimum ? &v.cellPressureMinimum->Owned() : nullptr};
        };
        const auto x = arrays(a), y = arrays(b);
        for (int k = 0; k < 7; ++k)
            if (x[k] && y[k])
            {
                const auto xr = x[k]->Read(), yr = y[k]->Read();
                const auto u = xr.View<R::UniformTableRows>(), v = yr.View<R::UniformTableRows>();
                REQUIRE(u.RowCount() == v.RowCount());
                for (I iRow = 0; iRow < u.RowCount(); ++iRow)
                    for (I j = 0; j < u.RowSize(iRow); ++j)
                        CHECK(u(iRow, j) == doctest::Approx(v(iRow, j)).epsilon(2e-11).scale(1));
            }
    }
}

TEST_CASE("Equation FV uses the established P3/gradient/WBAP flow5 kernels across CPU launchers and layouts")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    for (int variant = 0; variant < 3; ++variant)
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = variant == 0 ? "gradient" : "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 5;
        settings.numerics.limiter = variant == 2 ? "WBAP" : "none";
        settings.numerics.cfv.SORInstead = variant == 2;
        settings.numerics.characteristicLimiter = variant == 2;
        settings.numerics.faceFallback = false;
        settings.numerics.positivity = false;
        settings.numerics.riemann.useStencil = false;
        settings.numerics.viscousGradient = "polynomial";
        if (variant == 1)
        {
            settings.numerics.flux = "Roe_M1";
            settings.equation.transport = "NewtonianFourier";
            settings.equation.viscosity = .002;
            settings.equation.viscosityModel = "sutherland";
            settings.equation.viscosityReferenceTemperature = .8;
            settings.equation.sutherlandConstant = .3;
        }
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        const auto shape = variant == 0 ? R::RowLayout::Dense(0, 5) : variant == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                   : R::RowLayout::TiledColMajor(0, 5, 32);
        auto raw = fv.Metrics().BuildField(shape);
        {
            auto write = raw.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const auto center = fv.Metrics().GetCellCent(iCell);
                const auto state = equation.Gas().Conservative({1 + .01 * std::sin(6.283185307179586 * center[0]),
                                                                .3, -.1 + .01 * std::cos(6.283185307179586 * center[1]), .2, 1});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = state[j];
            }
        }
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(shape));
        A::Copy(state.CoupledField(), raw);
        auto generic = C::EquationDofs::Coupled(equation.Schema(), raw);
        auto expected = C::EquationDofOperations::NewVector(generic);
        auto result = C::EquationDofOperations::NewVector(state);
        auto genericReconstruction = fv.BuildReconstructionState(generic);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto genericWork = fv.BuildWorkspace(
            generic, std::nullopt,
            C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic});
        auto work = fv.BuildWorkspace(state);
        R::EagerEngine engine(variant == 0 ? 0 : 2);
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(generic, expected, *genericReconstruction, *genericWork, engine, stream);
        fv.Evaluate(state, result, *reconstruction, *work, engine, stream);
        Compare(result, expected, 2e-12);
        auto separate = C::EquationDofOperations::NewVector(state);
        auto faceFlux = fv.BuildFaceFluxBuffer(state.ComponentNumber());
        fv.EvaluateFlux(state, separate, *reconstruction, *work, DNDS::CFV::ReconstructionRegion::All, 0, std::ref(faceFlux));
        auto oracleFaces = fv.BuildFaceFluxBuffer(generic.ComponentNumber());
        fv.EvaluateFlux(generic, expected, *genericReconstruction, *genericWork,
                        DNDS::CFV::ReconstructionRegion::All, 0, std::ref(oracleFaces));
        {
            const auto actualRead = faceFlux.Read(), oracleRead = oracleFaces.Read();
            const auto actual = actualRead.View<R::FixedRows<5>>(), oracle = oracleRead.View<R::FixedRows<5>>();
            REQUIRE(actual.RowCount() == oracle.RowCount());
            for (I iFace = 0; iFace < actual.RowCount(); ++iFace)
                for (int j = 0; j < 5; ++j)
                    CHECK(actual(iFace, j) == doctest::Approx(oracle(iFace, j)).epsilon(2e-12).scale(1));
        }

        fv.AddSource(state, separate, *reconstruction, *work);
        Compare(result, separate, 2e-12);
        if (variant == 0)
        {
            auto preparedReconstruction = fv.BuildReconstructionState(state);
            auto prepared = fv.BuildWorkspace(state);
            fv.PrepareReconstruction(state, *preparedReconstruction, *prepared, engine, stream);
            fv.EvaluateFlux(state, separate, *preparedReconstruction, *prepared);
            fv.AddSource(state, separate, *preparedReconstruction, *prepared);
            Compare(result, separate, 2e-12);
        }
        ConservativeIntegral(result);
        if (variant == 1)
        {
            const auto &coefficients = fv.ReconstructionGuess(*reconstruction);
            auto snapshot = C::EquationDofOperations::NewVector(coefficients);
            auto replacement = C::EquationDofOperations::NewVector(coefficients);
            C::EquationDofOperations::Copy(snapshot, coefficients);
            C::EquationDofOperations::Fill(replacement, .125);
            std::optional<C::EquationDofs::ReadScope> retained;
            if (mpi.rank == 0)
                retained.emplace(coefficients.ReadOwned());
            CHECK_THROWS(fv.RestoreReconstructionGuess(*reconstruction, replacement));
            retained.reset();
            Compare(coefficients, snapshot, 1e-15);
            if (mpi.rank == 0)
            {
                auto write = replacement.WriteOwned();
                write.Block(0)(0, 0) = std::numeric_limits<double>::quiet_NaN();
            }
            CHECK_THROWS(fv.RestoreReconstructionGuess(*reconstruction, replacement));
            Compare(coefficients, snapshot, 1e-15);
            C::EquationDofOperations::Fill(replacement, .125);
            fv.RestoreReconstructionGuess(*reconstruction, replacement);
            Compare(coefficients, replacement, 1e-15);
        }
    }
}

TEST_CASE("Equation FV local stages and physical-field halos inherit the agreed safety policy")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    std::optional<C::EquationDofs> reference;
    int checkedReductions = 0;
    for (bool checks : {true, false})
    {
        C::DomainSettings settings;
        settings.implicit.collectiveChecks = checks;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 2;
        settings.numerics.reconstructionTolerance = 0;
        settings.numerics.limiter = "WBAP";
        settings.numerics.characteristicLimiter = false;
        settings.numerics.faceFallback = false;
        settings.numerics.positivity = false;
        settings.numerics.riemann.useStencil = false;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(5));
        {
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                const auto point = fv.Metrics().GetCellCent(iCell);
                const auto value = equation.Gas().Conservative({1 + .02 * std::sin(6.283185307179586 * point[0]), .2, -.1, .05, 1});
                for (I component = 0; component < 5; ++component)
                    flow(iCell, component) = value[component];
            }
        }
        auto result = C::EquationDofOperations::NewVector(state);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state);
        R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        struct CountingScope
        {
            CountingScope()
            {
                reductions = 0;
                countReductions = true;
            }
            ~CountingScope() { countReductions = false; }
        };
        {
            CountingScope count;
            fv.Evaluate(state, result, *reconstruction, *work, engine, stream);
        }
        if (checks)
        {
            checkedReductions = reductions;
            reference = result;
        }
        else
        {
            CHECK(checkedReductions > 0);
            CHECK(reductions.load() == 0); // Fixed sweeps; no numerical norm requested.
            Compare(result, *reference, 0);
        }
    }
}

TEST_CASE("Canonical equation face flux agrees with cell incidence on rotated and self-periodic faces")
{
    for (bool rotated : {false, true})
    {
        auto mesh = rotated ? CFVTest::Sector(mpi) : CauchyTest::PeriodicQuads(mpi, 1);
        C::DomainSettings settings;
        settings.numerics.reconstruction = "constant";
        settings.numerics.limiter = "none";
        settings.numerics.faceFallback = false;
        settings.numerics.characteristicLimiter = false;
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.flux = "Roe_M1";
        if (rotated)
            settings.boundaries["12"].kind = settings.boundaries["13"].kind = "extrapolate";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto raw = fv.Metrics().BuildField(5);
        {
            auto write = raw.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const auto center = fv.Metrics().GetCellCent(iCell);
                const auto value = equation.Gas().Conservative({1 + .1 * center[0], .2 + .03 * center[1], -.15, .07, 1});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = value[j];
            }
        }
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(5));
        A::Copy(state.CoupledField(), raw);
        auto generic = C::EquationDofs::Coupled(equation.Schema(), raw);
        auto expected = C::EquationDofOperations::NewVector(generic);
        auto result = C::EquationDofOperations::NewVector(state);
        auto genericWork = fv.BuildWorkspace(
            generic, std::nullopt,
            C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic});
        auto genericReconstruction = fv.BuildReconstructionState(generic);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state);
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(generic, expected, *genericReconstruction, *genericWork, engine, stream);
        fv.Evaluate(state, result, *reconstruction, *work, engine, stream);
        Compare(result, expected, 2e-12);
    }
}

TEST_CASE("Logical nonlinear flow5 retains native finite-sweep stage mathematics")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 2; ++scenario)
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = scenario ? "variational" : "constant";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 8;
        settings.numerics.reconstructionTolerance = scenario ? 1e-8 : 0;
        settings.numerics.faceFallback = false;
        settings.implicit.linear.method = scenario ? "gmres" : "sgs";
        settings.implicit.linear.relativeTolerance = 1e-5;
        settings.implicit.diagonal = scenario ? "block" : "entries";
        settings.execution.overlap = scenario;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto raw = fv.Metrics().BuildField(R::RowLayout::ColMajor(0, 5, {8, 1}));
        auto source = fv.Metrics().BuildField(5);
        const auto target = equation.Gas().Conservative({1.2, .3, -.1, .2, 1.1});
        {
            auto u = raw.Owned().Write(), b = source.Owned().Write();
            for (I iCell = 0; iCell < raw.Owned().RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    u.View()(iCell, j) = target[j] * (1 + .01 * std::sin(2 * iCell + mpi.rank + j));
                    b.View()(iCell, j) = 20 * target[j];
                }
        }
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(R::RowLayout::TiledColMajor(0, 5, 8)));
        A::Copy(state.CoupledField(), raw);
        auto generic = C::EquationDofs::Coupled(equation.Schema(), raw);
        auto right = C::EquationDofs::Coupled(equation.Schema(), source);
        R::EagerEngine engine(scenario ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        C::AuxiliaryRequest request;
        request.faceSpectralRadius = request.cellSpectralSum = true;
        C::NonlinearSolver reference(fv, generic,
                                     fv.BuildReconstructionState(generic),
                                     fv.BuildWorkspace(
                                         generic, std::nullopt,
                                         C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic}),
                                     engine, stream, settings, A::ComponentSelection(5));
        C::NonlinearSolver solver(fv, state, fv.BuildReconstructionState(state), fv.BuildWorkspace(state), engine, stream, settings, A::ComponentSelection(5));
        const auto expected = reference.Solve(generic, right, 20, .5, .1, 0, {2});
        const auto actual = solver.Solve(state, right, 20, .5, .1, 0, {2});
        CHECK(actual.iterations == expected.iterations);
        CHECK(actual.linearIterations == expected.linearIterations);
        CHECK(actual.converged == expected.converged);
        CHECK(actual.reconstructionIterations == expected.reconstructionIterations);
        CHECK(actual.reconstructionRelativeUpdate == doctest::Approx(expected.reconstructionRelativeUpdate).epsilon(1e-10));
        for (int j = 0; j < 5; ++j)
            CHECK(actual.residual[j] == doctest::Approx(expected.residual[j]).epsilon(1e-10).scale(1e-10));
        Compare(state, generic, 1e-12);
    }
}

TEST_CASE("Equation wall values preserve internal energy and support empty owners on a physical 3D mesh")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    for (int scenario = 0; scenario < 5; ++scenario)
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .002;
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.faceFallback = false;
        settings.boundaries["7"].kind = scenario < 2 || scenario == 4 ? "noSlipWall" : scenario == 2 ? "farfield"
                                                                                                     : "analytic";
        settings.boundaries["7"].primitive = {.9, -.3, .1, .2, 1.2};
        if (scenario == 2 || scenario == 3)
        {
            settings.numerics.reconstruction = "variational";
            settings.numerics.cfv.maxOrder = 3;
        }
        if (scenario == 1 || scenario == 4)
        {
            settings.boundaries["7"].thermal = "isothermal";
            settings.boundaries["7"].temperature = .7;
        }
        settings.numerics.noRsOnWall = scenario == 4;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto raw = fv.Metrics().BuildField(5);
        {
            auto write = raw.Owned().Write();
            const auto u = equation.Gas().Conservative({1, .1, .2, -.1, 1});
            for (I iCell = 0; iCell < raw.Owned().RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = u[j];
        }
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(5));
        A::Copy(state.CoupledField(), raw);
        auto generic = C::EquationDofs::Coupled(equation.Schema(), raw);
        auto expected = C::EquationDofOperations::NewVector(generic);
        auto result = C::EquationDofOperations::NewVector(state);
        auto genericWork = fv.BuildWorkspace(
            generic, std::nullopt,
            C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic});
        auto genericReconstruction = fv.BuildReconstructionState(generic);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state);
        R::EagerEngine engine(2);
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(generic, expected, *genericReconstruction, *genericWork, engine, stream);
        fv.Evaluate(state, result, *reconstruction, *work, engine, stream);
        Compare(result, expected, 2e-12);
        C::AuxiliaryRequest request;
        request.meanThermodynamics = request.faceSpectralRadius = request.cellSpectralSum = true;
        auto originalAux = fv.BuildAuxiliaryValues(request), equationAux = fv.BuildAuxiliaryValues(request);
        fv.PrepareAuxValues(generic, originalAux);
        fv.PrepareAuxValues(state, equationAux);
        CompareAuxiliary(originalAux, equationAux);
    }
}

TEST_CASE("Spatial initial input averages conservative polynomials on distributed native layouts")
{
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.equation.referenceScales = {{"rho0", 2}, {"U0", 3}, {"L0", 4}, {"T0", 5}};
    settings.initialField.exprtkInitializers.push_back({{"inRegion := 1; UExprtk[0] := 2*(1+x[0]); UExprtk[1] := 3*(2+x[0]);",
                                                         "UExprtk[4] := 18*(1+x[0]^2); 0;"},
                                                        "primRhoP_phy"});
    settings.initialField.integrationOrder = 2;
    const auto parsed = C::SimulationConfig::ResolveCollective({{"defaults", C::Json(settings)}}, mpi.comm);
    for (const auto *layout : {"aos", "soa", "aosoa32"})
    {
        settings = parsed.Domains()[0].settings;
        settings.execution.layout = layout;
        C::Domain domain("spatial", CauchyTest::PeriodicQuads(mpi, 4), settings);
        const auto read = domain.Dofs().ReadOwned();
        const auto flow = read.Block("flow");
        std::array<double, 5> integral{};
        for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                integral[j] += flow(iCell, j) / 16;
        MPI_Allreduce(MPI_IN_PLACE, integral.data(), 5, MPI_DOUBLE, MPI_SUM, mpi.comm);
        // Exact integrals on [0,1]^2: rho=1+x, u=2+x, p=1+x^2.
        // Energy must average rho*u^2/2, not use kinetic energy of the mean.
        CHECK(integral[0] == doctest::Approx(1.5).epsilon(2e-13));
        CHECK(integral[1] == doctest::Approx(23. / 6).epsilon(2e-13));
        CHECK(integral[2] == doctest::Approx(0).epsilon(2e-13));
        CHECK(integral[3] == doctest::Approx(0).epsilon(2e-13));
        CHECK(integral[4] == doctest::Approx(199. / 24).epsilon(2e-13));
    }
}

TEST_CASE("Face-centered assembled Jacobians match direct actions in stationary and moving frames")
{
    // Cover translated identity frames and a genuinely rotated periodic
    // frame against the independent directional-action implementation.
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        const auto mesh = scenario == 2 ? CFVTest::Sector(mpi) : CauchyTest::PeriodicQuads(mpi, 6);
        auto settings = C::DomainSettings{};
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 5;
        if (scenario)
            settings.motion.angularVelocity = {0, 0, .2};
        if (scenario == 2)
            settings.boundaries["12"].kind = settings.boundaries["13"].kind = "extrapolate";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(5));
        R::Array<double> tau(R::RowLayout::Dense(fv.Metrics().NumCell(), 1));
        {
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow");
            auto pseudo = tau.Write();
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto center = fv.Metrics().GetCellCent(iCell);
                const auto values = equation.Gas().Conservative({1 + .1 * center[0], .3, -.2 + .05 * center[1], .1, 1 + .1 * center[1]});
                for (int j = 0; j < 5; ++j)
                    flow(iCell, j) = values[j];
                pseudo.View()(iCell, 0) = 2 + center[0];
            }
        }
        fv.BindHalo(state.PhysicalField(0)).Begin().Wait();
        auto auxiliary = fv.BuildAuxiliaryValues({true, true});
        fv.PrepareAuxValues(state, auxiliary, .3);
        auto assembled = fv.BuildLinearProblem(state, C::DiagonalStorage::Block);
        auto pointwise = fv.BuildLinearProblem(state, C::DiagonalStorage::Block, C::OffDiagonalStorage::SemiMatrixFree);
        const C::JacobianPreparation controls{3, .7, false};
        fv.PrepareJacobian(state, auxiliary, tau, *assembled, controls, .3);
        fv.PrepareJacobian(state, auxiliary, tau, *pointwise, controls, .3);
        const auto geometry = fv.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        if (fv.Metrics().NumFaceProc())
            REQUIRE(metric.FaceQuadratureEnd(0) - metric.FaceQuadratureBegin(0) > 1);
        const auto a = assembled->ReadOffDiagonal(), b = pointwise->ReadOffDiagonal();
        Eigen::VectorXd directionStorage(10), direct(5);
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<2>> direction(directionStorage.data(), 5);
        for (int j = 0; j < 5; ++j)
            direction[j] = -.3 + .17 * j;
        a.WithBlocks([&](const auto &cached)
                     { b.WithBlocks([&](const auto &reference)
                                    {
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                for (I iBlock = 0; iBlock < I(assembled->Columns()[iCell].size()); ++iBlock)
                {
                    const auto observed = cached(iCell, iBlock);
                    const auto expected = reference(iCell, iBlock);
                    for (int j = 0; j < 5; ++j)
                        for (int i = 0; i < 5; ++i)
                            CHECK(observed(i, j) == doctest::Approx(expected(i, j)).epsilon(2e-14).scale(1));
                    pointwise->PreparedOffDiagonalEvaluator()->ApplyBlock(iCell, iBlock, direction, direct);
                    const auto product = (observed * direction).eval();
                    for (int j = 0; j < 5; ++j)
                        CHECK(direct[j] == doctest::Approx(product[j]).epsilon(3e-13).scale(1));
                } }); });
    }
}

#ifdef DNDS_USE_CANTERA
TEST_CASE("A single-species material uses empty independent composition and no chemical equation")
{
    const P::ReferenceScales scales{1, 340, 1, 300};
    P::Chemistry::ChemicalSource prototype(std::string(DNDS_TEST_SOURCE_ROOT) + "/test/cpp/Cauchy/argon.yaml", "argon", scales.U0, scales.rho0);
    P::Thermodynamics::IdealGasMixture material(prototype, scales);
    P::Chemistry::ChemicalWorkerPool workers(prototype, 1);
    const auto equation = C::EquationSystem::FromIdealGasMixture(material, false);
    REQUIRE(equation.Schema().blocks.size() == 1);
    double fraction = 0;
    prototype.massFractions(1, P::Chemistry::ConstSpeciesBufferView{}, {&fraction, 1});
    CHECK(fraction == 1);
    const P::Chemistry::ConstSpeciesBufferView fractions{&fraction, 1};
    const double energy = prototype.mixtureIntEnergy(300, fractions, prototype.mixtureR(fractions) * 300) / scales.SpecificEnergy();
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    for (const std::string source : {"none", "chemical"})
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.boundaries["7"].kind = "extrapolate";
        settings.equation.source = source;
        settings.numerics.reconstruction = "constant";
        settings.numerics.faceFallback = false;
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(5));
        {
            auto write = state.WriteOwned();
            const auto u = write.Block("flow");
            for (I iCell = 0; iCell < u.RowCount(); ++iCell)
            {
                u(iCell, 0) = 1;
                u(iCell, 4) = energy;
                for (int j = 1; j < 4; ++j)
                    u(iCell, j) = 0;
            }
        }
        auto residual = C::EquationDofOperations::NewVector(state);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state, workers);
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(state, residual, *reconstruction, *work, engine, stream);
        {
            const auto read = residual.ReadOwned();
            const auto r = read.Block("flow");
            for (I iCell = 0; iCell < r.owned.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(std::abs(r(iCell, j)) < 1e-12);
        }
        R::Array<double> jacobian(R::RowLayout::Dense(I(state.Space().OwnedRows()), 25));
        C::EvaluateChemicalSource(equation, state, residual, workers.Acquire(), std::ref(jacobian));
        const auto read = residual.ReadOwned();
        const auto matrix = jacobian.Read();
        for (I iCell = 0; iCell < I(state.Space().OwnedRows()); ++iCell)
        {
            for (int j = 0; j < 5; ++j)
                CHECK(read.Block("flow")(iCell, j) == 0);
            for (int j = 0; j < 25; ++j)
                CHECK(matrix.View()(iCell, j) == 0);
        }
    }
}
namespace
{
    struct MixtureFixture
    {
        const P::ReferenceScales scales;
        P::Chemistry::ChemicalSource prototype;
        P::Thermodynamics::IdealGasMixture material;
        P::Chemistry::ChemicalWorkerPool workers;
        C::EquationSystem equation;
        int species;
        explicit MixtureFixture(double length = 2) : scales{1.225, 340, length, 288.15}, prototype((std::getenv("DNDS_MECH_PATH") ? std::string(std::getenv("DNDS_MECH_PATH")) + "/" : "") + "h2o2.yaml", "", scales.U0, scales.rho0),
                                                     material(prototype, scales), workers(prototype, 1), equation(C::EquationSystem::FromIdealGasMixture(material, true)), species(prototype.nSpecies()) {}
        C::EquationDofs State(const A::DofSpace &space, int layout)
        {
            if (!layout)
                return C::EquationDofs::Coupled(equation.Schema(), A::DofField<double>(space, R::RowLayout::Dense(0, species + 4)));
            A::DofBundle fields(space);
            if (layout == 1)
            {
                fields.Add("flow", A::DofField<double>(space, mpi.rank % 2 ? R::RowLayout::Dense(0, 5, 8) : R::RowLayout::ColMajor(0, 5, {16, 1})));
                fields.Add("species", A::DofField<double>(space, R::RowLayout::TiledColMajor(0, species - 1, 32)));
                return C::EquationDofs::Separate(equation.Schema(), std::move(fields));
            }
            fields.Add("packed", A::DofField<double>(space, R::RowLayout::TiledColMajor(0, species + 4, 32)));
            return C::EquationDofs(equation.Schema(), std::move(fields), {{"flow", "packed", species - 1}, {"species", "packed", 0}});
        }
        void Fill(C::EquationDofs &state, const DNDS::CFV::FiniteVolumePlan &metrics, bool constant)
        {
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow"), rhoY = write.Block("species");
            std::vector<double> Y(size_t(species), 1e-6);
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                const auto center = metrics.GetCellCent(iCell);
                const double wave = constant ? 0 : .01 * std::sin(6.283185307179586 * center[0]) * std::cos(6.283185307179586 * center[1]);
                Y[0] = .027 + wave * .05;
                Y[3] = .22 - wave * .07;
                Y[5] = .02 + wave * .02;
                Y.back() = 0;
                Y.back() = 1 - std::accumulate(Y.begin(), Y.end(), 0.);
                const double rho = .8 + wave, temperature = 1400 + 50 * wave;
                const P::Chemistry::ConstSpeciesBufferView composition{Y.data(), species};
                const double pressure = rho * scales.rho0 * prototype.mixtureR(composition) * temperature;
                flow(iCell, 0) = rho;
                flow(iCell, 1) = rho * .2;
                flow(iCell, 2) = rho * -.04;
                flow(iCell, 3) = rho * .03;
                flow(iCell, 4) = rho * prototype.mixtureIntEnergy(temperature, composition, pressure) / scales.SpecificEnergy() + .5 * rho * (.04 + .0016 + .0009);
                for (int k = 0; k < species - 1; ++k)
                    rhoY(iCell, k) = rho * Y[k];
            }
        }
    };
}

TEST_CASE("Mixture characteristic WBAP consumes fresh distributed cell caches for modal and direct gradients")
{
    MixtureFixture mixture;
    const auto equation = C::EquationSystem::FromIdealGasMixture(mixture.material, false);
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
    for (const std::string reconstructionKind : {"gradient", "variational"})
    {
        std::optional<C::EquationDofs> baseline;
        for (int scenario = 0; scenario < 2; ++scenario)
        {
            INFO(reconstructionKind << ", derived/separate/worker=" << scenario);
            C::DomainSettings settings;
            settings.numerics.reconstruction = reconstructionKind;
            settings.numerics.cfv.maxOrder = 2;
            settings.numerics.reconstructionIterations = 3;
            settings.numerics.cfv.SORInstead = false;
            settings.numerics.limiter = "WBAP";
            settings.numerics.characteristicLimiter = true;
            settings.numerics.faceFallback = settings.numerics.positivity = false;
            settings.numerics.riemann.useStencil = false;
            settings.numerics.viscousGradient = "polynomial";
            settings.execution.overlap = bool(scenario);
            C::FiniteVolume fv(mesh, settings, equation);
            auto state = mixture.State(fv.Metrics().Space(), scenario);
            mixture.Fill(state, fv.Metrics(), false);
            auto residual = C::EquationDofOperations::NewVector(state);
            auto reconstruction = fv.BuildReconstructionState(state);
            auto work = fv.BuildWorkspace(state, mixture.workers);
            C::AuxiliaryRequest request;
            if (scenario)
                request.derived = {C::DerivedQuantity::Temperature};
            auto auxiliary = fv.BuildAuxiliaryValues(request);
            R::EagerEngine engine(scenario ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
            const auto stream = engine.CreateCPUStream();
            fv.Evaluate(state, residual, *reconstruction, *work, auxiliary, engine, stream);
            CHECK(fv.LastCharacteristicFallbacks(*work) == 0);
            fv.ValidateAuxValues(state, auxiliary);
            std::optional<R::Array<double>::ReadLease> dense;
            std::optional<C::DerivedFields::ReadScope> owned, ghost;
            C::MeanThermodynamicView thermal;
            if (scenario)
            {
                REQUIRE(auxiliary.derivedOwned);
                REQUIRE(auxiliary.derivedGhost);
                owned.emplace(auxiliary.derivedOwned->ReadCurrent(state, equation));
                ghost.emplace(auxiliary.derivedGhost->ReadCurrent(state, equation));
                thermal = C::MeanThermodynamicView::Derived(*owned, DNDS::OptionalRef<const C::DerivedFields::ReadScope>(*ghost));
            }
            else
            {
                REQUIRE(auxiliary.meanThermodynamics);
                dense.emplace(auxiliary.meanThermodynamics->Read());
                thermal = dense->View<R::UniformTableRows>();
            }
            REQUIRE(thermal.HasCharacteristics());
            REQUIRE(thermal.RowCount() == fv.Metrics().NumCellProc());
            const auto read = state.ReadWithGhost();
            const auto flow = read.Block("flow"), species = read.Block("species");
            auto evaluator = equation.MakePointEvaluator(mixture.workers.Acquire(), {});
            auto rhoY = evaluator.SpeciesInput();
            for (I iCell = 0; iCell < thermal.RowCount(); ++iCell)
            {
                C::FlowState value;
                for (int j = 0; j < 5; ++j)
                    value[j] = flow(iCell, j);
                for (int j = 0; j < rhoY.nSpecies; ++j)
                    rhoY[j] = species(iCell, j);
                const auto properties = evaluator.Evaluate(value, rhoY);
                const auto cached = thermal.Characteristics(iCell);
                CHECK(cached.pressure == doctest::Approx(properties.thermo.pressure).epsilon(3e-11));
                CHECK(cached.soundSpeed == doctest::Approx(properties.thermo.soundSpeed).epsilon(3e-11));
                CHECK(cached.totalEnthalpy == doctest::Approx((value[4] + properties.thermo.pressure) / value[0]).epsilon(3e-11));
                CHECK(cached.pressureInternalEnergyDerivative == doctest::Approx(properties.thermo.gasConstant / properties.thermo.cv).epsilon(3e-11));
            }
            ConservativeIntegral(residual);
            if (baseline)
                Compare(residual, *baseline, 3e-10);
            else
                baseline.emplace(residual.Clone());
        }
    }
}

TEST_CASE("Finite-volume admissibility uses one prepared complete mixture policy")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.faceFallback = false;
    C::FiniteVolume fv(mesh, settings, mixture.equation);
    auto state = mixture.State(fv.Metrics().Space(), 1);
    mixture.Fill(state, fv.Metrics(), true);
    const auto policy = fv.PrepareAdmissibility(mixture.workers);
    CHECK(fv.Admissible(state, policy));
    CHECK_THROWS(fv.Admissible(state));
    if (mpi.rank == 0)
    {
        auto write = state.WriteOwned();
        write.Block("species")(0, 0) = -.01;
    }
    CHECK_FALSE(fv.Admissible(state, policy));
}

TEST_CASE("Mixture interface packets retain complete equation rows and reject a non-simplex publication")
{
    MixtureFixture mixture;
    const auto mesh = CFVTest::CartesianHexes(mpi, 4);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.boundaries["7"].kind = "interface";
    settings.numerics.reconstruction = "gradient";
    settings.numerics.faceFallback = false;
    C::FiniteVolume fv(mesh, settings, mixture.equation);
    auto state = mixture.State(fv.Metrics().Space(), 2);
    mixture.Fill(state, fv.Metrics(), false);
    auto packet = fv.BuildBoundaryInputs(mixture.workers);
    const auto fill = [&](const auto &points, C::EquationDofs &destination)
    {
        const auto input = state.ReadOwned();
        const auto sourceLayout = input.ComponentLayout();
        const auto source = sourceLayout.View();
        auto output = destination.WriteOwned();
        const auto destinationLayout = output.ComponentLayout();
        const auto values = destinationLayout.View();
        REQUIRE(values.ComponentNumber() == source.ComponentNumber());
        for (I iPoint = 0; iPoint < I(points.size()); ++iPoint)
            for (I j = 0; j < values.ComponentNumber(); ++j)
                values(iPoint, j) = source(points[size_t(iPoint)].cell, j);
    };
    fill(fv.BoundaryPoints(), packet.QuadratureValues());
    fill(fv.BoundaryFaces(), packet.FaceCenterValues());
    CHECK_NOTHROW(fv.PublishBoundaryInputs(packet, .25));
    CHECK_NOTHROW(fv.ValidateBoundaryInputs(packet, .25));

    fv.BeginBoundaryUpdate(packet);
    fill(fv.BoundaryPoints(), packet.QuadratureValues());
    fill(fv.BoundaryFaces(), packet.FaceCenterValues());
    bool corrupted = false;
    {
        auto write = packet.QuadratureValues().WriteOwned();
        const auto flow = write.Block("flow"), species = write.Block("species");
        const auto &points = fv.BoundaryPoints();
        for (I iPoint = 0; iPoint < I(points.size()); ++iPoint)
            if (points[size_t(iPoint)].interface)
            {
                species(iPoint, 0) = -.01 * flow(iPoint, 0);
                corrupted = true;
                break;
            }
    }
    if (corrupted)
        CHECK_THROWS(fv.PublishBoundaryInputs(packet, .25));
    else
        CHECK_NOTHROW(fv.PublishBoundaryInputs(packet, .25));
}

TEST_CASE("Direct P1 mixture PP shares one thermal factor across split flow and species storage")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "gradient";
    settings.numerics.gradient = "leastSquares";
    settings.numerics.limiter = "none";
    settings.numerics.positivity = true;
    settings.numerics.positivityBetaCutoff = 0;
    settings.numerics.faceFallback = false;
    settings.numerics.viscousGradient = "polynomial";
    C::FiniteVolume fv(mesh, settings, mixture.equation);
    auto state = mixture.State(fv.Metrics().Space(), 1);
    mixture.Fill(state, fv.Metrics(), false);
    {
        auto write = state.WriteOwned();
        const auto flow = write.Block("flow");
        constexpr double oldKinetic = .5 * (.04 + .0016 + .0009);
        for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
        {
            const double rho = flow(iCell, 0);
            const double velocity = fv.Metrics().GetCellCent(iCell)[0] < .5
                                        ? -6
                                        : 6;
            flow(iCell, 1) = rho * velocity;
            flow(iCell, 2) = flow(iCell, 3) = 0;
            flow(iCell, 4) += .5 * rho * velocity * velocity -
                              rho * oldKinetic;
        }
    }
    auto reconstruction = fv.BuildReconstructionState(state);
    const auto storage = fv.InspectReconstructionStorage(*reconstruction);
    CHECK_FALSE(storage.polynomialPlan);
    CHECK(storage.coefficientFields == 0);
    CHECK(storage.coefficientTransferBindings == 0);
    auto workspace = fv.BuildWorkspace(state, mixture.workers);
    R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
    const auto stream = engine.CreateCPUStream();
    CHECK_NOTHROW(fv.PrepareReconstruction(
        state, *reconstruction, *workspace, engine, stream));
    const auto [limited, beta] = fv.PositivityReconstructionStats(*workspace);
    CHECK(limited > 0);
    CHECK(beta >= 0);
    CHECK(beta < 1);
}

TEST_CASE("Distributed prescribed mixture boundaries match independent nonconstant FV fluxes across layouts and launchers")
{
    MixtureFixture mixture;
    const auto equation = C::EquationSystem::FromIdealGasMixture(mixture.material, false);
    const auto mesh = CFVTest::CartesianHexes(mpi, 4);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.numerics.reconstruction = "constant";
    settings.numerics.faceFallback = false;
    auto &boundary = settings.boundaries["7"];
    boundary.kind = "analytic";
    std::vector<double> fractions(size_t(mixture.species), 0);
    fractions[0] = .04;
    fractions[3] = .21;
    fractions[5] = .03;
    fractions.back() = .72;
    const P::Chemistry::ConstSpeciesBufferView Y{fractions.data(), mixture.species};
    constexpr double density = .9, temperature = 1800;
    const double pressureSI = density * mixture.scales.rho0 * mixture.prototype.mixtureR(Y) * temperature;
    boundary.primitive = {density, -.2, .07, -.03, pressureSI / mixture.scales.Pressure()};
    for (int k = 0; k < mixture.species - 1; ++k)
        boundary.massFractions[mixture.material.SpeciesNames()[k]] = fractions[k];
    std::vector<double> prescribed(size_t(mixture.species + 4), 0);
    prescribed[0] = density;
    for (int d = 0; d < 3; ++d)
        prescribed[1 + d] = density * boundary.primitive[1 + d];
    prescribed[4] = density * mixture.prototype.mixtureIntEnergy(temperature, Y, pressureSI) / mixture.scales.SpecificEnergy() +
                    .5 * density * (.2 * .2 + .07 * .07 + .03 * .03);
    for (int k = 0; k < mixture.species - 1; ++k)
        prescribed[5 + k] = density * fractions[k];
    std::optional<C::EquationDofs> baseline;
    for (int scenario = 0; scenario < 2; ++scenario)
    {
        settings.execution.overlap = scenario;
        settings.implicit.collectiveChecks = scenario == 0;
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = mixture.State(fv.Metrics().Space(), scenario ? 2 : 1);
        fv.InitializeConstant(state, boundary.primitive, boundary.massFractions, temperature / mixture.scales.T0, mixture.workers);
        {
            const auto initial = state.ReadOwned();
            const auto flow = initial.Block("flow"), species = initial.Block("species");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                for (int j = 0; j < 5; ++j)
                    CHECK(flow(iCell, j) == doctest::Approx(prescribed[j]).epsilon(2e-12));
                for (int k = 0; k < mixture.species - 1; ++k)
                    CHECK(species(iCell, k) == doctest::Approx(prescribed[5 + k]).epsilon(2e-12));
            }
        }
        mixture.Fill(state, fv.Metrics(), false);
        auto result = C::EquationDofOperations::NewVector(state);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state, mixture.workers);
        R::EagerEngine engine(scenario ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(state, result, *reconstruction, *work, engine, stream);
        C::AuxiliaryRequest request;
        request.faceSpectralRadius = request.cellSpectralSum = true;
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        fv.PrepareAuxValues(state, auxiliary, 0, mixture.workers);
        request.meanThermodynamics = true;
        auto cachedAuxiliary = fv.BuildAuxiliaryValues(request);
        fv.PrepareAuxValues(state, cachedAuxiliary, 0, mixture.workers);
        {
            const auto a = auxiliary.faceSpectralRadius->Read(), b = cachedAuxiliary.faceSpectralRadius->Read();
            const auto direct = a.View<R::UniformTableRows>(), cached = b.View<R::UniformTableRows>();
            for (I iFace = 0; iFace < direct.RowCount(); ++iFace)
                for (I j = 0; j < direct.Width(); ++j)
                    CHECK(cached(iFace, j) == direct(iFace, j));
        }
        // Independent EOS backend and explicit face integration: no Cauchy
        // boundary, point producer or Riemann flux is used by this oracle.
        auto topologyRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
        auto metricRead = fv.Metrics().AcquireRead();
        const auto topology = topologyRead.HostView();
        const auto metric = metricRead.HostView();
        const auto values = state.ReadWithGhost(), residual = result.ReadOwned();
        const auto flow = values.Block("flow"), species = values.Block("species");
        const auto outFlow = residual.Block("flow"), outSpecies = residual.Block("species");
        std::vector<double> left(prescribed.size()), right(prescribed.size()), expected(prescribed.size());
        const auto gather = [&](I iCell, auto &point)
        {
            for (int j = 0; j < 5; ++j)
                point[j] = flow(iCell, j);
            for (int k = 0; k < mixture.species - 1; ++k)
                point[5 + k] = species(iCell, k);
        };
        const auto properties = [&](const auto &point)
        {
            double sum = 0, kinetic = 0;
            for (int k = 0; k < mixture.species - 1; ++k)
            {
                fractions[k] = point[5 + k] / point[0];
                sum += fractions[k];
            }
            fractions.back() = 1 - sum;
            for (int d = 1; d <= 3; ++d)
                kinetic += .5 * point[d] * point[d] / point[0];
            const double T = mixture.prototype.temperatureFromUV((point[4] - kinetic) / point[0] * mixture.scales.SpecificEnergy(),
                                                                 1 / (point[0] * mixture.scales.rho0), Y, 1400, 1e-12);
            const double p = point[0] * mixture.scales.rho0 * mixture.prototype.mixtureR(Y) * T;
            P::Thermodynamics::Properties thermo;
            mixture.prototype.EvaluateProperties(T, p, Y, thermo);
            return std::array<double, 2>{p / mixture.scales.Pressure(), thermo.soundSpeed / mixture.scales.U0};
        };
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            std::fill(expected.begin(), expected.end(), 0);
            gather(iCell, left);
            const auto lp = properties(left);
            const auto faces = topology.CellFaces(iCell);
            for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(faces.size()); ++slot)
            {
                const auto face = faces[slot], other = topology.CellFaceOther(iCell, face, slot);
                const int side = topology.CellIsFaceBack(iCell, face, slot) ? 0 : 1;
                if (other == DNDS::UnInitIndex)
                    right = prescribed;
                else
                    gather(other, right);
                const auto rp = properties(right);
                for (I q = metric.FaceQuadratureBegin(face); q < metric.FaceQuadratureEnd(face); ++q)
                {
                    const auto normal = ((side ? -1 : 1) * metric.GetFaceNorm(q)).eval();
                    double ul = 0, ur = 0;
                    for (int d = 0; d < 3; ++d)
                    {
                        ul += left[1 + d] / left[0] * normal[d];
                        ur += right[1 + d] / right[0] * normal[d];
                    }
                    const double alpha = std::max(std::abs(ul) + lp[1], std::abs(ur) + rp[1]);
                    // Flow uses acoustic Rusanov; conservative species use
                    // the independently specified velocity dissipation.
                    const double scalarAlpha = .5 * (std::abs(ul) + std::abs(ur));
                    for (I j = 0; j < I(expected.size()); ++j)
                    {
                        double fl = left[j] * ul, fr = right[j] * ur;
                        if (j >= 1 && j <= 3)
                        {
                            fl += lp[0] * normal[j - 1];
                            fr += rp[0] * normal[j - 1];
                        }
                        if (j == 4)
                        {
                            fl += lp[0] * ul;
                            fr += rp[0] * ur;
                        }
                        expected[j] -= metric.GetFaceWeight(q) * .5 * (fl + fr - (j < 5 ? alpha : scalarAlpha) * (right[j] - left[j]));
                    }
                }
            }
            for (int j = 0; j < 5; ++j)
                CHECK(outFlow(iCell, j) == doctest::Approx(expected[j]).epsilon(2e-10).scale(1e-10));
            for (int k = 0; k < mixture.species - 1; ++k)
                CHECK(outSpecies(iCell, k) == doctest::Approx(expected[5 + k]).epsilon(2e-10).scale(1e-10));
        }
        if (baseline)
            Compare(result, *baseline, 2e-12);
        else
            baseline = result;
    }
}

TEST_CASE("Mixture isothermal wall conducts the independent Fourier flux without species leakage")
{
    MixtureFixture mixture;
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.numerics.reconstruction = "constant";
    settings.numerics.faceFallback = false;
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.noRsOnWall = true;
    auto &wall = settings.boundaries["7"];
    wall.kind = "noSlipWall";
    wall.thermal = "isothermal";
    wall.temperature = 600 / mixture.scales.T0;
    C::FiniteVolume fv(mesh, settings, mixture.equation);
    auto state = mixture.State(fv.Metrics().Space(), 1), result = C::EquationDofOperations::NewVector(state);
    std::vector<double> Y(size_t(mixture.species), 0);
    Y[0] = .04;
    Y[3] = .21;
    Y[5] = .03;
    Y.back() = .72;
    std::map<std::string, double> composition;
    for (int k = 0; k < mixture.species; ++k)
        composition[mixture.material.SpeciesNames()[k]] = Y[k];
    fv.InitializeConstant(state, {.8, 0, 0, 0, 1}, composition, 1400 / mixture.scales.T0, mixture.workers);
    auto reconstruction = fv.BuildReconstructionState(state);
    auto work = fv.BuildWorkspace(state, mixture.workers);
    R::EagerEngine engine(2);
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(state, result, *reconstruction, *work, engine, stream);
    const P::Chemistry::ConstSpeciesBufferView fractions{Y.data(), mixture.species};
    const double r1 = .8, r2 = r1 * 1400 / 600;
    const double p = r1 * mixture.scales.rho0 * mixture.prototype.mixtureR(fractions) * 1400;
    const double e1 = mixture.prototype.mixtureIntEnergy(1400, fractions, p) / mixture.scales.SpecificEnergy();
    const double e2 = mixture.prototype.mixtureIntEnergy(600, fractions, p) / mixture.scales.SpecificEnergy();
    // Euler noRsOnWall uses the wall state for the flux's thermodynamics,
    // retaining the previously formed conservative exterior jump gradient.
    const double rho = r2, energy = e2, temperature = 600;
    P::Thermodynamics::Properties thermo;
    P::Transport::Properties transport;
    mixture.prototype.EvaluateProperties(temperature, rho * mixture.scales.rho0 * mixture.prototype.mixtureR(fractions) * temperature, fractions, thermo, &transport);
    // The unit cube's reflected cell-centroid separation is one; composition
    // is constant, so the independent full-energy chain rule has no dY term.
    const double gradient = (r2 * e2 - r1 * e1 - energy * (r2 - r1)) / (rho * thermo.cv / mixture.scales.HeatCapacity());
    const double expected = 6 * transport.conductivity / mixture.scales.Conductivity() * gradient;
    CHECK(expected < -1e-8);
    double actual = 0;
    {
        const auto read = result.ReadOwned();
        const auto flow = read.Block("flow"), species = read.Block("species");
        for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
        {
            actual += flow(iCell, 4);
            CHECK(std::abs(flow(iCell, 0)) < 1e-12);
            for (int k = 0; k < mixture.species - 1; ++k)
                CHECK(std::abs(species(iCell, k)) < 1e-12);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &actual, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(actual == doctest::Approx(expected).epsilon(2e-10));
}

TEST_CASE("Tagged mixture state initializes distributed Domain and prescribed boundary through one resolver")
{
    MixtureFixture mixture;
    auto values = std::vector<double>(5 + mixture.species - 1, 0);
    values[0] = .8 * mixture.scales.rho0;
    values[1] = .2 * mixture.scales.U0;
    values[2] = -.04 * mixture.scales.U0;
    values[3] = .03 * mixture.scales.U0;
    values[4] = 1400;
    values[5] = .027;
    values[8] = .22;
    values[10] = .02;
    C::DomainSettings settings;
    settings.equation.eos = "idealGasMixture";
    settings.equation.mixture.mechanismFile = mixture.prototype.mechanismFile();
    settings.equation.referenceScales = {{"rho0", mixture.scales.rho0}, {"U0", mixture.scales.U0}, {"L0", mixture.scales.L0}, {"T0", mixture.scales.T0}};
    settings.initialState = settings.referenceState = C::Json{{"type", "primRhoT_phy"}, {"state", values}};
    settings.boundaries["101"].kind = "analytic";
    settings.boundaries["101"].state = settings.initialState;
    settings.numerics.reconstruction = "constant";
    settings.numerics.faceFallback = false;
    settings.numerics.positivity = false;
    const auto config = C::SimulationConfig::ResolveCollective({{"defaults", C::Json(settings)}}, mpi.comm);
    const auto normalized = C::ResolvePhysicalStateInputs(config.Domains()[0].settings, mixture.equation, mixture.workers);
    const auto expected = normalized.initialConservative;
    for (const auto *layout : {"aos", "soa", "aosoa32"})
    {
        auto input = config.Domains()[0].settings;
        input.execution.layout = layout;
        // The periodic volume has no physical zone; exercise the prescribed
        // boundary separately below with the same resolved state.
        input.boundaries.clear();
        C::Domain domain("tagged", CauchyTest::PeriodicQuads(mpi, 4), input);
        const auto before = domain.Dofs().Clone();
        {
            auto read = domain.Dofs().ReadOwned();
            I offset = 0;
            for (const auto &block : mixture.equation.Schema().blocks)
            {
                const auto rows = read.Block(block.name);
                for (I iCell = 0; iCell < rows.owned.RowCount(); ++iCell)
                    for (I j = 0; j < rows.ComponentNumber(); ++j)
                        CHECK(rows(iCell, j) == doctest::Approx(expected[offset + j]).epsilon(2e-12));
                offset += I(block.components.size());
            }
        }
        REQUIRE(domain.Advance(1e-5).accepted);
        Compare(domain.Dofs(), before, 2e-12);
    }
    C::BoundarySystem boundary(normalized.settings.boundaries, mixture.equation);
    auto evaluator = mixture.equation.MakePointEvaluator(mixture.workers.Acquire(), {});
    auto rhoY = evaluator.SpeciesInput();
    for (int k = 0; k < rhoY.nSpecies; ++k)
        rhoY[k] = 0;
    C::FlowState inside{1, 0, 0, 0, 1};
    const auto value = boundary.At(101).FaceValue(evaluator, inside, rhoY, {1, 0, 0}, {0, 0, 0}, 0);
    for (int j = 0; j < 5; ++j)
        CHECK(value[j] == doctest::Approx(expected[j]).epsilon(2e-12));
    for (int k = 0; k < rhoY.nSpecies; ++k)
        CHECK(rhoY[k] == doctest::Approx(expected[5 + k]).epsilon(2e-12));
}

TEST_CASE("Explicit mixture rejects inadmissible raw flow or species and retries without increment compression")
{
    MixtureFixture mixture;
    bool reactive = false;
    SUBCASE("Flow density rejection") {}
    SUBCASE("Species rejection with conserved flow5") { reactive = true; }
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.equation.eos = "idealGasMixture";
    settings.equation.mixture.mechanismFile = mixture.prototype.mechanismFile();
    settings.equation.referenceScales = {{"rho0", mixture.scales.rho0}, {"U0", mixture.scales.U0}, {"L0", mixture.scales.L0}, {"T0", mixture.scales.T0}};
    settings.equation.source = reactive ? "chemical" : "none";
    if (reactive)
    {
        for (const auto &name : mixture.material.SpeciesNames())
            settings.initialMassFractions[name] = 1e-6;
        settings.initialMassFractions["H2"] = .027;
        settings.initialMassFractions["O2"] = .22;
        settings.initialMassFractions["H2O"] = .02;
        settings.initialMassFractions["N2"] = 1 - .267 - (mixture.species - 4) * 1e-6;
    }
    else
        settings.initialMassFractions = {{"O2", .22}, {"N2", .78}};
    settings.initialTemperature = (reactive ? 1400 : 1000) / mixture.scales.T0;
    settings.numerics.reconstruction = "constant";
    settings.numerics.faceFallback = false;
    settings.numerics.positivity = true;
    settings.execution.hostThreads = 2;
    settings.execution.layout = "soa";
    const auto config = C::SimulationConfig::ResolveCollective({{"defaults", C::Json(settings)}}, mpi.comm);
    const auto factory = [&](auto, const auto &, const auto &)
    { return mesh; };
    C::Simulation simulation(config, mpi.comm, factory);
    auto &domain = simulation.GetDomain(0);
    {
        auto state = domain.Dofs().NumericAlias();
        auto write = state.WriteOwned();
        for (I iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
        {
            const auto values = write.Block(iBlock);
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const double density = reactive ? 1. : domain.Discretization().Metrics().GetCellCent(iCell)[0] < .5 ? .2
                                                                                                                    : 2.;
                for (I j = 0; j < values.ComponentNumber(); ++j)
                    values(iCell, j) *= density;
            }
        }
    }
    const auto original = domain.Dofs().Clone();
    const C::EquationDofOperations::Binding accepted(domain.Dofs());
    double criticalDt = std::numeric_limits<double>::infinity();
    {
        const auto &rhs = domain.Residual();
        const auto initial = original.ReadOwned(), residual = rhs.ReadOwned();
        const auto values = initial.Block(reactive ? "species" : "flow"), rate = residual.Block(reactive ? "species" : "flow");
        for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            for (I j = 0; j < (reactive ? values.ComponentNumber() : 1); ++j)
                if (values(iCell, j) > 0 && rate(iCell, j) < 0)
                    criticalDt = std::min(criticalDt, -values(iCell, j) * domain.Discretization().Metrics().GetCellVol(iCell) / rate(iCell, j));
    }
    MPI_Allreduce(MPI_IN_PLACE, &criticalDt, 1, MPI_DOUBLE, MPI_MIN, mpi.comm);
    REQUIRE(std::isfinite(criticalDt));
    REQUIRE(criticalDt > 0);
    CHECK_FALSE(simulation.Advance(2 * criticalDt).accepted);
    CHECK_NOTHROW(accepted.Check(domain.Dofs(), true));
    CHECK(simulation.Time() == 0);
    CHECK(simulation.Step() == 0);
    double dt = criticalDt;
    bool completed = false;
    std::string lastRejection;
    for (int retry = 0; retry < 16; ++retry, dt *= .5)
    {
        const auto result = simulation.Advance(dt);
        if (result)
        {
            completed = true;
            break;
        }
        lastRejection = result.reason;
        CHECK_NOTHROW(accepted.Check(domain.Dofs(), true));
        CHECK(simulation.Time() == 0);
    }
    INFO("criticalDt=" << criticalDt << ", nextDt=" << dt << ", lastRejection=" << lastRejection);
    REQUIRE(completed);
    CHECK(simulation.Time() == dt);
    CHECK(simulation.Step() == 1);
    C::Simulation direct(config, mpi.comm, factory);
    auto fresh = direct.GetDomain(0).Dofs().NumericAlias();
    {
        const auto source = original.ReadOwned();
        auto destination = fresh.WriteOwned();
        for (I iBlock = 0; iBlock < original.BlockNumber(); ++iBlock)
        {
            const auto input = source.Block(iBlock);
            const auto output = destination.Block(iBlock);
            for (I iCell = 0; iCell < input.RowCount(); ++iCell)
                for (I j = 0; j < input.ComponentNumber(); ++j)
                    output(iCell, j) = input(iCell, j);
        }
    }
    REQUIRE(direct.Advance(dt).accepted);
    Compare(domain.Dofs(), direct.GetDomain(0).Dofs(), 0);
    auto oracle = original.Clone();
    auto rhs = C::EquationDofOperations::NewVector(oracle);
    const auto &fv = domain.Discretization();
    auto reconstruction = fv.BuildReconstructionState(oracle);
    auto work = fv.BuildWorkspace(oracle, domain.ChemicalWorkers());
    R::EagerEngine engine(2);
    const auto stream = engine.CreateCPUStream();
    const double history[3]{0, .75, 1. / 3}, weight[3]{1, .25, 2. / 3}, stageTime[3]{0, dt, .5 * dt};
    for (int stage = 0; stage < 3; ++stage)
    {
        fv.Evaluate(oracle, rhs, *reconstruction, *work, engine, stream, stageTime[stage]);
        const auto base = original.ReadOwned(), residual = rhs.ReadOwned();
        auto output = oracle.WriteOwned();
        for (I iBlock = 0; iBlock < oracle.BlockNumber(); ++iBlock)
        {
            const auto initial = base.Block(iBlock), rate = residual.Block(iBlock);
            const auto values = output.Block(iBlock);
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                for (I j = 0; j < values.ComponentNumber(); ++j)
                    values(iCell, j) = history[stage] * initial(iCell, j) + weight[stage] *
                                                                                (values(iCell, j) + dt * rate(iCell, j) / fv.Metrics().GetCellVol(iCell));
        }
    }
    Compare(domain.Dofs(), oracle, 0);
}

TEST_CASE("Configured mixture Domains advance every block through production time and publication paths")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    std::vector<double> fractions(size_t(mixture.species), 1e-6);
    fractions[0] = .027;
    fractions[3] = .22;
    fractions[5] = .02;
    fractions.back() = 0;
    fractions.back() = 1 - std::accumulate(fractions.begin(), fractions.end(), 0.);
    C::Json composition = C::Json::object();
    for (int k = 0; k < mixture.species; ++k)
        composition[mixture.material.SpeciesNames()[k]] = fractions[k];
    C::Json input{{"defaults", {{"equation", {{"eos", "idealGasMixture"}, {"source", "chemical"}, {"mixture", {{"mechanismFile", mixture.prototype.mechanismFile()}, {"rho0", mixture.scales.rho0}, {"U0", mixture.scales.U0}, {"L0", mixture.scales.L0}, {"T0", mixture.scales.T0}}}}}, {"initialPrimitive", {.8, .2, -.04, .03, 1}}, {"initialTemperature", 1400 / mixture.scales.T0}, {"initialMassFractions", composition}, {"execution", {{"layout", "soa"}, {"fieldGrouping", "equationBlocks"}, {"hostThreads", 2}}}, {"numerics", {{"reconstruction", "constant"}, {"faceFallback", false}}}, {"implicit", {{"diagonal", "block"}, {"pseudoCFL", 1e4}, {"maxPseudoCFL", 1e8}, {"relativeTolerance", 1e-10}, {"absoluteTolerance", 1e-8}, {"residualComponents", C::Json::array()}, {"linear", {{"method", "gmres"}, {"preconditioner", "ilu"}, {"relativeTolerance", 1e-10}, {"absoluteTolerance", 1e-13}}}}}}}, {"time", {{"method", "bdf1"}}}};
    const auto startFractions = fractions;
    constexpr double totalTime = 1e-5;
    double temperature = 1400;
    mixture.prototype.advanceConstVolume(temperature, .8 * mixture.scales.rho0, {fractions.data(), mixture.species}, 1,
                                         totalTime * mixture.scales.Time(), 1e-12, 1e-18);
    for (int scenario = 0; scenario < 4; ++scenario)
    {
        const int steps = scenario == 0 ? 8 : 4;
        input["time"]["method"] = scenario == 0 ? "ssprk3" : scenario == 1 ? "bdf1"
                                                         : scenario == 2   ? "esdirk4"
                                                                           : "ditr";
        if (scenario >= 3)
        {
            // Reaction-dominated production profile: use the requested U2R1.
            // U2R2 remains covered by the independent public time-method oracle
            // and nonreacting Domain tests; it need not converge this stiff
            // reactor within the same pseudo-time budget.
            input["time"]["ditr"] = C::Json{{"variant", "U2R1"}};
        }
        if (scenario == 3)
        {
            auto names = C::Json::array();
            for (int k = 0; k < mixture.species - 1; ++k)
                names.push_back("rhoY:" + mixture.material.SpeciesNames()[k]);
            input["defaults"]["implicit"]["residualComponents"] = std::move(names);
            // In this homogeneous reactor density cannot stop chemistry.
            // DITR observes after one pair of corrections. Density-only
            // acceptance need not imply convergence of the species equations.
            auto densityInput = input;
            densityInput["defaults"]["implicit"].erase("residualComponents");
            const auto densityConfig = C::SimulationConfig::ResolveCollective(densityInput, mpi.comm);
            C::Domain densityOnly("density-only", mesh, densityConfig.Domains()[0].settings, densityConfig.Time());
            const auto unchanged = densityOnly.Dofs().Clone();
            REQUIRE(densityOnly.Advance(totalTime / steps).accepted);
            CHECK(densityOnly.LastCoupledIterations() == 1);
            CHECK(densityOnly.LastCoupledTargetRatio() <= 1);
            REQUIRE(densityOnly.LastImplicitStages().size() == 1);
            CHECK(densityOnly.LastImplicitStages().front().converged);
            const auto after = densityOnly.Dofs().ReadOwned(), before = unchanged.ReadOwned();
            const auto flowAfter = after.Block("flow"), flowBefore = before.Block("flow");
            for (I iCell = 0; iCell < flowAfter.owned.RowCount(); ++iCell)
                for (I j = 0; j < flowAfter.ComponentNumber(); ++j)
                    CHECK(flowAfter(iCell, j) == doctest::Approx(flowBefore(iCell, j)).epsilon(1e-11));
        }
        input["defaults"]["execution"]["layout"] = scenario == 0 ? "aos" : scenario == 1 ? "soa"
                                                                                         : "aosoa32";
        input["defaults"]["execution"]["overlap"] = bool(scenario);
        input["defaults"]["execution"]["hostThreads"] = scenario ? 2 : 0;
        input["defaults"]["numerics"]["positivity"] = true;
        if (scenario == 2)
        {
            input["defaults"]["numerics"]["reconstruction"] = "variational";
            input["defaults"]["numerics"]["cfv"] = C::Json{{"maxOrder", 3}};
            input["defaults"]["numerics"]["limiter"] = "WBAP";
            input["defaults"]["numerics"]["characteristicLimiter"] = false;
        }
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        C::Domain domain("mixture", mesh, config.Domains()[0].settings, config.Time());
        REQUIRE(domain.Dofs().PhysicalFieldNumber() == 2);
        const auto initial = domain.Dofs().Clone();
        const auto dt = domain.StableStep(.5);
        CHECK(std::isfinite(dt));
        CHECK(dt > 0);
        CHECK(domain.Residual().ComponentNumber() == mixture.species + 4);
        int firstStep = 0;
        if (scenario == 1)
        {
            REQUIRE(domain.PrepareStep(totalTime / steps).accepted);
            const C::EquationDofOperations::Binding before(domain.Dofs());
            {
                const auto location = domain.Dofs().LocateBlock("species");
                const auto held = domain.Dofs().PhysicalField(location.physicalField).Owned().Read();
                CHECK_THROWS(domain.CommitStep()); // A species-only view must block all publication.
                CHECK_NOTHROW(before.Check(domain.Dofs(), true));
                CHECK(domain.Time() == 0);
                CHECK(domain.Epoch().acceptedStep == 0);
            }
            domain.CommitStep();
            firstStep = 1;
        }
        for (int step = firstStep; step < steps; ++step)
            REQUIRE(domain.Advance(totalTime / steps).accepted);
        CHECK(domain.Epoch().acceptedStep == uint64_t(steps));
        CHECK(domain.Time() == doctest::Approx(totalTime).epsilon(2e-15));
        if (scenario)
        {
            REQUIRE_FALSE(domain.LastImplicitStages().empty());
            for (const auto &report : domain.LastImplicitStages())
            {
                INFO("method=" << config.Time().method << " variant=" << config.Time().ditr.variant << " iterations=" << report.iterations << " target=" << report.residualTargetRatio
                               << " min_rhs_alpha=" << report.minimumRHSAlpha << " rhs_limited=" << report.rhsLimited
                               << " min_increment_alpha=" << report.minimumIncrementAlpha << " increment_limited=" << report.incrementLimited
                               << " residual=" << C::Json(report.residual).dump() << " initial=" << C::Json(report.initialResidual).dump());
                CHECK(report.converged);
                CHECK(report.residual.size() == size_t(mixture.species + 4));
            }
            if (scenario >= 3)
            {
                REQUIRE(domain.LastImplicitStages().size() == 1);
                CHECK(domain.LastCoupledTargetRatio() <= 1);
                CHECK(domain.LastImplicitStages().front().iterations == domain.LastCoupledIterations());
                CHECK(domain.LastImplicitStages().front().residualTargetRatio == domain.LastCoupledTargetRatio());
            }
        }
        double error = 0, change = 0;
        {
            const auto read = domain.Dofs().ReadOwned(), before = initial.ReadOwned();
            const auto flow = read.Block("flow"), oldFlow = before.Block("flow"), species = read.Block("species");
            for (I iCell = 0; iCell < flow.owned.RowCount(); ++iCell)
            {
                for (int j = 0; j < 5; ++j)
                    CHECK(flow(iCell, j) == doctest::Approx(oldFlow(iCell, j)).epsilon(2e-12).scale(1));
                for (int k = 0; k < mixture.species - 1; ++k)
                {
                    error = std::max(error, std::abs(species(iCell, k) - .8 * fractions[k]));
                    change = std::max(change, std::abs(species(iCell, k) - .8 * startFractions[k]));
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        CHECK(change > 1e-7);
        CHECK(error < 3e-7);
        const auto diagnostics = domain.Diagnostics(config.Domains()[0].settings, {});
        CHECK(diagnostics[2] > 0);
        CHECK(diagnostics[3] > 0);
        CHECK(diagnostics[13] > 0);
    }
    input["defaults"]["numerics"]["reconstruction"] = "constant";
    input["defaults"]["numerics"]["limiter"] = "none";
    input["time"]["method"] = "bdf1";
    input["domains"] = C::Json::array({{{"name", "a"}}, {{"name", "b"}}});
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    C::Simulation simulation(config, mpi.comm, [&](I, const auto &, const auto &)
                             { return mesh; });
    REQUIRE(simulation.Advance(totalTime).accepted);
    for (I iDomain = 0; iDomain < 2; ++iDomain)
    {
        const auto &domain = simulation.GetDomain(iDomain);
        const auto read = domain.Dofs().ReadOwned();
        const auto species = read.Block("species");
        double change = 0;
        for (I iCell = 0; iCell < species.owned.RowCount(); ++iCell)
            for (int k = 0; k < mixture.species - 1; ++k)
                change = std::max(change, std::abs(species(iCell, k) - .8 * startFractions[k]));
        MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        CHECK(change > 1e-7);
        CHECK(domain.Time() == simulation.Time());
    }
}

TEST_CASE("Physical mixture Jacobians match frozen-signal residual differences and solve through the logical driver")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.cfv.maxOrder = 1;
    settings.numerics.faceFallback = false;
    settings.equation.source = "chemical";
    C::FiniteVolume fv(mesh, settings, mixture.equation);
    auto state = mixture.State(fv.Metrics().Space(), 1), direction = mixture.State(fv.Metrics().Space(), 2);
    mixture.Fill(state, fv.Metrics(), false);
    const I nv = state.ComponentNumber();
    R::Array<double> inverseTau(R::RowLayout::Dense(fv.Metrics().NumCell(), 1));
    {
        const auto read = state.ReadOwned();
        const auto input = read.ComponentLayout();
        auto write = direction.WriteOwned();
        const auto output = write.ComponentLayout();
        auto pseudo = inverseTau.Write();
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const auto center = fv.Metrics().GetCellCent(iCell);
            for (I j = 0; j < nv; ++j)
                output.View()(iCell, j) = j < 5 ? .02 * (j + 1) * std::cos(2 * center[0] + center[1]) : .03 * input.View()(iCell, j) * std::cos(center[0] + 2 * center[1]);
            pseudo.View()(iCell, 0) = 2 + center[0];
        }
    }
    for (auto *field : {&state, &direction})
    {
        std::vector<R::TransferBinding> bindings;
        for (I iField = 0; iField < field->PhysicalFieldNumber(); ++iField)
            bindings.push_back(fv.Metrics().BindHalo(field->PhysicalField(iField)));
        std::vector<R::TransferOperation> active;
        for (auto &binding : bindings)
            active.push_back(binding.Begin());
        for (auto &operation : active)
            operation.Wait();
    }
    auto auxiliary = fv.BuildAuxiliaryValues({true, true});
    fv.PrepareAuxValues(state, auxiliary, 0, mixture.workers);
    C::JacobianPreparation controls{3, .7, false};
    auto assembled = fv.BuildLinearProblem(state, C::DiagonalStorage::Block);
    auto evaluated = fv.BuildLinearProblem(state, C::DiagonalStorage::Block, C::OffDiagonalStorage::SemiMatrixFree);
    fv.PrepareJacobian(state, auxiliary, inverseTau, *assembled, controls, 0, mixture.workers);
    fv.PrepareJacobian(state, auxiliary, inverseTau, *evaluated, controls, 0, mixture.workers);
    auto action = C::EquationDofOperations::NewVector(state), other = C::EquationDofOperations::NewVector(direction);
    assembled->Apply(direction, action);
    evaluated->Apply(direction, other);
    Compare(action, other, 2e-12);
    {
        const auto version = assembled->Diagonal().Version();
        const auto pin = assembled->OffDiagonal().Read();
        CHECK_THROWS(fv.PrepareJacobian(state, auxiliary, inverseTau, *assembled, controls, 0, mixture.workers));
        CHECK(assembled->Diagonal().Version() == version);
    }
    {
        auto topologyRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
        const auto topology = topologyRead.HostView();
        auto geometry = fv.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        const auto meanRead = state.ReadWithGhost(), deltaRead = direction.ReadWithGhost();
        const auto mean = meanRead.ComponentLayout(), delta = deltaRead.ComponentLayout();
        const auto resultRead = action.ReadOwned();
        const auto result = resultRead.ComponentLayout();
        const auto faceRead = auxiliary.faceSpectralRadius->Read(), pseudo = inverseTau.Read();
        const auto radius = faceRead.View();
        auto evaluator = mixture.equation.MakePointEvaluator(mixture.workers.Acquire(), {false, false, false, true, false});
        const auto point = [&](I iCell, double h)
        {
            Eigen::VectorXd value(nv);
            for (I j = 0; j < nv; ++j)
                value[j] = mean.View()(iCell, j) + h * delta.View()(iCell, j);
            return value;
        };
        const auto physicalFlux = [&](const Eigen::VectorXd &value, const C::Vector3 &normal)
        {
            const C::FlowState flow{value[0], value[1], value[2], value[3], value[4]};
            const auto properties = evaluator.Evaluate(flow, {value.data() + 5, mixture.species - 1}, 0, {});
            const auto first = C::Riemann::PhysicalFlux({flow, properties.thermo.pressure}, normal);
            Eigen::VectorXd flux(nv);
            double speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += flow[1 + d] / flow[0] * normal[d];
            for (I j = 0; j < nv; ++j)
                flux[j] = j < 5 ? first[j] : value[j] * speed;
            return flux;
        };
        const auto residual = [&](I iCell, double h)
        {
            const auto left = point(iCell, h);
            Eigen::VectorXd value = Eigen::VectorXd::Zero(nv);
            const auto source = evaluator.Evaluate({left[0], left[1], left[2], left[3], left[4]}, {left.data() + 5, mixture.species - 1}, 0, {false, false, false, true, false});
            for (int k = 0; k < mixture.species - 1; ++k)
                value[5 + k] = metric.GetCellVol(iCell) * source.chemicalSource[k];
            const auto faces = topology.CellFaces(iCell);
            for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(faces.size()); ++slot)
            {
                const auto iFace = faces[slot], neighbor = topology.CellFaceOther(iCell, iFace, slot);
                REQUIRE(neighbor != DNDS::UnInitIndex);
                const auto right = point(neighbor, h);
                const auto auxiliaryFaces = fv.AuxiliaryFaces();
                const auto found = std::lower_bound(auxiliaryFaces.begin(), auxiliaryFaces.end(), iFace);
                REQUIRE(found != auxiliaryFaces.end());
                const auto iAux = found - auxiliaryFaces.begin();
                const int sign = topology.CellIsFaceBack(iCell, iFace, slot) ? 1 : -1;
                for (I q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
                {
                    const DNDS::Geom::tPoint normal = sign * metric.GetFaceNorm(q);
                    const C::Vector3 n{normal[0], normal[1], normal[2]};
                    const auto a = physicalFlux(left, n), b = physicalFlux(right, n);
                    const double lambda = radius(iAux, 0) + radius(iAux, 1);
                    value.noalias() -= .5 * metric.GetFaceWeight(q) * (a + b - lambda * (right - left));
                }
            }
            return value;
        };
        constexpr double h = 2e-6;
        double signal = 0;
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const Eigen::VectorXd derivative = (residual(iCell, h) - residual(iCell, -h)) / (2 * h);
            Eigen::VectorXd expected(nv), actual(nv);
            for (I j = 0; j < nv; ++j)
            {
                expected[j] = metric.GetCellVol(iCell) * (pseudo.View()(iCell, 0) + controls.physicalMass) * delta.View()(iCell, j) - controls.residualScale * derivative[j];
                actual[j] = result.View()(iCell, j);
            }
            signal = std::max(signal, actual.lpNorm<Eigen::Infinity>());
            CHECK((actual - expected).lpNorm<Eigen::Infinity>() < 3e-5 * std::max(1., expected.lpNorm<Eigen::Infinity>()));
        }
        MPI_Allreduce(MPI_IN_PLACE, &signal, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        CHECK(signal > .01);
    }
    C::LinearSolveSettings linear;
    linear.method = "gmres";
    linear.preconditioner = "ilu";
    linear.subspace = 20;
    linear.maxIterations = 100;
    linear.relativeTolerance = 1e-10;
    linear.absoluteTolerance = 1e-12;
    C::DofLinearSolver solver(evaluated, direction, [&](auto &field)
                              { return fv.Metrics().BindHalo(field); }, linear);
    const auto report = solver.Solve(action);
    CHECK(report.converged);
    CHECK(report.usable);
    Compare(direction, solver.Solution(), 1e-8);
    const auto generation = evaluated->OffDiagonalGeneration(), diagonalVersion = evaluated->Diagonal().Version();
    {
        auto write = state.WriteOwned();
        const auto species = write.Block("species");
        if (species.RowCount())
            species(0, 0) += 1e-8;
    }
    CHECK_THROWS(fv.PrepareJacobian(state, auxiliary, inverseTau, *evaluated, controls, 0, mixture.workers));
    CHECK(evaluated->OffDiagonalGeneration() == generation);
    CHECK(evaluated->Diagonal().Version() == diagonalVersion);
    evaluated->Apply(direction, other); // The already-frozen operator remains unchanged.
    Compare(action, other, 2e-12);
}

TEST_CASE("Actual mixture FV residual preserves separate/permuted fields and core overlap without chemistry in face queries")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.reconstructionIterations = 5;
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.characteristicLimiter = false;
    settings.numerics.faceFallback = false;
    C::FiniteVolume baseline(mesh, settings, mixture.equation);
    auto referenceState = mixture.State(baseline.Metrics().Space(), 0);
    mixture.Fill(referenceState, baseline.Metrics(), false);
    auto expected = C::EquationDofOperations::NewVector(referenceState);
    auto baseReconstruction = baseline.BuildReconstructionState(referenceState);
    auto baseWork = baseline.BuildWorkspace(referenceState, mixture.workers);
    R::EagerEngine engine(2);
    const auto stream = engine.CreateCPUStream();
    baseline.Evaluate(referenceState, expected, *baseReconstruction, *baseWork, engine, stream);
    ConservativeIntegral(expected);
    const C::AuxiliaryRequest all{true, true, true, true, true, true, true};
    auto referenceAuxiliary = baseline.BuildAuxiliaryValues(all);
    baseline.PrepareAuxValues(referenceState, referenceAuxiliary, .25, mixture.workers);
    for (int layout = 1; layout < 3; ++layout)
    {
        settings.execution.overlap = layout == 1;
        C::FiniteVolume fv(mesh, settings, mixture.equation);
        auto state = mixture.State(fv.Metrics().Space(), layout);
        mixture.Fill(state, fv.Metrics(), false);
        auto result = C::EquationDofOperations::NewVector(state);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state, mixture.workers);
        const auto before = state.PhysicalField(0).Owned().Version();
        fv.Evaluate(state, result, *reconstruction, *work, engine, stream);
        Compare(expected, result, 5e-11);
        CHECK(state.PhysicalField(0).Owned().Version() == before);
        auto request = all;
        if (layout == 1)
            request.derived.enabled[int(C::DerivedQuantity::Temperature)] = true;
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        std::vector<R::TransferBinding> bindings;
        std::vector<R::TransferOperation> active;
        for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            bindings.push_back(fv.Metrics().BindHalo(state.PhysicalField(iField)));
        for (auto &binding : bindings)
            active.push_back(binding.Begin());
        auto preparation = fv.BeginAuxValues(state, auxiliary, .25, mixture.workers);
        preparation.Apply(DNDS::CFV::ReconstructionRegion::Core);
        for (auto &operation : active)
            operation.Wait();
        preparation.Apply(DNDS::CFV::ReconstructionRegion::HaloDependent);
        preparation.Finish(auxiliary);
        fv.ValidateAuxValues(state, auxiliary, .25);
        CompareAuxiliary(referenceAuxiliary, auxiliary);
        // One late physical output conflict must reject on all ranks before
        // a field halo is posted or an earlier output generation is published.
        const auto version = result.PhysicalField(0).Owned().Version();
        std::optional<R::Array<double>::ReadLease> pin;
        if (mpi.rank == 0)
            pin.emplace(result.PhysicalField(result.PhysicalFieldNumber() - 1).Owned().Read());
        CHECK_THROWS(fv.Evaluate(state, result, *reconstruction, *work, engine, stream));
        CHECK(result.PhysicalField(0).Owned().Version() == version);
        // A species-only mutation invalidates the property publication too.
        {
            const auto location = state.LocateBlock("species");
            auto write = state.PhysicalField(location.physicalField).Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            if (values.RowCount())
                values(0, location.firstComponent) += 1e-8;
        }
        CHECK_THROWS(fv.ValidateAuxValues(state, auxiliary, .25));
    }
    if (mpi.size > 1)
    {
        auto inconsistent = mixture.State(baseline.Metrics().Space(), mpi.rank == 0 ? 0 : 1);
        CHECK_THROWS(baseline.BuildWorkspace(inconsistent, mixture.workers));
    }
}

TEST_CASE("Mixture volume quadrature adds the conservative chemical rate and leaves full energy unchanged")
{
    MixtureFixture mixture;
    C::DomainSettings settings;
    settings.equation.source = "chemical";
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.cacheDiffBase = false;
    settings.numerics.reconstructionIterations = 2;
    settings.numerics.viscousGradient = "polynomial";
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    settings.numerics.faceFallback = false;
    C::FiniteVolume fv(mesh, settings, mixture.equation);
    auto state = mixture.State(fv.Metrics().Space(), 1);
    mixture.Fill(state, fv.Metrics(), true);
    auto result = C::EquationDofOperations::NewVector(state);
    auto reconstruction = fv.BuildReconstructionState(state);
    auto work = fv.BuildWorkspace(state, mixture.workers);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(state, result, *reconstruction, *work, engine, stream);
    auto evaluator = mixture.equation.MakePointEvaluator(mixture.workers.Acquire(), {false, false, false, true, false});
    const auto read = state.ReadOwned(), residual = result.ReadOwned();
    const auto flow = read.Block("flow"), species = read.Block("species"), rhsFlow = residual.Block("flow"), rhsSpecies = residual.Block("species");
    for (I iCell = 0; iCell < flow.owned.RowCount(); ++iCell)
    {
        C::FlowState u{};
        for (int j = 0; j < 5; ++j)
            u[j] = flow(iCell, j);
        auto input = evaluator.SpeciesInput();
        for (int k = 0; k < mixture.species - 1; ++k)
            input[k] = species(iCell, k);
        const auto point = evaluator.Evaluate(u, input);
        const double volume = fv.Metrics().GetCellVol(iCell);
        bool active = false;
        for (int k = 0; k < mixture.species - 1; ++k)
        {
            CHECK(rhsSpecies(iCell, k) == doctest::Approx(volume * point.chemicalSource[k]).epsilon(1e-8).scale(1e-8));
            active |= std::abs(point.chemicalSource[k]) > 1e-7;
        }
        CHECK(active);
        for (int j = 0; j < 5; ++j)
            CHECK(std::abs(rhsFlow(iCell, j)) < 1e-10);
    }
}

TEST_CASE("Shared quadrature keeps chemical source and P3 positivity caches unique without changing results")
{
    struct DiagnosticsScope
    {
        bool wasEnabled = R::AllocationDiagnostics::Enabled();
        DiagnosticsScope() { R::AllocationDiagnostics::Enable(); }
        ~DiagnosticsScope()
        {
            if (!wasEnabled)
                R::AllocationDiagnostics::Enable(false);
        }
    } diagnostics;
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    std::optional<C::EquationDofs> referenceResidual, referenceLimited;
    for (bool cache : {false, true})
    {
        C::DomainSettings settings;
        settings.equation.source = "chemical";
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 5;
        settings.numerics.cfv.cacheDiffBase = cache;
        settings.numerics.cfv.cacheDiffBaseOrder = 1;
        settings.numerics.reconstructionIterations = 0; // Prescribed nonconstant polynomial.
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.faceFallback = false;
        settings.numerics.positivity = true;
        // PP uses exactly the source volume rule with no independent setting.
        settings.numerics.temperatureFloor = 900 / mixture.scales.T0;
        C::FiniteVolume fv(mesh, settings, mixture.equation);
        auto state = mixture.State(fv.Metrics().Space(), 0);
        mixture.Fill(state, fv.Metrics(), true);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state, mixture.workers);
        const I modes = fv.PolynomialReconstruction()->NRecDOF();
        auto guess = fv.ReconstructionGuess(*reconstruction);
        C::EquationDofOperations::Fill(guess, 0);
        {
            auto write = guess.WriteOwned();
            const auto flow = write.Block("flow");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
                flow(iCell, 4 * modes) = .4 * (1 + .25 * std::sin(6.283185307179586 * fv.Metrics().GetCellCent(iCell)[0]));
        }
        auto residual = C::EquationDofOperations::NewVector(state);
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(state, residual, *reconstruction, *work, engine, stream);
        engine.Close();
        if (referenceResidual)
            Compare(residual, *referenceResidual, 2e-10);
        else
            referenceResidual.emplace(std::move(residual));

        // These named records identify physical allocations, not aliases.
        // One face bank plus one source bank serves RHS and PP simultaneously;
        // the source bank stores values only, with no unused direct point data.
        I sourceBanks = 0, hotBanks = 0;
        for (const auto &record : R::AllocationDiagnostics::Snapshot().allocations)
        {
            if (!record.bytes)
                continue;
            const auto named = [&](const char *name)
            { return std::find(record.names.begin(), record.names.end(), name) != record.names.end(); };
            if (named("Cauchy/quadrature/volume/basis"))
            {
                ++sourceBanks;
                CHECK(record.bytes == size_t(fv.Metrics().NumCell() * 9 * modes) * sizeof(double));
            }
            bool hotBasis = false;
            for (const auto &name : record.names)
                hotBasis |= (name.find("CFV/quadrature/") == 0 || name.find("Cauchy/quadrature/") == 0) &&
                            name.size() >= 6 && name.compare(name.size() - 6, 6, "/basis") == 0;
            hotBanks += hotBasis;
            if (cache)
                CHECK_FALSE(named("Cauchy/quadrature/volume/points"));
        }
        CHECK(sourceBanks == I(cache && fv.Metrics().NumCell() > 0));
        CHECK(hotBanks == (cache && fv.Metrics().NumCell() > 0 ? 2 : 0));

        {
            auto write = guess.WriteOwned();
            const auto flow = write.Block("flow");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
                flow(iCell, 4 * modes) = 300;
        }
        fv.LimitReconstructionPP(state, *reconstruction, *work);
        const auto statistics = fv.PositivityReconstructionStats(*work);
        if (fv.Metrics().NumCell())
            CHECK(statistics.first > 0);
        auto limited = fv.FluxReconstruction(*reconstruction, *work);
        if (referenceLimited)
            Compare(limited, *referenceLimited, 2e-11);
        else
        {
            referenceLimited.emplace(C::EquationDofOperations::NewVector(limited));
            C::EquationDofOperations::Copy(*referenceLimited, limited);
        }
    }
}

TEST_CASE("Nonconstant assembled mixture diffusion matches an independent face integration with direct backend coefficients")
{
    MixtureFixture mixture(2e-4); // Resolve a substantial code-unit diffusion signal.
    constexpr I width = 4;
    constexpr double h = 1. / width;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, width);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.reconstructionIterations = 5;
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.faceFallback = false;
    const auto inviscid = C::EquationSystem::FromIdealGasMixture(mixture.material, false);
    C::FiniteVolume noDiffusion(mesh, settings, inviscid), diffusion(mesh, settings, mixture.equation);
    auto a = C::EquationDofs::Coupled(inviscid.Schema(), noDiffusion.Metrics().BuildField(mixture.species + 4));
    auto b = C::EquationDofs::Coupled(mixture.equation.Schema(), diffusion.Metrics().BuildField(mixture.species + 4));
    mixture.Fill(a, noDiffusion.Metrics(), false);
    mixture.Fill(b, diffusion.Metrics(), false);
    auto reconstructionA = noDiffusion.BuildReconstructionState(a), reconstructionB = diffusion.BuildReconstructionState(b);
    auto wa = noDiffusion.BuildWorkspace(a, mixture.workers), wb = diffusion.BuildWorkspace(b, mixture.workers);
    auto ra = C::EquationDofOperations::NewVector(a), rb = C::EquationDofOperations::NewVector(b);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    noDiffusion.Evaluate(a, ra, *reconstructionA, *wa, engine, stream);
    diffusion.Evaluate(b, rb, *reconstructionB, *wb, engine, stream);
    struct FacePoint
    {
        I cell;
        DNDS::Geom::tPoint normal;
        double weight;
    };
    std::vector<FacePoint> entries;
    std::vector<I> queryCells;
    std::vector<DNDS::Geom::tPoint> queryPoints;
    auto meshRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
    const auto topology = meshRead.HostView();
    auto metricsRead = diffusion.Metrics().AcquireRead();
    const auto metric = metricsRead.HostView();
    for (I iCell = 0; iCell < diffusion.Metrics().NumCell(); ++iCell)
    {
        const auto faces = topology.CellFaces(iCell);
        for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(faces.size()); ++slot)
        {
            const I face = faces[slot], other = topology.CellFaceOther(iCell, face, slot);
            const int side = topology.CellIsFaceBack(iCell, face, slot) ? 0 : 1;
            REQUIRE(other >= 0);
            for (I q = metric.FaceQuadratureBegin(face); q < metric.FaceQuadratureEnd(face); ++q)
            {
                entries.push_back({iCell, (side ? -1 : 1) * metric.GetFaceNorm(q), metric.GetFaceWeight(q)});
                queryCells.push_back(iCell);
                queryCells.push_back(other);
                queryPoints.push_back(DNDS::CFV::Detail::FacePointInCell(topology, face, side, metric.GetFacePoint(q)));
                queryPoints.push_back(DNDS::CFV::Detail::FacePointInCell(topology, face, 1 - side, metric.GetFacePoint(q)));
            }
        }
    }
    const I count = I(queryCells.size()), components = mixture.species + 4;
    R::Array<I> cells(R::RowLayout::Dense(count, 1));
    R::Array<double> points(R::RowLayout::Dense(count, 3));
    {
        auto cw = cells.Write();
        auto pw = points.Write();
        for (I q = 0; q < count; ++q)
        {
            cw.View()(q, 0) = queryCells[q];
            for (int d = 0; d < 3; ++d)
                pw.View()(q, d) = queryPoints[q][d];
        }
    }
    DNDS::CFV::PolynomialSamplingPlan sampling(*diffusion.PolynomialReconstruction(), cells, points);
    std::vector<R::Array<double>> samples;
    for (int derivative = 0; derivative < 3; ++derivative)
    {
        samples.emplace_back(R::RowLayout::Dense(count, components));
        sampling.Evaluate(b.PhysicalField(0), diffusion.FluxReconstruction(*reconstructionB, *wb).PhysicalField(0), samples.back(), derivative);
    }
    const auto valueRead = samples[0].Read(), dxRead = samples[1].Read(), dyRead = samples[2].Read();
    const auto values = valueRead.View<R::UniformTableRows>();
    const R::ArrayView<const double, R::AccessBoundsChecks, R::UniformTableRows> derivatives[2]{dxRead.View<R::UniformTableRows>(), dyRead.View<R::UniformTableRows>()};
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(diffusion.Metrics().NumCell(), mixture.species - 1);
    std::vector<double> face(size_t(components), 0), grad(size_t(components), 0), Y(size_t(mixture.species)), D(size_t(mixture.species)), J(size_t(mixture.species));
    for (I entry = 0; entry < I(entries.size()); ++entry)
    {
        const auto &quadrature = entries[entry];
        const I left = 2 * entry, right = left + 1;
        for (I j = 0; j < components; ++j)
        {
            face[j] = .5 * (values(left, j) + values(right, j));
            grad[j] = (values(right, j) - values(left, j)) / h;
            for (int d = 0; d < 2; ++d)
                grad[j] += .5 * (derivatives[d](left, j) + derivatives[d](right, j)) * quadrature.normal[d];
        }
        double sumY = 0, kinetic = 0;
        for (int k = 0; k < mixture.species - 1; ++k)
        {
            Y[k] = face[5 + k] / face[0];
            sumY += Y[k];
        }
        Y.back() = 1 - sumY;
        for (int d = 1; d <= 3; ++d)
            kinetic += .5 * face[d] * face[d] / face[0];
        const P::Chemistry::ConstSpeciesBufferView fractions{Y.data(), mixture.species};
        const double T = mixture.prototype.temperatureFromUV((face[4] - kinetic) / face[0] * mixture.scales.SpecificEnergy(), 1 / (face[0] * mixture.scales.rho0), fractions, 1400, 1e-12);
        const double p = face[0] * mixture.scales.rho0 * mixture.prototype.mixtureR(fractions) * T;
        P::Thermodynamics::Properties thermo;
        P::Transport::Properties transport;
        mixture.prototype.EvaluateProperties(T, p, fractions, thermo, &transport, {D.data(), mixture.species});
        double sumGradient = 0, sumJ = 0;
        for (int k = 0; k < mixture.species - 1; ++k)
        {
            const double gradient = (grad[5 + k] - Y[k] * grad[0]) / face[0];
            J[k] = -face[0] * D[k] / mixture.scales.Diffusivity() * gradient;
            sumGradient += gradient;
            sumJ += J[k];
        }
        J.back() = face[0] * D.back() / mixture.scales.Diffusivity() * sumGradient;
        sumJ += J.back();
        for (int k = 0; k < mixture.species - 1; ++k)
            expected(quadrature.cell, k) += quadrature.weight * (-J[k] + Y[k] * sumJ);
    }
    const auto ar = ra.ReadOwned(), br = rb.ReadOwned();
    const auto av = ar.Block("species"), bv = br.Block("species");
    double signal = 0;
    for (I iCell = 0; iCell < expected.rows(); ++iCell)
        for (int k = 0; k < mixture.species - 1; ++k)
        {
            CHECK(bv(iCell, k) - av(iCell, k) == doctest::Approx(expected(iCell, k)).epsilon(2e-8).scale(1e-9));
            signal = std::max(signal, std::abs(expected(iCell, k)));
        }
    MPI_Allreduce(MPI_IN_PLACE, &signal, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    CHECK(signal > 1e-8); // Dropped/reversed diffusion cannot pass via a tiny signal.
}

TEST_CASE("Nonuniform chemical volume sampling agrees with independent Cartesian Gauss quadrature and direct backend rates")
{
    MixtureFixture mixture;
    const auto equation = C::EquationSystem::FromIdealGasMixture(mixture.material, false);
    constexpr I width = 4;
    constexpr double h = 1. / width;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, width);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.intOrder = 5;
    settings.numerics.reconstructionIterations = 0; // Prescribed nonconstant polynomial.
    settings.numerics.faceFallback = false;
    C::FiniteVolume withoutSource(mesh, settings, equation);
    settings.equation.source = "chemical";
    auto quadratureSettings = settings;
    quadratureSettings.implicit.sourceJacobianMode = "rhsQuadrature";
    C::FiniteVolume withSource(mesh, settings, equation);
    C::FiniteVolume withSourceQuadrature(mesh, quadratureSettings, equation);
    auto a = C::EquationDofs::Coupled(equation.Schema(), withoutSource.Metrics().BuildField(mixture.species + 4));
    auto b = C::EquationDofs::Coupled(equation.Schema(), withSource.Metrics().BuildField(mixture.species + 4));
    auto bq = C::EquationDofs::Coupled(equation.Schema(), withSourceQuadrature.Metrics().BuildField(mixture.species + 4));
    mixture.Fill(a, withoutSource.Metrics(), true);
    mixture.Fill(b, withSource.Metrics(), true);
    mixture.Fill(bq, withSourceQuadrature.Metrics(), true);
    auto reconstructionA = withoutSource.BuildReconstructionState(a), reconstructionB = withSource.BuildReconstructionState(b),
         reconstructionBq = withSourceQuadrature.BuildReconstructionState(bq);
    auto wa = withoutSource.BuildWorkspace(a, mixture.workers), wb = withSource.BuildWorkspace(b, mixture.workers),
         wbq = withSourceQuadrature.BuildWorkspace(bq, mixture.workers);
    for (const auto &[fv, reconstruction, work] :
         std::vector<std::tuple<const C::FiniteVolume *, std::shared_ptr<C::FiniteVolume::ReconstructionState>,
                                std::shared_ptr<C::FiniteVolume::Workspace>>>{
             {&withoutSource, reconstructionA, wa}, {&withSource, reconstructionB, wb}, {&withSourceQuadrature, reconstructionBq, wbq}})
    {
        auto coefficients = fv->ReconstructionGuess(*reconstruction);
        C::EquationDofOperations::Fill(coefficients, 0);
        auto write = coefficients.WriteOwned();
        const auto flow = write.Block("flow");
        const I modes = fv->PolynomialReconstruction()->NRecDOF();
        for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            flow(iCell, 4 * modes) = .4 * (1 + .25 * std::sin(6.283185307179586 * fv->Metrics().GetCellCent(iCell)[0]));
    }
    auto ra = C::EquationDofOperations::NewVector(a), rb = C::EquationDofOperations::NewVector(b);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    withoutSource.Evaluate(a, ra, *reconstructionA, *wa, engine, stream);
    withSource.Evaluate(b, rb, *reconstructionB, *wb, engine, stream);
    const I rows = withSource.Metrics().NumCell(), count = rows * 9;
    auto auxiliaryA = withoutSource.BuildAuxiliaryValues({false, true}), auxiliaryB = withSource.BuildAuxiliaryValues({false, true}),
         auxiliaryBq = withSourceQuadrature.BuildAuxiliaryValues({false, true});
    withoutSource.PrepareAuxValues(a, auxiliaryA, 0, mixture.workers);
    withSource.PrepareAuxValues(b, auxiliaryB, 0, mixture.workers);
    withSourceQuadrature.PrepareCorrectionAux(bq, *reconstructionBq, *wbq, auxiliaryBq, engine, stream, 0);
    R::Array<double> inverseTau(R::RowLayout::Dense(rows, 1));
    {
        auto write = inverseTau.Write();
        for (I iCell = 0; iCell < rows; ++iCell)
            write.View()(iCell, 0) = 0;
    }
    auto matrixA = withoutSource.BuildLinearProblem(a, C::DiagonalStorage::Block), matrixB = withSource.BuildLinearProblem(b, C::DiagonalStorage::Block),
         matrixBq = withSourceQuadrature.BuildLinearProblem(bq, C::DiagonalStorage::Block);
    C::JacobianPreparation controls;
    withoutSource.PrepareJacobian(a, auxiliaryA, inverseTau, *matrixA, controls, 0, mixture.workers);
    const auto unpreparedVersion = matrixB->Diagonal().Version();
    CHECK_THROWS(withSource.PrepareJacobian(b, auxiliaryB, inverseTau, *matrixB, controls, 0, mixture.workers));
    CHECK(matrixB->Diagonal().Version() == unpreparedVersion);
    withSource.PrepareJacobian(b, auxiliaryB, inverseTau, *matrixB, controls, 0, mixture.workers,
                               C::FiniteVolume::JacobianContext{*reconstructionB, *wb});
    withSourceQuadrature.PrepareJacobian(bq, auxiliaryBq, inverseTau, *matrixBq, controls, 0, mixture.workers,
                                         C::FiniteVolume::JacobianContext{*reconstructionBq, *wbq});

    // The fused mean-source producer also freezes the mean snapshot required
    // by evaluated off-diagonal actions. Exercise a nonzero physical vector,
    // while the independent chemical mean-source oracle below checks values.
    auto evaluated = withSource.BuildLinearProblem(b, C::DiagonalStorage::Block, C::OffDiagonalStorage::SemiMatrixFree);
    withSource.PrepareJacobian(b, auxiliaryB, inverseTau, *evaluated, controls, 0, mixture.workers,
                               C::FiniteVolume::JacobianContext{*reconstructionB, *wb});
    auto assembledAction = C::EquationDofOperations::NewVector(b), evaluatedAction = C::EquationDofOperations::NewVector(b);
    matrixB->Apply(b, assembledAction);
    evaluated->Apply(b, evaluatedAction);
    Compare(assembledAction, evaluatedAction, 2e-11);

    R::Array<I> cells(R::RowLayout::Dense(count, 1));
    R::Array<double> points(R::RowLayout::Dense(count, 3));
    const double location[3]{-std::sqrt(3. / 5), 0, std::sqrt(3. / 5)}, weights[3]{5. / 9, 8. / 9, 5. / 9};
    {
        auto cw = cells.Write();
        auto pw = points.Write();
        for (I iCell = 0; iCell < rows; ++iCell)
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                {
                    const I q = iCell * 9 + j * 3 + i;
                    const auto center = withSource.Metrics().GetCellCent(iCell);
                    cw.View()(q, 0) = iCell;
                    pw.View()(q, 0) = center[0] + h * .5 * location[i];
                    pw.View()(q, 1) = center[1] + h * .5 * location[j];
                    pw.View()(q, 2) = 0;
                }
    }
    DNDS::CFV::PolynomialSamplingPlan sampling(*withSource.PolynomialReconstruction(), cells, points, false);
    R::Array<double> sampled(R::RowLayout::Dense(count, mixture.species + 4));
    sampling.Evaluate(b.PhysicalField(0), withSource.FluxReconstruction(*reconstructionB, *wb).PhysicalField(0), sampled);
    const auto sampledRead = sampled.Read(), meanRead = b.PhysicalField(0).Owned().Read();
    const auto values = sampledRead.View<R::UniformTableRows>(), means = meanRead.View<R::UniformTableRows>();
    const auto lhs = ra.ReadOwned(), rhs = rb.ReadOwned();
    const auto x = lhs.Block("species"), y = rhs.Block("species");
    std::vector<double> fractions(size_t(mixture.species)), rates(size_t(mixture.species)), expected(size_t(mixture.species - 1));
    const int nv = mixture.species + 4;
    std::vector<double> rawJacobian(size_t(mixture.species * nv));
    Eigen::MatrixXd expectedJacobian(nv, nv);
    const auto rateAt = [&](const auto &input, I row)
    {
        double sum = 0, kinetic = 0;
        for (int k = 0; k < mixture.species - 1; ++k)
        {
            fractions[k] = input(row, 5 + k) / input(row, 0);
            sum += fractions[k];
        }
        fractions.back() = 1 - sum;
        for (int d = 1; d <= 3; ++d)
            kinetic += .5 * input(row, d) * input(row, d) / input(row, 0);
        const P::Chemistry::ConstSpeciesBufferView Y{fractions.data(), mixture.species};
        const double T = mixture.prototype.temperatureFromUV((input(row, 4) - kinetic) / input(row, 0) * mixture.scales.SpecificEnergy(), 1 / (input(row, 0) * mixture.scales.rho0), Y, 1400, 1e-12);
        const double p = input(row, 0) * mixture.scales.rho0 * mixture.prototype.mixtureR(Y) * T;
        mixture.prototype.productionRatesAndJacobian(T, p, input(row, 0), input(row, 4), input(row, 1), input(row, 2), input(row, 3), 4, Y,
                                                     {rates.data(), mixture.species}, {rawJacobian.data(), mixture.species, nv, mixture.species});
        for (int k = 0; k < mixture.species - 1; ++k)
            rates[k] *= mixture.prototype.molecularWeights()[k] / mixture.scales.SourceRate();
    };
    double signal[4]{}, quadratureError = 0, quadratureMagnitude = 0;
    {
        const auto da = matrixA->Diagonal().Read(), db = matrixB->Diagonal().Read(), dq = matrixBq->Diagonal().Read();
        for (I iCell = 0; iCell < rows; ++iCell)
        {
            std::fill(expected.begin(), expected.end(), 0);
            expectedJacobian.setZero();
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                {
                    rateAt(values, iCell * 9 + j * 3 + i);
                    for (int k = 0; k < mixture.species - 1; ++k)
                    {
                        expected[k] += h * h * .25 * weights[i] * weights[j] * rates[k];
                        for (int column = 0; column < nv; ++column)
                            expectedJacobian(5 + k, column) += h * h * .25 * weights[i] * weights[j] * mixture.prototype.molecularWeights()[k] / mixture.scales.SourceRate() * rawJacobian[k + mixture.species * column];
                    }
                }
            rateAt(means, iCell);
            for (int column = 0; column < nv; ++column)
                for (int row = 0; row < nv; ++row)
                {
                    const double observed = db.View()(iCell, row + nv * column) - da.View()(iCell, row + nv * column);
                    const double meanOnly = row >= 5 ? h * h * mixture.prototype.molecularWeights()[row - 5] / mixture.scales.SourceRate() * rawJacobian[row - 5 + mixture.species * column] : 0;
                    CHECK(observed == doctest::Approx(-meanOnly).epsilon(1e-6).scale(1e-8));
                    signal[2] = std::max(signal[2], std::abs(meanOnly));
                    if (row >= 5)
                    {
                        signal[3] = std::max(signal[3], std::abs(expectedJacobian(row, column) - meanOnly));
                        const double observedQuadrature =
                            dq.View()(iCell, row + nv * column) - db.View()(iCell, row + nv * column);
                        const double expectedQuadrature = -(expectedJacobian(row, column) - meanOnly);
                        quadratureError = std::max(quadratureError, std::abs(observedQuadrature - expectedQuadrature));
                        quadratureMagnitude = std::max(quadratureMagnitude, std::abs(expectedQuadrature));
                    }
                }
            for (int k = 0; k < mixture.species - 1; ++k)
            {
                const double observed = y(iCell, k) - x(iCell, k);
                CHECK(observed == doctest::Approx(expected[k]).epsilon(1e-8).scale(1e-9));
                signal[0] = std::max(signal[0], std::abs(expected[k]));
                signal[1] = std::max(signal[1], std::abs(expected[k] - h * h * rates[k]));
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, signal, 4, MPI_DOUBLE, MPI_MAX, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, &quadratureError, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, &quadratureMagnitude, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    CHECK(signal[0] > 1e-10);
    CHECK(signal[1] > 1e-6 * signal[0]); // A mean/centroid-only source must fail this gate.
    CHECK(signal[2] > 1e-8);
    CHECK(signal[3] > 1e-6 * signal[2]); // A quadrature-integrated approximate Jacobian must fail the mean-state oracle above.
    CHECK(quadratureMagnitude > 1e-8);
    CHECK(quadratureError <= 2e-7 * quadratureMagnitude + 1e-10);
    const auto version = matrixB->Diagonal().Version();
    {
        auto changed = withSource.FluxReconstruction(*reconstructionB, *wb).PhysicalField(0).Owned();
        auto write = changed.Write();
        const auto coefficients = write.View<R::UniformTableRows>();
        if (coefficients.RowCount())
            coefficients(0, 4 * withSource.PolynomialReconstruction()->NRecDOF()) += 1e-8;
    }
    CHECK_THROWS(withSource.PrepareJacobian(
        b, auxiliaryB, inverseTau, *matrixB, controls, 0, mixture.workers,
        C::FiniteVolume::JacobianContext{*reconstructionB, *wb}));
    CHECK(matrixB->Diagonal().Version() == version);
}
TEST_CASE("Equation P3 positivity protects actual EOS samples and preserves unlimited guesses in split and permuted layouts")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 2; ++scenario)
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 4;
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.faceFallback = false;
        settings.numerics.positivity = true;
        settings.numerics.temperatureFloor = 900 / mixture.scales.T0;
        settings.numerics.positivityCompressToMean = !scenario;
        settings.equation.source = "chemical"; // PP always protects this same volume rule.
        C::FiniteVolume fv(mesh, settings, mixture.equation);
        auto state = mixture.State(fv.Metrics().Space(), scenario + 1);
        mixture.Fill(state, fv.Metrics(), true);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state, mixture.workers);
        auto guess = fv.ReconstructionGuess(*reconstruction);
        const int modes = fv.PolynomialReconstruction()->NRecDOF();
        {
            auto write = guess.WriteOwned();
            const auto flow = write.Block("flow");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                flow(iCell, 4 * modes) = scenario ? .002 : 300;
                if (scenario)
                {
                    flow(iCell, 4 * modes + 2) = -300;
                    flow(iCell, 4 * modes + 5) = 100;
                }
            }
        }
        const C::EquationDofOperations::Binding meansBefore(state), guessBefore(guess);
        fv.LimitReconstructionPP(state, *reconstruction, *work);
        CHECK_NOTHROW(meansBefore.Check(state, true));
        CHECK_NOTHROW(guessBefore.Check(guess, true));
        const auto statistics = fv.PositivityReconstructionStats(*work);
        CHECK(statistics.first > 0);
        CHECK(statistics.second == 0);
        {
            const auto read = fv.FluxReconstruction(*reconstruction, *work).ReadOwned();
            const auto flow = read.Block("flow");
            for (I iCell = 0; iCell < flow.owned.RowCount(); ++iCell)
            {
                CHECK(flow(iCell, 4 * modes) == (scenario ? .002 : 0));
                for (int mode = 2; mode < modes; ++mode)
                    CHECK(flow(iCell, 4 * modes + mode) == 0);
            }
        }
        // Independently query centers and the chemical integration points,
        // then recover T with the actual EOS, not the limiter's bound checker.
        const I count = fv.Metrics().NumCell() * 9;
        R::Array<I> cells(R::RowLayout::Dense(count, 1));
        R::Array<double> points(R::RowLayout::Dense(count, 3));
        {
            auto cw = cells.Write();
            auto pw = points.Write();
            const auto c = cw.View();
            const auto p = pw.View();
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto center = fv.Metrics().GetCellCent(iCell);
                for (int k = 0; k < 9; ++k)
                {
                    const I q = 9 * iCell + k;
                    c(q, 0) = iCell;
                    p(q, 0) = center[0] + (k % 3 - 1) * .25 * .5 * std::sqrt(.6);
                    p(q, 1) = center[1] + (k / 3 - 1) * .25 * .5 * std::sqrt(.6);
                    p(q, 2) = center[2];
                }
            }
        }
        DNDS::CFV::PolynomialSamplingPlan sampler(*fv.PolynomialReconstruction(), cells, points, false);
        std::vector<R::Array<double>> values;
        for (I f = 0; f < state.PhysicalFieldNumber(); ++f)
        {
            values.emplace_back(R::RowLayout::Dense(count, I(state.PhysicalField(f).Width())));
            sampler.Evaluate(state.PhysicalField(f), fv.FluxReconstruction(*reconstruction, *work).PhysicalField(f), values.back());
        }
        std::vector<R::Array<double>::ReadLease> leases;
        for (const auto &value : values)
            leases.emplace_back(value.Read());
        const auto fl = state.LocateBlock("flow"), sp = state.LocateBlock("species");
        const auto flow = leases[fl.physicalField].View(), species = leases[sp.physicalField].View();
        {
            auto evaluator = mixture.equation.MakePointEvaluator(mixture.workers.Acquire(), {});
            auto rhoY = evaluator.SpeciesInput();
            for (I q = 0; q < count; ++q)
            {
                C::FlowState u;
                for (int j = 0; j < 5; ++j)
                    u[j] = flow(q, fl.firstComponent + j);
                for (int k = 0; k < mixture.species - 1; ++k)
                    rhoY[k] = species(q, sp.firstComponent + k);
                CHECK(evaluator.Evaluate(u, rhoY).thermo.temperature > settings.numerics.temperatureFloor);
            }
        }
        // Run the full P3 PP/source orchestration from a cold uniform guess.
        // A uniform periodic state has zero flux divergence, leaving V*S(U).
        C::EquationDofOperations::Fill(guess, 0);
        auto residual = C::EquationDofOperations::NewVector(state), rates = C::EquationDofOperations::NewVector(state);
        R::EagerEngine engine(2);
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(state, residual, *reconstruction, *work, engine, stream);
        C::EvaluateChemicalSource(mixture.equation, state, rates, mixture.workers.Acquire());
        const auto rr = residual.ReadOwned(), sr = rates.ReadOwned();
        const auto rl = rr.ComponentLayout(), sl = sr.ComponentLayout();
        const auto r = rl.View(), source = sl.View();
        for (I iCell = 0; iCell < r.RowCount(); ++iCell)
            for (I j = 0; j < r.ComponentNumber(); ++j)
                CHECK(r(iCell, j) == doctest::Approx(fv.Metrics().GetCellVol(iCell) * source(iCell, j)).epsilon(1e-10).scale(1));
        CHECK_NOTHROW(meansBefore.Check(state, true));
    }
}
TEST_CASE("Equation workspace agrees PP controls before binding conditional halo fields")
{
    if (mpi.size < 2)
        return;
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.faceFallback = false;
    settings.numerics.positivity = mpi.rank == 0;
    C::FiniteVolume fv(mesh, settings, mixture.equation);
    auto state = mixture.State(fv.Metrics().Space(), 1);
    mixture.Fill(state, fv.Metrics(), true);
    CHECK_THROWS_WITH(fv.BuildWorkspace(state, mixture.workers), doctest::Contains("operation controls"));
}
TEST_CASE("Equation implicit PP compresses RHS or pseudo-time and repairs species without changing flow")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 2; ++scenario)
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "constant";
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.faceFallback = false;
        settings.numerics.positivity = true;
        settings.numerics.temperatureFloor = 300 / mixture.scales.T0;
        settings.implicit.iterations = 1;
        settings.implicit.relativeTolerance = settings.implicit.absoluteTolerance = 0;
        settings.implicit.positivity.rhsFPPMode = scenario ? "dTau" : "rhs";
        settings.implicit.positivity.rhsFPPReduction = scenario ? "minimumAll" : "cell";
        C::FiniteVolume fv(mesh, settings, mixture.equation);
        auto state = mixture.State(fv.Metrics().Space(), scenario + 1);
        mixture.Fill(state, fv.Metrics(), true);
        auto source = C::EquationDofOperations::NewVector(state);
        C::EquationDofOperations::Copy(source, state);
        C::EquationDofOperations::Scale(source, 20);
        {
            auto write = source.WriteOwned();
            const auto flow = write.Block("flow");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
                if (fv.Metrics().Cells().LocalToGlobal(iCell) == 0)
                    flow(iCell, 0) = -1e8;
        }
        R::EagerEngine engine(2);
        const auto stream = engine.CreateCPUStream();
        C::NonlinearSolver solver(fv, state, fv.BuildReconstructionState(state), fv.BuildWorkspace(state, mixture.workers), engine, stream,
                                  settings, A::ComponentSelection(state.ComponentNumber()), mixture.workers);
        const auto report = solver.Solve(state, source, 20, 1, .1);
        CHECK(report.iterations == 1);
        CHECK_FALSE(report.converged);
        CHECK(report.rhsLimited == 1);
        CHECK(report.minimumRHSAlpha > 0);
        CHECK(report.minimumRHSAlpha < 1);
        auto old = C::EquationDofOperations::NewVector(state);
        C::EquationDofOperations::Copy(old, state);
        auto increment = C::EquationDofOperations::NewVector(state);
        C::EquationDofOperations::Fill(increment, 0);
        {
            auto write = increment.WriteOwned();
            const auto species = write.Block("species");
            for (I iCell = 0; iCell < species.RowCount(); ++iCell)
                species(iCell, 0) = -100;
        }
        std::pair<DNDS::index, DNDS::real> correction;
        CHECK_NOTHROW(correction = solver.ApplyCorrection(state, increment));
        // Species simplex repair is an exact target projection, not a common
        // flow/species thermal alpha. It must therefore leave the scalar
        // compression diagnostics untouched while publishing the repaired mean.
        CHECK(correction.first == 0);
        CHECK(correction.second == 1);
        {
            const auto read = state.ReadOwned(), before = old.ReadOwned();
            const auto flow = read.Block("flow"), prior = before.Block("flow"), species = read.Block("species");
            for (I iCell = 0; iCell < flow.owned.RowCount(); ++iCell)
            {
                for (int j = 0; j < 5; ++j)
                    CHECK(flow(iCell, j) == prior(iCell, j));
                CHECK(species(iCell, 0) >= 1e-30);
                double sum = 0;
                for (int k = 0; k < mixture.species - 1; ++k)
                    sum += species(iCell, k);
                CHECK(sum <= flow(iCell, 0));
            }
        }
        // A masked flow correction may not silently repair frozen species.
        // Remove kinetic-energy protection: a density-only trial can then
        // reach the simplex boundary while its thermal energy stays safe.
        {
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
                for (int d = 1; d <= 3; ++d)
                    flow(iCell, d) = 0;
        }
        C::EquationDofOperations::Fill(increment, 0);
        {
            const auto read = state.ReadOwned();
            const auto mean = read.Block("flow");
            auto write = increment.WriteOwned();
            const auto flow = write.Block("flow");
            // The power-two Euler compression makes an arbitrarily huge
            // increment almost vanish. A 1.01*rho decrement instead reaches
            // about .01*rho while retaining safe internal energy here.
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
                flow(iCell, 0) = -1.01 * mean(iCell, 0);
        }
        const C::EquationDofOperations::Binding before(state);
        CHECK_THROWS(solver.ApplyCorrection(state, increment, 1, {0}));
        CHECK_NOTHROW(before.Check(state, true));
    }
}
TEST_CASE("Coupled mixture nonlinear stage recovers a manufactured target through separate or permuted arrays")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 2; ++scenario)
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = scenario ? "variational" : "constant";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 40;
        settings.numerics.reconstructionTolerance = 1e-9;
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.faceFallback = false;
        settings.implicit.requireConvergedReconstruction = scenario;
        settings.implicit.linear.method = "gmres";
        settings.implicit.linear.preconditioner = scenario ? "sgs" : "ilu";
        settings.implicit.linear.relativeTolerance = 1e-5;
        settings.implicit.relativeTolerance = 1e-8;
        settings.implicit.absoluteTolerance = 1e-10;
        settings.implicit.diagonal = "block";
        settings.implicit.jacobianAction = scenario ? "semiMatrixFree" : "assembled";
        settings.implicit.collectiveChecks = scenario;
        settings.execution.haloFreshnessChecks = true; // Umbrella off/on both run the same halo protocol.
        settings.execution.overlap = scenario;
        C::FiniteVolume fv(mesh, settings, mixture.equation);
        auto target = mixture.State(fv.Metrics().Space(), 0), state = mixture.State(fv.Metrics().Space(), scenario + 1);
        mixture.Fill(target, fv.Metrics(), true);
        C::EquationDofOperations::Copy(state, target);
        {
            auto write = state.WriteOwned();
            const auto layout = write.ComponentLayout();
            const auto u = layout.View();
            for (I iCell = 0; iCell < u.RowCount(); ++iCell)
                for (I j = 0; j < u.ComponentNumber(); ++j)
                    u(iCell, j) *= 1 + .001 * std::sin(2 * iCell + mpi.rank + j);
        }
        auto source = C::EquationDofOperations::NewVector(target);
        C::EquationDofOperations::Copy(source, target);
        C::EquationDofOperations::Scale(source, 20);
        R::EagerEngine engine(scenario ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state, mixture.workers);
        C::NonlinearSolver solver(fv, state, reconstruction, work, engine, stream, settings, A::ComponentSelection(state.ComponentNumber()), mixture.workers);
        const auto report = solver.Solve(state, source, 20, .5, .1);
        CAPTURE(scenario);
        CAPTURE(report.iterations);
        CAPTURE(report.residualTargetRatio);
        REQUIRE(report.converged);
        CHECK(report.linearIterations > 0);
        CHECK(report.residual.size() == size_t(mixture.species + 4));
        CHECK(report.residual.back() < 1e-9);
        CHECK(solver.ReconstructionConverged());
        Compare(state, target, 2e-9);
        // A rejected negative species update must not mutate any physical field.
        auto increment = C::EquationDofOperations::NewVector(state);
        C::EquationDofOperations::Fill(increment, 0);
        {
            auto write = increment.WriteOwned();
            const auto species = write.Block("species");
            for (I iCell = 0; iCell < species.RowCount(); ++iCell)
                species(iCell, 0) = -100;
        }
        const C::EquationDofOperations::Binding before(state);
        CHECK_THROWS(solver.ApplyCorrection(state, increment));
        CHECK_NOTHROW(before.Check(state, true));
        CHECK_THROWS(solver.Solve(state, state, 20, .5, .1));
    }
}

TEST_CASE("Production equation nonlinear stages advance a distributed reacting FV state toward Cantera")
{
    MixtureFixture mixture;
    const auto equation = C::EquationSystem::FromIdealGasMixture(mixture.material, false);
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.equation.source = "chemical";
    settings.numerics.reconstruction = "constant";
    settings.numerics.faceFallback = false;
    settings.implicit.linear.method = "gmres";
    settings.implicit.linear.preconditioner = "ilu";
    settings.implicit.linear.relativeTolerance = 1e-10;
    settings.implicit.linear.absoluteTolerance = 1e-13;
    settings.implicit.diagonal = "block";
    settings.implicit.pseudoCFL = 1e4;
    settings.implicit.maxPseudoCFL = 1e8;
    settings.implicit.relativeTolerance = 1e-10;
    settings.implicit.absoluteTolerance = 1e-8;
    C::FiniteVolume fv(mesh, settings, equation);
    auto initial = mixture.State(fv.Metrics().Space(), 0);
    mixture.Fill(initial, fv.Metrics(), true);
    std::vector<double> expected(size_t(mixture.species + 4)), fractions(size_t(mixture.species), 1e-6);
    {
        // Every rank uses the same analytic initial point, including any empty
        // owner. Do not infer reference composition from a local ghost row.
        fractions[0] = .027;
        fractions[3] = .22;
        fractions[5] = .02;
        fractions.back() = 0;
        fractions.back() = 1 - std::accumulate(fractions.begin(), fractions.end(), 0.);
        const double rho = .8;
        const P::Chemistry::ConstSpeciesBufferView y{fractions.data(), mixture.species};
        const double p = rho * mixture.scales.rho0 * mixture.prototype.mixtureR(y) * 1400;
        expected[0] = rho;
        expected[1] = rho * .2;
        expected[2] = rho * -.04;
        expected[3] = rho * .03;
        expected[4] = rho * mixture.prototype.mixtureIntEnergy(1400, y, p) / mixture.scales.SpecificEnergy() + .5 * rho * (.04 + .0016 + .0009);
    }
    constexpr double totalTime = 1e-5;
    double temperature = 1400;
    mixture.prototype.advanceConstVolume(temperature, expected[0] * mixture.scales.rho0, {fractions.data(), mixture.species}, 1,
                                         totalTime * mixture.scales.Time(), 1e-11, 1e-20);
    for (int k = 0; k < mixture.species - 1; ++k)
        expected[5 + k] = expected[0] * fractions[k];
    const auto run = [&](int steps)
    {
        auto accepted = mixture.State(fv.Metrics().Space(), 2), trial = mixture.State(fv.Metrics().Space(), 1);
        C::EquationDofOperations::Copy(accepted, initial);
        C::EquationDofOperations::Copy(trial, initial);
        R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        auto reconstruction = fv.BuildReconstructionState(trial);
        auto work = fv.BuildWorkspace(trial, mixture.workers);
        C::NonlinearSolver nonlinear(fv, trial, reconstruction, work, engine, stream, settings, A::ComponentSelection(trial.ComponentNumber()), mixture.workers);
        C::EquationTimeStepper stepper(accepted, "bdf1");
        const auto evaluate = [&](auto &state, auto &rate, double time, int)
        {
            fv.Evaluate(state, rate, *reconstruction, *work, engine, stream, time);
            const auto geometry = fv.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto write = rate.WriteOwned();
            const auto layout = write.ComponentLayout();
            const auto values = layout.View();
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                for (I j = 0; j < values.ComponentNumber(); ++j)
                    values(iCell, j) /= metric.GetCellVol(iCell);
        };
        const auto solve = [&](auto &state, const auto &source, double mass, double scale, double time, int stage)
        {
            const auto report = nonlinear.Solve(state, source, mass, scale, time, stage);
            CHECK(report.converged);
            CHECK(report.residual.size() == expected.size());
            return report.converged;
        };
        for (int step = 0; step < steps; ++step)
        {
            stepper.BeginStep(accepted, trial, step * totalTime / steps,
                              totalTime / steps, evaluate);
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
        double error = 0, change = 0;
        const auto read = accepted.ReadOwned(), old = initial.ReadOwned();
        const auto layout = read.ComponentLayout(), oldLayout = old.ComponentLayout();
        const auto u = layout.View(), before = oldLayout.View();
        for (I iCell = 0; iCell < u.RowCount(); ++iCell)
            for (I j = 0; j < u.ComponentNumber(); ++j)
            {
                error = std::max(error, std::abs(u(iCell, j) - expected[j]));
                change = std::max(change, std::abs(u(iCell, j) - before(iCell, j)));
                if (j < 5)
                    CHECK(u(iCell, j) == doctest::Approx(before(iCell, j)).epsilon(1e-12));
            }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        CHECK(change > 1e-9);
        return error;
    };
    const double coarse = run(4), fine = run(8);
    CAPTURE(coarse);
    CAPTURE(fine);
    CHECK(coarse > 1e-12);
    CHECK(fine < .8 * coarse);
    CHECK(fine < 1e-5);
}

#endif

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
