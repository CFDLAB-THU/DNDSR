#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include "CartesianFixture.hpp"
#include <limits>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace G = DNDS::Geom;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

static C::DomainSettings Settings(const std::string &variant, int layout = 0)
{
    C::DomainSettings settings;
    settings.boundaries[std::to_string(G::BC_ID_DEFAULT_WALL)].kind = "extrapolate";
    settings.execution.layout = layout == 0 ? "aos" : layout == 1 ? "soa"
                                                                  : "aosoa32";
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.boundaryDerivativeWeights = {1, 0, 0, 0};
    settings.numerics.reconstructionIterations = 2;
    settings.numerics.flux = "Roe_M9";
    settings.numerics.limiter = "LVR";
    settings.numerics.limitedVR = DNDS::CFV::LimitedVRSettings::ForVariant(variant);
    settings.numerics.positivity = true;
    settings.numerics.faceFallback = false;
    return settings;
}
static void Initialize(const C::FiniteVolume &fv, DNDS::CFV::FVField &mean, const C::PerfectGas &gas, bool expansion = false, bool constant = false)
{
    auto write = mean.Owned().Write();
    const auto u = write.View<R::UniformTableRows>();
    for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
    {
        const bool left = fv.Metrics().GetCellCent(iCell)[0] < .5;
        const auto value = constant ? gas.Conservative({1, .1, .2, 0, 1}) : expansion ? gas.Conservative({1, left ? 0. : 1., 0, 0, left ? 1. : 2.})
                                                                                      : gas.Conservative({left ? 8. : 1.4, left ? 7. : 0., 0, 0, left ? 116.5 : 1.});
        for (int j = 0; j < 5; ++j)
            u(iCell, j) = value[j];
    }
}
static void Append(std::vector<double> &values, const R::Array<double> &array)
{
    auto read = array.Read();
    const auto view = read.View<R::UniformTableRows>();
    for (I iCell = 0; iCell < view.RowCount(); ++iCell)
        for (I j = 0; j < view.RowSize(iCell); ++j)
        {
            CHECK(std::isfinite(view(iCell, j)));
            values.push_back(view(iCell, j));
        }
}
static R::Array<double> Snapshot(const R::Array<double> &array)
{
    R::Array<double> result(array.Layout());
    auto read = array.Read();
    auto write = result.Write();
    const auto input = read.View<R::UniformTableRows>();
    const auto output = write.View<R::UniformTableRows>();
    for (I iCell = 0; iCell < input.RowCount(); ++iCell)
        for (I j = 0; j < input.RowSize(iCell); ++j)
            output(iCell, j) = input(iCell, j);
    return result;
}
TEST_CASE("LVR variants expand defaults per input layer with explicit overrides and strict validation")
{
    C::Json input{{"defaults", Settings("v3")}};
    input["defaults"]["numerics"]["limitedVR"] = {{"variant", "v2"}, {"alphaMax", .75}};
    input["domains"] = C::Json::array({C::Json{{"name", "fluid"}}});
    input["domains"][0]["overrides"]["numerics"]["limitedVR"] = {{"neighborLayers", 2}, {"compression", {.1, .3}}};
    const auto v2 = C::SimulationConfig::Resolve(input, mpi.size);
    const auto &settings = v2.Domains()[0].settings.numerics.limitedVR;
    CHECK(settings.variant == "v2");
    CHECK(settings.alphaMax == .75);
    CHECK(settings.contactAlphaMax == 0);
    CHECK(settings.densityJump[0] == .2);
    CHECK(settings.neighborLayers == 2);
    CHECK((settings.compression == std::array<double, 2>{.1, .3}));
    input["domains"][0]["overrides"]["numerics"]["limitedVR"]["variant"] = "v3";
    const auto v3 = C::SimulationConfig::Resolve(input, mpi.size);
    CHECK(v3.Domains()[0].settings.numerics.limitedVR.alphaMax == 1);
    CHECK(v3.Domains()[0].settings.numerics.limitedVR.contactAlphaMax == 1);
    CHECK(v3.Domains()[0].settings.numerics.limitedVR.neighborLayers == 2);
    input["domains"][0]["overrides"]["numerics"]["limitedVR"]["variant"] = "v4";
    CHECK_THROWS_AS(C::SimulationConfig::Resolve(input, mpi.size), std::exception);
    input["domains"][0]["overrides"]["numerics"]["limitedVR"] = {{"pressureJump", {.1, .1}}};
    CHECK_THROWS_AS(C::SimulationConfig::Resolve(input, mpi.size), std::exception);
}
TEST_CASE("Native Cauchy LVR prepares active v2/v3 targets identically across layouts and CPU schedules")
{
    const auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24).mesh;
    for (const std::string variant : {"v2", "v3"})
        for (int layout = 0; layout < 3; ++layout)
            for (bool sor : {false, true})
            {
                std::vector<double> expected;
                for (bool overlap : {false, true})
                    for (int workers : {0, 2})
                    {
                        auto settings = Settings(variant, layout);
                        settings.execution.overlap = overlap;
                        settings.numerics.cfv.SORInstead = sor;
                        const auto equation = C::EquationSystem::Resolve(settings.equation);
                        C::FiniteVolume fv(mesh, settings, equation);
                        const auto shape = layout == 0 ? R::RowLayout::Dense(0, 5, 7) : layout == 1 ? R::RowLayout::ColMajor(0, 5, {16, 1})
                                                                                                    : R::RowLayout::TiledColMajor(0, 5, 32);
                        auto mean = fv.Metrics().BuildField(shape), residual = fv.Metrics().BuildField(shape);
                        Initialize(fv, mean, equation.Gas());
                        auto state = C::EquationDofs::Coupled(equation.Schema(), mean);
                        auto output = C::EquationDofs::Coupled(equation.Schema(), residual);
                        const auto version = mean.Owned().Version();
                        auto reconstruction = fv.BuildReconstructionState(state);
                        auto work = fv.BuildWorkspace(state);
                        R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
                        const auto stream = engine.CreateCPUStream();
                        for (int iteration = 0; iteration < 2; ++iteration)
                            fv.Evaluate(state, output, *reconstruction, *work, engine, stream);
                        CHECK(mean.Owned().Version() == version);
                        CHECK(fv.LastFaceFallbacks(*work) == 0);
                        REQUIRE(fv.LimitedVRAlpha(*work));
                        REQUIRE(fv.LimitedVRHistory(*reconstruction));
                        double maximum = 0;
                        {
                            auto read = fv.LimitedVRAlpha(*work)->Read();
                            for (I iCell = 0; iCell < read.View().RowCount(); ++iCell)
                                maximum = std::max(maximum, read.View()(iCell, 0));
                        }
                        MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
                        CHECK(maximum == doctest::Approx(std::max(settings.numerics.limitedVR.alphaMax, settings.numerics.limitedVR.contactAlphaMax)));
                        std::vector<double> values;
                        Append(values, residual.Owned());
                        Append(values, fv.ReconstructionGuess(*reconstruction).CoupledField().Owned());
                        Append(values, fv.FluxReconstruction(*reconstruction, *work).CoupledField().Owned());
                        Append(values, *fv.LimitedVRAlpha(*work));
                        Append(values, *fv.LimitedVRHistory(*reconstruction));
                        if (!overlap && workers == 0)
                            expected = std::move(values);
                        else
                            CHECK(values == expected);
                    }
            }
}
TEST_CASE("Native Cauchy LVR expansion admission changes the shock gate while constant fields stay unlimited")
{
    const auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24).mesh;
    for (bool constant : {false, true})
        for (bool expansion : {false, true})
        {
            auto settings = Settings("v2");
            settings.numerics.limitedVR.detectExpansion = expansion;
            settings.numerics.positivity = false;
            settings.numerics.viscousLimited = !expansion;
            const auto equation = C::EquationSystem::Resolve(settings.equation);
            C::FiniteVolume fv(mesh, settings, equation);
            auto mean = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
            Initialize(fv, mean, equation.Gas(), true, constant);
            auto state = C::EquationDofs::Coupled(equation.Schema(), mean);
            auto output = C::EquationDofs::Coupled(equation.Schema(), residual);
            auto reconstruction = fv.BuildReconstructionState(state);
            CHECK_FALSE(fv.InspectReconstructionStorage(*reconstruction).distinctFinalized);
            auto work = fv.BuildWorkspace(state);
            R::EagerEngine engine(2);
            const auto stream = engine.CreateCPUStream();
            fv.Evaluate(state, output, *reconstruction, *work, engine, stream);
            double maximum = 0, norm = 0;
            {
                const auto alpha = fv.LimitedVRAlpha(*work)->Read(), rhs = residual.Owned().Read();
                const auto a = alpha.View<R::FixedRows<1>>();
                const auto r = rhs.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < a.RowCount(); ++iCell)
                {
                    maximum = std::max(maximum, a(iCell, 0));
                    for (int j = 0; j < 5; ++j)
                        norm = std::max(norm, std::abs(r(iCell, j)));
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            if (constant || !expansion)
                CHECK(maximum == 0);
            else
                CHECK(maximum > .1);
            if (constant)
                CHECK(norm < 1e-12);
        }
}
TEST_CASE("Canonical EquationDofs LVR retains the established target gates and residual")
{
    const auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24).mesh;
    auto settings = Settings("v3");
    settings.execution.overlap = true;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto genericField = fv.Metrics().BuildField(5), genericResidualField = fv.Metrics().BuildField(5);
    auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(5), "packed");
    auto residual = C::EquationDofOperations::NewVector(state);
    Initialize(fv, genericField, equation.Gas());
    auto generic = C::EquationDofs::Coupled(equation.Schema(), genericField);
    auto genericResidual = C::EquationDofs::Coupled(equation.Schema(), genericResidualField);
    {
        const auto source = genericField.Owned().Read();
        auto destination = state.CoupledField().Owned().Write();
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                destination.View()(iCell, j) = source.View()(iCell, j);
    }
    auto genericWork = fv.BuildWorkspace(
        generic, std::nullopt,
        C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic});
    auto genericReconstruction = fv.BuildReconstructionState(generic);
    auto reconstruction = fv.BuildReconstructionState(state);
    auto work = fv.BuildWorkspace(state);
    R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
    const auto stream = engine.CreateCPUStream();
    for (int iteration = 0; iteration < 2; ++iteration)
    {
        fv.Evaluate(generic, genericResidual, *genericReconstruction, *genericWork, engine, stream);
        fv.Evaluate(state, residual, *reconstruction, *work, engine, stream);
    }
    REQUIRE(fv.LimitedVRAlpha(*genericWork));
    REQUIRE(fv.LimitedVRAlpha(*work));
    REQUIRE(fv.LimitedVRHistory(*genericReconstruction));
    REQUIRE(fv.LimitedVRHistory(*reconstruction));
    const auto compare = [](const R::Array<double> &a, const R::Array<double> &b)
    {
        const auto ar = a.Read(), br = b.Read();
        REQUIRE(ar.View().RowCount() == br.View().RowCount());
        REQUIRE(ar.View().Width() == br.View().Width());
        for (I iRow = 0; iRow < ar.View().RowCount(); ++iRow)
            for (I j = 0; j < ar.View().RowSize(iRow); ++j)
                CHECK(br.View()(iRow, j) == doctest::Approx(ar.View()(iRow, j)).epsilon(3e-13));
    };
    compare(genericResidualField.Owned(), residual.CoupledField().Owned());
    compare(fv.ReconstructionGuess(*genericReconstruction).CoupledField().Owned(), fv.ReconstructionGuess(*reconstruction).CoupledField().Owned());
    compare(fv.FluxReconstruction(*genericReconstruction, *genericWork).CoupledField().Owned(),
            fv.FluxReconstruction(*reconstruction, *work).CoupledField().Owned());
    compare(*fv.LimitedVRAlpha(*genericWork), *fv.LimitedVRAlpha(*work));
    compare(*fv.LimitedVRHistory(*genericReconstruction), *fv.LimitedVRHistory(*reconstruction));
    const auto guess = fv.ReconstructionGuess(*reconstruction).Clone();
    const auto history = Snapshot(*fv.LimitedVRHistory(*reconstruction));
    fv.ResetReconstruction(*reconstruction, *work);
    fv.RestoreReconstructionGuess(*reconstruction, guess, std::cref(history));
    compare(guess.CoupledField().Owned(), fv.ReconstructionGuess(*reconstruction).CoupledField().Owned());
    compare(history, *fv.LimitedVRHistory(*reconstruction));
}
TEST_CASE("LVR layer or capability differences reject before building divergent transfer bindings")
{
    if (mpi.size < 2)
        return;
    const auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24).mesh;
    for (bool inactive : {false, true})
    {
        auto settings = Settings("v3");
        if (inactive && mpi.rank == 0)
            settings.numerics.limitedVR.alphaMax = settings.numerics.limitedVR.contactAlphaMax = 0;
        else
            settings.numerics.limitedVR.neighborLayers = mpi.rank == 0 ? 1 : 0;
        const auto prepare = [&]
        {
            C::FiniteVolume fv(mesh, settings, C::EquationSystem::Resolve(settings.equation));
            auto mean = fv.Metrics().BuildField(5);
            auto state = C::EquationDofs::Coupled(fv.Equation().Schema(), mean);
            fv.BuildWorkspace(state);
        };
        // Different operation support now rejects during CFV preparation,
        // earlier than the workspace's LVR routing/configuration agreement.
        CHECK_THROWS_WITH_AS(prepare(), doctest::Contains("differ"), std::exception);
    }
}
TEST_CASE("Zero LVR caps recover the unlimited operator and omit unnecessary gate work")
{
    const auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24).mesh;
    std::vector<double> expected;
    for (bool disabled : {false, true})
    {
        auto settings = Settings("v3");
        settings.numerics.limitedVR.alphaMax = settings.numerics.limitedVR.contactAlphaMax = 0;
        if (disabled)
            settings.numerics.limiter = "none";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto mean = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
        Initialize(fv, mean, equation.Gas());
        auto state = C::EquationDofs::Coupled(equation.Schema(), mean);
        auto output = C::EquationDofs::Coupled(equation.Schema(), residual);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state);
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();
        for (int iteration = 0; iteration < 2; ++iteration)
            fv.Evaluate(state, output, *reconstruction, *work, engine, stream);
        CHECK_FALSE(fv.LimitedVRHistory(*reconstruction));
        CHECK_FALSE(fv.LimitedVRAlpha(*work));
        std::vector<double> values;
        Append(values, residual.Owned());
        Append(values, fv.ReconstructionGuess(*reconstruction).CoupledField().Owned());
        Append(values, fv.FluxReconstruction(*reconstruction, *work).CoupledField().Owned());
        if (!disabled)
            expected = std::move(values);
        else
            CHECK(values == expected);
    }
}
TEST_CASE("LVR history errors preserve accepted means and replace failed reconstruction output")
{
    const auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24).mesh;
    const auto settings = Settings("v3");
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto mean = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
    Initialize(fv, mean, equation.Gas());
    auto state = C::EquationDofs::Coupled(equation.Schema(), mean);
    auto output = C::EquationDofs::Coupled(equation.Schema(), residual);
    auto reconstruction = fv.BuildReconstructionState(state);
    auto work = fv.BuildWorkspace(state);
    R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(state, output, *reconstruction, *work, engine, stream);
    auto guess = fv.ReconstructionGuess(*reconstruction).Clone();
    auto history = Snapshot(*fv.LimitedVRHistory(*reconstruction));
    const auto meanVersion = mean.Owned().Version(), residualVersion = residual.Owned().Version();
    std::optional<R::Array<double>::ReadLease> held;
    if (mpi.rank == 0)
        held.emplace(fv.LimitedVRHistory(*reconstruction)->Read());
    CHECK_THROWS_AS(fv.Evaluate(state, output, *reconstruction, *work, engine, stream), std::exception);
    held.reset();
    CHECK(mean.Owned().Version() == meanVersion);
    CHECK(residual.Owned().Version() == residualVersion);
    // A peer can already have submitted reconstruction work when another rank
    // reports a local lease failure. Accepted means remain unchanged, but both
    // solver-owned reconstruction output and scratch are replaceable after the
    // coordinated failure. The saved warm input is the recovery publication.
    std::vector<double> before, after;
    auto recoveredState = fv.BuildReconstructionState(state), referenceState = fv.BuildReconstructionState(state);
    auto recovered = fv.BuildWorkspace(state), reference = fv.BuildWorkspace(state);
    fv.RestoreReconstructionGuess(*recoveredState, guess, std::cref(history));
    fv.RestoreReconstructionGuess(*referenceState, guess, std::cref(history));
    auto referenceResidual = fv.Metrics().BuildField(5);
    auto referenceOutput = C::EquationDofs::Coupled(equation.Schema(), referenceResidual);
    fv.Evaluate(state, output, *recoveredState, *recovered, engine, stream);
    fv.Evaluate(state, referenceOutput, *referenceState, *reference, engine, stream);
    before.clear();
    after.clear();
    Append(before, residual.Owned());
    Append(after, referenceResidual.Owned());
    CHECK(before == after);
    auto badHistory = Snapshot(history);
    if (mpi.rank == 0)
    {
        auto write = badHistory.Write();
        write.View()(0, 0) = std::numeric_limits<double>::quiet_NaN();
    }
    before.clear();
    Append(before, fv.ReconstructionGuess(*referenceState).CoupledField().Owned());
    Append(before, *fv.LimitedVRHistory(*referenceState));
    CHECK_THROWS_AS(fv.RestoreReconstructionGuess(*referenceState, guess, std::cref(badHistory)), std::exception);
    after.clear();
    Append(after, fv.ReconstructionGuess(*referenceState).CoupledField().Owned());
    Append(after, *fv.LimitedVRHistory(*referenceState));
    CHECK(before == after);
    // A conflicting history destination cannot first overwrite coefficients.
    held.emplace(fv.LimitedVRHistory(*referenceState)->Read());
    CHECK_THROWS_AS(fv.RestoreReconstructionGuess(*referenceState, guess, std::cref(history)), std::exception);
    held.reset();
    after.clear();
    Append(after, fv.ReconstructionGuess(*referenceState).CoupledField().Owned());
    Append(after, *fv.LimitedVRHistory(*referenceState));
    CHECK(before == after);
    CHECK_NOTHROW(fv.RestoreReconstructionGuess(*referenceState, guess, std::cref(history)));
    before.clear();
    Append(before, fv.ReconstructionGuess(*referenceState).CoupledField().Owned());
    Append(before, *fv.LimitedVRHistory(*referenceState));
    held.emplace(fv.LimitedVRHistory(*referenceState)->Read());
    CHECK_THROWS_AS(fv.ResetReconstruction(*referenceState, *reference), std::exception);
    held.reset();
    after.clear();
    Append(after, fv.ReconstructionGuess(*referenceState).CoupledField().Owned());
    Append(after, *fv.LimitedVRHistory(*referenceState));
    CHECK(before == after); // A local reset conflict must not erase the polynomial first.
    fv.ResetReconstruction(*referenceState, *reference);
    after.clear();
    Append(after, fv.ReconstructionGuess(*referenceState).CoupledField().Owned());
    Append(after, *fv.LimitedVRHistory(*referenceState));
    Append(after, *fv.LimitedVRAlpha(*reference));
    for (const auto value : after)
        CHECK(value == 0);
}
TEST_CASE("LVR periodic mean and target halos agree across split and unsplit CPU execution")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    for (const std::string variant : {"v2", "v3"})
    {
        std::vector<double> expected;
        for (bool overlap : {false, true})
        {
            auto settings = Settings(variant, 1);
            settings.boundaries.clear(); // Fully periodic fixture has no physical wall zone.
            settings.execution.overlap = overlap;
            const auto equation = C::EquationSystem::Resolve(settings.equation);
            C::FiniteVolume fv(mesh, settings, equation);
            auto mean = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
            Initialize(fv, mean, equation.Gas());
            auto state = C::EquationDofs::Coupled(equation.Schema(), mean);
            auto output = C::EquationDofs::Coupled(equation.Schema(), residual);
            const auto reconstruction = fv.BuildReconstructionState(state);
            const auto workspace = fv.BuildWorkspace(state);
            R::EagerEngine engine(overlap ? 2 : 0);
            const auto stream = engine.CreateCPUStream();
            for (int evaluation = 0; evaluation < 2; ++evaluation)
                fv.Evaluate(state, output, *reconstruction, *workspace, engine, stream, 0);
            std::vector<double> actual;
            Append(actual, residual.Owned());
            Append(actual, fv.ReconstructionGuess(*reconstruction).CoupledField().Owned());
            Append(actual, *fv.LimitedVRHistory(*reconstruction));
            Append(actual, *fv.LimitedVRAlpha(*workspace));
            if (!overlap)
                expected = actual;
            else
                CHECK(actual == expected);
            double maximum = 0;
            {
                const auto read = fv.LimitedVRAlpha(*workspace)->Read();
                const auto alpha = read.View<R::FixedRows<1>>();
                for (I iCell = 0; iCell < alpha.RowCount(); ++iCell)
                    maximum = std::max(maximum, alpha(iCell, 0));
            }
            MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            CHECK(maximum > .25); // Exercise active periodic target coefficients.
        }
    }
}
int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    MPI_Finalize();
    return result;
}
