#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include <cstdlib>
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Core/StepResult.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include "CartesianFixture.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Storage/AllocationDiagnostics.hpp"
#include <filesystem>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <csignal>
#include <hdf5.h>
#include <cstring>
#include <atomic>
#include <limits>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
static DNDS::MPIInfo mpi;
static C::EquationDofs Bind(const C::FiniteVolume &fv, const DNDS::CFV::FVField &field)
{
    return C::EquationDofs::Coupled(fv.Equation().Schema(), field);
}
static std::atomic<bool> countReductions{false};
static std::atomic<int> reductions{0};
extern "C" int MPI_Allreduce(const void *send, void *receive, int count, MPI_Datatype type, MPI_Op operation, MPI_Comm comm)
{
    if (countReductions)
        ++reductions;
    return PMPI_Allreduce(send, receive, count, type, operation, comm);
}
TEST_CASE("Finite-volume workspace identity follows its shared plan and rejects foreign or expired plans")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 3);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.positivity = false;
    settings.numerics.limiter = "none";
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    std::shared_ptr<C::FiniteVolume::Workspace> saved;
    std::optional<C::EquationDofs> bound;
    {
        C::FiniteVolume original(mesh, settings, equation);
        bound.emplace(Bind(original, original.Metrics().BuildField(5)));
        saved = original.BuildWorkspace(*bound);
        auto sharedPlan = original;
        CHECK_NOTHROW(sharedPlan.ValidateWorkspace(*bound, *saved));
        const auto movedPlan = std::move(sharedPlan);
        CHECK_THROWS_WITH_AS(sharedPlan.ValidateWorkspace(*bound, *saved), doctest::Contains("foreign"), std::exception);
        CHECK_NOTHROW(movedPlan.ValidateWorkspace(*bound, *saved));
        C::FiniteVolume independent(mesh, settings, equation);
        CHECK_THROWS_WITH_AS(independent.ValidateWorkspace(*bound, *saved), doctest::Contains("foreign"), std::exception);
    }
    // A retained scratch handle cannot acquire the identity of a later plan,
    // even if an allocator reuses the old Data address.
    C::FiniteVolume replacement(mesh, settings, equation);
    CHECK_THROWS_WITH_AS(replacement.ValidateWorkspace(*bound, *saved), doctest::Contains("foreign"), std::exception);
}
TEST_CASE("Finalized reconstruction aliases unfiltered storage and retains publication provenance")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 3);
    size_t unfilteredOwnedBytes = 0, unfilteredGhostBytes = 0;
    for (const bool filtered : {false, true})
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 1;
        settings.numerics.reconstructionIterations = 0;
        settings.numerics.limiter = filtered ? "WBAP" : "none";
        settings.numerics.positivity = false;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto field = fv.Metrics().BuildField(5);
        auto logical = Bind(fv, field);
        auto reconstruction = fv.BuildReconstructionState(logical);
        const auto storage = fv.InspectReconstructionStorage(*reconstruction);
        CHECK(storage.distinctFinalized == filtered);
        CHECK(storage.coefficientOwnedBytes > 0);
        if (!filtered)
        {
            unfilteredOwnedBytes = storage.coefficientOwnedBytes;
            unfilteredGhostBytes = storage.coefficientGhostBytes;
        }
        else
        {
            CHECK(storage.coefficientOwnedBytes == 2 * unfilteredOwnedBytes);
            CHECK(storage.coefficientGhostBytes == 2 * unfilteredGhostBytes);
        }
        const auto publication = fv.AcquireFinalizedReconstruction(*reconstruction);
        CHECK(publication.UsesDistinctStorage() == filtered);
        const auto &warm = fv.ReconstructionGuess(*reconstruction);
        const auto &selected = publication.Values();
        REQUIRE(warm.PhysicalFieldNumber() == selected.PhysicalFieldNumber());
        for (DNDS::index iField = 0; iField < warm.PhysicalFieldNumber(); ++iField)
        {
            CHECK((warm.PhysicalField(iField).Owned().SharedStorage() ==
                   selected.PhysicalField(iField).Owned().SharedStorage()) == !filtered);
            CHECK((warm.PhysicalField(iField).Ghost().SharedStorage() ==
                   selected.PhysicalField(iField).Ghost().SharedStorage()) == !filtered);
        }
        auto replacement = C::EquationDofOperations::NewVector(warm);
        C::EquationDofOperations::Fill(replacement, .125);
        fv.RestoreReconstructionGuess(*reconstruction, replacement);
        CHECK_THROWS_WITH_AS(publication.Validate(), doctest::Contains("changed"), std::exception);
    }
}
TEST_CASE("One finite-volume scratch workspace serves independent durable reconstruction states")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 1;
    settings.numerics.reconstructionIterations = 0;
    settings.numerics.limiter = "none";
    settings.numerics.positivity = false;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto fieldA = fv.Metrics().BuildField(5), fieldB = fv.Metrics().BuildField(5);
    auto residualA = fv.Metrics().BuildField(5), residualB = fv.Metrics().BuildField(5);
    for (auto *field : {&fieldA, &fieldB})
    {
        auto write = field->Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const auto point = fv.Metrics().GetCellCent(iCell);
            const auto value = equation.Gas().Conservative({1 + .03 * point[0], .2, -.1, .05, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    auto stateA = Bind(fv, fieldA), stateB = Bind(fv, fieldB);
    auto outputA = Bind(fv, residualA), outputB = Bind(fv, residualB);
    auto reconstructionA = fv.BuildReconstructionState(stateA);
    auto reconstructionB = fv.BuildReconstructionState(stateB);
    auto workspace = fv.BuildWorkspace(stateA);
    auto seedA = C::EquationDofOperations::NewVector(fv.ReconstructionGuess(*reconstructionA));
    auto seedB = C::EquationDofOperations::NewVector(fv.ReconstructionGuess(*reconstructionB));
    C::EquationDofOperations::Fill(seedA, .01);
    C::EquationDofOperations::Fill(seedB, .02);
    fv.RestoreReconstructionGuess(*reconstructionA, seedA);
    fv.RestoreReconstructionGuess(*reconstructionB, seedB);
    const C::EquationDofOperations::Binding retainedA(fv.ReconstructionGuess(*reconstructionA));
    const C::EquationDofOperations::Binding retainedB(fv.ReconstructionGuess(*reconstructionB));
    R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(stateA, outputA, *reconstructionA, *workspace, engine, stream);
    const C::EquationDofOperations::Binding finalizedA(
        fv.FluxReconstruction(*reconstructionA, *workspace));
    CHECK_NOTHROW(retainedA.Check(fv.ReconstructionGuess(*reconstructionA), true));
    fv.Evaluate(stateB, outputB, *reconstructionB, *workspace, engine, stream);
    CHECK_NOTHROW(retainedA.Check(fv.ReconstructionGuess(*reconstructionA), true));
    CHECK_NOTHROW(retainedB.Check(fv.ReconstructionGuess(*reconstructionB), true));
    CHECK_NOTHROW(finalizedA.Check(fv.FluxReconstruction(*reconstructionA, *workspace), true));
    A::Axpy(residualB, -1., residualA);
    CHECK(A::NormGlobal(residualB) > 1e-8);
    // This is the DITR new-step contract: midpoint gets both the accepted
    // warm guess and finalized polynomial; the shared scratch stays untouched.
    fv.CopyReconstructionState(*reconstructionB, *reconstructionA);
    const auto equal = [](const C::EquationDofs &a, const C::EquationDofs &b)
    {
        const auto ar = a.ReadOwned(), br = b.ReadOwned();
        const auto al = ar.ComponentLayout(), bl = br.ComponentLayout();
        const auto av = al.View(), bv = bl.View();
        for (DNDS::index iCell = 0; iCell < av.RowCount(); ++iCell)
            for (DNDS::index j = 0; j < av.ComponentNumber(); ++j)
                CHECK(av(iCell, j) == bv(iCell, j));
    };
    equal(fv.ReconstructionGuess(*reconstructionA),
          fv.ReconstructionGuess(*reconstructionB));
    equal(fv.FluxReconstruction(*reconstructionA, *workspace),
          fv.FluxReconstruction(*reconstructionB, *workspace));
}
TEST_CASE("Default Cauchy CGNS loader propagates the configured preflight policy")
{
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    const auto path = (root / "data/mesh/IV10_2.cgns").string();
    std::vector<double> reference;
    for (bool checks : {true, false})
    {
        C::DomainSettings settings;
        settings.mesh.file = path;
        settings.mesh.dimension = 2;
        settings.mesh.periodicLengths = {10, 10, 0};
        settings.mesh.boundaryIds = {{"PERIODIC_1", -1}, {"PERIODIC_1_DONOR", -4}, {"PERIODIC_2", -2}, {"PERIODIC_2_DONOR", -5}};
        settings.implicit.collectiveChecks = checks;
        const auto config = C::SimulationConfig::ResolveCollective({{"defaults", C::Json(settings)}}, mpi.comm);
        C::Simulation simulation(config, mpi.comm);
        auto &domain = simulation.GetDomain(0);
        CHECK(domain.Discretization().Metrics().Communication()->ChecksCollectivePreflight() == checks);
        for (auto kind : {DNDS::Geom::EntityKind::Cell, DNDS::Geom::EntityKind::Node, DNDS::Geom::EntityKind::Face, DNDS::Geom::EntityKind::Bnd})
            CHECK(domain.Discretization().Mesh().Entities(kind).ChecksCollectivePreflight() == checks);
        REQUIRE(simulation.Advance(1e-4).accepted);
        auto read = domain.Dofs().CoupledField().Owned().Read();
        for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
            {
                const auto value = read.View()(iCell, j);
                CHECK(std::isfinite(value));
                if (checks)
                    reference.push_back(value);
                else
                    CHECK(value == reference[size_t(iCell * 5 + j)]);
            }
    }
}
TEST_CASE("Cauchy collective switch reaches prepared halos without changing the P3 RHS")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 3);
    std::vector<double> reference;
    int checkedReductions = 0;
    for (const bool checks : {true, false})
    {
        C::DomainSettings settings;
        settings.implicit.collectiveChecks = checks;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 3;
        settings.numerics.limiter = "WBAP";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        CHECK(fv.Metrics().Communication()->ChecksCollectivePreflight() == checks);
        auto state = fv.Metrics().BuildField(5), result = fv.Metrics().BuildField(5);
        {
            auto write = state.Owned().Write();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto center = fv.Metrics().GetCellCent(iCell);
                const auto values = equation.Gas().Conservative({1 + .05 * std::sin(6.283185307179586 * center[0]), .3, -.1, .2, 1});
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = values[j];
            }
        }
        struct CountingScope
        {
            CountingScope() { countReductions = true; }
            ~CountingScope() { countReductions = false; }
        };
        if (!checks)
        {
            const auto countBinding = [&](const auto &bind)
            {
                reductions = 0;
                CountingScope count;
                bind();
                return reductions.load();
            };
            const auto ordinary = countBinding([&]
                                               { (void)fv.Metrics().BindHalo(state); });
            const auto tracked = countBinding([&]
                                              { (void)fv.Metrics().BindTrackedHalo(state); });
            CHECK(tracked == ordinary); // Same required protocols; no safety-only reduction.
        }
        auto logical = Bind(fv, state), output = Bind(fv, result);
        auto reconstruction = fv.BuildReconstructionState(logical);
        auto workspace = fv.BuildWorkspace(logical);
        R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        reductions = 0;
        {
            CountingScope count;
            fv.Evaluate(logical, output, *reconstruction, *workspace, engine, stream);
        }
        if (checks)
            checkedReductions = reductions;
        else
        {
            CHECK(reductions.load() < checkedReductions);
            // The Cauchy master switch disables every safety-only collective.
            // Numerical reductions belong to the nonlinear/time controller,
            // not to this standalone finite-volume evaluation.
            CHECK(reductions.load() == 0);
        }
        const auto read = result.Owned().Read();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
            {
                const auto value = read.View()(iCell, j);
                CHECK(std::isfinite(value));
                if (checks)
                    reference.push_back(value);
                else
                    CHECK(value == reference[size_t(iCell * 5 + j)]);
            }
    }
}
TEST_CASE("Semantic finite-volume dispatch preserves the trusted flow kernel and rejects an incompatible model before communication")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    for (int layout = 0; layout < 3; ++layout)
    {
        C::DomainSettings settings;
        settings.execution.hostThreads = 2;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 3;
        settings.numerics.limiter = "WBAP";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        const auto shape = layout == 0 ? R::RowLayout::Dense(0, 5) : layout == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                 : R::RowLayout::TiledColMajor(0, 5, 32);
        auto raw = fv.Metrics().BuildField(shape);
        {
            auto write = raw.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const auto center = fv.Metrics().GetCellCent(iCell);
                const auto state = equation.Gas().Conservative({1 + .05 * std::sin(6.283185307179586 * center[0]), .3, -.1, .2, 1});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = state[j];
            }
        }
        auto semantic = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(shape), "packed");
        A::Copy(semantic.CoupledField(), raw);
        auto rawResult = fv.Metrics().BuildField(shape), semanticResult = fv.Metrics().BuildField(shape);
        auto generic = Bind(fv, raw), genericResult = Bind(fv, rawResult), semanticOutput = Bind(fv, semanticResult);
        auto genericReconstruction = fv.BuildReconstructionState(generic);
        auto semanticReconstruction = fv.BuildReconstructionState(semantic);
        auto rawWork = fv.BuildWorkspace(
            generic, std::nullopt,
            C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic});
        auto semanticWork = fv.BuildWorkspace(semantic);
        R::EagerEngine engine(2);
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(generic, genericResult, *genericReconstruction, *rawWork, engine, stream);
        fv.Evaluate(semantic, semanticOutput, *semanticReconstruction, *semanticWork, engine, stream);
        {
            const auto a = rawResult.Owned().Read(), b = semanticResult.Owned().Read();
            const auto x = a.View<R::UniformTableRows>(), y = b.View<R::UniformTableRows>();
            double maximumDifference = 0, maximumMagnitude = 0;
            for (DNDS::index iCell = 0; iCell < x.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    maximumDifference = std::max(maximumDifference, std::abs(x(iCell, j) - y(iCell, j)));
                    maximumMagnitude = std::max({maximumMagnitude, std::abs(x(iCell, j)), std::abs(y(iCell, j))});
                }
            // The generic equation and fixed-flow leaves retain intentionally
            // different point-accumulation order. Plan 59 therefore requires
            // tight numerical, not bitwise, agreement for this cross-oracle.
            CHECK(maximumDifference < 1e-12 * std::max(1., maximumMagnitude));
        }
        // One rank advertises pressure in place of energy while keeping the
        // same width and allocation. All ranks must reject before posting MPI.
        auto wrongSchema = equation.Schema();
        if (!mpi.rank)
            wrongSchema.blocks[0].components[4].name = "rhop";
        auto wrong = C::EquationDofs::Coupled(wrongSchema, semantic.CoupledField(), "packed");
        const auto resultVersion = semanticResult.Owned().Version();
        const auto stateVersion = semantic.CoupledField().Owned().Version();
        CHECK_THROWS(fv.BuildWorkspace(wrong));
        CHECK_THROWS(fv.Evaluate(wrong, semanticOutput, *semanticReconstruction, *semanticWork, engine, stream));
        CHECK(semanticResult.Owned().Version() == resultVersion);
        CHECK(semantic.CoupledField().Owned().Version() == stateVersion);
    }
}
TEST_CASE("Euler reconstruction PP selects a mean or progressively truncated reference across layouts during halo exchange")
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
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (bool compressToMean : {true, false})
        for (int quadratureOrder : {3, 5})
            for (int layout = 0; layout < 3; ++layout)
            {
                C::DomainSettings settings;
                settings.numerics.positivity = true;
                settings.numerics.positivityCompressToMean = compressToMean;
                settings.numerics.positivityBetaCutoff = 0;
                settings.numerics.reconstruction = "variational";
                settings.numerics.cfv.maxOrder = 3;
                settings.numerics.cfv.intOrder = quadratureOrder;
                settings.numerics.cfv.cacheDiffBase = false;
                C::FiniteVolume fv(mesh, settings, C::EquationSystem::Resolve(settings.equation));
                // No source is active: PP still uses the same volume rule.
                // Quad degree3 uses 2x2 points, degree5 uses 3x3, with no
                // independent maximum-order PP bank hidden alongside them.
                bool foundVolume = false;
                for (const auto &record : R::AllocationDiagnostics::Snapshot().allocations)
                    if (std::find(record.names.begin(), record.names.end(), "Cauchy/quadrature/volume/points") != record.names.end())
                    {
                        foundVolume = true;
                        // Cell-major offsets now imply cell identity: no PP
                        // per-point cell/scatter map remains. Inspect the
                        // actual shared coordinates to preserve the rule test.
                        CHECK(record.bytes == size_t(fv.Metrics().NumCell() * (quadratureOrder == 3 ? 4 : 9) * 3) * sizeof(DNDS::real));
                    }
                CHECK(foundVolume);
                const auto shape = layout == 0 ? R::RowLayout::Dense(0, 5) : layout == 1 ? R::RowLayout::ColMajor(0, 5)
                                                                                         : R::RowLayout::TiledColMajor(0, 5, 4);
                auto state = fv.Metrics().BuildField(shape);
                const C::FlowState mean{1, 0, 0, 0, 2.5};
                {
                    auto write = state.Owned().Write();
                    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                        for (int j = 0; j < 5; ++j)
                            write.View<R::UniformTableRows>()(iCell, j) = mean[j];
                }
                auto logical = Bind(fv, state);
                auto reconstruction = fv.BuildReconstructionState(logical);
                auto work = fv.BuildWorkspace(logical);
                const auto &rec = fv.ReconstructionGuess(*reconstruction);
                const DNDS::index modes = rec.CoupledField().Width() / 5;
                REQUIRE(modes == 9);
                auto input = rec.Clone();
                auto &inputArray = input.CoupledField().Owned();
                const auto fill = [&](DNDS::real density, DNDS::real cubicEnergy)
                {
                    auto write = inputArray.Write();
                    for (DNDS::index iCell = 0; iCell < inputArray.RowCount(); ++iCell)
                    {
                        for (DNDS::index j = 0; j < rec.CoupledField().Width(); ++j)
                            write.View()(iCell, j) = 0;
                        write.View()(iCell, 0) = density;
                        write.View()(iCell, 4 * modes + 5) = cubicEnergy;
                    }
                };
                // A good polynomial must remain exactly unchanged, even while
                // the state ghost buffer is held by a real MPI transfer.
                fill(1e-3, -1e-3);
                fv.RestoreReconstructionGuess(*reconstruction, input);
                auto halo = fv.Metrics().BindHalo(state);
                auto operation = halo.Begin();
                const auto meanVersion = state.Owned().Version();
                fv.LimitReconstructionPP(logical, *reconstruction, *work);
                operation.Wait();
                CHECK(state.Owned().Version() == meanVersion);
                CHECK(fv.PositivityReconstructionStats(*work) == std::pair<DNDS::index, DNDS::real>{0, 1});
                {
                    auto expected = inputArray.Read(), actual = fv.FluxReconstruction(*reconstruction, *work).CoupledField().Owned().Read();
                    for (DNDS::index iCell = 0; iCell < inputArray.RowCount(); ++iCell)
                        for (DNDS::index j = 0; j < rec.CoupledField().Width(); ++j)
                            CHECK(actual.View<R::UniformTableRows>()(iCell, j) == expected.View()(iCell, j));
                }
                // P3 and P1 are both inadmissible. Progressive reduction must
                // discard P3 before scaling P1 toward the mean; mean-reference
                // compression must retain a nonzero, uniformly scaled P3 mode.
                fill(4, -1000);
                fv.RestoreReconstructionGuess(*reconstruction, input);
                auto active = halo.Begin();
                fv.LimitReconstructionPP(logical, *reconstruction, *work);
                active.Wait();
                const auto [limited, beta] = fv.PositivityReconstructionStats(*work);
                CHECK(limited == 16);
                CHECK(beta > 0);
                CHECK(beta < 1);
                CHECK(state.Owned().Version() == meanVersion);
                {
                    auto coefficients = fv.FluxReconstruction(*reconstruction, *work).CoupledField().Owned().Read(), means = state.Owned().Read();
                    for (DNDS::index iCell = 0; iCell < inputArray.RowCount(); ++iCell)
                    {
                        CHECK(coefficients.View<R::UniformTableRows>()(iCell, 0) > 0);
                        CHECK(coefficients.View<R::UniformTableRows>()(iCell, 0) < 4);
                        const auto cubic = coefficients.View<R::UniformTableRows>()(iCell, 4 * modes + 5);
                        if (compressToMean)
                            CHECK(cubic / -1000 == doctest::Approx(coefficients.View<R::UniformTableRows>()(iCell, 0) / 4));
                        else
                            CHECK(cubic == 0);
                        for (int j = 0; j < 5; ++j)
                            CHECK(means.View<R::UniformTableRows>()(iCell, j) == mean[j]);
                    }
                }
                // A second standalone projection starts from the same unlimited
                // guess, even after the previous flux polynomial was compressed.
                // Raising the means makes that original polynomial admissible.
                {
                    auto write = state.Owned().Write();
                    const auto values = write.View<R::UniformTableRows>();
                    for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
                    {
                        values(iCell, 0) = 1000;
                        values(iCell, 4) = 1e9;
                    }
                }
                fv.LimitReconstructionPP(logical, *reconstruction, *work);
                CHECK(fv.PositivityReconstructionStats(*work) == std::pair<DNDS::index, DNDS::real>{0, 1});
                auto expected = inputArray.Read(), actual = fv.FluxReconstruction(*reconstruction, *work).CoupledField().Owned().Read();
                for (DNDS::index iCell = 0; iCell < inputArray.RowCount(); ++iCell)
                    for (DNDS::index j = 0; j < rec.CoupledField().Width(); ++j)
                        CHECK(actual.View<R::UniformTableRows>()(iCell, j) == expected.View()(iCell, j));
            }
}
TEST_CASE("WBAP and PP flux filtering do not overwrite the unlimited VR warm guess")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 3; ++scenario)
        for (bool pp : {false, true})
        {
            C::DomainSettings settings;
            settings.execution.overlap = scenario != 1;
            settings.numerics.reconstruction = "variational";
            settings.numerics.cfv.maxOrder = 3;
            settings.numerics.reconstructionIterations = 1;
            const auto equation = C::EquationSystem::Resolve(settings.equation);
            C::FiniteVolume reference(mesh, settings, equation);
            settings.numerics.limiter = pp ? "none" : "WBAP";
            settings.numerics.positivity = pp;
            C::FiniteVolume filtered(mesh, settings, equation);
            const auto layout = scenario == 0 ? R::RowLayout::Dense(0, 5) : scenario == 1 ? R::RowLayout::ColMajor(0, 5)
                                                                                          : R::RowLayout::TiledColMajor(0, 5, 4);
            auto mean = reference.Metrics().BuildField(layout), filteredMean = filtered.Metrics().BuildField(layout);
            for (auto *field : {&mean, &filteredMean})
            {
                auto write = field->Owned().Write();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                {
                    const auto point = reference.Metrics().GetCellCent(iCell);
                    const auto value = equation.Gas().Conservative({1 + .05 * std::sin(6.283185307179586 * point[0]), .3, .1, 0, 1});
                    for (int j = 0; j < 5; ++j)
                        write.View()(iCell, j) = value[j];
                }
            }
            auto referenceState = Bind(reference, mean), filteredState = Bind(filtered, filteredMean);
            auto referenceReconstruction = reference.BuildReconstructionState(referenceState);
            auto filteredReconstruction = filtered.BuildReconstructionState(filteredState);
            auto plain = reference.BuildWorkspace(referenceState), limited = filtered.BuildWorkspace(filteredState);
            auto referenceSeed = reference.ReconstructionGuess(*referenceReconstruction).Clone();
            auto filteredSeed = filtered.ReconstructionGuess(*filteredReconstruction).Clone();
            auto &seedArray = referenceSeed.CoupledField().Owned();
            {
                auto referenceWrite = seedArray.Write();
                auto filteredWrite = filteredSeed.CoupledField().Owned().Write();
                for (DNDS::index iCell = 0; iCell < seedArray.RowCount(); ++iCell)
                {
                    for (int j = 0; j < 45; ++j)
                    {
                        referenceWrite.View()(iCell, j) = 0;
                        filteredWrite.View()(iCell, j) = 0;
                    }
                    const auto id = mesh->OriginalEntityId(DNDS::Geom::EntityKind::Cell, iCell);
                    referenceWrite.View()(iCell, 0) = .1 * std::cos(id * .7);
                    filteredWrite.View()(iCell, 0) = .1 * std::cos(id * .7);
                    if (pp)
                    {
                        referenceWrite.View()(iCell, 4 * 9 + 5) = -1000 * std::cos(id * .3);
                        filteredWrite.View()(iCell, 4 * 9 + 5) = -1000 * std::cos(id * .3);
                    }
                }
            }
            reference.RestoreReconstructionGuess(*referenceReconstruction, referenceSeed);
            filtered.RestoreReconstructionGuess(*filteredReconstruction, filteredSeed);
            auto a = reference.Metrics().BuildField(5), b = filtered.Metrics().BuildField(5);
            auto outputA = Bind(reference, a), outputB = Bind(filtered, b);
            R::EagerEngine engine(scenario == 2 ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
            const auto stream = engine.CreateCPUStream();
            const auto meanVersion = filteredMean.Owned().Version();
            for (int evaluation = 0; evaluation < 2; ++evaluation)
            {
                reference.Evaluate(referenceState, outputA, *referenceReconstruction, *plain, engine, stream);
                filtered.Evaluate(filteredState, outputB, *filteredReconstruction, *limited, engine, stream);
                const auto expected = reference.ReconstructionGuess(*referenceReconstruction).CoupledField().Owned().Read();
                const auto actual = filtered.ReconstructionGuess(*filteredReconstruction).CoupledField().Owned().Read();
                const auto flux = filtered.FluxReconstruction(*filteredReconstruction, *limited).CoupledField().Owned().Read();
                double difference = 0, residualDifference = 0;
                for (DNDS::index iCell = 0; iCell < seedArray.RowCount(); ++iCell)
                    for (int j = 0; j < 45; ++j)
                    {
                        const DNDS::real x = actual.View<R::UniformTableRows>()(iCell, j), y = expected.View<R::UniformTableRows>()(iCell, j);
                        CHECK(std::memcmp(&x, &y, sizeof(x)) == 0);
                        difference += std::abs(flux.View<R::UniformTableRows>()(iCell, j) - x);
                    }
                auto first = a.Owned().Read(), second = b.Owned().Read();
                for (DNDS::index iCell = 0; iCell < seedArray.RowCount(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        residualDifference += std::abs(first.View<R::UniformTableRows>()(iCell, j) - second.View<R::UniformTableRows>()(iCell, j));
                MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
                MPI_Allreduce(MPI_IN_PLACE, &residualDifference, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
                if (evaluation == 0)
                {
                    CHECK(difference > 1e-8);         // Prove filtering is active, not a zero fixture.
                    CHECK(residualDifference > 1e-8); // RHS must consume the filtered field.
                }
                CHECK(filteredMean.Owned().Version() == meanVersion);
                if (pp && evaluation == 0)
                    CHECK(filtered.PositivityReconstructionStats(*limited).first > 0);
            }
            filtered.ResetReconstruction(*filteredReconstruction, *limited);
        }
}
TEST_CASE("Auxiliary provenance and persistent workspaces reject a different row view of the same allocation")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
    const auto layout = R::RowLayout::Dense(fv.Metrics().NumCell(), 5);
    const size_t bytes = size_t(fv.Metrics().NumCell()) * 5 * sizeof(double);
    const auto backing = R::Buffer::Allocate(2 * bytes);
    auto first = R::Array<double>::FromStorage(layout, backing.Subspan(0, bytes));
    auto second = R::Array<double>::FromStorage(layout, backing.Subspan(bytes, bytes));
    {
        auto a = first.Write();
        const auto value = equation.Gas().Conservative({1, .3, -.1, .2, 1});
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                a.View()(iCell, j) = value[j];
    }
    {
        auto b = second.Write();
        const auto value = equation.Gas().Conservative({2, .3, -.1, .2, 1.2});
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                b.View()(iCell, j) = value[j];
    }
    REQUIRE(first.Identity() == second.Identity());
    REQUIRE(first.Version() == second.Version());
    REQUIRE(first.ShapeVersion() == second.ShapeVersion());
    REQUIRE(first.SharedStorage() != second.SharedStorage());
    state.Owned() = first;
    fv.Metrics().BindHalo(state).Begin().Wait();
    auto logical = Bind(fv, state), output = Bind(fv, residual);
    C::AuxiliaryRequest request;
    request.meanThermodynamics = true;
    auto values = fv.BuildAuxiliaryValues(request);
    fv.PrepareAuxValues(logical, values);
    auto reconstruction = fv.BuildReconstructionState(logical);
    auto workspace = fv.BuildWorkspace(logical);
    state.Owned() = second;
    auto other = Bind(fv, state);
    CHECK_THROWS(fv.ValidateAuxValues(other, values));
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    CHECK_THROWS(fv.Evaluate(other, output, *reconstruction, *workspace, engine, stream));
    fv.PrepareAuxValues(other, values);
    CHECK_NOTHROW(fv.ValidateAuxValues(other, values));
    const auto auxLayout = values.meanThermodynamics->Layout();
    const size_t auxBytes = values.meanThermodynamics->Storage().Bytes();
    const auto auxBacking = R::Buffer::Allocate(2 * auxBytes);
    values.meanThermodynamics = R::Array<double>::FromStorage(auxLayout, auxBacking.Subspan(0, auxBytes));
    fv.PrepareAuxValues(other, values);
    values.meanThermodynamics = R::Array<double>::FromStorage(auxLayout, auxBacking.Subspan(auxBytes, auxBytes));
    CHECK_THROWS(fv.ValidateAuxValues(other, values));
}
TEST_CASE("Cauchy communication worker preserves nonconstant P3 RHS")
{
    int provided = 0;
    MPI_Query_thread(&provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        MESSAGE("communication worker coverage requires DNDS_TEST_COMMUNICATION_WORKER=1 and MPI_THREAD_MULTIPLE");
        return; // Runtime capability rejection is tested in the transfer suite.
    }
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 3);
    std::vector<double> reference;
    for (bool worker : {false, true})
    {
        C::DomainSettings settings;
        settings.execution.communicationExecution = worker ? "worker" : "caller";
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 3;
        settings.numerics.limiter = "WBAP";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = fv.Metrics().BuildField(5), result = fv.Metrics().BuildField(5);
        {
            auto write = state.Owned().Write();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto center = fv.Metrics().GetCellCent(iCell);
                const auto value = equation.Gas().Conservative({1 + .05 * std::sin(6.283185307179586 * center[0]), .3, -.1, .2, 1});
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = value[j];
            }
        }
        auto logical = Bind(fv, state), output = Bind(fv, result);
        auto reconstruction = fv.BuildReconstructionState(logical);
        auto workspace = fv.BuildWorkspace(logical);
        R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(logical, output, *reconstruction, *workspace, engine, stream);
        engine.Synchronize();
        const auto read = result.Owned().Read();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                if (worker)
                    CHECK(read.View()(iCell, j) == reference[size_t(iCell * 5 + j)]);
                else
                    reference.push_back(read.View()(iCell, j));
    }
}

TEST_CASE("Cauchy P0 explicit P1 and degree-one VR retain their distinct storage across layouts and workers")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    for (const std::string method : {"constant", "gradient", "variational"})
        for (int layout = 0; layout < 3; ++layout)
        {
            CAPTURE(method);
            CAPTURE(layout);
            C::DomainSettings settings;
            settings.mesh.dimension = 3;
            settings.boundaries["7"].kind = "farfield";
            settings.boundaries["7"].primitive = {1.2, .3, -.2, .4, 1.1};
            settings.numerics.reconstruction = method;
            settings.numerics.cfv.maxOrder = 1;
            settings.numerics.cfv.boundaryDerivativeWeights = {1, 0, 0, 0};
            const auto equation = C::EquationSystem::Resolve(settings.equation);
            C::FiniteVolume fv(mesh, settings, equation);
            const auto polynomial = fv.PolynomialReconstruction();
            CHECK(bool(polynomial) == (method == "variational"));
            if (polynomial)
            {
                CHECK(polynomial->Capabilities().sweeps == (method == "variational"));
                CHECK_FALSE(polynomial->Capabilities().linearSystem);
                CHECK(polynomial->NRecDOF() > 0);
            }
            const auto physical = layout == 0 ? R::RowLayout::Dense(0, 5, 8) : layout == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                           : R::RowLayout::TiledColMajor(0, 5, 32);
            auto state = fv.Metrics().BuildField(physical), residual = fv.Metrics().BuildField(physical);
            const auto exact = equation.Gas().Conservative(settings.boundaries["7"].primitive);
            {
                auto write = state.Owned().Write();
                for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        write.View<R::UniformTableRows>()(iCell, j) = exact[j];
            }
            auto logical = Bind(fv, state), output = Bind(fv, residual);
            auto reconstruction = fv.BuildReconstructionState(logical);
            const auto storage = fv.InspectReconstructionStorage(*reconstruction);
            CHECK(storage.polynomialPlan == (method == "variational"));
            if (method == "variational")
            {
                CHECK(storage.coefficientFields > 0);
                CHECK(storage.coefficientTransferBindings > 0);
                CHECK(storage.coefficientOwnedBytes > 0);
            }
            else
            {
                CHECK(storage.coefficientFields == 0);
                CHECK(storage.coefficientTransferBindings == 0);
                CHECK(storage.coefficientOwnedBytes == 0);
                CHECK(storage.coefficientGhostBytes == 0);
                CHECK_FALSE(storage.distinctFinalized);
            }
            auto workspace = fv.BuildWorkspace(logical);
            for (int workers : {0, 2})
            {
                CAPTURE(workers);
                R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
                const auto stream = engine.CreateCPUStream();
                fv.Evaluate(logical, output, *reconstruction, *workspace, engine, stream);
                engine.Synchronize();
                CHECK(A::NormGlobal(residual) < 2e-14);
            }
        }
}

TEST_CASE("Cauchy direct P1 owns Barth and thermal PP without variational storage")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    const auto run = [&](const std::string &limiter, bool positivity)
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "gradient";
        settings.numerics.gradient = "leastSquares";
        settings.numerics.limiter = limiter;
        settings.numerics.positivity = positivity;
        settings.numerics.positivityBetaCutoff = 0;
        settings.numerics.faceFallback = false;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = fv.Metrics().BuildField(
            R::RowLayout::ColMajor(0, 5, {8, 1}));
        {
            auto write = state.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const auto point = fv.Metrics().GetCellCent(iCell);
                const DNDS::real velocity = point[0] < .5 ? -10 : 10;
                const auto value = equation.Gas().Conservative(
                    {1, velocity, 0, 0, .01});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = value[j];
            }
        }
        auto logical = Bind(fv, state);
        auto reconstruction = fv.BuildReconstructionState(logical);
        const auto storage = fv.InspectReconstructionStorage(*reconstruction);
        CHECK_FALSE(storage.polynomialPlan);
        CHECK(storage.coefficientFields == 0);
        CHECK(storage.coefficientTransferBindings == 0);
        CHECK(storage.coefficientOwnedBytes == 0);
        CHECK(storage.coefficientGhostBytes == 0);
        auto workspace = fv.BuildWorkspace(logical);
        R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        CHECK_NOTHROW(fv.PrepareReconstruction(
            logical, *reconstruction, *workspace, engine, stream));
        return fv.PositivityReconstructionStats(*workspace);
    };

    const auto ppOnly = run("none", true);
    CHECK(ppOnly.first > 0);
    CHECK(ppOnly.second > 0);
    CHECK(ppOnly.second < 1);

    // Barth is a Cauchy equation policy applied before thermal PP. For this
    // momentum jump it bounds the P1 traces by adjacent conservative states,
    // so the subsequent thermodynamic projection is inactive.
    const auto barthAndPP = run("Barth", true);
    CHECK(barthAndPP.first == 0);
    CHECK(barthAndPP.second == 1);
    CHECK(run("Barth", false) ==
          std::pair<DNDS::index, DNDS::real>{0, 1});
}

TEST_CASE("Native viscous Cauchy residual reproduces linear Couette heat production")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 7);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .02;
    settings.boundaries["7"].kind = "extrapolate";
    settings.numerics.reconstruction = "gradient";
    settings.numerics.cfv.maxOrder = 1;
    settings.numerics.cfv.boundaryDerivativeWeights = {1, 0, 0, 0};
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
    const double shear = .3;
    {
        auto write = state.Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const auto center = fv.Metrics().GetCellCent(iCell);
            auto value = equation.Gas().Conservative({1, shear * center[1], 0, 0, 1});
            // Conservative cell average of u^2 includes the exact y variance.
            value[4] += .5 * shear * shear / (12 * 49);
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    auto logical = Bind(fv, state), output = Bind(fv, residual);
    auto reconstruction = fv.BuildReconstructionState(logical);
    auto work = fv.BuildWorkspace(logical);
    auto inviscidSettings = settings;
    inviscidSettings.equation.transport = "inviscid";
    inviscidSettings.equation.viscosity = 0;
    C::FiniteVolume inviscid(mesh, inviscidSettings, C::EquationSystem::Resolve(inviscidSettings.equation));
    auto inviscidState = inviscid.Metrics().BuildField(5), inviscidResidual = inviscid.Metrics().BuildField(5);
    {
        auto source = state.Owned().Read();
        auto target = inviscidState.Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                target.View()(iCell, j) = source.View()(iCell, j);
    }
    auto inviscidLogical = Bind(inviscid, inviscidState), inviscidOutput = Bind(inviscid, inviscidResidual);
    auto inviscidReconstruction = inviscid.BuildReconstructionState(inviscidLogical);
    auto inviscidWork = inviscid.BuildWorkspace(inviscidLogical);
    R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(logical, output, *reconstruction, *work, engine, stream);
    inviscid.Evaluate(inviscidLogical, inviscidOutput, *inviscidReconstruction, *inviscidWork, engine, stream);
    // Direct P1 owns no compatibility modal payload or coefficient transfer.
    const auto storage = fv.InspectReconstructionStorage(*reconstruction);
    CHECK_FALSE(storage.polynomialPlan);
    CHECK(storage.coefficientFields == 0);
    CHECK(storage.coefficientTransferBindings == 0);
    CHECK(storage.coefficientOwnedBytes == 0);
    CHECK(storage.coefficientGhostBytes == 0);
    const auto read = residual.Owned().Read();
    const auto base = inviscidResidual.Owned().Read();
    DNDS::index tested = 0;
    // Interior cells have an independent exact gradient/stress oracle; exclude
    // two-cell boundary closure whose extrapolation does not prescribe shear.
    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
    {
        const auto point = fv.Metrics().GetCellCent(iCell);
        if ((point.array() > .29).all() && (point.array() < .71).all())
        {
            ++tested;
            const double volume = fv.Metrics().GetCellVol(iCell);
            CHECK(std::abs((read.View()(iCell, 1) - base.View()(iCell, 1)) / volume) < 1e-12);
            CHECK((read.View()(iCell, 4) - base.View()(iCell, 4)) / volume == doctest::Approx(.02 * shear * shear).epsilon(2e-8));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &tested, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
    CHECK(tested == 27);
}
TEST_CASE("Cauchy SSPRK3 preserves periodic constants and rolls back rejected trials")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    settings.initialPrimitive = {1.1, .3, -.2, .4, 1.2};
    settings.execution.hostThreads = 2;
    C::Domain domain("periodic", mesh, settings);
    const auto exact = domain.Equation().Gas().Conservative(settings.initialPrimitive);
    for (int i = 0; i < 3; ++i)
        REQUIRE(domain.Advance(.001).accepted);
    CHECK(domain.Time() == doctest::Approx(.003));
    CHECK(domain.Epoch().acceptedStep == 3);
    CHECK(domain.LastFaceFallbacks() == 0);
    {
        const auto read = domain.Dofs().CoupledField().Owned().Read();
        for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(read.View()(iCell, j) == doctest::Approx(exact[j]).epsilon(2e-13));
    }
    REQUIRE(domain.PrepareStep(.001).accepted);
    std::optional<R::Array<double>::ReadLease> held;
    if (mpi.rank == 0)
        held.emplace(domain.Dofs().CoupledField().Owned().Read());
    CHECK_THROWS(domain.CommitStep());
    CHECK(domain.Time() == doctest::Approx(.003));
    held.reset();
    domain.CommitStep();
    CHECK(domain.Epoch().acceptedStep == 4);
    CHECK_THROWS(domain.Advance(-1));
    CHECK(domain.Time() == doctest::Approx(.004));
}
TEST_CASE("Cauchy periodic density transport has independently measured second-order spatial residual")
{
    double previous = 0;
    constexpr double pi = 3.14159265358979323846;
    for (DNDS::index width : {8, 16, 32})
    {
        C::DomainSettings settings;
        settings.numerics.cfv.maxOrder = 1;
        settings.execution.hostThreads = 2;
        C::Domain domain("transport", CauchyTest::PeriodicQuads(mpi, width), settings);
        const auto &fv = domain.Discretization().Metrics();
        const double averageFactor = std::sin(pi / width) / (pi / width);
        {
            auto write = domain.Dofs().CoupledField().Owned().Write();
            for (DNDS::index iCell = 0; iCell < fv.NumCell(); ++iCell)
            {
                const auto x = fv.GetCellCent(iCell);
                const double rho = 1 + .1 * averageFactor * averageFactor * std::sin(2 * pi * (x[0] + x[1]));
                const auto value = domain.Equation().Gas().Conservative({rho, .4, .2, .3, 1});
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = value[j];
            }
        }
        const auto &rhs = domain.Residual().CoupledField();
        auto read = rhs.Owned().Read();
        double error = 0, total = 0;
        for (DNDS::index iCell = 0; iCell < fv.NumCell(); ++iCell)
        {
            const auto x = fv.GetCellCent(iCell);
            const double volume = fv.GetCellVol(iCell);
            const double exact = -.1 * averageFactor * averageFactor * 2 * pi * .6 * std::cos(2 * pi * (x[0] + x[1]));
            error += volume * std::pow(read.View()(iCell, 0) / volume - exact, 2);
            total += read.View()(iCell, 0);
        }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
        error = std::sqrt(error);
        CHECK(std::abs(total) < 1e-12);
        CHECK(domain.LastFaceFallbacks() == 0);
        if (previous)
            CHECK(previous / error > 3.4);
        previous = error;
    }
}
TEST_CASE("Cauchy periodic transport SSPRK3 advances a smooth physical state and conserves mass")
{
    constexpr double pi = 3.14159265358979323846;
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    settings.execution.hostThreads = 2;
    constexpr DNDS::index width = 16;
    C::Domain domain("advection", CauchyTest::PeriodicQuads(mpi, width), settings);
    const auto &fv = domain.Discretization().Metrics();
    const double averageFactor = std::sin(pi / width) / (pi / width);
    {
        auto write = domain.Dofs().CoupledField().Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.NumCell(); ++iCell)
        {
            const auto point = fv.GetCellCent(iCell);
            const double rho = 1 + .1 * averageFactor * averageFactor * std::sin(2 * pi * (point[0] + point[1]));
            const auto state = domain.Equation().Gas().Conservative({rho, .4, .2, .3, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = state[j];
        }
    }
    for (int step = 0; step < 10; ++step)
        REQUIRE(domain.Advance(.001).accepted);
    double error = 0, mass = 0;
    const auto read = domain.Dofs().CoupledField().Owned().Read();
    for (DNDS::index iCell = 0; iCell < fv.NumCell(); ++iCell)
    {
        const auto point = fv.GetCellCent(iCell);
        const double volume = fv.GetCellVol(iCell), exact = 1 + .1 * averageFactor * averageFactor * std::sin(2 * pi * (point[0] + point[1] - .6 * domain.Time()));
        error += volume * DNDS::sqr(read.View()(iCell, 0) - exact);
        mass += volume * read.View()(iCell, 0);
    }
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(std::sqrt(error) < 2e-4);
    CHECK(mass == doctest::Approx(1).epsilon(2e-13));
    CHECK(domain.Epoch().acceptedStep == 10);
    CHECK(domain.LastFaceFallbacks() == 0);
}
TEST_CASE("Every Cauchy Riemann selection is integrated into periodic SSPRK3")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (const auto *name : {"rusanov", "Roe", "Roe_M1", "Roe_M2", "Roe_M3", "Roe_M4", "Roe_M5", "Roe_M6", "Roe_M7", "Roe_M8", "Roe_M9",
                             "HLLC", "HLLEP", "HLLEP_V1", "AUSMPWPlus", "AUSMPlusUp", "AUSMPlusUp2"})
    {
        CAPTURE(name);
        C::DomainSettings settings;
        settings.numerics.flux = name;
        settings.numerics.cfv.maxOrder = 1;
        settings.initialPrimitive = {1.2, .4, -.2, .3, 1.1};
        C::Domain domain("flow", mesh, settings);
        REQUIRE(domain.Advance(.001).accepted);
        const auto exact = domain.Equation().Gas().Conservative(settings.initialPrimitive);
        const auto read = domain.Dofs().CoupledField().Owned().Read();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(read.View()(iCell, j) == doctest::Approx(exact[j]).epsilon(3e-13));
    }
}
TEST_CASE("Cauchy variational reconstruction with WBAP advances a nonconstant conservative field")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.limiter = "WBAP";
    settings.numerics.flux = "HLLEP";
    settings.numerics.cfv.maxOrder = 2;
    settings.execution.hostThreads = 2;
    C::Domain domain("limited", mesh, settings);
    {
        auto write = domain.Dofs().CoupledField().Owned().Write();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
        {
            const auto point = domain.Discretization().Metrics().GetCellCent(iCell);
            const auto value = domain.Equation().Gas().Conservative({1 + .05 * std::sin(6.283185307179586 * point[0]), .3, .1, -.2, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    for (int step = 0; step < 3; ++step)
        REQUIRE(domain.Advance(.001).accepted);
    double mass = 0;
    const auto read = domain.Dofs().CoupledField().Owned().Read();
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        C::FlowState value;
        for (int j = 0; j < 5; ++j)
            value[j] = read.View()(iCell, j);
        CHECK(domain.Equation().Gas().Admissible(value));
        mass += value[0] * domain.Discretization().Metrics().GetCellVol(iCell);
    }
    MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(mass == doctest::Approx(1).epsilon(1e-12));
    CHECK(domain.LastFaceFallbacks() == 0);
    CHECK(domain.Epoch().acceptedStep == 3);
}
TEST_CASE("Cauchy physical execution includes ranks with empty owned cells")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 2);
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    C::Domain domain("empty", mesh, settings);
    CHECK(domain.StableStep(.5) > 0);
    REQUIRE(domain.Advance(.001).accepted);
    CHECK(domain.Epoch().acceptedStep == 1);
    CHECK(domain.Time() == .001);
    CHECK(A::NormGlobal(domain.Residual().CoupledField()) < 1e-13);
}
TEST_CASE("Cauchy rejects inadmissible trial collectively and retries without accepted-state corruption")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    C::Domain domain("rejection", mesh, settings);
    {
        auto write = domain.Dofs().CoupledField().Owned().Write();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
        {
            const auto x = domain.Discretization().Metrics().GetCellCent(iCell);
            const auto value = domain.Equation().Gas().Conservative({1, std::sin(6.283185307179586 * x[0]), 0, 0, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    const auto before = domain.Dofs().Fields().Clone();
    CHECK_FALSE(domain.Advance(1e4).accepted);
    CHECK(domain.Time() == 0);
    CHECK(domain.Epoch().acceptedStep == 0);
    {
        auto saved = before.Block("flow").Owned().Read(), actual = domain.Dofs().CoupledField().Owned().Read();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(actual.View()(iCell, j) == saved.View()(iCell, j));
    }
    REQUIRE(domain.Advance(.0001).accepted);
    CHECK(domain.Epoch().acceptedStep == 1);
    const auto &rhs = domain.Residual().CoupledField();
    std::optional<R::Array<double>::ReadLease> held;
    if (mpi.rank == 0)
        held.emplace(rhs.Owned().Read());
    CHECK_THROWS(domain.Residual().CoupledField());
    held.reset();
    CHECK_NOTHROW(domain.Residual().CoupledField());
}
TEST_CASE("Cauchy matching-partition restart preserves layouts continuation and guards mesh identity")
{
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto file = (std::filesystem::temp_directory_path() / ("cauchy-restart-" + std::to_string(pid) + ".h5")).string();
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (const std::string layout : {"aos", "soa", "aosoa32"})
    {
        C::DomainSettings settings;
        settings.numerics.cfv.maxOrder = 1;
        settings.execution.layout = layout;
        settings.initialPrimitive = {1.2, .3, -.2, .1, 1.1};
        C::Domain source("flow", mesh, settings), restored("flow", mesh, settings);
        const C::Json resolved{{"test", layout}};
        // Preserve a non-constant state, so a restart that silently restores
        // only the initial condition cannot pass continuation equivalence.
        {
            auto write = source.Dofs().CoupledField().Owned().Write();
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            {
                auto primitive = settings.initialPrimitive;
                primitive[0] += .01 * std::sin(6.283185307179586 * source.Discretization().Metrics().GetCellCent(iCell)[0]);
                const auto value = source.Equation().Gas().Conservative(primitive);
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = value[j];
            }
        }
        source.WriteCheckpoint(file, resolved, true);
        restored.ReadCheckpoint(file, resolved);
        CHECK(restored.Epoch().acceptedStep == 0);
        REQUIRE(source.Advance(.001).accepted);
        source.WriteCheckpoint(file, resolved, true);
        CHECK_THROWS(restored.ReadCheckpoint(file, C::Json{{"test", "wrong"}}));
        CHECK(restored.Time() == 0);
        restored.ReadCheckpoint(file, resolved);
        CHECK(restored.Dofs().CoupledField().Owned().Layout().Kind() == source.Dofs().CoupledField().Owned().Layout().Kind());
        REQUIRE(source.Advance(.001).accepted);
        REQUIRE(restored.Advance(.001).accepted);
        CHECK(restored.Time() == source.Time());
        CHECK(restored.Epoch().acceptedStep == source.Epoch().acceptedStep);
        {
            auto left = source.Dofs().CoupledField().Owned().Read(), right = restored.Dofs().CoupledField().Owned().Read();
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(left.View()(iCell, j) == right.View()(iCell, j));
        }
        REQUIRE(restored.PrepareStep(.001).accepted);
        restored.Residual().CoupledField();
        CHECK_THROWS(restored.CommitStep());
        const auto shifted = CauchyTest::PeriodicQuads(mpi, 4);
        shifted->ApplyAffineGeometryTransform(Eigen::Matrix3d::Identity(), DNDS::Geom::tPoint{.01, 0, 0});
        C::Domain different("flow", shifted, settings);
        CHECK_THROWS(different.ReadCheckpoint(file, resolved));
        CHECK(different.Time() == 0);
    }
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        CHECK(std::filesystem::remove(file));
    MPI_Barrier(mpi.comm);
}
TEST_CASE("Cauchy real single-domain Simulation owns its accepted clock and restart epoch")
{
    const auto config = C::SimulationConfig::ResolveCollective({{"defaults", {{"numerics", {{"cfv", {{"maxOrder", 1}}}}}}}}, mpi.comm);
    C::Simulation simulation(config, mpi.comm, [](DNDS::index, const auto &, const auto &domainMPI)
                             { return CauchyTest::PeriodicQuads(domainMPI, 4); });
    REQUIRE(simulation.Advance(.001).accepted);
    CHECK(simulation.Step() == 1);
    CHECK(simulation.Time() == simulation.GetDomain(0).Time());
    std::optional<R::Array<double>::ReadLease> held;
    if (mpi.rank == 0)
        held.emplace(simulation.GetDomain(0).Dofs().CoupledField().Owned().Read());
    CHECK_THROWS(simulation.Advance(.001));
    CHECK(simulation.Step() == 1);
    held.reset();
    REQUIRE(simulation.Advance(.001).accepted);
    CHECK(simulation.Step() == 2);
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto file = (std::filesystem::temp_directory_path() / ("cauchy-simulation-" + std::to_string(pid) + ".h5")).string();
    simulation.GetDomain(0).WriteCheckpoint(file, config.Root());
    REQUIRE(simulation.Advance(.001).accepted);
    simulation.ReadCheckpoint(file);
    CHECK(simulation.Step() == 2);
    CHECK(simulation.Time() == .002);
    REQUIRE(simulation.Advance(.001).accepted);
    CHECK(simulation.Step() == 3);
    CHECK(simulation.Time() == .003);
    if (mpi.rank == 0)
        CHECK(std::filesystem::remove(file));
    MPI_Barrier(mpi.comm);
}
TEST_CASE("Cauchy P3 non-PP vortex cell-average evolution converges and conserves on distributed torus")
{
    // Independently numbered 10x10 physical domain. The beta=5 field is the
    // standard nonlinear vortex, not the low-amplitude PP vortex control.
    double previous = 0;
    for (DNDS::index width : {12, 24})
    {
        C::DomainSettings settings;
        settings.initialPrimitive = {1, 1, 1, 0, 1};
        settings.initialProfile.kind = "isentropicVortex";
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 6;
        settings.numerics.cfv.jacobiRelax = .8;
        settings.numerics.reconstructionIterations = 240;
        settings.numerics.reconstructionTolerance = 1e-11;
        settings.numerics.faceFallback = false;
        C::Domain domain("vortex", CauchyTest::PeriodicQuads(mpi, width, 10), settings);
        const auto initial = domain.Diagnostics(settings, {});
        CHECK(initial[10] < 1e-14);
        for (int step = 0; step < 4; ++step)
            REQUIRE(domain.Advance(.0025).accepted);
        const auto final = domain.Diagnostics(settings, {});
        INFO("width=" << width << " error=" << final[10] << " update=" << final[9]);
        CHECK(domain.LastFaceFallbacks() == 0);
        CHECK(std::abs(final[0] - initial[0]) < 2e-11);
        CHECK(std::abs(final[1] - initial[1]) < 5e-11);
        CHECK(final[9] < 1e-9);
        CHECK(final[10] > 1e-10);
        if (previous)
            CHECK(previous / final[10] > 7);
        previous = final[10];
    }
}
TEST_CASE("Cauchy P3 characteristic 3WBAP advances shock data with inline and worker launchers")
{
    for (int workers : {0, 2})
    {
        C::DomainSettings settings;
        settings.execution.hostThreads = workers;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.limiter = "WBAP";
        settings.numerics.reconstructionIterations = 32;
        C::Domain domain("shock", CauchyTest::PeriodicQuads(mpi, 8), settings);
        {
            auto meshRead = domain.Discretization().Metrics().AcquireRead();
            auto write = domain.Dofs().CoupledField().Owned().Write();
            for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
            {
                const auto center = meshRead.HostView().GetCellBary(iCell);
                const auto state = domain.Equation().Gas().Conservative(center[0] < .5 ? C::FlowState{1, 0, 0, 0, 1} : C::FlowState{.125, 0, 0, 0, .1});
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = state[j];
            }
        }
        const auto before = domain.Diagnostics(settings, {});
        REQUIRE(domain.Advance(1e-4).accepted);
        const auto after = domain.Diagnostics(settings, {});
        CHECK(after[2] > 0);
        CHECK(after[3] > 0);
        CHECK(std::abs(after[0] - before[0]) < 1e-13);
        CHECK(std::abs(after[1] - before[1]) < 1e-12);
    }
}
TEST_CASE("Cauchy observation cadence leaves a fixed-sweep P3 trajectory unchanged")
{
    auto mesh = CauchyTest::PeriodicQuads(mpi, 8, 10);
    C::DomainSettings settings;
    settings.initialPrimitive = {1, 1, 1, 0, 1};
    settings.initialProfile.kind = "isentropicVortex";
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.reconstructionIterations = 4;
    C::Domain observed("observed", mesh, settings), control("control", mesh, settings);
    const auto first = observed.Diagnostics(settings, {});
    const auto repeat = observed.Diagnostics(settings, {});
    CHECK(first == repeat);
    for (int step = 0; step < 3; ++step)
    {
        observed.Diagnostics(settings, {});
        REQUIRE(observed.Advance(.001).accepted);
        REQUIRE(control.Advance(.001).accepted);
    }
    const auto accepted = observed.Diagnostics(settings, {});
    REQUIRE(observed.PrepareStep(.002).accepted);
    CHECK(observed.Diagnostics(settings, {}) == accepted);
    observed.DiscardStep();
    CHECK(observed.Diagnostics(settings, {}) == accepted);
    REQUIRE(observed.Advance(.001).accepted);
    REQUIRE(control.Advance(.001).accepted);
    auto a = observed.Dofs().CoupledField().Owned().Read(), b = control.Dofs().CoupledField().Owned().Read();
    for (DNDS::index iCell = 0; iCell < observed.Discretization().Metrics().NumCell(); ++iCell)
        for (int j = 0; j < 5; ++j)
            CHECK(a.View()(iCell, j) == b.View()(iCell, j));
}
TEST_CASE("Cauchy P3 boundary polynomial projection preserves an exact linear extrapolated flow")
{
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.boundaries["7"].kind = "extrapolate";
    settings.numerics.reconstruction = "variational";
    settings.numerics.flux = "Roe";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.boundaryDerivativeWeights = {1, 0, 0, 0};
    settings.numerics.reconstructionIterations = 6000;
    settings.numerics.reconstructionTolerance = 1e-11;
    settings.numerics.cfv.jacobiRelax = .8;
    const auto mesh = CFVTest::CartesianHexes(mpi, 4);
    C::FiniteVolume fv(mesh, settings, C::EquationSystem::Resolve(settings.equation));
    auto state = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
    {
        auto write = state.Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const auto x = fv.Metrics().GetCellCent(iCell);
            const auto value = C::PerfectGas{}.Conservative({1 + .2 * x[0], .3, 0, 0, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    auto logical = Bind(fv, state), output = Bind(fv, residual);
    auto reconstruction = fv.BuildReconstructionState(logical);
    auto workspace = fv.BuildWorkspace(logical);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(logical, output, *reconstruction, *workspace, engine, stream);
    INFO("boundary iteration sweeps=" << fv.LastReconstructionIterations(*workspace) << " update=" << fv.LastReconstructionUpdate(*workspace));
    CHECK(fv.LastReconstructionUpdate(*workspace) < 1e-10);
    auto geometry = fv.Metrics().AcquireRead();
    auto read = residual.Owned().Read();
    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
    {
        const double mass = read.View()(iCell, 0) / geometry.HostView().GetCellVol(iCell);
        CHECK(std::abs(mass + .06) < 2e-7);
    }
}
TEST_CASE("Cauchy P3 thermal PP preserves means and global conservation with forced temperature constraints")
{
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.flux = "Roe";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.reconstructionIterations = 24;
    settings.numerics.positivity = true;
    settings.numerics.temperatureFloor = .25;
    C::Domain domain("thermal", CauchyTest::PeriodicQuads(mpi, 8), settings);
    {
        auto geometry = domain.Discretization().Metrics().AcquireRead();
        auto write = domain.Dofs().CoupledField().Owned().Write();
        for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
        {
            const auto center = geometry.HostView().GetCellBary(iCell);
            const auto value = domain.Equation().Gas().Conservative(center[0] < .5 ? C::FlowState{1, 2, 0, 0, .251} : C::FlowState{.2, -2, 0, 0, .051});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    const auto before = domain.Diagnostics(settings, {});
    bool limited = false;
    bool stageLimited = false;
    for (int step = 0; step < 5; ++step)
    {
        REQUIRE(domain.Advance(1e-4).accepted);
        const auto pp = domain.LastPositivity();
        limited |= pp[0] > 0 || pp[2] > 0;
        stageLimited |= pp[2] > 0;
    }
    const auto after = domain.Diagnostics(settings, {});
    CHECK(std::abs(after[0] - before[0]) < 1e-12);
    CHECK(std::abs(after[1] - before[1]) < 1e-11);
    auto read = domain.Dofs().CoupledField().Owned().Read();
    for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
    {
        C::FlowState state;
        for (int j = 0; j < 5; ++j)
            state[j] = read.View()(iCell, j);
        CHECK(C::ThermallyAdmissible(state, domain.Discretization().ThermalConstraints()));
        CHECK(domain.Equation().Gas().Primitive(state).temperature >= .25);
    }
    CHECK(domain.LastFaceFallbacks() == 0);
    CHECK(limited);
    CHECK(stageLimited);
}
TEST_CASE("Inactive Cauchy P3 thermal PP leaves the full numerical trajectory bitwise unchanged")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8, 10);
    for (const std::string layout : {"aos", "soa", "aosoa32"})
        for (int workers : {0, 2})
        {
            C::DomainSettings settings;
            settings.execution.layout = layout;
            settings.execution.hostThreads = workers;
            settings.initialPrimitive = {1, .3, -.2, .1, 1};
            settings.initialProfile.kind = "isentropicVortex";
            settings.initialProfile.strength = .2;
            settings.numerics.reconstruction = "variational";
            settings.numerics.cfv.maxOrder = 3;
            settings.numerics.reconstructionIterations = 16;
            settings.numerics.faceFallback = false;
            C::Domain reference("reference", mesh, settings);
            settings.numerics.positivity = true;
            C::Domain protectedFlow("protected", mesh, settings);
            for (int step = 0; step < 3; ++step)
            {
                REQUIRE(reference.Advance(.001).accepted);
                REQUIRE(protectedFlow.Advance(.001).accepted);
                const auto factors = protectedFlow.LastPositivity();
                CHECK(factors[0] == 0);
                CHECK(factors[1] == 1);
                CHECK(factors[2] == 0);
                CHECK(factors[3] == 1);
            }
            auto expected = reference.Dofs().CoupledField().Owned().Read(), actual = protectedFlow.Dofs().CoupledField().Owned().Read();
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(actual.View()(iCell, j) == expected.View()(iCell, j));
            CHECK(protectedFlow.Time() == reference.Time());
            CHECK(protectedFlow.LastFaceFallbacks() == 0);
        }
}
TEST_CASE("Cauchy P3 fixed-sweep restart preserves the polynomial warm start and exact continuation")
{
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto path = (std::filesystem::temp_directory_path() / ("cauchy-p3-restart-" + std::to_string(pid) + ".h5")).string();
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8, 10);
    for (const auto *flux : {"rusanov", "Roe_M9"})
        for (const std::string layout : {"aos", "soa", "aosoa32"})
        {
            C::DomainSettings settings;
            settings.execution.layout = layout;
            settings.execution.hostThreads = 2;
            settings.initialPrimitive = {1, .3, -.2, .1, 1};
            settings.initialProfile.kind = "isentropicVortex";
            settings.initialProfile.strength = .2;
            settings.numerics.reconstruction = "variational";
            settings.numerics.flux = flux;
            settings.numerics.cfv.maxOrder = 3;
            // Deliberately few sweeps: losing the warm start must affect the next
            // accepted state, rather than being hidden by converged reconstruction.
            settings.numerics.reconstructionIterations = 4;
            settings.numerics.positivity = true;
            C::Domain uninterrupted("flow", mesh, settings), restored("flow", mesh, settings);
            for (int step = 0; step < 3; ++step)
                REQUIRE(uninterrupted.Advance(.001).accepted);
            const C::Json resolved{{"test", "p3-warm-start"}, {"layout", layout}, {"flux", flux}};
            uninterrupted.WriteCheckpoint(path, resolved, true);
            restored.ReadCheckpoint(path, resolved);
            for (int step = 0; step < 3; ++step)
            {
                REQUIRE(uninterrupted.Advance(.001).accepted);
                REQUIRE(restored.Advance(.001).accepted);
                CHECK(uninterrupted.LastPositivity() == restored.LastPositivity());
                auto expected = uninterrupted.Dofs().CoupledField().Owned().Read(), actual = restored.Dofs().CoupledField().Owned().Read();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        CHECK(actual.View()(iCell, j) == expected.View()(iCell, j));
            }
            CHECK(restored.Time() == uninterrupted.Time());
            CHECK(restored.Epoch().acceptedStep == uninterrupted.Epoch().acceptedStep);
        }
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        CHECK(std::filesystem::remove(path));
    MPI_Barrier(mpi.comm);
}
TEST_CASE("Cauchy thermal PP rejects invalid low-order stages collectively without clipping and permits retry")
{
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.cfv.maxOrder = 1;
    settings.numerics.positivity = true;
    settings.numerics.temperatureFloor = .2;
    settings.execution.hostThreads = 2;
    C::Domain domain("pp-rejection", CauchyTest::PeriodicQuads(mpi, 8), settings);
    {
        auto write = domain.Dofs().CoupledField().Owned().Write();
        for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
        {
            const auto x = domain.Discretization().Metrics().GetCellCent(iCell);
            const auto value = domain.Equation().Gas().Conservative({1, std::sin(6.283185307179586 * x[0]), 0, 0, .3});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    const auto saved = domain.Dofs().Fields().Clone();
    const auto rejected = domain.Advance(1e4);
    CHECK_FALSE(rejected.accepted);
    CHECK(rejected.reason.find("admissible low-order stage") != std::string::npos);
    CHECK(domain.Time() == 0);
    CHECK(domain.Epoch().acceptedStep == 0);
    {
        auto expected = saved.Block("flow").Owned().Read(), actual = domain.Dofs().CoupledField().Owned().Read();
        for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(actual.View()(iCell, j) == expected.View()(iCell, j));
    }
    REQUIRE(domain.Advance(1e-5).accepted);
    CHECK(domain.Time() == 1e-5);
    CHECK(domain.Epoch().acceptedStep == 1);
    CHECK(domain.Discretization().Admissible(domain.Dofs()));
}

TEST_CASE("Canonical flow5 positivity workspace produces the established high and low residuals without the archived FV frontend")
{
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.cfv.maxOrder = 1;
    settings.numerics.flux = "Roe_M9";
    settings.numerics.positivity = true;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto genericField = fv.Metrics().BuildField(5), genericResidualField = fv.Metrics().BuildField(5);
    auto nativeState = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(5), "packed");
    auto nativeResidual = C::EquationDofOperations::NewVector(nativeState);
    {
        auto a = genericField.Owned().Write(), b = nativeState.CoupledField().Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const auto x = fv.Metrics().GetCellCent(iCell);
            const auto value = equation.Gas().Conservative({1 + .1 * std::sin(6.283185307179586 * x[0]),
                                                            .2 + .05 * x[1], -.1, 0, 1});
            for (int j = 0; j < 5; ++j)
                a.View()(iCell, j) = b.View()(iCell, j) = value[j];
        }
    }
    auto genericState = Bind(fv, genericField), genericResidual = Bind(fv, genericResidualField);
    auto genericReconstruction = fv.BuildReconstructionState(genericState);
    auto nativeReconstruction = fv.BuildReconstructionState(nativeState);
    auto genericWork = fv.BuildWorkspace(
        genericState, std::nullopt,
        C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic});
    auto nativeWork = fv.BuildWorkspace(nativeState);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(genericState, genericResidual, *genericReconstruction, *genericWork, engine, stream);
    fv.Evaluate(nativeState, nativeResidual, *nativeReconstruction, *nativeWork, engine, stream);
    const auto legacyHigh = genericResidualField.Owned().Read();
    const auto nativeHigh = nativeResidual.CoupledField().Owned().Read();
    const auto legacyLow = fv.LowOrderResidual(*genericWork).CoupledField().Owned().Read();
    const auto nativeLow = fv.LowOrderResidual(*nativeWork).CoupledField().Owned().Read();
    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        for (int j = 0; j < 5; ++j)
        {
            CHECK(nativeHigh.View()(iCell, j) == doctest::Approx(legacyHigh.View()(iCell, j)).epsilon(2e-13));
            CHECK(nativeLow.View()(iCell, j) == doctest::Approx(legacyLow.View()(iCell, j)).epsilon(2e-13));
        }
}
TEST_CASE("Cauchy repeated parallel VTKHDF output preserves topology and nonconstant fields with explicit compression")
{
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto prefix = (std::filesystem::temp_directory_path() / ("cauchy-vtk-" + std::to_string(pid))).string();
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    // A small chunk deliberately crosses partition boundaries. Output must
    // preserve this shared mesh setting when a consumer overrides compression.
    mesh->SetHDF5OutSetting(13, 1, false, true);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.cfv.maxOrder = 1;
    C::Domain domain("output", mesh, settings);
    {
        auto write = domain.Dofs().CoupledField().Owned().Write();
        const auto cells = mesh->Entities(DNDS::Geom::EntityKind::Cell);
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
        {
            const auto value = domain.Equation().Gas().Conservative({1 + .001 * cells.LocalToGlobal(iCell), .3, .2, .1, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    for (int deflate : {0, 1})
        for (int repeat = 0; repeat < 4; ++repeat)
        {
            domain.WriteVisualization(prefix, deflate);
            CHECK(mesh->hdf5OutSetting.deflateLevel == 1);
            const auto file = H5Fopen((prefix + ".vtkhdf").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            CHECK(file >= 0);
            if (file >= 0)
            {
                std::array<DNDS::index, 65> offsets{};
                std::array<double, 64> rho{};
                const auto read = [&](const char *name, hid_t type, void *buffer, DNDS::index count)
                {
                    const auto dataset = H5Dopen(file, name, H5P_DEFAULT);
                    CHECK(dataset >= 0);
                    if (dataset < 0)
                        return;
                    const auto space = H5Dget_space(dataset);
                    const auto size = H5Sget_simple_extent_npoints(space);
                    CHECK(size == count);
                    if (size == count)
                        CHECK(H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) >= 0);
                    CHECK(H5Sclose(space) >= 0);
                    CHECK(H5Dclose(dataset) >= 0);
                };
                read("VTKHDF/Offsets", H5T_NATIVE_INT64, offsets.data(), 65);
                read("VTKHDF/CellData/rho", H5T_NATIVE_DOUBLE, rho.data(), 64);
                for (DNDS::index i = 0; i <= 64; ++i)
                    CHECK(offsets[i] == 4 * i);
                for (DNDS::index i = 0; i < 64; ++i)
                    CHECK(rho[i] == 1 + .001 * i);
                CHECK(H5Fclose(file) >= 0);
            }
            MPI_Barrier(mpi.comm);
        }
    CHECK_THROWS(domain.WriteVisualization(prefix, mpi.rank == 0 ? 10 : 0));
    if (mpi.size > 1)
        CHECK_THROWS(domain.WriteVisualization(prefix, mpi.rank == 0 ? 1 : 0));
    if (mpi.rank == 0)
    {
        CHECK(std::filesystem::remove(prefix + ".vtkhdf"));
        CHECK(std::filesystem::remove(prefix + "_bnd.vtkhdf"));
    }
    CHECK_FALSE(std::filesystem::exists(prefix + "_surface_rank" + std::to_string(mpi.rank) + ".csv"));
    MPI_Barrier(mpi.comm);
}
TEST_CASE("Riemann spatial stencils are computed by owners and exchanged across periodic partitions")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    const auto primitive = [](double x, double y) -> C::FlowState
    {
        constexpr double pi = 3.14159265358979323846;
        return {1 + .6 * std::sin(2 * pi * x) + .1 * std::cos(2 * pi * y),
                .05 * std::cos(2 * pi * y), .02 * std::sin(2 * pi * x), 0,
                1 + .2 * std::cos(2 * pi * x)};
    };
    for (const auto *flux : {"Roe_M9", "AUSMPWPlus"})
    {
        C::DomainSettings settings;
        settings.numerics.flux = flux;
        settings.numerics.reconstruction = "constant";
        settings.numerics.cfv.maxOrder = 1;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
        {
            auto write = state.Owned().Write();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto point = fv.Metrics().GetCellCent(iCell);
                const auto value = equation.Gas().Conservative(primitive(point[0], point[1]));
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = value[j];
            }
        }
        auto logical = Bind(fv, state), output = Bind(fv, residual);
        auto reconstruction = fv.BuildReconstructionState(logical);
        auto workspace = fv.BuildWorkspace(logical);
        const auto rhsRequest = fv.CompleteRHSAuxiliaryRequest();
        auto used = fv.BuildAuxiliaryValues(rhsRequest);
        CHECK_THROWS(fv.ValidateAuxValues(logical, used));
        R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(logical, output, *reconstruction, *workspace, used, engine, stream);
        CHECK(bool(used.cellWaveJump) == (std::string(flux) == "Roe_M9"));
        CHECK(bool(used.cellPressureMinimum) == (std::string(flux) == "AUSMPWPlus"));
        CHECK_FALSE(used.cellSpectralSum);
        // Request both reductions independently of the selected Riemann
        // consumer, and explicitly publish their owner-built ghost rows.
        C::AuxiliaryRequest request;
        request.cellWaveJump = request.cellPressureMinimum = true;
        request.cellSpectralSum = true;
        auto stencil = fv.BuildAuxiliaryValues(request);
        fv.PrepareAuxValues(logical, stencil);
        auto waveHalo = fv.Metrics().BindHalo(*stencil.cellWaveJump);
        auto pressureHalo = fv.Metrics().BindHalo(*stencil.cellPressureMinimum);
        waveHalo.Begin().Wait();
        pressureHalo.Begin().Wait();
        CHECK_NOTHROW(fv.ValidateAuxValues(logical, stencil));
        DNDS::index tested[2]{};
        {
            const auto ownedWave = stencil.cellWaveJump->Owned().Read(), ghostWave = stencil.cellWaveJump->Ghost().Read();
            const auto ownedPressure = stencil.cellPressureMinimum->Owned().Read(), ghostPressure = stencil.cellPressureMinimum->Ghost().Read();
            const auto sums = stencil.cellSpectralSum->Owned().Read();
            const R::PairReadView<DNDS::real> waves{ownedWave.View(), ghostWave.View()}, pressures{ownedPressure.View(), ghostPressure.View()};
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCellProc(); ++iCell)
            {
                const auto point = fv.Metrics().GetCellCent(iCell);
                const auto own = primitive(point[0], point[1]);
                const auto sound = std::sqrt(settings.equation.gamma * own[4] / own[0]);
                double expectedJump = 0, expectedPressure = own[4], expectedRadius = 0;
                // Independent Cartesian oracle. In particular, ghost rows
                // include all four neighbors of their owner, not only the
                // subset incident to this rank's owned cells.
                for (int direction = 0; direction < 2; ++direction)
                    for (int sign : {-1, 1})
                    {
                        auto neighbor = point;
                        neighbor[direction] += sign / 8.;
                        const auto next = primitive(neighbor[0], neighbor[1]);
                        const double acousticJump = std::sqrt(settings.equation.gamma * next[4] / next[0]) - sound;
                        const double velocityJump = sign * (next[1 + direction] - own[1 + direction]);
                        expectedJump = std::max({expectedJump, std::abs(velocityJump + acousticJump), std::abs(velocityJump - acousticJump), std::abs(acousticJump)});
                        expectedPressure = std::min(expectedPressure, next[4]);
                        expectedRadius += std::max(std::abs(own[1 + direction]) + sound,
                                                   std::abs(next[1 + direction]) + std::sqrt(settings.equation.gamma * next[4] / next[0])) /
                                          8.;
                    }
                CHECK(waves(iCell, 0) == doctest::Approx(expectedJump).epsilon(3e-12));
                CHECK(pressures(iCell, 0) == doctest::Approx(expectedPressure).epsilon(3e-12));
                if (iCell < fv.Metrics().NumCell())
                    CHECK(sums.View()(iCell, 0) == doctest::Approx(expectedRadius).epsilon(3e-12));
                ++tested[iCell < fv.Metrics().NumCell() ? 0 : 1];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, tested, 2, MPI_INT64_T, MPI_SUM, mpi.comm);
        CHECK(tested[0] == 64);
        if (mpi.size > 1)
            CHECK(tested[1] > 0);

        auto manualSettings = settings;
        manualSettings.numerics.riemann.useStencil = false;
        C::FiniteVolume manual(mesh, manualSettings, equation);
        auto manualState = manual.Metrics().BuildField(5), manualResidual = manual.Metrics().BuildField(5);
        {
            const auto read = state.Owned().Read();
            auto write = manualState.Owned().Write();
            for (DNDS::index iCell = 0; iCell < manual.Metrics().NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = read.View()(iCell, j);
        }
        auto manualLogical = Bind(manual, manualState), manualOutput = Bind(manual, manualResidual);
        auto manualReconstruction = manual.BuildReconstructionState(manualLogical);
        auto manualWork = manual.BuildWorkspace(manualLogical);
        manual.Evaluate(manualLogical, manualOutput, *manualReconstruction, *manualWork, engine, stream);
        const auto manualRequest = manual.CompleteRHSAuxiliaryRequest();
        CHECK_FALSE(manualRequest.cellWaveJump);
        CHECK_FALSE(manualRequest.cellPressureMinimum);
        double difference = 0;
        {
            const auto automatic = residual.Owned().Read(), explicitValue = manualResidual.Owned().Read();
            for (DNDS::index iCell = 0; iCell < manual.Metrics().NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    difference += std::abs(automatic.View()(iCell, j) - explicitValue.View()(iCell, j));
        }
        MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
        CHECK(difference > 1e-8);
    }
}
TEST_CASE("Stateless auxiliary preparation supports independently optional face and cell outputs")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    C::DomainSettings settings;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .03;
    settings.numerics.reconstruction = "constant";
    settings.numerics.cfv.maxOrder = 1;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    const C::FlowState primitive{1.2, .3, -.2, .1, 1.1};
    const auto conservative = equation.Gas().Conservative(primitive);
    for (const auto &physical : {R::RowLayout::Dense(0, 5, 8), R::RowLayout::ColMajor(0, 5, {8, 1}), R::RowLayout::TiledColMajor(0, 5, 32)})
    {
        auto state = fv.Metrics().BuildField(physical);
        {
            auto write = state.Owned().Write();
            const auto u = write.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    u(iCell, j) = conservative[j];
        }
        auto halo = fv.Metrics().BindHalo(state);
        halo.Begin().Wait();
        auto logical = Bind(fv, state);
        const auto ownedVersion = state.Owned().Version(), ghostVersion = state.Ghost().Version();
        auto all = fv.BuildAuxiliaryValues({true, true, true, true, true, true, true});
        fv.PrepareAuxValues(logical, all, .25);
        CHECK_NOTHROW(fv.ValidateAuxValues(logical, all, .25));
        CHECK(state.Owned().Version() == ownedVersion);
        CHECK(state.Ghost().Version() == ghostVersion);
        const double h = 1. / 8, a = std::sqrt(settings.equation.gamma * primitive[4] / primitive[0]);
        const double diffusion = settings.equation.viscosity / primitive[0] * std::max(4. / 3, settings.equation.gamma / settings.equation.prandtl);
        const double expected = 2 * h * (std::abs(primitive[1]) + std::abs(primitive[2]) + 2 * a) + 8 * diffusion;
        {
            const auto sumRead = all.cellSpectralSum->Owned().Read(), waveRead = all.cellWaveJump->Owned().Read(), pressureRead = all.cellPressureMinimum->Owned().Read();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                CHECK(sumRead.View()(iCell, 0) == doctest::Approx(expected).epsilon(3e-13));
                CHECK(waveRead.View()(iCell, 0) == 0);
                CHECK(pressureRead.View()(iCell, 0) == doctest::Approx(primitive[4]).epsilon(3e-13));
            }
        }
        CHECK(fv.StableStep(logical, all, .4, .25) == doctest::Approx(.4 * h * h / expected).epsilon(3e-13));
        CHECK_THROWS(fv.ValidateAuxValues(logical, all, .5));
        for (int selection = 0; selection < 5; ++selection)
        {
            C::AuxiliaryRequest request;
            request.meanThermodynamics = selection == 0 || selection == 4;
            request.faceSpectralRadius = request.faceWaveJump = request.facePressureMinimum = selection < 2;
            request.cellSpectralSum = request.cellWaveJump = request.cellPressureMinimum = selection == 0 || selection == 2;
            auto selected = fv.BuildAuxiliaryValues(request);
            fv.PrepareAuxValues(logical, selected, .25);
            CHECK_NOTHROW(fv.ValidateAuxValues(logical, selected, .25));
            const auto compare = [](const auto &reference, const auto &actual)
            {
                if (!actual)
                    return;
                const auto expectedRead = reference->Read(), actualRead = actual->Read();
                for (DNDS::index iRow = 0; iRow < actual->RowCount(); ++iRow)
                    for (DNDS::index j = 0; j < actual->RowSize(iRow); ++j)
                        CHECK(actualRead.View()(iRow, j) == expectedRead.View()(iRow, j));
            };
            compare(all.meanThermodynamics, selected.meanThermodynamics);
            compare(all.faceSpectralRadius, selected.faceSpectralRadius);
            compare(all.faceWaveJump, selected.faceWaveJump);
            compare(all.facePressureMinimum, selected.facePressureMinimum);
            for (const auto &pair : {std::make_pair(&all.cellSpectralSum, &selected.cellSpectralSum),
                                     std::make_pair(&all.cellWaveJump, &selected.cellWaveJump),
                                     std::make_pair(&all.cellPressureMinimum, &selected.cellPressureMinimum)})
                if (*pair.second)
                {
                    const std::optional<R::Array<DNDS::real>> reference{(*pair.first)->Owned()}, actual{(*pair.second)->Owned()};
                    compare(reference, actual);
                }
            if (selection == 3)
            {
                CHECK_FALSE(selected.meanThermodynamics);
                CHECK_FALSE(selected.faceSpectralRadius);
                CHECK_FALSE(selected.cellWaveJump);
                CHECK_FALSE(selected.cellPressureMinimum);
                CHECK_THROWS(fv.StableStep(logical, selected, .4, .25));
            }
        }
    }
}
TEST_CASE("Auxiliary reuse rejects active writers changed inputs replaced outputs and invalid shapes")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 2);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.cfv.maxOrder = 1;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5);
    auto logical = Bind(fv, state);
    fv.Initialize(logical, settings.initialProfile, settings.initialPrimitive);
    auto halo = fv.Metrics().BindHalo(state);
    halo.Begin().Wait();
    auto aux = fv.BuildAuxiliaryValues({true, true, true, true, true, true, true});
    CHECK_THROWS(fv.ValidateAuxValues(logical, aux));
    fv.PrepareAuxValues(logical, aux);
    {
        auto write = state.Owned().Write();
        CHECK_THROWS(fv.ValidateAuxValues(logical, aux));
        write.CancelUnmodified();
    }
    CHECK_NOTHROW(fv.ValidateAuxValues(logical, aux));
    {
        auto write = aux.meanThermodynamics->Write();
        CHECK_THROWS(fv.ValidateAuxValues(logical, aux));
    }
    CHECK_THROWS(fv.ValidateAuxValues(logical, aux));
    fv.PrepareAuxValues(logical, aux);
    {
        auto write = state.Ghost().Write();
    }
    CHECK_THROWS(fv.ValidateAuxValues(logical, aux));
    fv.PrepareAuxValues(logical, aux);
    {
        auto write = state.Owned().Write();
    }
    CHECK_THROWS(fv.ValidateAuxValues(logical, aux));
    fv.PrepareAuxValues(logical, aux);
    aux.faceWaveJump = aux.faceWaveJump->Clone();
    CHECK_THROWS(fv.ValidateAuxValues(logical, aux));
    fv.PrepareAuxValues(logical, aux);
    aux.faceWaveJump.reset();
    CHECK_THROWS(fv.ValidateAuxValues(logical, aux));
    aux.meanThermodynamics = R::Array<DNDS::real>(R::RowLayout::Dense(fv.Metrics().NumCellProc(), 2));
    CHECK_THROWS(fv.PrepareAuxValues(logical, aux));
    CHECK_THROWS(fv.ValidateAuxValues(logical, aux));
    // Correct shape with shared storage is still invalid: no source mutation.
    C::AuxiliaryValues alias;
    alias.cellWaveJump.emplace(fv.Metrics().BuildField(1));
    alias.cellPressureMinimum = alias.cellWaveJump;
    CHECK_THROWS(fv.PrepareAuxValues(logical, alias));
    C::AuxiliaryValues empty;
    fv.PrepareAuxValues(logical, empty);
    CHECK_NOTHROW(fv.ValidateAuxValues(logical, empty));
    empty.faceWaveJump.emplace(R::RowLayout::Dense(DNDS::index(fv.AuxiliaryFaces().size()), 1));
    CHECK_THROWS(fv.ValidateAuxValues(logical, empty));
    // The immutable stencil publication now pins its geometry generation.
    // Reject mutation while that publication is alive, rather than creating
    // an invalid mesh behind it. A failed edit leaves valid aux inputs usable.
    auto valid = fv.BuildAuxiliaryValues({true, true, true, true, true, true, true});
    fv.PrepareAuxValues(logical, valid);
    CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Geometry));
    CHECK_NOTHROW(fv.ValidateAuxValues(logical, valid));
}
TEST_CASE("Mean boundary spectra use stage kinematics while analytic RHS uses boundary stage time")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 4);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.numerics.reconstruction = "constant";
    settings.numerics.cfv.maxOrder = 1;
    settings.initialPrimitive = {1, .3, .2, .1, 1};
    settings.motion.angularVelocity = {0, 0, .4};
    settings.motion.translationVelocity = {.1, .07, 0};
    auto &bc = settings.boundaries["7"];
    bc.kind = "analytic";
    bc.primitive = settings.initialPrimitive;
    bc.profile.kind = "isentropicVortex";
    bc.profile.center = {.25, .2, 0};
    bc.profile.strength = 2;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5);
    auto logical = Bind(fv, state);
    fv.Initialize(logical, settings.initialProfile, settings.initialPrimitive);
    auto halo = fv.Metrics().BindHalo(state);
    halo.Begin().Wait();
    C::AuxiliaryRequest request;
    request.faceSpectralRadius = request.faceWaveJump = request.facePressureMinimum = true;
    auto first = fv.BuildAuxiliaryValues(request), second = fv.BuildAuxiliaryValues(request);
    fv.PrepareAuxValues(logical, first, 0);
    fv.PrepareAuxValues(logical, second, .6);
    CHECK_THROWS(fv.ValidateAuxValues(logical, first, .6));
    const auto geometry = fv.Metrics().AcquireRead();
    const auto metric = geometry.HostView();
    double difference = 0;
    for (const auto &entry : {std::make_pair(&first, 0.), std::make_pair(&second, .6)})
    {
        const auto spectral = entry.first->faceSpectralRadius->Read(), waves = entry.first->faceWaveJump->Read(), pressures = entry.first->facePressureMinimum->Read();
        const auto initial = first.faceSpectralRadius->Read();
        for (DNDS::index iAux = 0; iAux < DNDS::index(fv.AuxiliaryFaces().size()); ++iAux)
        {
            const auto iFace = fv.AuxiliaryFaces()[iAux];
            double speed = 0, jump = 0, pressure = 1;
            for (DNDS::index q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
            {
                const auto normal = metric.GetFaceNorm(q), point = metric.GetFacePoint(q);
                const double ownAcoustic = std::sqrt(settings.equation.gamma);
                // Independent rigid-grid velocity in mesh axes. Euler's
                // physical-boundary spectral estimate uses only the mean,
                // while kinematics still depend on the supplied stage time.
                const double angle = .4 * entry.second;
                const double gx = std::cos(angle) * .1 + std::sin(angle) * .07 - .4 * point[1];
                const double gy = -std::sin(angle) * .1 + std::cos(angle) * .07 + .4 * point[0];
                const double normalSpeed = (.3 - gx) * normal[0] + (.2 - gy) * normal[1] + .1 * normal[2];
                speed = std::max(speed, std::abs(normalSpeed) + ownAcoustic);
            }
            CHECK(spectral.View()(iAux, 0) == doctest::Approx(speed).epsilon(5e-13));
            CHECK(spectral.View()(iAux, 1) == 0);
            CHECK(waves.View()(iAux, 0) == doctest::Approx(jump).epsilon(5e-13));
            CHECK(pressures.View()(iAux, 0) == doctest::Approx(pressure).epsilon(5e-13));
            difference = std::max(difference, std::abs(spectral.View()(iAux, 0) - initial.View()(iAux, 0)));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    CHECK(difference > 1e-3);
    // Preserve the original boundary-time regression purpose in the RHS,
    // whose analytic exterior must still be evaluated at the physical stage.
    auto earlier = fv.Metrics().BuildField(5), later = fv.Metrics().BuildField(5);
    auto earlierState = Bind(fv, earlier), laterState = Bind(fv, later);
    auto reconstruction = fv.BuildReconstructionState(logical);
    auto workspace = fv.BuildWorkspace(logical);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    const auto meanVersion = state.Owned().Version();
    fv.Evaluate(logical, earlierState, *reconstruction, *workspace, engine, stream, 0);
    fv.Evaluate(logical, laterState, *reconstruction, *workspace, engine, stream, .6);
    A::Axpy(later, -1., earlier);
    CHECK(A::NormGlobal(later) > 1e-3);
    CHECK(state.Owned().Version() == meanVersion);
}
TEST_CASE("A checkpoint predating spatial Riemann options retains explicit manual behavior")
{
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto path = (std::filesystem::temp_directory_path() / ("cauchy-pre-stencil-" + std::to_string(pid) + ".h5")).string();
    C::Json input{{"defaults", {{"numerics", {{"flux", "Roe_M9"}, {"riemann", {{"useStencil", false}}}}}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::Domain source("fluid", mesh, config.Domains()[0].settings), restored("fluid", mesh, config.Domains()[0].settings);
    REQUIRE(source.Advance(.001).accepted);
    auto oldConfig = config.Root();
    oldConfig["domains"][0]["settings"]["numerics"]["riemann"].erase("useStencil");
    source.WriteCheckpoint(path, oldConfig);
    restored.ReadCheckpoint(path, config.Root());
    CHECK(restored.Time() == source.Time());
    CHECK(restored.Epoch().acceptedStep == source.Epoch().acceptedStep);
    {
        const auto expected = source.Dofs().CoupledField().Owned().Read(), actual = restored.Dofs().CoupledField().Owned().Read();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(actual.View()(iCell, j) == expected.View()(iCell, j));
    }
    input["defaults"]["numerics"]["riemann"]["useStencil"] = true;
    const auto automatic = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    C::Domain different("fluid", mesh, automatic.Domains()[0].settings);
    CHECK_THROWS(different.ReadCheckpoint(path, automatic.Root()));
    CHECK(different.Epoch().acceptedStep == 0);
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        CHECK(std::filesystem::remove(path));
    MPI_Barrier(mpi.comm);
}
TEST_CASE("Checkpoint PP policy preserves pre-Euler archives without silently changing active reconstruction limiting")
{
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto path = (std::filesystem::temp_directory_path() / ("cauchy-pre-euler-pp-" + std::to_string(pid) + ".h5")).string();
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (bool positivity : {false, true})
    {
        CAPTURE(positivity);
        C::Json input{{"defaults", {{"numerics", {{"positivity", positivity}, {"positivityReconstruction", "Segment"}}}}}};
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        C::Domain source("fluid", mesh, config.Domains()[0].settings), restored("fluid", mesh, config.Domains()[0].settings);
        REQUIRE(source.Advance(.001).accepted);
        auto oldConfig = config.Root();
        auto &oldNumerics = oldConfig["domains"][0]["settings"]["numerics"];
        for (const auto *name : {"positivityReconstruction", "positivityVolumePoints", "positivityCompressToMean", "positivityBetaPower"})
            oldNumerics.erase(name);
        source.WriteCheckpoint(path, oldConfig);
        CHECK_NOTHROW(restored.ReadCheckpoint(path, config.Root()));
        CHECK(restored.Epoch().acceptedStep == 1);
        input["defaults"]["numerics"]["positivityReconstruction"] = "Euler";
        const auto euler = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        C::Domain changed("fluid", mesh, euler.Domains()[0].settings);
        if (positivity)
        {
            CHECK_THROWS(changed.ReadCheckpoint(path, euler.Root()));
            CHECK(changed.Epoch().acceptedStep == 0);
        }
        else
        {
            CHECK_NOTHROW(changed.ReadCheckpoint(path, euler.Root()));
            CHECK(changed.Epoch().acceptedStep == 1);
        }
        auto expected = source.Dofs().CoupledField().Owned().Read(), actual = restored.Dofs().CoupledField().Owned().Read();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(actual.View()(iCell, j) == expected.View()(iCell, j));
    }
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        CHECK(std::filesystem::remove(path));
    MPI_Barrier(mpi.comm);
}
TEST_CASE("A rank-local Riemann stencil setting rejects collectively before optional halo construction")
{
    if (mpi.size == 1)
        return;
    C::DomainSettings settings;
    settings.numerics.flux = "Roe_M9";
    settings.numerics.cfv.maxOrder = 1;
    settings.numerics.riemann.useStencil = mpi.rank != 0;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    C::FiniteVolume fv(mesh, settings, C::EquationSystem::Resolve(settings.equation));
    auto state = fv.Metrics().BuildField(5);
    auto logical = Bind(fv, state);
    CHECK_THROWS_AS(fv.BuildWorkspace(logical), std::exception);
}
TEST_CASE("Optional derived auxiliaries refresh owned rows during halo exchange and feed the stateless RHS")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 8);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.boundaries["7"].kind = "extrapolate";
    settings.numerics.reconstruction = "constant";
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5), baseline = fv.Metrics().BuildField(5), result = fv.Metrics().BuildField(5);
    {
        auto write = state.Owned().Write();
        const auto u = write.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const auto p = fv.Metrics().GetCellCent(iCell);
            const auto value = equation.Gas().Conservative({1 + .02 * p[0], .2, -.1, .05, 1 + .01 * p[1]});
            for (int j = 0; j < 5; ++j)
                u(iCell, j) = value[j];
        }
    }
    auto typed = Bind(fv, state), baselineOutput = Bind(fv, baseline);
    auto reconstruction = fv.BuildReconstructionState(typed);
    auto work = fv.BuildWorkspace(typed);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(typed, baselineOutput, *reconstruction, *work, engine, stream);
    C::AuxiliaryRequest request;
    request.meanThermodynamics = true;
    request.faceSpectralRadius = true;
    request.derived = {C::DerivedQuantity::Temperature, C::DerivedQuantity::MolecularViscosity};
    auto aux = fv.BuildAuxiliaryValues(request);
    CHECK_FALSE(aux.meanThermodynamics); // One authority for p/a/diffusivity.
    REQUIRE(aux.derivedOwned);
    REQUIRE(aux.derivedGhost);
    auto preparation = fv.BeginAuxValues(typed, aux);
    auto halo = fv.Metrics().BindHalo(state).Begin();
    preparation.Apply(DNDS::CFV::ReconstructionRegion::Core);
    CHECK(aux.derivedOwned->Current(typed, equation));
    if (fv.Metrics().NumCellProc() > fv.Metrics().NumCell())
        CHECK_THROWS(preparation.Apply(DNDS::CFV::ReconstructionRegion::HaloDependent));
    halo.Wait();
    preparation.Apply(DNDS::CFV::ReconstructionRegion::HaloDependent);
    preparation.Finish(aux);
    CHECK_NOTHROW(fv.ValidateAuxValues(typed, aux));
    auto owned = aux.derivedOwned->ReadCurrent(typed, equation), ghost = aux.derivedGhost->ReadCurrent(typed, equation);
    auto meanRead = state.Owned().Read(), meanGhost = state.Ghost().Read();
    C::RHSReadViews input;
    input.mean = {meanRead.View<C::RHSFlowRows>(), meanGhost.View<C::RHSFlowRows>()};
    // P0 has no coefficient operand; optional derived auxiliaries must work
    // with the same empty coefficient view used by the production wrapper.
    input.auxiliary.meanThermodynamics = C::MeanThermodynamicView::Derived(
        owned, DNDS::OptionalRef<const C::DerivedFields::ReadScope>(ghost));
    C::RHSLocalDiagnostics diagnostics;
    {
        auto write = result.Owned().Write();
        diagnostics.Merge(fv.AcquireRHSOperator().ApplyBorrowedChecked(input, {write.View<C::RHSFlowRows>()}));
    }
    auto actual = result.Owned().Read(), expected = baseline.Owned().Read();
    const auto a = actual.View<R::UniformTableRows>(), b = expected.View<R::UniformTableRows>();
    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        for (int j = 0; j < 5; ++j)
            CHECK(a(iCell, j) == b(iCell, j));
}

TEST_CASE("Cauchy P3 viscous WBAP agrees with shared first-order, full and uncached basis")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 8);
    std::vector<DNDS::real> expected;
    for (int cacheOrder : {3, 1, 0, -1})
    {
        INFO("cache order=" << cacheOrder);
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .01;
        settings.boundaries["7"].kind = "extrapolate";
        settings.numerics.reconstruction = "variational";
        settings.numerics.reconstructionIterations = 2;
        settings.numerics.limiter = "WBAP";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.cacheDiffBase = cacheOrder >= 0;
        settings.numerics.cfv.cacheDiffBaseOrder = std::max(0, cacheOrder);
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
        {
            auto write = state.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto p = fv.Metrics().GetCellCent(iCell);
                const auto u = equation.Gas().Conservative({1 + .03 * std::sin(4 * p[0]), .2 + .02 * p[1], -.1, .05, 1 + .02 * std::cos(3 * p[2])});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = u[j];
            }
        }
        auto logical = Bind(fv, state), output = Bind(fv, residual);
        auto reconstruction = fv.BuildReconstructionState(logical);
        auto work = fv.BuildWorkspace(logical);
        R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(logical, output, *reconstruction, *work, engine, stream);
        auto read = residual.Owned().Read();
        const auto values = read.View<R::UniformTableRows>();
        std::vector<DNDS::real> actual;
        for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                actual.push_back(values(iCell, j));
        if (cacheOrder == 3)
            expected = actual;
        else
        {
            REQUIRE(actual.size() == expected.size());
            for (DNDS::index i = 0; i < DNDS::index(actual.size()); ++i)
                CHECK(actual[i] == doctest::Approx(expected[i]).epsilon(1e-12).scale(1));
        }
    }
}

TEST_CASE("Cauchy overlap and unsplit orchestration agree across CPU launchers")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 8);
    for (bool sor : {false, true})
    {
        std::vector<DNDS::real> expected;
        for (bool overlap : {false, true})
            for (int workers : {0, 2})
            {
                C::DomainSettings settings;
                settings.mesh.dimension = 3;
                settings.equation.transport = "NewtonianFourier";
                settings.equation.viscosity = .01;
                settings.boundaries["7"].kind = "extrapolate";
                settings.execution.overlap = overlap;
                settings.numerics.reconstruction = "variational";
                settings.numerics.reconstructionIterations = 3;
                settings.numerics.cfv.maxOrder = 2;
                settings.numerics.cfv.SORInstead = sor;
                // Cauchy resolves one effective SOR order for both policies;
                // changing only overlap must not change finite-sweep math.
                settings.numerics.cfv.coreFirst = false;
                settings.numerics.limiter = "WBAP";
                settings.numerics.flux = "Roe_M9";
                const auto equation = C::EquationSystem::Resolve(settings.equation);
                C::FiniteVolume fv(mesh, settings, equation);
                auto state = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
                {
                    auto write = state.Owned().Write();
                    const auto values = write.View<R::UniformTableRows>();
                    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    {
                        const auto point = fv.Metrics().GetCellCent(iCell);
                        const auto u = equation.Gas().Conservative({1 + .03 * std::sin(4 * point[0]), .2 + .02 * point[1], -.1, .05, 1 + .02 * std::cos(3 * point[2])});
                        for (int j = 0; j < 5; ++j)
                            values(iCell, j) = u[j];
                    }
                }
                const auto version = state.Owned().Version();
                auto logical = Bind(fv, state), output = Bind(fv, residual);
                auto reconstruction = fv.BuildReconstructionState(logical);
                auto work = fv.BuildWorkspace(logical);
                R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
                const auto stream = engine.CreateCPUStream();
                // A second evaluation also exercises the retained VR guess.
                for (int iteration = 0; iteration < 2; ++iteration)
                    fv.Evaluate(logical, output, *reconstruction, *work, engine, stream);
                CHECK(state.Owned().Version() == version);
                CHECK(fv.LastFaceFallbacks(*work) == 0);
                std::vector<DNDS::real> values;
                const std::array<const R::Array<DNDS::real> *, 3> outputs{
                    &residual.Owned(),
                    &fv.ReconstructionGuess(*reconstruction).CoupledField().Owned(),
                    &fv.FluxReconstruction(*reconstruction, *work).CoupledField().Owned()};
                for (const auto *array : outputs)
                {
                    auto read = array->Read();
                    const auto view = read.View<R::UniformTableRows>();
                    for (DNDS::index iCell = 0; iCell < view.RowCount(); ++iCell)
                        for (DNDS::index j = 0; j < view.RowSize(iCell); ++j)
                            values.push_back(view(iCell, j));
                }
                if (!overlap && workers == 0)
                    expected = values;
                else
                    CHECK(values == expected);
            }
    }
}

TEST_CASE("Cauchy P1 characteristic WBAP publishes direct physical gradients")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 12);
    std::vector<DNDS::real> reference;
    for (bool overlap : {false, true})
        for (int workers : {0, 2})
        {
            C::DomainSettings settings;
            settings.execution.overlap = overlap;
            settings.numerics.reconstruction = "gradient";
            settings.numerics.gradient = "leastSquares";
            settings.numerics.cfv.maxOrder = 1;
            settings.numerics.limiter = "WBAP";
            settings.numerics.characteristicLimiter = true;
            settings.numerics.positivity = false;
            const auto equation = C::EquationSystem::Resolve(settings.equation);
            C::FiniteVolume fv(mesh, settings, equation);
            auto state = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
            {
                auto write = state.Owned().Write();
                const auto values = write.View<R::UniformTableRows>();
                for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
                {
                    const auto point = fv.Metrics().GetCellCent(iCell);
                    const auto value = equation.Gas().Conservative(
                        {1 + .08 * std::sin(5 * point[0]), .25 + .03 * point[1], -.08, .02,
                         1 + .04 * std::cos(3 * point[1])});
                    for (int j = 0; j < 5; ++j)
                        values(iCell, j) = value[j];
                }
            }
            auto logical = Bind(fv, state), output = Bind(fv, residual);
            auto reconstruction = fv.BuildReconstructionState(logical);
            auto work = fv.BuildWorkspace(logical);
            R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
            const auto stream = engine.CreateCPUStream();
            fv.Evaluate(logical, output, *reconstruction, *work, engine, stream);
            engine.Synchronize();
            CHECK(fv.LastCharacteristicFallbacks(*work) == 0);
            auto read = residual.Owned().Read();
            const auto values = read.View<R::UniformTableRows>();
            std::vector<DNDS::real> actual;
            for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    REQUIRE(std::isfinite(values(iCell, j)));
                    actual.push_back(values(iCell, j));
                }
            if (!overlap && workers == 0)
                reference = actual;
            else
                CHECK(actual == reference);
        }
}
TEST_CASE("Primitive gradient overlap preserves conservative reconstruction across layouts and CPU launchers")
{
    // Row strips leave nonempty core/dependent regions even at np8, plus an
    // empty owner at np4/8. The gradient-plan test has an independent affine
    // oracle; here compare complete nonlinear orchestration and warm guesses.
    const auto mesh = DNDS::Geom::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24).mesh;
    for (const std::string reconstruction : {"gradient", "variational"})
        for (int layout = 0; layout < 3; ++layout)
        {
            std::vector<DNDS::real> expected;
            for (bool overlap : {false, true})
                for (int workers : {0, 2})
                {
                    C::DomainSettings settings;
                    settings.equation.transport = "NewtonianFourier";
                    settings.equation.viscosity = .01;
                    settings.boundaries[std::to_string(DNDS::Geom::BC_ID_DEFAULT_WALL)].kind = "extrapolate";
                    settings.execution.overlap = overlap;
                    settings.numerics.reconstruction = reconstruction;
                    settings.numerics.gradient = reconstruction == "gradient" ? "leastSquares" : "ggmp";
                    settings.numerics.reconstructionIterations = 2;
                    settings.numerics.cfv.maxOrder = reconstruction == "gradient" ? 1 : 2;
                    settings.numerics.flux = "Roe_M9";
                    const auto equation = C::EquationSystem::Resolve(settings.equation);
                    C::FiniteVolume fv(mesh, settings, equation);
                    const auto partition = fv.Metrics().CellWorkPartition();
                    if (fv.Metrics().NumCell())
                    {
                        REQUIRE_FALSE(partition.Core().empty());
                        if (mpi.size > 1)
                            REQUIRE_FALSE(partition.HaloDependent().empty());
                    }
                    const auto shape = layout == 0 ? R::RowLayout::Dense(0, 5, 7) : layout == 1 ? R::RowLayout::ColMajor(0, 5, {16, 1})
                                                                                                : R::RowLayout::TiledColMajor(0, 5, 16);
                    auto state = fv.Metrics().BuildField(shape), residual = fv.Metrics().BuildField(shape);
                    {
                        auto write = state.Owned().Write();
                        const auto values = write.View<R::UniformTableRows>();
                        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                        {
                            const auto point = fv.Metrics().GetCellCent(iCell);
                            const auto value = equation.Gas().Conservative({1 + .03 * point[0], .2 + .02 * point[1], -.1 + .01 * point[0] * point[0], .05, 1 + .02 * point[1] * point[1]});
                            for (int j = 0; j < 5; ++j)
                                values(iCell, j) = value[j];
                        }
                    }
                    const auto version = state.Owned().Version();
                    auto logical = Bind(fv, state), output = Bind(fv, residual);
                    auto reconstructionState = fv.BuildReconstructionState(logical);
                    auto work = fv.BuildWorkspace(logical);
                    // Both checked and caller-ordered eager MPI bridges occur
                    // in the matrix; every row includes inline and worker CPU.
                    R::EagerEngine engine(workers, layout == 1 ? nullptr : std::make_shared<R::AccessTrackingContext>());
                    const auto stream = engine.CreateCPUStream();
                    for (int iteration = 0; iteration < 2; ++iteration)
                        fv.Evaluate(logical, output, *reconstructionState, *work, engine, stream);
                    CHECK(state.Owned().Version() == version);
                    CHECK(fv.LastFaceFallbacks(*work) == 0);
                    std::vector<DNDS::real> values;
                    std::vector<const R::Array<DNDS::real> *> outputs{&residual.Owned()};
                    // Direct P1 has no modal guess: compare the complete RHS
                    // and unchanged mean generation above. VR additionally
                    // compares its genuine warm and finalized coefficients.
                    if (fv.PolynomialReconstruction())
                    {
                        outputs.push_back(&fv.ReconstructionGuess(*reconstructionState).CoupledField().Owned());
                        outputs.push_back(&fv.FluxReconstruction(*reconstructionState, *work).CoupledField().Owned());
                    }
                    for (const auto *array : outputs)
                    {
                        auto read = array->Read();
                        const auto view = read.View<R::UniformTableRows>();
                        for (DNDS::index iCell = 0; iCell < view.RowCount(); ++iCell)
                            for (DNDS::index j = 0; j < view.RowSize(iCell); ++j)
                            {
                                CHECK(std::isfinite(view(iCell, j)));
                                values.push_back(view(iCell, j));
                            }
                    }
                    if (!overlap && workers == 0)
                        expected = std::move(values);
                    else
                        CHECK(values == expected);
                }
        }
}
TEST_CASE("Stateless RHS consumes packed auxiliary views and preserves frozen inputs")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 16);
    for (const std::string flux : {"rusanov", "Roe_M9", "AUSMPWPlus"})
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.boundaries["7"].kind = "extrapolate";
        settings.numerics.reconstruction = "variational";
        settings.numerics.reconstructionIterations = 3;
        settings.numerics.cfv.maxOrder = 1;
        settings.numerics.flux = flux;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = fv.Metrics().BuildField(5), reference = fv.Metrics().BuildField(5);
        auto full = fv.Metrics().BuildField(5), split = fv.Metrics().BuildField(5);
        {
            auto write = state.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto point = fv.Metrics().GetCellCent(iCell);
                const auto u = equation.Gas().Conservative({1 + .01 * std::sin(point[0]), .2, -.1, .05, 1 + .02 * std::cos(point[1])});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = u[j];
            }
        }
        auto logical = Bind(fv, state), referenceOutput = Bind(fv, reference);
        const auto reconstruction = fv.BuildReconstructionState(logical);
        const auto work = fv.BuildWorkspace(logical);
        auto request = fv.CompleteRHSAuxiliaryRequest();
        request.meanThermodynamics = true;
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(logical, referenceOutput, *reconstruction, *work, auxiliary, engine, stream);
        auto coefficients = fv.FluxReconstruction(*reconstruction, *work).CoupledField();
        auto rhs = fv.AcquireRHSOperator();
        const auto meanVersion = state.Owned().Version(), coefficientVersion = coefficients.Owned().Version();
        auto mean = state.Owned().Read(), meanGhost = state.Ghost().Read(), rec = coefficients.Owned().Read();
        auto thermo = auxiliary.meanThermodynamics->Read();
        C::RHSReadViews input;
        input.mean = {mean.View<C::RHSFlowRows>(), meanGhost.View<C::RHSFlowRows>()};
        input.coefficients.owned = rec.View<R::UniformTableRows>();
        input.auxiliary.meanThermodynamics = thermo.View<R::UniformTableRows>();
        std::array<std::optional<R::Array<DNDS::real>::ReadLease>, 4> held;
        const auto readAuxiliary = [&](const auto &field, int first, auto &destination)
        {
            if (!field)
                return;
            held[first].emplace(field->Owned().Read());
            held[first + 1].emplace(field->Ghost().Read());
            destination = C::RHSFieldView{held[first]->View<R::UniformTableRows>(), held[first + 1]->View<R::UniformTableRows>()};
        };
        readAuxiliary(auxiliary.cellWaveJump, 0, input.auxiliary.cellWaveJump);
        readAuxiliary(auxiliary.cellPressureMinimum, 2, input.auxiliary.cellPressureMinimum);
        C::RHSLocalDiagnostics diagnostics;
        {
            auto ghost = coefficients.Ghost().Read();
            input.coefficients.ghost = ghost.View<R::UniformTableRows>();
            auto output = full.Owned().Write();
            diagnostics.Merge(rhs.ApplyBorrowedChecked(input, {output.View<C::RHSFlowRows>()}));
            auto uncached = input;
            uncached.auxiliary.meanThermodynamics.reset();
            diagnostics.Merge(rhs.ApplyBorrowedChecked(uncached, {output.View<C::RHSFlowRows>()}));
            auto missing = input;
            if (auxiliary.cellWaveJump)
            {
                missing.auxiliary.cellWaveJump.reset();
                CHECK_THROWS(rhs.ApplyBorrowedChecked(missing, {output.View<C::RHSFlowRows>()}));
            }
            if (auxiliary.cellPressureMinimum)
            {
                missing = input;
                missing.auxiliary.cellPressureMinimum.reset();
                CHECK_THROWS(rhs.ApplyBorrowedChecked(missing, {output.View<C::RHSFlowRows>()}));
            }
        }
        {
            auto output = split.Owned().Write();
            auto core = input;
            core.mean.ghost = {};
            core.coefficients.ghost = {};
            if (core.auxiliary.cellWaveJump)
                core.auxiliary.cellWaveJump->ghost = {};
            if (core.auxiliary.cellPressureMinimum)
                core.auxiliary.cellPressureMinimum->ghost = {};
            auto transfer = fv.Metrics().BindHalo(coefficients).Begin();
            diagnostics.Merge(rhs.ApplyBorrowedChecked(core, {output.View<C::RHSFlowRows>()}, DNDS::CFV::ReconstructionRegion::Core));
            transfer.Wait();
            auto ghost = coefficients.Ghost().Read();
            input.coefficients.ghost = ghost.View<R::UniformTableRows>();
            diagnostics.Merge(rhs.ApplyBorrowedChecked(input, {output.View<C::RHSFlowRows>()}, DNDS::CFV::ReconstructionRegion::HaloDependent));
        }
        auto a = reference.Owned().Read(), b = full.Owned().Read(), c = split.Owned().Read();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
            {
                CHECK(a.View()(iCell, j) == b.View()(iCell, j));
                CHECK(b.View()(iCell, j) == c.View()(iCell, j));
            }
        CHECK(state.Owned().Version() == meanVersion);
        CHECK(coefficients.Owned().Version() == coefficientVersion);
        CHECK(diagnostics.FaceFallbacks() == 0);

        // A separately supplied wave/pressure field must affect the flux. This
        // catches an operator that silently re-prepares or ignores the bundle.
        if (auxiliary.cellWaveJump || auxiliary.cellPressureMinimum)
        {
            auto substitute = fv.Metrics().BuildField(1), changed = fv.Metrics().BuildField(5);
            for (auto *array : {&substitute.Owned(), &substitute.Ghost()})
            {
                auto write = array->Write();
                const auto view = write.View<R::UniformTableRows>();
                for (DNDS::index iCell = 0; iCell < view.RowCount(); ++iCell)
                    view(iCell, 0) = auxiliary.cellWaveJump ? 100 : 1e-6;
            }
            auto own = substitute.Owned().Read(), ghost = substitute.Ghost().Read(), recGhost = coefficients.Ghost().Read();
            auto modified = input;
            modified.coefficients.ghost = recGhost.View<R::UniformTableRows>();
            const C::RHSFieldView replacement{own.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
            if (auxiliary.cellWaveJump)
                modified.auxiliary.cellWaveJump = replacement;
            if (auxiliary.cellPressureMinimum)
                modified.auxiliary.cellPressureMinimum = replacement;
            {
                auto output = changed.Owned().Write();
                diagnostics.Merge(rhs.ApplyBorrowedChecked(modified, {output.View<C::RHSFlowRows>()}));
            }
            auto changedRead = changed.Owned().Read();
            int differs = 0;
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    differs |= changedRead.View()(iCell, j) != b.View()(iCell, j);
            int anyDiffers = 0;
            MPI_Allreduce(&differs, &anyDiffers, 1, MPI_INT, MPI_MAX, mpi.comm);
            CHECK(anyDiffers == 1);
        }
    }
}

TEST_CASE("Direct FV consumes physical gradients without coefficients and splits flux from source")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 3);
    mesh->TransformCoords([](DNDS::Geom::tPoint p) -> DNDS::Geom::tPoint
                          { return {2 * p[0], .7 * p[1], 1.3 * p[2]}; });
    for (const bool p0 : {false, true})
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.numerics.reconstruction = "gradient";
        settings.numerics.cfv.maxOrder = 3; // Must not leak a VR width/basis requirement into direct FV.
        settings.numerics.firstOrderConvection = p0;
        settings.boundaries["7"].kind = "extrapolate";
        settings.motion.angularVelocity = {.1, -.2, .15};
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto mean = fv.Metrics().BuildField(5), gradient = fv.Metrics().BuildField(R::RowLayout::ColMajor(0, 15, {8, 1}));
        auto combined = fv.Metrics().BuildField(5), split = fv.Metrics().BuildField(5);
        const double g[5][3]{{.01, -.02, .03}, {.02, .01, -.01}, {-.01, .02, .01}, {.01, .01, .02}, {.03, -.01, .02}};
        const C::FlowState base{1, .2, -.1, .05, 3};
        const auto geometry = fv.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        {
            auto u = mean.Owned().Write(), grad = gradient.Owned().Write();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto x = metric.GetCellBary(iCell);
                for (int j = 0; j < 5; ++j)
                {
                    u.View()(iCell, j) = base[j];
                    for (int d = 0; d < 3; ++d)
                    {
                        u.View()(iCell, j) += g[j][d] * x[d];
                        grad.View()(iCell, j * 3 + d) = g[j][d];
                    }
                }
            }
        }
        fv.BindHalo(mean).Begin().Wait();
        fv.BindHalo(gradient).Begin().Wait();
        auto faces = fv.BuildFaceFluxBuffer(5);
        const auto u = mean.Owned().Read(), ug = mean.Ghost().Read(), grad = gradient.Owned().Read(), gg = gradient.Ghost().Read();
        C::RHSReadViews input;
        input.mean = {u.View<C::RHSFlowRows>(), ug.View<C::RHSFlowRows>()};
        if (!p0)
            input.conservativeGradient = C::RHSFieldView{grad.View<R::UniformTableRows>(), gg.View<R::UniformTableRows>()};
        C::RHSLocalDiagnostics diagnostics;
        const auto op = fv.AcquireRHSOperator();
        {
            auto a = combined.Owned().Write(), b = split.Owned().Write(), face = faces.Write();
            C::RHSWriteViews out{a.View<C::RHSFlowRows>()};
            diagnostics.Merge(op.ApplyBorrowedChecked(input, out, DNDS::CFV::ReconstructionRegion::All, .3));
            out.residual = b.View<C::RHSFlowRows>();
            out.faceFlux = face.View<C::RHSFlowRows>();
            diagnostics.Merge(op.ApplyFlux(input, out, DNDS::CFV::ReconstructionRegion::All, .3));
            op.AddSource(input, out, DNDS::CFV::ReconstructionRegion::All, .3);
            if (!p0)
            {
                auto missing = input;
                missing.conservativeGradient.reset();
                CHECK_THROWS(op.ApplyFlux(missing, out));
            }
        }
        const auto a = combined.Owned().Read(), b = split.Owned().Read(), flux = faces.Read();
        const auto topologyRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
        const auto topology = topologyRead.HostView();
        const auto frame = fv.Motion().At(.3);
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            C::FlowState meanState{}, gathered{}, independent{};
            for (int j = 0; j < 5; ++j)
                meanState[j] = u.View()(iCell, j);
            const auto source = frame.RotationSource(meanState);
            for (int j = 0; j < 5; ++j)
                gathered[j] = independent[j] = metric.GetCellVol(iCell) * source[j];
            const auto cellFaces = topology.CellFaces(iCell);
            for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(cellFaces.size()); ++slot)
            {
                const auto iFace = cellFaces[slot];
                const auto auxiliaryFaces = fv.AuxiliaryFaces();
                const auto row = std::lower_bound(auxiliaryFaces.begin(), auxiliaryFaces.end(), iFace) - auxiliaryFaces.begin();
                // The face buffer is canonical parent-zero flux even when
                // only parent one is owned here. Incidence slots also retain
                // the two sides of a periodic self-neighbor.
                const bool back = topology.CellIsFaceBack(iCell, iFace, slot);
                for (int j = 0; j < 5; ++j)
                    gathered[j] += (back ? 1 : -1) * flux.View()(row, j);
                if (!p0)
                {
                    const auto point = metric.GetFaceCent(iFace);
                    const auto normal = ((topology.CellIsFaceBack(iCell, iFace, slot) ? 1 : -1) * metric.GetFaceMeanNorm(iFace)).eval();
                    C::FlowState value = base;
                    for (int j = 0; j < 5; ++j)
                        for (int d = 0; d < 3; ++d)
                            value[j] += g[j][d] * point[d];
                    const auto physical = C::Riemann::PhysicalFlux(equation.PrepareFluxState(value).flow, {normal[0], normal[1], normal[2]}, frame.GridVelocity({point[0], point[1], point[2]}));
                    for (int j = 0; j < 5; ++j)
                        independent[j] -= metric.GetFaceArea(iFace) * physical[j];
                }
            }
            for (int j = 0; j < 5; ++j)
            {
                CHECK(a.View()(iCell, j) == b.View()(iCell, j));
                CHECK(a.View()(iCell, j) == doctest::Approx(gathered[j]).epsilon(3e-12).scale(1));
                if (!p0)
                    CHECK(a.View()(iCell, j) == doctest::Approx(independent[j]).epsilon(3e-12).scale(1));
            }
        }
        CHECK(diagnostics.FaceFallbacks() == 0);
    }
}

TEST_CASE("RHS checked and production-fatal policies share normal arithmetic")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.boundaries["7"].kind = "extrapolate";
    settings.numerics.reconstruction = "constant";
    settings.numerics.firstOrderConvection = true;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto mean = fv.Metrics().BuildField(5), checked = fv.Metrics().BuildField(5), fatal = fv.Metrics().BuildField(5);
    auto logical = Bind(fv, mean);
    fv.Initialize(logical, {}, settings.initialPrimitive);
    fv.BindHalo(mean).Begin().Wait();
    const auto owned = mean.Owned().Read(), ghost = mean.Ghost().Read();
    C::RHSReadViews input;
    input.mean = {owned.View<C::RHSFlowRows>(), ghost.View<C::RHSFlowRows>()};
    input.includeViscosity = false;
    const auto rhs = fv.AcquireRHSOperator();
    C::RHSKernelDiagnostics checkedDiagnostics, fatalDiagnostics;
    {
        auto a = checked.Owned().Write(), b = fatal.Owned().Write();
        checkedDiagnostics = rhs.ApplyBorrowedChecked(input, {a.View<C::RHSFlowRows>()});
        fatalDiagnostics = rhs.ApplyBorrowedFatal(input, {b.View<C::RHSFlowRows>()});
    }
    const auto a = checked.Owned().Read(), b = fatal.Owned().Read();
    for (DNDS::index iCell = 0; iCell < a.View().RowCount(); ++iCell)
        for (int j = 0; j < 5; ++j)
            CHECK(a.View()(iCell, j) == b.View()(iCell, j));
    CHECK(checkedDiagnostics.FaceFallbacks() == fatalDiagnostics.FaceFallbacks());

#if defined(DNDS_UNIX_LIKE)
    const auto child = fork();
    REQUIRE(child >= 0);
    if (child == 0)
    {
        struct rlimit limit
        {
            0, 0
        };
        setrlimit(RLIMIT_CORE, &limit);
        std::signal(SIGABRT, SIG_DFL);
        if (!std::freopen("/dev/null", "w", stderr))
            _exit(2);
        try
        {
            (void)rhs.ApplyBorrowedFatal(input, {}, DNDS::CFV::ReconstructionRegion::All,
                                         std::numeric_limits<DNDS::real>::quiet_NaN());
        }
        catch (...)
        {
            // A regression must fail the death-test assertion, not re-enter
            // doctest/MPI_Finalize in a fork child and hang the parent rank.
            _exit(3);
        }
        _exit(0);
    }
    int status = 0;
    REQUIRE(waitpid(child, &status, 0) == child);
    CHECK(WIFSIGNALED(status));
    if (WIFSIGNALED(status))
        CHECK(WTERMSIG(status) == SIGABRT);
#endif
}

int main(int argc, char **argv)
{
    int provided = 0;
    const int required = std::getenv("DNDS_TEST_COMMUNICATION_WORKER") ? MPI_THREAD_MULTIPLE : MPI_THREAD_FUNNELED;
    MPI_Init_thread(&argc, &argv, required, &provided);
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
