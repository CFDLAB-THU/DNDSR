#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Core/TimeStageOperators.hpp"
#include "Cauchy/State/MeanStateCheck.hpp"
#include "Cauchy/State/StateDiagnostics.hpp"
#include "Cauchy/Discretization/FlowErrorOperator.hpp"
#include "Cauchy/Discretization/BoundaryForceReduction.hpp"
#include "Cauchy/Interface/InterfacePacketOperator.hpp"
#include "Cauchy/IO/OutputFields.hpp"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Implicit/NonlinearOperators.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "CompactManagedStencilFixture.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using DNDS::real;
static DNDS::MPIInfo mpi;

namespace
{
    void Compare(const R::Array<real> &expected, const R::Array<real> &actual)
    {
        const auto a = expected.Read(), b = actual.Read();
        REQUIRE(a.View().RowCount() == b.View().RowCount());
        REQUIRE(a.View().Width() == b.View().Width());
        for (I iCell = 0; iCell < a.View().RowCount(); ++iCell)
            for (I j = 0; j < a.View().Width(); ++j)
            {
                CAPTURE(iCell);
                CAPTURE(j);
                CHECK(std::isfinite(b.View()(iCell, j)));
                CHECK(b.View()(iCell, j) == doctest::Approx(a.View()(iCell, j)).epsilon(3e-8).scale(1));
            }
    }
}

TEST_CASE("Prepared interface packet maps components and rotates vectors on every backend")
{
    R::Array<C::InterfacePacketRow> rows(R::RowLayout::Dense(2, 1));
    R::Array<real> source(R::RowLayout::Dense(2, 7));
    R::Array<real> initial(R::RowLayout::Dense(3, 7));
    {
        auto destinations = rows.Write();
        destinations.View()(0, 0) = {2, false};
        destinations.View()(1, 0) = {0, true};
        auto input = source.Write(), output = initial.Write();
        for (I iRow = 0; iRow < 2; ++iRow)
            for (I j = 0; j < 7; ++j)
                input.View()(iRow, j) = 10 * iRow + j + 1;
        for (I iRow = 0; iRow < 3; ++iRow)
            for (I j = 0; j < 7; ++j)
                output.View()(iRow, j) = 777;
    }
    const R::MetadataBuffer<I> components(std::vector<I>{0, 2, 1, 3, 4, 5, 6});
    const R::MetadataBuffer<I> vectors(std::vector<I>{1});
    for (int backend = 0; backend < 3; ++backend)
    {
        const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
        const R::ExecutionContext placement{device, nullptr};
        C::InterfacePacketOperator mapping(rows, {components}, vectors, 7, 3, 3, placement);
        auto values = source.To(device, placement);
        // Two distinct outputs, including on the CPU anchor.
        R::Array<real> quadrature(initial.Layout(), device), faces(initial.Layout(), device);
        quadrature.CopyFrom(initial, placement).Wait();
        faces.CopyFrom(initial, placement).Wait();
        R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto lane = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
        for (const bool rotate : {false, true})
        {
            C::InterfacePacketRotation rotation;
            rotation.donorActive = rotate;
            rotation.donor = {0, -1, 0, 1, 0, 0, 0, 0, 1};
            mapping.Submit(engine, lane, {values}, quadrature, faces, rotation);
            engine.Synchronize();
            const auto hostQ = quadrature.To(R::Device::Host(), placement), hostF = faces.To(R::Device::Host(), placement);
            const auto q = hostQ.Read(), f = hostF.Read();
            for (I j = 0; j < 7; ++j)
            {
                CHECK(q.View()(0, j) == 777);
                CHECK(f.View()(2, j) == 777);
            }
            CHECK(q.View()(2, 0) == 1);
            CHECK(f.View()(0, 0) == 11);
            CHECK(q.View()(2, 1) == (rotate ? -2 : 3));
            CHECK(q.View()(2, 2) == (rotate ? 3 : 2));
            CHECK(f.View()(0, 1) == (rotate ? -12 : 13));
            CHECK(f.View()(0, 2) == (rotate ? 13 : 12));
            CHECK(q.View()(2, 3) == 4);
            CHECK(q.View()(2, 6) == 7);
            CHECK(f.View()(0, 6) == 17);
        }
    }
}

TEST_CASE("Masked mean validation skips unused interface rows without downloading state")
{
    C::DomainSettings settings;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    const DNDS::Algebra::DofSpace space(2, 0, "interface check rows");
    auto host = C::EquationDofs::Coupled(equation.Schema(), DNDS::Algebra::DofField<real>(space, R::RowLayout::Dense(0, 5)));
    R::Array<uint8_t> selected(R::RowLayout::Dense(2, 1));
    {
        auto state = host.CoupledField().Owned().Write();
        for (I j = 0; j < 5; ++j)
        {
            state.View()(0, j) = j == 0 ? 1 : j == 4 ? 3
                                                     : 0;
            state.View()(1, j) = std::numeric_limits<real>::quiet_NaN();
        }
        auto mask = selected.Write();
        mask.View()(0, 0) = 1;
        mask.View()(1, 0) = 0;
    }
    for (int backend = 0; backend < 3; ++backend)
    {
        const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
        const R::ExecutionContext placement{device, nullptr};
        auto state = C::EquationDofOperations::CopyOwnedTo(host, device, placement);
        auto active = selected.To(device, placement);
        R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto lane = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
        const auto &n = settings.numerics;
        C::AdmissibilityPolicy policy(equation, n.densityFloor, n.temperatureFloor, n.sensibleEnergyFloor, n.pressureFloor);
        C::MeanStateCheck check(space, policy, placement, active);
        CHECK(bool(check.Evaluate(engine, lane, state)));
        C::MeanStateCheck allRows(space, policy, placement);
        CHECK_FALSE(bool(allRows.Evaluate(engine, lane, state)));
    }
}

TEST_CASE("Boundary packet publication and frozen snapshots retain device placement")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 2);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.SORInstead = false;
    settings.boundaries["7"].kind = "interface";
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    const C::FiniteVolume cpu(mesh, settings, equation);
    for (int backend = 0; backend < 3; ++backend)
    {
        const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
        const R::ExecutionContext placement{device, nullptr};
        auto plan = backend == 2 ? cpu.Placed(device, placement) : cpu;
        R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto lane = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
        auto packet = plan.BuildBoundaryInputs({}, lane);
        engine.Synchronize();
        for (auto *field : {&packet.QuadratureValues(), &packet.FaceCenterValues()})
        {
            CHECK(field->Device() == device);
            auto &output = field->CoupledField().Owned();
            R::Array<real> host(output.Layout());
            {
                auto write = host.Write();
                for (I iRow = 0; iRow < host.RowCount(); ++iRow)
                    for (I j = 0; j < 5; ++j)
                        write.View()(iRow, j) = j == 0 ? 1 : j == 4 ? 3
                                                                    : 0;
            }
            output.CopyFrom(host, lane.Context()).Wait();
        }
        plan.PublishBoundaryInputs(packet, .25, {}, {}, lane);
        plan.ValidateBoundaryInputs(packet, .25);
        auto frozen = plan.FreezeBoundaryInputs(packet, lane);
        CHECK(frozen.QuadratureValues().Device() == device);
        plan.BeginBoundaryUpdate(packet, lane);
        engine.Synchronize();
        // Clearing the reused staging bundle must not alter a retained stage.
        plan.ValidateBoundaryInputs(frozen, .25);
        const auto host = frozen.QuadratureValues().CoupledField().Owned().To(R::Device::Host(), lane.Context());
        const auto read = host.Read();
        for (I iRow = 0; iRow < host.RowCount(); ++iRow)
        {
            CHECK(read.View()(iRow, 0) == 1);
            CHECK(read.View()(iRow, 4) == 3);
        }
    }
}

TEST_CASE("Boundary momentum reduction selects compact face ranges across inline worker and CUDA")
{
    // More than one reduction block of real boundary faces. Both the center
    // and quadrature contracts exercise the same canonical registry; no mock
    // point-to-expanded-sample map masks a missing production topology input.
    for (const bool centers : {false, true})
    {
        CauchyTest::CompactManagedStencilFixture fixture(CFVTest::CartesianHexes(mpi, 5), centers);
        R::Array<real> momentum(R::RowLayout::Dense(fixture.boundaryRows, 3));
        C::Vector3 expected{};
        {
            auto values = momentum.Write();
            for (I iPoint = 0; iPoint < momentum.RowCount(); ++iPoint)
                for (int d = 0; d < 3; ++d)
                {
                    values.View()(iPoint, d) = (d + 1) * .01 * (iPoint + 1);
                    expected[d] += values.View()(iPoint, d);
                }
        }
        for (const int backend : {0, 1, 2})
        {
            const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
            R::EagerEngine engine(backend == 1 ? 1 : 0);
            const auto stream = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
            const auto metrics = backend == 2 ? fixture.metrics.Placed(device, stream.Context()) : fixture.metrics;
            const auto resident = backend == 2 ? fixture.stencil.Placed(metrics, stream.Context()) : fixture.stencil;
            const auto ranges = fixture.boundaryFaces.To(device, stream.Context());
            const auto flux = momentum.To(device, stream.Context());
            C::BoundaryForceReduction reduction(stream.Context());
            const auto force = reduction.Evaluate(engine, stream, resident, ranges, flux, {7, 9, 7});
            for (int d = 0; d < 3; ++d)
                CHECK(force[d] == doctest::Approx(expected[d]).epsilon(3e-12));
            CHECK(reduction.Evaluate(engine, stream, resident, ranges, flux, {7, 9, 7}) == force);
            CHECK(reduction.Evaluate(engine, stream, resident, ranges, flux, {9}) == C::Vector3{});
            CHECK(reduction.Evaluate(engine, stream, resident, ranges, flux, {}) == C::Vector3{});
        }
    }
}
TEST_CASE("Spatial residual uses one managed schedule for inline worker and CUDA")
{
    // Exercise the production operator entry, not an independently reproduced
    // schedule. Physical boundary traces, repeated VR/limiter publications,
    // source sampling and distributed halos are included. Canonical CPU uses
    // the generic equation kernel; CUDA is allowed its specialized leaf.
    auto mesh = CFVTest::CartesianHexes(mpi, 3);
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
    for (const int scenario : {0, 1, 2, 3})
    {
        CAPTURE(scenario);
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.execution.overlap = true;
        settings.numerics.reconstruction = scenario == 2 ? "gradient" : "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.SORInstead = false;
        settings.numerics.cfv.cacheDiffBase = scenario != 1;
        settings.numerics.reconstructionIterations = 2;
        // Positive but unreachable tolerance exercises scheduled norms at the
        // configured interval without making roundoff decide iteration count.
        settings.numerics.reconstructionTolerance = 1e-30;
        settings.numerics.reconstructionResidualCheckInterval = 2;
        settings.numerics.limiter = scenario == 1 ? "LVR" : "WBAP";
        settings.numerics.limitedVR = DNDS::CFV::LimitedVRSettings::ForVariant("v3");
        settings.numerics.characteristicLimiter = true;
        settings.numerics.positivity = true;
        settings.numerics.viscousGradient = "polynomial";
        settings.boundaries["7"].kind = "extrapolate";
        if (scenario >= 2)
        {
            settings.equation.transport = "NewtonianFourier";
            settings.equation.viscosity = .003;
            settings.equation.rans.model = scenario == 2 ? "SA" : "KOmegaSST";
            settings.equation.rans.viscosityReference = .01;
        }
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume cpu(mesh, settings, equation);
        const I components = scenario == 2 ? 6 : scenario == 3 ? 7
                                                               : 5;
        auto seed = C::EquationDofs::Coupled(equation.Schema(), cpu.Metrics().BuildField(components));
        {
            const auto geometry = cpu.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto write = seed.CoupledField().Owned().Write();
            const auto values = write.View<R::TableRows>();
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const auto x = metric.GetCellCent(iCell);
                const auto flow = equation.Gas().Conservative({1.1 + .02 * x[0], .2 + .03 * x[1], -.1, .02, 1.2 + .01 * x[2]});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = flow[j];
                if (components >= 6)
                    values(iCell, 5) = .02 + .001 * x[0];
                if (components == 7)
                    values(iCell, 6) = 20 + .1 * x[1];
            }
        }
        // The fixture tiles the unit cube with 3^3 equal-volume cells. The
        // spatial operator returns the integrated rate; use a small forward
        // step solely to exercise mean mutation -> fresh halo -> next residual.
        // This is not an ODE accuracy or implicit-driver acceptance test.
        constexpr real cellVolume = 1. / 27.;
        constexpr real dt = 1e-6;
        {
            const auto geometry = cpu.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            for (I iCell = 0; iCell < seed.Space().OwnedRows(); ++iCell)
                REQUIRE(metric.GetCellVol(iCell) == doctest::Approx(cellVolume));
        }
        std::vector<R::Array<real>> expectedResidual, expectedCoefficients, expectedState;
        std::vector<R::Array<real>> expectedDiagonal, expectedBlocks, expectedAction;
        std::vector<std::array<C::FlowState, 3>> expectedErrors;
        R::Array<real> hostTau(R::RowLayout::Dense(cpu.Metrics().NumCell(), 1));
        {
            auto write = hostTau.Write();
            for (I iCell = 0; iCell < hostTau.RowCount(); ++iCell)
                write.View()(iCell, 0) = 2 + .01 * iCell;
        }
        for (const int backend : {0, 1, 2})
        {
            CAPTURE(backend);
            const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
            const R::ExecutionContext copy{device, nullptr};
            auto plan = backend == 2 ? cpu.Placed(device, copy) : cpu;
            auto state = C::EquationDofs::Coupled(equation.Schema(), plan.Metrics().BuildField(components));
            state.CoupledField().Owned().CopyFrom(seed.CoupledField().Owned(), copy).Wait();
            auto residual = C::EquationDofOperations::NewVector(state);
            auto reconstruction = plan.BuildReconstructionState(state);
            auto workspace = plan.BuildWorkspace(state, {}, C::FiniteVolume::WorkspaceOptions(C::FiniteVolume::RHSKernelPolicy::Generic));
            auto execution = settings.execution;
            execution.device = backend == 2 ? "cuda" : "cpu";
            execution.hostThreads = backend == 1 ? 1 : 0;
            C::DomainExecutionContext context(execution);
            std::optional<C::FlowErrorOperator> errorOperator;
            if (scenario == 0)
                errorOperator.emplace(plan, context.ComputeStream().Context());
            C::AuxiliaryRequest auxiliaryRequest;
            auxiliaryRequest.faceSpectralRadius = true;
            auto auxiliary = plan.BuildAuxiliaryValues(auxiliaryRequest);
            auto tau = hostTau.To(device, copy);
            auto matrix = plan.BuildLinearProblem(state, C::DiagonalStorage::Block);
            auto evaluated = plan.BuildLinearProblem(state, C::DiagonalStorage::Block, C::OffDiagonalStorage::DirectEvaluated);
            auto action = C::EquationDofOperations::NewVector(state);
            for (int pass = 0; pass < 2; ++pass)
            {
                CAPTURE(pass);
                plan.Execute(state, *reconstruction, *workspace, context, C::FiniteVolumeEvaluation::Spatial(residual, 0, std::ref(auxiliary)));
                const auto result = residual.CoupledField().Owned().To(R::Device::Host());
                if (backend == 0)
                    expectedResidual.push_back(result.Clone());
                else
                    Compare(expectedResidual[pass], result);
                if (errorOperator)
                {
                    C::FlowProfileSettings profile;
                    profile.kind = "constant";
                    const auto error = errorOperator->Evaluate(context.Engine(), context.ComputeStream(), state, profile, {1, .2, -.1, .02, 1.2}, 0);
                    if (backend == 0)
                        expectedErrors.push_back(error);
                    else
                        for (int norm = 0; norm < 3; ++norm)
                            for (int j = 0; j < 5; ++j)
                                CHECK(error[norm][j] == doctest::Approx(expectedErrors[pass][norm][j]).epsilon(3e-10).scale(1));
                }
                if (scenario != 2)
                {
                    const auto coefficients = reconstruction->FinalizedValues().CoupledField().Owned().To(R::Device::Host());
                    if (backend == 0)
                        expectedCoefficients.push_back(coefficients.Clone());
                    else
                        Compare(expectedCoefficients[pass], coefficients);
                    if (pass == 0)
                    {
                        // Durable midpoint/snapshot restoration is distinct
                        // from spatial scratch; copying it must not invalidate
                        // the producer's RHS/Jacobian preparation.
                        auto restored = plan.BuildReconstructionState(state);
                        auto restoreWork = plan.BuildWorkspace(state);
                        plan.CopyReconstructionState(*restored, *reconstruction, context.ComputeStream());
                        CHECK(restored->FinalizedCurrent());
                        Compare(coefficients, restored->FinalizedValues().CoupledField().Owned().To(R::Device::Host()));
                        const auto history = plan.LimitedVRHistory(*reconstruction);
                        if (history)
                        {
                            const auto copiedHistory = plan.LimitedVRHistory(*restored);
                            REQUIRE(copiedHistory);
                            Compare(history->To(R::Device::Host()), copiedHistory->To(R::Device::Host()));
                        }
                        plan.RestoreReconstructionGuess(*restored, reconstruction->WarmValues(), {}, context.ComputeStream());
                        Compare(reconstruction->WarmValues().CoupledField().Owned().To(R::Device::Host()),
                                restored->WarmValues().CoupledField().Owned().To(R::Device::Host()));
                        plan.ResetReconstruction(*restored, *restoreWork, context.ComputeStream());
                        const auto reset = restored->WarmValues().CoupledField().Owned().To(R::Device::Host());
                        const auto read = reset.Read();
                        for (I iCell = 0; iCell < reset.RowCount(); ++iCell)
                            for (I j = 0; j < reset.Layout().Width(); ++j)
                                CHECK(read.View()(iCell, j) == 0);
                    }
                }
                // Production spatial publication -> Jacobian producer ->
                // queued sparse action, with no host reconstruction of device
                // numerical inputs. The second pass also exercises slot reuse
                // while the preceding evaluated publication remains alive.
                const C::FiniteVolume::JacobianContext jacobianContext{*reconstruction, *workspace, context.ComputeStream()};
                const C::JacobianPreparation controls{3, .7, false};
                plan.PrepareJacobian(state, auxiliary, tau, *matrix, controls, 0, {}, jacobianContext);
                const auto diagonal = matrix->Diagonal().To(R::Device::Host());
                const auto blocks = matrix->OffDiagonal().Values().To(R::Device::Host());
                if (backend == 0)
                {
                    expectedDiagonal.push_back(diagonal.Clone());
                    expectedBlocks.push_back(blocks.Clone());
                }
                else
                {
                    Compare(expectedDiagonal[pass], diagonal);
                    Compare(expectedBlocks[pass], blocks);
                }
                matrix->SubmitApply(context.Engine(), context.ComputeStream(), state, action);
                context.Synchronize();
                const auto assembledAction = action.CoupledField().Owned().To(R::Device::Host());
                if (backend == 0)
                    expectedAction.push_back(assembledAction.Clone());
                else
                    Compare(expectedAction[pass], assembledAction);
                plan.PrepareJacobian(state, auxiliary, tau, *evaluated, controls, 0, {}, jacobianContext);
                evaluated->SubmitApply(context.Engine(), context.ComputeStream(), state, action);
                evaluated->CheckQueuedActions(context.Engine(), context.ComputeStream());
                context.Synchronize();
                Compare(expectedAction[pass], action.CoupledField().Owned().To(R::Device::Host()));
                C::EquationDofOperations::Axpy(context.Engine(), context.ComputeStream(), state,
                                               dt / cellVolume, residual);
                context.Synchronize();
                const auto advanced = state.CoupledField().Owned().To(R::Device::Host());
                if (backend == 0)
                    expectedState.push_back(advanced.Clone());
                else
                    Compare(expectedState[pass], advanced);
            }
        }
    }
}

TEST_CASE("Managed nonlinear arithmetic retains CPU worker and CUDA state semantics")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 3);
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
    for (const int n : {5, 6, 7})
    {
        CAPTURE(n);
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.boundaries["7"].kind = "extrapolate";
        if (n > 5)
        {
            settings.equation.transport = "NewtonianFourier";
            settings.equation.viscosity = .01;
            settings.equation.rans.model = n == 6 ? "SA" : "KOmegaSST";
            settings.numerics.ransState.referenceConservative = n == 6
                                                                    ? std::map<std::string, real>{{"rho", 1.1}, {"rhoNuTilde", .02}}
                                                                    : std::map<std::string, real>{{"rho", 1.1}, {"rhoK", .02}, {"rhoOmega", 20}};
        }
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume cpu(mesh, settings, equation);
        const I rows = cpu.Metrics().NumCell();
        R::Array<real> initial(R::RowLayout::Dense(rows, n)), residual(initial.Layout()), constant(initial.Layout()), increment(initial.Layout());
        R::Array<real> spectral(R::RowLayout::Dense(rows, 1));
        {
            auto u = initial.Write(), r = residual.Write(), b = constant.Write(), d = increment.Write(), s = spectral.Write();
            for (I iCell = 0; iCell < rows; ++iCell)
            {
                const auto flow = equation.Gas().Conservative({1.1, .2, -.1, .02, 1.2});
                for (I j = 0; j < n; ++j)
                {
                    u.View()(iCell, j) = j < 5 ? flow[j] : j == 5 ? .02
                                                                  : 20;
                    r.View()(iCell, j) = (j + 1) * .001;
                    b.View()(iCell, j) = .1 * u.View()(iCell, j);
                    d.View()(iCell, j) = j == 4 ? (iCell % 2 ? -20 : .1) : j < 5 ? 0
                                                                                 : -100;
                }
                s.View()(iCell, 0) = 2;
            }
        }
        std::vector<R::Array<real>> expected;
        std::vector<real> expectedNorm, expectedScale;
        C::AlphaStatistics expectedAlpha;
        C::NonlinearCorrectionResult expectedCorrection;
        for (const int backend : {0, 1, 2})
        {
            CAPTURE(backend);
            const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
            const R::ExecutionContext copy{device, nullptr};
            const auto plan = backend == 2 ? cpu.Placed(device, copy) : cpu;
            R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
            const auto lane = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
            auto u = C::EquationDofs::Coupled(equation.Schema(), plan.Metrics().BuildField(n));
            auto r = C::EquationDofOperations::NewVector(u), b = C::EquationDofOperations::NewVector(u), d = C::EquationDofOperations::NewVector(u);
            auto defect = C::EquationDofOperations::NewVector(u), candidate = C::EquationDofOperations::NewVector(u);
            auto rate = C::EquationDofOperations::NewVector(u), recovered = C::EquationDofOperations::NewVector(u);
            u.CoupledField().Owned().CopyFrom(initial, copy).Wait();
            r.CoupledField().Owned().CopyFrom(residual, copy).Wait();
            b.CoupledField().Owned().CopyFrom(constant, copy).Wait();
            d.CoupledField().Owned().CopyFrom(increment, copy).Wait();
            auto spectrum = spectral.To(device, copy);
            R::Array<real> inverse(spectral.Layout(), device), alpha(spectral.Layout(), device);
            C::NonlinearOperators operators(plan.Metrics(), plan.PrepareAdmissibility(), lane.Context());
            operators.ValidateMeans(engine, lane, u);
            const auto norm = operators.AssembleDefect(engine, lane, u, r, b, defect, .2, .75, {}, 2, true);
            const auto scale = operators.MeasureMeanState(engine, lane, u);
            const auto prepared = operators.PreparePseudoTime(engine, lane, u, defect, spectrum, inverse, std::ref(alpha),
                                                              {2, 1, .9, 2, true, false});
            // The agreement is explicit controller work, shared by backends.
            const auto agreed = C::AgreeAlphaStatistics(prepared.limited, prepared.minimum, C::AlphaReduction::MinimumAll, mpi.comm);
            operators.SubmitAgreedRHSAlpha(engine, lane, alpha, inverse, defect, agreed.minimum, C::AlphaReduction::MinimumAll, false);
            const auto correction = operators.ApplyCorrection(engine, lane, u, d, candidate, {}, {1, .9, 2, true});
            operators.ValidateMeans(engine, lane, candidate);
            operators.SubmitScaleRate(engine, lane, r, rate, false);
            operators.SubmitScaleRate(engine, lane, rate, recovered, true);
            engine.Synchronize();
            Compare(residual, recovered.CoupledField().Owned().To(R::Device::Host()));
            const auto fields = std::vector<R::Array<real>>{
                defect.CoupledField().Owned().To(R::Device::Host()), inverse.To(R::Device::Host()),
                alpha.To(R::Device::Host()), candidate.CoupledField().Owned().To(R::Device::Host())};
            if (backend == 0)
            {
                for (const auto &field : fields)
                    expected.push_back(field.Clone());
                expectedNorm = norm;
                expectedScale = scale;
                expectedAlpha = prepared;
                expectedCorrection = correction;
                const auto read = initial.Read(), rhs = residual.Read(), source = constant.Read();
                real analytical = 0;
                for (I iCell = 0; iCell < rows; ++iCell)
                {
                    const real value = .75 * rhs.View()(iCell, 0) + (source.View()(iCell, 0) - .2 * read.View()(iCell, 0)) / 27;
                    analytical += value * value * 27;
                }
                CHECK(norm[0] == doctest::Approx(analytical).epsilon(1e-12));
                CHECK(correction.alpha.limited == rows / 2);
                CHECK(correction.frozen);
            }
            else
            {
                for (I j = 0; j <= n; ++j)
                {
                    CHECK(norm[j] == doctest::Approx(expectedNorm[j]).epsilon(1e-11).scale(1));
                    CHECK(scale[j] == doctest::Approx(expectedScale[j]).epsilon(1e-11).scale(1));
                }
                CHECK(prepared.limited == expectedAlpha.limited);
                CHECK(prepared.minimum == doctest::Approx(expectedAlpha.minimum).epsilon(1e-11));
                CHECK(correction.alpha.limited == expectedCorrection.alpha.limited);
                CHECK(correction.alpha.minimum == doctest::Approx(expectedCorrection.alpha.minimum).epsilon(1e-11));
                CHECK(correction.frozen == expectedCorrection.frozen);
                for (I i = 0; i < I(fields.size()); ++i)
                    Compare(expected[i], fields[i]);
            }
            CHECK_THROWS(operators.AssembleDefect(engine, lane, u, r, b, u, .2, 1, {}, 1, true));
        }
    }
}

TEST_CASE("Explicit stage algebra preserves conservative flow and independent RANS repair")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 3);
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
    for (const int n : {5, 6, 7})
    {
        CAPTURE(n);
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.boundaries["7"].kind = "extrapolate";
        if (n > 5)
        {
            settings.equation.transport = "NewtonianFourier";
            settings.equation.viscosity = .01;
            settings.equation.rans.model = n == 6 ? "SA" : "KOmegaSST";
            settings.numerics.ransState.referenceConservative = n == 6
                                                                    ? std::map<std::string, real>{{"rho", 1}, {"rhoNuTilde", .02}}
                                                                    : std::map<std::string, real>{{"rho", 1}, {"rhoK", .02}, {"rhoOmega", 20}};
        }
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume cpu(mesh, settings, equation);
        auto seed = C::EquationDofs::Coupled(equation.Schema(), cpu.Metrics().BuildField(n));
        auto residual = C::EquationDofOperations::NewVector(seed), low = C::EquationDofOperations::NewVector(seed);
        {
            auto u = seed.CoupledField().Owned().Write(), r = residual.CoupledField().Owned().Write(), l = low.CoupledField().Owned().Write();
            for (I iCell = 0; iCell < seed.Space().OwnedRows(); ++iCell)
                for (I j = 0; j < n; ++j)
                {
                    u.View()(iCell, j) = j == 0 ? 1 : j == 4 ? 3
                                                  : j == 5   ? .02
                                                  : j == 6   ? 20
                                                             : 0;
                    r.View()(iCell, j) = j == 4 ? (n == 5 ? -10 : .1) : j >= 5 ? -100
                                                                               : 0;
                    l.View()(iCell, j) = 0;
                }
        }
        R::Array<real> expected;
        C::ExplicitBlendStatistics expectedBlend;
        std::array<real, 14> expectedDiagnostics{};
        const C::ExplicitStageCoefficients coefficients{.1, .25, .75};
        const auto cpuPolicy = cpu.PrepareAdmissibility();
        const auto referenceLimit = cpu.LimitPhysicalStep(seed, residual, cpuPolicy.View().thermal, .1, .9, 1, 2);
        R::Array<real> spectral(R::RowLayout::Dense(cpu.Metrics().NumCell(), 1));
        {
            auto write = spectral.Write();
            for (I iCell = 0; iCell < spectral.RowCount(); ++iCell)
                write.View()(iCell, 0) = 1;
        }
        for (const int backend : {0, 1, 2})
        {
            CAPTURE(backend);
            const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
            const R::ExecutionContext copy{device, nullptr};
            auto plan = backend == 2 ? cpu.Placed(device, copy) : cpu;
            const auto placed = [&](const C::EquationDofs &source)
            {
                auto field = plan.Metrics().BuildField(n);
                field.Owned().CopyFrom(source.CoupledField().Owned(), copy).Wait();
                return C::EquationDofs::Coupled(equation.Schema(), std::move(field));
            };
            auto initial = placed(seed), stage = placed(seed), rhs = placed(residual), lowRHS = placed(low);
            R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
            const auto lane = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
            C::TimeStageOperators operators(plan.Metrics(), lane.Context());
            C::MeanStateCheck check(plan.Metrics().Space(), plan.PrepareAdmissibility(), lane.Context());
            CHECK(bool(check.Evaluate(engine, lane, stage)));
            C::StateDiagnostics diagnostics(plan.Metrics(), equation, lane.Context());
            const auto observed = diagnostics.Evaluate(engine, lane, initial, rhs);
            if (backend == 0)
                expectedDiagnostics = observed;
            for (int j = 0; j < 14; ++j)
                CHECK(observed[j] == doctest::Approx(expectedDiagnostics[j]).epsilon(3e-12).scale(1));
            CHECK(observed[0] == doctest::Approx(real(initial.Space().OwnedRows()) / 27).epsilon(3e-12));
            CHECK(observed[1] == doctest::Approx(3 * observed[0]).epsilon(3e-12));
            const auto repeatedDiagnostics = diagnostics.Evaluate(engine, lane, initial, rhs);
            CHECK(repeatedDiagnostics == observed);
            const auto residentSpectral = spectral.To(device, copy);
            CHECK(operators.MinimumStableStep(engine, lane, residentSpectral, 2) == doctest::Approx(2. / 27).epsilon(1e-12));
            const auto localLimit = operators.LimitPhysicalStep(engine, lane, initial, rhs, plan.PrepareAdmissibility(), .1, .9, 1, 2);
            const auto limited = DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Sum>(localLimit.limitedCells, mpi.comm);
            const real minimum = DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Min>(localLimit.minimumAlpha, mpi.comm);
            CHECK(limited == referenceLimit.limitedCells);
            CHECK(minimum == doctest::Approx(referenceLimit.minimumAlpha).epsilon(1e-12));
            CHECK((limited ? localLimit.dt * minimum : localLimit.dt) == doctest::Approx(referenceLimit.dt).epsilon(1e-12));
            if (n > 5)
                CHECK(minimum == 1); // RANS's wild trial is not a thermal restriction.
            if (n == 5)
            {
                auto blend = operators.MeasureBlend(engine, lane, initial, stage, rhs, lowRHS, plan.ThermalConstraints(), coefficients);
                CHECK(blend.safe[0] == 0);
                CHECK(blend.safe[1] == 1);
                CHECK(blend.alpha > 0);
                CHECK(blend.alpha < 1);
                CHECK(blend.limited == seed.Space().OwnedRows());
                // A physically rejected candidate is a status, not an
                // exception used for distributed recovery. Reset explicitly
                // before testing the conservative low/high blend.
                operators.UpdateFlow(engine, lane, initial, rhs, stage, {}, 1, coefficients);
                CHECK(check.Evaluate(engine, lane, stage).code == DNDS::NumericalError::InvalidState);
                C::EquationDofOperations::Copy(engine, lane, stage, initial);
                if (backend == 0)
                    expectedBlend = blend;
                else
                    CHECK(blend.alpha == doctest::Approx(expectedBlend.alpha).epsilon(1e-12));
                const real alpha = DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Min>(blend.alpha, mpi.comm);
                operators.UpdateFlow(engine, lane, initial, rhs, stage, std::cref(lowRHS), alpha, coefficients);
            }
            else
            {
                const auto policy = plan.PrepareAdmissibility();
                operators.UpdateExtended(engine, lane, initial, rhs, stage, policy.RANS(), coefficients);
                const auto host = stage.CoupledField().Owned().To(R::Device::Host());
                const auto read = host.Read();
                for (I iCell = 0; iCell < host.RowCount(); ++iCell)
                {
                    // Independent conservative formula: huge RANS decrease
                    // cannot shrink this energy increment (volume=1/27).
                    CHECK(read.View()(iCell, 4) == doctest::Approx(3 + .75 * .1 * .1 * 27).epsilon(1e-12));
                    CHECK(read.View()(iCell, 5) >= 0);
                }
            }
            const auto result = stage.CoupledField().Owned().To(R::Device::Host());
            CHECK(bool(check.Evaluate(engine, lane, stage)));
            if (backend == 0)
                expected = result.Clone();
            else
                Compare(expected, result);
        }
    }
}

TEST_CASE("Visualization field preparation is resident and matches independent perfect-gas values")
{
    for (const int components : {5, 6, 7})
    {
        C::DomainSettings settings;
        if (components > 5)
        {
            settings.equation.transport = "NewtonianFourier";
            settings.equation.viscosity = .01;
            settings.equation.rans.model = components == 6 ? "SA" : "KOmegaSST";
        }
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        for (const I rows : {I(0), I(257)})
        {
            const DNDS::Algebra::DofSpace space(size_t(rows), 0, "output-test");
            auto initial = C::EquationDofs::Coupled(equation.Schema(),
                                                    DNDS::Algebra::DofField<real>(space, R::RowLayout::Dense(0, components)), "equation");
            {
                auto write = initial.CoupledField().Owned().Write();
                for (I iRow = 0; iRow < rows; ++iRow)
                {
                    const auto point = equation.Gas().Conservative({1 + .001 * iRow, .2, -.3, .1, 1.7});
                    for (I j = 0; j < components; ++j)
                        write.View()(iRow, j) = j < 5 ? point[j] : .02 * j;
                }
            }
            R::Array<real> anchor;
            for (int backend = 0; backend < 3; ++backend)
            {
                CAPTURE(components);
                CAPTURE(rows);
                CAPTURE(backend);
                const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
                const R::ExecutionContext placement{device, nullptr};
                auto state = C::EquationDofOperations::CopyOwnedTo(initial, device, placement);
                R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
                const auto lane = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
                const auto prepared = C::OutputFields::Submit(engine, lane, state, equation);
                engine.Synchronize();
                const auto host = prepared.CopyToHost(lane.Context());
                if (backend == 0)
                    anchor = host;
                else
                    Compare(anchor, host);
                const auto read = host.Read();
                for (I iRow = 0; iRow < rows; ++iRow)
                {
                    const real density = 1 + .001 * iRow;
                    CHECK(read.View()(iRow, components) == doctest::Approx(1.7));
                    CHECK(read.View()(iRow, components + 1) == doctest::Approx(1.7 / density / equation.Gas().gasConstant));
                    CHECK(read.View()(iRow, components + 2) == doctest::Approx(.2));
                    CHECK(read.View()(iRow, components + 3) == doctest::Approx(-.3));
                    CHECK(read.View()(iRow, components + 5) == doctest::Approx(std::sqrt(.14 * density / (equation.Gas().gamma * 1.7))));
                    if (components > 5)
                        CHECK(read.View()(iRow, 5) == doctest::Approx(.1));
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    mpi = DNDS::MPIInfo(MPI_COMM_WORLD);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
