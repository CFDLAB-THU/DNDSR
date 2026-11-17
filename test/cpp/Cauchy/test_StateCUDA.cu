#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "Geom/Quadratures/Quadrature.hpp"

namespace C = DNDS::Cauchy;
namespace K = C::CUDA;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using DNDS::real;
static DNDS::MPIInfo mpi;

TEST_CASE("Queued limiter statistics keep integer counts and deterministic minima")
{
    const auto device = R::Device::CUDA(0);
    for (const I rows : {I(0), I(1025)})
    {
        R::Array<real> host(R::RowLayout::Dense(rows, 1));
        {
            auto write = host.Write();
            for (I iRow = 0; iRow < rows; ++iRow)
                write.View()(iRow, 0) = .5 * (iRow % 3);
        }
        auto beta = host.To(device);
        const I partials = K::StateReductionScratchRows(rows);
        R::Array<I> counts(R::RowLayout::Dense(partials, 1), device), count(R::RowLayout::Dense(1, 1), device);
        R::Array<real> minima(R::RowLayout::Dense(partials, 1), device), minimum(R::RowLayout::Dense(1, 1), device);
        R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto beta, auto counts, auto minima, auto count, auto minimum)
                        { K::ReduceLimiterStatistics(context, beta, counts, minima, count, minimum); }, R::ReadAs<R::TableRows>(beta), R::WriteAs<R::FixedRows<1>>(counts), R::WriteAs<R::FixedRows<1>>(minima), R::WriteAs<R::FixedRows<1>>(count), R::WriteAs<R::FixedRows<1>>(minimum));
        engine.Synchronize();
        const auto hostCount = count.To(R::Device::Host());
        const auto hostMinimum = minimum.To(R::Device::Host());
        CHECK(hostCount.Read().View()(0, 0) == (rows ? I(684) : I(0)));
        CHECK(hostMinimum.Read().View()(0, 0) == (rows ? 0. : 1.));
    }
}

TEST_CASE("Queued status reduction preserves earlier failures across reused producer buffers")
{
    const auto device = R::Device::CUDA(0);
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    for (const I rows : {I(0), I(1031)})
    {
        R::Array<DNDS::NumericalStatus> host(R::RowLayout::Dense(rows, 1));
        const auto failure = DNDS::NumericalStatus{DNDS::NumericalError::InvalidState, 7}.At(3, 2);
        {
            auto write = host.Write();
            for (I iRow = 0; iRow < rows; ++iRow)
                write.View()(iRow, 0) = {};
            if (rows)
            {
                write.View()(0, 0) = DNDS::NumericalStatus{DNDS::NumericalError::InvalidInput, 4}.At(8, 1);
                write.View()(rows - 1, 0) = failure;
            }
        }
        auto input = host.To(device);
        R::Array<DNDS::NumericalStatus> accumulator(R::RowLayout::Dense(1, 1), device);
        R::Array<DNDS::NumericalStatus> partial(R::RowLayout::Dense(K::StateReductionScratchRows(rows + 1), 1), device);
        R::SubmitArrays(engine, stream, [](auto context, auto result)
                        { K::ClearNumericalStatus(context, result); }, R::WriteAs<R::FixedRows<1>>(accumulator));
        const auto append = [&]
        {
            R::SubmitArrays(engine, stream, [](auto context, auto values, auto scratch, auto result)
                            { K::ReduceNumericalStatus(context, values, result.ConstView(), scratch, result); }, R::ReadAs<R::FixedRows<1>>(input), R::WriteAs<R::FixedRows<1>>(partial), R::ReadWriteAs<R::FixedRows<1>>(accumulator));
        };
        append();
        // A later successful producer must not erase an earlier failure in
        // the same queued segment. No host observation between submissions.
        R::SubmitArrays(engine, stream, [](auto context, auto values)
                        { K::ClearNumericalStatus(context, values); }, R::WriteAs<R::FixedRows<1>>(input));
        append();
        engine.Synchronize();
        const auto result = accumulator.To(R::Device::Host());
        const auto read = result.Read();
        const auto status = read.View()(0, 0);
        CHECK(bool(status) == (rows == 0));
        if (rows)
        {
            CHECK(status.code == failure.code);
            CHECK(status.detail == failure.detail);
            CHECK(status.row == failure.row);
            CHECK(status.point == failure.point);
        }
    }
}

TEST_CASE("Spatial placement retains metric identity and allocates VR boundary values without host staging")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    for (const std::string family : {"constant", "gradient", "variational"})
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.numerics.reconstruction = family;
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.SORInstead = false;
        settings.numerics.positivity = false;
        settings.numerics.limiter = "none";
        settings.boundaries["7"].kind = "extrapolate";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        std::optional<C::FiniteVolume> host{std::in_place, mesh, settings, equation};
        auto resident = host->Placed(device, context);
        CHECK(resident.Metrics().Space().Same(host->Metrics().Space()));
        CHECK(resident.Device() == device);
        CHECK(bool(resident.PolynomialReconstruction()) == (family == "variational"));
        host.reset();
        auto metric = resident.Metrics().AcquireRead(context);
        metric.View().ValidateDevice(device);
        if (const auto reconstruction = resident.PolynomialReconstruction())
        {
            CHECK_FALSE(reconstruction->RetainsHostPayloads());
            auto boundary = reconstruction->BuildBoundaryValues(5);
            CHECK(boundary.values.GetDevice() == device);
            CHECK(boundary.values.RowCount() == reconstruction->NumBoundaryQuadrature());
            CHECK(boundary.values.Layout().Width() == 5);
            auto coefficients = reconstruction->BuildCoefficients(5);
            // Native synchronous algebra dispatches to the array's device.
            DNDS::Algebra::Fill(coefficients, 0.);
            CHECK(DNDS::Algebra::DotLocal(coefficients, coefficients) == 0);
        }
    }
}

TEST_CASE("Queued boundary operator retains P3 basis and matches CPU physical and mirrored targets")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    for (const bool cached : {true, false})
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .003;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.SORInstead = false;
        settings.numerics.cfv.cacheDiffBase = cached;
        settings.numerics.limiter = "none";
        settings.numerics.positivity = false;
        settings.numerics.viscousGradient = "primitive";
        settings.numerics.vrBoundaryValue = cached ? "physicalMirrored" : "physicalTrace";
        settings.boundaries["7"].kind = "noSlipWall";
        settings.motion.angularVelocity = {0, 0, .2};
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume cpu(mesh, settings, equation);
        auto gpu = cpu.Placed(device, context);
        auto mean = C::EquationDofs::Coupled(equation.Schema(), cpu.Metrics().BuildField(5));
        cpu.InitializeConstant(mean, {1.1, .2, -.1, .05, 1.2});
        auto resident = C::EquationDofs::Coupled(equation.Schema(), gpu.Metrics().BuildField(5));
        resident.CoupledField().Owned().CopyFrom(mean.CoupledField().Owned(), context).Wait();
        auto cpuRec = cpu.BuildReconstructionState(mean), gpuRec = gpu.BuildReconstructionState(resident);
        {
            auto write = cpuRec->WarmValues().CoupledField().Owned().Write();
            const auto view = write.View<R::TableRows>();
            for (I iCell = 0; iCell < view.RowCount(); ++iCell)
                for (I j = 0; j < view.Width(); ++j)
                    view(iCell, j) = 1e-5 * (1 + j % 7);
        }
        gpuRec->WarmValues().CoupledField().Owned().CopyFrom(cpuRec->WarmValues().CoupledField().Owned(), context).Wait();
        auto cpuWork = cpu.BuildWorkspace(mean), gpuWork = gpu.BuildWorkspace(resident);
        auto cpuBinding = cpuWork->BindReconstruction(*cpuRec), gpuBinding = gpuWork->BindReconstruction(*gpuRec);
        {
            R::EagerEngine hostEngine;
            const auto hostStream = hostEngine.CreateCPUStream();
            cpuWork->SubmitBoundary(hostEngine, hostStream, mean, true, .4);
            hostEngine.Synchronize();
        }
        R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        gpuWork->ResetPreparationStatus(engine, stream);
        // Repeated VR updates reuse stable scratch. Time/frame are captured
        // per launch; immutable policy metadata is never rewritten between them.
        gpuWork->SubmitDeviceBoundary(engine, stream, resident, true, .2);
        gpuWork->SubmitDeviceBoundary(engine, stream, resident, true, .4);
        engine.Synchronize();
        gpuWork->CheckPreparationStatus();
        const auto compare = [](const auto &expected, const auto &actual)
        {
            const auto host = actual.To(R::Device::Host());
            const auto x = expected.Read(), y = host.Read();
            for (I iRow = 0; iRow < expected.RowCount(); ++iRow)
                for (I j = 0; j < expected.Layout().Width(); ++j)
                    CHECK(y.View()(iRow, j) == doctest::Approx(x.View()(iRow, j)).epsilon(3e-12).scale(1));
        };
        compare(cpuWork->fields.front().boundaryValues, gpuWork->fields.front().boundaryValues);
        REQUIRE(cpuWork->primitiveBoundary);
        REQUIRE(gpuWork->primitiveBoundary);
        compare(*cpuWork->primitiveBoundary, *gpuWork->primitiveBoundary);
        for (const int workers : {0, 2})
        {
            auto queuedWork = cpu.BuildWorkspace(mean);
            auto binding = queuedWork->BindReconstruction(*cpuRec);
            R::EagerEngine hostEngine(workers, std::make_shared<R::AccessTrackingContext>());
            const auto hostStream = hostEngine.CreateCPUStream();
            auto pointEvaluator = std::make_shared<C::PointEvaluator>(equation.MakePointEvaluator());
            queuedWork->SubmitBoundary(hostEngine, hostStream, mean, true, .2, pointEvaluator);
            queuedWork->SubmitBoundary(hostEngine, hostStream, mean, true, .4, pointEvaluator);
            // Consume a boundary buffer before draining: the producer must use
            // scheduled access, not acquire an immediate write on the worker.
            R::Array<double> copied(queuedWork->fields.front().boundaryValues.Layout());
            R::SubmitArrays(hostEngine, hostStream, [](R::ExecutionContext, auto input, auto output)
                            {
                for (I iRow = 0; iRow < output.RowCount(); ++iRow)
                    for (I j = 0; j < output.Width(); ++j) output(iRow, j) = input(iRow, j); }, R::ReadAs<R::TableRows>(queuedWork->fields.front().boundaryValues), R::WriteAs<R::TableRows>(copied));
            hostEngine.Synchronize();
            compare(cpuWork->fields.front().boundaryValues, copied);
            compare(*cpuWork->primitiveBoundary, *queuedWork->primitiveBoundary);
        }
    }
}

TEST_CASE("Managed CUDA volume sources retain distributed geometry and match CPU sampling")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 3);
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    for (const bool polynomial : {false, true})
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .003;
        settings.equation.rans.model = polynomial ? "KOmegaSST" : "SA";
        settings.equation.rans.viscosityReference = .01;
        settings.numerics.reconstruction = polynomial ? "variational" : "constant";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.SORInstead = false;
        settings.numerics.limiter = "none";
        settings.numerics.positivity = false;
        settings.numerics.viscousGradient = "polynomial";
        settings.boundaries["7"].kind = "extrapolate";
        settings.motion.angularVelocity = {0, 0, .2};
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        const I width = 5 + equation.RANS().ComponentNumber();
        C::FiniteVolume cpu(mesh, settings, equation);
        auto gpu = cpu.Placed(device, context);
        auto state = C::EquationDofs::Coupled(equation.Schema(), cpu.Metrics().BuildField(width));
        auto resident = C::EquationDofs::Coupled(equation.Schema(), gpu.Metrics().BuildField(width));
        {
            auto write = state.CoupledField().Owned().Write();
            const auto values = write.View<R::TableRows>();
            const auto flow = equation.Gas().Conservative({1.1, .2, -.1, .03, 1.2});
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                for (I j = 0; j < width; ++j)
                    values(iCell, j) = j < 5 ? flow[j] : j == 5 ? .02
                                                                : 20.;
        }
        resident.CoupledField().Owned().CopyFrom(state.CoupledField().Owned(), context).Wait();
        auto cpuRec = cpu.BuildReconstructionState(state), gpuRec = gpu.BuildReconstructionState(resident);
        auto cpuWork = cpu.BuildWorkspace(state), gpuWork = gpu.BuildWorkspace(resident);
        auto cpuBinding = cpuWork->BindReconstruction(*cpuRec), gpuBinding = gpuWork->BindReconstruction(*gpuRec);
        if (polynomial)
        {
            {
                auto write = cpuRec->WarmValues().CoupledField().Owned().Write();
                const auto values = write.View<R::TableRows>();
                for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                    for (I j = 0; j < values.Width(); ++j)
                        values(iCell, j) = 1e-6 * (1 + j % 7);
            }
            gpuRec->WarmValues().CoupledField().Owned().CopyFrom(cpuRec->WarmValues().CoupledField().Owned(), context).Wait();
        }
        else
        {
            CHECK(cpuWork->directGradients.front().Owned().SharedStorage() == cpuWork->fields.front().gradient->Owned().SharedStorage());
            CHECK(gpuWork->directGradients.front().Owned().SharedStorage() == gpuWork->fields.front().gradient->Owned().SharedStorage());
            auto write = cpuWork->fields.front().gradient->Owned().Write();
            const auto values = write.View<R::TableRows>();
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                for (I j = 0; j < values.Width(); ++j)
                    values(iCell, j) = 1e-3 * (1 + j % 5);
        }
        auto expected = C::EquationDofOperations::NewVector(state);
        auto actual = C::EquationDofOperations::NewVector(resident);
        auto expectedLow = C::EquationDofOperations::NewVector(state);
        auto actualLow = C::EquationDofOperations::NewVector(resident);
        DNDS::Algebra::Fill(expected.CoupledField(), .125);
        DNDS::Algebra::Fill(actual.CoupledField(), .125);
        DNDS::Algebra::Fill(expectedLow.CoupledField(), .75);
        DNDS::Algebra::Fill(actualLow.CoupledField(), .75);
        auto evaluator = equation.MakePointEvaluator({}, {true});
        cpuWork->PrepareVolumeSource(state, std::ref(expected), evaluator, .4, std::ref(expectedLow));
        if (!polynomial)
            gpuWork->directGradients.front().Owned().CopyFrom(cpuWork->directGradients.front().Owned(), context).Wait();
        R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        gpuWork->ResetPreparationStatus(engine, stream);
        gpuWork->SubmitVolumeSource(engine, stream, resident, std::ref(actual), .4, {}, std::ref(actualLow));
        engine.Synchronize();
        gpuWork->CheckPreparationStatus();
        const auto compare = [&](const C::EquationDofs &reference, const C::EquationDofs &value)
        {
            const auto copied = value.CoupledField().Owned().To(R::Device::Host());
            const auto x = reference.CoupledField().Owned().Read(), y = copied.Read();
            for (I iCell = 0; iCell < copied.RowCount(); ++iCell)
                for (I j = 0; j < width; ++j)
                    CHECK(y.View()(iCell, j) == doctest::Approx(x.View()(iCell, j)).epsilon(3e-10).scale(1));
        };
        compare(expected, actual);
        compare(expectedLow, actualLow);
        // The same source submission retains its evaluator and declares
        // output accesses for inline and worker CPU execution. Low-order
        // residuals receive rotation only, exactly as the synchronous API.
        for (const int workers : {0, 1})
        {
            auto queued = C::EquationDofOperations::NewVector(state);
            auto queuedLow = C::EquationDofOperations::NewVector(state);
            DNDS::Algebra::Fill(queued.CoupledField(), .125);
            DNDS::Algebra::Fill(queuedLow.CoupledField(), .75);
            auto queuedWork = cpu.BuildWorkspace(state);
            auto binding = queuedWork->BindReconstruction(*cpuRec);
            if (!polynomial)
                queuedWork->directGradients.front().Owned().CopyFrom(cpuWork->directGradients.front().Owned()).Wait();
            R::EagerEngine hostEngine(workers, std::make_shared<R::AccessTrackingContext>());
            const auto hostStream = hostEngine.CreateCPUStream();
            queuedWork->SubmitVolumeSource(hostEngine, hostStream, state, std::ref(queued), .4,
                                           workers == 0 ? std::make_shared<C::PointEvaluator>(equation.MakePointEvaluator({}, {true}))
                                                        : std::shared_ptr<C::PointEvaluator>{},
                                           std::ref(queuedLow));
            hostEngine.Synchronize();
            compare(expected, queued);
            compare(expectedLow, queuedLow);
        }
    }
}

TEST_CASE("SST mean-gradient auxiliary schedule is shared by inline worker and CUDA execution")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 3);
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
    for (const bool overlap : {false, true})
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.execution.overlap = overlap;
        settings.numerics.reconstruction = "constant";
        settings.numerics.limiter = "none";
        settings.numerics.positivity = false;
        settings.numerics.viscousGradient = "polynomial";
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .003;
        settings.equation.rans.model = "KOmegaSST";
        settings.equation.rans.viscosityReference = .01;
        settings.boundaries["7"].kind = "extrapolate";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume cpu(mesh, settings, equation);
        const auto device = R::Device::CUDA(0);
        const auto context = R::ExecutionContext::CUDA(0);
        auto gpu = cpu.Placed(device, context);
        auto means = C::EquationDofs::Coupled(equation.Schema(), cpu.Metrics().BuildField(7));
        {
            const auto geometry = cpu.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto write = means.CoupledField().Owned().Write();
            const auto values = write.View<R::TableRows>();
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const auto x = metric.GetCellCent(iCell);
                const auto flow = equation.Gas().Conservative({1.1, .2 + .03 * x[1], -.1, .02, 1.2});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = flow[j];
                values(iCell, 5) = .02 + .001 * x[0];
                values(iCell, 6) = 20 + .1 * x[1];
            }
        }
        auto resident = C::EquationDofs::Coupled(equation.Schema(), gpu.Metrics().BuildField(7));
        resident.CoupledField().Owned().CopyFrom(means.CoupledField().Owned(), context).Wait();
        C::AuxiliaryRequest request;
        request.meanThermodynamics = request.faceSpectralRadius = request.cellSpectralSum = true;
        auto reference = cpu.BuildAuxiliaryValues(request);
        for (const int backend : {0, 1, 2})
        {
            auto &plan = backend == 2 ? gpu : cpu;
            auto &state = backend == 2 ? resident : means;
            auto gradient = plan.BuildMeanGradients(state);
            auto result = plan.BuildAuxiliaryValues(request);
            R::EagerEngine engine(backend == 1 ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
            const auto compute = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
            const auto communication = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
            plan.PrepareAuxValues(state, result, gradient, engine, compute, communication, .3, {}, {});
            plan.ValidateAuxValues(state, result, .3);
            if (backend == 0)
                reference = result;
            else
            {
                const auto compare = [](const auto &expected, const auto &actual)
                {
                    const auto host = actual.To(R::Device::Host());
                    const auto x = expected.Read(), y = host.Read();
                    for (I iRow = 0; iRow < expected.RowCount(); ++iRow)
                        for (I j = 0; j < expected.Layout().Width(); ++j)
                            CHECK(y.View()(iRow, j) == doctest::Approx(x.View()(iRow, j)).epsilon(4e-11).scale(1));
                };
                compare(*reference.meanThermodynamics, *result.meanThermodynamics);
                compare(*reference.faceSpectralRadius, *result.faceSpectralRadius);
                compare(reference.cellSpectralSum->Owned(), result.cellSpectralSum->Owned());
            }
        }
    }
}

TEST_CASE("Queued auxiliary operator matches CPU on a distributed periodic mesh with optional outputs")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.limiter = "none";
    settings.numerics.positivity = false;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .003;
    settings.motion.angularVelocity = {0, 0, .2};
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume cpu(mesh, settings, equation);
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    auto gpu = cpu.Placed(device, context);
    auto means = C::EquationDofs::Coupled(equation.Schema(), cpu.Metrics().BuildField(5));
    {
        const auto geometry = cpu.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        auto write = means.CoupledField().Owned().Write();
        for (I iCell = 0; iCell < cpu.Metrics().NumCell(); ++iCell)
        {
            const auto center = metric.GetCellCent(iCell);
            const auto row = equation.Gas().Conservative({1 + .1 * center[0], .2 + .03 * center[1], -.1, .02, 1 + .07 * center[1]});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = row[j];
        }
    }
    cpu.BindHalo(means.CoupledField()).Begin().Wait();
    auto resident = C::EquationDofs::Coupled(equation.Schema(), gpu.Metrics().BuildField(5));
    resident.CoupledField().Owned().CopyFrom(means.CoupledField().Owned(), context).Wait();
    auto halo = gpu.BindHalo(resident.CoupledField());
    for (bool cacheMean : {false, true})
        for (bool faceOutputs : {false, true})
        {
            C::AuxiliaryRequest request;
            request.meanThermodynamics = cacheMean;
            request.faceSpectralRadius = request.faceWaveJump = request.facePressureMinimum = faceOutputs;
            request.cellSpectralSum = request.cellWaveJump = request.cellPressureMinimum = true;
            auto expected = cpu.BuildAuxiliaryValues(request), actual = gpu.BuildAuxiliaryValues(request);
            cpu.PrepareAuxValues(means, expected, .3);
            for (bool split : {true, false})
            {
                R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
                const auto compute = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
                const auto communication = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
                // The producer may still be queued when preparation captures
                // its input epoch; completion Version is deliberately distinct.
                DNDS::Algebra::Scale(engine, compute, resident.CoupledField(), 1.);
                auto preparation = gpu.BeginAuxValues(resident, actual, .3);
                auto transfer = halo.SubmitAfter(engine, communication, compute.Record());
                if (split)
                    preparation.Apply(engine, compute, DNDS::CFV::ReconstructionRegion::Core);
                transfer.EnqueueWait(compute);
                preparation.Apply(engine, compute, split ? DNDS::CFV::ReconstructionRegion::HaloDependent : DNDS::CFV::ReconstructionRegion::All);
                engine.Synchronize();
                preparation.Finish(actual);
                gpu.ValidateAuxValues(resident, actual, .3);
                const auto compare = [](const auto &host, const auto &deviceArray)
                {
                    const auto downloaded = deviceArray.To(R::Device::Host());
                    const auto a = host.Read(), b = downloaded.Read();
                    const auto x = a.template View<R::TableRows>(), y = b.template View<R::TableRows>();
                    REQUIRE(x.RowCount() == y.RowCount());
                    REQUIRE(x.Width() == y.Width());
                    for (I iRow = 0; iRow < x.RowCount(); ++iRow)
                        for (I j = 0; j < x.Width(); ++j)
                            CHECK(y(iRow, j) == doctest::Approx(x(iRow, j)).epsilon(3e-12).scale(1));
                };
                if (cacheMean)
                    compare(*expected.meanThermodynamics, *actual.meanThermodynamics);
                if (faceOutputs)
                {
                    compare(*expected.faceSpectralRadius, *actual.faceSpectralRadius);
                    compare(*expected.faceWaveJump, *actual.faceWaveJump);
                    compare(*expected.facePressureMinimum, *actual.facePressureMinimum);
                }
                compare(expected.cellSpectralSum->Owned(), actual.cellSpectralSum->Owned());
                compare(expected.cellWaveJump->Owned(), actual.cellWaveJump->Owned());
                compare(expected.cellPressureMinimum->Owned(), actual.cellPressureMinimum->Owned());
            }
        }
}

TEST_CASE("Managed CUDA flux matches generic CPU for P0 P1 and P3 with distributed inputs")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    for (const int degree : {0, 1, 3})
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = degree == 3 ? "variational" : degree == 1 ? "gradient"
                                                                                     : "constant";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.SORInstead = false;
        settings.numerics.cfv.cacheDiffBase = false;
        settings.numerics.limiter = degree == 1 ? "Barth" : "none";
        settings.numerics.positivity = degree == 1;
        settings.numerics.viscousGradient = degree == 1 ? "primitive" : "polynomial";
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .003;
        settings.motion.angularVelocity = {0, 0, .1};
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume cpu(mesh, settings, equation);
        auto gpu = cpu.Placed(device, context);
        auto mean = C::EquationDofs::Coupled(equation.Schema(), cpu.Metrics().BuildField(5));
        {
            const auto geometry = cpu.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto write = mean.CoupledField().Owned().Write();
            for (I iCell = 0; iCell < cpu.Metrics().NumCell(); ++iCell)
            {
                const auto x = metric.GetCellCent(iCell);
                const auto value = equation.Gas().Conservative({1 + .05 * x[0], .2 + .02 * x[1], -.1, .03, 1 + .04 * x[1]});
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = value[j];
            }
        }
        cpu.BindHalo(mean.CoupledField()).Begin().Wait();
        auto resident = C::EquationDofs::Coupled(equation.Schema(), gpu.Metrics().BuildField(5));
        resident.CoupledField().Owned().CopyFrom(mean.CoupledField().Owned(), context).Wait();
        auto cpuRec = cpu.BuildReconstructionState(mean), gpuRec = gpu.BuildReconstructionState(resident);
        if (degree == 3)
        {
            {
                auto write = cpuRec->WarmValues().CoupledField().Owned().Write();
                const auto view = write.View<R::TableRows>();
                for (I iCell = 0; iCell < view.RowCount(); ++iCell)
                    for (I j = 0; j < view.Width(); ++j)
                        view(iCell, j) = 1e-5 * (1 + j % 5);
            }
            cpu.BindHalo(cpuRec->WarmValues().CoupledField()).Begin().Wait();
            gpuRec->WarmValues().CoupledField().Owned().CopyFrom(cpuRec->WarmValues().CoupledField().Owned(), context).Wait();
        }
        auto expected = C::EquationDofOperations::NewVector(mean), actual = C::EquationDofOperations::NewVector(resident);
        auto cpuWork = cpu.BuildWorkspace(mean, {}, C::FiniteVolume::WorkspaceOptions(C::FiniteVolume::RHSKernelPolicy::Generic));
        auto gpuWork = gpu.BuildWorkspace(resident);
        auto cpuBinding = cpuWork->BindReconstruction(*cpuRec), gpuBinding = gpuWork->BindReconstruction(*gpuRec);
        // Match the production generic viscous RHS worker capabilities.
        auto evaluator = equation.MakePointEvaluator({}, {true, true, true});
        if (degree == 1)
        {
            {
                R::EagerEngine hostEngine;
                const auto hostStream = hostEngine.CreateCPUStream();
                cpuWork->SubmitBoundary(hostEngine, hostStream, mean, false, .3);
                hostEngine.Synchronize();
            }
            auto &field = cpuWork->fields.front();
            cpuWork->owner->gradient.Evaluate(mean.CoupledField(), *field.gradientBoundary, *field.gradient, {1});
            cpuWork->PreparePrimitiveGradient(mean, DNDS::CFV::ReconstructionRegion::All);
            cpuWork->PrepareDirectGradientLimiter(mean, true);
            cpuWork->ApplyDirectGradientPositivity(mean);
            cpu.BindHalo(cpuWork->directGradients.front()).Begin().Wait();
            cpu.BindHalo(*cpuWork->primitiveGradient).Begin().Wait();
            for (const int workers : {0, 1})
            {
                auto queued = cpu.BuildWorkspace(mean);
                queued->fields.front().gradient->Owned().CopyFrom(field.gradient->Owned()).Wait();
                R::EagerEngine hostEngine(workers, std::make_shared<R::AccessTrackingContext>());
                const auto hostStream = hostEngine.CreateCPUStream();
                queued->SubmitGradientLimiter(hostEngine, hostStream, mean, true);
                const auto pp = queued->SubmitGradientPositivity(hostEngine, hostStream, mean);
                hostEngine.Synchronize();
                queued->PublishGradientPositivity(pp);
                CHECK(queued->directPPLimited == cpuWork->directPPLimited);
                CHECK(queued->directPPMinimum == doctest::Approx(cpuWork->directPPMinimum));
                const auto x = cpuWork->directGradients.front().Owned().Read();
                const auto y = queued->directGradients.front().Owned().Read();
                for (I iCell = 0; iCell < cpu.Metrics().NumCell(); ++iCell)
                    for (I j = 0; j < cpuWork->directGradients.front().Width(); ++j)
                        CHECK(y.View()(iCell, j) == doctest::Approx(x.View()(iCell, j)).epsilon(3e-12).scale(1));
            }
        }
        cpuWork->ApplyRHS(mean, expected, DNDS::CFV::ReconstructionRegion::All, evaluator, .3);
        if (degree == 3)
            for (const auto policy : {C::FiniteVolume::RHSKernelPolicy::Generic, C::FiniteVolume::RHSKernelPolicy::FixedFlow5})
                for (const int workers : {0, 2})
                {
                    auto queuedWork = cpu.BuildWorkspace(mean, {}, C::FiniteVolume::WorkspaceOptions(policy));
                    auto binding = queuedWork->BindReconstruction(*cpuRec);
                    auto reference = C::EquationDofOperations::NewVector(mean);
                    auto queued = C::EquationDofOperations::NewVector(mean);
                    auto pointEvaluator = std::make_shared<C::PointEvaluator>(equation.MakePointEvaluator({}, {true, true, true}));
                    queuedWork->ApplyRHS(mean, reference, DNDS::CFV::ReconstructionRegion::All, *pointEvaluator, .3);
                    queuedWork->diagnostics.Reset();
                    queuedWork->faceFallbacks = {};
                    R::EagerEngine hostEngine(workers, std::make_shared<R::AccessTrackingContext>());
                    const auto compute = hostEngine.CreateCPUStream(), communication = hostEngine.CreateCPUStream();
                    auto meanBinding = cpu.BindHalo(mean.CoupledField());
                    auto recBinding = cpu.BindHalo(cpuRec->WarmValues().CoupledField());
                    auto meanTransfer = meanBinding.SubmitAfter(hostEngine, communication, compute.Record());
                    auto recTransfer = recBinding.SubmitAfter(hostEngine, communication, compute.Record());
                    const auto core = queuedWork->SubmitRHS(hostEngine, compute, mean, queued,
                                                            DNDS::CFV::ReconstructionRegion::Core, .3, pointEvaluator);
                    meanTransfer.EnqueueWait(compute);
                    recTransfer.EnqueueWait(compute);
                    const auto dependent = queuedWork->SubmitRHS(hostEngine, compute, mean, queued,
                                                                 DNDS::CFV::ReconstructionRegion::HaloDependent, .3, pointEvaluator);
                    hostEngine.Synchronize();
                    queuedWork->PublishRHS(core);
                    queuedWork->PublishRHS(dependent);
                    const auto x = reference.CoupledField().Owned().Read(), y = queued.CoupledField().Owned().Read();
                    for (I iCell = 0; iCell < cpu.Metrics().NumCell(); ++iCell)
                        for (int j = 0; j < 5; ++j)
                            CHECK(y.View()(iCell, j) == doctest::Approx(x.View()(iCell, j)).epsilon(3e-12).scale(1));
                }
        auto meanHalo = gpu.BindHalo(resident.CoupledField());
        std::optional<R::TransferBinding> coefficientHalo;
        if (degree == 3)
            coefficientHalo.emplace(gpu.BindHalo(gpuRec->WarmValues().CoupledField()));
        std::optional<R::TransferBinding> gradientHalo, primitiveHalo;
        if (degree == 1)
        {
            gradientHalo.emplace(gpu.BindHalo(gpuWork->directGradients.front()));
            primitiveHalo.emplace(gpu.BindHalo(*gpuWork->primitiveGradient));
        }
        for (bool split : {true, false})
        {
            std::shared_ptr<C::FiniteVolume::Workspace::GradientPositivitySubmission> gradientPP;
            R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
            const auto compute = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            const auto communication = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            gpuWork->ResetDeviceFluxStatus(engine, compute);
            if (degree == 1)
            {
                gpuWork->ResetPreparationStatus(engine, compute);
                gpuWork->SubmitDeviceBoundary(engine, compute, resident, false, .3);
            }
            auto meanTransfer = meanHalo.SubmitAfter(engine, communication, compute.Record());
            std::optional<R::TransferOperation> recTransfer;
            if (coefficientHalo)
                recTransfer.emplace(coefficientHalo->SubmitAfter(engine, communication, compute.Record()));
            if (split)
            {
                if (degree == 1)
                    gpuWork->SubmitGradients(engine, compute, resident, DNDS::CFV::ReconstructionRegion::Core);
                else
                    gpuWork->SubmitDeviceRHS(engine, compute, resident, actual, DNDS::CFV::ReconstructionRegion::Core, .3);
            }
            meanTransfer.EnqueueWait(compute);
            if (recTransfer)
                recTransfer->EnqueueWait(compute);
            if (degree == 1)
            {
                gpuWork->SubmitGradients(engine, compute, resident,
                                         split ? DNDS::CFV::ReconstructionRegion::HaloDependent : DNDS::CFV::ReconstructionRegion::All);
                gpuWork->SubmitGradientLimiter(engine, compute, resident, true);
                gradientPP = gpuWork->SubmitGradientPositivity(engine, compute, resident);
                auto gradientTransfer = gradientHalo->SubmitAfter(engine, communication, compute.Record());
                auto primitiveTransfer = primitiveHalo->SubmitAfter(engine, communication, compute.Record());
                if (split)
                    gpuWork->SubmitDeviceRHS(engine, compute, resident, actual, DNDS::CFV::ReconstructionRegion::Core, .3);
                gradientTransfer.EnqueueWait(compute);
                primitiveTransfer.EnqueueWait(compute);
            }
            gpuWork->SubmitDeviceRHS(engine, compute, resident, actual,
                                     split ? DNDS::CFV::ReconstructionRegion::HaloDependent : DNDS::CFV::ReconstructionRegion::All, .3);
            engine.Synchronize();
            if (degree == 1)
            {
                gpuWork->CheckPreparationStatus();
                gpuWork->PublishGradientPositivity(gradientPP);
                CHECK(gpuWork->directPPLimited == cpuWork->directPPLimited);
                CHECK(gpuWork->directPPMinimum == doctest::Approx(cpuWork->directPPMinimum).epsilon(3e-12));
            }
            gpuWork->CheckDeviceFluxStatus();
            const auto downloaded = actual.CoupledField().Owned().To(R::Device::Host());
            const auto x = expected.CoupledField().Owned().Read(), y = downloaded.Read();
            for (I iCell = 0; iCell < cpu.Metrics().NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(y.View()(iCell, j) == doctest::Approx(x.View()(iCell, j)).epsilon(3e-11).scale(1));
        }
    }
}

TEST_CASE("Queued CUDA owned-mean and CFL reductions match CPU including empty partitions and first failure")
{
    const auto device = R::Device::CUDA(0);
    for (const std::string model : {"none", "SA", "KOmegaWilcox", "KOmegaSST"})
    {
        C::EquationSettings settings;
        settings.transport = "NewtonianFourier";
        settings.viscosity = .01;
        settings.rans.model = model;
        const auto equation = C::EquationSystem::Resolve(settings);
        const I width = I(equation.Schema().LogicalComponents().size());
        const auto rans = equation.HasRANS() ? C::PrepareRANSStatePolicy(equation, {.1, .2}, 1e5, .01) : C::RANSStateView{};
        const C::AdmissibilityPolicy policy(equation, 1e-8, 1e-8, 0, 1e-8, rans);
        const auto bounds = policy.View();
        for (const I rows : {I(0), I(1), I(1025), I(32769)})
        {
            R::Array<real> mean(R::RowLayout::Dense(rows, width)), sums(R::RowLayout::Dense(rows, 1)), cells(R::RowLayout::Dense(rows + 3, 8));
            {
                auto u = mean.Write(), s = sums.Write(), m = cells.Write();
                for (I iCell = 0; iCell < rows; ++iCell)
                {
                    const auto flow = equation.Gas().Conservative({1 + .001 * iCell, .2, .1, -.1, 1.1});
                    for (I j = 0; j < width; ++j)
                        u.View()(iCell, j) = j < 5 ? flow[j] : .1;
                    s.View()(iCell, 0) = 2 + .07 * iCell;
                    m.View()(iCell, 0) = .1 + .003 * iCell;
                }
            }
            auto dm = mean.To(device), ds = sums.To(device), dc = cells.To(device);
            R::Array<K::StateReductionValue> valid(R::RowLayout::Dense(1, 1), device), step(valid.Layout(), device);
            R::Array<K::StateReductionValue> hostIdentity(R::RowLayout::Dense(1, 1));
            hostIdentity.Write().At(0, 0) = K::StableStepIdentity;
            const auto identity = hostIdentity.To(device);
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            const auto compare = [&]
            {
                bool expectedValid;
                DNDS::NumericalResult<real> expectedStep{real(0)};
                {
                    const auto u = mean.Read(), s = sums.Read(), m = cells.Read();
                    std::vector<DNDS::Algebra::DofComponentColumn<const real>> columns;
                    for (I j = 0; j < width; ++j)
                        columns.push_back({u.View<R::UniformTableRows>(), {}, j});
                    const DNDS::Algebra::DofComponentLayout<const real> components(std::move(columns), rows, 0);
                    expectedValid = C::CPU::AdmissibleMeans(components.View(), bounds);
                    DNDS::CFV::FiniteVolumeHostView metric;
                    metric.cells = m.View<R::FixedRows<8>, false>();
                    expectedStep = C::CPU::MinimumStableStep(s.View<R::UniformTableRows>(), metric, 3.5);
                }
                R::SubmitArrays(engine, stream, [bounds](R::ExecutionContext c, auto u, auto result)
                                { K::AdmissibleMeans(c, u, bounds, result); }, R::ReadAs<R::TableRows>(dm), R::WriteAs<R::FixedRows<1>>(valid));
                // Independent CFL atomic reduction on the same stream, no host wait.
                R::SubmitArrays(engine, stream, [](R::ExecutionContext c, auto s, auto m, auto seed, auto result)
                                { K::MinimumStableStep(c, s, m, 3.5, seed, result); }, R::ReadAs<R::TableRows>(ds), R::ReadAs<R::FixedRows<8>>(dc), R::ReadAs<R::FixedRows<1>>(identity), R::WriteAs<R::FixedRows<1>>(step));
                engine.Synchronize();
                const auto hv = valid.To(R::Device::Host()), hs = step.To(R::Device::Host());
                const auto v = hv.Read(), s = hs.Read();
                CHECK(v.View()(0, 0).value == real(expectedValid));
                CHECK((v.View()(0, 0).firstInvalid < 0) == expectedValid);
                CHECK((s.View()(0, 0).firstInvalid < 0) == bool(expectedStep));
                if (expectedStep)
                    CHECK(s.View()(0, 0).value == doctest::Approx(expectedStep.value).epsilon(2e-14));
                else
                    CHECK(s.View()(0, 0).firstInvalid == expectedStep.status.row);
            };
            compare();
            // A larger next minimum must replace the preceding smaller result.
            {
                auto write = sums.Write();
                for (I iCell = 0; iCell < rows; ++iCell)
                    write.View()(iCell, 0) *= .5;
            }
            ds.CopyFrom(sums, stream.Context()).Wait();
            compare();
            if (rows > 1)
            {
                mean.Write().At(rows / 2, 0) = -1;
                sums.Write().At(rows - 1, 0) = 0;
                dm.CopyFrom(mean, stream.Context()).Wait();
                ds.CopyFrom(sums, stream.Context()).Wait();
                compare();
            }
        }
    }
}

TEST_CASE("CUDA analytic error uses the CPU oracle quadrature and fixed-tree rank-local reductions")
{
    for (const int dimension : {2, 3})
    {
        const auto mesh = dimension == 2 ? CauchyTest::PeriodicQuads(mpi, 17) : CFVTest::CartesianHexes(mpi, 3);
        const auto metrics = DNDS::CFV::FiniteVolumePlan::ConstructMetrics(mesh);
        const auto geometry = metrics.AcquireRead();
        const auto metric = geometry.HostView();
        const auto meshRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
        const auto view = meshRead.HostView();
        const I rows = metrics.NumCell();
        std::vector<I> starts(size_t(rows + 1), 0);
        for (I iCell = 0; iCell < rows; ++iCell)
            starts[size_t(iCell + 1)] = starts[size_t(iCell)] + DNDS::Geom::Elem::Quadrature(
                                                                    view.GetCellElement(iCell), DNDS::Geom::Elem::INT_ORDER_MAX)
                                                                    .GetNumPoints();
        R::Array<I> offsets(R::RowLayout::Dense(rows + 1, 1));
        R::Array<real> points(R::RowLayout::Dense(starts.back(), 4)), cells(R::RowLayout::Dense(rows, 8));
        {
            auto o = offsets.Write();
            auto p = points.Write(), c = cells.Write();
            for (I iRow = 0; iRow <= rows; ++iRow)
                o.View()(iRow, 0) = starts[size_t(iRow)];
            for (I iCell = 0; iCell < rows; ++iCell)
            {
                for (int j = 0; j < 8; ++j)
                    c.View()(iCell, j) = metric.cells(iCell, j);
                DNDS::Geom::tSmallCoords coords;
                view.GetCoordsOnCell(iCell, coords);
                const auto element = view.GetCellElement(iCell);
                const DNDS::Geom::Elem::Quadrature quadrature(element, DNDS::Geom::Elem::INT_ORDER_MAX);
                for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                {
                    const auto &shape = DNDS::Geom::Elem::GetCachedShapeAtQuadrature(element.type, DNDS::Geom::Elem::INT_ORDER_MAX, iQ);
                    const DNDS::Geom::tPoint point = DNDS::Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                    for (int d = 0; d < 3; ++d)
                        p.View()(starts[size_t(iCell)] + iQ, d) = point[d];
                    p.View()(starts[size_t(iCell)] + iQ, 3) = DNDS::Geom::Elem::CellJacobianDet(dimension, coords, shape) * quadrature.GetWeight(iQ);
                }
            }
        }
        const auto device = R::Device::CUDA(0);
        auto doffsets = offsets.To(device);
        auto dpoints = points.To(device), dcells = cells.To(device);
        C::PerfectGas gas;
        C::RigidMotionSettings motion;
        motion.angularVelocity = {0, 0, .2};
        motion.translationVelocity = {.1, -.2, 0};
        const C::RigidMotion moving(motion, dimension);
        const real time = .35;
        const auto frame = moving.At(time);
        const C::FlowState primitive{1, .3, -.2, .1, 1};
        for (const std::string kind : {"constant", "isentropicVortex"})
        {
            C::FlowProfileSettings profile;
            profile.kind = kind;
            profile.strength = .8;
            profile.center = {.4, .6, 0};
            profile.period = 1;
            // Extra scalar components are deliberately unrelated to the flow
            // oracle; the kernel must use only the first five components.
            R::Array<real> means(R::RowLayout::Dense(rows, 7));
            {
                auto write = means.Write();
                REQUIRE(bool(C::CPU::InitializeFlowProfile(view, {write.View<R::UniformTableRows>(), 0, 5}, profile, primitive, gas, moving.At(0))));
                for (I iCell = 0; iCell < rows; ++iCell)
                {
                    write.View()(iCell, 0) += .001 * (iCell + 1);
                    write.View()(iCell, 5) = 100;
                    write.View()(iCell, 6) = -100;
                }
            }
            const auto read = means.Read();
            const auto expected = C::CPU::MeasureFlowError(view, metric, {read.View<R::UniformTableRows>(), 0, 5}, profile, primitive, gas, time, frame);
            REQUIRE(bool(expected));
            std::vector<DNDS::NumericalResult<C::CPU::FlowErrorStatistics>> cpuWorkers(
                3, DNDS::NumericalResult<C::CPU::FlowErrorStatistics>{C::CPU::FlowErrorStatistics{}});
            const auto partitioned = C::CPU::MeasureFlowError(view, metric, {read.View<R::UniformTableRows>(), 0, 5},
                                                              profile, primitive, gas, time, frame, {cpuWorkers.data(), I(cpuWorkers.size())});
            REQUIRE(bool(partitioned));
            CHECK(partitioned.value.volume == doctest::Approx(expected.value.volume).epsilon(2e-13));
            for (int norm = 0; norm < 3; ++norm)
                for (int j = 0; j < 5; ++j)
                    CHECK(partitioned.value.errors[norm][j] == doctest::Approx(expected.value.errors[norm][j]).epsilon(2e-13).scale(1));
            auto dmeans = means.To(device);
            R::Array<K::FlowErrorReduction> scratch(R::RowLayout::Dense(K::StateReductionScratchRows(rows), 1), device);
            R::Array<K::FlowErrorReduction> output(R::RowLayout::Dense(1, 1), device), repeat(output.Layout(), device);
            R::Array<unsigned int> completed(R::RowLayout::Dense(1, 1), device);
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            const auto submit = [&](R::Array<K::FlowErrorReduction> &result)
            {
                R::SubmitArrays(engine, stream, [parameters = C::PrepareFlowProfile(profile), primitive, gas, frame = frame.KernelParameters(), time](R::ExecutionContext context, auto offsets, auto points, auto cells, auto means, auto scratch, auto output, auto counter)
                                { K::MeasureFlowError(context, {offsets, points, cells}, means, parameters, primitive, gas, frame, time, scratch, output, counter); }, R::ReadAs<R::FixedRows<1>>(doffsets), R::ReadAs<R::FixedRows<4>>(dpoints), R::ReadAs<R::FixedRows<8>>(dcells), R::ReadAs<R::TableRows>(dmeans), R::WriteAs<R::FixedRows<1>>(scratch), R::WriteAs<R::FixedRows<1>>(result), R::WriteAs<R::FixedRows<1>>(completed));
            };
            submit(output);
            submit(repeat); // no host visibility or scratch reallocation between submissions
            engine.Synchronize();
            const auto host = output.To(R::Device::Host()), again = repeat.To(R::Device::Host());
            const auto a = host.Read(), b = again.Read();
            const auto actual = a.View()(0, 0), repeated = b.View()(0, 0);
            REQUIRE(bool(actual.status));
            CHECK(actual.volume == doctest::Approx(expected.value.volume).epsilon(2e-13));
            CHECK(actual.volume == repeated.volume);
            for (int norm = 0; norm < 3; ++norm)
                for (int j = 0; j < 5; ++j)
                {
                    CHECK(actual.errors[norm][j] == doctest::Approx(expected.value.errors[norm][j]).epsilon(2e-11).scale(1));
                    CHECK(actual.errors[norm][j] == repeated.errors[norm][j]);
                }
        }
    }
}

TEST_CASE("Managed characteristics feed WBAP without a host observation between operators")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.SORInstead = false;
    settings.numerics.limiter = "WBAP";
    settings.numerics.characteristicLimiter = true;
    settings.numerics.characteristicDiagnostics = true;
    settings.numerics.positivity = false;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume cpu(mesh, settings, equation);
    auto means = C::EquationDofs::Coupled(equation.Schema(), cpu.Metrics().BuildField(5));
    cpu.Initialize(means, {}, {1, .2, -.1, .03, 1});
    cpu.BindHalo(means.CoupledField()).Begin().Wait();
    auto seed = cpu.BuildReconstructionState(means);
    {
        auto write = seed->WarmValues().CoupledField().Owned().Write();
        for (I iCell = 0; iCell < write.View().RowCount(); ++iCell)
            for (I j = 0; j < write.View().Width(); ++j)
                write.View()(iCell, j) = 1e-3 * (1 + (iCell + j) % 7);
    }
    cpu.BindHalo(seed->WarmValues().CoupledField()).Begin().Wait();
    std::optional<R::Array<real>> expected, expectedForward;
    for (const int backend : {0, 1, 2})
    {
        CAPTURE(backend);
        const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
        const R::ExecutionContext transfer{device, nullptr};
        auto plan = backend == 2 ? cpu.Placed(device, transfer) : cpu;
        auto state = C::EquationDofs::Coupled(equation.Schema(), plan.Metrics().BuildField(5));
        state.CoupledField().Owned().CopyFrom(means.CoupledField().Owned(), transfer).Wait();
        state.CoupledField().Ghost().CopyFrom(means.CoupledField().Ghost(), transfer).Wait();
        auto reconstruction = plan.BuildReconstructionState(state);
        auto &coefficients = reconstruction->WarmValues().CoupledField();
        coefficients.Owned().CopyFrom(seed->WarmValues().CoupledField().Owned(), transfer).Wait();
        coefficients.Ghost().CopyFrom(seed->WarmValues().CoupledField().Ghost(), transfer).Wait();
        auto work = plan.BuildWorkspace(state);
        R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
        const auto record = work->SubmitCharacteristics(engine, stream, state);
        const auto vr = plan.PolynomialReconstruction();
        REQUIRE(vr);
        for (int order = 3; order >= 1; --order)
        {
            vr->SubmitLimiterWBAP_3OrderTo(engine, stream, coefficients, *work->fields.front().reconstruction, order,
                                           DNDS::CFV::FaceCharacteristicTransformRef(std::cref(*work->flowCharacteristic)));
            vr->SubmitCommitReconstruction(engine, stream, coefficients, *work->fields.front().reconstruction);
        }
        engine.Synchronize();
        work->PublishCharacteristics(record);
        CHECK(work->characteristicFallbacks == 0);
        const auto result = coefficients.Owned().To(R::Device::Host());
        const auto forward = work->flowCharacteristic->forward.To(R::Device::Host());
        if (!expected)
        {
            expected = result;
            expectedForward = forward;
        }
        else
            for (const auto pair : {std::make_pair(*expected, result), std::make_pair(*expectedForward, forward)})
            {
                const auto a = pair.first.Read(), b = pair.second.Read();
                for (I iRow = 0; iRow < a.View().RowCount(); ++iRow)
                    for (I j = 0; j < a.View().Width(); ++j)
                        CHECK(b.View()(iRow, j) == doctest::Approx(a.View()(iRow, j)).epsilon(2e-10).scale(1));
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
