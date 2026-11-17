#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Discretization/CUDA/FiniteVolumeGradientKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeGradientKernels.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "CompactStencilFixture.hpp"
#include <cuda_runtime.h>

namespace C = DNDS::Cauchy;
namespace K = DNDS::Cauchy::CUDA;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using DNDS::real;

namespace
{
    template <class T, class F>
    R::Array<T> Table(I rows, I width, F f)
    {
        R::Array<T> result(R::RowLayout::Dense(rows, width));
        auto write = result.Write();
        for (I i = 0; i < rows; ++i)
            for (I j = 0; j < width; ++j)
                write.View()(i, j) = f(i, j);
        return result;
    }
    void Compare(const R::Array<real> &device, const R::Array<real> &host)
    {
        const auto downloaded = device.To(R::Device::Host());
        const auto a = downloaded.Read(), b = host.Read();
        for (I i = 0; i < host.RowCount(); ++i)
            for (I j = 0; j < host.Layout().Width(); ++j)
                CHECK(a.View()(i, j) == doctest::Approx(b.View()(i, j)).epsilon(2e-12).scale(1));
    }
}

TEST_CASE("CUDA Cauchy centered Barth PP and primitive means match generic CPU")
{
    C::EquationSettings settings;
    settings.transport = "NewtonianFourier";
    settings.viscosity = .001;
    settings.rans.model = "KOmegaSST";
    const auto equation = C::EquationSystem::Resolve(settings);
    C::ThermalPolicy thermal(equation, 1e-6, 1e-6, 0, 1e-6);
    const auto state = equation.Gas().Conservative({1.2, .2, .1, -.1, 1.1});
    auto mean = Table<real>(1, 7, [&](I, I j)
                            { return j < 5 ? state[j] : .02; });
    auto ghost = Table<real>(1, 7, [&](I, I j)
                             { return j < 5 ? state[j] * 1.2 : .03; });
    auto gradient = Table<real>(1, 14, [](I, I j)
                                { return j == 8 ? real(-10) : j == 10 ? real(-.2)
                                                                      : .01 * (j + 1); });
    auto selected = Table<I>(1, 1, [](I, I)
                             { return I(0); });
    CauchyTest::CompactStencilFixture::Input fixture;
    fixture.cellFaces = {{0, 1}};
    fixture.faceCells = {{0, 1}, {0, DNDS::UnInitIndex}};
    fixture.zones = {-1, 1};
    fixture.cells = {1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1};
    fixture.faceOffsets = {0, 1, 2};
    fixture.faceMetrics = {.5, .1, 0, 1, 0, 0, 1, -.5, .1, 0, -1, 0, 0, 1};
    fixture.quadrature = fixture.faceMetrics;
    fixture.codes = {1, 0};
    fixture.parameters.generators.rotation[1].map() << 0, 1, 0, -1, 0, 0, 0, 0, 1;
    fixture.parameters.rotatedGenerators = 1;
    const CauchyTest::CompactStencilFixture cpuGeometry(fixture);
    // Preserve the old test's quarter-turn neighbor-to-cell frame explicitly;
    // the new canonical code is its left-to-right inverse, not an own Pbi flag.
    CHECK(cpuGeometry.View().metric.periodic.OtherRotation(0, 0)(0, 1) == -1);
    CHECK(cpuGeometry.View().metric.periodic.OtherRotation(0, 0)(1, 0) == 1);
    const auto device = R::Device::CUDA(0), host = R::Device::Host();
    const auto context = R::ExecutionContext::CUDA(0);
    const CauchyTest::CompactStencilFixture gpuGeometry(fixture, context);
    auto dm = mean.To(device), dghost = ghost.To(device), dg = gradient.To(device);
    auto rows = selected.To(device);
    R::Array<real> cpuBarth(R::RowLayout::Dense(1, 14)), gpuBarth(cpuBarth.Layout(), device);
    auto cpuPositive = gradient.To(host), gpuPositive = gradient.To(device);
    R::Array<real> cpuPrimitive(R::RowLayout::Dense(1, 5)), gpuPrimitive(cpuPrimitive.Layout(), device);
    const I summaries = K::GradientPositivitySummaryRows(1);
    R::Array<DNDS::NumericalStatus> partialStatus(R::RowLayout::Dense(summaries, 1), device), status(R::RowLayout::Dense(1, 1), device);
    R::Array<I> partialCount(R::RowLayout::Dense(summaries, 1), device), count(R::RowLayout::Dense(1, 1), device);
    R::Array<real> partialMinimum(R::RowLayout::Dense(summaries, 1), device), minimum(R::RowLayout::Dense(1, 1), device);
    C::CPU::GradientPositivityResult expectedStats;
    {
        const auto u = mean.Read(), g = ghost.Read(), grad = gradient.Read();
        std::vector<DNDS::Algebra::DofComponentColumn<const real>> columns;
        for (I j = 0; j < 7; ++j)
            columns.push_back({u.View<R::UniformTableRows>(), g.View<R::UniformTableRows>(), j});
        const DNDS::Algebra::DofComponentLayout<const real> layout(std::move(columns), 1, 1);
        const std::array<I, 7> logicalField{}, logicalComponent{0, 1, 2, 3, 4, 5, 6};
        const I vectorStart = 1;
        {
            auto out = cpuBarth.Write();
            R::Array<real> scratch(R::RowLayout::Dense(6, 7));
            auto work = scratch.Write();
            const auto source = grad.View<R::UniformTableRows>();
            const auto destination = out.View<R::UniformTableRows>();
            C::CPU::GradientLimiterRequest request;
            request.stencil = cpuGeometry.View();
            request.metric = request.stencil.metric;
            request.means = layout.View();
            request.source = {&source, 1};
            request.destination = {&destination, 1};
            request.logicalField = {logicalField.data(), 7};
            request.logicalComponent = {logicalComponent.data(), 7};
            request.vectorStarts = {&vectorStart, 1};
            request.scratch = work.View<R::TableRows>();
            request.components = 7;
            request.dimension = 2;
            request.barth = true;
            C::CPU::LimitPhysicalGradients(request);
        }
        {
            auto out = cpuPositive.Write();
            R::Array<real> values(R::RowLayout::Dense(2, 7));
            auto work = values.Write();
            const auto destination = out.View<R::UniformTableRows>();
            C::CPU::GradientPositivityRequest request;
            request.stencil = cpuGeometry.View();
            request.means = layout.View();
            request.gradients = {&destination, 1};
            request.logicalField = {logicalField.data(), 7};
            request.logicalComponent = {logicalComponent.data(), 7};
            request.samples = work.View<R::TableRows>();
            request.bounds = thermal.View();
            request.rans = C::ResolveRANSReconstructionPolicy(equation);
            request.components = 7;
            request.ransFirst = 5;
            request.dimension = 2;
            const auto result = C::CPU::LimitPositivePhysicalGradients(request);
            REQUIRE(bool(result));
            expectedStats = result.value;
        }
        auto primitive = cpuPrimitive.Write();
        C::CPU::FillPrimitiveMeans(u.View<R::UniformTableRows>(), primitive.View<R::UniformTableRows>(), equation.Gas());
    }
    {
        const auto u = dm.Read(context), g = dghost.Read(context), grad = dg.Read(context);
        const auto selection = rows.Read(context);
        auto out = gpuBarth.Write(context), positive = gpuPositive.Write(context), beta = partialMinimum.Write(context), resultMinimum = minimum.Write(context);
        auto errors = partialStatus.Write(context), resultStatus = status.Write(context);
        auto counts = partialCount.Write(context), resultCount = count.Write(context);
        const K::GradientLimiterGeometry geometry{gpuGeometry.Descriptor(), 1, 2};
        K::LimitPhysicalGradients(context, geometry, {u.View<R::TableRows>(), g.View<R::TableRows>()}, grad.View<R::TableRows>(), out.View<R::TableRows>(), selection.View<R::FixedRows<1>>(), 2, true);
        K::GradientPositivityParameters policy;
        policy.bounds = thermal.View();
        policy.rans = C::ResolveRANSReconstructionPolicy(equation);
        policy.dimension = 2;
        const K::ModalPositivitySummary partials{errors.View<R::FixedRows<1>>(), counts.View<R::FixedRows<1>>(), beta.View<R::FixedRows<1>>()};
        K::LimitPositivePhysicalGradients(context, geometry, u.View<R::TableRows>(), positive.View<R::TableRows>(), selection.View<R::FixedRows<1>>(), policy, partials);
        K::FinalizeModalPositivity(context, partials,
                                   {resultStatus.View<R::FixedRows<1>>(), resultCount.View<R::FixedRows<1>>(), resultMinimum.View<R::FixedRows<1>>()});
        REQUIRE(cudaStreamSynchronize(::DNDS::Runtime::CUDAStream(context)) == cudaSuccess);
    }
    {
        const auto checked = status.To(host);
        const auto min = minimum.To(host);
        const auto limited = count.To(host);
        REQUIRE(bool(checked.Read().View()(0, 0)));
        CHECK(min.Read().View()(0, 0) == doctest::Approx(expectedStats.minimum));
        CHECK(limited.Read().View()(0, 0) == expectedStats.limited);
        CHECK(expectedStats.limited == 1);
    }
    {
        const auto u = dm.Read(context);
        auto primitive = gpuPrimitive.Write(context);
        auto errors = status.Write(context);
        K::FillPrimitiveMeans(context, u.View<R::TableRows>(), primitive.View<R::TableRows>(), equation.Gas(), errors.View<R::FixedRows<1>>());
        REQUIRE(cudaStreamSynchronize(::DNDS::Runtime::CUDAStream(context)) == cudaSuccess);
    }
    Compare(gpuBarth, cpuBarth);
    Compare(gpuPositive, cpuPositive);
    Compare(gpuPrimitive, cpuPrimitive);
}
