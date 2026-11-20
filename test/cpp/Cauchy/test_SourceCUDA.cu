#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Discretization/CUDA/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include <cuda_runtime.h>

namespace C = DNDS::Cauchy;
namespace K = DNDS::Cauchy::CUDA;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using DNDS::real;

namespace
{
    template <class T, class Function>
    R::Array<T> Table(I rows, I width, Function value)
    {
        R::Array<T> result(R::RowLayout::Dense(rows, width));
        auto write = result.Write();
        for (I iRow = 0; iRow < rows; ++iRow)
            for (I j = 0; j < width; ++j)
                write.View()(iRow, j) = value(iRow, j);
        return result;
    }
    template <class T>
    DNDS::SpanView<const T> Span(const std::vector<T> &values) { return {values.data(), I(values.size())}; }
}

TEST_CASE("CUDA RANS quadrature sources and destruction diagonals match generic CPU")
{
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    for (const std::string model : {"SA", "KOmegaWilcox", "KOmegaSST"})
        for (int operation = 0; operation < 6; ++operation)
        {
            INFO("model=" << model << ", source operation=" << operation);
            C::EquationSettings settings;
            settings.transport = "NewtonianFourier";
            settings.viscosity = .002;
            settings.rans.model = model;
            settings.rans.viscosityReference = .01;
            const auto equation = C::EquationSystem::Resolve(settings);
            const int width = 5 + equation.RANS().ComponentNumber();
            const bool mean = operation == 1, jacobian = operation > 0 && operation < 4, fullBlock = operation == 3;
            const int outputWidth = fullBlock ? width * width : width;
            const auto parameters = C::PerfectGasSpatialParameters::FromEquation(equation, 2);
            auto parametersArray = Table<C::PerfectGasSpatialParameters>(1, 1, [&](I, I)
                                                                         { return parameters; })
                                       .To(device);
            auto offsets = Table<I>(3, 1, [](I i, I)
                                    { return i == 0 ? 0 : i == 1 ? 2
                                                                 : 5; });
            auto selected = Table<I>(2, 1, [](I i, I)
                                     { return 1 - i; })
                                .To(device);
            // Geometry includes a ghost row, as the production mesh caches do.
            // Source operands/outputs still have only two owned cell rows.
            auto lengths = Table<real>(3, 1, [](I i, I)
                                       { return .3 + .1 * i; });
            auto wallMean = Table<real>(3, 1, [](I i, I)
                                        { return .1 + .01 * i; });
            auto weights = Table<real>(5, 1, [](I i, I)
                                       { return std::array<real, 5>{.4, .6, .3, .7, 1}[i]; });
            auto metric = Table<real>(3, 8, [](I i, I j)
                                      { return j == 0 ? real(1 + i) : j == 7 ? 1.1 + .1 * i
                                                                             : 0.; });
            const auto state = [&](I iRow, I j)
            {
                const auto flow = equation.Gas().Conservative({1.1 + .01 * iRow, .3, .02, -.1, 1.2});
                return j < 5 ? flow[j] : j == 5 ? .02 + .001 * iRow
                                                : .3 + .01 * iRow;
            };
            const bool direct = operation == 4, modal = operation == 5;
            const auto parent = [](I q) -> I
            { return q < 2 ? 0 : 1; };
            const auto displacement = [](I q, int d)
            { return .03 * (q + 1) * (d == 0 ? 1 : -2); };
            const auto modalSlope = [](I cell, I j, int d)
            { return .001 * (cell + 1) * (j + 1) * (d + 1); };
            auto means = Table<real>(2, width, state);
            auto samples = Table<real>(5, width, [&](I q, I j)
                                       {
                                           real value = state(direct || modal ? parent(q) : q, j);
                                           if (modal)
                                               for (int d = 0; d < 2; ++d)
                                                   value += modalSlope(parent(q), j, d) * displacement(q, d);
                                           return value; });
            const I gradientRows = mean ? 2 : 5;
            std::array<R::Array<real>, 3> gradients;
            for (int d = 0; d < 3; ++d)
                gradients[d] = Table<real>(gradientRows, width, [=](I i, I j)
                                           { return modal ? modalSlope(parent(i), j, d)
                                                          : .001 * ((direct ? parent(i) : i) + 1) * (j + 1) * (d + 1); });
            R::Array<real> wallPoints(R::RowLayout::Ragged({2, 3, 1}));
            {
                auto wall = wallPoints.Write();
                for (I i = 0; i < 3; ++i)
                    for (I q = 0; q < wall.View().RowSize(i); ++q)
                        wall.View()(i, q) = .1 + .02 * (i + q);
            }
            auto expected = Table<real>(2, outputWidth, [](I, I)
                                        { return .125; });
            auto actual = expected.To(device);
            auto derivative = Table<real>(2, width * width, [](I, I)
                                          { return real(-1); });
            auto deviceDerivative = derivative.To(device);
            // For the direct-gradient case compare the fused output with a CPU
            // quadrature-only reference evaluated from explicitly sampled rows.
            const bool paired = operation == 4;
            {
                const auto u = means.Read(), q = samples.Read(), g0 = gradients[0].Read(), g1 = gradients[1].Read(), g2 = gradients[2].Read();
                const auto w = weights.Read(), len = lengths.Read(), wm = wallMean.Read(), wp = wallPoints.Read(), geo = metric.Read();
                const auto off = offsets.Read();
                auto output = expected.Write();
                const C::EquationBlockReadView flow{{u.View<R::UniformTableRows>(), 0, 5}, {}};
                const C::EquationBlockReadView rans{{u.View<R::UniformTableRows>(), 5, width - 5}, {}};
                const std::vector<C::EquationBlockReadView> blocks{flow, rans};
                const std::vector<C::EquationBlockLocation> locations{{0, 0, 5}, {0, 5, width - 5}};
                const std::vector<I> logical{0, 5};
                const std::vector<C::RHSScalarView> points{q.View<R::UniformTableRows>()};
                const std::vector<std::array<C::CPU::SourceGradientView, 3>> derivatives{{{{g0.View<R::UniformTableRows>(), 1, 0}, {g1.View<R::UniformTableRows>(), 1, 0}, {g2.View<R::UniformTableRows>(), 1, 0}}}};
                C::CPU::SourceKernelRequest request;
                request.metric.cells = geo.View<R::FixedRows<8>, false>();
                request.offsets = off.View<R::FixedRows<1>>();
                request.weights = w.View<R::FixedRows<1>>();
                request.lengths = len.View<R::FixedRows<1>>();
                request.blocks = Span(blocks);
                request.locations = Span(locations);
                request.logicalFirst = Span(logical);
                request.samples = Span(points);
                request.gradients = Span(derivatives);
                request.walls.cellMean = wm.View<R::FixedRows<1>, false>();
                request.walls.cells = wp.View<R::CSRRows, false>();
                request.ransMean = rans;
                request.ransOutput = {output.View<R::UniformTableRows>(), 5, width - 5};
                request.matrix = output.View<R::UniformTableRows>();
                request.cells = 2;
                request.components = width;
                request.flowFirst = 0;
                request.ransFirst = 5;
                request.dimension = 2;
                request.fullBlock = fullBlock;
                request.scale = .7;
                auto worker = equation.MakePointEvaluator();
                const auto status = mean       ? C::CPU::AssembleCenterSourceJacobian(request, equation, worker)
                                    : jacobian ? C::CPU::AddQuadratureSourceJacobian(request, equation, worker)
                                               : C::CPU::EvaluateVolumeSource(request, equation, worker);
                REQUIRE(bool(status));
                if (paired)
                {
                    auto record = derivative.Write();
                    for (I i = 0; i < 2; ++i)
                        for (I k = 0; k < width * width; ++k)
                            record.View()(i, k) = 0;
                    auto reference = request;
                    reference.matrix = record.View<R::UniformTableRows>();
                    reference.fullBlock = true;
                    reference.rowMajorBlocks = true;
                    reference.scale = 1;
                    REQUIRE(bool(C::CPU::AddQuadratureSourceJacobian(reference, equation, worker)));
                }
                std::vector<real> serial(size_t(2 * outputWidth));
                for (I iCell = 0; iCell < 2; ++iCell)
                    for (I j = 0; j < outputWidth; ++j)
                    {
                        serial[size_t(iCell * outputWidth + j)] = output.View()(iCell, j);
                        output.View()(iCell, j) = .125;
                    }
                std::vector<C::CPU::SourceWorkerScratch> workers;
                for (int iWorker = 0; iWorker < 3; ++iWorker)
                    workers.emplace_back(equation.MakePointEvaluator(), width);
                const DNDS::SpanView<C::CPU::SourceWorkerScratch> slots{workers.data(), I(workers.size())};
                const auto threaded = mean       ? C::CPU::AssembleCenterSourceJacobian(request, equation, slots)
                                      : jacobian ? C::CPU::AddQuadratureSourceJacobian(request, equation, slots)
                                                 : C::CPU::EvaluateVolumeSource(request, equation, slots);
                REQUIRE(bool(threaded));
                for (I iCell = 0; iCell < 2; ++iCell)
                    for (I j = 0; j < outputWidth; ++j)
                        CHECK(output.View()(iCell, j) == serial[size_t(iCell * outputWidth + j)]);
            }
            auto du = means.To(device), dq = samples.To(device), dw = weights.To(device), dl = lengths.To(device), dm = wallMean.To(device), dp = wallPoints.To(device), dg = metric.To(device);
            auto offDevice = offsets.To(device);
            std::optional<R::Array<real>> modalBasis, modalCoefficients;
            if (modal)
            {
                modalBasis = Table<real>(5, 2 * 3, [&](I q, I col)
                                         {
                                                const int mode = int(col / 3), derivative = int(col % 3);
                                                const real scale = 1.1 + .1 * parent(q);
                                                return derivative ? real(derivative - 1 == mode) / scale
                                                                  : displacement(q, mode) / scale; })
                                 .To(device);
                modalCoefficients = Table<real>(2, width * 2, [&](I cell, I j)
                                                { return (1.1 + .1 * cell) * modalSlope(cell, j / 2, int(j % 2)); })
                                        .To(device);
            }
            std::array<R::Array<real>, 3> derivativeDevice;
            for (int d = 0; d < 3; ++d)
                derivativeDevice[d] = direct ? Table<real>(2, width, [d](I cell, I j)
                                                           { return .001 * (cell + 1) * (j + 1) * (d + 1); })
                                                   .To(device)
                                             : gradients[d].To(device);
            R::Array<DNDS::NumericalStatus> status(R::RowLayout::Dense(2, 1), device);
            {
                const auto params = parametersArray.Read(context);
                const auto u = du.Read(context), q = dq.Read(context), w = dw.Read(context), len = dl.Read(context), wm = dm.Read(context), wp = dp.Read(context), geo = dg.Read(context);
                const auto g0 = derivativeDevice[0].Read(context), g1 = derivativeDevice[1].Read(context), g2 = derivativeDevice[2].Read(context);
                const auto off = offDevice.Read(context), rows = selected.Read(context);
                auto output = actual.Write(context);
                auto errors = status.Write(context);
                auto record = deviceDerivative.Write(context);
                K::SourceRequest request;
                request.parameters = params.View<R::FixedRows<1>>();
                request.means = u.View<R::TableRows>();
                request.cellInputs = direct;
                if (!direct && !modal)
                    request.samples = q.View<R::TableRows>();
                request.gradients = {{{g0.View<R::TableRows>(), 1, 0}, {g1.View<R::TableRows>(), 1, 0}, {g2.View<R::TableRows>(), 1, 0}}};
                request.offsets = off.View<R::FixedRows<1>>();
                request.weights = w.View<R::FixedRows<1>>();
                request.lengths = len.View<R::FixedRows<1>>();
                request.wallMean = wm.View<R::FixedRows<1>>();
                request.wallPoints = wp.View<R::CSRRows>();
                request.cells = geo.View<R::FixedRows<8>, false>();
                request.output = output.View<R::UniformTableRows>();
                request.status = errors.View<R::FixedRows<1>>();
                if (paired)
                    request.preparedDerivative = record.View<R::UniformTableRows>();
                request.components = width;
                request.dimension = 2;
                request.fullBlock = fullBlock;
                request.scale = .7;
                request.evaluation = mean ? K::SourceEvaluation::MeanDestruction : jacobian ? K::SourceEvaluation::QuadratureDestruction
                                                                                            : K::SourceEvaluation::QuadratureResidual;
                std::optional<R::Array<real>::ReadLease> modalBasisRead, modalCoefficientRead;
                if (modal)
                {
                    modalBasisRead.emplace(modalBasis->Read(context));
                    modalCoefficientRead.emplace(modalCoefficients->Read(context));
                    request.coefficients = modalCoefficientRead->View<R::TableRows>();
                    request.basis.polynomial = DNDS::CFV::Math::TaylorBasis(2, 1).KernelParameters();
                    request.basis.cached = modalBasisRead->View<R::UniformTableRows>();
                    request.basis.cachedDerivatives = 3;
                }
                K::AddRANSSources(context, request, rows.View<R::FixedRows<1>>());
                REQUIRE(cudaStreamSynchronize(::DNDS::Runtime::CUDAStream(context)) == cudaSuccess);
            }
            if (paired)
            {
                const auto downloadedDerivative = deviceDerivative.To(R::Device::Host());
                const auto actualRecord = downloadedDerivative.Read(), expectedRecord = derivative.Read();
                for (I i = 0; i < 2; ++i)
                    for (I k = 0; k < width * width; ++k)
                        CHECK(actualRecord.View()(i, k) == doctest::Approx(expectedRecord.View()(i, k)).epsilon(5e-11).scale(1));
            }
            const auto downloaded = actual.To(R::Device::Host());
            const auto errors = status.To(R::Device::Host());
            const auto a = downloaded.Read(), b = expected.Read();
            const auto e = errors.Read();
            for (I iCell = 0; iCell < 2; ++iCell)
            {
                CHECK(bool(e.View()(iCell, 0)));
                for (I j = 0; j < outputWidth; ++j)
                    CHECK(a.View()(iCell, j) == doctest::Approx(b.View()(iCell, j)).epsilon(5e-11).scale(1));
            }
        }
}

TEST_CASE("CUDA moving-axis source preserves scalar components")
{
    const auto device = R::Device::CUDA(0), host = R::Device::Host();
    const auto context = R::ExecutionContext::CUDA(0);
    auto mean = Table<real>(2, 7, [](I i, I j)
                            { return 1 + .2 * i + .1 * j; });
    auto geometry = Table<real>(2, 8, [](I i, I j)
                                { return j == 0 ? 2 + i : 0; });
    auto selected = Table<I>(2, 1, [](I i, I)
                             { return 1 - i; })
                        .To(device);
    auto dm = mean.To(device), dg = geometry.To(device);
    auto output = Table<real>(2, 7, [](I, I)
                              { return .5; })
                      .To(device);
    C::RigidFrame frame;
    frame.moving = true;
    frame.angularVelocity = {.2, -.3, .5};
    {
        const auto m = dm.Read(context), g = dg.Read(context);
        const auto rows = selected.Read(context);
        auto result = output.Write(context);
        K::AddRotationSource(context, g.View<R::FixedRows<8>, false>(), frame.KernelParameters(), m.View<R::TableRows>(), result.View<R::TableRows>(), {}, false, rows.View<R::FixedRows<1>>());
        REQUIRE(cudaStreamSynchronize(::DNDS::Runtime::CUDAStream(context)) == cudaSuccess);
    }
    const auto result = output.To(host);
    const auto r = result.Read(), m = mean.Read(), g = geometry.Read();
    for (I iCell = 0; iCell < 2; ++iCell)
    {
        C::FlowState state;
        for (int j = 0; j < 5; ++j)
            state[j] = m.View()(iCell, j);
        const auto source = frame.RotationSource(state);
        for (int j = 0; j < 7; ++j)
            CHECK(r.View()(iCell, j) == doctest::Approx(.5 + (j < 5 ? source[j] * g.View()(iCell, 0) : 0)));
    }
}
