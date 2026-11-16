#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "CompactStencilFixture.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeRHSKernels.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Runtime/Array/ArrayCopy.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

namespace C = DNDS::Cauchy;
namespace K = DNDS::Cauchy::CUDA;
namespace R = DNDS::Runtime;
namespace F = DNDS::CFV;
using I = DNDS::index;
using DNDS::real;

namespace
{
    template <class T>
    DNDS::SpanView<const T> Span(const std::vector<T> &values) { return {values.data(), I(values.size())}; }
    template <class T>
    R::Array<T> Rows(const std::vector<T> &values, I width = 1)
    {
        R::Array<T> result(R::RowLayout::Dense(I(values.size()) / width, width));
        auto write = result.Write();
        for (I iRow = 0; iRow < result.RowCount(); ++iRow)
            for (I j = 0; j < width; ++j)
                write.View()(iRow, j) = values[iRow * width + j];
        return result;
    }
    // Two owned cells, one remote cell, two points per face. One owned-owned
    // face has a rotated periodic image; the wall and interface have separate
    // packet/force slots. This is a numerical incidence fixture, not a mesh
    // closure or geometry-builder test. The oracle is the actual generic CPU
    // RHS, not the host instantiation of the new CUDA arithmetic.
    struct Fixture
    {
        std::vector<C::BoundaryCondition> conditions{3};
        // Independent legacy-shaped point oracle for preparing cached basis
        // values only. Production leaves consume the compact publication below.
        std::vector<C::FiniteVolumeDetail::SampleRecord> samples;
        CauchyTest::CompactStencilFixture::Input geometry;
        R::Array<I> offsets = Rows<I>({0, 6, 10}), entries = Rows<I>({0, 2, 4, 8});
        R::Array<I> faceOffsets = Rows<I>({0, 2, 4, 6, 8});
        R::Array<I> coreFaces = Rows<I>({3, 1, 0}), dependentFaces = Rows<I>({2});
        R::Array<I> coreCells = Rows<I>({1}), dependentCells = Rows<I>({0});
        R::Array<real> cells, quadrature, wallMean = Rows<real>({.1, .2, .3});
        R::Array<real> wallFaces{R::RowLayout::Ragged({2, 2, 2, 2})};
        R::Array<C::PerfectGasBoundaryParameters> boundary;
        Fixture(bool periodic)
        {
            conditions[0].kind = C::BoundaryKind::Extrapolate;
            conditions[1].kind = C::BoundaryKind::NoSlipWall;
            conditions[1].isothermal = true;
            conditions[1].temperature = .9;
            conditions[2].kind = C::BoundaryKind::Interface;
            std::vector<C::PerfectGasBoundaryParameters> parameters;
            for (const auto &condition : conditions)
                parameters.push_back(condition.PerfectGasKernelParameters());
            boundary = Rows(parameters);
            cells = Rows<real>({1, 0, 0, 0, 0, 0, 0, 1,
                                1, 1, 0, 0, 1, 0, 0, 1,
                                1, 0, 1, 0, 0, 1, 0, 1},
                               8);
            const std::array<real, 9> identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
            const std::array<real, 9> rotate{0, 1, 0, -1, 0, 0, 0, 0, 1};
            const std::array<real, 9> inverse{0, -1, 0, 1, 0, 0, 0, 0, 1};
            std::vector<C::FiniteVolumeDetail::SampleRecord> records;
            std::vector<real> points(8 * 7);
            for (I incidence = 0; incidence < 5; ++incidence)
                for (I iQ = 0; iQ < 2; ++iQ)
                {
                    const I face = incidence < 3 ? incidence : incidence == 3 ? 1
                                                                              : 3;
                    C::FiniteVolumeDetail::SampleRecord s{};
                    s.cell = incidence < 3 ? 0 : 1;
                    s.other = incidence == 1 ? 1 : incidence == 2 ? 2
                                               : incidence == 3   ? 0
                                                                  : DNDS::UnInitIndex;
                    s.face = s.auxiliaryFace = face;
                    s.point = face * 2 + iQ;
                    s.side = incidence == 3 ? 1 : 0;
                    s.condition = incidence == 0 ? 1 : incidence == 4 ? 2
                                                                      : 0;
                    s.boundaryPoint = incidence == 0 ? iQ : incidence == 4 ? 2 + iQ
                                                                           : DNDS::UnInitIndex;
                    s.forceSlot = s.jacobianColumn = DNDS::UnInitIndex;
                    s.rotation = periodic && face == 1 ? (s.side ? inverse : rotate) : identity;
                    s.changesFrame = periodic && face == 1;
                    s.normal = face == 0 ? C::Vector3{-1, 0, 0} : face == 1 ? C::Vector3{1, 0, 0}
                                                                            : C::Vector3{0, 1, 0};
                    s.ownPoint = face == 0   ? C::Vector3{-.5, .1 * (2 * iQ - 1), 0}
                                 : face == 1 ? C::Vector3{.5, .1 * (2 * iQ - 1), 0}
                                 : face == 2 ? C::Vector3{.1 * (2 * iQ - 1), .5, 0}
                                             : C::Vector3{1 + .1 * (2 * iQ - 1), .5, 0};
                    s.otherPoint = face == 1 ? C::Vector3{1, 0, 0} : C::Vector3{0, 1, 0};
                    s.viscousDistance = 1;
                    if (incidence == 3)
                    {
                        // Coordinates in cell1's native frame; the opposite
                        // periodic vector map sends the anchor flux here.
                        const auto anchor = records[2 + iQ];
                        for (int d = 0; d < 3; ++d)
                        {
                            s.ownPoint[d] = d == 0 ? 1 : 0;
                            s.otherPoint[d] = d == 0 ? 1 : 0;
                            s.normal[d] = 0;
                            for (int k = 0; k < 3; ++k)
                            {
                                s.ownPoint[d] += s.rotation[d + 3 * k] * (anchor.ownPoint[k] - anchor.otherPoint[k]);
                                s.otherPoint[d] -= s.rotation[d + 3 * k] * anchor.otherPoint[k];
                                s.normal[d] -= s.rotation[d + 3 * k] * anchor.normal[k];
                            }
                        }
                    }
                    else
                    {
                        for (int d = 0; d < 3; ++d)
                        {
                            points[7 * s.point + d] = s.ownPoint[d];
                            points[7 * s.point + 3 + d] = s.normal[d];
                        }
                        points[7 * s.point + 6] = .5;
                    }
                    records.push_back(s);
                }
            samples = std::move(records);
            quadrature = Rows(points, 7);
            geometry.cellFaces = {{0, 1, 2}, {1, 3}};
            geometry.faceCells = {{{0, DNDS::UnInitIndex}}, {{0, 1}}, {{0, 2}}, {{1, DNDS::UnInitIndex}}};
            geometry.cells = {1, 0, 0, 0, 0, 0, 0, 1,
                              1, 1, 0, 0, 1, 0, 0, 1,
                              1, 0, 1, 0, 0, 1, 0, 1};
            geometry.quadrature = points;
            geometry.faceOffsets = {0, 2, 4, 6, 8};
            geometry.codes = {0, uint8_t(periodic ? 4 : 0), 0, 0};
            geometry.parameters.rotatedGenerators = periodic ? 1 : 0;
            geometry.parameters.generators.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
            geometry.parameters.generators.rotationCenter[1].map() = DNDS::Geom::tPoint{1, 0, 0};
            geometry.parameters.generators.translation[1].map().setZero();
            geometry.routing = {{0, 1, 0, 0, DNDS::UnInitIndex},
                                {1, 0, DNDS::UnInitIndex, DNDS::UnInitIndex, DNDS::UnInitIndex},
                                {2, 0, DNDS::UnInitIndex, DNDS::UnInitIndex, DNDS::UnInitIndex},
                                {3, 2, 1, 2, DNDS::UnInitIndex}};
            geometry.coreCells = {1};
            geometry.dependentCells = {0};
            geometry.coreFaces = {3, 1, 0};
            geometry.dependentFaces = {2};
            for (I iFace = 0; iFace < 4; ++iFace)
            {
                for (int d = 0; d < 6; ++d)
                    geometry.faceMetrics.push_back(d < 3 ? .5 * (points[14 * iFace + d] + points[14 * iFace + 7 + d]) : points[14 * iFace + d]);
                geometry.faceMetrics.push_back(1);
            }
            auto wall = wallFaces.Write();
            for (I iFace = 0; iFace < 4; ++iFace)
                for (I iQ = 0; iQ < 2; ++iQ)
                    wall.View()(iFace, iQ) = .05 + .02 * iFace;
        }
    };
    template <class T>
    struct Layout
    {
        std::vector<C::EquationBlockArrayView<T>> blocks;
        std::vector<I> offsets{0, 5};
        explicit Layout(R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows> view)
        {
            blocks.push_back({view, 0, 5});
            if (view.Width() > 5)
            {
                blocks.push_back({view, 5, view.Width() - 5});
                offsets.push_back(view.Width());
            }
        }
        C::EquationLayoutView<T> View() const { return {Span(blocks), Span(offsets), 0, blocks.size() > 1 ? 1 : -1, -1, true}; }
    };
}

TEST_CASE("CUDA integrated RHS matches generic CPU with periodic halo wall and interface incidences")
{
    const auto device = R::Device::CUDA(0), host = R::Device::Host();
    const auto context = R::ExecutionContext::CUDA(0);
    using Precision = DNDS::Math::ComputePrecision;
    // Exercise all five reduced-arithmetic combinations per scalar, including
    // the two coupled choices whose launch definitions live in separate TUs.
    constexpr std::array precisionCases{
        std::array{Precision::FP64, Precision::FP64, Precision::FP64},
        std::array{Precision::FP64, Precision::FP64, Precision::FP64}, // tiled input
        std::array{Precision::FP32, Precision::FP64, Precision::FP64},
        std::array{Precision::FP64, Precision::FP32, Precision::FP64},
        std::array{Precision::FP64, Precision::FP64, Precision::FP32},
        std::array{Precision::FP32, Precision::FP32, Precision::FP64},
        std::array{Precision::FP32, Precision::FP64, Precision::FP32},
#ifdef DNDS_USE_CCCL_COMPUTE
        std::array{Precision::FloatFloat, Precision::FP64, Precision::FP64},
        std::array{Precision::FP64, Precision::FloatFloat, Precision::FP64},
        std::array{Precision::FP64, Precision::FP64, Precision::FloatFloat},
        std::array{Precision::FloatFloat, Precision::FloatFloat, Precision::FP64},
        std::array{Precision::FloatFloat, Precision::FP64, Precision::FloatFloat},
#endif
    };
    constexpr int firstMapping = int(precisionCases.size());
    constexpr int firstSegmented = firstMapping + 4;
    constexpr int experimentalVariants = firstSegmented + 12;
    for (const std::string model : {"none", "SA", "KOmegaWilcox", "KOmegaSST"})
        for (const int reconstruction : {0, 1, 2, 3})
            for (int experiment = 0; experiment < (model == "none" && reconstruction == 3 ? experimentalVariants : 1); ++experiment)
            {
                // Retain the canonical model/reconstruction matrix. Experimental
                // policies add only one uncached P3 flow5 fixture per selection.
                INFO("model=" << model << ", reconstruction=" << reconstruction << ", experiment=" << experiment);
                const auto arithmetic = precisionCases[experiment < firstMapping ? experiment : 0];
                const bool fp32 = std::find(arithmetic.begin(), arithmetic.end(), Precision::FP32) != arithmetic.end();
                const bool floatFloat = std::find(arithmetic.begin(), arithmetic.end(), Precision::FloatFloat) != arithmetic.end();
                const real tolerance = fp32 ? 5e-5 : floatFloat ? 2e-10
                                                                : 3e-11;
                C::EquationSettings settings;
                settings.transport = "NewtonianFourier";
                settings.viscosity = .002;
                settings.viscosityModel = "sutherland";
                settings.sutherlandConstant = .3;
                settings.rans.model = model;
                settings.rans.viscosityReference = .01;
                const auto equation = C::EquationSystem::Resolve(settings);
                const int width = 5 + (equation.HasRANS() ? equation.RANS().ComponentNumber() : 0);
                const int modes = reconstruction < 2 ? 2 : 9;
                Fixture fixture(reconstruction != 0);
                CauchyTest::CompactStencilFixture hostStencil(fixture.geometry), deviceStencil(fixture.geometry, context);
                const auto frame = C::RigidMotion({}, 2).At(.2);
                C::ThermalPolicy thermal(equation, 1e-10, 1e-10, 0, 1e-10);
                K::RHSKernelParameters parameters;
                parameters.physics = C::PerfectGasSpatialParameters::FromEquation(equation, 2);
                parameters.thermal = thermal.View();
                parameters.Validate(width);
                K::RHSKernelControls controls;
                controls.frame = frame.KernelParameters();
                controls.time = .2;
                controls.components = width;
                controls.dimension = 2;
                controls.modes = modes;
                controls.direct = reconstruction < 2;
                controls.linear = reconstruction == 1;
                controls.firstOrder = reconstruction == 0;
                controls.viscous = true;
                controls.distinctViscous = reconstruction != 0;
                controls.produceLow = width == 5;
                controls.positivity = true;
                controls.solver = model == "SA" ? C::Riemann::Solver::Roe_M9 : C::Riemann::Solver::AUSMPWPlus;
                controls.polynomialPrecision = arithmetic[0];
                controls.dissipationPrecision = arithmetic[1];
                controls.fluxPrecision = arithmetic[2];
                if (controls.dissipationPrecision != Precision::FP64)
                    controls.solver = C::Riemann::Solver::Roe_M1;
                controls.mapping = experiment == firstMapping || experiment == firstMapping + 2 ? K::RHSFaceMapping::WarpQuadrature : K::RHSFaceMapping::Thread;
                controls.doubleBuffer = experiment > firstMapping && experiment < firstSegmented;
                controls.blockSize = experiment == firstMapping + 1 ? 64 : experiment == firstMapping + 3 ? 256
                                                                                                          : 128;
                if (experiment >= firstSegmented)
                {
                    const int variant = experiment - firstSegmented;
                    controls.mapping = variant / 6 ? K::RHSFaceMapping::CTASegments : K::RHSFaceMapping::WarpSegments;
                    controls.reduction = (variant / 3) % 2 ? K::RHSFaceReduction::Segmented : K::RHSFaceReduction::Serial;
                    controls.blockSize = std::array<int, 3>{128, 256, 512}[variant % 3];
                    // 512 lanes with high+low vectors exceed the portable
                    // 48KiB budget. Qualify that block with high-order output
                    // only, matching the implicit campaign's real contract.
                    controls.produceLow = controls.blockSize != 512;
                }
                // Two points per face exercise idle lanes and fixed warp
                // quadrature accumulation against the same CPU oracle.
                controls.fluxPolicy.required = C::Riemann::RequiredInputs(controls.solver);
                controls.fluxPolicy.noRsOnWall = reconstruction % 2;
                controls.fluxPolicy.noGRPOnWall = reconstruction == 2;
                controls.useStencil = controls.fluxPolicy.transverseSignal = controls.fluxPolicy.pressureStencil = true;
                controls.Validate();
                std::vector<real> mean(3 * width), coef(3 * width * modes), viscous(coef.size()), prepared(3 * 2);
                for (I iCell = 0; iCell < 3; ++iCell)
                {
                    const auto flow = equation.Gas().Conservative({1.2 + .02 * iCell, .2, .05, -.03, 1.1});
                    const auto acoustic = equation.PrepareFluxState(flow);
                    prepared[2 * iCell] = acoustic.flow.pressure;
                    prepared[2 * iCell + 1] = acoustic.soundSpeed;
                    for (int j = 0; j < width; ++j)
                    {
                        mean[iCell * width + j] = j < 5 ? flow[j] : j == 5 ? .02
                                                                           : .3;
                        for (int m = 0; m < modes; ++m)
                        {
                            coef[(iCell * width + j) * modes + m] = .001 * (iCell + 1) * (j + 1) / (m + 1);
                            viscous[(iCell * width + j) * modes + m] = -.3 * coef[(iCell * width + j) * modes + m];
                        }
                    }
                }
                auto u = Rows<real>({mean.begin(), mean.begin() + 2 * width}, width);
                auto ug = Rows<real>({mean.begin() + 2 * width, mean.end()}, width);
                auto rec = Rows<real>({coef.begin(), coef.begin() + 2 * width * modes}, width * modes);
                auto recg = Rows<real>({coef.begin() + 2 * width * modes, coef.end()}, width * modes);
                auto vis = Rows<real>({viscous.begin(), viscous.begin() + 2 * width * modes}, width * modes);
                auto visg = Rows<real>({viscous.begin() + 2 * width * modes, viscous.end()}, width * modes);
                std::vector<real> packet(4 * width);
                for (I iRow = 0; iRow < 4; ++iRow)
                    for (int j = 0; j < width; ++j)
                        packet[iRow * width + j] = mean[width + j];
                auto incoming = Rows(packet, width), meanAux = Rows(prepared, 2);
                auto wave = Rows<real>({.03, .07}), waveg = Rows<real>({.09});
                auto pressure = Rows<real>({.7, .8}), pressureg = Rows<real>({.6});
                const auto polynomial = F::Math::TaylorBasis(2, 3).KernelParameters();
                R::Array<real> moments(R::RowLayout::Dense(3, 10)), cache(R::RowLayout::Dense(16, 27));
                {
                    auto mom = moments.Write(), cached = cache.Write();
                    auto cells = fixture.cells.Read();

                    auto entries = fixture.entries.Read();
                    for (I iCell = 0; iCell < 3; ++iCell)
                        for (int j = 0; j < 10; ++j)
                            mom.View()(iCell, j) = j == 0 ? 1 : 0;
                    for (I iFace = 0; iFace < 4; ++iFace)
                        for (I iQ = 0; iQ < 2; ++iQ)
                            for (int side = 0; side < 2; ++side)
                            {
                                const auto &s = fixture.samples[size_t(entries.View()(iFace, 0) + iQ)];
                                const I iCell = side && s.other != DNDS::UnInitIndex ? s.other : s.cell;
                                C::Vector3 x{};
                                for (int d = 0; d < 3; ++d)
                                {
                                    if (side && s.other != DNDS::UnInitIndex)
                                        for (int k = 0; k < 3; ++k)
                                            x[d] += s.rotation[k + 3 * d] * (s.ownPoint[k] - s.otherPoint[k]);
                                    else
                                        x[d] = s.ownPoint[d] - cells.View()(iCell, 1 + d);
                                }
                                for (int m = 0; m < 9; ++m)
                                    for (int derivative = 0; derivative < 3; ++derivative)
                                        cached.View()(2 * s.point + side, 3 * m + derivative) = polynomial.Derivative(derivative, m + 1, x, 1);
                            }
                }
                R::Array<real> cpuFaces(R::RowLayout::Dense(4, width)), cpuLowFaces(R::RowLayout::Dense(4, width));
                R::Array<real> cpuRHS(R::RowLayout::Dense(2, width)), cpuLow(R::RowLayout::Dense(2, width)), cpuForces(R::RowLayout::Dense(4, 3));
                I cpuFallbacks = -1;
                {
                    const auto a = u.Read(), b = ug.Read(), r = rec.Read(), rg = recg.Read(), v = vis.Read(), vg = visg.Read();
                    const auto geo = fixture.cells.Read(), quad = fixture.quadrature.Read(), wall = fixture.wallFaces.Read(), wm = fixture.wallMean.Read();
                    const auto off = fixture.offsets.Read(), e = fixture.entries.Read(), fo = fixture.faceOffsets.Read();
                    const auto packetRead = incoming.Read(), aux = meanAux.Read(), w = wave.Read(), wg = waveg.Read(), p = pressure.Read(), pg = pressureg.Read(), cached = cache.Read();
                    auto face = cpuFaces.Write(), lowFace = cpuLowFaces.Write(), rhs = cpuRHS.Write(), low = cpuLow.Write(), force = cpuForces.Write();
                    Layout<const real> own(a.View<R::UniformTableRows>()), ghost(b.View<R::UniformTableRows>());
                    Layout<real> result(rhs.View<R::UniformTableRows>()), lowResult(low.View<R::UniformTableRows>());
                    std::vector<C::EquationBlockReadView> means;
                    for (I iBlock = 0; iBlock < I(own.blocks.size()); ++iBlock)
                        means.push_back({own.blocks[iBlock], ghost.blocks[iBlock]});
                    const std::vector<I> starts{1}, logical = width == 5 ? std::vector<I>{0} : std::vector<I>{0, 5};
                    std::vector<C::EquationBlockLocation> locations;
                    std::vector<C::EquationBlockReadView> recBlocks, visBlocks;
                    for (const auto &block : own.blocks)
                    {
                        locations.push_back({0, block.first, block.components});
                        recBlocks.push_back({{r.View<R::UniformTableRows>(), block.first * modes, block.components * modes}, {rg.View<R::UniformTableRows>(), block.first * modes, block.components * modes}});
                        visBlocks.push_back({{v.View<R::UniformTableRows>(), block.first * modes, block.components * modes}, {vg.View<R::UniformTableRows>(), block.first * modes, block.components * modes}});
                    }
                    std::vector<C::RHSFieldView> gradients{{r.View<R::UniformTableRows>(), rg.View<R::UniformTableRows>()}}, viscousGradients{{v.View<R::UniformTableRows>(), vg.View<R::UniformTableRows>()}};
                    C::CPU::EquationRHSRequest request;
                    request.stencil = hostStencil.View();
                    request.metric = request.stencil.metric;
                    F::ReconstructionBasisEvaluationView basis;
                    basis.cached = cached.View<R::TableRows>();
                    basis.dimension = 2;
                    basis.degree = 3;
                    basis.modes = 9;
                    basis.cachedDerivatives = 3;
                    request.basis = basis;
                    request.conditions = Span(fixture.conditions);
                    request.meanBlocks = Span(means);
                    request.recBlocks = Span(recBlocks);
                    request.viscousRecBlocks = Span(visBlocks);
                    request.directLocations = Span(locations);
                    request.directViews = Span(gradients);
                    request.directViscousViews = Span(viscousGradients);
                    request.logicalFirst = Span(logical);
                    request.vectorStarts = Span(starts);
                    if (width > 5)
                        request.ransMean = means[1];
                    request.preparedMean = aux.View<R::UniformTableRows>();
                    request.incoming = packetRead.View<R::UniformTableRows>();
                    request.waveJump = {w.View<R::UniformTableRows>(), wg.View<R::UniformTableRows>()};
                    request.pressureMinimum = {p.View<R::UniformTableRows>(), pg.View<R::UniformTableRows>()};
                    request.faceValues = face.View<R::UniformTableRows>();
                    request.lowFaceValues = lowFace.View<R::UniformTableRows>();
                    request.wallFlux = force.View<R::FixedRows<3>>();
                    request.wall.faces = wall.View<R::CSRRows, false>();
                    request.wall.cellMean = wm.View<R::FixedRows<1>, false>();
                    request.thermal = thermal.View();
                    request.frame = frame;
                    request.fluxPolicy = controls.fluxPolicy;
                    request.ownedCells = 2;
                    request.width = width;
                    request.flowFirst = 0;
                    request.ransFirst = width > 5 ? 5 : -1;
                    request.dimension = 2;
                    request.modes = modes;
                    request.direct = controls.direct;
                    request.linear = controls.linear;
                    request.firstOrder = controls.firstOrder;
                    request.viscous = request.hasWall = request.positivity = request.useStencil = request.preparedWaveJump = request.preparedPressureMinimum = true;
                    request.produceLow = controls.produceLow;
                    request.time = controls.time;
                    C::BoundaryPointOperator bc(equation);
                    auto worker = equation.MakePointEvaluator({}, {true, true, true});
                    const auto flux = C::Riemann::RiemannFlux::Resolve(controls.solver == C::Riemann::Solver::Roe_M1 ? "Roe_M1" : model == "SA" ? "Roe_M9"
                                                                                                                                                : "AUSMPWPlus");
                    const auto status = C::CPU::EvaluateEquationFlux(request, equation, bc, flux, worker, own.View(), ghost.View(), result.View(), lowResult.View(), F::ReconstructionRegion::All);
                    CAPTURE(int(status.status.code));
                    CAPTURE(status.status.detail);
                    CAPTURE(status.status.row);
                    CAPTURE(status.status.point);
                    REQUIRE(bool(status));
                    cpuFallbacks = status.value;
                    // Three worker slots also exercise an empty cell range. The
                    // face pass must join before gather; otherwise a different
                    // worker could consume an unfinished neighboring face value.
                    std::vector<real> serialFaces(size_t(4 * width)), serialRHS(size_t(2 * width));
                    for (I iFace = 0; iFace < 4; ++iFace)
                        for (I j = 0; j < width; ++j)
                            serialFaces[size_t(iFace * width + j)] = face.View()(iFace, j);
                    for (I iCell = 0; iCell < 2; ++iCell)
                        for (I j = 0; j < width; ++j)
                            serialRHS[size_t(iCell * width + j)] = rhs.View()(iCell, j);
                    std::vector<C::CPU::EquationRHSWorkerScratch> workers;
                    for (int iWorker = 0; iWorker < 3; ++iWorker)
                        workers.emplace_back(equation.MakePointEvaluator({}, {true, true, true}), width, request.species, request.produceLow);
                    const auto threaded = C::CPU::EvaluateEquationFlux(request, equation, bc, flux,
                                                                       DNDS::SpanView<C::CPU::EquationRHSWorkerScratch>{workers.data(), I(workers.size())},
                                                                       own.View(), ghost.View(), result.View(), lowResult.View(), F::ReconstructionRegion::All);
                    REQUIRE(bool(threaded));
                    CHECK(threaded.value == cpuFallbacks);
                    for (I iFace = 0; iFace < 4; ++iFace)
                        for (I j = 0; j < width; ++j)
                            CHECK(face.View()(iFace, j) == serialFaces[size_t(iFace * width + j)]);
                    for (I iCell = 0; iCell < 2; ++iCell)
                        for (I j = 0; j < width; ++j)
                            CHECK(rhs.View()(iCell, j) == serialRHS[size_t(iCell * width + j)]);
                }
                if (experiment == 1)
                {
                    // Exercise physical tiled input storage, including independent
                    // owned/ghost coefficient buffers; output and geometry stay AoS.
                    const auto tiled = [](const R::Array<real> &array)
                    { return R::ConvertLayout(array, R::RowLayout::TiledColMajor(array.RowCount(), array.Layout().Width(), 32)); };
                    u = tiled(u);
                    ug = tiled(ug);
                    rec = tiled(rec);
                    recg = tiled(recg);
                    vis = tiled(vis);
                    visg = tiled(visg);
                }
                auto du = u.To(device), dug = ug.To(device), dr = rec.To(device), drg = recg.To(device), dv = vis.To(device), dvg = visg.To(device);
                auto di = incoming.To(device), da = meanAux.To(device), dw = wave.To(device), dwg = waveg.To(device), dp = pressure.To(device), dpg = pressureg.To(device);
                auto db = fixture.boundary.To(device);
                auto doff = fixture.offsets.To(device), de = fixture.entries.To(device), dfo = fixture.faceOffsets.To(device);
                auto dcells = fixture.cells.To(device), dquad = fixture.quadrature.To(device), dwall = fixture.wallFaces.To(device), dwm = fixture.wallMean.To(device), dcache = cache.To(device), dm = moments.To(device);
                auto dcore = fixture.coreFaces.To(device), ddependent = fixture.dependentFaces.To(device), dcoreCells = fixture.coreCells.To(device), ddependentCells = fixture.dependentCells.To(device);
                auto dparameters = Rows<K::RHSKernelParameters>({parameters}).To(device);
                std::optional<K::RHSFacePackages> corePackages, dependentPackages;
                if (controls.mapping == K::RHSFaceMapping::WarpSegments || controls.mapping == K::RHSFaceMapping::CTASegments)
                {
                    const std::array<I, 4> pointCounts{2, 2, 2, 2};
                    const DNDS::SpanView<const I> counts{pointCounts.data(), I(pointCounts.size())};
                    corePackages.emplace(K::RHSFacePackages::Prepare(dcore, counts, controls, context));
                    dependentPackages.emplace(K::RHSFacePackages::Prepare(ddependent, counts, controls, context));
                }

                R::Array<real> faces(R::RowLayout::Dense(4, width), device), lowFaces(R::RowLayout::Dense(4, width), device), rhs(R::RowLayout::Dense(2, width), device), low(R::RowLayout::Dense(2, width), device), forces(R::RowLayout::Dense(4, 3), device);
                R::Array<DNDS::NumericalStatus> faceStatus(R::RowLayout::Dense(4, 1), device), cellStatus(R::RowLayout::Dense(2, 1), device);
                R::Array<I> fallback(R::RowLayout::Dense(4, 1), device);
                {
                    const auto a = du.Read(context), b = dug.Read(context), r = dr.Read(context), rg = drg.Read(context), v = dv.Read(context), vg = dvg.Read(context);
                    const auto geo = dcells.Read(context), quad = dquad.Read(context), wall = dwall.Read(context), wm = dwm.Read(context), cached = dcache.Read(context), mom = dm.Read(context);
                    const auto conditions = db.Read(context);
                    const auto off = doff.Read(context), e = de.Read(context), fo = dfo.Read(context);
                    const auto parametersRead = dparameters.Read(context);
                    const auto packetRead = di.Read(context), aux = da.Read(context), w = dw.Read(context), wg = dwg.Read(context), p = dp.Read(context), pg = dpg.Read(context);
                    const auto core = dcore.Read(context), dependent = ddependent.Read(context), coreCells = dcoreCells.Read(context), dependentCells = ddependentCells.Read(context);
                    K::RHSGeometry geometry;
                    geometry.stencil = deviceStencil.View();
                    geometry.faceOffsets = geometry.stencil.metric.faceOffsets;
                    geometry.cells = geometry.stencil.metric.cells;
                    geometry.quadrature = geometry.stencil.metric.quadrature;
                    geometry.conditions = conditions.View<R::FixedRows<1>>();
                    geometry.wallMean = wm.View<R::FixedRows<1>>();
                    geometry.wallFaces = wall.View<R::CSRRows>();
                    geometry.hasWall = true;
                    geometry.hasBasis = reconstruction >= 2;
                    geometry.basis.cached = cached.View<R::TableRows, false>();
                    geometry.basis.moments = mom.View<R::TableRows, false>();
                    geometry.basis.cells = geometry.cells;
                    geometry.basis.polynomial = polynomial;
                    geometry.basis.cachedDerivatives = reconstruction == 3 ? 0 : 3;
                    geometry.basis.availableDerivatives = 3;
                    K::RHSGeometryPublication published(geometry, context);
                    K::RHSFields fields;
                    fields.mean = {a.View<R::UniformTableRows>(), b.View<R::UniformTableRows>()};
                    fields.reconstruction = {r.View<R::UniformTableRows>(), rg.View<R::UniformTableRows>()};
                    fields.viscousData = {v.View<R::UniformTableRows>(), vg.View<R::UniformTableRows>()};
                    fields.preparedMean = aux.View<R::TableRows>();
                    fields.incoming = packetRead.View<R::TableRows>();
                    fields.waveJump = {w.View<R::UniformTableRows>(), wg.View<R::UniformTableRows>()};
                    fields.pressureMinimum = {p.View<R::UniformTableRows>(), pg.View<R::UniformTableRows>()};
                    fields.hasPreparedMean = fields.hasIncoming = fields.hasWave = fields.hasPressure = true;
                    {
                        auto face = faces.Write(context), lowFace = lowFaces.Write(context), force = forces.Write(context);
                        auto errors = faceStatus.Write(context);
                        auto counts = fallback.Write(context);
                        K::RHSFaceOutputs outputs{face.View<R::TableRows>(), lowFace.View<R::TableRows>(), force.View<R::FixedRows<3>>(), errors.View<R::FixedRows<1>>(), counts.View<R::FixedRows<1>>()};
                        auto coreFields = fields;
                        coreFields.mean.ghost = {};
                        coreFields.reconstruction.ghost = {};
                        coreFields.viscousData.ghost = {};
                        coreFields.waveJump.ghost = {};
                        coreFields.pressureMinimum.ghost = {};
                        K::EvaluateFluxFaces(context, published, parametersRead.View<R::FixedRows<1>>(), controls, coreFields, outputs, core.View<R::FixedRows<1>>(), false,
                                             corePackages ? corePackages->View() : K::RHSFacePackageView{});
                        K::EvaluateFluxFaces(context, published, parametersRead.View<R::FixedRows<1>>(), controls, fields, outputs, dependent.View<R::FixedRows<1>>(), true,
                                             dependentPackages ? dependentPackages->View() : K::RHSFacePackageView{});
                        REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
                    }
                    const auto face = faces.Read(context), lowFace = lowFaces.Read(context);
                    const auto errors = faceStatus.Read(context);
                    auto result = rhs.Write(context), lowResult = low.Write(context);
                    auto resultStatus = cellStatus.Write(context);
                    for (const auto rows : {coreCells.View<R::FixedRows<1>>(), dependentCells.View<R::FixedRows<1>>()})
                        K::GatherFluxCells(context, published, face.View<R::TableRows>(), lowFace.View<R::TableRows>(), errors.View<R::FixedRows<1>>(), result.View<R::TableRows>(), lowResult.View<R::TableRows>(), resultStatus.View<R::FixedRows<1>>(), rows, controls.produceLow);
                    // Borrowed-view launchers do not own these read/write scopes.
                    // Keep them through completion; production uses queued access.
                    REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
                }
                const auto check = [tolerance](const R::Array<real> &actual, const R::Array<real> &expected)
                {
                    const auto downloaded = actual.To(R::Device::Host());
                    const auto a = downloaded.Read(), b = expected.Read();
                    for (I iRow = 0; iRow < expected.RowCount(); ++iRow)
                        for (I j = 0; j < expected.Layout().Width(); ++j)
                        {
                            CHECK(std::isfinite(a.View()(iRow, j)));
                            CHECK(std::abs(a.View()(iRow, j) - b.View()(iRow, j)) <= tolerance * std::max(real(1), std::abs(b.View()(iRow, j))));
                        }
                };
                check(faces, cpuFaces);
                check(rhs, cpuRHS);
                check(forces, cpuForces);
                if (controls.produceLow)
                {
                    check(lowFaces, cpuLowFaces);
                    check(low, cpuLow);
                }
                const auto faceErrors = faceStatus.To(host), cellErrors = cellStatus.To(host);
                const auto counts = fallback.To(host);
                auto fe = faceErrors.Read(), ce = cellErrors.Read();
                auto n = counts.Read();
                I sum = 0;
                for (I iFace = 0; iFace < 4; ++iFace)
                {
                    CHECK(bool(fe.View()(iFace, 0)));
                    sum += n.View()(iFace, 0);
                }
                for (I iCell = 0; iCell < 2; ++iCell)
                    CHECK(bool(ce.View()(iCell, 0)));
                CHECK(sum == cpuFallbacks);
            }
}

TEST_CASE("CUDA canonical ghost-left and self-periodic faces preserve owned signs and incidence frames")
{
    const auto context = R::ExecutionContext::CUDA(0);
    const auto equation = C::EquationSystem::Resolve(C::EquationSettings{});
    const auto gas = equation.Gas();
    const auto rotate = [](C::FlowState state, bool inverse)
    {
        const real x = state[1], y = state[2];
        state[1] = inverse ? y : -y;
        state[2] = inverse ? -x : x;
        return state;
    };
    const auto physicalFlux = [&](const C::FlowState &state)
    {
        const real pressure = gas.Pressure(state), velocity = state[1] / state[0];
        return C::FlowState{state[1], state[1] * velocity + pressure,
                            state[2] * velocity, state[3] * velocity, (state[4] + pressure) * velocity};
    };
    for (bool self : {false, true})
    {
        CAPTURE(self);
        CauchyTest::CompactStencilFixture::Input input;
        input.cellFaces = self ? std::vector<std::vector<I>>{{0, 0}} : std::vector<std::vector<I>>{{0}};
        input.incidenceBits = self ? std::vector<std::vector<uint8_t>>{{0, 1}} : std::vector<std::vector<uint8_t>>{{0}};
        input.faceCells = self ? std::vector<std::array<I, 2>>{{{0, 0}}} : std::vector<std::array<I, 2>>{{{1, 0}}};
        input.faceOffsets = {0, 1};
        input.cells = {1, 1, 0, 0, 1, 0, 0, 1};
        if (!self)
            input.cells.insert(input.cells.end(), {1, -1, 0, 0, -1, 0, 0, 1});
        constexpr real area = .75;
        input.faceMetrics = input.quadrature = {.25, .2, 0, 1, 0, 0, area};
        input.codes = {4};
        input.parameters.rotatedGenerators = 1;
        input.parameters.generators.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
        input.parameters.generators.translation[1].map().setZero();
        input.quadratureBoundaryValues = false;
        input.coreCells = self ? std::vector<I>{0} : std::vector<I>{};
        input.dependentCells = self ? std::vector<I>{} : std::vector<I>{0};
        input.coreFaces = self ? std::vector<I>{0} : std::vector<I>{};
        input.dependentFaces = self ? std::vector<I>{} : std::vector<I>{0};
        CauchyTest::CompactStencilFixture hostGeometry(input), geometry(input, context);
        const auto leftRecord = hostGeometry.View().RHSFaceSample(0, 0, 0);
        CHECK(leftRecord.cell == (self ? 0 : 1)); // The canonical ghost stays on the left.
        const auto rightRecord = hostGeometry.View().GradientSample(0, self ? 1 : 0);
        CHECK(rightRecord.side == 1);
        if (self)
        {
            CHECK(hostGeometry.View().topology.CellFaces(0).size() == 2);
            CHECK(hostGeometry.View().GradientSample(0, 0).side == 0);
        }
        const auto owned = gas.Conservative({1.2, .3, -.2, .04, 1.1});
        const auto remote = gas.Conservative({.9, -.1, .15, -.02, .8});
        const auto left = self ? owned : remote;
        const auto right = rotate(owned, false); // right stored -> canonical left
        const auto fl = physicalFlux(left), fr = physicalFlux(right);
        const real signal = std::max(std::abs(left[1] / left[0]) + std::sqrt(gas.gamma * gas.Pressure(left) / left[0]),
                                     std::abs(right[1] / right[0]) + std::sqrt(gas.gamma * gas.Pressure(right) / right[0]));
        C::FlowState flux{}, expected{};
        for (int j = 0; j < 5; ++j)
            flux[j] = .5 * (fl[j] + fr[j]) - .5 * signal * (right[j] - left[j]);
        const auto rightFlux = rotate(flux, true); // canonical left -> stored right
        for (int j = 0; j < 5; ++j)
            expected[j] = area * (rightFlux[j] - (self ? flux[j] : 0));
        auto means = Rows<real>({owned.begin(), owned.end()}, 5).To(context.device);
        auto ghosts = Rows<real>(self ? std::vector<real>{} : std::vector<real>{remote.begin(), remote.end()}, 5).To(context.device);
        auto selected = Rows<I>({0}).To(context.device), noRows = Rows<I>({}).To(context.device);
        C::PerfectGasBoundaryParameters interior;
        interior.kind = C::BoundaryKind::Interior;
        auto conditions = Rows<C::PerfectGasBoundaryParameters>({interior}).To(context.device);
        C::ThermalPolicy thermal(equation, 1e-12, 1e-12, 0, 1e-12);
        K::RHSKernelParameters parameters;
        parameters.physics = C::PerfectGasSpatialParameters::FromEquation(equation, 2);
        parameters.thermal = thermal.View();
        auto policy = Rows<K::RHSKernelParameters>({parameters}).To(context.device);
        const auto meanRead = means.Read(context), ghostRead = ghosts.Read(context);
        const auto conditionRead = conditions.Read(context);
        const auto selectedRead = selected.Read(context), emptyRead = noRows.Read(context);
        const auto policyRead = policy.Read(context);
        K::RHSGeometry numerical;
        numerical.stencil = geometry.View();
        numerical.cells = numerical.stencil.metric.cells;
        numerical.faceOffsets = numerical.stencil.metric.faceOffsets;
        numerical.quadrature = numerical.stencil.metric.quadrature;
        numerical.conditions = conditionRead.View<R::FixedRows<1>>();
        K::RHSGeometryPublication publication(numerical, context);
        K::RHSFields fields;
        fields.mean = {meanRead.View<R::UniformTableRows>(), ghostRead.View<R::UniformTableRows>()};
        K::RHSKernelControls controls;
        controls.components = 5;
        controls.dimension = 2;
        controls.direct = controls.firstOrder = true;
        controls.solver = C::Riemann::Solver::Rusanov;
        controls.fluxPolicy.rusanov = true;
        controls.fluxPolicy.required = C::Riemann::RequiredInputs(controls.solver);
        R::Array<real> faces(R::RowLayout::Dense(1, 5), context.device), result(R::RowLayout::Dense(1, 5), context.device);
        R::Array<DNDS::NumericalStatus> faceStatus(R::RowLayout::Dense(1, 1), context.device), cellStatus(R::RowLayout::Dense(1, 1), context.device);
        R::Array<I> counts(R::RowLayout::Dense(1, 1), context.device);
        {
            auto out = faces.Write(context);
            auto status = faceStatus.Write(context);
            auto count = counts.Write(context);
            K::RHSFaceOutputs outputs{out.View<R::TableRows>(), {}, {}, status.View<R::FixedRows<1>>(), count.View<R::FixedRows<1>>()};
            auto core = fields;
            core.mean.ghost = {};
            // The core launch intentionally has no ghost allocation. For the
            // right-owned case it is empty; dependent work sees the ghost.
            K::EvaluateFluxFaces(context, publication, policyRead.View<R::FixedRows<1>>(), controls, core, outputs,
                                 self ? selectedRead.View<R::FixedRows<1>>() : emptyRead.View<R::FixedRows<1>>(), false);
            if (!self)
                K::EvaluateFluxFaces(context, publication, policyRead.View<R::FixedRows<1>>(), controls, fields, outputs,
                                     selectedRead.View<R::FixedRows<1>>(), true);
            REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
        }
        {
            const auto face = faces.Read(context);
            const auto status = faceStatus.Read(context);
            auto out = result.Write(context);
            auto error = cellStatus.Write(context);
            K::GatherFluxCells(context, publication, face.View<R::TableRows>(), {}, status.View<R::FixedRows<1>>(),
                               out.View<R::TableRows>(), {}, error.View<R::FixedRows<1>>(), selectedRead.View<R::FixedRows<1>>(), false);
            REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
        }
        const auto downloaded = result.To(R::Device::Host());
        const auto checked = cellStatus.To(R::Device::Host());
        const auto values = downloaded.Read();
        const auto status = checked.Read();
        REQUIRE(bool(status.View()(0, 0)));
        for (int j = 0; j < 5; ++j)
            CHECK(values.View()(0, j) == doctest::Approx(expected[j]).epsilon(3e-12).scale(1));
    }
}
