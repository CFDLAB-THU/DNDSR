#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "CompactStencilFixture.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeAuxiliaryKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeBoundaryKernels.hpp"
#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace K = DNDS::Cauchy::CUDA;
using I = DNDS::index;
using DNDS::real;

namespace
{
    template <class T>
    R::Array<T> DeviceRows(const std::vector<T> &values)
    {
        R::Array<T> host(R::RowLayout::Dense(I(values.size()), 1));
        {
            auto write = host.Write();
            for (I iRow = 0; iRow < I(values.size()); ++iRow)
                write.View()(iRow, 0) = values[iRow];
        }
        return host.To(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
    }
    R::Array<real> DeviceStates(const std::vector<C::FlowState> &values)
    {
        R::Array<real> host(R::RowLayout::Dense(I(values.size()), 5));
        {
            auto write = host.Write();
            for (I iRow = 0; iRow < I(values.size()); ++iRow)
                for (int j = 0; j < 5; ++j)
                    write.View()(iRow, j) = values[iRow][j];
        }
        return host.To(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
    }
}

TEST_CASE("CUDA boundary producers match generic CPU face exterior and interface states")
{
    const auto device = R::Device::CUDA(0), host = R::Device::Host();
    const auto context = R::ExecutionContext::CUDA(0);
    for (const std::string model : {"none", "SA", "KOmegaWilcox", "KOmegaSST"})
    {
        INFO("boundary model=" << model);
        C::EquationSettings settings;
        settings.transport = "NewtonianFourier";
        settings.viscosity = .002;
        settings.viscosityModel = "sutherland";
        settings.sutherlandConstant = .3;
        settings.rans.model = model;
        settings.rans.viscosityReference = .01;
        const auto equation = C::EquationSystem::Resolve(settings);
        const int components = 5 + (equation.HasRANS() ? equation.RANS().ComponentNumber() : 0);
        const auto frame = C::RigidMotion({}, 2).At(.2);
        C::ThermalPolicy thermal(equation, 1e-10, 1e-10, 0, 1e-10);
        K::BoundaryKernelParameters parameters;
        parameters.gas = equation.Gas();
        parameters.rans = equation.HasRANS() ? equation.RANS().Kind() : DNDS::Physics::RANS::ModelKind::None;
        parameters.thermal = thermal.View();
        parameters.allowPointFallback = true;
        parameters.Validate(components);
        auto params = DeviceRows<K::BoundaryKernelParameters>({parameters});
        std::vector<C::BoundaryCondition> conditions(3);
        conditions[0].kind = C::BoundaryKind::NoSlipWall;
        conditions[0].isothermal = true;
        conditions[0].temperature = .8;
        conditions[1].kind = C::BoundaryKind::TotalConditionInlet;
        conditions[1].totalCondition = {1.3, 1.1, 1, 0, 0};
        conditions[1].totalRANSPrimitive = {.03, .4};
        conditions[2].kind = C::BoundaryKind::Interface;
        std::vector<C::PerfectGasBoundaryParameters> physical;
        std::vector<C::BoundaryTraceGeometry> points;
        for (I i = 0; i < 3; ++i)
        {
            physical.push_back(conditions[i].PerfectGasKernelParameters());
            points.push_back({i, i, i, i, 0, 0, {-1, 0, 0}, {.2 * i, .1, 0}});
        }
        auto conditionRows = DeviceRows(physical);
        CauchyTest::CompactStencilFixture::Input geometry;
        geometry.cellFaces = {{0}, {1}, {2}};
        geometry.faceCells = {{{0, DNDS::UnInitIndex}}, {{1, DNDS::UnInitIndex}}, {{2, DNDS::UnInitIndex}}};
        geometry.faceOffsets = {0, 1, 2, 3};
        geometry.routing = {{0, 0, 1, 1, DNDS::UnInitIndex}, {1, 1, 2, 2, DNDS::UnInitIndex}, {2, 2, 0, 0, DNDS::UnInitIndex}};
        for (I iCell = 0; iCell < 3; ++iCell)
        {
            geometry.cells.insert(geometry.cells.end(), {1, 0, 0, 0, 0, 0, 0, 1});
            const std::initializer_list<real> metric{.2 * iCell, .1, 0, -1, 0, 0, 1};
            geometry.faceMetrics.insert(geometry.faceMetrics.end(), metric);
            geometry.quadrature.insert(geometry.quadrature.end(), metric);
        }
        const auto stencil = std::make_shared<CauchyTest::CompactStencilFixture>(std::move(geometry), context);
        auto faceRanges = DeviceRows<C::BoundaryFaceRange>({{2, 0, 1, true}, {0, 1, 1, false}, {1, 2, 1, false}});
        auto walls = DeviceRows<real>({.03, .04, .05});
        R::Array<real> means(R::RowLayout::Dense(3, components)), coefficients(R::RowLayout::Dense(3, 2 * components));
        R::Array<real> incoming(R::RowLayout::Dense(3, components)), basis(R::RowLayout::Dense(6, 2));
        {
            auto u = means.Write(), rec = coefficients.Write(), packet = incoming.Write(), b = basis.Write();
            for (I i = 0; i < 3; ++i)
            {
                const auto flow = equation.Gas().Conservative({1.2 + .02 * i, .2, .05, -.03, 1.1});
                const auto incomingFlow = equation.Gas().Conservative({1.2, .2, .05, -.03, 1.1});
                for (int j = 0; j < components; ++j)
                {
                    u.View()(i, j) = j < 5 ? flow[j] : j == 5 ? .02
                                                              : .3;
                    rec.View()(i, 2 * j) = .001 * (j + 1);
                    rec.View()(i, 2 * j + 1) = -.002 * (j + 1);
                    packet.View()(i, j) = j < 5 ? incomingFlow[j] : j == 5 ? .02
                                                                           : .3;
                }
                for (int side = 0; side < 2; ++side)
                {
                    b.View()(2 * i + side, 0) = .2;
                    b.View()(2 * i + side, 1) = -.1;
                }
            }
            // One unlimited VR boundary trace fails thermal admissibility and
            // must use the cell mean before physical BC evaluation.
            rec.View()(0, 8) = -100;
        }
        auto u = means.To(device), rec = coefficients.To(device), packet = incoming.To(device), cache = basis.To(device);
        R::Array<real> output(R::RowLayout::Dense(3, components), device);
        R::Array<real> primitiveOutput(R::RowLayout::Dense(3, 5), device);
        R::Array<DNDS::NumericalStatus> errors(R::RowLayout::Dense(3, 1), device);
        for (const int operation : {0, 1, 2})
        {
            const bool exterior = operation == 1, primitive = operation == 2;
            auto &target = primitive ? primitiveOutput : output;
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            const auto polynomial = DNDS::CFV::Math::TaylorBasis(2, 1).KernelParameters();
            R::SubmitArrays(engine, stream, [components, exterior, primitive, polynomial, frame, stencil](auto execution, auto p, auto bc, auto ranges, auto wall, auto mean, auto coefs, auto incoming, auto basis, auto out, auto status)
                            {
                K::BoundaryTraceRequest request;
                request.frame = frame.KernelParameters(); request.time = .2;
                request.parameters = p; request.conditions = bc; request.stencil = stencil->Descriptor(); request.faces = ranges;
                request.wallMean = wall; request.means = mean; request.coefficients = coefs; request.incoming = incoming;
                request.basis.cached = basis; request.basis.polynomial = polynomial;
                request.basis.availableDerivatives = request.basis.cachedDerivatives = 1;
                request.output = out; request.status = status; request.components = components; request.modes = 2;
                request.polynomial = !primitive; request.centers = primitive; request.hasIncoming = true; request.exterior = exterior;
                if (primitive) K::FillPrimitiveBoundary(execution, request);
                else K::FillBoundaryTraces(execution, request); }, R::ReadAs<R::FixedRows<1>>(params), R::ReadAs<R::FixedRows<1>>(conditionRows), R::ReadAs<R::FixedRows<1>>(faceRanges), R::ReadAs<R::FixedRows<1>>(walls), R::ReadAs<R::TableRows>(u), R::ReadAs<R::TableRows>(rec), R::ReadAs<R::TableRows>(packet), R::ReadAs<R::TableRows>(cache), R::WriteAs<R::TableRows>(target), R::WriteAs<R::FixedRows<1>>(errors));
            engine.Close();
            const auto downloaded = target.To(host);
            const auto downloadedErrors = errors.To(host);
            const auto result = downloaded.Read();
            const auto status = downloadedErrors.Read();
            const auto meanRead = means.Read(), coefficientRead = coefficients.Read(), packetRead = incoming.Read();
            C::BoundaryPointOperator boundary(equation);
            auto evaluator = equation.MakePointEvaluator();
            const I order[]{2, 0, 1};
            for (I iPoint = 0; iPoint < 3; ++iPoint)
            {
                const I iCell = order[iPoint];
                std::vector<real> mean(components), point(components);
                for (int j = 0; j < components; ++j)
                {
                    mean[j] = meanRead.View()(iCell, j);
                    point[j] = primitive ? mean[j] : mean[j] + .2 * coefficientRead.View()(iCell, 2 * j) - .1 * coefficientRead.View()(iCell, 2 * j + 1);
                }
                if (!thermal.View().ClippedReferenceAdmissible(point))
                    point = mean;
                const auto checked = boundary.TryEvaluate(point, conditions[iCell], 0, points[iCell].normal, points[iCell].location,
                                                          evaluator, frame, .2, exterior ? C::BoundaryPointStateKind::ExteriorState : C::BoundaryPointStateKind::FaceTrace,
                                                          DNDS::OptionalRef<const std::vector<real>>(mean), .03 + .01 * iCell, packetRead.View<R::UniformTableRows>());
                REQUIRE(checked);
                if (primitive)
                {
                    C::FlowState flow;
                    for (int j = 0; j < 5; ++j)
                        flow[j] = point[j]; // Primitive gradients consume the physical center FaceTrace.
                    const auto value = equation.Gas().Primitive(flow);
                    point = {value.density, value.velocity[0], value.velocity[1], value.velocity[2], value.temperature};
                }
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                if (!primitive && model == "SA" && iCell == 0)
                    point[5] = 0;
#endif
                REQUIRE(bool(status.View()(iPoint, 0)));
                for (int j = 0; j < (primitive ? 5 : components); ++j)
                    CHECK(result.View()(iPoint, j) == doctest::Approx(point[j]).epsilon(2e-11).scale(1));
            }
        }
    }
}

TEST_CASE("CUDA auxiliary means match generic CPU thermodynamics and RANS transport with ghost gradients")
{
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    for (const std::string model : {"none", "SA", "KOmegaWilcox", "KOmegaSST"})
    {
        INFO("model=" << model);
        C::EquationSettings settings;
        settings.transport = "NewtonianFourier";
        settings.viscosity = .001;
        settings.viscosityModel = "sutherland";
        settings.sutherlandConstant = .3;
        settings.rans.model = model;
        settings.rans.viscosityReference = .01;
        const auto equation = C::EquationSystem::Resolve(settings);
        const int components = 5 + (equation.HasRANS() ? equation.RANS().ComponentNumber() : 0);
        const auto parameters = C::PerfectGasSpatialParameters::FromEquation(equation, 2);
        R::Array<C::PerfectGasSpatialParameters> hostParameters(R::RowLayout::Dense(1, 1));
        {
            auto write = hostParameters.Write();
            write.View()(0, 0) = parameters;
        }
        R::Array<real> own(R::RowLayout::Dense(3, components)), ghost(R::RowLayout::Dense(2, components));
        R::Array<real> ownGradient(R::RowLayout::Dense(3, components * 2)), ghostGradient(R::RowLayout::Dense(2, components * 2));
        const bool viscosityOnly = model == "KOmegaWilcox";
        const bool writeViscosity = viscosityOnly || model == "KOmegaSST";
        const int auxiliaryWidth = writeViscosity ? 4 : 3;
        R::Array<real> wall(R::RowLayout::Dense(5, 1)), expected(R::RowLayout::Dense(5, auxiliaryWidth));
        {
            auto a = own.Write(), b = ghost.Write(), ga = ownGradient.Write(), gb = ghostGradient.Write();
            auto distance = wall.Write(), result = expected.Write();
            auto evaluator = equation.MakePointEvaluator();
            for (I iCell = 0; iCell < 5; ++iCell)
            {
                C::EquationFVDetail::PointScratch point(components);
                const auto flow = equation.Gas().Conservative({1 + .04 * iCell, .21, -.03 * iCell, .07, 1.1 + .01 * iCell});
                for (int j = 0; j < components; ++j)
                {
                    const auto value = j < 5 ? flow[j] : j == 5 ? .013
                                                                : .2;
                    point.values[j] = value;
                    const I row = iCell < 3 ? iCell : iCell - 3;
                    (iCell < 3 ? a.View() : b.View())(row, j) = value;
                    for (int d = 0; d < 2; ++d)
                    {
                        point.gradient(j, d) = .001 * (iCell + 1) * (j + 1) * (d == 0 ? 1 : -.3);
                        (iCell < 3 ? ga.View() : gb.View())(row, 2 * j + d) = point.gradient(j, d);
                    }
                }
                distance.View()(iCell, 0) = .03 + .01 * iCell;
                C::PointProperties properties;
                REQUIRE(bool(evaluator.TryEvaluate(flow, {}, 0, properties)));
                real bound = properties.diffusionBound;
                if (equation.HasRANS())
                {
                    const auto transport = equation.RANS().TryTransport(C::EquationFVDetail::RANSPoint(point, 0, 5, equation.RANS(), properties, distance.View()(iCell, 0), 0));
                    REQUIRE(bool(transport));
                    if (auxiliaryWidth == 4)
                        result.View()(iCell, 3) = transport.value.viscosity;
                    bound = std::max({bound, real(4. / 3.) * (properties.transport.viscosity + transport.value.viscosity) / flow[0],
                                      (properties.transport.conductivity + transport.value.conductivity) / (flow[0] * properties.thermo.cv)});
                }
                result.View()(iCell, 0) = properties.thermo.pressure;
                result.View()(iCell, 1) = properties.thermo.soundSpeed;
                result.View()(iCell, 2) = bound;
            }
        }
        auto p = hostParameters.To(device, context);
        auto u = own.To(device, context), ug = ghost.To(device, context);
        auto g = ownGradient.To(device, context), gg = ghostGradient.To(device, context), w = wall.To(device, context);
        R::Array<real> output(R::RowLayout::Dense(viscosityOnly ? 0 : 5, 3), device);
        R::Array<real> viscosity(R::RowLayout::Dense(writeViscosity ? 5 : 0, 1), device);
        R::Array<DNDS::NumericalStatus> status(R::RowLayout::Dense(5, 1), device);
        R::EagerEngine engine;
        const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        // Both partitions are queued without a host join between them. The
        // same output lease is scheduled twice on one lane, with disjoint rows.
        for (const bool dependent : {false, true})
            R::SubmitArrays(engine, stream, [=](R::ExecutionContext execution, auto params, auto mean, auto halo, auto gradient, auto gradientHalo, auto walls, auto out, auto eddy, auto errors)
                            {
                K::AuxiliaryStateInput input{params, {mean, halo}, {gradient, gradientHalo}, walls, components, 2, true, true};
                K::PrepareAuxiliaryMeans(execution, input, out, eddy, errors, dependent ? 3 : 0, dependent ? 5 : 3); }, R::ReadAs<R::FixedRows<1>>(p), R::ReadAs<R::TableRows>(u), R::ReadAs<R::TableRows>(ug), R::ReadAs<R::TableRows>(g), R::ReadAs<R::TableRows>(gg), R::ReadAs<R::FixedRows<1>>(w), R::WriteAs<R::TableRows>(output), R::WriteAs<R::TableRows>(viscosity), R::WriteAs<R::FixedRows<1>>(status));
        engine.Close();
        const auto result = output.To(R::Device::Host());
        const auto errors = status.To(R::Device::Host());
        const auto hostViscosity = viscosity.To(R::Device::Host());
        const auto actual = result.Read(), reference = expected.Read(), eddy = hostViscosity.Read();
        const auto error = errors.Read();
        for (I iCell = 0; iCell < 5; ++iCell)
        {
            REQUIRE(bool(error.View()(iCell, 0)));
            if (!viscosityOnly)
                for (int j = 0; j < 3; ++j)
                    CHECK(actual.View()(iCell, j) == doctest::Approx(reference.View()(iCell, j)).epsilon(2e-11).scale(1));
            if (writeViscosity)
                CHECK(eddy.View()(iCell, 0) == doctest::Approx(reference.View()(iCell, 3)).epsilon(2e-11).scale(1));
        }
    }
}

TEST_CASE("CUDA face auxiliaries distinguish rotated neighbor physical boundary and interface then reduce by cell")
{
    const auto device = R::Device::CUDA(0);
    const auto context = R::ExecutionContext::CUDA(0);
    C::PerfectGasSpatialParameters parameters;
    parameters.dimension = 2;
    parameters.Validate(5);
    const auto &gas = parameters.gas;
    const auto left = gas.Conservative({1., .3, -.1, .04, 1.});
    const auto neighbor = gas.Conservative({.8, .2, .4, -.03, .7});
    const auto incoming = gas.Conservative({1.2, -.15, .21, 0., 1.3});
    auto params = DeviceRows<C::PerfectGasSpatialParameters>({parameters});
    auto own = DeviceStates({left}), ghost = DeviceStates({neighbor}), packet = DeviceStates({incoming, incoming});
    auto selected = DeviceRows<I>({2, 0, 1}), selectedCells = DeviceRows<I>({0});
    auto kinds = DeviceRows<C::BoundaryKind>({C::BoundaryKind::Interior, C::BoundaryKind::Farfield, C::BoundaryKind::Interface});
    CauchyTest::CompactStencilFixture::Input input;
    input.cellFaces = {{0, 1, 2}};
    input.faceCells = {{{0, 1}}, {{0, DNDS::UnInitIndex}}, {{0, DNDS::UnInitIndex}}};
    input.faceOffsets = {0, 1, 2, 3};
    input.cells = {1, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 1, 0, 0, 1};
    input.codes = {4, 0, 0};
    input.parameters.rotatedGenerators = 1;
    input.parameters.generators.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    input.parameters.generators.translation[1].map().setZero();
    input.routing = {{0, 0, DNDS::UnInitIndex, DNDS::UnInitIndex, DNDS::UnInitIndex},
                     {1, 1, 0, 0, DNDS::UnInitIndex},
                     {2, 2, 1, 1, DNDS::UnInitIndex}};
    for (I iFace = 0; iFace < 3; ++iFace)
    {
        const std::initializer_list<real> metric{.2, .1 * iFace, 0, 1, 0, 0, std::array<real, 3>{1., .8, 1.2}[iFace]};
        input.faceMetrics.insert(input.faceMetrics.end(), metric);
        input.quadrature.insert(input.quadrature.end(), metric);
    }
    CauchyTest::CompactStencilFixture geometry(std::move(input), context);
    R::Array<real> spectral(R::RowLayout::Dense(3, 2), device), wave(R::RowLayout::Dense(3, 1), device), pressure(R::RowLayout::Dense(3, 1), device);
    R::Array<DNDS::NumericalStatus> status(R::RowLayout::Dense(3, 1), device);
    C::RigidFrameParameters frame;
    frame.moving = true;
    frame.angularVelocity = {0, 0, .2};
    frame.centerVelocity = {.03, -.04, 0};
    {
        auto p = params.Read(context);
        auto u = own.Read(context), ug = ghost.Read(context), in = packet.Read(context);
        auto conditions = kinds.Read(context);
        auto selection = selected.Read(context);
        auto lambda = spectral.Write(context), jump = wave.Write(context), minimum = pressure.Write(context);
        auto errors = status.Write(context);
        K::AuxiliaryFaceRequest request;
        request.state.parameters = p.View<R::FixedRows<1>>();
        request.state.mean = {u.View<R::TableRows>(), ug.View<R::TableRows>()};
        request.state.dimension = 2;
        request.stencil = geometry.Descriptor();
        request.faceNumber = 3;
        request.allCells = 2;
        request.conditions = conditions.View<R::FixedRows<1>>();
        request.incoming = in.View<R::TableRows>();
        request.spectral = lambda.View<R::TableRows>();
        request.wave = jump.View<R::TableRows>();
        request.pressure = minimum.View<R::TableRows>();
        request.status = errors.View<R::FixedRows<1>>();
        request.frame = frame;
        request.needSpectral = request.needWave = request.needPressure = true;
        K::PrepareAuxiliaryFaces(context, request, selection.View<R::FixedRows<1>>());
        // This borrowed-view test deliberately owns its explicit completion
        // boundary. The launcher itself must not wait or release these leases.
        auto done = R::Completion::Record(context);
        lambda.Complete(done);
        jump.Complete(done);
        minimum.Complete(done);
        errors.Complete(done);
        done.Wait();
    }
    auto hSpectral = spectral.To(R::Device::Host()), hWave = wave.To(R::Device::Host()), hPressure = pressure.To(R::Device::Host());
    auto hStatus = status.To(R::Device::Host());
    const auto l = hSpectral.Read(), w = hWave.Read(), p = hPressure.Read();
    const auto errors = hStatus.Read();
    real sum = 0, maxJump = 0, minPressure = 1;
    for (I iFace = 0; iFace < 3; ++iFace)
    {
        REQUIRE(bool(errors.View()(iFace, 0)));
        auto right = iFace == 0 ? neighbor : iFace == 1 ? left
                                                        : incoming;
        if (iFace == 0)
        {
            right[1] = -neighbor[2];
            right[2] = neighbor[1];
        }
        const auto a = gas.Primitive(left), b = gas.Primitive(right);
        const real gridSpeed = .03 - .2 * (.1 * iFace);
        const real expectedLambda = std::max(std::abs(a.velocity[0] - gridSpeed) + a.soundSpeed,
                                             std::abs(b.velocity[0] - gridSpeed) + b.soundSpeed);
        const real expectedJump = std::abs(b.velocity[0] - a.velocity[0]) + std::abs(b.soundSpeed - a.soundSpeed);
        const real expectedPressure = std::min(a.pressure, b.pressure);
        CHECK(l.View()(iFace, 0) == doctest::Approx(expectedLambda));
        CHECK(l.View()(iFace, 1) == 0);
        CHECK(w.View()(iFace, 0) == doctest::Approx(expectedJump));
        CHECK(p.View()(iFace, 0) == doctest::Approx(expectedPressure));
        sum += std::array<real, 3>{1., .8, 1.2}[iFace] * expectedLambda;
        maxJump = std::max(maxJump, expectedJump);
        minPressure = std::min(minPressure, expectedPressure);
    }
    R::Array<real> cellLambda(R::RowLayout::Dense(1, 1), device), cellJump(R::RowLayout::Dense(1, 1), device), cellPressure(R::RowLayout::Dense(1, 1), device);
    {
        auto selection = selectedCells.Read(context);
        auto lambda = spectral.Read(context), jump = wave.Read(context), minimum = pressure.Read(context);
        auto ls = cellLambda.Write(context), jm = cellJump.Write(context), pm = cellPressure.Write(context);
        K::AuxiliaryCellRequest request{geometry.Descriptor(), 1, 3, lambda.View<R::TableRows>(), jump.View<R::TableRows>(), minimum.View<R::TableRows>(), {}, ls.View<R::TableRows>(), jm.View<R::TableRows>(), pm.View<R::TableRows>(), .2};
        K::ReduceAuxiliaryCells(context, request, selection.View<R::FixedRows<1>>());
        auto done = R::Completion::Record(context);
        ls.Complete(done);
        jm.Complete(done);
        pm.Complete(done);
        done.Wait();
    }
    const auto cl = cellLambda.To(R::Device::Host()), cj = cellJump.To(R::Device::Host()), cp = cellPressure.To(R::Device::Host());
    CHECK(cl.Read().View()(0, 0) == doctest::Approx(sum + .2));
    CHECK(cj.Read().View()(0, 0) == doctest::Approx(maxJump));
    CHECK(cp.Read().View()(0, 0) == doctest::Approx(minPressure));
}
