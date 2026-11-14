#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Implicit/CUDA/FrozenJacobianKernels.hpp"
#include "Cauchy/Implicit/CUDA/FiniteVolumeJacobianKernels.hpp"
#include "Cauchy/Implicit/CPU/FiniteVolumeJacobianKernels.hpp"
#include "Cauchy/Implicit/FrozenJacobianOperator.hpp"
#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include <cuda_runtime.h>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;
using DNDS::real;

namespace
{
    template <class T, class Fill>
    R::Array<T> Table(I rows, I width, Fill fill)
    {
        R::Array<T> result(R::RowLayout::Dense(rows, width));
        auto write = result.Write();
        const auto view = write.template View<R::TableRows>();
        for (I iRow = 0; iRow < rows; ++iRow)
            for (I j = 0; j < width; ++j)
                view(iRow, j) = fill(iRow, j);
        return result;
    }
    struct Geometry
    {
        R::Array<I> columns{R::RowLayout::Ragged({2, 1, 0, 1})};
        R::Array<I> offsets = Table<I>(5, 1, [](I i, I)
                                       { return std::array<I, 5>{0, 2, 3, 3, 4}[i]; });
        R::Array<C::JacobianPointRecord> points{R::RowLayout::Ragged({2, 1, 1, 1})};
        Geometry()
        {
            auto c = columns.Write();
            c.View()(0, 0) = 1;
            c.View()(0, 1) = 4;
            c.View()(1, 0) = 2;
            c.View()(3, 0) = 0;
            auto p = points.Write();
            const auto view = p.View();
            I sample = 0;
            for (I iBlock = 0; iBlock < 4; ++iBlock)
                for (I j = 0; j < view.RowSize(iBlock); ++j, ++sample)
                {
                    const bool rotated = sample % 2 == 0;
                    // Two images of cell 1 share a block. The rotated image
                    // deliberately mixes selected and unselected momenta.
                    const std::array<real, 9> rotation = rotated ? std::array<real, 9>{0, 1, 0, -1, 0, 0, 0, 0, 1}
                                                                 : std::array<real, 9>{1, 0, 0, 0, 1, 0, 0, 0, 1};
                    view(iBlock, j) = {std::array<I, 4>{1, 4, 2, 0}[iBlock], sample, sample, {.6, .8, 0}, {.2 * sample, -.3, .1}, rotation, .7 + .1 * sample, rotated};
                }
        }
    };
    std::shared_ptr<const C::FrozenJacobianOperator::Geometry> ManagedGeometry(const Geometry &source, R::ExecutionContext context)
    {
        const std::vector<std::vector<I>> columns{{1, 4}, {2}, {}, {0}};
        auto pattern = std::make_shared<C::LinearPatternStorage>(columns,
                                                                 std::vector<I>{1, 2, 3, 0}, std::vector<I>{3, 0, 1, 2}, std::vector<I>{1, 2, 3}, std::vector<I>{0});
        R::Array<C::FiniteVolumeDetail::SampleRecord> samples(R::RowLayout::Dense(5, 1));
        R::Array<I> offsets(R::RowLayout::Dense(5, 1));
        R::Array<real> faceMeans(R::RowLayout::Dense(5, 7));
        {
            const auto c = source.columns.Read(), o = source.offsets.Read();
            const auto p = source.points.Read();
            auto s = samples.Write();
            auto off = offsets.Write();
            auto f = faceMeans.Write();
            I next = 0;
            for (I iCell = 0; iCell < 4; ++iCell)
            {
                off.View()(iCell, 0) = next;
                for (I iBlock = 0; iBlock < c.View().RowSize(iCell); ++iBlock)
                    for (const auto &point : p.View()[o.View()(iCell, 0) + iBlock])
                    {
                        C::FiniteVolumeDetail::SampleRecord value{};
                        value.cell = iCell;
                        value.other = point.other;
                        value.point = next;
                        value.face = next;
                        value.auxiliaryFace = point.auxiliary;
                        value.jacobianColumn = iBlock;
                        value.normal = point.normal;
                        value.ownPoint = point.location;
                        value.rotation = point.rotation;
                        value.changesFrame = point.changesFrame;
                        s.View()(next, 0) = value;
                        for (int j = 0; j < 7; ++j)
                            f.View()(next, j) = j == 6 ? point.weight : 0;
                        ++next;
                    }
            }
            off.View()(4, 0) = next;
            REQUIRE(next == 5);
        }
        const auto s = samples.Read();
        const auto o = offsets.Read();
        const auto f = faceMeans.Read();
        C::FiniteVolumeStencilView stencil;
        stencil.jacobianSamples = s.View<R::FixedRows<1>>();
        stencil.jacobianOffsets = o.View<R::FixedRows<1>>();
        DNDS::CFV::FiniteVolumeHostView metric;
        const auto face = f.View<R::FixedRows<7>>();
        metric.faceMean = {face.data, nullptr, face.rows, face.width, face.stride, nullptr, face.kind, face.device};
        C::JacobianGeometryStorage host(stencil, metric, columns);
        auto resident = std::make_shared<C::JacobianGeometryStorage>(host.Placed(context.device, context));
        return std::make_shared<C::FrozenJacobianOperator::Geometry>(pattern, resident, context);
    }
}

TEST_CASE("CUDA frozen Jacobians match CPU blocks and actions including ALE periodic projection and RANS stress")
{
    Geometry geometry;
    const auto context = R::ExecutionContext::CUDA(0);
    const auto dc = geometry.columns.To(context.device), doff = geometry.offsets.To(context.device);
    const auto dp = geometry.points.To(context.device);
    auto all = Table<I>(4, 1, [](I i, I)
                        { return i; });
    auto core = Table<I>(3, 1, [](I i, I)
                         { return i + 1; });
    auto dependent = Table<I>(1, 1, [](I, I)
                              { return I(0); });
    const auto dall = all.To(context.device), dcore = core.To(context.device), ddep = dependent.To(context.device);
    const auto cRead = geometry.columns.Read(), oRead = geometry.offsets.Read();
    const auto pRead = geometry.points.Read();
    const C::JacobianGeometryView hostGeometry{cRead.View<R::CSRRows>(), oRead.View<R::FixedRows<1>>(), pRead.View<R::CSRRows>()};
    const auto cDevice = dc.Read(context), oDevice = doff.Read(context);
    const auto pDevice = dp.Read(context);
    const C::JacobianGeometryView deviceGeometry{cDevice.View<R::CSRRows>(), oDevice.View<R::FixedRows<1>>(), pDevice.View<R::CSRRows>()};
    const auto allRead = dall.Read(context), coreRead = dcore.Read(context), depRead = ddep.Read(context);
    const C::EquationSettings settings;
    const auto equation = C::EquationSystem::Resolve(settings);
    const auto gas = equation.Gas();
    C::RigidFrame frame;
    frame.moving = true;
    frame.angularVelocity = {.1, -.2, .3};
    frame.centerVelocity = {.03, .01, -.04};
    for (int profile : {5, 6, 7, 50})
        for (bool subset : {false, true})
        {
            CAPTURE(profile);
            CAPTURE(subset);
            const int n = profile == 50 ? 5 : profile;
            auto means = Table<real>(5, n, [&](I i, I j)
                                     {
                const auto flow = gas.Conservative({1 + .1 * i, .3 + .03 * i, -.2, .1, 1.2 + .1 * i});
                return j < 5 ? flow[j] : .02 * (i + j); });
            auto pressure = Table<real>(5, 1, [&](I i, I)
                                        {
                const auto read = means.Read(); C::FlowState state{};
                for (int j = 0; j < 5; ++j) state[j] = read.View()(i, j);
                return gas.Pressure(state); });
            auto derivative = Table<real>(5, n, [&](I i, I j)
                                          {
                const auto read = means.Read(); const auto u = read.View()[i];
                if (j == 0) return (gas.gamma - 1) * .5 * (u[1] * u[1] + u[2] * u[2] + u[3] * u[3]) / (u[0] * u[0]);
                if (j < 4) return -(gas.gamma - 1) * u[j] / u[0];
                return j == 4 ? gas.gamma - 1 : 0.; });
            auto radius = Table<real>(5, 2, [](I i, I j)
                                      { return j == 0 ? 1.7 + .03 * i : .04; });
            auto magnitudes = Table<real>(5, 3, [](I i, I j)
                                          { return .5 + .13 * i + .7 * j; });
            C::CUDA::FrozenJacobianParameters parameters;
            parameters.gas = gas;
            parameters.frame = frame.KernelParameters();
            parameters.components = n;
            parameters.kineticComponent = n == 7 ? 5 : -1;
            parameters.roe = profile == 50;
            parameters.scale = .73;
            if (subset)
                parameters.active[2] = 0;
            const auto m = means.Read(), p = pressure.Read(), d = derivative.Read(), r = radius.Read(), mag = magnitudes.Read();
            C::CPU::EquationJacobianView reference;
            reference.geometry = hostGeometry;
            reference.means = m.View<R::TableRows>();
            reference.pressure = p.View<R::TableRows>();
            reference.pressureDerivative = d.View<R::TableRows>();
            reference.radius = r.View<R::TableRows>();
            reference.frame = frame;
            reference.scale = parameters.scale;
            reference.components = n;
            reference.flowFirst = 0;
            reference.kineticFirst = parameters.kineticComponent;
            reference.allComponents = !subset;
            reference.active = {parameters.active.data(), n};
            A::BlockMatrixStorage expected({2, 1, 0, 1}, n);
            {
                auto write = expected.Write();
                for (I iCell = 0; iCell < 4; ++iCell)
                    for (I iBlock = 0; iBlock < hostGeometry.columns.RowSize(iCell); ++iBlock)
                    {
                        Eigen::MatrixXd block(n, n);
                        if (parameters.roe)
                        {
                            C::CPU::FlowJacobianView flow;
                            flow.geometry = hostGeometry;
                            flow.means = m.View<R::FixedRows<5>>();
                            flow.radius = r.View<R::TableRows>();
                            flow.magnitudes = mag.View<R::TableRows>();
                            flow.frame = frame;
                            flow.scale = parameters.scale;
                            flow.roe = true;
                            flow.selected = subset;
                            for (int j = 0; j < 5; ++j)
                                flow.active[j] = parameters.active[j];
                            C::CPU::EvaluateFlowJacobianBlock(flow, equation, iCell, iBlock, block);
                        }
                        else
                            C::CPU::EvaluateEquationJacobianBlock(reference, iCell, iBlock, block);
                        write.View()(iCell, iBlock) = block;
                    }
            }
            const auto meanOwned = Table<real>(4, n, [&](I i, I j)
                                               { return m.View()(i, j); });
            const auto meanGhost = Table<real>(1, n, [&](I i, I j)
                                               { return m.View()(4 + i, j); });
            const auto ownedDevice = meanOwned.To(context.device), ghostDevice = meanGhost.To(context.device);
            R::Array<real> dm(R::RowLayout::Dense(5, n), context.device), dpr(R::RowLayout::Dense(5, 1), context.device), dd(R::RowLayout::Dense(5, n), context.device);
            R::Array<DNDS::NumericalStatus> meanStatus(R::RowLayout::Dense(5, 1), context.device);
            {
                const auto own = ownedDevice.Read(context), ghost = ghostDevice.Read(context);
                auto meanWrite = dm.Write(context), pressureWrite = dpr.Write(context), derivativeWrite = dd.Write(context);
                auto statusWrite = meanStatus.Write(context);
                C::CUDA::FreezeJacobianMeans(context, gas, own.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>(),
                                             meanWrite.View<R::UniformTableRows>(), pressureWrite.View<R::TableRows>(), derivativeWrite.View<R::TableRows>(), statusWrite.View<R::FixedRows<1>>());
                REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
            }
            {
                const auto checked = meanStatus.To(R::Device::Host());
                const auto read = checked.Read();
                for (I iCell = 0; iCell < 5; ++iCell)
                    REQUIRE(bool(read.View()(iCell, 0)));
            }
            const auto dr = radius.To(context.device), dmag = magnitudes.To(context.device);
            const auto mr = dm.Read(context), pr = dpr.Read(context), der = dd.Read(context), rr = dr.Read(context), mags = dmag.Read(context);
            C::CUDA::FrozenJacobianInput snapshot{deviceGeometry, mr.View<R::UniformTableRows>(), pr.View<R::TableRows>(), der.View<R::TableRows>(),
                                                  rr.View<R::TableRows>(), parameters.roe ? mags.View<R::TableRows>() : C::CUDA::LinearReadRows{}, parameters};
            A::BlockMatrixStorage actual({2, 1, 0, 1}, n, context.device);
            R::Array<DNDS::NumericalStatus> status(R::RowLayout::Dense(4, 1), context.device);
            {
                auto write = actual.Write(context);
                auto st = status.Write(context);
                C::CUDA::AssembleFrozenJacobian(context, snapshot, allRead.View<R::FixedRows<1>>(), write.ArrayView(), st.View<R::FixedRows<1>>());
                REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
            }
            const auto actualHost = actual.Placed(R::Device::Host(), context);
            {
                const auto checked = status.To(R::Device::Host());
                const auto read = checked.Read();
                for (I iCell = 0; iCell < 4; ++iCell)
                    REQUIRE(bool(read.View()(iCell, 0)));
            }
            const auto ar = actualHost.Read(), er = expected.Read();
            for (I iCell = 0; iCell < 4; ++iCell)
                for (I iBlock = 0; iBlock < hostGeometry.columns.RowSize(iCell); ++iBlock)
                    for (int i = 0; i < n; ++i)
                        for (int j = 0; j < n; ++j)
                            CHECK(ar.View()(iCell, iBlock)(i, j) == doctest::Approx(er.View()(iCell, iBlock)(i, j)).epsilon(3e-12).scale(1));
            auto input = Table<real>(4, n, [](I i, I j)
                                     { return .07 * (1 + i - 2 * j); });
            auto ghost = Table<real>(1, n, [](I, I j)
                                     { return -.03 * (j + 1); });
            auto diagonal = Table<real>(4, n * n, [&](I i, I j)
                                        { return j % n == j / n ? 2 + .1 * i : .003 * (j + 1); });
            auto rhs = Table<real>(4, n, [](I i, I j)
                                   { return .4 + .04 * (i + j); });
            const auto dx = input.To(context.device), dg = ghost.To(context.device), db = rhs.To(context.device), diag = diagonal.To(context.device);
            const auto x = input.Read(), g = ghost.Read(), b = rhs.Read(), di = diagonal.Read();
            const auto xr = dx.Read(context), gr = dg.Read(context), br = db.Read(context), dir = diag.Read(context);
            for (bool jacobi : {false, true})
            {
                CAPTURE(jacobi);
                R::Array<real> output(R::RowLayout::Dense(4, n), context.device);
                {
                    auto out = output.Write(context);
                    auto st = status.Write(context);
                    // Exercise the same split launches as production: a core
                    // launch cannot see the vector ghost allocation at all.
                    C::CUDA::ApplyFrozenJacobian(context, snapshot, coreRead.View<R::FixedRows<1>>(), dir.View<R::TableRows>(), true,
                                                 br.View<R::UniformTableRows>(), xr.View<R::UniformTableRows>(), {}, out.View<R::TableRows>(), st.View<R::FixedRows<1>>(), jacobi);
                    C::CUDA::ApplyFrozenJacobian(context, snapshot, depRead.View<R::FixedRows<1>>(), dir.View<R::TableRows>(), true,
                                                 br.View<R::UniformTableRows>(), xr.View<R::UniformTableRows>(), gr.View<R::UniformTableRows>(), out.View<R::TableRows>(), st.View<R::FixedRows<1>>(), jacobi);
                    REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
                }
                const auto result = output.To(R::Device::Host());
                const auto checked = status.To(R::Device::Host());
                const auto resultRead = result.Read();
                const auto checkedRead = checked.Read();
                for (I iCell = 0; iCell < 4; ++iCell)
                {
                    REQUIRE(bool(checkedRead.View()(iCell, 0)));
                    Eigen::VectorXd local(n), remote(n), value(n);
                    for (int j = 0; j < n; ++j)
                        local[j] = x.View()(iCell, j);
                    const Eigen::Map<const Eigen::MatrixXd> block(di.View()[iCell].data(), n, n);
                    if (jacobi)
                        for (int j = 0; j < n; ++j)
                            value[j] = b.View()(iCell, j);
                    else
                        value = block * local;
                    for (I iBlock = 0; iBlock < hostGeometry.columns.RowSize(iCell); ++iBlock)
                    {
                        const I column = hostGeometry.columns(iCell, iBlock);
                        for (int j = 0; j < n; ++j)
                            remote[j] = column < 4 ? x.View()(column, j) : g.View()(column - 4, j);
                        value += (jacobi ? -1 : 1) * er.View()(iCell, iBlock) * remote;
                    }
                    if (jacobi)
                        value = (block * value).eval();
                    for (int j = 0; j < n; ++j)
                        CHECK(resultRead.View()(iCell, j) == doctest::Approx(value[j]).epsilon(4e-12).scale(1));
                }
            }
            if (n == 7 && !subset)
            {
                // Exercise the managed LinearProblem -> resident provider ->
                // eager operation entry, not just a bare launch descriptor.
                const A::DofSpace space(4, 1);
                auto problem = std::make_shared<C::LinearProblemContext>(space, n,
                                                                         std::vector<std::vector<I>>{{1, 4}, {2}, {}, {0}}, std::vector<I>{1, 2, 3},
                                                                         C::DiagonalStorage::Block, C::OffDiagonalStorage::DirectEvaluated, context.device);
                problem->Diagonal().CopyFrom(diag, context).Wait();
                auto provider = std::make_shared<C::FrozenJacobianOperator>(space.Pin(), ManagedGeometry(geometry, context),
                                                                            C::FrozenJacobianOperator::Values{dm, dpr, dd, dr, dmag}, parameters, context);
                problem->PublishOffDiagonalEvaluator(provider);
                A::DofField<real> state(space, dx, dg), result(space, n, 0, context.device);
                R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
                const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
                problem->SubmitApply(engine, stream, state, result, C::LinearRegion::Core);
                problem->SubmitApply(engine, stream, state, result, C::LinearRegion::HaloDependent);
                provider.reset();
                problem->CheckQueuedActions(engine, stream);
                const auto host = result.Owned().To(R::Device::Host());
                const auto read = host.Read();
                for (I iCell = 0; iCell < 4; ++iCell)
                {
                    Eigen::VectorXd value(n), local(n);
                    for (int j = 0; j < n; ++j)
                        local[j] = x.View()(iCell, j);
                    value = Eigen::Map<const Eigen::MatrixXd>(di.View()[iCell].data(), n, n) * local;
                    for (I iBlock = 0; iBlock < hostGeometry.columns.RowSize(iCell); ++iBlock)
                    {
                        const I column = hostGeometry.columns(iCell, iBlock);
                        for (int j = 0; j < n; ++j)
                            local[j] = column < 4 ? x.View()(column, j) : g.View()(column - 4, j);
                        value += er.View()(iCell, iBlock) * local;
                    }
                    for (int j = 0; j < n; ++j)
                        CHECK(read.View()(iCell, j) == doctest::Approx(value[j]).epsilon(5e-12).scale(1));
                }
            }
        }
}

TEST_CASE("CUDA diagonal initialization preserves CPU mass and moving-frame block convention")
{
    const auto context = R::ExecutionContext::CUDA(0);
    const auto cells = Table<real>(4, 8, [](I i, I j)
                                   { return j == 0 ? .7 + .3 * i : 0; });
    const auto tau = Table<real>(4, 1, [](I i, I)
                                 { return .1 * i; });
    const auto rows = Table<I>(4, 1, [](I i, I)
                               { return 3 - i; });
    const auto dc = cells.To(context.device), dt = tau.To(context.device);
    const auto dr = rows.To(context.device);
    C::RigidFrame frame;
    frame.moving = true;
    frame.angularVelocity = {.1, -.2, .3};
    const auto rotation = frame.NegativeSourceJacobian();
    const auto cr = dc.Read(context), tr = dt.Read(context);
    const auto rr = dr.Read(context);
    for (int n : {5, 6, 7})
        for (bool full : {false, true})
        {
            CAPTURE(n);
            CAPTURE(full);
            R::Array<real> diagonal(R::RowLayout::Dense(4, full ? n * n : n), context.device);
            R::Array<DNDS::NumericalStatus> status(R::RowLayout::Dense(4, 1), context.device);
            {
                auto write = diagonal.Write(context);
                auto st = status.Write(context);
                // CFV's metric view intentionally has its own unchecked policy.
                const auto source = cr.View<R::FixedRows<8>>();
                const C::CUDA::JacobianCellMetrics metric{source.data, nullptr, source.rows, source.width, source.stride, nullptr, source.kind, source.device};
                C::CUDA::InitializeJacobianDiagonal(context, metric, tr.View<R::TableRows>(), write.View<R::TableRows>(),
                                                    st.View<R::FixedRows<1>>(), rr.View<R::FixedRows<1>>(), frame.KernelParameters(), .2, .73, n, full);
                REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
            }
            const auto result = diagonal.To(R::Device::Host());
            const auto checked = status.To(R::Device::Host());
            const auto read = result.Read();
            const auto st = checked.Read();
            for (I iCell = 0; iCell < 4; ++iCell)
            {
                REQUIRE(bool(st.View()(iCell, 0)));
                const real volume = .7 + .3 * iCell;
                Eigen::MatrixXd expected = Eigen::MatrixXd::Identity(n, n) * volume * (.1 * iCell + .2);
                if (full)
                    expected.topLeftCorner<5, 5>() += .73 * volume * Eigen::Map<const Eigen::Matrix<real, 5, 5>>(rotation.data());
                else
                    expected.diagonal().head<5>().array() += .73 * volume * std::sqrt(.14);
                for (int j = 0; j < (full ? n * n : n); ++j)
                    CHECK(read.View()(iCell, j) == doctest::Approx(full ? expected.data()[j] : expected(j, j)).epsilon(2e-14).scale(1));
            }
        }
}

TEST_CASE("CUDA centered face diagonal matches CPU wall chains Roe matrices and self periodic contribution")
{
    using Matrix = Eigen::Matrix<real, 5, 5>;
    const auto context = R::ExecutionContext::CUDA(0);
    const auto equation = C::EquationSystem::Resolve(C::EquationSettings{});
    const auto gas = equation.Gas();
    const auto flux = C::Riemann::RiemannFlux::Resolve("Roe_M9");
    C::RigidFrame frame;
    frame.moving = true;
    frame.angularVelocity = {0, 0, .17};
    frame.centerVelocity = {.03, -.02, 0};
    std::array<C::BoundaryCondition, 3> conditions;
    conditions[0].kind = C::BoundaryKind::NoSlipWall;
    conditions[0].velocity = {.04, -.02, .01};
    conditions[0].isothermal = true;
    conditions[0].temperature = 1.8;
    conditions[1].kind = C::BoundaryKind::SlipWall;
    conditions[1].velocity = {-.02, .04, 0};
    conditions[2].kind = C::BoundaryKind::Farfield;
    conditions[2].farfield = gas.Conservative({1.1, .12, .02, 0, 1.3});
    auto zoneData = Table<C::PerfectGasBoundaryParameters>(3, 1, [&](I i, I)
                                                           { return conditions[i].PerfectGasKernelParameters(); });
    auto samples = Table<C::FiniteVolumeDetail::SampleRecord>(4, 1, [](I i, I)
                                                              {
        C::FiniteVolumeDetail::SampleRecord sample{};
        sample.cell = i / 2; sample.other = i == 1 ? 0 : DNDS::UnInitIndex;
        sample.point = i; sample.condition = i < 2 ? 0 : i - 1;
        sample.normal = i < 2 ? C::Vector3{.6, .8, 0} : i == 2 ? C::Vector3{0, 1, 0} : C::Vector3{1, 0, 0};
        sample.ownPoint = {.1 * i, .2, 0};
        sample.rotation = i == 1 ? std::array<real, 9>{0, 1, 0, -1, 0, 0, 0, 0, 1}
                                 : std::array<real, 9>{1, 0, 0, 0, 1, 0, 0, 0, 1};
        sample.changesFrame = i == 1; sample.face = i; sample.auxiliaryFace = i;
        sample.jacobianColumn = DNDS::UnInitIndex; sample.boundaryPoint = DNDS::UnInitIndex;
        return sample; });
    auto offsets = Table<I>(3, 1, [](I i, I)
                            { return 2 * i; });
    auto selected = Table<I>(2, 1, [](I i, I)
                             { return i; });
    auto faces = Table<real>(4, 7, [](I i, I j)
                             { return j == 6 ? .5 + .1 * i : 0; });
    auto means = Table<real>(2, 5, [&](I i, I j)
                             { return gas.Conservative({1.2 + .1 * i, .3, -.1, .04, 1.4})[j]; });
    const auto meanRead = means.Read();
    auto pressure = Table<real>(2, 1, [&](I i, I)
                                {
        C::FlowState u{}; for (int j = 0; j < 5; ++j) u[j] = meanRead.View()(i, j);
        return gas.Pressure(u); });
    auto derivative = Table<real>(2, 5, [&](I i, I j)
                                  {
        C::FlowState u{}; for (int k = 0; k < 5; ++k) u[k] = meanRead.View()(i, k);
        return gas.PressureDerivative(u)[j]; });
    auto radius = Table<real>(4, 2, [](I i, I j)
                              { return j == 0 ? 1.6 + .1 * i : .03; });
    const auto ds = samples.To(context.device);
    const auto dz = zoneData.To(context.device);
    const auto doff = offsets.To(context.device), dsel = selected.To(context.device);
    const auto df = faces.To(context.device), dm = means.To(context.device), dp = pressure.To(context.device), dd = derivative.To(context.device), dr = radius.To(context.device);
    const auto sr = ds.Read(context);
    const auto zr = dz.Read(context);
    const auto offr = doff.Read(context), selr = dsel.Read(context);
    const auto fr = df.Read(context), mr = dm.Read(context), pr = dp.Read(context), der = dd.Read(context), rr = dr.Read(context);
    const auto hostSamples = samples.Read();
    const auto hostOffsets = offsets.Read();
    const auto hostFaces = faces.Read(), hostRadius = radius.Read();
    for (int mode : {0, 1, 2}) // Euler scalar, physical scalar, Roe matrix
        for (bool full : {false, true})
        {
            if (mode == 2 && !full)
                continue;
            CAPTURE(mode);
            CAPTURE(full);
            const auto seed = Table<real>(2, full ? 25 : 5, [&](I, I j)
                                          { return !full || j % 5 == j / 5 ? 3. : 0.; });
            auto diagonal = seed.To(context.device);
            auto statusHost = Table<DNDS::NumericalStatus>(2, 1, [](I, I)
                                                           { return DNDS::NumericalStatus{}; });
            auto status = statusHost.To(context.device);
            R::Array<real> magnitudes(R::RowLayout::Dense(mode == 2 ? 4 : 0, 3), context.device);
            C::CUDA::JacobianFaceInput request;
            request.samples = sr.View<R::FixedRows<1>>();
            request.conditions = zr.View<R::FixedRows<1>>();
            request.offsets = offr.View<R::FixedRows<1>>();
            const auto faceView = fr.View<R::FixedRows<7>>();
            request.faces = {faceView.data, nullptr, faceView.rows, faceView.width, faceView.stride, nullptr, faceView.kind, faceView.device};
            request.means = mr.View<R::UniformTableRows>();
            request.pressure = pr.View<R::TableRows>();
            request.pressureDerivative = der.View<R::TableRows>();
            request.radius = rr.View<R::TableRows>();
            request.parameters.gas = gas;
            request.parameters.frame = frame.KernelParameters();
            request.parameters.scale = .73;
            request.parameters.roe = mode == 2;
            request.parameters.active[2] = 0;
            request.physicalBoundary = mode != 0;
            request.solver = flux.Kind();
            request.time = .2;
            {
                auto out = diagonal.Write(context), waves = magnitudes.Write(context);
                auto st = status.Write(context);
                C::CUDA::AddJacobianFaces(context, request, selr.View<R::FixedRows<1>>(), out.View<R::TableRows>(), full, waves.View<R::TableRows>(), st.View<R::FixedRows<1>>());
                REQUIRE(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) == cudaSuccess);
            }
            const auto actual = diagonal.To(R::Device::Host());
            const auto checked = status.To(R::Device::Host());
            const auto actualRead = actual.Read();
            const auto checkedRead = checked.Read();
            for (I iCell = 0; iCell < 2; ++iCell)
            {
                REQUIRE(bool(checkedRead.View()(iCell, 0)));
                Matrix expected = 3 * Matrix::Identity();
                C::FlowState left{};
                for (int j = 0; j < 5; ++j)
                    left[j] = meanRead.View()(iCell, j);
                for (I entry = hostOffsets.View()(iCell, 0); entry < hostOffsets.View()(iCell + 1, 0); ++entry)
                {
                    const auto &sample = hostSamples.View()(entry, 0);
                    const real factor = .5 * .73 * hostFaces.View()(sample.face, 6);
                    const real diffusion = hostRadius.View()(entry, 1), lambda = hostRadius.View()(entry, 0) + diffusion;
                    const bool boundary = sample.other == DNDS::UnInitIndex;
                    if (mode != 2 && boundary && !mode)
                    {
                        expected.diagonal().array() += factor * lambda;
                        continue;
                    }
                    auto right = left;
                    Matrix rotation = Matrix::Identity();
                    if (boundary)
                        right = conditions[sample.condition].Exterior(gas, left, sample.normal, sample.ownPoint, .2, std::cref(frame));
                    else
                    {
                        rotation.block<3, 3>(1, 1) = Eigen::Map<const Eigen::Matrix3d>(sample.rotation.data());
                        const Eigen::Matrix<real, 5, 1> transformed = rotation * Eigen::Map<const Eigen::Matrix<real, 5, 1>>(left.data());
                        for (int j = 0; j < 5; ++j)
                            right[j] = transformed[j];
                    }
                    Matrix dissipation = lambda * Matrix::Identity();
                    const auto grid = frame.GridVelocity(sample.ownPoint);
                    if (mode == 2)
                    {
                        const auto basis = equation.PrepareCharacteristics(left, right, sample.normal);
                        const auto ld = equation.PrepareFluxState(left), rd = equation.PrepareFluxState(right);
                        const auto model = equation.PrepareRiemann(flux, ld, rd, ld, rd, sample.normal, grid);
                        const auto waves = flux.Evaluate(model.Input(ld, rd, sample.normal, grid), request.fluxSettings);
                        Eigen::Matrix<real, 5, 1> signals;
                        signals << waves.contact, waves.contact, waves.contact, waves.acousticPlus, waves.acousticMinus;
                        dissipation = Eigen::Map<const Matrix>(basis.inverse.data()) * signals.asDiagonal() * Eigen::Map<const Matrix>(basis.forward.data());
                        dissipation.diagonal().array() += diffusion;
                    }
                    expected += factor * dissipation;
                    const auto derivative = equation.PrepareFluxJacobian(right, sample.normal, grid);
                    const Matrix other = factor * (Eigen::Map<const Matrix>(derivative.data()) - dissipation);
                    if (boundary)
                    {
                        const auto exterior = conditions[sample.condition].ApproximateExteriorJacobian(left, sample.normal, sample.ownPoint, .2, std::cref(frame));
                        expected += other * Eigen::Map<const Matrix>(exterior.data());
                    }
                    else
                        expected += other * rotation;
                }
                for (int i = 0; i < 5; ++i)
                    for (int j = 0; j < 5; ++j)
                        if (i == 2 || j == 2)
                            expected(i, j) = real(i == j);
                for (int j = 0; j < (full ? 25 : 5); ++j)
                    CHECK(actualRead.View()(iCell, j) == doctest::Approx(full ? expected.data()[j] : expected(j, j)).epsilon(5e-12).scale(1));
            }
        }
}
