#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Implicit/CPU/JacobianKernels.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "../CFV/RotatedSectorFixture.hpp"
#include "CartesianFixture.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
static DNDS::MPIInfo mpi;
namespace
{
    C::EquationDofs Bind(const C::FiniteVolume &fv, const DNDS::CFV::FVField &field)
    {
        return C::EquationDofs::Coupled(fv.Equation().Schema(), field);
    }
    void InitializeFlow(const C::FiniteVolume &fv, DNDS::CFV::FVField &field,
                        const C::FlowProfileSettings &profile, const C::FlowState &primitive)
    {
        auto state = Bind(fv, field);
        fv.Initialize(state, profile, primitive);
    }
}

TEST_CASE("Six-component frozen face leaves retain EOS coupling and additive matrix and strided product semantics" * doctest::test_suite("rank-local"))
{
    std::array<DNDS::real, 6> mean{1.2, .31, -.18, .07, 3.1, .13};
    // A nonzero composition column rules out a perfect-gas/passive shortcut.
    std::array<DNDS::real, 6> dp{.08, -.09, .04, -.02, .37, .17};
    std::array<DNDS::real, 36> matrix{};
    for (int j = 0; j < 36; ++j)
        matrix[j] = .001 * (j + 1);
    const auto initial = matrix;
    const C::Vector3 normal{.6, .8, 0}, grid{.03, -.07, .02};
    constexpr DNDS::real pressure = 1.1, factor = .27, lambda = 1.8;
    C::CPU::AddFrozenFluxBlock6({mean.data(), 6}, {dp.data(), 6}, pressure, normal, grid, factor, lambda, {matrix.data(), 36});
    const C::FlowState flow{mean[0], mean[1], mean[2], mean[3], mean[4]};
    for (int column = 0; column < 6; ++column)
    {
        C::FlowState direction{};
        if (column < 5)
            direction[column] = 1;
        const auto derivative = C::PrepareCompressibleFluxDerivative(flow, direction, pressure, dp[column], normal, grid);
        DNDS::real speed = 0, deltaSpeed = 0;
        for (int d = 0; d < 3; ++d)
        {
            speed += (flow[1 + d] / flow[0] - grid[d]) * normal[d];
            deltaSpeed += (direction[1 + d] - flow[1 + d] / flow[0] * direction[0]) / flow[0] * normal[d];
        }
        for (int row = 0; row < 6; ++row)
        {
            const auto value = row < 5 ? derivative[row] : C::PrepareAdvectedScalarFluxDerivative(mean[5], DNDS::real(column == 5), speed, deltaSpeed);
            CHECK(matrix[6 * column + row] == doctest::Approx(initial[6 * column + row] + factor * (value - lambda * DNDS::real(row == column))).epsilon(3e-12).scale(1));
        }
    }
    for (DNDS::index stride : {1, 3})
    {
        std::array<DNDS::real, 18> direction{};
        std::array<DNDS::real, 6> product{};
        for (DNDS::index j = 0; j < 6; ++j)
        {
            direction[j * stride] = .03 * (j + 1) - .11;
            product[j] = -.02 * (j + 1);
        }
        const auto original = product;
        C::CPU::AddFrozenFluxProduct6({mean.data(), 6}, {dp.data(), 6}, pressure, normal, grid,
                                      factor, lambda, {direction.data(), 6, stride}, {product.data(), 6});
        for (DNDS::index row = 0; row < 6; ++row)
        {
            DNDS::real expected = original[row];
            for (DNDS::index column = 0; column < 6; ++column)
                expected += (matrix[row + 6 * column] - initial[row + 6 * column]) * direction[column * stride];
            CHECK(product[row] == doctest::Approx(expected).epsilon(3e-12).scale(1));
        }
    }
}

TEST_CASE("Integrated approximate Jacobian matches frozen-spectral flux differences on periodic and physical boundaries")
{
    for (bool periodic : {false, true})
    {
        const auto mesh = periodic ? CauchyTest::PeriodicQuads(mpi, 6) : CFVTest::CartesianHexes(mpi, 3);
        C::DomainSettings settings;
        settings.mesh.dimension = periodic ? 2 : 3;
        settings.numerics.reconstruction = "constant";
        settings.numerics.cfv.maxOrder = 1;
        if (!periodic)
        {
            settings.boundaries["7"].kind = "slipWall";
            settings.motion.angularVelocity = {.2, -.1, .3};
            settings.motion.translationVelocity = {.07, -.03, .02};
            settings.motion.center = {.3, -.2, .1};
        }
        const double time = .4;
        const auto frame = C::RigidMotion(settings.motion, settings.mesh.dimension).At(time);
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::BoundarySystem boundaries(settings.boundaries, equation.Gas());
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = fv.Metrics().BuildField(5), delta = fv.Metrics().BuildField(5), action = fv.Metrics().BuildField(5);
        R::Array<double> inverseTau(R::RowLayout::Dense(fv.Metrics().NumCell(), 1));
        {
            auto write = state.Owned().Write(), direction = delta.Owned().Write(), pseudo = inverseTau.Write();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto point = fv.Metrics().GetCellCent(iCell);
                const auto value = equation.Gas().Conservative({1 + .02 * point[0], .2 + .03 * point[1], -.1, .05, 1 + .01 * point[0]});
                for (int j = 0; j < 5; ++j)
                {
                    write.View()(iCell, j) = value[j];
                    direction.View()(iCell, j) = .03 * (j + 1) * std::cos(2 * point[0] + point[1]) + .02 * point[2];
                }
                pseudo.View()(iCell, 0) = 2 + point[0];
            }
        }
        auto meanBinding = fv.Metrics().BindHalo(state), directionBinding = fv.Metrics().BindHalo(delta);
        meanBinding.Begin().Wait();
        directionBinding.Begin().Wait();
        C::AuxiliaryRequest request;
        request.faceSpectralRadius = request.meanThermodynamics = true;
        // Exercise the same independent flux-difference oracle through the
        // optional shared property table, including a non-thermo column.
        if (periodic)
            request.derived = {C::DerivedQuantity::Temperature, C::DerivedQuantity::MolecularViscosity};
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        auto logicalState = Bind(fv, state);
        auto workspace = fv.BuildWorkspace(logicalState);
        fv.PrepareAuxValues(logicalState, auxiliary, time);
        const auto stateVersion = state.Owned().Version();
        for (auto diagonal : {C::DiagonalStorage::Entries, C::DiagonalStorage::Block})
            for (auto mode : {C::OffDiagonalStorage::Assembled, C::OffDiagonalStorage::SemiMatrixFree})
            {
                // The scalar Euler diagonal intentionally omits physical-BC
                // derivatives. Full block/Roe mode supplies them for this oracle.
                if (!periodic && diagonal == C::DiagonalStorage::Entries)
                    continue;
                auto problem = fv.BuildLinearProblem(logicalState, diagonal, mode);
                C::JacobianPreparation controls;
                controls.physicalMass = 3;
                controls.residualScale = .7;
                controls.roeDissipation = !periodic;
                fv.PrepareJacobian(logicalState, auxiliary, inverseTau, *problem, controls, time, {},
                                   C::FiniteVolume::JacobianContext{*workspace});
                problem->InvertDiagonal();
                problem->Apply(delta, action);
                auto geometry = fv.Metrics().AcquireRead();
                const auto metric = geometry.HostView();
                auto topologyRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
                const auto topology = topologyRead.HostView();
                const auto own = state.Owned().Read(), ghost = state.Ghost().Read();
                const auto direction = delta.Owned().Read(), remoteDirection = delta.Ghost().Read();
                const C::RHSFieldView u{own.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
                const C::RHSFieldView du{direction.View<R::UniformTableRows>(), remoteDirection.View<R::UniformTableRows>()};
                const auto faceRead = auxiliary.faceSpectralRadius->Read(), pseudo = inverseTau.Read(), result = action.Owned().Read();
                const auto radius = faceRead.View();
                const auto perturbed = [&](DNDS::index iCell, double h)
                {
                    C::FlowState value;
                    for (int j = 0; j < 5; ++j)
                        value[j] = u(iCell, j) + h * du(iCell, j);
                    return value;
                };
                const auto residual = [&](DNDS::index iCell, double h)
                {
                    const auto left = perturbed(iCell, h);
                    auto value = frame.RotationSource(left);
                    for (auto &component : value)
                        component *= metric.GetCellVol(iCell);
                    const auto faces = topology.CellFaces(iCell);
                    for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(faces.size()); ++slot)
                    {
                        const auto iFace = faces[slot];
                        const auto auxiliaryFaces = fv.AuxiliaryFaces();
                        const auto found = std::lower_bound(auxiliaryFaces.begin(), auxiliaryFaces.end(), iFace);
                        REQUIRE(found != auxiliaryFaces.end());
                        const auto iAux = found - auxiliaryFaces.begin();
                        const auto other = topology.CellFaceOther(iCell, iFace, slot);
                        const int sign = topology.CellIsFaceBack(iCell, iFace, slot) ? 1 : -1;
                        for (DNDS::index q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
                        {
                            const DNDS::Geom::tPoint normal = sign * metric.GetFaceNorm(q);
                            const C::Vector3 n{normal[0], normal[1], normal[2]};
                            const auto point = metric.GetFacePoint(q);
                            const C::Vector3 location{point[0], point[1], point[2]};
                            const auto gridVelocity = frame.GridVelocity(location);
                            const auto right = other == DNDS::UnInitIndex ? boundaries.At(topology.GetFaceZone(iFace)).Exterior(equation.Gas(), left, n, location, time, DNDS::OptionalRef<const C::RigidFrame>(frame)) : perturbed(other, h);
                            double lambda = radius(iAux, 0) + radius(iAux, 1);
                            if (controls.roeDissipation)
                            {
                                // Full-block Rusanov uses pointwise frozen
                                // signals; the auxiliary radius is a face max.
                                // Rotation makes w.n vary along one face.
                                const auto baseLeft = perturbed(iCell, 0);
                                const auto baseRight = other == DNDS::UnInitIndex ? boundaries.At(topology.GetFaceZone(iFace)).Exterior(equation.Gas(), baseLeft, n, location, time, DNDS::OptionalRef<const C::RigidFrame>(frame)) : perturbed(other, 0);
                                const auto signal = [&](const C::FlowState &u)
                                {
                                    double speed = 0;
                                    for (int d = 0; d < 3; ++d)
                                        speed += (u[1 + d] / u[0] - gridVelocity[d]) * n[d];
                                    return std::abs(speed) + std::sqrt(equation.Gas().gamma * equation.Gas().Pressure(u) / u[0]);
                                };
                                lambda = std::max(signal(baseLeft), signal(baseRight)) + radius(iAux, 1);
                            }
                            const auto a = C::Riemann::PhysicalFlux(equation.PrepareFluxState(left).flow, n, gridVelocity);
                            const auto b = C::Riemann::PhysicalFlux(equation.PrepareFluxState(right).flow, n, gridVelocity);
                            for (int j = 0; j < 5; ++j)
                                value[j] -= .5 * metric.GetFaceWeight(q) * (a[j] + b[j] - lambda * (right[j] - left[j]));
                        }
                    }
                    return value;
                };
                constexpr double h = 1e-6;
                for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                {
                    const auto plus = residual(iCell, h), minus = residual(iCell, -h);
                    for (int j = 0; j < 5; ++j)
                    {
                        const double expected = metric.GetCellVol(iCell) * (pseudo.View()(iCell, 0) + controls.physicalMass) * du(iCell, j) - controls.residualScale * (plus[j] - minus[j]) / (2 * h);
                        CHECK(result.View()(iCell, j) == doctest::Approx(expected).epsilon(8e-8).scale(1));
                    }
                }
                CHECK(state.Owned().Version() == stateVersion);
            }
        auto problem = fv.BuildLinearProblem(Bind(fv, state), C::DiagonalStorage::Entries);
        CHECK_THROWS(fv.PrepareJacobian(Bind(fv, state), auxiliary, inverseTau, *problem, {0, 1, true}));
        CHECK_THROWS(fv.PrepareJacobian(Bind(fv, state), auxiliary, inverseTau, *problem, {-1, 1, false}));
        const auto outputVersion = problem->Diagonal().Version();
        {
            auto writer = state.Owned().Write();
        }
        CHECK_THROWS(fv.PrepareJacobian(Bind(fv, state), auxiliary, inverseTau, *problem));
        CHECK(problem->Diagonal().Version() == outputVersion);
    }
}
namespace
{
    void EqualFields(const DNDS::CFV::FVField &a, const DNDS::CFV::FVField &b)
    {
        const auto x = a.Owned().Read(), y = b.Owned().Read();
        const auto u = x.View<R::UniformTableRows>(), v = y.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < u.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(u(iCell, j) == doctest::Approx(v(iCell, j)).epsilon(8e-13).scale(1));
    }
}
TEST_CASE("Frozen physical semi-matrix-free blocks agree with assembled action, split SGS and local ILU")
{
    for (int meshKind = 0; meshKind < 4; ++meshKind)
    {
        const auto mesh = meshKind == 0 ? CauchyTest::PeriodicQuads(mpi, 6) : meshKind == 1 ? CFVTest::CartesianHexes(mpi, 3)
                                                                          : meshKind == 2   ? CFVTest::Sector(mpi)
                                                                                            : CauchyTest::PeriodicQuads(mpi, 1);
        C::DomainSettings settings;
        settings.mesh.dimension = meshKind == 1 ? 3 : 2;
        settings.numerics.reconstruction = "constant";
        settings.numerics.cfv.maxOrder = 1;
        settings.numerics.flux = "Roe_M1";
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .03;
        if (meshKind == 1)
        {
            settings.boundaries["7"].kind = "noSlipWall";
            settings.motion.angularVelocity = {.2, -.1, .3};
            settings.motion.translationVelocity = {.07, -.03, .02};
        }
        if (meshKind == 2)
            settings.boundaries["12"].kind = settings.boundaries["13"].kind = "slipWall";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        for (int layout = 0; layout < 3; ++layout)
        {
            const auto shape = layout == 0 ? R::RowLayout::Dense(0, 5, 8) : layout == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                        : R::RowLayout::TiledColMajor(0, 5, 32);
            auto state = fv.Metrics().BuildField(shape), delta = fv.Metrics().BuildField(shape), rhs = fv.Metrics().BuildField(shape);
            auto assembled = fv.Metrics().BuildField(shape), evaluated = fv.Metrics().BuildField(shape), split = fv.Metrics().BuildField(shape);
            auto forwardA = fv.Metrics().BuildField(shape), forwardM = fv.Metrics().BuildField(shape);
            R::Array<double> inverseTau(R::RowLayout::Dense(fv.Metrics().NumCell(), 1));
            {
                auto u = state.Owned().Write(), x = delta.Owned().Write(), b = rhs.Owned().Write(), tau = inverseTau.Write();
                for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                {
                    const auto p = fv.Metrics().GetCellCent(iCell);
                    const auto mean = equation.Gas().Conservative({1 + .02 * p[0], .4 + .03 * p[1], -.2 + .01 * p[0], .1, 1 + .02 * p[1]});
                    for (int j = 0; j < 5; ++j)
                    {
                        u.View<R::UniformTableRows>()(iCell, j) = mean[j];
                        x.View<R::UniformTableRows>()(iCell, j) = .02 * (j + 1) * std::cos(2 * p[0] + p[1]) + .01 * p[2];
                        b.View<R::UniformTableRows>()(iCell, j) = .2 + .03 * p[0] + .04 * j;
                    }
                    tau.View()(iCell, 0) = 2 + p[0];
                }
            }
            auto meanHalo = fv.Metrics().BindHalo(state), deltaHalo = fv.Metrics().BindHalo(delta);
            auto forwardAHalo = fv.Metrics().BindHalo(forwardA), forwardMHalo = fv.Metrics().BindHalo(forwardM);
            meanHalo.Begin().Wait();
            deltaHalo.Begin().Wait();
            C::AuxiliaryRequest request;
            request.faceSpectralRadius = request.meanThermodynamics = request.cellWaveJump = request.cellPressureMinimum = true;
            auto auxiliary = fv.BuildAuxiliaryValues(request);
            auto logicalState = Bind(fv, state);
            auto workspace = fv.BuildWorkspace(logicalState);
            fv.PrepareAuxValues(logicalState, auxiliary);
            fv.Metrics().BindHalo(*auxiliary.cellWaveJump).Begin().Wait();
            fv.Metrics().BindHalo(*auxiliary.cellPressureMinimum).Begin().Wait();
            for (int approximation = 0; approximation < 3; ++approximation)
            {
                const auto mode = approximation == 0 ? C::DiagonalStorage::Entries : C::DiagonalStorage::Block;
                auto a = fv.BuildLinearProblem(logicalState, mode);
                auto m = fv.BuildLinearProblem(logicalState, mode, C::OffDiagonalStorage::SemiMatrixFree);
                CHECK_FALSE(m->HasAssembledOffDiagonal());
                CHECK_THROWS(m->OffDiagonal());
                CHECK_THROWS(m->ReadOffDiagonal());
                C::JacobianPreparation controls{3, .7, approximation == 2};
                fv.PrepareJacobian(logicalState, auxiliary, inverseTau, *a, controls, 0, {},
                                   C::FiniteVolume::JacobianContext{*workspace});
                fv.PrepareJacobian(logicalState, auxiliary, inverseTau, *m, controls, 0, {},
                                   C::FiniteVolume::JacobianContext{*workspace});
                if (layout == 1 && approximation < 2)
                {
                    // Compare the logical physical producer with the established
                    // flow5 approximation, including periodic images, self
                    // incidences, moving-frame sources and both storage modes.
                    auto semantic = C::EquationDofs::Coupled(equation.Schema(), state);
                    auto packet = fv.BuildAuxiliaryValues(request);
                    fv.PrepareAuxValues(semantic, packet);
                    for (auto storage : {C::OffDiagonalStorage::Assembled, C::OffDiagonalStorage::SemiMatrixFree})
                    {
                        auto logical = fv.BuildLinearProblem(semantic, mode, storage);
                        fv.PrepareJacobian(semantic, packet, inverseTau, *logical, controls);
                        if (meshKind == 0 && approximation == 0 && storage == C::OffDiagonalStorage::Assembled)
                        {
                            auto unsupported = controls;
                            unsupported.roeDissipation = true;
                            const auto version = logical->Diagonal().Version();
                            CHECK_THROWS(fv.PrepareJacobian(semantic, packet, inverseTau, *logical, unsupported));
                            CHECK(logical->Diagonal().Version() == version);
                        }
                        const auto oldD = a->Diagonal().Read(), newD = logical->Diagonal().Read();
                        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                            for (DNDS::index j = 0; j < oldD.View().RowSize(iCell); ++j)
                                CHECK(newD.View()(iCell, j) == doctest::Approx(oldD.View()(iCell, j)).epsilon(3e-12).scale(1));
                        const auto expected = a->ReadOffDiagonal(), actual = logical->ReadOffDiagonal();
                        expected.WithBlocks([&](const auto &x)
                                            { actual.WithBlocks([&](const auto &y)
                                                                {
                                for (DNDS::index iCell=0;iCell<fv.Metrics().NumCell();++iCell)
                                    for (DNDS::index k=0;k<DNDS::index(a->Columns()[iCell].size());++k)
                                    {
                                        const auto first=x(iCell,k);
                                        const auto second=y(iCell,k);
                                        CHECK((first-second).cwiseAbs().maxCoeff()<3e-12*std::max(1.,first.cwiseAbs().maxCoeff()));
                                    } }); });
                    }
                }
                a->InvertDiagonal();
                m->InvertDiagonal();
                a->Apply(delta, assembled);
                m->Apply(delta, evaluated);
                EqualFields(assembled, evaluated);
                {
                    const auto generation = m->OffDiagonalGeneration(), diagonalVersion = m->Diagonal().Version();
                    auto held = m->ReadOffDiagonal();
                    CHECK_THROWS(m->PublishOffDiagonalEvaluator(m->PreparedOffDiagonalEvaluator()));
                    CHECK_THROWS(fv.PrepareJacobian(Bind(fv, state), auxiliary, inverseTau, *m, controls));
                    CHECK(m->OffDiagonalGeneration() == generation);
                    CHECK(m->Diagonal().Version() == diagonalVersion);
                }
                {
                    auto exchange = deltaHalo.Begin();
                    m->Apply(delta, split, C::LinearRegion::Core);
                    bool ghostColumn = false;
                    for (const auto &row : m->Columns())
                        for (auto column : row)
                            ghostColumn |= column >= fv.Metrics().NumCell();
                    if (ghostColumn)
                        CHECK_THROWS(m->Apply(delta, split, C::LinearRegion::HaloDependent));
                    exchange.Wait();
                    m->Apply(delta, split, C::LinearRegion::HaloDependent);
                }
                EqualFields(assembled, split);
                for (bool middle : {false, true})
                {
                    auto exchange = deltaHalo.Begin();
                    auto sa = a->BeginSGS(rhs, delta, forwardA, assembled, middle);
                    auto sm = m->BeginSGS(rhs, delta, forwardM, evaluated, middle);
                    sa.Lower(C::LinearRegion::Core);
                    sm.Lower(C::LinearRegion::Core);
                    exchange.Wait();
                    sa.Lower(C::LinearRegion::HaloDependent);
                    sm.Lower(C::LinearRegion::HaloDependent);
                    if (middle)
                    {
                        forwardAHalo.Begin().Wait();
                        forwardMHalo.Begin().Wait();
                    }
                    sa.Upper(C::LinearRegion::HaloDependent);
                    sm.Upper(C::LinearRegion::HaloDependent);
                    sa.Upper(C::LinearRegion::Core);
                    sm.Upper(C::LinearRegion::Core);
                    sa.Finish();
                    sm.Finish();
                    EqualFields(assembled, evaluated);
                }
                for (int fill : {0, 1})
                {
                    a->PrepareILUStructure(fill);
                    m->PrepareILUStructure(fill);
                    a->FactorizeILU();
                    m->FactorizeILU();
                    a->ApplyILU(rhs, forwardA, assembled);
                    m->ApplyILU(rhs, forwardM, evaluated);
                    EqualFields(assembled, evaluated);
                    const auto diagonalVersion = m->Diagonal().Version();
                    m->PublishOffDiagonalEvaluator(m->PreparedOffDiagonalEvaluator());
                    CHECK(m->Diagonal().Version() == diagonalVersion);
                    CHECK_FALSE(m->ILUFactorsCurrent());
                    CHECK_THROWS(m->ApplyILU(rhs, forwardM, evaluated));
                    m->FactorizeILU();
                    CHECK(m->ILUFactorsCurrent());
                }
                // A fresh snapshot may reuse geometry without changing action.
                a->Apply(delta, assembled);
                const auto old = m->PreparedOffDiagonalEvaluator();
                fv.PrepareJacobian(logicalState, auxiliary, inverseTau, *m, controls, 0, {},
                                   C::FiniteVolume::JacobianContext{*workspace});
                CHECK(old != m->PreparedOffDiagonalEvaluator());
                m->Apply(delta, evaluated);
                EqualFields(assembled, evaluated);
            }
            // Live mean mutation must not change an already prepared operator.
            auto frozen = fv.BuildLinearProblem(Bind(fv, state), C::DiagonalStorage::Entries, C::OffDiagonalStorage::SemiMatrixFree);
            fv.PrepareJacobian(Bind(fv, state), auxiliary, inverseTau, *frozen);
            frozen->Apply(delta, assembled);
            {
                auto u = state.Owned().Write();
                const auto values = u.View<R::UniformTableRows>();
                for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
                    values(iCell, 1) += .1;
            }
            meanHalo.Begin().Wait();
            frozen->Apply(delta, evaluated);
            EqualFields(assembled, evaluated);
            CHECK_THROWS(fv.PrepareJacobian(Bind(fv, state), auxiliary, inverseTau, *frozen)); // stale aux, before writes
        }
    }
}
TEST_CASE("Selected physical Jacobians are principal matrices after periodic frame transformation")
{
    for (bool rotated : {false, true})
    {
        const auto mesh = rotated ? CFVTest::Sector(mpi) : CauchyTest::PeriodicQuads(mpi, 1);
        C::DomainSettings settings;
        settings.numerics.reconstruction = "constant";
        settings.numerics.cfv.maxOrder = 1;
        if (rotated)
            settings.boundaries["12"].kind = settings.boundaries["13"].kind = "slipWall";
        C::FiniteVolume fv(mesh, settings, C::EquationSystem::Resolve(settings.equation));
        auto state = fv.Metrics().BuildField(5), direction = fv.Metrics().BuildField(5), projected = fv.Metrics().BuildField(5);
        auto expected = fv.Metrics().BuildField(5), actual = fv.Metrics().BuildField(5);
        InitializeFlow(fv, state, {}, {1.1, .4, -.2, .1, 1});
        {
            auto x = direction.Owned().Write(), p = projected.Owned().Write();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    const double value = .03 * (j + 1) * std::cos(fv.Metrics().Cells().LocalToGlobal(iCell) + j);
                    x.View()(iCell, j) = value;
                    p.View()(iCell, j) = j == 1 || j == 4 ? value : 0;
                }
        }
        fv.Metrics().BindHalo(state).Begin().Wait();
        fv.Metrics().BindHalo(direction).Begin().Wait();
        fv.Metrics().BindHalo(projected).Begin().Wait();
        C::AuxiliaryRequest request;
        request.faceSpectralRadius = true;
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        fv.PrepareAuxValues(Bind(fv, state), auxiliary);
        R::Array<double> inverseTau(R::RowLayout::Dense(fv.Metrics().NumCell(), 1));
        {
            auto write = inverseTau.Write();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                write.View()(iCell, 0) = 3;
        }
        auto semantic = C::EquationDofs::Coupled(C::EquationSystem::Resolve(settings.equation).Schema(), state);
        auto workspace = fv.BuildWorkspace(semantic);
        auto packet = fv.BuildAuxiliaryValues(request);
        fv.PrepareAuxValues(semantic, packet);
        for (auto diagonal : {C::DiagonalStorage::Entries, C::DiagonalStorage::Block})
            for (auto storage : {C::OffDiagonalStorage::Assembled, C::OffDiagonalStorage::SemiMatrixFree})
            {
                auto full = fv.BuildLinearProblem(semantic, diagonal, storage),
                     subset = fv.BuildLinearProblem(semantic, diagonal, storage);
                // The logical producer's scalar approximation must project
                // after periodic transformation in both diagonal layouts.
                auto logicalFull = fv.BuildLinearProblem(semantic, diagonal, storage), logicalSubset = fv.BuildLinearProblem(semantic, diagonal, storage);
                C::JacobianPreparation logicalControls{2, .7, false};
                fv.PrepareJacobian(semantic, packet, inverseTau, *logicalFull, logicalControls);
                logicalControls.updateComponents = {4, 1};
                fv.PrepareJacobian(semantic, packet, inverseTau, *logicalSubset, logicalControls);
                auto reference = C::EquationDofs::Coupled(semantic.Schema(), fv.Metrics().BuildField(5));
                auto result = C::EquationDofs::Coupled(semantic.Schema(), fv.Metrics().BuildField(5));
                logicalFull->Apply(C::EquationDofs::Coupled(semantic.Schema(), projected), reference);
                logicalSubset->Apply(C::EquationDofs::Coupled(semantic.Schema(), direction), result);
                {
                    const auto x = direction.Owned().Read(), a = result.CoupledField().Owned().Read(), b = reference.CoupledField().Owned().Read();
                    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                        for (int j = 0; j < 5; ++j)
                            CHECK(a.View()(iCell, j) == doctest::Approx(j == 1 || j == 4 ? b.View()(iCell, j) : x.View()(iCell, j)).epsilon(3e-12).scale(1));
                }
                C::JacobianPreparation controls{2, .7, diagonal == C::DiagonalStorage::Block};
                fv.PrepareJacobian(semantic, auxiliary, inverseTau, *full, controls, 0, {},
                                   C::FiniteVolume::JacobianContext{*workspace});
                controls.updateComponents = {4, 1};
                fv.PrepareJacobian(semantic, auxiliary, inverseTau, *subset, controls, 0, {},
                                   C::FiniteVolume::JacobianContext{*workspace});
                full->Apply(projected, expected);
                subset->Apply(direction, actual);
                const auto x = direction.Owned().Read(), a = actual.Owned().Read(), b = expected.Owned().Read();
                for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        CHECK(a.View()(iCell, j) == doctest::Approx(j == 1 || j == 4 ? b.View()(iCell, j) : x.View()(iCell, j)).epsilon(1e-13).scale(1));
                // Every stored/evaluated entry is compared to the original
                // matrix, independent of the action implementation above.
                const auto fd = full->Diagonal().Read(), sd = subset->Diagonal().Read();
                for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        for (int k = 0; k < 5; ++k)
                        {
                            if (diagonal == C::DiagonalStorage::Entries && j != k)
                                continue;
                            const auto slot = diagonal == C::DiagonalStorage::Block ? 5 * k + j : j;
                            const bool selected = (j == 1 || j == 4) && (k == 1 || k == 4);
                            CHECK(sd.View()(iCell, slot) == (selected ? fd.View()(iCell, slot) : double(j == k)));
                        }
                auto fRead = full->ReadOffDiagonal(), sRead = subset->ReadOffDiagonal();
                fRead.WithBlocks([&](const auto &f)
                                 { sRead.WithBlocks([&](const auto &s)
                                                    {
                    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                        for (DNDS::index entry = 0; entry < DNDS::index(full->Columns()[iCell].size()); ++entry)
                        {
                            const auto fb = f(iCell, entry); const auto sb = s(iCell, entry);
                            for (int j = 0; j < 5; ++j)
                                for (int k = 0; k < 5; ++k)
                                    CHECK(sb(j, k) == ((j == 1 || j == 4) && (k == 1 || k == 4) ? fb(j, k) : 0));
                        } }); });
            }
    }
}
int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
