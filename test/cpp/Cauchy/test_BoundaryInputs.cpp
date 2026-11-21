#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp"
#include "CompactManagedStencilFixture.hpp"
#include "Cauchy/Input/FlowProfile.hpp"
#include "Cauchy/Input/StateInput.hpp"
#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "../CFV/ReconstructionFixture.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Analytic boundary numerical failures return status through both equation entries" * doctest::test_suite("rank-local"))
{
    C::DomainSettings settings;
    auto &input = settings.boundaries["4"];
    input.kind = "analytic";
    input.profile.kind = "isentropicVortex";
    input.profile.strength = 100;
    input.profile.center = {0, 0, 0};
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::BoundarySystem boundaries(settings.boundaries, equation);
    const auto &condition = boundaries.At(4);
    const auto inside = equation.Gas().Conservative({1, 0, 0, 0, 1});
    const C::Vector3 normal{1, 0, 0};
    const auto fixed = condition.TryExterior(equation.Gas(), inside, normal);
    CHECK_FALSE(fixed);
    CHECK(fixed.status.code == DNDS::NumericalError::InvalidState);
    auto evaluator = equation.MakePointEvaluator({}, {});
    C::BoundaryPointOperator boundary(equation);
    std::vector<DNDS::real> point(inside.begin(), inside.end());
    const auto generic = boundary.TryEvaluate(point, condition, 0, normal, {}, evaluator,
                                              {}, 0, C::BoundaryPointStateKind::ExteriorState);
    CHECK_FALSE(generic);
    CHECK(generic.code == DNDS::NumericalError::InvalidState);
    // Only the checked setup/binding convenience is allowed to throw.
    CHECK_THROWS(condition.Exterior(equation.Gas(), inside, normal));
}

TEST_CASE("DMR bottom producer resolves prescribed and reflecting pieces in world coordinates" * doctest::test_suite("rank-local"))
{
    C::DomainSettings settings;
    settings.boundaries["4"].kind = "doubleMachBottom";
    settings.boundaries["4"].profile.kind = "doubleMachReflection";
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::BoundarySystem boundaries(settings.boundaries, equation);
    const auto &condition = boundaries.At(4);
    CHECK_NOTHROW(condition.ValidateEquationProducer());
    C::BoundaryPointOperator boundary(equation);
    auto evaluator = equation.MakePointEvaluator({}, {});
    const auto inside = equation.Gas().Conservative({2, 3, -4, 1, 5});
    const C::Vector3 normal{0, -1, 0};
    for (const auto kind : {C::BoundaryPointStateKind::FaceTrace, C::BoundaryPointStateKind::ExteriorState})
        for (double x : {.1, 1. / 6, .3})
        {
            CAPTURE(x);
            CAPTURE(int(kind));
            std::vector<DNDS::real> point(inside.begin(), inside.end());
            REQUIRE(boundary.TryEvaluate(point, condition, 0, normal, {x, 0, 0}, evaluator, {}, 0, kind));
            const auto expected = x < 1. / 6
                                      ? C::FlowState{8, 57.157676649772960, -33, 0, 563.5}
                                      : equation.Gas().Conservative({2, 3, kind == C::BoundaryPointStateKind::FaceTrace ? 0. : 4., 1, 5});
            for (int j = 0; j < 5; ++j)
                CHECK(point[j] == doctest::Approx(expected[j]).epsilon(1e-12));
            Eigen::MatrixXd jacobian(5, 5);
            REQUIRE(boundary.TryApproximateExteriorJacobian(jacobian, condition,
                                                            std::vector<DNDS::real>(inside.begin(), inside.end()),
                                                            normal, {x, 0, 0}, {}, 0));
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 5; ++j)
                    CHECK(jacobian(i, j) == (x < 1. / 6 || i != j ? 0. : i == 2 ? -1.
                                                                                : 1.));
        }
    C::RigidMotionSettings motion;
    motion.translationVelocity = {1, 0, 0};
    const auto frame = C::RigidMotion(motion, 2).At(.2);
    CHECK(condition.KindAt({}, DNDS::OptionalRef<const C::RigidFrame>(frame)) == C::BoundaryKind::SlipWall);
}

TEST_CASE("VR wall target selection preserves the selected physical boundary equation")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    for (bool mirrored : {false, true})
    {
        CAPTURE(mirrored);
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.boundaries["7"].kind = "slipWall";
        settings.numerics.reconstruction = "variational";
        settings.numerics.vrBoundaryValue = mirrored ? "physicalMirrored" : "physicalTrace";
        settings.numerics.faceFallback = false;
        settings.numerics.reconstructionIterations = 1;
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.boundaryDerivativeWeights = {1, 0, 0, 0};
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        const auto plan = DNDS::CFV::VariationalReconstructionPlan::ConstructRecCoeff(fv.Metrics(), settings.numerics.cfv);
        auto state = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
        {
            auto write = state.Owned().Write();
            const auto u = write.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < u.RowCount(); ++iCell)
            {
                const auto x = fv.Metrics().GetCellCent(iCell);
                const auto value = equation.Gas().Conservative({1 + .01 * x[0], .02 + .01 * x[1], -.03, .01, 10});
                for (int j = 0; j < 5; ++j)
                    u(iCell, j) = value[j];
            }
        }
        auto expected = plan.BuildWorkspace(5);
        auto expectedCoefficients = plan.BuildCoefficients(5);
        DNDS::Algebra::Fill(expectedCoefficients, 0.);
        auto boundary = plan.BuildBoundaryValues(5);
        {
            const auto read = state.Owned().Read();
            const auto u = read.View<R::UniformTableRows>();
            auto write = boundary.values.Write();
            const auto values = write.View<R::UniformTableRows>();
            for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
            {
                const auto point = plan.GetBoundaryQuadrature(iPoint);
                C::FlowState inside{};
                for (int j = 0; j < 5; ++j)
                    inside[j] = u(point.cell, j);
                auto trace = equation.Gas().Primitive(inside);
                double normalVelocity = 0;
                for (int d = 0; d < 3; ++d)
                    normalVelocity += trace.velocity[d] * point.normal[d];
                // Independent slip target: retain rho/p, remove the normal
                // velocity once for a face trace or twice for a mirrored target.
                for (int d = 0; d < 3; ++d)
                    trace.velocity[d] -= (mirrored ? 2 : 1) * normalVelocity * point.normal[d];
                const auto expectedTrace = equation.Gas().Conservative(
                    {trace.density, trace.velocity[0], trace.velocity[1],
                     trace.velocity[2], trace.pressure});
                for (int j = 0; j < 5; ++j)
                    values(iPoint, j) = expectedTrace[j];
            }
        }
        fv.Metrics().BindHalo(state).Begin().Wait();
        fv.Metrics().BindHalo(expectedCoefficients).Begin().Wait();
        plan.DoReconstructionIter(state, expectedCoefficients, expected, boundary);
        const auto check = [&](const auto &coefficients)
        {
            const auto a = coefficients.Owned().Read(), b = expectedCoefficients.Owned().Read();
            const auto actual = a.template View<R::UniformTableRows>(), oracle = b.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < actual.RowCount(); ++iCell)
                for (I j = 0; j < actual.Width(); ++j)
                    CHECK(actual(iCell, j) == doctest::Approx(oracle(iCell, j)).epsilon(1e-12).scale(1));
        };
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();
        auto logical = C::EquationDofs::Coupled(equation.Schema(), state);
        auto output = C::EquationDofs::Coupled(equation.Schema(), residual);
        auto reconstruction = fv.BuildReconstructionState(logical);
        auto equationWorkspace = fv.BuildWorkspace(logical);
        fv.Evaluate(logical, output, *reconstruction, *equationWorkspace, engine, stream);
        check(fv.ReconstructionGuess(*reconstruction).CoupledField());
    }
}

TEST_CASE("Total inlet resolves physical vector input and applies static density to SA" * doctest::test_suite("rank-local"))
{
    C::DomainSettings input;
    input.equation.rans.model = "SA";
    input.equation.transport = "NewtonianFourier";
    input.equation.viscosity = .01;
    input.equation.gasConstant = 287. * 50 / 100;
    input.equation.referenceScales = {{"rho0", 2}, {"U0", 10}, {"L0", 3}, {"T0", 50}};
    auto &inlet = input.boundaries["7"];
    inlet.kind = "totalConditionInlet";
    inlet.state = {{"type", "totalPT_phy"}, {"state", {101325., 300., -2., 0., 0., 2.}}};
    const auto equation = C::EquationSystem::Resolve(input.equation);
    const auto prepared = C::ResolvePhysicalStateInputs(input, equation, {});
    C::BoundarySystem boundaries(prepared.settings.boundaries, equation);
    C::BoundaryPointOperator boundary(equation);
    auto evaluator = equation.MakePointEvaluator({}, {});
    const auto flow = equation.Gas().Conservative({1, .2, .1, 0, 4});
    for (const auto kind : {C::BoundaryPointStateKind::FaceTrace, C::BoundaryPointStateKind::ExteriorState})
    {
        std::vector<DNDS::real> point(flow.begin(), flow.end());
        point.push_back(7.);
        REQUIRE(boundary.TryEvaluate(point, boundaries.At(7), 0, {1, 0, 0}, {}, evaluator, {}, 0, kind));
        const double totalT = 6, totalP = 101325. / 200;
        const double cp = 1.4 / .4 * input.equation.gasConstant;
        const double temperature = totalT - .05 / (2 * cp);
        const double pressure = totalP * std::pow(temperature / totalT, 3.5);
        CHECK(point[0] == doctest::Approx(pressure / (input.equation.gasConstant * temperature)));
        CHECK(point[1] / point[0] == doctest::Approx(-std::sqrt(.05)));
        CHECK(point[2] == 0);
        CHECK(point[5] == doctest::Approx(2 * point[0]));
    }
    std::vector<DNDS::real> point(flow.begin(), flow.end());
    point.push_back(7.);
    CHECK_FALSE(boundary.TryEvaluate(point, boundaries.At(7), 0, {-1, 0, 0}, {}, evaluator, {}, 0,
                                     C::BoundaryPointStateKind::ExteriorState));
    const auto &condition = boundaries.At(7);
    const auto jacobian = condition.ExteriorJacobian(equation.Gas(), flow, {1, 0, 0}, {}, 0);
    for (int column = 0; column < 5; ++column)
    {
        auto plus = flow, minus = flow;
        const double epsilon = 1e-6;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        const auto a = condition.Exterior(equation.Gas(), plus, {1, 0, 0}, {}, 0);
        const auto b = condition.Exterior(equation.Gas(), minus, {1, 0, 0}, {}, 0);
        for (int row = 0; row < 5; ++row)
            CHECK(jacobian[column * 5 + row] == doctest::Approx((a[row] - b[row]) / (2 * epsilon)).epsilon(2e-6).scale(1));
    }
    Eigen::MatrixXd approximate(6, 6);
    REQUIRE(boundary.TryApproximateExteriorJacobian(approximate, condition, point, {1, 0, 0}, {}, {}, 0));
    CHECK(approximate.isZero());
}

TEST_CASE("An ACM interface publishes negative relative pressure and zero velocity")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 2);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.equation.system = "incompressibleACM";
    settings.equation.eos = "none";
    settings.numerics.reconstruction = "gradient";
    settings.numerics.positivity = false;
    settings.boundaries["7"].kind = "interface";
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto packet = fv.BuildBoundaryInputs();
    for (auto *section : {&packet.FaceCenterValues(), &packet.QuadratureValues()})
    {
        auto write = section->WriteOwned();
        const auto rows = write.Block("flow");
        for (I i = 0; i < rows.RowCount(); ++i)
            for (int j = 0; j < 4; ++j)
                rows(i, j) = j == 3 ? -3. : 0.;
    }
    fv.PublishBoundaryInputs(packet, 0.);
    fv.ValidateBoundaryInputs(packet, 0.);
    const auto read = packet.FaceCenterValues().ReadOwned();
    const auto published = read.Block("flow");
    REQUIRE(published.RowCount() > 0);
    CHECK(published(0, 3) == -3.);
}

TEST_CASE("P1 least squares uses affine receiver face-center values from an interface packet")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.boundaries["7"].kind = "interface";
    settings.numerics.reconstruction = "gradient";
    settings.numerics.gradient = "leastSquares";
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    const auto &metrics = fv.Metrics();
    const auto affine = [&](const auto &p)
    { return equation.Gas().Conservative({1 + .2 * p[0] - .1 * p[1] + .05 * p[2], .3, -.2, .1, 1.}); };
    auto field = metrics.BuildField(5), residual = metrics.BuildField(5);
    auto state = C::EquationDofs::Coupled(equation.Schema(), field);
    auto rhs = C::EquationDofs::Coupled(equation.Schema(), residual);
    {
        const auto geometryRead = metrics.AcquireRead();
        const auto geometry = geometryRead.HostView();
        auto write = field.Owned().Write();
        const auto rows = write.View();
        for (I iCell = 0; iCell < rows.RowCount(); ++iCell)
        {
            const auto value = affine(geometry.GetCellBary(iCell));
            for (int j = 0; j < 5; ++j)
                rows(iCell, j) = value[j];
        }
    }
    auto packet = fv.BuildBoundaryInputs();
    {
        auto write = packet.FaceCenterValues().WriteOwned();
        const auto rows = write.Block("flow");
        const auto &faces = fv.BoundaryFaces();
        REQUIRE(rows.RowCount() == I(faces.size()));
        for (I iFace = 0; iFace < rows.RowCount(); ++iFace)
        {
            const auto value = affine(faces[iFace].point);
            for (int j = 0; j < 5; ++j)
                rows(iFace, j) = value[j];
        }
    }
    fv.PublishBoundaryInputs(packet, 0.);
    auto reconstruction = fv.BuildReconstructionState(state);
    auto workspace = fv.BuildWorkspace(state);
    fv.SetBoundaryInputs(*workspace, packet);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(state, rhs, *reconstruction, *workspace, engine, stream);
    const auto read = residual.Owned().Read();
    constexpr double densityRhs = -(.3 * .2 + (-.2) * (-.1) + .1 * .05);
    for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        CHECK(std::abs(read.View()(iCell, 0) - densityRhs * metrics.GetCellVol(iCell)) < 2e-13);
}

TEST_CASE("Published interface packets reproduce analytic P3 boundaries through auxiliary RHS and frozen Jacobian operators")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.motion.angularVelocity = {.1, -.2, .3};
    settings.motion.center = {.3, .4, .5};
    settings.initialPrimitive = {1, .2, -.1, .05, 1.1};
    settings.initialProfile.kind = "isentropicVortex";
    settings.initialProfile.strength = 1.2;
    settings.initialProfile.center = {.4, .5, 0};
    settings.boundaries["7"].kind = "analytic";
    settings.boundaries["7"].primitive = settings.initialPrimitive;
    settings.boundaries["7"].profile = settings.initialProfile;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.reconstructionIterations = 2;
    settings.numerics.vrBoundaryValue = "physicalMirrored";
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume reference(mesh, settings, equation);
    settings.boundaries["7"].kind = "interface";
    C::FiniteVolume coupled(mesh, settings, equation);
    auto a = reference.Metrics().BuildField(5), b = coupled.Metrics().BuildField(5);
    auto ra = reference.Metrics().BuildField(5), rb = coupled.Metrics().BuildField(5);
    auto equationA = C::EquationDofs::Coupled(equation.Schema(), a);
    auto equationB = C::EquationDofs::Coupled(equation.Schema(), b);
    auto equationRA = C::EquationDofs::Coupled(equation.Schema(), ra);
    auto equationRB = C::EquationDofs::Coupled(equation.Schema(), rb);
    reference.Initialize(equationA, settings.initialProfile, settings.initialPrimitive);
    coupled.Initialize(equationB, settings.initialProfile, settings.initialPrimitive);
    auto reconstructionA = reference.BuildReconstructionState(equationA);
    auto reconstructionB = coupled.BuildReconstructionState(equationB);
    auto workA = reference.BuildWorkspace(equationA);
    auto workB = coupled.BuildWorkspace(equationB);
    auto incoming = coupled.BuildBoundaryInputs();
    constexpr double time = .3;
    CHECK_THROWS(coupled.ValidateBoundaryInputs(incoming, time));
    if (!coupled.BoundaryPoints().empty())
        CHECK_THROWS(coupled.PublishBoundaryInputs(incoming, time));
    const auto frame = coupled.Motion().At(time);
    {
        auto write = incoming.QuadratureValues().WriteOwned();
        const auto fill = [&](const auto &points, const auto &values)
        {
            for (I iPoint = 0; iPoint < I(points.size()); ++iPoint)
            {
                const auto state = frame.LocalState(C::ProfileValue(settings.initialProfile, settings.initialPrimitive, equation.Gas(), frame.WorldPoint(points[iPoint].point), time));
                for (int j = 0; j < 5; ++j)
                    values(iPoint, j) = state[j];
            }
        };
        fill(coupled.BoundaryPoints(), write.Block("flow"));
        // Quadrature values alone cannot publish a packet for centered
        // Jacobians: the donor must explicitly sample the face centers too.
    }
    if (!coupled.BoundaryFaces().empty())
        CHECK_THROWS(coupled.PublishBoundaryInputs(incoming, time));
    {
        auto write = incoming.FaceCenterValues().WriteOwned();
        const auto values = write.Block("flow");
        const auto &points = coupled.BoundaryFaces();
        for (I iFace = 0; iFace < I(points.size()); ++iFace)
        {
            CHECK(points[iFace].quadrature == -1);
            const auto state = frame.LocalState(C::ProfileValue(settings.initialProfile, settings.initialPrimitive, equation.Gas(), frame.WorldPoint(points[iFace].point), time));
            for (int j = 0; j < 5; ++j)
                values(iFace, j) = state[j];
        }
    }
    auto routeOwner = std::make_shared<int>(7);
    C::BoundaryInterfaceSource source;
    source.donorDomain = 3;
    source.receiverDomain = 4;
    source.routeGeneration = 9;
    source.donorPose.translation[0] = .1;
    source.receiverPose.translation[1] = -.2;
    source.routeOwner = routeOwner;
    source.localDonor = std::make_shared<const C::InterfaceDonorPublication>(
        reference.PublishInterfaceDonor(equationA, *reconstructionA, *workA, false));
    const std::weak_ptr<const C::InterfaceDonorPublication> retainedDonor = source.localDonor;
    const auto sampledMeanVersions = source.localDonor->MeanVersions();
    coupled.PublishBoundaryInputs(incoming, time, {0, 1, 2, 0, 0, 0},
                                  {source});
    source.localDonor.reset();
    CHECK_FALSE(retainedDonor.expired());
    REQUIRE(incoming.InterfaceSources().size() == 1);
    CHECK(incoming.InterfaceSources()[0].routeGeneration == 9);
    CHECK(incoming.InterfaceSources()[0].routeOwner == routeOwner);
    coupled.SetBoundaryInputs(*workB, incoming);
    R::EagerEngine engine(2);
    const auto stream = engine.CreateCPUStream();
    reference.Evaluate(equationA, equationRA, *reconstructionA, *workA, engine, stream, time);
    coupled.Evaluate(equationB, equationRB, *reconstructionB, *workB, engine, stream, time);
    const auto compare = [](const auto &left, const auto &right)
    {
        const auto ar = left.Owned().Read(), br = right.Owned().Read();
        const auto u = ar.template View<R::UniformTableRows>(), v = br.template View<R::UniformTableRows>();
        for (I iCell = 0; iCell < u.RowCount(); ++iCell)
            for (I j = 0; j < u.Width(); ++j)
                CHECK(u(iCell, j) == v(iCell, j));
    };
    compare(ra, rb);
    compare(reference.FluxReconstruction(*reconstructionA, *workA).CoupledField(),
            coupled.FluxReconstruction(*reconstructionB, *workB).CoupledField());
    C::AuxiliaryRequest request;
    request.faceSpectralRadius = true;
    auto auxA = reference.BuildAuxiliaryValues(request), auxB = coupled.BuildAuxiliaryValues(request);
    reference.PrepareAuxValues(equationA, auxA, time);
    C::AuxiliaryInputs auxiliaryInputs;
    auxiliaryInputs.boundary.Emplace(incoming);
    coupled.PrepareAuxValues(equationB, auxB, time, std::nullopt,
                             auxiliaryInputs);
    // The split must reproduce the full auxiliary publication exactly, while
    // its independent phase has neither a packet nor permission to read ghosts.
    const C::AuxiliaryRequest all{true, true, true, true, true, true, true};
    auto readyAux = coupled.BuildAuxiliaryValues(all);
    auto deferredAux = coupled.BuildAuxiliaryValues(all);
    coupled.PrepareAuxValues(equationB, readyAux, time, {}, auxiliaryInputs);
    C::AuxiliaryInputs deferredInputs;
    deferredInputs.deferBoundary = true;
    auto preparation = coupled.BeginAuxValues(equationB, deferredAux, time, {}, deferredInputs);
    CHECK_THROWS(preparation.Apply(DNDS::CFV::ReconstructionRegion::Core));
    {
        auto blockedGhost = b.Ghost().Write();
        preparation.ApplyInterfaceIndependent();
        blockedGhost.CancelUnmodified();
    }
    preparation.BindBoundary(incoming);
    preparation.Apply();
    preparation.Finish(deferredAux);
    CHECK_NOTHROW(coupled.ValidateAuxValues(equationB, deferredAux, time));
    const auto arrays = [](const C::AuxiliaryValues &values)
    {
        return std::array<const R::Array<double> *, 7>{
            &*values.meanThermodynamics, &*values.faceSpectralRadius,
            &*values.faceWaveJump, &*values.facePressureMinimum,
            &values.cellSpectralSum->Owned(), &values.cellWaveJump->Owned(),
            &values.cellPressureMinimum->Owned()};
    };
    const auto expectedArrays = arrays(readyAux), actualArrays = arrays(deferredAux);
    for (int k = 0; k < 7; ++k)
    {
        const auto expectedRead = expectedArrays[k]->Read(), actualRead = actualArrays[k]->Read();
        const auto expected = expectedRead.View<R::UniformTableRows>(), actual = actualRead.View<R::UniformTableRows>();
        for (I iRow = 0; iRow < expected.RowCount(); ++iRow)
            for (I j = 0; j < expected.RowSize(iRow); ++j)
                CHECK(actual(iRow, j) == expected(iRow, j));
    }
    R::Array<double> tau(R::RowLayout::Dense(reference.Metrics().NumCell(), 1));
    {
        auto write = tau.Write();
        for (I iCell = 0; iCell < tau.RowCount(); ++iCell)
            write.View()(iCell, 0) = 10;
    }
    auto ja = reference.BuildLinearProblem(equationA, C::DiagonalStorage::Block),
         jb = coupled.BuildLinearProblem(equationB, C::DiagonalStorage::Block);
    const C::JacobianPreparation controls{3, .7, true};
    reference.PrepareJacobian(equationA, auxA, tau, *ja, controls, time, {},
                              C::FiniteVolume::JacobianContext{*reconstructionA, *workA});
    coupled.PrepareJacobian(equationB, auxB, tau, *jb, controls, time, {},
                            C::FiniteVolume::JacobianContext{*reconstructionB, *workB});
    // Full physical field a/b is an independent nonconstant action input.
    ja->Apply(a, ra);
    jb->Apply(b, rb);
    compare(ra, rb);
    settings.numerics.cfv.maxOrder = 1;
    C::FiniteVolume coarse(mesh, settings, equation);
    const auto alias = coarse.AliasBoundaryInputs(incoming);
    CHECK(alias.QuadratureValues().CoupledField().Owned().SharedStorage() ==
          incoming.QuadratureValues().CoupledField().Owned().SharedStorage());
    CHECK(alias.FaceCenterValues().CoupledField().Owned().SharedStorage() ==
          incoming.FaceCenterValues().CoupledField().Owned().SharedStorage());
    CHECK_NOTHROW(coarse.ValidateBoundaryInputs(alias, time));
    CHECK(alias.Epoch().stage == 2);
    REQUIRE(alias.InterfaceSources().size() == 1);
    CHECK(alias.InterfaceSources()[0].routeOwner == routeOwner);
    CHECK(alias.SamePublication(incoming));
    CHECK(coupled.AuxiliaryBoundaryCurrent(auxB, *workB));
    const auto frozen = coupled.FreezeBoundaryInputs(incoming);
    CHECK_FALSE(frozen.SamePublication(incoming));
    coupled.SetBoundaryInputs(*workB, frozen);
    // Same means and physical time do not make an auxiliary cache current
    // after a new interface publication (DITR's cached-rate correction path).
    CHECK_FALSE(coupled.AuxiliaryBoundaryCurrent(auxB, *workB));
    coupled.SetBoundaryInputs(*workB, incoming);
    CHECK(coupled.AuxiliaryBoundaryCurrent(auxB, *workB));
    CHECK(frozen.InterfaceSources()[0].localDonor == retainedDonor.lock());
    CHECK(frozen.QuadratureValues().CoupledField().Owned().SharedStorage() !=
          incoming.QuadratureValues().CoupledField().Owned().SharedStorage());
    CHECK(frozen.FaceCenterValues().CoupledField().Owned().SharedStorage() !=
          incoming.FaceCenterValues().CoupledField().Owned().SharedStorage());
    CHECK_NOTHROW(coupled.ValidateBoundaryInputs(frozen, time));
    // A frozen packet describes the sampled past state. Subsequent donor
    // writes invalidate re-sampling from that publication, not this packet.
    {
        auto write = equationA.WriteOwned();
    }
    CHECK(frozen.InterfaceSources()[0].localDonor->MeanVersions() == sampledMeanVersions);
    CHECK_THROWS(frozen.InterfaceSources()[0].localDonor->Validate());
    CHECK_NOTHROW(coupled.ValidateBoundaryInputs(frozen, time));
    {
        auto write = incoming.FaceCenterValues().WriteOwned(); // Either table's debug write invalidates publication.
    }
    CHECK_THROWS(coupled.ValidateAuxValues(equationB, auxB, time));
    CHECK_NOTHROW(coupled.ValidateBoundaryInputs(frozen, time));
}

TEST_CASE("Physical and interface gradient targets evaluate nonconstant fields at actual face centers")
{
    // Use real three-dimensional P5 face quadrature: on faces tangent to y,
    // y^2 differs at the center and every noncentral quadrature point. A copy
    // of the first quadrature trace or a quadrature average cannot pass.
    const auto mesh = CFVTest::CartesianHexes(mpi, 2);
    CauchyTest::CompactManagedStencilFixture geometry(mesh);
    DNDS::CFV::ReconstructionSettings settings;
    settings.maxOrder = 2;
    settings.cacheDiffBase = true;
    const auto reconstruction = DNDS::CFV::VariationalReconstructionPlan::ConstructRecCoeff(geometry.metrics, settings);
    const auto basisRead = reconstruction.AcquireBasisRead();
    const auto &basis = basisRead.EvaluationView();
    const auto polynomial = DNDS::CFV::Math::TaylorBasis(3, 2).KernelParameters();
    I quadratic = -1;
    for (int mode = 0; mode < reconstruction.NRecDOF(); ++mode)
        if (polynomial.orders[mode + 1] == std::array<int, 3>{0, 2, 0})
            quadratic = mode;
    REQUIRE(quadratic >= 0);
    const C::EquationSystem equation = C::EquationSystem::Resolve(C::EquationSettings{});
    const auto flow = equation.Gas().Conservative({1, 0, 0, 0, 1});
    const I cells = geometry.metrics.NumCell(), faces = geometry.boundaryFaces.RowCount();
    const int modes = reconstruction.NRecDOF();
    R::Array<double> means(R::RowLayout::Dense(cells, 5)), coefficients(R::RowLayout::Dense(cells, 5 * modes)),
        centers(R::RowLayout::Dense(faces, 5)), incoming(R::RowLayout::Dense(faces, 5));
    const auto stencilRead = geometry.stencil.Read();
    const auto stencil = stencilRead.View();
    const auto rangesRead = geometry.boundaryFaces.Read();
    const auto ranges = rangesRead.View<R::FixedRows<1>>();
    {
        auto mean = means.Write(), coefficient = coefficients.Write(), packet = incoming.Write();
        for (I iCell = 0; iCell < cells; ++iCell)
        {
            for (int j = 0; j < 5; ++j)
                mean.View()(iCell, j) = flow[j];
            for (int j = 0; j < 5 * modes; ++j)
                coefficient.View()(iCell, j) = j == quadratic ? .2 : 0;
        }
        for (I iFace = 0; iFace < faces; ++iFace)
        {
            const auto location = stencil.metric.GetFaceCent(ranges(iFace, 0).face);
            const auto value = equation.Gas().Conservative({2 + .2 * location[0] + .3 * location[1] * location[1], 0, 0, 0, 1});
            for (int j = 0; j < 5; ++j)
                packet.View()(iFace, j) = value[j];
        }
    }
    const auto mean = means.Read(), coefficient = coefficients.Read(), packet = incoming.Read();
    const C::EquationBlockReadView meanBlock{{mean.View<R::UniformTableRows>(), 0, 5}, {{}, 0, 5}};
    const C::EquationBlockReadView coefficientBlock{{coefficient.View<R::UniformTableRows>(), 0, 5 * modes}, {{}, 0, 5 * modes}};
    const I logicalFirst = 0;
    const C::BoundaryPointOperator boundary(equation);
    for (const auto kind : {C::BoundaryKind::Extrapolate, C::BoundaryKind::Interface})
    {
        auto evaluator = equation.MakePointEvaluator();
        C::BoundaryCondition condition;
        condition.kind = kind;
        auto output = centers.Write();
        const C::EquationBlockArrayView<double> target{output.View<R::UniformTableRows>(), 0, 5};
        C::CPU::BoundaryTraceRequest request;
        request.stencil = stencil;
        request.faces = ranges;
        request.means = {&meanBlock, 1};
        request.coefficients = {&coefficientBlock, 1};
        request.output = {&target, 1};
        request.logicalFirst = {&logicalFirst, 1};
        request.basis = basis;
        request.componentNumber = 5;
        request.modes = modes;
        request.polynomial = request.centers = true;
        request.incoming = packet.View<R::UniformTableRows>();
        REQUIRE(bool(C::CPU::FillBoundaryTraces(request, {&condition, 1}, boundary, evaluator)));
        I distinguished = 0;
        for (I iFace = 0; iFace < faces; ++iFace)
        {
            const I face = ranges(iFace, 0).face;
            const auto location = stencil.metric.GetFaceCent(face);
            const I cell = stencil.topology.FaceCells(face)[0];
            const auto prepared = DNDS::CFV::Math::TaylorPointEvaluation::FromPoint(
                std::array<double, 3>{location[0], location[1], location[2]},
                std::array<double, 3>{stencil.metric.cells(cell, 1), stencil.metric.cells(cell, 2), stencil.metric.cells(cell, 3)},
                stencil.metric.cells(cell, 7));
            const double centered = flow[0] + .2 * (polynomial.Derivative(0, int(quadratic + 1), prepared) - basis.moments(cell, quadratic + 1));
            if (kind == C::BoundaryKind::Extrapolate)
            {
                CHECK(output.View()(iFace, 0) == doctest::Approx(centered).epsilon(1e-13));
                const I q = stencil.metric.FaceQuadratureBegin(face);
                const double firstQuadrature = flow[0] + .2 * basis.cached(2 * q, quadratic * basis.cachedDerivatives);
                distinguished += std::abs(firstQuadrature - centered) > 1e-6;
            }
            else
                for (int j = 0; j < 5; ++j)
                    CHECK(output.View()(iFace, j) == packet.View()(iFace, j));
        }
        if (kind == C::BoundaryKind::Extrapolate)
            CHECK(DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Sum>(distinguished, mpi.comm) > 0);
    }
}

int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    int status;
    {
        doctest::Context context(argc, argv);
        status = context.run();
    }
    DNDS::MPI::Finalize();
    return status;
}
