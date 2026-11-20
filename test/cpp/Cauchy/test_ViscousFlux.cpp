#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "Geom/Access/EntityWorkPartition.hpp"
#include "CFV/Math/TaylorBasis.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "CartesianFixture.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
static DNDS::MPIInfo mpi;

TEST_CASE("P3 viscous face derivatives reproduce polynomial heat conduction and Couette production")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 5);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.gasConstant = 2.5;
    settings.equation.viscosity = .02;
    settings.boundaries["7"].kind = "extrapolate";
    settings.numerics.reconstruction = "variational";
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.intOrder = 6;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    const auto &gas = equation.Gas();
    C::FiniteVolume fv(mesh, settings, equation);
    const auto rec = DNDS::CFV::VariationalReconstructionPlan::ConstructRecCoeff(fv.Metrics(), settings.numerics.cfv);
    const auto partition = DNDS::Geom::EntityWorkPartition::CellNodeCell(*mesh);
    auto topologyRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
    const auto topology = topologyRead.HostView();
    auto metricsRead = fv.Metrics().AcquireRead();
    const auto metric = metricsRead.HostView();
    constexpr double h = .2, shear = .3, ax = .2, by = .1;
    const auto valueAt = [&](const DNDS::Geom::tPoint &point)
    {
        const double temperature = 1 + ax * point[0] * point[0] + by * point[1] * point[1] * point[1] + .05 * point[2];
        return gas.Conservative({1, .2 + shear * point[1], 0, 0, gas.gasConstant * temperature});
    };
    for (int layout = 0; layout < 3; ++layout)
    {
        const auto shape = [&](DNDS::index width)
        {
            return layout == 0   ? R::RowLayout::Dense(0, width, width + 3)
                   : layout == 1 ? R::RowLayout::ColMajor(0, width, {8, 1})
                                 : R::RowLayout::TiledColMajor(0, width, 4);
        };
        auto mean = fv.Metrics().BuildField(shape(5)), coefficients = fv.Metrics().BuildField(shape(5 * rec.NRecDOF()));
        auto full = fv.Metrics().BuildField(shape(5)), split = fv.Metrics().BuildField(shape(5)), inviscid = fv.Metrics().BuildField(shape(5));
        {
            auto u = mean.Owned().Write(), c = coefficients.Owned().Write();
            const auto state = u.View<R::UniformTableRows>(), coefficient = c.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto center = metric.GetCellCent(iCell);
                auto average = valueAt(center);
                // Exact conservative volume average; temperature is cubic and
                // kinetic energy is quadratic. This is not a point-value input.
                average[4] += gas.gasConstant / (gas.gamma - 1) * (ax * h * h / 12 + by * center[1] * h * h / 4) + .5 * shear * shear * h * h / 12;
                for (int j = 0; j < 5; ++j)
                    state(iCell, j) = average[j];
                DNDS::index rows = 0;
                for (auto iFace : topology.CellFaces(iCell))
                    rows += metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
                Eigen::MatrixXd basis(rows, rec.NRecDOF()), values(rows, 5);
                DNDS::index row = 0;
                const auto faces = topology.CellFaces(iCell);
                for (DNDS::index iSlot = 0; iSlot < DNDS::index(faces.size()); ++iSlot)
                {
                    const auto iFace = faces[iSlot];
                    const int side = topology.CellIsFaceBack(iCell, iFace, iSlot) ? 0 : 1;
                    for (auto q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q, ++row)
                    {
                        const auto b = rec.GetCachedFaceBasis(iFace, side, int(q - metric.FaceQuadratureBegin(iFace)));
                        basis.row(row) = b.row(0);
                        const auto value = valueAt(metric.GetFacePoint(q));
                        for (int j = 0; j < 5; ++j)
                            values(row, j) = value[j] - average[j];
                    }
                }
                // An independent point-value fit supplies an exact polynomial;
                // the production RHS must obtain its derivatives from the
                // derivative cache, not reuse this fitting calculation.
                const auto qr = basis.colPivHouseholderQr();
                REQUIRE(qr.rank() == rec.NRecDOF());
                const Eigen::MatrixXd exact = qr.solve(values);
                CHECK((basis * exact - values).norm() < 1e-12);
                for (int j = 0; j < 5; ++j)
                    for (int mode = 0; mode < rec.NRecDOF(); ++mode)
                        coefficient(iCell, j * rec.NRecDOF() + mode) = exact(mode, j);
            }
        }
        auto meanHalo = fv.Metrics().BindHalo(mean), recHalo = fv.Metrics().BindHalo(coefficients);
        const auto rhs = fv.AcquireRHSOperator();
        for (int workers : {0, 2})
        {
            R::EagerEngine engine(workers);
            const auto stream = engine.CreateCPUStream();
            auto owned = mean.Owned().Read(), recOwned = coefficients.Owned().Read();
            C::RHSReadViews input;
            input.mean.owned = owned.View<C::RHSFlowRows>();
            input.coefficients.owned = recOwned.View<R::UniformTableRows>();
            C::RHSLocalDiagnostics diagnostics;
            auto splitWrite = split.Owned().Write();
            auto meanOperation = meanHalo.Begin(), recOperation = recHalo.Begin();
            engine.Submit(stream, [rhs, input, output = splitWrite.View<C::RHSFlowRows>(), diagnostics = std::ref(diagnostics)](auto)
                          { diagnostics.get().Merge(rhs.ApplyBorrowedChecked(input, {output}, DNDS::CFV::ReconstructionRegion::Core)); });
            meanOperation.Wait();
            recOperation.Wait();
            auto ghost = mean.Ghost().Read(), recGhost = coefficients.Ghost().Read();
            input.mean.ghost = ghost.View<C::RHSFlowRows>();
            input.coefficients.ghost = recGhost.View<R::UniformTableRows>();
            engine.Submit(stream, [rhs, input, output = splitWrite.View<C::RHSFlowRows>(), diagnostics = std::ref(diagnostics)](auto)
                          { diagnostics.get().Merge(rhs.ApplyBorrowedChecked(input, {output}, DNDS::CFV::ReconstructionRegion::HaloDependent)); });
            auto fullWrite = full.Owned().Write(), inviscidWrite = inviscid.Owned().Write();
            engine.Submit(stream, [rhs, input, output = fullWrite.View<C::RHSFlowRows>(), base = inviscidWrite.View<C::RHSFlowRows>(), diagnostics = std::ref(diagnostics)](auto) mutable
                          {
                diagnostics.get().Merge(rhs.ApplyBorrowedChecked(input, {output}));
                input.includeViscosity = false;
                diagnostics.get().Merge(rhs.ApplyBorrowedChecked(input, {base})); });
            stream.Record().WaitHost();
            CHECK(diagnostics.FaceFallbacks() == 0);
        }
        const auto actual = full.Owned().Read(), staged = split.Owned().Read(), base = inviscid.Owned().Read();
        DNDS::index tested = 0;
        const double conductivity = gas.viscosity * gas.gamma * gas.gasConstant / ((gas.gamma - 1) * gas.prandtl);
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            for (int j = 0; j < 5; ++j)
                CHECK(actual.View<R::UniformTableRows>()(iCell, j) == staged.View<R::UniformTableRows>()(iCell, j));
            const auto center = metric.GetCellCent(iCell);
            if ((center.array() > .2).all() && (center.array() < .8).all())
            {
                ++tested;
                for (int j = 0; j < 4; ++j)
                    CHECK(std::abs(actual.View<R::UniformTableRows>()(iCell, j) - base.View<R::UniformTableRows>()(iCell, j)) < 1e-13);
                const double expected = gas.viscosity * shear * shear + conductivity * (2 * ax + 6 * by * center[1]);
                CHECK((actual.View<R::UniformTableRows>()(iCell, 4) - base.View<R::UniformTableRows>()(iCell, 4)) / metric.GetCellVol(iCell) == doctest::Approx(expected).epsilon(2e-10));
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &tested, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
        CHECK(tested == 27);
    }
}

TEST_CASE("Direct P1 Barth preserves an independently selectable unlimited viscous gradient")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    const auto run = [&](bool viscousLimited)
    {
        C::DomainSettings settings;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .02;
        settings.numerics.reconstruction = "gradient";
        settings.numerics.limiter = "Barth";
        settings.numerics.positivity = false;
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.viscousLimited = viscousLimited;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto mean = fv.Metrics().BuildField(R::RowLayout::Dense(0, 5));
        auto rhs = fv.Metrics().BuildField(mean.Owned().Layout());
        {
            auto write = mean.Owned().Write();
            for (DNDS::index iCell = 0; iCell < mean.Owned().RowCount(); ++iCell)
            {
                // An oblique shear patch exercises tangential derivatives as well
                // as the face-normal jump correction on this orthogonal mesh.
                const auto x = fv.Metrics().GetCellCent(iCell);
                const auto u = equation.Gas().Conservative({1, 0, x[0] + x[1] < .8 ? .3 : 0., 0, 1});
                for (int j = 0; j < 5; ++j)
                    write.View<R::UniformTableRows>()(iCell, j) = u[j];
            }
        }
        auto state = C::EquationDofs::Coupled(equation.Schema(), mean);
        auto result = C::EquationDofs::Coupled(equation.Schema(), rhs);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state);
        R::EagerEngine engine;
        fv.Evaluate(state, result, *reconstruction, *work, engine, engine.CreateCPUStream());
        return rhs.Owned();
    };
    const auto limited = run(true), unlimited = run(false);
    const auto a = limited.Read(), b = unlimited.Read();
    double difference = 0.;
    for (DNDS::index iCell = 0; iCell < limited.RowCount(); ++iCell)
        for (int j = 1; j <= 2; ++j)
            difference = std::max(difference, std::abs(a.View()(iCell, j) - b.View()(iCell, j)));
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    CHECK(difference > 1e-10);
}

TEST_CASE("Polynomial viscosity exchanges the selected limited or unlimited coefficients after the last VR commit")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (bool limited : {false, true})
        for (bool overlap : {false, true})
            for (int layout = 0; layout < 3; ++layout)
            {
                C::DomainSettings settings;
                settings.execution.overlap = overlap;
                settings.equation.transport = "NewtonianFourier";
                settings.equation.viscosity = .02;
                settings.numerics.reconstruction = "variational";
                settings.numerics.viscousGradient = "polynomial";
                settings.numerics.viscousLimited = limited;
                settings.numerics.limiter = "WBAP";
                settings.numerics.characteristicLimiter = false;
                settings.numerics.reconstructionIterations = 5;
                settings.numerics.cfv.maxOrder = 3;
                const auto equation = C::EquationSystem::Resolve(settings.equation);
                C::FiniteVolume fv(mesh, settings, equation);
                const auto shape = layout == 0   ? R::RowLayout::Dense(0, 5, 8)
                                   : layout == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                 : R::RowLayout::TiledColMajor(0, 5, 4);
                auto mean = fv.Metrics().BuildField(shape), actual = fv.Metrics().BuildField(shape), expected = fv.Metrics().BuildField(shape), alternative = fv.Metrics().BuildField(shape);
                {
                    auto write = mean.Owned().Write();
                    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    {
                        const auto center = fv.Metrics().GetCellCent(iCell);
                        const auto value = equation.Gas().Conservative({1 + .1 * std::sin(2 * DNDS::pi * center[0]), .2, -.1, .05,
                                                                        1 + .15 * std::cos(2 * DNDS::pi * center[1])});
                        for (int j = 0; j < 5; ++j)
                            write.View<R::UniformTableRows>()(iCell, j) = value[j];
                    }
                }
                auto state = C::EquationDofs::Coupled(equation.Schema(), mean);
                auto result = C::EquationDofs::Coupled(equation.Schema(), actual);
                auto reconstruction = fv.BuildReconstructionState(state);
                auto work = fv.BuildWorkspace(state);
                R::EagerEngine engine(overlap ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
                const auto stream = engine.CreateCPUStream();
                fv.Evaluate(state, result, *reconstruction, *work, engine, stream);
                // Independently refresh both published coefficient fields before
                // the direct stateless operator. A stale unlimited halo from the
                // final VR commit would disagree with this reference on np>1.
                auto flux = fv.FluxReconstruction(*reconstruction, *work).CoupledField();
                auto guess = fv.ReconstructionGuess(*reconstruction).CoupledField();
                auto fluxOperation = fv.Metrics().BindHalo(flux).Begin();
                fluxOperation.Wait();
                auto guessOperation = fv.Metrics().BindHalo(guess).Begin();
                guessOperation.Wait();
                const auto u = mean.Owned().Read(), ug = mean.Ghost().Read(), c = flux.Owned().Read(), cg = flux.Ghost().Read();
                const auto v = guess.Owned().Read(), vg = guess.Ghost().Read();
                C::RHSReadViews input;
                input.mean = {u.View<C::RHSFlowRows>(), ug.View<C::RHSFlowRows>()};
                input.coefficients = {c.View<R::UniformTableRows>(), cg.View<R::UniformTableRows>()};
                if (!limited)
                    input.viscousCoefficients = C::RHSFieldView{v.View<R::UniformTableRows>(), vg.View<R::UniformTableRows>()};
                C::RHSLocalDiagnostics diagnostics;
                {
                    auto write = expected.Owned().Write();
                    const auto rhs = fv.AcquireRHSOperator();
                    diagnostics.Merge(rhs.ApplyBorrowedChecked(input, {write.View<C::RHSFlowRows>()}));
                    auto bad = input;
                    bad.viscousCoefficients = C::RHSFieldView{};
                    CHECK_THROWS(rhs.ApplyBorrowedChecked(bad, {write.View<C::RHSFlowRows>()}));
                }
                if (!limited)
                {
                    auto write = alternative.Owned().Write();
                    input.viscousCoefficients.reset();
                    diagnostics.Merge(fv.AcquireRHSOperator().ApplyBorrowedChecked(input, {write.View<C::RHSFlowRows>()}));
                }
                const auto a = actual.Owned().Read(), b = expected.Owned().Read(), other = alternative.Owned().Read();
                int differs = 0;
                for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                    {
                        CHECK(a.View<R::UniformTableRows>()(iCell, j) == b.View<R::UniformTableRows>()(iCell, j));
                        if (!limited)
                            differs |= std::abs(a.View<R::UniformTableRows>()(iCell, j) - other.View<R::UniformTableRows>()(iCell, j)) > 1e-12;
                    }
                if (!limited)
                {
                    MPI_Allreduce(MPI_IN_PLACE, &differs, 1, MPI_INT, MPI_MAX, mpi.comm);
                    CHECK(differs == 1);
                }
                CHECK(diagnostics.FaceFallbacks() == 0);
            }
}
TEST_CASE("Viscous policy disagreement is rejected before differing transfer schedules are bound")
{
    if (mpi.size == 1)
        return;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 2);
    for (int field = 0; field < 3; ++field)
    {
        C::DomainSettings settings;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .02;
        if (field == 0)
            settings.numerics.viscousGradient = mpi.rank == 0 ? "polynomial" : "primitive";
        if (field == 1)
            settings.numerics.viscousLimited = mpi.rank != 0;
        if (field == 2)
            settings.numerics.noGRPOnWall = mpi.rank == 0;
        C::FiniteVolume fv(mesh, settings, C::EquationSystem::Resolve(settings.equation));
        auto mean = fv.Metrics().BuildField(5);
        auto state = C::EquationDofs::Coupled(fv.Equation().Schema(), mean);
        CHECK_THROWS(fv.BuildWorkspace(state));
    }
}
TEST_CASE("Unified equation orchestration preserves primitive-gradient flow5 viscosity")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (bool overlap : {false, true})
    {
        CAPTURE(overlap);
        C::DomainSettings settings;
        settings.execution.overlap = overlap;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .015;
        settings.numerics.reconstruction = "variational";
        settings.numerics.viscousGradient = "primitive";
        settings.numerics.reconstructionIterations = 4;
        settings.numerics.cfv.maxOrder = 2;
        settings.numerics.faceFallback = false;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto raw = fv.Metrics().BuildField(5), expected = fv.Metrics().BuildField(5);
        {
            auto write = raw.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const auto center = fv.Metrics().GetCellCent(iCell);
                const auto value = equation.Gas().Conservative(
                    {1 + .05 * std::sin(2 * DNDS::pi * center[0]),
                     .2 + .1 * center[1], -.08 + .04 * center[0], .03,
                     1 + .07 * std::cos(2 * DNDS::pi * center[1])});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = value[j];
            }
        }
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(5));
        auto generic = C::EquationDofs::Coupled(equation.Schema(), raw);
        C::EquationDofOperations::Copy(state, generic);
        auto expectedState = C::EquationDofs::Coupled(equation.Schema(), expected);
        auto actual = C::EquationDofOperations::NewVector(state);
        auto genericWork = fv.BuildWorkspace(
            generic, std::nullopt,
            C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic});
        auto genericReconstruction = fv.BuildReconstructionState(generic);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state);
        R::EagerEngine engine(overlap ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(generic, expectedState, *genericReconstruction, *genericWork, engine, stream);
        fv.Evaluate(state, actual, *reconstruction, *work, engine, stream);
        const auto a = actual.CoupledField().Owned().Read(), b = expected.Owned().Read();
        const auto av = a.View<R::UniformTableRows>(), bv = b.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < av.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(av(iCell, j) == doctest::Approx(bv(iCell, j)).epsilon(2e-12).scale(1));
    }
}
TEST_CASE("Polynomial wall viscosity has the Euler normal jump and noGRP switch")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    for (int policy = 0; policy < 3; ++policy)
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .03;
        settings.boundaries["7"].kind = policy == 0 ? "farfield" : "noSlipWall";
        settings.initialPrimitive = {1, .4, -.2, .1, 2};
        settings.boundaries["7"].primitive = settings.initialPrimitive;
        settings.numerics.reconstruction = "constant";
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.noGRPOnWall = policy == 2;
        settings.numerics.cfv.maxOrder = 1;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto mean = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5), inviscid = fv.Metrics().BuildField(5);
        auto state = C::EquationDofs::Coupled(equation.Schema(), mean);
        auto output = C::EquationDofs::Coupled(equation.Schema(), residual);
        fv.InitializeConstant(state, settings.initialPrimitive);
        auto reconstruction = fv.BuildReconstructionState(state);
        auto work = fv.BuildWorkspace(state);
        R::EagerEngine engine(2);
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(state, output, *reconstruction, *work, engine, stream);
        {
            const auto u = mean.Owned().Read(), ug = mean.Ghost().Read();
            C::RHSReadViews input;
            input.mean = {u.View<C::RHSFlowRows>(), ug.View<C::RHSFlowRows>()};
            // The constant inviscid reference has no modal coefficients or
            // gradients. Keep those inputs empty; the wall-jump oracle below
            // isolates the viscous contribution from the full evaluation.
            input.includeViscosity = false;
            auto output = inviscid.Owned().Write();
            C::RHSLocalDiagnostics diagnostics;
            diagnostics.Merge(fv.AcquireRHSOperator().ApplyBorrowedChecked(input, {output.View<C::RHSFlowRows>()}));
            CHECK(diagnostics.FaceFallbacks() == 0);
        }
        const auto actual = residual.Owned().Read(), base = inviscid.Owned().Read();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            for (int j = 0; j < 5; ++j)
            {
                // Unit cube: d_GRP=1 on each wall. Six mirrored velocity
                // jumps give sum(tau*n)=-(40/3)*mu*u. The wall velocity and
                // adiabatic heat flux are zero, so its energy flux is zero.
                const double expected = policy == 1 && j >= 1 && j <= 3
                                            ? -(40. / 3) * equation.Gas().viscosity * settings.initialPrimitive[j]
                                            : 0;
                CHECK(actual.View()(iCell, j) - base.View()(iCell, j) == doctest::Approx(expected).epsilon(2e-12).scale(1));
            }
        }
    }
}
TEST_CASE("Nonzero quadratic temperature does not introduce viscosity at Euler open or slip boundaries")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    const DNDS::CFV::Math::TaylorBasis basis(3, 2);
    int quadratic = 0;
    for (int mode = 1; mode < basis.Size(); ++mode)
        if (basis.MultiIndex(mode) == std::array<int, 3>{2, 0, 0})
            quadratic = mode;
    REQUIRE(quadratic > 0);
    for (const auto *kind : {"farfield", "extrapolate", "slipWall"})
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .03;
        settings.boundaries["7"].kind = kind;
        settings.boundaries["7"].primitive = {1, 0, 0, 0, 2};
        settings.numerics.reconstruction = "variational"; // This oracle injects a quadratic polynomial, not a direct-FV gradient.
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.cfv.maxOrder = 2;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        const int modes = basis.Size() - 1;
        auto mean = fv.Metrics().BuildField(5), coefficient = fv.Metrics().BuildField(5 * modes);
        auto residual = fv.Metrics().BuildField(5), inviscid = fv.Metrics().BuildField(5);
        auto logicalMean = C::EquationDofs::Coupled(equation.Schema(), mean);
        fv.Initialize(logicalMean, {}, settings.boundaries["7"].primitive);
        DNDS::Algebra::Fill(coefficient, 0.);
        {
            auto write = coefficient.Owned().Write();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                write.View()(iCell, 4 * modes + quadratic - 1) = .05;
        }
        const auto u = mean.Owned().Read(), ug = mean.Ghost().Read(), c = coefficient.Owned().Read(), cg = coefficient.Ghost().Read();
        C::RHSReadViews input;
        input.mean = {u.View<C::RHSFlowRows>(), ug.View<C::RHSFlowRows>()};
        input.coefficients = {c.View<R::UniformTableRows>(), cg.View<R::UniformTableRows>()};
        const auto rhs = fv.AcquireRHSOperator();
        C::RHSLocalDiagnostics diagnostics;
        {
            auto output = residual.Owned().Write(), base = inviscid.Owned().Write();
            diagnostics.Merge(rhs.ApplyBorrowedChecked(input, {output.View<C::RHSFlowRows>()}));
            input.includeViscosity = false;
            diagnostics.Merge(rhs.ApplyBorrowedChecked(input, {base.View<C::RHSFlowRows>()}));
        }
        const auto a = residual.Owned().Read(), b = inviscid.Owned().Read();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(a.View()(iCell, j) == b.View()(iCell, j));
        CHECK(diagnostics.FaceFallbacks() == 0);
    }
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
