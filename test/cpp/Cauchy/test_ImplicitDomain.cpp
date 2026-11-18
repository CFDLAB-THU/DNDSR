#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include <optional>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
static DNDS::MPIInfo mpi;
namespace
{
    std::array<double, 5> Integral(const C::Domain &domain)
    {
        std::array<double, 5> value{};
        const auto read = domain.Dofs().CoupledField().Owned().Read();
        const auto data = read.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < data.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                value[j] += data(iCell, j) * domain.Discretization().Metrics().GetCellVol(iCell);
        MPI_Allreduce(MPI_IN_PLACE, value.data(), 5, MPI_DOUBLE, MPI_SUM, mpi.comm);
        return value;
    }
    void DensityWave(C::Domain &domain)
    {
        constexpr double pi = 3.14159265358979323846;
        auto write = domain.Dofs().CoupledField().Owned().Write();
        const auto values = write.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
        {
            const auto point = domain.Discretization().Metrics().GetCellCent(iCell);
            const double rho = 1 + .05 * std::sin(2 * pi * (point[0] + point[1]));
            const auto state = domain.Equation().Gas().Conservative({rho, .4, .2, .1, 1});
            for (int j = 0; j < 5; ++j)
                values(iCell, j) = state[j];
        }
    }
}
TEST_CASE("Periodic ACM physical step transports velocity and preserves the pressure gauge")
{
    C::DomainSettings settings;
    settings.equation.system = "incompressibleACM";
    settings.equation.eos = "none";
    settings.equation.referenceScales = {{"rho0", 1}, {"U0", 1}, {"L0", 1}, {"T0", 300}};
    settings.initialState = {{"type", "primUP"}, {"state", {1., .1, 0., -.25}}};
    settings.numerics.reconstruction = "constant";
    settings.numerics.positivity = false;
    settings.numerics.faceFallback = false;
    settings.numerics.flux = "rusanov";
    settings.implicit.jacobianAction = "assembled";
    settings.implicit.diagonal = "block";
    SUBCASE("single grid") {}
    SUBCASE("same-cell FAS with a gradient coarse level")
    {
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 2;
        settings.numerics.cfv.SORInstead = false;
        settings.numerics.reconstructionIterations = 10;
        settings.numerics.reconstructionTolerance = 0;
        settings.implicit.requireConvergedReconstruction = false;
        settings.implicit.multigrid.levels = 1;
        settings.implicit.multigrid.coarseSmooth = 1;
    }
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (const auto method : {"bdf1", "esdirk2", "ditr"})
    {
        CAPTURE(method);
        C::TimeSettings time;
        time.method = method;
        C::Domain domain("acm", mesh, settings, time);
        auto &owned = domain.Dofs().CoupledField().Owned();
        {
            auto write = owned.Write();
            const auto view = write.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < view.RowCount(); ++iCell)
                view(iCell, 1) += .02 * std::sin(2 * 3.141592653589793 * domain.Discretization().Metrics().GetCellCent(iCell)[0]);
        }
        REQUIRE(domain.Advance(.01).accepted);
        REQUIRE_FALSE(domain.LastImplicitStages().empty());
        for (const auto &stage : domain.LastImplicitStages())
        {
            CHECK(stage.converged);
            if (settings.implicit.multigrid.levels)
                CHECK(stage.multigridCycles > 0);
        }
        const auto read = owned.Read();
        const auto values = read.View<R::UniformTableRows>();
        std::array<double, 2> sums{};
        for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
        {
            const double vol = domain.Discretization().Metrics().GetCellVol(iCell);
            const double x = domain.Discretization().Metrics().GetCellCent(iCell)[0];
            sums[0] += vol * (values(iCell, 3) + .25);
            sums[1] += vol * std::abs(values(iCell, 1) - (.1 + .02 * std::sin(2 * 3.141592653589793 * x)));
        }
        MPI_Allreduce(MPI_IN_PLACE, sums.data(), 2, MPI_DOUBLE, MPI_SUM, mpi.comm);
        CHECK(std::abs(sums[0]) < 1e-12);
        CHECK(sums[1] > 1e-6);
    }
}
TEST_CASE("ACM BDF observes continuity separately from named momentum convergence")
{
    C::DomainSettings settings;
    settings.equation.system = "incompressibleACM";
    settings.equation.eos = "none";
    settings.equation.referenceScales = {{"rho0", 1}, {"U0", 1}, {"L0", 1}, {"T0", 300}};
    settings.initialState = {{"type", "primUP"}, {"state", {1., 0., 0., 0.}}};
    settings.numerics.reconstruction = "constant";
    settings.implicit.jacobianAction = "assembled";
    settings.implicit.diagonal = "block";
    settings.implicit.residualComponents = {"w"};
    settings.implicit.iterations = 1;
    settings.implicit.algebraicRateTolerance = 1e-14;
    settings.implicit.rejectUnconvergedStages = true;
    C::TimeSettings time;
    time.method = "bdf1";
    C::Domain domain("acm", CauchyTest::PeriodicQuads(mpi, 4), settings, time);
    {
        auto write = domain.Dofs().WriteOwned();
        const auto flow = write.Block("flow");
        for (DNDS::index iCell = 0; iCell < flow.RowCount(); ++iCell)
            flow(iCell, 0) += .1 * std::sin(2 * 3.141592653589793 * domain.Discretization().Metrics().GetCellCent(iCell)[0]);
    }
    CHECK_FALSE(domain.Advance(.01).accepted);
    REQUIRE_FALSE(domain.LastImplicitStages().empty());
    CHECK(domain.LastImplicitStages().back().residual[2] == 0);
    CHECK(domain.LastImplicitStages().back().residualTargetRatio > 1);
}
TEST_CASE("Production DITR controller uses common limits unless an override is supplied")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.implicit.iterations = settings.implicit.minIterations = 3;
    C::TimeSettings time;
    time.method = "ditr";
    C::Domain inherited("inherited", mesh, settings, time);
    REQUIRE(inherited.Advance(.01).accepted);
    REQUIRE(inherited.LastImplicitStages().size() == 1);
    CHECK(inherited.LastImplicitStages()[0].converged);
    CHECK(inherited.LastImplicitStages()[0].iterations == 3);
    time.ditr.iterations = time.ditr.minIterations = 1;
    C::Domain overridden("overridden", mesh, settings, time);
    REQUIRE(overridden.Advance(.01).accepted);
    REQUIRE(overridden.LastImplicitStages().size() == 1);
    CHECK(overridden.LastImplicitStages()[0].converged);
    CHECK(overridden.LastImplicitStages()[0].iterations == 1);
}
TEST_CASE("Native physical implicit methods preserve uniform periodic flow and transactional commit")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (const std::string method : {"bdf1", "bdf2", "trapz", "esdirk2", "esdirk3", "esdirk4", "ditr"})
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "constant";
        settings.implicit.minIterations = 0;
        settings.initialPrimitive = {1.1, .3, -.2, .4, 1.2};
        settings.execution.hostThreads = method == "esdirk4" ? 2 : 0;
        C::TimeSettings time;
        time.method = method;
        C::Domain domain("uniform", mesh, settings, time);
        const auto initial = Integral(domain);
        REQUIRE(domain.Advance(.01).accepted);
        REQUIRE(domain.Advance(.02).accepted);
        CHECK(domain.Time() == doctest::Approx(.03));
        CHECK(domain.Epoch().acceptedStep == 2);
        for (const auto &report : domain.LastImplicitStages())
            CHECK(report.converged);
        REQUIRE(domain.PrepareStep(.01).accepted);
        std::optional<R::Array<double>::ReadLease> held;
        if (mpi.rank == 0)
            held.emplace(domain.Dofs().CoupledField().Owned().Read());
        CHECK_THROWS(domain.CommitStep());
        CHECK(domain.Time() == doctest::Approx(.03));
        held.reset();
        domain.CommitStep();
        CHECK(domain.Time() == doctest::Approx(.04));
        CHECK(domain.Epoch().acceptedStep == 3);
        const auto final = Integral(domain);
        for (int j = 0; j < 5; ++j)
            CHECK(final[j] == doctest::Approx(initial[j]).epsilon(5e-14));
    }
}
TEST_CASE("P3 implicit periodic transport conserves flow and discard restores the VR warm guess")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
    for (const std::string method : {"esdirk4", "ditr"})
        for (const std::string action : {"assembled", "semiMatrixFree"})
        {
            C::DomainSettings settings;
            settings.implicit.jacobianAction = action;
            settings.numerics.reconstruction = "variational";
            settings.numerics.cfv.maxOrder = 3;
            settings.numerics.reconstructionIterations = 120;
            settings.numerics.reconstructionTolerance = 1e-10;
            settings.numerics.faceFallback = false;
            settings.implicit.relativeTolerance = 1e-9;
            settings.implicit.absoluteTolerance = 1e-11;
            settings.implicit.linear.relativeTolerance = 1e-3;
            settings.implicit.minIterations = 0;
            C::TimeSettings time;
            time.method = method;
            C::Domain retried("retry", mesh, settings, time), direct("direct", mesh, settings, time);
            DensityWave(retried);
            DensityWave(direct);
            const auto initial = Integral(retried);
            REQUIRE(retried.PrepareStep(.005).accepted);
            retried.DiscardStep();
            CHECK(retried.Time() == 0);
            CHECK(retried.Epoch().acceptedStep == 0);
            REQUIRE(retried.Advance(.005).accepted);
            REQUIRE(direct.Advance(.005).accepted);
            CHECK(retried.Time() == .005);
            CHECK(retried.LastFaceFallbacks() == 0);
            {
                const auto a = retried.Dofs().CoupledField().Owned().Read(), b = direct.Dofs().CoupledField().Owned().Read();
                for (DNDS::index iCell = 0; iCell < a.View().RowCount(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        CHECK(a.View()(iCell, j) == b.View()(iCell, j));
            }
            const auto final = Integral(retried);
            for (int j = 0; j < 5; ++j)
                CHECK(final[j] == doctest::Approx(initial[j]).epsilon(2e-11));
            if (method == "ditr")
            {
                CHECK(retried.LastCoupledIterations() > 0);
                CHECK(retried.LastCoupledTargetRatio() <= 1);
                REQUIRE(retried.LastImplicitStages().size() == 1);
                CHECK(retried.LastImplicitStages().front().iterations == retried.LastCoupledIterations());
                CHECK(retried.LastImplicitStages().front().residualTargetRatio == retried.LastCoupledTargetRatio());
            }
            else
                CHECK(retried.LastImplicitStages().size() == 5);
            for (const auto &report : retried.LastImplicitStages())
            {
                CHECK(report.converged);
                CHECK(report.iterations > 0);
                CHECK(report.reconstructionIterations > 0);
                CHECK(report.reconstructionIterations <= settings.numerics.reconstructionIterations);
                CHECK(report.reconstructionRelativeUpdate <= settings.numerics.reconstructionTolerance);
            }
            CHECK(retried.Discretization().Admissible(retried.Dofs()));
        }
}
TEST_CASE("Steady BDF1 infinite-dt mode removes a nonconstant residual without advancing physical time")
{
    for (const auto *layout : {"aos", "soa", "aosoa32"})
    {
        C::Json input{{"time", {{"steady", true}, {"method", "esdirk4"}}},
                      {"defaults", {{"execution", {{"layout", layout}}}, {"numerics", {{"reconstruction", "constant"}, {"flux", "rusanov"}}}, {"implicit", {{"iterations", 1000}, {"relativeTolerance", 1e-8}, {"absoluteTolerance", 1e-11}, {"pseudoCFL", 2}, {"maxPseudoCFL", 2}, {"pseudoGrowth", 1}}}}}};
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        C::Simulation simulation(config, mpi.comm, [&](auto, const auto &, const auto &domainMPI)
                                 { return CauchyTest::PeriodicQuads(domainMPI, 4); });
        auto &domain = simulation.GetDomain(0);
        DensityWave(domain);
        const auto before = domain.Diagnostics(config.Domains()[0].settings, {});
        REQUIRE(before[4] > 1e-3);
        REQUIRE(simulation.Advance(.01).accepted); // Irrelevant finite placeholder in steady mode.
        CHECK(domain.Steady());
        CHECK(domain.Time() == 0);
        CHECK(simulation.Time() == 0);
        CHECK(domain.Epoch().acceptedStep == 1);
        CHECK(simulation.Step() == 1);
        REQUIRE(domain.LastImplicitStages().size() == 1);
        CHECK(domain.LastImplicitStages()[0].converged);
        const auto after = domain.Diagnostics(config.Domains()[0].settings, {});
        CHECK(after[4] < before[4] * 1e-6);
    }
}
TEST_CASE("Capped steady stages advance by default and reject only with explicit control")
{
    for (bool fas : {false, true})
        for (bool reject : {false, true})
        {
            C::DomainSettings settings;
            settings.numerics.reconstruction = "variational";
            settings.numerics.cfv.maxOrder = 3;
            settings.numerics.reconstructionIterations = 3;
            settings.implicit.iterations = 1;
            settings.implicit.relativeTolerance = 0;
            settings.implicit.absoluteTolerance = 1e-14;
            if (fas)
            {
                settings.implicit.multigrid.levels = 2;
                settings.implicit.multigrid.cycles = 1;
                settings.implicit.multigrid.coarseSmooth = 1;
            }
            settings.implicit.rejectUnconvergedStages = reject;
            C::TimeSettings time;
            time.steady = true;
            time.method = "esdirk4"; // Domain construction also applies the steady override.
            C::Domain domain("capped-steady", CauchyTest::PeriodicQuads(mpi, 4), settings, time);
            DensityWave(domain);
            auto initial = domain.Discretization().Metrics().BuildField(domain.Dofs().CoupledField().Owned().Layout());
            DNDS::Algebra::Copy(initial, domain.Dofs().CoupledField());
            if (reject)
                CHECK_FALSE(domain.Advance(.01).accepted);
            else
                REQUIRE(domain.Advance(.01).accepted);
            CHECK(domain.Time() == 0);
            CHECK(domain.Epoch().acceptedStep == (reject ? 0 : 1));
            REQUIRE(domain.LastImplicitStages().size() == 1);
            CHECK_FALSE(domain.LastImplicitStages()[0].converged);
            CHECK_THROWS(domain.CommitStep());
            CHECK(domain.Discretization().Admissible(domain.Dofs()));
            const auto expected = initial.Owned().Read(), actual = domain.Dofs().CoupledField().Owned().Read();
            bool changed = false;
            for (DNDS::index iCell = 0; iCell < actual.View().RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    changed = changed || actual.View()(iCell, j) != expected.View()(iCell, j);
                    if (reject)
                        CHECK(actual.View()(iCell, j) == expected.View()(iCell, j));
                }
            int anyChanged = changed;
            MPI_Allreduce(MPI_IN_PLACE, &anyChanged, 1, MPI_INT, MPI_MAX, mpi.comm);
            CHECK(bool(anyChanged) == !reject);
        }
}
TEST_CASE("Strict implicit acceptance requires settled VR in addition to a loose mean target")
{
    for (bool fas : {false, true})
        for (bool strict : {false, true})
        {
            C::DomainSettings settings;
            settings.numerics.reconstruction = "variational";
            settings.numerics.cfv.maxOrder = 3;
            settings.numerics.reconstructionIterations = 1;
            settings.numerics.reconstructionTolerance = 1e-30;
            settings.implicit.requireConvergedReconstruction = strict;
            settings.implicit.rejectUnconvergedStages = true;
            settings.implicit.minIterations = 0;
            settings.implicit.iterations = 1;
            // Isolate VR acceptance: every finite mean residual meets this.
            settings.implicit.absoluteTolerance = 1e100;
            if (fas)
            {
                settings.implicit.multigrid.levels = 2;
                settings.implicit.multigrid.cycles = 1;
                settings.implicit.multigrid.coarseSmooth = 1;
            }
            C::TimeSettings time;
            time.steady = true;
            C::Domain domain("strict-vr", CauchyTest::PeriodicQuads(mpi, 4), settings, time);
            DensityWave(domain);
            if (strict)
            {
                CHECK_FALSE(domain.Advance(.01).accepted);
                CHECK(domain.Epoch().acceptedStep == 0);
            }
            else
            {
                REQUIRE(domain.Advance(.01).accepted);
                CHECK(domain.Epoch().acceptedStep == 1);
            }
            CHECK(domain.Time() == 0);
            REQUIRE(domain.LastImplicitStages().size() == 1);
            CHECK(domain.LastImplicitStages()[0].converged == !strict);
            CHECK(domain.LastImplicitStages()[0].reconstructionRelativeUpdate > settings.numerics.reconstructionTolerance);
        }
}
TEST_CASE("DITR coupled cap advances by default and rejects only with explicit control")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.implicit.minIterations = 0;
    settings.implicit.relativeTolerance = 1e-10;
    settings.implicit.absoluteTolerance = 1e-12;
    bool reject = false;
    SUBCASE("Euler cap policy") {}
    SUBCASE("Explicit rejection") { reject = true; }
    SUBCASE("One spatial FAS cycle per coupledStage")
    {
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.implicit.requireConvergedReconstruction = false;
        settings.implicit.multigrid.levels = 2;
        settings.implicit.multigrid.preSmooth = 1;
        settings.implicit.multigrid.postSmooth = 1;
        settings.implicit.multigrid.coarseSmooth = 1;
    }
    settings.implicit.rejectUnconvergedStages = reject;
    C::TimeSettings time;
    time.method = "ditr";
    time.ditr.iterations = 1;
    time.ditr.relativeTolerance = 1e-10;
    time.ditr.absoluteTolerance = 1e-12;
    C::Domain domain("capped", mesh, settings, time);
    DensityWave(domain);
    const auto before = Integral(domain);
    const auto version = domain.Dofs().CoupledField().Owned().Version();
    const auto prepared = domain.PrepareStep(.01);
    CHECK(prepared.accepted == !reject);
    if (reject)
        CHECK(prepared.reason.find("last_residual") != std::string::npos);
    CHECK(domain.Time() == 0);
    CHECK(domain.Epoch().acceptedStep == 0);
    CHECK(domain.Dofs().CoupledField().Owned().Version() == version);
    CHECK(Integral(domain) == before);
    CHECK(domain.LastCoupledIterations() == 1);
    CHECK(domain.LastCoupledTargetRatio() > 1);
    REQUIRE(domain.LastImplicitStages().size() == 1);
    CHECK_FALSE(domain.LastImplicitStages().front().converged);
    CHECK(domain.LastImplicitStages().front().iterations == 1);
    CHECK(domain.LastImplicitStages().front().residualTargetRatio == domain.LastCoupledTargetRatio());
    if (settings.implicit.multigrid.levels)
    {
        const auto &report = domain.LastImplicitStages().front();
        CHECK(report.multigridCycles == 2);           // Intermediate then endpoint, one each.
        REQUIRE(report.levelCorrections.size() == 3); // Fine plus two coarse levels.
        CHECK(report.levelCorrections[0] == 4);       // Two visits, pre+post each.
        CHECK(report.levelCorrections[1] == 4);
        CHECK(report.levelCorrections[2] == 2); // One coarsest correction per visit.
    }
    if (reject)
        CHECK_THROWS(domain.CommitStep());
    else
    {
        domain.CommitStep();
        CHECK(domain.Time() == doctest::Approx(.01));
        CHECK(domain.Epoch().acceptedStep == 1);
        CHECK(domain.Discretization().Admissible(domain.Dofs()));
    }
}
TEST_CASE("Rotating physical domains retain a uniform inertial flow with explicit and implicit stage times")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    for (const std::string method : {"ssprk3", "esdirk4"})
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.motion.angularVelocity = {.1, -.2, .3};
        settings.motion.translationVelocity = {.02, -.01, .03};
        settings.motion.center = {.3, .4, .5};
        settings.initialPrimitive = {1.2, .3, -.2, .1, 1.1};
        settings.boundaries["7"].kind = "analytic";
        settings.boundaries["7"].primitive = settings.initialPrimitive;
        settings.numerics.reconstruction = "constant";
        settings.numerics.cfv.maxOrder = 1;
        settings.implicit.diagonal = "block";
        settings.implicit.roeDissipation = true;
        settings.implicit.absoluteTolerance = 1e-12;
        settings.implicit.relativeTolerance = 1e-10;
        C::TimeSettings time;
        time.method = method;
        C::Domain domain("rotating", mesh, settings, time);
        const auto world = domain.Equation().Gas().Conservative(settings.initialPrimitive);
        for (int step = 0; step < 4; ++step)
            REQUIRE(domain.Advance(.001).accepted);
        const auto frame = domain.Discretization().Motion().At(domain.Time());
        const auto exact = frame.LocalState(world);
        const auto read = domain.Dofs().CoupledField().Owned().Read();
        const auto values = read.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(values(iCell, j) == doctest::Approx(exact[j]).epsilon(2e-9).scale(1));
        for (const auto &stage : domain.LastImplicitStages())
            CHECK(stage.converged);
    }
}
int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
