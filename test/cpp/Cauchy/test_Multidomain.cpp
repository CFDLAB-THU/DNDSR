#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "CartesianFixture.hpp"
#include "RotatingVortexFixture.hpp"
#include <cstdlib>
#include <filesystem>
#include <unistd.h>
#include <iostream>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
namespace
{
    C::Json Domains(bool split)
    {
        std::vector<int> a, b;
        for (int rank = mpi.size - 1; rank >= 0; --rank)
        {
            if (!split || rank % 2 == 0)
                a.push_back(rank);
            if (!split || rank % 2 == 1)
                b.push_back(rank);
        }
        if (b.empty())
            b = a;
        return C::Json::array({{{"name", "outer"}, {"ranks", a}}, {{"name", "disk"}, {"ranks", b}}});
    }
    std::vector<double> Values(const C::Domain &domain)
    {
        const auto &owned = domain.Dofs().CoupledField().Owned();
        const auto read = owned.GetDevice() == R::Device::Host() ? owned.Read() : owned.To(R::Device::Host()).Read();
        const auto values = read.View<R::UniformTableRows>();
        std::vector<double> result;
        for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            for (I j = 0; j < values.Width(); ++j)
                result.push_back(values(iCell, j));
        return result;
    }
    std::array<double, 2> Integrals(const C::Simulation &simulation)
    {
        std::array<double, 2> value{};
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
            {
                const auto &domain = simulation.GetDomain(iDomain);
                const auto geometry = domain.Discretization().Metrics().AcquireRead();
                const auto metric = geometry.HostView();
                const auto read = domain.Dofs().CoupledField().Owned().Read();
                const auto u = read.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < u.RowCount(); ++iCell)
                {
                    value[0] += metric.GetCellVol(iCell) * u(iCell, 0);
                    value[1] += metric.GetCellVol(iCell) * u(iCell, 4);
                }
            }
        MPI_Allreduce(MPI_IN_PLACE, value.data(), 2, MPI_DOUBLE, MPI_SUM, mpi.comm);
        return value;
    }
    std::string CheckpointPath(const std::string &label)
    {
        int pid = int(getpid());
        MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
        return (std::filesystem::temp_directory_path() / ("cauchy-multidomain-" + label + std::to_string(pid) + ".h5")).string();
    }
    void RemoveCheckpoint(const std::string &path)
    {
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0)
        {
            CHECK(std::filesystem::remove(path));
            std::filesystem::remove_all(path + ".parts");
        }
        MPI_Barrier(mpi.comm);
    }
    // Independent fixture-coordinate oracle for a 4x4 torus. Dense global
    // storage is confined to this 160-value test snapshot, never mesh setup.
    std::vector<double> TorusValues(const C::Simulation &simulation)
    {
        std::vector<double> result(2 * 16 * 5, 0);
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
            {
                const auto &domain = simulation.GetDomain(iDomain);
                const auto read = domain.Dofs().CoupledField().Owned().Read();
                for (I iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
                {
                    const auto point = domain.Discretization().Metrics().GetCellCent(iCell);
                    const I global = I(std::floor(point[0] * .4)) + 4 * I(std::floor(point[1] * .4));
                    for (int j = 0; j < 5; ++j)
                        result[(iDomain * 16 + global) * 5 + j] = read.View()(iCell, j);
                }
            }
        MPI_Allreduce(MPI_IN_PLACE, result.data(), int(result.size()), MPI_DOUBLE, MPI_SUM, mpi.comm);
        return result;
    }
}
TEST_CASE("Multi-domain BDF2 restart reserves every domain and redistributes by persistent IDs")
{
    const auto factory = [](I, const auto &, const auto &domainMPI)
    { return CauchyTest::PeriodicQuads(domainMPI, 4, 10); };
    for (bool split : {false, true})
    {
        if (split && mpi.size == 1)
            continue;
        CAPTURE(split);
        const auto path = CheckpointPath(split ? "split-" : "shared-");
        C::Json input{{"domains", Domains(split)}, {"time", {{"method", "bdf2"}}}, {"defaults", {{"initialPrimitive", {1, .1, .2, .03, 1}}, {"initialProfile", {{"kind", "isentropicVortex"}, {"strength", .2}, {"period", 10}}}, {"numerics", {{"reconstruction", "variational"}, {"limiter", "none"}, {"reconstructionIterations", 4}, {"cfv", {{"maxOrder", 2}}}}}, {"implicit", {{"iterations", 30}, {"absoluteTolerance", 1e-12}}}}}};
        C::Simulation source(C::SimulationConfig::ResolveCollective(input, mpi.comm), mpi.comm, factory);
        REQUIRE(source.Advance(.001).accepted);
        REQUIRE(source.Advance(.002).accepted);
        const auto saved = TorusValues(source);
        const C::Json outputController{{"schema_version", 1}, {"test_cursor", 7}};
        source.WriteCheckpoint(path, true, outputController);
        REQUIRE(source.Advance(.0015).accepted);
        const auto continuation = TorusValues(source);
        std::array<uint64_t, 2> versions{};
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (source.Local(iDomain))
                versions[iDomain] = source.GetDomain(iDomain).Dofs().CoupledField().Owned().Version();
        std::optional<R::Array<double>::ReadLease> held;
        if (source.Placement().DomainRank(1) == 0)
            held.emplace(source.GetDomain(1).Dofs().CoupledField().Owned().Read());
        CHECK_THROWS(source.ReadCheckpoint(path));
        held.reset();
        CHECK(source.Step() == 3);
        CHECK(TorusValues(source) == continuation);
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (source.Local(iDomain))
                CHECK(source.GetDomain(iDomain).Dofs().CoupledField().Owned().Version() == versions[iDomain]);
        source.ReadCheckpoint(path);
        CHECK(source.Step() == 2);
        CHECK(source.Time() == .003);
        REQUIRE(source.RestoredOutputController());
        CHECK(*source.RestoredOutputController() == outputController);
        CHECK(TorusValues(source) == saved);
        REQUIRE(source.Advance(.0015).accepted);
        CHECK(TorusValues(source) == continuation);
        // Republishing uses a new child generation. Reverse sharing and use
        // SoA/P3 to exercise actual redistribution and intentional cold VR.
        source.WriteCheckpoint(path, true, outputController);
        input["domains"] = Domains(!split && mpi.size > 1);
        input["defaults"]["execution"]["layout"] = "soa";
        input["defaults"]["numerics"]["cfv"]["maxOrder"] = 3;
        C::Simulation redistributed(C::SimulationConfig::ResolveCollective(input, mpi.comm), mpi.comm, factory);
        redistributed.ReadCheckpoint(path);
        REQUIRE(redistributed.RestoredOutputController());
        CHECK(*redistributed.RestoredOutputController() == outputController);
        CHECK(redistributed.Step() == source.Step());
        CHECK(redistributed.Time() == source.Time());
        CHECK(TorusValues(redistributed) == continuation);
        REQUIRE(redistributed.Advance(.0005).accepted);
        RemoveCheckpoint(path);
    }
}
TEST_CASE("Moving ACM disk follows the pressure datum of the connected outer domain through restart")
{
    const auto factory = [](I iDomain, const auto &, const auto &domainMPI)
    {
        return CauchyTest::VortexPatch(domainMPI, iDomain == 1, 8, 20,
                                       CauchyTest::VortexPointDistribution::JitteredGrid, false);
    };
    const auto state = [](double pressure) -> C::Json
    { return {{"type", "primUP"}, {"state", {0., 0., 0., pressure}}}; };
    C::Json input{
        {"domains", Domains(mpi.size > 1)},
        {"time", {{"method", "bdf1"}}},
        {"defaults", {{"equation", {{"system", "incompressibleACM"}, {"eos", "none"}, {"acmPressureMean", 9.}, {"referenceScales", {{"rho0", 1}, {"U0", 1}, {"L0", 1}, {"T0", 300}}}}}, {"initialState", state(-2.)}, {"boundaries", {{"101", {{"kind", "interface"}}}}}, {"numerics", {{"reconstruction", "gradient"}, {"flux", "Roe"}, {"positivity", false}}}, {"implicit", {{"diagonal", "block"}, {"jacobianAction", "assembled"}, {"iterations", 3}, {"rejectUnconvergedStages", false}}}}},
        {"interfaces", C::Json::array({{{"donor", "outer"}, {"receiver", "disk"}, {"donorZone", 101}, {"receiverZone", 101}, {"maxGap", .03}, {"refreshCadence", "nonlinearIteration"}},
                                       {{"donor", "disk"}, {"receiver", "outer"}, {"donorZone", 101}, {"receiverZone", 101}, {"maxGap", .03}, {"refreshCadence", "nonlinearIteration"}}})}};
    input["domains"][0]["overrides"]["boundaries"]["102"] = {{"kind", "pressureOutlet"}, {"state", state(-2.)}};
    input["domains"][1]["overrides"] = {{"initialState", state(-1.99)}, {"equation", {{"acmPressureMean", 7.}}}, {"motion", {{"angularVelocity", {0., 0., .02}}, {"center", {5., 5., 0.}}}}};
    // Match the numerical choices on both backends, including the linear update.
    input["defaults"]["implicit"]["linear"] = {{"method", "jacobi"}, {"stopping", "fixedCount"}, {"iterations", 3}};
    input["defaults"]["numerics"]["viscousGradient"] = "polynomial";
    bool expectMultigrid = false;
    std::optional<C::SimulationConfig> cpuConfig;
    if (std::getenv("DNDS_TEST_ACM_INTERFACE_CUDA"))
    {
        SUBCASE("direct P1 with nonlinear packet refresh")
        {
            input["defaults"]["implicit"]["multigrid"]["levels"] = 0;
        }
        SUBCASE("modal P2 with one coarse level and frozen stage packets")
        {
            expectMultigrid = true;
            auto &numerics = input["defaults"]["numerics"];
            numerics["reconstruction"] = "variational";
            numerics["cfv"] = {{"maxOrder", 2}, {"intOrder", 3}, {"SORInstead", false}};
            numerics["reconstructionIterations"] = 1;
            numerics["reconstructionTolerance"] = 0;
            auto &implicit = input["defaults"]["implicit"];
            implicit["requireConvergedReconstruction"] = false;
            implicit["multigrid"] = {{"levels", 1}, {"coarseSmooth", 1}};
            for (auto &connection : input["interfaces"])
                connection["refreshCadence"] = "stageFrozen";
        }
        cpuConfig.emplace(C::SimulationConfig::ResolveCollective(input, mpi.comm));
        input["defaults"]["execution"]["device"] = "cuda";
    }
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    C::Simulation simulation(config, mpi.comm, factory);
    for (I d = 0; d < 2; ++d)
        if (simulation.Local(d))
        {
            const auto &domain = simulation.GetDomain(d);
            CHECK(domain.Discretization().HasPressureBoundary() == (d == 0));
            CHECK_FALSE(domain.CaptureStateSnapshot(false).PressureGauge().has_value());
        }
    constexpr double dt = .001;
    REQUIRE(simulation.Advance(dt).accepted);
    CHECK(simulation.GetInterfaceDiagnostics().queryPoints > 0);
    auto path = CheckpointPath("acm-outlet-");
    simulation.WriteCheckpoint(path, true);
    C::Simulation resumed(config, mpi.comm, factory);
    resumed.ReadCheckpoint(path);
    REQUIRE(simulation.Advance(dt).accepted);
    REQUIRE(resumed.Advance(dt).accepted);
    for (I d = 0; d < 2; ++d)
        if (simulation.Local(d))
        {
            const auto actual = Values(resumed.GetDomain(d)), expected = Values(simulation.GetDomain(d));
            REQUIRE(actual.size() == expected.size());
            for (size_t i = 0; i < actual.size(); ++i)
                CHECK(actual[i] == doctest::Approx(expected[i]).epsilon(1e-11).scale(1));
            if (d == 1)
            {
                double speed = 0;
                for (size_t i = 0; i < actual.size(); i += 4)
                {
                    speed = std::max(speed, std::hypot(actual[i], actual[i + 1]));
                    CHECK(std::abs(actual[i + 3] + 2.) < .1);
                }
                CHECK(DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Max>(speed, resumed.GetDomain(d).Communicator()) > 1e-8);
            }
        }
    if (cpuConfig)
    {
        C::Simulation cpu(*cpuConfig, mpi.comm, factory);
        REQUIRE(cpu.Advance(dt).accepted);
        REQUIRE(cpu.Advance(dt).accepted);
        for (I d = 0; d < 2; ++d)
            if (simulation.Local(d))
            {
                REQUIRE(simulation.GetDomain(d).Dofs().Device() == R::Device::CUDA(0));
                if (expectMultigrid)
                {
                    const auto &stages = simulation.GetDomain(d).LastImplicitStages();
                    REQUIRE_FALSE(stages.empty());
                    CHECK(stages.back().multigridCycles > 0);
                }
                const auto actual = Values(simulation.GetDomain(d)), reference = Values(cpu.GetDomain(d));
                REQUIRE(actual.size() == reference.size());
                for (size_t i = 0; i < actual.size(); ++i)
                    CHECK(actual[i] == doctest::Approx(reference[i]).epsilon(1e-9).scale(1));
            }
    }
    RemoveCheckpoint(path);
}

TEST_CASE("Coordinated domains retain single-domain time equations and atomic publication")
{
    for (const std::string method : {"ssprk3", "bdf2", "esdirk4", "ditr"})
    {
        CAPTURE(method);
        C::Json input{{"domains", Domains(mpi.size > 1)}, {"time", {{"method", method}}}, {"defaults", {{"numerics", {{"reconstruction", "constant"}, {"flux", "rusanov"}}}, {"initialPrimitive", {1, .1, .2, .03, 1}}, {"initialProfile", {{"kind", "isentropicVortex"}, {"strength", .2}, {"period", 10}}}, {"implicit", {{"iterations", 30}, {"absoluteTolerance", 1e-12}}}}}};
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        const auto factory = [](I, const auto &, const auto &domainMPI)
        { return CauchyTest::PeriodicQuads(domainMPI, 4, 10); };
        C::Simulation simulation(config, mpi.comm, factory);
        std::vector<std::unique_ptr<C::Domain>> reference(2);
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
            {
                const auto &domain = config.Domains()[iDomain];
                reference[iDomain] = std::make_unique<C::Domain>(domain.placement.name,
                                                                 factory(iDomain, domain, DNDS::MPIInfo(simulation.Placement().Context(iDomain)->Communicator())), domain.settings, config.Time());
            }
        for (int step = 0; step < 2; ++step)
        {
            REQUIRE(simulation.Advance(.001).accepted);
            for (I iDomain = 0; iDomain < 2; ++iDomain)
                if (reference[iDomain])
                {
                    REQUIRE(reference[iDomain]->Advance(.001).accepted);
                    CHECK(Values(simulation.GetDomain(iDomain)) == Values(*reference[iDomain]));
                    CHECK(simulation.GetDomain(iDomain).Time() == simulation.Time());
                }
        }
        // Only one rank in the second domain retains a reader. Every domain's
        // accepted field and BDF history must remain at the previous step.
        std::vector<std::vector<double>> before(2);
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
                before[iDomain] = Values(simulation.GetDomain(iDomain));
        std::optional<R::Array<double>::ReadLease> held;
        if (simulation.Placement().DomainRank(1) == 0)
            held.emplace(simulation.GetDomain(1).Dofs().CoupledField().Owned().Read());
        CHECK_THROWS(simulation.Advance(.001));
        CHECK(simulation.Step() == 2);
        CHECK(simulation.Time() == .002);
        held.reset();
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
                CHECK(Values(simulation.GetDomain(iDomain)) == before[iDomain]);
        REQUIRE(simulation.Advance(.001).accepted);
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (reference[iDomain])
            {
                REQUIRE(reference[iDomain]->Advance(.001).accepted);
                CHECK(Values(simulation.GetDomain(iDomain)) == Values(*reference[iDomain]));
            }
    }
}
TEST_CASE("P3 flow5 crosses a posed disk interface with dense and sparse distributed matching")
{
    for (bool split : {false, true})
    {
        if (split && mpi.size == 1)
            continue;
        for (double omega : {0., .02})
        {
            CAPTURE(split);
            CAPTURE(omega);
            C::Json input{{"domains", Domains(split)}, {"time", {{"method", !split && omega == 0 ? "esdirk2" : "ssprk3"}}}, {"defaults", {{"initialPrimitive", {1, .1, .2, .03, 1}}, {"initialProfile", {{"kind", "isentropicVortex"}, {"strength", .5}, {"period", 10}}}, {"boundaries", {{"101", {{"kind", "interface"}}}}}, {"numerics", {{"reconstruction", "variational"}, {"flux", "rusanov"}, {"limiter", "none"}, {"positivity", false}, {"reconstructionIterations", 12}, {"cfv", {{"maxOrder", 3}, {"intOrder", 5}, {"jacobiRelax", .8}, {"boundaryDerivativeWeights", {1, 0, 0, 0}}}}}}}}, {"interfaces", C::Json::array({{{"donor", "outer"}, {"receiver", "disk"}, {"donorZone", 101}, {"receiverZone", 101}, {"maxGap", .02}}, {{"donor", "disk"}, {"receiver", "outer"}, {"donorZone", 101}, {"receiverZone", 101}, {"maxGap", .02}}})}};
            input["domains"][1]["overrides"] = {{"motion", {{"angularVelocity", {0, 0, omega}}, {"center", {5, 5, 0}}}}};
            const auto factory = [](I iDomain, const auto &, const auto &domainMPI)
            { return CauchyTest::VortexPatch(domainMPI, iDomain == 1, 12, 32); };
            const auto sparseConfig = C::SimulationConfig::ResolveCollective(input, mpi.comm);
            C::Simulation sparse(sparseConfig, mpi.comm, factory);
            const auto initial = Integrals(sparse);
            for (int step = 0; step < 3; ++step)
                REQUIRE(sparse.Advance(.001).accepted);
            const auto final = Integrals(sparse);
            const auto diagnostics = sparse.GetInterfaceDiagnostics();
            CHECK(diagnostics.queryPoints > 0);
            CHECK(diagnostics.maximumGap <= .02);
            CHECK(diagnostics.mapBuilds == (omega == 0 ? 2 : 20));
            double error = 0;
            for (I iDomain = 0; iDomain < 2; ++iDomain)
                if (sparse.Local(iDomain))
                {
                    const auto &domain = sparse.GetDomain(iDomain);
                    const auto &settings = sparseConfig.Domains()[iDomain].settings;
                    error = std::max(error, domain.Discretization().Error(domain.Dofs(), settings.initialProfile, settings.initialPrimitive, sparse.Time())[2][0]);
                    CHECK(domain.Discretization().Admissible(domain.Dofs()));
                }
            MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            if (mpi.rank == 0)
                std::cout << "IV interfaces split=" << split << " omega=" << omega << " rho_Linf=" << error << " mass_drift=" << final[0] - initial[0] << " energy_drift=" << final[1] - initial[1] << '\n';
            CHECK(error > 1e-10);
            CHECK(error < 2e-3);
            CHECK(std::abs(final[0] - initial[0]) < 2e-3);
            CHECK(std::abs(final[1] - initial[1]) < 5e-3);
            // Same physical meshes and poses, independent dense routing. Every
            // resulting field component must agree, not merely the match count.
            for (auto &connection : input["interfaces"])
                connection["routing"] = "allGatherBoundary";
            C::Simulation dense(C::SimulationConfig::ResolveCollective(input, mpi.comm), mpi.comm, factory);
            for (int step = 0; step < 3; ++step)
                REQUIRE(dense.Advance(.001).accepted);
            for (I iDomain = 0; iDomain < 2; ++iDomain)
                if (sparse.Local(iDomain))
                    CHECK(Values(sparse.GetDomain(iDomain)) == Values(dense.GetDomain(iDomain)));
            if (!split && omega != 0)
            {
                const auto old = Values(sparse.GetDomain(0));
                std::optional<R::Array<double>::ReadLease> held;
                if (mpi.rank == 0)
                    held.emplace(sparse.GetDomain(1).Dofs().CoupledField().Owned().Read());
                CHECK_THROWS(sparse.Advance(.001));
                held.reset();
                CHECK(sparse.Step() == 3);
                CHECK(Values(sparse.GetDomain(0)) == old);
                CHECK(sparse.GetDomain(0).StableStep(.2) > 0);
                CHECK_NOTHROW(sparse.GetDomain(0).Diagnostics(sparseConfig.Domains()[0].settings, {}));
                const auto path = CheckpointPath("rotating-");
                sparse.WriteCheckpoint(path, true);
                {
                    std::optional<R::Array<double>::ReadLease> heldRestart;
                    if (mpi.rank == 0)
                        heldRestart.emplace(sparse.GetDomain(1).Dofs().CoupledField().Owned().Read());
                    CHECK_THROWS(sparse.ReadCheckpoint(path));
                }
                // Failed publication must also restore accepted interface
                // packets; diagnostics/CFL must not consume trial-time traces.
                CHECK(sparse.Step() == 3);
                CHECK(sparse.GetDomain(0).StableStep(.2) > 0);
                REQUIRE(sparse.Advance(.001).accepted);
                const std::array<std::vector<double>, 2> continuation{Values(sparse.GetDomain(0)), Values(sparse.GetDomain(1))};
                sparse.ReadCheckpoint(path);
                CHECK(sparse.Step() == 3);
                CHECK(sparse.GetDomain(0).StableStep(.2) > 0);
                REQUIRE(sparse.Advance(.001).accepted);
                for (I iDomain = 0; iDomain < 2; ++iDomain)
                    CHECK(Values(sparse.GetDomain(iDomain)) == continuation[iDomain]);
                RemoveCheckpoint(path);
            }
        }
    }
}
TEST_CASE("P0 and physical-gradient P1 publish complete multidomain traces without modal storage")
{
    for (const std::string reconstruction : {"constant", "gradient"})
        for (const bool split : {false, true})
        {
            if (split && mpi.size == 1)
                continue;
            CAPTURE(reconstruction);
            CAPTURE(split);
            C::Json input{
                {"domains", Domains(split)},
                {"time", {{"method", "ssprk3"}}},
                {"defaults",
                 {{"initialPrimitive", {1, .1, .2, .03, 1}},
                  {"initialProfile",
                   {{"kind", "isentropicVortex"}, {"strength", .2}, {"period", 10}}},
                  {"boundaries", {{"101", {{"kind", "interface"}}}}},
                  {"numerics",
                   {{"reconstruction", reconstruction},
                    {"flux", "rusanov"},
                    {"limiter", reconstruction == "gradient" ? "Barth" : "none"},
                    {"positivity", false}}}}},
                {"interfaces",
                 C::Json::array(
                     {{{"donor", "outer"},
                       {"receiver", "disk"},
                       {"donorZone", 101},
                       {"receiverZone", 101},
                       {"maxGap", .03}},
                      {{"donor", "disk"},
                       {"receiver", "outer"},
                       {"donorZone", 101},
                       {"receiverZone", 101},
                       {"maxGap", .03}}})}};
            input["domains"][1]["overrides"] = {
                {"motion",
                 {{"angularVelocity", {0, 0, .02}}, {"center", {5, 5, 0}}}}};
            const auto factory = [](I iDomain, const auto &, const auto &domainMPI)
            { return CauchyTest::VortexPatch(domainMPI, iDomain == 1, 8, 20); };
            C::Simulation simulation(
                C::SimulationConfig::ResolveCollective(input, mpi.comm), mpi.comm,
                factory);
            I interfaceCenters = 0;
            for (I iDomain = 0; iDomain < 2; ++iDomain)
                if (simulation.Local(iDomain))
                {
                    const auto &fv = simulation.GetDomain(iDomain).Discretization();
                    const auto packet = fv.BuildBoundaryInputs();
                    // Direct FV requests centers only. Matching must not also
                    // send the same location as a one-point quadrature role.
                    CHECK(fv.ReadBoundaryPoints().View().RowCount() == 0);
                    CHECK(packet.QuadratureValues().Space().OwnedRows() == 0);
                    const auto faces = fv.ReadBoundaryFaces();
                    CHECK(packet.FaceCenterValues().Space().OwnedRows() == size_t(faces.View().RowCount()));
                    for (I iFace = 0; iFace < faces.View().RowCount(); ++iFace)
                        interfaceCenters += faces.View()(iFace, 0).interface;
                }
            interfaceCenters = DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Sum>(interfaceCenters, mpi.comm);
            REQUIRE(simulation.Advance(.001).accepted);
            const auto diagnostics = simulation.GetInterfaceDiagnostics();
            CHECK(diagnostics.queryPoints == interfaceCenters);
            CHECK(interfaceCenters > 0);
            CHECK(diagnostics.maximumGap <= .03);
            for (I iDomain = 0; iDomain < 2; ++iDomain)
                if (simulation.Local(iDomain))
                    CHECK(simulation.GetDomain(iDomain)
                              .Discretization()
                              .Admissible(simulation.GetDomain(iDomain).Dofs()));
        }
}
TEST_CASE("DITR freezes distinct rotating midpoint and endpoint interface packets")
{
    for (const bool split : {false, true})
    {
        if (split && mpi.size == 1)
            continue;
        CAPTURE(split);
        C::Json input{
            {"domains", Domains(split)},
            {"time", {{"method", "ditr"}}},
            {"defaults",
             {{"initialPrimitive", {1, .1, .2, .03, 1}},
              {"initialProfile",
               {{"kind", "isentropicVortex"},
                {"strength", .2},
                {"period", 10}}},
              {"boundaries", {{"101", {{"kind", "interface"}}}}},
              {"implicit", {{"iterations", 3}}},
              {"numerics",
               {{"reconstruction", "gradient"},
                {"gradient", "leastSquares"},
                {"flux", "rusanov"},
                {"limiter", "Barth"},
                {"positivity", false}}}}},
            {"interfaces",
             C::Json::array(
                 {{{"donor", "outer"},
                   {"receiver", "disk"},
                   {"donorZone", 101},
                   {"receiverZone", 101},
                   {"maxGap", .03}},
                  {{"donor", "disk"},
                   {"receiver", "outer"},
                   {"donorZone", 101},
                   {"receiverZone", 101},
                   {"maxGap", .03}}})}};
        input["domains"][1]["overrides"] = {
            {"motion",
             {{"angularVelocity", {0, 0, .02}}, {"center", {5, 5, 0}}}}};
        // The shared receiver schedule also supports different access-check
        // policies and launchers without a host wait at the packet boundary.
        input["domains"][0]["overrides"]["execution"]["checkAccess"] = false;
        input["domains"][1]["overrides"]["execution"]["hostThreads"] = 1;
        for (auto &connection : input["interfaces"])
            connection["refreshCadence"] = "nonlinearIteration";
        const auto factory = [](I iDomain, const auto &, const auto &domainMPI)
        { return CauchyTest::VortexPatch(domainMPI, iDomain == 1, 8, 20); };
        C::Simulation simulation(
            C::SimulationConfig::ResolveCollective(input, mpi.comm), mpi.comm,
            factory);
        REQUIRE(simulation.Advance(1e-4).accepted);
        const auto diagnostics = simulation.GetInterfaceDiagnostics();
        CHECK(diagnostics.queryPoints > 0);
        CHECK(diagnostics.maximumGap <= .03);
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
            {
                const auto &domain = simulation.GetDomain(iDomain);
                CHECK(domain.LastCoupledIterations() > 0);
                CHECK(domain.Discretization().Admissible(domain.Dofs()));
            }
        // Exercise the actual matcher/sampler/return route with distinguishable
        // stage data. Solver admissibility alone cannot detect packet aliasing
        // or midpoint/endpoint route reuse at the wrong rigid-body pose.
        std::vector<std::optional<C::FiniteVolume>> local(2);
        std::vector<std::optional<C::EquationDofs>> values(2);
        std::vector<std::shared_ptr<C::FiniteVolume::ReconstructionState>> reconstruction(2);
        std::vector<std::shared_ptr<C::FiniteVolume::Workspace>> workspace(2);
        std::array<std::vector<std::optional<C::BoundaryInputs>>, 2> frozen{
            std::vector<std::optional<C::BoundaryInputs>>(2),
            std::vector<std::optional<C::BoundaryInputs>>(2)};
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
            {
                local[iDomain] = simulation.GetDomain(iDomain).Discretization();
                values[iDomain] = C::EquationDofs::Coupled(
                    local[iDomain]->Equation().Schema(), local[iDomain]->Metrics().BuildField(5));
                reconstruction[iDomain] = local[iDomain]->BuildReconstructionState(*values[iDomain]);
                workspace[iDomain] = local[iDomain]->BuildWorkspace(*values[iDomain]);
            }
        C::InterfaceCoupling coupling(simulation.Config(), simulation.Placement().ControlContext(), local);
        for (int round = 0; round < 2; ++round)
            for (int slot = 0; slot < 2; ++slot)
            {
                const double time = slot == 0 ? 5e-5 : 1e-4;
                const double density = 2 + slot + 2 * round;
                std::vector<C::InterfaceDomainInput> donors(2);
                for (I iDomain = 0; iDomain < 2; ++iDomain)
                    if (local[iDomain])
                    {
                        local[iDomain]->Initialize(*values[iDomain], {}, {density, 0, 0, 0, 1});
                        donors[iDomain].values = local[iDomain]->PublishInterfaceDonor(
                            *values[iDomain], *reconstruction[iDomain], *workspace[iDomain], false);
                    }
                C::StateEpoch epoch;
                epoch.trial = 1;
                epoch.coupledStage = slot;
                epoch.couplingIteration = round;
                auto refresh = round == 0 ? coupling.BeginRefresh(donors, time, epoch)
                                          : coupling.BeginValueRefresh(donors, time, epoch);
                // A receiver can retain its exact round before completion;
                // copying the handle must not lose the deferred publication.
                std::vector<std::optional<C::BoundaryInputs>> pending(2);
                for (I iDomain = 0; iDomain < 2; ++iDomain)
                    if (local[iDomain])
                    {
                        pending[size_t(iDomain)] = refresh.PendingInputs(iDomain);
                        REQUIRE(pending[size_t(iDomain)]);
                        CHECK_THROWS(pending[size_t(iDomain)]->Time());
                    }
                refresh.Complete();
                for (I iDomain = 0; iDomain < 2; ++iDomain)
                    if (local[iDomain])
                    {
                        const auto packet = refresh.Inputs(iDomain);
                        REQUIRE(packet);
                        CHECK(pending[size_t(iDomain)]->SamePublication(*packet));
                        CHECK_NOTHROW(local[iDomain]->ValidateBoundaryInputs(*pending[size_t(iDomain)], time));
                        CHECK(packet->Epoch().coupledStage == uint64_t(slot));
                        CHECK(packet->Epoch().couplingIteration == uint64_t(round));
                        CHECK(packet->Time() == time);
                        const auto rows = packet->QuadratureValues().ReadOwned();
                        const auto flow = rows.Block("flow");
                        const auto &points = local[iDomain]->BoundaryPoints();
                        for (I iPoint = 0; iPoint < I(points.size()); ++iPoint)
                            if (points[iPoint].interface)
                                CHECK(flow(iPoint, 0) == doctest::Approx(density));
                        if (round == 0)
                            frozen[slot][iDomain] = local[iDomain]->FreezeBoundaryInputs(*packet);
                        else
                        {
                            const auto &old = *frozen[slot][iDomain];
                            REQUIRE(packet->InterfaceSources().size() == old.InterfaceSources().size());
                            for (I iSource = 0; iSource < I(old.InterfaceSources().size()); ++iSource)
                                CHECK(packet->InterfaceSources()[iSource].routeGeneration == old.InterfaceSources()[iSource].routeGeneration);
                            local[iDomain]->ValidateBoundaryInputs(old, time);
                            const auto oldRows = old.QuadratureValues().ReadOwned();
                            for (I iPoint = 0; iPoint < I(points.size()); ++iPoint)
                                if (points[iPoint].interface)
                                    CHECK(oldRows.Block("flow")(iPoint, 0) == doctest::Approx(2 + slot));
                        }
                    }
            }
    }
}
TEST_CASE("Split flow and SA fields cross a rotating interface as one equation packet")
{
    if (mpi.size < 2)
        return;
    C::Json input{
        {"domains", Domains(true)},
        {"time", {{"method", "ssprk3"}}},
        {"defaults",
         {{"equation",
           {{"transport", "NewtonianFourier"},
            {"viscosity", 1e-3},
            {"rans", {{"model", "SA"}}}}},
          {"execution",
           {{"fieldGrouping", "equationBlocks"}, {"layout", "soa"}}},
          {"initialPrimitive", {1, .1, .2, .03, 1}},
          {"initialRANS", {{"rhoNuTilde", .01}}},
          {"boundaries", {{"101", {{"kind", "interface"}}}}},
          {"numerics",
           {{"reconstruction", "gradient"},
            {"gradient", "leastSquares"},
            {"viscousGradient", "polynomial"},
            {"flux", "rusanov"},
            {"faceFallback", false},
            {"limiter", "Barth"},
            {"positivity", false}}}}},
        {"interfaces",
         C::Json::array(
             {{{"donor", "outer"},
               {"receiver", "disk"},
               {"donorZone", 101},
               {"receiverZone", 101},
               {"maxGap", .03}},
              {{"donor", "disk"},
               {"receiver", "outer"},
               {"donorZone", 101},
               {"receiverZone", 101},
               {"maxGap", .03}}})}};
    input["domains"][1]["overrides"] = {
        {"motion",
         {{"angularVelocity", {0, 0, .02}}, {"center", {5, 5, 0}}}}};
    const auto factory = [](I iDomain, const auto &, const auto &domainMPI)
    { return CauchyTest::VortexPatch(domainMPI, iDomain == 1, 8, 20); };
    C::Simulation simulation(
        C::SimulationConfig::ResolveCollective(input, mpi.comm), mpi.comm,
        factory);
    REQUIRE(simulation.Advance(1e-4).accepted);
    for (I iDomain = 0; iDomain < 2; ++iDomain)
        if (simulation.Local(iDomain))
        {
            const auto &domain = simulation.GetDomain(iDomain);
            CHECK(domain.Dofs().PhysicalFieldNumber() == 2);
            const auto read = domain.Dofs().ReadOwned();
            const auto flow = read.Block("flow");
            const auto rans = read.Block("rans");
            for (I iCell = 0; iCell < flow.owned.RowCount(); ++iCell)
            {
                CHECK(std::isfinite(flow.owned(iCell, 0)));
                CHECK(std::isfinite(rans.owned(iCell, 0)));
                CHECK(flow.owned(iCell, 0) > 0);
                CHECK(rans.owned(iCell, 0) >= 0);
            }
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
