#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/IO/StateArchive.hpp"
#include "Cauchy/IO/AsyncOutput.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Physics/Thermodynamics/ConservativeConversion.hpp"
#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "DNDS/Runtime/Array/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <hdf5.h>
#include <filesystem>
#include <unistd.h>
#include <limits>
#include <fstream>
#include <sstream>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
static DNDS::MPIInfo mpi;
namespace
{
    std::string Path(const std::string &label)
    {
        int pid = int(getpid());
        MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
        return (std::filesystem::temp_directory_path() / (label + std::to_string(pid) + ".h5")).string();
    }
    C::FlowState Wave(const C::Domain &domain, DNDS::index iCell)
    {
        const auto point = domain.Discretization().Metrics().GetCellCent(iCell);
        return domain.Equation().Gas().Conservative({1 + .05 * std::sin(6.283185307179586 * (point[0] + point[1])), .3, -.1, .2, 1});
    }
    void Initialize(C::Domain &domain)
    {
        auto write = domain.Dofs().CoupledField().Owned().Write();
        for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
        {
            const auto value = Wave(domain, iCell);
            for (int j = 0; j < 5; ++j)
                write.View<R::UniformTableRows>()(iCell, j) = value[j];
        }
    }
    void Remove(const std::vector<std::string> &paths)
    {
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0)
            for (const auto &path : paths)
                CHECK(std::filesystem::remove(path));
        MPI_Barrier(mpi.comm);
    }
}

TEST_CASE("Accepted domain snapshots pin one immutable mean generation")
{
    C::DomainSettings settings;
    C::Domain domain("snapshot", CauchyTest::PeriodicQuads(mpi, 3), settings);
    {
        const auto snapshot = domain.CaptureStateSnapshot();
        const auto retained = snapshot;
        CHECK(snapshot.Name() == "snapshot");
        CHECK(snapshot.Time() == doctest::Approx(0));
        CHECK(snapshot.Epoch().acceptedStep == 0);
        CHECK(snapshot.Means().ComponentNumber() == domain.Dofs().ComponentNumber());
        CHECK_FALSE(snapshot.Reconstruction());
        CHECK_THROWS(domain.Dofs().WriteOwned());
        CHECK(retained.Means().PhysicalField(0).Owned().SharedStorage() ==
              snapshot.Means().PhysicalField(0).Owned().SharedStorage());
    }
    auto write = domain.Dofs().WriteOwned();
    write.CancelUnmodified();
}
TEST_CASE("Prepared checkpoint owns its fields metadata and communicator beyond the domain")
{
    const auto path = Path("cauchy-prepared-");
    C::DomainSettings settings;
    std::optional<C::PreparedCheckpoint> prepared;
    R::Array<DNDS::real> expected;
    {
        C::Domain source("prepared", CauchyTest::PeriodicQuads(mpi, 3), settings);
        Initialize(source);
        expected = source.Dofs().CoupledField().Owned().Clone();
        auto io = R::CommunicationContext::Duplicate(source.Discretization().Metrics().Communicator());
        prepared.emplace(source.PrepareCheckpoint(io));
        CHECK(prepared->Detached());
        io.reset();
        auto write = source.Dofs().WriteOwned();
        const auto flow = write.Block("flow");
        for (DNDS::index iCell = 0; iCell < flow.RowCount(); ++iCell)
            flow(iCell, 0) = 99;
    }
    prepared->Write(path, C::Json::object());
    C::Domain restored("prepared", CauchyTest::PeriodicQuads(mpi, 3), settings);
    restored.ReadCheckpoint(path, C::Json::object());
    const auto a = expected.Read(), b = restored.Dofs().CoupledField().Owned().Read();
    for (DNDS::index iCell = 0; iCell < expected.RowCount(); ++iCell)
        for (DNDS::index j = 0; j < 5; ++j)
            CHECK(a.View()(iCell, j) == b.View()(iCell, j));
    prepared.reset();
    Remove({path});
}
TEST_CASE("Detached host snapshots retain accepted values after the domain mutates")
{
    C::DomainSettings settings;
    C::Domain domain("detached", CauchyTest::PeriodicQuads(mpi, 3), settings);
    Initialize(domain);
    const auto detached = domain.CaptureStateSnapshot().DetachedHost(R::ExecutionContext::Host());
    CHECK(detached.Means().PhysicalField(0).Owned().SharedStorage() != domain.Dofs().PhysicalField(0).Owned().SharedStorage());
    C::EquationDofOperations::Fill(domain.Dofs(), 17);
    const auto read = detached.Means().ReadOwned();
    for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
    {
        const auto expected = Wave(domain, iCell);
        for (int j = 0; j < 5; ++j)
            CHECK(read.Block("flow")(iCell, j) == expected[j]);
    }
    CHECK(detached.Epoch().acceptedStep == 0);
    CHECK(detached.Time() == 0);
}

TEST_CASE("Asynchronous HDF output snapshots survive foreground field mutation and MPI")
{
    const auto prefix = Path("cauchy-async-");
    const C::Json input{{"defaults", {{"mesh", {{"dimension", 3}}}, {"boundaries", {{"7", {{"kind", "extrapolate"}}}}}}}, {"output", {{"prefix", prefix}, {"asynchronous", true}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    const auto factory = [](DNDS::index, const auto &, const auto &domainMPI)
    { return CFVTest::CartesianHexes(domainMPI, 2); };
    C::Simulation source(config, mpi.comm, factory);
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        CHECK_THROWS(C::AsyncOutput(source));
        return;
    }
    auto &domain = source.GetDomain(0);
    {
        auto write = domain.Dofs().WriteOwned();
        const auto flow = write.Block("flow");
        const auto value = domain.Equation().Gas().Conservative({1.3, .1, -.02, 0, 1});
        for (DNDS::index iCell = 0; iCell < flow.RowCount(); ++iCell)
            for (DNDS::index j = 0; j < 5; ++j)
                flow(iCell, j) = value[j];
    }
    const C::OutputPlan plan(config.Output(), 1);
    const std::string token = "-S0";
    C::AsyncOutput output(source);
    const C::Json marker{{"schema", "test.completed-output"}, {"step", 0}};
    output.Submit(source, config, plan, token, true, true, {}, marker);
    {
        auto write = domain.Dofs().WriteOwned();
        const auto flow = write.Block("flow");
        for (DNDS::index iCell = 0; iCell < flow.RowCount(); ++iCell)
            flow(iCell, 0) = 99;
    }
    // A foreground collective uses the original communicator while the file
    // worker uses independent I/O contexts. No source lease blocks this update.
    CHECK(DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Sum>(1, mpi.comm) == mpi.size);
    output.Drain();
    C::Simulation restored(config, mpi.comm, factory);
    restored.ReadCheckpoint(plan.CheckpointPath(token));
    {
        const auto read = restored.GetDomain(0).Dofs().ReadOwned();
        const auto flow = read.Block("flow");
        for (DNDS::index iCell = 0; iCell < flow.owned.RowCount(); ++iCell)
            CHECK(flow(iCell, 0) == 1.3);
    }
    const auto surface = plan.VisualizationPrefix(0, token) + "_surface_rank" + std::to_string(mpi.rank) + ".csv";
    std::ifstream rows(surface);
    std::string line;
    REQUIRE(bool(std::getline(rows, line)));
    while (std::getline(rows, line))
    {
        std::stringstream fields(line);
        std::string value;
        for (int column = 0; column <= 9; ++column)
            REQUIRE(bool(std::getline(fields, value, ',')));
        CHECK(std::stod(value) == doctest::Approx(1.3));
    }
    rows.close();
    CHECK(std::filesystem::remove(surface));
    if (mpi.rank == 0)
    {
        std::ifstream file(plan.SnapshotManifestPath(token));
        C::Json published;
        file >> published;
        CHECK(published == marker);
    }
    Remove({plan.CheckpointPath(token), plan.VisualizationPrefix(0, token) + ".vtkhdf", plan.SnapshotManifestPath(token)});
}
TEST_CASE("Equation archive remaps coupled and split fields without confusing component ranges")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 3);
    C::DomainSettings settings;
    C::FiniteVolume fv(mesh, settings, C::EquationSystem::Resolve(settings.equation));
    auto schema = C::EquationSystem::Resolve(settings.equation).Schema();
    schema.blocks.push_back({"tracers", {{"a", "1", "tracer"}, {"b", "1", "tracer"}}, {}});
    auto coupled = C::EquationDofs::Coupled(schema, fv.Metrics().BuildField(R::RowLayout::ColMajor(0, 7)));
    {
        auto write = coupled.WriteOwned();
        for (DNDS::index iBlock = 0; iBlock < coupled.BlockNumber(); ++iBlock)
        {
            const auto values = write.Block(iBlock);
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                for (DNDS::index j = 0; j < values.ComponentNumber(); ++j)
                    values(iCell, j) = 1000 * mpi.rank + 100 * iCell + 10 * iBlock + j;
        }
    }
    DNDS::Algebra::DofBundle fields(fv.Metrics().Space());
    fields.Add("flow", fv.Metrics().BuildField(R::RowLayout::Dense(0, 5)));
    fields.Add("tracers", fv.Metrics().BuildField(R::RowLayout::TiledColMajor(0, 2, 32)));
    auto split = C::EquationDofs::Separate(schema, std::move(fields));
    auto context = std::make_shared<R::CommunicationContext>(mpi.comm);
    std::vector<R::RowRequest> requests;
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
        requests.push_back({mpi.rank, mesh->NumCell() - 1 - iCell, iCell});
    R::RowTransferPlan route(context, mesh->NumCell(), mesh->NumCell(), requests);
    C::StateArchive archive{C::DescribeEquationStorage(coupled), {coupled.PhysicalField(0).Owned()}};
    archive.Restore(split, route, context);
    {
        const auto read = split.ReadOwned();
        for (DNDS::index iBlock = 0; iBlock < split.BlockNumber(); ++iBlock)
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                for (DNDS::index j = 0; j < read.Block(iBlock).ComponentNumber(); ++j)
                    CHECK(read.Block(iBlock)(iCell, j) == 1000 * mpi.rank + 100 * (mesh->NumCell() - 1 - iCell) + 10 * iBlock + j);
    }
    C::StateArchive splitArchive{C::DescribeEquationStorage(split), {split.PhysicalField(0).Owned(), split.PhysicalField(1).Owned()}};
    splitArchive.Restore(coupled, route, context);
    const auto read = coupled.ReadOwned();
    for (DNDS::index iBlock = 0; iBlock < coupled.BlockNumber(); ++iBlock)
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            for (DNDS::index j = 0; j < read.Block(iBlock).ComponentNumber(); ++j)
                CHECK(read.Block(iBlock)(iCell, j) == 1000 * mpi.rank + 100 * iCell + 10 * iBlock + j);
}
#ifdef DNDS_USE_CANTERA
namespace
{
    C::DomainSettings MixtureSettings()
    {
        C::DomainSettings settings;
        settings.equation.eos = "idealGasMixture";
        settings.equation.mixture.mechanismFile = "h2o2.yaml";
        settings.initialTemperature = 1000 / settings.equation.mixture.T0;
        settings.initialMassFractions = {{"H2", .1}, {"O2", .2}};
        settings.initialPrimitive = {1, .2, -.1, .03, 1};
        settings.execution.layout = "soa";
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 2;
        settings.numerics.reconstructionIterations = 4;
        settings.numerics.faceFallback = false;
        return settings;
    }
    void InitializeMixtureWave(C::Domain &domain)
    {
        auto state = domain.Dofs();
        auto write = state.WriteOwned();
        for (DNDS::index iBlock = 0; iBlock < domain.Dofs().BlockNumber(); ++iBlock)
        {
            const auto values = write.Block(iBlock);
            for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const auto point = domain.Discretization().Metrics().GetCellCent(iCell);
                const double scale = 1 + .05 * std::sin(6.283185307179586 * (point[0] + point[1]));
                for (DNDS::index j = 0; j < values.ComponentNumber(); ++j)
                    values(iCell, j) *= scale;
            }
        }
    }
    void CompareEquation(const C::EquationDofs &a, const C::EquationDofs &b, double tolerance = 0)
    {
        const auto left = a.ReadOwned(), right = b.ReadOwned();
        for (DNDS::index iBlock = 0; iBlock < a.BlockNumber(); ++iBlock)
            for (DNDS::index iCell = 0; iCell < left.Block(iBlock).owned.RowCount(); ++iCell)
                for (DNDS::index j = 0; j < left.Block(iBlock).ComponentNumber(); ++j)
                    if (tolerance == 0)
                        CHECK(left.Block(iBlock)(iCell, j) == right.Block(iBlock)(iCell, j));
                    else
                        CHECK(left.Block(iBlock)(iCell, j) == doctest::Approx(right.Block(iBlock)(iCell, j)).epsilon(tolerance));
    }
}
TEST_CASE("Mixture restart redistributes every block and cold starts changed polynomial degree")
{
    const auto path = Path("cauchy-mixture-redistributed-"), cold = path + ".cold.h5";
    auto settings = MixtureSettings();
    const C::Json metadata{{"test", "mixture persistent IDs"}};
    MPI_Comm writers = MPI_COMM_NULL;
    MPI_Comm_split(mpi.comm, mpi.rank < std::max(1, mpi.size / 2) ? 0 : MPI_UNDEFINED, mpi.rank, &writers);
    if (writers != MPI_COMM_NULL)
    {
        {
            C::Domain source("mixture", CauchyTest::PeriodicQuads(DNDS::MPIInfo(writers), 4), settings);
            InitializeMixtureWave(source);
            source.Residual();
            source.WriteCheckpoint(path, metadata, true);
        }
        MPI_Comm_free(&writers);
    }
    MPI_Barrier(mpi.comm);
    settings.execution.layout = "aosoa32";
    settings.numerics.cfv.maxOrder = 3;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::Domain restored("mixture", mesh, settings), expected("mixture", mesh, settings);
    InitializeMixtureWave(expected);
    restored.ReadCheckpoint(path, metadata);
    CompareEquation(restored.Dofs(), expected.Dofs());
    restored.WriteCheckpoint(cold, metadata, true);
    auto file = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
    file->OpenFile(cold, true);
    std::string text;
    file->ReadString("metadata_json", text);
    const auto description = C::Json::parse(text).at("reconstruction_storage");
    const auto range = DNDS::EvenSplitRange(mpi.rank, mpi.size, 16);
    const auto coefficients = C::StateArchive::Read(file, "reconstruction_field_", description, mpi, {range.second - range.first, range.first});
    for (const auto &field : coefficients.fields)
    {
        const auto read = field.Read();
        for (DNDS::index iCell = 0; iCell < field.RowCount(); ++iCell)
            for (DNDS::index j = 0; j < field.RowSize(iCell); ++j)
                CHECK(read.View()(iCell, j) == 0);
    }
    file->CloseFile();
    Remove({path, cold});
}
TEST_CASE("Mixture BDF2 restart restores species history and warm reconstruction transactionally")
{
    const auto path = Path("cauchy-mixture-history-");
    auto settings = MixtureSettings();
    C::TimeSettings time;
    time.method = "bdf2";
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 3);
    C::Domain source("mixture", mesh, settings, time);
    InitializeMixtureWave(source);
    REQUIRE(source.Advance(1e-5).accepted);
    REQUIRE(source.Advance(2e-5).accepted);
    const C::Json metadata{{"test", "mixture history"}};
    source.WriteCheckpoint(path, metadata, true);
    settings.execution.layout = "aosoa32";
    C::Domain restored("mixture", mesh, settings, time);
    const auto before = restored.Dofs().Clone();
    const auto speciesField = restored.Dofs().LocateBlock("species").physicalField;
    std::optional<R::Array<double>::ReadLease> held;
    if (mpi.rank == 0)
        held.emplace(restored.Dofs().PhysicalField(speciesField).Owned().Read());
    CHECK_THROWS(restored.ReadCheckpoint(path, metadata));
    CompareEquation(restored.Dofs(), before);
    CHECK(restored.Time() == 0);
    held.reset();
    restored.ReadCheckpoint(path, metadata);
    CompareEquation(restored.Dofs(), source.Dofs());
    CHECK(restored.PreviousTimeStep() == source.PreviousTimeStep());
    REQUIRE(source.Advance(1.5e-5).accepted);
    REQUIRE(restored.Advance(1.5e-5).accepted);
    CompareEquation(restored.Dofs(), source.Dofs(), 2e-13);
    Remove({path});
}
TEST_CASE("Physical Strang source composes with transport and preserves retry and restart history semantics")
{
    for (const std::string method : {"ssprk3", "esdirk2", "bdf2"})
    {
        CAPTURE(method);
        auto settings = MixtureSettings();
        settings.equation.source = "chemical";
        settings.numerics.positivity = true;
        settings.implicit.linear.method = "ilu";
        settings.implicit.multigrid.levels = method == "esdirk2" ? 1 : 0;
        C::TimeSettings time;
        time.method = method;
        time.sourceSplitting = "strang";
        const auto mesh = CauchyTest::PeriodicQuads(mpi, 3);
        C::Domain source("mixture", mesh, settings, time);
        CHECK(source.UsesStrangSource());
        InitializeMixtureWave(source);
        auto transportSettings = settings;
        transportSettings.equation.source = "none";
        auto transportTime = time;
        transportTime.sourceSplitting = "coupled";
        // Euler resets BDF history around each source jump: its BDF2
        // transport takes the startup BDF1 equation on every split step.
        if (method == "bdf2")
            transportTime.method = "bdf1";
        C::Domain reference("mixture", mesh, transportSettings, transportTime);
        InitializeMixtureWave(reference);
        auto firstStep = source.Dofs().Clone();
        const auto scales = source.Equation().Mixture().Scales();
        const auto *directory = std::getenv("DNDS_MECH_PATH");
        const auto mechanism = directory ? std::string(directory) + "/h2o2.yaml" : "h2o2.yaml";
        DNDS::Physics::Chemistry::ChemicalSource oracle(mechanism, "", scales.U0, scales.rho0);
        const auto halfStep = [&](C::Domain &domain, double duration)
        {
            auto state = domain.Dofs();
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow"), rhoY = write.Block("species");
            std::vector<double> fractions(size_t(oracle.nSpecies()));
            for (DNDS::index iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                double sum = 0;
                for (int k = 0; k < oracle.nSpecies() - 1; ++k)
                    sum += fractions[k] = std::max(0., rhoY(iCell, k) / flow(iCell, 0));
                fractions.back() = std::max(0., 1 - sum);
                sum += fractions.back();
                for (auto &value : fractions)
                    value /= sum;
                C::FlowState u;
                for (int j = 0; j < 5; ++j)
                    u[j] = flow(iCell, j);
                const double energy = DNDS::Physics::Thermodynamics::ConservativeSpecificInternalEnergy<3>(u) * scales.SpecificEnergy();
                double temperature = oracle.temperatureFromUV(energy, 1 / (u[0] * scales.rho0), {fractions.data(), oracle.nSpecies()});
                oracle.advanceConstVolume(temperature, u[0] * scales.rho0, {fractions.data(), oracle.nSpecies()}, 1,
                                          duration * scales.Time(), time.reactor.rtol, time.reactor.atol, time.reactor.maxOrder, time.reactor.maxSteps);
                sum = 0;
                for (int k = 0; k < oracle.nSpecies() - 1; ++k)
                    sum += fractions[k] = std::max(0., fractions[k]);
                for (int k = 0; k < oracle.nSpecies() - 1; ++k)
                    rhoY(iCell, k) = u[0] * fractions[k] * (sum >= 1 ? (1 - 1e-14) / sum : 1);
            }
        };
        for (const double dt : {1e-5, 1.5e-5})
        {
            REQUIRE(source.Advance(dt).accepted);
            halfStep(reference, .5 * dt);
            REQUIRE(reference.Advance(dt).accepted);
            halfStep(reference, .5 * dt);
            CompareEquation(source.Dofs(), reference.Dofs(), 3e-10);
            if (dt == 1e-5)
                C::EquationDofOperations::Copy(firstStep, source.Dofs());
        }
        if (method == "esdirk2")
        {
            const C::Json config{{"defaults", C::Json(settings)}, {"time", C::Json(time)}, {"domains", C::Json::array({{{"name", "a"}}, {{"name", "b"}}})}};
            C::Simulation coordinated(C::SimulationConfig::ResolveCollective(config, mpi.comm), mpi.comm,
                                      [](DNDS::index, const auto &, const auto &domainMPI)
                                      { return CauchyTest::PeriodicQuads(domainMPI, 3); });
            for (DNDS::index iDomain = 0; iDomain < 2; ++iDomain)
                InitializeMixtureWave(coordinated.GetDomain(iDomain));
            REQUIRE(coordinated.Advance(1e-5).accepted);
            for (DNDS::index iDomain = 0; iDomain < 2; ++iDomain)
                CompareEquation(coordinated.GetDomain(iDomain).Dofs(), firstStep, 3e-10);
        }
        const auto path = Path("cauchy-strang-" + method);
        const C::Json metadata{{"test", "physical Strang source"}};
        source.WriteCheckpoint(path, metadata, true);
        auto file = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
        file->OpenFile(path, true);
        std::string serialized;
        file->ReadString("metadata_json", serialized);
        CHECK(C::Json::parse(serialized)["source_integration"] == "strang");
        CHECK_FALSE(C::Json::parse(serialized)["history"].get<bool>());
        file->CloseFile();
        C::Domain restored("mixture", mesh, settings, time);
        restored.ReadCheckpoint(path, metadata);
        const auto accepted = source.Dofs().Clone();
        REQUIRE(source.PrepareStep(1.2e-5).accepted);
        source.DiscardStep();
        CompareEquation(source.Dofs(), accepted);
        REQUIRE(source.Advance(1.2e-5).accepted);
        REQUIRE(restored.Advance(1.2e-5).accepted);
        CompareEquation(source.Dofs(), restored.Dofs(), 2e-13);
        Remove({path});
    }
}
TEST_CASE("Mixture visualization writes conservative species and physical boundary traces")
{
    const auto prefix = Path("cauchy-mixture-output-");
    auto settings = MixtureSettings();
    settings.mesh.dimension = 3;
    settings.boundaries["7"].kind = "extrapolate";
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    C::Domain domain("mixture", mesh, settings);
    const auto before = domain.Dofs().Clone();
    domain.WriteVisualization(prefix, 1);
    CompareEquation(domain.Dofs(), before);
    const auto file = H5Fopen((prefix + ".vtkhdf").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    for (const auto &field : std::vector<std::pair<std::string, double>>{{"rho", 1}, {"rhoY:H2", .1}, {"Y:H2", .1}, {"Y:O2", .2}, {"Y:N2", .7}, {"T", settings.initialTemperature}})
    {
        const auto dataset = H5Dopen(file, ("VTKHDF/CellData/" + field.first).c_str(), H5P_DEFAULT);
        REQUIRE(dataset >= 0);
        const auto space = H5Dget_space(dataset);
        REQUIRE(H5Sget_simple_extent_npoints(space) == 1);
        double value = 0;
        CHECK(H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) >= 0);
        CHECK(value == doctest::Approx(field.second).epsilon(1e-12));
        H5Sclose(space);
        H5Dclose(dataset);
    }
    H5Fclose(file);
    const auto surfacePath = prefix + "_surface_rank" + std::to_string(mpi.rank) + ".csv";
    std::ifstream surface(surfacePath);
    std::string line;
    REQUIRE(bool(std::getline(surface, line)));
    CHECK(line.find("w,T,Y:H2,") != std::string::npos);
    DNDS::index rows = 0;
    while (std::getline(surface, line))
    {
        std::istringstream record(line);
        std::vector<double> values;
        std::string token;
        while (std::getline(record, token, ','))
            values.push_back(std::stod(token));
        REQUIRE(values.size() == 25);
        CHECK(values[14] == doctest::Approx(settings.initialTemperature).epsilon(1e-12));
        CHECK(values[15] == doctest::Approx(.1).epsilon(1e-12));
        ++rows;
    }
    MPI_Allreduce(MPI_IN_PLACE, &rows, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
    CHECK(rows > 0);
    surface.close();
    CHECK(std::filesystem::remove(surfacePath));
    Remove({prefix + ".vtkhdf"});
}
#endif
TEST_CASE("Configured RANS BDF2 restores all fields across layouts and rejects changed model metadata")
{
    for (bool twoEquation : {false, true})
    {
        const auto path = Path("cauchy-sa-history-");
        C::DomainSettings settings;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .001;
        settings.equation.rans.model = "SA";
        settings.equation.rans.viscosityReference = .01;
        settings.initialPrimitive = {1, .1, 0, 0, 1};
        settings.initialRANS = {{"rhoNuTilde", .02}};
        if (twoEquation)
        {
            settings.equation.rans.model = "KOmegaSST";
            settings.initialRANS = {{"rhoK", .02}, {"rhoOmega", .1}};
            settings.numerics.ransState.referenceConservative = settings.initialRANS;
        }
        settings.execution.layout = "soa";
        settings.execution.fieldGrouping = "equationBlocks";
        settings.execution.hostThreads = 2;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 2;
        settings.numerics.reconstructionIterations = 4;
        settings.numerics.faceFallback = false;
        settings.numerics.limiter = "none";
        settings.numerics.viscousGradient = "polynomial";
        settings.implicit.diagonal = "block";
        settings.implicit.residualComponents = twoEquation ? std::vector<std::string>{"rho", "rhoK", "rhoOmega"} : std::vector<std::string>{"rho", "rhoNuTilde"};
        settings.implicit.linear.method = "gmres";
        settings.implicit.linear.preconditioner = "ilu";
        C::TimeSettings time;
        time.method = "bdf2";
        const auto config = C::SimulationConfig::Resolve({{"defaults", settings}, {"time", time}}, mpi.size);
        const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
        CHECK_FALSE(mesh->HasNodeWallDist());
        C::Domain source("sa", mesh, config.Domains()[0].settings, config.Time());
        CHECK(mesh->HasNodeWallDist());
        CHECK(source.Dofs().PhysicalFieldNumber() == 2);
        {
            auto state = source.Dofs();
            auto write = state.WriteOwned();
            const auto sa = write.Block("rans");
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                sa(iCell, 0) *= 1 + .02 * std::sin(6.283185307179586 * source.Discretization().Metrics().GetCellCent(iCell)[0]);
        }
        CHECK(source.StableStep(.5) > 0);
        REQUIRE(source.Advance(1e-4).accepted);
        REQUIRE(source.Advance(2e-4).accepted);
        REQUIRE(!source.LastImplicitStages().empty());
        CHECK(source.LastImplicitStages().back().converged);
        const C::Json metadata = config.Root();
        auto olderConfig = metadata;
        olderConfig["domains"][0]["settings"]["equation"]["rans"].erase("useQCR");
        source.WriteCheckpoint(path, olderConfig, true);
        std::string olderModel;
        {
            auto file = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
            file->OpenFile(path, true);
            std::string text;
            file->ReadString("metadata_json", text);
            auto description = C::Json::parse(text);
            CHECK(description.at("flow").at("rans").at("model") == settings.equation.rans.model);
            description["flow"]["rans"].erase("useQCR");
            olderModel = description.dump();
            file->CloseFile();
        }
        // Emulate the actual earlier on-disk model metadata, not just an old
        // caller config. Write the shorter JSON into its fixed-size attribute.
        DNDS::MPI::LocalStage(mpi, [&]
                              {
            if (mpi.rank != 0) return;
            const auto file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
            DNDS_check_throw_info(file >= 0, "cannot edit checkpoint fixture metadata");
            const auto attribute = H5Aopen(file, "metadata_json", H5P_DEFAULT), type = H5Aget_type(attribute);
            const auto size = H5Tget_size(type);
            DNDS_check_throw_info(H5Tis_variable_str(type) == 0 && olderModel.size() <= size, "fixture metadata string shape changed");
            olderModel.resize(size, '\0');
            CHECK(H5Awrite(attribute, type, olderModel.data()) >= 0);
            H5Tclose(type); H5Aclose(attribute); H5Fclose(file); });
        settings.execution.layout = "aosoa32";
        settings.execution.fieldGrouping = "coupled";
        C::Domain restored("sa", mesh, settings, time);
        CHECK(restored.Dofs().PhysicalFieldNumber() == 1);
        restored.ReadCheckpoint(path, metadata);
        const auto compare = [&](double tolerance)
        {
            const auto a = source.Dofs().ReadOwned(), b = restored.Dofs().ReadOwned();
            for (const auto *name : {"flow", "rans"})
            {
                const auto u = a.Block(name), v = b.Block(name);
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                    for (DNDS::index j = 0; j < u.ComponentNumber(); ++j)
                        CHECK(std::abs(u(iCell, j) - v(iCell, j)) <= tolerance);
            }
        };
        compare(0);
        CHECK(restored.PreviousTimeStep() == source.PreviousTimeStep());
        REQUIRE(source.Advance(1.5e-4).accepted);
        REQUIRE(restored.Advance(1.5e-4).accepted);
        compare(2e-12);
        settings.equation.rans.useQCR = true;
        C::Domain different("sa", mesh, settings, time);
        const C::EquationDofOperations::Binding before(different.Dofs());
        CHECK_THROWS(different.ReadCheckpoint(path, metadata));
        CHECK_NOTHROW(before.Check(different.Dofs(), true));
        Remove({path});
    }
}
TEST_CASE("Older viscous archives retain primitive gradients and reject a silent polynomial method change")
{
    const auto path = Path("cauchy-viscous-policy-");
    const C::Json input{{"defaults", {{"equation", {{"transport", "NewtonianFourier"}, {"viscosity", .02}}}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 2);
    C::Domain source("flow", mesh, config.Domains()[0].settings);
    Initialize(source);
    auto old = config.Root();
    auto &numerics = old["domains"][0]["settings"]["numerics"];
    for (const auto *name : {"viscousGradient", "viscousLimited", "noGRPOnWall"})
        numerics.erase(name);
    numerics.erase("firstOrderConvection");
    source.WriteCheckpoint(path, old);
    C::Domain restored("flow", mesh, config.Domains()[0].settings);
    CHECK_NOTHROW(restored.ReadCheckpoint(path, config.Root()));
    {
        const auto a = source.Dofs().CoupledField().Owned().Read(), b = restored.Dofs().CoupledField().Owned().Read();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(a.View<R::UniformTableRows>()(iCell, j) == b.View<R::UniformTableRows>()(iCell, j));
    }
    auto polynomialInput = input;
    polynomialInput["defaults"]["numerics"]["viscousGradient"] = "polynomial";
    const auto polynomial = C::SimulationConfig::ResolveCollective(polynomialInput, mpi.comm);
    C::Domain changed("flow", mesh, polynomial.Domains()[0].settings);
    const auto version = changed.Dofs().CoupledField().Owned().Version();
    CHECK_THROWS(changed.ReadCheckpoint(path, polynomial.Root()));
    CHECK(changed.Dofs().CoupledField().Owned().Version() == version);
    auto meanConvection = config.Root();
    meanConvection["domains"][0]["settings"]["numerics"]["firstOrderConvection"] = true;
    CHECK_THROWS(restored.ReadCheckpoint(path, meanConvection));
    Remove({path});
}
TEST_CASE("Older physical-boundary archives omit inactive composition without invalidating restart")
{
    const auto path = Path("cauchy-boundary-composition-");
    const C::Json input{{"defaults", {{"mesh", {{"dimension", 3}}}, {"boundaries", {{"7", {{"kind", "extrapolate"}}}}}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    C::Domain source("flow", mesh, config.Domains()[0].settings);
    auto old = config.Root();
    old["domains"][0]["settings"]["boundaries"]["7"].erase("massFractions");
    source.WriteCheckpoint(path, old);
    C::Domain restored("flow", mesh, config.Domains()[0].settings);
    CHECK_NOTHROW(restored.ReadCheckpoint(path, config.Root()));
    const auto version = restored.Dofs().CoupledField().Owned().Version();
    auto changed = config.Root();
    changed["domains"][0]["settings"]["boundaries"]["7"]["massFractions"] = C::Json{{"H2", .1}};
    CHECK_THROWS(restored.ReadCheckpoint(path, changed));
    CHECK(restored.Dofs().CoupledField().Owned().Version() == version);
    Remove({path});
}
TEST_CASE("Checkpoint changes MPI partition and polynomial degree using persistent cell IDs with cold reconstruction")
{
    const auto plain = Path("cauchy-id-plain-"), polynomial = plain + ".rec.h5", cold = plain + ".cold.h5";
    C::Json input{{"defaults", {{"numerics", {{"reconstruction", "variational"}, {"reconstructionIterations", 40}, {"cfv", {{"maxOrder", 2}}}}}}}};
    MPI_Comm writerComm = MPI_COMM_NULL;
    const int writerRanks = std::max(1, mpi.size / 2);
    MPI_Comm_split(mpi.comm, mpi.rank < writerRanks ? 0 : MPI_UNDEFINED, mpi.rank, &writerComm);
    if (writerComm != MPI_COMM_NULL)
    {
        {
            const DNDS::MPIInfo writers(writerComm);
            const auto config = C::SimulationConfig::ResolveCollective(input, writerComm);
            C::Domain source("fluid", CauchyTest::PeriodicQuads(writers, 6), config.Domains()[0].settings);
            Initialize(source);
            source.Residual().CoupledField();
            source.WriteCheckpoint(plain, config.Root());
            source.WriteCheckpoint(polynomial, config.Root(), true);
            auto serializer = std::make_shared<DNDS::Serializer::SerializerH5>(writers);
            serializer->OpenFile(plain, true);
            CHECK_FALSE(serializer->ListCurrentPath().count("reconstruction_field_0"));
            CHECK(serializer->ListCurrentPath().count("metadata_json") == 1);
            serializer->CloseFile();
            serializer->OpenFile(polynomial, true);
            std::string metadataText;
            serializer->ReadString("metadata_json", metadataText);
            const auto descriptor = C::Json::parse(metadataText).at("reconstruction_storage");
            const auto archive = C::StateArchive::Read(serializer, "reconstruction_field_", descriptor, writers,
                                                       DNDS::Serializer::ArrayGlobalOffset_Unknown);
            REQUIRE(archive.fields.size() == 1);
            const auto &rec = archive.fields[0];
            serializer->CloseFile();
            double norm = 0;
            {
                const auto read = rec.Read();
                for (DNDS::index iCell = 0; iCell < rec.RowCount(); ++iCell)
                    for (DNDS::index j = 0; j < rec.RowSize(iCell); ++j)
                        norm += std::abs(read.View()(iCell, j));
            }
            MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, writerComm);
            CHECK(norm > 1e-8); // A zero fixture could not prove cold-start conversion.
        }
        MPI_Comm_free(&writerComm);
    }
    MPI_Barrier(mpi.comm);
    input["defaults"]["numerics"]["cfv"]["maxOrder"] = 3;
    // Iteration budgets/tolerances are restart controls, not stored physics.
    // Retune them together with the existing degree/layout/partition migration.
    input["defaults"]["numerics"]["reconstructionIterations"] = 80;
    input["defaults"]["numerics"]["reconstructionTolerance"] = 1e-12;
    input["defaults"]["execution"]["layout"] = "soa";
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    auto targetMesh = CauchyTest::PeriodicQuads(mpi, 6);
    std::vector<DNDS::index> before;
    for (DNDS::index iCell = 0; iCell < targetMesh->NumCell(); ++iCell)
        before.push_back(targetMesh->OriginalEntityId(DNDS::Geom::EntityKind::Cell, iCell));
    targetMesh->ReorderLocalCells();
    DNDS::index changed = 0;
    for (DNDS::index iCell = 0; iCell < targetMesh->NumCell(); ++iCell)
        changed += before[iCell] != targetMesh->OriginalEntityId(DNDS::Geom::EntityKind::Cell, iCell);
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
    CHECK(changed > 0); // Prove this fixture exercises a real mesh permutation.
    C::Domain target("fluid", targetMesh, config.Domains()[0].settings);
    for (const auto &path : {plain, polynomial})
    {
        target.ReadCheckpoint(path, config.Root());
        const auto read = target.Dofs().CoupledField().Owned().Read();
        const auto values = read.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
        {
            const auto expected = Wave(target, iCell);
            for (int j = 0; j < 5; ++j)
                CHECK(values(iCell, j) == expected[j]);
        }
    }
    auto incompatible = config.Root();
    incompatible["domains"][0]["settings"]["numerics"]["flux"] = "Roe";
    const auto version = target.Dofs().CoupledField().Owned().Version();
    CHECK_THROWS(target.ReadCheckpoint(plain, incompatible));
    CHECK(target.Dofs().CoupledField().Owned().Version() == version);
    target.WriteCheckpoint(cold, config.Root(), true);
    auto serializer = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
    serializer->OpenFile(cold, true);
    std::string metadataText;
    serializer->ReadString("metadata_json", metadataText);
    const auto descriptor = C::Json::parse(metadataText).at("reconstruction_storage");
    const auto archive = C::StateArchive::Read(serializer, "reconstruction_field_", descriptor, mpi,
                                               DNDS::Serializer::ArrayGlobalOffset_Unknown);
    REQUIRE(archive.fields.size() == 1);
    const auto &coefficients = archive.fields[0];
    serializer->CloseFile();
    {
        const auto read = coefficients.Read();
        CHECK(coefficients.Layout().Width() == 45);
        for (DNDS::index iCell = 0; iCell < coefficients.RowCount(); ++iCell)
            for (DNDS::index j = 0; j < coefficients.RowSize(iCell); ++j)
                CHECK(read.View()(iCell, j) == 0);
    }
    Remove({plain, polynomial, cold});
}
TEST_CASE("ACM BDF2 restart preserves the pressure gauge and differential history")
{
    const auto path = Path("cauchy-acm-history-");
    C::DomainSettings settings;
    settings.equation.system = "incompressibleACM";
    settings.equation.eos = "none";
    settings.equation.referenceScales = {{"rho0", 1}, {"U0", 1}, {"L0", 1}, {"T0", 300}};
    settings.initialState = {{"type", "primUP"}, {"state", {.5, 0, 0, -2}}};
    settings.numerics.reconstruction = "constant";
    settings.implicit.jacobianAction = "assembled";
    settings.implicit.diagonal = "block";
    settings.implicit.residualComponents = {"u", "v", "p"};
    C::TimeSettings time;
    time.method = "bdf2";
    auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::Domain direct("acm", mesh, settings, time), resumed("acm", mesh, settings, time);
    {
        auto write = direct.Dofs().WriteOwned();
        const auto flow = write.Block("flow");
        for (DNDS::index iCell = 0; iCell < flow.RowCount(); ++iCell)
            flow(iCell, 1) = .1 * std::sin(2 * 3.141592653589793 * direct.Discretization().Metrics().GetCellCent(iCell)[0]);
    }
    REQUIRE(direct.Advance(.01).accepted);
    REQUIRE(direct.Advance(.007).accepted);
    const C::Json metadata{{"case", "ACM restart"}};
    direct.WriteCheckpoint(path, metadata, true);
    resumed.ReadCheckpoint(path, metadata);
    REQUIRE(resumed.PreviousTimeStep() == .007);
    REQUIRE(direct.Advance(.008).accepted);
    REQUIRE(resumed.Advance(.008).accepted);
    auto a = direct.Dofs().ReadOwned(), b = resumed.Dofs().ReadOwned();
    double difference = 0;
    for (DNDS::index iCell = 0; iCell < a.Block("flow").RowCount(); ++iCell)
        for (int j = 0; j < 4; ++j)
            difference = std::max(difference, std::abs(a.Block("flow")(iCell, j) - b.Block("flow")(iCell, j)));
    CHECK(difference < 1e-14);
    Remove({path});
}
TEST_CASE("A mean-only initial checkpoint can seed a different nonlinear stage policy")
{
    const auto path = Path("cauchy-initial-method-");
    C::DomainSettings settings;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::TimeSettings time;
    time.method = "bdf1";
    C::Domain source("fluid", mesh, settings, time);
    Initialize(source);
    time.method = "ditr";
    C::Domain restored("fluid", mesh, settings, time);
    const C::Json metadata{{"case", "initial means"}};
    source.WriteCheckpoint(path, metadata, false);
    restored.ReadCheckpoint(path, metadata);
    REQUIRE(restored.Time() == 0);
    const auto expected = source.Dofs().ReadOwned();
    const auto actual = restored.Dofs().ReadOwned();
    for (DNDS::index iCell = 0; iCell < actual.Block("flow").RowCount(); ++iCell)
        for (int j = 0; j < 5; ++j)
            CHECK(actual.Block("flow")(iCell, j) == expected.Block("flow")(iCell, j));
    Remove({path});
}

TEST_CASE("Implicit restart preserves variable BDF2 history and rebuilds live DITR stage bindings")
{
    for (const std::string method : {"bdf2", "ditr"})
    {
        const auto path = Path("cauchy-" + method + "-history-");
        C::DomainSettings settings;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 2;
        settings.numerics.reconstructionIterations = 80;
        settings.numerics.reconstructionTolerance = 1e-11;
        settings.implicit.minIterations = 0;
        settings.implicit.relativeTolerance = 1e-9;
        settings.implicit.linear.relativeTolerance = 1e-3;
        C::TimeSettings time;
        time.method = method;
        const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
        C::Domain source("flow", mesh, settings, time), restored("flow", mesh, settings, time);
        Initialize(source);
        REQUIRE(source.Advance(.001).accepted);
        REQUIRE(source.Advance(.002).accepted);
        const C::Json metadata{{"test", method + " native continuation"}};
        source.WriteCheckpoint(path, metadata, true);
        if (method == "ditr")
            REQUIRE(restored.Advance(.0007).accepted); // build an old intermediate binding before replacement
        const auto oldTime = restored.Time();
        const auto oldDt = restored.PreviousTimeStep();
        const auto oldStep = restored.Epoch().acceptedStep;
        std::optional<R::Array<double>::ReadLease> held;
        if (mpi.rank == 0)
            held.emplace(restored.Dofs().CoupledField().Owned().Read());
        CHECK_THROWS(restored.ReadCheckpoint(path, metadata));
        CHECK(restored.Time() == oldTime);
        CHECK(restored.PreviousTimeStep() == oldDt);
        CHECK(restored.Epoch().acceptedStep == oldStep);
        held.reset();
        restored.ReadCheckpoint(path, metadata);
        CHECK(restored.Time() == source.Time());
        CHECK(restored.PreviousTimeStep() == .002);
        CHECK(restored.Epoch().acceptedStep == source.Epoch().acceptedStep);
        REQUIRE(source.Advance(.0015).accepted);
        REQUIRE(restored.Advance(.0015).accepted);
        const auto a = source.Dofs().CoupledField().Owned().Read(), b = restored.Dofs().CoupledField().Owned().Read();
        for (DNDS::index iCell = 0; iCell < source.Discretization().Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(a.View()(iCell, j) == b.View()(iCell, j));
        Remove({path});
    }
}
TEST_CASE("Restart dt growth reads the accepted scalar and supports files predating it")
{
    for (const std::string method : {"bdf2", "trapz"})
    {
        const auto path = Path("cauchy-previous-step-" + method);
        const C::Json metadata{{"test", "physical timestep history"}};
        C::DomainSettings settings;
        settings.numerics.reconstruction = "constant";
        settings.implicit.minIterations = 0;
        C::TimeSettings time;
        time.method = method;
        time.positivity.enabled = true;
        auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
        C::Domain source("flow", mesh, settings, time), restored("flow", mesh, settings, time);
        REQUIRE(source.Advance(.001).accepted);
        REQUIRE(source.Advance(.002).accepted);
        source.WriteCheckpoint(path, metadata);
        restored.ReadCheckpoint(path, metadata);
        CHECK(restored.PreviousTimeStep() == .002);
        CHECK(restored.LimitPhysicalStep(.006).dt == .004);
        DNDS::MPI::LocalStage(mpi, [&]
                              {
            if (mpi.rank == 0)
            {
                const auto file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                DNDS_check_throw_info(file >= 0, "cannot open legacy-dt test fixture");
                const auto removed = H5Adelete(file, "last_accepted_dt");
                H5Fclose(file);
                DNDS_check_throw_info(removed >= 0, "cannot remove accepted-dt fixture attribute");
            } });
        restored.ReadCheckpoint(path, metadata);
        CHECK(restored.PreviousTimeStep() == (method == "bdf2" ? .002 : 0));
        CHECK(restored.LimitPhysicalStep(.006).dt == (method == "bdf2" ? .004 : .006));
        // Nonfinite, impossible-clock and conflicting-history scalar input
        // must fail before publishing means or changing accepted history.
        for (const double invalid : {std::numeric_limits<double>::quiet_NaN(), .1, method == "bdf2" ? .001 : -.001})
        {
            source.WriteCheckpoint(path, metadata);
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                if (mpi.rank == 0)
                {
                    const auto file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                    DNDS_check_throw_info(file >= 0, "cannot open invalid-dt test fixture");
                    const auto attribute = H5Aopen(file, "last_accepted_dt", H5P_DEFAULT);
                    DNDS_check_throw_info(attribute >= 0, "missing accepted-dt fixture attribute");
                    const auto written = H5Awrite(attribute, H5T_NATIVE_DOUBLE, &invalid);
                    H5Aclose(attribute); H5Fclose(file);
                    DNDS_check_throw_info(written >= 0, "cannot write invalid accepted-dt fixture");
                } });
            const auto version = restored.Dofs().CoupledField().Owned().Version();
            const auto previous = restored.PreviousTimeStep(), clock = restored.Time();
            CHECK_THROWS(restored.ReadCheckpoint(path, metadata));
            CHECK(restored.Dofs().CoupledField().Owned().Version() == version);
            CHECK(restored.PreviousTimeStep() == previous);
            CHECK(restored.Time() == clock);
        }
        Remove({path});
    }
}
TEST_CASE("CGNS checkpoint remains physical across independently built MPI partitions")
{
    const auto path = Path("cauchy-cgns-repartition-");
    // Check fixture availability before splitting writer participants. An absent
    // mesh must fail all ranks, not leave nonwriters waiting after a writer exits.
    DNDS::MPI::LocalStage(mpi, []
                          { DNDS_check_throw_info(std::filesystem::is_regular_file(std::string(DNDS_TEST_SOURCE_ROOT) + "/data/mesh/IV10_20.cgns"),
                                                  "checkpoint partition test requires data/mesh/IV10_20.cgns"); });
    const C::Json input{{"defaults", {{"mesh", {{"file", std::string(DNDS_TEST_SOURCE_ROOT) + "/data/mesh/IV10_20.cgns"}, {"dimension", 2}, {"periodicLengths", {10, 10, 0}}, {"boundaryIds", {{"PERIODIC_1", -1}, {"PERIODIC_1_DONOR", -4}, {"PERIODIC_2", -2}, {"PERIODIC_2_DONOR", -5}}}}}, {"numerics", {{"reconstruction", "constant"}, {"cfv", {{"maxOrder", 1}}}}}}}};
    const auto value = [](const C::Domain &domain, DNDS::index iCell)
    {
        const auto id = domain.Discretization().Mesh().OriginalEntityId(DNDS::Geom::EntityKind::Cell, iCell);
        return domain.Equation().Gas().Conservative({1 + .0001 * id, .31, -.12, .07, 1.2});
    };
    MPI_Comm writerComm = MPI_COMM_NULL;
    MPI_Comm_split(mpi.comm, mpi.rank < std::max(1, mpi.size / 2) ? 0 : MPI_UNDEFINED, mpi.rank, &writerComm);
    if (writerComm != MPI_COMM_NULL)
    {
        {
            const auto config = C::SimulationConfig::ResolveCollective(input, writerComm);
            C::Simulation simulation(config, writerComm);
            auto &domain = simulation.GetDomain(0);
            {
                auto write = domain.Dofs().CoupledField().Owned().Write();
                for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
                {
                    const auto expected = value(domain, iCell);
                    for (int j = 0; j < 5; ++j)
                        write.View()(iCell, j) = expected[j];
                }
            }
            domain.WriteCheckpoint(path, config.Root());
        }
        MPI_Comm_free(&writerComm);
    }
    MPI_Barrier(mpi.comm);
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    C::Simulation restored(config, mpi.comm);
    restored.ReadCheckpoint(path);
    const auto &domain = restored.GetDomain(0);
    const auto read = domain.Dofs().CoupledField().Owned().Read();
    for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
    {
        const auto expected = value(domain, iCell);
        for (int j = 0; j < 5; ++j)
            CHECK(read.View()(iCell, j) == expected[j]);
    }
    Remove({path});
}
TEST_CASE("Rigid motion restart retains absolute-state convention and BDF history")
{
    const auto path = Path("cauchy-rigid-frame-");
    const C::Json input{{"defaults", {{"motion", {{"angularVelocity", {0, 0, .2}}, {"center", {.4, .3, 0}}}}, {"numerics", {{"reconstruction", "constant"}}}}}, {"time", {{"method", "bdf2"}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 3);
    C::Domain source("flow", mesh, config.Domains()[0].settings, config.Time());
    REQUIRE(source.Advance(.001).accepted);
    const C::Json outputController{{"schema_version", 1}, {"test_cursor", 9}};
    source.WriteCheckpoint(path, config.Root(), false, outputController);
    {
        auto serializer = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, true);
        std::string metadata;
        serializer->ReadString("metadata_json", metadata);
        std::string output;
        serializer->ReadString("output_controller_json", output);
        CHECK(C::Json::parse(output) == outputController);
        const auto frame = C::Json::parse(metadata)["frame"];
        CHECK(frame["convention"] == "absolute-velocity-and-energy-in-mesh-axes");
        CHECK(frame["law"]["angularVelocity"][2] == .2);
        CHECK(frame["rotation_column_major"].size() == 9);
        serializer->CloseFile();
    }
    C::Domain restored("flow", mesh, config.Domains()[0].settings, config.Time());
    restored.ReadCheckpoint(path, config.Root());
    CHECK(restored.Time() == source.Time());
    REQUIRE(source.Advance(.002).accepted);
    REQUIRE(restored.Advance(.002).accepted);
    const auto a = source.Dofs().CoupledField().Owned().Read(), b = restored.Dofs().CoupledField().Owned().Read();
    const auto u = a.View<R::UniformTableRows>(), v = b.View<R::UniformTableRows>();
    for (DNDS::index iCell = 0; iCell < u.RowCount(); ++iCell)
        for (int j = 0; j < 5; ++j)
            CHECK(u(iCell, j) == v(iCell, j));
    Remove({path});
}
TEST_CASE("Exact internal checkpoints resume BDF DIRK and DITR nonlinear iterations")
{
    const auto metadata = C::Json{{"test", "exact internal nonlinear continuation"}};
    for (const std::string method : {"bdf2", "esdirk4", "ditr"})
    {
        CAPTURE(method);
        const auto path = Path("cauchy-internal-" + method + "-");
        C::DomainSettings settings;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 2;
        settings.numerics.reconstructionIterations = 3;
        settings.numerics.faceFallback = false;
        settings.implicit.iterations = settings.implicit.minIterations = 3;
        settings.implicit.relativeTolerance = 1;
        settings.implicit.absoluteTolerance = 0;
        C::TimeSettings time;
        time.method = method;
        const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
        C::Domain uninterrupted("flow", mesh, settings, time);
        Initialize(uninterrupted);
        bool written = false;
        uninterrupted.SetQuiescentObserver([&](const C::NonlinearSolveState &state)
                                           {
            if (!written && !state.terminal)
            {
                uninterrupted.WriteInternalCheckpoint(path, metadata, state);
                written = true;
            } });
        REQUIRE(uninterrupted.Advance(.002).accepted);
        REQUIRE(written);

        C::Domain resumed("flow", mesh, settings, time);
        resumed.ReadCheckpoint(path, metadata);
        REQUIRE(resumed.Advance(.002).accepted);
        const auto compare = [&](const C::Domain &a, const C::Domain &b)
        {
            const auto left = a.Dofs().ReadOwned(), right = b.Dofs().ReadOwned();
            for (DNDS::index iBlock = 0; iBlock < a.Dofs().BlockNumber(); ++iBlock)
                for (DNDS::index iCell = 0; iCell < left.Block(iBlock).owned.RowCount(); ++iCell)
                    for (DNDS::index j = 0; j < left.Block(iBlock).ComponentNumber(); ++j)
                        CHECK(left.Block(iBlock)(iCell, j) == right.Block(iBlock)(iCell, j));
        };
        compare(uninterrupted, resumed);
        REQUIRE(uninterrupted.Advance(.0015).accepted);
        REQUIRE(resumed.Advance(.0015).accepted);
        compare(uninterrupted, resumed);
        Remove({path});
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
