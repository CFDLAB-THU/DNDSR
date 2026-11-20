#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/IO/AsyncOutput.hpp"
#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "../CFV/RotatedSectorFixture.hpp"
#include "Cauchy/Config/CUDAKernelDefaults.hpp"
#include <filesystem>
#include <hdf5.h>
#include <unistd.h>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("ACM device correction matches CPU on flow4 and mechanical SA and SST")
{
    for (const std::string profile : {"none", "SA", "KOmegaSST", "SST-eulerScalar", "flow4-pMG"})
    {
        CAPTURE(profile);
        const bool multigrid = profile == "flow4-pMG";
        const bool eulerScalar = profile == "SST-eulerScalar";
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.equation.system = "incompressibleACM";
        settings.equation.eos = "none";
        settings.equation.acmPressureMean = 0;
        settings.equation.transport = "Newtonian";
        settings.equation.viscosity = .001;
        settings.equation.rans.model = multigrid ? "none" : eulerScalar ? "KOmegaSST"
                                                                        : profile;
        settings.equation.rans.viscosityReference = .001;
        settings.equation.referenceScales = {{"rho0", 1}, {"U0", 1}, {"L0", 1}, {"T0", 300}};
        std::vector<double> initial{.2, -.03, .04, -.1};
        if (settings.equation.rans.model != "none")
            initial.push_back(.03);
        if (settings.equation.rans.model == "KOmegaSST")
            initial.push_back(2.);
        settings.initialState = {{"type", "primUP"}, {"state", initial}};
        settings.referenceState = settings.initialState;
        settings.boundaries["7"].kind = "noSlipWall";
        if (settings.equation.rans.model == "KOmegaSST")
            settings.motion.angularVelocity = {0, 0, .19};
        settings.numerics.reconstruction = "constant";
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.faceFallback = false;
        settings.numerics.ransMeanFaceValues = eulerScalar;
        settings.numerics.ransP1Overlay = eulerScalar;
        if (eulerScalar)
        {
            settings.numerics.limiter = "Barth";
            settings.numerics.viscousLimited = false;
        }
        settings.numerics.ransCellCenterViscosity = eulerScalar;
        settings.implicit.residualComponents.clear();
        settings.implicit.diagonal = "block";
        settings.implicit.jacobianAction = "assembled";
        settings.implicit.boundaryJacobian = eulerScalar ? "eulerScalar" : "physical";
        settings.implicit.linear.method = "jacobi";
        settings.implicit.linear.stopping = "fixedCount";
        settings.implicit.linear.iterations = 3;
        settings.implicit.iterations = settings.implicit.minIterations = 2;
        const bool polynomial = multigrid || settings.equation.rans.model != "none";
        if (polynomial)
        {
            settings.numerics.reconstruction = "variational";
            settings.numerics.cfv.maxOrder = multigrid ? 2 : profile == "SA" ? 1
                                                                             : 3;
            settings.numerics.cfv.SORInstead = false;
            settings.numerics.reconstructionIterations = 4;
            settings.numerics.reconstructionTolerance = 0;
            settings.implicit.requireConvergedReconstruction = false;
            if (multigrid)
            {
                settings.implicit.multigrid.levels = 1;
                settings.implicit.multigrid.coarseSmooth = 1;
            }
        }
        C::TimeSettings time;
        time.method = "bdf1";
        auto mesh = CFVTest::CartesianHexes(mpi, 3);
        std::optional<R::Array<double>> reference;
        std::vector<DNDS::real> referenceBounds;
        for (const auto *mapping : {"cpu", "thread", "ctaSegments"})
        {
            CAPTURE(std::string_view(mapping));
            const bool cpu = std::string_view(mapping) == "cpu";
            // Compare the regional device update against a whole-domain CPU correction.
            settings.execution.overlap = !cpu;
            settings.execution.device = cpu ? "cpu" : "cuda";
            settings.execution.kernels.rhsMapping = cpu ? "auto" : mapping;
            settings.execution.reconstructionKernel = !cpu && polynomial ? "fp64Pipelined" : "fp64";
            C::Domain domain("acm", mesh, settings, time);
            CHECK(domain.CaptureStateSnapshot(false).PressureGauge() == std::optional<DNDS::real>(0));
            REQUIRE(domain.Advance(.0001).accepted);
            REQUIRE_FALSE(domain.LastImplicitStages().empty());
            if (multigrid)
                CHECK(domain.LastImplicitStages().front().multigridCycles > 0);
            auto values = domain.Dofs().CoupledField().Owned().To(R::Device::Host());
            const auto bounds = domain.MeasureStateBounds(false);
            REQUIRE(bounds.size() == 2 * initial.size());
            if (!reference)
            {
                reference = values.Clone();
                referenceBounds = bounds;
                continue;
            }
            for (I j = 0; j < I(bounds.size()); ++j)
                CHECK(bounds[j] == doctest::Approx(referenceBounds[j]).epsilon(1e-8).scale(1));
            auto expected = reference->Read(), actual = values.Read();
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                for (I j = 0; j < I(initial.size()); ++j)
                    CHECK(actual.View()(iCell, j) == doctest::Approx(expected.View()(iCell, j)).epsilon(1e-8).scale(1));
        }
    }
}

TEST_CASE("ACM CUDA exports zero tangential slip-wall flux and nonzero no-slip shear")
{
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.equation.system = "incompressibleACM";
    settings.equation.eos = "none";
    settings.equation.transport = "Newtonian";
    settings.equation.viscosity = .01;
    settings.equation.referenceScales = {{"rho0", 1}, {"U0", 1}, {"L0", 1}, {"T0", 300}};
    settings.equation.rans.model = "KOmegaSST";
    settings.boundaries["7"].kind = "noSlipWall";
    settings.boundaries["8"].kind = "slipWall";
    settings.initialState = {{"type", "primUP"}, {"state", {.25, 0, 0, .7, .03, 2}}};
    settings.referenceState = settings.initialState;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 1;
    settings.numerics.cfv.SORInstead = false;
    settings.numerics.viscousGradient = "polynomial";
    settings.execution.device = "cuda";
    settings.implicit.diagonal = "block";
    settings.implicit.jacobianAction = "assembled";
    settings.implicit.linear.method = "jacobi";
    settings.implicit.linear.stopping = "fixedCount";
    settings.boundaryFacialFluxes = true;
    C::TimeSettings time;
    time.method = "bdf1";
    C::Domain domain("wall", CFVTest::CartesianHexes(mpi, 3, true), settings, time);
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto prefix = (std::filesystem::temp_directory_path() / ("cauchy-cuda-wall-" + std::to_string(pid))).string();
    domain.WriteVisualization(prefix, 0);
    const auto file = H5Fopen((prefix + "_bnd.vtkhdf").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    std::array<int, 54> zones{};
    std::array<double, 3 * 54> tangents{};
    const auto readField = [&](const char *name, hid_t datatype, void *values, size_t count)
    {
        const auto data = H5Dopen(file, name, H5P_DEFAULT);
        REQUIRE(data >= 0);
        const auto space = H5Dget_space(data);
        REQUIRE(H5Sget_simple_extent_npoints(space) == DNDS::index(count));
        REQUIRE(H5Dread(data, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) >= 0);
        H5Sclose(space);
        H5Dclose(data);
    };
    readField("VTKHDF/CellData/bnd_zone_id", H5T_NATIVE_INT, zones.data(), zones.size());
    readField("VTKHDF/CellData/force_tangential", H5T_NATIVE_DOUBLE, tangents.data(), tangents.size());
    H5Fclose(file);
    int slip = 0, noSlip = 0;
    for (size_t iFace = 0; iFace < zones.size(); ++iFace)
        if (zones[iFace] == 8)
        {
            ++slip;
            CHECK(std::abs(tangents[3 * iFace]) < 1e-13);
        }
        else if (std::abs(tangents[3 * iFace]) > 1e-8)
            ++noSlip;
    CHECK(slip == 9);
    CHECK(noSlip > 0);
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
    {
        CHECK(std::filesystem::remove(prefix + ".vtkhdf"));
        CHECK(std::filesystem::remove(prefix + "_bnd.vtkhdf"));
    }
}

TEST_CASE("ACM polynomial CUDA face mappings match the host on a rotational seam")
{
    C::DomainSettings settings;
    settings.motion.angularVelocity = {0, 0, .19};
    settings.equation.system = "incompressibleACM";
    settings.equation.eos = "none";
    settings.equation.referenceScales = {{"rho0", 1}, {"U0", 1}, {"L0", 1}, {"T0", 300}};
    settings.initialState = {{"type", "primUP"}, {"state", {.2, -.03, .04, -.1}}};
    settings.boundaries["12"].kind = settings.boundaries["13"].kind = "slipWall";
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 2;
    settings.numerics.cfv.intOrder = 3;
    settings.numerics.cfv.cacheDiffBase = false;
    settings.numerics.cfv.SORInstead = false;
    settings.numerics.viscousGradient = "polynomial";
    settings.implicit.jacobianAction = "assembled";
    settings.implicit.diagonal = "block";
    settings.implicit.linear.method = "jacobi";
    settings.implicit.linear.stopping = "fixedCount";
    settings.implicit.linear.iterations = 2;
    settings.implicit.iterations = settings.implicit.minIterations = 2;
    settings.implicit.rejectUnconvergedStages = false;
    const auto mesh = CFVTest::Sector(mpi);
    C::TimeSettings time;
    time.method = "bdf1";
    std::optional<R::Array<double>> reference;
    for (const std::string mapping : {"cpu", "thread", "ctaSegments"})
    {
        CAPTURE(mapping);
        settings.execution.device = mapping == "cpu" ? "cpu" : "cuda";
        settings.execution.kernels.rhsMapping = mapping == "cpu" ? "auto" : mapping;
        C::Domain domain("acm-sector", mesh, settings, time);
        REQUIRE(domain.Advance(.0001).accepted);
        const auto values = domain.Dofs().CoupledField().Owned().To(R::Device::Host());
        if (!reference)
            reference = values.Clone();
        const auto actualRead = values.Read(), expectedRead = reference->Read();
        const auto actual = actualRead.View<R::FixedRows<4>>(), expected = expectedRead.View<R::FixedRows<4>>();
        for (I iCell = 0; iCell < actual.RowCount(); ++iCell)
            for (I j = 0; j < 4; ++j)
                CHECK(actual(iCell, j) == doctest::Approx(expected(iCell, j)).epsilon(1e-8).scale(1));
    }
}

TEST_CASE("Automatic A100 kernels advance raw P3 flow with frozen explicit reference selectors")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    const auto metrics = DNDS::CFV::FiniteVolumePlan::ConstructMetrics(mesh);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.execution.device = "cuda";
    settings.execution.overlap = false;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.intOrder = 5;
    settings.numerics.cfv.cacheDiffBase = false;
    settings.numerics.cfv.storage.representation = "raw";
    settings.numerics.cfv.storage.faceCoupling = true;
    settings.numerics.cfv.SORInstead = false;
    settings.numerics.reconstructionIterations = 1;
    settings.numerics.reconstructionTolerance = 0;
    settings.numerics.positivity = true;
    settings.numerics.faceFallback = false;
    settings.implicit.iterations = settings.implicit.minIterations = 2;
    settings.implicit.requireConvergedReconstruction = false;
    settings.implicit.rejectUnconvergedStages = false;
    settings.implicit.linear.method = "jacobi";
    settings.implicit.linear.stopping = "fixedCount";
    settings.implicit.linear.iterations = 8;
    settings.boundaries["7"].kind = "extrapolate";
    const auto gas = C::EquationSystem::Resolve(settings.equation).Gas();
    R::Array<double> initial(R::RowLayout::Dense(metrics.NumCell(), 5));
    {
        auto write = initial.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto x = metrics.GetCellCent(iCell);
            const auto value = gas.Conservative({1 + .01 * x[0], .3 + .01 * x[1], .2, .1, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    C::TimeSettings time;
    time.method = "bdf1";
    std::optional<R::Array<double>> reference;
    for (bool automatic : {false, true})
    {
        auto selected = settings;
        if (!automatic)
        {
            selected.execution.reconstructionKernel = "fp64";
            selected.execution.kernels.rhsMapping = "ctaSegments";
            selected.execution.kernels.rhsReduction = "serial";
            selected.execution.kernels.polynomialPPThreads = 128;
            selected.implicit.linear.cudaDiagonalInverse = "batchedSVD";
        }
        C::Domain domain("fluid", mesh, selected, time);
        const auto &resolved = domain.Discretization().Settings().execution;
        if (automatic && C::IsA100Device(R::Device::CUDA(0)))
        {
            CHECK(resolved.reconstructionKernel == "fp64Pipelined");
            CHECK(resolved.kernels.rhsMapping == "auto");
            CHECK(resolved.kernels.rhsReduction == "auto");
            CHECK(resolved.kernels.polynomialPPThreads == 512);
        }
        domain.Dofs().CoupledField().Owned().CopyFrom(initial, {R::Device::CUDA(0), nullptr}).Wait();
        REQUIRE(domain.Advance(1e-4).accepted);
        const auto final = domain.Dofs().CoupledField().Owned().To(R::Device::Host());
        if (!automatic)
            reference = final.Clone();
        const auto actual = final.Read(), expected = reference->Read();
        for (I iCell = 0; iCell < final.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
            {
                CHECK(std::isfinite(actual.View()(iCell, j)));
                CHECK(actual.View()(iCell, j) == doctest::Approx(expected.View()(iCell, j)).epsilon(2e-8).scale(1));
            }
    }
}

TEST_CASE("Device domains advance nonuniform implicit flow through single grid and pMG with restart")
{
    // Unequal cell widths also exercise geometric pseudo-time smoothing.
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6, 1, 1. / 12);
    const auto metrics = DNDS::CFV::FiniteVolumePlan::ConstructMetrics(mesh);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.SORInstead = false;
    settings.numerics.reconstructionIterations = 12;
    settings.numerics.reconstructionTolerance = 0;
    settings.numerics.faceFallback = false;
    settings.numerics.limiter = "none";
    settings.implicit.iterations = 80;
    settings.implicit.minIterations = 1;
    settings.implicit.relativeTolerance = 1e-8;
    settings.implicit.absoluteTolerance = 1e-11;
    settings.implicit.linear.method = "gmres";
    settings.implicit.linear.preconditioner = "jacobi";
    settings.implicit.linear.relativeTolerance = 1e-3;
    settings.implicit.diagonal = "block";
    settings.implicit.multigrid.cycles = 40;
    settings.implicit.multigrid.preSmooth = 1;
    settings.implicit.multigrid.postSmooth = 1;
    settings.implicit.multigrid.coarseSmooth = 3;
    const auto gas = C::EquationSystem::Resolve(settings.equation).Gas();
    R::Array<double> initial(R::RowLayout::Dense(metrics.NumCell(), 5));
    {
        auto write = initial.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto point = metrics.GetCellCent(iCell);
            const auto u = gas.Conservative({1 + .02 * std::sin(2 * DNDS::pi * (point[0] + point[1])), .3, .2, .1, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = u[j];
        }
    }
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto checkpoint = (std::filesystem::temp_directory_path() / ("cauchy-device-domain-" + std::to_string(pid) + ".h5")).string();
    for (int profile = 0; profile < 2; ++profile)
    {
        CAPTURE(profile);
        C::TimeSettings time;
        time.method = profile ? "esdirk4" : "bdf2";
        settings.implicit.multigrid.levels = profile ? 2 : 0;
        settings.implicit.linear.method = profile ? "jacobi" : "gmres";
        std::optional<R::Array<double>> reference;
        for (int backend = 0; backend < 3; ++backend)
        {
            CAPTURE(backend);
            settings.execution.device = backend == 2 ? "cuda" : "cpu";
            settings.execution.deviceOrdinal = 0; // np2 launcher exposes one GPU per rank.
            settings.execution.hostThreads = backend == 1 ? 1 : 0;
            C::Domain domain("fluid", mesh, settings, time);
            const auto device = domain.Dofs().Device();
            REQUIRE(device == (backend == 2 ? R::Device::CUDA(0) : R::Device::Host()));
            domain.Dofs().CoupledField().Owned().CopyFrom(initial, {device, nullptr}).Wait();
            for (int step = 0; step < 2; ++step)
            {
                REQUIRE(domain.Advance(1e-4).accepted);
                REQUIRE_FALSE(domain.LastImplicitStages().empty());
                for (const auto &stage : domain.LastImplicitStages())
                    CHECK(stage.converged);
            }
            const auto final = domain.Dofs().CoupledField().Owned().To(R::Device::Host());
            if (!backend)
                reference = final.Clone();
            const auto actual = final.Read(), expected = reference->Read();
            for (I iCell = 0; iCell < final.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(actual.View()(iCell, j) == doctest::Approx(expected.View()(iCell, j)).epsilon(2e-7).scale(1));
            if (backend == 2 && profile == 0)
            {
                // Actual domain archive: preserve BDF history and VR warm
                // state, then compare the next unequal physical step.
                domain.WriteCheckpoint(checkpoint, C::Json::object(), true);
                C::Domain restored("fluid", mesh, settings, time);
                restored.ReadCheckpoint(checkpoint, C::Json::object());
                CHECK(restored.Time() == domain.Time());
                REQUIRE(domain.Advance(1.5e-4).accepted);
                REQUIRE(restored.Advance(1.5e-4).accepted);
                const auto a = domain.Dofs().CoupledField().Owned().To(R::Device::Host());
                const auto b = restored.Dofs().CoupledField().Owned().To(R::Device::Host());
                const auto ar = a.Read(), br = b.Read();
                for (I iCell = 0; iCell < a.RowCount(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        CHECK(ar.View()(iCell, j) == doctest::Approx(br.View()(iCell, j)).epsilon(2e-12).scale(1));
                MPI_Barrier(mpi.comm);
                if (mpi.rank == 0)
                    CHECK(std::filesystem::remove(checkpoint));
                MPI_Barrier(mpi.comm);
            }
        }
    }
}

TEST_CASE("Domain reconstruction selection runs quantized maps and shared or direct quadrature with FP64 evolving fields")
{
    // Quantized comparison selectors deliberately retain only C=5/M=19.
    // Keep the evolving-field/cache oracle on that supported 3D P3 fixture;
    // the preceding domain matrix retains translational-periodic 2D coverage.
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    const auto metrics = DNDS::CFV::FiniteVolumePlan::ConstructMetrics(mesh);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.execution.device = "cuda";
    settings.execution.deviceOrdinal = 0;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.SORInstead = false;
    settings.numerics.reconstructionIterations = 12;
    settings.numerics.reconstructionTolerance = 0;
    settings.numerics.limiter = "none";
    settings.numerics.faceFallback = false;
    settings.numerics.positivity = true;
    settings.implicit.linear.method = "jacobi";
    settings.boundaries["7"].kind = "extrapolate";
    C::TimeSettings time;
    time.method = "ssprk3";
    const auto gas = C::EquationSystem::Resolve(settings.equation).Gas();
    R::Array<double> initial(R::RowLayout::Dense(metrics.NumCell(), 5));
    {
        auto write = initial.Write();
        for (I iCell = 0; iCell < initial.RowCount(); ++iCell)
        {
            const auto x = metrics.GetCellCent(iCell);
            const auto value = gas.Conservative({1 + .02 * std::sin(2 * DNDS::pi * (x[0] + x[1])), .3, .2, .1, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    std::optional<R::Array<double>> reference;
    for (const std::string kernel : {"fp64", "fp32", "fp64Direct"})
    {
        CAPTURE(kernel);
        settings.execution.reconstructionKernel = kernel == "fp64Direct" ? "fp64" : kernel;
        settings.numerics.cfv.cacheDiffBase = kernel != "fp64Direct";
        C::Domain domain("fluid", mesh, settings, time);
        domain.Dofs().CoupledField().Owned().CopyFrom(initial, {R::Device::CUDA(0), nullptr}).Wait();
        for (int step = 0; step < 3; ++step)
            REQUIRE(domain.Advance(1e-3).accepted);
        const auto result = domain.Dofs().CoupledField().Owned().To(R::Device::Host());
        if (!reference)
            reference = result.Clone();
        const auto actual = result.Read(), expected = reference->Read(), seed = initial.Read();
        double change = 0;
        for (I iCell = 0; iCell < result.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
            {
                CHECK(actual.View()(iCell, j) == doctest::Approx(expected.View()(iCell, j)).epsilon(2e-6).scale(1));
                change = std::max(change, std::abs(actual.View()(iCell, j) - seed.View()(iCell, j)));
            }
        // Nonconstant data and an actual change prevent a vacuous constant
        // state test from passing when sweep selection is disconnected.
        CHECK(change > 1e-6);
    }
}

TEST_CASE("Asynchronous CUDA checkpoint captures the accepted epoch before a new device write")
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Query_thread(&provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        MESSAGE("asynchronous checkpoint qualification requires MPI_THREAD_MULTIPLE");
        return;
    }
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto prefix = (std::filesystem::temp_directory_path() / ("cauchy-device-writer-" + std::to_string(pid))).string();
    const C::Json input{{"defaults", {{"execution", {{"device", "cuda"}}}, {"implicit", {{"iterations", 4}, {"minIterations", 2}, {"linear", {{"method", "jacobi"}}}}}}},
                        {"time", {{"method", "bdf1"}}},
                        {"output", {{"prefix", prefix}, {"asynchronous", true}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    const auto factory = [](I, const auto &, const auto &domainMPI)
    { return CauchyTest::PeriodicQuads(domainMPI, 4); };
    C::Simulation simulation(config, mpi.comm, factory);
    const C::OutputPlan plan(config.Output(), 1);
    const std::string token = "-S0";
    auto &field = simulation.GetDomain(0).Dofs().CoupledField().Owned();
    const auto placement = R::ExecutionContext::CUDA(0);
    auto expected = field.To(R::Device::Host(), placement);
    {
        auto seed = expected.Write();
        for (I i = 0; i < expected.RowCount(); ++i)
            seed.At(i, 0) += .01 * std::sin(double(i + 1 + mpi.rank));
    }
    field.CopyFrom(expected, placement).Wait();
    auto changed = expected.Clone();
    {
        auto write = changed.Write();
        for (I i = 0; i < changed.RowCount(); ++i)
            write.At(i, 0) += 1;
    }
    {
        C::AsyncOutput output(simulation);
        output.Submit(simulation, config, plan, token, true, false, {}, C::Json{{"step", 0}});
        field.CopyFrom(changed, placement).Wait();
        // Also exercise worker drain/join during destruction.
    }
    simulation.ReadCheckpoint(plan.CheckpointPath(token));
    const auto observed = simulation.GetDomain(0).Dofs().CoupledField().Owned().To(R::Device::Host(), placement);
    const auto read = observed.Read(), original = expected.Read();
    REQUIRE(observed.RowCount() == expected.RowCount());
    for (I i = 0; i < observed.RowCount(); ++i)
        CHECK(read.At(i, 0) == original.At(i, 0));
    // Freeze the first internal iteration while the foreground finishes the step.
    const std::string internalToken = "-S1_0_0";
    bool captured = false;
    {
        C::AsyncOutput output(simulation);
        simulation.GetDomain(0).SetQuiescentObserver([&](const C::NonlinearSolveState &state)
                                                     {
            if (captured) return;
            output.SubmitInternal(simulation, config, plan, 0, internalToken, state, true, false, {}, {});
            captured = true; });
        REQUIRE(simulation.Advance(.001).accepted);
        simulation.GetDomain(0).SetQuiescentObserver({});
    }
    REQUIRE(captured);
    C::Simulation resumed(config, mpi.comm, factory);
    resumed.ReadCheckpoint(plan.InternalCheckpointPath(0, internalToken));
    REQUIRE(resumed.Advance(.001).accepted);
    const auto reference = simulation.GetDomain(0).Dofs().CoupledField().Owned().To(R::Device::Host(), placement);
    const auto replay = resumed.GetDomain(0).Dofs().CoupledField().Owned().To(R::Device::Host(), placement);
    const auto a = reference.Read(), b = replay.Read();
    double change = 0;
    for (I i = 0; i < replay.RowCount(); ++i)
        change = std::max(change, std::abs(a.At(i, 0) - original.At(i, 0)));
    CHECK(DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Max>(change, mpi.comm) > 1e-12);
    for (I i = 0; i < replay.RowCount(); ++i)
        for (I j = 0; j < replay.Layout().Width(); ++j)
            CHECK(a.At(i, j) == b.At(i, j));
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
    {
        CHECK(std::filesystem::remove(plan.InternalCheckpointPath(0, internalToken)));
        CHECK(std::filesystem::remove(plan.InternalSnapshotManifestPath(0, internalToken)));
        CHECK(std::filesystem::remove(plan.CheckpointPath(token)));
        CHECK(std::filesystem::remove(plan.SnapshotManifestPath(token)));
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    mpi = DNDS::MPIInfo(MPI_COMM_WORLD);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
