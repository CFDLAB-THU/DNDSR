#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "Cauchy/Config/CUDAKernelDefaults.hpp"
#include <filesystem>
#include <unistd.h>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

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
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
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
