#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/LimitedVRPlan.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include "RotatedSectorFixture.hpp"
#include <limits>

namespace C = DNDS::CFV;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

static void Equal(const R::Array<double> &expected, const R::Array<double> &actual, double tolerance = 2e-10)
{
    const auto host = actual.To(R::Device::Host());
    const auto left = expected.Read(), right = host.Read();
    REQUIRE(expected.RowCount() == actual.RowCount());
    for (I iRow = 0; iRow < expected.RowCount(); ++iRow)
        for (I j = 0; j < expected.RowSize(iRow); ++j)
            CHECK(right.View()(iRow, j) == doctest::Approx(left.View()(iRow, j)).epsilon(tolerance).scale(1));
}

TEST_CASE("Queued LVR seeds and neighbor rounds retain CPU semantics across host workers and CUDA")
{
    auto mesh = DNDS::Geom::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    const auto plan = C::LimitedVRPlan::Prepare(metrics);
    auto scalar = metrics.BuildField(2), gate = metrics.BuildField(3), expected = metrics.BuildField(3);
    R::Array<double> history(R::RowLayout::Dense(metrics.NumCell(), 2));
    {
        const auto geometry = metrics.AcquireRead();
        const auto view = geometry.HostView();
        auto s = scalar.Owned().Write(), g = gate.Owned().Write(), h = history.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            // Transverse equal neighbors, alternating extrema and a retained
            // history produce nonzero correction seeds away from boundaries.
            const int stripe = int(std::floor(view.GetCellBary(iCell)[0] * 24));
            s.View()(iCell, 0) = stripe % 3 == 1 ? 1.02 : 1.;
            s.View()(iCell, 1) = stripe % 3 == 1 ? 1.04 : 1.;
            g.View()(iCell, 0) = stripe % 4 == 0 ? 1. : .3;
            g.View()(iCell, 1) = .7;
            g.View()(iCell, 2) = stripe % 4 == 2 ? 1. : .2;
            h.View()(iCell, 0) = .4;
            h.View()(iCell, 1) = .6;
        }
    }
    metrics.BindHalo(scalar).Begin().Wait();
    metrics.BindHalo(gate).Begin().Wait();
    for (const bool cuda : {false, true})
    {
        const auto device = cuda ? R::Device::CUDA(0) : R::Device::Host();
        const auto context = cuda ? R::ExecutionContext::CUDA(0) : R::ExecutionContext::Host();
        auto residentMetrics = cuda ? metrics.Placed(device, context) : metrics;
        const auto resident = plan.Placed(residentMetrics, context);
        auto ds = residentMetrics.BuildField(2), dg = residentMetrics.BuildField(3), result = residentMetrics.BuildField(3);
        ds.Owned().CopyFrom(scalar.Owned(), context).Wait();
        ds.Ghost().CopyFrom(scalar.Ghost(), context).Wait();
        dg.Owned().CopyFrom(gate.Owned(), context).Wait();
        dg.Ghost().CopyFrom(gate.Ghost(), context).Wait();
        auto dh = history.To(device, context);
        R::EagerEngine engine(2);
        const auto stream = cuda ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
        for (const std::string variant : {"v2", "v3"})
        {
            const auto settings = C::LimitedVRSettings::ForVariant(variant);
            plan.BuildExtremumSeeds(scalar, gate, history, expected, settings);
            {
                // Core must not acquire the separately allocated ghost buffer.
                auto ghost = ds.Ghost().Write(context);
                resident.BuildExtremumSeeds(engine, stream, ds, dg, dh, result, settings, C::ReconstructionRegion::Core);
            }
            resident.BuildExtremumSeeds(engine, stream, ds, dg, dh, result, settings, C::ReconstructionRegion::HaloDependent);
            engine.Synchronize();
            Equal(expected.Owned(), result.Owned());
        }
        for (const bool saturation : {false, true})
        {
            plan.NeighborMaximum(gate, expected, {true, false, true}, true, saturation, {1, 1, 1});
            resident.NeighborMaximum(engine, stream, dg, result, {true, false, true}, true, saturation, {1, 1, 1}, C::ReconstructionRegion::Core);
            resident.NeighborMaximum(engine, stream, dg, result, {true, false, true}, true, saturation, {1, 1, 1}, C::ReconstructionRegion::HaloDependent);
            engine.Synchronize();
            Equal(expected.Owned(), result.Owned());
        }
    }
}

TEST_CASE("Limited CUDA Jacobi matches CPU relaxation and excludes neighbors for a fixed target")
{
    const auto context = R::ExecutionContext::CUDA(0);
    auto mesh = CFVTest::Sector(mpi);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 3;
    settings.jacobiRelax = .73;
    const auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    const auto resident = plan.Place(context.device, context);
    auto mean = metrics.BuildField(5), input = plan.BuildCoefficients(5), target = plan.BuildCoefficients(5);
    auto reference = plan.BuildWorkspace(5, {1});
    auto bc = plan.BuildBoundaryValues(5);
    A::Fill(mean, 1.1);
    A::Fill(input, .02);
    A::Fill(target, -.03);
    {
        auto write = bc.values.Write();
        for (I iRow = 0; iRow < bc.values.RowCount(); ++iRow)
            for (I j = 0; j < 5; ++j)
                write.View()(iRow, j) = 1.2 + .1 * j;
    }
    metrics.BindHalo(mean).Begin().Wait();
    plan.BindHalo(input).Begin().Wait();
    R::Array<double> alpha(R::RowLayout::Dense(metrics.NumCell(), 1));
    {
        auto write = alpha.Write();
        for (I iCell = 0; iCell < alpha.RowCount(); ++iCell)
            write.View()(iCell, 0) = (iCell % 3) * .5;
    }
    auto dm = metrics.BuildField(R::RowLayout::Dense(0, 5), context.device), di = resident.BuildCoefficients(5), dt = resident.BuildCoefficients(5);
    auto work = resident.BuildWorkspace(5, {1});
    dm.Owned().CopyFrom(mean.Owned(), context).Wait();
    dm.Ghost().CopyFrom(mean.Ghost(), context).Wait();
    di.Owned().CopyFrom(input.Owned(), context).Wait();
    di.Ghost().CopyFrom(input.Ghost(), context).Wait();
    dt.Owned().CopyFrom(target.Owned(), context).Wait();
    auto da = alpha.To(context.device, context);
    auto db = resident.PlaceBoundaryValues(bc);
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto compare = [&]
    {
        plan.DoReconstructionIterLimitedTo(mean, input, reference, bc, target, alpha);
        for (const auto kernel : {C::ReconstructionDeviceKernel::FP64, C::ReconstructionDeviceKernel::FP64Pipelined})
        {
            {
                auto ghost = di.Ghost().Write(context);
                resident.SubmitReconstructionIterLimitedTo(engine, stream, dm, di, work, db, dt, da,
                                                           C::ReconstructionRegion::Core, kernel);
            }
            resident.SubmitReconstructionIterLimitedTo(engine, stream, dm, di, work, db, dt, da,
                                                       C::ReconstructionRegion::HaloDependent, kernel);
            engine.Synchronize();
            resident.CheckStatus(work);
            Equal(reference.temporary.Owned(), work.temporary.Owned());
        }
    };
    compare();
    // Both implementations must omit *all* neighbor/BC contributions for alpha
    // one, including NaN means. Current coefficients still have finite damping.
    A::Fill(mean, std::numeric_limits<double>::quiet_NaN());
    {
        auto write = alpha.Write();
        for (I iCell = 0; iCell < alpha.RowCount(); ++iCell)
            write.View()(iCell, 0) = 1;
    }
    dm.Owned().CopyFrom(mean.Owned(), context).Wait();
    da.CopyFrom(alpha, context).Wait();
    compare();
    if (alpha.RowCount())
        alpha.Write().At(0, 0) = -1;
    da.CopyFrom(alpha, context).Wait();
    resident.SubmitReconstructionIterLimitedTo(engine, stream, dm, di, work, db, dt, da);
    engine.Synchronize();
    if (alpha.RowCount())
        CHECK_THROWS(resident.CheckStatus(work));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi = DNDS::MPIInfo(MPI_COMM_WORLD);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
