#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/CUDA/FiniteVolumeLimitedVRKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeLimitedVRKernels.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "CompactManagedStencilFixture.hpp"

namespace C = DNDS::Cauchy;
namespace K = C::CUDA;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using DNDS::real;
static DNDS::MPIInfo mpi;

namespace
{
    void Equal(const R::Array<real> &expected, const R::Array<real> &actual)
    {
        const auto downloaded = actual.To(R::Device::Host());
        const auto a = expected.Read(), b = downloaded.Read();
        REQUIRE(expected.RowCount() == downloaded.RowCount());
        for (I iRow = 0; iRow < expected.RowCount(); ++iRow)
            for (I j = 0; j < expected.RowSize(iRow); ++j)
                CHECK(b.View()(iRow, j) == doctest::Approx(a.View()(iRow, j)).epsilon(2e-12).scale(1));
    }
    void Good(const R::Array<DNDS::NumericalStatus> &array)
    {
        const auto host = array.To(R::Device::Host());
        const auto read = host.Read();
        for (I iRow = 0; iRow < array.RowCount(); ++iRow)
            CHECK(bool(read.View()(iRow, 0)));
    }

}

TEST_CASE("LVR center target guard and direct sensors preserve CPU CUDA branch gates")
{
    CauchyTest::CompactManagedStencilFixture fixture(CFVTest::CartesianHexes(DNDS::MPIInfo(MPI_COMM_SELF), 2));
    const C::PerfectGas gas;
    const C::ThermalBounds bounds{1e-5, 1e-5, 1e-5};
    const auto device = R::Device::CUDA(0);
    const R::ExecutionContext context{device, nullptr};
    const auto geometry = fixture.metrics.AcquireRead();
    const auto metric = geometry.HostView();
    auto means = fixture.metrics.BuildField(5), target = fixture.metrics.BuildField(5 * 19);
    const auto fill = [&](R::Array<real> &values, R::Array<real> &coefficients, I first)
    {
        auto u = values.Write(), c = coefficients.Write();
        for (I iRow = 0; iRow < values.RowCount(); ++iRow)
        {
            const auto center = metric.GetCellBary(first + iRow);
            const auto state = gas.Conservative({center[0] < .5 ? 1.8 : 1., center[0] < .5 ? .4 : -.2, .1, 0, center[0] < .5 ? 1.3 : .8});
            for (int j = 0; j < 5; ++j)
            {
                u.View()(iRow, j) = state[j];
                for (int mode = 0; mode < 19; ++mode)
                    c.View()(iRow, j * 19 + mode) = 0;
            }
            // An active thermal guard: the stored affine state would have
            // negative energy on one face without limiting.
            c.View()(iRow, 4 * 19) = 20;
        }
    };
    fill(means.Owned(), target.Owned(), 0);
    fill(means.Ghost(), target.Ghost(), fixture.metrics.NumCell());
    DNDS::CFV::FVField dm(means.Space(), means.Owned().To(device, context), means.Ghost().To(device, context));
    DNDS::CFV::FVField dt(target.Space(), target.Owned().To(device, context), target.Ghost().To(device, context));
    const auto deviceMetrics = fixture.metrics.Placed(device, context);
    const auto deviceStencil = fixture.stencil.Placed(deviceMetrics, context);
    const auto hostStencil = fixture.stencil.Read();
    const auto resident = std::make_shared<C::FiniteVolumeStencilStorage::ReadAccess>(deviceStencil.Read(context));
    R::Array<DNDS::NumericalStatus> targetStatus(R::RowLayout::Dense(std::min<I>(1024, (means.Owned().RowCount() + 127) / 128), 1), device);
    {
        const auto u = means.Owned().Read();
        auto c = target.Owned().Write();
        REQUIRE(bool(C::CPU::LimitLVRTarget(hostStencil.View(), u.View<R::UniformTableRows>(), c.View<R::UniformTableRows>(), 3, bounds, 1, 1e-3)));
    }
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    R::SubmitArrays(engine, stream, [resident, bounds](R::ExecutionContext c, auto u, auto target, auto status)
                    { K::LimitLVRTarget(c, resident->KernelDescriptor(), u, target, 3, bounds, 1, 1e-3, status); }, R::ReadAs<R::TableRows>(dm.Owned()), R::ReadWriteAs<R::TableRows>(dt.Owned()), R::WriteAs<R::FixedRows<1>>(targetStatus));
    engine.Synchronize();
    Good(targetStatus);
    Equal(target.Owned(), dt.Owned());
    {
        const auto u = means.Owned().Read(), coefficients = target.Owned().Read();
        for (I iCell = 0; iCell < means.Owned().RowCount(); ++iCell)
            for (const I iFace : hostStencil.View().topology.CellFaces(iCell))
            {
                // This nonperiodic fixture's center is in the cell frame.
                const auto value = C::EvaluateLVRLinearTarget(hostStencil.View(), u.View<R::UniformTableRows>(),
                                                              coefficients.View<R::UniformTableRows>(), iCell, metric.GetFaceCent(iFace), 3, 19);
                CHECK(C::ThermallyAdmissible(value, bounds));
            }
    }
    const I faces = fixture.stencil.stencilFaces.RowCount(), cells = fixture.metrics.NumCell();
    R::Array<real> sensors(R::RowLayout::Dense(faces, 3)), branches(R::RowLayout::Dense(cells, 3)), base(R::RowLayout::Dense(cells, 2));
    R::Array<real> ds(sensors.Layout(), device), db(branches.Layout(), device), dbase(base.Layout(), device);
    R::Array<DNDS::NumericalStatus> faceStatus(R::RowLayout::Dense(faces, 1), device), cellStatus(R::RowLayout::Dense(cells, 1), device);
    for (const std::string variant : {"v2", "v3"})
        for (const bool expansion : {false, true})
        {
            auto settings = DNDS::CFV::LimitedVRSettings::ForVariant(variant);
            settings.detectExpansion = expansion;
            {
                const auto u = means.Owned().Read(), g = means.Ghost().Read(), a = target.Owned().Read(), b = target.Ghost().Read();
                auto s = sensors.Write();
                REQUIRE(bool(C::CPU::BuildLVRFaceSensors(hostStencil.View(), {u.View<R::UniformTableRows>(), g.View<R::UniformTableRows>()},
                                                         {a.View<R::UniformTableRows>(), b.View<R::UniformTableRows>()}, 3, gas, bounds, settings, s.View<R::UniformTableRows>())));
            }
            {
                const auto s = sensors.Read();
                auto b = branches.Write(), v = base.Write();
                C::CPU::GatherLVRBranches(hostStencil.View(), s.View<R::UniformTableRows>(), b.View<R::UniformTableRows>(), v.View<R::FixedRows<2>>());
            }
            const auto parameters = DNDS::CFV::LimitedVRKernelParameters::FromSettings(settings);
            R::SubmitArrays(engine, stream, [resident, gas, bounds, parameters](R::ExecutionContext c, auto u, auto g, auto a, auto b, auto s, auto status)
                            { K::BuildLVRFaceSensors(c, resident->KernelDescriptor(), {u, g}, {a, b}, 3, gas, bounds, parameters, s, status); }, R::ReadAs<R::TableRows>(dm.Owned()), R::ReadAs<R::TableRows>(dm.Ghost()), R::ReadAs<R::TableRows>(dt.Owned()), R::ReadAs<R::TableRows>(dt.Ghost()), R::WriteAs<R::TableRows>(ds), R::WriteAs<R::FixedRows<1>>(faceStatus));
            R::SubmitArrays(engine, stream, [resident](R::ExecutionContext c, auto s, auto fs, auto b, auto v, auto status)
                            { K::GatherLVRBranches(c, resident->KernelDescriptor(), s, fs, b, v, status); }, R::ReadAs<R::TableRows>(ds), R::ReadAs<R::FixedRows<1>>(faceStatus), R::WriteAs<R::TableRows>(db), R::WriteAs<R::FixedRows<2>>(dbase), R::WriteAs<R::FixedRows<1>>(cellStatus));
            engine.Synchronize();
            Good(faceStatus);
            Good(cellStatus);
            Equal(sensors, ds);
            Equal(branches, db);
            Equal(base, dbase);
        }
    engine.Close();
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
