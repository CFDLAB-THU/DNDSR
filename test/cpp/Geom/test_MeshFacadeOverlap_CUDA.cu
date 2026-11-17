#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "MeshFacadeOverlapKernel.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

namespace T = DNDS::Geom::OverlapTest;
namespace R = DNDS::Runtime;
namespace G = DNDS::Geom;

static int OverlapGPU(DNDS::MPIInfo mpi)
{
    int devices = 0;
    REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
    const char *visible = std::getenv("CUDA_VISIBLE_DEVICES");
    const bool isolated = devices == 1 && visible && *visible && std::string(visible).find(',') == std::string::npos;
    REQUIRE((isolated || devices >= mpi.size));
    const auto ordinal = isolated ? 0 : mpi.rank;
    REQUIRE(cudaSetDevice(ordinal) == cudaSuccess);
    REQUIRE(DNDS::MPI::isCudaAware());
    return ordinal;
}

static void CheckGPUOverlap(T::Fixture &fixture, int ordinal)
{
    auto &mesh = *fixture.mesh;
    auto hostAccess = mesh.AcquireRead(G::MeshReadRequirements::CellFlux());
    const auto partition = T::Classify(mesh, hostAccess.HostView());
    auto hostCore = T::RowList(partition.core), hostRing = T::RowList(partition.ring);
    auto hostOwned = T::OwnedField(fixture);
    const auto device = R::Device::CUDA(ordinal);
    R::EagerEngine engine;
    auto core = engine.CreateStream(std::make_shared<R::CUDALauncher>(ordinal));
    auto ring = engine.CreateStream(std::make_shared<R::CUDALauncher>(ordinal));
    auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(ordinal));
    mesh.PlaceNative(device, core.Context());
    auto access = mesh.AcquireRead(G::MeshReadRequirements::CellFlux(), core.Context());
    const auto view = access.DeviceView();
    auto coreRows = hostCore.To(device, core.Context()), ringRows = hostRing.To(device, ring.Context());
    auto owned = hostOwned.To(device, comm.Context());
    R::Array<DNDS::real> ghost(R::RowLayout::Dense(size_t(mesh.NumCellGhost()), 1), device);
    R::Array<DNDS::real> coreResult(R::RowLayout::Dense(partition.core.size(), 1), device);
    R::Array<DNDS::real> ringResult(R::RowLayout::Dense(partition.ring.size(), 1), device);
    REQUIRE(T::PreloadOverlapKernels() == cudaSuccess);
    const auto entities = mesh.Entities(G::EntityKind::Cell);
    auto route = entities.BuildRowTransferPlan();
    auto binding = R::PayloadPlan(route, owned.Layout(), ghost.Layout(), R::ElementSchema::Arithmetic<DNDS::real>())
                       .Bind(owned, ghost, R::TransferTransport::Direct);
    auto operation = binding.BeginScheduled(comm.Context());
    CHECK_THROWS(ghost.Read(comm.Context()));
    R::SubmitArrays(engine, core, [view](R::ExecutionContext context, auto rows, auto u, auto output)
                    { DNDS_check_throw_info(T::LaunchCore(view, rows, u, output, ::DNDS::Runtime::CUDAStream(context)) == cudaSuccess,
                                            "real mesh owned-only CUDA core kernel failed"); }, R::Read(coreRows), R::Read(owned), R::Write(coreResult));
    const auto coreDone = core.Record();
    // Deterministic completion-observation gate: do not acknowledge the actual
    // MPI operation until the independent core CUDA prefix has completed.
    // Thus the receive lease remains active throughout device core execution,
    // even when the network eagerly finishes this small mesh payload.
    coreDone.WaitHost();
    CHECK(coreDone.Ready());
    CHECK_THROWS(ghost.Read(comm.Context()));
    CHECK_FALSE(ghost.Storage().IsIdle());
    R::MPIProgress progress;
    const auto received = progress.Import(std::move(operation));
    progress.Drain();
    CHECK(received.Ready());
    ring.WaitFor(received); // Portable MPI must be observed before a CUDA wait.
    ring.WaitFor(coreDone);
    R::SubmitArrays(engine, ring, [view](R::ExecutionContext context, auto rows, auto u, auto halo, auto output)
                    { DNDS_check_throw_info(T::LaunchRing(view, rows, u, halo, output, ::DNDS::Runtime::CUDAStream(context)) == cudaSuccess,
                                            "real mesh CUDA ring kernel failed"); }, R::Read(ringRows), R::Read(owned), R::Read(ghost), R::Write(ringResult));
    const auto ringDone = ring.Record();
    core.WaitFor(ringDone);
    access.Complete(R::Completion::Record(core.Context())); // Joins both mesh users.
    engine.Synchronize();
    access.Wait();
    partition.stamp.Validate();
    auto coreHost = coreResult.To(R::Device::Host()), ringHost = ringResult.To(R::Device::Host());
    const auto coreValues = coreHost.Read(), ringValues = ringHost.Read();
    for (size_t i = 0; i < partition.core.size(); ++i)
        CHECK(coreValues.View()(i, 0) == doctest::Approx(T::Expected(fixture, partition.core[i])).epsilon(1e-12));
    for (size_t i = 0; i < partition.ring.size(); ++i)
        CHECK(ringValues.View()(i, 0) == doctest::Approx(T::Expected(fixture, partition.ring[i])).epsilon(1e-12));
    DNDS::index counts[]{DNDS::index(partition.core.size()), DNDS::index(partition.ring.size())};
    MPI_Allreduce(MPI_IN_PLACE, counts, 2, DNDS::DNDS_MPI_INDEX, MPI_SUM, mesh.mpi.comm);
    CHECK(counts[0] > 0);
    DNDS::index owners = mesh.NumCell() > 0;
    MPI_Allreduce(MPI_IN_PLACE, &owners, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mesh.mpi.comm);
    if (owners > 1)
        CHECK(counts[1] > 0);
    CHECK(counts[0] + counts[1] == DNDS::index(fixture.expected.size()));
    if (mesh.mpi.rank == 0)
        std::cout << "real mesh CUDA direct MPI core=" << counts[0] << " ring=" << counts[1]
                  << "; device core completed within active receive-lease interval; no network speedup claim\n";
}

TEST_CASE("Real mesh CUDA finite-volume core uses owned solution during direct multi-GPU MPI halo")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size >= 2);
    const int ordinal = OverlapGPU(mpi);
    bool emptyPeers = false;
    SUBCASE("balanced native ownership") {}
    SUBCASE("native empty owners")
    {
        emptyPeers = true;
    }
    auto fixture = T::BuildNativeFixture(mpi, emptyPeers);
    DNDS::index empty = fixture.mesh->NumCell() == 0;
    MPI_Allreduce(MPI_IN_PLACE, &empty, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
    if (emptyPeers && mpi.size > 1)
        CHECK(empty > 0);
    CheckGPUOverlap(fixture, ordinal);
    const auto old = fixture.mesh->Entities(G::EntityKind::Cell);
    auto oldRoute = old.BuildRowTransferPlan();
    fixture.mesh->ReorderLocalCells(2, 1);
    CHECK_FALSE(old.Valid());
    CHECK_FALSE(oldRoute.Valid());
    CheckGPUOverlap(fixture, ordinal);
    T::RedistributeCurrent(fixture);
    CheckGPUOverlap(fixture, ordinal);
    T::RestartCurrent(fixture);
    CheckGPUOverlap(fixture, ordinal);
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    DNDS_check_throw_info(provided >= MPI_THREAD_FUNNELED, "mesh overlap test requires MPI FUNNELED");
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
