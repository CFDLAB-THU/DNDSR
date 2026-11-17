#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "Geom/Access/EntityField.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using FixedState = R::Array<double, R::FixedRows<5>>;
using TypedState = G::EntityField<double>;

namespace
{
    DNDS_DEVICE_CALLABLE double Expected(DNDS::index original, size_t component, int phase)
    {
        return 1000.0 + 17.0 * double(original) + 0.25 * double(component) +
               double(phase) * 0.5 * double(component + 1);
    }

    __global__ void CheckState(R::ArrayView<const double, false, R::FixedRows<5>> values,
                               R::ArrayView<const DNDS::index> original,
                               R::ArrayView<double> difference, int phase)
    {
        for (size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < values.RowCount(); row += blockDim.x * gridDim.x)
            for (size_t component = 0; component < 5; ++component)
                difference(row, component) = values(row, component) - Expected(original(row, 0), component, phase);
    }

    __global__ void UpdateState(R::ArrayView<double, false, R::FixedRows<5>> values)
    {
        for (size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < values.RowCount(); row += blockDim.x * gridDim.x)
            for (size_t component = 0; component < 5; ++component)
                values(row, component) += 0.5 * double(component + 1);
    }

    struct Stream
    {
        std::shared_ptr<R::StreamResource> owner;
        cudaStream_t value;
        explicit Stream(int ordinal) : owner(R::CreateCUDAStream(ordinal)),
                                       value(R::CUDAStream(R::ExecutionContext::CUDA(ordinal, owner))) {}
        ~Stream() { owner->Synchronize(); }
    };

    struct Gate
    {
        std::atomic<bool> open{false};
        cudaStream_t stream;
        explicit Gate(cudaStream_t value) : stream(value) {}
        static void CUDART_CB Wait(void *pointer)
        {
            auto &gate = *static_cast<Gate *>(pointer);
            while (!gate.open.load(std::memory_order_acquire))
                std::this_thread::yield();
        }
        ~Gate()
        {
            open.store(true, std::memory_order_release);
            cudaStreamSynchronize(stream);
        }
    };

    int SelectGPU(const DNDS::MPIInfo &mpi)
    {
        int count = 0;
        REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
        const char *visible = std::getenv("CUDA_VISIBLE_DEVICES");
        const bool isolated = count == 1 && visible && *visible && std::string(visible).find(',') == std::string::npos;
        REQUIRE((isolated || count >= mpi.size));
        const int ordinal = isolated ? 0 : mpi.rank;
        REQUIRE(cudaSetDevice(ordinal) == cudaSuccess);
        cudaDeviceProp properties{};
        REQUIRE(cudaGetDeviceProperties(&properties, ordinal) == cudaSuccess);
        const auto uuid = properties.uuid;
        std::vector<unsigned char> identities(size_t(mpi.size) * sizeof(uuid.bytes));
        MPI_Allgather(uuid.bytes, int(sizeof(uuid.bytes)), MPI_BYTE, identities.data(), int(sizeof(uuid.bytes)), MPI_BYTE, mpi.comm);
        for (int a = 0; a < mpi.size; ++a)
            for (int b = a + 1; b < mpi.size; ++b)
                REQUIRE_FALSE(std::equal(identities.begin() + size_t(a) * sizeof(uuid.bytes),
                                         identities.begin() + size_t(a + 1) * sizeof(uuid.bytes),
                                         identities.begin() + size_t(b) * sizeof(uuid.bytes)));
        REQUIRE(DNDS::MPI::isCudaAware());
        return ordinal;
    }

    std::vector<DNDS::index> GlobalOriginalIds(G::UnstructuredMesh &mesh)
    {
        const auto space = mesh.Entities(G::EntityKind::Cell);
        std::vector<int> counts(size_t(mesh.mpi.size)), offsets(size_t(mesh.mpi.size) + 1, 0);
        int count = int(space.NumOwned());
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, mesh.mpi.comm);
        for (int rank = 0; rank < mesh.mpi.size; ++rank)
            offsets[size_t(rank + 1)] = offsets[size_t(rank)] + counts[size_t(rank)];
        std::vector<DNDS::index> owned(size_t(count), DNDS::UnInitIndex), global(size_t(offsets.back()), DNDS::UnInitIndex);
        for (int row = 0; row < count; ++row)
            owned[size_t(row)] = mesh.OriginalEntityId(G::EntityKind::Cell, row);
        MPI_Allgatherv(owned.data(), count, DNDS::DNDS_MPI_INDEX, global.data(), counts.data(), offsets.data(), DNDS::DNDS_MPI_INDEX, mesh.mpi.comm);
        REQUIRE(global.size() == size_t(space.NumGlobal()));
        for (DNDS::index row = 0; row < space.NumLocal(); ++row)
            CHECK(global.at(size_t(space.LocalToGlobal(row))) == mesh.OriginalEntityId(G::EntityKind::Cell, row));
        return global;
    }

    R::Array<DNDS::index> DeviceOriginalIds(G::UnstructuredMesh &mesh, R::ExecutionContext context)
    {
        R::Array<DNDS::index> host(R::RowLayout::Dense(size_t(mesh.NumCell()), 1));
        {
            auto write = host.Write();
            for (DNDS::index row = 0; row < mesh.NumCell(); ++row)
                write.View()(size_t(row), 0) = mesh.OriginalEntityId(G::EntityKind::Cell, row);
        }
        return host.To(context.device, context);
    }

    void CheckZero(const FixedState &difference, R::ExecutionContext context)
    {
        const auto host = difference.To(R::Device::Host(), context);
        const auto read = host.Read();
        for (size_t row = 0; row < host.RowCount(); ++row)
            for (size_t component = 0; component < 5; ++component)
                CHECK(read.View()(row, component) == 0.0);
    }

    void Probe(TypedState &field, G::UnstructuredMesh &mesh, R::ExecutionContext context, int phase, bool gated)
    {
        const auto space = mesh.Entities(G::EntityKind::Cell);
        auto original = DeviceOriginalIds(mesh, context);
        FixedState difference(R::RowLayout::Dense(size_t(space.NumOwned()), 5), context.device);
        {
            auto read = field.Read(space, context);
            auto ids = original.Read(context);
            auto output = difference.Write(context);
            R::Completion done; // Destroy after Gate releases any queued callback.
            Gate gate(::DNDS::Runtime::CUDAStream(context));
            CHECK_FALSE(mesh.versions.CanMutate());
            CHECK_THROWS(field.Write(space, context));
            CHECK_THROWS(mesh.versions.BeginMutation(R::Change::Geometry));
            if (gated)
                REQUIRE(cudaLaunchHostFunc(gate.stream, Gate::Wait, &gate) == cudaSuccess);
            CheckState<<<4, 64, 0, gate.stream>>>(read.View<R::FixedRows<5>>(), ids.View(), output.View(), phase);
            REQUIRE(cudaGetLastError() == cudaSuccess);
            done = R::Completion::Record(context);
            read.Complete(done);
            ids.Complete(done);
            output.Complete(done);
            CHECK_THROWS(read.View());
            if (gated)
                CHECK_FALSE(done.Ready());
            gate.open.store(true, std::memory_order_release);
            done.Wait();
            CHECK_FALSE(mesh.versions.CanMutate());
        }
        CHECK(mesh.versions.CanMutate());
        CheckZero(difference, context);
    }

    void CheckGhosts(const R::Array<double> &ghost, G::UnstructuredMesh &mesh,
                     const std::vector<DNDS::index> &globalOriginal, R::ExecutionContext context)
    {
        const auto space = mesh.Entities(G::EntityKind::Cell);
        REQUIRE(ghost.RowCount() == size_t(space.NumGhost()));
        auto host = ghost.To(R::Device::Host(), context);
        auto read = host.Read();
        for (size_t row = 0; row < ghost.RowCount(); ++row)
        {
            const auto global = space.LocalToGlobal(space.NumOwned() + DNDS::index(row));
            const auto original = globalOriginal.at(size_t(global));
            for (size_t component = 0; component < 5; ++component)
                CHECK(read.View()(row, component) == Expected(original, component, 1));
        }
        DNDS::index ghosts = space.NumGhost();
        MPI_Allreduce(MPI_IN_PLACE, &ghosts, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mesh.mpi.comm);
        CHECK(ghosts > 0);
    }
}

TEST_CASE("Native EntityField CUDA scopes and direct multi-GPU halo retain checked association")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size >= 2);
    const int ordinal = SelectGPU(mpi);
    R::EagerEngine engine;
    auto migration = engine.CreateStream(std::make_shared<R::CUDALauncher>(ordinal));
    Stream compute(ordinal), communication(ordinal);
    const auto context = R::ExecutionContext::CUDA(ordinal, compute.owner);
    const auto haloContext = R::ExecutionContext::CUDA(ordinal, communication.owner);
    // Resolve lazy module loading before callbacks intentionally hold a stream.
    cudaFuncAttributes attributes{};
    REQUIRE(cudaFuncGetAttributes(&attributes, CheckState) == cudaSuccess);
    REQUIRE(cudaFuncGetAttributes(&attributes, UpdateState) == cudaSuccess);

    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    auto mesh = fixture.mesh;
    auto space = mesh->Entities(G::EntityKind::Cell);
    const auto original = GlobalOriginalIds(*mesh);
    FixedState host(R::RowLayout::Dense(size_t(space.NumOwned()), 5));
    {
        auto write = host.Write();
        for (DNDS::index row = 0; row < space.NumOwned(); ++row)
            for (size_t component = 0; component < 5; ++component)
                write.View()(size_t(row), component) = Expected(mesh->OriginalEntityId(G::EntityKind::Cell, row), component, 0);
    }
    TypedState field(space, host.To(context.device, context), R::ElementSchema::Arithmetic<double>(), context);
    Probe(field, *mesh, context, 0, true);
    {
        auto write = field.Write(space, context);
        R::Completion done;
        Gate gate(compute.value);
        CHECK_FALSE(mesh->versions.CanMutate());
        CHECK_THROWS(field.Read(space, context));
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Geometry));
        REQUIRE(cudaLaunchHostFunc(compute.value, Gate::Wait, &gate) == cudaSuccess);
        UpdateState<<<4, 64, 0, compute.value>>>(write.View<R::FixedRows<5>>());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        done = R::Completion::Record(context);
        write.Complete(done);
        CHECK_THROWS(write.View());
        CHECK_FALSE(done.Ready());
        gate.open.store(true, std::memory_order_release);
        done.Wait();
    }
    CHECK(mesh->versions.CanMutate());
    Probe(field, *mesh, context, 1, false);

    auto route = field.BuildHaloRoute(space);
    auto binding = route.Bind(field, space);
    auto ghosts = binding.GhostValues();
    {
        auto operation = binding.BeginScheduled(haloContext);
        CHECK_FALSE(mesh->versions.CanMutate());
        CHECK_THROWS(ghosts.Read(haloContext));
        CHECK_THROWS(field.Write(space, context));
        operation.Wait();
        CheckGhosts(ghosts, *mesh, original, haloContext);
    }
    CHECK(mesh->versions.CanMutate());

    // CUDA values migrate through the actual detached mesh/field transaction.
    G::MeshTransitionHandle handle(mesh);
    G::ReorderInput input;
    input.explicitMaps.push_back({G::EntityKind::Cell,
                                  std::vector<DNDS::MPI_int>(size_t(space.NumOwned()), (mpi.rank + 1) % mpi.size)});
    auto registration = field.Register("cuda.entity.state", {}, migration);
    auto prepared = G::PreparedMeshTransition::Prepare(handle, input, {registration});
    CHECK_THROWS(field.Write(space, context)); // Prepared values retain the source read.
    prepared.Publish();
    auto next = handle.Get();
    REQUIRE(next->HasNativeState());
    auto nextSpace = next->Entities(G::EntityKind::Cell);
    CHECK(space.Valid());               // Old source is retained; new field association differs.
    CHECK(space.SameDomain(nextSpace)); // Redistribution retains the logical mesh family.
    CHECK_FALSE(space.SameOrder(nextSpace));
    CHECK_THROWS(field.Read(space, context));
    CHECK_THROWS(binding.BeginScheduled(haloContext));
    CHECK_THROWS(route.Bind(field, nextSpace));
    Probe(field, *next, context, 1, false);
    auto nextOriginal = GlobalOriginalIds(*next);
    auto nextRoute = field.BuildHaloRoute(nextSpace);
    auto nextBinding = nextRoute.Bind(field, nextSpace);
    {
        auto operation = nextBinding.BeginScheduled(haloContext);
        operation.Wait();
    }
    CheckGhosts(nextBinding.GhostValues(), *next, nextOriginal, haloContext);
    if (mpi.rank == 0)
        std::cout << "EntityField fixed-width CUDA read/write + direct halo + detached CUDA field migration verified on "
                  << mpi.size << " distinct GPU UUIDs\n";
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    DNDS_check_throw_info(provided >= MPI_THREAD_FUNNELED, "EntityField CUDA test requires MPI FUNNELED");
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
