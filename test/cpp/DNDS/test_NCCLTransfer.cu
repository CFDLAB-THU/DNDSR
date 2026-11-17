#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/Communication/NCCLTransfer.hpp"
#include "DNDS/Runtime/Communication/MPIProgress.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <thread>

namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
static const auto device = R::Device::CUDA(0);
static const auto context = R::ExecutionContext::CUDA(0);
static std::shared_ptr<R::NCCLTransferProvider> provider;
static R::TransferOptions Options()
{
    R::TransferOptions result;
    result.provider = provider;
    return result;
}
static R::RowLayout Shape(int kind, I rows, I width)
{
    return kind == 0 ? R::RowLayout::Dense(rows, width) : kind == 1 ? R::RowLayout::ColMajor(rows, width, {32, 1})
                                                                    : R::RowLayout::TiledColMajor(rows, width, 8);
}
__host__ __device__ static double Value(int rank, I row, I component, int pass)
{
    return pass * 100000000. + rank * 1000000. + row * 100. + component + .25;
}
static void Fill(R::Array<double> &array, int rank, int pass = 0)
{
    auto host = array.To(R::Device::Host());
    {
        auto write = host.Write();
        for (I iRow = 0; iRow < host.RowCount(); ++iRow)
            for (I j = 0; j < I(host.RowSize(iRow)); ++j)
                write.View()(iRow, j) = Value(rank, iRow, j, pass);
    }
    array.CopyFrom(host, context).Wait();
}
__global__ void Produce(R::ArrayView<double> output, int rank, int pass)
{
    for (I i = threadIdx.x; i < output.rows * output.width; i += blockDim.x)
        output(i / output.width, i % output.width) = Value(rank, i / output.width, i % output.width, pass);
}
__global__ void CheckRows(R::ArrayView<const double> input, R::ArrayView<int> errors, int previous, int pass)
{
    for (I i = threadIdx.x; i < 11 * input.width; i += blockDim.x)
    {
        const I iRow = i / input.width, j = i % input.width;
        if (input(iRow, j) != Value(previous, ((10 - iRow) * 7) % 17, j, pass))
            atomicAdd(&errors(0, 0), 1);
    }
}
__global__ void Core(R::ArrayView<const double> input, R::ArrayView<double> output)
{
    for (I i = threadIdx.x; i < output.rows * output.width; i += blockDim.x)
        output(i / output.width, i % output.width) = 2 * input(i / output.width, i % output.width);
}
static void LaunchOK() { DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "NCCL fixture kernel launch failed"); }
struct TestGate
{
    std::shared_ptr<std::atomic<bool>> open = std::make_shared<std::atomic<bool>>(false);
    std::atomic<bool> finished{false}, timedOut{false};
    std::thread watchdog;
    TestGate() : watchdog([this]
                          { const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!finished && std::chrono::steady_clock::now() < limit) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (!finished) { timedOut = true; *open = true; } }) {}
    ~TestGate()
    {
        *open = true;
        finished = true;
        watchdog.join();
    }
    void Release()
    {
        *open = true;
        finished = true;
    }
    void Enqueue(R::EagerEngine &engine, const R::ExecutionStream &stream)
    {
        engine.Submit(stream, [flag = open](auto c)
                      {
            // The callback owns its flag even on an exceptional test exit.
            auto retained = std::make_unique<std::shared_ptr<std::atomic<bool>>>(flag);
            const auto status = cudaLaunchHostFunc(::DNDS::Runtime::CUDAStream(c), [](void *p)
            {
                std::unique_ptr<std::shared_ptr<std::atomic<bool>>> owner(static_cast<std::shared_ptr<std::atomic<bool>> *>(p));
                while (!(*owner)->load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
            }, retained.get());
            if (status == cudaSuccess)
                retained.release();
            DNDS_check_throw_info(status == cudaSuccess, "gate submission failed"); });
    }
};

TEST_CASE("NCCL packed rows match exact values across layouts and queued producer consumer epochs")
{
    const int previous = (mpi.rank + mpi.size - 1) % mpi.size;
    for (I width : {I(1), I(5), I(37)})
        for (int from = 0; from < 3; ++from)
            for (int to = 0; to < 3; ++to)
            {
                R::Array<double> source(Shape(from, 17, width), device), destination(Shape(to, 13, width), device);
                Fill(source, mpi.rank);
                Fill(destination, -1);
                std::vector<R::RowRequest> requests;
                for (I iRow = 0; iRow < 11; ++iRow)
                    requests.push_back({previous, (iRow * 7) % 17, 10 - iRow});
                R::PayloadPlan payload(R::RowTransferPlan(mpi.comm, 17, 13, requests), source.Layout(), destination.Layout(), R::ElementSchema::For<double>());
                auto binding = payload.Bind(source, destination, R::TransferTransport::DevicePacked, Options());
                const auto version = destination.Version();
                auto manual = binding.Begin(context);
                manual.WaitDestinationReady();
                manual.Wait();
                CHECK(manual.Test());
                CHECK(destination.Version() == version + 1);
                R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
                auto producer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
                auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
                auto consumer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
                R::Array<int> errors(R::RowLayout::Dense(1, 1), device);
                for (int pass = 1; pass <= 3; ++pass)
                {
                    R::SubmitArrays(engine, producer, [pass](auto c, auto output)
                                    { Produce<<<1,128,0,::DNDS::Runtime::CUDAStream(c)>>>(output, mpi.rank, pass); LaunchOK(); }, R::Write(source));
                    auto transfer = binding.SubmitAfter(engine, comm, producer.Record());
                    CHECK_FALSE(transfer.RequiresHostProgress());
                    transfer.EnqueueWait(consumer);
                    R::SubmitArrays(engine, consumer, [previous, pass](auto c, auto input, auto errors)
                                    { CheckRows<<<1,128,0,::DNDS::Runtime::CUDAStream(c)>>>(input, errors, previous, pass); LaunchOK(); }, R::Read(destination), R::ReadWrite(errors));
                    producer.WaitFor(consumer.Record());
                }
                engine.Synchronize();
                CHECK(errors.To(R::Device::Host()).Read().View()(0, 0) == 0);
                const auto host = destination.To(R::Device::Host());
                const auto read = host.Read();
                for (I iRow = 11; iRow < 13; ++iRow)
                    for (I j = 0; j < width; ++j)
                        CHECK(read.View()(iRow, j) == Value(-1, iRow, j, 0));
            }
}
TEST_CASE("NCCL direct contiguous copies and explicit unsupported bindings")
{
    R::Array<double> source(Shape(0, 9, 5), device), destination(Shape(0, 9, 5), device);
    Fill(source, mpi.rank);
    std::vector<R::RowRequest> requests;
    for (I iRow = 0; iRow < 9; ++iRow)
        requests.push_back({(mpi.rank + 1) % mpi.size, iRow, iRow});
    R::PayloadPlan payload(R::RowTransferPlan(mpi.comm, 9, 9, requests), source.Layout(), destination.Layout(), R::ElementSchema::For<double>());
    auto binding = payload.Bind(source, destination, R::TransferTransport::Direct, Options());
    auto active = binding.BeginScheduled(context);
    CHECK_THROWS(binding.BeginScheduled(context));
    R::EagerEngine engine;
    auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    CHECK_THROWS(binding.Enqueue(engine, stream));
    auto moved = std::move(active);
    CHECK(active.Test());
    moved.Wait();
    CHECK(moved.Test());
    R::EagerEngine otherEngine;
    const auto otherStream = otherEngine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto version = destination.Version();
    CHECK_THROWS(binding.Enqueue(engine, otherStream));
    CHECK(destination.Version() == version);
    const auto host = destination.To(R::Device::Host());
    const auto read = host.Read();
    for (I iRow = 0; iRow < 9; ++iRow)
        for (I j = 0; j < 5; ++j)
            CHECK(read.View()(iRow, j) == Value((mpi.rank + 1) % mpi.size, iRow, j, 0));
    CHECK_THROWS(binding.Begin(R::ExecutionContext::Host()));
    CHECK_THROWS(payload.Bind(source, destination, R::TransferTransport::HostPacked, Options()));
    CHECK_THROWS(payload.Bind(source, destination, R::Reducer::Builtin<double>(R::Reduction::Sum), R::TransferTransport::DevicePacked, Options()));
    auto bad = Options();
    bad.tag = 1;
    CHECK_THROWS(payload.Bind(source, destination, R::TransferTransport::Direct, bad));
    R::Array<double> colSource(R::RowLayout::ColMajor(9, 5, {8, 1}), device);
    Fill(colSource, mpi.rank);
    R::PayloadPlan columnPayload(R::RowTransferPlan(mpi.comm, 9, 9, requests), colSource.Layout(), destination.Layout(), R::ElementSchema::For<double>());
    CHECK_THROWS(columnPayload.Bind(colSource, destination, R::TransferTransport::Direct, Options()));
    auto hostSource = source.To(R::Device::Host());
    CHECK_THROWS(payload.Bind(hostSource, destination, R::TransferTransport::Direct, Options()));
    {
        auto held = destination.Write(context);
        CHECK_THROWS(binding.Begin(context));
    }
    // A rejected destination lease must cancel the source reservation and
    // active-binding flag without publishing an unperformed transfer.
    binding.Begin(context).Wait();
}
TEST_CASE("NCCL packed CSR empty ownership and omitted POD bytes preserve receiver contents")
{
    for (auto transport : {R::TransferTransport::Direct, R::TransferTransport::DevicePacked})
    {
        R::Array<double> emptySource(Shape(0, 0, 5), device), emptyDestination(Shape(0, 0, 5), device);
        R::PayloadPlan emptyPayload(R::RowTransferPlan(mpi.comm, 0, 0, {}), emptySource.Layout(), emptyDestination.Layout(), R::ElementSchema::For<double>());
        auto empty = emptyPayload.Bind(emptySource, emptyDestination, transport, Options());
        empty.Begin(context).Wait();
        R::EagerEngine engine;
        auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        empty.Enqueue(engine, stream);
        engine.Synchronize();
        CHECK(emptyDestination.RowCount() == 0);
    }
    R::Array<double> source(mpi.rank == 0 ? R::RowLayout::Ragged({0, 3, 7}) : R::RowLayout::Ragged({}), device);
    R::Array<double> destination(R::RowLayout::Padded({7, 0, 3}, 9), device);
    Fill(source, mpi.rank);
    R::PayloadPlan payload(R::RowTransferPlan(mpi.comm, source.RowCount(), 3, {{0, 2, 0}, {0, 0, 1}, {0, 1, 2}}), source.Layout(), destination.Layout(), R::ElementSchema::For<double>());
    payload.Bind(source, destination, R::TransferTransport::DevicePacked, Options()).Begin(context).Wait();
    const auto host = destination.To(R::Device::Host());
    const auto read = host.Read();
    for (I j = 0; j < 7; ++j)
        CHECK(read.View()(0, j) == Value(0, 2, j, 0));
    for (I j = 0; j < 3; ++j)
        CHECK(read.View()(2, j) == Value(0, 1, j, 0));

    struct Record
    {
        int32_t id;
        double value;
        int32_t omitted;
    };
    const auto schema = R::ElementSchema::Register<Record>(R::ElementSchema::Record("NCCL.Record.v1", sizeof(Record), {{"id", offsetof(Record, id), R::ElementScalar::I32}, {"value", offsetof(Record, value), R::ElementScalar::F64}}, "Record", alignof(Record)));
    R::Array<Record> src(Shape(2, 5, 3)), dst(Shape(1, 5, 3));
    {
        auto w = src.Write();
        for (I iRow = 0; iRow < 5; ++iRow)
            for (I j = 0; j < 3; ++j)
                w.View()(iRow, j) = {int32_t(iRow + 10 * mpi.rank), Value(mpi.rank, iRow, j, 0), 123};
        auto raw = dst.Storage().Write();
        std::fill_n(raw.View<uint8_t>().data, dst.Storage().Bytes(), uint8_t(0x77));
    }
    auto gpuSrc = src.To(device), gpuDst = dst.To(device);
    std::vector<R::RowRequest> requests;
    const int previous = (mpi.rank + mpi.size - 1) % mpi.size;
    for (I iRow = 0; iRow < 5; ++iRow)
        requests.push_back({previous, 4 - iRow, iRow});
    R::PayloadPlan pod(R::RowTransferPlan(mpi.comm, 5, 5, requests), gpuSrc.Layout(), gpuDst.Layout(), schema);
    pod.Bind(gpuSrc, gpuDst, R::TransferTransport::DevicePacked, Options()).Begin(context).Wait();
    const auto result = gpuDst.To(R::Device::Host());
    const auto values = result.Read();
    const auto bytes = result.Storage().Read();
    for (I iRow = 0; iRow < 5; ++iRow)
        for (I j = 0; j < 3; ++j)
        {
            CHECK(values.View()(iRow, j).id == 4 - iRow + previous * 10);
            CHECK(values.View()(iRow, j).value == Value(previous, 4 - iRow, j, 0));
            const auto offset = result.Layout().ElementOffset(iRow, j) * I(sizeof(Record));
            for (I byte = 0; byte < I(sizeof(Record)); ++byte)
                if (!(byte < I(sizeof(int32_t)) || (byte >= I(offsetof(Record, value)) && byte < I(offsetof(Record, value) + sizeof(double)))))
                    CHECK(bytes.View<uint8_t>()[offset + byte] == 0x77);
        }
}
TEST_CASE("NCCL enqueue overlaps core work with gated transfer and retains leases and workspace")
{
    R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
    auto control = R::CommunicationContext::Duplicate(mpi.comm);
    std::weak_ptr<R::CommunicationContext> retainedControl = control;
    auto ownedProvider = R::NCCLTransferProvider::Create(control, device);
    CHECK(ownedProvider->ControlContext()->Communicator() == control->Communicator());
    auto options = Options();
    options.provider = ownedProvider;
    auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto compute = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    R::Array<double> source(Shape(1, 17, 5), device), destination(Shape(2, 17, 5), device), output(Shape(0, 17, 5), device);
    Fill(source, mpi.rank);
    std::vector<R::RowRequest> requests;
    for (I iRow = 0; iRow < 17; ++iRow)
        requests.push_back({(mpi.rank + 1) % mpi.size, iRow, iRow});
    auto binding = std::make_unique<R::TransferBinding>(R::PayloadPlan(R::RowTransferPlan(mpi.comm, 17, 17, requests), source.Layout(), destination.Layout(), R::ElementSchema::For<double>()).Bind(source, destination, R::TransferTransport::DevicePacked, options));
    options.provider.reset();
    ownedProvider.reset();
    control.reset();
    const auto core = [](auto c, auto input, auto result)
    { Core<<<1,128,0,::DNDS::Runtime::CUDAStream(c)>>>(input, result); LaunchOK(); };
    R::SubmitArrays(engine, compute, core, R::Read(source), R::Write(output));
    binding->Enqueue(engine, comm);
    engine.Synchronize();
    TestGate gate;
    gate.Enqueue(engine, comm);
    binding->Enqueue(engine, comm);
    const auto haloReady = comm.Record();
    CHECK_FALSE(haloReady.Ready());
    CHECK_THROWS(source.Write(context));
    CHECK_THROWS(destination.Read(context));
    R::SubmitArrays(engine, compute, core, R::Read(source), R::Write(output));
    compute.Record().WaitHost();
    CHECK_FALSE(gate.timedOut.load());
    CHECK_FALSE(haloReady.Ready());
    auto retained = destination;
    source = {};
    destination = {};
    binding.reset();
    CHECK_FALSE(retainedControl.expired());
    auto cpu = engine.CreateCPUStream();
    std::atomic<bool> consumed{false};
    cpu.WaitFor(haloReady); // May retire numerical captures on this worker.
    engine.Submit(cpu, [&](R::ExecutionContext)
                  { consumed = true; });
    gate.Release();
    engine.Close(); // Progress provider status and retire its owners on controller.
    CHECK(consumed.load());
    CHECK(haloReady.Ready());
    CHECK(retainedControl.expired());
    const auto result = retained.To(R::Device::Host());
    const auto read = result.Read();
    for (I iRow = 0; iRow < 17; ++iRow)
        for (I j = 0; j < 5; ++j)
            CHECK(read.View()(iRow, j) == Value((mpi.rank + 1) % mpi.size, iRow, j, 0));
}
TEST_CASE("CUDA producer chains device-packed MPI and NCCL across streams through pending shutdown")
{
    R::EagerEngine engine(1, std::make_shared<R::AccessTrackingContext>());
    auto control = R::CommunicationContext::Duplicate(mpi.comm);
    std::weak_ptr<R::CommunicationContext> retainedControl = control;
    auto localProvider = R::NCCLTransferProvider::Create(control, device);
    R::TransferOptions ncclOptions;
    ncclOptions.provider = localProvider;

    auto producer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto mpiStream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto ncclStream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto observer = engine.CreateCPUStream();
    R::Array<double> source(Shape(1, 17, 5), device);
    R::Array<double> mpiRows(Shape(2, 17, 5), device);
    R::Array<double> ncclRows(Shape(0, 17, 5), device);
    const int previous = (mpi.rank + mpi.size - 1) % mpi.size;
    const int twicePrevious = (previous + mpi.size - 1) % mpi.size;
    std::vector<R::RowRequest> requests;
    for (I iRow = 0; iRow < 17; ++iRow)
        requests.push_back({previous, iRow, iRow});
    auto route = R::RowTransferPlan(mpi.comm, 17, 17, requests);
    auto mpiBinding = std::make_unique<R::TransferBinding>(
        R::PayloadPlan(route, source.Layout(), mpiRows.Layout(), R::ElementSchema::For<double>())
            .Bind(source, mpiRows, R::TransferTransport::DevicePacked));
    auto ncclBinding = std::make_unique<R::TransferBinding>(
        R::PayloadPlan(std::move(route), mpiRows.Layout(), ncclRows.Layout(), R::ElementSchema::For<double>())
            .Bind(mpiRows, ncclRows, R::TransferTransport::DevicePacked, ncclOptions));

    R::SubmitArrays(engine, producer, [](auto c, auto output)
                    { Produce<<<1,128,0,::DNDS::Runtime::CUDAStream(c)>>>(output, mpi.rank, 7); LaunchOK(); }, R::Write(source));
    // CUDA-aware MPI is not stream ordered: the controller must observe the
    // producer before MPI may reserve/read its device buffer. The subsequent
    // MPI completion is exported as an event for the stream-ordered NCCL leg.
    auto mpiOperation = mpiBinding->SubmitAfter(engine, mpiStream, producer.Record());
    CHECK(mpiOperation.RequiresHostProgress());
    // MPI completion is a host-progress event, not a CUDA-stream event. The
    // controller observes unpack completion before entering the fully
    // stream-ordered NCCL segment.
    mpiOperation.EnqueueWait(ncclStream);
    mpiOperation = {};
    TestGate gate;
    gate.Enqueue(engine, ncclStream);
    auto ncclOperation = ncclBinding->SubmitReady(engine, ncclStream);
    CHECK_FALSE(ncclOperation.RequiresHostProgress());
    const auto resultReady = ncclOperation.CompletionEvent();
    ncclOperation = {}; // The runtime retains provider/payload access until retirement.
    CHECK_FALSE(resultReady.Ready());

    auto retained = ncclRows;
    source = {};
    mpiRows = {};
    ncclRows = {};
    mpiBinding.reset();
    ncclBinding.reset();
    ncclOptions.provider.reset();
    localProvider.reset();
    control.reset();
    CHECK_FALSE(retainedControl.expired());

    std::atomic<bool> observed{false};
    observer.WaitFor(resultReady);
    engine.Submit(observer, [&](R::ExecutionContext)
                  { observed = true; });
    gate.Release();
    engine.Close();
    CHECK(observed.load());
    CHECK(resultReady.Ready());
    CHECK(retainedControl.expired());
    const auto host = retained.To(R::Device::Host());
    const auto read = host.Read();
    for (I iRow = 0; iRow < 17; ++iRow)
        for (I j = 0; j < 5; ++j)
            CHECK(read.View()(iRow, j) == Value(twicePrevious, iRow, j, 7));
}
TEST_CASE("NCCL DOF halo publishes the retired producer version without a submission host wait")
{
    namespace A = DNDS::Algebra;
    auto control = std::make_shared<R::CommunicationContext>(mpi.comm);
    A::DofSpace space(control, 17, 1);
    A::DofField<> field(space, R::Array<double>(Shape(1, 17, 5), device),
                        R::Array<double>(Shape(2, 1, 5), device));
    const int previous = (mpi.rank + mpi.size - 1) % mpi.size;
    R::RowTransferPlan route(control, 17, 1, {{previous, 3, 0}});
    A::DofHalo<> halo(field, route, Options(), R::TransferTransport::DevicePacked);
    R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
    auto producer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto consumer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    TestGate gate;
    gate.Enqueue(engine, producer);
    R::SubmitArrays(engine, producer, [](auto context, auto output)
                    { Produce<<<1,128,0,::DNDS::Runtime::CUDAStream(context)>>>(output, mpi.rank, 9); LaunchOK(); }, R::Write(field.Owned()));
    auto publication = std::make_shared<A::DofHalo<>::Operation>(halo.Begin(9, engine, producer));
    publication->EnqueuePublication(engine, consumer, publication);
    publication.reset();
    // Reaching this release proves no producer-completion host wait was added
    // to native halo launch/publication, even with a pending source generation.
    gate.Release();
    engine.Close();
    halo.RequireCurrent(field, 9);
    const auto ghost = field.Ghost().To(R::Device::Host());
    for (I j = 0; j < 5; ++j)
        CHECK(ghost.Read().At(0, j) == Value(previous, 3, j, 9));
}

TEST_CASE("Unchecked NCCL missing producer edge demonstrably transfers a stale epoch")
{
    R::Array<double> source(Shape(0, 17, 5), device), destination(Shape(0, 17, 5), device), warm(Shape(0, 17, 5), device);
    Fill(source, mpi.rank, 0);
    std::vector<R::RowRequest> requests;
    const int previous = (mpi.rank + mpi.size - 1) % mpi.size;
    for (I iRow = 0; iRow < 17; ++iRow)
        requests.push_back({previous, iRow, iRow});
    auto binding = R::PayloadPlan(R::RowTransferPlan(mpi.comm, 17, 17, requests), source.Layout(), destination.Layout(), R::ElementSchema::For<double>()).Bind(source, destination, R::TransferTransport::DevicePacked, Options());
    binding.Begin(context).Wait();
    R::EagerEngine engine;
    auto producer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto produce = [](auto c, auto output)
    { Produce<<<1,128,0,::DNDS::Runtime::CUDAStream(c)>>>(output, mpi.rank, 1); LaunchOK(); };
    R::SubmitArrays(engine, producer, produce, R::Write(warm));
    engine.Synchronize();
    TestGate gate;
    gate.Enqueue(engine, producer);
    R::SubmitArrays(engine, producer, produce, R::Write(source));
    // Deliberately omit comm.WaitFor(producer.Record()). The closed gate
    // ensures a stale read, without creating an actual overlapping data race.
    binding.Enqueue(engine, comm);
    comm.Record().WaitHost();
    CHECK_FALSE(gate.timedOut.load());
    CHECK_FALSE(producer.Record().Ready());
    gate.Release();
    engine.Synchronize();
    const auto result = destination.To(R::Device::Host());
    const auto read = result.Read();
    for (I iRow = 0; iRow < 17; ++iRow)
        for (I j = 0; j < 5; ++j)
        {
            CHECK(read.View()(iRow, j) == Value(previous, iRow, j, 0));
            CHECK(read.View()(iRow, j) != Value(previous, iRow, j, 1));
        }
    CHECK(source.To(R::Device::Host()).Read().View()(0, 0) == Value(mpi.rank, 0, 0, 1));
}
TEST_CASE("NCCL collisions and aborted communicator reject explicitly")
{
    R::Array<double> source(Shape(0, 2, 1), device), destination(Shape(0, 1, 1), device);
    R::PayloadPlan collided(R::RowTransferPlan(mpi.comm, 2, 1, {{0, 0, 0}, {0, 1, 0}}), source.Layout(), destination.Layout(), R::ElementSchema::For<double>());
    CHECK_THROWS(collided.Bind(source, destination, R::TransferTransport::DevicePacked, Options()));
    CHECK_THROWS(collided.Bind(source, destination, R::Reducer::Builtin<double>(R::Reduction::Sum), R::TransferTransport::DevicePacked, Options()));
    auto isolated = R::NCCLTransferProvider::Create(mpi.comm, device);
    auto options = Options();
    options.provider = isolated;
    R::PayloadPlan payload(R::RowTransferPlan(mpi.comm, 2, 1, {{0, 0, 0}}), source.Layout(), destination.Layout(), R::ElementSchema::For<double>());
    auto binding = payload.Bind(source, destination, R::TransferTransport::DevicePacked, options);
    isolated->CheckAsyncError();
    isolated->Abort();
    CHECK_THROWS(isolated->CheckAsyncError());
    CHECK_THROWS(binding.Begin(context));
}
TEST_CASE("NCCL binding checks communicator identity beyond congruent MPI groups")
{
    if (mpi.size < 2)
        return;
    auto first = R::NCCLTransferProvider::Create(mpi.comm, device);
    auto second = R::NCCLTransferProvider::Create(mpi.comm, device);
    R::Array<double> source(Shape(0, 1, 1), device), destination(Shape(0, 1, 1), device);
    auto options = Options();
    options.provider = mpi.rank == 0 ? first : second;
    R::PayloadPlan payload(R::RowTransferPlan(mpi.comm, 1, 1, {{(mpi.rank + 1) % mpi.size, 0, 0}}), source.Layout(), destination.Layout(), R::ElementSchema::For<double>());
    CHECK_THROWS(payload.Bind(source, destination, R::TransferTransport::DevicePacked, options));
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi = DNDS::MPIInfo(MPI_COMM_WORLD);
    provider = R::NCCLTransferProvider::Create(mpi.comm, device);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    const int result = ctx.run();
    provider.reset();
    MPI_Finalize();
    return result;
}
