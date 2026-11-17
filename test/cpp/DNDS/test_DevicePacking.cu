#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Communication/MPIProgress.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#include <chrono>

namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
static const auto device = R::Device::CUDA(0);
static const auto context = R::ExecutionContext::CUDA(0);
static double Value(int rank, I row, I component) { return rank * 1000000 + row * 100 + component + .25; }
static R::RowLayout Shape(int kind, I rows, I width)
{
    return kind == 0 ? R::RowLayout::Dense(rows, width, width + 3) : kind == 1 ? R::RowLayout::ColMajor(rows, width, {32, 1})
                                                                               : R::RowLayout::TiledColMajor(rows, width, 8);
}
static void Fill(R::Array<double> &array, int rank)
{
    auto host = array.To(R::Device::Host());
    auto write = host.Write();
    const auto view = write.View();
    for (I row = 0; row < host.RowCount(); ++row)
        for (I j = 0; j < I(host.RowSize(row)); ++j)
            view(row, j) = Value(rank, row, j);
    write.Wait();
    array.CopyFrom(host, context).Wait();
}
TEST_CASE("Device packing preserves wire row order across layouts and persistent transient modes")
{
    const int previous = (mpi.rank + mpi.size - 1) % mpi.size;
    for (I width : {I(1), I(5), I(37)})
        for (int from = 0; from < 3; ++from)
            for (int to = 0; to < 3; ++to)
            {
                R::Array<double> source(Shape(from, 17, width), device), destination(Shape(to, 13, width), device);
                Fill(source, mpi.rank);
                Fill(destination, -1);
                std::vector<R::RowRequest> rows;
                for (I row = 0; row < 11; ++row)
                    rows.push_back({previous, (row * 7) % 17, 10 - row});
                R::RowTransferPlan route(mpi.comm, 17, 13, rows);
                R::PayloadPlan payload(route, source.Layout(), destination.Layout(), R::ElementSchema::For<double>());
                for (auto mode : {R::TransferMode::Persistent, R::TransferMode::Transient})
                {
                    R::TransferOptions options;
                    options.mode = mode;
                    auto binding = payload.Bind(source, destination, R::TransferTransport::DevicePacked, options);
                    for (int pass = 0; pass < 2; ++pass)
                    {
                        const auto version = destination.Version();
                        auto operation = binding.Begin(context);
                        operation.WaitDestinationReady();
                        operation.Wait();
                        CHECK(operation.Test());
                        CHECK(destination.Version() == version + 1);
                        const auto host = destination.To(R::Device::Host());
                        const auto read = host.Read();
                        for (I row = 0; row < 13; ++row)
                            for (I j = 0; j < width; ++j)
                                CHECK(read.View()(row, j) == Value(row < 11 ? previous : -1, row < 11 ? ((10 - row) * 7) % 17 : row, j));
                    }
                }
            }
}
TEST_CASE("Device packing handles CSR zero rows empty source peers and distributed fanout")
{
    auto shape = mpi.rank == 0 ? R::RowLayout::Ragged({0, 3, 7}) : R::RowLayout::Ragged({});
    R::Array<double> source(shape, device), destination(R::RowLayout::Padded({7, 0, 3}, 9), device);
    Fill(source, mpi.rank);
    R::RowTransferPlan route(mpi.comm, source.RowCount(), 3, {{0, 2, 0}, {0, 0, 1}, {0, 1, 2}});
    R::PayloadPlan payload(route, source.Layout(), destination.Layout(), R::ElementSchema::For<double>());
    auto binding = payload.Bind(source, destination, R::TransferTransport::DevicePacked);
    auto operation = binding.Begin(context);
    while (!operation.Test())
        std::this_thread::yield();
    const auto host = destination.To(R::Device::Host());
    const auto read = host.Read();
    for (I row = 0; row < 3; ++row)
        for (I j = 0; j < I(host.RowSize(row)); ++j)
            CHECK(read.View()(row, j) == Value(0, row == 0 ? 2 : 1, j));
}
struct alignas(512) Record
{
    int32_t id;
    double value;
    int64_t untouched;
};
TEST_CASE("Device packed POD uses the registered schema extent and preserves omitted fields and padding")
{
    auto schema = R::ElementSchema::Register<Record>(R::ElementSchema::Record("DevicePacking.Record.v1", sizeof(Record), {{"id", offsetof(Record, id), R::ElementScalar::I32}, {"value", offsetof(Record, value), R::ElementScalar::F64}}, "Record", alignof(Record)));
    R::Array<Record> host(R::RowLayout::TiledColMajor(9, 3, 8)), initial(R::RowLayout::ColMajor(9, 3, {8, 1}));
    {
        auto w = host.Write();
        for (I row = 0; row < 9; ++row)
            for (I j = 0; j < 3; ++j)
                w.View()(row, j) = {int32_t(mpi.rank * 1000 + row * 10 + j), Value(mpi.rank, row, j), 123};
    }
    {
        auto w = initial.Storage().Write();
        std::fill_n(w.View<uint8_t>().data, initial.Storage().Bytes(), uint8_t(0x77));
    }
    auto source = host.To(device), destination = initial.To(device);
    const int previous = (mpi.rank + mpi.size - 1) % mpi.size;
    std::vector<R::RowRequest> rows;
    for (I row = 0; row < 9; ++row)
        rows.push_back({previous, 8 - row, row});
    R::PayloadPlan payload(R::RowTransferPlan(mpi.comm, 9, 9, rows), source.Layout(), destination.Layout(), schema);
    auto binding = payload.Bind(source, destination, R::TransferTransport::DevicePacked);
    binding.Begin(context).Wait();
    const auto result = destination.To(R::Device::Host());
    const auto read = result.Read();
    const auto bytes = result.Storage().Read();
    const auto raw = bytes.View<uint8_t>();
    for (I row = 0; row < 9; ++row)
        for (I j = 0; j < 3; ++j)
        {
            CHECK(read.View()(row, j).id == previous * 1000 + (8 - row) * 10 + j);
            CHECK(read.View()(row, j).value == Value(previous, 8 - row, j));
            const auto offset = result.Layout().ElementOffset(row, j) * I(sizeof(Record));
            for (I byte = 0; byte < I(sizeof(Record)); ++byte)
                if (!(byte < I(sizeof(int32_t)) || (byte >= I(offsetof(Record, value)) && byte < I(offsetof(Record, value) + sizeof(double)))))
                    CHECK(raw[offset + byte] == 0x77);
        }
}
TEST_CASE("Device packed collisions require a reducer and use stable receiver reduction")
{
    R::Array<int64_t> host(R::RowLayout::ColMajor(3, 7, {8, 1}));
    {
        auto w = host.Write();
        for (I row = 0; row < 3; ++row)
            for (I j = 0; j < 7; ++j)
                w.View()(row, j) = mpi.rank * 100 + row * 10 + j;
    }
    auto source = host.To(device);
    R::Array<int64_t> destination(R::RowLayout::TiledColMajor(2, 7, 8), device);
    std::vector<R::RowRequest> requests;
    for (int rank = 0; rank < mpi.size; ++rank)
        for (I row : {I(0), I(2)})
            requests.push_back({rank, row, 0});
    R::PayloadPlan payload(R::RowTransferPlan(mpi.comm, 3, 2, requests), source.Layout(), destination.Layout(), R::ElementSchema::For<int64_t>());
    CHECK_THROWS(payload.Bind(source, destination, R::TransferTransport::DevicePacked));
    for (auto reduction : {R::Reduction::Sum, R::Reduction::BitXor})
    {
        auto binding = payload.Bind(source, destination, R::Reducer::Builtin<int64_t>(reduction, R::DestinationInit::Identity), R::TransferTransport::DevicePacked);
        binding.BeginScheduled(context).Wait();
        const auto result = destination.To(R::Device::Host());
        const auto read = result.Read();
        for (I j = 0; j < 7; ++j)
        {
            int64_t expected = 0;
            for (int rank = 0; rank < mpi.size; ++rank)
                for (I row : {I(0), I(2)})
                {
                    const int64_t value = rank * 100 + row * 10 + j;
                    if (reduction == R::Reduction::Sum)
                        expected += value;
                    else
                        expected ^= value;
                }
            CHECK(read.View()(0, j) == expected);
            CHECK(read.View()(1, j) == 0);
        }
    }
}
__global__ void Core(R::ArrayView<const double> source, R::ArrayView<double> output)
{
    for (I row = blockIdx.x; row < output.rows; row += gridDim.x)
        for (I j = threadIdx.x; j < output.width; j += blockDim.x)
            output(row, j) = source(row, j) * 2;
}
TEST_CASE("Device packing begins eagerly and overlaps owned compute with a gated communication stream")
{
    R::EagerEngine engine;
    auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto compute = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto open = std::make_shared<std::atomic<bool>>(false);
    struct OpenOnExit
    {
        std::shared_ptr<std::atomic<bool>> value;
        ~OpenOnExit() { *value = true; }
    } cleanup{open};
    R::Array<double> source(Shape(1, 17, 5), device), destination(Shape(2, 17, 5), device), output(Shape(0, 17, 5), device);
    Fill(source, mpi.rank);
    std::vector<R::RowRequest> rows;
    for (I row = 0; row < 17; ++row)
        rows.push_back({(mpi.rank + mpi.size - 1) % mpi.size, row, row});
    auto binding = std::make_unique<R::TransferBinding>(R::PayloadPlan(R::RowTransferPlan(mpi.comm, 17, 17, rows), source.Layout(), destination.Layout(), R::ElementSchema::For<double>()).Bind(source, destination, R::TransferTransport::DevicePacked));
    // Load the real compute kernel before blocking a native CUDA stream;
    // lazy module loading itself may synchronize the context.
    R::SubmitArrays(engine, compute, [](R::ExecutionContext c, auto input, auto result)
                    {Core<<<17,32,0,::DNDS::Runtime::CUDAStream(c)>>>(input,result);DNDS_check_throw_info(cudaGetLastError()==cudaSuccess,"core preload failed"); }, R::Read(source), R::Write(output));
    engine.Synchronize();
    binding->Begin(comm.Context()).Wait(); // warm MPI and packing modules too
    engine.Submit(comm, [open](R::ExecutionContext c)
                  { DNDS_check_throw_info(cudaLaunchHostFunc(::DNDS::Runtime::CUDAStream(c), [](void *p)
                                                             {while(!static_cast<std::atomic<bool> *>(p)->load())std::this_thread::sleep_for(std::chrono::microseconds(50)); }, open.get()) == cudaSuccess,
                                          "gate launch failed"); });
    std::atomic<bool> returned{false}, timedOut{false};
    std::thread watchdog([&]
                         {const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(5);while(!returned&&std::chrono::steady_clock::now()<limit)std::this_thread::sleep_for(std::chrono::milliseconds(1));if(!returned){timedOut=true;*open=true;} });
    auto operation = binding->BeginScheduled(comm);
    CHECK_FALSE(timedOut.load());
    CHECK_FALSE(operation.Test());
    R::SubmitArrays(engine, compute, [](R::ExecutionContext c, auto input, auto result)
                    {Core<<<17,32,0,::DNDS::Runtime::CUDAStream(c)>>>(input,result);DNDS_check_throw_info(cudaGetLastError()==cudaSuccess,"core launch failed"); }, R::Read(source), R::Write(output));
    compute.Record().WaitHost();
    returned = true;
    watchdog.join();
    CHECK_FALSE(timedOut.load());
    CHECK_FALSE(operation.Test());
    CHECK_THROWS(source.Write(context));
    CHECK_THROWS(destination.Read(context));
    auto retained = destination;
    source = {};
    destination = {};
    binding.reset();
    *open = true;
    R::MPIProgress progress;
    auto arrived = progress.Import(std::move(operation));
    engine.Close(); // Runtime progresses posted MPI and late unpack before closing streams.
    CHECK(arrived.Ready());
    const auto result = retained.To(R::Device::Host()), core = output.To(R::Device::Host());
    const auto a = result.Read(), b = core.Read();
    for (I row = 0; row < 17; ++row)
        for (I j = 0; j < 5; ++j)
        {
            CHECK(a.View()(row, j) == Value((mpi.rank + mpi.size - 1) % mpi.size, row, j));
            CHECK(b.View()(row, j) == Value(mpi.rank, row, j) * 2);
        }
}
TEST_CASE("Device packed preflight rejects invalid placement and rolls back unmodified destination")
{
    R::Array<double> host(R::RowLayout::Dense(3, 5)), other(host.Layout());
    R::RowTransferPlan route(mpi.comm, 3, 3, {{mpi.rank, 1, 2}});
    R::PayloadPlan payload(route, host.Layout(), other.Layout(), R::ElementSchema::For<double>());
    CHECK_THROWS(payload.Bind(host, other, R::TransferTransport::DevicePacked));
    auto source = host.To(device), destination = other.To(device);
    auto binding = payload.Bind(source, destination, R::TransferTransport::DevicePacked);
    const auto version = destination.Version();
    std::optional<R::Array<double>::WriteLease> held;
    if (mpi.rank == 0)
        held.emplace(destination.Write(context));
    CHECK_THROWS(binding.Begin(context));
    if (held)
        held.reset();
    CHECK(destination.IsValid());
    // Rank0's caller-owned write publishes once; other ranks' cancelled receive
    // reservations publish nothing and the binding remains reusable.
    CHECK(destination.Version() == version + (mpi.rank == 0));
    binding.Begin(context).Wait();
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi = DNDS::MPIInfo(MPI_COMM_WORLD);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    const auto result = ctx.run();
    MPI_Finalize();
    return result;
}
