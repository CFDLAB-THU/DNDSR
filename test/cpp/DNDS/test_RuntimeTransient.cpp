#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/Communication/DynamicRowTransfer.hpp"
#include "DNDS/Runtime/Communication/MPIProgress.hpp"
#include "DNDS/Runtime/Communication/MPIGlobalReduction.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/EagerCopy.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include <chrono>
#include <optional>

using namespace DNDS::Runtime;

TEST_CASE("Known route rejects negative extents on every rank before publishing")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    std::vector<std::vector<DNDS::index>> peers(size);
    CHECK_THROWS(RowTransferPlan::Known(context, rank == 0 ? -1 : 0, 0, peers, peers));
    CHECK_THROWS(RowTransferPlan::Known(context, 0, rank == 0 ? -1 : 0, peers, peers));
    auto route = RowTransferPlan::Known(context, 0, 0, peers, peers);
    CHECK(route.Valid());
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    int result = context.run();
    MPI_Finalize();
    return result;
}

TEST_CASE("MPI allreduce queues behind a producer and orders the next reduction and consumer")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (const int workers : {0, 1})
    {
        EagerEngine engine(workers, std::make_shared<AccessTrackingContext>());
        const auto compute = engine.CreateCPUStream(), communication = engine.CreateCPUStream();
        const DNDS::MPIInfo laneMPI(workers ? mpi.comm : MPI_COMM_SELF);
        auto context = CommunicationContext::Duplicate(laneMPI.comm);
        Array<double> src(RowLayout::Dense(1, 2)), result(src.Layout()), observed(src.Layout());
        auto sum = BindMPIAllreduce(context, src, result, Reduction::Sum);
        auto maximum = BindMPIAllreduce(context, result, result, Reduction::Max);
        auto gate = std::make_shared<std::atomic<bool>>(!workers || mpi.rank != 0);
        SubmitArrays(engine, compute, [gate, rank = mpi.rank](ExecutionContext, auto values)
                     {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!gate->load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            values(0, 0) = gate->load() ? rank + 1 : -1000;
            values(0, 1) = rank * 2; }, Write(src));
        communication.WaitFor(compute.Record());
        sum->Submit(engine, communication);
        const auto done = maximum->Submit(engine, communication);
        compute.WaitFor(done);
        SubmitArrays(engine, compute, [](ExecutionContext, auto in, auto out)
                     {
            out(0, 0) = 2 * in(0, 0);
            out(0, 1) = 2 * in(0, 1); }, Read(result), Write(observed));
        // Other host collectives use the caller's original communicator.
        // Rank zero may still be waiting for its producer on the queued lane.
        CHECK(DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Sum>(mpi.rank + 1, mpi.comm) == mpi.size * (mpi.size + 1) / 2);
        *gate = true;
        sum.reset();
        maximum.reset();
        context.reset();
        compute.Record().WaitHost();
        const auto values = observed.Read();
        CHECK(values.At(0, 0) == (workers ? mpi.size * (mpi.size + 1) : 2 * (mpi.rank + 1)));
        CHECK(values.At(0, 1) == (workers ? 2 * mpi.size * (mpi.size - 1) : 4 * mpi.rank));
    }
}

TEST_CASE("Direct transfers support SELF and reordered split communicators without inherited tag attributes")
{
    int worldRank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm split = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank % 2, -worldRank, &split);
    for (auto communicator : {MPI_COMM_SELF, split})
    {
        int rank = -1, size = 0;
        MPI_Comm_rank(communicator, &rank);
        MPI_Comm_size(communicator, &size);
        std::vector<int> worldRanks(size_t(size), -1);
        MPI_Allgather(&worldRank, 1, MPI_INT, worldRanks.data(), 1, MPI_INT, communicator);
        const int peer = (rank + 1) % size;
        auto context = std::make_shared<CommunicationContext>(communicator);
        CHECK_THROWS(context->Reserve(-1));
        Array<int64_t> source(RowLayout::Dense(3, 5, 8));
        {
            auto values = source.Write();
            for (size_t row = 0; row < 3; ++row)
                for (size_t column = 0; column < 5; ++column)
                    values.At(row, column) = 1000 * worldRank + 10 * row + column;
        }
        RowTransferPlan route(context, 3, 2, {{peer, 2, 0}, {rank, 0, 1}});
        for (auto mode : {TransferMode::Persistent, TransferMode::Transient})
        {
            Array<int64_t> destination(RowLayout::Dense(2, 5, 7));
            auto binding = PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>())
                               .Bind(source, destination, TransferTransport::Direct, {mode, context, 32767});
            binding.Begin().Wait();
            auto values = destination.Read();
            for (size_t column = 0; column < 5; ++column)
            {
                CHECK(values.At(0, column) == 1000 * worldRanks[size_t(peer)] + 20 + column);
                CHECK(values.At(1, column) == 1000 * worldRank + column);
            }
        }
    }
    MPI_Comm_free(&split);
}

TEST_CASE("Transient reduction rejects missing collision resolution before publishing")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    Array<int64_t> source(RowLayout::Dense(3, 7, 8));
    {
        auto w = source.Write();
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 7; ++j)
                w.At(i, j) = rank * 10 + i + j;
    }
    RowTransferPlan route(context, 3, 1, {{rank, 0, 0}, {(rank + 1) % size, 2, 0}});
    CHECK_THROWS(DynamicRowTransfer(route, source.SharedStorage(), ElementSchema::Arithmetic<int64_t>(), Device::Host(),
                                    TransferTransport::Direct, {TransferMode::Transient, context, 61}));
    for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
    {
        DynamicRowTransfer op(route, source.SharedStorage(), ElementSchema::Arithmetic<int64_t>(), Device::Host(),
                              transport, {TransferMode::Transient, context, 61}, Reducer::Builtin<int64_t>(Reduction::Sum, DestinationInit::Identity));
        auto result = Array<int64_t>::FromSharedStorage(op.Commit());
        auto read = result.Read();
        for (size_t j = 0; j < 7; ++j)
            CHECK(read.At(0, j) == rank * 10 + (rank + 1) % size * 10 + 2 + 2 * j);
    }
    // A reducer failure after MPI posting is tested in a bounded subprocess.
}

TEST_CASE("Transfer completion imported into eager progress orders its consumer")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    constexpr size_t rows = 47;
    Array<double> source(RowLayout::Dense(rows, 6, 8));
    {
        auto write = source.Write();
        for (DNDS::index iRow = 0; iRow < DNDS::index(rows); ++iRow)
            for (DNDS::index j = 0; j < 6; ++j)
                write.View()(iRow, j) = 100 * rank + 10 * iRow + j;
    }
    std::vector<RowRequest> requests;
    for (size_t i = 0; i < rows; ++i)
    {
        requests.push_back({rank, int64_t(i), i});
        requests.push_back({(rank + 1) % size, int64_t(rows - 1 - i), i});
    }
    RowTransferPlan route(context, rows, rows, requests);
    for (auto mode : {TransferMode::Persistent, TransferMode::Transient})
        for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
            for (bool scheduled : {false, true})
            {
                Array<double> destination(RowLayout::Dense(rows, 6, 9));
                int calls = 0;
                auto reducer = Reducer::Custom<double>(ElementSchema::Arithmetic<double>(), "audit.completion", [&calls](double &a, const double &b)
                                                       {
                        ++calls;
                        a += b; }, 0.0);
                auto binding = PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<double>())
                                   .Bind(source, destination, reducer, transport, {mode, context, 81});
                auto operation = scheduled ? binding.BeginScheduled() : binding.Begin();
                MPIProgress progress;
                auto done = progress.Import(std::move(operation));
                EagerEngine engine(scheduled ? 2 : 0);
                auto consumer = engine.CreateCPUStream();
                bool ran = false;
                consumer.WaitFor(done);
                engine.Submit(consumer, [&ran](ExecutionContext)
                              { ran = true; });
                progress.Drain();
                done.WaitHost();
                CHECK(done.Ready());
                engine.Synchronize();
                CHECK(ran);
                CHECK(calls == int(2 * rows * 6));
                const auto read = destination.Read();
                for (DNDS::index iRow = 0; iRow < DNDS::index(rows); ++iRow)
                    for (DNDS::index j = 0; j < 6; ++j)
                        CHECK(read.View()(iRow, j) == 100 * (rank + (rank + 1) % size) + 10 * (rows - 1) + 2 * j);
            }
}

#ifdef DNDS_USE_CUDA
TEST_CASE("CUDA eager producer portable packed MPI and consumer retain ordering")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked, TransferTransport::DevicePacked})
    {
        EagerEngine engine(2, std::make_shared<AccessTrackingContext>());
        auto stream = engine.CreateStream(std::make_shared<CUDALauncher>(0));
        Array<double> host(RowLayout::Dense(67, 6, 8));
        {
            auto w = host.Write();
            for (size_t i = 0; i < 67; ++i)
                for (size_t j = 0; j < 6; ++j)
                    w.At(i, j) = rank * 10000 + i * 10 + j;
        }
        auto source = transport == TransferTransport::Direct ? Array<double>(host.Layout(), Device::CUDA(0)) : AllocateArrayAsync<double>(engine, stream, host.Layout());
        auto destination = transport == TransferTransport::Direct ? Array<double>(host.Layout(), Device::CUDA(0)) : AllocateLikeAsync(engine, stream, source);
        auto sourceBuffer = source.Storage();
        CopyAsync(engine, stream, host.Storage(), sourceBuffer);
        std::vector<RowRequest> requests;
        for (size_t i = 0; i < 67; ++i)
            requests.push_back({(rank + 1) % size, int64_t(66 - i), i});
        RowTransferPlan route(MPI_COMM_WORLD, 67, 67, requests);
        auto binding = PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<double>())
                           .Bind(source, destination, transport, {TransferMode::Transient, {}, 0});
        MPIProgress progress;
        auto event = progress.Import(binding.BeginChecked(stream));
        event.WaitHost();
        stream.WaitFor(event);
        Array<double> output(host.Layout());
        auto outputBuffer = output.Storage();
        CopyAsync(engine, stream, destination.Storage(), outputBuffer);
        stream.Record().WaitHost();
        progress.Drain();
        engine.Synchronize();
        auto read = output.Read();
        for (size_t i = 0; i < 67; ++i)
            for (size_t j = 0; j < 6; ++j)
                CHECK(read.At(i, j) == (rank + 1) % size * 10000 + (66 - i) * 10 + j);
    }
    EagerEngine engine(0, std::make_shared<AccessTrackingContext>());
    auto stream = engine.CreateStream(std::make_shared<CUDALauncher>(0));
    auto pooled = AllocateArrayAsync<double>(engine, stream, RowLayout::Dense(3, 5));
    Array<double> target(pooled.Layout(), Device::CUDA(0));
    PayloadPlan payload(RowTransferPlan(MPI_COMM_WORLD, 3, 3, {{(rank + 1) % size, 1, 0}}), pooled.Layout(), target.Layout(), ElementSchema::For<double>());
    CHECK_THROWS(payload.Bind(pooled, target, TransferTransport::Direct));
    engine.Synchronize();
}
#endif

TEST_CASE("Transient and persistent contexts transfer permuted SoA rows")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size, previous = (rank + size - 1) % size;
    constexpr size_t rows = 193;
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    std::vector<std::vector<DNDS::index>> send(size), receive(size);
    for (DNDS::index i = 0; i < DNDS::index(rows); ++i)
    {
        send[previous].push_back(i * 17 % DNDS::index(rows));
        receive[next].push_back(i);
    }
    auto route = RowTransferPlan::Known(context, rows, rows, send, receive);
    std::vector<size_t> lengths(rows);
    for (size_t i = 0; i < rows; ++i)
        lengths[i] = i % 8;
    Array<double> source(RowLayout::Padded(lengths, 11));
    Array<int64_t> labels(RowLayout::Dense(rows, 2, 3));
    {
        auto x = source.Write();
        auto ids = labels.Write();
        for (size_t i = 0; i < rows; ++i)
        {
            ids.At(i, 0) = rank * 10000 + i;
            ids.At(i, 1) = -int64_t(i);
            for (size_t j = 0; j < lengths[i]; ++j)
                x.At(i, j) = rank * 10000 + i * 10 + j;
        }
    }
    Array<double> target(route.DestinationLayout(source.Layout()));
    Array<int64_t> targetIDs(RowLayout::Dense(rows, 2, 4));
    PayloadPlan values(route, source.Layout(), target.Layout(), ElementSchema::Arithmetic<double>());
    PayloadPlan ids(route, labels.Layout(), targetIDs.Layout(), ElementSchema::Arithmetic<int64_t>());
    for (auto mode : {TransferMode::Transient, TransferMode::Persistent})
        for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
        {
            auto first = values.Bind(source, target, transport, {mode, context, 37});
            auto second = ids.Bind(labels, targetIDs, transport, {mode, context, 38});
            CHECK_THROWS(values.Bind(source, target, transport, {mode, context, 37}));
            for (int repeat = 0; repeat < 3; ++repeat)
            {
                auto a = first.Begin();
                auto b = second.Begin();
                CHECK_THROWS(first.Begin());
                b.Wait();
                a.Wait();
                auto y = target.Read();
                auto indices = targetIDs.Read();
                for (size_t i = 0; i < rows; ++i)
                {
                    const auto from = i * 17 % rows;
                    CHECK(target.RowSize(i) == lengths[from]);
                    CHECK(indices.At(i, 0) == next * 10000 + from);
                    CHECK(indices.At(i, 1) == -int64_t(from));
                    for (size_t j = 0; j < lengths[from]; ++j)
                        CHECK(y.At(i, j) == next * 10000 + from * 10 + j);
                }
            }
        }
    auto badSend = send;
    if (rank == 0)
        badSend[previous].pop_back();
    CHECK_THROWS(RowTransferPlan::Known(context, rows, rows, badSend, receive));
}

TEST_CASE("Dynamic CSR migration privately prepares shapes and publishes once")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int peer = (rank + 1) % size;
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    for (size_t epoch = 0; epoch < 4; ++epoch)
    {
        const size_t rows = 67 + epoch * 2;
        std::vector<size_t> widths(rows);
        for (size_t i = 0; i < rows; ++i)
            widths[i] = (i + rank + epoch) % 9;
        Array<int64_t> source(RowLayout::Ragged(widths));
        {
            auto w = source.Write();
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < widths[i]; ++j)
                    w.At(i, j) = rank * 100000 + i * 100 + j;
        }
        std::vector<RowRequest> requests;
        for (size_t i = 0; i < rows; ++i)
            requests.push_back({peer, int64_t(rows - i - 1), i});
        RowTransferPlan route(context, rows, rows + 1, requests);
        DynamicRowTransfer operation(route, source.SharedStorage(), ElementSchema::Arithmetic<int64_t>(), Device::Host(),
                                     TransferTransport::Direct, {TransferMode::Transient, context, 51});
        CHECK_THROWS(source.Write());
        source.Reset(RowLayout::Dense(1, 1)); // Operation retains the old snapshot.
        auto destination = Array<int64_t>::FromSharedStorage(operation.Commit());
        CHECK_THROWS(operation.Commit());
        REQUIRE(destination.RowSize(rows) == 0);
        auto read = destination.Read();
        for (size_t i = 0; i < rows; ++i)
        {
            const auto from = rows - i - 1;
            REQUIRE(destination.RowSize(i) == (from + peer + epoch) % 9);
            for (size_t j = 0; j < destination.RowSize(i); ++j)
                CHECK(read.At(i, j) == peer * 100000 + from * 100 + j);
        }
    }
}

TEST_CASE("Dynamic migration preserves a strided table instead of publishing temporary CSR")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Array<int64_t> source(RowLayout::Dense(3, 2, 5 + rank));
    {
        auto write = source.Write();
        for (DNDS::index iRow = 0; iRow < 3; ++iRow)
            for (DNDS::index j = 0; j < 2; ++j)
                write.View()(iRow, j) = 100 * rank + 10 * iRow + j;
    }
    RowTransferPlan route(MPI_COMM_WORLD, 3, 3, {{(rank + 1) % size, 2, 0}, {(rank + 1) % size, 1, 1}, {(rank + 1) % size, 0, 2}});
    DynamicRowTransfer transfer(route, source.SharedStorage(), ElementSchema::Arithmetic<int64_t>());
    auto result = Array<int64_t>::FromSharedStorage(transfer.Commit());
    CHECK(result.Layout() == RowLayout::Dense(3, 2, 5 + rank));
    auto read = result.Read();
    for (DNDS::index iRow = 0; iRow < 3; ++iRow)
        for (DNDS::index j = 0; j < 2; ++j)
            CHECK(read.View<FixedRows<2>>()(iRow, j) == 100 * ((rank + 1) % size) + 10 * (2 - iRow) + j);
#ifdef DNDS_USE_CUDA
    if (DNDS::MPI::isCudaAware())
    {
        const auto context = ExecutionContext::CUDA(0);
        auto deviceSource = source.To(Device::CUDA(0), context);
        DynamicRowTransfer deviceTransfer(route, deviceSource.SharedStorage(), ElementSchema::Arithmetic<int64_t>(),
                                          Device::CUDA(0), TransferTransport::Direct,
                                          {TransferMode::Transient, {}, 0}, {}, context);
        auto deviceResult = Array<int64_t>::FromSharedStorage(deviceTransfer.Commit());
        CHECK(deviceResult.GetDevice() == Device::CUDA(0));
        CHECK(deviceResult.Layout() == source.Layout());
        auto hostResult = deviceResult.To(Device::Host(), context);
        auto hostRead = hostResult.Read();
        for (DNDS::index iRow = 0; iRow < 3; ++iRow)
            for (DNDS::index j = 0; j < 2; ++j)
                CHECK(hostRead.View<FixedRows<2>>()(iRow, j) == read.View<FixedRows<2>>()(iRow, j));
    }
    else
        MESSAGE("Direct device MPI unavailable; preserved-layout GPU migration is not exercised");
#endif
}

TEST_CASE("Dynamic migration rejects a missing source collectively and leaves the route usable")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Array<int64_t> source(RowLayout::Dense(1, 2));
    source.Write().At(0, 0) = 100 + rank;
    source.Write().At(0, 1) = 200 + rank;
    RowTransferPlan route(MPI_COMM_WORLD, 1, 1, {{(rank + 1) % size, 0, 0}});
    const auto schema = ElementSchema::Arithmetic<int64_t>();
    auto storage = rank == 0 ? ArrayStorage::Shared{} : source.SharedStorage();
    CHECK_THROWS_WITH(DynamicRowTransfer(route, storage, schema),
                      doctest::Contains("dynamic transfer source reservation failed on a rank"));
    CHECK(source.Storage().IsIdle());
    CHECK(source.Read().At(0, 0) == 100 + rank);
    DynamicRowTransfer retry(route, source.SharedStorage(), schema);
    auto result = Array<int64_t>::FromSharedStorage(retry.Commit());
    CHECK(result.Read().At(0, 0) == 100 + (rank + 1) % size);
    CHECK(result.Read().At(0, 1) == 200 + (rank + 1) % size);
}

TEST_CASE("Main-thread MPI progression joins worker computation with scheduled leases")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    constexpr size_t rows = 257;
    for (size_t workers : {size_t(0), size_t(2)})
        for (bool checking : {false, true})
        {
            EagerEngine engine(workers, checking ? std::make_shared<AccessTrackingContext>() : nullptr);
            auto core = engine.CreateCPUStream(), ring = engine.CreateCPUStream(), bridge = engine.CreateCPUStream();
            Array<double> owned(RowLayout::Dense(rows, 6, 8)), ghost(RowLayout::Dense(19, 6, 8));
            Array<double> output(RowLayout::Dense(rows, 6));
            {
                auto w = owned.Write();
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < 6; ++j)
                        w.At(i, j) = rank * 10000 + i * 10 + j;
            }
            std::vector<RowRequest> requests;
            for (size_t i = 0; i < 19; ++i)
                requests.push_back({(rank + 1) % size, int64_t(i * 13), i});
            RowTransferPlan route(MPI_COMM_WORLD, rows, 19, requests);
            auto binding = PayloadPlan(route, owned.Layout(), ghost.Layout(), ElementSchema::Arithmetic<double>())
                               .Bind(owned, ghost, TransferTransport::Direct, {TransferMode::Transient, {}, 0});
            MPIProgress progress;
            auto event = progress.Import(checking ? binding.BeginChecked(bridge) : binding.BeginScheduled());
            CHECK_THROWS(ghost.Read());
            SubmitArrays(engine, core, [](ExecutionContext, auto u, auto rhs)
                         { for (size_t i = 0; i < rows; ++i) for (size_t j = 0; j < 6; ++j) rhs(i, j) = 2 * u(i, j); }, Read(owned), Write(output));
            ring.WaitFor(event);
            ring.WaitFor(core.Record());
            SubmitArrays(engine, ring, [](ExecutionContext, auto g, auto rhs)
                         { for (size_t i = 0; i < 19; ++i) for (size_t j = 0; j < 6; ++j) rhs(i, j) += g(i, j); }, Read(ghost), ReadWrite(output));
            progress.Drain();
            engine.Synchronize();
            auto r = output.Read();
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < 6; ++j)
                    CHECK(r.At(i, j) == 2 * (rank * 10000 + i * 10 + j) + (i < 19 ? (rank + 1) % size * 10000 + i * 130 + j : 0));
        }
}

TEST_CASE("Checked MPI rejects a rank-local missing reader edge collectively and permits retry")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    EagerEngine engine(2, std::make_shared<AccessTrackingContext>());
    auto reader = engine.CreateCPUStream(), bridge = engine.CreateCPUStream();
    Array<double> source(RowLayout::Dense(6, 5)), ghost(RowLayout::Dense(1, 5));
    {
        auto write = source.Write();
        for (DNDS::index iRow = 0; iRow < 6; ++iRow)
            for (DNDS::index j = 0; j < 5; ++j)
                write.View()(iRow, j) = rank * 100 + iRow * 10 + j;
    }
    auto binding = PayloadPlan(RowTransferPlan(MPI_COMM_WORLD, 6, 1, {{(rank + 1) % size, 3, 0}}), source.Layout(), ghost.Layout(), ElementSchema::For<double>()).Bind(source, ghost);
    auto gate = std::make_shared<std::atomic<bool>>(false);
    struct Release
    {
        std::shared_ptr<std::atomic<bool>> gate;
        ~Release() { *gate = true; }
    } cleanup{gate};
    if (rank == 0)
        SubmitArrays(engine, reader, [gate](auto, auto)
                     {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!gate->load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield(); }, Read(ghost));
    CHECK_THROWS(binding.BeginChecked(bridge));
    *gate = true;
    engine.Synchronize();
    auto operation = binding.BeginChecked(bridge);
    operation.Wait();
    const auto values = ghost.Read();
    for (DNDS::index j = 0; j < 5; ++j)
        CHECK(values.View()(0, j) == ((rank + 1) % size) * 100 + 30 + j);
}

TEST_CASE("MPI completes while a core worker is active across Euler field widths")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int peer = (rank + 1) % size;
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    std::vector<RowRequest> requests;
    for (size_t i = 0; i < 19; ++i)
        requests.push_back({peer, int64_t(i * 13 % 67), i});
    RowTransferPlan route(context, 67, 19, requests);
    for (size_t workers : {size_t(1), size_t(2)})
        for (size_t width : {size_t(1), size_t(4), size_t(5), size_t(6), size_t(7), size_t(13), size_t(114)})
            for (auto mode : {TransferMode::Transient, TransferMode::Persistent})
                for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
                {
                    Array<double> owned(RowLayout::Dense(67, width, width + 3));
                    Array<double> ghost(RowLayout::Dense(19, width, width + 2));
                    Array<double> rhs(RowLayout::Dense(67, width, width + 1));
                    auto binding = PayloadPlan(route, owned.Layout(), ghost.Layout(), ElementSchema::Arithmetic<double>())
                                       .Bind(owned, ghost, transport, {mode, context, 91});
                    EagerEngine engine(workers);
                    auto core = engine.CreateCPUStream(), ring = engine.CreateCPUStream();
                    for (int epoch = 0; epoch < 3; ++epoch)
                    {
                        {
                            auto w = owned.Write();
                            for (size_t i = 0; i < 67; ++i)
                                for (size_t j = 0; j < width; ++j)
                                    w.At(i, j) = epoch * 1000000 + rank * 10000 + 10 * i + j;
                        }
                        MPIProgress progress;
                        auto received = progress.Import(binding.BeginScheduled());
                        std::atomic<bool> entered{false}, mpiDone{false}, exited{false};
                        SubmitArrays(engine, core, [&](ExecutionContext, auto u, auto r)
                                     {
                            entered = true;
                            for (size_t i = 0; i < r.RowCount(); ++i)
                                for (size_t j = 0; j < r.RowSize(i); ++j) r(i, j) = u(i, j) + u((17 * i + 3) % 67, j);
                            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                            while (!mpiDone.load())
                            {
                                DNDS_check_throw_info(std::chrono::steady_clock::now() < deadline, "MPI/core rendezvous timed out");
                                std::this_thread::yield();
                            }
                            exited = true; }, Read(owned), Write(rhs));
                        auto coreDone = core.Record();
                        ring.WaitFor(received);
                        ring.WaitFor(coreDone);
                        SubmitArrays(engine, ring, [](ExecutionContext, auto g, auto r)
                                     { for (size_t i = 0; i < g.RowCount(); ++i) for (size_t j = 0; j < g.RowSize(i); ++j) r(i, j) += g(i, j); }, Read(ghost), ReadWrite(rhs));
                        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                        while (!entered.load() && std::chrono::steady_clock::now() < deadline)
                            std::this_thread::yield();
                        CHECK(entered.load());
                        CHECK_FALSE(exited.load());
                        progress.Drain(); // Actual MPI requests finish while the core body is live.
                        CHECK(received.Ready());
                        CHECK_FALSE(coreDone.Ready());
                        mpiDone = true;
                        engine.Synchronize();
                        auto r = rhs.Read();
                        std::vector<double> norms(width, 0);
                        for (size_t i = 0; i < 67; ++i)
                            for (size_t j = 0; j < width; ++j)
                            {
                                const double local = 2 * (epoch * 1000000 + rank * 10000 + j) + 10 * (i + (17 * i + 3) % 67);
                                const double remote = i < 19 ? epoch * 1000000 + peer * 10000 + 10 * (i * 13 % 67) + j : 0;
                                CHECK(r.At(i, j) == local + remote);
                                norms[j] += r.At(i, j);
                            }
                        MPI_Allreduce(MPI_IN_PLACE, norms.data(), int(width), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                        size_t remoteRowSum = 0;
                        for (size_t i = 0; i < 19; ++i)
                            remoteRowSum += i * 13 % 67;
                        for (size_t j = 0; j < width; ++j)
                            CHECK(norms[j] == size * (153.0 * (epoch * 1000000 + j) + 44220 + 10 * remoteRowSum) +
                                                  153.0 * 10000 * size * (size - 1) / 2);
                    }
                }
}

TEST_CASE("Rank-local payload and lease mistakes fail collectively without changing values")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Array<int64_t> source(RowLayout::Dense(31, 7, 9)), destination(RowLayout::Dense(31, 7, 10));
    RowTransferPlan route(MPI_COMM_WORLD, 31, 31, {{(rank + 1) % size, 17, 3}});
    PayloadPlan payload(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>());
    Array<int64_t> wrong(RowLayout::Dense(31, 7, rank == 0 ? 11 : 10));
    CHECK_THROWS(payload.Bind(source, wrong));
    CHECK_THROWS(PayloadPlan(route, source.Layout(), RowLayout::Dense(31, rank == 0 ? 6 : 7), ElementSchema::Arithmetic<int64_t>()));
    if (size > 1)
    {
        CHECK_THROWS(PayloadPlan(route, source.Layout(), destination.Layout(),
                                 rank == 0 ? ElementSchema::Arithmetic<double>() : ElementSchema::Arithmetic<int64_t>()));
        CHECK_THROWS(payload.Bind(source, destination, Reducer::Builtin<int64_t>(rank == 0 ? Reduction::Sum : Reduction::Max)));
    }
    for (auto mode : {TransferMode::Transient, TransferMode::Persistent})
        for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
        {
            auto binding = payload.Bind(source, destination, transport, {mode, {}, 0});
            std::optional<ReadAccess> pinned;
            if (rank == 0)
                pinned.emplace(destination.Storage().Read());
            CHECK_THROWS(binding.Begin());
            CHECK_THROWS(binding.BeginScheduled());
            pinned.reset();
            CHECK(source.Storage().IsIdle());
            CHECK(destination.Storage().IsIdle());
            CHECK(destination.IsValid());
            {
                auto r = destination.Read();
                for (size_t i = 0; i < 31; ++i)
                    for (size_t j = 0; j < 7; ++j)
                        CHECK(r.At(i, j) == 0);
            }
            binding.Begin().Wait(); // A rejected start must leave the binding reusable.
        }
}

TEST_CASE("Repeated halo gathers and reverse residual assembly share routes across fields")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size, previous = (rank + size - 1) % size;
    constexpr size_t rows = 67, halo = 29, width = 7;
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    std::vector<RowRequest> requests;
    for (size_t i = 0; i < halo; ++i)
        requests.push_back({next, int64_t(i % 11), i}); // Duplicated source rows, disjoint ghost rows.
    RowTransferPlan route(context, rows, halo, requests);
    CHECK_FALSE(route.HasCollisions());
    CHECK(route.Reverse().HasCollisions());
    for (auto mode : {TransferMode::Persistent, TransferMode::Transient})
        for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
        {
            Array<double> u(RowLayout::Dense(rows, width, 10)), ghost(RowLayout::Dense(halo, width, 9));
            Array<int64_t> ids(RowLayout::Dense(rows, 1, 2)), ghostIds(RowLayout::Dense(halo, 1));
            Array<double> contributions(ghost.Layout()), assembled(u.Layout());
            auto pull = PayloadPlan(route, u.Layout(), ghost.Layout(), ElementSchema::Arithmetic<double>())
                            .Bind(u, ghost, transport, {mode, context, 101});
            auto pullIds = PayloadPlan(route, ids.Layout(), ghostIds.Layout(), ElementSchema::Arithmetic<int64_t>())
                               .Bind(ids, ghostIds, transport, {mode, context, 102});
            PayloadPlan reverse(route.Reverse(), contributions.Layout(), assembled.Layout(), ElementSchema::Arithmetic<double>());
            CHECK_THROWS(reverse.Bind(contributions, assembled));
            auto push = reverse.Bind(contributions, assembled, Reducer::Builtin<double>(Reduction::Sum), transport, {mode, context, 103});
            EagerEngine engine(2);
            auto compute = engine.CreateCPUStream();
            for (int epoch = 0; epoch < 4; ++epoch)
            {
                {
                    auto state = u.Write();
                    auto labels = ids.Write();
                    auto residual = assembled.Write();
                    for (size_t i = 0; i < rows; ++i)
                    {
                        labels.At(i, 0) = rank * 1000 + i;
                        for (size_t j = 0; j < width; ++j)
                        {
                            state.At(i, j) = epoch * 100000 + rank * 1000 + i * 10 + j;
                            residual.At(i, j) = -10 - double(j);
                        }
                    }
                }
                MPIProgress progress;
                auto stateReady = progress.Import(pull.BeginScheduled());
                auto idsReady = progress.Import(pullIds.BeginScheduled());
                compute.WaitFor(stateReady);
                compute.WaitFor(idsReady);
                SubmitArrays(engine, compute, [rank](ExecutionContext, auto g, auto labels, auto flux)
                             {
                    for (size_t i = 0; i < halo; ++i)
                        for (size_t j = 0; j < width; ++j)
                            flux(i, j) = g(i, j) + labels(i, 0) + rank + i; }, Read(ghost), Read(ghostIds), Write(contributions));
                progress.Drain();
                compute.Record().WaitHost(); // CPU producer must finish before MPI Begin.
                auto gather = push.Begin();
                CHECK_THROWS(assembled.Read());
                gather.Wait();
                auto r = assembled.Read();
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < width; ++j)
                    {
                        const size_t count = i < 11 ? 1 + (halo - 1 - i) / 11 : 0;
                        // Independent closed-form sum of indices i, i+11, ... .
                        const double expected = -10 - double(j) + count * (epoch * 100000 + rank * 2000 + 12 * i + j + previous) +
                                                11 * count * (count ? count - 1 : 0) / 2;
                        CHECK(r.At(i, j) == expected);
                    }
            }
            engine.Synchronize();
        }
}

TEST_CASE("Uneven empty-rank field bundles migrate out and back with epoch invalidation")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int peer = (rank + 1) % size;
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    VersionTracker meshVersion;
    auto count = [size](int owner, int epoch) -> size_t
    { return size > 1 && owner == 0 ? 0 : 37 + owner + epoch; };
    for (int epoch = 0; epoch < 4; ++epoch)
    {
        const size_t rows = count(rank, epoch), receivedRows = count(peer, epoch);
        Array<double> state(RowLayout::Dense(rows, 6, 8));
        Array<int64_t> labels(RowLayout::Dense(rows, 2, 3));
        RowBuilder builder(rows, sizeof(double));
        {
            auto u = state.Write();
            auto ids = labels.Write();
            for (size_t i = 0; i < rows; ++i)
            {
                ids.At(i, 0) = rank * 1000 + i;
                ids.At(i, 1) = epoch;
                for (size_t j = 0; j < 6; ++j)
                    u.At(i, j) = epoch * 100000 + rank * 1000 + 10 * i + j;
                builder.ResizeRow(i, (i + rank + epoch) % 5 * 19); // Empty and high-order rows.
                auto r = builder.Row<double>(i);
                for (size_t j = 0; j < r.size; ++j)
                    r.data[j] = epoch * 100000 + rank * 1000 + 100 * i + j;
            }
        }
        auto rec = Array<double>::FromSharedStorage(builder.Commit());
        std::vector<RowRequest> requests;
        for (size_t i = 0; i < receivedRows; ++i)
            requests.push_back({peer, int64_t(receivedRows - 1 - i), i});
        RowTransferPlan route(context, rows, receivedRows, requests, {meshVersion.Stamp(Change::Distribution)});
        DynamicRowTransfer moveU(route, state.SharedStorage(), ElementSchema::Arithmetic<double>(), Device::Host(), TransferTransport::Direct, {TransferMode::Transient, context, 111});
        DynamicRowTransfer moveIDs(route, labels.SharedStorage(), ElementSchema::Arithmetic<int64_t>(), Device::Host(), TransferTransport::HostPacked, {TransferMode::Transient, context, 112});
        DynamicRowTransfer moveRec(route, rec.SharedStorage(), ElementSchema::Arithmetic<double>(), Device::Host(), TransferTransport::Direct, {TransferMode::Transient, context, 113});
        auto oldState = state;
        state.Reset(RowLayout::Dense(1, 6)); // Replacement must not retarget the migration.
        auto newU = Array<double>::FromSharedStorage(moveU.Commit());
        auto newIDs = Array<int64_t>::FromSharedStorage(moveIDs.Commit());
        auto newRec = Array<double>::FromSharedStorage(moveRec.Commit());
        CHECK(state.Read().At(0, 0) == 0);
        {
            auto u = newU.Read();
            auto ids = newIDs.Read();
            auto r = newRec.Read();
            for (size_t i = 0; i < receivedRows; ++i)
            {
                const size_t from = receivedRows - 1 - i;
                CHECK(ids.At(i, 0) == peer * 1000 + from);
                CHECK(ids.At(i, 1) == epoch);
                CHECK(newRec.RowSize(i) == (from + peer + epoch) % 5 * 19);
                for (size_t j = 0; j < 6; ++j)
                    CHECK(u.At(i, j) == epoch * 100000 + peer * 1000 + 10 * from + j);
                for (size_t j = 0; j < newRec.RowSize(i); ++j)
                    CHECK(r.At(i, j) == epoch * 100000 + peer * 1000 + 100 * from + j);
            }
        }
        DynamicRowTransfer back(route.Reverse(), newRec.SharedStorage(), ElementSchema::Arithmetic<double>(), Device::Host(), TransferTransport::HostPacked, {TransferMode::Transient, context, 114});
        auto original = Array<double>::FromSharedStorage(back.Commit());
        auto read = original.Read();
        REQUIRE(original.RowCount() == rows);
        for (size_t i = 0; i < rows; ++i)
        {
            REQUIRE(original.RowSize(i) == (i + rank + epoch) % 5 * 19);
            for (size_t j = 0; j < original.RowSize(i); ++j)
                CHECK(read.At(i, j) == epoch * 100000 + rank * 1000 + 100 * i + j);
        }
        auto oldPayload = PayloadPlan(route, oldState.Layout(), newU.Layout(), ElementSchema::Arithmetic<double>()).Bind(oldState, newU);
        // Distribution changes may happen only after old operations are drained.
        auto mutation = meshVersion.BeginMutation(Change::Distribution);
        mutation.Commit();
        CHECK_THROWS(oldPayload.Begin());
    }
}
