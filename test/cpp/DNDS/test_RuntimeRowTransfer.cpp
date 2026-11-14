#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Communication/CommunicationWorker.hpp"
#include "DNDS/Runtime/Communication/NCCLTransfer.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Communication/MPIProgress.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include <optional>
#include <atomic>
#include <string>
#include <chrono>

using namespace DNDS::Runtime;

namespace
{
    std::atomic<bool> countReductions{false};
    std::atomic<int> reductionNumber{0};
    struct CountReductions
    {
        CountReductions()
        {
            reductionNumber = 0;
            countReductions = true;
        }
        ~CountReductions() { countReductions = false; }
    };
}
// MPI's profiling interface counts real Runtime/library calls, rather than a
// test-only policy callback. Only this executable interposes the MPI symbol.
extern "C" int MPI_Allreduce(const void *send, void *receive, int count, MPI_Datatype type, MPI_Op operation, MPI_Comm comm)
{
    if (countReductions)
        ++reductionNumber;
    return PMPI_Allreduce(send, receive, count, type, operation, comm);
}

int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    if (argc == 2 && std::string(argv[1]) == "--preflight-off-failure")
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD, false);
        Array<double> source(RowLayout::Dense(1, 1)), destination(RowLayout::Dense(1, 1));
        RowTransferPlan route(context, 1, 1, {{rank, 0, 0}});
        auto binding = PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::For<double>()).Bind(source, destination);
        binding.BeginScheduled(ExecutionContext::Host(), rank != 0).Wait();
        return 0; // Reaching this line is a failed fail-fast contract.
    }
    if (argc == 2 && std::string(argv[1]) == "--test-abort-transfer-conflict")
    {
        int rank = 0, size = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        RowTransferPlan route(MPI_COMM_WORLD, 1, 1, {{(rank + 1) % size, 0, 0}});
        Array<int64_t> source(RowLayout::Dense(1, 5)), destination(RowLayout::Dense(1, 5));
        TransferOptions options;
        options.preflight = TransferPreflight::AbortOnFailure;
        auto binding = PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>()).Bind(source, destination, TransferTransport::Direct, options);
        // Only rank0 conflicts. Peers may already post requests: the failing
        // rank must abort MPI immediately, never throw into local recovery.
        std::optional<Array<int64_t>::WriteLease> write;
        if (!rank)
            write.emplace(source.Write());
        binding.Begin().Wait();
        MPI_Finalize();
        return 0; // CTest expects nonzero; reaching this point is a failure.
    }
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int result = ctx.run();
    DNDS::MPI::Finalize();
    return result;
}

TEST_CASE("Prepared CPU transfer groups retain separate payloads and perform one optional agreement")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int active = size > 2 ? size - 1 : size;
    const DNDS::index rows = rank < active ? 35 : 0, ghosts = rank < active ? 3 : 0;
    const int previous = (rank + active - 1) % active;
    const DNDS::index order[]{33, 0, 17};
    for (const bool checked : {true, false})
        for (const auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
        {
            CAPTURE(checked);
            CAPTURE(transport == TransferTransport::HostPacked);
            auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD, checked);
            std::vector<RowRequest> requests;
            for (DNDS::index iRow = 0; iRow < ghosts; ++iRow)
                requests.push_back({previous, order[iRow], iRow});
            RowTransferPlan route(context, rows, ghosts, requests);
            std::vector<Array<double>> input, output;
            std::vector<TransferBinding> bindings;
            for (int field = 0; field < 3; ++field)
            {
                const auto layout = field == 0 ? RowLayout::Dense(rows, 5, 8) : field == 1 ? RowLayout::ColMajor(rows, 12, {8, 1})
                                                                                           : RowLayout::TiledColMajor(rows, 7, 8);
                input.emplace_back(layout);
                output.emplace_back(layout.WithRowCount(ghosts));
                PayloadPlan payload(route, input.back().Layout(), output.back().Layout(), ElementSchema::For<double>());
                bindings.push_back(payload.Bind(input.back(), output.back(), transport, {TransferMode::Persistent, context, 10 + field}));
            }
            TransferGroup group(context, bindings);
            for (int iteration = 0; iteration < 2; ++iteration)
            {
                for (int field = 0; field < 3; ++field)
                {
                    auto write = input[field].Write();
                    const auto values = write.View<UniformTableRows>();
                    for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                        for (DNDS::index j = 0; j < values.Width(); ++j)
                            values(iRow, j) = 10000 * rank + 1000 * field + 100000 * iteration + 10 * iRow + j;
                }
                {
                    CountReductions count;
                    auto operations = group.Begin();
                    REQUIRE(operations.size() == 3);
                    for (auto &operation : operations)
                        operation.Wait();
                }
                CHECK(reductionNumber == (checked ? 1 : 0));
                for (int field = 0; field < 3; ++field)
                {
                    auto read = output[field].Read();
                    const auto values = read.View<UniformTableRows>();
                    for (DNDS::index iRow = 0; iRow < ghosts; ++iRow)
                        for (DNDS::index j = 0; j < values.Width(); ++j)
                            CHECK(values(iRow, j) == 10000 * previous + 1000 * field + 100000 * iteration + 10 * order[iRow] + j);
                }
                // The same retained bindings still support ordinary starts.
                for (const auto &binding : bindings)
                    binding.Begin().Wait();
            }
        }
}

TEST_CASE("Grouped reservation rejection releases all fields before any payload is posted")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    RowTransferPlan route(context, 1, 1, {{(rank + size - 1) % size, 0, 0}});
    Array<double> a(RowLayout::Dense(1, 1)), b(RowLayout::Dense(1, 1));
    Array<double> x(RowLayout::Dense(1, 1)), y(RowLayout::Dense(1, 1));
    a.Write().View()(0, 0) = rank + 1;
    b.Write().View()(0, 0) = rank + 10;
    PayloadPlan payload(route, a.Layout(), x.Layout(), ElementSchema::For<double>());
    const auto first = payload.Bind(a, x, TransferTransport::Direct, {TransferMode::Persistent, context, 21});
    const auto second = payload.Bind(b, y, TransferTransport::Direct, {TransferMode::Persistent, context, 22});
    if (size > 1)
        CHECK_THROWS(TransferGroup(context, rank == 0 ? std::vector<TransferBinding>{second, first} : std::vector<TransferBinding>{first, second}));
    TransferGroup group(context, {first, second});
    const auto xVersion = x.Version(), yVersion = y.Version();
    {
        std::optional<Array<double>::WriteLease> busy;
        if (rank == 0)
            busy.emplace(b.Write());
        CHECK_THROWS(group.Begin());
        CHECK(x.Version() == xVersion);
        CHECK(y.Version() == yVersion);
    }
    // Both the earlier successful reservation and the failing member are usable
    // afterward. A hidden active request/lease would reject or hang this retry.
    auto operations = group.Begin();
    for (auto &operation : operations)
        operation.Wait();
    CHECK(x.Read().View()(0, 0) == (rank + size - 1) % size + 1);
    CHECK(y.Read().View()(0, 0) == (rank + size - 1) % size + 10);
}

TEST_CASE("Optional collective preflight removes only successful launch reductions")
{
    int worldRank = 0, worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    // np4 exercises independent subgroups; every nontrivial group is reversed.
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldSize >= 4 ? worldRank % 2 : 0, -worldRank, &comm);
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const int previous = (rank + size - 1) % size;
    const bool sourceEmpty = size > 1 && rank == size - 1;
    const bool donorEmpty = size > 1 && previous == size - 1;
    const DNDS::index sourceRows = sourceEmpty ? 0 : 4, ghostRows = donorEmpty ? 0 : 3;
    for (const bool enabled : {true, false})
    {
        CAPTURE(enabled);
        auto context = std::make_shared<CommunicationContext>(comm, enabled);
        CHECK(context->ChecksCollectivePreflight() == enabled);
        std::vector<RowRequest> requests;
        const DNDS::index order[]{3, 1, 2};
        for (DNDS::index iRow = 0; iRow < ghostRows; ++iRow)
            requests.push_back({previous, order[iRow], iRow});
        RowTransferPlan route(context, sourceRows, ghostRows, requests);
        for (int layout = 0; layout < 3; ++layout)
        {
            CAPTURE(layout);
            std::vector<size_t> sourceSizes = sourceEmpty ? std::vector<size_t>{} : std::vector<size_t>{2, 0, 3, 1};
            std::vector<size_t> ghostSizes = donorEmpty ? std::vector<size_t>{} : std::vector<size_t>{1, 0, 3};
            const auto sourceShape = layout == 0 ? RowLayout::Dense(sourceRows, 3, 5) : layout == 1 ? RowLayout::Padded(sourceSizes, 5)
                                                                                                    : RowLayout::Ragged(sourceSizes);
            const auto ghostShape = layout == 0 ? RowLayout::Dense(ghostRows, 3, 7) : layout == 1 ? RowLayout::Padded(ghostSizes, 7)
                                                                                                  : RowLayout::Ragged(ghostSizes);
            Array<int64_t> source(sourceShape), destination(ghostShape);
            {
                auto write = source.Write();
                for (DNDS::index iRow = 0; iRow < sourceRows; ++iRow)
                    for (DNDS::index j = 0; j < sourceShape.RowSize(iRow); ++j)
                        write.View()(iRow, j) = rank * 1000 + iRow * 10 + j;
            }
            std::vector<int64_t> expected(ghostShape.Elements(), -77);
            DNDS::index csrOffset = 0;
            for (DNDS::index iRow = 0; iRow < ghostRows; ++iRow)
            {
                const DNDS::index width = layout == 0 ? 3 : DNDS::index(ghostSizes[size_t(iRow)]);
                for (DNDS::index j = 0; j < width; ++j)
                    expected[size_t((layout == 2 ? csrOffset : iRow * 7) + j)] = previous * 1000 + order[iRow] * 10 + int64_t(j);
                csrOffset += width;
            }
            const PayloadPlan payload(route, sourceShape, ghostShape, ElementSchema::For<int64_t>());
            for (const auto mode : {TransferMode::Transient, TransferMode::Persistent})
            {
                CAPTURE(mode == TransferMode::Persistent);
                auto binding = payload.Bind(source, destination, TransferTransport::Direct, {mode, context, 23});
                EagerEngine engine(0, std::make_shared<AccessTrackingContext>());
                const auto stream = engine.CreateCPUStream();
                for (int kind = 0; kind < 3; ++kind)
                {
                    CAPTURE(kind);
                    {
                        auto write = destination.Storage().Write();
                        std::fill_n(write.View<int64_t>().data, expected.size(), int64_t(-77));
                    }
                    {
                        CountReductions counter;
                        auto operation = kind == 0 ? binding.Begin() : kind == 1 ? binding.BeginScheduled()
                                                                                 : binding.BeginChecked(stream);
                        operation.Wait();
                    }
                    CHECK(reductionNumber == (enabled ? (kind == 2 ? 3 : 1) : 0));
                    auto read = destination.Storage().Read();
                    for (DNDS::index iElement = 0; iElement < DNDS::index(expected.size()); ++iElement)
                        CHECK(read.View<int64_t>()[iElement] == expected[iElement]);
                }
            }
        }
    }
    MPI_Comm_free(&comm);
}

TEST_CASE("Managed MPI requests survive dropped handles and progress worker consumers during close")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (const bool checked : {false, true})
    {
        EagerEngine engine(2, checked ? std::make_shared<AccessTrackingContext>() : nullptr);
        auto stream = engine.CreateCPUStream();
        auto communication = engine.CreateCPUStream();
        Array<double> source(RowLayout::Dense(17, 5)), destination(source.Layout()), output(source.Layout());
        {
            auto write = source.Write();
            for (DNDS::index iRow = 0; iRow < 17; ++iRow)
                for (DNDS::index j = 0; j < 5; ++j)
                    write.View()(iRow, j) = 1000 * rank + 10 * iRow + j;
        }
        std::vector<RowRequest> requests;
        for (DNDS::index iRow = 0; iRow < 17; ++iRow)
            requests.push_back({(rank + 1) % size, 16 - iRow, iRow});
        auto binding = std::make_unique<TransferBinding>(PayloadPlan(RowTransferPlan(MPI_COMM_WORLD, 17, 17, requests), source.Layout(), destination.Layout(),
                                                                     ElementSchema::For<double>())
                                                             .Bind(source, destination));
        // No operation handle/event remains; Close must still inventory it.
        {
            auto operation = checked ? binding->BeginChecked(stream) : binding->BeginScheduled(stream);
        }
        engine.Synchronize();
        CHECK(destination.Read().At(0, 0) == 1000 * ((rank + 1) % size) + 160);
        auto operation = binding->SubmitReady(engine, communication);
        CHECK(operation.RequiresHostProgress());
        auto event = operation.CompletionEvent();
        std::atomic<bool> core{false};
        engine.Submit(stream, [&](ExecutionContext)
                      { core = true; });
        operation.EnqueueWait(stream);
        operation = {}; // Runtime inventory and event retain the actual transfer.
        SubmitArrays(engine, stream, [](ExecutionContext, auto input, auto result)
                     {
            for (DNDS::index iRow = 0; iRow < 17; ++iRow)
                for (DNDS::index j = 0; j < 5; ++j) result(iRow, j) = 2 * input(iRow, j); }, Read(destination), Write(output));
        source = {};
        destination = {};
        binding.reset(); // Only runtime/operation scopes retain the payload owners.
        engine.Close();  // No bridge.Poll/Drain; registered operation drives the event.
        CHECK(core.load());
        CHECK(event.Ready());
        const auto read = output.Read();
        for (DNDS::index iRow = 0; iRow < 17; ++iRow)
            for (DNDS::index j = 0; j < 5; ++j)
                CHECK(read.View()(iRow, j) == 2 * (1000 * ((rank + 1) % size) + 10 * (16 - iRow) + j));
    }
}

TEST_CASE("Inherited named channels share a communicator and preserve concurrent payloads")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    auto unevenReservation = context->Reserve(1 + rank % 2);
    RowTransferPlan route(context, 3, 3, {{(rank + 1) % size, 2, 0}, {(rank + 1) % size, 0, 1}, {(rank + 1) % size, 1, 2}});
    Array<double> a(RowLayout::Dense(3, 5)), b(a.Layout()), x(a.Layout()), y(a.Layout());
    {
        auto aw = a.Write(), bw = b.Write();
        for (DNDS::index iRow = 0; iRow < 3; ++iRow)
            for (DNDS::index j = 0; j < 5; ++j)
            {
                aw.View()(iRow, j) = 1000 * rank + 10 * iRow + j;
                bw.View()(iRow, j) = -aw.View()(iRow, j) - 7;
            }
    }
    PayloadPlan payload(route, a.Layout(), x.Layout(), ElementSchema::For<double>());
    TransferOptions first, second;
    first.channel = "mean";
    second.channel = "coefficients";
    auto left = payload.Bind(a, x, TransferTransport::Direct, first);
    auto right = payload.Bind(b, y, TransferTransport::Direct, second);
    // Group compatibility proves both default bindings use this exact context.
    TransferGroup group(context, {left, right});
    auto operations = group.Begin();
    for (auto &operation : operations)
        operation.Wait();
    const auto xr = x.Read(), yr = y.Read();
    const DNDS::index rows[]{2, 0, 1};
    for (DNDS::index iRow = 0; iRow < 3; ++iRow)
        for (DNDS::index j = 0; j < 5; ++j)
        {
            CHECK(xr.View()(iRow, j) == 1000 * ((rank + 1) % size) + 10 * rows[iRow] + j);
            CHECK(yr.View()(iRow, j) == -xr.View()(iRow, j) - 7);
        }
    TransferOptions isolated;
    isolated.isolatedNamespace = true;
    auto independent = payload.Bind(a, y, TransferTransport::Direct, isolated);
    CHECK_THROWS(TransferGroup(context, {independent}));
}
TEST_CASE("Communication context adoption and policy aliases retain one resource and tag namespace")
{
    MPI_Comm raw = MPI_COMM_NULL;
    MPI_Comm_dup(MPI_COMM_WORLD, &raw);
    const MPI_Comm adopted = raw;
    auto owner = CommunicationContext::Adopt(raw);
    CHECK(raw == MPI_COMM_NULL);
    auto alias = owner->WithPreflight(false);
    CHECK(owner->SharesResource(*alias));
    CHECK(owner->ChecksCollectivePreflight());
    CHECK_FALSE(alias->ChecksCollectivePreflight());
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(adopted, alias->Communicator(), &comparison);
    CHECK(comparison == MPI_IDENT);
    auto reservation = owner->Reserve(71);
    CHECK_THROWS(alias->Reserve(71));
    owner.reset();
    int size = 0, sum = 0, one = 1;
    MPI_Comm_size(alias->Communicator(), &size);
    MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, alias->Communicator());
    CHECK(sum == size);
    reservation.reset();
    CHECK_NOTHROW(alias->Reserve(71));
}

TEST_CASE("Communication context agrees its immutable safety policy before preparing routes")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size > 1)
    {
        CHECK_THROWS(std::make_shared<CommunicationContext>(MPI_COMM_WORLD, rank != 0));
        const auto checked = std::make_shared<CommunicationContext>(MPI_COMM_WORLD, true);
        const auto unchecked = std::make_shared<CommunicationContext>(MPI_COMM_WORLD, false);
        RowTransferPlan route(checked, 1, 1, {{rank, 0, 0}});
        Array<int64_t> source(RowLayout::Dense(1, 1)), destination(RowLayout::Dense(1, 1));
        PayloadPlan payload(route, source.Layout(), destination.Layout(), ElementSchema::For<int64_t>());
        // Each context is coherent, but different ranks choose different modes.
        // The existing binding-control agreement must reject that before posting.
        CHECK_THROWS(payload.Bind(source, destination, TransferTransport::Direct,
                                  {TransferMode::Persistent, rank == 0 ? checked : unchecked, 11}));
        payload.Bind(source, destination, TransferTransport::Direct, {TransferMode::Persistent, checked, 11}).Begin().Wait();
    }
}

TEST_CASE("Fail-fast MPI launches preserve padded rows in eager and checked execution")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    RowTransferPlan route(MPI_COMM_WORLD, 3, 2, {{(rank + 1) % size, 2, 0}, {rank, 0, 1}});
    Array<int64_t> source(RowLayout::Dense(3, 5, 7)), destination(RowLayout::Dense(2, 5, 8));
    CHECK(TransferOptions{}.preflight == TransferPreflight::Inherit);
    EagerEngine engine(0, std::make_shared<AccessTrackingContext>());
    const auto stream = engine.CreateCPUStream();
    for (auto mode : {TransferMode::Persistent, TransferMode::Transient})
    {
        TransferOptions options;
        options.mode = mode;
        options.preflight = TransferPreflight::AbortOnFailure;
        auto binding = PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>()).Bind(source, destination, TransferTransport::Direct, options);
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            {
                auto write = source.Write();
                for (DNDS::index iRow = 0; iRow < 3; ++iRow)
                    for (DNDS::index j = 0; j < 5; ++j)
                        write.At(iRow, j) = 1000 * iteration + 100 * rank + 10 * iRow + j;
            }
            auto operation = iteration == 0 ? binding.Begin() : iteration == 1 ? binding.BeginScheduled(stream.Context())
                                                                               : binding.BeginChecked(stream);
            operation.Wait();
            const auto read = destination.Read();
            for (DNDS::index j = 0; j < 5; ++j)
            {
                CHECK(read.At(0, j) == 1000 * iteration + 100 * ((rank + 1) % size) + 20 + j);
                CHECK(read.At(1, j) == 1000 * iteration + 100 * rank + j);
            }
        }
    }
    // A disagreement must still fail collectively at binding construction,
    // before either rank can enter a different launch protocol.
    if (size > 1)
    {
        TransferOptions options;
        if (!rank)
            options.preflight = TransferPreflight::AbortOnFailure;
        CHECK_THROWS(PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>()).Bind(source, destination, TransferTransport::Direct, options));
    }
}

TEST_CASE("Large permuted routes preserve row order padding and rank-local collision flags")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    constexpr DNDS::index rows = 8193, width = 3, step = 4099;
    const int previous = (rank + size - 1) % size, next = (rank + 1) % size;
    const auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    Array<int64_t> source(RowLayout::Dense(rows, width, 5)), destination(RowLayout::Dense(rows, width, 6));
    {
        auto write = source.Storage().Write();
        std::fill_n(write.View<int64_t>().data, rows * 5, int64_t(-999));
        for (DNDS::index iRow = 0; iRow < rows; ++iRow)
            for (DNDS::index iCol = 0; iCol < width; ++iCol)
                write.View<int64_t>()[iRow * 5 + iCol] = rank * 1000000 + iRow * 16 + iCol;
    }
    for (bool duplicate : {false, true})
    {
        std::vector<std::vector<DNDS::index>> send(size), receive(size);
        for (DNDS::index iRow = 0; iRow < rows; ++iRow)
        {
            send[next].push_back(iRow * step % rows);
            receive[previous].push_back(iRow);
        }
        if (duplicate && rank == 0)
            send[next].back() = send[next].front();
        auto route = RowTransferPlan::Known(context, rows, rows, send, receive);
        CHECK_FALSE(route.HasCollisions());
        CHECK(route.Reverse().HasCollisions() == (duplicate && rank == 0));
        {
            auto write = destination.Storage().Write();
            std::fill_n(write.View<int64_t>().data, rows * 6, int64_t(-777));
        }
        const auto schema = ElementSchema::Arithmetic<int64_t>();
        PayloadPlan payload(route, source.Layout(), destination.Layout(), schema);
        payload.Bind(source, destination).Begin().Wait();
        const auto verify = [&]
        {
            auto input = source.Storage().Read(), output = destination.Storage().Read();
            const auto in = input.View<int64_t>(), out = output.View<int64_t>();
            for (DNDS::index iRow = 0; iRow < rows; ++iRow)
            {
                const auto original = duplicate && previous == 0 && iRow == rows - 1 ? 0 : iRow * step % rows;
                for (DNDS::index iCol = 0; iCol < width; ++iCol)
                    CHECK(out[iRow * 6 + iCol] == previous * 1000000 + original * 16 + iCol);
                for (DNDS::index iCol = width; iCol < 6; ++iCol)
                    CHECK(out[iRow * 6 + iCol] == -777);
                for (DNDS::index iCol = width; iCol < 5; ++iCol)
                    CHECK(in[iRow * 5 + iCol] == -999);
            }
        };
        verify();
        if (duplicate)
        {
            PayloadPlan reverse(route.Reverse(), destination.Layout(), source.Layout(), schema);
            CHECK_THROWS(reverse.Bind(destination, source));
            // A failed reverse admission must leave the matching namespace
            // reusable and must not alter the forward row correspondence.
            payload.Bind(source, destination).Begin().Wait();
            verify();
        }
    }
}

TEST_CASE("Complete small row relations agree with independent sorted collision oracle")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    for (DNDS::index rows = 0; rows <= 4; ++rows)
    {
        DNDS::index combinations = 1;
        for (DNDS::index iRow = 0; iRow < rows; ++iRow)
            combinations *= rows;
        for (DNDS::index pattern = 0; pattern < combinations; ++pattern)
        {
            std::vector<std::vector<DNDS::index>> send(size), receive(size);
            auto value = pattern;
            for (DNDS::index iRow = 0; iRow < rows; ++iRow)
            {
                send[rank].push_back(value % rows);
                value /= rows;
                receive[rank].push_back(iRow);
            }
            auto oracle = send[rank];
            std::sort(oracle.begin(), oracle.end());
            const bool duplicate = std::adjacent_find(oracle.begin(), oracle.end()) != oracle.end();
            auto route = RowTransferPlan::Known(context, rows, rows, send, receive);
            CHECK_FALSE(route.HasCollisions());
            CHECK(route.Reverse().HasCollisions() == duplicate);
        }
    }
}

TEST_CASE("Typed binding rejects same-size arithmetic schema confusion")
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    RowTransferPlan route(MPI_COMM_WORLD, 31, 31, {{rank, 17, 3}});
    Array<int64_t> source(RowLayout::Dense(31, 7, 9)), destination(source.Layout());
    PayloadPlan wrong(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<double>());
    CHECK_THROWS(wrong.Bind(source, destination));
    CHECK_THROWS(wrong.Bind(source, destination, Reducer::Builtin<double>(Reduction::Sum)));
    CHECK_THROWS(Reducer::Custom<int64_t>(ElementSchema::Arithmetic<double>(), "wrong.scalar", [](int64_t &a, const int64_t &b)
                                          { a += b; }, int64_t(0)));
    CHECK(source.Storage().IsIdle());
    CHECK(destination.Storage().IsIdle());
}

TEST_CASE("Sender ordering derives a checked route without changing peer row correspondence")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size, previous = (rank + size - 1) % size;
    const std::vector<DNDS::index> sourceOrder{4, 1, 5, 0, 3, 2}, destinationOrder{3, 0, 5, 2, 1, 4};
    std::vector<std::vector<DNDS::index>> send(size), receive(size);
    send[next] = sourceOrder;
    receive[previous] = destinationOrder;
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    VersionTracker epoch;
    auto route = RowTransferPlan::Known(context, 6, 6, send, receive, {epoch.Stamp(Change::Distribution)});
    auto ordered = route.ReorderSourceByPeer();
    CHECK(ordered.first.Context() == context);
    CHECK(route.SendRows() == send);
    CHECK(ordered.first.ReceiveRows() == receive);
    CHECK_FALSE(ordered.first.HasCollisions());
    CHECK(ordered.first.SendRows()[next] == std::vector<DNDS::index>{0, 1, 2, 3, 4, 5});
    Array<int64_t> original(RowLayout::Ragged({1, 2, 3, 1, 2, 3}));
    std::vector<size_t> widths(6);
    {
        auto write = original.Write();
        const auto view = write.View();
        for (DNDS::index iRow = 0; iRow < 6; ++iRow)
        {
            widths[size_t(ordered.second[size_t(iRow)])] = size_t(view.RowSize(iRow));
            for (DNDS::index iSlot = 0; iSlot < view.RowSize(iRow); ++iSlot)
                view(iRow, iSlot) = rank * 100 + iRow * 10 + iSlot;
        }
    }
    Array<int64_t> permuted(RowLayout::Ragged(widths));
    {
        auto read = original.Read();
        auto write = permuted.Write();
        for (DNDS::index iRow = 0; iRow < 6; ++iRow)
            std::copy(read.View()[iRow].begin(), read.View()[iRow].end(), write.View()[ordered.second[size_t(iRow)]].begin());
    }
    PayloadPlan payload(ordered.first, permuted.Layout(), ElementSchema::Arithmetic<int64_t>());
    Array<int64_t> destination(payload.DestinationLayout());
    payload.Bind(permuted, destination).Begin().Wait();
    auto read = destination.Read();
    for (DNDS::index iEdge = 0; iEdge < 6; ++iEdge)
    {
        const auto iRow = destinationOrder[size_t(iEdge)], sourceRow = sourceOrder[size_t(iEdge)];
        CHECK(read.View().RowSize(iRow) == 1 + sourceRow % 3);
        for (DNDS::index iSlot = 0; iSlot < read.View().RowSize(iRow); ++iSlot)
            CHECK(read.View()(iRow, iSlot) == previous * 100 + sourceRow * 10 + iSlot);
    }
    const auto reverse = route.Reverse().ReorderSourceByPeer();
    CHECK(reverse.first.ReceiveRows() == send);
    if (size > 1)
        CHECK_THROWS((rank == 0 ? route.Reverse() : route).ReorderSourceByPeer());
    auto mutation = epoch.BeginMutation(Change::Distribution);
    mutation.Commit();
    CHECK_FALSE(ordered.first.Valid());
    CHECK_FALSE(reverse.first.Valid());
    CHECK_THROWS(route.ReorderSourceByPeer());

    auto incomplete = RowTransferPlan::Known(context, rank == 0 ? 7 : 6, 6, send, receive);
    CHECK_THROWS(incomplete.ReorderSourceByPeer());
    if (rank == 0)
        send[next].back() = send[next].front();
    auto duplicate = RowTransferPlan::Known(context, 6, 6, send, receive);
    CHECK_THROWS(duplicate.ReorderSourceByPeer());
    send[next] = {1, 0};
    receive[previous] = {0, 0};
    auto fanIn = RowTransferPlan::Known(context, 2, 1, send, receive).ReorderSourceByPeer();
    CHECK(fanIn.first.HasCollisions());
    CHECK_THROWS(fanIn.first.Reverse().ReorderSourceByPeer());
}

TEST_CASE("Native direct ragged halo retains leases and reusable routing")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size;
    Array<double> owned(RowLayout::Ragged({2, 0, 3}));
    {
        auto write = owned.Write();
        for (size_t i = 0; i < owned.RowCount(); ++i)
            for (size_t j = 0; j < owned.RowSize(i); ++j)
                write.At(i, j) = rank * 100 + i * 10 + j;
    }
    RowTransferPlan route(MPI_COMM_WORLD, 3, 3, {{next, 2, 0}, {next, 1, 1}, {rank, 0, 2}});
    Array<double> ghost(route.DestinationLayout(owned.Layout()));
    PayloadPlan payload(route, owned.Layout(), ghost.Layout(), ElementSchema::Arithmetic<double>());
    auto binding = payload.Bind(owned, ghost);
    const auto version = ghost.Version();
    {
        auto operation = binding.Begin();
        auto coreRead = owned.Read();
        CHECK(coreRead.At(0, 0) == rank * 100);
        CHECK_THROWS(owned.Write());
        CHECK_THROWS(ghost.Read());
        CHECK_THROWS(binding.Begin());
        CHECK(ghost.Version() == version);
        operation.WaitDestinationReady();
        CHECK(ghost.Version() == version + 1);
        auto ringRead = ghost.Read();
        CHECK(ringRead.At(0, 2) == next * 100 + 22);
        CHECK(ringRead.At(2, 1) == rank * 100 + 1);
        CHECK_THROWS(owned.Write()); // Send lease remains owned until full completion.
        operation.Wait();
    }
    binding.Begin().Wait();
    payload.Bind(owned, ghost, TransferTransport::HostPacked).Begin().Wait();
    CHECK(ghost.Read().At(0, 2) == next * 100 + 22);
    auto badShape = Array<double>(RowLayout::Dense(1, owned.Layout().Elements()));
    CHECK_THROWS(payload.Bind(badShape, ghost));
    {
        std::optional<Array<double>::ReadLease> conflict;
        if (rank == 0)
            conflict.emplace(ghost.Read());
        CHECK_THROWS(binding.Begin());
    }
    binding.Begin().Wait();
    Array<int32_t> second(RowLayout::Dense(3, 4));
    Array<int32_t> secondGhost(route.DestinationLayout(second.Layout()));
    PayloadPlan(route, second.Layout(), secondGhost.Layout(), ElementSchema::Arithmetic<int32_t>())
        .Bind(second, secondGhost)
        .Begin()
        .Wait();
}

TEST_CASE("Inferred payload shapes retain sparse holes widths and collective policy checks")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size;
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    Array<int64_t> source(RowLayout::Ragged({0, size_t(rank + 2), 3}));
    {
        auto write = source.Write();
        auto view = write.View();
        for (DNDS::index iRow = 0; iRow < view.RowCount(); ++iRow)
            for (DNDS::index iCol = 0; iCol < view.RowSize(iRow); ++iCol)
                view(iRow, iCol) = rank * 1000 + iRow * 10 + iCol;
    }
    RowTransferPlan route(context, 3, 4, {{next, 1, 3}, {next, 0, 1}, {rank, 2, 0}});
    CHECK(route.Context() == context);
    PayloadPlan payload(route, source.Layout(), ElementSchema::Arithmetic<int64_t>());
    CHECK(payload.DestinationLayout() == route.DestinationLayout(source.Layout()));
    Array<int64_t> destination(payload.DestinationLayout());
    for (auto mode : {TransferMode::Transient, TransferMode::Persistent})
    {
        auto binding = payload.Bind(source, destination, TransferTransport::Direct, {mode, context, 7});
        binding.Begin().Wait();
        auto read = destination.Read();
        CHECK(read.View().RowSize(1) == 0);
        CHECK(read.View().RowSize(2) == 0);
        CHECK(read.View().RowSize(3) == next + 2);
        CHECK(read.At(0, 2) == rank * 1000 + 22);
        CHECK(read.At(3, next + 1) == next * 1000 + 10 + next + 1);
    }
    RowTransferPlan collision(context, 3, 1, {{rank, 0, 0}, {rank, 2, 0}});
    CHECK_THROWS(PayloadPlan(collision, source.Layout(), ElementSchema::Arithmetic<int64_t>()));
    if (size > 1)
    {
        CHECK_THROWS(payload.Bind(source, destination, rank == 0 ? TransferTransport::HostPacked : TransferTransport::Direct,
                                  {TransferMode::Transient, context, 8}));
        auto mixed = [&]
        {
            if (rank == 0)
                return PayloadPlan(route, source.Layout(), ElementSchema::Arithmetic<int64_t>());
            return PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>());
        }();
        mixed.Bind(source, destination, TransferTransport::Direct, {TransferMode::Transient, context, 8}).Begin().Wait();
        CHECK_THROWS([&]
                     {
            if (rank == 0)
                return PayloadPlan(route, source.Layout(), RowLayout::Dense(4, 9), ElementSchema::Arithmetic<int64_t>());
            return PayloadPlan(route, source.Layout(), ElementSchema::Arithmetic<int64_t>()); }());
    }
    payload.Bind(source, destination, TransferTransport::Direct, {TransferMode::Transient, context, 7}).Begin().Wait();
}

TEST_CASE("Table width agreement preserves sparse rows and handles rank-varying layouts")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size;
    RowTransferPlan route(MPI_COMM_WORLD, 3, 4, {{next, 2, 3}, {rank, 1, 0}});
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        const DNDS::index width = scenario == 1 ? rank + 2 : 5;
        // Common-width tables, different widths on different ranks, and a
        // CSR/table mixture all follow one compatible collective protocol.
        const auto shape = scenario == 2 && rank == 0
                               ? RowLayout::Ragged({5, 5, 5})
                               : RowLayout::Dense(3, width, width + 2);
        Array<int64_t> source(shape);
        {
            auto write = source.Write();
            const auto view = write.View();
            for (DNDS::index iRow = 0; iRow < 3; ++iRow)
                for (DNDS::index iCol = 0; iCol < width; ++iCol)
                    view(iRow, iCol) = 1000 * rank + 100 * iRow + iCol;
        }
        PayloadPlan payload(route, source.Layout(), ElementSchema::Arithmetic<int64_t>(), TransferLayout::Compact());
        Array<int64_t> output(payload.DestinationLayout());
        payload.Bind(source, output).Begin().Wait();
        auto read = output.Read();
        const auto view = read.View();
        CHECK(view.RowSize(0) == width);
        CHECK(view.RowSize(1) == 0);
        CHECK(view.RowSize(2) == 0);
        const DNDS::index remoteWidth = scenario == 1 ? next + 2 : 5;
        CHECK(view.RowSize(3) == remoteWidth);
        for (DNDS::index iCol = 0; iCol < width; ++iCol)
            CHECK(view(0, iCol) == 1000 * rank + 100 + iCol);
        for (DNDS::index iCol = 0; iCol < remoteWidth; ++iCol)
            CHECK(view(3, iCol) == 1000 * next + 200 + iCol);
    }
}

TEST_CASE("Default inferred transfers preserve table stride padded capacity and CSR through repeated halos")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size;
    RowTransferPlan route(MPI_COMM_WORLD, 3, 3, {{next, 2, 0}, {next, 0, 1}, {next, 1, 2}});
    for (int kind = 0; kind < 3; ++kind)
    {
        CAPTURE(kind);
        const auto shape = kind == 0   ? RowLayout::Dense(3, 3, 5 + rank)
                           : kind == 1 ? RowLayout::Padded({0, 2, 3}, 6 + rank)
                                       : RowLayout::Ragged({0, 2, 3});
        Array<int64_t> source(shape);
        {
            auto write = source.Storage().Write();
            std::fill_n(write.View<int64_t>().data, shape.Elements(), int64_t(-777));
        }
        PayloadPlan payload(route, shape, ElementSchema::Arithmetic<int64_t>());
        const auto targetShape = payload.DestinationLayout();
        CHECK(targetShape.Kind() == shape.Kind());
        if (!shape.IsRagged())
            CHECK(targetShape.Stride() == shape.Stride());
        if (kind == 0)
        {
            CHECK(targetShape.Width() == 3);
            CHECK(targetShape.Offsets().empty());
            CHECK(targetShape.Sizes().empty());
        }
        CHECK(targetShape == route.DestinationLayout(shape));
        Array<int64_t> destination(targetShape);
        for (auto mode : {TransferMode::Transient, TransferMode::Persistent})
        {
            auto binding = payload.Bind(source, destination, TransferTransport::Direct, {mode});
            for (int stage = 0; stage < 2; ++stage)
            {
                {
                    auto write = source.Write();
                    for (DNDS::index iRow = 0; iRow < 3; ++iRow)
                        for (DNDS::index j = 0; j < source.RowSize(iRow); ++j)
                            write.View()(iRow, j) = 10000 * stage + 1000 * rank + 10 * iRow + j;
                }
                {
                    auto write = destination.Storage().Write();
                    std::fill_n(write.View<int64_t>().data, targetShape.Elements(), int64_t(-999));
                }
                binding.Begin().Wait();
                auto read = destination.Storage().Read();
                const auto flat = read.View<int64_t>();
                for (DNDS::index iRow = 0; iRow < 3; ++iRow)
                {
                    const DNDS::index oldRow = (iRow + 2) % 3;
                    const DNDS::index width = kind == 0 ? 3 : oldRow == 0 ? 0
                                                          : oldRow == 1   ? 2
                                                                          : 3;
                    CHECK(targetShape.RowSize(iRow) == width);
                    for (DNDS::index j = 0; j < width; ++j)
                        CHECK(flat[targetShape.Offset(iRow) + j] == 10000 * stage + 1000 * next + 10 * oldRow + j);
                    if (!targetShape.IsRagged())
                        for (DNDS::index j = width; j < targetShape.Stride(); ++j)
                            CHECK(flat[targetShape.Offset(iRow) + j] == -999);
                }
            }
        }
#ifdef DNDS_USE_CUDA
        if (DNDS::MPI::isCudaAware())
        {
            // Run the same preserved shapes through direct device MPI too.
            // Each MPI rank may select its local GPU with CUDA_VISIBLE_DEVICES.
            const auto context = ExecutionContext::CUDA(0);
            auto deviceSource = source.To(Device::CUDA(0), context);
            auto deviceDestination = destination.To(Device::CUDA(0), context);
            for (auto mode : {TransferMode::Transient, TransferMode::Persistent})
            {
                payload.Bind(deviceSource, deviceDestination, TransferTransport::Direct, {mode}).Begin(context).Wait();
                auto hostResult = deviceDestination.To(Device::Host(), context);
                auto read = hostResult.Storage().Read();
                const auto flat = read.View<int64_t>();
                for (DNDS::index iRow = 0; iRow < 3; ++iRow)
                {
                    const DNDS::index oldRow = (iRow + 2) % 3;
                    const auto width = targetShape.RowSize(iRow);
                    for (DNDS::index j = 0; j < width; ++j)
                        CHECK(flat[targetShape.Offset(iRow) + j] == 10000 + 1000 * next + 10 * oldRow + j);
                    if (!targetShape.IsRagged())
                        for (DNDS::index j = width; j < targetShape.Stride(); ++j)
                            CHECK(flat[targetShape.Offset(iRow) + j] == -999);
                }
            }
        }
        else
            MESSAGE("Direct device MPI unavailable; preserved-layout GPU transfer is not exercised");
#endif
    }
}

TEST_CASE("Preserved inference retains typed empty ranks and rejects remote rows that do not fit")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const DNDS::index owned = rank == 0 ? 2 : 0;
    const DNDS::index ghosts = rank == 0 ? 0 : 1;
    RowTransferPlan route(MPI_COMM_WORLD, owned, ghosts, rank == 0 ? std::vector<RowRequest>{} : std::vector<RowRequest>{{0, 1, 0}});
    for (int kind = 0; kind < 2; ++kind)
    {
        const auto shape = kind == 0 ? RowLayout::Dense(owned, 3, 5 + rank)
                                     : RowLayout::Padded(rank == 0 ? std::vector<size_t>{1, 3} : std::vector<size_t>{}, 5 + rank);
        Array<int64_t> source(shape);
        {
            auto write = source.Write();
            for (DNDS::index iRow = 0; iRow < owned; ++iRow)
                for (DNDS::index j = 0; j < shape.RowSize(iRow); ++j)
                    write.View()(iRow, j) = iRow * 10 + j;
        }
        PayloadPlan payload(route, shape, ElementSchema::Arithmetic<int64_t>());
        CHECK(payload.DestinationLayout().Kind() == shape.Kind());
        CHECK(payload.DestinationLayout().Stride() == shape.Stride());
        Array<int64_t> destination(payload.DestinationLayout());
        payload.Bind(source, destination).Begin().Wait();
        if (rank != 0)
            CHECK(destination.Read().At(0, 2) == 12);
    }
    if (size > 1)
    {
        // One receiver has insufficient local capacity. All ranks must reject
        // before any data binding; the following valid transfer must still work.
        const auto bad = RowLayout::Padded(rank == 0 ? std::vector<size_t>{1, 3} : std::vector<size_t>{}, rank == 1 ? 2 : 5);
        CHECK_THROWS(PayloadPlan(route, bad, ElementSchema::Arithmetic<int64_t>()));
        CHECK_THROWS(route.DestinationLayout(bad));
        CHECK_NOTHROW(PayloadPlan(route, bad, ElementSchema::Arithmetic<int64_t>(), TransferLayout::Compact()));
        const auto badTable = RowLayout::Dense(owned, rank == 1 ? 2 : 3, 5);
        CHECK_THROWS(PayloadPlan(route, badTable, ElementSchema::Arithmetic<int64_t>()));
        CHECK_THROWS(route.DestinationLayout(badTable));
        CHECK_NOTHROW(PayloadPlan(route, badTable, ElementSchema::Arithmetic<int64_t>(), TransferLayout::Compact()));
    }
}

TEST_CASE("Preserved sparse table slots keep width and explicit compact retains zero-length holes")
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const auto shape = RowLayout::Dense(2, 3, 7);
    RowTransferPlan route(MPI_COMM_WORLD, 2, 4, {{rank, 1, 3}});
    PayloadPlan table(route, shape, ElementSchema::Arithmetic<int64_t>());
    CHECK(table.DestinationLayout() == RowLayout::Dense(4, 3, 7));
    PayloadPlan compact(route, shape, ElementSchema::Arithmetic<int64_t>(), TransferLayout::Compact());
    CHECK(compact.DestinationLayout() == RowLayout::Ragged({0, 0, 0, 3}));
}

TEST_CASE("Binding rejects reordered communicator and rank-local tag reservation failure")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    RowTransferPlan route(context, 1, 1, {{(rank + 1) % size, 0, 0}});
    Array<int64_t> source(RowLayout::Dense(1, 1)), destination(RowLayout::Dense(1, 1));
    {
        auto write = source.Write();
        write.At(0, 0) = rank + 19;
    }
    PayloadPlan payload(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>());
    if (size > 1)
    {
        MPI_Comm reordered = MPI_COMM_NULL;
        MPI_Comm_split(MPI_COMM_WORLD, 0, size - rank, &reordered);
        {
            auto wrong = std::make_shared<CommunicationContext>(reordered);
            CHECK_THROWS(payload.Bind(source, destination, TransferTransport::Direct,
                                      {TransferMode::Transient, wrong, 12}));
        }
        MPI_Comm_free(&reordered);
    }
    std::shared_ptr<const int> held;
    if (rank == 0)
        held = context->Reserve(12);
    CHECK_THROWS(payload.Bind(source, destination, TransferTransport::Direct,
                              {TransferMode::Transient, context, 12}));
    held.reset();
    // Failed ranks must retire any successful local reservations, so the
    // identical tag remains usable after the collectively rejected binding.
    payload.Bind(source, destination, TransferTransport::Direct,
                 {TransferMode::Transient, context, 12})
        .Begin()
        .Wait();
    CHECK(destination.Read().At(0, 0) == (rank + 1) % size + 19);
}

TEST_CASE("Combined payload and binding preflight reject rank-local mismatches before requests")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    VersionTracker versions;
    RowTransferPlan route(context, 3, 2, {{(rank + 1) % size, 2, 0}, {rank, 0, 1}}, {versions.Stamp()});
    Array<int64_t> source(RowLayout::Dense(3, 5, 7)), destination(RowLayout::Dense(2, 5, 8));
    {
        auto write = source.Write();
        for (DNDS::index iRow = 0; iRow < 3; ++iRow)
            for (DNDS::index iCol = 0; iCol < 5; ++iCol)
                write.At(iRow, iCol) = rank * 100 + iRow * 10 + iCol;
    }
    CHECK_THROWS(PayloadPlan(route, RowLayout::Dense(rank == 0 ? 2 : 3, 5), ElementSchema::Arithmetic<int64_t>()));
    PayloadPlan payload(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>());
    // Same logical widths, different physical strides only on
    // one rank. Both typed copy and reduction must fail collectively.
    auto wrong = rank == 0 ? Array<int64_t>(RowLayout::Dense(2, 5, 9)) : destination;
    CHECK_THROWS(payload.Bind(source, wrong, TransferTransport::Direct, {TransferMode::Transient, context, 19}));
    CHECK_THROWS(payload.Bind(source, wrong, Reducer::Builtin<int64_t>(Reduction::Sum),
                              TransferTransport::Direct, {TransferMode::Transient, context, 19}));
    CHECK(source.Storage().IsIdle());
    CHECK(destination.Storage().IsIdle());
    // A rejected preflight must not retain the matching tag or any leases.
    payload.Bind(source, destination, TransferTransport::Direct, {TransferMode::Transient, context, 19}).Begin().Wait();
    CHECK(destination.Read().At(0, 4) == ((rank + 1) % size) * 100 + 24);
    // Application-level binding checks join the existing transport preflight.
    // One rejecting rank must stop every peer before any request is posted;
    // both caller-ordered and checked entry points remain reusable afterward.
    auto binding = payload.Bind(source, destination, TransferTransport::Direct, {TransferMode::Transient, context, 19});
    EagerEngine engine(0, std::make_shared<AccessTrackingContext>());
    const auto stream = engine.CreateCPUStream();
    CHECK_THROWS(binding.BeginScheduled(stream.Context(), rank != 0));
    binding.BeginScheduled(stream.Context()).Wait();
    CHECK_THROWS(binding.BeginChecked(stream, rank != 0));
    binding.BeginChecked(stream).Wait();
    CHECK(source.Storage().IsIdle());
    CHECK(destination.Storage().IsIdle());
    CHECK(destination.Read().At(0, 4) == ((rank + 1) % size) * 100 + 24);
    if (rank == 0)
    {
        auto mutation = versions.BeginMutation(Change::Distribution);
        mutation.Commit();
    }
    CHECK_THROWS(PayloadPlan(route, source.Layout(), ElementSchema::Arithmetic<int64_t>()));
}

TEST_CASE("Typed MPI bindings reject one unpublished shape collectively and recover after commit")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    RowTransferPlan route(context, 3, 2, {{(rank + 1) % size, 2, 0}, {rank, 0, 1}});
    Array<int64_t> source(RowLayout::Dense(3, 5, 7)), destination(RowLayout::Dense(2, 5, 8));
    {
        auto write = source.Write();
        for (DNDS::index iRow = 0; iRow < 3; ++iRow)
            for (DNDS::index iCol = 0; iCol < 5; ++iCol)
                write.At(iRow, iCol) = rank * 100 + iRow * 10 + iCol;
    }
    PayloadPlan payload(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>());
    std::optional<ExclusiveRows> editor;
    if (rank == 0)
        editor.emplace(destination.EditRows());
    CHECK_THROWS(payload.Bind(source, destination, TransferTransport::Direct, {TransferMode::Transient, context, 23}));
    CHECK_THROWS(payload.Bind(source, destination, Reducer::Builtin<int64_t>(Reduction::Sum),
                              TransferTransport::Direct, {TransferMode::Transient, context, 23}));
    if (editor)
        editor->Commit();
    payload.Bind(source, destination, TransferTransport::Direct, {TransferMode::Transient, context, 23}).Begin().Wait();
    {
        auto read = destination.Read();
        for (DNDS::index iCol = 0; iCol < 5; ++iCol)
        {
            CHECK(read.At(0, iCol) == ((rank + 1) % size) * 100 + 20 + iCol);
            CHECK(read.At(1, iCol) == rank * 100 + iCol);
        }
    }
    // Abandoned edits and consumed handles take the same collective path.
    if (rank == 0)
    {
        auto abandoned = destination.EditRows();
    }
    CHECK_THROWS(payload.Bind(source, destination));
    destination.Reset(RowLayout::Dense(2, 5, 8));
    std::optional<Array<int64_t>> retained;
    if (rank == 0)
        retained.emplace(std::move(source));
    CHECK_THROWS(payload.Bind(source, destination));
}

TEST_CASE("Direct permutation honors imported buffer origins and padded state strides")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const auto peer = (rank + 1) % size;
    constexpr size_t rows = 31, width = 7, prefix = 13, suffix = 17;
    auto sourceOwner = std::make_shared<std::vector<double>>(prefix + rows * 8 + suffix, -9);
    auto targetOwner = std::make_shared<std::vector<double>>(prefix + rows * 9 + suffix, -19);
    auto sourceAllocation = Allocation::Import(std::shared_ptr<void>(sourceOwner, sourceOwner->data()), sourceOwner->size() * sizeof(double));
    auto targetAllocation = Allocation::Import(std::shared_ptr<void>(targetOwner, targetOwner->data()), targetOwner->size() * sizeof(double));
    auto source = Array<double>::FromStorage(RowLayout::Dense(rows, width, 8),
                                             Buffer::FromAllocation(sourceAllocation, prefix * sizeof(double), rows * 8 * sizeof(double)));
    auto target = Array<double>::FromStorage(RowLayout::Dense(rows, width, 9),
                                             Buffer::FromAllocation(targetAllocation, prefix * sizeof(double), rows * 9 * sizeof(double)));
    {
        auto write = source.Write();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < width; ++j)
                write.At(i, j) = rank * 10000 + i * 100 + j;
    }
    std::vector<RowRequest> requests;
    for (size_t i = 0; i < rows; ++i)
        requests.push_back({peer, int64_t(i * 7 % rows), i});
    RowTransferPlan route(MPI_COMM_WORLD, rows, rows, requests);
    PayloadPlan payload(route, source.Layout(), target.Layout(), ElementSchema::Arithmetic<double>());
    for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
    {
        payload.Bind(source, target, transport).Begin().Wait();
        auto read = target.Read();
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < width; ++j)
                CHECK(read.At(i, j) == peer * 10000 + (i * 7 % rows) * 100 + j);
            CHECK((*targetOwner)[prefix + i * 9 + 7] == -19);
            CHECK((*targetOwner)[prefix + i * 9 + 8] == -19);
            CHECK((*sourceOwner)[prefix + i * 8 + 7] == -9);
        }
        for (size_t i = 0; i < prefix; ++i)
            CHECK((*targetOwner)[i] == -19);
        for (size_t i = prefix + rows * 9; i < targetOwner->size(); ++i)
            CHECK((*targetOwner)[i] == -19);
    }
}

TEST_CASE("One route transfers padded reconstruction rows into a packed CSR destination")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    constexpr size_t count = 37;
    std::vector<size_t> lengths(count);
    std::vector<RowRequest> requests;
    for (size_t i = 0; i < count; ++i)
    {
        lengths[i] = i % 11 == 0 ? 0 : (i % 7 + 1) * 5;
        requests.push_back({(rank + 1) % size, int64_t(i * 5 % count), i});
    }
    Array<double> source(RowLayout::Padded(lengths, 40));
    {
        auto write = source.Write();
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < lengths[i]; ++j)
                write.At(i, j) = rank * 10000 + i * 100 + j;
    }
    RowTransferPlan route(MPI_COMM_WORLD, count, count, requests);
    Array<double> destination(route.DestinationLayout(source.Layout(), TransferLayout::Compact()));
    CHECK(destination.Layout().IsRagged());
    auto payload = PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<double>());
    for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
    {
        payload.Bind(source, destination, transport).Begin().Wait();
        auto read = destination.Read();
        for (size_t i = 0; i < count; ++i)
        {
            const auto original = i * 5 % count;
            CHECK(destination.RowSize(i) == lengths[original]);
            for (size_t j = 0; j < lengths[original]; ++j)
                CHECK(read.At(i, j) == ((rank + 1) % size) * 10000 + original * 100 + j);
        }
    }
}

TEST_CASE("Native reverse fan-in rejects copy and supports reductions")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Array<int32_t> root(RowLayout::Dense(rank == 0 ? 1 : 0, 2));
    Array<int32_t> leaves(RowLayout::Dense(2, 2));
    RowTransferPlan halo(MPI_COMM_WORLD, root.RowCount(), 2, {{0, 0, 0}, {0, 0, 1}});
    PayloadPlan reverse(halo.Reverse(), leaves.Layout(), root.Layout(), ElementSchema::Arithmetic<int32_t>());
    CHECK_THROWS(reverse.Bind(leaves, root));
    {
        auto write = leaves.Write();
        write.At(0, 0) = rank + 1;
        write.At(1, 0) = rank + 2;
        write.At(0, 1) = 1;
        write.At(1, 1) = 2;
    }
    for (auto op : {Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::BitOr, Reduction::BitXor, Reduction::BitAnd})
    {
        reverse.Bind(leaves, root, Reducer::Builtin<int32_t>(op, DestinationInit::Identity)).Begin().Wait();
        if (rank == 0)
        {
            auto result = root.Read();
            if (op == Reduction::Sum)
                CHECK(result.At(0, 0) == size * (size + 2));
            if (op == Reduction::Min)
                CHECK(result.At(0, 0) == 1);
            if (op == Reduction::Max)
                CHECK(result.At(0, 0) == size + 1);
            if (op == Reduction::BitOr)
                CHECK(result.At(0, 1) == 3);
            if (op == Reduction::BitXor)
                CHECK(result.At(0, 1) == (size % 2 ? 3 : 0));
            if (op == Reduction::BitAnd)
                CHECK(result.At(0, 1) == 0);
        }
    }
    if (rank == 0)
        root.Write().At(0, 0) = 7;
    reverse.Bind(leaves, root, Reducer::Builtin<int32_t>(Reduction::Sum)).Begin().Wait();
    if (rank == 0)
        CHECK(root.Read().At(0, 0) == 7 + size * (size + 2));
}

TEST_CASE("Native POD transfers use MPI extent and support custom reductions")
{
    struct Metadata
    {
        int32_t tag;
        double weight;
    };
    MPI_Datatype raw, pod;
    int blocks[] = {1, 1};
    MPI_Aint offsets[] = {offsetof(Metadata, tag), offsetof(Metadata, weight)};
    MPI_Datatype types[] = {MPI_INT32_T, MPI_DOUBLE};
    MPI_Type_create_struct(2, blocks, offsets, types, &raw);
    MPI_Type_create_resized(raw, 0, sizeof(Metadata), &pod);
    MPI_Type_commit(&pod);
    ElementSchema schema(pod, sizeof(Metadata), "test.metadata.v1");
    MPI_Type_free(&raw);
    MPI_Type_free(&pod);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Array<Metadata> src(RowLayout::Dense(3, 2)), dst(RowLayout::Dense(3, 2));
    {
        auto write = src.Write();
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 2; ++j)
                write.At(i, j) = {rank, double(i * 2 + j)};
    }
    RowTransferPlan route(MPI_COMM_WORLD, 3, 3, {{(rank + 1) % size, 0, 0}, {(rank + 1) % size, 1, 1}, {(rank + 1) % size, 2, 2}});
    PayloadPlan payload(route, src.Layout(), dst.Layout(), schema);
    CHECK(payload.SendSegments() == 1);
    CHECK(payload.ReceiveSegments() == 1);
    payload.Bind(src, dst).Begin().Wait();
    CHECK(dst.Read().At(2, 1).tag == (rank + 1) % size);
    CHECK(dst.Read().At(2, 1).weight == 5);
    payload.Bind(src, dst, TransferTransport::HostPacked).Begin().Wait();
    CHECK(dst.Read().At(2, 1).tag == (rank + 1) % size);
    auto add = Reducer::Custom<Metadata>(schema, "metadata.sum", [](Metadata &a, const Metadata &b)
                                         { a.tag += b.tag; a.weight += b.weight; }, Metadata{}, DestinationInit::Identity);
    payload.Bind(src, dst, add).Begin().Wait();
    CHECK(dst.Read().At(2, 1).weight == 5);
    for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
    {
        payload.Bind(src, dst, transport, {TransferMode::Transient, {}, 0}).Begin().Wait();
        CHECK(dst.Read().At(2, 1).tag == (rank + 1) % size);
        CHECK(dst.Read().At(2, 1).weight == 5);
        payload.Bind(src, dst, add, transport, {TransferMode::Transient, {}, 0}).Begin().Wait();
        CHECK(dst.Read().At(2, 1).tag == (rank + 1) % size);
        CHECK(dst.Read().At(2, 1).weight == 5);
    }
#ifdef DNDS_USE_CUDA
    auto deviceSource = src.To(Device::CUDA(0));
    auto deviceDestination = dst.To(Device::CUDA(0));
    payload.Bind(deviceSource, deviceDestination, DNDS::MPI::isCudaAware() ? TransferTransport::Direct : TransferTransport::HostPacked)
        .Begin(ExecutionContext::CUDA(0))
        .Wait();
    auto deviceResult = deviceDestination.To(Device::Host());
    CHECK(deviceResult.Read().At(2, 1).tag == (rank + 1) % size);
    CHECK(deviceResult.Read().At(2, 1).weight == 5);
#endif
}

TEST_CASE("Native empty routes and generation validation")
{
    VersionTracker tracker;
    RowTransferPlan route(MPI_COMM_WORLD, 0, 0, {}, {tracker.Stamp(Change::Layout)});
    Array<double> src, dst;
    auto binding = PayloadPlan(route, src.Layout(), dst.Layout(), ElementSchema::Arithmetic<double>()).Bind(src, dst);
    CHECK(binding.Begin().Test());
    PayloadPlan(route, src.Layout(), dst.Layout(), ElementSchema::Arithmetic<double>())
        .Bind(src, dst, TransferTransport::HostPacked)
        .Begin()
        .Wait();
    auto mutation = tracker.BeginMutation(Change::Layout);
    mutation.Commit();
    CHECK_THROWS(binding.Begin());
}

// Unsafe reducer completion is exercised by the distributed-failure fixture.

#ifdef DNDS_USE_CUDA
TEST_CASE_TEMPLATE("Native CUDA reduction dispatch covers wide rows without requiring device MPI", T,
                   int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double)
{
    const auto ctx = ExecutionContext::CUDA(0);
    Array<T> source(RowLayout::Ragged({129, 129, 129, 257, 257, 257, 257}));
    Array<T> initial(RowLayout::Dense(3, 257, 263));
    std::vector<Contribution> groups;
    {
        auto w = source.Write();
        for (size_t i = 0; i < source.RowCount(); ++i)
        {
            groups.push_back({source.Layout().Offset(i), i < 3 ? size_t(0) : size_t(526), source.RowSize(i)});
            for (size_t j = 0; j < source.RowSize(i); ++j)
                w.At(i, j) = j == 0 ? std::numeric_limits<T>::max() : T(int((i * 5 + j) % 27) - 13);
        }
        auto seed = initial.Storage().Write();
        std::fill_n(seed.template View<T>().data, initial.Layout().Elements(), T(9));
    }
    auto gpuSource = source.To(Device::CUDA(0));
    auto hostGroups = Buffer::Allocate(groups.size() * sizeof(Contribution));
    {
        auto w = hostGroups.Write();
        std::copy(groups.begin(), groups.end(), w.View<Contribution>().data);
    }
    auto deviceGroups = Buffer::Allocate(hostGroups.Bytes(), Device::CUDA(0));
    Copy(hostGroups, deviceGroups, ctx).Wait();
    for (auto op : {Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::BitOr, Reduction::BitXor, Reduction::BitAnd})
    {
        if (!std::is_integral_v<T> && op > Reduction::Max)
            continue;
        for (auto init : {DestinationInit::Identity, DestinationInit::Preserve})
        {
            auto expected = initial.Clone();
            auto result = initial.To(Device::CUDA(0));
            auto reducer = Reducer::Builtin<T>(op, init);
            {
                auto src = source.Read();
                auto dst = expected.Write();
                reducer.host(src.View().data, dst.View().data, groups, ExecutionContext::Host());
            }
            {
                auto src = gpuSource.Read(ctx);
                auto dst = result.Write(ctx);
                auto mapping = deviceGroups.Read(ctx);
                REQUIRE(bool(reducer.device));
                reducer.device(src.View().data, dst.View().data, mapping.View<Contribution>().data, groups.size(), ctx);
                auto done = Completion::Record(ctx);
                src.Complete(done);
                dst.Complete(done);
                mapping.Complete(done);
            }
            auto hostResult = result.To(Device::Host());
            auto actual = hostResult.Storage().Read();
            auto reference = expected.Storage().Read();
            // Includes untouched row, short-row tail and physical stride padding.
            for (size_t j = 0; j < initial.Layout().Elements(); ++j)
                CHECK(actual.template View<T>().data[j] == reference.template View<T>().data[j]);
        }
    }
}

TEST_CASE("Native CUDA integer reduction families agree with the CPU reference")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (!DNDS::MPI::isCudaAware())
        return;
    Array<int32_t> source(RowLayout::Dense(1, 2)), cpu(RowLayout::Dense(1, 2));
    source.Write().At(0, 0) = rank + 1;
    source.Write().At(0, 1) = 1 << (rank % 8);
    auto gpuSource = source.To(Device::CUDA(0));
    auto gpu = cpu.To(Device::CUDA(0));
    std::vector<RowRequest> requests;
    for (int peer = 0; peer < size; ++peer)
    {
        requests.push_back({peer, 0, 0});
        requests.push_back({peer, 0, 0});
    }
    RowTransferPlan route(MPI_COMM_WORLD, 1, 1, requests);
    PayloadPlan payload(route, source.Layout(), cpu.Layout(), ElementSchema::Arithmetic<int32_t>());
    for (auto op : {Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::BitOr, Reduction::BitXor, Reduction::BitAnd})
    {
        auto reducer = Reducer::Builtin<int32_t>(op, DestinationInit::Identity);
        payload.Bind(source, cpu, reducer).Begin().Wait();
        payload.Bind(gpuSource, gpu, reducer).Begin(ExecutionContext::CUDA(0)).Wait();
        auto result = gpu.To(Device::Host());
        CHECK(result.Read().At(0, 0) == cpu.Read().At(0, 0));
        CHECK(result.Read().At(0, 1) == cpu.Read().At(0, 1));
    }
}

TEST_CASE("Native CUDA direct MPI and device fan-in reduction")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto ctx = ExecutionContext::CUDA(0);
    Array<double> host(RowLayout::Ragged({0, 3}));
    {
        auto w = host.Write();
        for (size_t j = 0; j < 3; ++j)
            w.At(1, j) = rank + j;
    }
    auto source = host.To(Device::CUDA(0), ctx);
    Array<double> destination(source.Layout(), Device::CUDA(0));
    RowTransferPlan route(MPI_COMM_WORLD, 2, 2, {{(rank + 1) % size, 0, 0}, {(rank + 1) % size, 1, 1}, {rank, 1, 1}});
    PayloadPlan payload(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<double>());
    auto reducer = Reducer::Builtin<double>(Reduction::Sum, DestinationInit::Identity);
    auto packed = payload.Bind(source, destination, reducer, TransferTransport::HostPacked).Begin(ctx);
    while (!packed.Test())
        std::this_thread::yield();
    CHECK(destination.To(Device::Host(), ctx).Read().At(1, 2) == rank + (rank + 1) % size + 4);
    if (!DNDS::MPI::isCudaAware())
    {
        CHECK_THROWS(payload.Bind(source, destination, reducer));
        MESSAGE("Direct device MPI unavailable; capability rejection verified");
        return;
    }
    auto binding = payload.Bind(source, destination, reducer);
    binding.Begin(ctx).Wait();
    auto result = destination.To(Device::Host(), ctx);
    CHECK(result.Read().At(1, 2) == rank + (rank + 1) % size + 4);
}
#endif

TEST_CASE("Mixed physical tables use direct datatype routes and preserve logical rows")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    constexpr DNDS::index nRow = 33, width = 7;
    const auto formats = std::vector<RowLayout>{RowLayout::Dense(nRow, width, 9), RowLayout::ColMajor(nRow, width, {32, 1}), RowLayout::TiledColMajor(nRow, width, 8)};
    std::vector<RowRequest> requests;
    for (DNDS::index iRow = 0; iRow < nRow; ++iRow)
        requests.push_back({(rank + 1) % size, (iRow * 13) % nRow, iRow});
    RowTransferPlan route(context, nRow, nRow, requests);
    for (const auto &sourceShape : formats)
        for (const auto &destinationShape : formats)
        {
            Array<int64_t> source(sourceShape), destination(destinationShape);
            {
                auto write = source.Write();
                for (DNDS::index iRow = 0; iRow < nRow; ++iRow)
                    for (DNDS::index iComp = 0; iComp < width; ++iComp)
                        write.View()(iRow, iComp) = rank * 100000 + iRow * 100 + iComp;
            }
            for (auto mode : {TransferMode::Transient, TransferMode::Persistent})
                for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
                {
                    {
                        auto write = destination.Storage().Write();
                        std::fill_n(write.View<int64_t>().data, destinationShape.Elements(), int64_t(-991));
                    }
                    PayloadPlan payload(route, sourceShape, destinationShape, ElementSchema::Arithmetic<int64_t>());
                    payload.Bind(source, destination, transport, {mode, context, 9}).Begin().Wait();
                    const auto read = destination.Read();
                    for (DNDS::index iRow = 0; iRow < nRow; ++iRow)
                        for (DNDS::index iComp = 0; iComp < width; ++iComp)
                            CHECK(read.View()(iRow, iComp) == ((rank + 1) % size) * 100000 + ((iRow * 13) % nRow) * 100 + iComp);
                    for (DNDS::index iElement = 0; iElement < DNDS::index(destinationShape.Elements()); ++iElement)
                    {
                        bool logical = false;
                        for (DNDS::index iRow = 0; iRow < nRow && !logical; ++iRow)
                            for (DNDS::index iComp = 0; iComp < width; ++iComp)
                                logical = logical || destinationShape.ElementOffset(iRow, iComp) == iElement;
                        if (!logical)
                            CHECK(read.View().data[iElement] == -991);
                    }
                }
            PayloadPlan preserving(route, sourceShape, ElementSchema::Arithmetic<int64_t>());
            CHECK(preserving.DestinationLayout() == sourceShape);
        }
    RowTransferPlan small(context, nRow, 3, {{rank, 1, 0}, {rank, 2, 1}, {rank, 3, 2}});
    PayloadPlan preserving(small, formats[1], ElementSchema::Arithmetic<int64_t>());
    CHECK(preserving.DestinationLayout().Kind() == RowLayoutKind::ColMajorTable);
    CHECK(preserving.DestinationLayout().Stride() == 32);
}

TEST_CASE("Strided receive collisions use scalar contribution groups for builtin and custom POD reductions")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const DNDS::index width = 5;
    std::vector<RowRequest> requests;
    for (int peer = 0; peer < size; ++peer)
    {
        requests.push_back({peer, 0, 0});
        requests.push_back({peer, 0, 0});
    }
    RowTransferPlan route(MPI_COMM_WORLD, 1, 1, requests);
    struct Record
    {
        int64_t tag;
        double value;
    };
    const auto schema = ElementSchema::Register<Record>(ElementSchema::Record("test.physical.Record.v1", sizeof(Record),
                                                                              {{"tag", offsetof(Record, tag), ElementScalar::I64}, {"value", offsetof(Record, value), ElementScalar::F64}}, "Record", alignof(Record)));
    for (const auto &shape : {RowLayout::ColMajor(1, width, {8, 0}), RowLayout::TiledColMajor(1, width, 8)})
    {
        Array<int64_t> source(shape), destination(shape);
        Array<Record> podSource(shape), podDestination(shape);
        {
            auto write = source.Write();
            auto pod = podSource.Write();
            for (DNDS::index iComp = 0; iComp < width; ++iComp)
            {
                write.View()(0, iComp) = rank + 1 + iComp;
                pod.View()(0, iComp) = {rank + 1, double(iComp + rank)};
            }
        }
        PayloadPlan payload(route, shape, shape, ElementSchema::Arithmetic<int64_t>());
        CHECK_THROWS(payload.Bind(source, destination));
        payload.Bind(source, destination, Reducer::Builtin<int64_t>(Reduction::Sum, DestinationInit::Identity)).Begin().Wait();
        auto reducer = Reducer::Custom<Record>(schema, "physical.Record.sum", [](Record &a, const Record &b)
                                               { a.tag += b.tag; a.value += b.value; }, Record{}, DestinationInit::Identity);
        PayloadPlan podPayload(route, shape, shape, schema);
        podPayload.Bind(podSource, podDestination, reducer).Begin().Wait();
        const auto read = destination.Read();
        const auto pod = podDestination.Read();
        for (DNDS::index iComp = 0; iComp < width; ++iComp)
        {
            CHECK(read.View()(0, iComp) == size * (size + 1) + 2 * size * iComp);
            CHECK(pod.View()(0, iComp).tag == size * (size + 1));
            CHECK(pod.View()(0, iComp).value == 2 * size * iComp + size * (size - 1));
        }
    }
}

TEST_CASE("Communication worker preserves repeated packed and direct sparse transfers")
{
    int rank = 0, size = 0, provided = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Query_thread(&provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        CHECK_THROWS(CommunicationWorker{});
        return;
    }
    auto worker = std::make_shared<CommunicationWorker>();
    const int active = size > 2 ? size - 1 : size;
    const DNDS::index rows = rank < active ? 8 : 0, ghosts = rank < active ? 2 : 0;
    const int previous = (rank + active - 1) % active;
    for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
        for (auto mode : {TransferMode::Persistent, TransferMode::Transient})
        {
            Array<double> input(RowLayout::ColMajor(rows, 5)), output(RowLayout::ColMajor(ghosts, 5));
            std::vector<RowRequest> requests;
            if (ghosts)
                requests = {{previous, 7, 0}, {previous, 1, 1}};
            PayloadPlan payload(RowTransferPlan(MPI_COMM_WORLD, rows, ghosts, requests), input.Layout(), output.Layout(), ElementSchema::For<double>());
            TransferOptions options;
            options.mode = mode;
            options.communicationWorker = worker;
            auto binding = payload.Bind(input, output, transport, options);
            for (int epoch = 0; epoch < 3; ++epoch)
            {
                {
                    auto write = input.Write();
                    auto values = write.View<UniformTableRows>();
                    for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                        for (DNDS::index j = 0; j < 5; ++j)
                            values(iRow, j) = 1000 * epoch + 100 * rank + 10 * iRow + j;
                }
                // Gate worker execution explicitly: the main-thread core read
                // must finish while the transfer's write reservation is live.
                std::mutex mutex;
                std::condition_variable condition;
                bool release = false;
                worker->Submit([&]
                               {
                    std::unique_lock<std::mutex> lock(mutex);
                    condition.wait(lock, [&] { return release; }); });
                auto operation = binding.Begin();
                CHECK_FALSE(operation.Test());
                CHECK_THROWS(output.Read());
                {
                    auto read = input.Read();
                    const auto values = read.View<UniformTableRows>();
                    for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                        CHECK(values(iRow, 0) == 1000 * epoch + 100 * rank + 10 * iRow);
                }
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    release = true;
                    condition.notify_one();
                }
                operation.WaitDestinationReady();
                {
                    auto read = output.Read();
                    auto values = read.View<UniformTableRows>();
                    for (DNDS::index iRow = 0; iRow < ghosts; ++iRow)
                        for (DNDS::index j = 0; j < 5; ++j)
                            CHECK(values(iRow, j) == 1000 * epoch + 100 * previous + 10 * (iRow == 0 ? 7 : 1) + j);
                }
                operation.Wait();
                CHECK(operation.Test());
            }
        }
}

TEST_CASE("Communication worker overlaps core reads with active receiver completion")
{
    int rank = 0, size = 0, provided = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Query_thread(&provided);
    if (provided != MPI_THREAD_MULTIPLE)
        return;
    auto worker = std::make_shared<CommunicationWorker>();
    Array<double> input(RowLayout::Dense(1, 1)), output(RowLayout::Dense(1, 1));
    {
        auto write = input.Write();
        write.View()(0, 0) = rank + .5;
    }
    std::mutex mutex;
    std::condition_variable ready;
    bool entered = false, release = false;
    const auto schema = ElementSchema::For<double>();
    auto reducer = Reducer::Custom<double>(schema, "gated-sum", [&](double &destination, const double &source)
                                           {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        ready.notify_one();
        ready.wait(lock, [&] { return release; });
        destination += source; }, 0., DestinationInit::Identity);
    TransferOptions options;
    options.communicationWorker = worker;
    auto binding = PayloadPlan(RowTransferPlan(MPI_COMM_WORLD, 1, 1, {{(rank + 1) % size, 0, 0}}), input.Layout(), output.Layout(), schema)
                       .Bind(input, output, reducer, TransferTransport::Direct, options);
    auto operation = binding.Begin();
    {
        std::unique_lock<std::mutex> lock(mutex);
        const bool active = ready.wait_for(lock, std::chrono::seconds(5), [&]
                                           { return entered; });
        CHECK(active);
        // This is inside a real transfer's receiver completion, not simply a
        // pending queue entry. Computation can read owned data concurrently.
        if (active)
        {
            const auto read = input.Read();
            CHECK(read.View()(0, 0) * 2 == 2 * rank + 1);
            CHECK_FALSE(operation.Test());
        }
        release = true;
        ready.notify_one();
    }
    operation.Wait();
    const auto read = output.Read();
    CHECK(read.View()(0, 0) == (rank + 1) % size + .5);
}

TEST_CASE("DevicePacked is explicit and rejects host endpoints collectively")
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Array<double> source(RowLayout::Dense(3, 5)), destination(RowLayout::Dense(2, 5));
    PayloadPlan payload(RowTransferPlan(MPI_COMM_WORLD, 3, 2, {{rank, 2, 1}}), source.Layout(), destination.Layout(), ElementSchema::For<double>());
    CHECK_THROWS(payload.Bind(source, destination, TransferTransport::DevicePacked));
    // Failed backend selection does not poison the default direct path.
    payload.Bind(source, destination).Begin().Wait();
}
TEST_CASE("MPI enqueue and unavailable NCCL capabilities reject explicitly")
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Array<double> source(RowLayout::Dense(2, 1)), destination(RowLayout::Dense(1, 1));
    auto binding = PayloadPlan(RowTransferPlan(MPI_COMM_WORLD, 2, 1, {{rank, 1, 0}}), source.Layout(), destination.Layout(), ElementSchema::For<double>()).Bind(source, destination);
    EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    CHECK_THROWS(binding.Enqueue(engine, stream));
    binding.Begin().Wait();
#ifndef DNDS_USE_CUDA
    CHECK_FALSE(NCCLTransferProvider::Available());
#endif
    if (!NCCLTransferProvider::Available())
        CHECK_THROWS(NCCLTransferProvider::Create(MPI_COMM_WORLD, Device::CUDA(0)));
}

TEST_CASE("Empty direct routes retain stream ordering without payload access or host readiness")
{
    struct DeferredEvent final : StreamEvent::State
    {
        mutable std::atomic<bool> ready{false}, hostWait{false};
        bool Ready() const override { return ready.load(); }
        void Wait() const override
        {
            hostWait.store(true);
            ready.store(true);
        }
    };
    for (const bool checked : {false, true})
    {
        EagerEngine engine(2, checked ? std::make_shared<AccessTrackingContext>() : nullptr);
        const auto communication = engine.CreateCPUStream();
        Array<double> source(RowLayout::Dense(1, 1)), destination(source.Layout());
        destination.Write().View()(0, 0) = 17;
        const auto version = destination.Version();
        auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
        RowTransferPlan route(context, 1, 1, {});
        auto binding = PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::For<double>())
                           .Bind(source, destination, TransferTransport::Direct);
        auto ready = std::make_shared<DeferredEvent>();
        // An unrelated immediate reader must not conflict: no destination row
        // is written by this route, despite its nonempty destination allocation.
        auto read = destination.Read();
        auto operation = binding.SubmitAfter(engine, communication, StreamEvent(ready));
        auto repeated = binding.SubmitAfter(engine, communication, StreamEvent(ready));
        CHECK_FALSE(ready->hostWait.load());
        ready->ready.store(true);
        ready->NotifyCompletion();
        operation.Wait();
        repeated.Wait();
        engine.Synchronize();
        CHECK(destination.Version() == version);
        CHECK(read.View()(0, 0) == 17);
        // Caller validity still participates in the ordinary collective checks.
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        CHECK_THROWS(binding.SubmitReady(engine, communication, rank != 0));
    }
}

TEST_CASE("Locally empty direct participants retain the nonempty peer preflight sequence")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 3)
        return;
    for (const bool checked : {false, true})
    {
        EagerEngine engine(2, checked ? std::make_shared<AccessTrackingContext>() : nullptr);
        const auto communication = engine.CreateCPUStream();
        Array<double> source(RowLayout::Dense(1, 1)), destination(source.Layout());
        source.Write().View()(0, 0) = rank + 10;
        auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
        const std::vector<RowRequest> requests = rank < 2 ? std::vector<RowRequest>{{1 - rank, 0, 0}} : std::vector<RowRequest>{};
        RowTransferPlan route(context, 1, 1, requests);
        auto binding = PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::For<double>())
                           .Bind(source, destination, TransferTransport::Direct);
        TransferOperation operation;
        {
            CountReductions count;
            operation = binding.SubmitReady(engine, communication);
        }
        CHECK(reductionNumber == (checked ? 3 : 1));
        operation.Wait();
        engine.Synchronize();
        if (rank < 2)
            CHECK(destination.Read().View()(0, 0) == 11 - rank);
        CHECK_THROWS(binding.SubmitReady(engine, communication, rank != size - 1));
    }
}
