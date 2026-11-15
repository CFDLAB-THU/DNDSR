#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/Storage/Buffer.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Array/ArrayCopy.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Legacy/Array.hpp"
#include <cuda_runtime.h>
#include <map>
#include <atomic>
#include <thread>

using namespace DNDS::Runtime;

TEST_CASE("Pinned host snapshots retain DMA leases and are accounted once")
{
    AllocationDiagnostics::Enable();
    const auto baseline = AllocationDiagnostics::Snapshot().pinnedHostBytes;
    constexpr DNDS::index rows = 257;
    {
        auto target = Array<double>::FromStorage(RowLayout::Dense(rows, 1), Buffer::AllocatePinnedHost(rows * sizeof(double)));
        target.SetName("test/pinnedSnapshot", DNDS_ALLOCATION_SITE());
        const auto alias = target;
        CHECK(AllocationDiagnostics::Snapshot().pinnedHostBytes == baseline + rows * sizeof(double));
        {
            auto read = target.Read();
            cudaPointerAttributes attributes{};
            REQUIRE(cudaPointerGetAttributes(&attributes, read.View().data) == cudaSuccess);
            CHECK(attributes.type == cudaMemoryTypeHost);
            for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                CHECK(read.View()(iRow, 0) == 0);
        }
        Array<double> seed(RowLayout::Dense(rows, 1));
        {
            auto write = seed.Write();
            for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                write.View()(iRow, 0) = iRow + .25;
        }
        const auto context = ExecutionContext::CUDA(0);
        auto device = seed.To(Device::CUDA(0), context);
        auto copy = target.CopyFrom(device, context);
        // Completion may already have occurred physically, but access remains
        // reserved until this operation publishes/retires its leases.
        CHECK_THROWS(alias.Read());
        copy.Wait();
        const auto read = alias.Read();
        for (DNDS::index iRow = 0; iRow < rows; ++iRow)
            CHECK(read.View()(iRow, 0) == iRow + .25);
    }
    CHECK(AllocationDiagnostics::Snapshot().pinnedHostBytes == baseline);
    AllocationDiagnostics::Enable(false);
}

__global__ void ApplyMetadataWeights(DNDS::SpanView<const double> weights,
                                     ArrayView<double, AccessBoundsChecks, FixedRows<1>> output)
{
    const DNDS::index iRow = blockIdx.x * blockDim.x + threadIdx.x;
    if (iRow >= output.RowCount())
        return;
    double value = 0;
    for (DNDS::index j = 0; j < weights.size(); ++j)
        value += weights[j] * double(iRow + j);
    output(iRow, 0) = value;
}

TEST_CASE("Backend metadata spans survive asynchronous launch after handles are replaced")
{
    const auto context = ExecutionContext::CUDA(0);
    const MetadataBuffer<double> host(std::vector<double>{1, 2, 4});
    auto placed = host.CopyTo(Device::CUDA(0), context);
    CHECK_THROWS(placed.HostView());
    const auto roundTrip = placed.CopyTo(Device::Host(), context);
    CHECK(roundTrip == host);
    Array<double> output(RowLayout::Dense(128, 1), Device::CUDA(0));
    {
        auto read = placed.Read(context);
        auto write = output.Write(context);
        const auto values = read.View<double>();
        ApplyMetadataWeights<<<4, 32>>>(
            {values.data, DNDS::index(values.size)},
            write.View<FixedRows<1>>());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        const auto done = Completion::Record(context);
        read.Complete(done);
        write.Complete(done);
        placed = MetadataBuffer<double>{};
        read.Wait();
        write.Wait();
    }
    const auto result = output.To(Device::Host(), context);
    const auto read = result.Read();
    const auto values = read.View<FixedRows<1>>();
    for (DNDS::index iRow = 0; iRow < values.RowCount(); ++iRow)
        CHECK(values(iRow, 0) == 7 * iRow + 10);
}

template <class Policy>
__global__ void FillStaticRows(ArrayView<double, AccessBoundsChecks, Policy> values)
{
    static_assert(!decltype(values)::is_dynamic);
    static_assert(std::is_trivially_copyable_v<decltype(values)>);
    for (size_t row = blockIdx.x; row < values.RowCount(); row += gridDim.x)
    {
        auto entries = values[row];
        for (size_t column = threadIdx.x; column < entries.size(); column += blockDim.x)
            entries[column] = double(100 * row + column + 1);
    }
}

template <class Policy>
__global__ void ReadStaticRows(ArrayView<const double, AccessBoundsChecks, Policy> values, ArrayView<double> sums)
{
    static_assert(std::is_same_v<decltype(values[0][0]), const double &>);
    if (threadIdx.x)
        return;
    for (size_t row = blockIdx.x; row < values.RowCount(); row += gridDim.x)
    {
        double sum = 0;
        for (auto value : values[row])
            sum += value;
        sums(row, 0) = sum;
    }
}

template <class Policy>
static void CheckStaticCUDAAccess(RowLayout shape)
{
    Array<double, Policy> host(shape);
    {
        auto raw = host.Storage().Write();
        const auto flat = raw.template View<double>();
        for (size_t i = 0; i < flat.size; ++i)
            flat[i] = -51;
    }
    const auto context = ExecutionContext::CUDA(0);
    auto device = host.To(Device::CUDA(0), context);
    auto retained = device;
    {
        auto write = device.Write(context);
        FillStaticRows<Policy><<<3, 32>>>(write.template View<Policy>());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        write.Complete(Completion::Record(context));
        device.Reset(Policy::Default());
        write.Wait();
    }
    Array<double, FixedRows<1>> sums(RowLayout::Dense(shape.RowCount(), 1), Device::CUDA(0));
    {
        auto read = retained.Read(context);
        auto write = sums.Write(context);
        ReadStaticRows<Policy><<<3, 32>>>(read.template View<Policy>(), write.template View<FixedRows<1>>());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        const auto done = Completion::Record(context);
        read.Complete(done);
        write.Complete(done);
        read.Wait();
        write.Wait();
    }
    auto actual = retained.To(Device::Host(), context);
    auto hostSums = sums.To(Device::Host(), context);
    auto raw = actual.Storage().Read();
    auto logical = actual.Read();
    auto totals = hostSums.Read();
    const auto flat = raw.template View<double>();
    for (size_t row = 0; row < shape.RowCount(); ++row)
    {
        double sum = 0;
        CHECK(logical.View()[row].size() == shape.RowSize(row));
        for (size_t column = 0; column < shape.RowSize(row); ++column)
        {
            const double expected = double(100 * row + column + 1);
            CHECK(logical.View()(row, column) == expected);
            sum += expected;
        }
        if (shape.IsContiguousRows() && !shape.IsRagged())
            for (size_t column = shape.RowSize(row); column < shape.Stride(); ++column)
                CHECK(flat[row * shape.Stride() + column] == -51);
        CHECK(totals.View()(row, 0) == sum);
    }
    if (!shape.IsContiguousRows())
        for (DNDS::index iElement = 0; iElement < DNDS::index(shape.Elements()); ++iElement)
        {
            const auto iRow = shape.Kind() == RowLayoutKind::ColMajorTable ? iElement % shape.Stride() : (iElement / (shape.Width() * shape.Stride())) * shape.Stride() + iElement % shape.Stride();
            if (iRow >= shape.RowCount())
                CHECK(flat.data[iElement] == -51);
        }
}

TEST_CASE("CUDA five layout policies static width stride and capacity reach actual kernels")
{
    CheckStaticCUDAAccess<TableRows>(RowLayout::Dense(3, 3, 7));
    CheckStaticCUDAAccess<FixedRows<3>>(RowLayout::Dense(3, 3, 7));
    CheckStaticCUDAAccess<FixedRows<3, 7>>(RowLayout::Dense(3, 3, 7));
    CheckStaticCUDAAccess<PaddedRows>(RowLayout::Padded({0, 3, 7}, 7));
    CheckStaticCUDAAccess<StaticMaxRows<7>>(RowLayout::Padded({0, 3, 7}, 7));
    CheckStaticCUDAAccess<CSRRows>(RowLayout::Ragged({0, 3, 7}));
    CheckStaticCUDAAccess<TableRows>(RowLayout::Dense(0, 3, 7));
    CheckStaticCUDAAccess<FixedRows<3, 7>>(RowLayout::Dense(0, 3, 7));
    CheckStaticCUDAAccess<PaddedRows>(RowLayout::Padded({}, 7));
    CheckStaticCUDAAccess<StaticMaxRows<7>>(RowLayout::Padded({}, 7));
    CheckStaticCUDAAccess<CSRRows>(RowLayout::Ragged({}));
    // These projections must inspect the host shape, never dereference CUDA
    // offsets/sizes during host-side view construction.
    CheckStaticCUDAAccess<FixedRows<3, 3>>(RowLayout::Ragged({3, 3, 3}));
    CheckStaticCUDAAccess<TableRows>(RowLayout::Padded({3, 3, 3}, 3));
    Array<double> ragged(RowLayout::Ragged({2, 3}), Device::CUDA(0));
    Array<double> padded(RowLayout::Padded({2, 2}, 3), Device::CUDA(0));
    CHECK_THROWS(ragged.Read(ExecutionContext::CUDA(0)).template View<TableRows>());
    CHECK_THROWS(padded.Read(ExecutionContext::CUDA(0)).template View<TableRows>());
}

template <class Policy>
static void CheckEagerTypedCUDA(RowLayout shape)
{
    EagerEngine engine(2);
    auto stream = engine.CreateStream(std::make_shared<CUDALauncher>(0));
    Array<double, Policy> values(shape, Device::CUDA(0));
    auto retained = values;
    Array<double, FixedRows<1>> sums(RowLayout::Dense(shape.RowCount(), 1), Device::CUDA(0));
    SubmitArrays(engine, stream, [](ExecutionContext context, auto output)
                 {
        static_assert(std::is_same_v<decltype(output), ArrayView<double, AccessBoundsChecks, Policy>>);
        FillStaticRows<Policy><<<3, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(output);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "typed eager fill launch failed"); }, WriteAs<Policy>(values));
    SubmitArrays(engine, stream, [](ExecutionContext context, auto input, auto output)
                 {
        static_assert(std::is_same_v<decltype(input), ArrayView<const double, AccessBoundsChecks, Policy>>);
        static_assert(decltype(output)::row_width == 1);
        ReadStaticRows<Policy><<<3, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(input, output);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "typed eager read launch failed"); }, ReadAs<Policy>(values), WriteAs<FixedRows<1>>(sums));
    values.Reset(Policy::Default());
    stream.Record().WaitHost();
    auto actual = sums.To(Device::Host(), stream.Context());
    auto source = retained.To(Device::Host(), stream.Context());
    auto read = actual.Read();
    auto input = source.Read();
    for (size_t row = 0; row < shape.RowCount(); ++row)
    {
        double expected = 0;
        for (size_t column = 0; column < shape.RowSize(row); ++column)
        {
            expected += double(100 * row + column + 1);
            CHECK(input.View()(row, column) == double(100 * row + column + 1));
        }
        CHECK(read.View()(row, 0) == expected);
    }
}

TEST_CASE("CUDA eager submissions preserve all five layout policies in launcher and device views")
{
    CheckEagerTypedCUDA<TableRows>(RowLayout::Dense(3, 3, 5));
    CheckEagerTypedCUDA<FixedRows<3, 5>>(RowLayout::Dense(3, 3, 5));
    CheckEagerTypedCUDA<PaddedRows>(RowLayout::Padded({0, 2, 5}, 5));
    CheckEagerTypedCUDA<StaticMaxRows<5>>(RowLayout::Padded({0, 2, 5}, 5));
    CheckEagerTypedCUDA<CSRRows>(RowLayout::Ragged({0, 2, 5}));
    CheckEagerTypedCUDA<FixedRows<3, 3>>(RowLayout::Ragged({3, 3, 3}));
    CheckEagerTypedCUDA<TableRows>(RowLayout::Padded({3, 3, 3}, 3));
}

TEST_CASE("CUDA array allocation accounting includes host and device row metadata without alias duplication")
{
    Array<double> csr(RowLayout::Ragged({2, 0, 3}), Device::CUDA(0));
    auto alias = csr;
    auto other = Array<double>::FromSharedStorage(csr.SharedStorage()->WithValues(Buffer::Allocate(5 * sizeof(double), Device::CUDA(0)), sizeof(double)));
    Array<int> padded(RowLayout::Padded({1, 3}, 5), Device::CUDA(0));
    std::map<uint64_t, size_t> allocations;
    size_t host = 0, device = 0;
    auto visit = [&](uint64_t identity, size_t capacity, Device placement)
    {
        if (allocations.emplace(identity, capacity).second)
            (placement == Device::Host() ? host : device) += capacity;
    };
    csr.SharedStorage()->VisitAllocations(visit);
    alias.SharedStorage()->VisitAllocations(visit);
    other.SharedStorage()->VisitAllocations(visit);
    padded.SharedStorage()->VisitAllocations(visit);
    CHECK(host == 6 * sizeof(size_t));
    CHECK(device == 2 * 5 * sizeof(double) + 2 * 5 * sizeof(int) + 6 * sizeof(size_t));
    CHECK(allocations.size() == 7);
}

__global__ void FillRows(ArrayView<double> values)
{
    for (size_t i = blockIdx.x; i < values.RowCount(); i += gridDim.x)
    {
        auto row = values[i];
        for (size_t j = threadIdx.x; j < row.size(); j += blockDim.x)
            row[j] = 100 * i + j;
    }
}

TEST_CASE("Native ragged Array CUDA views retain metadata through replacement")
{
    DNDS::Runtime::Array<double> values(RowLayout::Ragged({2, 0, 513}), Device::CUDA(0));
    auto retained = values;
    auto shared = Array<double>::FromSharedStorage(values.SharedStorage());
    CHECK(shared.SharedStorage() == values.SharedStorage());
    CHECK(shared.SharedStorage()->SharedLayout() == values.SharedStorage()->SharedLayout());
    CHECK(shared.SharedStorage()->SharedLayout()->DeviceOffsetIdentity() == values.SharedStorage()->SharedLayout()->DeviceOffsetIdentity());
    auto ctx = ExecutionContext::CUDA(0);
    {
        auto write = values.Write(ctx);
        FillRows<<<3, 128>>>(write.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        write.Complete(Completion::Record(ctx));
        values.Reset(RowLayout::Dense(1, 1));
        CHECK_THROWS(retained.Read(ctx));
        write.Wait();
    }
    auto host = retained.To(Device::Host(), ctx);
    CHECK(host.Read().At(2, 512) == 712);
    CHECK(host.RowSize(1) == 0);
    auto copy = retained.Clone(ctx);
    CHECK(copy.Identity() != retained.Identity());
    CHECK(copy.To(Device::Host(), ctx).Read().At(0, 1) == 1);
}

TEST_CASE("Padded CUDA views use logical row sizes and preserve capacity padding")
{
    cudaStream_t stream;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    const auto context = ExecutionContext::CUDA(0, stream);
    {
        Array<double> values(RowLayout::Padded({0, 5, 7, 33, 0, 19}, 40), Device::CUDA(0));
        auto retained = values;
        auto write = values.Write(context);
        FillRows<<<3, 64, 0, stream>>>(write.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        write.Complete(Completion::Record(context));
        values.Reset(RowLayout::Dense(0, 0));
        write.Wait();
        auto host = retained.To(Device::Host(), context);
        CHECK(host.Layout().IsPadded());
        CHECK(host.Read().At(3, 32) == 332);
        CHECK_THROWS(host.Read().At(3, 33));
        auto raw = host.Storage().Read();
        auto flat = raw.View<double>();
        for (size_t i = 0; i < host.RowCount(); ++i)
            for (size_t j = host.RowSize(i); j < 40; ++j)
                CHECK(flat[i * 40 + j] == 0);
    }
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

struct RowEntry
{
    int64_t id;
    double weight;
};

__global__ void CopyPairRows(PairReadView<RowEntry, true, FixedRows<2>> input, ArrayView<RowEntry, true, FixedRows<2>> output)
{
    static_assert(decltype(input)::row_width == 2 && decltype(output)::row_width == 2);
    static_assert(decltype(input[0])::row_width == 2);
    for (size_t row = blockIdx.x; row < input.RowCount(); row += gridDim.x)
    {
        const auto source = input[row];
        auto target = output[row];
        for (size_t column = threadIdx.x; column < source.size(); column += blockDim.x)
        {
            target[column] = source.cbegin()[column];
            target[column].weight += target[column].id;
        }
    }
}

TEST_CASE("Fixed POD CUDA rows use sized owned and combined read spans")
{
    ArrayPair<RowEntry, FixedRows<2>> host;
    host.owned.Reset(RowLayout::Dense(1, 2, 3));
    host.ghost.Reset(RowLayout::Dense(1, 2));
    {
        auto owned = host.owned.Write();
        auto ghost = host.ghost.Write();
        owned.View()[0][0] = {1, 1.5};
        owned.View()[0][1] = {2, 2.5};
        ghost.View()[0][0] = {3, 3.5};
        ghost.View()[0][1] = {4, 4.5};
    }
    const auto context = ExecutionContext::CUDA(0);
    ArrayPair<RowEntry, FixedRows<2>> device{host.owned.To(Device::CUDA(0)), host.ghost.To(Device::CUDA(0))};
    Array<RowEntry, FixedRows<2>> output(RowLayout::Dense(2, 2, 4), Device::CUDA(0));
    auto read = device.ReadCombined(context);
    auto write = output.Write(context);
    CopyPairRows<<<2, 32>>>(read.View<FixedRows<2>, true>(), write.View<FixedRows<2>, true>());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    auto done = Completion::Record(context);
    read.Complete(done);
    write.Complete(done);
    device.owned.Reset(RowLayout::Dense(0, 2));
    device.ghost.Reset(RowLayout::Dense(0, 2));
    read.Wait();
    write.Wait();
    auto result = output.To(Device::Host(), context);
    auto values = result.Read();
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 2; ++column)
        {
            const auto id = int64_t(2 * row + column + 1);
            CHECK(values.View()[row][column].id == id);
            CHECK(values.View()[row][column].weight == 2 * id + 0.5);
        }
}

__global__ void Scale(Span<const double> input, Span<double> output)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < input.size)
        output[i] = 3 * input[i];
}

TEST_CASE("External CUDA allocation and shared CSR metadata survive pending kernels")
{
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    cudaStream_t stream;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    const auto context = ExecutionContext::CUDA(0, stream);
    int releases = 0;
    bool releaseOK = true;
    {
        const auto shape = RowLayout::Ragged({0, 5, 19, 0, 513, 7});
        void *pointer = nullptr;
        const auto bytes = shape.Elements() * sizeof(double);
        REQUIRE(cudaMalloc(&pointer, bytes) == cudaSuccess);
        auto owner = std::shared_ptr<void>(pointer, [&releases, &releaseOK](void *p)
                                           { ++releases; releaseOK = cudaFree(p) == cudaSuccess; });
        auto allocation = Allocation::Import(owner, bytes, Device::CUDA(0));
        auto array = Array<double>::FromStorage(shape, Buffer::FromAllocation(allocation, 0, bytes));
        auto second = Array<double>::FromSharedStorage(array.SharedStorage()->WithValues(Buffer::Allocate(bytes, Device::CUDA(0)), sizeof(double)));
        CHECK(second.SharedStorage()->SharedLayout() == array.SharedStorage()->SharedLayout());
        CHECK(second.SharedStorage()->SharedLayout()->DeviceOffsetIdentity() == array.SharedStorage()->SharedLayout()->DeviceOffsetIdentity());
        auto write = array.Write(context);
        FillRows<<<7, 128, 0, stream>>>(write.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        write.Complete(Completion::Record(context));
        auto retained = array;
        array.Reset(RowLayout::Dense(0, 0));
        owner.reset();
        allocation = {};
        CHECK(releases == 0);
        write.Wait();
        auto host = retained.To(Device::Host(), context);
        CHECK(host.Read().At(4, 512) == 912);
        CHECK(host.RowSize(0) == 0);
        retained.Reset(RowLayout::Dense(0, 0));
        CHECK(releases == 1);
        auto edit = second.Write(context);
        FillRows<<<3, 64, 0, stream>>>(edit.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        edit.Complete(Completion::Record(context));
        edit.Wait();
        CHECK(second.To(Device::Host(), context).Read().At(5, 6) == 506);
    }
    CHECK(releaseOK);
    CHECK(releases == 1);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("Runtime CUDA residency, cross-stream transfer and retained leases")
{
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    cudaStream_t stream1, stream2;
    REQUIRE(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking) == cudaSuccess);
    {
        constexpr size_t count = 513;
        auto h = Buffer::Allocate(count * sizeof(double));
        {
            auto w = h.Write();
            for (size_t i = 0; i < count; ++i)
                w.View<double>()[i] = double(i);
        }
        auto d = Buffer::Allocate(h.Bytes(), Device::CUDA(0));
        auto out = Buffer::Allocate(h.Bytes(), Device::CUDA(0));
        auto c1 = ExecutionContext::CUDA(0, stream1);
        auto c2 = ExecutionContext::CUDA(0, stream2);
#if CUDART_VERSION >= 13000
        int deviceCount = 0;
        REQUIRE(cudaGetDeviceCount(&deviceCount) == cudaSuccess);
        if (deviceCount > 1)
        {
            auto second = Buffer::Allocate(8, Device::CUDA(1));
            CHECK_THROWS(second.Read(ExecutionContext::CUDA(1, stream1)));
        }
#endif
        {
            auto write = d.Write(c1);
            CHECK_THROWS(write.Complete(Completion::Record(c2)));
        }
        CHECK_THROWS(d.Read());
        CHECK_THROWS(d.Write(ExecutionContext::CUDA(1)));
        auto upload = Copy(h, d, c1);
        CHECK_THROWS(h.Write());
        CHECK_THROWS(d.Read(c1));
        upload.Wait();
        auto r = d.Read(c2);
        auto w = out.Write(c2);
        CHECK_THROWS(r.View<double>()[0]);
        Scale<<<3, 256, 0, stream2>>>(r.View<double>(), w.View<double>());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        auto done = Completion::Record(c2);
        r.Complete(done);
        w.Complete(done);
        CHECK_THROWS(w.View<double>());
        d = {}; // r still owns the in-flight input allocation
        r.Wait();
        w.Wait();
        auto result = Buffer::Allocate(h.Bytes());
        auto download = Copy(out, result, c1);
        out = {}; // download retains both allocations
        download.Wait();
        auto read = result.Read();
        for (size_t i = 0; i < count; ++i)
            CHECK(read.View<double>()[i] == 3 * double(i));
    }
    REQUIRE(cudaStreamDestroy(stream1) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(stream2) == cudaSuccess);
}

TEST_CASE("Copied CUDA completions share terminal observation without releasing attached leases")
{
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    const auto context = ExecutionContext::CUDA(0, stream);
    std::atomic<int> released{0};
    std::weak_ptr<void> lifetime;
    {
        constexpr DNDS::index rows = 257;
        void *pointer = nullptr;
        REQUIRE(cudaMalloc(&pointer, rows * sizeof(double)) == cudaSuccess);
        auto owner = std::shared_ptr<void>(pointer, [&released](void *value)
                                           {
                                               if (cudaFree(value) != cudaSuccess)
                                                   std::terminate();
                                               ++released; });
        lifetime = owner;
        auto values = Array<double>::FromStorage(RowLayout::Dense(rows, 1),
                                                 Buffer::FromAllocation(Allocation::Import(owner, rows * sizeof(double), Device::CUDA(0)),
                                                                        0, rows * sizeof(double)));
        auto write = values.Write(context);
        FillRows<<<7, 64, 0, stream>>>(write.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        const auto done = Completion::Record(context);
        write.Complete(done);
        CHECK_THROWS(values.Read(context));
        owner.reset();
        values = {};
        std::atomic<bool> failed{false};
        std::vector<std::thread> observers;
        for (int i = 0; i < 4; ++i)
            observers.emplace_back([completion = done, &failed]
                                   {
                                       try
                                       {
                                           for (int repeat = 0; repeat < 8; ++repeat)
                                           {
                                               completion.Wait();
                                               if (!completion.Ready())
                                                   failed = true;
                                           }
                                       }
                                       catch (...)
                                       {
                                           failed = true;
                                       } });
        for (auto &observer : observers)
            observer.join();
        CHECK_FALSE(failed.load());
        CHECK_FALSE(lifetime.expired()); // Observing completion does not retire its lease.
        CHECK(released.load() == 0);
        write.Wait(); // Reuses the terminal result, then releases the allocation.
        CHECK(lifetime.expired());
        CHECK(released.load() == 1);
        CHECK(done.Ready());
        CHECK_NOTHROW(done.Wait());
    }
    CHECK(released.load() == 1);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("Legacy device mirror cannot supply stale native host access")
{
    DNDS::Array<double, 1> array;
    array.Resize(2);
    array(0, 0) = 5;
    array.to_device(DNDS::DeviceBackend::CUDA);
    CHECK_THROWS(array.HostReadData());
    CHECK_THROWS(array.HostWriteData());
    auto before = array.HostDataVersion();
    array.to_host();
    array.clear_device();
    CHECK(array.HostDataVersion() > before);
    CHECK(array.HostReadData().View<double>()[0] == 5);
}

TEST_CASE("Runtime CUDA destructor fences an unsubmitted launch")
{
    auto a = Buffer::Allocate(128 * sizeof(double), Device::CUDA(0));
    auto b = Buffer::Allocate(a.Bytes(), Device::CUDA(0));
    auto context = ExecutionContext::CUDA(0);
    {
        auto r = a.Read(context);
        auto w = b.Write(context);
        Scale<<<1, 128>>>(r.View<double>(), w.View<double>());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        // No Complete: lease destructors must fence before allowing buffer reuse.
    }
    CHECK(a.IsIdle());
    CHECK(b.Version() == 1);
    auto empty = Buffer::Allocate(0, Device::CUDA(0));
    auto emptyHost = Buffer::Allocate(0);
    Copy(empty, emptyHost, context).Wait();
}

TEST_CASE("CUDA column and tiled row views preserve physical padding and static strides")
{
    CheckStaticCUDAAccess<ColMajorTable>(RowLayout::ColMajor(33, 5, {32, 1}));
    CheckStaticCUDAAccess<FixedWidthRows<ColMajorTable, 5>>(RowLayout::ColMajor(9, 5, {8, 1}));
    CheckStaticCUDAAccess<TiledColMajorTable<8>>(RowLayout::TiledColMajor(33, 21, 8));
    CheckStaticCUDAAccess<ColMajorTable>(RowLayout::ColMajor(0, 5, {32, 1}));
    CheckStaticCUDAAccess<TiledColMajorTable<8>>(RowLayout::TiledColMajor(0, 21, 8));
    CheckEagerTypedCUDA<ColMajorTable>(RowLayout::ColMajor(33, 5, {32, 1}));
    CheckEagerTypedCUDA<TiledColMajorTable<8>>(RowLayout::TiledColMajor(33, 21, 8));
}

TEST_CASE("CUDA explicit format conversion never reads device row metadata on the host")
{
    using I = DNDS::index;
    const auto context = ExecutionContext::CUDA(0);
    Array<double> source(RowLayout::Dense(9, 5));
    {
        auto write = source.Write();
        for (I iRow = 0; iRow < 9; ++iRow)
            for (I j = 0; j < 5; ++j)
                write.View()(iRow, j) = 100 * iRow + j + 1;
    }
    auto current = source.To(Device::CUDA(0));
    for (const auto &shape : std::vector<RowLayout>{RowLayout::ColMajor(9, 5, {8, 1}), RowLayout::TiledColMajor(9, 5, 8),
                                                    RowLayout::Ragged(std::vector<size_t>(9, 5)), RowLayout::Padded(std::vector<size_t>(9, 5), 7), RowLayout::Dense(9, 5, 8)})
    {
        current = ConvertLayout(current, shape);
        const auto host = current.To(Device::Host());
        const auto read = host.Read();
        for (I iRow = 0; iRow < 9; ++iRow)
            for (I j = 0; j < 5; ++j)
                CHECK(read.View()(iRow, j) == 100 * iRow + j + 1);
    }
    Array<double> ragged(RowLayout::Ragged({0, 2, 5}));
    {
        auto write = ragged.Write();
        for (I iRow = 0; iRow < 3; ++iRow)
            for (I j = 0; j < I(ragged.RowSize(iRow)); ++j)
                write.View()(iRow, j) = 10 * iRow + j;
    }
    auto device = ConvertLayout(ragged, RowLayout::Padded({0, 2, 5}, 7), Device::CUDA(0), context);
    auto packed = ConvertLayout(device, RowLayout::Ragged({0, 2, 5}));
    const auto host = ConvertLayout(packed, RowLayout::Padded({0, 2, 5}, 8), Device::Host(), context);
    const auto read = host.Read();
    for (I iRow = 0; iRow < 3; ++iRow)
        for (I j = 0; j < I(host.RowSize(iRow)); ++j)
            CHECK(read.View()(iRow, j) == 10 * iRow + j);
}
