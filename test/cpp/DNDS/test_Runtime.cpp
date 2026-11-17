#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/Storage/Buffer.hpp"
#include "DNDS/Runtime/Storage/AllocationResource.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Legacy/Array.hpp"
#include "DNDS/Runtime/Storage/Version.hpp"
#include "DNDS/Runtime/Array/Field.hpp"
#include "DNDS/Core/OptionalRef.hpp"
#include <functional>
#include <thread>
#include <type_traits>

using namespace DNDS::Runtime;

static_assert(std::is_constructible_v<DNDS::OptionalRef<const int>, DNDS::OptionalRef<int>>);
static_assert(!std::is_constructible_v<DNDS::OptionalRef<int>, DNDS::OptionalRef<const int>>);
static_assert(std::is_convertible_v<std::reference_wrapper<int>, DNDS::OptionalRef<int>>);
static_assert(std::is_convertible_v<std::reference_wrapper<const int>, DNDS::OptionalRef<const int>>);
static_assert(!std::is_convertible_v<int &, DNDS::OptionalRef<int>>);

TEST_CASE("OptionalRef preserves constness and checked synchronous rebinding")
{
    int first = 3, second = 7;
    DNDS::OptionalRef<int> value;
    CHECK_FALSE(value.HasValue());
    CHECK(value.Get() == nullptr);
    CHECK_THROWS(value.Value());
    CHECK_THROWS(*value);
    CHECK_THROWS(value.operator->());

    value.Emplace(first);
    CHECK(value.HasValue());
    CHECK(&value.Value() == &first);
    *value = 5;
    CHECK(first == 5);

    DNDS::OptionalRef<const int> readOnly(value);
    CHECK(readOnly.Value() == 5);
    DNDS::OptionalRef<const int> wrapped(std::cref(second));
    CHECK(wrapped.Value() == 7);

    value.emplace(second); // Retain the legacy spelling while callers migrate.
    CHECK(value.get() == &second);
    value.reset();
    CHECK_FALSE(value.has_value());
}

namespace
{
    struct ReleaseObservation
    {
        std::atomic<int> releases{0};
        std::atomic<bool> explicitContext{false};
    };
    class RecordedResource final : public AllocationResource
    {
        std::shared_ptr<ReleaseObservation> _observation;
        std::unique_ptr<int64_t[]> _values = std::make_unique<int64_t[]>(131 * 7);

    public:
        explicit RecordedResource(std::shared_ptr<ReleaseObservation> value) : _observation(std::move(value)) {}
        ~RecordedResource() override { Release(nullptr); }
        void *Data() const noexcept override { return _values.get(); }
        size_t Bytes() const noexcept override { return 131 * 7 * sizeof(int64_t); }
        Device GetDevice() const noexcept override { return Device::Host(); }
        void Release(const ExecutionContext *context) noexcept override
        {
            if (!_values)
                return;
            _observation->explicitContext = context != nullptr;
            _values.reset();
            ++_observation->releases;
        }
    };
}

TEST_CASE("Final allocation ownership passes only the actual final release context to its provider")
{
    auto observed = std::make_shared<ReleaseObservation>();
    auto a = Buffer::FromResource(std::make_unique<RecordedResource>(observed));
    auto b = a;
    CHECK(a.CanReleaseOrdered(ExecutionContext::Host()));
    CHECK_FALSE(a.CanReleaseOrdered(ExecutionContext::CUDA(0)));
    struct WrongReleaseStream final : StreamResource
    {
        void SynchronizeBackend() const override {}
        Device GetDevice() const noexcept override { return Device::CUDA(0); }
    };
    CHECK_THROWS(a.ReleaseOn({Device::Host(), std::make_shared<WrongReleaseStream>()}));
    CHECK(a.Identity() != 0);
    a.ReleaseOn(ExecutionContext::Host());
    CHECK(a.Identity() == 0);
    CHECK(observed->releases == 0);
    b.ReleaseOn(ExecutionContext::Host());
    CHECK(observed->releases == 1);
    CHECK(observed->explicitContext);
    b.ReleaseOn(ExecutionContext::Host());
    CHECK(observed->releases == 1);

    auto fallback = std::make_shared<ReleaseObservation>();
    a = Buffer::FromResource(std::make_unique<RecordedResource>(fallback));
    auto allocation = a.AllocationHandle();
    a.ReleaseOn(ExecutionContext::Host());
    CHECK(fallback->releases == 0);
    allocation = {}; // A surviving ordinary owner uses its conservative fallback.
    CHECK(fallback->releases == 1);
    CHECK_FALSE(fallback->explicitContext);

    auto pinned = std::make_shared<ReleaseObservation>();
    a = Buffer::FromResource(std::make_unique<RecordedResource>(pinned));
    auto read = a.Read();
    a.ReleaseOn(ExecutionContext::Host());
    CHECK(pinned->releases == 0);
    CHECK(read.View<int64_t>().size == 131 * 7);
    read.Wait();
    CHECK(pinned->releases == 1);
    CHECK_FALSE(pinned->explicitContext);
}

TEST_CASE("Concurrent contextual releases never free a shared allocation more than once")
{
    auto observed = std::make_shared<ReleaseObservation>();
    auto buffer = Buffer::FromResource(std::make_unique<RecordedResource>(observed));
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i)
        threads.emplace_back([buffer]() mutable
                             {
            for (int j = 0; j < 97; ++j)
            {
                auto copy = buffer;
                copy.ReleaseOn(ExecutionContext::Host());
            }
            buffer.ReleaseOn(ExecutionContext::Host()); });
    buffer.ReleaseOn(ExecutionContext::Host());
    for (auto &thread : threads)
        thread.join();
    CHECK(observed->releases == 1);
    CHECK(observed->explicitContext);
}

TEST_CASE("Async Array construction releases a rejected provider allocation with its explicit context")
{
    struct WrongShape : StreamLauncher
    {
        std::shared_ptr<ReleaseObservation> observation;
        explicit WrongShape(std::shared_ptr<ReleaseObservation> value) : observation(std::move(value)) {}
        ExecutionContext Context() const override { return ExecutionContext::Host(); }
        void Submit(Work, CommandKind, Retirement) override { throw std::runtime_error("unused"); }
        void WaitFor(StreamEvent) override {}
        StreamEvent Record() override { return {}; }
        void Drain() override {}
        Buffer Allocate(size_t) override { return Buffer::FromResource(std::make_unique<RecordedResource>(observation)); }
    };
    auto observed = std::make_shared<ReleaseObservation>();
    EagerEngine engine;
    auto stream = engine.CreateStream(std::make_shared<WrongShape>(observed));
    CHECK_THROWS(AllocateArrayAsync<int64_t>(engine, stream, RowLayout::Dense(1, 1)));
    CHECK(observed->releases == 1);
    CHECK(observed->explicitContext);
}

TEST_CASE("Allocator policy and pooled storage outlive asynchronous access owners")
{
    struct Pool : Allocator
    {
        std::shared_ptr<void> Allocate(size_t bytes, Device device) override
        {
            DNDS_check_throw(device == Device::Host());
            return {new std::byte[bytes](), [](void *p)
                    { delete[] static_cast<std::byte *>(p); }};
        }
    };
    auto pool = std::make_shared<Pool>();
    std::weak_ptr<Pool> alive = pool;
    auto buffer = Buffer::Allocate(16, Device::Host(), pool);
    auto read = buffer.Read();
    pool.reset();
    buffer = {};
    CHECK_FALSE(alive.expired());
    read.Wait();
    CHECK(alive.expired());
}

TEST_CASE("Field keeps association and distribution outside allocation")
{
    Field<double> a({3, 5, 8}, {42, 7, 2, 1}, "cell");
    Field<double> b({3, 5, 8}, {42, 7, 2, 1}, "cell");
    a.Write().View<double>()[8] = 2;
    b.CopyFrom(a).Wait();
    CHECK(b.Read().View<double>()[8] == 2);
    Field<double> other({3, 5, 8}, {42, 8, 2, 1}, "cell");
    CHECK_THROWS(other.CopyFrom(a));
    CHECK_THROWS(Field<double>({3, 5, 4}, {42, 7, 2, 1}, "cell"));
    CHECK_THROWS(Field<double>({3, 5, 8}, {42, 7, 3, 1}, "cell"));
}

TEST_CASE("Mutation versions invalidate selected dependencies and fail closed")
{
    VersionTracker state;
    auto geometry = state.Stamp(Change::Geometry);
    auto layout = state.Stamp(Change::Layout);
    {
        auto read = state.Read();
        CHECK_FALSE(state.CanMutate());
        CHECK_THROWS(state.BeginMutation(Change::Geometry));
    }
    {
        auto edit = state.BeginMutation(Change::Layout);
        CHECK_FALSE(geometry.Valid());
        CHECK_THROWS(state.Read());
        edit.Commit();
    }
    CHECK(geometry.Valid());
    CHECK_FALSE(layout.Valid());
    {
        auto failed = state.BeginMutation(Change::Geometry);
        // An exception/abandoned destructive update must not publish valid caches.
    }
    CHECK_FALSE(geometry.Valid());
    CHECK_FALSE(state.CanMutate());
    CHECK_THROWS(state.Stamp());
}

TEST_CASE("Shallow mesh controller aliases share exclusion and invalidation")
{
    VersionTracker original;
    auto alias = original;
    auto stamp = original.Stamp(Change::Geometry);
    {
        auto reader = alias.Read();
        CHECK_THROWS(original.BeginMutation(Change::Geometry));
    }
    alias.BeginMutation(Change::Geometry).Commit();
    CHECK_FALSE(stamp.Valid());
    VersionTracker replaced;
    auto old = replaced.Stamp();
    replaced = original;
    CHECK_FALSE(old.Valid());
    auto read = replaced.Read();
    CHECK_THROWS(original.BeginMutation(Change::Layout));
    CHECK_THROWS(original.Stamp(Change(128)));
}

TEST_CASE("Runtime aliases share access state and versions, copies do not")
{
    auto a = Buffer::Allocate(4 * sizeof(double));
    auto alias = a;
    auto b = Buffer::Allocate(a.Bytes());
    {
        auto write = a.Write();
        write.View<double>()[2] = 7;
        CHECK_THROWS(alias.Read());
        CHECK_THROWS(alias.Write());
        CHECK(a.Version() == 0);
    }
    CHECK(alias.Version() == 1);
    {
        auto r1 = a.Read();
        auto r2 = alias.Read();
        CHECK(r2.View<double>()[2] == 7);
        CHECK_THROWS(a.Write());
    }
    Copy(a, b).Wait();
    CHECK(b.Identity() != a.Identity());
    CHECK(b.Read().View<double>()[2] == 7);
    CHECK_THROWS(Copy(a, alias));
    struct WrongDeviceStream final : StreamResource
    {
        Device GetDevice() const noexcept override { return Device::CUDA(0); }
        void SynchronizeBackend() const override {}
    };
    CHECK_THROWS(Copy(a, b, ExecutionContext{Device::Host(), std::make_shared<WrongDeviceStream>()}));
}

TEST_CASE("Runtime access owns allocation after handle replacement")
{
    auto a = Buffer::Allocate(2 * sizeof(double));
    auto old = a.Write();
    old.View<double>()[0] = 11;
    a = Buffer::Allocate(sizeof(double));
    CHECK(a.Read().View<double>()[0] == 0);
    CHECK(old.View<double>()[0] == 11);
    old.Wait();
    CHECK_THROWS(old.View<double>());
    CHECK_THROWS(a.Read().View<uint64_t>()[1]);
    auto empty = Buffer::Allocate(0);
    CHECK(empty.Read().View<double>().size == 0);
    auto odd = Buffer::Allocate(3);
    CHECK_THROWS(odd.Read().View<double>());
#ifndef DNDS_USE_CUDA
    CHECK_THROWS(Buffer::Allocate(1, Device::CUDA(0)));
#endif
    CHECK_THROWS(Buffer::Allocate(1, Device{MemorySpace::Host, 0}));
}

TEST_CASE("Legacy Array shares native host allocation without changing row API")
{
    DNDS::Array<double, 3> array;
    array.Resize(2);
    const auto identity = array.HostDataIdentity();
    {
        auto write = array.HostWriteData();
        write.View<double>()[4] = 23;
        CHECK_THROWS(array.HostReadData());
        CHECK_THROWS(array.to_device(DNDS::DeviceBackend::Host));
    }
    CHECK(array(1, 1) == 23);
    CHECK(array.HostDataVersion() == 1);
    auto read = array.HostReadData();
    auto copy = array;
    CHECK(copy.HostDataIdentity() != identity);
    CHECK(copy(1, 1) == 23);
    array.Resize(4);
    CHECK(array.HostDataIdentity() != identity);
    CHECK(read.View<double>()[4] == 23);
}
