#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"
#include "DNDS/Runtime/Execution/EagerCopy.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include <atomic>
#include <thread>
#include <stdexcept>
#include <chrono>

namespace R = DNDS::Runtime;

TEST_CASE("Array submission captures survive launch callback disposal until retirement")
{
    // Simulate a provider that submits immediately but destroys its Work
    // callback before execution completion. Runtime must retain the capture
    // through Retirement, independently of CUDA's own command ownership.
    class DeferredRetirement : public R::StreamLauncher
    {
        std::vector<Retirement> pending;

    public:
        R::ExecutionContext Context() const override { return R::ExecutionContext::Host(); }
        void Submit(Work work, R::CommandKind, Retirement retire) override
        {
            work(Context());
            pending.push_back(std::move(retire));
        }
        void WaitFor(R::StreamEvent event) override { event.WaitHost(); }
        R::StreamEvent Record() override
        {
            Drain();
            return {};
        }
        void Drain() override
        {
            for (auto &retire : pending)
                if (retire)
                    retire({});
            pending.clear();
        }
    };
    for (const bool batch : {false, true})
    {
        R::EagerEngine engine;
        auto launcher = std::make_shared<DeferredRetirement>();
        const auto stream = engine.CreateStream(launcher);
        R::Array<double> output(R::RowLayout::Dense(1, 1));
        auto owner = std::make_shared<int>(37);
        const std::weak_ptr<int> observed = owner;
        if (batch)
        {
            std::vector<R::ArrayOperand<double, false, R::TableRows>> inputs;
            std::vector<R::ArrayOperand<double, true, R::TableRows>> outputs{R::WriteAs<R::TableRows>(output)};
            R::SubmitArrayBatch(engine, stream, [owner](auto, auto, auto out)
                                { out[0](0, 0) = *owner; }, std::move(inputs), std::move(outputs));
        }
        else
            R::SubmitArrays(engine, stream, [owner](auto, auto out)
                            { out(0, 0) = *owner; }, R::WriteAs<R::TableRows>(output));
        owner.reset();
        CHECK_FALSE(observed.expired());
        engine.Synchronize();
        CHECK(observed.expired());
        CHECK(output.Read().At(0, 0) == 37);
    }
}

struct Gate : R::StreamEvent::State
{
    std::atomic<bool> open{false};
    void Signal()
    {
        open = true;
        NotifyCompletion();
    }
    bool Ready() const override { return open.load(); }
    void Wait() const override
    {
        while (!Ready())
            std::this_thread::yield();
    }
};

static void Await(const std::atomic<bool> &flag)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!flag.load())
    {
        DNDS_check_throw_info(std::chrono::steady_clock::now() < deadline, "mock overlap rendezvous timed out");
        std::this_thread::yield();
    }
}

TEST_CASE("Controller resources survive inline or worker work and retire on the controller")
{
    for (int workers : {0, 2})
    {
        INFO("workers=" << workers);
        struct Resource
        {
            std::thread::id &destroyedOn;
            explicit Resource(std::thread::id &thread) : destroyedOn(thread) {}
            ~Resource() { destroyedOn = std::this_thread::get_id(); }
        };
        const auto controller = std::this_thread::get_id();
        std::thread::id destroyedOn;
        R::EagerEngine engine(workers);
        const auto stream = engine.CreateCPUStream();
        auto resource = std::make_shared<Resource>(destroyedOn);
        std::weak_ptr<Resource> weak = resource;
        engine.RetainUntilSynchronize(resource);
        engine.Submit(stream, [resource](R::ExecutionContext)
                      { (void)resource; });
        resource.reset();
        stream.Record().WaitHost();
        CHECK_FALSE(weak.expired());
        CHECK(destroyedOn == std::thread::id{});
        engine.Synchronize();
        CHECK(weak.expired());
        CHECK(destroyedOn == controller);
        engine.Close();
    }
}

TEST_CASE("Completion notifications wake CPU lanes without controller polling")
{
    struct NotifiedGate final : R::StreamEvent::State
    {
        const std::thread::id controller = std::this_thread::get_id();
        std::atomic<bool> done{false};
        mutable std::atomic<int> pendingWorkerQueries{0};
        bool Ready() const override
        {
            const bool ready = done.load();
            if (!ready && std::this_thread::get_id() != controller)
                ++pendingWorkerQueries;
            return ready;
        }
        void Wait() const override { DNDS_check_throw_info(done.load(), "test gate was not signalled"); }
        void Signal()
        {
            done = true;
            NotifyCompletion();
        }
    };
    // Separate executors exercise cross-runtime notification, with one worker
    // each: a blocked lane must not consume the producer's only worker.
    R::EagerEngine producer(1), consumer(1);
    const auto blocked = producer.CreateCPUStream(), independent = producer.CreateCPUStream();
    const auto target = consumer.CreateCPUStream();
    auto gate = std::make_shared<NotifiedGate>();
    std::atomic<bool> independentDone{false}, consumed{false};
    int payload = 0;
    blocked.WaitFor(R::StreamEvent(gate));
    producer.Submit(blocked, [&](R::ExecutionContext)
                    { payload = 19; });
    const auto ready = blocked.Record();
    target.WaitFor(ready);
    consumer.Submit(target, [&](R::ExecutionContext)
                    { CHECK(payload == 19); consumed = true; });
    producer.Submit(independent, [&](R::ExecutionContext)
                    { independentDone = true; });
    Await(independentDone);
    CHECK_FALSE(consumed);
    gate->Signal();
    Await(consumed); // No Ready/WaitHost/Synchronize drives either runtime here.
    CHECK(gate->pendingWorkerQueries == 0);
    bool lateNotification = false;
    R::StreamEvent(gate).OnCompletion([&]
                                      { lateNotification = true; });
    CHECK(lateNotification); // Completion before registration is not lost.
    producer.Synchronize();
    consumer.Synchronize();
}

TEST_CASE("Immediate context borrows prevent closure and weak runtime handles reject expiry")
{
    R::ExecutionStream saved;
    {
        R::EagerEngine engine(2);
        saved = engine.CreateCPUStream();
        {
            auto borrow = saved.BorrowContext();
            CHECK_THROWS(engine.Close());
            CHECK_FALSE(engine.IsClosed());
            auto call = saved.LockEngine();
            CHECK(call.Owns(saved));
            CHECK_THROWS(call.Close());
            call.Submit(saved, [](R::ExecutionContext) {});
            call.Synchronize();
        }
        engine.Close();
        CHECK_THROWS(saved.LockEngine());
    }
    CHECK_THROWS(saved.BorrowContext());
}
TEST_CASE("Context shutdown progresses controller transport and CPU lanes before releasing owners")
{
    for (const size_t workers : {size_t(0), size_t(2)})
        for (const int finish : {0, 1, 2}) // synchronize, close, destructor
        {
            const bool close = finish == 1;
            struct Signals
            {
                std::shared_ptr<Gate> sendReady = std::make_shared<Gate>();
                std::shared_ptr<Gate> received = std::make_shared<Gate>();
                std::atomic<bool> coreDone{false}, ringDone{false};
                std::thread::id disposed;
                int payload = 0;
            };
            struct Transport final : R::ExecutionProgress
            {
                std::shared_ptr<Signals> signals;
                explicit Transport(std::shared_ptr<Signals> s) : signals(std::move(s)) {}
                ~Transport() override { signals->disposed = std::this_thread::get_id(); }
                bool Poll() override
                {
                    signals->sendReady->Signal();
                    if (!signals->coreDone.load())
                        return false;
                    signals->payload = 37;
                    signals->received->Signal();
                    return true;
                }
            };
            auto signals = std::make_shared<Signals>();
            R::ExecutionStream surviving;
            R::StreamEvent completed;
            {
                R::EagerEngine engine(workers);
                // Consumer first: sequential stream draining would block it
                // before the controller can progress transport or the producer.
                auto ring = engine.CreateCPUStream(), core = engine.CreateCPUStream();
                surviving = ring;
                auto operation = std::make_shared<Transport>(signals);
                ring.RegisterProgress(operation);
                operation.reset(); // Runtime is the only transport owner.
                ring.WaitFor(R::StreamEvent(signals->received));
                engine.Submit(ring, [signals](R::ExecutionContext)
                              { signals->ringDone = signals->payload == 37; });
                core.WaitFor(R::StreamEvent(signals->sendReady));
                engine.Submit(core, [signals](R::ExecutionContext)
                              { signals->coreDone = true; });
                completed = ring.Record();
                if (finish != 2)
                {
                    if (close)
                        engine.Close();
                    else
                        engine.Synchronize();
                    CHECK(signals->coreDone.load());
                    CHECK(signals->ringDone.load());
                    CHECK(signals->disposed == std::this_thread::get_id());
                    CHECK(completed.Ready());
                    CHECK(engine.IsClosed() == close);
                    if (!close)
                        CHECK_NOTHROW(engine.Submit(core, [](R::ExecutionContext) {}));
                    engine.Close();
                    CHECK_NOTHROW(engine.Close());
                    CHECK_THROWS(surviving.Record());
                    CHECK_THROWS(engine.Submit(surviving, [](R::ExecutionContext) {}));
                }
            }
            CHECK(completed.Ready());
            CHECK(signals->ringDone.load());
            CHECK(signals->disposed == std::this_thread::get_id());
            CHECK_THROWS(surviving.Context());
        }
}

TEST_CASE("Drain waits autonomous prefixes by priority only after controller progress retires")
{
    struct Progress final : R::ExecutionProgress
    {
        int polls = 0;
        bool retired = false;
        bool Poll() override { return retired = ++polls == 3; }
    };
    struct Prefix final : R::StreamEvent::State
    {
        std::shared_ptr<Progress> progress;
        std::vector<unsigned> &order;
        unsigned priority;
        mutable bool done = false, prematureWait = false;
        mutable int readyCalls = 0, waitCalls = 0;
        Prefix(std::shared_ptr<Progress> p, std::vector<unsigned> &o, unsigned rank)
            : progress(std::move(p)), order(o), priority(rank) {}
        bool Ready() const override
        {
            ++readyCalls;
            return done || (!priority && progress->retired);
        }
        void Wait() const override
        {
            ++waitCalls;
            prematureWait = prematureWait || !progress->retired;
            order.push_back(priority);
            done = true;
        }
        unsigned AutonomousWaitPriority() const noexcept override { return priority; }
    };
    struct Launcher final : R::StreamLauncher
    {
        std::shared_ptr<Prefix> prefix;
        explicit Launcher(std::shared_ptr<Prefix> p) : prefix(std::move(p)) {}
        R::ExecutionContext Context() const override { return R::ExecutionContext::Host(); }
        void Submit(Work work, R::CommandKind, Retirement retire) override
        {
            work(Context());
            if (retire)
                retire({});
        }
        void WaitFor(R::StreamEvent event) override { event.WaitHost(); }
        R::StreamEvent Record() override { return R::StreamEvent(prefix); }
        void Drain() override {}
    };
    // The provider declarations drive ordering; these mocks require no CUDA.
    R::EagerEngine engine;
    auto progress = std::make_shared<Progress>();
    std::vector<unsigned> order;
    auto consumer = std::make_shared<Prefix>(progress, order, 1);
    auto producer = std::make_shared<Prefix>(progress, order, 2);
    auto custom = std::make_shared<Prefix>(progress, order, 0);
    engine.CreateStream(std::make_shared<Launcher>(consumer));
    engine.CreateStream(std::make_shared<Launcher>(producer));
    engine.CreateStream(std::make_shared<Launcher>(custom));
    engine.RegisterProgress(progress);
    engine.Synchronize();
    CHECK(progress->polls == 3);
    const std::vector<unsigned> expected{2, 1};
    CHECK(order == expected);
    CHECK_FALSE(consumer->prematureWait);
    CHECK_FALSE(producer->prematureWait);
    CHECK(consumer->waitCalls == 1);
    CHECK(producer->waitCalls == 1);
    CHECK(custom->waitCalls == 0);
    CHECK(custom->readyCalls > 0);
    CHECK(Gate{}.AutonomousWaitPriority() == 0);
    engine.Close();
}

TEST_CASE("Active mock communication overlaps inline and worker core kernels")
{
    // A separate executor models transport for inline/one-worker compute;
    // two-worker compute uses the same engine for both lanes. Both bodies must
    // be live: neither can finish before the other enters.
    for (size_t workers : {size_t(0), size_t(1), size_t(2)})
        for (size_t width : {size_t(1), size_t(4), size_t(5), size_t(6), size_t(7), size_t(13), size_t(114)})
        {
            R::EagerEngine engine(workers), transport(1);
            auto core = engine.CreateCPUStream(), ring = engine.CreateCPUStream();
            auto &commEngine = workers == 2 ? engine : transport;
            auto comm = commEngine.CreateCPUStream();
            R::Array<double> owned(R::RowLayout::Dense(257, width, width + 3));
            R::Array<double> ghost(R::RowLayout::Dense(19, width, width + 2));
            R::Array<double> output(R::RowLayout::Dense(257, width, width + 1));
            {
                auto w = owned.Write();
                for (size_t i = 0; i < 257; ++i)
                    for (size_t j = 0; j < width; ++j)
                        w.At(i, j) = 10 * i + j;
            }
            std::atomic<bool> commEntered{false}, coreEntered{false}, coreFinished{false};
            R::SubmitArrays(commEngine, comm, [&](R::ExecutionContext, auto u, auto g)
                            {
                commEntered = true;
                Await(coreEntered);
                for (size_t i = 0; i < g.RowCount(); ++i)
                    for (size_t j = 0; j < g.RowSize(i); ++j) g(i, j) = u(13 * i, j);
                Await(coreFinished); }, R::Read(owned), R::Write(ghost));
            auto received = comm.Record();
            R::SubmitArrays(engine, core, [&](R::ExecutionContext, auto u, auto rhs)
                            {
                coreEntered = true;
                Await(commEntered);
                for (size_t i = 0; i < rhs.RowCount(); ++i)
                    for (size_t j = 0; j < rhs.RowSize(i); ++j) rhs(i, j) = 2 * u(i, j);
                coreFinished = true; }, R::Read(owned), R::Write(output));
            ring.WaitFor(core.Record());
            ring.WaitFor(received);
            R::SubmitArrays(engine, ring, [](R::ExecutionContext, auto g, auto rhs)
                            {
                for (size_t i = 0; i < g.RowCount(); ++i)
                    for (size_t j = 0; j < g.RowSize(i); ++j) rhs(i, j) += g(i, j); }, R::Read(ghost), R::ReadWrite(output));
            engine.Synchronize();
            transport.Synchronize();
            CHECK(commEntered.load());
            CHECK(coreFinished.load());
            auto r = output.Read();
            for (size_t i = 0; i < 257; ++i)
                for (size_t j = 0; j < width; ++j)
                    CHECK(r.At(i, j) == 20 * i + 2 * j + (i < 19 ? 130 * i + j : 0));
            auto raw = output.Storage().Read();
            for (size_t i = 0; i < 257; ++i)
                CHECK(raw.View<double>().data[i * (width + 1) + width] == 0);
        }
}

TEST_CASE("Missing eager join produces a deterministic stale result without a data race")
{
    for (size_t workers : {size_t(0), size_t(1), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto comm = engine.CreateCPUStream(), compute = engine.CreateCPUStream();
        R::Array<double> ghost(R::RowLayout::Dense(19, 7, 9)), stale(ghost.Layout()), correct(ghost.Layout());
        auto gate = std::make_shared<Gate>();
        comm.WaitFor(R::StreamEvent(gate));
        R::SubmitArrays(engine, comm, [](R::ExecutionContext, auto g)
                        { for (size_t i = 0; i < g.RowCount(); ++i) for (size_t j = 0; j < 7; ++j) g(i, j) = 100 + i + j; }, R::Write(ghost));
        CHECK_THROWS(ghost.Read()); // Immediate access catches the pending writer.
        auto copy = [](R::ExecutionContext, auto g, auto r)
        { for (size_t i = 0; i < g.RowCount(); ++i) for (size_t j = 0; j < 7; ++j) r(i, j) = 2 * g(i, j); };
        // Scheduled reads intentionally trust caller ordering. Keep the writer
        // gated until this read completes so the negative case has no C++ UB.
        R::SubmitArrays(engine, compute, copy, R::Read(ghost), R::Write(stale));
        compute.Record().WaitHost();
        gate->Signal();
        compute.WaitFor(comm.Record());
        R::SubmitArrays(engine, compute, copy, R::Read(ghost), R::Write(correct));
        engine.Synchronize();
        CHECK(stale.IsValid()); // Access validity is not a numerical oracle.
        auto bad = stale.Read(), good = correct.Read();
        size_t mismatches = 0;
        for (size_t i = 0; i < 19; ++i)
            for (size_t j = 0; j < 7; ++j)
            {
                CHECK(good.At(i, j) == 2 * (100 + i + j));
                mismatches += bad.At(i, j) != good.At(i, j);
            }
        CHECK(mismatches == 19 * 7);
    }
}

TEST_CASE("Manufactured triangular sweeps preserve in-place ordering and eager stopping")
{
    // Euler's SGS path reads and writes the same increment in directional
    // order, with host decisions between iterations. Manufacture an exact
    // solution so verification does not repeat the implementation recurrence.
    for (size_t workers : {size_t(0), size_t(1), size_t(2)})
        for (size_t width : {size_t(4), size_t(5), size_t(6), size_t(7), size_t(13)})
        {
            R::EagerEngine engine(workers);
            auto stream = engine.CreateCPUStream();
            R::Array<double> increment(R::RowLayout::Dense(67, width, width + 3));
            for (bool backward : {false, true})
            {
                auto rhs = R::AllocateLikeAsync(engine, stream, increment);
                R::SubmitArrays(engine, stream, [backward](R::ExecutionContext, auto r)
                                {
                    for (size_t i = 0; i < r.RowCount(); ++i)
                        for (size_t j = 0; j < r.RowSize(i); ++j)
                        {
                            const bool boundary = backward ? i + 1 == r.RowCount() : i == 0;
                            const double neighbor = boundary ? 0 : double(i) + (backward ? 1 : -1) + j + 1;
                            r(i, j) = 2 * (i + j + 1) - 0.25 * neighbor;
                        } }, R::Write(rhs));
                int iterations = 0;
                for (; iterations < 4;)
                {
                    ++iterations;
                    R::SubmitArrays(engine, stream, [backward](R::ExecutionContext, auto r, auto x)
                                    {
                        for (size_t k = 0; k < x.RowCount(); ++k)
                        {
                            const size_t i = backward ? x.RowCount() - 1 - k : k;
                            for (size_t j = 0; j < x.RowSize(i); ++j)
                                x(i, j) = (r(i, j) + (k ? 0.25 * x(backward ? i + 1 : i - 1, j) : 0)) / 2;
                        } }, R::Read(rhs), R::ReadWrite(increment));
                    stream.Record().WaitHost();
                    auto solution = increment.Read();
                    double error = 0;
                    for (size_t i = 0; i < 67; ++i)
                        for (size_t j = 0; j < width; ++j)
                            error = std::max(error, std::abs(solution.At(i, j) - double(i + j + 1)));
                    if (error == 0)
                        break; // Fully eager data-dependent stopping.
                }
                CHECK(iterations == 1);
                engine.ReleaseOn(std::move(rhs), stream);
            }
            engine.Synchronize();
        }
}

TEST_CASE("Inline and worker launchers execute the same eager fork join loop")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto compute = engine.CreateCPUStream();
        auto comm = engine.CreateCPUStream();
        std::vector<double> state(67 * 7, 1);
        double norm = 0;
        int iterations = 0;
        for (int iteration = 0; iteration < 19; ++iteration)
        {
            engine.Submit(compute, [&](R::ExecutionContext context)
                          {
                CHECK(context.device == R::Device::Host());
                for (auto &x : state) x *= 0.5; });
            comm.WaitFor(compute.Record());
            engine.Submit(comm, [&](R::ExecutionContext)
                          {
                norm = 0;
                for (auto x : state) norm += x; });
            auto reduced = comm.Record();
            reduced.WaitHost();
            ++iterations;
            if (norm < 1)
                break;
            compute.WaitFor(reduced);
        }
        engine.Synchronize();
        CHECK(iterations == 9);
        CHECK(state[66 * 7 + 6] == doctest::Approx(1.0 / 512));
    }
}

TEST_CASE("Record describes a fixed prefix and a waiting lane cannot exhaust workers")
{
    for (size_t workers : {size_t(0), size_t(1)})
    {
        R::EagerEngine engine(workers);
        auto first = engine.CreateCPUStream();
        auto other = engine.CreateCPUStream();
        std::atomic<int> value{0};
        engine.Submit(first, [&](R::ExecutionContext)
                      { value = 1; });
        auto prefix = first.Record();
        prefix.WaitHost();
        auto gate = std::make_shared<Gate>();
        first.WaitFor(R::StreamEvent(gate));
        engine.Submit(first, [&](R::ExecutionContext)
                      { value = 2; });
        auto tail = first.Record();
        CHECK(prefix.Ready());
        CHECK_FALSE(tail.Ready());
        CHECK(value == 1);
        engine.Submit(other, [gate](R::ExecutionContext)
                      { gate->Signal(); });
        engine.Synchronize();
        CHECK(value == 2);
        CHECK(tail.Ready());
    }
}

TEST_CASE("Imported CPU prefixes progress without blocking another executor")
{
    R::EagerEngine producer;
    R::EagerEngine consumer(1);
    auto source = producer.CreateCPUStream();
    auto target = consumer.CreateCPUStream();
    auto gate = std::make_shared<Gate>();
    std::atomic<int> state{0};
    source.WaitFor(R::StreamEvent(gate));
    producer.Submit(source, [&](R::ExecutionContext)
                    { state = 7; });
    target.WaitFor(source.Record());
    consumer.Submit(target, [&](R::ExecutionContext)
                    { CHECK(state == 7); state = 9; });
    gate->Signal();
    consumer.Synchronize();
    CHECK(state == 9);
    CHECK_THROWS(consumer.Submit(source, [](R::ExecutionContext) {}));
}

TEST_CASE("Queued temporary ownership and releases follow CPU completion on either lane")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto compute = engine.CreateCPUStream();
        auto comm = engine.CreateCPUStream();
        auto gate = std::make_shared<Gate>();
        std::atomic<int> deletes{0}, sum{0};
        auto owner = std::shared_ptr<std::vector<int>>(new std::vector<int>(31 * 5, 3), [&](auto *p)
                                                       { delete p; ++deletes; });
        std::weak_ptr<std::vector<int>> weak = owner;
        auto alias = owner;
        compute.WaitFor(R::StreamEvent(gate));
        engine.Submit(compute, [owner, &sum](R::ExecutionContext)
                      {
            for (auto x : *owner) sum += x; });
        engine.ReleaseOn(std::move(owner), compute);
        comm.WaitFor(compute.Record());
        engine.Submit(comm, [alias, &sum](R::ExecutionContext)
                      { sum += alias->back(); });
        engine.ReleaseOn(std::move(alias), comm);
        CHECK_FALSE(owner);
        CHECK_FALSE(alias);
        CHECK_FALSE(weak.expired());
        CHECK(deletes == 0);
        gate->Signal();
        engine.Synchronize();
        CHECK(weak.expired());
        CHECK(deletes == 1);
        CHECK(sum == 31 * 5 * 3 + 3);
        engine.ReleaseOn(std::move(owner), comm); // Releasing an empty handle is harmless.
        engine.Synchronize();
        CHECK(deletes == 1);
    }
}

TEST_CASE("Errors skip dependent work but retain ordered cleanup and drain other lanes")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto compute = engine.CreateCPUStream();
        auto comm = engine.CreateCPUStream();
        auto before = compute.Record();
        std::atomic<bool> skipped{true}, cleaned{false}, unrelated{false};
        engine.Submit(compute, [](R::ExecutionContext)
                      { throw std::runtime_error("operator failed"); });
        comm.WaitFor(compute.Record());
        engine.Submit(comm, [&](R::ExecutionContext)
                      { skipped = false; });
        engine.SubmitCleanup(comm, [&](R::ExecutionContext)
                             { cleaned = true; });
        auto independent = engine.CreateCPUStream();
        engine.Submit(independent, [&](R::ExecutionContext)
                      { unrelated = true; });
        CHECK_THROWS(engine.Synchronize());
        CHECK(before.Ready());
        CHECK(skipped);
        CHECK(cleaned);
        CHECK(unrelated);
    }
}

TEST_CASE("Concurrent host submitters linearize commands and preserve every accepted use")
{
    R::EagerEngine engine(2);
    auto a = engine.CreateCPUStream();
    auto b = engine.CreateCPUStream();
    std::atomic<size_t> total{0};
    std::vector<std::thread> submitters;
    for (int thread = 0; thread < 4; ++thread)
        submitters.emplace_back([&, thread]
                                {
            for (int i = 0; i < 257; ++i)
                engine.Submit(thread % 2 ? a : b, [&](R::ExecutionContext) { total += 7; }); });
    for (auto &thread : submitters)
        thread.join();
    engine.Synchronize();
    CHECK(total == 4 * 257 * 7);
}

TEST_CASE("Failed lanes still honor joins before releasing shared resources")
{
    R::EagerEngine engine(2);
    auto failed = engine.CreateCPUStream();
    auto peer = engine.CreateCPUStream();
    auto gate = std::make_shared<Gate>();
    std::atomic<int> deletes{0};
    auto owner = std::shared_ptr<int>(new int(17), [&](int *p)
                                      { delete p; ++deletes; });
    std::weak_ptr<int> weak = owner;
    peer.WaitFor(R::StreamEvent(gate));
    engine.Submit(peer, [owner](R::ExecutionContext)
                  { CHECK(*owner == 17); });
    engine.Submit(failed, [](R::ExecutionContext)
                  { throw std::runtime_error("failed before join"); });
    auto error = failed.Record();
    CHECK_THROWS(error.WaitHost());
    failed.WaitFor(peer.Record());
    engine.ReleaseOn(std::move(owner), failed);
    CHECK_FALSE(weak.expired());
    CHECK(deletes == 0);
    gate->Signal();
    CHECK_THROWS(engine.Synchronize());
    CHECK(weak.expired());
    CHECK(deletes == 1);
}

TEST_CASE("Rejected release returns the caller handle and never guesses a stream")
{
    class Reject : public R::StreamLauncher
    {
    public:
        R::ExecutionContext Context() const override { return R::ExecutionContext::Host(); }
        void Submit(Work, R::CommandKind, Retirement) override { throw std::runtime_error("queue rejected"); }
        void WaitFor(R::StreamEvent) override {}
        R::StreamEvent Record() override { return {}; }
        void Drain() override {}
    };
    R::EagerEngine engine;
    auto owner = std::make_shared<int>(42);
    CHECK_THROWS(engine.ReleaseOn(std::move(owner), R::ExecutionStream{}));
    REQUIRE(owner);
    auto rejected = engine.CreateStream(std::make_shared<Reject>());
    CHECK_THROWS(engine.ReleaseOn(std::move(owner), rejected));
    REQUIRE(owner);
    CHECK(*owner == 42);
}

TEST_CASE("Array reservations publish at execution and preserve same stream read write ordering")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto stream = engine.CreateCPUStream();
        auto gate = std::make_shared<Gate>();
        stream.WaitFor(R::StreamEvent(gate));
        R::Array<double, R::FixedRows<6>> state(R::RowLayout::Dense(257, 6, 8));
        auto alias = state;
        const auto initial = state.Version();
        for (int iteration = 0; iteration < 31; ++iteration)
            R::SubmitArrays(engine, stream, [](R::ExecutionContext, R::ArrayView<double> u)
                            {
                for (size_t row = 0; row < u.RowCount(); ++row)
                    for (size_t col = 0; col < u.RowSize(row); ++col)
                        u(row, col) += double(row + col + 1); }, R::ReadWrite(state));
        CHECK(state.Version() == initial);
        CHECK_FALSE(state.Storage().IsIdle());
        CHECK_THROWS(alias.Read());
        CHECK_THROWS(alias.Write());
        CHECK_THROWS(state.EditRows());
        gate->Signal();
        stream.Record().WaitHost();
        CHECK(state.Storage().IsIdle());
        CHECK(state.Version() == initial + 31);
        auto read = state.Read();
        auto flatAccess = state.Storage().Read();
        auto flat = flatAccess.View<double>();
        for (size_t row = 0; row < state.RowCount(); ++row)
        {
            CHECK(read.At(row, 5) == 31 * double(row + 6));
            CHECK(flat.data[row * 8 + 6] == 0);
            CHECK(flat.data[row * 8 + 7] == 0);
        }
    }
}

TEST_CASE("Owned only CSR stencil can complete while ghost receives remain pending")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto compute = engine.CreateCPUStream();
        auto comm = engine.CreateCPUStream();
        R::ArrayPair<double, R::FixedRows<5>> state{
            R::Array<double, R::FixedRows<5>>(R::RowLayout::Dense(257, 5, 8)),
            R::Array<double, R::FixedRows<5>>(R::RowLayout::Dense(19, 5, 7))};
        {
            auto write = state.owned.Write();
            for (size_t row = 0; row < 257; ++row)
                for (size_t col = 0; col < 5; ++col)
                    write.At(row, col) = row + col;
        }
        std::vector<size_t> lengths(239);
        for (size_t row = 0; row < lengths.size(); ++row)
            lengths[row] = 1 + row % 5;
        R::Array<size_t> adjacency(R::RowLayout::Ragged(lengths));
        {
            auto write = adjacency.Write();
            for (size_t row = 0; row < lengths.size(); ++row)
                for (size_t j = 0; j < lengths[row]; ++j)
                    write.At(row, j) = (row + 7 * j) % 257;
        }
        R::Array<double> residual(R::RowLayout::Dense(239, 5));
        auto gate = std::make_shared<Gate>();
        comm.WaitFor(R::StreamEvent(gate));
        R::SubmitArrays(engine, comm, [](R::ExecutionContext, R::ArrayView<double> ghost)
                        {
            for (size_t row = 0; row < ghost.RowCount(); ++row)
                for (size_t col = 0; col < ghost.RowSize(row); ++col) ghost(row, col) = 1000 + row + col; }, R::Write(state.ghost));
        R::SubmitArrays(engine, compute, [](R::ExecutionContext, R::ArrayView<const double> owned, R::ArrayView<const size_t> c2c, R::ArrayView<double> rhs)
                        {
                for (size_t row = 0; row < rhs.RowCount(); ++row)
                    for (size_t col = 0; col < rhs.RowSize(row); ++col)
                        for (size_t n = 0; n < c2c.RowSize(row); ++n) rhs(row, col) += owned(c2c(row, n), col); }, R::Read(state.owned), R::Read(adjacency), R::Write(residual));
        compute.Record().WaitHost();
        CHECK_THROWS(state.ghost.Read());
        auto output = residual.Read();
        for (size_t row = 0; row < lengths.size(); ++row)
        {
            double expected = 0;
            for (size_t n = 0; n < lengths[row]; ++n)
                expected += (row + 7 * n) % 257 + 4;
            CHECK(output.At(row, 4) == expected);
        }
        gate->Signal();
        compute.WaitFor(comm.Record());
        R::Array<double> ring(R::RowLayout::Dense(19, 5));
        R::SubmitArrays(engine, compute, [](R::ExecutionContext, R::ArrayView<const double> owned, R::ArrayView<const double> ghost, R::ArrayView<double> rhs)
                        {
                R::PairReadView<double> u{owned, ghost};
                for (size_t row = 0; row < rhs.RowCount(); ++row)
                    for (size_t col = 0; col < rhs.RowSize(row); ++col) rhs(row, col) = u(row, col) + u(owned.RowCount() + row, col); }, R::Read(state.owned), R::Read(state.ghost), R::Write(ring));
        engine.Synchronize();
        CHECK(ring.Read().At(18, 4) == 1044);
    }
}

TEST_CASE("Pending operands retain old values and buffered row sizes across handle reset")
{
    R::EagerEngine engine(2);
    auto stream = engine.CreateCPUStream();
    auto gate = std::make_shared<Gate>();
    stream.WaitFor(R::StreamEvent(gate));
    std::vector<size_t> lengths(97);
    for (size_t i = 0; i < lengths.size(); ++i)
        lengths[i] = i % 8;
    R::Array<int64_t> input(R::RowLayout::Padded(lengths, 11));
    {
        auto write = input.Write();
        for (size_t i = 0; i < lengths.size(); ++i)
            for (size_t j = 0; j < lengths[i]; ++j)
                write.At(i, j) = i * 100 + j;
    }
    std::weak_ptr<const R::ArrayStorage> storage = input.SharedStorage();
    std::weak_ptr<const R::Detail::ArrayLayout> metadata = input.SharedStorage()->SharedLayout();
    R::Array<int64_t> output(R::RowLayout::Ragged(lengths));
    R::SubmitArrays(engine, stream, [](R::ExecutionContext, R::ArrayView<const int64_t> in, R::ArrayView<int64_t> out)
                    {
        for (size_t i = 0; i < out.RowCount(); ++i)
            for (size_t j = 0; j < out.RowSize(i); ++j) out(i, j) = in(i, j) + 1; }, R::Read(input), R::Write(output));
    input.Reset(R::RowLayout::Dense(1, 2));
    CHECK_FALSE(storage.expired());
    CHECK_FALSE(metadata.expired());
    gate->Signal();
    stream.Record().WaitHost();
    CHECK(storage.expired());
    CHECK(metadata.expired());
    auto values = output.Read();
    for (size_t i = 0; i < lengths.size(); ++i)
        for (size_t j = 0; j < lengths[i]; ++j)
            CHECK(values.At(i, j) == int64_t(i * 100 + j + 1));
}

TEST_CASE("Dynamic array batches retain segmented fields and order queued producers")
{
    using Input = R::ArrayOperand<double, false, R::UniformTableRows>;
    using Output = R::ArrayOperand<double, true, R::UniformTableRows>;
    for (int workers : {0, 2})
    {
        R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        const auto gate = std::make_shared<Gate>();
        stream.WaitFor(R::StreamEvent(gate));
        R::Array<double> flow(R::RowLayout::Dense(17, 5));
        R::Array<double> species(R::RowLayout::ColMajorWithPitch(17, 11, 32));
        R::Array<double> sum(R::RowLayout::Dense(17, 1));
        const auto epoch = flow.MutationEpoch(), version = flow.Version();
        std::weak_ptr<const R::ArrayStorage> retained = flow.SharedStorage();
        R::SubmitArrayBatch(engine, stream, [](R::ExecutionContext, auto inputs, auto outputs)
                            {
            CHECK(inputs.empty());
            for (DNDS::index iField = 0; iField < outputs.size(); ++iField)
                for (DNDS::index iRow = 0; iRow < outputs[iField].RowCount(); ++iRow)
                    for (DNDS::index j = 0; j < outputs[iField].Width(); ++j)
                        outputs[iField](iRow, j) = 100 * iField + iRow + j; }, std::vector<Input>{}, std::vector<Output>{R::WriteAs<R::UniformTableRows>(flow), R::WriteAs<R::UniformTableRows>(species)});
        CHECK(flow.MutationEpoch() > epoch);
        CHECK(flow.Version() == version);
        CHECK_THROWS(flow.Read());
        R::SubmitArrayBatch(engine, stream, [](R::ExecutionContext, auto inputs, auto outputs)
                            {
            for (DNDS::index iRow = 0; iRow < outputs[0].RowCount(); ++iRow)
            {
                double total = 0;
                for (const auto &field : inputs)
                    for (DNDS::index j = 0; j < field.Width(); ++j) total += field(iRow, j);
                outputs[0](iRow, 0) = total;
            } }, std::vector<Input>{R::ReadAs<R::UniformTableRows>(flow), R::ReadAs<R::UniformTableRows>(species)}, std::vector<Output>{R::WriteAs<R::UniformTableRows>(sum)});
        flow.Reset(R::RowLayout::Dense(1, 1));
        CHECK_FALSE(retained.expired());
        gate->Signal();
        engine.Synchronize();
        CHECK(retained.expired());
        const auto read = sum.Read();
        for (DNDS::index iRow = 0; iRow < 17; ++iRow)
            CHECK(read.View()(iRow, 0) == 1165 + 16 * iRow);
    }
}

TEST_CASE("Dynamic array batch failure invalidates every declared output")
{
    using Input = R::ArrayOperand<double, false, R::TableRows>;
    using Output = R::ArrayOperand<double, true, R::TableRows>;
    R::EagerEngine engine(2);
    const auto stream = engine.CreateCPUStream();
    R::Array<double> a(R::RowLayout::Dense(3, 5)), b(R::RowLayout::Dense(3, 2));
    R::SubmitArrayBatch(engine, stream, [](R::ExecutionContext, auto, auto output)
                        {
        output[0](0, 0) = 1;
        throw std::runtime_error("test-only partial batch failure"); }, std::vector<Input>{}, std::vector<Output>{R::WriteAs<R::TableRows>(a), R::WriteAs<R::TableRows>(b)});
    CHECK_THROWS(engine.Synchronize());
    CHECK_FALSE(a.IsValid());
    CHECK_FALSE(b.IsValid());
    CHECK(a.Storage().IsIdle());
    CHECK(b.Storage().IsIdle());
}

TEST_CASE("Rejected reservations cancel while failed or skipped writes stay invalid")
{
    R::EagerEngine engine(2), other;
    auto stream = engine.CreateCPUStream();
    auto gate = std::make_shared<Gate>();
    stream.WaitFor(R::StreamEvent(gate));
    R::Array<double> input(R::RowLayout::Dense(43, 6)), output(input.Layout()), skipped(input.Layout());
    auto kernel = [](R::ExecutionContext, R::ArrayView<const double>, R::ArrayView<double>) {};
    auto initial = output.Version();
    {
        auto pin = input.Write();
        CHECK_THROWS(R::SubmitArrays(engine, stream, kernel, R::Read(input), R::Write(output)));
    }
    CHECK(output.Storage().IsIdle());
    CHECK(output.IsValid());
    CHECK(output.Version() == initial);
    CHECK_THROWS(R::SubmitArrays(other, stream, kernel, R::Read(input), R::Write(output)));
    CHECK(output.Storage().IsIdle());
    {
        auto pin = input.Read();
        R::SubmitArrays(engine, stream, [](R::ExecutionContext, R::ArrayView<const double>) {}, R::Read(input));
        CHECK_THROWS(R::SubmitArrays(engine, stream, [](R::ExecutionContext, R::ArrayView<double>) {}, R::Write(input)));
    }
    CHECK_THROWS(input.Write()); // Pending read excludes immediate writer.
    CHECK_NOTHROW(input.Read());
    R::SubmitArrays(engine, stream, [](R::ExecutionContext, R::ArrayView<double> value)
                    {
        value(17, 4) = 9;
        throw std::runtime_error("partial update"); }, R::Write(output));
    R::SubmitArrays(engine, stream, kernel, R::Read(output), R::Write(skipped));
    gate->Signal();
    CHECK_THROWS(engine.Synchronize());
    CHECK(input.IsValid());
    CHECK_FALSE(output.IsValid());
    CHECK_FALSE(skipped.IsValid());
    CHECK(output.Version() == initial);
    CHECK(output.Storage().IsIdle());
    CHECK_THROWS(output.Read());
    CHECK_THROWS(skipped.Read());
}

TEST_CASE("Eager byte copies use the same pending access contract on inline and worker CPU")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto stream = engine.CreateCPUStream();
        auto gate = std::make_shared<Gate>();
        stream.WaitFor(R::StreamEvent(gate));
        auto source = R::Buffer::Allocate(131 * 7 * sizeof(int64_t));
        auto target = R::Buffer::Allocate(source.Bytes());
        {
            auto write = source.Write();
            auto view = write.View<int64_t>();
            for (size_t i = 0; i < view.size; ++i)
                view.data[i] = int64_t(13 * i);
        }
        R::CopyAsync(engine, stream, source, target);
        CHECK_THROWS(source.Write());
        CHECK_THROWS(target.Read());
        CHECK_THROWS(R::CopyAsync(engine, stream, source, source));
        auto small = R::Buffer::Allocate(3);
        CHECK_THROWS(R::CopyAsync(engine, stream, source, small));
        CHECK(small.IsIdle());
        source = {};
        gate->Signal();
        engine.Synchronize();
        auto read = target.Read();
        auto values = read.View<int64_t>();
        for (size_t i = 0; i < values.size; ++i)
            CHECK(values.data[i] == int64_t(13 * i));
        auto empty = R::Buffer::Allocate(0), other = R::Buffer::Allocate(0);
        R::CopyAsync(engine, stream, empty, other);
        engine.Synchronize();
        CHECK(other.Version() == 1);
    }
#ifndef DNDS_USE_CUDA
    CHECK_THROWS(R::CUDALauncher(0));
#endif
}

TEST_CASE("CPU async allocation facade preserves shared variable shape metadata and handle ownership")
{
    R::EagerEngine engine(2);
    auto stream = engine.CreateCPUStream();
    auto table = R::AllocateArrayAsync<double, R::FixedRows<6>>(engine, stream, R::RowLayout::Dense(31, 6, 8));
    CHECK(table.Read().At(30, 5) == 0);
    auto csr = R::AllocateArrayAsync<int64_t>(engine, stream, R::RowLayout::Ragged({0, 7, 19, 5}));
    auto second = R::AllocateLikeAsync(engine, stream, csr);
    CHECK(csr.SharedStorage()->SharedLayout() == second.SharedStorage()->SharedLayout());
    auto alias = table;
    auto gate = std::make_shared<Gate>();
    stream.WaitFor(R::StreamEvent(gate));
    engine.ReleaseOn(std::move(table), stream);
    CHECK(alias.IsValid());
    gate->Signal();
    engine.Synchronize();
    CHECK(alias.Read().At(30, 5) == 0);
    engine.ReleaseOn(std::move(alias), stream);
    engine.ReleaseOn(std::move(alias), stream);
    engine.Synchronize();
}
