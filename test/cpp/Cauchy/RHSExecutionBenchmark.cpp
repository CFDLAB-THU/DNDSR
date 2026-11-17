// Private executable instrumentation, never linked into a production library.
#include "RHSExecutionBenchmark.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "DNDS/Runtime/Communication/TransferProvider.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <sched.h>
#include <thread>

namespace Plan53
{
    namespace R = DNDS::Runtime;
    namespace F = DNDS::CFV;
    using Clock = std::chrono::steady_clock;
    using J = nlohmann::ordered_json;
    using I = DNDS::index;
    namespace
    {
        double Time() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
        struct Event
        {
            double producer = Time(), enqueue = 0, start = 0, ready = 0, hostWait = 0;
            double submit = 0;
            double staging = 0, pack = 0, post = 0, progress = 0, unpack = 0;
            double firstPost = 0, received = 0;
            MPI_Request *receiveRequests = nullptr; // Runtime starts the receive group before the send group.
            I stagingBytes = 0, stagingCalls = 0, packs = 0, unpacks = 0, requests = 0;
            I receiveBytes = 0;
        };
        struct Trace
        {
            double origin = Time();
            std::vector<std::shared_ptr<Event>> events;  // Controller appends; event bodies have one MPI owner.
            std::vector<std::pair<double, double>> core; // Exactly one compute lane appends.
        };
        std::atomic<Trace *> trace{nullptr};
        thread_local Event *event = nullptr;
        struct EventScope
        {
            Event *previous;
            explicit EventScope(Event *value) : previous(event) { event = value; }
            ~EventScope() { event = previous; }
        };
        struct Pending
        {
            R::TransferBinding binding;
            std::optional<R::TransferOperation> operation;
            std::promise<void> done;
            std::shared_future<void> future = done.get_future().share();
            std::shared_ptr<Event> observation;
            explicit Pending(R::TransferBinding value) : binding(std::move(value)) {}
        };
        void Pin(int cpu)
        {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(cpu, &mask);
            DNDS_check_throw_info(sched_setaffinity(0, sizeof(mask), &mask) == 0, "cannot pin assessment thread");
        }
        int Topology(int cpu, const char *name)
        {
            std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/" + name);
            int result = -1;
            file >> result;
            DNDS_check_throw_info(result >= 0, "assessment requires Linux physical-core topology");
            return result;
        }
        int NUMA(int cpu)
        {
            for (const auto &entry : std::filesystem::directory_iterator("/sys/devices/system/cpu/cpu" + std::to_string(cpu)))
            {
                const auto name = entry.path().filename().string();
                if (name.size() > 4 && name.compare(0, 4, "node") == 0 &&
                    std::all_of(name.begin() + 4, name.end(), [](char c)
                                { return c >= '0' && c <= '9'; }))
                    return std::stoi(name.substr(4));
            }
            return -1;
        }
        /// One MPI owner. Background Test/Wait runs only here; main-thread
        /// control RPCs join their own call. No main thread tests worker requests.
        /// Requests can coexist, so moving work does not serialize fields into
        /// complete transfer-by-transfer waits. Shutdown drains before MPI owners.
        class Broker
        {
            std::mutex _mutex;
            std::condition_variable _changed;
            std::deque<std::function<void()>> _queue;
            std::vector<std::shared_ptr<Pending>> _active;
            bool _stop = false;
            std::thread _worker;
            void Retire(const std::shared_ptr<Pending> &p, bool wait)
            {
                EventScope scope(p->observation.get());
                try
                {
                    if (wait)
                        p->operation->Wait();
                    else if (!p->operation->Test())
                        return;
                    if (event)
                        event->ready = Time();
                    p->operation.reset();
                    p->done.set_value();
                }
                catch (...)
                {
                    p->operation.reset();
                    p->done.set_exception(std::current_exception());
                }
            }
            void Run()
            {
                for (;;)
                {
                    std::function<void()> action;
                    {
                        std::unique_lock<std::mutex> lock(_mutex);
                        if (_active.empty())
                            _changed.wait(lock, [&]
                                          { return _stop || !_queue.empty(); });
                        if (_stop && _queue.empty())
                            return;
                        if (!_queue.empty())
                        {
                            action = std::move(_queue.front());
                            _queue.pop_front();
                        }
                    }
                    if (action)
                        action();
                    for (auto it = _active.begin(); it != _active.end();)
                    {
                        Retire(*it, false);
                        if ((*it)->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                            it = _active.erase(it);
                        else
                            ++it;
                    }
                }
            }

        public:
            explicit Broker(int cpu) : _worker([this]
                                               { Run(); })
            {
                try
                {
                    Call([=]
                         { Pin(cpu); return 0; });
                }
                catch (...)
                {
                    {
                        std::lock_guard<std::mutex> lock(_mutex);
                        _stop = true;
                    }
                    _changed.notify_one();
                    _worker.join();
                    throw;
                }
            }
            template <class Action>
            auto Call(Action action) -> decltype(action())
            {
                if (std::this_thread::get_id() == _worker.get_id())
                    return action();
                auto task = std::make_shared<std::packaged_task<decltype(action())()>>(std::move(action));
                auto result = task->get_future();
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _queue.push_back([task]
                                     { (*task)(); });
                }
                _changed.notify_one();
                return result.get();
            }
            std::shared_ptr<Pending> Begin(R::TransferBinding binding, bool scheduled, std::shared_ptr<Event> observation)
            {
                auto p = std::make_shared<Pending>(std::move(binding));
                p->observation = std::move(observation);
                if (p->observation)
                    p->observation->enqueue = Time();
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _queue.push_back([this, p, scheduled]
                                     {
                        EventScope scope(p->observation.get());
                        try
                        {
                            if (event) event->start = Time();
                            p->operation.emplace(scheduled ? p->binding.BeginScheduled() : p->binding.Begin());
                            _active.push_back(p);
                        }
                        catch (...) { p->done.set_exception(std::current_exception()); } });
                }
                _changed.notify_one();
                return p;
            }
            ~Broker()
            {
                // All public operation owners should already have observed
                // completion. Drain nevertheless before stopping this MPI owner.
                Call([&]
                     { for (auto &p : _active) Retire(p, true); _active.clear(); return 0; });
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _stop = true;
                }
                _changed.notify_one();
                _worker.join();
            }
        };
        Broker *broker = nullptr;
        template <class Action>
        int MPIOnOwner(Action action) { return broker ? broker->Call(std::move(action)) : action(); }

        class Operation final : public R::Detail::TransferProviderOperation
        {
            std::shared_ptr<Broker> _owner;
            std::shared_ptr<Pending> _pending;
            std::optional<R::TransferOperation> _caller;
            std::shared_ptr<Event> _event;
            bool _observed = false;

        public:
            Operation(R::TransferBinding binding, bool scheduled, std::shared_ptr<Broker> owner, std::shared_ptr<Event> observation)
                : _owner(std::move(owner)), _event(std::move(observation))
            {
                const double before = _event ? Time() : 0;
                if (_owner)
                    _pending = _owner->Begin(std::move(binding), scheduled, _event);
                else
                {
                    EventScope scope(_event.get());
                    if (event)
                        event->start = event->enqueue = Time();
                    _caller.emplace(scheduled ? binding.BeginScheduled() : binding.Begin());
                }
                if (_event)
                    _event->submit = Time() - before;
            }
            void WaitDestinationReady() override { Wait(); } // Conservative full readiness in this bounded provider.
            void Wait() override
            {
                if (_observed)
                    return;
                const double start = _event ? Time() : 0;
                if (_pending)
                    _pending->future.get();
                else if (_caller)
                {
                    EventScope scope(_event.get());
                    _caller->Wait();
                    if (event)
                        event->ready = Time();
                    _caller.reset();
                }
                if (_event)
                    _event->hostWait += Time() - start;
                _observed = true;
            }
            bool Test() override
            {
                if (_observed)
                    return true;
                if (_pending)
                {
                    if (_pending->future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                        return false;
                    _pending->future.get();
                    _observed = true;
                    return true;
                }
                EventScope scope(_event.get());
                if (_caller && !_caller->Test())
                    return false;
                if (event)
                    event->ready = Time();
                _caller.reset();
                _observed = true;
                return true;
            }
            ~Operation() override
            {
                try
                {
                    Wait();
                }
                catch (...)
                {
                }
            }
        };
        class Binding final : public R::Detail::TransferProviderBinding
        {
            R::TransferBinding _inner;
            std::shared_ptr<Broker> _owner;
            I _receiveBytes;

        public:
            Binding(R::TransferBinding inner, std::shared_ptr<Broker> owner, I bytes)
                : _inner(std::move(inner)), _owner(std::move(owner)), _receiveBytes(bytes) {}
            std::unique_ptr<R::Detail::TransferProviderOperation> Begin(R::ExecutionContext context, bool scheduled) const override
            {
                DNDS_check_throw_info(context.device == R::Device::Host() && (!context.stream || context.stream->GetDevice() == R::Device::Host()), "assessment provider is host-only");
                std::shared_ptr<Event> observation;
                if (auto *sample = trace.load())
                {
                    observation = std::make_shared<Event>();
                    observation->receiveBytes = _receiveBytes;
                    sample->events.push_back(observation);
                }
                return std::make_unique<Operation>(_inner, scheduled, _owner, std::move(observation));
            }
            void Enqueue(R::EagerEngine &, const R::ExecutionStream &) const override
            {
                DNDS_check_throw_info(false, "assessment provider requires ordinary CPU Begin/Wait");
            }
        };
        class Provider final : public R::TransferProvider
        {
            std::shared_ptr<Broker> _owner;
            R::TransferTransport _transport;

        public:
            Provider(std::shared_ptr<Broker> owner, R::TransferTransport transport) : _owner(std::move(owner)), _transport(transport) {}
            const char *Name() const override { return "plan53-bounded-host-provider"; }
            std::shared_ptr<R::Detail::TransferProviderBinding> Bind(const R::TransferPayload &p, R::Buffer source, R::Buffer destination,
                                                                     std::shared_ptr<const R::Reducer> reducer, R::TransferTransport, const R::TransferOptions &, bool valid) const override
            {
                DNDS_check_throw_info(valid && !reducer && source.GetDevice() == R::Device::Host() && destination.GetDevice() == R::Device::Host() &&
                                          !p.destination.IsVariable(),
                                      "unsupported assessment payload");
                R::PayloadPlan inner(p.route, p.source, p.destination, p.schema);
                I bytes = 0;
                for (const auto &rows : p.route.ReceiveRows())
                    bytes += I(rows.size()) * p.destination.Width() * sizeof(double);
                return std::make_shared<Binding>(inner.Bind(std::move(source), std::move(destination), _transport), _owner, bytes);
            }
        };
        std::shared_ptr<R::TransferProvider> provider;
        bool originalBindings = false;
    }

    struct RHSExecution::Data
    {
        std::string mode;
        int controller = -1, worker = -1;
        std::vector<int> placements;
        std::shared_ptr<Broker> owner;
        std::unique_ptr<Trace> sample;
    };
    RHSExecution::RHSExecution(const DNDS::MPIInfo &mpi, std::string mode, std::string transport) : _data(std::make_unique<Data>())
    {
        auto &d = *_data;
        d.mode = std::move(mode);
        DNDS_check_throw_info((d.mode == "production" || d.mode == "caller" || d.mode == "compute-worker" || d.mode == "mpi-worker") &&
                                  (transport == "direct" || transport == "packed") && (d.mode != "production" || transport == "direct"),
                              "invalid execution assessment mode");
        int provided = 0;
        MPI_Query_thread(&provided);
        DNDS_check_throw_info(provided >= MPI_THREAD_SERIALIZED && !broker && !provider, "assessment needs SERIALIZED support and one controller");
        cpu_set_t mask;
        CPU_ZERO(&mask);
        DNDS_check_throw_info(sched_getaffinity(0, sizeof(mask), &mask) == 0, "cannot query rank affinity");
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &mask))
            {
                if (d.controller < 0)
                    d.controller = cpu;
                else if (Topology(cpu, "core_id") != Topology(d.controller, "core_id") ||
                         Topology(cpu, "physical_package_id") != Topology(d.controller, "physical_package_id"))
                {
                    d.worker = cpu;
                    break;
                }
            }
        DNDS_check_throw_info(d.worker >= 0, "reserve two physical cores per rank in every control");
        int here[]{d.controller, d.worker, Topology(d.controller, "physical_package_id"), Topology(d.controller, "core_id"), NUMA(d.controller),
                   Topology(d.worker, "physical_package_id"), Topology(d.worker, "core_id"), NUMA(d.worker)};
        d.placements.resize(size_t(mpi.size) * 8);
        MPI_Allgather(here, 8, MPI_INT, d.placements.data(), 8, MPI_INT, mpi.comm);
        std::vector<std::pair<int, int>> physical;
        for (int rank = 0; rank < mpi.size; ++rank)
        {
            physical.emplace_back(d.placements[8 * rank + 2], d.placements[8 * rank + 3]);
            physical.emplace_back(d.placements[8 * rank + 5], d.placements[8 * rank + 6]);
        }
        std::sort(physical.begin(), physical.end());
        DNDS_check_throw_info(std::adjacent_find(physical.begin(), physical.end()) == physical.end(), "assessment oversubscribes physical cores");
        Pin(d.controller);
        if (d.mode == "mpi-worker")
        {
            d.owner = std::make_shared<Broker>(d.worker);
            broker = d.owner.get();
        }
        provider = std::make_shared<Provider>(d.owner, transport == "direct" ? R::TransferTransport::Direct : R::TransferTransport::HostPacked);
        originalBindings = true; // Mesh/CFV preparation and the reference use the original path.
    }
    RHSExecution::~RHSExecution()
    {
        trace = nullptr;
        provider.reset();
        broker = nullptr; // No MPI owners/operations may outlive this fixture owner.
        _data.reset();
    }
    int RHSExecution::HostThreads() const { return _data->mode == "compute-worker" ? 1 : 0; }
    void RHSExecution::PinCompute(R::EagerEngine &engine, const R::ExecutionStream &stream)
    {
        if (HostThreads())
        {
            engine.Submit(stream, [cpu = _data->worker](auto)
                          { Pin(cpu); });
            stream.Record().WaitHost();
        }
    }
    void RHSExecution::OriginalBindings(bool selected) { originalBindings = selected || _data->mode == "production"; }
    void RHSExecution::BeginSample(bool instrumented)
    {
        _data->sample.reset();
        if (instrumented)
        {
            _data->sample = std::make_unique<Trace>();
            _data->sample->events.reserve(256);
            _data->sample->core.reserve(128);
        }
        trace = _data->sample.get();
    }
    J RHSExecution::EndSample()
    {
        trace = nullptr; // Evaluate has observed every operation and its compute prefix.
        J result;
        if (!_data->sample)
            return result;
        auto &sample = *_data->sample;
        double staging = 0, packing = 0, posting = 0, progress = 0, unpack = 0, queue = 0, waits = 0, submit = 0;
        I bytes = 0, calls = 0, packs = 0, unpacks = 0, requests = 0, payload = 0;
        std::vector<std::pair<double, double>> windows;
        J timeline = J::array();
        for (const auto &p : sample.events)
        {
            DNDS_check_throw_info(p->start >= p->producer && p->ready >= p->start, "missing transfer completion observation");
            staging += p->staging;
            packing += p->pack;
            posting += p->post;
            progress += p->progress;
            unpack += p->unpack;
            queue += p->start - p->enqueue;
            waits += p->hostWait;
            submit += p->submit;
            bytes += p->stagingBytes;
            calls += p->stagingCalls;
            packs += p->packs;
            unpacks += p->unpacks;
            requests += p->requests;
            payload += p->receiveBytes;
            windows.emplace_back(p->start, p->ready);
            timeline.push_back({{"producer", p->producer - sample.origin}, {"enqueue", p->enqueue - sample.origin}, {"start", p->start - sample.origin}, {"first_post", p->firstPost ? p->firstPost - sample.origin : -1}, {"received", p->received ? p->received - sample.origin : -1}, {"ghost_ready", p->ready - sample.origin}});
        }
        std::sort(windows.begin(), windows.end());
        std::vector<std::pair<double, double>> united;
        for (auto window : windows)
            if (united.empty() || window.first > united.back().second)
                united.push_back(window);
            else
                united.back().second = std::max(united.back().second, window.second);
        double core = 0, overlap = 0;
        for (auto interval : sample.core)
        {
            core += interval.second - interval.first;
            for (auto window : united)
                overlap += std::max(0., std::min(interval.second, window.second) - std::max(interval.first, window.first));
        }
        result = {{"starts", sample.events.size()}, {"staging_seconds", staging}, {"pack_seconds", packing}, {"posting_seconds", posting}, {"progress_seconds", progress}, {"unpack_seconds", unpack}, {"queue_seconds", queue}, {"host_wait_seconds", waits}, {"controller_submit_seconds", submit}, {"staging_bytes", bytes}, {"staging_calls", calls}, {"pack_calls", packs}, {"unpack_calls", unpacks}, {"requests", requests}, {"receive_bytes", payload}, {"vr_core_seconds", core}, {"vr_core_overlap_seconds", overlap}, {"vr_core_calls", sample.core.size()}};
        result["timeline"] = std::move(timeline);
        return result;
    }
    J RHSExecution::Placement() const { return _data->placements; }
}

// All hot controller MPI calls are RPCs in mpi-worker mode. Setup/resource
// calls outside these wrappers happen with no pending payload, before/after
// workspace execution. Kernel callbacks perform no MPI. Original reference
// work uses the same owner but no background provider operations.
#define PLAN53_MPI(name, signature, arguments)                        \
    extern "C" int name signature                                     \
    {                                                                 \
        return Plan53::MPIOnOwner([&] { return P##name arguments; }); \
    }
PLAN53_MPI(MPI_Comm_rank, (MPI_Comm comm, int *rank), (comm, rank))
PLAN53_MPI(MPI_Comm_size, (MPI_Comm comm, int *size), (comm, size))
PLAN53_MPI(MPI_Comm_compare, (MPI_Comm a, MPI_Comm b, int *result), (a, b, result))
PLAN53_MPI(MPI_Barrier, (MPI_Comm comm), (comm))
PLAN53_MPI(MPI_Allreduce, (const void *in, void *out, int n, MPI_Datatype type, MPI_Op op, MPI_Comm comm), (in, out, n, type, op, comm))
#undef PLAN53_MPI

extern "C" int MPI_Pack(const void *in, int n, MPI_Datatype type, void *out, int bytes, int *position, MPI_Comm comm)
{
    return Plan53::MPIOnOwner([&]
                              { auto *p = Plan53::event; const double t = p ? Plan53::Time() : 0;
        int result = PMPI_Pack(in, n, type, out, bytes, position, comm); if (p) { p->pack += Plan53::Time() - t; ++p->packs; } return result; });
}
extern "C" int MPI_Unpack(const void *in, int bytes, int *position, void *out, int n, MPI_Datatype type, MPI_Comm comm)
{
    return Plan53::MPIOnOwner([&]
                              { auto *p = Plan53::event; const double t = p ? Plan53::Time() : 0;
        int result = PMPI_Unpack(in, bytes, position, out, n, type, comm); if (p) { p->unpack += Plan53::Time() - t; ++p->unpacks; } return result; });
}
extern "C" int MPI_Startall(int n, MPI_Request *requests)
{
    return Plan53::MPIOnOwner([&]
                              { auto *p = Plan53::event; const double t = p ? Plan53::Time() : 0;
        int result = PMPI_Startall(n, requests); if (p) { if (!p->firstPost) { p->firstPost = t; p->receiveRequests = requests; } p->post += Plan53::Time() - t; p->requests += n; } return result; });
}
extern "C" int MPI_Waitall(int n, MPI_Request *requests, MPI_Status *status)
{
    return Plan53::MPIOnOwner([&]
                              { auto *p = Plan53::event; const double t = p ? Plan53::Time() : 0;
        int result = PMPI_Waitall(n, requests, status); if (p) { p->progress += Plan53::Time() - t; if (requests == p->receiveRequests && !p->received) p->received = Plan53::Time(); } return result; });
}
extern "C" int MPI_Testall(int n, MPI_Request *requests, int *ready, MPI_Status *status)
{
    return Plan53::MPIOnOwner([&]
                              { auto *p = Plan53::event; const double t = p ? Plan53::Time() : 0;
        int result = PMPI_Testall(n, requests, ready, status); if (p) { p->progress += Plan53::Time() - t; if (*ready && requests == p->receiveRequests && !p->received) p->received = Plan53::Time(); } return result; });
}

using Metrics = DNDS::CFV::FiniteVolumePlan;
using Field = DNDS::CFV::FVField;
using TransferOptions = DNDS::Runtime::TransferOptions;
extern "C" DNDS::Runtime::TransferBinding RealHalo(const Metrics *, Field &, TransferOptions) asm("__real__ZNK4DNDS3CFV16FiniteVolumePlan8BindHaloERNS_7Algebra8DofFieldIdLm18446744073709551615EEENS_7Runtime15TransferOptionsE");
extern "C" DNDS::Runtime::TransferBinding WrapHalo(const Metrics *, Field &, TransferOptions) asm("__wrap__ZNK4DNDS3CFV16FiniteVolumePlan8BindHaloERNS_7Algebra8DofFieldIdLm18446744073709551615EEENS_7Runtime15TransferOptionsE");
extern "C" DNDS::Runtime::TransferBinding WrapHalo(const Metrics *m, Field &field, TransferOptions options)
{
    if (Plan53::originalBindings)
        return RealHalo(m, field, std::move(options));
    m->ValidateField(field, field.Width());
    auto route = m->Cells().BuildRowTransferPlan(m->Communication());
    options.provider = Plan53::provider;
    return DNDS::Runtime::PayloadPlan(route, field.Owned().Layout(), field.Ghost().Layout(), DNDS::Runtime::ElementSchema::For<double>()).Bind(field.Owned(), field.Ghost(), DNDS::Runtime::TransferTransport::Direct, options);
}
extern "C" DNDS::Algebra::DofHalo<> RealTracked(const Metrics *, Field &, TransferOptions) asm("__real__ZNK4DNDS3CFV16FiniteVolumePlan15BindTrackedHaloERNS_7Algebra8DofFieldIdLm18446744073709551615EEENS_7Runtime15TransferOptionsE");
extern "C" DNDS::Algebra::DofHalo<> WrapTracked(const Metrics *, Field &, TransferOptions) asm("__wrap__ZNK4DNDS3CFV16FiniteVolumePlan15BindTrackedHaloERNS_7Algebra8DofFieldIdLm18446744073709551615EEENS_7Runtime15TransferOptionsE");
extern "C" DNDS::Algebra::DofHalo<> WrapTracked(const Metrics *m, Field &field, TransferOptions options)
{
    if (Plan53::originalBindings)
        return RealTracked(m, field, std::move(options));
    m->ValidateField(field, field.Width());
    options.provider = Plan53::provider;
    return DNDS::Algebra::DofHalo<>(field, m->Cells().BuildRowTransferPlan(m->Communication()), options);
}
using Device = DNDS::Runtime::Device;
using Context = DNDS::Runtime::ExecutionContext;
using Completion = DNDS::Runtime::Completion;
extern "C" Completion RealCopy(const void *, Device, void *, Device, size_t, Context) asm("__real__ZN4DNDS7Runtime6Detail9CopyBytesEPKvNS0_6DeviceEPvS4_mNS0_16ExecutionContextE");
extern "C" Completion WrapCopy(const void *, Device, void *, Device, size_t, Context) asm("__wrap__ZN4DNDS7Runtime6Detail9CopyBytesEPKvNS0_6DeviceEPvS4_mNS0_16ExecutionContextE");
extern "C" Completion WrapCopy(const void *in, Device source, void *out, Device destination, size_t bytes, Context context)
{
    auto *p = Plan53::event;
    const double start = p ? Plan53::Time() : 0;
    auto result = RealCopy(in, source, out, destination, bytes, context);
    if (p)
    {
        p->staging += Plan53::Time() - start;
        p->stagingBytes += bytes;
        ++p->stagingCalls;
    }
    return result;
}
using VR = DNDS::CFV::VariationalReconstructionPlan;
using Work = DNDS::CFV::ReconstructionWorkspace;
using Boundary = DNDS::CFV::ReconstructionBoundaryValues;
using Region = DNDS::CFV::ReconstructionRegion;
extern "C" void RealSweep(const VR *, const Field &, const Field &, Work &, const Boundary &, Region) asm("__real__ZNK4DNDS3CFV29VariationalReconstructionPlan22DoReconstructionIterToERKNS_7Algebra8DofFieldIdLm18446744073709551615EEES6_RNS0_23ReconstructionWorkspaceERKNS0_28ReconstructionBoundaryValuesENS0_20ReconstructionRegionE");
extern "C" void WrapSweep(const VR *, const Field &, const Field &, Work &, const Boundary &, Region) asm("__wrap__ZNK4DNDS3CFV29VariationalReconstructionPlan22DoReconstructionIterToERKNS_7Algebra8DofFieldIdLm18446744073709551615EEES6_RNS0_23ReconstructionWorkspaceERKNS0_28ReconstructionBoundaryValuesENS0_20ReconstructionRegionE");
extern "C" void WrapSweep(const VR *vr, const Field &mean, const Field &input, Work &work, const Boundary &boundary, Region region)
{
    auto *p = region == Region::Core ? Plan53::trace.load() : nullptr;
    const double start = p ? Plan53::Time() : 0;
    RealSweep(vr, mean, input, work, boundary, region);
    if (p)
        p->core.emplace_back(start, Plan53::Time());
}
