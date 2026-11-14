#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/MPI/Failure.hpp"
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <algorithm>

namespace DNDS::Runtime
{
    void StreamEvent::State::NotifyCompletion() const
    {
        std::vector<std::function<void()>> notifications;
        {
            std::lock_guard<std::mutex> lock(_notificationMutex);
            if (_completed)
                return;
            _completed = true;
            notifications.swap(_notifications);
        }
        for (const auto &notify : notifications)
            notify();
    }
    void StreamEvent::State::OnCompletion(std::function<void()> notify) const
    {
        DNDS_check_throw_info(notify, "empty completion notification");
        {
            std::lock_guard<std::mutex> lock(_notificationMutex);
            if (!_completed)
            {
                _notifications.push_back(std::move(notify));
                return;
            }
        }
        notify();
    }
    bool StreamEvent::Ready() const
    {
        if (!_state)
            return true;
        bool ready;
        try
        {
            ready = _state->Ready();
        }
        catch (...)
        {
            _state->NotifyCompletion(); // Providers report only retired failures.
            throw;
        }
        if (ready)
            _state->NotifyCompletion();
        return ready;
    }
    void StreamEvent::WaitHost() const
    {
        if (!_state)
            return;
        try
        {
            _state->Wait();
        }
        catch (...)
        {
            _state->NotifyCompletion();
            throw;
        }
        _state->NotifyCompletion();
    }
    void StreamEvent::OnCompletion(std::function<void()> notify) const
    {
        if (_state)
            _state->OnCompletion(std::move(notify));
        else
            notify();
    }
    namespace
    {
        struct CPUCommand
        {
            StreamLauncher::Work work;
            StreamEvent dependency;
            bool wait = false, cleanup = false;
            StreamLauncher::Retirement retire;
            std::shared_ptr<std::atomic<bool>> dependencyCompleted;
        };
        struct CPULane
        {
            std::deque<CPUCommand> commands;
            uint64_t submitted = 0, completed = 0, failedAt = 0;
            std::exception_ptr error;
            bool running = false;
            std::vector<std::pair<uint64_t, std::weak_ptr<const StreamEvent::State>>> events;
        };

        class CPUExecutor : public std::enable_shared_from_this<CPUExecutor>
        {
            std::mutex _mutex;
            std::condition_variable _changed;
            std::vector<std::weak_ptr<CPULane>> _lanes;
            std::vector<std::thread> _workers;
            size_t _next = 0;
            bool _stopping = false;
            uint64_t _wakeGeneration = 0;
            const std::thread::id _controller = std::this_thread::get_id();
            bool _progressingDependencies = false; // Controller thread only.
            const bool _solverFailures = MPI::SolverFailureScopeActive();

            bool Step()
            {
                std::shared_ptr<CPULane> lane;
                CPUCommand command;
                std::exception_ptr priorError;
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    const auto count = _lanes.size();
                    for (size_t i = 0; i < count; ++i)
                    {
                        const auto candidate = (_next + i) % count;
                        auto current = _lanes[candidate].lock();
                        if (current && !current->running && !current->commands.empty() &&
                            (!current->commands.front().wait || current->commands.front().dependencyCompleted->load()))
                        {
                            lane = std::move(current);
                            lane->running = true;
                            command = std::move(lane->commands.front());
                            lane->commands.pop_front();
                            priorError = lane->error;
                            _next = (candidate + 1) % count;
                            break;
                        }
                    }
                }
                if (!lane)
                    return false;
                std::exception_ptr error;
                try
                {
                    // Invoke backend/event/user code with no executor lock held.
                    // Only signalled waits reach a worker. Observe the completed
                    // result here to propagate errors, never to progress MPI.
                    if (command.wait)
                        DNDS_check_throw_info(command.dependency.Ready(),
                                              "event notified before completion");
                    if (!command.wait && (!priorError || command.cleanup))
                        command.work(ExecutionContext::Host());
                }
                catch (...)
                {
                    if (_solverFailures)
                    {
                        MPI::AbortIfSolverFailure("solver CPU command failed");
                        MPI::AbortCaught(MPI_COMM_NULL, "solver CPU worker failed");
                    }
                    error = std::current_exception();
                }
                if (command.retire)
                    command.retire(error ? error : priorError);
                // Destroy captures before advancing completion: owning CPU
                // resources and cleanup belong to this ordered command.
                command = {};
                std::vector<std::shared_ptr<const StreamEvent::State>> notifications;
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    ++lane->completed;
                    if (error && !lane->error)
                    {
                        lane->error = error;
                        lane->failedAt = lane->completed;
                    }
                    lane->running = false;
                    for (auto it = lane->events.begin(); it != lane->events.end();)
                        if (it->first <= lane->completed || it->second.expired())
                        {
                            if (auto event = it->second.lock())
                                notifications.push_back(std::move(event));
                            it = lane->events.erase(it);
                        }
                        else
                            ++it;
                    ++_wakeGeneration;
                }
                for (const auto &event : notifications)
                    event->NotifyCompletion();
                _changed.notify_all();
                return true;
            }
            void Worker()
            {
                for (;;)
                {
                    uint64_t observed;
                    {
                        std::lock_guard<std::mutex> lock(_mutex);
                        observed = _wakeGeneration;
                    }
                    if (Step())
                        continue;
                    std::unique_lock<std::mutex> lock(_mutex);
                    if (_stopping)
                        return;
                    _changed.wait(lock, [&]
                                  { return _stopping || _wakeGeneration != observed; });
                }
            }

        public:
            void Wake()
            {
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    ++_wakeGeneration;
                }
                _changed.notify_all();
            }
            /// Observe foreign events only on the controller. Portable MPI and
            /// CUDA host retirement may need progress; compute workers do not.
            void ProgressDependencies()
            {
                if (std::this_thread::get_id() != _controller || _progressingDependencies)
                    return;
                struct ProgressScope
                {
                    bool &active;
                    explicit ProgressScope(bool &value) : active(value) { active = true; }
                    ~ProgressScope() { active = false; }
                } scope(_progressingDependencies);
                std::vector<StreamEvent> events;
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    for (const auto &weak : _lanes)
                        if (auto lane = weak.lock())
                            for (const auto &command : lane->commands)
                                if (command.wait && !command.dependencyCompleted->load())
                                    events.push_back(command.dependency);
                }
                for (const auto &event : events)
                    try
                    {
                        event.Ready();
                    }
                    catch (...)
                    {
                        // Ready notified the terminal failure. The queued wait
                        // observes it on its own stream, preserving cleanup order.
                        MPI::AbortIfSolverFailure("solver dependency progress failed");
                    }
            }
            void ObservePrefix(const std::shared_ptr<CPULane> &lane, uint64_t prefix,
                               const std::shared_ptr<const StreamEvent::State> &event)
            {
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    if (lane->completed < prefix)
                    {
                        lane->events.emplace_back(prefix, event);
                        return;
                    }
                }
                event->NotifyCompletion();
            }
            explicit CPUExecutor(size_t threads)
            {
                try
                {
                    for (size_t i = 0; i < threads; ++i)
                        _workers.emplace_back([this]
                                              { Worker(); });
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("solver CPU worker creation failed");
                    {
                        std::lock_guard<std::mutex> lock(_mutex);
                        _stopping = true;
                    }
                    _changed.notify_all();
                    for (auto &worker : _workers)
                        worker.join();
                    throw;
                }
            }
            ~CPUExecutor() { Shutdown(); }
            void Shutdown()
            {
                if (std::uncaught_exceptions())
                    MPI::AbortIfSolverFailure("solver unwinding before CPU worker join");
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _stopping = true;
                }
                _changed.notify_all();
                for (auto &worker : _workers)
                    if (worker.joinable())
                        worker.join();
            }
            std::shared_ptr<CPULane> NewLane()
            {
                auto lane = std::make_shared<CPULane>();
                std::lock_guard<std::mutex> lock(_mutex);
                // Reuse expired slots instead of retaining all historical lanes.
                for (auto &slot : _lanes)
                    if (slot.expired())
                    {
                        slot = lane;
                        return lane;
                    }
                _lanes.emplace_back(lane);
                return lane;
            }
            void Submit(const std::shared_ptr<CPULane> &lane, CPUCommand command)
            {
                const bool importsEvent = command.wait;
                if (command.wait)
                {
                    command.dependencyCompleted = std::make_shared<std::atomic<bool>>(false);
                    command.dependency.OnCompletion([weak = weak_from_this(), done = command.dependencyCompleted]
                                                    {
                        done->store(true);
                        if (auto executor = weak.lock())
                            executor->Wake(); });
                }
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    DNDS_check_throw_info(!_stopping && lane->submitted < UINT64_MAX, "CPU execution lane unavailable");
                    lane->commands.push_back(std::move(command));
                    ++lane->submitted;
                    ++_wakeGeneration;
                }
                _changed.notify_all();
                if (importsEvent)
                    ProgressDependencies();
                if (_workers.empty())
                    while (Step())
                    {
                    }
            }
            uint64_t Tail(const std::shared_ptr<CPULane> &lane)
            {
                std::lock_guard<std::mutex> lock(_mutex);
                return lane->submitted;
            }
            bool Ready(const std::shared_ptr<CPULane> &lane, uint64_t prefix)
            {
                std::lock_guard<std::mutex> lock(_mutex);
                if (lane->completed < prefix)
                    return false;
                if (lane->error && lane->failedAt <= prefix)
                    std::rethrow_exception(lane->error);
                return true;
            }
            bool Poll(const std::shared_ptr<CPULane> &lane, uint64_t prefix)
            {
                if (Ready(lane, prefix))
                    return true;
                ProgressDependencies();
                if (_workers.empty() && std::this_thread::get_id() == _controller)
                    Step(); // Allows an imported inline-executor dependency to progress.
                return Ready(lane, prefix);
            }
            void ProgressInline()
            {
                ProgressDependencies();
                if (_workers.empty())
                    Step();
            }
            void Wait(const std::shared_ptr<CPULane> &lane, uint64_t prefix)
            {
                while (!Ready(lane, prefix))
                {
                    ProgressDependencies();
                    if (_workers.empty() && Step())
                        continue;
                    // Controller waits drive foreign backend progress. Worker
                    // waits sleep; normal queued dependencies never enter here.
                    if (std::this_thread::get_id() == _controller)
                    {
                        std::this_thread::yield();
                        continue;
                    }
                    std::unique_lock<std::mutex> lock(_mutex);
                    _changed.wait(lock, [&]
                                  { return lane->completed >= prefix || _stopping; });
                }
            }
        };

        class CPUEvent : public StreamEvent::State
        {
            std::shared_ptr<CPUExecutor> _executor;
            std::shared_ptr<CPULane> _lane;
            uint64_t _prefix;

        public:
            CPUEvent(std::shared_ptr<CPUExecutor> executor, std::shared_ptr<CPULane> lane, uint64_t prefix)
                : _executor(std::move(executor)), _lane(std::move(lane)), _prefix(prefix) {}
            bool Ready() const override { return _executor->Poll(_lane, _prefix); }
            void Wait() const override { _executor->Wait(_lane, _prefix); }
            unsigned AutonomousWaitPriority() const noexcept override { return 1; }
        };

        class CPULauncher : public StreamLauncher
        {
            std::shared_ptr<CPUExecutor> _executor;
            std::shared_ptr<CPULane> _lane;

        public:
            explicit CPULauncher(std::shared_ptr<CPUExecutor> executor)
                : _executor(std::move(executor)), _lane(_executor->NewLane()) {}
            ExecutionContext Context() const override { return ExecutionContext::Host(); }
            void Submit(Work work, CommandKind kind, Retirement retire) override
            {
                DNDS_check_throw_info(work, "empty CPU command");
                _executor->Submit(_lane, {std::move(work), {}, false, kind != CommandKind::Compute, std::move(retire)});
            }
            void WaitFor(StreamEvent event) override { _executor->Submit(_lane, {{}, std::move(event), true, false}); }
            StreamEvent Record() override
            {
                const auto prefix = _executor->Tail(_lane);
                auto event = std::make_shared<CPUEvent>(_executor, _lane, prefix);
                _executor->ObservePrefix(_lane, prefix, event);
                return StreamEvent(std::move(event));
            }
            void Drain() override { _executor->Wait(_lane, _executor->Tail(_lane)); }
            Buffer Allocate(size_t bytes) override { return Buffer::Allocate(bytes); }
            Buffer Allocate(size_t bytes, AllocationOptions options) override { return Buffer::Allocate(bytes, Device::Host(), options); }
        };
    }

    struct ExecutionRuntime
    {
        enum class Phase
        {
            Open,
            Closing,
            Closed
        };
        std::atomic<Phase> phase{Phase::Open};
        const std::thread::id controller = std::this_thread::get_id();
        std::shared_ptr<CPUExecutor> host;
        std::vector<std::shared_ptr<StreamLauncher>> streams;
        std::vector<std::shared_ptr<ExecutionProgress>> pending;
        std::vector<std::shared_ptr<const void>> retainedUntilDrain;
        std::shared_ptr<AccessTrackingContext> tracking;
        std::mutex mutex;
        bool draining = false;
        std::atomic<size_t> openBorrows{0};
        explicit ExecutionRuntime(size_t threads) : host(std::make_shared<CPUExecutor>(threads)) {}
        void CheckOpen() const
        {
            DNDS_check_throw_info(phase.load() == Phase::Open, "execution context is closing or closed");
        }
        void CheckController() const
        {
            DNDS_check_throw_info(std::this_thread::get_id() == controller, "execution progress requires the creating controller thread");
        }
        void Register(std::shared_ptr<ExecutionProgress> operation)
        {
            CheckController();
            DNDS_check_throw_info(operation, "cannot register empty execution progress");
            {
                std::lock_guard<std::mutex> lock(mutex);
                CheckOpen();
                DNDS_check_throw_info(!draining, "cannot register external work during synchronization");
                DNDS_check_throw_info(std::find(pending.begin(), pending.end(), operation) == pending.end(), "operation already registered");
                draining = true;
            }
            try
            {
                // Explicit Wait may have completed earlier operations without
                // a whole-engine Synchronize. Retire their registry entries,
                // rather than accumulating one owner per solver iteration.
                // Poll is nonblocking and no registry lock spans backend code.
                for (auto it = pending.begin(); it != pending.end();)
                    if ((*it)->Poll())
                        it = pending.erase(it);
                    else
                        ++it;
                pending.push_back(std::move(operation));
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("solver operation registration/progress failed");
                std::lock_guard<std::mutex> lock(mutex);
                draining = false;
                throw;
            }
            std::lock_guard<std::mutex> lock(mutex);
            draining = false;
        }
    };
    ExecutionBorrow::ExecutionBorrow(std::shared_ptr<ExecutionRuntime> runtime, ExecutionContext context)
        : _runtime(std::move(runtime)), _context(context)
    {
        DNDS_check_throw_info(_runtime, "execution context no longer exists");
        _runtime->CheckOpen();
        ++_runtime->openBorrows;
    }
    ExecutionBorrow::ExecutionBorrow(ExecutionBorrow &&other) noexcept
        : _runtime(std::move(other._runtime)), _context(other._context) {}
    ExecutionBorrow &ExecutionBorrow::operator=(ExecutionBorrow &&other) noexcept
    {
        if (this != &other)
        {
            if (_runtime)
                --_runtime->openBorrows;
            _runtime = std::move(other._runtime);
            _context = other._context;
        }
        return *this;
    }
    ExecutionBorrow::~ExecutionBorrow()
    {
        if (_runtime)
            --_runtime->openBorrows;
    }
    ExecutionContext ExecutionBorrow::Context() const
    {
        DNDS_check_throw_info(_runtime, "execution borrow was moved");
        _runtime->CheckOpen();
        return _context;
    }
    EagerEngine::EagerEngine(size_t threads, std::shared_ptr<AccessTrackingContext> tracking) : _state(std::make_shared<ExecutionRuntime>(threads))
    {
        _state->tracking = std::move(tracking);
    }
    EagerEngine::EagerEngine(std::shared_ptr<ExecutionRuntime> state) : _state(std::move(state)), _shutdownOwner(false)
    {
        _state->CheckOpen();
        ++_state->openBorrows;
    }
    EagerEngine::~EagerEngine()
    {
        if (!_shutdownOwner)
        {
            --_state->openBorrows;
            return;
        }
        if (_state->openBorrows.load())
            std::terminate();
        if (std::this_thread::get_id() != _state->controller)
            std::terminate();
        if (std::uncaught_exceptions())
            MPI::AbortIfSolverFailure("solver unwinding before stream shutdown");
        try
        {
            Close();
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("solver stream shutdown failed");
        } // Explicit Synchronize reports errors; shutdown still drains.
    }
    ExecutionStream EagerEngine::CreateCPUStream() { return CreateStream(std::make_shared<CPULauncher>(_state->host)); }
    bool EagerEngine::ChecksAccess() const { return bool(_state->tracking); }
    ExecutionStream EagerEngine::CreateStream(std::shared_ptr<StreamLauncher> launcher)
    {
        DNDS_check_throw_info(launcher, "stream needs a launcher");
        std::lock_guard<std::mutex> lock(_state->mutex);
        _state->CheckOpen();
        _state->streams.push_back(launcher);
        return ExecutionStream(std::move(launcher), _state, _state->tracking,
                               _state->tracking ? _state->tracking->NewLane() : 0);
    }
    void EagerEngine::Submit(const ExecutionStream &stream, StreamLauncher::Work work)
    {
        DNDS_check_throw_info(stream._runtime.lock() == _state && work, "invalid eager submission or foreign stream");
        SubmitOperation(stream, std::move(work), [](std::exception_ptr) {});
    }
    Buffer EagerEngine::AllocateBuffer(const ExecutionStream &stream, size_t bytes, AllocationOptions options)
    {
        DNDS_check_throw_info(stream._runtime.lock() == _state, "invalid allocation stream");
        const auto launcher = stream.LockLauncher();
        if (!_state->tracking)
            return launcher->Allocate(bytes, options);
        auto buffer = launcher->AllocateChecked(bytes, options);
        // Allocation/zero initialization is itself a producer, even when native
        // allocation returned a host handle before its device work completed.
        SubmitOperation(stream, [](ExecutionContext) {}, [](std::exception_ptr) {}, {{buffer, AccessMode::Write}});
        return buffer;
    }
    void EagerEngine::SubmitOperation(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire,
                                      std::vector<ScheduledBufferAccess> accesses)
    {
        DNDS_check_throw_info(stream._runtime.lock() == _state && work && retire,
                              "invalid eager operation or foreign stream");
        const auto launcher = stream.LockLauncher();
        if (!_state->tracking)
            launcher->Submit(std::move(work), CommandKind::Compute, std::move(retire));
        else
            _state->tracking->Submit(stream._lane, accesses, [&](AccessTrackingContext::Submission submission)
                                     { launcher->Submit(std::move(work), CommandKind::Compute,
                                                        [submission, retire = std::move(retire)](std::exception_ptr error) noexcept
                                                        {
                                                            retire(error);
                                                            submission.Complete();
                                                        }); });
    }
    void EagerEngine::SubmitCleanup(const ExecutionStream &stream, StreamLauncher::Work work, std::vector<ScheduledBufferAccess> accesses)
    {
        DNDS_check_throw_info(stream._runtime.lock() == _state && work, "invalid eager cleanup or foreign stream");
        const auto launcher = stream.LockLauncher();
        if (!_state->tracking)
            launcher->Submit(std::move(work), CommandKind::Cleanup);
        else
            _state->tracking->Submit(stream._lane, accesses, [&](AccessTrackingContext::Submission submission)
                                     { launcher->Submit(std::move(work), CommandKind::Cleanup,
                                                        [submission](std::exception_ptr) noexcept
                                                        { submission.Complete(); }); });
    }
    void EagerEngine::SubmitOrderedRelease(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire,
                                           std::vector<ScheduledBufferAccess> accesses)
    {
        DNDS_check_throw_info(stream._runtime.lock() == _state && work && retire,
                              "invalid eager release or foreign stream");
        const auto launcher = stream.LockLauncher();
        if (!_state->tracking)
            launcher->Submit(std::move(work), CommandKind::OrderedRelease, std::move(retire));
        else
            _state->tracking->Submit(stream._lane, accesses, [&](AccessTrackingContext::Submission submission)
                                     { launcher->Submit(std::move(work), CommandKind::OrderedRelease,
                                                        [submission, retire = std::move(retire)](std::exception_ptr error) noexcept
                                                        { retire(error); submission.Complete(); }); });
    }
    void EagerEngine::RegisterProgress(std::shared_ptr<ExecutionProgress> operation) { _state->Register(std::move(operation)); }
    void EagerEngine::RetainUntilSynchronize(std::shared_ptr<const void> owner)
    {
        _state->CheckController();
        DNDS_check_throw_info(owner, "cannot retain an empty controller resource");
        std::lock_guard<std::mutex> lock(_state->mutex);
        _state->CheckOpen();
        DNDS_check_throw_info(!_state->draining, "cannot retain a resource during synchronization");
        _state->retainedUntilDrain.push_back(std::move(owner));
    }
    bool EagerEngine::IsClosed() const { return _state->phase.load() == ExecutionRuntime::Phase::Closed; }
    void EagerEngine::Synchronize() { Drain(false); }
    void EagerEngine::Close()
    {
        DNDS_check_throw_info(_shutdownOwner, "only the owning engine may close the execution runtime");
        Drain(true);
    }
    void EagerEngine::Drain(bool closing)
    {
        _state->CheckController();
        std::vector<std::shared_ptr<StreamLauncher>> streams;
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            if (IsClosed())
                return;
            DNDS_check_throw_info(!closing || !_state->openBorrows.load(), "cannot close execution context with open immediate borrows");
            DNDS_check_throw_info(!_state->draining, "recursive execution synchronization");
            streams = _state->streams; // Allocate before changing lifecycle state.
            if (closing)
                _state->phase = ExecutionRuntime::Phase::Closing;
            _state->draining = true;
        }
        std::exception_ptr first;
        auto failure = [&]
        {
            MPI::AbortIfSolverFailure("solver execution progress failed");
            if (!first)
                first = std::current_exception();
        };
        try
        {
            // A prefix is one boundary per stream, not an event per kernel.
            std::vector<StreamEvent> prefixes;
            prefixes.reserve(streams.size());
            for (const auto &stream : streams)
                prefixes.push_back(stream->Record());
            std::vector<bool> complete(prefixes.size(), false);
            std::vector<index> waitOrder;
            waitOrder.reserve(prefixes.size());
            for (index i = 0; i < index(prefixes.size()); ++i)
                if (prefixes[i].Provider() && prefixes[i].Provider()->AutonomousWaitPriority())
                    waitOrder.push_back(i);
            std::stable_sort(waitOrder.begin(), waitOrder.end(), [&](index a, index b)
                             { return prefixes[a].Provider()->AutonomousWaitPriority() > prefixes[b].Provider()->AutonomousWaitPriority(); });
            for (;;)
            {
                // Submission is serialized with this loop. Each callback runs
                // without the registry lock; completion drops owners here on
                // the controller, never on a CPU event-observer thread.
                for (auto it = _state->pending.begin(); it != _state->pending.end();)
                {
                    bool done = false;
                    try
                    {
                        done = (*it)->Poll();
                    }
                    catch (...)
                    {
                        failure();
                        done = true;
                    }
                    if (done)
                        it = _state->pending.erase(it);
                    else
                        ++it;
                }
                // Register rejects new progress while draining, and registered
                // operations have now released their owners on this controller.
                // Only after all such work retires may a provider's autonomous
                // Wait replace repeated Ready queries. Higher-priority native
                // producers finish before self-progressing CPU consumers.
                if (_state->pending.empty())
                    for (const auto i : waitOrder)
                        if (!complete[i])
                            try
                            {
                                prefixes[i].WaitHost();
                                complete[i] = true;
                            }
                            catch (...)
                            {
                                failure();
                                complete[i] = true;
                            }
                bool all = _state->pending.empty();
                for (size_t i = 0; i < prefixes.size(); ++i)
                {
                    if (!complete[i])
                        try
                        {
                            complete[i] = prefixes[i].Ready();
                        }
                        catch (...)
                        {
                            failure();
                            complete[i] = true;
                        }
                    all = all && complete[i];
                }
                if (all)
                    break;
                std::this_thread::yield();
            }
        }
        catch (...)
        {
            // Failure to record a prefix cannot justify blocking on a stream
            // while registered external work still needs controller progress.
            failure();
            while (!_state->pending.empty())
            {
                _state->host->ProgressInline();
                for (auto it = _state->pending.begin(); it != _state->pending.end();)
                {
                    bool done = false;
                    try
                    {
                        done = (*it)->Poll();
                    }
                    catch (...)
                    {
                        failure();
                        done = true;
                    }
                    if (done)
                        it = _state->pending.erase(it);
                    else
                        ++it;
                }
            }
        }
        // Registered work is retired, including any late device unpack. Drain
        // the final native tails before resource release, even after an error.
        for (const auto &stream : streams)
            try
            {
                stream->Drain();
            }
            catch (...)
            {
                failure();
            }
        if (closing)
            _state->host->Shutdown();
        // No registry lock around destruction: a resource may release MPI or
        // backend allocations. Every stream tail is retired at this point,
        // including failure paths. Destruction runs only on the controller.
        _state->retainedUntilDrain.clear();
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            if (closing)
            {
                _state->streams.clear();
                _state->phase = ExecutionRuntime::Phase::Closed;
            }
            _state->draining = false;
        }
        if (first)
            std::rethrow_exception(first);
    }
    std::shared_ptr<StreamLauncher> ExecutionStream::LockLauncher() const
    {
        const auto runtime = _runtime.lock();
        DNDS_check_throw_info(runtime, "execution context no longer exists");
        runtime->CheckOpen();
        auto launcher = _launcher.lock();
        DNDS_check_throw_info(launcher, "empty execution stream");
        return launcher;
    }
    void ExecutionStream::RegisterProgress(std::shared_ptr<ExecutionProgress> operation) const
    {
        const auto runtime = _runtime.lock();
        DNDS_check_throw_info(runtime, "execution context no longer exists");
        runtime->Register(std::move(operation));
    }
    ExecutionContext ExecutionStream::Context() const
    {
        return LockLauncher()->Context();
    }
    ExecutionBorrow ExecutionStream::BorrowContext() const
    {
        const auto launcher = LockLauncher();
        return ExecutionBorrow(_runtime.lock(), launcher->Context());
    }
    EagerEngine ExecutionStream::LockEngine() const
    {
        const auto runtime = _runtime.lock();
        DNDS_check_throw_info(runtime, "execution context no longer exists");
        runtime->CheckOpen();
        return EagerEngine(runtime);
    }
    void ExecutionStream::WaitFor(StreamEvent event) const
    {
        const auto launcher = LockLauncher();
        if (_tracking)
        {
            const auto order = event._order;
            _tracking->WaitFor(_lane, order, [&]
                               { launcher->WaitFor(std::move(event)); });
        }
        else
            launcher->WaitFor(std::move(event));
    }
    StreamEvent ExecutionStream::Record() const
    {
        const auto launcher = LockLauncher();
        if (!_tracking)
            return launcher->Record();
        StreamEvent event;
        const auto order = _tracking->Record(_lane, [&]
                                             { event = launcher->Record(); });
        event._order = order;
        return event;
    }
}
