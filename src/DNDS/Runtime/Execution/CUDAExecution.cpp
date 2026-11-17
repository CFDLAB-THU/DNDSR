#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Diagnostics/Timeline.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <deque>
#ifdef DNDS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif

namespace DNDS::Runtime
{
#ifdef DNDS_USE_CUDA
    namespace
    {
        void Check(cudaError_t status) { DNDS_check_throw_info(status == cudaSuccess, cudaGetErrorString(status)); }
        class DeviceScope
        {
            int _previous = -1;

        public:
            explicit DeviceScope(int ordinal)
            {
                Check(cudaGetDevice(&_previous));
                Check(cudaSetDevice(ordinal));
            }
            ~DeviceScope()
            {
                if (cudaSetDevice(_previous) != cudaSuccess)
                    std::terminate();
            }
        };
        using NativeStream = StreamResource;
        struct RetainedCommand
        {
            StreamLauncher::Work work;
            StreamLauncher::Retirement retire;
            bool cleanup = false;
            std::exception_ptr error;
        };

        /// A closed, immutable native prefix. All dependencies refer to earlier
        /// recorded prefixes, never unrecorded events or future promises.
        class CUDAPrefix final : public StreamEvent::State
        {
            mutable std::shared_ptr<NativeStream> _stream;
            // Identity without extending the native stream's lifetime after
            // retirement. A raw CUDA stream handle can be reused after destroy.
            const std::weak_ptr<NativeStream> _origin;
            const int _ordinal;
            cudaEvent_t _event = nullptr;
            mutable std::vector<RetainedCommand> _commands;
            mutable std::vector<StreamEvent> _dependencies;
            mutable std::once_flag _retired;
            mutable std::exception_ptr _error;
            std::atomic<bool> _recorded{false};
            mutable std::atomic<bool> _finished{false};

            bool Finished() const
            {
                if (!_finished.load(std::memory_order_acquire))
                    return false;
                // The release store follows every error write and capture
                // retirement. The cached path must preserve terminal errors.
                if (_error)
                    std::rethrow_exception(_error);
                return true;
            }

            void Finish(std::exception_ptr nativeError) const
            {
                std::call_once(_retired, [&]
                               {
                                   if (!_error)
                                       _error = nativeError;
                                   for (const auto &dependency : _dependencies)
                                       try
                                       {
                                           dependency.WaitHost();
                                       }
                                       catch (...)
                                       {
                                           if (!_error)
                                               _error = std::current_exception();
                                       }
                                   _dependencies.clear();
                                   for (auto &command : _commands)
                                   {
                                       if (!_error)
                                           _error = command.error;
                                       if (command.cleanup)
                                           try
                                           {
                                               command.work(ExecutionContext{_stream->GetDevice(), _stream});
                                           }
                                           catch (...)
                                           {
                                               if (!_error)
                                                   _error = std::current_exception();
                                           }
                                       if (command.retire)
                                           command.retire(_error);
                                       command = {}; // Drop captured owners before completion is visible.
                                   }
                                   _commands.clear();
                                   _stream.reset(); // A completed event does not own a live stream.
                                   _finished.store(true, std::memory_order_release); });
                NotifyCompletion(); // CPU consumers wake only after host retirement.
                if (_error)
                    std::rethrow_exception(_error);
            }

        public:
            explicit CUDAPrefix(std::shared_ptr<NativeStream> stream)
                : _stream(std::move(stream)), _origin(_stream), _ordinal(_stream->GetDevice().ordinal)
            {
                DeviceScope scope(_ordinal);
                // Host waits occur only at explicit observation/drain points.
                // Let the OS sleep there: a default CUDA event can spin inside
                // cudaEventSynchronize even after engine-side polling ceased.
                // Query and device stream-wait semantics are unchanged; this
                // flag adds no synchronization to queued kernel submission.
                Check(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming | cudaEventBlockingSync));
            }
            ~CUDAPrefix()
            {
                // The launcher retains every prefix until observation/drain;
                // events may outlive it, but their work has already been retired.
                DeviceScope scope(_ordinal);
                if (cudaEventDestroy(_event) != cudaSuccess)
                    std::terminate();
            }
            void Record(std::vector<RetainedCommand> &commands, std::vector<StreamEvent> &dependencies,
                        std::exception_ptr error)
            {
                DeviceScope scope(_ordinal);
                Check(cudaEventRecord(_event, CUDAStream(*_stream)));
                _commands.swap(commands);
                _dependencies.swap(dependencies);
                _error = error;
                _recorded = true;
            }
            bool Ready() const override
            {
                DNDS_check_throw_info(_recorded, "unrecorded CUDA prefix");
                if (Finished())
                    return true;
                DeviceScope scope(_ordinal);
                auto status = cudaEventQuery(_event);
                if (status == cudaErrorNotReady)
                    return false;
                std::exception_ptr error;
                try
                {
                    Check(status);
                }
                catch (...)
                {
                    error = std::current_exception();
                }
                Finish(error);
                return true;
            }
            void Wait() const override
            {
                DNDS_check_throw_info(_recorded, "unrecorded CUDA prefix");
                if (Finished())
                    return;
                DeviceScope scope(_ordinal);
                std::exception_ptr error;
                try
                {
                    Check(cudaEventSynchronize(_event));
                }
                catch (...)
                {
                    error = std::current_exception();
                }
                Finish(error);
            }
            unsigned AutonomousWaitPriority() const noexcept override { return 2; }
            bool BelongsTo(const NativeStream &consumer) const
            {
                const auto origin = _origin.lock();
                return origin && origin.get() == &consumer;
            }
            void WaitOn(const NativeStream &consumer) const
            {
                DNDS_check_throw_info(_recorded && consumer.GetDevice().ordinal == _ordinal,
                                      "CUDA wait requires a recorded prefix on the same device");
                Check(cudaStreamWaitEvent(CUDAStream(consumer), _event, 0));
            }
        };
    }

    struct CUDALauncher::State
    {
        std::shared_ptr<NativeStream> native;
        std::vector<RetainedCommand> pending;
        std::vector<StreamEvent> dependencies;
        std::deque<std::shared_ptr<CUDAPrefix>> prefixes;
        StreamEvent previous;
        std::exception_ptr error;
        std::atomic<uint64_t> events{0};
        std::mutex mutex;
        std::condition_variable changed;
        std::thread::id issuer;
        explicit State(int device) : native(CreateCUDAStream(device)) {}

        /// Serialize native issue and Record across submitting threads, but
        /// invoke kernels/providers with no queue mutex held. Recursive issue
        /// on the same lane is rejected instead of deadlocking.
        struct Issue
        {
            State &state;
            explicit Issue(State &value) : state(value)
            {
                std::unique_lock<std::mutex> lock(state.mutex);
                DNDS_check_throw_info(state.issuer != std::this_thread::get_id(), "recursive CUDA stream submission");
                state.changed.wait(lock, [&]
                                   { return state.issuer == std::thread::id{}; });
                state.issuer = std::this_thread::get_id();
            }
            ~Issue()
            {
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    state.issuer = {};
                }
                state.changed.notify_all();
            }
        };
        void Poll()
        {
            while (!prefixes.empty())
            {
                try
                {
                    if (!prefixes.front()->Ready())
                        break;
                }
                catch (...)
                {
                    if (!error)
                        error = std::current_exception();
                }
                prefixes.pop_front();
            }
        }
        StreamEvent Record()
        {
            if (pending.empty() && dependencies.empty())
                return previous;
            auto retainedDependencies = dependencies;
            if (previous.Provider())
                retainedDependencies.push_back(previous);
            auto prefix = std::make_shared<CUDAPrefix>(native);
            prefixes.push_back(prefix); // Allocate before recording/consuming commands.
            try
            {
                prefix->Record(pending, retainedDependencies, error);
            }
            catch (...)
            {
                prefixes.pop_back();
                throw;
            }
            dependencies.clear();
            ++events;
            previous = StreamEvent(prefix);
            return previous;
        }
    };

    CUDALauncher::CUDALauncher(int device) : _state(std::make_unique<State>(device)) {}
    CUDALauncher::~CUDALauncher()
    {
        try
        {
            if (!_state->native->IsRetired())
                Drain();
        }
        catch (...)
        {
        } // Explicit Drain/Synchronize reports accepted-work errors.
    }
    ExecutionContext CUDALauncher::Context() const { return ExecutionContext{_state->native->GetDevice(), _state->native}; }
    uint64_t CUDALauncher::NativeEventsRecorded() const { return _state->events; }
    Buffer CUDALauncher::Allocate(size_t bytes) { return Allocate(bytes, {}); }
    Buffer CUDALauncher::Allocate(size_t bytes, AllocationOptions options)
    {
        return AllocateImpl(bytes, options, false);
    }
    Buffer CUDALauncher::AllocateChecked(size_t bytes, AllocationOptions options)
    {
        return AllocateImpl(bytes, options, true);
    }
    Buffer CUDALauncher::AllocateImpl(size_t bytes, AllocationOptions options, bool checked)
    {
        State::Issue issue(*_state);
        _state->Poll();
        if (_state->error)
            std::rethrow_exception(_state->error);
        const auto context = Context();
        auto result = Buffer::AllocateCUDAAsync(bytes, context, options);
        std::shared_ptr<ScheduledAccess> initialization;
        try
        {
            initialization = std::make_shared<ScheduledAccess>(result.Reserve(AccessMode::Write, context.device, checked));
            _state->pending.push_back({{}, [initialization, owner = result, context](std::exception_ptr error) mutable noexcept
                                       {
                                           initialization->Complete(!error);
                                           try
                                           {
                                               owner.ReleaseOn(context);
                                           }
                                           catch (...)
                                           {
                                               std::terminate();
                                           }
                                       },
                                       false,
                                       {}});
        }
        catch (...)
        {
            if (initialization)
                initialization->Cancel();
            result.ReleaseOn(context); // Roll back unpublished ownership without a host fence.
            throw;
        }
        return result;
    }
    void CUDALauncher::Submit(Work work, CommandKind kind, Retirement retire)
    {
        DNDS_check_throw_info(work, "empty CUDA command");
        State::Issue issue(*_state);
        DeviceScope device(_state->native->GetDevice().ordinal);
        _state->Poll();
        _state->pending.push_back({std::move(work), std::move(retire), kind == CommandKind::Cleanup, _state->error});
        auto &command = _state->pending.back(); // Accepted from this point on.
        if (kind != CommandKind::Cleanup && (!_state->error || kind == CommandKind::OrderedRelease))
            try
            {
                const TimelineScope timeline("Runtime/CUDA/Launch");
                command.work(Context());
            }
            catch (...)
            {
                command.error = _state->error = std::current_exception();
            }
    }
    void CUDALauncher::WaitFor(StreamEvent event)
    {
        State::Issue issue(*_state);
        DeviceScope device(_state->native->GetDevice().ordinal);
        auto native = std::dynamic_pointer_cast<const CUDAPrefix>(event.Provider());
        if (!native)
        {
            DNDS_check_throw_info(event.Ready(), "pending non-CUDA dependency needs an explicit host wait before CUDA submission");
            return;
        }
        // The launcher's previous-prefix chain already retains every earlier
        // prefix from this same native stream, including errors and captured
        // owners. Stream order supplies the device dependency. Do not turn a
        // self-wait into another dependency-only event at the next Record().
        // ExecutionStream still imports the logical checked-access frontier.
        if (native->BelongsTo(*_state->native))
            return;
        // Repeating an identical foreign prefix before Record needs one native
        // wait and one retained dependency. Distinct prefixes are not merged:
        // a newer producer may contain work absent from the older prefix.
        for (const auto &dependency : _state->dependencies)
            if (dependency.Provider() == event.Provider())
                return;
        _state->dependencies.push_back(event); // Retain provider until the consumer prefix completes.
        try
        {
            native->WaitOn(*_state->native);
        }
        catch (...)
        {
            _state->dependencies.pop_back();
            throw;
        }
    }
    StreamEvent CUDALauncher::Record()
    {
        State::Issue issue(*_state);
        _state->Poll();
        return _state->Record();
    }
    void CUDALauncher::Drain()
    {
        State::Issue issue(*_state);
        try
        {
            _state->Record();
        }
        catch (...)
        {
            // Failure to allocate/record a boundary cannot discard captures of
            // work already issued. Drain that native stream before fallback
            // retirement; no new work is accepted while this Issue is held.
            if (!_state->error)
                _state->error = std::current_exception();
            DeviceScope device(_state->native->GetDevice().ordinal);
            try
            {
                _state->native->Synchronize();
            }
            catch (...)
            {
            }
            for (const auto &dependency : _state->dependencies)
                try
                {
                    dependency.WaitHost();
                }
                catch (...)
                {
                }
            _state->dependencies.clear();
            for (auto &command : _state->pending)
            {
                if (command.cleanup)
                    try
                    {
                        command.work(Context());
                    }
                    catch (...)
                    {
                    }
                if (command.retire)
                    command.retire(_state->error);
            }
            _state->pending.clear();
        }
        for (const auto &prefix : _state->prefixes)
            try
            {
                prefix->Wait();
            }
            catch (...)
            {
                if (!_state->error)
                    _state->error = std::current_exception();
            }
        _state->prefixes.clear();
        if (_state->error)
            std::rethrow_exception(_state->error);
    }
#else
    struct CUDALauncher::State
    {
    };
    CUDALauncher::CUDALauncher(int) { DNDS_check_throw_info(false, "runtime CUDA launcher is unavailable in this build"); }
    CUDALauncher::~CUDALauncher() = default;
    ExecutionContext CUDALauncher::Context() const { return {}; }
    uint64_t CUDALauncher::NativeEventsRecorded() const { return 0; }
    void CUDALauncher::Submit(Work, CommandKind, Retirement) { DNDS_check_throw_info(false, "runtime CUDA launcher unavailable"); }
    void CUDALauncher::WaitFor(StreamEvent) { DNDS_check_throw_info(false, "runtime CUDA launcher unavailable"); }
    StreamEvent CUDALauncher::Record()
    {
        DNDS_check_throw_info(false, "runtime CUDA launcher unavailable");
        return {};
    }
    void CUDALauncher::Drain() {}
    Buffer CUDALauncher::Allocate(size_t bytes) { return Allocate(bytes, {}); }
    Buffer CUDALauncher::Allocate(size_t, AllocationOptions)
    {
        DNDS_check_throw_info(false, "runtime CUDA allocation unavailable");
        return {};
    }
    Buffer CUDALauncher::AllocateChecked(size_t bytes, AllocationOptions options) { return Allocate(bytes, options); }
#endif
}
