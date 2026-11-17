#pragma once

#include "DNDS/Runtime/Storage/Buffer.hpp"
#include "DNDS/Runtime/Execution/AccessTracking.hpp"
#include <functional>
#include <vector>
#include <optional>
#include <mutex>

namespace DNDS::Runtime
{
    struct ExecutionRuntime;
    class EagerEngine;
    /// Immediate, unsubmitted use of a runtime's native context. Keep this
    /// guard outside the raw kernel view and destroy it after its access scope.
    /// Close rejects live guards; queued operations use retained commands instead.
    class ExecutionBorrow
    {
        std::shared_ptr<ExecutionRuntime> _runtime;
        ExecutionContext _context;
        friend class ExecutionStream;
        ExecutionBorrow(std::shared_ptr<ExecutionRuntime>, ExecutionContext);

    public:
        ExecutionBorrow(const ExecutionBorrow &) = delete;
        ExecutionBorrow &operator=(const ExecutionBorrow &) = delete;
        ExecutionBorrow(ExecutionBorrow &&) noexcept;
        ExecutionBorrow &operator=(ExecutionBorrow &&) noexcept;
        ~ExecutionBorrow();
        ExecutionContext Context() const;
    };
    /// Controller-progressed accepted work. Register before initiating external
    /// activity. Poll must not block: other streams/operations may enable it.
    /// Return true only after access publication and request retirement. A
    /// throw denotes a fully retired failure, never an outstanding request.
    /// Retain actual backend/operand owners here, but never the owning engine.
    class ExecutionProgress
    {
    public:
        virtual ~ExecutionProgress() = default;
        virtual bool Poll() = 0;
    };

    enum class CommandKind
    {
        Compute,
        Cleanup,
        OrderedRelease
    };
    namespace Detail
    {
        template <class T, class = void>
        struct HasOrderedRelease : std::false_type
        {
        };
        template <class T>
        struct HasOrderedRelease<T, std::void_t<decltype(std::declval<const T &>().CanReleaseOrdered(ExecutionContext{})),
                                                decltype(std::declval<T &>().ReleaseOn(ExecutionContext{}))>> : std::true_type
        {
        };
        template <class T, class = void>
        struct HasReleaseBuffers : std::false_type
        {
        };
        template <class T>
        struct HasReleaseBuffers<T, std::void_t<decltype(std::declval<const T &>().ReleaseBuffers())>> : std::true_type
        {
        };
    }
    /// Backend-owned completion for an explicit stream prefix. Ordinary kernel
    /// submission does not create or return one of these objects.
    class StreamEvent
    {
    public:
        class State
        {
            mutable std::mutex _notificationMutex;
            mutable bool _completed = false;
            mutable std::vector<std::function<void()>> _notifications;

        public:
            virtual ~State() = default;
            /// Nonblocking observation. Throws report terminal, retired errors;
            /// a provider must not throw merely because work is still pending.
            virtual bool Ready() const = 0;
            virtual void Wait() const = 0;
            /// Opt in to a blocking host wait after controller-managed external
            /// progress is fully retired. Zero preserves nonblocking polling for
            /// custom providers. A positive priority promises that Wait advances
            /// its own dependencies without another engine progress pass.
            /// Larger priorities are observed first: native submitted work uses
            /// 2, self-progressing CPU executors use 1, so CPU dependency scans
            /// need not repeatedly query a native producer. This is an ordering
            /// preference, not a backend identity or a dependency replacement.
            virtual unsigned AutonomousWaitPriority() const noexcept { return 0; }
            /// Backend completion notification, including terminal failure.
            /// Publish result/error and release completed accesses first. This
            /// wakes consumers; it never runs their numerical work inline.
            void NotifyCompletion() const;
            /// Subscribe without racing completion. The callback must not throw
            /// or perform backend work; it may run immediately on this thread.
            /// Capture weak executor handles to avoid event/executor cycles.
            void OnCompletion(std::function<void()> notify) const;
        };

    private:
        std::shared_ptr<const State> _state;
        std::shared_ptr<const AccessOrder> _order;
        friend class ExecutionStream;
        friend class TransferBinding;
        friend class TransferOperation;
        friend class MPIProgress;

    public:
        StreamEvent() = default;
        explicit StreamEvent(std::shared_ptr<const State> state) : _state(std::move(state)) {}
        bool Ready() const;
        void WaitHost() const;
        void OnCompletion(std::function<void()> notify) const;
        const std::shared_ptr<const State> &Provider() const { return _state; }
    };

    /// One ordered lane. Launchers own queueing, dependencies and completion;
    /// the engine uses this contract for inline/worker CPU and native devices.
    class StreamLauncher
    {
    public:
        using Work = std::function<void(ExecutionContext)>;
        using Retirement = std::function<void(std::exception_ptr)>;
        virtual ~StreamLauncher() = default;
        virtual ExecutionContext Context() const = 0;
        /// Rejection throws before accepting ownership; an accepted command's
        /// execution errors are reported by its completion/drain path instead.
        /// Retirement runs once after actual completion (or accepted work is
        /// skipped after failure), before the stream prefix becomes complete.
        /// It must not throw. Rejected work is not retired by the launcher.
        virtual void Submit(Work work, CommandKind kind, Retirement retire = {}) = 0;
        virtual void WaitFor(StreamEvent event) = 0;
        virtual StreamEvent Record() = 0;
        /// Retire all accepted work before returning or reporting a terminal
        /// error. A backend which cannot safely retire work must abort rather
        /// than throw a recoverable, still-active shutdown to its caller.
        virtual void Drain() = 0;
        virtual Buffer Allocate(size_t)
        {
            DNDS_check_throw_info(false, "stream provider has no allocation capability");
            return {};
        }
        /// Provider extension for requested alignment. Existing custom launchers
        /// keep their allocation hook; this fallback fails if it cannot honor
        /// the requested address alignment. Native providers override it.
        virtual Buffer Allocate(size_t bytes, AllocationOptions options)
        {
            DNDS_check_throw_info(!options.alignmentBytes || !(options.alignmentBytes & (options.alignmentBytes - 1)),
                                  "allocation alignment must be a power of two");
            auto result = Allocate(bytes);
            DNDS_check_throw_info(!options.alignmentBytes ||
                                      result.IsAligned(options.alignmentBytes),
                                  "stream allocator cannot satisfy requested alignment");
            return result;
        }
        /// Native asynchronous initialization must use checked reservations when
        /// called by a checked engine. Ready CPU providers use the default.
        virtual Buffer AllocateChecked(size_t bytes, AllocationOptions options) { return Allocate(bytes, options); }
    };

    class ExecutionStream
    {
        friend class EagerEngine;
        friend class TransferBinding;
        std::weak_ptr<StreamLauncher> _launcher;
        std::weak_ptr<ExecutionRuntime> _runtime;
        std::shared_ptr<StreamLauncher> LockLauncher() const;
        std::shared_ptr<AccessTrackingContext> _tracking;
        uint64_t _lane = 0;
        std::shared_ptr<void> _laneLifetime;
        ExecutionStream(std::shared_ptr<StreamLauncher> launcher, const std::shared_ptr<ExecutionRuntime> &runtime,
                        std::shared_ptr<AccessTrackingContext> tracking = {}, uint64_t lane = 0)
            : _launcher(std::move(launcher)), _runtime(runtime), _tracking(std::move(tracking)), _lane(lane)
        {
            if (_tracking)
                _laneLifetime = std::shared_ptr<void>(_tracking.get(), [tracking = _tracking, lane](void *)
                                                      { tracking->RetireLane(lane); });
        }

    public:
        ExecutionStream() = default;
        ExecutionContext Context() const;
        /// Compare live launcher identity without exposing a native pointer.
        /// This does not establish readiness or retain either runtime.
        bool SameLane(const ExecutionStream &other) const;
        ExecutionBorrow BorrowContext() const;
        /// Lock the actual runtime for one synchronous operator call. The
        /// returned facade shares queues but does not own shutdown. Closed or
        /// expired contexts reject; no borrowed EagerEngine address is retained.
        EagerEngine LockEngine() const;
        /// Caller supplies every cross-stream dependency; waiting is not inferred
        /// from Array aliases. CUDA rejects an unfinished portable MPI/CPU event.
        void WaitFor(StreamEvent event) const;
        /// Capture only the already-submitted prefix. Later work is not covered;
        /// caller must record again before exporting that work's readiness.
        StreamEvent Record() const;
        /// Backend integration: register controller work before posting/packing.
        /// The runtime retains it even if all public operation handles vanish.
        void RegisterProgress(std::shared_ptr<ExecutionProgress> operation) const;
    };

    /// Fully eager execution. The caller supplies dependency/release ordering.
    /// HostThreads=0 runs ready CPU work inline; >0 shares a bounded worker pool
    /// between all CPU lanes. Waiting lanes never consume blocked workers.
    /// CPU prefix completion directly notifies dependent executors. Foreign
    /// backends without push notification are progressed by the controller at
    /// submission/wait/drain boundaries; workers never periodically probe them.
    /// @pre Caller orders all conflicting scheduled accesses and joins every
    /// independent use before a final release. An optional shared tracking
    /// context checks declared accesses at submission. Unchecked eager mode
    /// leaves this proof with the caller; omitted edges can corrupt values.
    /// @pre CPU work must not block waiting for work that needs the same bounded
    /// pool; express dependencies with WaitFor. Synchronize explicitly to observe
    /// failures. Concurrent mutation of one public Array handle is not supported.
    /// See docs/architecture/RuntimeCallerContract.md for the complete contract.
    class EagerEngine
    {
        std::shared_ptr<ExecutionRuntime> _state;
        bool _shutdownOwner = true;
        friend class ExecutionStream;
        explicit EagerEngine(std::shared_ptr<ExecutionRuntime> state);
        void Drain(bool closing);

    public:
        explicit EagerEngine(size_t hostThreads = 0, std::shared_ptr<AccessTrackingContext> tracking = {});
        EagerEngine(const EagerEngine &) = delete;
        EagerEngine &operator=(const EagerEngine &) = delete;
        ~EagerEngine();
        ExecutionStream CreateCPUStream();
        ExecutionStream CreateStream(std::shared_ptr<StreamLauncher> launcher);
        bool ChecksAccess() const;
        bool Owns(const ExecutionStream &stream) const { return stream._runtime.lock() == _state; }
        /// CPU allocation is ready immediately; CUDA allocation/zeroing follows
        /// this stream. Record/Wait exports readiness when another stream needs it.
        Buffer AllocateBuffer(const ExecutionStream &stream, size_t bytes, AllocationOptions options = {});
        void Submit(const ExecutionStream &stream, StreamLauncher::Work work);
        /// Backend/operator integration; no per-kernel public completion token.
        void SubmitOperation(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire,
                             std::vector<ScheduledBufferAccess> accesses = {});
        /// Ordered cleanup still runs after a preceding computation failed.
        void SubmitCleanup(const ExecutionStream &stream, StreamLauncher::Work work, std::vector<ScheduledBufferAccess> accesses = {});
        void SubmitOrderedRelease(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire,
                                  std::vector<ScheduledBufferAccess> accesses = {});
        /// Consume one handle, not all aliases. Caller must join every independent
        /// use onto stream before any potentially final release. Different alias
        /// release streams are legal only when the actual final decrement follows
        /// all uses; reference counting alone does not establish that ordering.
        template <class T>
        void ReleaseOn(T &&handle, const ExecutionStream &stream)
        {
            static_assert(!std::is_lvalue_reference_v<T>, "ReleaseOn consumes a moved handle");
            static_assert(std::is_nothrow_move_assignable_v<std::decay_t<T>>, "release handles need noexcept move assignment");
            stream.LockLauncher(); // Reject a dead/closed context before consuming the handle.
            std::vector<ScheduledBufferAccess> accesses;
            if (ChecksAccess())
            {
                if constexpr (Detail::HasReleaseBuffers<std::decay_t<T>>::value)
                {
                    for (const auto &buffer : handle.ReleaseBuffers())
                        if (buffer.Identity())
                            accesses.push_back({buffer, AccessMode::Write});
                }
                else
                    DNDS_check_throw_info(false, "checked release needs explicit ReleaseBuffers allocation declarations");
            }
            bool ordered = false;
            if constexpr (Detail::HasOrderedRelease<std::decay_t<T>>::value)
                ordered = handle.CanReleaseOrdered(stream.Context());
            auto retained = std::make_shared<std::optional<std::decay_t<T>>>(std::in_place, std::move(handle));
            try
            {
                if constexpr (Detail::HasOrderedRelease<std::decay_t<T>>::value)
                    if (ordered)
                    {
                        SubmitOrderedRelease(stream, [retained](ExecutionContext context)
                                             { retained->value().ReleaseOn(context); retained->reset(); }, [retained](std::exception_ptr) noexcept
                                             { retained->reset(); }, std::move(accesses));
                        return;
                    }
                SubmitCleanup(stream, [retained](ExecutionContext)
                              { retained->reset(); }, std::move(accesses));
            }
            catch (...)
            {
                handle = std::move(retained->value()); // Rejected: return ownership.
                throw;
            }
        }
        /// Drain all streams even if one fails, then rethrow the first failure.
        void Synchronize();
        /// Idempotent controller-only shutdown. Reject further submissions,
        /// progress all registered work and streams together, then join workers.
        /// Serialize submission with Close/Synchronize; no peer work is invented.
        void Close();
        bool IsClosed() const;
        /// Register before initiating external work. No backend callbacks run
        /// under the runtime registry lock. Only the creating thread progresses
        /// and retires these owners. Completion is not a dependency edge.
        void RegisterProgress(std::shared_ptr<ExecutionProgress> operation);
        /// Pin a host resource through the next Synchronize/Close, then release
        /// this copy on the creating controller after all accepted work retires.
        /// Use before submitting work that may otherwise drop the last mesh or
        /// MPI resource owner on a CPU worker/device retirement path. No event
        /// is recorded and no work is submitted. This is not an Array access
        /// lease or a stream dependency; numerical operands still use tracked
        /// submissions. Do not retain the engine itself (that would form a cycle).
        void RetainUntilSynchronize(std::shared_ptr<const void> owner);
    };
}
