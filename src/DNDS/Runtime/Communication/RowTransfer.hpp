#pragma once

#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Communication/TransferLayout.hpp"
#include "DNDS/Runtime/Storage/ElementSchema.hpp"
#include "DNDS/Runtime/Storage/Version.hpp"
#include "DNDS/MPI/MPI.hpp"
#include "DNDS/Runtime/Communication/CommunicationContext.hpp"
#include "DNDS/Runtime/Communication/TransferProvider.hpp"
#include <functional>
#include <string>
#include <cstring>
#include <cmath>
#include <typeinfo>
#include <optional>
#include <utility>

namespace DNDS::Runtime
{
    class AccessTrackingContext;
    class StreamEvent;
    struct AccessOrder;
    /// One incoming relation edge. Destination rows need not be consecutive.
    struct RowRequest
    {
        int sourceRank = 0;
        index sourceRow = 0;
        index destinationRow = 0;
    };

    /// Collective immutable row relation. Caller serializes setup/Begin in the
    /// same order on every rank, supplies correct semantic row identities and
    /// stamps affected mesh epochs. Mutation requires draining old operations;
    /// a stamp is invalidation detection, not a lock held through execution.
    class RowTransferPlan
    {
        struct State;
        std::shared_ptr<const State> _state;
        bool _reverse = false;
        RowTransferPlan() = default;

    public:
        RowTransferPlan(MPI_Comm comm, index sourceRows, index destinationRows,
                        const std::vector<RowRequest> &requests, std::vector<VersionStamp> dependencies = {});
        RowTransferPlan(std::shared_ptr<CommunicationContext> context, index sourceRows, index destinationRows,
                        const std::vector<RowRequest> &requests, std::vector<VersionStamp> dependencies = {});
        /// Peer row lists in matching per-peer order bypass owner/row discovery.
        /// Counts, bounds and collisions are checked collectively.
        /// Caller guarantees that corresponding send/receive entries mean the
        /// same row; equal counts cannot detect a semantically wrong permutation.
        static RowTransferPlan Known(std::shared_ptr<CommunicationContext> context, index sourceRows, index destinationRows,
                                     std::vector<std::vector<index>> send, std::vector<std::vector<index>> receive,
                                     std::vector<VersionStamp> dependencies = {});
        RowTransferPlan Reverse() const;
        /// Collective schedule derivation for a complete source permutation.
        /// Every source row must occur exactly once across the send lists.
        /// Returns {adjusted route, old-to-new source row permutation}; caller
        /// applies the permutation to its ordinary arrays before binding.
        /// Peer correspondence and receive slots are unchanged. No payload,
        /// row-ID/count exchange or communicator duplication is performed.
        std::pair<RowTransferPlan, std::vector<index>> ReorderSourceByPeer() const;
        MPI_Comm Communicator() const;
        /// Shared matching namespace for sequential payload bindings. Concurrent
        /// bindings must reserve distinct explicit tags or use separate contexts.
        std::shared_ptr<CommunicationContext> Context() const;
        index SourceRows() const;
        index DestinationRows() const;
        const std::vector<std::vector<index>> &SendRows() const;
        const std::vector<std::vector<index>> &ReceiveRows() const;
        bool HasCollisions() const;
        bool Valid() const;
        /// Setup exchange only: preserve local storage by default. Unrequested
        /// variable rows have length zero; table slots keep their declared width.
        RowLayout DestinationLayout(const RowLayout &source, TransferLayout layout = {}) const;
    };

    enum class Reduction
    {
        Sum,
        Min,
        Max,
        BitOr,
        BitXor,
        BitAnd
    };
    enum class DestinationInit
    {
        Preserve,
        Identity
    };
    enum class TransferTransport
    {
        Direct,
        HostPacked,
        /// Explicit CUDA pack/contiguous transport/unpack. With the default MPI
        /// provider, Begin posts receives and Test/Wait advances sends/unpack on
        /// the MPI thread. A stream-native provider instead enqueues the whole
        /// chain. Both endpoints must share a CUDA device.
        DevicePacked
    };

    struct Contribution
    {
        size_t sourceElement = 0, destinationElement = 0, count = 0;
    };

    struct Reducer;
#ifdef DNDS_USE_CUDA
    void InstallBuiltinCUDA(Reducer &, Reduction, size_t bytes, bool integral, bool isSigned, DestinationInit);
#endif

    /// Dispatch once per operation. Custom CPU/CUDA/vendor implementations use
    /// the same contribution groups; no MPI or Array ownership in the callback.
    /// Caller gives identical combine/initialization semantics to a shared name
    /// on every rank. Do not read untransmitted POD fields; do not retain pointers
    /// or issue unmatched MPI collectives in a combine. Device combines must use
    /// the supplied stream. Floating reductions are not partition-bitwise stable.
    struct Reducer
    {
        std::string identity;
        std::string schema;
        size_t elementBytes = 0;
        DestinationInit initialization = DestinationInit::Preserve;
        std::function<void(const void *, void *, const std::vector<Contribution> &, ExecutionContext)> host;
        std::function<void(const void *, void *, const Contribution *, size_t, ExecutionContext)> device;

        template <class T, class Combine>
        static Reducer Custom(const ElementSchema &element, std::string name, Combine combine, T identityValue,
                              DestinationInit init = DestinationInit::Preserve)
        {
            DNDS_check_throw_info(element.Extent() == sizeof(T) && element.Compatible(ElementSchema::TypeIdentity<T>()), "reducer element type/stride mismatch");
            Reducer result;
            result.identity = std::move(name);
            result.schema = element.Identity();
            result.elementBytes = sizeof(T);
            result.initialization = init;
            result.host = [combine, identityValue, init](const void *input, void *output,
                                                         const std::vector<Contribution> &groups, ExecutionContext)
            {
                auto *src = static_cast<const T *>(input);
                auto *dst = static_cast<T *>(output);
                // Groups are stably ordered by destination then contributor.
                size_t previous = SIZE_MAX;
                for (const auto &group : groups)
                {
                    if (init == DestinationInit::Identity && group.destinationElement != previous)
                        std::fill_n(dst + group.destinationElement, group.count, identityValue);
                    for (size_t j = 0; j < group.count; ++j)
                        combine(dst[group.destinationElement + j], src[group.sourceElement + j]);
                    previous = group.destinationElement;
                }
            };
            return result;
        }

        template <class T>
        static Reducer Builtin(Reduction op, DestinationInit init = DestinationInit::Preserve)
        {
            static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>);
            DNDS_check_throw_info(std::is_integral_v<T> || op <= Reduction::Max, "bitwise reduction requires integers");
            T identityValue{};
            if (op == Reduction::Min)
                identityValue = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
            if (op == Reduction::Max)
                identityValue = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
            if constexpr (std::is_integral_v<T>)
                if (op == Reduction::BitAnd)
                    identityValue = T(~T(0));
            auto result = Custom<T>(ElementSchema::Arithmetic<T>(), "builtin." + std::to_string(int(op)), [op](T &dst, const T &src)
                                    {
                                 switch (op)
                                 {
                                 case Reduction::Sum:
                                     if constexpr (std::is_integral_v<T>)
                                     {
                                         using U = std::make_unsigned_t<T>;
                                         U bits = U(dst) + U(src); // Defined modulo width, including signed payloads.
                                         std::memcpy(&dst, &bits, sizeof(T));
                                     }
                                     else dst += src;
                                     break;
                                 case Reduction::Min:
                                     if constexpr (std::is_floating_point_v<T>) dst = std::fmin(dst, src);
                                     else dst = std::min(dst, src);
                                     break;
                                 case Reduction::Max:
                                     if constexpr (std::is_floating_point_v<T>) dst = std::fmax(dst, src);
                                     else dst = std::max(dst, src);
                                     break;
                                 default:
                                     if constexpr (std::is_integral_v<T>)
                                     {
                                         if (op == Reduction::BitOr) dst |= src;
                                         if (op == Reduction::BitXor) dst ^= src;
                                         if (op == Reduction::BitAnd) dst &= src;
                                     }
                                 } }, identityValue, init);
#ifdef DNDS_USE_CUDA
            InstallBuiltinCUDA(result, op, sizeof(T), std::is_integral_v<T>, std::is_signed_v<T>, init);
#endif
            return result;
        }
    };

    class TransferBinding;
    class TransferOperation;
    class TransferGroup;

    /// Immutable setup description shared by payload providers. Layouts include
    /// physical strides; schemas describe transmitted fields, not POD padding.
    struct TransferPayload
    {
        RowTransferPlan route;
        RowLayout source, destination;
        ElementSchema schema;
    };

    class PayloadPlan
    {
        struct State;
        std::shared_ptr<const State> _state;
        friend class TransferBinding;
        friend class TransferOperation;
        PayloadPlan(RowTransferPlan route, RowLayout source, std::optional<RowLayout> destination, ElementSchema schema, TransferLayout layout);
        bool LayoutsMatch(const RowLayout &source, const RowLayout &destination, size_t elementBytes, bool elementCompatible) const;
        template <class T>
        bool CaptureBuffers(const Array<T> &source, const Array<T> &destination, Buffer &input, Buffer &output) const
        {
            try
            {
                // Pin the published shapes while checking and obtaining their
                // value buffers. An unpublished/failed shape on one rank must
                // reach binding's collective rejection, not strand its peers.
                const auto sourceStorage = source.SharedStorage(), destinationStorage = destination.SharedStorage();
                DNDS_check_throw_info(sourceStorage && destinationStorage, "binding requires live Array handles");
                const bool valid = LayoutsMatch(sourceStorage->Layout(), destinationStorage->Layout(), sizeof(T),
                                                Compatible(ElementSchema::TypeIdentity<T>()));
                input = sourceStorage->Values();
                output = destinationStorage->Values();
                return valid;
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("MPI payload buffer validation failed");
                return false;
            }
        }

    public:
        PayloadPlan(RowTransferPlan route, RowLayout source, RowLayout destination, ElementSchema schema);
        /// Infer and preserve destination storage from one owner-width exchange.
        /// Explicit Compact is available for setup scratch with varying widths.
        /// Rank-local fit failures reject collectively before payload binding.
        PayloadPlan(RowTransferPlan route, RowLayout source, ElementSchema schema, TransferLayout layout = {});
        RowLayout DestinationLayout() const;
        void CheckLayouts(const RowLayout &source, const RowLayout &destination, size_t elementBytes, bool elementCompatible = true) const;
        bool Compatible(const std::string &type) const;
        size_t SendSegments() const;
        size_t ReceiveSegments() const;
        TransferBinding Bind(Buffer source, Buffer destination, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
        TransferBinding Bind(Buffer source, Buffer destination, Reducer reducer, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
        template <class T>
        TransferBinding Bind(const Array<T> &source, const Array<T> &destination, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
        template <class T>
        TransferBinding Bind(const Array<T> &source, const Array<T> &destination, Reducer reducer, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
    };

    class TransferBinding
    {
        struct State;
        std::shared_ptr<State> _state;
        std::shared_ptr<Detail::TransferProviderBinding> _provider;
        friend class PayloadPlan;
        friend class TransferOperation;
        friend class TransferGroup;
        TransferBinding(std::shared_ptr<const PayloadPlan::State> plan, Buffer source, Buffer destination,
                        std::shared_ptr<const Reducer> reducer, TransferTransport transport, TransferOptions options,
                        bool layoutsValid = true);
        TransferOperation StartChecked(const ExecutionStream &stream, bool callerValid,
                                       bool observeProducer) const;
        TransferOperation StartManaged(const ExecutionStream &stream, bool checked,
                                       bool callerValid, bool observeProducer) const;
        bool HasEmptyDirectPayload() const;
        TransferOperation SubmitEmpty(EagerEngine &engine, const ExecutionStream &stream,
                                      bool callerValid) const;
        TransferOperation Start(ExecutionContext context, bool scheduled, std::shared_ptr<AccessTrackingContext> tracking = {}, uint64_t lane = 0, bool callerValid = true) const;

    public:
        /// Whether this binding supplies a native stream enqueue operation.
        /// False selects portable MPI's controller-side producer-readiness
        /// bridge. This is a provider capability, not a device-location test.
        bool SupportsStreamEnqueue() const { return bool(_provider); }
        /// Common managed launch for CPU-visible input (MPI), or input already
        /// ordered on the communication lane (native providers). Native launch
        /// retains payload/provider owners through actual stream retirement.
        TransferOperation SubmitReady(EagerEngine &engine, const ExecutionStream &communication,
                                      bool callerValid = true) const;
        /// Import the producer prefix, then submit on the communication lane.
        /// Native providers use stream events without a host wait. Portable MPI
        /// requires controller progress/readiness and is not graph-capturable.
        /// A direct MPI route with no local sends/receives and no reducer has
        /// no payload effect: it retains the usual preflights and binding owner
        /// but exports only an ordered stream prefix, without a producer host
        /// wait or payload access reservation. This is independent of rank count.
        TransferOperation SubmitAfter(EagerEngine &engine, const ExecutionStream &communication,
                                      StreamEvent producer, bool callerValid = true) const;
        /// One active Begin operation per binding. Default remains isolated persistent
        /// requests. Options select shared context/tags and transient requests.
        /// Collective: all ranks call in the same order, including empty ranks.
        /// This binding retains old buffers after Array::Reset; rebuild it after
        /// changing storage/layout or semantic row order. Finish before mutation
        /// or MPI_Finalize, on a thread allowed by the initialized MPI level.
        /// The bound communication context selects optional safety agreement.
        /// Disabled mode adds no launch handshake; local rejection aborts the
        /// communicator instead of allowing rank-asymmetric exception recovery.
        /// Local leases/readiness and required setup protocols remain enabled.
        TransferOperation Begin(ExecutionContext context = ExecutionContext::Host()) const;
        /// Caller-ordered eager reservations allow dependent scheduled kernels.
        /// CPU producers must already be complete; CUDA readiness is observed
        /// on the supplied stream before portable MPI is posted.
        /// Caller joins consumers explicitly and finishes CPU producers first;
        /// eager reservations permit conflicts but do not make races safe.
        /// DevicePacked queues pack in this context and posts receives; call
        /// Test/Wait on the MPI-permitted thread to advance sends/unpack. Its
        /// CUDA inputs may have prior scheduled producers on this same stream.
        /// callerValid joins the existing MPI preflight when enabled; false
        /// aborts with preflight disabled. Native providers require
        /// caller-coordinated validation.
        TransferOperation BeginScheduled(ExecutionContext context = ExecutionContext::Host(), bool callerValid = true) const;
        /// Managed counterpart: register requests with the stream's runtime
        /// before posting. Dropping the returned handle does not lose progress;
        /// engine Synchronize/Close retires it on the controller. The raw-context
        /// overload above remains explicitly caller-managed interop.
        TransferOperation BeginScheduled(const ExecutionStream &stream, bool callerValid = true) const;
        /// MPI bridge for a checked stream. Captures its producer frontier but
        /// gives MPI a distinct lane: later stream commands do not implicitly
        /// follow outstanding requests. CPU producers are observed on the host;
        /// CUDA Direct observes this prefix, DevicePacked queues packing on it.
        /// Full Wait/Test completion makes checked consumers/reuse eligible.
        /// All ranks call this collectively, including rank-local check errors.
        /// Registered with the runtime before posting; Close owns retirement.
        /// Stream-native providers use Enqueue instead.
        /// callerValid joins the checked-stream preflight when enabled; false
        /// aborts with preflight disabled. Scheduled access checking is separate.
        TransferOperation BeginChecked(const ExecutionStream &stream, bool callerValid = true) const;
        /// Managed begin for payloads already visible to the CPU controller.
        /// This communication-lane entry does not record or host-wait the lane
        /// prefix. Checked mode still reserves all transfer accesses and rejects
        /// any unordered outstanding producer; unchecked mode leaves that proof
        /// with the caller. Use the ordinary methods when this lane produces the
        /// payload before the exchange.
        TransferOperation BeginReadyScheduled(const ExecutionStream &stream,
                                              bool callerValid = true) const;
        TransferOperation BeginReadyChecked(const ExecutionStream &stream,
                                            bool callerValid = true) const;
        /// Stream-native providers enqueue pack/transfer/unpack without host
        /// completion waits or per-exchange MPI preflight. Caller supplies input
        /// readiness and joins consumers using stream events. Private scratch
        /// reuse and shared-provider peer order must be ordered too. Ordinary
        /// MPI rejects this API; use its Begin/Test/Wait host-progress bridge.
        void Enqueue(EagerEngine &, const ExecutionStream &) const;
    };

    class TransferOperation
    {
        struct ManagedState;
        std::shared_ptr<ManagedState> _managed;
        std::shared_ptr<StreamEvent> _nativeCompletion;
        StreamEvent ManagedCompletion() const;
        struct WorkerState;
        std::unique_ptr<WorkerState> _worker;
        struct State;
        std::unique_ptr<State> _state;
        std::unique_ptr<Detail::TransferProviderOperation> _provider;
        friend class TransferBinding;
        friend class MPIProgress;
        friend class TransferGroup;
        std::shared_ptr<const AccessOrder> CheckedOrder() const;
        explicit TransferOperation(std::unique_ptr<State> state);
        explicit TransferOperation(std::unique_ptr<Detail::TransferProviderOperation> provider);
        bool PostPackedSends(bool wait);
        bool CompleteDestination(bool wait);

    public:
        TransferOperation();
        TransferOperation(const TransferOperation &) = delete;
        TransferOperation &operator=(const TransferOperation &) = delete;
        TransferOperation(TransferOperation &&) noexcept;
        TransferOperation &operator=(TransferOperation &&) noexcept;
        ~TransferOperation();
        /// Failed completion remains failed across repeated observation and
        /// moves (including MPIProgress import). Destruction only drains once.
        /// Caller explicitly observes errors. WaitDestinationReady releases only
        /// receives; source send ownership lasts until Wait/Test completion.
        /// Raw failure is rank-local; coordinate solver recovery across ranks.
        void WaitDestinationReady();
        void Wait();
        bool Test();
        /// Full completion of a managed MPI or native stream submission. The creating runtime
        /// retains and progresses the transfer after this handle is released.
        /// Raw-context and unmanaged provider operations reject this query.
        /// Native events can be imported by matching device streams directly;
        /// use EnqueueWait for a portable MPI -> device consumer bridge.
        StreamEvent CompletionEvent() const;
        /// Insert consumer ordering using this transport's completion contract.
        /// Native transfers import a native event. Portable MPI -> CUDA must
        /// progress/wait on the controller before device submission: this is an
        /// explicit non-capturable bridge, never a fabricated CUDA event.
        /// CPU consumers remain queued and do not block a worker on MPI.
        void EnqueueWait(const ExecutionStream &consumer) const;
        /// Portable MPI may need controller work and cannot be graph captured.
        bool RequiresHostProgress() const { return !_nativeCompletion; }
    };

    /// Opt-in prepared CPU MPI bundle; existing separate starts remain default.
    /// Members retain their allocations, datatypes, tags and separate messages.
    /// Construction is collective on context: every rank supplies the same
    /// ordered tags, including empty participants. Members must be persistent
    /// host Direct/HostPacked bindings using that exact context. Other providers,
    /// devices and scheduled/checked-stream launches are explicitly unsupported.
    ///
    /// Begin acquires every local member before its one optional safety agreement;
    /// rejection cancels all unmodified reservations before any request is posted.
    /// Serialize group/member/controller access, finish old operations and release
    /// producer/consumer views first. Returned operations retain the normal
    /// destination-ready versus full-send-completion distinction. Drain all of
    /// them before reusing the group, changing buffers or MPI_Finalize.
    class TransferGroup
    {
        std::shared_ptr<CommunicationContext> _context;
        std::vector<TransferBinding> _bindings;

    public:
        TransferGroup(std::shared_ptr<CommunicationContext> context, std::vector<TransferBinding> bindings);
        std::vector<TransferOperation> Begin() const;
    };

    template <class T>
    TransferBinding PayloadPlan::Bind(const Array<T> &source, const Array<T> &destination, TransferTransport transport, TransferOptions options) const
    {
        Buffer input, output;
        const bool valid = CaptureBuffers(source, destination, input, output);
        return TransferBinding(_state, std::move(input), std::move(output), {}, transport, std::move(options), valid);
    }
    template <class T>
    TransferBinding PayloadPlan::Bind(const Array<T> &source, const Array<T> &destination, Reducer reducer, TransferTransport transport, TransferOptions options) const
    {
        Buffer input, output;
        const bool valid = CaptureBuffers(source, destination, input, output);
        return TransferBinding(_state, std::move(input), std::move(output), std::make_shared<const Reducer>(std::move(reducer)),
                               transport, std::move(options), valid);
    }
}
