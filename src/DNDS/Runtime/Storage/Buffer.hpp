#pragma once

#include "DNDS/Core/Errors.hpp"
#include "DNDS/Core/Defines.hpp"
#include "DNDS/Core/CheckedSize.hpp"
#include "DNDS/Runtime/Execution/Device.hpp"
#include "DNDS/Runtime/Execution/StreamResource.hpp"
#include "DNDS/Runtime/Storage/AllocationDiagnostics.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

namespace DNDS::Runtime
{
    /// Placement and shared abstract stream resource, without submission rights.
    /// A null resource denotes immediate host work or the CUDA default stream.
    /// External kernels use this lane or join their other lanes before completion.
    struct ExecutionContext
    {
        Device device;
        std::shared_ptr<StreamResource> stream;
        static ExecutionContext Host() { return {}; }
        static ExecutionContext CUDA(int ordinal, std::shared_ptr<StreamResource> stream = {}) { return {Device::CUDA(ordinal), std::move(stream)}; }
        void Validate() const
        {
            DNDS_check_throw_info(device.IsValid() && (!stream || stream->GetDevice() == device), "execution context stream/placement mismatch");
        }
        /// Observe this lane, or do nothing after a completed engine close.
        void Synchronize() const;
    };

    /// Base-address alignment is independent of table pitch. Zero requests the
    /// provider default; a nonzero request must be a power of two. A provider
    /// either honors the request or throws before publishing the allocation.
    struct AllocationOptions
    {
        size_t alignmentBytes = 0;
    };

    class Completion
    {
        struct State;
        std::shared_ptr<State> _state;
        explicit Completion(std::shared_ptr<State> state) : _state(std::move(state)) {}

    public:
        Completion() = default;
        static Completion Record(ExecutionContext context);
        /// Backend completion proof (e.g. successful CUDA-aware MPI request).
        /// No event is recorded and no stream is fenced. Caller guarantees all
        /// work covered by this access has completed on the declared context.
        static Completion FromExternalCompletion(ExecutionContext context);
        void Wait() const;
        bool Ready() const;
        Device GetDevice() const;
        bool Matches(ExecutionContext context) const;
    };

    namespace Detail
    {
        /// Backend primitive for operation implementations holding all required
        /// leases. It does not acquire/release access or publish generations.
        Completion CopyBytes(const void *source, Device sourceDevice, void *destination, Device destinationDevice,
                             size_t bytes, ExecutionContext context);
        /// Same primitive without an event. The enclosing launcher owns actual
        /// completion and both operands; callers must not drop DMA owners early.
        void EnqueueCopyBytes(const void *source, Device sourceDevice, void *destination, Device destinationDevice,
                              size_t bytes, ExecutionContext context);
    }

    template <class T>
    struct Span
    {
        T *data = nullptr;
        size_t size = 0;
        MemorySpace space = MemorySpace::Host;

        DNDS_DEVICE_CALLABLE T &operator[](size_t i) const
        {
#ifdef __CUDA_ARCH__
            DNDS_HD_assert(space == MemorySpace::CUDA && i < size);
#else
            DNDS_check_throw_info(space == MemorySpace::Host && i < size, "runtime span: invalid host access or index");
#endif
            return data[i];
        }
    };

    struct BufferState;
    class Buffer;
    class CopyOperation;
    class AllocationResource;
    enum class AccessMode
    {
        Read,
        Write,
        ReadWrite
    };

    /// A declared eager access, not an exclusive execution lock. The caller
    /// orders pending accesses through streams. Immediate accesses still check
    /// these reservations. The launcher retires them at actual completion.
    class ScheduledAccess
    {
        friend class Buffer;
        std::shared_ptr<BufferState> _state;
        std::weak_ptr<BufferState> _issued;
        size_t _offset = 0, _bytes = 0;
        bool _write = false;
        bool _checked = false;
        ScheduledAccess(std::shared_ptr<BufferState>, size_t, size_t, Device, AccessMode, bool);

    public:
        ScheduledAccess() = default;
        ScheduledAccess(const ScheduledAccess &) = delete;
        ScheduledAccess &operator=(const ScheduledAccess &) = delete;
        ScheduledAccess(ScheduledAccess &&) noexcept = default;
        ScheduledAccess &operator=(ScheduledAccess &&) noexcept;
        ~ScheduledAccess(); // Unaccepted reservation: cancel without publication.
        void Validate() const;
        void *Pointer(size_t alignment, size_t elementBytes) const;
        /// Native provider handoff after successful GPU submission. Publication
        /// retains a weak controller; ordered allocation release replaces the
        /// strong execution pin. Foreign/DMA owners must stay retained instead.
        void Handoff(ExecutionContext context);
        void Complete(bool success) noexcept;
        void Cancel() noexcept;
    };

    /// Shared allocation and access domain. Import once, then create Buffer
    /// ranges from copied handles so overlapping views share exclusion/versioning.
    /// The provider guarantees capacity, placement and readiness before import;
    /// owner may be an aliasing shared_ptr retaining a tensor or foreign allocator.
    class Allocation
    {
        friend class Buffer;
        std::shared_ptr<BufferState> _state;
        explicit Allocation(std::shared_ptr<BufferState> state) : _state(std::move(state)) {}

    public:
        Allocation() = default;
        /// @pre owner retains the allocation (aliasing shared_ptr is supported),
        /// whose address, capacity, placement and readiness match this descriptor.
        /// Import each allocation once and derive Buffer ranges from that handle;
        /// importing aliases separately creates independent access controllers.
        /// External writers must obey the same leases and stream dependencies.
        /// Mark pool/VMM memory explicitly; foreign MPI capability checks cannot
        /// infer allocator semantics from a shared_ptr owner.
        /// Nonempty CUDA imports verify memory type/device ordinal once. This
        /// does not discover capacity or replace the producer-readiness contract.
        static Allocation Import(std::shared_ptr<void> owner, size_t bytes,
                                 Device device = Device::Host(), bool writable = true, bool streamOrderedMemory = false);
        size_t Bytes() const;
        Device GetDevice() const;
    };

    /// Optional allocator policy for pools/pinned memory. Returned ownership
    /// must free on the named device and remain valid until its last lease ends.
    /// Implementations return zero-initialized storage before publication.
    class Allocator
    {
    public:
        virtual ~Allocator() = default;
        virtual bool StreamOrderedMemory(Device) const noexcept { return false; }
        virtual std::shared_ptr<void> Allocate(size_t bytes, Device device) = 0;
        /// Compatibility default validates the returned address. Providers with
        /// stronger alignment support may override without changing old clients.
        virtual std::shared_ptr<void> Allocate(size_t bytes, Device device, AllocationOptions options)
        {
            auto owner = Allocate(bytes, device);
            const auto alignment = options.alignmentBytes;
            DNDS_check_throw_info(!alignment || (!(alignment & (alignment - 1)) &&
                                                 reinterpret_cast<uintptr_t>(owner.get()) % alignment == 0),
                                  "allocator cannot satisfy requested alignment");
            return owner;
        }
    };

    /// Move-only access owns its allocation until completion. Complete records
    /// a launch event; Wait releases access. Destruction waits as a safe fallback.
    /// A CUDA lease without Complete fences its stream before releasing storage.
    class Access
    {
        friend class Buffer;
        friend CopyOperation Copy(const Buffer &, Buffer &, ExecutionContext);
        std::shared_ptr<BufferState> _state;
        ExecutionContext _context;
        Completion _completion;
        bool _hasCompletion = false;
        bool _write = false;
        size_t _offset = 0, _bytes = 0;
        int _uncaughtOnAcquire = 0;
        Access(std::shared_ptr<BufferState> state, size_t offset, size_t bytes, ExecutionContext context, bool write);

    protected:
        void *Pointer(size_t alignment, size_t elementSize) const;

    public:
        Access() = default;
        Access(const Access &) = delete;
        Access &operator=(const Access &) = delete;
        Access(Access &&other) noexcept;
        Access &operator=(Access &&other) noexcept;
        ~Access();
        void Complete(Completion completion);
        void Wait();
        /// Discard an unissued access after collective preflight failed. The
        /// caller guarantees no read/write work has used this lease.
        void CancelUnmodified();
        /// Drain submitted work, then make a potentially partial write unreadable.
        /// Recovery requires replacing the allocation, not accepting stale values.
        void Fail();
        bool Active() const { return bool(_state); }
        size_t Bytes() const;
        Device GetDevice() const;
    };

    class ReadAccess : public Access
    {
        friend class Buffer;
        explicit ReadAccess(Access access) : Access(std::move(access)) {}

    public:
        ReadAccess() = default;
        template <class T>
        Span<const T> View() const
        {
            static_assert(std::is_trivially_copyable_v<T>);
            auto *p = static_cast<const T *>(Pointer(alignof(T), sizeof(T)));
            return {p, Bytes() / sizeof(T), GetDevice().space};
        }
    };

    class WriteAccess : public Access
    {
        friend class Buffer;
        explicit WriteAccess(Access access) : Access(std::move(access)) {}

    public:
        WriteAccess() = default;
        template <class T>
        Span<T> View() const
        {
            static_assert(std::is_trivially_copyable_v<T>);
            auto *p = static_cast<T *>(Pointer(alignof(T), sizeof(T)));
            return {p, Bytes() / sizeof(T), GetDevice().space};
        }
    };

    /// A shallow, shared handle to exactly one allocation and access controller.
    /// Allocate creates independent storage; Copy creates independent values.
    /// Layout, mesh association and distributed routing belong in other objects.
    class Buffer
    {
        friend class CUDALauncher;
        std::shared_ptr<BufferState> _state;
        size_t _offset = 0, _bytes = 0;
        uintptr_t Address() const;
        explicit Buffer(std::shared_ptr<BufferState> state);
        static Buffer AllocateCUDAAsync(size_t bytes, ExecutionContext context, AllocationOptions options = {});

    public:
        Buffer() = default;
        static Buffer Allocate(size_t bytes, Device device = Device::Host(), AllocationOptions options = {});
        /// Zero-initialized portable CUDA page-locked host storage. Views still
        /// carry Device::Host(); allocation accounting records the pinned kind.
        /// CopyOperation retains this allocation and its leases until DMA is
        /// complete. No pageable fallback: rejects when CUDA is not compiled.
        static Buffer AllocatePinnedHost(size_t bytes);
        static Buffer Allocate(size_t bytes, Device device, std::shared_ptr<Allocator> allocator, AllocationOptions options = {});
        /// Ready allocation import for backend providers. Include
        /// AllocationResource.hpp when implementing/importing a provider.
        static Buffer FromResource(std::unique_ptr<AllocationResource> resource);
        static Buffer FromAllocation(const Allocation &allocation, size_t offset, size_t bytes);
        Allocation AllocationHandle() const { return Allocation(_state); }
        Buffer Subspan(size_t offset, size_t bytes) const;
        size_t Bytes() const;
        Device GetDevice() const;
        /// Address/placement metadata only: no access lease, readiness wait or
        /// payload dereference. Detect physical overlap even for independently
        /// imported aliases; such aliases still have separate access controllers.
        bool Overlaps(const Buffer &other) const;
        bool SameAddress(const Buffer &other) const;
        bool SameRange(const Buffer &other) const;
        /// Zero alignment is invalid (false); a null/empty address is aligned.
        /// This validates address alignment, not capacity or producer completion.
        bool IsAligned(size_t alignment) const;
        uint64_t Identity() const;
        /// Add an allocation alias label without retaining storage in diagnostics.
        void SetDiagnosticName(const std::string &name, AllocationSite site = {}) const;
        /// Weak identity pin for host tracking registries; never retains payload
        /// storage or exposes its pointer. Buffer ranges share this identity.
        std::weak_ptr<void> AccessLifetime() const { return _state; }
        std::vector<Buffer> ReleaseBuffers() const { return {*this}; }
        uint64_t Version() const;
        /// Monotone mutation-intent epoch, advanced when a write lease or
        /// scheduled write reservation is issued, not when it completes.
        /// Cache provenance may use this across queued producers; it is NOT
        /// a readiness/completion test or permission to read a pointer. Canceled
        /// writes conservatively invalidate old epochs, preventing ABA reuse.
        uint64_t MutationEpoch() const;
        bool IsIdle() const;
        bool IsUnique() const;
        bool IsValid() const;
        bool StreamOrderedMemory() const;
        bool CanReleaseOrdered(ExecutionContext context) const;
        /// Consume this handle, preserving aliases. Final resource release
        /// receives this explicit context; ordinary destruction is conservative.
        void ReleaseOn(ExecutionContext context);
        ReadAccess Read(ExecutionContext context = ExecutionContext::Host()) const;
        WriteAccess Write(ExecutionContext context = ExecutionContext::Host());
        /// Checked and unchecked reservations cannot have simultaneous conflicting
        /// uses. Drain/import at an execution-policy boundary; read/read is safe.
        ScheduledAccess Reserve(AccessMode mode, Device device, bool checked = false) const;
        /// Non-mutating preflight for a later scheduled reservation. Rejects
        /// conflicting immediate leases, placement and access-policy mismatches;
        /// permits pending uses from the same scheduling policy. No wait, access
        /// acquisition, version/epoch change or stream-order proof occurs here.
        /// Compound collective operators may call this before posting work.
        /// The real Reserve still validates atomically; callers must serialize
        /// controller submissions and provide the required stream/event edges.
        void ValidateReservation(AccessMode mode, Device device, bool checked = false) const;

        /// Compatibility boundary only. Raw pointers cannot enforce future
        /// writes, synchronization or lifetimes. Native kernels use access leases.
        void *LegacyPointer() const;
    };

    class CopyOperation
    {
        friend CopyOperation Copy(const Buffer &, Buffer &, ExecutionContext);
        ReadAccess _source;
        WriteAccess _destination;

    public:
        CopyOperation() = default;
        CopyOperation(const CopyOperation &) = delete;
        CopyOperation &operator=(const CopyOperation &) = delete;
        CopyOperation(CopyOperation &&) noexcept = default;
        CopyOperation &operator=(CopyOperation &&) noexcept = default;
        void Wait();
    };

    /// Explicit equal-size copy. CUDA copies require a context matching every
    /// CUDA allocation involved; cross-device transfers are rejected for now.
    CopyOperation Copy(const Buffer &source, Buffer &destination,
                       ExecutionContext context = ExecutionContext::Host());
}
