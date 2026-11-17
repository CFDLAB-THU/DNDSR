#include "DNDS/Runtime/Storage/Buffer.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Runtime/Storage/AllocationResource.hpp"
#include <atomic>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <new>
#include <utility>
#ifdef DNDS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif

namespace DNDS::Runtime
{
    namespace
    {
        std::atomic<uint64_t> nextIdentity{1};

        void ValidateDevice(Device device)
        {
            DNDS_check_throw_info((device.space == MemorySpace::Host && device.ordinal == -1) ||
                                      (device.space == MemorySpace::CUDA && device.ordinal >= 0),
                                  "runtime: invalid device descriptor");
#ifndef DNDS_USE_CUDA
            DNDS_check_throw_info(device.space == MemorySpace::Host, "runtime: CUDA is not enabled in this build");
#endif
        }

#ifdef DNDS_USE_CUDA
        void CheckCUDA(cudaError_t status)
        {
            DNDS_check_throw_info(status == cudaSuccess, cudaGetErrorString(status));
        }

        class DeviceScope
        {
            int _previous = -1;

        public:
            explicit DeviceScope(int ordinal)
            {
                CheckCUDA(cudaGetDevice(&_previous));
                CheckCUDA(cudaSetDevice(ordinal));
            }
            ~DeviceScope()
            {
                if (cudaSetDevice(_previous) != cudaSuccess)
                    std::terminate();
            }
        };

        void ValidateStream(ExecutionContext context)
        {
            DeviceScope scope(context.device.ordinal);
#    if CUDART_VERSION >= 13000
            int ordinal = -1;
            CheckCUDA(cudaStreamGetDevice(CUDAStream(context), &ordinal));
            DNDS_check_throw_info(ordinal == context.device.ordinal, "runtime: stream belongs to another CUDA device");
#    else
            // Older CUDA runtimes lack this query. The caller must supply a
            // stream from the declared device; event recording validates usage.
            unsigned flags = 0;
            CheckCUDA(cudaStreamGetFlags(CUDAStream(context), &flags));
#    endif
        }
#endif

        class PinnedHostResource final : public AllocationResource
        {
            void *_pointer = nullptr;
            size_t _bytes = 0;

        public:
            explicit PinnedHostResource(size_t bytes) : _bytes(bytes)
            {
#ifdef DNDS_USE_CUDA
                if (bytes)
                {
                    CheckCUDA(cudaHostAlloc(&_pointer, bytes, cudaHostAllocPortable));
                    std::memset(_pointer, 0, bytes);
                }
#else
                DNDS_check_throw_info(false, "pinned host allocation requires CUDA");
#endif
            }
            ~PinnedHostResource() override { Release(nullptr); }
            void *Data() const noexcept override { return _pointer; }
            size_t Bytes() const noexcept override { return _bytes; }
            Device GetDevice() const noexcept override { return Device::Host(); }
            bool ExternalProvider() const noexcept override { return false; }
            bool PinnedHostMemory() const noexcept override { return true; }
            void Release(const ExecutionContext *) noexcept override
            {
#ifdef DNDS_USE_CUDA
                if (auto *pointer = std::exchange(_pointer, nullptr))
                    if (cudaFreeHost(pointer) != cudaSuccess)
                        std::terminate();
#endif
            }
        };

        void Fence(ExecutionContext context)
        {
            context.Synchronize();
        }

        class SharedResource final : public AllocationResource
        {
            std::shared_ptr<Allocator> _allocator;
            std::shared_ptr<void> _owner;
            size_t _bytes;
            Device _device;
            bool _streamOrderedMemory;

        public:
            SharedResource(std::shared_ptr<void> owner, size_t bytes, Device device, std::shared_ptr<Allocator> allocator = {}, bool streamOrderedMemory = false)
                : _allocator(std::move(allocator)), _owner(std::move(owner)), _bytes(bytes), _device(device), _streamOrderedMemory(streamOrderedMemory) {}
            void *Data() const noexcept override { return _owner.get(); }
            size_t Bytes() const noexcept override { return _bytes; }
            Device GetDevice() const noexcept override { return _device; }
            bool StreamOrderedMemory() const noexcept override { return _streamOrderedMemory; }
            void Release(const ExecutionContext *) noexcept override
            {
                _owner.reset();
                _allocator.reset();
            }
        };

        size_t ValidateAlignment(AllocationOptions options)
        {
            const auto alignment = options.alignmentBytes;
            DNDS_check_throw_info(!alignment || !(alignment & (alignment - 1)),
                                  "allocation alignment must be a power of two");
            return std::max(size_t(64), alignment);
        }

        class NativeResource final : public AllocationResource
        {
            void *_pointer = nullptr, *_allocation = nullptr;
            size_t _alignment;
            size_t _bytes;
            Device _device;
            bool _orderedCUDA = false;
            bool _asyncAllocation = false, _unpublished = false;
            ExecutionContext _construction;

        public:
            NativeResource(size_t bytes, Device device, AllocationOptions options = {}, const ExecutionContext *async = nullptr)
                : _alignment(ValidateAlignment(options)), _bytes(bytes), _device(device), _asyncAllocation(async != nullptr), _unpublished(async != nullptr),
                  _construction(async ? *async : ExecutionContext{})
            {
                if (!bytes)
                    return;
                if (device == Device::Host())
                {
                    _allocation = _pointer = ::operator new(bytes, std::align_val_t(_alignment));
                    std::memset(_pointer, 0, bytes);
                }
#ifdef DNDS_USE_CUDA
                else
                {
                    DeviceScope scope(device.ordinal);
                    // Keep the provider base for cudaFree[Async]; a stronger
                    // requested alignment may expose an interior payload address.
                    const auto allocatedBytes = CheckedSize::Add(bytes, _alignment - 1);
                    auto exposeAligned = [&]()
                    {
                        auto address = reinterpret_cast<uintptr_t>(_allocation);
                        _pointer = reinterpret_cast<void *>((address + _alignment - 1) & ~uintptr_t(_alignment - 1));
                    };
                    try
                    {
#    if CUDART_VERSION >= 11020
                        int pools = 0;
                        CheckCUDA(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, device.ordinal));
                        _orderedCUDA = bool(pools);
#    endif
                        if (async)
                        {
#    if CUDART_VERSION >= 11020
                            DNDS_check_throw_info(_orderedCUDA, "device does not support stream ordered allocation");
                            auto stream = CUDAStream(*async);
                            CheckCUDA(cudaMallocAsync(&_allocation, allocatedBytes, stream));
                            exposeAligned();
                            CheckCUDA(cudaMemsetAsync(_pointer, 0, bytes, stream));
#    else
                            DNDS_check_throw_info(false, "CUDA runtime lacks stream ordered allocation");
#    endif
                        }
                        else
                        {
                            CheckCUDA(cudaMalloc(&_allocation, allocatedBytes));
                            exposeAligned();
                            CheckCUDA(cudaMemset(_pointer, 0, bytes));
                            CheckCUDA(cudaStreamSynchronize(nullptr));
                        }
                    }
                    catch (...)
                    {
                        Release(async);
                        throw;
                    }
                }
#endif
            }
            ~NativeResource() override { Release(_unpublished ? &_construction : nullptr); }
            bool ExternalProvider() const noexcept override { return false; }
            void Published() { _unpublished = false; }
            bool StreamOrderedMemory() const noexcept override { return _asyncAllocation; }
            void *Data() const noexcept override { return _pointer; }
            size_t Bytes() const noexcept override { return _bytes; }
            Device GetDevice() const noexcept override { return _device; }
            bool CanReleaseOrdered(ExecutionContext context) const noexcept override
            {
                return context.device == _device && (!context.stream || (context.stream->GetDevice() == _device && !context.stream->IsRetired())) &&
                       (_device == Device::Host() || _orderedCUDA || !_bytes);
            }
            void Release(const ExecutionContext *context) noexcept override
            {
                _pointer = nullptr;
                auto *pointer = std::exchange(_allocation, nullptr);
                if (!pointer)
                    return;
                if (_device == Device::Host())
                    ::operator delete(pointer, std::align_val_t(_alignment));
#ifdef DNDS_USE_CUDA
                else
                {
                    DeviceScope device(_device.ordinal);
                    cudaError_t status;
#    if CUDART_VERSION >= 11020
                    if (context && (!context->stream || !context->stream->IsRetired()))
                        status = cudaFreeAsync(pointer, CUDAStream(*context));
                    else
#    endif
                    {
                        // cudaFree does not synchronize a cudaMallocAsync
                        // allocation. Unplaced RAII uses the conservative path.
                        if (_asyncAllocation && cudaDeviceSynchronize() != cudaSuccess)
                            std::terminate();
                        status = cudaFree(pointer);
                    }
                    if (status != cudaSuccess)
                        std::terminate();
                }
#endif
            }
        };
    }

    struct Completion::State
    {
        Device device;
        std::shared_ptr<StreamResource> stream;
#ifdef DNDS_USE_CUDA
        cudaEvent_t event = nullptr;
        mutable std::mutex observation;
        mutable std::atomic<bool> completed{false};
        mutable std::exception_ptr error;

        bool Completed() const
        {
            if (!completed.load(std::memory_order_acquire))
                return false;
            if (error)
                std::rethrow_exception(error);
            return true;
        }
        /// One terminal observation is shared by the copy and every attached
        /// lease. A nonblocking Ready must not wait behind another observer's
        /// blocking CUDA wait. Completion/error publication precedes the release
        /// store; later observers neither query nor synchronize the event again.
        bool Observe(bool wait) const
        {
            if (Completed())
                return true;
            std::unique_lock<std::mutex> lock(observation, std::defer_lock);
            if (wait)
                lock.lock();
            else if (!lock.try_lock())
                return false;
            if (Completed())
                return true;
            DeviceScope scope(device.ordinal);
            const auto status = wait ? cudaEventSynchronize(event) : cudaEventQuery(event);
            if (!wait && status == cudaErrorNotReady)
                return false;
            try
            {
                CheckCUDA(status);
            }
            catch (...)
            {
                error = std::current_exception();
            }
            completed.store(true, std::memory_order_release);
            if (error)
                std::rethrow_exception(error);
            return true;
        }
        ~State()
        {
            if (event)
            {
                try
                {
                    Observe(true); // Last ownership release still waits if unobserved.
                }
                catch (...)
                {
                    std::terminate();
                }
                DeviceScope scope(device.ordinal);
                if (cudaEventDestroy(event) != cudaSuccess)
                    std::terminate();
            }
        }
#endif
    };

    void ExecutionContext::Synchronize() const
    {
        ValidateDevice(device);
        Validate();
        if (stream)
        {
            stream->Synchronize();
            return;
        }
#ifdef DNDS_USE_CUDA
        if (device.space == MemorySpace::CUDA)
        {
            DeviceScope scope(device.ordinal);
            CheckCUDA(cudaStreamSynchronize(nullptr));
        }
#endif
    }

    Completion Completion::Record(ExecutionContext context)
    {
        ValidateDevice(context.device);
        context.Validate();
        if (context.device.space == MemorySpace::Host && !context.stream)
            return {};
        auto state = std::make_shared<State>();
        state->device = context.device;
        state->stream = context.stream;
        // This is the immediate-access completion API. CPU work has no native
        // asynchronous event here: observe the lane before claiming readiness.
        // Queued/nonblocking CPU prefixes use ExecutionStream::Record instead.
        if (context.device == Device::Host())
            context.Synchronize();
#ifdef DNDS_USE_CUDA
        if (context.device.space == MemorySpace::CUDA)
        {
            ValidateStream(context);
            DeviceScope device(context.device.ordinal);
            // Immediate completions are observed at host-access boundaries, just
            // like queued prefixes. Sleeping there avoids driver-side busy waiting;
            // device stream ordering and nonblocking Ready semantics are unchanged.
            CheckCUDA(cudaEventCreateWithFlags(&state->event, cudaEventDisableTiming | cudaEventBlockingSync));
            CheckCUDA(cudaEventRecord(state->event, CUDAStream(context)));
        }
#endif
        return Completion(std::move(state));
    }

    Completion Completion::FromExternalCompletion(ExecutionContext context)
    {
        ValidateDevice(context.device);
        context.Validate();
        if (context.device == Device::Host() && !context.stream)
            return {};
        auto state = std::make_shared<State>();
        state->device = context.device;
        state->stream = context.stream;
        return Completion(std::move(state));
    }

    void Completion::Wait() const
    {
#ifdef DNDS_USE_CUDA
        if (_state && _state->event)
            _state->Observe(true);
#endif
    }

    bool Completion::Ready() const
    {
#ifdef DNDS_USE_CUDA
        if (_state && _state->event)
            return _state->Observe(false);
#endif
        return true;
    }

    Device Completion::GetDevice() const { return _state ? _state->device : Device::Host(); }
    bool Completion::Matches(ExecutionContext context) const
    {
        return GetDevice() == context.device && (_state ? _state->stream : nullptr) == context.stream;
    }

    struct BufferState
    {
        Detail::AllocationReference allocation;
        size_t bytes = 0;
        Device device;
        uint64_t identity = nextIdentity.fetch_add(1);
        uint64_t version = 0;
        uint64_t mutationEpoch = 0;
        size_t readers = 0;
        size_t pendingReaders = 0, pendingWriters = 0;
        size_t checkedReaders = 0, checkedWriters = 0;
        bool writer = false;
        bool failed = false;
        bool writable = true;
        bool streamOrderedMemory = false;
        mutable std::mutex mutex;
        ~BufferState() { Detail::ForgetAllocation(identity); }
    };

    namespace
    {
        // Caller holds state.mutex. Keep dry-run preflight and real reservation
        // admission identical without publishing a canceled write intent.
        void ValidateReservationLocked(const BufferState &state, AccessMode mode, Device device, bool checked)
        {
            DNDS_check_throw_info(mode == AccessMode::Read || mode == AccessMode::Write || mode == AccessMode::ReadWrite,
                                  "runtime: invalid scheduled access mode");
            const bool write = mode != AccessMode::Read;
            DNDS_check_throw_info(state.device == device, "runtime: scheduled access placement mismatch");
            DNDS_check_throw_info(!state.failed, "runtime: failed allocation must be replaced");
            DNDS_check_throw_info(!write || state.writable, "runtime: imported allocation is read-only");
            DNDS_check_throw_info(!state.writer && (!write || !state.readers), "runtime: conflicting immediate buffer access");
            const auto otherWriters = checked ? state.pendingWriters - state.checkedWriters : state.checkedWriters;
            const auto otherReaders = checked ? state.pendingReaders - state.checkedReaders : state.checkedReaders;
            DNDS_check_throw_info(!otherWriters && (!write || !otherReaders),
                                  "runtime: conflicting checked/unchecked access; observe completion at the policy boundary");
            DNDS_check_throw_info((write ? state.pendingWriters : state.pendingReaders) < SIZE_MAX,
                                  "runtime: scheduled access count overflow");
            DNDS_check_throw_info(!write || state.mutationEpoch != UINT64_MAX, "runtime: mutation epoch overflow");
        }

        std::shared_ptr<BufferState> MakeState(std::unique_ptr<AllocationResource> resource)
        {
            DNDS_check_throw_info(resource, "runtime: null allocation resource");
            ValidateDevice(resource->GetDevice());
            DNDS_check_throw_info(!resource->Bytes() || resource->Data(), "runtime: allocation resource has null storage");
            DNDS_check_throw_info(resource->Bytes() <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(resource->Data()),
                                  "runtime: allocation address range overflow");
            auto state = std::make_shared<BufferState>();
            state->bytes = resource->Bytes();
            state->device = resource->GetDevice();
            state->streamOrderedMemory = resource->StreamOrderedMemory();
            const bool external = resource->ExternalProvider();
            const bool pinnedHost = resource->PinnedHostMemory();
            state->allocation = Detail::AllocationReference(std::move(resource));
            if (AllocationDiagnostics::Enabled())
            {
                const std::weak_ptr<BufferState> weak = state;
                Detail::TrackAllocation(state->identity, state->bytes, state->device, external, state->streamOrderedMemory, pinnedHost, [weak]
                                        {
                    AllocationActivity result;
                    if (const auto live = weak.lock())
                    {
                        std::lock_guard<std::mutex> lock(live->mutex);
                        result = {live->readers, live->pendingReaders, live->pendingWriters, live->writer};
                    }
                    return result; });
            }
            return state;
        }
    }

    Allocation Allocation::Import(std::shared_ptr<void> owner, size_t bytes, Device device, bool writable, bool streamOrderedMemory)
    {
        ValidateDevice(device);
        DNDS_check_throw_info(owner.use_count() && (!bytes || owner.get()), "runtime: imported allocation requires shared ownership");
        DNDS_check_throw_info(bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(owner.get()), "runtime: imported address range overflow");
        DNDS_check_throw_info(!streamOrderedMemory || device.space == MemorySpace::CUDA, "pool/VMM memory needs a device allocation");
#ifdef DNDS_USE_CUDA
        if (bytes && device.space == MemorySpace::CUDA)
        {
            // Validate foreign CUDA placement once at import, not at every view
            // access/launch. Capacity, readiness and owner lifetime remain the
            // foreign provider's contract; pointer attributes do not prove them.
            DeviceScope scope(device.ordinal);
            cudaPointerAttributes attributes{};
            CheckCUDA(cudaPointerGetAttributes(&attributes, owner.get()));
            DNDS_check_throw_info((attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged) &&
                                      attributes.device == device.ordinal,
                                  "runtime: imported CUDA pointer disagrees with declared placement");
        }
#endif
        auto state = MakeState(std::make_unique<SharedResource>(std::move(owner), bytes, device, nullptr, streamOrderedMemory));
        state->writable = writable;
        return Allocation(std::move(state));
    }

    size_t Allocation::Bytes() const { return _state ? _state->bytes : 0; }
    Device Allocation::GetDevice() const { return _state ? _state->device : Device::Host(); }
    Buffer::Buffer(std::shared_ptr<BufferState> state) : _state(std::move(state)), _bytes(_state->bytes) {}

    Buffer Buffer::FromAllocation(const Allocation &allocation, size_t offset, size_t bytes)
    {
        DNDS_check_throw_info(allocation._state && offset <= allocation.Bytes() && bytes <= allocation.Bytes() - offset,
                              "runtime: buffer range outside allocation");
        Buffer result(allocation._state);
        result._offset = offset;
        result._bytes = bytes;
        return result;
    }

    Buffer Buffer::Subspan(size_t offset, size_t bytes) const
    {
        DNDS_check_throw_info(offset <= _bytes && bytes <= _bytes - offset, "runtime: subspan outside buffer");
        return FromAllocation(AllocationHandle(), _offset + offset, bytes);
    }

    Buffer Buffer::Allocate(size_t bytes, Device device, AllocationOptions options)
    {
        ValidateDevice(device);
        return FromResource(std::make_unique<NativeResource>(bytes, device, options));
    }

    Buffer Buffer::AllocatePinnedHost(size_t bytes)
    {
        return FromResource(std::make_unique<PinnedHostResource>(bytes));
    }

    Buffer Buffer::Allocate(size_t bytes, Device device, std::shared_ptr<Allocator> allocator, AllocationOptions options)
    {
        ValidateDevice(device);
        DNDS_check_throw_info(allocator, "runtime: null allocator policy");
        ValidateAlignment(options);
        auto owner = allocator->Allocate(bytes, device, options);
        DNDS_check_throw_info(!bytes || (owner && (!options.alignmentBytes ||
                                                   reinterpret_cast<uintptr_t>(owner.get()) % options.alignmentBytes == 0)),
                              "allocator returned a misaligned allocation");
        const bool streamOrdered = allocator->StreamOrderedMemory(device);
        return FromResource(std::make_unique<SharedResource>(std::move(owner), bytes, device, std::move(allocator), streamOrdered));
    }
    Buffer Buffer::FromResource(std::unique_ptr<AllocationResource> resource) { return Buffer(MakeState(std::move(resource))); }
    Buffer Buffer::AllocateCUDAAsync(size_t bytes, ExecutionContext context, AllocationOptions options)
    {
        ValidateDevice(context.device);
        DNDS_check_throw_info(context.device.space == MemorySpace::CUDA, "async device allocation requires CUDA");
#ifdef DNDS_USE_CUDA
        ValidateStream(context);
#endif
        auto resource = std::make_unique<NativeResource>(bytes, context.device, options, &context);
        auto *native = resource.get();
        auto result = FromResource(std::move(resource));
        native->Published();
        return result;
    }

    size_t Buffer::Bytes() const { return _state ? _bytes : 0; }
    Device Buffer::GetDevice() const { return _state ? _state->device : Device::Host(); }
    uint64_t Buffer::Identity() const { return _state ? _state->identity : 0; }
    void Buffer::SetDiagnosticName(const std::string &name, AllocationSite site) const
    {
        DNDS_check_throw_info(_state, "cannot name an unallocated buffer");
        Detail::NameAllocation(_state->identity, name, site);
    }
    uint64_t Buffer::Version() const
    {
        if (!_state)
            return 0;
        std::lock_guard<std::mutex> lock(_state->mutex);
        return _state->version;
    }
    uint64_t Buffer::MutationEpoch() const
    {
        if (!_state)
            return 0;
        std::lock_guard<std::mutex> lock(_state->mutex);
        return _state->mutationEpoch;
    }
    bool Buffer::IsIdle() const
    {
        if (!_state)
            return true;
        std::lock_guard<std::mutex> lock(_state->mutex);
        return !_state->writer && !_state->readers && !_state->pendingReaders && !_state->pendingWriters;
    }
    bool Buffer::IsUnique() const { return _state && _state.use_count() == 1; }
    uintptr_t Buffer::Address() const
    {
        auto *base = _state ? static_cast<std::byte *>(_state->allocation.Data()) : nullptr;
        return reinterpret_cast<uintptr_t>(base ? base + _offset : nullptr);
    }
    bool Buffer::Overlaps(const Buffer &other) const
    {
        if (!Bytes() || !other.Bytes() || GetDevice() != other.GetDevice())
            return false;
        const auto a = Address(), b = other.Address();
        // Differences avoid overflowing the end address of a foreign range.
        return a <= b ? b - a < Bytes() : a - b < other.Bytes();
    }
    bool Buffer::SameAddress(const Buffer &other) const { return GetDevice() == other.GetDevice() && Address() == other.Address(); }
    bool Buffer::SameRange(const Buffer &other) const { return SameAddress(other) && Bytes() == other.Bytes(); }
    bool Buffer::IsAligned(size_t alignment) const { return alignment && Address() % alignment == 0; }
    void *Buffer::LegacyPointer() const { return reinterpret_cast<void *>(Address()); }
    bool Buffer::IsValid() const
    {
        if (!_state)
            return false;
        std::lock_guard<std::mutex> lock(_state->mutex);
        return !_state->failed;
    }
    bool Buffer::StreamOrderedMemory() const { return _state && _state->streamOrderedMemory; }
    bool Buffer::CanReleaseOrdered(ExecutionContext context) const
    {
        return !_state || _state->allocation.CanReleaseOrdered(context);
    }
    void Buffer::ReleaseOn(ExecutionContext context)
    {
        if (!_state)
            return;
        ValidateDevice(context.device);
        context.Validate();
#ifdef DNDS_USE_CUDA
        if (context.device.space == MemorySpace::CUDA)
            ValidateStream(context);
#endif
        DNDS_check_throw_info(CanReleaseOrdered(context), "allocation provider cannot release on this stream");
        // The extra payload reference carries the explicit context through
        // controller destruction. It never tests use_count or force-frees aliases.
        auto payload = _state->allocation;
        _state.reset();
        _offset = _bytes = 0;
        payload.ReleaseOn(context);
    }
    ReadAccess Buffer::Read(ExecutionContext context) const { return ReadAccess(Access(_state, _offset, _bytes, context, false)); }
    WriteAccess Buffer::Write(ExecutionContext context) { return WriteAccess(Access(_state, _offset, _bytes, context, true)); }
    ScheduledAccess Buffer::Reserve(AccessMode mode, Device device, bool checked) const
    {
        return ScheduledAccess(_state, _offset, _bytes, device, mode, checked);
    }
    void Buffer::ValidateReservation(AccessMode mode, Device device, bool checked) const
    {
        DNDS_check_throw_info(_state, "runtime: access to an unallocated buffer");
        std::lock_guard<std::mutex> lock(_state->mutex);
        ValidateReservationLocked(*_state, mode, device, checked);
    }

    ScheduledAccess::ScheduledAccess(std::shared_ptr<BufferState> state, size_t offset, size_t bytes, Device device, AccessMode mode, bool checked)
        : _state(std::move(state)), _offset(offset), _bytes(bytes), _write(mode != AccessMode::Read), _checked(checked)
    {
        DNDS_check_throw_info(_state, "runtime: access to an unallocated buffer");
        std::lock_guard<std::mutex> lock(_state->mutex);
        ValidateReservationLocked(*_state, mode, device, checked);
        auto &count = _write ? _state->pendingWriters : _state->pendingReaders;
        if (_write)
        {
            ++_state->mutationEpoch;
        }
        ++count;
        if (_checked)
            ++(_write ? _state->checkedWriters : _state->checkedReaders);
    }
    ScheduledAccess &ScheduledAccess::operator=(ScheduledAccess &&other) noexcept
    {
        if (this != &other)
        {
            Cancel();
            _state = std::move(other._state);
            _issued = std::move(other._issued);
            _offset = other._offset;
            _bytes = other._bytes;
            _write = other._write;
            _checked = other._checked;
        }
        return *this;
    }
    ScheduledAccess::~ScheduledAccess() { Cancel(); }
    void ScheduledAccess::Validate() const
    {
        auto state = _state ? _state : _issued.lock();
        DNDS_check_throw_info(state, "runtime: inactive scheduled access");
        std::lock_guard<std::mutex> lock(state->mutex);
        DNDS_check_throw_info(!state->failed, "runtime: scheduled operand has failed");
    }
    void *ScheduledAccess::Pointer(size_t alignment, size_t elementBytes) const
    {
        DNDS_check_throw_info(_state, "runtime: scheduled view requested after handoff");
        Validate();
        auto *base = static_cast<std::byte *>(_state->allocation.Data());
        auto *pointer = base ? base + _offset : nullptr;
        DNDS_check_throw_info(elementBytes && alignment && _bytes % elementBytes == 0 &&
                                  reinterpret_cast<uintptr_t>(pointer) % alignment == 0,
                              "runtime: incompatible scheduled view size/alignment");
        return pointer;
    }
    void ScheduledAccess::Handoff(ExecutionContext context)
    {
        DNDS_check_throw_info(_state && _state->device.space == MemorySpace::CUDA &&
                                  _state->allocation.CanReleaseOrdered(context),
                              "runtime: unsupported scheduled handoff");
        auto payload = _state->allocation;
        _issued = _state;
        _state.reset();
        payload.ReleaseOn(context);
    }
    void ScheduledAccess::Complete(bool success) noexcept
    {
        auto state = _state ? std::move(_state) : _issued.lock();
        _issued.reset();
        if (!state)
            return;
        std::lock_guard<std::mutex> lock(state->mutex);
        if (_checked)
            --(_write ? state->checkedWriters : state->checkedReaders);
        if (_write)
        {
            --state->pendingWriters;
            if (success && !state->failed)
                ++state->version;
            else
                state->failed = true;
        }
        else
            --state->pendingReaders;
    }
    void ScheduledAccess::Cancel() noexcept
    {
        auto state = _state ? std::move(_state) : _issued.lock();
        _issued.reset();
        if (!state)
            return;
        std::lock_guard<std::mutex> lock(state->mutex);
        --(_write ? state->pendingWriters : state->pendingReaders);
        if (_checked)
            --(_write ? state->checkedWriters : state->checkedReaders);
    }

    Access::Access(std::shared_ptr<BufferState> state, size_t offset, size_t bytes, ExecutionContext context, bool write)
        : _state(std::move(state)), _context(context), _write(write), _offset(offset), _bytes(bytes), _uncaughtOnAcquire(std::uncaught_exceptions())
    {
        ValidateDevice(context.device);
        context.Validate();
        DNDS_check_throw_info(!context.stream || !context.stream->IsRetired(), "runtime: execution stream has retired");
        DNDS_check_throw_info(_state, "runtime: access to an unallocated buffer");
        DNDS_check_throw_info(_state->device == context.device,
                              "runtime: placement mismatch; copy explicitly before accessing");
#ifdef DNDS_USE_CUDA
        if (context.device.space == MemorySpace::CUDA)
            ValidateStream(context);
#endif
        std::lock_guard<std::mutex> lock(_state->mutex);
        DNDS_check_throw_info(!_state->failed, "runtime: failed allocation must be replaced");
        DNDS_check_throw_info(!write || _state->writable, "runtime: imported allocation is read-only");
        DNDS_check_throw_info(!_state->writer && (!write || !_state->readers), "runtime: conflicting buffer access");
        DNDS_check_throw_info(!_state->pendingWriters && (!write || !_state->pendingReaders),
                              "runtime: pending scheduled access; observe stream completion first");
        if (write)
        {
            DNDS_check_throw_info(_state->mutationEpoch != UINT64_MAX, "runtime: mutation epoch overflow");
            ++_state->mutationEpoch;
            _state->writer = true;
        }
        else
            ++_state->readers;
    }

    Access::Access(Access &&other) noexcept
        : _state(std::move(other._state)), _context(std::move(other._context)), _completion(std::move(other._completion)),
          _hasCompletion(other._hasCompletion), _write(other._write), _offset(other._offset), _bytes(other._bytes), _uncaughtOnAcquire(other._uncaughtOnAcquire) {}

    Access &Access::operator=(Access &&other) noexcept
    {
        if (this != &other)
        {
            try
            {
                Wait();
            }
            catch (...)
            {
                std::terminate();
            }
            _state = std::move(other._state);
            _context = std::move(other._context);
            _completion = std::move(other._completion);
            _hasCompletion = other._hasCompletion;
            _write = other._write;
            _offset = other._offset;
            _bytes = other._bytes;
            _uncaughtOnAcquire = other._uncaughtOnAcquire;
        }
        return *this;
    }

    Access::~Access()
    {
        try
        {
            if (_write && std::uncaught_exceptions() > _uncaughtOnAcquire)
                Fail();
            else
                Wait();
        }
        catch (...)
        {
            std::terminate();
        }
    }

    void Access::Complete(Completion completion)
    {
        DNDS_check_throw_info(_state && !_hasCompletion, "runtime: access already submitted or released");
        DNDS_check_throw_info(completion.Matches(_context), "runtime: completion device/stream mismatch");
        _completion = std::move(completion);
        _hasCompletion = true;
    }

    void Access::Wait()
    {
        if (!_state)
            return;
        std::exception_ptr error;
        try
        {
            if (_hasCompletion)
                _completion.Wait();
            else
                Fence(_context);
        }
        catch (...)
        {
            error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            if (_write)
            {
                ++_state->version;
                _state->failed |= bool(error);
                _state->writer = false;
            }
            else
                --_state->readers;
        }
        _state.reset();
        _context = {};
        _completion = {};
        _hasCompletion = false;
        if (error)
            std::rethrow_exception(error);
    }

    void Access::CancelUnmodified()
    {
        if (!_state)
            return;
        DNDS_check_throw_info(!_hasCompletion, "cannot cancel submitted access");
        auto state = std::move(_state);
        _context = {};
        _completion = {};
        _hasCompletion = false;
        std::lock_guard<std::mutex> lock(state->mutex);
        if (_write)
            state->writer = false;
        else
            --state->readers;
    }

    void Access::Fail()
    {
        if (!_state)
            return;
        if (_write)
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            _state->failed = true;
        }
        Wait();
    }

    size_t Access::Bytes() const
    {
        DNDS_check_throw(_state);
        return _bytes;
    }
    Device Access::GetDevice() const
    {
        DNDS_check_throw(_state);
        return _state->device;
    }
    void *Access::Pointer(size_t alignment, size_t elementSize) const
    {
        DNDS_check_throw_info(_state && !_hasCompletion, "runtime: view requested after submit/release");
        auto *base = static_cast<std::byte *>(_state->allocation.Data());
        auto *p = base ? base + _offset : nullptr;
        DNDS_check_throw_info(_bytes % elementSize == 0 && reinterpret_cast<uintptr_t>(p) % alignment == 0,
                              "runtime: incompatible typed view size/alignment");
        return p;
    }

    void Detail::EnqueueCopyBytes(const void *source, Device sourceDevice, void *destination, Device destinationDevice,
                                  size_t bytes, ExecutionContext context)
    {
        ValidateDevice(context.device);
        const bool gpu = sourceDevice != Device::Host() || destinationDevice != Device::Host();
        DNDS_check_throw_info(gpu == (context.device.space == MemorySpace::CUDA), "copy execution device mismatch");
        DNDS_check_throw_info((sourceDevice == Device::Host() || sourceDevice == context.device) &&
                                  (destinationDevice == Device::Host() || destinationDevice == context.device),
                              "copy device ordinal mismatch");
        if (!gpu)
        {
            context.Validate();
            if (bytes)
                std::memcpy(destination, source, bytes);
        }
#ifdef DNDS_USE_CUDA
        else
        {
            DeviceScope device(context.device.ordinal);
            ValidateStream(context);
            if (bytes)
                CheckCUDA(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDefault, CUDAStream(context)));
        }
#endif
    }

    Completion Detail::CopyBytes(const void *source, Device sourceDevice, void *destination, Device destinationDevice,
                                 size_t bytes, ExecutionContext context)
    {
        EnqueueCopyBytes(source, sourceDevice, destination, destinationDevice, bytes, context);
        return Completion::Record(context);
    }

    CopyOperation Copy(const Buffer &source, Buffer &destination, ExecutionContext context)
    {
        DNDS_check_throw_info(source.Identity() && destination.Identity() && source.Identity() != destination.Identity(),
                              "runtime: copy requires distinct allocated buffers");
        DNDS_check_throw_info(source.Bytes() == destination.Bytes(), "runtime: copy size mismatch");
        ValidateDevice(context.device);
        auto sourceContext = source.GetDevice().space == MemorySpace::Host ? ExecutionContext::Host() : context;
        context.Validate();
        auto destinationContext = destination.GetDevice().space == MemorySpace::Host ? ExecutionContext::Host() : context;
        const bool onDevice = source.GetDevice().space == MemorySpace::CUDA || destination.GetDevice().space == MemorySpace::CUDA;
        DNDS_check_throw_info(onDevice == (context.device.space == MemorySpace::CUDA), "runtime: copy execution device mismatch");
        CopyOperation operation;
        operation._source = source.Read(sourceContext);
        operation._destination = destination.Write(destinationContext);
        auto src = operation._source.View<std::byte>();
        auto dst = operation._destination.View<std::byte>();
        if (!onDevice)
        {
            if (source.Bytes())
                std::memcpy(dst.data, src.data, source.Bytes());
            operation.Wait();
        }
#ifdef DNDS_USE_CUDA
        else
        {
            DeviceScope device(context.device.ordinal);
            operation._source._context = context;
            operation._destination._context = context;
            if (source.Bytes())
                CheckCUDA(cudaMemcpyAsync(dst.data, src.data, source.Bytes(), cudaMemcpyDefault,
                                          CUDAStream(context)));
            // Host staging leases also have to wait on the CUDA copy. Attach
            // the copy context before recording its event (no host-side access).
            auto done = Completion::Record(context);
            operation._source.Complete(done);
            operation._destination.Complete(done);
        }
#endif
        return operation;
    }

    void CopyOperation::Wait()
    {
        _source.Wait();
        _destination.Wait();
    }
}
