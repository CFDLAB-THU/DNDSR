#pragma once

#include "DNDS/Runtime/Storage/Buffer.hpp"
#include <atomic>
#include <utility>

namespace DNDS::Runtime
{
    /// One backend-owned allocation. Its provider owns any registration/pool
    /// context. Release is called once by final ownership, with an explicit
    /// stream only when the provider accepts ordered release there. Destruction
    /// must also reclaim an unpublished resource if construction/import fails.
    class AllocationResource
    {
    public:
        virtual ~AllocationResource() = default;
        virtual void *Data() const noexcept = 0;
        virtual size_t Bytes() const noexcept = 0;
        virtual Device GetDevice() const noexcept = 0;
        /// True for imported/user-provider memory. Capacity remains the provider's
        /// declared extent; accounting does not infer allocator-pool reservations.
        virtual bool ExternalProvider() const noexcept { return true; }
        /// Pool/VMM allocations require extra foreign-provider capability (e.g.
        /// CUDA-aware MPI alone does not imply support for cudaMallocAsync).
        virtual bool StreamOrderedMemory() const noexcept { return false; }
        /// Page-locked host allocation suitable for asynchronous CUDA DMA.
        /// This is an allocation property, not a different view residency.
        virtual bool PinnedHostMemory() const noexcept { return false; }
        virtual bool CanReleaseOrdered(ExecutionContext context) const noexcept
        {
            return GetDevice() == Device::Host() && context.device == Device::Host() &&
                   (!context.stream || (context.stream->GetDevice() == Device::Host() && !context.stream->IsRetired()));
        }
        /// Null context is conservative ordinary RAII. A provider with native
        /// pending uses must make that fallback safe; no last-stream guessing.
        virtual void Release(const ExecutionContext *context) noexcept = 0;
    };

    namespace Detail
    {
        /// Ordinary shared ownership with a context-aware final decrement.
        /// This replaces the payload's shared_ptr control block; it is not a
        /// second pending-use registry or a count of public/engine handles.
        class AllocationReference
        {
            struct State
            {
                std::atomic<size_t> references{1};
                std::unique_ptr<AllocationResource> resource;
                explicit State(std::unique_ptr<AllocationResource> value) : resource(std::move(value)) {}
            };
            State *_state = nullptr;
            void Drop(const ExecutionContext *context) noexcept
            {
                auto *state = std::exchange(_state, nullptr);
                if (state && state->references.fetch_sub(1, std::memory_order_acq_rel) == 1)
                {
                    state->resource->Release(context);
                    delete state;
                }
            }

        public:
            AllocationReference() = default;
            explicit AllocationReference(std::unique_ptr<AllocationResource> resource)
            {
                DNDS_check_throw_info(resource, "null allocation resource");
                _state = new State(std::move(resource));
            }
            AllocationReference(const AllocationReference &other) noexcept : _state(other._state)
            {
                if (_state)
                    _state->references.fetch_add(1, std::memory_order_relaxed);
            }
            AllocationReference(AllocationReference &&other) noexcept : _state(std::exchange(other._state, nullptr)) {}
            AllocationReference &operator=(const AllocationReference &other) noexcept
            {
                if (this != &other)
                {
                    AllocationReference retained(other);
                    std::swap(_state, retained._state);
                }
                return *this;
            }
            AllocationReference &operator=(AllocationReference &&other) noexcept
            {
                if (this != &other)
                {
                    Drop(nullptr);
                    _state = std::exchange(other._state, nullptr);
                }
                return *this;
            }
            ~AllocationReference() { Drop(nullptr); }
            void *Data() const noexcept { return _state ? _state->resource->Data() : nullptr; }
            bool CanReleaseOrdered(ExecutionContext context) const noexcept
            {
                return !_state || _state->resource->CanReleaseOrdered(context);
            }
            void ReleaseOn(ExecutionContext context) noexcept { Drop(&context); }
        };
    }
}
