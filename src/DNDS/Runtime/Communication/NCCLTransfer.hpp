#pragma once

#include "DNDS/Runtime/Communication/GlobalReduction.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"

namespace DNDS::Runtime
{
    /// One rank per CUDA device. MPI bootstraps only; prepared payload operations
    /// are stream ordered. Managed Enqueue retains the provider on the runtime's
    /// controller through completion; manual Begin remains caller-managed.
    /// Shared-communicator submissions must have matching per-peer call order
    /// across ranks. Use separate providers for independent communication lanes.
    /// Direct requires contiguous described fields; DevicePacked handles general
    /// supported layouts. Unsupported reducers fail at binding, never silently
    /// fall back to MPI. No global CUDA synchronization is implied.
    class NCCLTransferProvider final : public TransferProvider
    {
        struct State;
        std::shared_ptr<State> _state;
        explicit NCCLTransferProvider(std::shared_ptr<State>);
        std::shared_ptr<const GlobalReductionBinding> BindAllreduceBuffers(
            Buffer, Buffer, const RowLayout &, const RowLayout &, ElementScalar, Reduction) const;

    public:
        /// Build capability only; Create validates runtime device availability.
        static bool Available();
        /// Collective setup on comm, outside hot execution/capture. Device
        /// ordinal is rank-local; each rank must select a distinct physical GPU.
        static std::shared_ptr<NCCLTransferProvider> Create(MPI_Comm comm, Runtime::Device device);
        /// Share the actual bootstrap/control owner without duplicating MPI.
        /// Raw-communicator Create is the compatibility isolated-namespace path.
        static std::shared_ptr<NCCLTransferProvider> Create(std::shared_ptr<CommunicationContext> context, Runtime::Device device);
        std::shared_ptr<const CommunicationContext> ControlContext() const;
        /// Nonblocking error observation for manual progress. Managed runtime
        /// progress polls this too and aborts on asynchronous transport failure;
        /// collaborative solver paths never recover from exceptions.
        void CheckAsyncError() const;
        /// Explicit cancellation of this communicator and all its bindings.
        /// Drain engine work before releasing arrays or their native streams.
        void Abort() const noexcept;
        /// Prepare an elementwise sum, minimum or maximum for two equally shaped
        /// compact tables. Exact in-place reduction is supported. All ranks must
        /// participate in binding; repeated submissions perform no MPI preflight.
        template <class T>
        std::shared_ptr<const GlobalReductionBinding> BindAllreduce(
            const Array<T> &source, Array<T> &destination, Reduction operation) const
        {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>);
            return BindAllreduceBuffers(source.Storage(), destination.Storage(), source.Layout(), destination.Layout(),
                                        ElementSchema::Arithmetic<T>().Fields().front().scalar, operation);
        }
        const char *Name() const override { return "nccl"; }
        std::shared_ptr<Detail::TransferProviderBinding> Bind(
            const TransferPayload &, Buffer, Buffer, std::shared_ptr<const Reducer>,
            TransferTransport, const TransferOptions &, bool layoutsValid) const override;
    };
}
