#pragma once

#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/Runtime/Storage/BufferAlias.hpp"

namespace DNDS::Runtime
{
    struct CopyAsyncOptions
    {
        /// Advanced caller-managed lifetime, like WithOrderedHandoff operands.
        /// Remaining aliases must use correctly joined explicit ReleaseOn.
        bool orderedSource = false, orderedDestination = false;
    };
    /// Explicit byte copy in an eager stream. Host DMA buffers remain retained
    /// through actual completion. No event or host wait is inserted per copy.
    inline void CopyAsync(EagerEngine &engine, const ExecutionStream &stream, const Buffer &source, Buffer &destination, CopyAsyncOptions options = {})
    {
        const auto context = stream.Context();
        const auto srcDevice = source.GetDevice(), dstDevice = destination.GetDevice();
        const bool gpu = srcDevice != Device::Host() || dstDevice != Device::Host();
        DNDS_check_throw_info(Detail::IndependentBuffers(source, destination) &&
                                  source.Bytes() == destination.Bytes(),
                              "eager copy requires independent nonoverlapping equal-size buffers");
        DNDS_check_throw_info(gpu == (context.device.space == MemorySpace::CUDA) &&
                                  (srcDevice == Device::Host() || srcDevice == context.device) &&
                                  (dstDevice == Device::Host() || dstDevice == context.device),
                              "eager copy placement mismatch");
        struct Pending
        {
            Buffer sourceOwner, destinationOwner;
            ScheduledAccess source, destination;
            ExecutionContext context;
            const void *src;
            void *dst;
            bool handoffSource, handoffDestination;
            Pending(const Buffer &in, const Buffer &out, ExecutionContext context, bool checked, CopyAsyncOptions options)
                : sourceOwner(in), destinationOwner(out), source(in.Reserve(AccessMode::Read, in.GetDevice(), checked)), destination(out.Reserve(AccessMode::Write, out.GetDevice(), checked)),
                  context(context), src(source.Pointer(1, 1)), dst(destination.Pointer(1, 1)),
                  handoffSource(options.orderedSource && in.GetDevice().space == MemorySpace::CUDA && in.CanReleaseOrdered(context)),
                  handoffDestination(options.orderedDestination && out.GetDevice().space == MemorySpace::CUDA && out.CanReleaseOrdered(context)) {}
        };
        auto retained = std::make_shared<Pending>(source, destination, context, engine.ChecksAccess(), options);
        engine.SubmitOperation(stream, [retained, srcDevice, dstDevice, bytes = source.Bytes()](ExecutionContext execution)
                               {
                retained->source.Validate(); retained->destination.Validate();
                Detail::EnqueueCopyBytes(retained->src, srcDevice, retained->dst, dstDevice, bytes, execution);
                if (retained->handoffSource) { retained->sourceOwner.ReleaseOn(execution); retained->source.Handoff(execution); }
                if (retained->handoffDestination) { retained->destinationOwner.ReleaseOn(execution); retained->destination.Handoff(execution); } }, [retained](std::exception_ptr error) noexcept
                               {
                retained->source.Complete(!error);
                retained->destination.Complete(!error);
                try
                {
                    for (auto *owner : {&retained->sourceOwner, &retained->destinationOwner})
                        if (owner->GetDevice().space == MemorySpace::CUDA && owner->CanReleaseOrdered(retained->context))
                            owner->ReleaseOn(retained->context);
                }
                catch (...) { std::terminate(); } }, {{source, AccessMode::Read}, {destination, AccessMode::Write}});
    }
}
