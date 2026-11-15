#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include <tuple>

namespace DNDS::Runtime
{
    /// Borrow an immutable metadata publication as a residency-tagged whole
    /// array view. Its owner must be captured until the operation completes.
    /// Metadata has no mutable aliases; ordinary Arrays still need operands.
    template <class T>
    ArrayView<const T, AccessBoundsChecks, FixedRows<1>> MetadataArrayView(const MetadataBuffer<T> &metadata, Device device)
    {
        const auto values = metadata.View(device);
        return {values.data(), nullptr, index(values.size()), 1, 1, nullptr, RowLayoutKind::Table, device};
    }
    /// Structural pin and declared intent; construction does not reserve access.
    /// SubmitArrays reserves all operands before accepting the command.
    template <class T, bool Writable, class LayoutPolicy = RowLayout>
    struct ArrayOperand
    {
        using Value = std::conditional_t<Writable, T, const T>;
        using Policy = LayoutPolicy;
        ArrayStorage::Shared storage;
        AccessMode mode;
        bool orderedHandoff = false;
    };

    template <class T>
    ArrayOperand<T, false, RowLayout> Read(const Array<T> &array) { return {array.SharedStorage(), AccessMode::Read}; }
    template <class T>
    ArrayOperand<T, true, RowLayout> Write(Array<T> &array) { return {array.SharedStorage(), AccessMode::Write}; }
    template <class T>
    ArrayOperand<T, true, RowLayout> ReadWrite(Array<T> &array) { return {array.SharedStorage(), AccessMode::ReadWrite}; }
    template <class Policy, class T>
    ArrayOperand<T, false, Policy> ReadAs(const Array<T> &array) { return {array.SharedStorage(), AccessMode::Read}; }
    template <class Policy, class T>
    ArrayOperand<T, true, Policy> WriteAs(Array<T> &array) { return {array.SharedStorage(), AccessMode::Write}; }
    template <class Policy, class T>
    ArrayOperand<T, true, Policy> ReadWriteAs(Array<T> &array) { return {array.SharedStorage(), AccessMode::ReadWrite}; }

    /// Advanced opt-in: after GPU launch, release operand ownership on that
    /// stream instead of retaining it to observed completion. Every remaining
    /// caller alias must use an explicit, correctly joined ReleaseOn; ordinary
    /// RAII of the last alias may otherwise synchronize before host progress.
    /// CPU execution retains the normal execution-lifetime behavior.
    template <class Operand>
    Operand WithOrderedHandoff(Operand operand)
    {
        operand.orderedHandoff = true;
        return operand;
    }

    /// Values are initialized in stream order. New GPU tables need no metadata
    /// upload. For variable GPU shapes, prepare/upload metadata during setup and
    /// use AllocateLikeAsync to share it without allocation or upload per field.
    template <class T, class L = RowLayout>
    Array<T, L> AllocateArrayAsync(EagerEngine &engine, const ExecutionStream &stream, RowLayout layout, AllocationOptions options = {})
    {
        if constexpr (!std::is_same_v<L, RowLayout>)
            L::Validate(layout);
        DNDS_check_throw_info(stream.Context().device == Device::Host() || !layout.IsVariable(),
                              "new variable GPU metadata needs setup; use AllocateLikeAsync with a prepared Array");
        DNDS_check_throw_info(!options.alignmentBytes || !(options.alignmentBytes & (options.alignmentBytes - 1)),
                              "allocation alignment must be a power of two");
        options.alignmentBytes = std::max(options.alignmentBytes, size_t(alignof(T)));
        auto values = engine.AllocateBuffer(stream, CheckedSize::Multiply(layout.Elements(), sizeof(T)), options);
        try
        {
            auto result = Array<T, L>::FromStorage(std::move(layout), values);
            values.ReleaseOn(stream.Context());
            return result;
        }
        catch (...)
        {
            // Keep one allocation owner until metadata/handle construction
            // succeeds, so failure can also free in the allocation stream.
            values.ReleaseOn(stream.Context());
            throw;
        }
    }
    template <class T>
    Array<T> AllocateLikeAsync(EagerEngine &engine, const ExecutionStream &stream, const Array<T> &prototype)
    {
        auto storage = prototype.SharedStorage();
        auto values = storage->Values();
        DNDS_check_throw_info(values.GetDevice() == stream.Context().device, "async prototype placement mismatch");
        auto fresh = engine.AllocateBuffer(stream, values.Bytes(), AllocationOptions{alignof(T)});
        try
        {
            auto result = Array<T>::FromSharedStorage(storage->WithValues(fresh, sizeof(T)));
            fresh.ReleaseOn(stream.Context());
            return result;
        }
        catch (...)
        {
            fresh.ReleaseOn(stream.Context());
            throw;
        }
    }

    namespace Detail
    {
        template <class Operand>
        struct PreparedOperand
        {
            using Value = typename Operand::Value;
            using Policy = typename Operand::Policy;
            ArrayStorage::Shared storage;
            std::shared_ptr<const ArrayLayout> layout;
            ScheduledAccess access;
            ExecutionContext releaseContext;
            bool orderedHandoff;
            ArrayView<Value, AccessBoundsChecks, Policy> view;

            PreparedOperand(Operand operand, ExecutionContext context, bool checked)
                : storage(std::move(operand.storage)), layout(storage->SharedLayout()),
                  access(storage->Values().Reserve(operand.mode, context.device, checked)),
                  releaseContext(context), orderedHandoff(operand.orderedHandoff)
            {
                DNDS_check_throw_info(storage->ElementBytes() == sizeof(Value), "eager operand element size mismatch");
                auto *pointer = static_cast<Value *>(access.Pointer(alignof(Value), sizeof(Value)));
                const auto device = context.device;
                const auto &rows = layout->rows;
                ArrayView<Value, AccessBoundsChecks> dynamic{pointer, layout->Offsets(device), rows.RowCount(), rows.Width(), rows.Stride(), layout->Sizes(device), rows.Kind(), device};
                view = ProjectView<Value, AccessBoundsChecks, Policy>(dynamic, rows);
            }
            void Handoff(ExecutionContext context)
            {
                if (orderedHandoff && storage && context.device.space == MemorySpace::CUDA && storage->CanReleaseOrdered(context) && layout->CanReleaseOrdered(context))
                {
                    ArrayStorage::ReleaseHandle(storage, context);
                    ArrayLayout::ReleaseHandle(layout, context);
                    access.Handoff(context);
                }
            }
            void Complete(bool success) noexcept
            {
                access.Complete(success);
                // Ordinary submission owns its operands through completion.
                // Use the known device stream for the final decrement rather
                // than a potentially global-synchronizing cudaFree fallback.
                try
                {
                    if (storage && releaseContext.device.space == MemorySpace::CUDA &&
                        storage->CanReleaseOrdered(releaseContext) && layout->CanReleaseOrdered(releaseContext))
                    {
                        ArrayStorage::ReleaseHandle(storage, releaseContext);
                        ArrayLayout::ReleaseHandle(layout, releaseContext);
                    }
                }
                catch (...)
                {
                    std::terminate();
                } // Retirement cannot recover.
            }
        };
    }

    /// Same orchestration for inline/worker CPU and device launchers. Kernel
    /// receives only bare views. Its return means CPU completion or device
    /// submission; the launcher decides when the retirement callback is safe.
    /// @pre Declare every accessed allocation and its correct intent; do not
    /// capture untracked raw pointers or escape views beyond this operation.
    /// Caller proves stencil bounds and orders conflicting accesses by stream
    /// edges. A Write declaration neither zeros nor proves full initialization.
    /// CUDA kernels must launch on the supplied context stream and report launch
    /// errors; launching on another stream escapes the lifetime contract.
    /// Captured owners (e.g. immutable metadata/mesh read scopes) are retained
    /// by the retirement record, not merely the host launch callback. Their
    /// lifetime therefore ends per completed operation, without an engine drain.
    template <class Kernel, class... Operands>
    void SubmitArrays(EagerEngine &engine, const ExecutionStream &stream, Kernel kernel, Operands... operands)
    {
        using Prepared = std::tuple<Detail::PreparedOperand<Operands>...>;
        struct Retained
        {
            Prepared operands;
            Kernel kernel;
            Retained(Kernel body, ExecutionContext context, bool checked, Operands... values)
                : operands(Detail::PreparedOperand<Operands>(std::move(values), context, checked)...), kernel(std::move(body)) {}
        };
        const std::vector<ScheduledBufferAccess> accesses{{operands.storage->Values(), operands.mode}...};
        const auto context = stream.Context();
        auto retained = std::make_shared<Retained>(std::move(kernel), context, engine.ChecksAccess(), std::move(operands)...);
        engine.SubmitOperation(stream, [retained](ExecutionContext context) mutable
                               { std::apply([&](auto &...operand)
                                            {
                    try
                    {
                        (operand.access.Validate(), ...);
                        retained->kernel(context, operand.view...);
                    }
                    catch (...)
                    {
                        // A failed launcher may already have issued native work.
                        // Ordered release follows that prefix; publication still
                        // retires as failed. Unsupported providers stay retained.
                        auto error = std::current_exception();
                        auto handoff = [context](auto &value) { try { value.Handoff(context); } catch (...) {} };
                        (handoff(operand), ...);
                        std::rethrow_exception(error);
                    }
                    (operand.Handoff(context), ...); }, retained->operands); }, [retained](std::exception_ptr error) noexcept
                               { std::apply([&](auto &...operand)
                                            { (operand.Complete(!error), ...); }, retained->operands); }, accesses);
    }
}
