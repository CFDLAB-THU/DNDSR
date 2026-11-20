#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Runtime/Communication/NCCLTransfer.hpp"
#include "DNDS/Runtime/Communication/DevicePacking.hpp"
#include "DNDS/Runtime/Communication/TransferAccess.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>

#ifdef DNDS_USE_NCCL
#    include <cuda_runtime.h>
#    include <nccl.h>
#endif

namespace DNDS::Runtime
{
#ifdef DNDS_USE_NCCL
    namespace
    {
        void CheckCUDA(cudaError_t error)
        {
            DNDS_check_throw_info(error == cudaSuccess, cudaGetErrorString(error));
        }
        void CheckNCCL(ncclResult_t error)
        {
            DNDS_check_throw_info(error == ncclSuccess, ncclGetErrorString(error));
        }
        template <class T>
        Buffer Upload(const std::vector<T> &values, Device device)
        {
            auto host = Buffer::Allocate(CheckedSize::Multiply(values.size(), sizeof(T)));
            {
                auto write = host.Write();
                std::copy(values.begin(), values.end(), write.template View<T>().data);
            }
            auto result = Buffer::Allocate(host.Bytes(), device);
            Copy(host, result, {device, nullptr}).Wait();
            return result;
        }
    }
    struct NCCLTransferProvider::State
    {
        struct Binding;
        struct Allreduce;
        struct Pending final : ExecutionProgress
        {
            std::shared_ptr<State> owner;
            std::shared_ptr<const void> binding;
            std::atomic<bool> retired{false};
            std::exception_ptr failure;

            Pending(std::shared_ptr<State> state, std::shared_ptr<const void> resource)
                : owner(std::move(state)), binding(std::move(resource)) {}
            bool Poll() override
            {
                // Only the controller polls and releases the communicator. The
                // event observer publishes the failure before setting retired.
                const bool done = retired.load(std::memory_order_acquire);
                try
                {
                    owner->Check();
                }
                catch (...)
                {
                    MPI::Abort(owner->control, "managed NCCL asynchronous failure");
                }
                if (!done)
                    return false;
                if (failure)
                    MPI::Abort(owner->control, "managed NCCL retirement failed");
                binding.reset();
                owner.reset();
                return true;
            }
        };
        Device device;
        std::shared_ptr<CommunicationContext> controlOwner;
        MPI_Comm control = MPI_COMM_NULL;
        const std::thread::id controller = std::this_thread::get_id();
        ncclComm_t comm = nullptr;
        ncclUniqueId identity;
        std::mutex submission;
        std::atomic<bool> aborted{false};
        void Abort() noexcept
        {
            if (comm && !aborted.exchange(true))
                ncclCommAbort(comm);
        }
        void Check() const
        {
            DNDS_check_throw_info(!aborted.load(), "NCCL communicator was aborted");
            ncclResult_t error;
            CheckNCCL(ncclCommGetAsyncError(comm, &error));
            CheckNCCL(error);
        }
        ~State()
        {
            // NCCL/MPI disposal is controller work, even when a CPU worker
            // observes the device event that retires the numerical accesses.
            if (std::this_thread::get_id() != controller)
                std::terminate();
            if (comm && !aborted.load())
                ncclCommDestroy(comm);
        }
    };
    struct NCCLTransferProvider::State::Binding final : Detail::TransferProviderBinding,
                                                        std::enable_shared_from_this<Binding>
    {
        struct Peer
        {
            size_t sendOffset = 0, receiveOffset = 0, sendBytes = 0, receiveBytes = 0;
        };
        std::shared_ptr<State> owner;
        TransferPayload payload;
        Buffer source, destination, send, receive, packRows, unpackRows, fields;
        std::vector<Peer> peers;
        index packCount = 0, unpackCount = 0, fieldCount = 0;
        bool packed;
        mutable std::atomic<bool> manualActive{false};
        Binding(std::shared_ptr<State> state, const TransferPayload &description, Buffer src, Buffer dst, bool packing)
            : owner(std::move(state)), payload(description), source(std::move(src)), destination(std::move(dst)), packed(packing) {}

        void Prepare()
        {
            const auto extent = payload.schema.Extent();
            peers.resize(payload.route.SendRows().size());
            std::vector<Detail::DevicePackRow> gather, scatter;
            index sendElements = 0, receiveElements = 0;
            auto compile = [&](const auto &rows, const RowLayout &layout, std::vector<Detail::DevicePackRow> &metadata,
                               index &elements, size_t &offset, size_t &bytes)
            {
                const index begin = elements;
                index first = -1;
                for (auto iRow : rows)
                {
                    const auto width = index(layout.RowSize(iRow));
                    DNDS_check_throw_info(width <= INT64_MAX - elements, "NCCL packed extent overflow");
                    const auto base = width ? layout.ElementOffset(iRow, 0) : 0;
                    if (width)
                    {
                        if (first < 0)
                            first = base;
                        DNDS_check_throw_info(packed || ((width == 1 || layout.ComponentStride() == 1) && base == first + elements - begin),
                                              "NCCL Direct requires contiguous rows; select DevicePacked");
                    }
                    metadata.push_back({base, layout.ComponentStride(), elements, width});
                    elements += width;
                }
                offset = CheckedSize::Multiply(size_t(packed ? begin : std::max(index(0), first)), extent);
                bytes = CheckedSize::Multiply(size_t(elements - begin), extent);
            };
            for (index peer = 0; peer < index(peers.size()); ++peer)
            {
                auto &p = peers[peer];
                compile(payload.route.SendRows()[peer], payload.source, gather, sendElements, p.sendOffset, p.sendBytes);
                compile(payload.route.ReceiveRows()[peer], payload.destination, scatter, receiveElements, p.receiveOffset, p.receiveBytes);
            }
            if (!packed)
            {
                size_t next = 0;
                for (const auto &field : payload.schema.Fields())
                {
                    DNDS_check_throw_info(field.offset == next, "NCCL Direct requires a fully described contiguous schema");
                    next = CheckedSize::Add(next, CheckedSize::Multiply(field.count, ElementSchema::ScalarBytes(field.scalar)));
                }
                DNDS_check_throw_info(next == extent, "NCCL Direct cannot transmit omitted POD fields; select DevicePacked");
                return;
            }
            std::vector<Detail::DevicePackField> wordFields;
            for (const auto &field : payload.schema.Fields())
                wordFields.push_back({index(field.offset), index(CheckedSize::Multiply(field.count, ElementSchema::ScalarBytes(field.scalar)))});
            packCount = index(gather.size());
            unpackCount = index(scatter.size());
            fieldCount = index(wordFields.size());
            packRows = Upload(gather, owner->device);
            unpackRows = Upload(scatter, owner->device);
            fields = Upload(wordFields, owner->device);
            const auto sendBytes = CheckedSize::Multiply(size_t(sendElements), extent);
            const auto receiveBytes = CheckedSize::Multiply(size_t(receiveElements), extent);
            DNDS_check_throw_info(sendBytes <= size_t(INT64_MAX) && receiveBytes <= size_t(INT64_MAX), "NCCL packed byte extent exceeds signed indexing");
            send = Buffer::Allocate(sendBytes, owner->device);
            receive = Buffer::Allocate(receiveBytes, owner->device);
        }
        void Launch(ExecutionContext context, const void *input, void *output) const
        {
            DNDS_check_throw_info(payload.route.Valid(), "NCCL row mapping is stale");
            DNDS_check_throw_info(context.device == owner->device, "NCCL execution placement mismatch");
            CheckCUDA(cudaSetDevice(owner->device.ordinal));
            std::lock_guard<std::mutex> lock(owner->submission);
            owner->Check();
            try
            {
                if (packed)
                    Detail::LaunchDevicePacking(context, input, send.LegacyPointer(),
                                                static_cast<const Detail::DevicePackRow *>(packRows.LegacyPointer()), packCount,
                                                static_cast<const Detail::DevicePackField *>(fields.LegacyPointer()), fieldCount, index(payload.schema.Extent()), false);
                const auto *src = static_cast<const uint8_t *>(packed ? send.LegacyPointer() : input);
                auto *dst = static_cast<uint8_t *>(packed ? receive.LegacyPointer() : output);
                CheckNCCL(ncclGroupStart());
                ncclResult_t error = ncclSuccess;
                auto remember = [&](ncclResult_t result)
                { if (error == ncclSuccess) error = result; };
                for (index peer = 0; peer < index(peers.size()); ++peer)
                {
                    const auto &p = peers[peer];
                    if (p.receiveBytes)
                        remember(ncclRecv(dst + p.receiveOffset, p.receiveBytes, ncclUint8, int(peer), owner->comm, ::DNDS::Runtime::CUDAStream(context)));
                    if (p.sendBytes)
                        remember(ncclSend(src + p.sendOffset, p.sendBytes, ncclUint8, int(peer), owner->comm, ::DNDS::Runtime::CUDAStream(context)));
                }
                remember(ncclGroupEnd());
                CheckNCCL(error);
                if (packed)
                    Detail::LaunchDevicePacking(context, receive.LegacyPointer(), output,
                                                static_cast<const Detail::DevicePackRow *>(unpackRows.LegacyPointer()), unpackCount,
                                                static_cast<const Detail::DevicePackField *>(fields.LegacyPointer()), fieldCount, index(payload.schema.Extent()), true);
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("solver NCCL launch failed");
                owner->Abort();
                throw;
            }
        }
        struct Operation final : Detail::TransferProviderOperation
        {
            std::shared_ptr<const Binding> binding;
            Detail::TransferAccess source, destination, send, receive;
            Completion completion;
            std::exception_ptr error;
            bool finished = false, manual = false;
            Operation(std::shared_ptr<const Binding> b, ExecutionContext context, bool scheduled, bool exclusive, bool checked = false)
                : binding(std::move(b)), manual(exclusive)
            {
                DNDS_check_throw_info(context.device == binding->owner->device && binding->payload.route.Valid(), "NCCL execution placement or row mapping mismatch");
                if (manual)
                    DNDS_check_throw_info(!binding->manualActive.exchange(true), "NCCL binding has an active manual operation");
                else
                    DNDS_check_throw_info(!binding->manualActive.load(), "NCCL binding has an active manual operation");
                try
                {
                    auto src = binding->source, dst = binding->destination;
                    source.Acquire(src, context, false, scheduled, checked);
                    destination.Acquire(dst, context, true, scheduled, checked);
                    if (binding->packed)
                    {
                        auto packedSend = binding->send, packedReceive = binding->receive;
                        send.Acquire(packedSend, context, true, scheduled, checked);
                        receive.Acquire(packedReceive, context, true, scheduled, checked);
                    }
                }
                catch (...)
                {
                    source.CancelUnmodified();
                    destination.CancelUnmodified();
                    send.CancelUnmodified();
                    receive.CancelUnmodified();
                    if (manual)
                        binding->manualActive = false;
                    throw;
                }
            }
            void Retire(std::exception_ptr failure) noexcept
            {
                error = failure;
                if (!error)
                    try
                    {
                        binding->owner->Check();
                    }
                    catch (...)
                    {
                        error = std::current_exception();
                        // Managed progress reports this terminal failure on
                        // the controller; an event observer may be a worker.
                    }
                source.pending.Complete(!error);
                destination.pending.Complete(!error);
                send.pending.Complete(!error);
                receive.pending.Complete(!error);
                finished = true;
            }
            void WaitDestinationReady() override { Wait(); }
            void Wait() override
            {
                if (error)
                    std::rethrow_exception(error);
                if (finished)
                    return;
                try
                {
                    while (!completion.Ready())
                    {
                        binding->owner->Check();
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
                    binding->owner->Check();
                    source.Wait();
                    destination.Wait();
                    send.Wait();
                    receive.Wait();
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("solver NCCL completion failed");
                    error = std::current_exception();
                    binding->owner->Abort();
                    try
                    {
                        source.Fail();
                    }
                    catch (...)
                    {
                    }
                    try
                    {
                        destination.Fail();
                    }
                    catch (...)
                    {
                    }
                    try
                    {
                        send.Fail();
                    }
                    catch (...)
                    {
                    }
                    try
                    {
                        receive.Fail();
                    }
                    catch (...)
                    {
                    }
                }
                finished = true;
                if (manual)
                    binding->manualActive = false;
                if (error)
                    std::rethrow_exception(error);
            }
            bool Test() override
            {
                if (error)
                    std::rethrow_exception(error);
                binding->owner->Check();
                if (!finished && !completion.Ready())
                    return false;
                Wait();
                return true;
            }
        };
        std::unique_ptr<Detail::TransferProviderOperation> Begin(ExecutionContext context, bool scheduled) const override
        {
            auto op = std::make_unique<Operation>(shared_from_this(), context, scheduled, true);
            try
            {
                Launch(context, op->source.View<uint8_t>().data, op->destination.View<uint8_t>().data);
                op->completion = Completion::Record(context);
                op->source.Complete(op->completion);
                op->destination.Complete(op->completion);
                if (packed)
                {
                    op->send.Complete(op->completion);
                    op->receive.Complete(op->completion);
                }
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("solver NCCL operation failed before completion");
                auto error = std::current_exception();
                // A failed immediate launch may have submitted packing before
                // producing its completion event. Drain that prefix too before
                // releasing immediate read/write access. Scheduled Fail also
                // retains this rule through its supplied context.
                try
                {
                    Completion::Record(context).Wait();
                }
                catch (...)
                {
                }
                try
                {
                    op->source.Fail();
                }
                catch (...)
                {
                }
                try
                {
                    op->destination.Fail();
                }
                catch (...)
                {
                }
                try
                {
                    op->send.Fail();
                }
                catch (...)
                {
                }
                try
                {
                    op->receive.Fail();
                }
                catch (...)
                {
                }
                manualActive = false;
                std::rethrow_exception(error);
            }
            return op;
        }
        void Enqueue(EagerEngine &engine, const ExecutionStream &stream) const override
        {
            auto pending = std::make_shared<State::Pending>(owner, shared_from_this());
            engine.RegisterProgress(pending); // Before issuing any native work.
            try
            {
                auto op = std::make_shared<Operation>(shared_from_this(), stream.Context(), true, false, engine.ChecksAccess());
                engine.SubmitOperation(stream, [op](ExecutionContext context)
                                       {
                                       op->source.pending.Validate();
                                       op->destination.pending.Validate();
                                       op->binding->Launch(context, op->source.View<uint8_t>().data, op->destination.View<uint8_t>().data); }, [op, pending](std::exception_ptr error) noexcept
                                       {
                                       op->Retire(error);
                                       pending->failure = op->error;
                                       op->binding.reset(); // pending is the controller-side anchor.
                                       pending->retired.store(true, std::memory_order_release); }, packed ? std::vector<ScheduledBufferAccess>{{source, AccessMode::Read}, {destination, AccessMode::Write}, {send, AccessMode::Write}, {receive, AccessMode::Write}} : std::vector<ScheduledBufferAccess>{{source, AccessMode::Read}, {destination, AccessMode::Write}});
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("managed NCCL submission rejected");
                pending->retired.store(true, std::memory_order_release); // No accepted command.
                throw;
            }
        }
    };

    struct NCCLTransferProvider::State::Allreduce final : GlobalReductionBinding, std::enable_shared_from_this<Allreduce>
    {
        std::shared_ptr<State> owner;
        Buffer source, destination;
        size_t count = 0, scalarBytes = 0;
        ncclDataType_t type{};
        ncclRedOp_t operation{};

        StreamEvent Submit(EagerEngine &engine, const ExecutionStream &stream) const override
        {
            const auto context = stream.Context();
            DNDS_check_throw_info(engine.Owns(stream) && context.device == owner->device &&
                                      std::this_thread::get_id() == owner->controller,
                                  "NCCL collective stream or controller differs");
            const bool inPlace = source.SameRange(destination);
            struct Operands
            {
                ScheduledAccess source, destination;
            };
            auto operands = std::make_shared<Operands>();
            auto pending = std::make_shared<Pending>(owner, shared_from_this());
            engine.RegisterProgress(pending);
            try
            {
                std::vector<ScheduledBufferAccess> accesses;
                if (!inPlace)
                {
                    operands->source = source.Reserve(AccessMode::Read, context.device, engine.ChecksAccess());
                    accesses.emplace_back(source, AccessMode::Read);
                }
                const auto mode = inPlace ? AccessMode::ReadWrite : AccessMode::Write;
                operands->destination = destination.Reserve(mode, context.device, engine.ChecksAccess());
                accesses.emplace_back(destination, mode);
                // The pending record pins this binding until controller retirement.
                engine.SubmitOperation(stream, [this, operands, inPlace](ExecutionContext ctx)
                                       {
                    if (!inPlace) operands->source.Validate();
                    operands->destination.Validate();
                    if (!count) return;
                    auto *out = operands->destination.Pointer(scalarBytes, scalarBytes);
                    const auto *in = inPlace ? out : operands->source.Pointer(scalarBytes, scalarBytes);
                    CheckCUDA(cudaSetDevice(ctx.device.ordinal));
                    std::lock_guard<std::mutex> lock(owner->submission);
                    owner->Check();
                    CheckNCCL(ncclAllReduce(in, out, count, type, operation, owner->comm, CUDAStream(ctx))); }, [operands, pending](std::exception_ptr error) noexcept
                                       {
                    operands->source.Complete(!error);
                    operands->destination.Complete(!error);
                    pending->failure = error;
                    pending->retired.store(true, std::memory_order_release); }, std::move(accesses));
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("solver NCCL allreduce submission rejected");
                pending->retired.store(true, std::memory_order_release);
                throw;
            }
            return stream.Record();
        }
    };

    std::shared_ptr<const GlobalReductionBinding> NCCLTransferProvider::BindAllreduceBuffers(
        Buffer src, Buffer dst, const RowLayout &in, const RowLayout &out, ElementScalar scalar, Reduction op) const
    {
        auto result = std::make_shared<State::Allreduce>();
        result->owner = _state;
        const auto comm = _state->control;
        const bool valid = src.GetDevice() == _state->device && dst.GetDevice() == _state->device &&
                           in.Kind() == RowLayoutKind::Table && out.Kind() == RowLayoutKind::Table &&
                           in.Width() == in.Stride() && out.Width() == out.Stride() &&
                           in.Width() == out.Width() && in.Rows() == out.Rows() &&
                           (!src.Overlaps(dst) || src.SameRange(dst)) &&
                           (op == Reduction::Sum || op == Reduction::Min || op == Reduction::Max);
        std::array<uint64_t, 4> descriptor{uint64_t(scalar), uint64_t(op), uint64_t(in.Rows()), uint64_t(in.Width())};
        auto reference = descriptor;
        auto identity = _state->identity;
        MPI_Bcast(reference.data(), int(reference.size()), MPI_UINT64_T, 0, comm);
        MPI_Bcast(&identity, sizeof(identity), MPI_BYTE, 0, comm);
        CommunicationDetail::RequireAll(valid && reference == descriptor && !std::memcmp(&identity, &_state->identity, sizeof(identity)),
                                        comm, "NCCL allreduce requires matching scalar, operation, compact table shapes and provider");
        result->source = std::move(src);
        result->destination = std::move(dst);
        result->count = in.Elements();
        result->scalarBytes = ElementSchema::ScalarBytes(scalar);
        switch (scalar)
        {
        case ElementScalar::I32:
            result->type = ncclInt32;
            break;
        case ElementScalar::I64:
            result->type = ncclInt64;
            break;
        case ElementScalar::F32:
            result->type = ncclFloat32;
            break;
        case ElementScalar::F64:
            result->type = ncclFloat64;
            break;
        default:
            DNDS_check_throw_info(false, "unsupported NCCL allreduce scalar");
        }
        result->operation = op == Reduction::Sum ? ncclSum : op == Reduction::Max ? ncclMax
                                                                                  : ncclMin;
        return result;
    }

    bool NCCLTransferProvider::Available() { return true; }
    void NCCLTransferProvider::CheckAsyncError() const { _state->Check(); }
    void NCCLTransferProvider::Abort() const noexcept { _state->Abort(); }
    std::shared_ptr<NCCLTransferProvider> NCCLTransferProvider::Create(MPI_Comm comm, Runtime::Device device)
    {
        return Create(CommunicationContext::Duplicate(comm), device);
    }
    std::shared_ptr<const CommunicationContext> NCCLTransferProvider::ControlContext() const { return _state->controlOwner; }
    std::shared_ptr<NCCLTransferProvider> NCCLTransferProvider::Create(std::shared_ptr<CommunicationContext> context, Runtime::Device device)
    {
        DNDS_check_throw_info(context, "NCCL setup needs a communication context");
        const auto comm = context->Communicator();
        DNDS_check_throw_info(comm != MPI_COMM_NULL, "NCCL setup needs an MPI communicator");
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        CommunicationDetail::RequireAll(status == cudaSuccess && device.space == MemorySpace::CUDA && device.ordinal >= 0 && device.ordinal < count,
                                        comm, "NCCL setup needs a valid rank-local CUDA device");
        auto state = std::make_shared<State>();
        state->device = device;
        CheckCUDA(cudaSetDevice(device.ordinal));
        state->controlOwner = std::move(context);
        state->control = comm;
        int rank = 0, size = 0;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        ncclUniqueId identifier;
        const auto unique = rank ? ncclSuccess : ncclGetUniqueId(&identifier);
        CommunicationDetail::RequireAll(unique == ncclSuccess, comm, "NCCL communicator identifier failed");
        MPI_Bcast(&identifier, sizeof(identifier), MPI_BYTE, 0, comm);
        state->identity = identifier;
        CheckNCCL(ncclCommInitRank(&state->comm, size, identifier, rank));
        return std::shared_ptr<NCCLTransferProvider>(new NCCLTransferProvider(std::move(state)));
    }
    std::shared_ptr<Detail::TransferProviderBinding> NCCLTransferProvider::Bind(
        const TransferPayload &payload, Buffer source, Buffer destination, std::shared_ptr<const Reducer> reducer,
        TransferTransport transport, const TransferOptions &options, bool layoutsValid) const
    {
        int comparison = MPI_UNEQUAL;
        MPI_Comm_compare(payload.route.Communicator(), _state->control, &comparison);
        // Congruent MPI groups do not prove that ranks selected the same NCCL
        // communicator. Agree its actual bootstrap identity during setup only.
        auto identity = _state->identity;
        MPI_Bcast(&identity, sizeof(identity), MPI_BYTE, 0, payload.route.Communicator());
        const bool matchingProvider = std::memcmp(&identity, &_state->identity, sizeof(identity)) == 0;
        const auto extent = payload.schema.Extent();
        bool valid = layoutsValid && matchingProvider && !_state->aborted.load() && (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) && !options.context && options.tag == 0 &&
                     !reducer && !payload.route.HasCollisions() && payload.route.Valid() &&
                     (transport == TransferTransport::Direct || transport == TransferTransport::DevicePacked) &&
                     source.Identity() && destination.Identity() && source.Identity() != destination.Identity() &&
                     source.GetDevice() == _state->device && destination.GetDevice() == _state->device &&
                     extent && extent <= size_t(INT64_MAX) && !payload.schema.Fields().empty() &&
                     source.Bytes() <= size_t(INT64_MAX) && destination.Bytes() <= size_t(INT64_MAX) &&
                     source.Bytes() / extent == payload.source.Elements() && source.Bytes() % extent == 0 &&
                     destination.Bytes() / extent == payload.destination.Elements() && destination.Bytes() % extent == 0;
        CommunicationDetail::RequireAll(valid, payload.route.Communicator(), "invalid NCCL binding: context/tag, collision/reducer, schema, layout, placement or communicator");
        std::shared_ptr<State::Binding> binding;
        try
        {
            binding = std::make_shared<State::Binding>(_state, payload, std::move(source), std::move(destination), transport == TransferTransport::DevicePacked);
            binding->Prepare();
        }
        catch (...)
        {
            valid = false;
        }
        CommunicationDetail::RequireAll(valid, payload.route.Communicator(), "NCCL workspace preparation failed; Direct requires contiguous rows and a full schema");
        return binding;
    }
#else
    struct NCCLTransferProvider::State
    {
    };
    std::shared_ptr<const GlobalReductionBinding> NCCLTransferProvider::BindAllreduceBuffers(
        Buffer, Buffer, const RowLayout &, const RowLayout &, ElementScalar, Reduction) const
    {
        DNDS_check_throw_info(false, "NCCL provider is unavailable");
        return {};
    }
    bool NCCLTransferProvider::Available() { return false; }
    void NCCLTransferProvider::CheckAsyncError() const { DNDS_check_throw_info(false, "NCCL provider is unavailable"); }
    void NCCLTransferProvider::Abort() const noexcept {}
    std::shared_ptr<NCCLTransferProvider> NCCLTransferProvider::Create(MPI_Comm, Runtime::Device)
    {
        DNDS_check_throw_info(false, "NCCL provider is unavailable; configure DNDS_USE_CUDA and DNDS_USE_NCCL");
        return {};
    }
    std::shared_ptr<NCCLTransferProvider> NCCLTransferProvider::Create(std::shared_ptr<CommunicationContext>, Runtime::Device)
    {
        DNDS_check_throw_info(false, "NCCL provider is unavailable; configure DNDS_USE_CUDA and DNDS_USE_NCCL");
        return {};
    }
    std::shared_ptr<const CommunicationContext> NCCLTransferProvider::ControlContext() const { return {}; }
    std::shared_ptr<Detail::TransferProviderBinding> NCCLTransferProvider::Bind(
        const TransferPayload &, Buffer, Buffer, std::shared_ptr<const Reducer>, TransferTransport, const TransferOptions &, bool) const
    {
        DNDS_check_throw_info(false, "NCCL provider is unavailable");
        return {};
    }
#endif
    NCCLTransferProvider::NCCLTransferProvider(std::shared_ptr<State> state) : _state(std::move(state)) {}
}
