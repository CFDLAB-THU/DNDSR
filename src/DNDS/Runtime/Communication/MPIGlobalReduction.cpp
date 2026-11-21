#include "DNDS/Runtime/Communication/MPIGlobalReduction.hpp"
#include "DNDS/Runtime/Storage/BufferAlias.hpp"
#include <atomic>
#include <thread>

namespace DNDS::Runtime
{
    namespace
    {
        struct Operands
        {
            ScheduledAccess src, dst;
        };

        /// The controller alone posts/tests the request. A worker observes only
        /// the published result. The CPU stream imports this private event before
        /// its tracked retirement command, so even a one-worker engine can run
        /// independent producer work while a remote rank is delayed.
        struct Pending final : ExecutionProgress, StreamEvent::State
        {
            std::shared_ptr<CommunicationContext> communication;
            std::shared_ptr<Operands> operands;
            StreamEvent producer;
            MPI_Request request = MPI_REQUEST_NULL;
            MPI_Op op;
            MPI_Datatype type;
            int count;
            size_t bytes;
            bool inPlace, armed = false, polling = false;
            const std::thread::id controller = std::this_thread::get_id();
            std::atomic<bool> done{false};

            void Finish()
            {
                producer = {};
                communication.reset(); // Release the MPI owner on its controller.
                done.store(true, std::memory_order_release);
                NotifyCompletion();
            }
            bool Poll() override
            {
                if (done.load(std::memory_order_acquire))
                    return true;
                if (std::this_thread::get_id() != controller || !armed || polling)
                    return false;
                // An inline producer can progress dependencies on this same executor.
                polling = true;
                try
                {
                    const auto comm = communication->Communicator();
                    if (request == MPI_REQUEST_NULL && producer.Ready())
                    {
                        auto *out = operands->dst.Pointer(bytes, bytes);
                        const void *in = inPlace ? MPI_IN_PLACE : operands->src.Pointer(bytes, bytes);
                        if (count && MPI_Iallreduce(in, out, count, type, op, comm, &request) != MPI_SUCCESS)
                            MPI::Abort(comm, "streamed MPI_Iallreduce failed to post");
                        producer = {};
                        if (request == MPI_REQUEST_NULL)
                            Finish();
                    }
                    if (request != MPI_REQUEST_NULL)
                    {
                        int complete = 0;
                        if (MPI_Test(&request, &complete, MPI_STATUS_IGNORE) != MPI_SUCCESS)
                            MPI::Abort(comm, "streamed MPI_Iallreduce failed during progress");
                        if (complete)
                            Finish();
                    }
                }
                catch (...)
                {
                    MPI::AbortCaught(communication->Communicator(), "MPI allreduce producer or request failed");
                }
                polling = false;
                return done.load(std::memory_order_acquire);
            }
            bool Ready() const override { return const_cast<Pending *>(this)->Poll(); }
            void Wait() const override
            {
                DNDS_check_throw_info(done.load(std::memory_order_acquire) || std::this_thread::get_id() == controller,
                                      "MPI allreduce progress requires its submitting thread");
                while (!Ready())
                    std::this_thread::yield();
            }
        };

        struct Binding final : GlobalReductionBinding
        {
            std::shared_ptr<CommunicationContext> communication;
            Buffer src, dst;
            MPI_Datatype type;
            MPI_Op op;
            int count;
            size_t bytes;
            const std::thread::id controller = std::this_thread::get_id();

            StreamEvent Submit(EagerEngine &engine, const ExecutionStream &stream) const override
            {
                DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == Device::Host() &&
                                          std::this_thread::get_id() == controller,
                                      "MPI allreduce needs a CPU stream on its binding controller");
                auto pending = std::make_shared<Pending>();
                auto operands = std::make_shared<Operands>();
                const bool inPlace = Detail::SameBufferOperand(src, dst);
                const auto mode = inPlace ? AccessMode::ReadWrite : AccessMode::Write;
                std::vector<ScheduledBufferAccess> accesses{{dst, mode}};
                if (!inPlace)
                {
                    operands->src = src.Reserve(AccessMode::Read, Device::Host(), engine.ChecksAccess());
                    accesses.emplace_back(src, AccessMode::Read);
                }
                operands->dst = dst.Reserve(mode, Device::Host(), engine.ChecksAccess());
                pending->communication = communication;
                pending->operands = operands;
                pending->inPlace = inPlace;
                pending->op = op;
                pending->type = type;
                pending->count = count;
                pending->bytes = bytes;
                pending->producer = stream.Record();
                engine.RegisterProgress(pending);
                try
                {
                    stream.WaitFor(StreamEvent(pending));
                    // Reserve the tracked access before allowing MPI to post.
                    // Completion of this lane prefix publishes the mutation.
                    engine.SubmitOperation(stream, [](ExecutionContext) {}, [operands](std::exception_ptr error) noexcept
                                           {
                        operands->src.Complete(!error);
                        operands->dst.Complete(!error); }, std::move(accesses));
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("MPI allreduce submission rejected");
                    pending->Finish();
                    throw;
                }
                pending->armed = true;
                pending->Poll();
                return stream.Record();
            }
        };
    }

    std::shared_ptr<const GlobalReductionBinding> Detail::BindMPIAllreduceBuffers(
        std::shared_ptr<CommunicationContext> context, Buffer src, Buffer dst, const RowLayout &in, const RowLayout &out, ElementScalar scalar, Reduction op)
    {
        const auto comm = context->Communicator();
        const bool valid = src.Identity() && dst.Identity() && src.GetDevice() == Device::Host() && dst.GetDevice() == Device::Host() &&
                           in.Kind() == RowLayoutKind::Table && out.Kind() == RowLayoutKind::Table &&
                           in.Stride() == in.Width() && out.Stride() == out.Width() && in.Width() == out.Width() && in.Rows() == out.Rows() &&
                           in.Elements() <= INT_MAX && (IndependentBuffers(src, dst) || SameBufferOperand(src, dst)) &&
                           (op == Reduction::Sum || op == Reduction::Min || op == Reduction::Max);
        std::array<uint64_t, 5> descriptor{uint64_t(scalar), uint64_t(op), uint64_t(in.Rows()), uint64_t(in.Width()), uint64_t(Detail::SameBufferOperand(src, dst))};
        auto reference = descriptor;
        MPI_Bcast(reference.data(), int(reference.size()), MPI_UINT64_T, 0, comm);
        CommunicationDetail::RequireAll(valid && reference == descriptor, comm, "MPI allreduce requires matching compact host tables, alias mode, scalar and operation");
        auto binding = std::make_shared<Binding>();
        binding->communication = std::move(context);
        binding->src = std::move(src);
        binding->dst = std::move(dst);
        binding->count = int(in.Elements());
        binding->bytes = ElementSchema::ScalarBytes(scalar);
        binding->op = op == Reduction::Sum ? MPI_SUM : op == Reduction::Min ? MPI_MIN
                                                                            : MPI_MAX;
        switch (scalar)
        {
        case ElementScalar::F32:
            binding->type = MPI_FLOAT;
            break;
        case ElementScalar::F64:
            binding->type = MPI_DOUBLE;
            break;
        case ElementScalar::I32:
            binding->type = MPI_INT32_T;
            break;
        case ElementScalar::I64:
            binding->type = MPI_INT64_T;
            break;
        default:
            DNDS_check_throw_info(false, "unsupported MPI reduction scalar");
        }
        return binding;
    }
}
