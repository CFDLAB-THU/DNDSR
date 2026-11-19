#pragma once

#include "Cauchy/State/EquationDofAlgebra.hpp"

namespace DNDS::Cauchy
{
    /** @brief Instance-bound owned algebra for method and hierarchy controllers.
     * @details With a lane, operations enqueue the ordinary DNDS array algebra
     * on CPU-inline, CPU-worker or CUDA; no field staging, MPI, or implicit
     * synchronization occurs. Callers order dependent operators on that lane
     * and drain before host observation or synchronous I/O. The engine owns
     * pending operands, while this adapter retains only a weak stream handle.
     * An unbound adapter preserves standalone synchronous CPU use and rejects
     * device storage; it never creates a hidden execution engine.
     */
    class QueuedEquationOperations : public EquationDofOperations
    {
        std::optional<Runtime::ExecutionStream> _stream;
        template <class Queue, class Inline>
        void Apply(Queue queue, Inline run) const
        {
            if (_stream)
            {
                auto engine = _stream->LockEngine();
                queue(engine, *_stream);
            }
            else
                run();
        }

    public:
        QueuedEquationOperations() = default;
        explicit QueuedEquationOperations(Runtime::ExecutionStream stream) : _stream(std::move(stream)) {}
        using EquationDofOperations::NewVector;
        /// Give time-method storage a stable allocation label at setup.
        static State NewVector(const State &prototype, const char *name)
        {
            auto field = EquationDofOperations::NewVector(prototype);
            field.Fields().SetName(std::string("Cauchy/time/") + name, DNDS_ALLOCATION_SITE());
            return field;
        }
        /// Binding identity and submitted-write epochs, not only completed
        /// host-visible versions. A method may freeze resOther immediately
        /// after enqueueing its producer: completion of that already-declared
        /// write is not a subsequent mutation. Another submitted writer is.
        struct Binding
        {
            EquationDofOperations::Binding storage;
            std::vector<uint64_t> epochs;
            explicit Binding(const State &state) : storage(state)
            {
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    epochs.push_back(state.PhysicalField(iField).Owned().MutationEpoch());
            }
            void Check(const State &state, bool unchangedOwned = false) const
            {
                storage.Check(state);
                if (unchangedOwned)
                    for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                        DNDS_check_throw_info(epochs[iField] == state.PhysicalField(iField).Owned().MutationEpoch(),
                                              "time/FAS callback submitted a write to its immutable owned input");
            }
        };
        static void CheckCompatible(const State &a, const State &b) { a.RequireCompatible(b); }
        void Copy(State &out, const State &x) const
        {
            Apply([&](auto &engine, const auto &lane)
                  { EquationDofOperations::Copy(engine, lane, out, x); },
                  [&]
                  { EquationDofOperations::Copy(out, x); });
        }
        void CopyGuess(State &out, const State &x) const { Copy(out, x); }
        void Fill(State &out, real value) const
        {
            Apply([&](auto &engine, const auto &lane)
                  { EquationDofOperations::Fill(engine, lane, out, value); },
                  [&]
                  { EquationDofOperations::Fill(out, value); });
        }
        void Scale(State &out, real value) const
        {
            Apply([&](auto &engine, const auto &lane)
                  { EquationDofOperations::Scale(engine, lane, out, value); },
                  [&]
                  { EquationDofOperations::Scale(out, value); });
        }
        void Axpy(State &out, real value, const State &x) const
        {
            Apply([&](auto &engine, const auto &lane)
                  { EquationDofOperations::Axpy(engine, lane, out, value, x); },
                  [&]
                  { EquationDofOperations::Axpy(out, value, x); });
        }
        void LinearCombination(State &out, real a, const State &x, real b, const State &y) const
        {
            Apply([&](auto &engine, const auto &lane)
                  { EquationDofOperations::LinearCombination(engine, lane, out, a, x, b, y); },
                  [&]
                  { EquationDofOperations::LinearCombination(out, a, x, b, y); });
        }
        /// Physical-time arithmetic on this lane retains queued operand leases.
        void PhysicalCopyScale(State &out, real a, const State &x) const
        {
            if (out.Schema().AccumulationKind() == PhysicalAccumulationKind::Identity)
            {
                Copy(out, x);
                Scale(out, a);
            }
            else
                Apply([&](auto &e, const auto &s)
                      { SubmitPhysical(e, s, PhysicalOperation::CopyScale, out, x, {}, {}, a, 0); },
                      [&]
                      { EquationDofOperations::PhysicalCopyScale(out, a, x); });
        }
        void PhysicalLinearCombination(State &out, real a, const State &x, real b, const State &y) const
        {
            if (out.Schema().AccumulationKind() == PhysicalAccumulationKind::Identity)
                LinearCombination(out, a, x, b, y);
            else
                Apply([&](auto &e, const auto &s)
                      { SubmitPhysical(e, s, PhysicalOperation::Combination, out, x, std::cref(y), {}, a, b); },
                      [&]
                      { EquationDofOperations::PhysicalLinearCombination(out, a, x, b, y); });
        }
        void PhysicalStateAndRate(State &out, real a, const State &x, real b, const State &rate) const
        {
            if (out.Schema().AccumulationKind() == PhysicalAccumulationKind::Identity)
                LinearCombination(out, a, x, b, rate);
            else
                Apply([&](auto &e, const auto &s)
                      { SubmitPhysical(e, s, PhysicalOperation::StateAndRate, out, x, std::cref(rate), {}, a, b); },
                      [&]
                      { EquationDofOperations::PhysicalStateAndRate(out, a, x, b, rate); });
        }
        void PhysicalAxpy(State &out, real a, const State &x) const
        {
            if (out.Schema().AccumulationKind() == PhysicalAccumulationKind::Identity)
                Axpy(out, a, x);
            else
                Apply([&](auto &e, const auto &s)
                      { SubmitPhysical(e, s, PhysicalOperation::Axpy, out, x, {}, {}, a, 0); },
                      [&]
                      { EquationDofOperations::PhysicalAxpy(out, a, x); });
        }
        void AssembleTimeResidual(State &out, const State &constant, const State &state,
                                  const State &rate, real m, real a) const
        {
            if (out.Schema().AccumulationKind() == PhysicalAccumulationKind::Identity)
            {
                LinearCombination(out, 1, constant, -m, state);
                Axpy(out, a, rate);
            }
            else
                Apply([&](auto &e, const auto &s)
                      { SubmitPhysical(e, s, PhysicalOperation::TimeResidual, out, constant, std::cref(state), std::cref(rate), m, a); },
                      [&]
                      { EquationDofOperations::AssembleTimeResidual(out, constant, state, rate, m, a); });
        }
        /// Explicit host observation boundary, also progressing host MPI.
        void Synchronize() const
        {
            if (_stream)
                _stream->LockEngine().Synchronize();
        }
    };
}
