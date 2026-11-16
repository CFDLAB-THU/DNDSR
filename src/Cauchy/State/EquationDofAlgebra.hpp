#pragma once

#include "Cauchy/State/EquationViews.hpp"
#include "DNDS/Algebra/LocalProducts.hpp"
#include "DNDS/Core/OptionalRef.hpp"
#include <optional>

namespace DNDS::Cauchy
{
    /// Placement-copy allocation policy. Page locking is explicit, accounted
    /// and valid only for a host destination; there is no pageable fallback.
    struct EquationSnapshotOptions
    {
        bool pinnedHost = false;
    };
    /// Semantic adapter over DNDS pointwise algebra. Both synchronous CPU and
    /// queued backend operations touch owned values only, with no halo/collective or global
    /// repacking. All output reservations precede numerical writes. Binary
    /// operands must have distinct allocations; Scale is explicitly in-place.
    /// Caller serializes state rebinding. Queued operations retain numeric
    /// allocation handles; MPI resource ownership stays with the controller.
    struct EquationDofOperations
    {
        using State = EquationDofs;
        static State NewVector(const State &prototype)
        {
            prototype.Validate();
            // Allocation preserves the prototype backend independently of
            // arithmetic dispatch. This creates uninitialized scratch, with
            // no copy, host fallback or change to the equation grouping.
            const auto device = prototype.Device();
            Algebra::DofBundle fields(prototype.Space());
            for (index iField = 0; iField < prototype.PhysicalFieldNumber(); ++iField)
            {
                const auto &field = prototype.PhysicalField(iField);
                fields.Add(prototype.Fields().Name(iField), Algebra::DofField<real>(prototype.Space(),
                                                                                    Runtime::Array<real>(field.Owned().Layout(), device), Runtime::Array<real>(field.Ghost().Layout(), device)));
            }
            return prototype.RebindFields(std::move(fields));
        }
        /// Explicit setup/I/O allocation on another device. Preserves physical
        /// layouts, logical bindings and row identity; neither owned nor ghost
        /// values are initialized. This is never an arithmetic fallback.
        static State NewVectorOn(const State &prototype, Runtime::Device device);
        /// Synchronous owned-value snapshot/placement transfer. Host snapshots
        /// are deep copies even for CPU input. Ghosts are deliberately not copied
        /// or marked fresh. The caller must first order/drain source producers;
        /// context identifies the device stream for a CUDA transfer.
        static State CopyOwnedTo(const State &source, Runtime::Device device, Runtime::ExecutionContext context,
                                 EquationSnapshotOptions options = {});
        static void CheckCompatible(const State &prototype, const State &state)
        {
            prototype.RequireCompatible(state);
            DNDS_check_throw_info(state.Device() == Runtime::Device::Host(), "CPU equation algebra has no device fallback");
        }
        static void RequireDistinct(const State &x, const State &y)
        {
            for (index iField = 0; iField < x.PhysicalFieldNumber(); ++iField)
                for (index jField = 0; jField < y.PhysicalFieldNumber(); ++jField)
                {
                    const auto &a = x.PhysicalField(iField), &b = y.PhysicalField(jField);
                    for (const auto *input : {&a.Owned(), &a.Ghost()})
                        for (const auto *output : {&b.Owned(), &b.Ghost()})
                            DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), output->Storage()), "equation algebra operands alias");
                }
        }
        struct Binding
        {
            State snapshot;
            std::vector<uint64_t> versions;
            explicit Binding(const State &state) : snapshot(state)
            {
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    versions.push_back(state.PhysicalField(iField).Owned().Version());
            }
            void Check(const State &state, bool unchangedOwned = false) const
            {
                snapshot.RequireCompatible(state);
                DNDS_check_throw_info(snapshot.PhysicalFieldNumber() == state.PhysicalFieldNumber(), "time callback changed physical equation grouping");
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto before = snapshot.LocateBlock(iBlock), after = state.LocateBlock(iBlock);
                    const auto &a = snapshot.PhysicalField(before.physicalField), &b = state.PhysicalField(after.physicalField);
                    DNDS_check_throw_info(before.firstComponent == after.firstComponent &&
                                              a.Owned().SharedStorage() == b.Owned().SharedStorage() && a.Ghost().SharedStorage() == b.Ghost().SharedStorage(),
                                          "time callback rebound an equation block");
                    DNDS_check_throw_info(!unchangedOwned || versions[before.physicalField] == b.Owned().Version(),
                                          "time callback modified an owned equation input/source");
                }
            }
        };

    private:
        static void SubmitApply(Runtime::EagerEngine &, const Runtime::ExecutionStream &, Algebra::PointwiseOperation,
                                State &, OptionalRef<const State>, OptionalRef<const State>, real, real);
        static bool SameOwnedBinding(const State &a, const State &b)
        {
            for (index iBlock = 0; iBlock < a.BlockNumber(); ++iBlock)
            {
                const auto x = a.LocateBlock(iBlock), y = b.LocateBlock(iBlock);
                if (x.firstComponent != y.firstComponent ||
                    a.PhysicalField(x.physicalField).Owned().SharedStorage() != b.PhysicalField(y.physicalField).Owned().SharedStorage())
                    return false;
            }
            return true;
        }
        static void Apply(Algebra::PointwiseOperation operation, State &output, const State *x, const State *y, real a, real b = 0)
        {
            CheckCompatible(output, output);
            if (x)
            {
                CheckCompatible(output, *x);
                if (operation == Algebra::PointwiseOperation::Copy && SameOwnedBinding(output, *x))
                    return;
                RequireDistinct(output, *x);
            }
            if (y)
            {
                CheckCompatible(output, *y);
                RequireDistinct(output, *y);
            }
            std::optional<State::ReadScope> readX, readY;
            if (x)
                readX.emplace(x->ReadOwned());
            if (y)
                readY.emplace(y->ReadOwned());
            auto write = output.WriteOwned();
            const EquationViewLayout<real> destination(output, write);
            if (readY)
            {
                const EquationViewLayout<const real> left(*x, *readX), right(*y, *readY);
                const auto kernel = [&](auto out, auto u, auto v)
                { Algebra::ApplyHostRows(operation, out, u, v, a, b); };
                DispatchCPUEquationLayouts(kernel, destination, left, right);
            }
            else if (readX)
            {
                const EquationViewLayout<const real> source(*x, *readX);
                const auto kernel = [&](auto out, auto u)
                { Algebra::ApplyHostRows(operation, out, u, EquationBlockedView<const real>{}, a, b); };
                DispatchCPUEquationLayouts(kernel, destination, source);
            }
            else
            {
                const auto kernel = [&](auto out)
                { Algebra::ApplyHostRows(operation, out, EquationBlockedView<const real>{}, EquationBlockedView<const real>{}, a, b); };
                DispatchCPUEquationLayouts(kernel, destination);
            }
        }

    public:
        /// Synchronous publication using scopes already reserved across all
        /// participating domains. Launches through the same engine lane on
        /// inline CPU, worker CPU and CUDA, then drains before returning.
        /// Scopes must belong to x/out and were acquired on stream.Context()
        /// after prior queued work completed. They retain all allocations until
        /// completion; this function neither rebinds nor reacquires arrays.
        /// Intended for accepted-state publication, not iterative arithmetic.
        static void CopyReserved(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                 State &out, const State &x, State::WriteScope &, State::ReadScope &);
        /// Queued owned-only algebra. All physical allocations are reserved in
        /// one operation; segmented/general CPU layouts keep logical component
        /// indexing. CUDA currently requires the declared combined-AoS profile.
        /// Inputs and output are distinct except the explicit Scale/Axpy target.
        static void Copy(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, State &out, const State &x)
        {
            SubmitApply(engine, stream, Algebra::PointwiseOperation::Copy, out, std::cref(x), {}, 0, 0);
        }
        static void Fill(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, State &out, real value)
        {
            SubmitApply(engine, stream, Algebra::PointwiseOperation::Fill, out, {}, {}, value, 0);
        }
        static void Scale(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, State &out, real value)
        {
            SubmitApply(engine, stream, Algebra::PointwiseOperation::Scale, out, {}, {}, value, 0);
        }
        static void Axpy(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, State &out, real value, const State &x)
        {
            SubmitApply(engine, stream, Algebra::PointwiseOperation::Axpy, out, std::cref(x), {}, value, 0);
        }
        static void LinearCombination(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                                      State &out, real a, const State &x, real b, const State &y)
        {
            SubmitApply(engine, stream, Algebra::PointwiseOperation::Combination, out, std::cref(x), std::cref(y), a, b);
        }
        static void Copy(State &out, const State &x) { Apply(Algebra::PointwiseOperation::Copy, out, &x, nullptr, 0); }
        static void CopyGuess(State &out, const State &x) { Copy(out, x); }
        static void Fill(State &out, real value) { Apply(Algebra::PointwiseOperation::Fill, out, nullptr, nullptr, value); }
        static void Scale(State &out, real value) { Apply(Algebra::PointwiseOperation::Scale, out, nullptr, nullptr, value); }
        static void Axpy(State &out, real value, const State &x) { Apply(Algebra::PointwiseOperation::Axpy, out, &x, nullptr, value); }
        static void LinearCombination(State &out, real a, const State &x, real b, const State &y)
        {
            Apply(Algebra::PointwiseOperation::Combination, out, &x, &y, a, b);
        }
        /// Owned local dot-product batch in logical row/component order. The
        /// caller owns its communicator/reduction and distributed error policy.
        /// Descriptors/scopes are resolved once; no per-element schema checks,
        /// halo access or global payload flattening occurs.
        static std::vector<real> LocalProducts(const State &x, const std::vector<const State *> &probes,
                                               const std::vector<real> &inverseScales = {})
        {
            CheckCompatible(x, x);
            DNDS_check_throw_info(inverseScales.empty() || index(inverseScales.size()) == x.ComponentNumber(), "equation product scale width differs");
            for (const auto *probe : probes)
            {
                DNDS_check_throw_info(probe, "null equation product probe");
                CheckCompatible(x, *probe);
            }
            const auto read = x.ReadOwned();
            const EquationViewLayout<const real> values(x, read);
            std::vector<State::ReadScope> readers;
            readers.reserve(probes.size());
            for (const auto *probe : probes)
                readers.push_back(probe->ReadOwned());
            std::vector<EquationViewLayout<const real>> inputs;
            inputs.reserve(probes.size());
            for (index k = 0; k < index(probes.size()); ++k)
                inputs.emplace_back(*probes[k], readers[k]);
            std::vector<real> result(probes.size(), 0);
            OMP::ForRows(0, index(probes.size()), [&](index k)
                         {
                const auto kernel = [&](auto u, auto v)
                { result[k] = Algebra::Detail::AccumulateLocalProduct(u, v, u.RowCount(), u.ComponentNumber(), inverseScales); };
                DispatchCPUEquationLayouts(kernel, values, inputs[k]); });
            return result;
        }
    };
}
