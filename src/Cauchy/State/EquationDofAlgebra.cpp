#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/CPU/EquationDofAlgebraKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/State/CUDA/FieldKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    EquationDofs EquationDofOperations::NewVectorOn(const State &prototype, R::Device device)
    {
        prototype.Validate();
        if (device == prototype.Device())
            return NewVector(prototype);
        Algebra::DofBundle fields(prototype.Space());
        std::vector<EquationDofBinding> bindings;
        for (index iField = 0; iField < prototype.PhysicalFieldNumber(); ++iField)
        {
            const auto &field = prototype.PhysicalField(iField);
            fields.Add(prototype.Fields().Name(iField), Algebra::DofField<real>(prototype.Space(),
                                                                                R::Array<real>(field.Owned().Layout(), device), R::Array<real>(field.Ghost().Layout(), device)));
        }
        for (index iBlock = 0; iBlock < prototype.BlockNumber(); ++iBlock)
        {
            const auto location = prototype.LocateBlock(iBlock);
            bindings.push_back({prototype.Schema().blocks[iBlock].name,
                                prototype.Fields().Name(location.physicalField), location.firstComponent});
        }
        return {prototype.Schema(), std::move(fields), std::move(bindings)};
    }
    EquationDofs EquationDofOperations::CopyOwnedTo(const State &source, R::Device device, R::ExecutionContext context,
                                                    EquationSnapshotOptions options)
    {
        source.Validate();
        const auto executionDevice = device == R::Device::Host() ? source.Device() : device;
        DNDS_check_throw_info(context.device == executionDevice, "equation placement copy context differs");
        DNDS_check_throw_info(!options.pinnedHost || device == R::Device::Host(), "pinned snapshot requires a host destination");
        const auto allocate = [&]() -> State
        {
            if (!options.pinnedHost)
                return NewVectorOn(source, device);
            Algebra::DofBundle fields(source.Space());
            std::vector<EquationDofBinding> bindings;
            for (index iField = 0; iField < source.PhysicalFieldNumber(); ++iField)
            {
                const auto &field = source.PhysicalField(iField);
                const auto layout = field.Owned().Layout();
                auto owned = R::Array<real>::FromStorage(layout,
                                                         R::Buffer::AllocatePinnedHost(CheckedSize::Multiply(layout.Elements(), sizeof(real))));
                // Archives consume owned rows only. Keep an ordinary uninitialized
                // ghost allocation to preserve the public DofSpace/layout contract.
                fields.Add(source.Fields().Name(iField), Algebra::DofField<real>(source.Space(),
                                                                                 std::move(owned), R::Array<real>(field.Ghost().Layout(), device)));
            }
            for (index iBlock = 0; iBlock < source.BlockNumber(); ++iBlock)
            {
                const auto location = source.LocateBlock(iBlock);
                bindings.push_back({source.Schema().blocks[iBlock].name,
                                    source.Fields().Name(location.physicalField), location.firstComponent});
            }
            return {source.Schema(), std::move(fields), std::move(bindings)};
        };
        auto result = allocate();
        std::vector<R::CopyOperation> copies;
        copies.reserve(size_t(source.PhysicalFieldNumber()));
        for (index iField = 0; iField < source.PhysicalFieldNumber(); ++iField)
            copies.push_back(result.PhysicalField(iField).Owned().CopyFrom(source.PhysicalField(iField).Owned(), context));
        // Enqueue every physical field on the copy lane before observing it.
        // CopyOperation owns both leases until completion; a caller cannot
        // mutate a source while its D2H copy is still in flight.
        for (auto &copy : copies)
            copy.Wait();
        return result;
    }
    void EquationDofOperations::CopyReserved(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                             State &out, const State &x, State::WriteScope &write, State::ReadScope &read)
    {
        out.RequireCompatible(x);
        RequireDistinct(out, x);
        DNDS_check_throw_info(engine.Owns(stream) && out.Device() == stream.Context().device && x.Device() == out.Device(),
                              "reserved equation copy device or engine differs");
        if (out.Device() != R::Device::Host())
        {
            DNDS_check_throw_info(out.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table &&
                                      x.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table,
                                  "CUDA reserved copy requires combined AoS");
        }
        const EquationViewLayout<real> target(out, write);
        const EquationViewLayout<const real> source(x, read);
        const auto wholeOutput = write.Block(0).parent;
        const auto wholeInput = read.Block(0).owned.parent;
        // The caller retains the immediate scopes throughout this synchronous
        // operation. Descriptor buffers are captured by value for worker launch;
        // no bare pointer escapes into an unobserved operation record.
        engine.Submit(stream, [target, source, wholeOutput, wholeInput](R::ExecutionContext context)
                      {
            if (context.device == R::Device::Host())
                DispatchCPUEquationLayouts([](auto y, auto x)
                { Algebra::ApplyHostRows(Algebra::PointwiseOperation::Copy, y, x, EquationBlockedView<const real>{}); }, target, source);
            else
                Algebra::ApplyRowViews(context, Algebra::PointwiseOperation::Copy, wholeOutput.DynamicView(), wholeInput.DynamicView()); });
        engine.Synchronize();
        const auto completed = R::Completion::FromExternalCompletion(stream.Context());
        write.Complete(completed);
        read.Complete(completed);
    }
    namespace
    {
        using ReadRows = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
        using WriteRows = R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>;
        struct ReadBlocks
        {
            const EquationDofs &state;
            SpanView<const ReadRows> views;
            index first;
            EquationBlockReadView Block(index iBlock) const
            {
                const auto location = state.LocateBlock(iBlock);
                return {{views[first + location.physicalField], location.firstComponent, location.componentNumber},
                        {{}, location.firstComponent, location.componentNumber}};
            }
        };
        struct WriteBlocks
        {
            const EquationDofs &state;
            SpanView<const WriteRows> views;
            EquationBlockArrayView<real> Block(index iBlock) const
            {
                const auto location = state.LocateBlock(iBlock);
                return {views[location.physicalField], location.firstComponent, location.componentNumber};
            }
        };
    }
    void EquationDofOperations::SubmitApply(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                            Algebra::PointwiseOperation operation, State &out,
                                            OptionalRef<const State> x, OptionalRef<const State> y, real a, real b)
    {
        out.Validate();
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == out.Device(),
                              "equation algebra stream or placement differs");
        const auto validate = [&](const State &input)
        {
            out.RequireCompatible(input);
            DNDS_check_throw_info(input.Device() == out.Device(), "equation algebra operand placement differs");
        };
        if (x)
            validate(*x);
        if (y)
            validate(*y);
        if (operation == Algebra::PointwiseOperation::Copy && x && SameOwnedBinding(out, *x))
            return;
        if (x)
            RequireDistinct(out, *x);
        if (y)
            RequireDistinct(out, *y);
        const auto requireCombined = [](const State &state)
        {
            const auto &field = state.CoupledField();
            DNDS_check_throw_info(field.Owned().Layout().Kind() == R::RowLayoutKind::Table,
                                  "CUDA equation algebra currently requires combined AoS");
        };
        if (out.Device() != R::Device::Host())
        {
            requireCombined(out);
            if (x)
                requireCombined(*x);
            if (y)
                requireCombined(*y);
        }
        std::optional<State> left, right;
        std::vector<R::ArrayOperand<real, false, R::UniformTableRows>> reads;
        std::vector<R::ArrayOperand<real, true, R::UniformTableRows>> writes;
        if (x)
        {
            left = x->NumericAlias();
            for (index iField = 0; iField < x->PhysicalFieldNumber(); ++iField)
                reads.push_back(R::ReadAs<R::UniformTableRows>(x->PhysicalField(iField).Owned()));
        }
        const index firstRight = index(reads.size());
        if (y)
        {
            right = y->NumericAlias();
            for (index iField = 0; iField < y->PhysicalFieldNumber(); ++iField)
                reads.push_back(R::ReadAs<R::UniformTableRows>(y->PhysicalField(iField).Owned()));
        }
        for (index iField = 0; iField < out.PhysicalFieldNumber(); ++iField)
        {
            auto &array = out.PhysicalField(iField).Owned();
            writes.push_back(operation == Algebra::PointwiseOperation::Scale || operation == Algebra::PointwiseOperation::Axpy
                                 ? R::ReadWriteAs<R::UniformTableRows>(array)
                                 : R::WriteAs<R::UniformTableRows>(array));
        }
        R::SubmitArrayBatch(engine, stream, [output = out.NumericAlias(), left, right, firstRight, operation, a, b](R::ExecutionContext context, auto inputs, auto outputs)
                            {
            if (context.device != R::Device::Host())
            {
                const R::ArrayView<const real> u = left ? inputs[0].DynamicView() : R::ArrayView<const real>{};
                const R::ArrayView<const real> v = right ? inputs[firstRight].DynamicView() : R::ArrayView<const real>{};
                Algebra::ApplyRowViews(context, operation, outputs[0].DynamicView(), u, v, a, b);
                return;
            }
            const EquationViewLayout<real> destination(output, WriteBlocks{output, outputs});
            if (right)
            {
                const EquationViewLayout<const real> u(*left, ReadBlocks{*left, inputs, 0});
                const EquationViewLayout<const real> v(*right, ReadBlocks{*right, inputs, firstRight});
                DispatchCPUEquationLayouts([&](auto out, auto x, auto y) { Algebra::ApplyHostRows(operation, out, x, y, a, b); }, destination, u, v);
            }
            else if (left)
            {
                const EquationViewLayout<const real> u(*left, ReadBlocks{*left, inputs, 0});
                DispatchCPUEquationLayouts([&](auto out, auto x) { Algebra::ApplyHostRows(operation, out, x, EquationBlockedView<const real>{}, a, b); }, destination, u);
            }
            else
                DispatchCPUEquationLayouts([&](auto out) { Algebra::ApplyHostRows(operation, out, EquationBlockedView<const real>{}, EquationBlockedView<const real>{}, a, b); }, destination); }, std::move(reads), std::move(writes));
    }

    void EquationDofOperations::SubmitPhysical(R::EagerEngine &engine, const R::ExecutionStream &lane,
                                               PhysicalOperation op, State &out, const State &x,
                                               OptionalRef<const State> y, OptionalRef<const State> z, real a, real b)
    {
        DNDS_check_throw_info(engine.Owns(lane) && out.Device() == lane.Context().device &&
                                  out.Schema().blocks.front().components.size() == 4,
                              "invalid ACM accumulation binding");
        std::vector<R::ArrayOperand<real, false, R::UniformTableRows>> reads;
        std::vector<R::ArrayOperand<real, true, R::UniformTableRows>> writes;
        std::vector<State> states;
        std::array<index, 3> first{};
        auto checkLayout = [&](const State &state)
        {
            if (lane.Context().device != R::Device::Host())
                DNDS_check_throw_info(state.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table, "ACM CUDA requires combined AoS");
        };
        auto bind = [&](const State &input)
        {
            out.RequireCompatible(input);
            RequireDistinct(out, input);
            checkLayout(input);
            first[states.size()] = index(reads.size());
            states.emplace_back(input.NumericAlias());
            for (index f = 0; f < input.PhysicalFieldNumber(); ++f)
                reads.push_back(R::ReadAs<R::UniformTableRows>(input.PhysicalField(f).Owned()));
        };
        checkLayout(out);
        bind(x);
        if (y)
            bind(*y);
        if (z)
            bind(*z);
        for (index f = 0; f < out.PhysicalFieldNumber(); ++f)
        {
            auto &v = out.PhysicalField(f).Owned();
            writes.push_back(op == PhysicalOperation::Axpy ? R::ReadWriteAs<R::UniformTableRows>(v) : R::WriteAs<R::UniformTableRows>(v));
        }
        R::SubmitArrayBatch(engine, lane, [out = out.NumericAlias(), states = std::move(states), first, op, a, b](R::ExecutionContext context, auto inputs, auto outputs)
                            {
#ifdef DNDS_USE_CUDA
            if (context.device != R::Device::Host())
            {
                CUDA::ApplyAccumulation(context, op, out.ComponentNumber(), outputs[0].DynamicView(), inputs[0].DynamicView(),
                    states.size() >= 2 ? inputs[first[1]].DynamicView() : R::ArrayView<const real>{},
                    states.size() >= 3 ? inputs[first[2]].DynamicView() : R::ArrayView<const real>{}, a, b);
                return;
            }
#endif
            const EquationViewLayout<real> dst(out, WriteBlocks{out, outputs});
            auto source = [&](index k)
            { return EquationViewLayout<const real>(states[k], ReadBlocks{states[k], inputs, first[k]}); };
            const auto empty = EquationBlockedView<const real>{};
            if (states.size() == 3)
                DispatchCPUEquationLayouts([&](auto o, auto u, auto v, auto w) { CPU::ApplyConstantDensityAccumulation(op,o,u,v,w,a,b); }, dst,source(0),source(1),source(2));
            else if (states.size() == 2)
                DispatchCPUEquationLayouts([&](auto o, auto u, auto v) { CPU::ApplyConstantDensityAccumulation(op,o,u,v,empty,a,b); }, dst,source(0),source(1));
            else
                DispatchCPUEquationLayouts([&](auto o, auto u) { CPU::ApplyConstantDensityAccumulation(op,o,u,empty,empty,a,b); }, dst,source(0)); }, std::move(reads), std::move(writes));
    }
    void EquationDofOperations::ApplyPhysical(PhysicalOperation operation, State &out,
                                              const State *x, const State *y, const State *z,
                                              real a, real b)
    {
        CheckCompatible(out, out);
        DNDS_check_throw_info(out.Schema().AccumulationKind() == PhysicalAccumulationKind::ConstantDensityVelocityPressure,
                              "unsupported nonidentity physical accumulation");
        DNDS_check_throw_info(out.ComponentNumber() >= 4 && out.Schema().blocks.front().name == "flow" &&
                                  out.Schema().blocks.front().components.size() == 4,
                              "constant-density accumulation requires leading flow4");
        for (const auto *input : {x, y, z})
            if (input)
            {
                CheckCompatible(out, *input);
                RequireDistinct(out, *input);
            }
        DNDS_check_throw_info(x &&
                                  (operation == PhysicalOperation::CopyScale || operation == PhysicalOperation::Axpy
                                       ? !y && !z
                                   : operation == PhysicalOperation::Combination || operation == PhysicalOperation::StateAndRate ? y && !z
                                                                                                                                 : y && z),
                              "physical accumulation operands differ from operation");
        const auto readX = x->ReadOwned();
        std::optional<State::ReadScope> readY, readZ;
        if (y)
            readY.emplace(y->ReadOwned());
        if (z)
            readZ.emplace(z->ReadOwned());
        auto write = out.WriteOwned();
        const EquationViewLayout<real> output(out, write);
        const EquationViewLayout<const real> inputX(*x, readX);
        const auto empty = EquationBlockedView<const real>{};
        const auto launch = [&](auto target, auto left, auto right, auto extra)
        {
            CPU::ApplyConstantDensityAccumulation(operation, target, left, right, extra, a, b);
        };
        if (readZ)
        {
            const EquationViewLayout<const real> inputY(*y, *readY), inputZ(*z, *readZ);
            DispatchCPUEquationLayouts(launch, output, inputX, inputY, inputZ);
        }
        else if (readY)
        {
            const EquationViewLayout<const real> inputY(*y, *readY);
            DispatchCPUEquationLayouts([&](auto target, auto left, auto right)
                                       { launch(target, left, right, empty); },
                                       output, inputX, inputY);
        }
        else
            DispatchCPUEquationLayouts([&](auto target, auto left)
                                       { launch(target, left, empty, empty); },
                                       output, inputX);
    }
}
