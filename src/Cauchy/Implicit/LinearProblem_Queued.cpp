#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Cauchy/Implicit/CPU/LinearBlockKernels.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"
#include "DNDS/Algebra/ArrayInputCache.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Implicit/CUDA/LinearBlockKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace A = Algebra;
    struct LinearProblemContext::InputWorkspace
    {
        Algebra::ArrayInputCache rhs, owned, ghost;
        InputWorkspace(const Algebra::DofSpace &space, index components, R::Device device)
            : rhs(R::RowLayout::Dense(space.OwnedRows(), components), device, "Cauchy/linear/tiledRHS"),
              owned(R::RowLayout::Dense(space.OwnedRows(), components), device, "Cauchy/linear/tiledInput"),
              ghost(R::RowLayout::Dense(space.GhostRows(), components), device, "Cauchy/linear/tiledGhost") {}
    };
    void LinearProblemContext::ConfigureInputLayout(LinearInputLayout layout)
    {
        DNDS_check_throw_info(layout == LinearInputLayout::Native || layout == LinearInputLayout::Tiled32,
                              "unknown linear input layout");
        if (layout == LinearInputLayout::Native)
            _inputWorkspace.reset();
        else if (!_inputWorkspace)
        {
            DNDS_check_throw_info(Device().space == R::MemorySpace::CUDA, "tiled linear shadow requires CUDA");
            _inputWorkspace = std::make_shared<InputWorkspace>(_space, _components, Device());
        }
    }
    namespace
    {
        using Field = A::DofField<real>;
        using ReadRows = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
        using WriteRows = R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>;
        using Input = R::ArrayOperand<real, false, R::UniformTableRows>;
        using Output = R::ArrayOperand<real, true, R::UniformTableRows>;
        template <class State, class Visit>
        void Fields(State &state, Visit visit)
        {
            if constexpr (std::is_same_v<std::remove_const_t<State>, EquationDofs>)
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    visit(state.PhysicalField(iField));
            else
                visit(state);
        }
        template <class State>
        index Width(const State &state)
        {
            if constexpr (std::is_same_v<State, EquationDofs>)
                return state.ComponentNumber();
            else
                return index(state.Width());
        }
        template <class T>
        auto Table(R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows> view)
        {
            DNDS_check_throw_info(view.kind == R::RowLayoutKind::Table, "queued linear CUDA/matrix operand requires AoS");
            return R::ArrayView<T, R::AccessBoundsChecks, R::TableRows>{view.data, nullptr, view.rows,
                                                                        view.width, view.stride, nullptr, view.kind, view.device};
        }
        LinearColumnsView SymbolicColumns(LinearColumnsView::Wide columns)
        {
            return {columns};
        }
        LinearColumnsView SymbolicColumns(LinearColumnsView::Narrow columns, LinearColumnsView::Narrow offsets)
        {
            return {offsets, columns};
        }
        struct ColumnSlot
        {
            index owned, ghost, component;
        };
        /// Only component-scale metadata. Arrays themselves are retained by
        /// SubmitArrayBatch, and no DOF/communicator owner enters the callback.
        struct ComponentBinding
        {
            R::MetadataBuffer<ColumnSlot> columns;
            index ownedRows = 0, ghostRows = 0;
            template <class T>
            auto View(SpanView<const R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows>> values) const
            {
                std::vector<A::DofComponentColumn<T>> result;
                result.reserve(columns.size());
                for (const auto &column : columns.HostView())
                    result.push_back({values[column.owned], column.ghost >= 0 ? values[column.ghost] : R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows>{}, column.component});
                return A::DofComponentLayout<T>(std::move(result), ownedRows, ghostRows);
            }
            index FirstOwned() const { return columns.HostView()[0].owned; }
            index FirstGhost() const { return columns.HostView()[0].ghost; }
        };
        template <class State>
        ComponentBinding Describe(const State &state, index ownedSlot, index ghostSlot)
        {
            std::vector<ColumnSlot> columns;
            if constexpr (std::is_same_v<State, EquationDofs>)
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto location = state.LocateBlock(iBlock);
                    for (index j = 0; j < location.componentNumber; ++j)
                        columns.push_back({ownedSlot + location.physicalField,
                                           ghostSlot >= 0 ? ghostSlot + location.physicalField : -1, location.firstComponent + j});
                }
            else
                for (index j = 0; j < index(state.Width()); ++j)
                    columns.push_back({ownedSlot, ghostSlot, j});
            return {R::MetadataBuffer<ColumnSlot>(std::move(columns)), index(state.Space().OwnedRows()),
                    ghostSlot >= 0 ? index(state.Space().GhostRows()) : 0};
        }
        template <class State>
        ComponentBinding AddInput(const State &state, bool withGhost, std::vector<Input> &inputs)
        {
            const index own = index(inputs.size());
            Fields(state, [&](const auto &field)
                   { inputs.push_back(R::ReadAs<R::UniformTableRows>(field.Owned())); });
            const index ghost = withGhost ? index(inputs.size()) : -1;
            if (withGhost)
                Fields(state, [&](const auto &field)
                       { inputs.push_back(R::ReadAs<R::UniformTableRows>(field.Ghost())); });
            return Describe(state, own, ghost);
        }
    }

    template <class State>
    void LinearProblemContext::SubmitRows(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                          OptionalRef<const State> rhs, OptionalRef<const State> input, State &output, LinearRegion region, LinearKernelOptions options) const
    {
        ValidateMatrix();
        const auto context = stream.Context();
        const bool jacobi = bool(rhs), zero = !input;
        DNDS_check_throw_info(engine.Owns(stream) && context.device == Device() && (rhs || input),
                              "queued linear stream, device or operands differ");
        DNDS_check_throw_info(region == LinearRegion::All || region == LinearRegion::Core || region == LinearRegion::HaloDependent,
                              "invalid queued linear region");
        const auto validate = [&](const State &state)
        {
            state.Validate();
            DNDS_check_throw_info(state.Space().Same(_space) && Width(state) == _components && state.Device() == Device(),
                                  "queued linear state space, width or device differs");
            if (Device() != R::Device::Host())
            {
                if constexpr (std::is_same_v<State, EquationDofs>)
                    (void)state.CoupledField();
                Fields(state, [](const auto &field)
                       { DNDS_check_throw_info(field.Owned().Layout().Kind() == R::RowLayoutKind::Table &&
                                                   field.Ghost().Layout().Kind() == R::RowLayoutKind::Table,
                                               "queued CUDA linear state requires combined AoS"); });
            }
        };
        validate(output);
        const auto distinct = [&](const State &source)
        {
            validate(source);
            if constexpr (std::is_same_v<State, EquationDofs>)
                source.RequireCompatible(output);
            Fields(output, [&](const auto &destination)
                   { Fields(source, [&](const auto &field)
                            { DNDS_check_throw_info(!destination.Owned().Storage().Overlaps(field.Owned().Storage()) &&
                                                        !destination.Owned().Storage().Overlaps(field.Ghost().Storage()),
                                                    "queued linear output aliases an input"); }); });
        };
        if (rhs)
            distinct(*rhs);
        if (input)
            distinct(*input);
        Fields(output, [&](const auto &field)
               {
            const auto values = field.Owned().Storage();
            DNDS_check_throw_info(!values.Overlaps(_diagonal.Storage()) && !values.Overlaps(_inverse.Storage()) &&
                                      (!_offDiagonal || !values.Overlaps(_offDiagonal->Values().Storage())),
                                  "queued linear output aliases matrix storage"); });
        if (jacobi)
            DNDS_check_throw_info(_inverseCurrent && _inverseSourceEpoch == _diagonal.MutationEpoch() &&
                                      _inverseEpoch == _inverse.MutationEpoch(),
                                  "queued Jacobi inverse is missing or invalidated by a submitted write");
        const auto coupled = [](auto &state) -> decltype(auto)
        {
            if constexpr (std::is_same_v<State, EquationDofs>)
                return state.CoupledField();
            else
                return (state);
        };
        const auto shadow = [&](Algebra::ArrayInputCache &cache, const R::Array<real> &source) -> const R::Array<real> &
        {
            cache.Refresh(engine, stream, source);
            return cache.Values();
        };

        if (!zero && !_offDiagonal && Device() != R::Device::Host())
        {
            DNDS_check_throw_info(!options.componentGroups, "component-group CUDA Jacobi requires an assembled operator");
            DNDS_check_throw_info(_offDiagonalEvaluator && _offDiagonalEvaluator->Device() == Device(),
                                  "queued evaluated Jacobian is missing or belongs to another device");
            EvaluatedLinearAction action;
            action.selection = _pattern->rows.Selections().For(region);
            action.rows = region == LinearRegion::All    ? _pattern->rows.OrderArray()
                          : region == LinearRegion::Core ? _pattern->rows.CoreArray()
                                                         : _pattern->rows.DependentArray();
            action.diagonal = jacobi ? _inverse : _diagonal;
            action.owned = coupled(*input).Owned();
            if (region != LinearRegion::Core && _pattern->hasGhostColumns)
                action.ghost = coupled(*input).Ghost();
            if (rhs)
                action.rhs = coupled(*rhs).Owned();
            if (_inputWorkspace)
            {
                action.owned = shadow(_inputWorkspace->owned, action.owned);
                if (action.ghost)
                    action.ghost = shadow(_inputWorkspace->ghost, *action.ghost);
                if (action.rhs)
                    action.rhs = shadow(_inputWorkspace->rhs, *action.rhs);
            }
            action.output = coupled(output).Owned();
            action.status = ActionStatus();
            action.fullDiagonal = _diagonalStorage == DiagonalStorage::Block;
            action.rowMajorBlocks = _rowMajorBlocks;
            _offDiagonalEvaluator->SubmitRows(engine, stream, std::move(action));
            return;
        }

        // Pattern/offset arrays are operands too: an immediate read scope here
        // would fence its CUDA stream when the completed command is retired.
        // Scheduled reads retain CSR shape buffers and reject mutation until
        // the same operation completion that retires the numerical operands.
        const auto &patternArrays = _pattern->rows;
        const auto evaluator = _offDiagonalEvaluator;
        std::vector<Input> reads{R::ReadAs<R::UniformTableRows>(_diagonal)};
        const index diagonalSlot = jacobi ? index(reads.size()) : 0;
        if (jacobi)
            reads.push_back(R::ReadAs<R::UniformTableRows>(_inverse));
        index blocksSlot = -1;
        if (!zero)
        {
            if (_offDiagonal)
            {
                blocksSlot = index(reads.size());
                reads.push_back(R::ReadAs<R::UniformTableRows>(_offDiagonal->Values()));
            }
            else
            {
                DNDS_check_throw_info(context.device == R::Device::Host() && _offDiagonalEvaluator,
                                      "queued evaluated device Jacobian producer is not installed");
            }
        }
        ComponentBinding right, source;
        if (rhs)
            right = AddInput(*rhs, false, reads);
        if (input)
            source = AddInput(*input, region != LinearRegion::Core && _pattern->hasGhostColumns, reads);
        if (_inputWorkspace)
        {
            if (rhs)
                reads[right.FirstOwned()] = R::ReadAs<R::UniformTableRows>(shadow(_inputWorkspace->rhs, coupled(*rhs).Owned()));
            if (input)
            {
                reads[source.FirstOwned()] = R::ReadAs<R::UniformTableRows>(shadow(_inputWorkspace->owned, coupled(*input).Owned()));
                if (source.FirstGhost() >= 0)
                    reads[source.FirstGhost()] = R::ReadAs<R::UniformTableRows>(shadow(_inputWorkspace->ghost, coupled(*input).Ghost()));
            }
        }
        std::vector<Output> writes;
        Fields(output, [&](auto &field)
               { writes.push_back(R::WriteAs<R::UniformTableRows>(field.Owned())); });
        const auto destination = Describe(output, 0, -1);
        const auto components = _components;
        const bool fullDiagonal = _diagonalStorage == DiagonalStorage::Block;
        const auto submit = [&](auto symbolic)
        {
            R::SubmitArrayBatch(engine, stream, [evaluator, right, source, destination, components, fullDiagonal, region, diagonalSlot, blocksSlot, jacobi, zero, options, rowMajorBlocks = _rowMajorBlocks, ownedRows = _space.OwnedRows(), selections = patternArrays.Selections()](R::ExecutionContext context, SpanView<const ReadRows> values, SpanView<const WriteRows> outputs, auto order, auto position, auto core, auto dependent, auto offsets, auto... symbolicViews)
                                {
            const auto symbolicColumns = SymbolicColumns(symbolicViews...);
            const LinearPatternView pattern{symbolicColumns, order, position, core, dependent, index(ownedRows), selections};
            const auto rows = pattern.Rows(region);
            A::BlockMatrixArrayView<const real> blocks;
            if (blocksSlot >= 0)
                blocks = {values[blocksSlot], offsets, components, rowMajorBlocks};
            if (context.device != R::Device::Host())
            {
#ifdef DNDS_USE_CUDA
                const CUDA::LinearBlockInput request{blocks, zero ? decltype(pattern.columns){} : pattern.columns,
                    rows, values[diagonalSlot], pattern.ownedRows, int(components), fullDiagonal, rowMajorBlocks};
                const auto x = zero ? CUDA::VectorReadRows{} : values[source.FirstOwned()];
                const auto ghost = !zero && source.FirstGhost() >= 0 ? values[source.FirstGhost()] : CUDA::VectorReadRows{};
                if (jacobi)
                    CUDA::JacobiLinearRows(context, request, values[right.FirstOwned()], x, ghost, Table(outputs[0]), zero, options.componentGroups);
                else
                    CUDA::ApplyLinearRows(context, request, x, ghost, Table(outputs[0]));
#else
                DNDS_check_throw_info(false, "CUDA linear operator is not compiled");
#endif
                return;
            }
            CPU::LinearBlockInput request{{{}, evaluator.get(), components}, pattern, rows,
                values[diagonalSlot], components, fullDiagonal, {}, rowMajorBlocks};
            if (blocksSlot >= 0) request.blocks.assembled.emplace(blocks);
            const auto y = destination.template View<real>(outputs);
            const auto x = (zero ? right : source).View<const real>(values);
            if (jacobi)
            {
                const auto b = right.View<const real>(values);
                CPU::JacobiLinearRows(request, b.View().OwnedPart(), x.View().OwnedPart(), x.View().GhostPart(), y.View(), zero);
            }
            else
                CPU::ApplyLinearRows(request, x.View().OwnedPart(), x.View().GhostPart(), y.View()); }, std::move(reads), std::move(writes), std::tuple_cat(std::make_tuple(R::ReadAs<R::FixedRows<1>>(patternArrays.OrderArray()), R::ReadAs<R::FixedRows<1>>(patternArrays.PositionArray()), R::ReadAs<R::FixedRows<1>>(patternArrays.CoreArray()), R::ReadAs<R::FixedRows<1>>(patternArrays.DependentArray()), R::ReadAs<R::FixedRows<1>>(!zero && _offDiagonal ? _offDiagonal->OffsetArray() : patternArrays.OrderArray())), std::move(symbolic)));
        };
        if (patternArrays.UsesInt32())
            submit(std::make_tuple(R::ReadAs<R::FixedRows<1>>(patternArrays.Columns32()), R::ReadAs<R::FixedRows<1>>(patternArrays.Offsets32())));
        else
            submit(std::make_tuple(R::ReadAs<R::CSRRows>(patternArrays.ColumnsArray())));
    }

    void LinearProblemContext::SubmitApply(R::EagerEngine &engine, const R::ExecutionStream &stream, const Field &input, Field &output, LinearRegion region) const
    {
        SubmitRows<Field>(engine, stream, {}, std::cref(input), output, region);
    }
    void LinearProblemContext::SubmitApply(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &input, EquationDofs &output, LinearRegion region) const
    {
        SubmitRows<EquationDofs>(engine, stream, {}, std::cref(input), output, region);
    }
    void LinearProblemContext::SubmitJacobi(R::EagerEngine &engine, const R::ExecutionStream &stream, const Field &rhs, const Field &input, Field &output, LinearRegion region, LinearKernelOptions options) const
    {
        SubmitRows<Field>(engine, stream, std::cref(rhs), std::cref(input), output, region, options);
    }
    void LinearProblemContext::SubmitJacobi(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &rhs, const EquationDofs &input, EquationDofs &output, LinearRegion region, LinearKernelOptions options) const
    {
        SubmitRows<EquationDofs>(engine, stream, std::cref(rhs), std::cref(input), output, region, options);
    }
    void LinearProblemContext::SubmitJacobiZero(R::EagerEngine &engine, const R::ExecutionStream &stream, const Field &rhs, Field &output, LinearRegion region, LinearKernelOptions options) const
    {
        SubmitRows<Field>(engine, stream, std::cref(rhs), {}, output, region, options);
    }
    void LinearProblemContext::SubmitJacobiZero(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &rhs, EquationDofs &output, LinearRegion region, LinearKernelOptions options) const
    {
        SubmitRows<EquationDofs>(engine, stream, std::cref(rhs), {}, output, region, options);
    }
}
