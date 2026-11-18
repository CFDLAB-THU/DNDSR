#pragma once

#include "Cauchy/State/EquationDofs.hpp"
#include "DNDS/Core/Span.hpp"

namespace DNDS::Cauchy
{
    enum class EquationViewKind
    {
        Generic,
        SeparateFlow5,
        Combined,
        CombinedAoS,
        CombinedSoA,
        CombinedAoSoA,
        SeparateFlow5AoS
    };

    namespace EquationViewDetail
    {
        template <EquationViewKind Kind, size_t TileWidth>
        struct ParentPolicy
        {
            using type = Runtime::UniformTableRows;
        };
        template <size_t TileWidth>
        struct ParentPolicy<EquationViewKind::CombinedAoS, TileWidth>
        {
            using type = Runtime::TableRows;
        };
        template <size_t TileWidth>
        struct ParentPolicy<EquationViewKind::CombinedSoA, TileWidth>
        {
            using type = Runtime::ColMajorTable;
        };
        template <size_t TileWidth>
        struct ParentPolicy<EquationViewKind::CombinedAoSoA, TileWidth>
        {
            static_assert(TileWidth != Runtime::DynamicExtent,
                          "AoSoA equation view needs a compile-time tile width");
            using type = Runtime::TiledColMajorTable<TileWidth>;
        };
    }

    /// Borrowed whole-equation view. The layout owner and every original access
    /// scope must outlive use. Device execution also needs device-resident block
    /// descriptors/offsets. No ownership, schema lookup or lease operation occurs
    /// in row access; total width never determines physical equation semantics.
    /// CUDA plan: dispatch dedicated expected-layout/physics kernels with
    /// direct component arguments or statically composed inline segment views.
    /// Device-callability here is not a requirement to upload or interpret the
    /// generic CPU descriptor registry in production GPU kernels.
    template <class T, EquationViewKind Kind = EquationViewKind::Generic,
              size_t Width = Runtime::DynamicExtent,
              size_t TileWidth = Runtime::DynamicExtent>
    struct EquationBlockedView
    {
        using value_type = T;
        static constexpr EquationViewKind kind = Kind;
        static constexpr bool aos = Kind == EquationViewKind::CombinedAoS;
        static constexpr bool soa = Kind == EquationViewKind::CombinedSoA;
        static constexpr bool aosoa = Kind == EquationViewKind::CombinedAoSoA;
        static constexpr bool table = aos; // compatibility name for existing kernels
        static constexpr bool combined = Kind == EquationViewKind::Combined || aos || soa || aosoa;
        static constexpr size_t row_width = Width;
        static constexpr size_t flow_width = Kind == EquationViewKind::SeparateFlow5 ||
                                                     Kind == EquationViewKind::SeparateFlow5AoS || aos || soa || aosoa
                                                 ? 5
                                                 : Runtime::DynamicExtent;
        using ParentPolicy = typename EquationViewDetail::ParentPolicy<Kind, TileWidth>::type;
        using FlowPolicy = std::conditional_t<combined, ParentPolicy,
                                              std::conditional_t<Kind == EquationViewKind::SeparateFlow5AoS,
                                                                 Runtime::TableRows,
                                                                 Runtime::UniformTableRows>>;
        using WholeView = EquationBlockArrayView<T, Runtime::AccessBoundsChecks, Width, ParentPolicy>;
        using FlowView = EquationBlockArrayView<T, Runtime::AccessBoundsChecks, flow_width, FlowPolicy>;
        static constexpr bool has_single_parent = combined,
                              has_static_layout = aos || soa || aosoa;
        static constexpr size_t component_stride = combined ? WholeView::component_stride : Runtime::DynamicExtent;
        static constexpr size_t row_stride = combined ? WholeView::row_stride : Runtime::DynamicExtent;
        SpanView<const EquationBlockArrayView<T>> blocks;
        SpanView<const index> offsets;
        index blockNumber = 0, rows = 0, components = 0, rans = -1, species = -1, flowFirst = -1;
        WholeView whole;
        FlowView flow;
        DNDS_DEVICE_CALLABLE index RowCount() const { return rows; }
        DNDS_DEVICE_CALLABLE index ComponentNumber() const { return Width == Runtime::DynamicExtent ? components : index(Width); }
        DNDS_DEVICE_CALLABLE index RowSize(index) const { return ComponentNumber(); }
        DNDS_DEVICE_CALLABLE T &operator()(index iRow, index j) const
        {
            if constexpr (Runtime::AccessBoundsChecks)
                DNDS_HD_assert(iRow >= 0 && iRow < rows && j >= 0 && j < ComponentNumber());
            if constexpr (combined)
                return whole(iRow, j);
            else
            {
                if constexpr (flow_width == 5)
                    if (j >= flowFirst && j < flowFirst + 5)
                        return flow(iRow, j - flowFirst);
                index iBlock = 0;
                while (j >= offsets[iBlock + 1])
                    ++iBlock;
                return blocks[iBlock](iRow, j - offsets[iBlock]);
            }
        }
        struct Row;
        DNDS_DEVICE_CALLABLE Row operator[](index iRow) const;
    };

    template <class T, EquationViewKind Kind, size_t Width, size_t TileWidth>
    struct EquationBlockedView<T, Kind, Width, TileWidth>::Row
    {
        using value_type = T;
        static constexpr EquationViewKind kind = Kind;
        static constexpr bool has_single_parent = combined,
                              has_static_layout = aos || soa || aosoa;
        static constexpr size_t row_width = Width, component_stride = EquationBlockedView::component_stride;
        EquationBlockedView view;
        index iRow;
        DNDS_DEVICE_CALLABLE index size() const { return view.ComponentNumber(); }
        DNDS_DEVICE_CALLABLE T &operator[](index j) const { return view(iRow, j); }
        /// A split logical row has no single raw pointer. Request Flow(),
        /// RANS(), Species() or Block() and retain that segment's stride.
        /// A combined raw row still needs ElementStride(), including SoA and
        /// tiled layouts; data()+j is valid only when that stride is one.
        template <bool C = combined, std::enable_if_t<C && combined, int> = 0>
        DNDS_DEVICE_CALLABLE T *data() const { return view.whole[iRow].data(); }
        template <bool C = combined, std::enable_if_t<C && combined, int> = 0>
        DNDS_DEVICE_CALLABLE index ElementStride() const { return view.whole[iRow].ElementStride(); }
        DNDS_DEVICE_CALLABLE auto Flow() const
        {
            if constexpr (flow_width == Runtime::DynamicExtent)
                if (view.flowFirst < 0)
                    return typename decltype(view.flow)::RowType{nullptr, 0};
            return view.flow[iRow];
        }
        DNDS_DEVICE_CALLABLE auto Block(index iBlock) const
        {
            using BlockRow = Runtime::RowSpan<T, Runtime::AccessBoundsChecks,
                                              Runtime::DynamicExtent,
                                              Runtime::DynamicExtent,
                                              component_stride>;
            if (iBlock < 0)
                return BlockRow{};
            if constexpr (combined)
            {
                const auto row = view.whole[iRow];
                const index first = view.offsets[iBlock];
                return BlockRow{row.data() + first * row.ElementStride(),
                                view.offsets[iBlock + 1] - first,
                                row.ElementStride()};
            }
            else
            {
                const auto row = view.blocks[iBlock][iRow];
                return BlockRow{row.data(), row.size(), row.ElementStride()};
            }
        }
        DNDS_DEVICE_CALLABLE auto RANS() const { return Block(view.rans); }
        DNDS_DEVICE_CALLABLE auto Species() const { return Block(view.species); }
    };
    template <class T, EquationViewKind Kind, size_t Width, size_t TileWidth>
    DNDS_DEVICE_CALLABLE typename EquationBlockedView<T, Kind, Width, TileWidth>::Row
    EquationBlockedView<T, Kind, Width, TileWidth>::operator[](index iRow) const { return {*this, iRow}; }

    /// Borrowed dispatch descriptor. Its block/offset ranges are small metadata,
    /// retained by the caller along with payload leases. No owner is passed to a
    /// kernel; the same frozen shape selects CPU or future device specializations.
    template <class T>
    class EquationLayoutView
    {
        SpanView<const EquationBlockArrayView<T>> _blocks;
        SpanView<const index> _offsets;
        index _flow = -1, _rans = -1, _species = -1;
        bool _combined = false;

    public:
        EquationLayoutView() = default;
        EquationLayoutView(SpanView<const EquationBlockArrayView<T>> blocks,
                           SpanView<const index> offsets, index flow, index rans,
                           index species, bool combined)
            : _blocks(blocks), _offsets(offsets), _flow(flow), _rans(rans),
              _species(species), _combined(combined) {}
        bool Combined() const { return _combined; }
        bool Flow5() const { return _flow >= 0 && _blocks[_flow].ComponentNumber() == 5; }
        bool FlowTable() const { return Flow5() && _blocks[_flow].HasTableLayout(); }
        bool Table() const { return _combined && _blocks[0].HasTableLayout(); }
        bool ColMajor() const
        {
            return _combined && _blocks[0].parent.kind == Runtime::RowLayoutKind::ColMajorTable;
        }
        bool TiledColMajor(index tileWidth) const
        {
            return _combined &&
                   _blocks[0].parent.kind == Runtime::RowLayoutKind::TiledColMajorTable &&
                   _blocks[0].parent.stride == tileWidth;
        }
        /// Compact whole-equation projection is independent of which components
        /// represent flow. An empty halo may be represented by an absent view.
        bool CompactHostTable(index width) const
        {
            if (!Table() || ComponentNumber() != width)
                return false;
            const auto parent = _blocks[0].parent;
            return (!parent.rows && !parent.data) || (parent.device == Runtime::Device::Host() &&
                                                      parent.width == width && parent.stride == width);
        }
        template <size_t Width>
        auto CompactRows() const
        {
            using Rows = Runtime::FixedRows<Width, Width>;
            DNDS_assert(CompactHostTable(Width));
            const auto parent = _blocks[0].parent;
            return EquationBlockArrayView<T, Runtime::AccessBoundsChecks, Width, Rows>{
                {parent.data, nullptr, parent.rows, Width, Width, nullptr, Runtime::RowLayoutKind::Table, parent.device},
                0,
                Width};
        }
        index ComponentNumber() const { return _offsets[_offsets.size() - 1]; }
        template <EquationViewKind Kind = EquationViewKind::Generic,
                  size_t Width = Runtime::DynamicExtent,
                  size_t TileWidth = Runtime::DynamicExtent>
        auto View() const
        {
            using V = EquationBlockedView<T, Kind, Width, TileWidth>;
            DNDS_check_throw_info(!V::combined || _combined, "combined view requires ordered single-parent storage");
            DNDS_check_throw_info(!V::aos || Table(), "combined AoS view requires row-major table storage");
            DNDS_check_throw_info(!V::soa || ColMajor(), "combined SoA view requires column-major table storage");
            DNDS_check_throw_info(!V::aosoa || TiledColMajor(index(TileWidth)),
                                  "combined AoSoA view requires the selected tiled storage");
            DNDS_check_throw_info(V::flow_width != 5 || Flow5(), "static flow view requires flow5 block");
            DNDS_check_throw_info(!(V::aos || Kind == EquationViewKind::SeparateFlow5AoS) || FlowTable(), "static AoS flow requires table storage");
            DNDS_check_throw_info(Width == Runtime::DynamicExtent || index(Width) == ComponentNumber(), "whole-equation static width differs");
            V view;
            view.blocks = _blocks;
            view.offsets = _offsets;
            view.blockNumber = index(_blocks.size());
            view.rows = _blocks[0].RowCount();
            view.components = ComponentNumber();
            view.rans = _rans;
            view.species = _species;
            const auto parent = _blocks[0].parent;
            view.whole = {{parent.data, parent.offsets, parent.rows, parent.width, parent.stride, parent.sizes, parent.kind, parent.device}, 0, ComponentNumber()};
            if (_flow >= 0)
            {
                const auto block = _blocks[_flow];
                view.flow = {{block.parent.data, block.parent.offsets, block.parent.rows, block.parent.width, block.parent.stride, block.parent.sizes, block.parent.kind, block.parent.device}, block.first, block.components};
                view.flowFirst = _offsets[_flow];
            }
            return view;
        }
    };

    /// Immutable backend-aware descriptor ownership, built under payload scopes.
    /// Moving preserves views because MetadataBuffer allocations are shared;
    /// destruction still requires all borrowed consumers to have completed.
    template <class T>
    class EquationViewLayout : public EquationLayoutView<T>
    {
        Runtime::MetadataBuffer<EquationBlockArrayView<T>> _blockStorage;
        Runtime::MetadataBuffer<index> _offsetStorage;

    public:
        template <class Scope>
        EquationViewLayout(const EquationDofs &state, const Scope &scope, bool ghost = false)
        {
            bool _combined = state.PhysicalFieldNumber() == 1;
            index _flow = -1, _rans = -1, _species = -1;
            std::vector<EquationBlockArrayView<T>> _blocks;
            std::vector<index> _offsets{0};
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            {
                const auto &name = state.Schema().blocks[iBlock].name;
                if (name == "flow")
                    _flow = iBlock;
                if (name == "rans")
                    _rans = iBlock;
                if (name == "species")
                    _species = iBlock;
                if constexpr (std::is_const_v<T>)
                {
                    const auto part = scope.Block(iBlock);
                    _blocks.push_back(ghost ? part.ghost : part.owned);
                }
                else
                {
                    DNDS_check_throw_info(!ghost, "write equation view has no ghost part");
                    _blocks.push_back(scope.Block(iBlock));
                }
                const auto location = state.LocateBlock(iBlock);
                _combined = _combined && location.firstComponent == _offsets.back();
                _offsets.push_back(_offsets.back() + location.componentNumber);
            }
            _blockStorage = Runtime::MetadataBuffer<EquationBlockArrayView<T>>(std::move(_blocks));
            _offsetStorage = Runtime::MetadataBuffer<index>(std::move(_offsets));
            static_cast<EquationLayoutView<T> &>(*this) = EquationLayoutView<T>(
                _blockStorage.HostView(), _offsetStorage.HostView(), _flow, _rans, _species, _combined);
        }

        EquationLayoutView<T> Descriptor() const { return *this; }
    };

    /// Explicit operator-selected dispatch, never an automatic kernel policy.
    /// Each operator chooses supported families in its own preparation code.
    /// Joint construction prevents input/output specialization products.
    /// CPU placement, schema compatibility, row spaces and nonaliasing remain
    /// operator-entry responsibilities. Callback return types must agree.
    template <EquationViewKind Kind = EquationViewKind::Generic,
              size_t Width = Runtime::DynamicExtent,
              size_t TileWidth = Runtime::DynamicExtent,
              class Kernel, class First, class... Other>
    decltype(auto) DispatchCPUEquationViews(Kernel &&kernel, const First &first, const Other &...other)
    {
        DNDS_check_throw_info(((first.ComponentNumber() == other.ComponentNumber()) && ...), "equation view widths differ");
        return std::forward<Kernel>(kernel)(first.template View<Kind, Width, TileWidth>(),
                                            other.template View<Kind, Width, TileWidth>()...);
    }

    /// Operator-selected CPU layout dispatch. All operands must share the same
    /// combined physical layout; otherwise use the fully generic view. The
    /// numerical kernel and logical component order are identical. AoSoA tile
    /// widths outside this bounded instantiation list remain supported by the
    /// generic path rather than growing every translation unit.
    template <size_t Width = Runtime::DynamicExtent, class Kernel, class First, class... Other>
    decltype(auto) DispatchCPUEquationLayouts(Kernel &&kernel,
                                              const First &first,
                                              const Other &...other)
    {
        DNDS_check_throw_info(((first.ComponentNumber() == other.ComponentNumber()) && ...),
                              "equation view widths differ");
        const auto all = [&](auto predicate)
        { return predicate(first) && (predicate(other) && ...); };
        if (all([](const auto &layout)
                { return layout.Table() && layout.Flow5(); }))
            return DispatchCPUEquationViews<EquationViewKind::CombinedAoS, Width>(
                std::forward<Kernel>(kernel), first, other...);
        if (all([](const auto &layout)
                { return layout.ColMajor() && layout.Flow5(); }))
            return DispatchCPUEquationViews<EquationViewKind::CombinedSoA, Width>(
                std::forward<Kernel>(kernel), first, other...);
        if (all([](const auto &layout)
                { return layout.TiledColMajor(8) && layout.Flow5(); }))
            return DispatchCPUEquationViews<EquationViewKind::CombinedAoSoA, Width, 8>(
                std::forward<Kernel>(kernel), first, other...);
        if (all([](const auto &layout)
                { return layout.TiledColMajor(16) && layout.Flow5(); }))
            return DispatchCPUEquationViews<EquationViewKind::CombinedAoSoA, Width, 16>(
                std::forward<Kernel>(kernel), first, other...);
        if (all([](const auto &layout)
                { return layout.TiledColMajor(32) && layout.Flow5(); }))
            return DispatchCPUEquationViews<EquationViewKind::CombinedAoSoA, Width, 32>(
                std::forward<Kernel>(kernel), first, other...);
        return DispatchCPUEquationViews<EquationViewKind::Generic, Width>(
            std::forward<Kernel>(kernel), first, other...);
    }
}
