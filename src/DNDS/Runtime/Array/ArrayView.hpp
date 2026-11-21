#pragma once

#include "DNDS/Runtime/Array/RowLayout.hpp"
#include <iterator>

namespace DNDS::Runtime
{
    /// A borrowed row iterator records a logical position rather than creating
    /// an out-of-allocation end pointer for a strided row.
    template <class T>
    struct StridedRowIterator
    {
        using value_type = std::remove_const_t<T>;
        using difference_type = index;
        using reference = T &;
        using pointer = T *;
        using iterator_category = std::random_access_iterator_tag;
        T *base = nullptr;
        index stride = 1, position = 0;
        DNDS_DEVICE_CALLABLE reference operator*() const { return base[position * stride]; }
        DNDS_DEVICE_CALLABLE reference operator[](index n) const { return base[(position + n) * stride]; }
        DNDS_DEVICE_CALLABLE StridedRowIterator &operator++()
        {
            ++position;
            return *this;
        }
        DNDS_DEVICE_CALLABLE StridedRowIterator operator++(int)
        {
            auto old = *this;
            ++*this;
            return old;
        }
        DNDS_DEVICE_CALLABLE StridedRowIterator &operator--()
        {
            --position;
            return *this;
        }
        DNDS_DEVICE_CALLABLE StridedRowIterator operator--(int)
        {
            auto old = *this;
            --*this;
            return old;
        }
        DNDS_DEVICE_CALLABLE StridedRowIterator &operator+=(index n)
        {
            position += n;
            return *this;
        }
        DNDS_DEVICE_CALLABLE StridedRowIterator &operator-=(index n)
        {
            position -= n;
            return *this;
        }
        DNDS_DEVICE_CALLABLE StridedRowIterator operator+(index n) const
        {
            auto out = *this;
            return out += n;
        }
        DNDS_DEVICE_CALLABLE StridedRowIterator operator-(index n) const
        {
            auto out = *this;
            return out -= n;
        }
        DNDS_DEVICE_CALLABLE index operator-(StridedRowIterator other) const { return position - other.position; }
        DNDS_DEVICE_CALLABLE bool operator==(StridedRowIterator other) const { return position == other.position; }
        DNDS_DEVICE_CALLABLE bool operator!=(StridedRowIterator other) const { return !(*this == other); }
        DNDS_DEVICE_CALLABLE bool operator<(StridedRowIterator other) const { return position < other.position; }
        DNDS_DEVICE_CALLABLE bool operator>(StridedRowIterator other) const { return other < *this; }
        DNDS_DEVICE_CALLABLE bool operator<=(StridedRowIterator other) const { return !(*this > other); }
        DNDS_DEVICE_CALLABLE bool operator>=(StridedRowIterator other) const { return !(*this < other); }
        friend DNDS_DEVICE_CALLABLE StridedRowIterator operator+(index n, StridedRowIterator value) { return value + n; }
    };

    namespace Detail
    {
        // A fully fixed row crosses kernel boundaries as one pointer. General
        // rows retain their original three-word descriptor: shrinking those
        // cases also changes their calling convention, which can hurt generic
        // kernels. Both cases stay plain, trivially copied host/device views.
        template <class T, bool DynamicWidth, bool DynamicStride>
        struct RowSpanStorage
        {
            T *data = nullptr;
            index size = 0, stride = 1;
            DNDS_DEVICE_CALLABLE RowSpanStorage() = default;
            DNDS_DEVICE_CALLABLE RowSpanStorage(T *dataIn, index sizeIn, index strideIn)
                : data(dataIn), size(sizeIn), stride(strideIn) {}
        };
        template <class T>
        struct RowSpanStorage<T, false, false>
        {
            T *data = nullptr;
            DNDS_DEVICE_CALLABLE RowSpanStorage() = default;
            DNDS_DEVICE_CALLABLE RowSpanStorage(T *dataIn, index, index) : data(dataIn) {}
        };
    }

    /// Borrowed logical row. General element access preserves component stride;
    /// raw data() is reserved for dense/interop/low-level adapters. The outer
    /// scope, not this trivially copied row, retains values and shape metadata.
    template <class T, bool Checked = AccessBoundsChecks, size_t Width = DynamicExtent,
              size_t Capacity = DynamicExtent, size_t ComponentStride = DynamicExtent>
    class RowSpan
    {
        Detail::RowSpanStorage<T, Width == DynamicExtent, ComponentStride == DynamicExtent> _storage;

    public:
        using value_type = T;
        static constexpr size_t row_width = Width, row_capacity = Capacity, component_stride = ComponentStride;
        DNDS_DEVICE_CALLABLE RowSpan() = default;
        DNDS_DEVICE_CALLABLE RowSpan(T *data, index size, index stride = 1) : _storage(data, size, stride) {}
        template <class U, bool OtherChecked, size_t W, size_t C, size_t S,
                  std::enable_if_t<std::is_convertible_v<U *, T *> && (Width == DynamicExtent || Width == W) &&
                                       (Capacity == DynamicExtent || Capacity == C) &&
                                       (ComponentStride == DynamicExtent || ComponentStride == S),
                                   int> = 0>
        DNDS_DEVICE_CALLABLE RowSpan(const RowSpan<U, OtherChecked, W, C, S> &other)
            : _storage(other.data(), other.size(), other.ElementStride()) {}
        DNDS_DEVICE_CALLABLE index ElementStride() const
        {
            if constexpr (ComponentStride != DynamicExtent)
                return index(ComponentStride);
            else
                return _storage.stride;
        }
        DNDS_DEVICE_CALLABLE T &operator[](index column) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(column >= 0 && column < this->size());
            return _storage.data[column * ElementStride()];
        }
        DNDS_DEVICE_CALLABLE T *data() const { return _storage.data; }
        DNDS_DEVICE_CALLABLE index size() const
        {
            if constexpr (Width != DynamicExtent)
                return Width;
            else
                return _storage.size;
        }
        DNDS_DEVICE_CALLABLE bool empty() const { return size() == 0; }
        DNDS_DEVICE_CALLABLE auto begin() const
        {
            if constexpr (ComponentStride == 1)
                return _storage.data;
            else
                return StridedRowIterator<T>{_storage.data, ElementStride(), 0};
        }
        DNDS_DEVICE_CALLABLE auto end() const
        {
            if constexpr (ComponentStride == 1)
                return size() ? _storage.data + size() : _storage.data;
            else
                return StridedRowIterator<T>{_storage.data, ElementStride(), size()};
        }
        DNDS_DEVICE_CALLABLE auto cbegin() const { return StridedRowIterator<const std::remove_const_t<T>>{_storage.data, ElementStride(), 0}; }
        DNDS_DEVICE_CALLABLE auto cend() const { return StridedRowIterator<const std::remove_const_t<T>>{_storage.data, ElementStride(), size()}; }
    };

    /// Plain kernel view: no ownership, placement dispatch or tracking per element.
    /// Placement covers values and offsets/sizes. ValidateDevice is a host-wrapper
    /// check, never an element-access check. A copied view does not retain storage.
    /// Checked=true is a separate development policy, independent of NDEBUG.
    /// @pre Caller retains the lease/operation through actual use, validates row
    /// and component indices (including stencil indirection), and uses the correct
    /// device. An owned-only core stencil must be reclassified after mesh changes.
    /// Copying this view does not retain values or offset/size metadata.
    template <class T, bool Checked = AccessBoundsChecks, class LayoutPolicy = RowLayout>
    struct ArrayView
    {
        using value_type = T;
        using Policy = Detail::RowPolicyTraits<LayoutPolicy>;
        static constexpr size_t component_stride = Policy::component_stride;
        static constexpr size_t row_width = Policy::row_width, row_capacity = Policy::row_capacity, row_stride = Policy::row_stride;
        static constexpr RowLayoutKind layout_kind = Policy::layout_kind;
        static constexpr bool is_dynamic = Policy::is_dynamic;
        using RowType = RowSpan<T, Checked, row_width, row_capacity, Policy::component_stride>;
        T *data = nullptr;
        const size_t *offsets = nullptr;
        index rows = 0, width = 0, stride = 0;
        const size_t *sizes = nullptr;
        RowLayoutKind kind = RowLayoutKind::Table;
        Device device = Device::Unspecified();

        DNDS_DEVICE_CALLABLE Device GetDevice() const { return device; }
        /// Check borrowed operands before a CPU loop or CUDA launch. Empty
        /// optional views may be unbound; a nonempty view must have provenance.
        /// This verifies placement, not pointer lifetime or stream readiness.
        void ValidateDevice(Device expected) const
        {
            DNDS_check_throw_info(expected.IsValid(), "array view: invalid execution device");
            DNDS_check_throw_info(rows >= 0, "array view: negative row count");
            if (!rows && device == Device::Unspecified())
                return;
            DNDS_check_throw_info(device == expected, "array view: placement mismatch (including device ordinal)");
        }

        DNDS_DEVICE_CALLABLE index Width() const { return width; }
        DNDS_DEVICE_CALLABLE index RowCount() const { return rows; }
        DNDS_DEVICE_CALLABLE index Rows() const { return RowCount(); }
        DNDS_DEVICE_CALLABLE index RowSize(index row) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < rows);
            if constexpr (row_width != DynamicExtent)
                return row_width;
            else if constexpr (!is_dynamic && layout_kind == RowLayoutKind::Table)
                return width;
            else if constexpr (!is_dynamic && layout_kind == RowLayoutKind::Padded)
                return sizes ? sizes[row] : width;
            else if constexpr (!is_dynamic && layout_kind == RowLayoutKind::CSR)
                return offsets ? offsets[row + 1] - offsets[row] : width;
            else
                return offsets ? offsets[row + 1] - offsets[row] : sizes ? sizes[row]
                                                                         : width;
        }
        DNDS_DEVICE_CALLABLE T *Row(index row) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < rows);
            const index logicalWidth = row_width == DynamicExtent ? width : index(row_width);
            index offset;
            if constexpr (is_dynamic)
            {
                if (kind == RowLayoutKind::ColMajorTable)
                    offset = Detail::ColumnRowOffset(row, logicalWidth);
                else if (kind == RowLayoutKind::TiledColMajorTable)
                    offset = Detail::TiledColumnRowOffset(row, logicalWidth, stride);
                else
                    offset = offsets ? offsets[row] : row * stride;
            }
            else if constexpr (layout_kind == RowLayoutKind::ColMajorTable)
                offset = Detail::ColumnRowOffset(row, logicalWidth);
            else if constexpr (layout_kind == RowLayoutKind::TiledColMajorTable)
                offset = Detail::TiledColumnRowOffset(row, logicalWidth, index(Policy::component_stride));
            else if constexpr (row_stride != DynamicExtent)
                offset = row * row_stride;
            else if constexpr (!is_dynamic && layout_kind != RowLayoutKind::CSR)
                offset = row * stride;
            else if constexpr (!is_dynamic && layout_kind == RowLayoutKind::CSR)
                offset = offsets ? offsets[row] : row * stride;
            else
                offset = offsets ? offsets[row] : row * stride;
            return data ? data + offset : nullptr;
        }
        DNDS_DEVICE_CALLABLE index ElementStride() const
        {
            if constexpr (Policy::component_stride != DynamicExtent)
                return index(Policy::component_stride);
            else if constexpr (is_dynamic)
                return kind == RowLayoutKind::ColMajorTable || kind == RowLayoutKind::TiledColMajorTable ? stride : 1;
            else
                return stride;
        }
        DNDS_DEVICE_CALLABLE T &operator()(index row, index component) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < rows && component >= 0 && component < RowSize(row));
            return Row(row)[component * ElementStride()];
        }
        DNDS_DEVICE_CALLABLE RowType operator[](index row) const { return {Row(row), RowSize(row), ElementStride()}; }
        DNDS_DEVICE_CALLABLE ArrayView<T, Checked> DynamicView() const { return {data, offsets, rows, width, stride, sizes, kind, device}; }
        /// Read-only projection of the same borrowed storage. This does not
        /// acquire access or extend lifetime; the original scope remains owner.
        DNDS_DEVICE_CALLABLE ArrayView<const std::remove_const_t<T>, Checked, LayoutPolicy> ConstView() const
        {
            return {data, offsets, rows, width, stride, sizes, kind, device};
        }
        template <bool C = Checked, std::enable_if_t<C, int> = 0>
        DNDS_DEVICE_CALLABLE operator ArrayView<T, false, LayoutPolicy>() const
        {
            return {data, offsets, rows, width, stride, sizes, kind, device};
        }
        template <class P = LayoutPolicy, std::enable_if_t<!std::is_same_v<P, RowLayout>, int> = 0>
        DNDS_DEVICE_CALLABLE operator ArrayView<T, Checked>() const { return DynamicView(); }
    };

    namespace Detail
    {
        /// Construct a typed view with a one-time projection check. Table
        /// storage may be exposed as padded/uniform CSR; variable storage may
        /// be exposed as a table/fixed view only when every row is uniform and
        /// physically contiguous. Metadata and values remain borrowed.
        template <class T, bool Checked, class Policy>
        ArrayView<T, Checked, Policy> ProjectView(ArrayView<T, Checked> source, const RowLayout &shape)
        {
            if constexpr (std::is_same_v<Policy, RowLayout>)
                return source;
            else
            {
                using Traits = RowPolicyTraits<Policy>;
                if constexpr (Traits::is_dynamic)
                {
                    Policy::Validate(shape);
                    return {source.data, source.offsets, source.rows, source.width, source.stride, source.sizes, source.kind, source.device};
                }
                if (!shape.IsContiguousRows())
                {
                    Policy::Validate(shape);
                    DNDS_check_throw_info(Traits::layout_kind == shape.Kind(), "noncontiguous rows need a matching physical view");
                    return {source.data, source.offsets, source.rows, source.width, source.stride, source.sizes, source.kind, source.device};
                }
                DNDS_check_throw_info(Traits::layout_kind != RowLayoutKind::ColMajorTable && Traits::layout_kind != RowLayoutKind::TiledColMajorTable,
                                      "column-major view requires actual column-major storage");
                const bool sourceTable = !source.offsets && !source.sizes;
                const bool sourcePadded = source.sizes;
                const bool sourceCSR = source.offsets;
                const bool keepPadded = Traits::layout_kind == RowLayoutKind::Padded && sourcePadded;
                const bool keepCSR = Traits::layout_kind == RowLayoutKind::CSR && sourceCSR;
                index width = source.width, stride = source.stride;
                const size_t *offsets = nullptr, *sizes = nullptr;
                if (keepPadded)
                    sizes = source.sizes;
                else if (keepCSR)
                    offsets = source.offsets;
                else if (!sourceTable)
                {
                    // Projection runs on the submitting host. The view may contain
                    // CUDA metadata pointers; only the retained host shape is read.
                    width = shape.RowCount() ? shape.RowSize(0) : 0;
                    for (index row = 0; row < source.rows; ++row)
                        DNDS_check_throw_info(shape.RowSize(row) == width &&
                                                  shape.Offset(row) == row * width,
                                              "row view projection requires uniform contiguous rows");
                    stride = width;
                }
                if constexpr (Traits::row_width != DynamicExtent)
                    DNDS_check_throw_info(width == Traits::row_width, "fixed row width view mismatch");
                if constexpr (Traits::row_stride != DynamicExtent)
                    DNDS_check_throw_info(stride == Traits::row_stride, "fixed row stride view mismatch");
                return {source.data, offsets, source.rows, width, stride, sizes, shape.Kind(), source.device};
            }
        }
    }

    /// Layout-generic debugging/setup view over table, padded or CSR storage.
    template <class T, bool Checked = AccessBoundsChecks>
    using GenericRowView = ArrayView<T, Checked, RowLayout>;

    /// Logical combined indexing without combining storage or access controllers.
    template <class T, bool Checked = AccessBoundsChecks, class LayoutPolicy = RowLayout>
    struct PairReadView
    {
        static constexpr size_t row_width = Detail::RowPolicyTraits<LayoutPolicy>::row_width;
        static constexpr size_t row_capacity = Detail::RowPolicyTraits<LayoutPolicy>::row_capacity;
        using RowType = RowSpan<const T, Checked, row_width, row_capacity, Detail::RowPolicyTraits<LayoutPolicy>::component_stride>;
        ArrayView<const T, Checked, LayoutPolicy> owned, ghost;
        /// Host-wrapper validation; owned and ghost must have compatible placement.
        void ValidateDevice(Device expected) const
        {
            owned.ValidateDevice(expected);
            ghost.ValidateDevice(expected);
        }
        DNDS_DEVICE_CALLABLE index RowCount() const { return owned.RowCount() + ghost.RowCount(); }
        DNDS_DEVICE_CALLABLE index Rows() const { return RowCount(); }
        DNDS_DEVICE_CALLABLE const T *Row(index row) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < RowCount());
            return row < owned.RowCount() ? owned.Row(row) : ghost.Row(row - owned.RowCount());
        }
        DNDS_DEVICE_CALLABLE index RowSize(index row) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < RowCount());
            return row < owned.RowCount() ? owned.RowSize(row) : ghost.RowSize(row - owned.RowCount());
        }
        DNDS_DEVICE_CALLABLE const T &operator()(index row, index column) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < RowCount() && column >= 0 && column < RowSize(row));
            return row < owned.RowCount() ? owned(row, column) : ghost(row - owned.RowCount(), column);
        }
        DNDS_DEVICE_CALLABLE RowType operator[](index row) const
        {
            return row < owned.RowCount() ? owned[row] : ghost[row - owned.RowCount()];
        }
        DNDS_DEVICE_CALLABLE PairReadView<T, Checked> DynamicView() const { return {owned.DynamicView(), ghost.DynamicView()}; }
        template <class P = LayoutPolicy, std::enable_if_t<!std::is_same_v<P, RowLayout>, int> = 0>
        DNDS_DEVICE_CALLABLE operator PairReadView<T, Checked>() const { return DynamicView(); }
    };

    /// Validate whole-array operands once at a numerical C++ entry boundary.
    /// Row views/spans deliberately carry no placement and cannot be used here.
    template <class... Views>
    void ValidateViewDevices(Device expected, const Views &...views)
    {
        DNDS_check_throw_info(expected.IsValid(), "array views: invalid execution device");
        (views.ValidateDevice(expected), ...);
    }
}
