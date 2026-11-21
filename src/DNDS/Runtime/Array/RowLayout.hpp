#pragma once

#include "DNDS/Runtime/Storage/Buffer.hpp"
#include "DNDS/Runtime/Storage/Metadata.hpp"
#include <algorithm>
#include <vector>
#include <limits>

namespace DNDS::Runtime
{
    enum class RowLayoutKind
    {
        Table,
        Padded,
        CSR,
        ColMajorTable,
        TiledColMajorTable
    };

    struct ColMajorTableOptions
    {
        index rowPaddingMultiple = 1;
        index extraRows = 0;
    };

#ifndef DNDS_RUNTIME_ARRAY_BOUNDS_CHECKS
#    define DNDS_RUNTIME_ARRAY_BOUNDS_CHECKS 0
#endif
    inline constexpr bool AccessBoundsChecks = DNDS_RUNTIME_ARRAY_BOUNDS_CHECKS != 0;

    namespace Detail
    {
        // A zero-width column table has no first component plane. All its
        // empty rows start at offset zero, including in a partial final tile.
        DNDS_DEVICE_CALLABLE inline index ColumnRowOffset(index row, index width)
        {
            return width ? row : 0;
        }

        DNDS_DEVICE_CALLABLE inline index TiledColumnRowOffset(index row, index width, index tile)
        {
            return width ? (row / tile) * (width * tile) + row % tile : 0;
        }
    }

    /// Immutable row shape in element units. Padding belongs to storage, not a row.
    class RowLayout
    {
        size_t _rows = 0, _width = 0, _stride = 0;
        RowLayoutKind _kind = RowLayoutKind::Table;
        MetadataBuffer<size_t> _offsets;
        MetadataBuffer<size_t> _sizes;
        ColMajorTableOptions _padding;

    public:
        RowLayout() = default;
        static RowLayout Dense(size_t rows, size_t width, size_t stride = 0)
        {
            DNDS_check_throw_info(rows <= size_t(INT64_MAX) && width <= size_t(INT64_MAX) && stride <= size_t(INT64_MAX),
                                  "row shape exceeds signed index range");
            RowLayout layout;
            layout._rows = rows;
            layout._width = width;
            layout._stride = stride ? stride : width;
            DNDS_check_throw_info(layout._stride >= width, "row stride is smaller than width");
            DNDS_check_throw_info(CheckedSize::Multiply(rows, layout._stride) <= size_t(INT64_MAX),
                                  "row offsets exceed signed index range");
            return layout;
        }
        static RowLayout Padded(const std::vector<size_t> &lengths, size_t capacity)
        {
            DNDS_check_throw_info(lengths.size() <= size_t(INT64_MAX) && capacity <= size_t(INT64_MAX),
                                  "row shape exceeds signed index range");
            RowLayout layout;
            layout._kind = RowLayoutKind::Padded;
            layout._rows = lengths.size();
            layout._stride = capacity;
            DNDS_check_throw_info(CheckedSize::Multiply(layout._rows, capacity) <= size_t(INT64_MAX),
                                  "row offsets exceed signed index range");
            for (auto n : lengths)
                DNDS_check_throw_info(n <= capacity, "row size exceeds padded capacity");
            layout._sizes = MetadataBuffer<size_t>(lengths);
            return layout;
        }
        static RowLayout ColMajor(size_t rows, size_t width, ColMajorTableOptions options = {})
        {
            DNDS_check_throw_info(rows <= size_t(INT64_MAX) && width <= size_t(INT64_MAX) &&
                                      options.rowPaddingMultiple > 0 && options.extraRows >= 0,
                                  "invalid column-major table dimensions or padding");
            RowLayout layout;
            layout._kind = RowLayoutKind::ColMajorTable;
            layout._rows = rows;
            layout._width = width;
            layout._padding = options;
            if (rows)
            {
                const auto padded = CheckedSize::Add(rows, size_t(options.extraRows));
                const auto multiple = size_t(options.rowPaddingMultiple);
                layout._stride = CheckedSize::Multiply(CheckedSize::Add(padded, multiple - 1) / multiple, multiple);
            }
            DNDS_check_throw_info(layout._stride <= size_t(INT64_MAX) &&
                                      CheckedSize::Multiply(width, layout._stride) <= size_t(INT64_MAX),
                                  "column-major table extent overflow");
            return layout;
        }
        /// Import an existing physical pitch without deriving a new one.
        static RowLayout ColMajorWithPitch(size_t rows, size_t width, size_t pitch, ColMajorTableOptions options = {})
        {
            auto layout = ColMajor(rows, width, options);
            DNDS_check_throw_info(pitch >= rows && pitch <= size_t(INT64_MAX) &&
                                      CheckedSize::Multiply(width, pitch) <= size_t(INT64_MAX),
                                  "invalid column-major physical pitch");
            layout._stride = pitch;
            return layout;
        }
        static RowLayout TiledColMajor(size_t rows, size_t width, size_t tile)
        {
            DNDS_check_throw_info(rows <= size_t(INT64_MAX) && width <= size_t(INT64_MAX) && tile && tile <= size_t(INT64_MAX),
                                  "invalid tiled column-major table shape");
            RowLayout layout;
            layout._kind = RowLayoutKind::TiledColMajorTable;
            layout._rows = rows;
            layout._width = width;
            layout._stride = tile;
            DNDS_check_throw_info(CheckedSize::Multiply(width, tile) <= size_t(INT64_MAX) &&
                                      layout.Elements() <= size_t(INT64_MAX),
                                  "tiled table extent overflow");
            return layout;
        }
        static RowLayout Ragged(const std::vector<size_t> &lengths)
        {
            DNDS_check_throw_info(lengths.size() <= size_t(INT64_MAX), "row count exceeds signed index range");
            RowLayout layout;
            layout._kind = RowLayoutKind::CSR;
            layout._rows = lengths.size();
            size_t offset = 0;
            layout._offsets = MetadataBuffer<size_t>::Generate(CheckedSize::Add(lengths.size(), size_t(1)), [&](index i)
                                                               {
                if (i)
                {
                    const auto n = lengths[size_t(i - 1)];
                    DNDS_check_throw_info(n <= size_t(INT64_MAX) - offset, "row offsets exceed signed index range");
                    offset += n;
                }
                return offset; });
            return layout;
        }
        index RowCount() const { return index(_rows); }
        index Rows() const { return RowCount(); }
        RowLayoutKind Kind() const { return _kind; }
        bool IsRagged() const { return _kind == RowLayoutKind::CSR; }
        bool IsPadded() const { return _kind == RowLayoutKind::Padded; }
        bool IsVariable() const { return IsRagged() || IsPadded(); }
        bool IsContiguousRows() const { return _kind != RowLayoutKind::ColMajorTable && _kind != RowLayoutKind::TiledColMajorTable; }
        index Width() const { return index(_width); }
        index Stride() const { return index(_stride); }
        /// Physical pitch: AoS row capacity, SoA plane pitch, or AoSoA tile width.
        index ComponentStride() const { return IsContiguousRows() ? 1 : index(_stride); }
        ColMajorTableOptions PaddingOptions() const { return _padding; }
        size_t Elements() const
        {
            if (_kind == RowLayoutKind::ColMajorTable)
                return CheckedSize::Multiply(_width, _stride);
            if (_kind == RowLayoutKind::TiledColMajorTable)
                return CheckedSize::Multiply(CheckedSize::Multiply(_rows / _stride + (_rows % _stride != 0), _stride), _width);
            return IsRagged() ? _offsets.back() : _rows * _stride;
        }
        RowLayout WithRowCount(size_t rows) const
        {
            DNDS_check_throw_info(!IsVariable(), "uniform layout required for row-count rebinding");
            if (rows == _rows)
                return *this;
            if (_kind == RowLayoutKind::ColMajorTable)
                return ColMajor(rows, _width, _padding);
            if (_kind == RowLayoutKind::TiledColMajorTable)
                return TiledColMajor(rows, _width, _stride);
            return Dense(rows, _width, _stride);
        }
        const MetadataBuffer<size_t> &Offsets() const { return _offsets; }
        const MetadataBuffer<size_t> &Sizes() const { return _sizes; }
        index RowSize(index row) const
        {
            DNDS_check_throw_info(row >= 0 && size_t(row) < _rows, "row outside layout");
            return IsRagged() ? _offsets[row + 1] - _offsets[row] : IsPadded() ? _sizes[row]
                                                                               : _width;
        }
        index Offset(index row) const
        {
            DNDS_check_throw_info(row >= 0 && size_t(row) < _rows, "row outside layout");
            if (_kind == RowLayoutKind::ColMajorTable)
                return Detail::ColumnRowOffset(row, index(_width));
            if (_kind == RowLayoutKind::TiledColMajorTable)
                return Detail::TiledColumnRowOffset(row, index(_width), index(_stride));
            return IsRagged() ? _offsets[row] : row * _stride;
        }
        index ElementOffset(index row, index column) const
        {
            DNDS_check_throw_info(column >= 0 && column < RowSize(row), "column outside layout");
            return Offset(row) + column * ComponentStride();
        }
        bool operator==(const RowLayout &other) const
        {
            return _kind == other._kind && _rows == other._rows && _width == other._width && _stride == other._stride &&
                   _offsets == other._offsets && _sizes == other._sizes &&
                   _padding.rowPaddingMultiple == other._padding.rowPaddingMultiple && _padding.extraRows == other._padding.extraRows;
        }
    };

    inline constexpr size_t DynamicExtent = std::numeric_limits<size_t>::max();

    /// Layout-specific access with runtime widths/capacities. These policies
    /// remove layout dispatch from kernels without fixing the row extent.
    struct TableRows
    {
        static constexpr size_t row_width = DynamicExtent, row_capacity = DynamicExtent, row_stride = DynamicExtent;
        static constexpr size_t component_stride = 1;
        static constexpr RowLayoutKind layout_kind = RowLayoutKind::Table;
        static constexpr bool is_dynamic = false, is_static_max = false;
        static RowLayout Default() { return RowLayout::Dense(0, 0); }
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(layout.Kind() == layout_kind, "table row layout mismatch");
        }
    };

    struct PaddedRows
    {
        static constexpr size_t row_width = DynamicExtent, row_capacity = DynamicExtent, row_stride = DynamicExtent;
        static constexpr size_t component_stride = 1;
        static constexpr RowLayoutKind layout_kind = RowLayoutKind::Padded;
        static constexpr bool is_dynamic = false, is_static_max = false;
        static RowLayout Default() { return RowLayout::Padded({}, 0); }
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(layout.Kind() == layout_kind, "padded row layout mismatch");
        }
    };

    struct CSRRows
    {
        static constexpr size_t row_width = DynamicExtent, row_capacity = DynamicExtent, row_stride = DynamicExtent;
        static constexpr size_t component_stride = 1;
        static constexpr RowLayoutKind layout_kind = RowLayoutKind::CSR;
        static constexpr bool is_dynamic = false, is_static_max = false;
        static RowLayout Default() { return RowLayout::Ragged({}); }
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(layout.Kind() == layout_kind, "CSR row layout mismatch");
        }
    };

    /// Fixed logical width, with either a runtime stride or a static physical
    /// stride. The shape remains shared with dynamic language and MPI adapters.
    template <size_t Width, size_t Stride = DynamicExtent>
    struct FixedRows : TableRows
    {
        static_assert(Width != DynamicExtent && (Stride == DynamicExtent || Stride >= Width));
        static constexpr size_t row_width = Width, row_capacity = Stride, row_stride = Stride;
        static constexpr bool is_static_max = false;
        static RowLayout Default() { return RowLayout::Dense(0, Width, Stride == DynamicExtent ? Width : Stride); }
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(layout.Kind() == RowLayoutKind::Table && layout.Width() == Width, "fixed row width mismatch");
            DNDS_check_throw_info(Stride == DynamicExtent || layout.Stride() == Stride, "static row stride mismatch");
        }
    };

    /// Legacy static-max equivalent: variable logical lengths in fixed-capacity
    /// rows. Capacity is a compile-time bound and physical stride, never a width.
    template <size_t Capacity>
    struct StaticMaxRows : PaddedRows
    {
        static_assert(Capacity != DynamicExtent);
        static constexpr size_t row_width = DynamicExtent, row_capacity = Capacity, row_stride = Capacity;
        static constexpr bool is_static_max = true;
        static RowLayout Default() { return RowLayout::Padded({}, Capacity); }
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(layout.IsPadded() && layout.Stride() == Capacity, "static row capacity mismatch");
        }
    };

    struct ColMajorTable : TableRows
    {
        static constexpr RowLayoutKind layout_kind = RowLayoutKind::ColMajorTable;
        static constexpr size_t component_stride = DynamicExtent;
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(layout.Kind() == layout_kind, "column-major view layout mismatch");
        }
    };
    template <size_t W>
    struct TiledColMajorTable : TableRows
    {
        static_assert(W > 0 && W <= size_t(INT64_MAX));
        static constexpr RowLayoutKind layout_kind = RowLayoutKind::TiledColMajorTable;
        static constexpr size_t component_stride = W;
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(layout.Kind() == layout_kind && layout.Stride() == W, "tiled view layout mismatch");
        }
    };
    template <class Physical, size_t Width>
    struct FixedWidthRows : Physical
    {
        static constexpr size_t row_width = Width;
        static void Validate(const RowLayout &layout)
        {
            Physical::Validate(layout);
            DNDS_check_throw_info(layout.Width() == Width, "fixed logical row width mismatch");
        }
    };
    /// A uniform logical table with a runtime physical format. Row selection
    /// resolves its base/stride once; scalar row indexing has no kind dispatch.
    struct UniformTableRows : TableRows
    {
        static constexpr bool is_dynamic = true;
        static constexpr size_t component_stride = DynamicExtent;
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(!layout.IsVariable(), "uniform table view requires equal row sizes");
        }
    };

    namespace Detail
    {
        template <class Policy>
        struct RowPolicyTraits : Policy
        {
        };
        template <>
        struct RowPolicyTraits<RowLayout>
        {
            static constexpr size_t row_width = DynamicExtent, row_capacity = DynamicExtent, row_stride = DynamicExtent;
            static constexpr size_t component_stride = DynamicExtent;
            static constexpr RowLayoutKind layout_kind = RowLayoutKind::Table;
            static constexpr bool is_dynamic = true, is_static_max = false;
        };
    }

}
