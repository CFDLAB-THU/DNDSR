#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Array/RowSelection.hpp"
#include <optional>

namespace DNDS::Cauchy
{
    enum class LinearRegion
    {
        All,
        Core,
        HaloDependent
    };
    /// Compress the existing logical scan order once. A permutation with more
    /// than two runs keeps its original order through the published index table.
    struct LinearRowSelections
    {
        Runtime::RowSelectionShape all, core, dependent;

        Runtime::RowSelectionShape For(LinearRegion region) const
        {
            return region == LinearRegion::All ? all : region == LinearRegion::Core ? core
                                                                                    : dependent;
        }
    };

    enum class LinearIndexStorage
    {
        Int64,
        Int32
    };
    /** Borrowed sparse columns with explicitly selected native index width.
     * The 32-bit representation is published directly during setup, never
     * manufactured by an AMG adapter. Numeric consumers use the same data.
     */
    struct LinearColumnsView
    {
        using Wide = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::CSRRows>;
        using Narrow = Runtime::ArrayView<const int, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
        Wide wide;
        Narrow offsets, columns;
        struct NarrowCSR
        {
            Narrow offsets, columns;
            DNDS_DEVICE_CALLABLE index RowCount() const { return offsets.RowCount() - 1; }
            DNDS_DEVICE_CALLABLE index RowSize(index iRow) const { return offsets(iRow + 1, 0) - offsets(iRow, 0); }
            DNDS_DEVICE_CALLABLE index operator()(index iRow, index j) const { return columns(offsets(iRow, 0) + j, 0); }
        };
        DNDS_DEVICE_CALLABLE LinearColumnsView() = default;
        DNDS_DEVICE_CALLABLE LinearColumnsView(Wide value) : wide(value) {}
        DNDS_DEVICE_CALLABLE LinearColumnsView(Narrow rowOffsets, Narrow columnIndices) : offsets(rowOffsets), columns(columnIndices) {}
        DNDS_DEVICE_CALLABLE bool IsInt32() const { return offsets.RowCount() != 0; }
        DNDS_DEVICE_CALLABLE index RowCount() const { return IsInt32() ? offsets.RowCount() - 1 : wide.RowCount(); }
        DNDS_DEVICE_CALLABLE index RowSize(index iRow) const { return IsInt32() ? offsets(iRow + 1, 0) - offsets(iRow, 0) : wide.RowSize(iRow); }
        DNDS_DEVICE_CALLABLE index operator()(index iRow, index j) const
        {
            return IsInt32() ? columns(offsets(iRow, 0) + j, 0) : wide(iRow, j);
        }
        struct Row
        {
            const index *wide = nullptr;
            const int *narrow = nullptr;
            index count = 0;
            struct Iterator
            {
                using value_type = index;
                using difference_type = index;
                using reference = index;
                using pointer = void;
                using iterator_category = std::random_access_iterator_tag;
                const index *wide = nullptr;
                const int *narrow = nullptr;
                index position = 0;
                DNDS_DEVICE_CALLABLE index operator*() const { return narrow ? narrow[position] : wide[position]; }
                DNDS_DEVICE_CALLABLE Iterator &operator++()
                {
                    ++position;
                    return *this;
                }
                DNDS_DEVICE_CALLABLE Iterator &operator--()
                {
                    --position;
                    return *this;
                }
                DNDS_DEVICE_CALLABLE Iterator &operator+=(index n)
                {
                    position += n;
                    return *this;
                }
                DNDS_DEVICE_CALLABLE Iterator &operator-=(index n)
                {
                    position -= n;
                    return *this;
                }
                DNDS_DEVICE_CALLABLE Iterator operator+(index n) const
                {
                    auto result = *this;
                    return result += n;
                }
                DNDS_DEVICE_CALLABLE Iterator operator-(index n) const
                {
                    auto result = *this;
                    return result -= n;
                }
                DNDS_DEVICE_CALLABLE index operator-(Iterator other) const { return position - other.position; }
                DNDS_DEVICE_CALLABLE bool operator==(Iterator other) const { return position == other.position; }
                DNDS_DEVICE_CALLABLE bool operator!=(Iterator other) const { return !(*this == other); }
                DNDS_DEVICE_CALLABLE bool operator<(Iterator other) const { return position < other.position; }
            };
            DNDS_DEVICE_CALLABLE index size() const { return count; }
            DNDS_DEVICE_CALLABLE index operator[](index j) const
            {
                if constexpr (Runtime::AccessBoundsChecks)
                    DNDS_HD_assert(j >= 0 && j < count);
                return narrow ? narrow[j] : wide[j];
            }
            DNDS_DEVICE_CALLABLE Iterator begin() const { return {wide, narrow, 0}; }
            DNDS_DEVICE_CALLABLE Iterator end() const { return {wide, narrow, count}; }
        };
        DNDS_DEVICE_CALLABLE Row operator[](index iRow) const
        {
            if (IsInt32())
                return {nullptr, columns.data + offsets(iRow, 0), RowSize(iRow)};
            return {wide[iRow].data(), nullptr, RowSize(iRow)};
        }
        void ValidateDevice(Runtime::Device device) const
        {
            if (IsInt32())
                Runtime::ValidateViewDevices(device, offsets, columns);
            else
                wide.ValidateDevice(device);
        }
    };
    /// Plain sparse-row and logical-scan input. Every index refers to the same
    /// owned+ghost row space; no communicator, allocation or lease lives here.
    struct LinearPatternView
    {
        LinearColumnsView columns;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> order, position, core, dependent;
        index ownedRows = 0;
        LinearRowSelections selections;

        Runtime::RowSelection ScanRows(LinearRegion region) const
        {
            return {selections.For(region), region == LinearRegion::All ? order : region == LinearRegion::Core ? core
                                                                                                               : dependent};
        }
        /// An independent matrix action can traverse all owned cells naturally,
        /// even when the triangular sweeps use a nonmonotone symbolic permutation.
        Runtime::RowSelection IndependentRows(LinearRegion region) const
        {
            return region == LinearRegion::All ? Runtime::RowSelection{Runtime::RowSelectionShape::AllRows(ownedRows)} : ScanRows(region);
        }
        /// C++ launcher check; no residency branches inside row arithmetic.
        void ValidateDevice(Runtime::Device device) const
        {
            columns.ValidateDevice(device);
            Runtime::ValidateViewDevices(device, order, position, core, dependent);
        }
    };

    /** @brief Canonical rank-local symbolic fields for matrix actions/scans.
     * @details Variable-length columns use a CSR Array. Logical order and
     * inverse positions are one-column Arrays. Setup vectors are discarded
     * after publication; numerical callers acquire one read scope per operation.
     */
    class LinearPatternStorage
    {
        LinearRowSelections _selections;
        Runtime::Array<index> _columns{Runtime::RowLayout::Ragged({})};
        std::optional<Runtime::Array<int>> _offsets32, _columns32;
        Runtime::Array<index> _order{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> _position{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> _core{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> _dependent{Runtime::RowLayout::Dense(0, 1)};

    public:
        class ReadAccess
        {
            LinearRowSelections _selections;
            Runtime::Array<index>::ReadLease _columns, _order, _position, _core, _dependent;
            std::optional<Runtime::Array<int>::ReadLease> _offsets32, _columns32;

        public:
            explicit ReadAccess(const LinearPatternStorage &storage, Runtime::ExecutionContext context)
                : _selections(storage._selections), _columns(storage._columns.Read(context)), _order(storage._order.Read(context)),
                  _position(storage._position.Read(context)), _core(storage._core.Read(context)),
                  _dependent(storage._dependent.Read(context))
            {
                if (storage._offsets32)
                {
                    _offsets32.emplace(storage._offsets32->Read(context));
                    _columns32.emplace(storage._columns32->Read(context));
                }
            }
            LinearPatternView View() const
            {
                const auto columns = _offsets32 ? LinearColumnsView{_offsets32->View<Runtime::FixedRows<1>>(), _columns32->View<Runtime::FixedRows<1>>()} : LinearColumnsView{_columns.View<Runtime::CSRRows>()};
                return {columns, _order.View<Runtime::FixedRows<1>>(),
                        _position.View<Runtime::FixedRows<1>>(), _core.View<Runtime::FixedRows<1>>(),
                        _dependent.View<Runtime::FixedRows<1>>(), columns.RowCount(), _selections};
            }
        };
        LinearPatternStorage() = default;
        LinearPatternStorage(const std::vector<std::vector<index>> &columns,
                             const std::vector<index> &order, const std::vector<index> &position,
                             const std::vector<index> &core, const std::vector<index> &dependent,
                             LinearIndexStorage indices = LinearIndexStorage::Int64);
        /// Retain this scope until the submitted operation completes.
        ReadAccess Read(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const { return ReadAccess(*this, context); }
        /// Completed setup-time placement. No host numerical payload is retained
        /// by the returned object; this does not migrate during an iteration.
        LinearPatternStorage Placed(Runtime::Device device, Runtime::ExecutionContext context) const;
        /// Immutable operand handles for queued operators. Copies retain the
        /// symbolic publication; callers must not write through an alias.
        const Runtime::Array<index> &ColumnsArray() const
        {
            DNDS_check_throw_info(!UsesInt32(), "int32 linear pattern has no int64 column mirror");
            return _columns;
        }
        LinearRowSelections Selections() const { return _selections; }
        bool UsesInt32() const { return bool(_columns32); }
        const Runtime::Array<int> &Offsets32() const
        {
            DNDS_check_throw_info(_offsets32, "linear pattern is not native int32");
            return *_offsets32;
        }
        const Runtime::Array<int> &Columns32() const
        {
            DNDS_check_throw_info(_columns32, "linear pattern is not native int32");
            return *_columns32;
        }
        const Runtime::Array<index> &OrderArray() const { return _order; }
        const Runtime::Array<index> &PositionArray() const { return _position; }
        const Runtime::Array<index> &CoreArray() const { return _core; }
        const Runtime::Array<index> &DependentArray() const { return _dependent; }
        /// Setup/diagnostic snapshot. Numerical operators use Read(), never a
        /// per-iteration host reconstruction of the symbolic pattern.
        std::vector<std::vector<index>> CopyColumns() const;
        std::vector<index> CopyOrder() const;
        std::vector<index> CopyCore() const;
        std::vector<index> CopyDependent() const;
    };
}
