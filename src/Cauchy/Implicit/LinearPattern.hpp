#pragma once

#include "DNDS/Runtime/Array/Array.hpp"

namespace DNDS::Cauchy
{
    /// Plain sparse-row and logical-scan input. Every index refers to the same
    /// owned+ghost row space; no communicator, allocation or lease lives here.
    struct LinearPatternView
    {
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::CSRRows> columns;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> order, position, core, dependent;
        index ownedRows = 0;
        /// C++ launcher check; no residency branches inside row arithmetic.
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, columns, order, position, core, dependent);
        }
    };

    /** @brief Canonical rank-local symbolic fields for matrix actions/scans.
     * @details Variable-length columns use a CSR Array. Logical order and
     * inverse positions are one-column Arrays. Setup vectors are discarded
     * after publication; numerical callers acquire one read scope per operation.
     */
    class LinearPatternStorage
    {
        Runtime::Array<index> _columns{Runtime::RowLayout::Ragged({})};
        Runtime::Array<index> _order{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> _position{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> _core{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> _dependent{Runtime::RowLayout::Dense(0, 1)};

    public:
        class ReadAccess
        {
            Runtime::Array<index>::ReadLease _columns, _order, _position, _core, _dependent;

        public:
            explicit ReadAccess(const LinearPatternStorage &storage, Runtime::ExecutionContext context)
                : _columns(storage._columns.Read(context)), _order(storage._order.Read(context)),
                  _position(storage._position.Read(context)), _core(storage._core.Read(context)),
                  _dependent(storage._dependent.Read(context)) {}
            LinearPatternView View() const
            {
                return {_columns.View<Runtime::CSRRows>(), _order.View<Runtime::FixedRows<1>>(),
                        _position.View<Runtime::FixedRows<1>>(), _core.View<Runtime::FixedRows<1>>(),
                        _dependent.View<Runtime::FixedRows<1>>(), _columns.Layout().RowCount()};
            }
        };
        LinearPatternStorage() = default;
        LinearPatternStorage(const std::vector<std::vector<index>> &columns,
                             const std::vector<index> &order, const std::vector<index> &position,
                             const std::vector<index> &core, const std::vector<index> &dependent);
        /// Retain this scope until the submitted operation completes.
        ReadAccess Read(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const { return ReadAccess(*this, context); }
        /// Completed setup-time placement. No host numerical payload is retained
        /// by the returned object; this does not migrate during an iteration.
        LinearPatternStorage Placed(Runtime::Device device, Runtime::ExecutionContext context) const;
        /// Immutable operand handles for queued operators. Copies retain the
        /// symbolic publication; callers must not write through an alias.
        const Runtime::Array<index> &ColumnsArray() const { return _columns; }
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
