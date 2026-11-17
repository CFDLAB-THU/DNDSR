#pragma once

#include "DNDS/Algebra/BlockIndex.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Algebra/EigenView.hpp"
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace DNDS::Algebra
{
    /// Physical storage of independent matrix blocks. State-vector layout is
    /// deliberately independent of these choices.
    enum class MatrixTableLayout
    {
        AoS,
        SoA,
        AoSoA32
    };
    struct MatrixTableOptions
    {
        MatrixTableLayout layout = MatrixTableLayout::AoS;
        /// SoA physical row pitch; AoS and tiled32 determine their own pitch.
        index rowPaddingMultiple = 1;
        /// AoS row stride rounded to this many real elements, not bytes.
        /// Logical row width is unchanged. Other physical layouts ignore it.
        index rowStridePaddingMultiple = 1;
        /// Base address alignment in bytes; zero selects the allocator default.
        size_t allocationAlignmentBytes = 0;
        Runtime::AllocationOptions Allocation() const { return {allocationAlignmentBytes}; }
        Runtime::RowLayout MakeLayout(index rows, index width) const
        {
            DNDS_check_throw_info(rows >= 0 && width >= 0 && rowPaddingMultiple > 0 && rowStridePaddingMultiple > 0,
                                  "invalid matrix table dimensions or padding");
            if (layout == MatrixTableLayout::AoS)
            {
                const auto stride = CheckedSize::Multiply(CheckedSize::Add(size_t(width), size_t(rowStridePaddingMultiple) - 1) /
                                                              size_t(rowStridePaddingMultiple),
                                                          size_t(rowStridePaddingMultiple));
                return Runtime::RowLayout::Dense(rows, width, stride);
            }
            if (layout == MatrixTableLayout::SoA)
                return Runtime::RowLayout::ColMajor(rows, width, {rowPaddingMultiple, 0});
            DNDS_check_throw_info(layout == MatrixTableLayout::AoSoA32, "invalid matrix table layout");
            return Runtime::RowLayout::TiledColMajor(rows, width, 32);
        }
    };
    /** @brief Borrowed dense-block rows for CPU and device kernels.
     * @details Each value row is one square block in the declared dense order. Sparse-row
     * offsets are field data with the same residency as values. The owner must
     * retain both leases through completion. No Eigen object or owning handle
     * crosses a CUDA launch boundary.
     */
    template <class Scalar>
    struct BlockMatrixArrayView
    {
        Runtime::ArrayView<Scalar, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> values;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> offsets;
        index blockSize = 0;
        bool rowMajorBlocks = false;
        DNDS_DEVICE_CALLABLE index RowNumber() const { return offsets.RowCount() - 1; }
        DNDS_DEVICE_CALLABLE index BlockNumber(index iRow) const
        {
            if constexpr (Runtime::AccessBoundsChecks)
                DNDS_HD_assert(iRow >= 0 && iRow < RowNumber());
            return offsets(iRow + 1, 0) - offsets(iRow, 0);
        }
        DNDS_DEVICE_CALLABLE auto operator()(index iRow, index iBlock) const
        {
            if constexpr (Runtime::AccessBoundsChecks)
                DNDS_HD_assert(iRow >= 0 && iRow < RowNumber() && iBlock >= 0 && iBlock < BlockNumber(iRow));
            return values[offsets(iRow, 0) + iBlock];
        }
        DNDS_DEVICE_CALLABLE Scalar &Element(index iRow, index iBlock, index i, index j) const
        {
            if constexpr (Runtime::AccessBoundsChecks)
                DNDS_HD_assert(i >= 0 && i < blockSize && j >= 0 && j < blockSize);
            return (*this)(iRow, iBlock)[DenseBlockEntry(i, j, blockSize, rowMajorBlocks)];
        }
        /// Validate once at the host launcher, never per matrix entry.
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, values, offsets);
        }
    };

    /// Borrowed CPU dense-block access. Retain the storage access object while
    /// using this view. Logical sparse rows contain consecutive dense blocks;
    /// each block uses the centralized strided Eigen adapter without packing.
    template <class Scalar, class Policy = Runtime::UniformTableRows>
    class BlockMatrixView
    {
        using ArrayView = Runtime::ArrayView<Scalar, Runtime::AccessBoundsChecks, Policy>;
        EigenRowMatrices<ArrayView> _blocks;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> _offsets;
        index _rows, _componentStride;
        bool _rowMajorBlocks;
        index _blockSize;

    public:
        explicit BlockMatrixView(BlockMatrixArrayView<Scalar> view)
            : BlockMatrixView(ArrayView{view.values.data, view.values.offsets, view.values.rows, view.values.width, view.values.stride,
                                        view.values.sizes, view.values.kind, view.values.device},
                              view.offsets, view.blockSize, view.rowMajorBlocks)
        {
            view.ValidateDevice(Runtime::Device::Host());
        }
        BlockMatrixView(ArrayView values, decltype(_offsets) offsets, index blockSize, bool rowMajorBlocks = false)
            : _blocks(values, blockSize, blockSize, Runtime::Device::Host(), rowMajorBlocks), _offsets(offsets),
              _rows(offsets.RowCount() - 1), _componentStride(values.ElementStride()), _rowMajorBlocks(rowMajorBlocks), _blockSize(blockSize) {}
        index RowNumber() const { return _rows; }
        /// Scalar spacing inside each dense block; contiguous-only leaf
        /// specializations must test this once before their row loop.
        index ElementStride() const { return _componentStride; }
        bool RowMajorBlocks() const { return _rowMajorBlocks; }
        /// Resolve physical row policy once before numerical scans. Canonical
        /// AoS column-major blocks keep Eigen's compile-time unit inner stride;
        /// experimental layouts use honest dynamic strides without packing.
        template <class Callback>
        void WithLayout(Callback &&callback) const
        {
            const auto values = _blocks.ValuesView();
            if constexpr (std::is_same_v<Policy, Runtime::UniformTableRows>)
            {
                if (values.kind == Runtime::RowLayoutKind::Table && !_rowMajorBlocks)
                {
                    using Contiguous = Runtime::ArrayView<Scalar, Runtime::AccessBoundsChecks, Runtime::TableRows>;
                    callback(BlockMatrixView<Scalar, Runtime::TableRows>{Contiguous{values.data, nullptr, values.rows, values.width,
                                                                                    values.stride, nullptr, values.kind, values.device},
                                                                         _offsets, _blockSize});
                    return;
                }
            }
            callback(*this);
        }
        index BlockNumber(index iRow) const
        {
            if constexpr (Runtime::AccessBoundsChecks)
                DNDS_assert(iRow >= 0 && iRow < _rows);
            return _offsets(iRow + 1, 0) - _offsets(iRow, 0);
        }
        auto operator()(index iRow, index iBlock) const
        {
            if constexpr (Runtime::AccessBoundsChecks)
                DNDS_assert(iRow >= 0 && iRow < _rows && iBlock >= 0 && iBlock < BlockNumber(iRow));
            return _blocks[_offsets(iRow, 0) + iBlock];
        }
    };

    /// Native numerical storage, independent of mesh adjacency and factorization.
    /// Shared copies retain both the Array allocation and immutable block-row
    /// offsets. All ownership/leases are acquired outside block arithmetic.
    class BlockMatrixStorage
    {
        struct Layout
        {
            index rows;
            index blockSize;
            bool rowMajorBlocks;
            Runtime::AllocationOptions allocation;
        };
        std::shared_ptr<const Layout> _layout;
        Runtime::Array<real> _values;
        Runtime::Array<index> _offsets;
        BlockMatrixStorage(std::shared_ptr<const Layout> layout, Runtime::Array<real> values, Runtime::Array<index> offsets)
            : _layout(std::move(layout)), _values(std::move(values)), _offsets(std::move(offsets)) {}

    public:
        class ReadAccess
        {
            std::shared_ptr<const Layout> _layout;
            Runtime::Array<real>::ReadLease _lease;
            Runtime::Array<index>::ReadLease _offsets;

        public:
            ReadAccess(std::shared_ptr<const Layout> layout, Runtime::Array<real>::ReadLease lease, Runtime::Array<index>::ReadLease offsets)
                : _layout(std::move(layout)), _lease(std::move(lease)), _offsets(std::move(offsets)) {}
            BlockMatrixArrayView<const real> ArrayView() const
            {
                return {_lease.View<Runtime::UniformTableRows>(), _offsets.View<Runtime::FixedRows<1>>(), _layout->blockSize, _layout->rowMajorBlocks};
            }
            auto View() const
            {
                return BlockMatrixView<const real>(ArrayView());
            }
        };
        class WriteAccess
        {
            std::shared_ptr<const Layout> _layout;
            Runtime::Array<real>::WriteLease _lease;
            Runtime::Array<index>::ReadLease _offsets;

        public:
            WriteAccess(std::shared_ptr<const Layout> layout, Runtime::Array<real>::WriteLease lease, Runtime::Array<index>::ReadLease offsets)
                : _layout(std::move(layout)), _lease(std::move(lease)), _offsets(std::move(offsets)) {}
            BlockMatrixArrayView<real> ArrayView() const
            {
                return {_lease.View<Runtime::UniformTableRows>(), _offsets.View<Runtime::FixedRows<1>>(), _layout->blockSize, _layout->rowMajorBlocks};
            }
            auto View() const
            {
                return BlockMatrixView<real>(ArrayView());
            }
            /// Match Array write transactions: cancel only before mutation;
            /// fail a partially filled matrix rather than publishing its values.
            void CancelUnmodified() { _lease.CancelUnmodified(); }
            void Fail() { _lease.Fail(); }
        };

        BlockMatrixStorage(const std::vector<index> &blocksPerRow, index blockSize,
                           Runtime::Device device = Runtime::Device::Host(), MatrixTableOptions options = {}, bool rowMajorBlocks = false,
                           std::optional<Runtime::Array<real>> values = {})
        {
            DNDS_check_throw_info(blockSize > 0 && blockSize <= std::numeric_limits<rowsize>::max() / blockSize,
                                  "invalid dense block dimension");
            auto layout = std::make_shared<Layout>();
            layout->blockSize = blockSize;
            layout->rowMajorBlocks = rowMajorBlocks;
            layout->allocation = options.Allocation();
            layout->rows = index(blocksPerRow.size());
            _offsets = Runtime::Array<index>(Runtime::RowLayout::Dense(layout->rows + 1, 1));
            index totalBlocks = 0;
            {
                auto write = _offsets.Write();
                const auto offsets = write.View<Runtime::FixedRows<1>>();
                offsets(0, 0) = 0;
                for (index iRow = 0; iRow < layout->rows; ++iRow)
                {
                    const index count = blocksPerRow[iRow];
                    DNDS_check_throw_info(count >= 0 && count <= std::numeric_limits<index>::max() - totalBlocks,
                                          "invalid block row length or total block overflow");
                    totalBlocks += count;
                    offsets(iRow + 1, 0) = totalBlocks;
                }
            }
            if (values)
            {
                DNDS_check_throw_info(values->GetDevice() == device && values->RowCount() == totalBlocks &&
                                          values->Layout().Width() == blockSize * blockSize,
                                      "shared block-matrix values differ from the declared block shape/device");
                _values = std::move(*values);
            }
            else
                _values = Runtime::Array<real>(options.MakeLayout(totalBlocks, blockSize * blockSize), device, layout->allocation);
            if (device != Runtime::Device::Host())
                _offsets = _offsets.To(device);
            _layout = std::move(layout);
        }
        index RowNumber() const { return _layout->rows; }
        index BlockSize() const { return _layout->blockSize; }
        bool RowMajorBlocks() const { return _layout->rowMajorBlocks; }
        /// Host setup/diagnostic snapshot; kernels use retained ArrayView().
        /// Device storage must be explicitly placed on host before this call.
        std::vector<index> Offsets() const
        {
            const auto read = _offsets.Read();
            const auto view = read.View<Runtime::FixedRows<1>>();
            std::vector<index> result(view.RowCount());
            for (index iRow = 0; iRow < view.RowCount(); ++iRow)
                result[iRow] = view(iRow, 0);
            return result;
        }
        const Runtime::Array<index> &OffsetArray() const { return _offsets; }
        const Runtime::Array<real> &Values() const { return _values; }
        Runtime::Device GetDevice() const { return _values.GetDevice(); }
        /// Explicit completed placement/snapshot. No host payload mirror is
        /// retained; only the small immutable dimension descriptor is shared.
        BlockMatrixStorage Placed(Runtime::Device device, Runtime::ExecutionContext context) const
        {
            Runtime::Array<real> values(_values.Layout(), device, _layout->allocation);
            if (!_values.Name().empty())
                values.SetName(_values.Name());
            values.CopyFrom(_values, context).Wait();
            return {_layout, std::move(values), _offsets.To(device, context)};
        }
        ReadAccess Read(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const
        {
            return {_layout, _values.Read(context), _offsets.Read(context)};
        }
        WriteAccess Write(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host())
        {
            return {_layout, _values.Write(context), _offsets.Read(context)};
        }
    };
}
