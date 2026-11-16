#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"
#include <Eigen/Core>

namespace DNDS::Algebra
{
    /// Explicit CPU packing for a dense numerical kernel. Logical row/scalar
    /// indexing itself never packs or creates an Eigen map. The caller owns
    /// and sizes scratch before its loops; the returned reference borrows it
    /// until the next packing operation into the same scratch.
    template <class T, class Vector>
    auto PackEigenRow(DofComponentView<T> rows, index iRow, Vector &scratch)
    {
        for (index j = 0; j < rows.ComponentNumber(); ++j)
            scratch[j] = rows(iRow, j);
        return Eigen::Ref<const Vector>(scratch);
    }
    /// Explicit inverse of PackEigenRow. No allocation or synchronization;
    /// the caller retains the logical output's write lease through this copy.
    template <class Vector>
    void UnpackEigenRow(DofComponentView<real> rows, index iRow, const Vector &value)
    {
        for (index j = 0; j < rows.ComponentNumber(); ++j)
            rows(iRow, j) = value[j];
    }

    /// Borrowed per-row dense matrices. Construct once in the
    /// operator prologue while retaining the Array access lease. Scalar rows may
    /// be strided (SoA/AoSoA); no packing or allocation occurs. Row-major
    /// local blocks use a dynamic-stride view with rowMajor=true; ordinary
    /// column-major typed views retain static inner strides. CPU only.
    template <class ArrayView, int MatrixRows = Eigen::Dynamic, int MatrixColumns = Eigen::Dynamic>
    class EigenRowMatrices
    {
        ArrayView _view;
        index _matrixRows, _matrixColumns;
        bool _rowMajor;
        using Scalar = std::remove_const_t<typename ArrayView::value_type>;
        using Matrix = Eigen::Matrix<Scalar, MatrixRows, MatrixColumns>;
        using QualifiedMatrix = std::conditional_t<std::is_const_v<typename ArrayView::value_type>, const Matrix, Matrix>;
        static constexpr int Inner = ArrayView::component_stride == Runtime::DynamicExtent || ArrayView::component_stride > INT_MAX ? Eigen::Dynamic : int(ArrayView::component_stride);
        using Stride = Eigen::Stride<Eigen::Dynamic, Inner>;

    public:
        EigenRowMatrices(ArrayView view, index rows, index columns, Runtime::Device device, bool rowMajor = false)
            : _view(view), _matrixRows(rows), _matrixColumns(columns), _rowMajor(rowMajor)
        {
            DNDS_check_throw_info(device == Runtime::Device::Host() && rows >= 0 && columns >= 0,
                                  "Eigen row matrices require host placement and nonnegative dimensions");
            DNDS_check_throw_info((MatrixRows == Eigen::Dynamic || rows == MatrixRows) &&
                                      (MatrixColumns == Eigen::Dynamic || columns == MatrixColumns),
                                  "Eigen row matrix static dimensions differ from the requested shape");
            const auto extent = CheckedSize::Multiply(size_t(rows), size_t(columns));
            DNDS_check_throw_info(extent <= size_t(INT64_MAX), "local matrix extent exceeds signed index range");
            const auto componentStride = view.ElementStride();
            const index leading = rowMajor ? columns : rows;
            DNDS_check_throw_info(componentStride >= 0 && (!leading || componentStride <= INT64_MAX / std::max<index>(1, leading)),
                                  "local matrix stride exceeds signed index range");
            // Runtime block order needs an honest dynamic inner stride. Callers
            // use UniformTableRows for row-major blocks; ordinary typed AoS
            // views keep their compile-time unit stride and packetization.
            DNDS_check_throw_info(!rowMajor || Inner == Eigen::Dynamic || columns == 1,
                                  "row-major local matrices require a dynamic-inner-stride view");
            // This adapter is intended for uniform tables. Variable-row callers
            // construct an explicitly checked row adapter instead of rescanning
            // CSR lengths during every matrix access.
            DNDS_check_throw_info(!ArrayView::is_dynamic || (!view.offsets && !view.sizes),
                                  "Eigen row matrices require uniform logical rows");
            DNDS_check_throw_info(!view.RowCount() || view.Width() >= index(extent), "local matrix exceeds logical row width");
        }
        auto operator[](index iRow) const
        {
            const auto row = _view[iRow];
            const auto inner = row.ElementStride();
            return Eigen::Map<QualifiedMatrix, Eigen::Unaligned, Stride>(row.data(), _matrixRows, _matrixColumns,
                                                                         Stride(_rowMajor ? inner : _matrixRows * inner,
                                                                                _rowMajor ? _matrixColumns * inner : inner));
        }
        index RowNumber() const { return _view.RowCount(); }
    };

    /// Optional static matrix dimensions specialize arithmetic, never Array
    /// ownership/layout. Construction checks them once; row strides stay intact.
    template <int Rows = Eigen::Dynamic, int Columns = Eigen::Dynamic, class View>
    auto MakeEigenRowMatrices(View view, index rows, index columns, Runtime::Device device = Runtime::Device::Host(), bool rowMajor = false)
    {
        return EigenRowMatrices<View, Rows, Columns>(view, rows, columns, device, rowMajor);
    }
}

namespace DNDS::Algebra
{
    template <class T, bool Checked, class Policy>
    class EigenPairRowMatrices
    {
        EigenRowMatrices<Runtime::ArrayView<const T, Checked, Policy>> _owned, _ghost;
        index _ownedRows;

    public:
        EigenPairRowMatrices(Runtime::PairReadView<T, Checked, Policy> view, index rows, index columns, Runtime::Device device)
            : _owned(view.owned, rows, columns, device), _ghost(view.ghost, rows, columns, device), _ownedRows(view.owned.RowCount()) {}
        auto operator[](index iRow) const { return iRow < _ownedRows ? _owned[iRow] : _ghost[iRow - _ownedRows]; }
    };
    template <class T, bool Checked, class Policy>
    auto MakeEigenRowMatrices(Runtime::PairReadView<T, Checked, Policy> view, index rows, index columns,
                              Runtime::Device device = Runtime::Device::Host())
    {
        return EigenPairRowMatrices<T, Checked, Policy>(view, rows, columns, device);
    }
}
