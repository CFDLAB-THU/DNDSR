#pragma once
#include "../DeviceView.hpp"

namespace DNDS
{
    template <int a, int b>
    constexpr rowsize EigenSize_Mul_RowSize()
    {
        if constexpr (a >= 0 && b >= 0)
        {
            return a * b;
        }
        if constexpr (a == Eigen::Dynamic || b == Eigen::Dynamic)
        {
            return DynamicSize;
        }
        return DNDS_ROWSIZE_MIN;
    }

    template <DeviceBackend B, int _n_row, int _n_col>
    class ArrayEigenUniMatrixBatchDeviceView : public ArrayDeviceView<B, real, NonUniformSize>
    {
        static_assert(_n_row >= 0 || _n_row == Eigen::Dynamic, "invalid _n_row");
        static_assert(_n_col >= 0 || _n_col == Eigen::Dynamic, "invalid _n_col");

        using t_base = ArrayDeviceView<B, real, NonUniformSize>;
        using t_base::t_base;

        using t_EigenMatrix = Eigen::Matrix<real, _n_row, _n_col,
                                            Eigen::AutoAlign |
                                                ((_n_row == 1 && _n_col != 1) ? Eigen ::RowMajor : (_n_col == 1 && _n_row != 1) ? Eigen ::ColMajor // ColMajor except for row-vector
                                                                                                                                : Eigen ::ColMajor)>;
        using t_EigenMap = Eigen::Map<t_EigenMatrix, Eigen::Unaligned>;             // default no buffer align and stride
        using t_EigenMap_const = Eigen::Map<const t_EigenMatrix, Eigen::Unaligned>; // default no buffer align and stride

    private:
        int _row_dynamic = _n_row > 0 ? _n_row : 0;
        int _col_dynamic = _n_col > 0 ? _n_col : 0;
        int _m_size = this->Rows() * this->Cols(); //! extra data!

    public:
        DNDS_DEVICE_CALLABLE ArrayEigenUniMatrixBatchDeviceView(const t_base &base_view,
                                                                int n_row_dynamic, int n_col_dynamic, int n_m_size)
            : t_base(base_view), _row_dynamic(n_row_dynamic), _col_dynamic(n_col_dynamic), _m_size(n_m_size) {}

        DNDS_DEVICE_CALLABLE [[nodiscard]] int Rows() const { return _n_row > 0 ? _n_row : _row_dynamic; }
        DNDS_DEVICE_CALLABLE [[nodiscard]] int Cols() const { return _n_col > 0 ? _n_col : _col_dynamic; }
        DNDS_DEVICE_CALLABLE [[nodiscard]] int MSize() const
        {
            if constexpr (_n_row >= 0 && _n_col >= 0)
                return _n_row * _n_col;
            else
                return _m_size;
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize BatchSize(index i) const
        {
            return this->RowSize(i);
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize RowSize(index i) const
        {
            rowsize row_size_c = this->t_base::RowSize(i);
            DNDS_assert(MSize() != 0 && row_size_c % MSize() == 0);
            return row_size_c / MSize();
        }

        DNDS_DEVICE_CALLABLE t_EigenMap operator()(index i, rowsize j)
        {
            DNDS_assert(j >= 0 && j < this->RowSize(i));
            // if constexpr (_n_row >= 0 && _n_col >= 0)
            return {this->t_base::operator[](i) + MSize() * j, Rows(), Cols()};
        }

        DNDS_DEVICE_CALLABLE t_EigenMap_const operator()(index i, rowsize j) const
        {
            DNDS_assert(j >= 0 && j < this->RowSize(i));
            // if constexpr (_n_row >= 0 && _n_col >= 0)
            return {this->t_base::operator[](i) + MSize() * j, Rows(), Cols()};
        }
    };
}
