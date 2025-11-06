#pragma once

#include "../DeviceView.hpp"

namespace DNDS
{
    template <rowsize _mat_ni, rowsize _mat_nj>
    constexpr rowsize __OneMatGetRowSize()
    {
        if constexpr (_mat_ni >= 0 && _mat_nj >= 0)
        {
            return _mat_ni * _mat_nj;
        }
        else if constexpr (_mat_ni == NonUniformSize || _mat_nj == NonUniformSize)
        {
            return NonUniformSize;
        }
        else
        {
            return DynamicSize;
        }
    }

    template <DeviceBackend B, class real_T, rowsize _mat_ni = 1, rowsize _mat_nj = 1,
              rowsize _mat_ni_max = _mat_ni, rowsize _mat_nj_max = _mat_nj, rowsize _align = NoAlign>
    class ArrayEigenMatrixDeviceView : public ArrayDeviceView<B, real_T,
                                                              __OneMatGetRowSize<_mat_ni, _mat_nj>(),
                                                              __OneMatGetRowSize<_mat_ni_max, _mat_nj_max>(),
                                                              _align>
    {
        const rowsize *_mat_nRows = nullptr;
        rowsize _mat_nRow_dynamic = 0;

    public:
        using t_base = ArrayDeviceView<B, real_T,
                                       __OneMatGetRowSize<_mat_ni, _mat_nj>(),
                                       __OneMatGetRowSize<_mat_ni_max, _mat_nj_max>(),
                                       _align>;
        // using t_base::t_base;
        using t_self = ArrayEigenMatrixDeviceView<B, real_T, _mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ArrayEigenMatrixDeviceView, t_self)

        DNDS_DEVICE_CALLABLE ArrayEigenMatrixDeviceView(const t_base &base_view,
                                                        const rowsize *n_mat_nRows, rowsize n_mat_nRow_dynamic)
            : t_base(base_view), _mat_nRows(n_mat_nRows), _mat_nRow_dynamic(n_mat_nRow_dynamic) {};

        using t_EigenMatrix = Eigen::Matrix<std::remove_cv_t<real_T>, RowSize_To_EigenSize(_mat_ni), RowSize_To_EigenSize(_mat_nj)>;
        using t_EigenMap_const = Eigen::Map<const t_EigenMatrix, Eigen::Unaligned>; // default no buffer align and stride
        using t_EigenMap = std::conditional_t<std::is_const_v<real_T>,
                                              t_EigenMap_const,
                                              Eigen::Map<t_EigenMatrix, Eigen::Unaligned>>; // default no buffer align and stride

        DNDS_DEVICE_CALLABLE void operator()(index i, rowsize j)
        {
            // just don't call
        }

        DNDS_DEVICE_CALLABLE t_EigenMap operator[](index i)
        {
            DNDS_HD_assert_infof(i >= 0 && i < this->Size(), "invalid index %lld / %lld", i, this->Size());
            rowsize c_nRow;
            if constexpr (_mat_ni == NonUniformSize)
                c_nRow = _mat_nRows[i];
            else if constexpr (_mat_ni == DynamicSize)
                c_nRow = _mat_nRow_dynamic;
            else
                c_nRow = _mat_ni;
            // std::cout << c_nRow << "  " << t_base::RowSize(i) << std::endl;

            return {t_base::operator[](i), c_nRow, t_base::RowSize(i) / c_nRow}; // need static dispatch?
        }

        DNDS_DEVICE_CALLABLE t_EigenMap_const operator[](index i) const
        {
            DNDS_HD_assert_infof(i >= 0 && i < this->Size(), "invalid index %lld / %lld", i, this->Size());
            rowsize c_nRow;
            if constexpr (_mat_ni == NonUniformSize)
                c_nRow = _mat_nRows[i];
            else if constexpr (_mat_ni == DynamicSize)
                c_nRow = _mat_nRow_dynamic;
            else
                c_nRow = _mat_ni;
            // std::cout << c_nRow << "  " << t_base::RowSize(i) << std::endl;

            return {t_base::operator[](i), c_nRow, t_base::RowSize(i) / c_nRow}; // need static dispatch?
        }
    };
}
