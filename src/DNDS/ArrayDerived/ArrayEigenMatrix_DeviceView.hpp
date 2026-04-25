#pragma once
/// @file ArrayEigenMatrix_DeviceView.hpp
/// @brief Device-callable view for @ref DNDS::ArrayEigenMatrix "ArrayEigenMatrix". `operator[]` returns
/// an `Eigen::Map<Matrix<real, Ni, Nj>>`.

#include "DNDS/Device/DeviceView.hpp"
#include "../EigenUtil.hpp"

namespace DNDS
{
    /**
     * @brief Compute the underlying per-row element count for an `Ni x Nj`
     * matrix cell.
     *
     * @details Resolves to `Ni*Nj` when both are compile-time fixed,
     * @ref NonUniformSize when either uses per-row sizing, or @ref DynamicSize
     * for the remaining runtime-determined cases.
     */
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

    /**
     * @brief Device-callable view onto @ref DNDS::ArrayEigenMatrix "ArrayEigenMatrix" rows.
     *
     * @details Mirrors the host `operator[]` -> `Eigen::Map<Matrix>` but works
     * inside CUDA kernels. The underlying @ref DNDS::ArrayDeviceView "ArrayDeviceView" stores the flat row
     * of `Ni*Nj` reals; this class reinterprets that row as an Eigen matrix.
     *
     * @tparam real_T `DNDS::real` or `const DNDS::real`.
     */
    template <DeviceBackend B, class real_T, rowsize _mat_ni = 1, rowsize _mat_nj = 1,
              rowsize _mat_ni_max = _mat_ni, rowsize _mat_nj_max = _mat_nj, rowsize _align = NoAlign>
    class ArrayEigenMatrixDeviceView : public ArrayDeviceView<B, real_T,
                                                              __OneMatGetRowSize<_mat_ni, _mat_nj>(),
                                                              __OneMatGetRowSize<_mat_ni_max, _mat_nj_max>(),
                                                              _align>
    {
    public:
        using t_base = ArrayDeviceView<B, real_T,
                                       __OneMatGetRowSize<_mat_ni, _mat_nj>(),
                                       __OneMatGetRowSize<_mat_ni_max, _mat_nj_max>(),
                                       _align>;
        // using t_base::t_base;
        using t_self = ArrayEigenMatrixDeviceView<B, real_T, _mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>;

    protected:
        std::conditional_t<_mat_ni == DynamicSize, rowsize, EmptyNoDefault> _mat_nRow_dynamic = 0;
        std::conditional_t<_mat_ni == NonUniformSize, const rowsize *, EmptyNoDefault> _mat_nRows = nullptr;

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ArrayEigenMatrixDeviceView, t_self)

        DNDS_DEVICE_CALLABLE ArrayEigenMatrixDeviceView(const t_base &base_view,
                                                        const rowsize *n_mat_nRows, rowsize n_mat_nRow_dynamic)
            : t_base(base_view), _mat_nRow_dynamic(n_mat_nRow_dynamic), _mat_nRows(n_mat_nRows)
        {
            if constexpr (_mat_ni != NonUniformSize)
                DNDS_HD_assert(n_mat_nRows == nullptr);
            if constexpr (_mat_ni != DynamicSize)
                DNDS_HD_assert(n_mat_nRow_dynamic == 0);
        }

        using t_EigenMatrix = Eigen::Matrix<std::remove_cv_t<real_T>, RowSize_To_EigenSize(_mat_ni), RowSize_To_EigenSize(_mat_nj)>;
        using t_EigenMap_const = Eigen::Map<const t_EigenMatrix, Eigen::Unaligned>; // default no buffer align and stride
        using t_EigenMap = std::conditional_t<std::is_const_v<real_T>,
                                              t_EigenMap_const,
                                              Eigen::Map<t_EigenMatrix, Eigen::Unaligned>>; // default no buffer align and stride
        using t_EigenView = EigenMatrixView<B, real_T, RowSize_To_EigenSize(_mat_ni), RowSize_To_EigenSize(_mat_nj)>;
        using t_EigenView_const = EigenMatrixView<B, const real_T, RowSize_To_EigenSize(_mat_ni), RowSize_To_EigenSize(_mat_nj)>;

        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize MatRowSize(index iMat = 0) const
        {
            if constexpr (_mat_ni >= 0)
                return _mat_ni;
            if constexpr (_mat_ni == NonUniformSize)
            {
                DNDS_HD_assert(iMat >= 0 && iMat < this->Size());
                return _mat_nRows[iMat];
            }
            if constexpr (_mat_ni == DynamicSize)
                return _mat_nRow_dynamic;
            return UnInitRowsize; // invalid branch
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize MatColSize(index iMat = 0) const
        {
            if constexpr (_mat_nj >= 0)
                return _mat_nj;
            if constexpr (_mat_nj == NonUniformSize)
                return this->t_base::RowSize(iMat) / this->MatRowSize(iMat);
            if constexpr (_mat_nj == DynamicSize)
                return this->t_base::RowSize(iMat) / this->MatRowSize(iMat);
            return UnInitRowsize; // invalid branch
        }

#define DNDS_ARRAYEIGENMATRIXVIEW_GETTER_PREREQ                                                              \
    DNDS_HD_assert_infof(iRow >= 0 && iRow < this->Size(), "invalid index %lld / %lld", iRow, this->Size()); \
    rowsize c_nRow;                                                                                          \
    if constexpr (_mat_ni == NonUniformSize)                                                                 \
        c_nRow = _mat_nRows[iRow];                                                                           \
    else if constexpr (_mat_ni == DynamicSize)                                                               \
        c_nRow = _mat_nRow_dynamic;                                                                          \
    else                                                                                                     \
        c_nRow = _mat_ni;

        DNDS_DEVICE_CALLABLE t_EigenMap operator[](index iRow)
        {
            DNDS_ARRAYEIGENMATRIXVIEW_GETTER_PREREQ
            return {t_base::operator[](iRow), c_nRow, t_base::RowSize(iRow) / c_nRow}; // need static dispatch?
        }

        DNDS_DEVICE_CALLABLE t_EigenMap_const operator[](index iRow) const
        {
            DNDS_ARRAYEIGENMATRIXVIEW_GETTER_PREREQ
            return {t_base::operator[](iRow), c_nRow, t_base::RowSize(iRow) / c_nRow}; // need static dispatch?
        }

        DNDS_DEVICE_CALLABLE std::conditional_t<_mat_ni == 1 && _mat_nj == 1,
                                                real &, void>
        operator()(index iRow)
        {
            if constexpr (_mat_ni == 1 && _mat_nj == 1)
                return *t_base::operator[](iRow);
        }

        DNDS_DEVICE_CALLABLE std::conditional_t<_mat_ni == 1 && _mat_nj == 1,
                                                real, void>
        operator()(index iRow) const
        {
            if constexpr (_mat_ni == 1 && _mat_nj == 1)
                return *t_base::operator[](iRow);
        }

        DNDS_DEVICE_CALLABLE std::conditional_t<_mat_ni == 1 || _mat_nj == 1,
                                                real &, void>
        operator()(index iRow, rowsize j)
        {
            if constexpr (_mat_ni == 1 || _mat_nj == 1)
                return t_base::operator()(iRow, j);
        }

        DNDS_DEVICE_CALLABLE std::conditional_t<_mat_ni == 1 || _mat_nj == 1,
                                                real, void>
        operator()(index iRow, rowsize j) const
        {
            if constexpr (_mat_ni == 1 || _mat_nj == 1)
                return t_base::operator()(iRow, j);
        }

        DNDS_DEVICE_CALLABLE t_EigenView MatView(index iRow)
        {
            DNDS_ARRAYEIGENMATRIXVIEW_GETTER_PREREQ
            return {t_base::operator[](iRow), c_nRow, t_base::RowSize(iRow) / c_nRow}; // need static dispatch?
        }

        DNDS_DEVICE_CALLABLE t_EigenView MatView(index iRow) const
        {
            DNDS_ARRAYEIGENMATRIXVIEW_GETTER_PREREQ
            return {t_base::operator[](iRow), c_nRow, t_base::RowSize(iRow) / c_nRow}; // need static dispatch?
        }
    };
}
