#pragma once
/// @file ArrayEigenVector_DeviceView.hpp
/// @brief Device-callable view type for @ref DNDS::ArrayEigenVector "ArrayEigenVector"; `operator[]` returns
/// an `Eigen::Map<Vector>` suitable for use inside CUDA kernels.

#include "DNDS/Device/DeviceView.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/Device/DeviceStorage.hpp"
#include "Eigen/src/Core/util/Constants.h"

namespace DNDS
{
    /**
     * @brief Device-callable view onto @ref DNDS::ArrayEigenVector "ArrayEigenVector" rows.
     *
     * @details Extends the generic @ref DNDS::ArrayDeviceView "ArrayDeviceView" to yield an
     * `Eigen::Map<Vector>` on `operator[]`. The Eigen maps use
     * `Eigen::DontAlign` to avoid assumptions about the backing pointer's
     * alignment (which is device-allocator specific).
     *
     * @tparam real_T  `DNDS::real` or `const DNDS::real` for a const view.
     * @tparam _vec_size / _row_max / _align  Same meaning as in the host class.
     */
    template <DeviceBackend B, class real_T, rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    class ArrayEigenVectorDeviceView : public ArrayDeviceView<B, real_T, _vec_size, _row_max, _align>
    {
    public:
        using t_base = ArrayDeviceView<B, real_T, _vec_size, _row_max, _align>;
        using t_base::t_base;

        using t_self = ArrayEigenVectorDeviceView<B, real_T, _vec_size, _row_max, _align>;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ArrayEigenVectorDeviceView, t_self)

        DNDS_DEVICE_CALLABLE ArrayEigenVectorDeviceView(const t_base &base_view) : t_base(base_view) {};

        using t_EigenVector = Eigen::Matrix<std::remove_cv_t<real_T>, RowSize_To_EigenSize(_vec_size), 1,
                                            Eigen::DontAlign | Eigen::ColMajor, RowSize_To_EigenSize(_row_max), 1>;
        using t_EigenMap_Const = Eigen::Map<const t_EigenVector, Eigen::Unaligned>; // default no buffer align and stride
        using t_EigenMap =
            std::conditional_t<std::is_const_v<real_T>,
                               t_EigenMap_Const,
                               Eigen::Map<t_EigenVector, Eigen::Unaligned>>; // default no buffer align and stride

        DNDS_DEVICE_CALLABLE t_EigenMap operator[](index i)
        {
            return {t_base::operator[](i), t_base::RowSize(i)};
        }

        DNDS_DEVICE_CALLABLE t_EigenMap_Const operator[](index i) const
        {
            return {static_cast<const t_base &>(*this).operator[](i), t_base::RowSize(i)};
        }
    };
}