#pragma once

#include "../DeviceView.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "Eigen/src/Core/util/Constants.h"

namespace DNDS
{
    template <DeviceBackend B, rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    class ArrayEigenVectorDeviceView : public ArrayDeviceView<B, real, _vec_size, _row_max, _align>
    {
    public:
        using t_base = ArrayDeviceView<B, real, _vec_size, _row_max, _align>;
        using t_base::t_base;

        using t_self = ArrayEigenVectorDeviceView<B, _vec_size, _row_max, _align>;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ArrayEigenVectorDeviceView, t_self)

        DNDS_DEVICE_CALLABLE ArrayEigenVectorDeviceView(const t_base &base_view) : t_base(base_view) {};

        using t_EigenVector = Eigen::Matrix<real, RowSize_To_EigenSize(_vec_size), 1,
                                            Eigen::DontAlign | Eigen::ColMajor, RowSize_To_EigenSize(_row_max), 1>;
        using t_EigenMap = Eigen::Map<t_EigenVector, Eigen::Unaligned>;             // default no buffer align and stride
        using t_EigenMap_Const = Eigen::Map<const t_EigenVector, Eigen::Unaligned>; // default no buffer align and stride

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