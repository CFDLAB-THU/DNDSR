#pragma once

#include "../ArrayTransformer.hpp"
#include "ArrayEigenVector_DeviceView.hpp"
#include "DNDS/DeviceStorage.hpp"

namespace DNDS
{
    template <rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    class ArrayEigenVector : public ParArray<real, _vec_size, _row_max, _align>
    {
    public:
        using t_base = ParArray<real, _vec_size, _row_max, _align>;
        using t_base::t_base;

        template <DeviceBackend B>
        using t_deviceView = ArrayEigenVectorDeviceView<B, real, _vec_size, _row_max, _align>;
        template <DeviceBackend B>
        using t_deviceViewConst = ArrayEigenVectorDeviceView<B, const real, _vec_size, _row_max, _align>;

        using t_EigenVector = typename t_deviceView<DeviceBackend::Host>::t_EigenVector;
        using t_EigenMap = typename t_deviceView<DeviceBackend::Host>::t_EigenMap;
        using t_EigenMap_Const = typename t_deviceView<DeviceBackend::Host>::t_EigenMap_Const;

        using t_copy = t_EigenVector;

    public:
        t_EigenMap operator[](index i)
        {
            return {t_base::operator[](i), t_base::RowSize(i)}; // need static dispatch?
        }

        t_EigenMap_Const operator[](index i) const
        {
            return {t_base::operator[](i), t_base::RowSize(i)};
        }

        using t_base::ReadSerializer;
        using t_base::WriteSerializer; //! because no extra data than Array<>

        template <DeviceBackend B>
        auto deviceView()
        {
            return t_deviceView<B>{this->t_base::template deviceView<B>()};
        }

        template <DeviceBackend B>
        auto deviceView() const
        {
            return t_deviceViewConst<B>{this->t_base::template deviceView<B>()};
        }
    };
}
