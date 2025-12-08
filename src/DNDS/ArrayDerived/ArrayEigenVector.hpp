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
        using t_self = ArrayEigenVector<_vec_size, _row_max, _align>;
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
        // default copy
        ArrayEigenVector(const t_self &R) = default;
        t_self &operator=(const t_self &R) = default;
        // operator= handled automatically

        void clone(const t_self &R)
        {
            this->operator=(R);
        }

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

        template <DeviceBackend B>
        class iterator : public ArrayIteratorBase<iterator<B>>
        {
        public:
            using view_type = t_deviceView<B>;
            using t_base_iter = ArrayIteratorBase<iterator<B>>;
            using typename t_base_iter::difference_type;
            using reference = t_EigenMap;
            using iterator_category = std::random_access_iterator_tag;

        protected:
            view_type view;

        public:
            auto getView() const { return view; }
            DNDS_DEVICE_CALLABLE iterator(const iterator &) = default;
            DNDS_DEVICE_CALLABLE ~iterator() = default;
            DNDS_DEVICE_CALLABLE explicit iterator(const view_type &n_view, index n_iRow) : view(n_view), t_base_iter(n_iRow)
            {
            }

            DNDS_DEVICE_CALLABLE reference operator*() { return view.operator[](this->iRow); }
        };

        template <DeviceBackend B>
        iterator<B> begin()
        {
            return {deviceView<B>(), 0};
        }

        template <DeviceBackend B>
        iterator<B> end()
        {
            return {deviceView<B>(), this->Size()};
        }
    };
}
