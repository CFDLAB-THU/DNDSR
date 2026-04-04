#pragma once
/// @file ArrayAdjacency.hpp
/// @brief Adjacency array (CSR-like index storage) built on ParArray.
/// @par Unit Test Coverage (test_ArrayDerived.cpp, MPI np=1,2,4)
/// - Basics: Resize, ResizeRow, Compress, RowSize, operator[], rowPtr
/// - Ghost communication: pull-based ghost exchange verifying row sizes and values
/// - Clone independence
/// - Fixed-size variant: ArrayAdjacency<3> (TABLE_StaticFixed)
/// @par Not Yet Tested
/// - ArrayIndex (ArrayAdjacency<1> subclass)
/// - AdjacencyRow::operator= from std::vector, conversion operator
/// - Device views
#ifndef DNDS_ARRAY_PAIR_HPP
#    define DNDS_ARRAY_PAIR_HPP
#    include "DNDS/ArrayDerived/ArrayAdjacency_DeviceView.hpp"
#    include "DNDS/Defines.hpp"
#    include "DNDS/DeviceStorage.hpp"
#    include "../ArrayTransformer.hpp"

namespace DNDS
{
    /// @brief CSR-like index array for mesh connectivity, extending ParArray<index>.
    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayAdjacency : public ParArray<index, _row_size, _row_max, _align>
    {
    public:
        using t_self = ArrayAdjacency<_row_size, _row_max, _align>;
        using t_base = ParArray<index, _row_size, _row_max, _align>;
        using t_base::t_base;

        // default copy
        ArrayAdjacency(const t_self &R) = default;
        t_self &operator=(const t_self &R) = default;
        // operator= handled automatically

        void clone(const t_self &R)
        {
            this->operator=(R);
        }

        AdjacencyRow<index> operator[](index i)
        {
            DNDS_assert_info(
                i < this->Size(),
                fmt::format("i {}, Size {}, sig: {}",
                            i, this->Size(), this->GetArrayName())); //! disable past-end input
            return AdjacencyRow(t_base::operator[](i), t_base::RowSize(i));
        }

        AdjacencyRow<const index> operator[](index i) const
        {
            return const_cast<t_self *>(this)->operator[](i);
        }

        index *rowPtr(index i) { return t_base::operator[](i); }

        template <DeviceBackend B>
        using t_deviceView = ArrayAdjacencyDeviceView<B, index, _row_size, _row_max, _align>;

        template <DeviceBackend B>
        using t_deviceViewConst = ArrayAdjacencyDeviceView<B, const index, _row_size, _row_max, _align>;

        template <DeviceBackend B>
        auto deviceView()
        {
            return t_deviceView<B>{t_base::template deviceView<B>()};
        }

        template <DeviceBackend B>
        auto deviceView() const
        {
            return t_deviceViewConst<B>{this->t_base::template deviceView<B>()};
        }

        using t_base::to_device;
        using t_base::to_host;

        /// @brief Row iterator for ArrayAdjacency, yielding AdjacencyRow per element.
        template <DeviceBackend B>
        class iterator : public ArrayIteratorBase<iterator<B>>
        {
        public:
            using view_type = t_deviceView<B>;
            using t_base_iter = ArrayIteratorBase<iterator<B>>;
            using typename t_base_iter::difference_type;
            using reference = AdjacencyRow<index>;
            using iterator_category = std::random_access_iterator_tag;

        protected:
            view_type view;

        public:
            auto getView() const { return view; }
            DNDS_DEVICE_CALLABLE iterator(const iterator &) = default;
            DNDS_DEVICE_CALLABLE ~iterator() = default;
            DNDS_DEVICE_CALLABLE iterator(const view_type &n_view, index n_iRow) : view(n_view), t_base_iter(n_iRow)
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

namespace DNDS
{
    /// @brief Single-column index array (ArrayAdjacency with fixed row size 1).
    class ArrayIndex : public ArrayAdjacency<1>
    {
    public:
        using t_base = ArrayAdjacency<1>;
        using t_base::t_base;

        index &operator[](index i)
        {
            DNDS_assert(i < this->Size()); //! disable past-end input
            return t_base::operator()(i, 0);
        }

        index *rowPtr(index i) { return t_base::rowPtr(i); }

        using t_base::ReadSerializer;
        using t_base::WriteSerializer; //! because no extra data than Array<>
    };
}
#endif
