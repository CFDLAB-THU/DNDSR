#pragma once
/// @file ArrayAdjacency_DeviceView.hpp
/// @brief Device-callable view type for @ref DNDS::ArrayAdjacency "ArrayAdjacency". Mirrors the host
/// `operator[]` returning an @ref DNDS::AdjacencyRow "AdjacencyRow", but with `__device__` methods.

#include "AdjacencyRow.hpp"
#include "DNDS/Device/DeviceView.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/Device/DeviceStorage.hpp"

namespace DNDS
{
    /**
     * @brief Device-callable adjacency view: extends @ref DNDS::ArrayDeviceView "ArrayDeviceView" so that
     * indexing into a row returns an @ref DNDS::AdjacencyRow "AdjacencyRow" of indices.
     *
     * @tparam B         Target device backend.
     * @tparam index_T   `DNDS::index` or `const DNDS::index`.
     * @tparam _row_size / _row_max / _align  Same meaning as in @ref DNDS::Array "Array".
     */
    template <DeviceBackend B, typename index_T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayAdjacencyDeviceView : public ArrayDeviceView<B, index, _row_size, _row_max, _align>
    {
    public:
        using t_base = ArrayDeviceView<B, index_T, _row_size, _row_max, _align>;
        using t_base::t_base;

        using t_self = ArrayAdjacencyDeviceView<B, index_T, _row_size, _row_max, _align>;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ArrayAdjacencyDeviceView, t_self)

        DNDS_DEVICE_CALLABLE ArrayAdjacencyDeviceView(const t_base &base_view) : t_base(base_view) {};

        DNDS_DEVICE_CALLABLE AdjacencyRow<index_T> operator[](index i)
        {
            return {t_base::operator[](i), t_base::RowSize(i)};
        }

        DNDS_DEVICE_CALLABLE AdjacencyRow<const index_T> operator[](index i) const
        {
            return {t_base::operator[](i), t_base::RowSize(i)};
        }

        DNDS_DEVICE_CALLABLE index_T *rowPtr(index i) { return t_base::operator[](i); }
    };
}