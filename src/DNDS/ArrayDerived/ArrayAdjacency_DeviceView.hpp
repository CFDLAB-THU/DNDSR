#pragma once

#include "AdjacencyRow.hpp"
#include "../DeviceView.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"

namespace DNDS
{
    template <DeviceBackend B, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayAdjacencyDeviceView : public ArrayDeviceView<B, index, _row_size, _row_max, _align>
    {
    public:
        using t_base = ArrayDeviceView<B, index, _row_size, _row_max, _align>;
        using t_base::t_base;

        ArrayAdjacencyDeviceView(const t_base &base_view) : t_base(base_view) {};

        DNDS_DEVICE_CALLABLE AdjacencyRow operator[](index i)
        {
            return AdjacencyRow(t_base::operator[](i), t_base::RowSize(i));
        }

        DNDS_DEVICE_CALLABLE index *rowPtr(index i) { return t_base::operator[](i); }
    };
}