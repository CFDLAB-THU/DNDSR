#pragma once
#ifndef DNDS_ARRAY_PAIR_HPP
#define DNDS_ARRAY_PAIR_HPP
#include "DNDS/ArrayDerived/ArrayAdjacency_DeviceView.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "../ArrayTransformer.hpp"

namespace DNDS
{
    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayAdjacency : public ParArray<index, _row_size, _row_max, _align>
    {
    public:
        using t_this = ArrayAdjacency<_row_size, _row_max, _align>;
        using t_base = ParArray<index, _row_size, _row_max, _align>;
        using t_base::t_base;

        AdjacencyRow operator[](index i)
        {
            DNDS_assert_info(
                i < this->Size(),
                fmt::format("i {}, Size {}, sig: {}",
                            i, this->Size(), this->GetArrayName())); //! disable past-end input
            return AdjacencyRow(t_base::operator[](i), t_base::RowSize(i));
        }

        const AdjacencyRow operator[](index i) const
        {
            return const_cast<t_this *>(this)->operator[](i);
        }

        index *rowPtr(index i) { return t_base::operator[](i); }

        template <DeviceBackend B>
        auto deviceView()
        {
            return ArrayAdjacencyDeviceView<B, _row_size, _row_max, _align>{t_base::template deviceView<B>()};
        }
    };

}

namespace DNDS
{
    class ArrayIndex : public ParArray<index, 1, 1, -1>
    {
    public:
        using t_base = ParArray<index, 1, 1, -1>;
        using t_base::t_base;

        index &operator[](index i)
        {
            DNDS_assert(i < this->Size()); //! disable past-end input
            return t_base::operator()(i, 0);
        }

        index *rowPtr(index i) { return t_base::operator[](i); }

        using t_base::ReadSerializer;
        using t_base::WriteSerializer; //! because no extra data than Array<>
    };
}
#endif
