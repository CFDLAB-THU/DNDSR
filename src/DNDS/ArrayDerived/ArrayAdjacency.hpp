#pragma once
#ifndef DNDS_ARRAY_PAIR_HPP
#    define DNDS_ARRAY_PAIR_HPP
#    include "DNDS/ArrayDerived/ArrayAdjacency_DeviceView.hpp"
#    include "DNDS/Defines.hpp"
#    include "DNDS/DeviceStorage.hpp"
#    include "../ArrayTransformer.hpp"

namespace DNDS
{
    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayAdjacency : public ParArray<index, _row_size, _row_max, _align>
    {
    public:
        using t_this = ArrayAdjacency<_row_size, _row_max, _align>;
        using t_base = ParArray<index, _row_size, _row_max, _align>;
        using t_base::t_base;

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
            return const_cast<t_this *>(this)->operator[](i);
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
    };

}

namespace DNDS
{
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
