#pragma once
#ifndef DNDS_ARRAY_PAIR_HPP
#define DNDS_ARRAY_PAIR_HPP

#include "../ArrayTransformer.hpp"

namespace DNDS
{
    class AdjacencyRow // instead of std::vector<index> for building on raw buffer as a "mapping" object
    {
        index *__p_indices;
        rowsize __Row_size;

    public:
        AdjacencyRow(index *ptr, rowsize siz) : __p_indices(ptr), __Row_size(siz) {} // default actually

        index &operator[](rowsize j)
        {
            DNDS_assert(j >= 0 && j < __Row_size);
            return __p_indices[j];
        }

        index operator[](rowsize j) const
        {
            DNDS_assert(j >= 0 && j < __Row_size);
            return __p_indices[j];
        }

        operator std::vector<index>() const // copies to a new std::vector<index>
        {
            return std::vector<index>(__p_indices, __p_indices + __Row_size);
        }

        void operator=(const std::vector<index> &r)
        {
            DNDS_assert(__Row_size == r.size());
            std::copy(r.begin(), r.end(), __p_indices);
        }

        index *begin() { return __p_indices; }
        index *end() { return __p_indices + __Row_size; } // past-end
        rowsize size() const { return __Row_size; }
    };

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayAdjacency : public ParArray<index, _row_size, _row_max, _align>
    {
    public:
        using t_base = ParArray<index, _row_size, _row_max, _align>;
        using t_base::t_base;

        AdjacencyRow operator[](index i)
        {
            DNDS_assert_info(
                i < this->Size(),
                fmt::format("i {}, Size {}, sig: {}",
                            i, this->Size(), this->GetArraySignature())); //! disable past-end input
            return AdjacencyRow(t_base::operator[](i), t_base::RowSize(i));
        }

        index *rowPtr(index i) { return t_base::operator[](i); }
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
