#pragma once
#include "../Defines.hpp"

namespace DNDS
{
    template <typename index_T = index>
    class AdjacencyRow // instead of std::vector<index> for building on raw buffer as a "mapping" object
    {
        index_T *__p_indices;
        rowsize __Row_size;

    public:
        //! the copy is not trivial!
        // DNDS_DEVICE_TRIVIAL_COPY_DEFINE(AdjacencyRow, AdjacencyRow)

        DNDS_DEVICE_CALLABLE AdjacencyRow() = default;
        DNDS_DEVICE_CALLABLE AdjacencyRow(const AdjacencyRow &) = default;
        DNDS_DEVICE_CALLABLE ~AdjacencyRow() = default;
        DNDS_DEVICE_CALLABLE AdjacencyRow(index_T *ptr, rowsize siz) : __p_indices(ptr), __Row_size(siz) {} // default actually

        DNDS_DEVICE_CALLABLE index_T &operator[](rowsize j)
        {
            DNDS_assert(j >= 0 && j < __Row_size);
            return __p_indices[j];
        }

        DNDS_DEVICE_CALLABLE index_T operator[](rowsize j) const
        {
            DNDS_assert(j >= 0 && j < __Row_size);
            return __p_indices[j];
        }

        operator std::vector<index>() const // copies to a new std::vector<index>
        {
            return {__p_indices, __p_indices + __Row_size};
        }

        void operator=(const std::vector<index> &r)
        {
            DNDS_assert(__Row_size == r.size());
            std::copy(r.begin(), r.end(), __p_indices);
        }

        DNDS_DEVICE_CALLABLE void operator=(const AdjacencyRow &r)
        {
            DNDS_assert(__Row_size == r.size());
            std::copy(r.cbegin(), r.cend(), __p_indices);
        }

        DNDS_DEVICE_CALLABLE index_T *begin() { return __p_indices; }
        DNDS_DEVICE_CALLABLE index_T *end() { return __p_indices + __Row_size; } // past-end
        DNDS_DEVICE_CALLABLE index_T *cbegin() const { return __p_indices; }
        DNDS_DEVICE_CALLABLE index_T *cend() const { return __p_indices + __Row_size; } // past-end
        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize size() const { return __Row_size; }
    };
}