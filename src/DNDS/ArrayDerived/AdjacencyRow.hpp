#pragma once
/// @file AdjacencyRow.hpp
/// @brief Non-owning row view for adjacency arrays.

#include "../Defines.hpp"

namespace DNDS
{
    /**
     * @brief Non-owning span `(pointer, size)` into an @ref DNDS::ArrayAdjacency "ArrayAdjacency" row.
     *
     * @details Serves as the return value of `ArrayAdjacency::operator[]`.
     * It is the "typed row handle" equivalent of `std::span<index_T>` with a
     * few DNDSR-specific conveniences:
     *  - indexing is bounds-checked in debug (@ref DNDS_assert);
     *  - assignment from / conversion to `std::vector<index>`;
     *  - device-callable accessors so kernels can walk adjacency rows.
     *
     * The span does not own the underlying storage; it must not outlive the
     * backing @ref DNDS::ArrayAdjacency "ArrayAdjacency".
     *
     * @tparam index_T  Either `DNDS::index` (mutable) or `const DNDS::index`.
     */
    template <typename index_T = index>
    class AdjacencyRow // instead of std::vector<index> for building on raw buffer as a "mapping" object
    {
        index_T *p_indices;
        rowsize Row_size{};

    public:
        //! the copy is not trivial!
        // DNDS_DEVICE_TRIVIAL_COPY_DEFINE(AdjacencyRow, AdjacencyRow)

        DNDS_DEVICE_CALLABLE AdjacencyRow() = default;
        DNDS_DEVICE_CALLABLE AdjacencyRow(const AdjacencyRow &) = default;
        DNDS_DEVICE_CALLABLE ~AdjacencyRow() = default;
        // Rule-of-five closure. The custom `operator=(const AdjacencyRow&)`
        // below returns `void` (it copies the pointed-to contents, not the
        // pointer), so the compiler does not synthesise the canonical move
        // operations. Restore them explicitly — moving this view wrapper is
        // just a pointer + size copy.
        DNDS_DEVICE_CALLABLE AdjacencyRow(AdjacencyRow &&) noexcept = default;
        DNDS_DEVICE_CALLABLE AdjacencyRow &operator=(AdjacencyRow &&) noexcept = default;
        /// @brief Construct a span from raw pointer and size.
        DNDS_DEVICE_CALLABLE AdjacencyRow(index_T *ptr, rowsize siz) : p_indices(ptr), Row_size(siz) {} // default actually

        /// @brief Bounds-checked (debug) element access.
        DNDS_DEVICE_CALLABLE index_T &operator[](rowsize j)
        {
            DNDS_assert(j >= 0 && j < Row_size);
            return p_indices[j];
        }

        DNDS_DEVICE_CALLABLE index_T operator[](rowsize j) const
        {
            DNDS_assert(j >= 0 && j < Row_size);
            return p_indices[j];
        }

        /// @brief Copy the row into a new `std::vector<index>`.
        operator std::vector<index>() const // copies to a new std::vector<index>
        {
            return {p_indices, p_indices + Row_size};
        }

        /// @brief Overwrite the row from a vector of the same size.
        void operator=(const std::vector<index> &r)
        {
            DNDS_assert(Row_size == r.size());
            std::copy(r.begin(), r.end(), p_indices);
        }

        /// @brief Copy contents of another span (same size required).
        /// @details Guarded against self-assignment: when `&r == this`,
        /// `p_indices` and `r.cbegin()` point into the same buffer, and
        /// `std::copy` on overlapping ranges is undefined behaviour.
        DNDS_DEVICE_CALLABLE void operator=(const AdjacencyRow &r)
        {
            if (this == &r)
                return;
            DNDS_assert(Row_size == r.size());
            std::copy(r.cbegin(), r.cend(), p_indices);
        }

        DNDS_DEVICE_CALLABLE index_T *begin() { return p_indices; }
        DNDS_DEVICE_CALLABLE index_T *end() { return p_indices + Row_size; } // past-end
        DNDS_DEVICE_CALLABLE [[nodiscard]] index_T *cbegin() const { return p_indices; }
        DNDS_DEVICE_CALLABLE [[nodiscard]] index_T *cend() const { return p_indices + Row_size; } // past-end
        /// @brief Row width in number of `index_T` elements.
        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize size() const { return Row_size; }
    };
}