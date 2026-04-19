#pragma once
/// @file VectorUtils.hpp
/// @brief Small utilities for MPI-indexed type layouts (hindexed optimisation).

#include "Defines.hpp"
#include "MPI.hpp"

namespace DNDS
{
    /**
     * @brief Coalesce contiguous blocks in an `MPI_Type_create_hindexed` layout.
     *
     * @details MPI derived-type performance depends heavily on the number of
     * blocks. If two consecutive blocks happen to be adjacent in memory
     * (`disps[i+1] == disps[i] + blk_sizes[i] * sizeofElem`), they can be
     * merged into a single larger block. This helper walks the input arrays
     * and returns a compacted `(n_size, blk_sizes, disps)` tuple.
     *
     * Used by #ArrayTransformer::createMPITypes before calling the actual
     * `MPI_Type_create_hindexed`.
     *
     * @tparam TBlkSiz  Element type of the block-size array (e.g., `MPI_int`).
     * @tparam TDisp    Element type of the displacement array (e.g., `MPI_Aint`).
     * @tparam TSizeof  Element-size type (usually `MPI_Aint`).
     *
     * @param o_size       Number of input blocks.
     * @param blk_sizes    Block sizes (in element count).
     * @param disps        Displacements (in bytes).
     * @param sizeofElem   Element size in bytes (matches `MPI_Type_extent`).
     * @return `(new_size, new_blk_sizes, new_disps)` -- a merged layout.
     */
    template <class TBlkSiz, class TDisp, class TSizeof>
    auto optimize_hindexed_layout(index o_size, TBlkSiz *blk_sizes, TDisp *disps, TSizeof sizeofElem)
    {
        index n_size = 0;
        for (index i = 0; i < o_size; i++)
        {
            while (i + 1 < o_size && (blk_sizes[i] * sizeofElem + disps[i] == disps[i + 1]))
                i++;
            n_size++;
        }
        std::vector<TBlkSiz> new_blk_sizes(n_size, 0);
        std::vector<TDisp> new_disps(n_size);
        n_size = 0;
        for (index i = 0; i < o_size; i++)
        {
            new_blk_sizes[n_size] += blk_sizes[i];
            new_disps[n_size] = disps[i];
            while (i + 1 < o_size && (blk_sizes[i] * sizeofElem + disps[i] == disps[i + 1]))
            {
                i++;
                new_blk_sizes[n_size] += blk_sizes[i];
            }
            n_size++;
        }
        return std::make_tuple(n_size, new_blk_sizes, new_disps);
    }
}