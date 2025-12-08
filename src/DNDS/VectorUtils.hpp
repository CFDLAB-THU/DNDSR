#pragma once
#include "Defines.hpp"
#include "MPI.hpp"

namespace DNDS
{
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