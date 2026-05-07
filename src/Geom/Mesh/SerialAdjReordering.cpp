#include "SerialAdjReordering.hpp"

namespace DNDS::Geom
{
    /**
     * @brief Partition a sub-graph into nParts and optionally apply RCM reordering within each part.
     *
     * This function operates on the sub-range [mat_begin, mat_end), which represents
     * n_elem rows of an adjacency graph. Adjacency entries use absolute indices
     * (offset by ind_offset), i.e., cell i's neighbors are stored as (ind_offset + local_index).
     *
     * When called for second-level (inner) partitioning, the sub-range's adjacency rows
     * may contain cross-sub-graph references pointing outside [ind_offset, ind_offset + n_elem).
     * These arise because the first-level partitioning reindexed the full graph, and boundary
     * cells retain edges to cells in other first-level partitions. Two problems follow:
     *
     *   1. Metis and the partition-adjacency builder require a self-contained local graph;
     *      cross-sub-graph edges must be excluded from these operations.
     *
     *   2. When cells within this sub-range are permuted, all references to them -- including
     *      those in other sub-ranges' rows -- must be updated to maintain the bidirectional
     *      graph invariant. The optional full_mat_begin / full_n_elem parameters provide
     *      access to the full graph for this cross-sub-graph fixup.
     *
     * The RCM section trims cross-sub-partition edges before reordering (each sub-partition
     * only needs internal connectivity for bandwidth reduction). The RCM reindex uses
     * (ind_offset + local_offset) as the base for correct absolute index arithmetic.
     *
     * @param full_mat_begin  Iterator to the start of the full graph (for cross-sub-graph
     *                        reference updates). Pass default (empty) when this sub-range
     *                        IS the full graph (first-level call).
     * @param full_n_elem     Total number of rows in the full graph. 0 means no cross-sub-graph
     *                        fixup is needed.
     */
    std::vector<index> ReorderSerialAdj_PartitionMetisC(
        tLocalMatStruct::iterator mat_begin, tLocalMatStruct::iterator mat_end,
        std::vector<index>::iterator i_new2old_begin, std::vector<index>::iterator i_new2old_end,
        int nParts,
        index ind_offset,
        bool do_rcm,
        index &bwOldM, index &bwNewM,
        tLocalMatStruct::iterator full_mat_begin,
        index full_n_elem)
    {
        index n_elem = mat_end - mat_begin;
        DNDS_check_throw(i_new2old_end - i_new2old_begin == n_elem);

        nParts = std::max(1, nParts);
        std::vector<index> part_start(nParts + 1, 0);

        if (nParts > 1)
        {
            //! PartitionSerialAdj_Metis now handles the filtering and rebasing internally
            //! via its ind_offset parameter.
            auto partition_local = PartitionSerialAdj_Metis(mat_begin, mat_end, nParts, ind_offset);

            //! Build partition-level adjacency (which partitions neighbor which).
            //! Only intra-sub-graph edges matter for partition adjacency.
            std::vector<std::set<int>> localPartsAdjV(nParts);
            for (index iC = 0; iC < n_elem; iC++)
            {
                auto iP = partition_local.at(iC);
                for (auto iCOther : mat_begin[iC])
                {
                    index iLocal = iCOther - ind_offset;
                    if (iLocal >= 0 && iLocal < n_elem)
                    {
                        auto iPOther = partition_local.at(iLocal);
                        if (iPOther != iP)
                            localPartsAdjV.at(iP).insert(iPOther);
                    }
                }
            }
            tLocalMatStruct localPartsAdj;
            localPartsAdj.reserve(nParts);
            for (auto &row : localPartsAdjV)
            {
                localPartsAdj.emplace_back();
                localPartsAdj.back().reserve(row.size());
                for (const auto &v : row)
                    localPartsAdj.back().emplace_back(v);
            }

            index bwOld, bwNew;
            auto [partNew2Old_, partOld2New_] = ReorderSerialAdj_CorrectRCM(
                localPartsAdj.begin(), localPartsAdj.end(), bwOld, bwNew, 0);
            for (auto &iP : partition_local)
            {
                DNDS_assert(partOld2New_[iP] < nParts);
                iP = partOld2New_[iP];
            }

            std::vector<index> n_elem_part(nParts, 0);
            for (index iC = 0; iC < n_elem; iC++)
            {
                auto iP = partition_local.at(iC);
                n_elem_part[iP]++;
            }
            for (int i = 0; i < nParts; i++)
                part_start[i + 1] = part_start[i] + n_elem_part[i];

            //! Permute rows according to partition assignment; build i_old2new mapping.
            tLocalMatStruct new_mat(n_elem);
            std::vector<index> new_i_new2old(n_elem);
            std::vector<index> i_old2new(n_elem);
            n_elem_part.assign(n_elem, 0);
            for (index iC = 0; iC < n_elem; iC++)
            {
                auto iP = partition_local[iC];
                index iNew = part_start[iP] + n_elem_part[iP];
                DNDS_assert(iNew < n_elem);
                new_i_new2old[iNew] = i_new2old_begin[iC];
                new_mat[iNew] = std::move(mat_begin[iC]);
                i_old2new[iC] = iNew;
                n_elem_part[iP]++;
            }

            //! Remap in-range references within the permuted sub-range rows.
            //! Cross-sub-graph references (outside [ind_offset, ind_offset + n_elem)) are
            //! left unchanged here; they point to cells not touched by this permutation.
            for (auto &row : new_mat)
                for (auto &iCOther : row)
                {
                    index iLocal = iCOther - ind_offset;
                    if (iLocal >= 0 && iLocal < n_elem)
                        iCOther = ind_offset + i_old2new[iLocal];
                }

            for (index iC = 0; iC < n_elem; iC++)
            {
                i_new2old_begin[iC] = new_i_new2old[iC]; // composing the oldest mapping
                mat_begin[iC] = std::move(new_mat[iC]);
            }

            //! Update cross-sub-graph references: rows OUTSIDE [mat_begin, mat_end) may
            //! contain edges pointing into this sub-range with pre-permutation indices.
            //! Walk the full graph and apply i_old2new to any such references.
            if (full_n_elem > 0)
            {
                index sub_begin = mat_begin - full_mat_begin;
                index sub_end = mat_end - full_mat_begin;
                for (index iC = 0; iC < full_n_elem; iC++)
                {
                    if (iC >= sub_begin && iC < sub_end)
                        continue; // already remapped above
                    for (auto &iCOther : full_mat_begin[iC])
                    {
                        index iLocal = iCOther - ind_offset;
                        if (iLocal >= 0 && iLocal < n_elem)
                            iCOther = ind_offset + i_old2new[iLocal];
                    }
                }
            }
        }
        else
            part_start[1] = n_elem;

        bwOldM = 0, bwNewM = 0;
        if (!do_rcm)
            return part_start;

        //! Per-partition RCM reordering. Cross-sub-partition edges are trimmed before RCM
        //! since bandwidth reduction only considers intra-partition connectivity.
        //! The reindex step uses (ind_offset + local_offset) as the absolute base.
        for (int iPart = 0; iPart < nParts; iPart++)
        {
            index local_offset = part_start[iPart];
            index local_nelem = part_start[iPart + 1] - part_start[iPart];
            //! Trim: remove edges outside [ind_offset + local_offset, ind_offset + local_offset + local_nelem)
            for (index iC = 0; iC < local_nelem; iC++)
            {
                auto &row = mat_begin[iC + local_offset];
                auto last = std::remove_if(row.begin(), row.end(), [&](index v)
                                           { return (v < local_offset + ind_offset) ||
                                                    (v >= local_offset + ind_offset + local_nelem); });
                row.erase(last, row.end());
            }
            index bwOld, bwNew;
            auto [cNew2Old_, cOld2New_] = ReorderSerialAdj_CorrectRCM(
                mat_begin + local_offset,
                mat_begin + local_offset + local_nelem,
                bwOld, bwNew,
                ind_offset + local_offset);
            bwNewM = std::max(bwNewM, bwNew);
            bwOldM = std::max(bwOldM, bwOld);

            tLocalMatStruct new_mat(local_nelem);
            std::vector<index> new_i_new2old(local_nelem);
            for (index iC = 0; iC < local_nelem; iC++)
            {
                index iNew = cOld2New_[iC];
                new_i_new2old[iNew] = i_new2old_begin[iC + local_offset];
                new_mat[iNew] = std::move(mat_begin[iC + local_offset]);
            }
            for (auto &row : new_mat)
                for (auto &iCOther : row)
                    iCOther = ind_offset + local_offset + cOld2New_.at(iCOther - ind_offset - local_offset);

            for (index iC = 0; iC < local_nelem; iC++)
            {
                i_new2old_begin[iC + local_offset] = new_i_new2old[iC]; // composing the oldest mapping
                mat_begin[iC + local_offset] = std::move(new_mat[iC]);
            }
        }

        return part_start;
    }
}