#include "SerialAdjReordering.hpp"

namespace DNDS::Geom
{
    std::vector<index> ReorderSerialAdj_PartitionMetisC(
        tLocalMatStruct::iterator mat_begin, tLocalMatStruct::iterator mat_end,
        std::vector<index>::iterator i_new2old_begin, std::vector<index>::iterator i_new2old_end,
        int nParts,
        index ind_offset,
        bool do_rcm,
        index &bwOldM, index &bwNewM)
    {
        index n_elem = mat_end - mat_begin;
        DNDS_check_throw(i_new2old_end - i_new2old_begin == n_elem);

        // std::cout << "here 0-" << ind_offset << "  " << n_elem << std::endl;
        nParts = std::max(1, nParts);
        std::vector<index> part_start(nParts + 1, 0);

        if (nParts > 1)
        {
            auto partition_local = PartitionSerialAdj_Metis(mat_begin, mat_end, nParts);
            std::vector<std::set<int>> localPartsAdjV(nParts);

            for (index iC = 0; iC < n_elem; iC++)
            {
                auto iP = partition_local.at(iC);

                for (auto iCOther : mat_begin[iC])
                {
                    auto iPOther = partition_local.at(iCOther - ind_offset);
                    if (iPOther != iP)
                        localPartsAdjV.at(iP).insert(iPOther);
                }
            }
            tLocalMatStruct localPartsAdj;
            localPartsAdj.reserve(nParts);
            for (auto &row : localPartsAdjV)
            {
                localPartsAdj.emplace_back();
                localPartsAdj.back().reserve(row.size());
                for (auto &v : row)
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
            for (auto &row : new_mat)
                for (auto &iCOther : row)
                    iCOther = ind_offset + i_old2new.at(iCOther - ind_offset);

            for (index iC = 0; iC < n_elem; iC++)
            {
                i_new2old_begin[iC] = new_i_new2old[iC]; // which is oldest...
                mat_begin[iC] = std::move(new_mat[iC]);
            }
            // // asserting perumtation
            // std::set<index> set;
            // for (auto v : i_old2new)
            // {
            //     DNDS_assert(v < n_elem);
            //     DNDS_assert(set.count(v) == 0);
            //     set.insert(v);
            // }
            // // asserting symmetry:
            // for (index iC = 0; iC < n_elem; iC++)
            // {
            //     for (auto jC : mat_begin[iC])
            //     {
            //         int c = 0;
            //         for (auto kC : mat_begin[jC])
            //             if (kC == iC)
            //                 c++;
            //         DNDS_assert(c == 1);
            //     }
            // }
        }
        else
            part_start[1] = n_elem;

        bwOldM = 0, bwNewM = 0;
        if (!do_rcm)
            return part_start;

        for (int iPart = 0; iPart < nParts; iPart++)
        {
            index local_offset = part_start[iPart];
            index local_nelem = part_start[iPart + 1] - part_start[iPart];
            //! trim the local matrix
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
                    iCOther = local_offset + cOld2New_.at(iCOther - local_offset);

            for (index iC = 0; iC < local_nelem; iC++)
            {
                i_new2old_begin[iC + local_offset] = new_i_new2old[iC]; // which is oldest...
                mat_begin[iC + local_offset] = std::move(new_mat[iC]);
            }
        }

        return part_start;
    }
}