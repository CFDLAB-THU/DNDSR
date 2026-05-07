#pragma once
/// @file Mesh_CellPermutation.hpp
/// @brief Helper for computing cell reordering permutations via Metis.
///
/// Extracted from Mesh.cpp for use by both the legacy ReorderLocalCellsLegacy
/// and the new ReorderLocalCells (in Mesh_Reorder.cpp).

#include "DNDS/Defines.hpp"
#include "DNDS/ArrayPair.hpp"
#include "SerialAdjReordering.hpp"

#include <vector>

#include <fmt/core.h>

namespace DNDS::Geom::detail
{
    /// Result of local cell permutation computation.
    struct CellPermutationResult
    {
        std::vector<index> cellOld2New;
        std::vector<index> cellNew2Old;
        std::vector<index> localPartitionStarts;
        index bwOld = 0;
        index bwNew = 0;
    };

    /// Compute a cell reordering permutation using Metis partitioning
    /// with optional inner partitioning and contiguous sorting.
    ///
    /// 1. Partition via Metis + RCM.
    /// 2. Optionally sub-partition each first-level partition.
    /// 3. Within each partition, sort cells so interior (private) cells come
    ///    before cells that touch ghost neighbors.
    /// 4. Build inverse permutation.
    ///
    /// @param cell2cellFaceV  Local face-adjacency graph (no ghost edges).
    /// @param cell2cell       Full cell-to-cell adjacency (with ghost).
    /// @param nCell           Number of local (father) cells.
    /// @param nParts          Number of first-level partitions.
    /// @param nPartsInner     Number of inner partitions per first-level part.
    inline CellPermutationResult ComputeCellPermutation(
        tLocalMatStruct &cell2cellFaceV,
        const tAdjPair &cell2cell,
        index nCell,
        int nParts,
        int nPartsInner)
    {
        CellPermutationResult result;
        result.cellOld2New.resize(nCell, -1);
        result.cellNew2Old.resize(nCell);
        for (index i = 0; i < nCell; i++)
            result.cellNew2Old[i] = i;

        result.localPartitionStarts = ReorderSerialAdj_PartitionMetisC(
            cell2cellFaceV.begin(),
            cell2cellFaceV.end(),
            result.cellNew2Old.begin(),
            result.cellNew2Old.end(), nParts, 0, nPartsInner <= 1, result.bwOld, result.bwNew);

        if (nPartsInner > 1)
        {
            auto dbgCheckSubGraphRanges = [&]([[maybe_unused]] const char *tag)
            {
                for (int p = 0; p < static_cast<int>(result.localPartitionStarts.size()) - 1; p++)
                {
                    index pStart = result.localPartitionStarts[p];
                    index pEnd = result.localPartitionStarts[p + 1];
                    for (index iC = pStart; iC < pEnd; iC++)
                        for (auto jC : cell2cellFaceV[iC])
                            DNDS_assert_infof(
                                jC >= 0 && jC < nCell,
                                "%s: partition %d [%lld,%lld): cell %lld has neighbor %lld outside [0,%lld)",
                                tag, p, (long long)pStart, (long long)pEnd,
                                (long long)iC, (long long)jC, (long long)nCell);
                }
            };
            auto dbgCheckBidir = [&]([[maybe_unused]] const char *tag)
            {
                for (index iC = 0; iC < nCell; iC++)
                    for (auto jC : cell2cellFaceV[iC])
                    {
                        bool found = false;
                        for (auto kC : cell2cellFaceV[jC])
                            if (kC == iC)
                            {
                                found = true;
                                break;
                            }
                        DNDS_assert_infof(found,
                                          "%s: edge %lld->%lld exists but reverse %lld->%lld missing",
                                          tag, (long long)iC, (long long)jC, (long long)jC, (long long)iC);
                    }
            };

#ifndef DNDS_NDEBUG
            dbgCheckSubGraphRanges("before inner partitioning");
            dbgCheckBidir("before inner partitioning");
#endif

            for (int iPart = 0; iPart < static_cast<int>(result.localPartitionStarts.size()) - 1; iPart++)
            {
                index bwOldC{0}, bwNewC{0};
                index offset = result.localPartitionStarts[iPart];
                index offsetN = result.localPartitionStarts[iPart + 1];
                auto inner_parts_start = ReorderSerialAdj_PartitionMetisC(
                    cell2cellFaceV.begin() + offset,
                    cell2cellFaceV.begin() + offsetN,
                    result.cellNew2Old.begin() + offset,
                    result.cellNew2Old.begin() + offsetN, nPartsInner, offset, true, bwOldC, bwNewC,
                    cell2cellFaceV.begin(), nCell);
                result.bwOld = std::max(result.bwOld, bwOldC);
                result.bwNew = std::max(result.bwNew, bwNewC);

#ifndef DNDS_NDEBUG
                dbgCheckSubGraphRanges(fmt::format("after inner part {}", iPart).c_str());
#endif
            }

#ifndef DNDS_NDEBUG
            dbgCheckBidir("after all inner partitioning");
#endif
        }

        // Contiguous sorting: within each partition, put interior cells before
        // cells that touch ghost neighbors.
        {
            auto cellIsNotPrivate = [&](index iCell)
            {
                for (auto iCellOther : cell2cell[iCell])
                {
                    if (iCellOther >= nCell)
                        return 1;
                }
                return 0;
            };
            int nLocalParts = !result.localPartitionStarts.empty()
                                  ? static_cast<int>(result.localPartitionStarts.size()) - 1
                                  : 1;
            auto localPartStart = [&](int iPart) -> index
            { return !result.localPartitionStarts.empty() ? result.localPartitionStarts.at(iPart) : 0; };
            auto localPartEnd = [&](int iPart) -> index
            { return !result.localPartitionStarts.empty() ? result.localPartitionStarts.at(iPart + 1) : nCell; };

            std::vector<index> cellNew2Old_new;
            cellNew2Old_new.reserve(nCell);
            for (index i = 0; i < nCell; i++)
                result.cellOld2New[i] = cellIsNotPrivate(result.cellNew2Old[i]);
            for (int iPart = 0; iPart < nLocalParts; iPart++)
            {
                for (index i = localPartStart(iPart); i < localPartEnd(iPart); i++)
                    if (!result.cellOld2New[i])
                        cellNew2Old_new.push_back(result.cellNew2Old[i]);
                for (index i = localPartStart(iPart); i < localPartEnd(iPart); i++)
                    if (result.cellOld2New[i])
                        cellNew2Old_new.push_back(result.cellNew2Old[i]);
            }

            result.cellNew2Old = std::move(cellNew2Old_new);
            DNDS_assert(static_cast<index>(result.cellNew2Old.size()) == nCell);
            for (auto v : result.cellNew2Old)
                DNDS_assert(v < nCell && v >= 0);
        }

        // Build inverse permutation
        {
            std::vector<bool> seen(nCell, false);
            for (index i = 0; i < nCell; i++)
            {
                DNDS_assert(result.cellNew2Old[i] >= 0 && result.cellNew2Old[i] < nCell);
                DNDS_assert(!seen[result.cellNew2Old[i]]);
                seen[result.cellNew2Old[i]] = true;
                result.cellOld2New.at(result.cellNew2Old[i]) = i;
            }
        }
        return result;
    }

} // namespace DNDS::Geom::detail
