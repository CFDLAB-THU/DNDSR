#pragma once

#include "DNDS/Core/CheckedSize.hpp"
#include "DNDS/Core/OMP.hpp"
#include <vector>

namespace DNDS::Geom::ConnectivityRows
{
    /** @brief Append independent setup-row expansions in their original order.
     * @details visit(row, emit) reads retained immutable inputs and emits zero or
     * more indices. Parallel execution counts, prefixes and fills disjoint row
     * ranges; scratch scales with the local visited rows, never global IDs.
     * The serial path visits once and preserves its original append order.
     * @pre visit is deterministic, has no side effects except emit, and performs
     * no MPI, publication or lease acquisition. Setup exceptions join the team
     * before reaching the caller's collective failure boundary.
     */
    template <class Visit>
    void AppendSetupRows(index rowNumber, std::vector<index> &output, Visit &&visit)
    {
        DNDS_check_throw_info(rowNumber >= 0, "negative connectivity setup row count");
        if (OMP::DistributedThreads() == 1 || rowNumber <= 1)
        {
            for (index iRow = 0; iRow < rowNumber; ++iRow)
                visit(iRow, [&](index value)
                      { output.push_back(value); });
            return;
        }
        std::vector<size_t> starts(CheckedSize::Add(size_t(rowNumber), size_t(1)), 0);
        OMP::ForSetupRows(0, rowNumber, [&](index iRow)
                          {
            size_t count = 0;
            visit(iRow, [&](index) { count = CheckedSize::Add(count, size_t(1)); });
            starts[size_t(iRow) + 1] = count; });
        starts[0] = output.size();
        for (index iRow = 0; iRow < rowNumber; ++iRow)
            starts[size_t(iRow) + 1] = CheckedSize::Add(starts[size_t(iRow)], starts[size_t(iRow) + 1]);
        output.resize(starts.back());
        OMP::ForSetupRows(0, rowNumber, [&](index iRow)
                          {
            size_t next = starts[size_t(iRow)];
            const auto end = starts[size_t(iRow) + 1];
            visit(iRow, [&](index value)
                  {
                DNDS_check_throw_info(next < end, "connectivity setup expansion changed after counting");
                output[next++] = value; });
            DNDS_check_throw_info(next == end, "connectivity setup expansion changed after counting"); });
    }
}
