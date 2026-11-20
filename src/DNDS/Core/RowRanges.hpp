#pragma once

#include "DNDS/Core/Defines.hpp"
#include <algorithm>
#include <array>
#include <vector>

namespace DNDS
{
    struct RowInterval
    {
        index begin = 0, end = 0;
        DNDS_DEVICE_CALLABLE index Size() const { return end - begin; }
        DNDS_DEVICE_CALLABLE bool Contains(index row) const { return begin <= row && row < end; }
    };

    /// Half-open intervals owned by one worker, in physical traversal order.
    /// A missing second run is empty. Neither interval includes another worker's cells.
    struct WorkerRowRanges
    {
        RowInterval first, second;
        DNDS_DEVICE_CALLABLE bool Contains(index row) const { return first.Contains(row) || second.Contains(row); }
        DNDS_DEVICE_CALLABLE index Size() const { return first.Size() + second.Size(); }
        DNDS_DEVICE_CALLABLE index RowAt(index position) const
        {
            return position < first.Size() ? first.begin + position : second.begin + (position - first.Size());
        }
        std::array<RowInterval, 2> Intervals() const { return {first, second}; }
        RowInterval Contiguous() const
        {
            DNDS_check_throw_info(second.Size() == 0, "worker owns disjoint cell intervals; use LocalPartRanges");
            return first;
        }
    };

    inline std::vector<WorkerRowRanges> WorkerRangesFromStarts(const std::vector<index> &starts, index extent)
    {
        DNDS_check_throw_info(starts.size() >= 2 && starts.front() == 0 && starts.back() == extent &&
                                  std::is_sorted(starts.begin(), starts.end()),
                              "invalid worker partition boundaries");
        std::vector<WorkerRowRanges> result;
        result.reserve(starts.size() - 1);
        for (size_t part = 1; part < starts.size(); ++part)
            result.push_back({{starts[part - 1], starts[part]}, {}});
        return result;
    }

    /// Normalize adjacent/empty runs and prove exact coverage without a cell-sized bitmap.
    inline void ValidateWorkerRanges(std::vector<WorkerRowRanges> &workers, index extent)
    {
        DNDS_check_throw_info(!workers.empty(), "missing cell partitions");
        std::vector<RowInterval> runs;
        for (auto &worker : workers)
        {
            for (auto run : {worker.first, worker.second})
            {
                DNDS_check_throw_info(0 <= run.begin && run.begin <= run.end && run.end <= extent, "invalid worker cell interval");
                if (run.Size())
                    runs.push_back(run);
            }
            if (!worker.second.Size())
                continue;
            if (!worker.first.Size())
                worker.first = worker.second;
            else if (worker.first.end == worker.second.begin)
                worker.first.end = worker.second.end;
            else
            {
                DNDS_check_throw_info(worker.first.end < worker.second.begin, "worker intervals are out of order");
                continue;
            }
            worker.second = {};
        }
        std::sort(runs.begin(), runs.end(), [](auto a, auto b)
                  { return a.begin < b.begin; });
        index covered = 0;
        for (auto run : runs)
        {
            DNDS_check_throw_info(run.begin == covered, "overlap or gap in cell partitions");
            covered = run.end;
        }
        DNDS_check_throw_info(covered == extent, "incomplete cell partitions");
    }
}
