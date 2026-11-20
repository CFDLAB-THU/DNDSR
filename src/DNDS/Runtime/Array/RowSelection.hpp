#pragma once

#include "DNDS/Core/RowRanges.hpp"
#include "DNDS/Runtime/Array/ArrayView.hpp"

namespace DNDS::Runtime
{
    enum class RowSelectionForm
    {
        AllRows,
        IndexList,
        RowRange,
        RowRangePair
    };

    using ::DNDS::RowInterval;

    /// Small immutable description of a work list. Default means use the supplied indices.
    /// The setup pass recognizes at most two consecutive runs without reordering the list.
    struct RowSelectionShape
    {
        RowSelectionForm form = RowSelectionForm::IndexList;
        RowInterval first, second;

        static RowSelectionShape AllRows(index extent)
        {
            DNDS_check_throw_info(extent >= 0, "negative owned row extent");
            return {RowSelectionForm::AllRows, {0, extent}, {}};
        }

        static RowSelectionShape FromIndices(const std::vector<index> &ids, index extent)
        {
            RowSelectionShape result{RowSelectionForm::RowRange, {}, {}};
            auto *run = &result.first;
            for (auto id : ids)
            {
                DNDS_check_throw_info(id >= 0 && id < extent, "work row outside local row space");
                if (result.form == RowSelectionForm::IndexList)
                    continue;
                if (run->begin == run->end)
                    run->begin = id;
                else if (id != run->end)
                {
                    if (run == &result.second)
                    {
                        result.form = RowSelectionForm::IndexList;
                        continue;
                    }
                    result.form = RowSelectionForm::RowRangePair;
                    run = &result.second;
                    run->begin = id;
                }
                run->end = id + 1;
            }
            if (result.form == RowSelectionForm::RowRangePair &&
                result.first.begin < result.second.end && result.second.begin < result.first.end)
                result.form = RowSelectionForm::IndexList;
            return result;
        }
    };

    /// A borrowed explicit list or up to two intervals. Dispatch outside the numerical
    /// loop. The caller retains the explicit-index Array through completion on its stream.
    struct RowSelection
    {
        using Indices = ArrayView<const index, AccessBoundsChecks, FixedRows<1>>;
        RowSelectionShape shape;
        Indices indices;

        RowSelection() = default;
        RowSelection(Indices ids) : indices(ids) {}
        RowSelection(RowSelectionShape selection, Indices ids = {}) : shape(selection), indices(ids) {}

        void ValidateDevice(Device device) const
        {
            if (shape.form == RowSelectionForm::IndexList)
                indices.ValidateDevice(device);
        }

        bool SameTraversal(const RowSelection &other) const
        {
            if (shape.form != other.shape.form)
                return false;
            if (shape.form == RowSelectionForm::IndexList)
                return indices.data == other.indices.data && indices.rows == other.indices.rows &&
                       indices.stride == other.indices.stride && indices.device == other.indices.device;
            return shape.first.begin == other.shape.first.begin && shape.first.end == other.shape.first.end &&
                   shape.second.begin == other.shape.second.begin && shape.second.end == other.shape.second.end;
        }

        DNDS_DEVICE_CALLABLE index RowCount() const
        {
            return shape.form == RowSelectionForm::IndexList ? indices.RowCount() : (shape.first.end - shape.first.begin) + (shape.second.end - shape.second.begin);
        }

        struct IntervalRows
        {
            index begin;
            DNDS_DEVICE_CALLABLE index operator()(index i) const { return begin + i; }
        };
        struct IntervalPairRows
        {
            RowInterval first, second;
            DNDS_DEVICE_CALLABLE index operator()(index i) const
            {
                const index size = first.end - first.begin;
                return i < size ? first.begin + i : second.begin + (i - size);
            }
        };
        struct IndexedRows
        {
            Indices ids;
            DNDS_DEVICE_CALLABLE index operator()(index i) const { return ids(i, 0); }
        };
        template <class F>
        DNDS_DEVICE_CALLABLE decltype(auto) Visit(F &&f) const
        {
            if (shape.form == RowSelectionForm::IndexList)
                return f(IndexedRows{indices});
            if (shape.form == RowSelectionForm::RowRangePair)
                return f(IntervalPairRows{shape.first, shape.second});
            return f(IntervalRows{shape.first.begin});
        }
        /// Resolve once per numerical row. Heavy point and matrix loops reuse this ID.
        DNDS_DEVICE_CALLABLE index RowAt(index position) const
        {
            return Visit([&](auto rows)
                         { return rows(position); });
        }
    };
    static_assert(std::is_trivially_copyable_v<RowSelection>);
}
