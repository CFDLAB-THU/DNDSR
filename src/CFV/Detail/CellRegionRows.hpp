#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Runtime/Array/RowSelection.hpp"

namespace DNDS::CFV::Detail
{
    /// Host submission metadata and the optional index payload for one cell traversal.
    /// An implicit selection carries an empty table for the scheduled-access interface.
    struct CellRegionRows
    {
        Runtime::RowSelectionShape shape;
        Runtime::Array<index> indices{Runtime::RowLayout::Dense(0, 1)};

        static CellRegionRows Prepare(const std::vector<index> &rows, index owned)
        {
            CellRegionRows result{Runtime::RowSelectionShape::FromIndices(rows, owned)};
            if (result.shape.form == Runtime::RowSelectionForm::IndexList)
            {
                result.indices = Runtime::Array<index>(Runtime::RowLayout::Dense(rows.size(), 1));
                auto write = result.indices.Write();
                std::copy(rows.begin(), rows.end(), write.View().data);
            }
            return result;
        }
    };

    /// Preserve Geom's exact work order, including irregular core and dependent sets.
    struct ReconstructionRegionRows
    {
        CellRegionRows all, core, dependent;
        ReconstructionRegionRows(const Geom::EntityWorkPartition &partition, index owned)
            : all{Runtime::RowSelectionShape::AllRows(owned)}, core(CellRegionRows::Prepare(partition.Core(), owned)),
              dependent(CellRegionRows::Prepare(partition.HaloDependent(), owned)) {}

        const CellRegionRows &Get(ReconstructionRegion region) const
        {
            if (region == ReconstructionRegion::All)
                return all;
            if (region == ReconstructionRegion::Core)
                return core;
            DNDS_check_throw_info(region == ReconstructionRegion::HaloDependent, "unknown reconstruction region");
            return dependent;
        }
        void Place(Runtime::ExecutionContext context, const std::string &prefix)
        {
            for (auto *rows : {&all, &core, &dependent})
                rows->indices = rows->indices.To(context.device, context);
            core.indices.SetName(prefix + "/core", DNDS_ALLOCATION_SITE());
            dependent.indices.SetName(prefix + "/dependent", DNDS_ALLOCATION_SITE());
        }
    };
}
