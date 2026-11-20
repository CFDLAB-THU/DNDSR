#include "CFV/Plans/LimitedVRPlan.hpp"
#include "CFV/Detail/ReconstructionData.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV
{
    LimitedVRPlan LimitedVRPlan::Prepare(FiniteVolumePlan metrics)
    {
        LimitedVRPlan result(std::move(metrics));
        DNDS_check_throw_info(result.Device() == Runtime::Device::Host(), "LVR topology preparation requires host metrics");
        auto read = result._metrics._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto mesh = read.HostView();
        index count = 0;
        for (index iCell = 0; iCell < result._metrics.NumCell(); ++iCell)
            count += index(mesh.CellFaces(iCell).size());
        result._offsets = Runtime::Array<index>(Runtime::RowLayout::Dense(result._metrics.NumCell() + 1, 1));
        result._neighbors = Runtime::Array<index>(Runtime::RowLayout::Dense(count, 1));
        result._boundary = Runtime::Array<index>(Runtime::RowLayout::Dense(result._metrics.NumCell(), 1));
        auto offsets = result._offsets.Write(), neighbors = result._neighbors.Write(), boundary = result._boundary.Write();
        const auto o = offsets.View<Runtime::FixedRows<1>>(), n = neighbors.View<Runtime::FixedRows<1>>(), b = boundary.View<Runtime::FixedRows<1>>();
        index next = 0;
        // Prefix allocation precedes independent row fill; no thread appends
        // to shared incidence storage. Slot multiplicity/order are unchanged.
        for (index iCell = 0; iCell < result._metrics.NumCell(); ++iCell)
        {
            o(iCell, 0) = next;
            next += index(mesh.CellFaces(iCell).size());
        }
        o(result._metrics.NumCell(), 0) = next;
        std::vector<uint8_t> valid(size_t(result._metrics.NumCell()), 1);
        OMP::ForRows(0, result._metrics.NumCell(), [&](index iCell)
                     {
            index position = o(iCell, 0);
            b(iCell, 0) = 0;
            for (index slot = 0; slot < index(mesh.CellFaces(iCell).size()); ++slot)
            {
                const auto other = mesh.CellFaceOther(iCell, mesh.CellFaces(iCell)[slot], rowsize(slot));
                valid[iCell] &= other == UnInitIndex || (other >= 0 && other < result._metrics.NumCellProc());
                n(position++, 0) = other;
                b(iCell, 0) |= other == UnInitIndex;
            } });
        OMP::ForSetupRows(0, result._metrics.NumCell(), [&](index iCell)
                          { DNDS_check_throw_info(valid[iCell], "LVR requires complete face-neighbor ghosts"); });
        const auto &core = result._partition.Core();
        OMP::ForSetupRows(0, index(core.size()), [&](index iRow)
                          {
            const index iCell = core[iRow];
            for (index entry = o(iCell, 0); entry < o(iCell + 1, 0); ++entry)
                DNDS_check_throw_info(n(entry, 0) == UnInitIndex || n(entry, 0) < result._metrics.NumCell(), "LVR core depends on ghost input"); });
        return result;
    }
}
