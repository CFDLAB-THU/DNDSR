#include "CFV/Detail/GradientKernels.hpp"
#include "CFV/Math/GradientEvaluation.hpp"
#include "CFV/Math/GradientLimiting.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV::Detail
{
    void EvaluateGradient(GradientKernelView plan, DeviceWorkSelection selection, GradientVectorRanges vectors,
                          GradientField mean, GradientInput boundary, GradientOutput output)
    {
        Runtime::ValidateViewDevices(Runtime::Device::Host(), plan, selection, mean, boundary, output);
        OMP::ForRows(0, selection.count, [&](index iWork) noexcept
                     { Math::EvaluateGradientCell(plan, vectors, mean, boundary, output, selection.Cell(iWork), mean.owned.Width()); });
    }
    void ConvertGradient(GradientCellGeometry cells, GradientInput input, GradientOutput output,
                         index firstCell, int dimension, index modes, bool toCoefficients, index firstComponent, index componentCount)
    {
        Runtime::ValidateViewDevices(Runtime::Device::Host(), cells, input, output);
        OMP::ForRows(0, output.RowCount(), [&](index iRow) noexcept
                     { Math::ConvertGradientRow(cells, input, output, iRow, firstCell, dimension, modes, toCoefficients, firstComponent, componentCount); });
    }
    void LimitGradient(GradientLimiterView plan, DeviceWorkSelection selection, GradientVectorRanges vectors,
                       GradientLimiterValues values, GradientOutput output, GradientOutput scratch)
    {
        Runtime::ValidateViewDevices(Runtime::Device::Host(), plan, selection, values, output, scratch);
        DNDS_check_throw_info(scratch.RowCount() > 0 && scratch.Width() >= 6 * output.Width() &&
                                  scratch.kind == Runtime::RowLayoutKind::Table,
                              "CPU gradient limiter scratch is incomplete");
        // A logical scratch slot belongs to one independent loop chunk. No
        // dependency on worker thread IDs or mutable OpenMP team size exists.
        const index slots = std::min(scratch.RowCount(), selection.count);
        OMP::ForRows(0, slots, [&](index slot) noexcept
                     {
            const auto row = scratch[slot];
            for (index iWork = slot; iWork < selection.count; iWork += slots)
                Math::LimitGradientCell(plan, vectors, values, output, selection.Cell(iWork), {row.data(), row.size()}); });
    }
}
