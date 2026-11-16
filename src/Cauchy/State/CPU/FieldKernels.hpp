#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Core/Errors.hpp"
#include "DNDS/Core/Span.hpp"
#include "DNDS/Core/OMP.hpp"
#include <cmath>
#include <array>
namespace DNDS::Cauchy::CPU
{
    /// Simple owned-row kernels. The operator reserves access and passes the
    /// logical width explicitly; physical pitch remains part of each view.
    template <class Input, class Output>
    void CopyRows(Input input, Output output, index components)
    {
        OMP::ForRows(0, output.RowCount(), [&](index iCell)
                     {
            for (index j = 0; j < components; ++j)
                output(iCell, j) = input(iCell, j); });
    }
    template <class Output>
    void FillRows(Output output, index components, real value)
    {
        OMP::ForRows(0, output.RowCount(), [&](index iCell)
                     {
            for (index j = 0; j < components; ++j)
                output(iCell, j) = value; });
    }
    /// Return the first invalid row; error emission belongs to the caller.
    template <class Input>
    NumericalStatus RequireFiniteRows(Input input, index components) noexcept
    {
        return OMP::ForRowsStatus(0, input.RowCount(), [&](index iCell) -> NumericalStatus
                                  {
            for (index j = 0; j < components; ++j)
                if (!(std::isfinite(input(iCell, j))))
                    return NumericalStatus{NumericalError::InvalidState, 1, iCell};
            return {}; });
    }
    template <class Input>
    NumericalStatus RequireCappedRows(Input input, SpanView<const real> caps) noexcept
    {
        return OMP::ForRowsStatus(0, input.RowCount(), [&](index iCell) -> NumericalStatus
                                  {
            for (index j = 0; j < caps.size(); ++j)
                if (!(std::isfinite(input(iCell, j)) && input(iCell, j) >= 0 && input(iCell, j) <= caps[j]))
                    return NumericalStatus{NumericalError::InvalidState, 1, iCell};
            return {}; });
    }
    template <class Values, class Metric>
    void AccumulateMeanScale(Values values, const Metric &metric, index components, SpanView<real> sum, SpanView<real> partials = {})
    {
        (void)OMP::SumRows(values.RowCount(), sum, partials, [&](index iCell, SpanView<real> local) -> NumericalStatus
                           {
            const real volume = metric.GetCellVol(iCell);
            local[components] += volume;
            for (index j = 0; j < components; ++j)
                local[j] += volume * std::abs(values(iCell, j));
            return {}; });
    }

    template <class Rows>
    bool FiniteRows(Rows values, index components)
    {
        return bool(OMP::ForRowsStatus(0, values.RowCount(), [&](index iRow) -> NumericalStatus
                                       {
            for (index j = 0; j < components; ++j)
                if (!std::isfinite(values(iRow, j)))
                    return {NumericalError::InvalidState, 1, iRow};
            return {}; }));
    }
    template <class Input, class Output, class Metric>
    void DivideRowsByVolume(Input input, Output output, const Metric &metric, index components)
    {
        OMP::ForRows(0, input.RowCount(), [&](index iCell)
                     {
            for (index j = 0; j < components; ++j)
                output(iCell, j) = input(iCell, j) / metric.GetCellVol(iCell); });
    }
    template <class Input, class Metric>
    NumericalStatus AccumulateVolumeNorm(Input values, const Metric &metric, index components, int normOrder, SpanView<real> sum) noexcept
    {
        for (index iCell = 0; iCell < values.RowCount(); ++iCell)
        {
            const real volume = metric.GetCellVol(iCell);
            sum[components] += volume;
            for (index j = 0; j < components; ++j)
            {
                const real value = values(iCell, j);
                if (!(std::isfinite(value)))
                    return NumericalStatus{NumericalError::InvalidState, 1, iCell};
                sum[j] += volume * (normOrder == 1 ? std::abs(value) : value * value);
            }
        }
        return {};
    }
    template <class Input>
    void AccumulateReconstructionUpdateNorm(Input a, Input b, SpanView<real> norms)
    {
        for (index iCell = 0; iCell < a.RowCount(); ++iCell)
            for (index j = 0; j < a.Width(); ++j)
            {
                const real delta = b(iCell, j) - a(iCell, j);
                norms[0] += delta * delta;
                norms[1] += b(iCell, j) * b(iCell, j);
            }
    }

    /// Caller supplies one two-scalar row per worker. Accumulation within each
    /// contiguous partition and the final merge both have fixed order; no FP
    /// atomic sums or allocations occur in this kernel. One worker reproduces
    /// the serial arithmetic (including accumulation across physical fields).
    template <class Input, class Scratch>
    void AccumulateReconstructionUpdateNorm(Input a, Input b, SpanView<real> norms, Scratch workers)
    {
        if (workers.RowCount() == 1)
        {
            AccumulateReconstructionUpdateNorm(a, b, norms);
            return;
        }
        const index quotient = a.RowCount() / workers.RowCount(), remainder = a.RowCount() % workers.RowCount();
        OMP::ForRows(0, workers.RowCount(), [&](index iWorker)
                     {
            const index first = iWorker * quotient + std::min(iWorker, remainder);
            const index end = first + quotient + (iWorker < remainder);
            real difference = 0, magnitude = 0;
            for (index iCell = first; iCell < end; ++iCell)
                for (index j = 0; j < a.Width(); ++j)
                {
                    const real delta = b(iCell, j) - a(iCell, j);
                    difference += delta * delta;
                    magnitude += b(iCell, j) * b(iCell, j);
                }
            workers(iWorker, 0) = difference;
            workers(iWorker, 1) = magnitude; });
        for (index iWorker = 0; iWorker < workers.RowCount(); ++iWorker)
        {
            norms[0] += workers(iWorker, 0);
            norms[1] += workers(iWorker, 1);
        }
    }

    template <class Output>
    void FillConstantRows(Output rows, SpanView<const real> values)
    {
        OMP::ForRows(0, rows.RowCount(), [&](index iCell)
                     {
            for (index j = 0; j < values.size(); ++j)
                rows(iCell, j) = values[j]; });
    }
}
