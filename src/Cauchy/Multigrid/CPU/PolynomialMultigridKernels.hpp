#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Core/OMP.hpp"
#include <cstring>

namespace DNDS::Cauchy::CPU
{
    /// Physical-column mask: 0=frozen, 1=selected, 2=unused storage column.
    /// Only logical equation components participate; unbound columns survive.
    template <class View>
    index FirstChangedFrozenRow(View values, View saved, SpanView<const uint8_t> mask)
    {
        index first = values.RowCount();
        const int threads = OMP::DistributedThreads();
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1) reduction(min : first)
#endif
        for (index iRow = 0; iRow < values.RowCount(); ++iRow)
            for (index j = 0; j < values.Width(); ++j)
                if (mask[j] == 0 && std::memcmp(&values(iRow, j), &saved(iRow, j), sizeof(real)))
                {
                    first = std::min(first, iRow);
                    break;
                }
        return first == values.RowCount() ? -1 : first;
    }
    template <bool Add, class View, class Input>
    void UpdateSelectedRows(View values, Input delta, SpanView<const uint8_t> mask)
    {
        OMP::ForRows(0, values.RowCount(), [&](index iRow)
                     {
            for (index j = 0; j < values.Width(); ++j)
                if constexpr (Add)
                {
                    if (mask[j] == 1) values(iRow, j) += delta(iRow, j);
                }
                else if (mask[j] == 0) values(iRow, j) = 0; });
    }
    NumericalStatus CheckFrozenComponents(Algebra::DofComponentView<const real> values,
                                          Algebra::DofComponentView<const real> saved, SpanView<const uint8_t> active) noexcept;
    void ProjectSelectedCorrection(Algebra::DofComponentView<real> values, SpanView<const uint8_t> active);
    void AddSelectedCorrection(Algebra::DofComponentView<real> values,
                               Algebra::DofComponentView<const real> delta, SpanView<const index> components);
    void CopyPolynomialLevel(Algebra::DofComponentView<const real> source, Algebra::DofComponentView<real> destination);
    void AccumulateFASNorm(Algebra::DofComponentView<const real> source, Algebra::DofComponentView<const real> value,
                           CFV::FiniteVolumeHostView metric, int normOrder, SpanView<real> sum);
}
