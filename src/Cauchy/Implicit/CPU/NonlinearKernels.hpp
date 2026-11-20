#pragma once

#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Core/Errors.hpp"
#include "DNDS/Core/OMP.hpp"
#include "Cauchy/State/EquationViews.hpp"
#include "Cauchy/Implicit/StageAssembly.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "Cauchy/Implicit/PositivityReduction.hpp"

#include <cmath>

namespace DNDS::Cauchy::CPU
{
    /// Scale an already evaluated cell rate by cell volume. The caller owns
    /// every lease and validates the row/component shapes before entering.
    template <class Input, class Output, class Metric>
    void ScaleRateByVolume(Input input, Output output, const Metric &metric, index components)
    {
        OMP::ForRows(0, input.RowCount(), [&](index iCell)
                     {
            const real volume = metric.GetCellVol(iCell);
            for (index j = 0; j < components; ++j)
                output(iCell, j) = volume * input(iCell, j); });
    }

    /// Divide a cell-integrated residual by cell volume. This is the CPU leaf
    /// used by rate-only evaluation; it performs no ownership or MPI work.
    template <class Input, class Output, class Metric>
    void NormalizeRateByVolume(Input input, Output output, const Metric &metric, index components)
    {
        OMP::ForRows(0, input.RowCount(), [&](index iCell)
                     {
            const real inverseVolume = 1 / metric.GetCellVol(iCell);
            for (index j = 0; j < components; ++j)
                output(iCell, j) = inverseVolume * input(iCell, j); });
    }

    /// One cell of the time equation. The physical mass projection applies
    /// only to the current mean; resOther and the spatial rate keep every row.
    template <StageAssembly Op, PhysicalAccumulationKind Accumulation, int Width, class Mean, class Residual, class Constant, class Rate, class Publish>
    inline NumericalStatus AssembleStageRow(index iCell, index components, real volume, real inverseDt, real rhsScale,
                                            Mean mean, Residual residual, Constant constant, Rate rate, Publish publish) noexcept
    {
        for (index j = 0; j < (Width ? Width : components); ++j)
        {
            const real integrated = residual(iCell, j);
            const bool pressure = Accumulation == PhysicalAccumulationKind::ConstantDensityVelocityPressure && j == 3;
            real value;
            if constexpr (Op == StageAssembly::Defect)
            {
                const real history = constant(iCell, j);
                value = rhsScale * integrated + volume * (pressure ? history : history - inverseDt * mean(iCell, j));
            }
            else
            {
                real weightedRate;
                if constexpr (Op == StageAssembly::EquationAndRate)
                {
                    const real normalized = integrated / volume;
                    rate(iCell, j) = normalized;
                    weightedRate = rhsScale * normalized;
                }
                else
                    weightedRate = rhsScale * integrated / volume;
                value = pressure ? -weightedRate : inverseDt * mean(iCell, j) - weightedRate;
            }
            if (const auto status = publish(j, value); !status)
                return status;
        }
        return {};
    }

    /// Assemble the implicit stage defect and, when requested, its local norm
    /// contribution. Reduction and stopping decisions remain with the caller.
    template <PhysicalAccumulationKind Accumulation, int Width, class Mean, class Residual, class Constant, class Output, class Metric, class Active, class Norm>
    NumericalStatus AssembleDefectRows(Mean mean, Residual residual, Constant constant, Output output,
                                       const Metric &metric, index components, real inverseDt, real rhsScale, const Active &active,
                                       int normOrder, bool measureNorm, Norm &norm, SpanView<real> partials = {}) noexcept
    {
        const auto row = [&](index iCell, SpanView<real> sum) -> NumericalStatus
        {
            const real volume = metric.GetCellVol(iCell);
            if (measureNorm)
                sum[components] += volume;
            const auto publish = [&](index j, real value) -> NumericalStatus
            {
                if (!(std::isfinite(value)))
                    return NumericalStatus{NumericalError::InvalidState, 1, iCell};
                output(iCell, j) = active(j) ? value : 0;
                if (measureNorm)
                    sum[j] += normOrder == 1 ? std::abs(value) : value * value / volume;
                return {};
            };
            if constexpr (Accumulation == PhysicalAccumulationKind::ConstantDensityVelocityPressure)
                if (measureNorm)
                {
                    const real continuity = residual(iCell, 3);
                    sum[components + 1] += continuity * continuity / volume;
                }
            return AssembleStageRow<StageAssembly::Defect, Accumulation, Width>(
                iCell, components, volume, inverseDt, rhsScale, mean, residual, constant, nullptr, publish);
        };
        if (!measureNorm)
            return OMP::ForRowsStatus(0, mean.RowCount(), [&](index iCell)
                                      { return row(iCell, {}); });
        constexpr bool algebraicPressure = Accumulation == PhysicalAccumulationKind::ConstantDensityVelocityPressure;
        return OMP::SumRows(mean.RowCount(), {norm.data(), components + 1 + algebraicPressure}, partials, row);
    }

    /// Write (M U)/dt - rhsScale * R_int/V. The fused variant also
    /// publishes R_int/V, including the continuity rate at the pressure row.
    template <StageAssembly Op, PhysicalAccumulationKind Accumulation, int Width, class Mean, class Residual, class Output, class Rate, class Metric>
    void AssembleEquationRows(Mean mean, Residual residual, Output output, Rate rate, const Metric &metric,
                              index components, real inverseDt, real rhsScale)
    {
        OMP::ForRows(0, mean.RowCount(), [&](index iCell)
                     {
            const real volume = metric.GetCellVol(iCell);
            AssembleStageRow<Op, Accumulation, Width>(
                iCell, components, volume, inverseDt, rhsScale, mean, residual, nullptr, rate,
                [&](index j, real value) -> NumericalStatus
                {
                    output(iCell, j) = value;
                    return {};
                }); });
    }

    struct CorrectionControls
    {
        ThermalView bounds;
        RANSStateView rans;
        SpanView<const uint8_t> active;
        index components, flowFirst, speciesFirst, ransFirst;
        int independentSpecies;
        real scale, incrementRelax, compressionPower;
        bool positivity;
        real ransRelaxation = 1;
    };
    struct CorrectionResult
    {
        index limited = 0;
        real minimum = 1;
        bool frozen = true;
    };
    /// Operator-owned scratch, one result and independent-species slice per
    /// contiguous row partition. No allocation or shared counter in the leaf.
    struct CorrectionWorkspace
    {
        SpanView<real> species;
        SpanView<CorrectionResult> partials;
    };
    /// Euler-compatible solved-increment compression, repaired species target,
    /// final thermal line search, then independent RANS repair. No MPI/access.
    NumericalResult<CorrectionResult> ApplyCorrection(const CorrectionControls &,
                                                      Algebra::DofComponentView<const real> mean, Algebra::DofComponentView<const real> delta,
                                                      Algebra::DofComponentView<real> candidate, CorrectionWorkspace) noexcept;
    NumericalStatus MeasureRateNorm(Algebra::DofComponentView<const real> rate,
                                    Algebra::DofComponentView<const real> reference, CFV::FiniteVolumeHostView metric,
                                    int normOrder, SpanView<real> sum, SpanView<real> partials = {},
                                    Algebra::DofComponentView<const real> rawACMRate = {}) noexcept;
    NumericalStatus AssembleDefect(EquationLayoutView<const real> mean, EquationLayoutView<const real> residual,
                                   EquationLayoutView<const real> constant, EquationLayoutView<real> output,
                                   CFV::FiniteVolumeHostView metric, PhysicalAccumulationKind accumulation,
                                   real inverseDt, real rhsScale, SpanView<const uint8_t> active,
                                   int normOrder, bool measureNorm, SpanView<real> norm, SpanView<real> partials = {}) noexcept;

    struct PseudoTimeRequest
    {
        CFV::FiniteVolumeHostView metric;
        Algebra::DofComponentView<const real> means;
        Algebra::DofComponentView<real> defect;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> spectral;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> smoothScale;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> inverse;
        std::optional<Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>> alphas;
        ThermalView bounds;
        real pseudoCFL, forecastScale, relax, compressionPower;
        bool positivity, compressRHS;
    };
    NumericalResult<AlphaStatistics> PreparePseudoTime(const PseudoTimeRequest &) noexcept;
    /// Uncompressed pseudo-time. Invalid spectra publish NaN, detected by
    /// matrix preparation alongside verification of the other assembled data.
    template <class Read, class Write>
    void PrepareSpectralPseudoTime(CFV::FiniteVolumeHostView metric, Read spectral, Read smoothScale, Write inverse, real cfl) noexcept
    {
        OMP::ForRows(0, inverse.RowCount(), [&](index iCell)
                     {
            const real volume = metric.GetCellVol(iCell);
            const real factor = smoothScale.RowCount() ? smoothScale(iCell, 0) : 1;
            const real value = spectral(iCell, 0) / (volume * cfl * factor);
            inverse(iCell, 0) = volume > 0 && value > 0 && std::isfinite(value)
                                    ? value : std::numeric_limits<real>::quiet_NaN(); });
    }
    void ApplyAgreedRHSAlpha(
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> alpha,
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> inverse,
        Algebra::DofComponentView<real> defect, real minimum, AlphaReduction reduction, bool compressRHS);
}
