#pragma once

#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Core/Errors.hpp"
#include "DNDS/Core/OMP.hpp"
#include "Cauchy/State/EquationViews.hpp"
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

    /// Assemble the implicit stage defect and, when requested, its local norm
    /// contribution. Reduction and stopping decisions remain with the caller.
    template <PhysicalAccumulationKind Accumulation, class Mean, class Residual, class Constant, class Output, class Metric, class Active, class Norm>
    NumericalStatus AssembleDefectRows(Mean mean, Residual residual, Constant constant, Output output,
                                       const Metric &metric, index components, real mass, real rhsScale, const Active &active,
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
            const auto differential = [&](index j)
            {
                return publish(j, rhsScale * residual(iCell, j) +
                                      volume * (constant(iCell, j) - mass * mean(iCell, j)));
            };
            if constexpr (Accumulation == PhysicalAccumulationKind::ConstantDensityVelocityPressure)
            {
                for (index j = 0; j < 3; ++j)
                    if (const auto status = differential(j); !status)
                        return status;
                if (const auto status = publish(3, rhsScale * residual(iCell, 3) + volume * constant(iCell, 3)); !status)
                    return status;
                for (index j = 4; j < components; ++j)
                    if (const auto status = differential(j); !status)
                        return status;
            }
            else
                for (index j = 0; j < components; ++j)
                    if (const auto status = differential(j); !status)
                        return status;
            return {};
        };
        if (!measureNorm)
            return OMP::ForRowsStatus(0, mean.RowCount(), [&](index iCell)
                                      { return row(iCell, {}); });
        return OMP::SumRows(mean.RowCount(), {norm.data(), components + 1}, partials, row);
    }

    /// Assemble mass*u-rhsScale*R(u)/volume and optionally publish the spatial
    /// rate. The two entry points avoid a nullable output inside the hot loop.
    template <PhysicalAccumulationKind Accumulation, class Mean, class Residual, class Output, class Metric>
    void AssembleEquationRows(Mean mean, Residual residual, Output output, const Metric &metric,
                              index components, real mass, real rhsScale)
    {
        OMP::ForRows(0, mean.RowCount(), [&](index iCell)
                     {
            const real volume = metric.GetCellVol(iCell);
            const auto differential = [&](index j)
            { output(iCell, j) = mass * mean(iCell, j) - rhsScale * residual(iCell, j) / volume; };
            if constexpr (Accumulation == PhysicalAccumulationKind::ConstantDensityVelocityPressure)
            {
                for (index j = 0; j < 3; ++j)
                    differential(j);
                output(iCell, 3) = -rhsScale * residual(iCell, 3) / volume;
                for (index j = 4; j < components; ++j)
                    differential(j);
            }
            else
                for (index j = 0; j < components; ++j)
                    output(iCell, j) = mass * mean(iCell, j) - rhsScale * residual(iCell, j) / volume; });
    }

    template <PhysicalAccumulationKind Accumulation, class Mean, class Residual, class Output, class Rate, class Metric>
    void AssembleEquationAndRateRows(Mean mean, Residual residual, Output output, Rate rate,
                                     const Metric &metric, index components, real mass, real rhsScale)
    {
        OMP::ForRows(0, mean.RowCount(), [&](index iCell)
                     {
            const real volume = metric.GetCellVol(iCell);
            const auto differential = [&](index j)
            {
                const real rateValue = residual(iCell, j) / volume;
                output(iCell, j) = mass * mean(iCell, j) - rhsScale * rateValue;
                rate(iCell, j) = rateValue;
            };
            if constexpr (Accumulation == PhysicalAccumulationKind::ConstantDensityVelocityPressure)
            {
                for (index j = 0; j < 3; ++j)
                    differential(j);
                const real pressureRate = residual(iCell, 3) / volume;
                output(iCell, 3) = -rhsScale * pressureRate;
                rate(iCell, 3) = pressureRate;
                for (index j = 4; j < components; ++j)
                    differential(j);
            }
            else
                for (index j = 0; j < components; ++j)
                    differential(j); });
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
                                    int normOrder, SpanView<real> sum, SpanView<real> partials = {}) noexcept;
    NumericalStatus AssembleDefect(EquationLayoutView<const real> mean, EquationLayoutView<const real> residual,
                                   EquationLayoutView<const real> constant, EquationLayoutView<real> output,
                                   CFV::FiniteVolumeHostView metric, PhysicalAccumulationKind accumulation,
                                   real mass, real rhsScale, SpanView<const uint8_t> active,
                                   int normOrder, bool measureNorm, SpanView<real> norm, SpanView<real> partials = {}) noexcept;

    struct PseudoTimeRequest
    {
        CFV::FiniteVolumeHostView metric;
        Algebra::DofComponentView<const real> means;
        Algebra::DofComponentView<real> defect;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> spectral;
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
    void PrepareSpectralPseudoTime(CFV::FiniteVolumeHostView metric, Read spectral, Write inverse, real cfl) noexcept
    {
        OMP::ForRows(0, inverse.RowCount(), [&](index iCell)
                     {
            const real volume = metric.GetCellVol(iCell);
            const real value = spectral(iCell, 0) / (volume * cfl);
            inverse(iCell, 0) = volume > 0 && value > 0 && std::isfinite(value)
                                    ? value : std::numeric_limits<real>::quiet_NaN(); });
    }
    void ApplyAgreedRHSAlpha(
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> alpha,
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> inverse,
        Algebra::DofComponentView<real> defect, real minimum, AlphaReduction reduction, bool compressRHS);
}
