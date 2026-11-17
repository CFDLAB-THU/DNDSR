#include "Cauchy/Implicit/CPU/NonlinearKernels.hpp"

namespace DNDS::Cauchy::CPU
{
    NumericalStatus MeasureRateNorm(Algebra::DofComponentView<const real> rate,
                                    Algebra::DofComponentView<const real> reference, CFV::FiniteVolumeHostView metric,
                                    int normOrder, SpanView<real> sum, SpanView<real> partials) noexcept
    {
        return OMP::SumRows(rate.RowCount(), sum, partials, [&](index iCell, SpanView<real> local) -> NumericalStatus
                            {
            const real volume = metric.GetCellVol(iCell);
            local[rate.ComponentNumber()] += volume;
            for (index j = 0; j < rate.ComponentNumber(); ++j)
            {
                const real value = rate(iCell, j) - (reference.RowCount() ? reference(iCell, j) : 0);
                if (!std::isfinite(value) || !(volume > 0)) return {NumericalError::InvalidState, 1, iCell};
                local[j] += normOrder == 1 ? volume * std::abs(value) : volume * value * value;
            }
            return {}; });
    }
    NumericalStatus AssembleDefect(EquationLayoutView<const real> mean, EquationLayoutView<const real> residual,
                                   EquationLayoutView<const real> constant, EquationLayoutView<real> output,
                                   CFV::FiniteVolumeHostView metric, PhysicalAccumulationKind accumulation,
                                   real mass, real rhsScale, SpanView<const uint8_t> active,
                                   int normOrder, bool measureNorm, SpanView<real> norm, SpanView<real> partials) noexcept
    {
        const auto identity = [&](auto u, auto r, auto b, auto y) -> NumericalStatus
        { return AssembleDefectRows<PhysicalAccumulationKind::Identity>(u, r, b, y, metric, mean.ComponentNumber(), mass, rhsScale, [&](index j)
                                                                        { return active.empty() || active[j]; }, normOrder, measureNorm, norm, partials); };
        if (accumulation == PhysicalAccumulationKind::Identity)
            return DispatchCPUEquationLayouts(identity, mean, residual, constant, output);
        DNDS_assert_info(accumulation == PhysicalAccumulationKind::ConstantDensityVelocityPressure,
                         "unsupported physical accumulation");
        return DispatchCPUEquationLayouts([&](auto u, auto r, auto b, auto y) -> NumericalStatus
                                          { return AssembleDefectRows<PhysicalAccumulationKind::ConstantDensityVelocityPressure>(u, r, b, y, metric, mean.ComponentNumber(), mass, rhsScale, [&](index j)
                                                                                                                                 { return active.empty() || active[j]; }, normOrder, measureNorm, norm, partials); }, mean, residual, constant, output);
    }

    // Keep these helpers at namespace scope: clangd 21.1.0 and 23.1.0
    // segfault in Sema::tryCaptureVariable while instantiating a local class
    // whose member type depends on a generic-lambda parameter.
    namespace
    {
        template <class IncrementView>
        struct SelectedIncrement
        {
            IncrementView values;
            index row;
            real scale;
            SpanView<const uint8_t> active;
            real operator[](index j) const
            {
                return active.empty() || active[j] ? scale * values(row, j) : 0;
            }
        };

        template <class Increment>
        struct LimitedIncrement
        {
            const Increment &values;
            real factor;
            real operator[](index j) const { return factor * values[j]; }
        };

        template <class Increment, class OldState>
        struct StoredIncrement
        {
            const Increment &values;
            const OldState &oldState;
            SpanView<const real> species;
            index first, count;
            real operator[](index j) const
            {
                return !species.empty() && j >= first && j < first + count
                           ? species[j - first] - oldState[j]
                           : values[j];
            }
        };
    }

    NumericalResult<CorrectionResult> ApplyCorrection(const CorrectionControls &control,
                                                      Algebra::DofComponentView<const real> inputMean, Algebra::DofComponentView<const real> inputDelta,
                                                      Algebra::DofComponentView<real> output, CorrectionWorkspace workspace) noexcept
    {
        CorrectionResult result;
        const auto &bounds = control.bounds;
        const auto &rans = control.rans;
        const auto active = control.active;
        const auto scale = control.scale;
        const index components = control.components, independent = control.independentSpecies;
        const index partitions = workspace.partials.size();
        DNDS_assert(partitions > 0 && workspace.species.size() >= partitions * independent);
        const auto apply = [&](auto mean, auto delta, auto candidate) -> NumericalStatus
        {
            return OMP::ForRowsStatus(0, partitions, [&](index part) -> NumericalStatus
                                      {
            auto &partial = workspace.partials[part];
            partial = {};
            auto &limited = partial.limited;
            auto &minimum = partial.minimum;
            auto &frozen = partial.frozen;
            SpanView<real> repairedSpecies = independent
                ? SpanView<real>{workspace.species.data() + part * independent, independent}
                : SpanView<real>{};
            const index begin = part * (mean.RowCount() / partitions) + std::min(part, mean.RowCount() % partitions);
            const index end = begin + mean.RowCount() / partitions + (part < mean.RowCount() % partitions);
            for (index iCell = begin; iCell < end; ++iCell)
            {
                const ThermalAdmissibilityDetail::PointRow<decltype(mean)> old{mean, iCell};
                using Selected = SelectedIncrement<decltype(delta)>;
                Selected change{delta, iCell, scale, active};
                real alpha = 1;
                if (control.positivity)
                {
                    // PP POLICY -- preliminary solved-increment forecast.
                    // Match Euler EvaluateCellRHSAlpha: one clipped-reference
                    // scalar scales the complete flow/species/RANS correction.
                    // This is not acceptance; the exact species target and
                    // raw-affine FixFraction line search below remain mandatory.
                    alpha = bounds.IncrementFraction(old, change, control.incrementRelax, 2,
                                                     control.compressionPower);
                    if (!(std::isfinite(alpha) && alpha >= 0 && alpha <= 1))
                        return NumericalStatus{NumericalError::InvalidState, 1, iCell};
                }
                using Limited = LimitedIncrement<Selected>;
                Limited limitedChange{change, alpha};
                if (control.positivity && independent)
                {
                    // PP POLICY -- exact stored species target. Repair once
                    // before final thermal restriction; this clipped rhoY is
                    // the actual accepted/exchanged mean and EOS/T state. Both
                    // endpoints are in the simplex, so the cheaper raw-affine
                    // reference equals the clipped reference along the entire
                    // line and merely proves the clipped stored-state invariant.
                    // A raw out-of-simplex mean is never accepted, and no
                    // projection may follow FixFraction. In contrast,
                    // reconstruction retains raw polynomial rhoY for transport
                    // and clips only its temporary point-property view.
                    for (index k = 0; k < independent; ++k)
                        repairedSpecies[size_t(k)] = old[control.speciesFirst + k] + limitedChange[control.speciesFirst + k];
                    struct SpeciesScratch
                    {
                        SpanView<real> values;
                        real &operator[](index k) const { return values[k]; }
                    } species{{repairedSpecies.data(), index(repairedSpecies.size())}};
                    Physics::Chemistry::RepairSpeciesDensities(
                        old[control.flowFirst] + limitedChange[control.flowFirst], species, independent);
                }
                StoredIncrement<Limited, decltype(old)> storedChange{limitedChange, old,
                                                                     control.positivity && independent ? SpanView<const real>{repairedSpecies.data(), index(repairedSpecies.size())} : SpanView<const real>{},
                                                                     control.speciesFirst, independent};
                real fix = 1;
                if (control.positivity)
                {
                    fix = bounds.FixFraction(old, storedChange);
                    if (!(std::isfinite(fix) && fix >= 0 && fix <= 1))
                        return NumericalStatus{NumericalError::InvalidState, 2, iCell};
                    const real combined = alpha * fix;
                    if (combined < 1)
                    {
                        ++limited;
                        minimum = std::min(minimum, combined);
                    }
                }
                for (index j = 0; j < components; ++j)
                {
                    if (control.positivity && independent &&
                        j >= control.speciesFirst && j < control.speciesFirst + independent)
                    {
                        const real target = repairedSpecies[size_t(j - control.speciesFirst)];
                        // Publish the exact repaired target when the final
                        // thermal line needs no contraction. Computing
                        // old+(target-old) can erase a tiny positive species
                        // floor by cancellation. The convex form is retained
                        // when FixFraction genuinely restricts the line.
                        candidate(iCell, j) = fix == 1
                                                  ? target
                                                  : (1 - fix) * old[j] + fix * target;
                    }
                    else
                        candidate(iCell, j) = mean(iCell, j) + fix * storedChange[j];
                    if (!(active.empty() || active[j]))
                        frozen = frozen && candidate(iCell, j) == mean(iCell, j);
                    // Exponential scalar repair must not hide an infinite or
                    // NaN linear correction behind a finite floor.
                    if (!(std::isfinite(candidate(iCell, j))))
                        return NumericalStatus{NumericalError::InvalidState, 3, iCell};
                }
                if (rans.components)
                {
                    std::array<real, 2> oldRANS{}, changeRANS{}, nextRANS{};
                    for (int j = 0; j < rans.components; ++j)
                    {
                        oldRANS[j] = mean(iCell, control.ransFirst + j);
                        changeRANS[j] = fix * storedChange[control.ransFirst + j];
                    }
                    if (!(rans.Admissible(oldRANS)))
                        return NumericalStatus{NumericalError::InvalidState, 4, iCell};
                    // Preserve Euler's order: whole-vector thermal compression,
                    // then component-wise RANS repair of that scaled increment.
                    rans.RepairIncrement(oldRANS, changeRANS);
                    for (int j = 0; j < rans.components; ++j)
                        nextRANS[j] = oldRANS[j] + changeRANS[j];
                    const real rho = control.flowFirst < 0 ? real(1) : candidate(iCell, control.flowFirst);
                    rans.RepairMean(rho, nextRANS);
                    for (int j = 0; j < rans.components; ++j)
                    {
                        const index component = control.ransFirst + j;
                        candidate(iCell, component) = nextRANS[j];
                        if (!(active.empty() || active[component]))
                            frozen = frozen && nextRANS[j] == oldRANS[j];
                    }
                }
            }
            return {}; });
        };
        const auto meanTable = Algebra::TryOwnedTable(inputMean), deltaTable = Algebra::TryOwnedTable(inputDelta);
        const auto candidateTable = Algebra::TryOwnedTable(output);
        const auto status = meanTable && deltaTable && candidateTable
                                ? apply(*meanTable, *deltaTable, *candidateTable)
                                : apply(inputMean, inputDelta, output);
        if (!status)
            return status;
        for (index part = 0; part < partitions; ++part)
        {
            result.limited += workspace.partials[part].limited;
            result.minimum = std::min(result.minimum, workspace.partials[part].minimum);
            result.frozen = result.frozen && workspace.partials[part].frozen;
        }
        return result;
    }
    NumericalResult<AlphaStatistics> PreparePseudoTime(const PseudoTimeRequest &request) noexcept
    {
        index rhsLimited = 0;
        real rhsAlpha = 1;
        const auto metric = request.metric;
        const auto spectral = request.spectral;
        const auto inverse = request.inverse;
        const auto means = request.means;
        const auto defect = request.defect;
        const auto bounds = request.bounds;
        const index nv = defect.ComponentNumber();
        index firstBadTime = inverse.RowCount(), firstBadAlpha = inverse.RowCount();
        const int threads = OMP::DistributedThreads();
        // Independent cell updates; only exact counts and minima are merged.
        // Numerical failures remain explicit and report the first logical row.
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1) reduction(+ : rhsLimited) reduction(min : rhsAlpha, firstBadTime, firstBadAlpha)
#endif
        for (index iCell = 0; iCell < inverse.RowCount(); ++iCell)
        {
            const real volume = metric.GetCellVol(iCell);
            real value = spectral(iCell, 0) / (request.pseudoCFL * volume);
            if (!(std::isfinite(value) && value > 0))
            {
                firstBadTime = std::min(firstBadTime, iCell);
                continue;
            }
            if (request.positivity)
            {
                // PP POLICY -- whole-vector forecast. The trial
                // includes every coupled component, while its
                // mixture energy reference uses only a temporary
                // clipped/normalized composition. No species
                // simplex alpha or storage mutation occurs. The
                // resulting scalar applies to the complete defect
                // or common dTau; ApplyCorrectionImpl owns the
                // exact repaired-mean guarantee.
                const ThermalAdmissibilityDetail::PointRow<decltype(means)> mean{means, iCell};
                struct TrialIncrement
                {
                    decltype(defect) values;
                    index row;
                    real factor;
                    real operator[](index j) const { return values(row, j) * factor; }
                } delta{defect, iCell, request.forecastScale / (value * volume)};
                const real alpha = bounds.IncrementFraction(mean, delta, request.relax, 2, request.compressionPower);
                if (!(std::isfinite(alpha) && alpha >= 0 && alpha <= 1))
                {
                    firstBadAlpha = std::min(firstBadAlpha, iCell);
                    continue;
                }
                if (alpha < 1)
                {
                    ++rhsLimited;
                    rhsAlpha = std::min(rhsAlpha, alpha);
                }
                if (request.alphas)
                    (*request.alphas)(iCell, 0) = alpha;
                else if (request.compressRHS)
                    for (index j = 0; j < nv; ++j)
                        defect(iCell, j) *= alpha;
                else
                    value /= std::max(alpha, smallReal);
            }
            inverse(iCell, 0) = value;
        }
        if (firstBadTime < inverse.RowCount() && firstBadTime <= firstBadAlpha)
            return NumericalStatus{NumericalError::InvalidState, 1, firstBadTime};
        if (firstBadAlpha < inverse.RowCount())
            return NumericalStatus{NumericalError::InvalidState, 2, firstBadAlpha};
        return AlphaStatistics{rhsLimited, rhsAlpha};
    }
    void ApplyAgreedRHSAlpha(
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> alpha,
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> inverse,
        Algebra::DofComponentView<real> defect, real minimum, AlphaReduction reduction, bool compressRHS)
    {
        OMP::ForRows(0, alpha.RowCount(), [&](index iCell)
                     {
            const real fraction = SelectAlpha(alpha(iCell, 0), minimum, reduction);
            if (compressRHS)
                for (index j = 0; j < defect.ComponentNumber(); ++j)
                    defect(iCell, j) *= fraction;
            else
                inverse(iCell, 0) /= std::max(fraction, smallReal); });
    }
}
