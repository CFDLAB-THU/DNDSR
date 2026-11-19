#pragma once

#include "Cauchy/Implicit/CUDA/LinearBlockKernels.hpp"
#include "Cauchy/Implicit/StageAssembly.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Implicit/PositivityReduction.hpp"

namespace DNDS::Cauchy::CUDA
{
    using NonlinearCells = Runtime::ArrayView<const real, false, Runtime::FixedRows<8>>;

    /** @brief Compact per-block/final nonlinear reduction, never per-cell scratch.
     * @details norms contains raw component integrals and volume at index N.
     * For ACM defect observation it also contains the raw continuity squared integral.
     * Counts/minimum describe thermal compression, not independent RANS repair.
     * frozen verifies inactive components were not changed by scalar repair.
     * A failed status invalidates all numerical values. Empty ranks contribute
     * zeros, minimum=1 and frozen=true. Deterministic trees use no FP atomics.
     */
    struct NonlinearSummary
    {
        std::array<real, 8> norms{};
        index limited = 0;
        real minimum = 1;
        bool frozen = true;
        NumericalStatus status;
    };
    using NonlinearSummaryOutput = Runtime::ArrayView<NonlinearSummary, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using NonlinearSummaryInput = Runtime::ArrayView<const NonlinearSummary, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    /// Caller allocates bounded scratch once for its owned row count.
    index NonlinearScratchRows(index rows);

    struct NonlinearSelection
    {
        std::array<uint8_t, 7> active{1, 1, 1, 1, 1, 1, 1};
    };
    /// Integrated defect = rhsScale*R + volume*(resOther-inverseDt*(M U)).
    /// Norms use the unmasked defect, matching CPU; the output masks inactive
    /// components. L1 sums abs(defect); L2 sums defect^2/volume. No MPI or host
    /// observation is performed, even when measureNorm is enabled.
    void AssembleDefect(Runtime::ExecutionContext, NonlinearCells, LinearReadRows mean,
                        LinearReadRows residual, LinearReadRows constant, LinearWriteRows output,
                        NonlinearSelection, real inverseDt, real rhsScale, int normOrder, bool measureNorm,
                        NonlinearSummaryOutput scratch, NonlinearSummaryOutput result, PhysicalAccumulationKind accumulation = PhysicalAccumulationKind::Identity);

    struct PseudoTimeParameters
    {
        ThermalView thermal;
        real pseudoCFL = 1, forecastScale = 1, relax = 1;
        int compressionPower = 1;
        bool positivity = false, compressRHS = false, deferAlpha = false;
    };
    /// Compute the inverse pseudo-step using face spectra and the optional owned geometry factors.
    void PrepareSpectralPseudoTime(Runtime::ExecutionContext, NonlinearCells, LinearReadRows spectral, LinearReadRows smoothScale,
                                   LinearWriteRows inverseTau, real cfl);
    void PressureGaugeSum(Runtime::ExecutionContext, NonlinearCells, LinearReadRows candidate,
                          NonlinearSummaryOutput scratch, NonlinearSummaryOutput result);
    void ShiftPressure(Runtime::ExecutionContext, LinearWriteRows candidate, real shift);
    /** @brief Euler-compatible RHS/dTau forecast on combined 5/6/7 states.
     * @details No RANS scalar admissibility condition enters alpha. A thermal
     * alpha scales the entire coupled defect or the common pseudo-time. With
     * deferAlpha, publish cell factors without applying them; the controller
     * agrees its selected global policy, then calls ApplyAgreedRHSAlpha.
     * This is a forecast, not final mean acceptance or increment repair.
     */
    void PreparePseudoTime(Runtime::ExecutionContext, NonlinearCells, LinearReadRows mean,
                           LinearWriteRows defect, LinearReadRows spectral, LinearReadRows smoothScale, LinearWriteRows inverseTau,
                           LinearWriteRows alpha, PseudoTimeParameters,
                           NonlinearSummaryOutput scratch, NonlinearSummaryOutput result);
    void ApplyAgreedRHSAlpha(Runtime::ExecutionContext, LinearReadRows alpha, LinearWriteRows inverseTau,
                             LinearWriteRows defect, real minimum, AlphaReduction, bool compressRHS);

    struct CorrectionParameters
    {
        AdmissibilityView admissibility;
        NonlinearSelection selection;
        real scale = 1, incrementRelax = 1;
        int compressionPower = 1;
        bool positivity = true;
    };
    /** @brief Implicit solved-increment repair, preserving CPU/Euler order.
     * @details Select/relax delta, estimate thermal alpha, apply final thermal
     * FixFraction, then separately repair RANS scalars. Species are unsupported
     * in this specialization. Candidate is distinct from both inputs; it is
     * never committed here. Final admissibility and rank agreement belong to
     * the operator/controller before publishing the mean. Do not use this
     * per-cell flow compression in an explicit physical-step update.
     */
    void ApplyCorrection(Runtime::ExecutionContext, LinearReadRows mean, LinearReadRows increment,
                         LinearWriteRows candidate, CorrectionParameters,
                         NonlinearSummaryOutput scratch, NonlinearSummaryOutput result);

    /// Pure rate conversion. multiply=true converts a rate to integrated RHS;
    /// false converts an integrated RHS to rate. Geometry is already validated.
    void ScaleRateByVolume(Runtime::ExecutionContext, NonlinearCells, LinearReadRows input,
                           LinearWriteRows output, bool multiply);
    /// inverseDt*(M U)-rhsScale*R/volume, with optional R/volume publication. An unbound
    /// empty rate view omits that output. All provided buffers are distinct.
    void AssembleEquation(Runtime::ExecutionContext, NonlinearCells, LinearReadRows mean,
                          LinearReadRows residual, LinearWriteRows output, LinearWriteRows rate,
                          real inverseDt, real rhsScale, PhysicalAccumulationKind accumulation = PhysicalAccumulationKind::Identity);
    /// Raw volume-weighted component magnitudes for Euler's GMRES metric.
    /// Momentum/energy normalization and MPI agreement stay on the controller.
    void MeasureMeanState(Runtime::ExecutionContext, NonlinearCells, LinearReadRows mean,
                          NonlinearSummaryOutput scratch, NonlinearSummaryOutput result);
    /// Local integral of abs(rate-reference) or its square. An unbound empty
    /// reference omits subtraction. No payload staging or host observation.
    void MeasureRateNorm(Runtime::ExecutionContext, NonlinearCells, LinearReadRows rate,
                         LinearReadRows reference, int normOrder,
                         NonlinearSummaryOutput scratch, NonlinearSummaryOutput result, LinearReadRows rawACMRate = {});
}
