#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Implicit/CUDA/NonlinearKernels_Detail.hpp"
#include <limits>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace D = NonlinearDetail;
    namespace
    {
        struct SelectedIncrement
        {
            LinearReadRows values;
            index row;
            real scale;
            NonlinearSelection selection;
            DNDS_DEVICE_CALLABLE real operator[](index j) const
            {
                return selection.active[j] ? scale * values(row, j) : 0;
            }
        };
        struct ScaledIncrement
        {
            SelectedIncrement values;
            real factor;
            DNDS_DEVICE_CALLABLE real operator[](index j) const { return factor * values[j]; }
        };
        __device__ void Fail(NonlinearSummary &summary, index row, int detail)
        {
            if (summary.status)
                summary.status = {NumericalError::InvalidState, detail, row};
        }
        __global__ void SpectralPseudoTime(NonlinearCells cells, LinearReadRows spectral, LinearReadRows smoothScale, LinearWriteRows inverse, real cfl)
        {
            for (index row = index(blockIdx.x) * blockDim.x + threadIdx.x; row < inverse.RowCount(); row += index(gridDim.x) * blockDim.x)
            {
                const real volume = cells(row, 0);
                const real factor = smoothScale.RowCount() ? smoothScale(row, 0) : 1;
                const real value = spectral(row, 0) / (volume * cfl * factor);
                inverse(row, 0) = volume > 0 && value > 0 && std::isfinite(value)
                                      ? value
                                      : std::numeric_limits<real>::quiet_NaN();
            }
        }
        __global__ void PseudoTime(NonlinearCells cells, LinearReadRows mean, LinearWriteRows defect,
                                   LinearReadRows spectral, LinearReadRows smoothScale, LinearWriteRows inverseTau, LinearWriteRows alpha,
                                   PseudoTimeParameters parameters, NonlinearSummaryOutput scratch)
        {
            NonlinearSummary local;
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < mean.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const real volume = cells(iCell, 0);
                const real factor = smoothScale.RowCount() ? smoothScale(iCell, 0) : 1;
                real inverse = spectral(iCell, 0) / (parameters.pseudoCFL * volume * factor);
                if (!(std::isfinite(inverse) && inverse > 0 && volume > 0))
                {
                    Fail(local, iCell, 1);
                    inverseTau(iCell, 0) = std::numeric_limits<real>::quiet_NaN();
                    if (parameters.deferAlpha)
                        alpha(iCell, 0) = std::numeric_limits<real>::quiet_NaN();
                    continue;
                }
                real fraction = 1;
                if (parameters.positivity)
                {
                    // PP POLICY -- forecast the WHOLE coupled defect. Only
                    // flow thermal bounds restrict this scalar; RANS scalar
                    // bounds never restrict RHS/dTau alpha. This is not final
                    // mean acceptance and is not an explicit time-step repair.
                    const ThermalAdmissibilityDetail::PointRow<LinearReadRows> old{mean, iCell};
                    const SelectedIncrement delta{defect.ConstView(), iCell, parameters.forecastScale / (inverse * volume), {}};
                    fraction = parameters.thermal.IncrementFraction(old, delta, parameters.relax, 2, parameters.compressionPower);
                    if (!(std::isfinite(fraction) && fraction >= 0 && fraction <= 1))
                    {
                        Fail(local, iCell, 2);
                        inverseTau(iCell, 0) = std::numeric_limits<real>::quiet_NaN();
                        if (parameters.deferAlpha)
                            alpha(iCell, 0) = std::numeric_limits<real>::quiet_NaN();
                        continue;
                    }
                    if (fraction < 1)
                    {
                        ++local.limited;
                        local.minimum = std::min(local.minimum, fraction);
                    }
                    if (!parameters.deferAlpha)
                    {
                        if (parameters.compressRHS)
                            for (index j = 0; j < mean.Width(); ++j)
                                defect(iCell, j) *= fraction;
                        else
                            inverse /= std::max(fraction, smallReal);
                    }
                }
                if (parameters.deferAlpha)
                    alpha(iCell, 0) = fraction;
                inverseTau(iCell, 0) = inverse;
            }
            D::PublishBlock(local, scratch);
        }
        __global__ void AgreedAlpha(LinearReadRows alpha, LinearWriteRows inverseTau, LinearWriteRows defect,
                                    real minimum, AlphaReduction reduction, bool compressRHS)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < alpha.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const real fraction = SelectAlpha(alpha(iCell, 0), minimum, reduction);
                if (compressRHS)
                    for (index j = 0; j < defect.Width(); ++j)
                        defect(iCell, j) *= fraction;
                else
                    inverseTau(iCell, 0) /= std::max(fraction, smallReal);
            }
        }
        __global__ void Correction(LinearReadRows mean, LinearReadRows increment, LinearWriteRows candidate,
                                   CorrectionParameters parameters, NonlinearSummaryOutput scratch)
        {
            NonlinearSummary local;
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < mean.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const ThermalAdmissibilityDetail::PointRow<LinearReadRows> old{mean, iCell};
                const SelectedIncrement change{increment, iCell, parameters.scale, parameters.selection};
                real alpha = 1, fix = 1;
                // PP POLICY -- preserve CPU/Euler solved-increment order:
                // thermal forecast -> final thermal line search -> separately
                // repaired RANS increment and mean. No species specialization
                // is allowed here. The operator validates the complete final
                // candidate before any commit; failed rows are poisoned.
                if (parameters.positivity)
                    alpha = parameters.admissibility.thermal.IncrementFraction(old, change,
                                                                               parameters.incrementRelax, 2, parameters.compressionPower);
                const ScaledIncrement scaled{change, alpha};
                if (!(std::isfinite(alpha) && alpha >= 0 && alpha <= 1))
                    Fail(local, iCell, 1);
                else if (parameters.positivity)
                    fix = parameters.admissibility.thermal.FixFraction(old, scaled);
                bool valid = std::isfinite(alpha) && alpha >= 0 && alpha <= 1 &&
                             std::isfinite(fix) && fix >= 0 && fix <= 1;
                if (!valid)
                    Fail(local, iCell, 2);
                if (parameters.positivity && valid && alpha * fix < 1)
                {
                    ++local.limited;
                    local.minimum = std::min(local.minimum, alpha * fix);
                }
                for (index j = 0; j < mean.Width(); ++j)
                {
                    const real value = old[j] + fix * scaled[j];
                    candidate(iCell, j) = value;
                    if (!parameters.selection.active[j])
                        local.frozen = local.frozen && value == old[j];
                    if (!std::isfinite(value))
                    {
                        valid = false;
                        Fail(local, iCell, 3);
                    }
                }
                const auto rans = parameters.admissibility.rans;
                const index first = parameters.admissibility.ransFirst;
                if (valid && rans.components)
                {
                    std::array<real, 2> oldRANS{}, deltaRANS{}, nextRANS{};
                    for (int j = 0; j < rans.components; ++j)
                    {
                        oldRANS[j] = old[first + j];
                        deltaRANS[j] = fix * scaled[first + j];
                    }
                    if (!rans.Admissible(oldRANS))
                    {
                        valid = false;
                        Fail(local, iCell, 4);
                    }
                    else
                    {
                        rans.RepairIncrement(oldRANS, deltaRANS);
                        for (int j = 0; j < rans.components; ++j)
                            nextRANS[j] = oldRANS[j] + deltaRANS[j];
                        rans.RepairMean(parameters.admissibility.thermal.flowFirst < 0 ? real(1) : candidate(iCell, 0), nextRANS);
                        for (int j = 0; j < rans.components; ++j)
                        {
                            candidate(iCell, first + j) = nextRANS[j];
                            if (!parameters.selection.active[first + j])
                                local.frozen = local.frozen && nextRANS[j] == oldRANS[j];
                        }
                    }
                }
                if (!valid)
                    for (index j = 0; j < mean.Width(); ++j)
                        candidate(iCell, j) = std::numeric_limits<real>::quiet_NaN();
            }
            D::PublishBlock(local, scratch);
        }
    }
    void PrepareSpectralPseudoTime(R::ExecutionContext context, NonlinearCells cells, LinearReadRows spectral, LinearReadRows smoothScale,
                                   LinearWriteRows inverse, real cfl)
    {
        const index rows = spectral.RowCount();
        D::CheckCells(context, cells, rows);
        D::CheckRows(context, spectral, rows, 1);
        if (smoothScale.RowCount())
            D::CheckRows(context, smoothScale, rows, 1);
        D::CheckRows(context, inverse, rows, 1);
        if (rows)
        {
            SpectralPseudoTime<<<unsigned(NonlinearScratchRows(rows)), D::Threads, 0, R::CUDAStream(context)>>>(cells, spectral, smoothScale, inverse, cfl);
            D::CheckLaunch();
        }
    }
    void PreparePseudoTime(R::ExecutionContext context, NonlinearCells cells, LinearReadRows mean,
                           LinearWriteRows defect, LinearReadRows spectral, LinearReadRows smoothScale, LinearWriteRows inverseTau,
                           LinearWriteRows alpha, PseudoTimeParameters parameters,
                           NonlinearSummaryOutput scratch, NonlinearSummaryOutput result)
    {
        D::CheckInput(context, mean);
        D::CheckCells(context, cells, mean.RowCount());
        D::CheckRows(context, defect, mean.RowCount(), mean.Width());
        D::CheckRows(context, spectral, mean.RowCount(), 1);
        if (smoothScale.RowCount())
            D::CheckRows(context, smoothScale, mean.RowCount(), 1);
        D::CheckRows(context, inverseTau, mean.RowCount(), 1);
        D::CheckSummary(context, mean.RowCount(), scratch, result);
        if (parameters.positivity)
            D::CheckThermal(parameters.thermal, int(mean.Width()));
        if (parameters.deferAlpha)
            D::CheckRows(context, alpha, mean.RowCount(), 1);
        DNDS_check_throw_info(std::isfinite(parameters.pseudoCFL) && parameters.pseudoCFL > 0 &&
                                  std::isfinite(parameters.forecastScale) && parameters.forecastScale >= 0 &&
                                  std::isfinite(parameters.relax) && parameters.relax >= 0 && parameters.relax <= 1 && parameters.compressionPower > 0,
                              "invalid nonlinear CUDA pseudo-time policy");
        for (const auto target : {defect, inverseTau, parameters.deferAlpha ? alpha : LinearWriteRows{}})
            DNDS_check_throw_info(!target.data || (target.data != mean.data && target.data != cells.data && target.data != spectral.data && target.data != smoothScale.data),
                                  "CUDA pseudo-time output aliases input");
        DNDS_check_throw_info(!defect.data || (defect.data != inverseTau.data && (!parameters.deferAlpha ||
                                                                                  (alpha.data != inverseTau.data && alpha.data != defect.data))),
                              "CUDA pseudo-time outputs alias");
        R::CheckKernelArgumentBudget<NonlinearCells, LinearReadRows, LinearWriteRows, LinearReadRows, LinearReadRows,
                                     LinearWriteRows, LinearWriteRows, PseudoTimeParameters, NonlinearSummaryOutput>();
        PseudoTime<<<unsigned(scratch.RowCount()), D::Threads, 0, ::DNDS::Runtime::CUDAStream(context)>>>(
            cells, mean, defect, spectral, smoothScale, inverseTau, alpha, parameters, scratch);
        D::Finish(context, scratch, result);
    }
    void ApplyAgreedRHSAlpha(R::ExecutionContext context, LinearReadRows alpha, LinearWriteRows inverseTau,
                             LinearWriteRows defect, real minimum, AlphaReduction reduction, bool compressRHS)
    {
        D::CheckInput(context, defect.ConstView());
        D::CheckRows(context, alpha, defect.RowCount(), 1);
        D::CheckRows(context, inverseTau, defect.RowCount(), 1);
        DNDS_check_throw_info(std::isfinite(minimum) && minimum >= 0 && minimum <= 1 && int(reduction) >= 0 && int(reduction) <= 2,
                              "invalid CUDA agreed positivity policy");
        DNDS_check_throw_info(!defect.data || (defect.data != inverseTau.data && defect.data != alpha.data && inverseTau.data != alpha.data),
                              "CUDA agreed alpha fields alias");
        R::CheckKernelArgumentBudget<LinearReadRows, LinearWriteRows, LinearWriteRows, real, AlphaReduction, bool>();
        if (!defect.RowCount())
            return;
        AgreedAlpha<<<unsigned(NonlinearScratchRows(defect.RowCount())), D::Threads, 0, ::DNDS::Runtime::CUDAStream(context)>>>(
            alpha, inverseTau, defect, minimum, reduction, compressRHS);
        D::CheckLaunch();
    }
    void ApplyCorrection(R::ExecutionContext context, LinearReadRows mean, LinearReadRows increment,
                         LinearWriteRows candidate, CorrectionParameters parameters,
                         NonlinearSummaryOutput scratch, NonlinearSummaryOutput result)
    {
        D::CheckInput(context, mean);
        D::CheckRows(context, increment, mean.RowCount(), mean.Width());
        D::CheckRows(context, candidate, mean.RowCount(), mean.Width());
        D::CheckSummary(context, mean.RowCount(), scratch, result);
        const int flowWidth = parameters.admissibility.thermal.flowFirst < 0 ? 4 : 5;
        if (flowWidth == 5)
            D::CheckThermal(parameters.admissibility.thermal, int(mean.Width()));
        else
            DNDS_check_throw_info(!parameters.positivity && parameters.admissibility.thermal.components == mean.Width(),
                                  "ACM correction requires the nonthermal policy");
        const auto rans = parameters.admissibility.rans;
        const bool ransValid = (mean.Width() == flowWidth && rans.components == 0) ||
                               (parameters.admissibility.ransFirst == flowWidth && rans.components == mean.Width() - flowWidth &&
                                ((rans.components == 1 && rans.kind == Physics::RANS::ModelKind::SA) ||
                                 (rans.components == 2 && (rans.kind == Physics::RANS::ModelKind::KOmegaSST || rans.kind == Physics::RANS::ModelKind::KOmegaWilcox))));
        DNDS_check_throw_info(ransValid && std::isfinite(parameters.scale) && parameters.scale > 0 &&
                                  std::isfinite(parameters.incrementRelax) && parameters.incrementRelax >= 0 && parameters.incrementRelax <= 1 && parameters.compressionPower > 0,
                              "invalid CUDA correction policy");
        DNDS_check_throw_info(!candidate.data || (candidate.data != mean.data && candidate.data != increment.data),
                              "CUDA correction candidate aliases input");
        R::CheckKernelArgumentBudget<LinearReadRows, LinearReadRows, LinearWriteRows, CorrectionParameters, NonlinearSummaryOutput>();
        Correction<<<unsigned(scratch.RowCount()), D::Threads, 0, ::DNDS::Runtime::CUDAStream(context)>>>(
            mean, increment, candidate, parameters, scratch);
        D::Finish(context, scratch, result);
    }
}
