#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumePositivityKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        enum class Operation
        {
            ValidateMeans,
            MarkDegrees,
            Truncate,
            SelectDegree,
            Compress,
            MarkRecovery,
            Recover,
            Verify
        };
        struct Fields
        {
            PPInput samples, base, reference;
            PPOutput coefficients;
            PPIndexInput degrees;
            PPIndexOutput outputDegrees;
            PPFlagInput recover;
            PPFlagOutput flag;
            PPScalarInput recoveryScale;
            PPScalarOutput beta, outputRecoveryScale;
            bool hasBase = false, hasReference = false;
            int order = 0;
        };
        struct RepeatedMean
        {
            PPInput values;
            index cell;
            DNDS_DEVICE_CALLABLE real operator()(index, index component) const { return values(cell, component); }
        };
        DNDS_DEVICE_CALLABLE bool ReferenceGood(PPInput values, index q, const ThermalView &bounds)
        {
            const ThermalAdmissibilityDetail::PointRow<PPInput> point{values, q};
            return bounds.RawThermallyAdmissible(point) && bounds.ClippedReferenceAdmissible(point);
        }
        template <Operation Op>
        DNDS_DEVICE_CALLABLE NumericalStatus Process(PolynomialPositivityRequest r, Fields f, index iCell) noexcept
        {
            const auto &p = r.policy;
            const index begin = r.offsets(iCell, 0), end = r.offsets(iCell + 1, 0);
            const ThermalAdmissibilityDetail::PointRow<PPInput> mean{r.means, iCell};
            const RepeatedMean repeated{r.means, iCell};
            if constexpr (Op == Operation::ValidateMeans)
            {
                if (!p.bounds.Admissible(mean))
                    return NumericalStatus{NumericalError::InvalidState, 1}.At(iCell);
            }
            if constexpr (Op == Operation::MarkDegrees)
            {
                f.flag(iCell, 0) = 0;
                bool good = true;
                for (index q = begin; q < end; ++q)
                {
                    for (int j = 0; j < p.components; ++j)
                        if (!std::isfinite(f.samples(q, j)))
                            return NumericalStatus{NumericalError::InvalidState, 1}.At(iCell, q);
                    good = good && ReferenceGood(f.samples, q, p.bounds);
                }
                f.outputDegrees(iCell, 0) = good ? p.maxOrder : -1;
                f.flag(iCell, 0) = uint8_t(good && (!p.rans.components || p.rans.Compute(repeated, f.samples, 5, begin, end) == 1));
            }
            if constexpr (Op == Operation::Truncate)
            {
                if (f.degrees(iCell, 0) < 0)
                    for (int j = 0; j < 5; ++j)
                        for (int mode = f.order; mode < p.modes; ++mode)
                            f.coefficients(iCell, j * p.modes + mode) = 0;
            }
            if constexpr (Op == Operation::SelectDegree)
            {
                if (f.outputDegrees(iCell, 0) < 0)
                {
                    bool good = true;
                    for (index q = begin; q < end; ++q)
                        good = good && ReferenceGood(f.samples, q, p.bounds);
                    if (good)
                        f.outputDegrees(iCell, 0) = f.order;
                    else
                        for (int j = 0; j < 5 * p.modes; ++j)
                            f.coefficients(iCell, j) = f.reference(iCell, j);
                }
            }
            if constexpr (Op == Operation::Compress)
            {
                f.beta(iCell, 0) = 1;
                const real scale = p.rans.Compute(repeated, f.samples, 5, begin, end);
                if (!(std::isfinite(scale) && scale >= 0 && scale <= 1))
                    return NumericalStatus{NumericalError::InvalidState, 1}.At(iCell);
                // PP POLICY -- RANS polynomials are independent from flow5.
                if (scale < 1)
                    for (int j = 5; j < p.components; ++j)
                        for (int mode = 0; mode < p.modes; ++mode)
                            f.coefficients(iCell, j * p.modes + mode) *= scale;
                if (f.degrees(iCell, 0) == p.maxOrder)
                    return {};
                real beta = 1;
                if (p.segment)
                {
                    const auto flow = p.bounds.Flow(mean);
                    for (index q = begin; q < end; ++q)
                        beta = std::min(beta, AdmissibleFraction(flow, p.bounds.Flow(ThermalAdmissibilityDetail::PointRow<PPInput>{f.samples, q}), p.segmentBounds));
                    if (beta < p.betaCutoff)
                        beta = 0;
                }
                else
                    beta = f.hasBase ? p.bounds.ReconstructionFraction(mean, f.base, f.samples, begin, end, p.betaPower, p.betaCutoff, 1, p.clippedPolicy)
                                     : p.bounds.ReconstructionFraction(mean, repeated, f.samples, begin, end, p.betaPower, p.betaCutoff, 1, p.clippedPolicy);
                if (!(std::isfinite(beta) && beta >= 0 && beta <= 1))
                    return NumericalStatus{NumericalError::InvalidState, 2}.At(iCell);
                f.beta(iCell, 0) = beta;
                if (beta < 1)
                    for (int j = 0; j < 5 * p.modes; ++j)
                    {
                        const real base = f.hasReference ? f.reference(iCell, j) : 0;
                        f.coefficients(iCell, j) = base + beta * (f.coefficients(iCell, j) - base);
                    }
            }
            if constexpr (Op == Operation::MarkRecovery)
            {
                bool good = true;
                for (index q = begin; q < end; ++q)
                    good = good && p.bounds.ClippedReferenceAdmissible(ThermalAdmissibilityDetail::PointRow<PPInput>{f.samples, q});
                f.flag(iCell, 0) = uint8_t(!good);
                if (!good)
                {
                    const real previous = f.beta(iCell, 0);
                    f.beta(iCell, 0) = p.clippedPolicy == ReconstructionClippedReferencePolicy::FallbackToBase ? 0 : previous * .9;
                    if (f.beta(iCell, 0) < p.betaCutoff)
                        f.beta(iCell, 0) = 0;
                    if (!(previous > 0))
                        return NumericalStatus{NumericalError::InvalidState, 1}.At(iCell);
                    f.outputRecoveryScale(iCell, 0) = f.beta(iCell, 0) / previous;
                }
            }
            if constexpr (Op == Operation::Recover)
            {
                if (f.recover(iCell, 0))
                    for (int j = 0; j < 5 * p.modes; ++j)
                    {
                        const real base = f.hasReference ? f.reference(iCell, j) : 0;
                        f.coefficients(iCell, j) = base + f.recoveryScale(iCell, 0) * (f.coefficients(iCell, j) - base);
                    }
            }
            if constexpr (Op == Operation::Verify)
            {
                // PP POLICY -- check actual post-storage point values. No
                // coefficient acceptance is inferred from a computed beta.
                for (index q = begin; q < end; ++q)
                    if (!p.bounds.ClippedReferenceAdmissible(ThermalAdmissibilityDetail::PointRow<PPInput>{f.samples, q}))
                        return NumericalStatus{NumericalError::InvalidState, 1}.At(iCell, q);
            }
            return {};
        }
        template <Operation Op>
        __global__ void Cells(PolynomialPositivityRequest r, Fields fields)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < r.means.RowCount(); iCell += index(gridDim.x) * blockDim.x)
                r.status(iCell, 0) = Process<Op>(r, fields, iCell);
        }
        template <int Components>
        __global__ void PhysicalStep(PPInput mean, PPInput rhs, R::ArrayView<const real, false, R::FixedRows<8>> cells,
                                     ThermalView thermal, real dt, real relax, real scale, int power, PPScalarOutput alpha, PPStatus status)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < mean.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                NumericalStatus resultStatus;
                std::array<real, Components> increment;
                const auto row = mean[iCell];
                if (!thermal.Admissible(row))
                    resultStatus = NumericalStatus{NumericalError::InvalidState, 1}.At(iCell);
                for (int j = 0; j < Components; ++j)
                {
                    increment[j] = (dt * scale) * rhs(iCell, j) / cells(iCell, 0);
                    if (!std::isfinite(increment[j]))
                        resultStatus = NumericalStatus{NumericalError::InvalidState, 2}.At(iCell);
                }
                if (resultStatus)
                {
                    const real value = thermal.IncrementFraction(row, increment, relax, 1, power);
                    if (std::isfinite(value) && value >= 0 && value <= 1)
                        alpha(iCell, 0) = value;
                    else
                        resultStatus = NumericalStatus{NumericalError::InvalidState, 3}.At(iCell);
                }
                status(iCell, 0) = resultStatus;
            }
        }
        template <class View>
        void CheckRows(R::ExecutionContext context, View view, index rows)
        {
            view.ValidateDevice(context.device);
            DNDS_check_throw_info(view.RowCount() == rows, "CUDA PP cell field row count differs");
        }
        template <class View>
        void CheckCoefficients(R::ExecutionContext context, const PolynomialPositivityRequest &r, View view)
        {
            CheckRows(context, view, r.means.RowCount());
            DNDS_check_throw_info(view.kind == R::RowLayoutKind::Table && view.Width() == r.policy.components * r.policy.modes,
                                  "CUDA PP coefficient layout/width differs");
        }
        void CheckSamples(R::ExecutionContext context, const PolynomialPositivityRequest &r, PPInput view)
        {
            view.ValidateDevice(context.device);
            DNDS_check_throw_info(view.kind == R::RowLayoutKind::Table && view.Width() == r.policy.components,
                                  "CUDA PP sample layout/width differs");
            // The immutable setup offsets describe this complete point set;
            // reading their terminal value here would force a device/host sync.
        }
        template <Operation Op>
        void Launch(R::ExecutionContext context, PolynomialPositivityRequest r, Fields f = {})
        {
            r.Validate(context);
            R::CheckKernelArgumentBudget<PolynomialPositivityRequest, Fields>();
            if (!r.means.RowCount())
                return;
            const auto blocks = unsigned(std::min<index>((r.means.RowCount() + 127) / 128, 1024));
            Cells<Op><<<blocks, 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(r, f);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA polynomial PP launch failed");
        }
    }
    void PolynomialPositivityParameters::Validate() const
    {
        DNDS_check_throw_info(components >= 5 && components <= 7 && modes > 0 && modes <= 19 && maxOrder >= 1 && maxOrder <= 3 &&
                                  betaPower >= 0 && std::isfinite(betaCutoff) && betaCutoff >= 0 && betaCutoff <= 1 &&
                                  bounds.components == components && bounds.flowFirst == 0 && bounds.independent == 0 &&
                                  bounds.speciesEnergies[0].empty() && bounds.speciesEnergies[1].empty() &&
                                  rans.components + 5 == components && (!segment || components == 5),
                              "unsupported CUDA perfect-gas reconstruction PP policy");
        DNDS_check_throw_info(clippedPolicy == ReconstructionClippedReferencePolicy::FallbackToBase || clippedPolicy == ReconstructionClippedReferencePolicy::Backtracking,
                              "unknown CUDA PP endpoint policy");
    }
    void PolynomialPositivityRequest::Validate(R::ExecutionContext context) const
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA PP requires a CUDA context");
        policy.Validate();
        R::ValidateViewDevices(context.device, means, offsets, status);
        DNDS_check_throw_info(means.kind == R::RowLayoutKind::Table && means.Width() == policy.components &&
                                  offsets.RowCount() == means.RowCount() + 1 && status.RowCount() == means.RowCount(),
                              "CUDA PP input shape/layout differs");
    }
    void ValidateReconstructionMeans(R::ExecutionContext c, PolynomialPositivityRequest r) { Launch<Operation::ValidateMeans>(c, r); }
    void MarkAdmissibleDegrees(R::ExecutionContext c, PolynomialPositivityRequest r, PPInput samples, PPIndexOutput degrees, PPFlagOutput good)
    {
        CheckSamples(c, r, samples);
        CheckRows(c, degrees, r.means.RowCount());
        CheckRows(c, good, r.means.RowCount());
        Fields f;
        f.samples = samples;
        f.outputDegrees = degrees;
        f.flag = good;
        Launch<Operation::MarkDegrees>(c, r, f);
    }
    void TruncateReferenceModes(R::ExecutionContext c, PolynomialPositivityRequest r, PPIndexInput degrees, PPOutput coefficients, int keep)
    {
        CheckRows(c, degrees, r.means.RowCount());
        CheckCoefficients(c, r, coefficients);
        DNDS_check_throw_info(keep >= 0 && keep <= r.policy.modes, "CUDA PP truncation is outside modal range");
        Fields f;
        f.degrees = degrees;
        f.coefficients = coefficients;
        f.order = keep;
        Launch<Operation::Truncate>(c, r, f);
    }
    void SelectReferenceDegree(R::ExecutionContext c, PolynomialPositivityRequest r, PPInput samples, PPInput reference, PPIndexOutput degrees, PPOutput coefficients, int order)
    {
        CheckSamples(c, r, samples);
        CheckCoefficients(c, r, reference);
        CheckCoefficients(c, r, coefficients);
        CheckRows(c, degrees, r.means.RowCount());
        DNDS_check_throw_info(order >= 0 && order < r.policy.maxOrder, "CUDA PP reference order differs");
        Fields f;
        f.samples = samples;
        f.reference = reference;
        f.outputDegrees = degrees;
        f.coefficients = coefficients;
        f.order = order;
        Launch<Operation::SelectDegree>(c, r, f);
    }
    void CompressPolynomial(R::ExecutionContext c, PolynomialPositivityRequest r, PPInput higher, PPInput base, PPInput reference,
                            bool hasBase, bool hasReference, PPIndexInput degrees, PPOutput coefficients, PPScalarOutput beta)
    {
        CheckSamples(c, r, higher);
        CheckRows(c, degrees, r.means.RowCount());
        CheckRows(c, beta, r.means.RowCount());
        CheckCoefficients(c, r, coefficients);
        if (hasBase)
        {
            CheckSamples(c, r, base);
            CheckRows(c, base, higher.RowCount());
        }
        if (hasReference)
            CheckCoefficients(c, r, reference);
        Fields f;
        f.samples = higher;
        f.base = base;
        f.reference = reference;
        f.hasBase = hasBase;
        f.hasReference = hasReference;
        f.degrees = degrees;
        f.coefficients = coefficients;
        f.beta = beta;
        Launch<Operation::Compress>(c, r, f);
    }
    void MarkPolynomialRecovery(R::ExecutionContext c, PolynomialPositivityRequest r, PPInput samples, PPScalarOutput beta, PPFlagOutput recover, PPScalarOutput scale)
    {
        CheckSamples(c, r, samples);
        CheckRows(c, beta, r.means.RowCount());
        CheckRows(c, recover, r.means.RowCount());
        CheckRows(c, scale, r.means.RowCount());
        Fields f;
        f.samples = samples;
        f.beta = beta;
        f.flag = recover;
        f.outputRecoveryScale = scale;
        Launch<Operation::MarkRecovery>(c, r, f);
    }
    void RecoverPolynomial(R::ExecutionContext c, PolynomialPositivityRequest r, PPInput reference, bool hasReference, PPOutput coefficients, PPFlagInput recover, PPScalarInput scale)
    {
        CheckCoefficients(c, r, coefficients);
        CheckRows(c, recover, r.means.RowCount());
        CheckRows(c, scale, r.means.RowCount());
        if (hasReference)
            CheckCoefficients(c, r, reference);
        Fields f;
        f.reference = reference;
        f.hasReference = hasReference;
        f.coefficients = coefficients;
        f.recover = recover;
        f.recoveryScale = scale;
        Launch<Operation::Recover>(c, r, f);
    }
    void VerifyReconstructedPoints(R::ExecutionContext c, PolynomialPositivityRequest r, PPInput samples)
    {
        CheckSamples(c, r, samples);
        Fields f;
        f.samples = samples;
        Launch<Operation::Verify>(c, r, f);
    }
    void LimitPhysicalStepRows(R::ExecutionContext context, PPInput means, PPInput rhs,
                               R::ArrayView<const real, false, R::FixedRows<8>> cells, ThermalView thermal,
                               real dt, real relax, real scale, int power, PPScalarOutput alpha, PPStatus status)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA physical-dt forecast requires a CUDA context");
        R::ValidateViewDevices(context.device, means, rhs, cells, alpha, status);
        DNDS_check_throw_info(means.kind == R::RowLayoutKind::Table && rhs.kind == R::RowLayoutKind::Table &&
                                  means.Width() >= 5 && means.Width() <= 7 && rhs.Width() == means.Width() && rhs.RowCount() == means.RowCount() &&
                                  cells.RowCount() >= means.RowCount() && alpha.RowCount() == means.RowCount() && status.RowCount() == means.RowCount() &&
                                  thermal.components == means.Width() && thermal.flowFirst == 0 && thermal.independent == 0 &&
                                  thermal.speciesEnergies[0].empty() && thermal.speciesEnergies[1].empty() &&
                                  std::isfinite(dt) && dt >= 0 && std::isfinite(relax) && std::isfinite(scale) && power >= 0,
                              "CUDA physical-dt forecast shape/profile differs");
        R::CheckKernelArgumentBudget<PPInput, PPInput, decltype(cells), ThermalView, real, real, real, int, PPScalarOutput, PPStatus>();
        if (!means.RowCount())
            return;
        const auto blocks = unsigned(std::min<index>((means.RowCount() + 127) / 128, 1024));
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        switch (means.Width())
        {
        case 5:
            PhysicalStep<5><<<blocks, 128, 0, stream>>>(means, rhs, cells, thermal, dt, relax, scale, power, alpha, status);
            break;
        case 6:
            PhysicalStep<6><<<blocks, 128, 0, stream>>>(means, rhs, cells, thermal, dt, relax, scale, power, alpha, status);
            break;
        case 7:
            PhysicalStep<7><<<blocks, 128, 0, stream>>>(means, rhs, cells, thermal, dt, relax, scale, power, alpha, status);
            break;
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA physical-dt forecast launch failed");
    }
}
