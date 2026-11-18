#pragma once
#include "Cauchy/Discretization/VolumeSourceBasis.hpp"

#include "Cauchy/Discretization/PerfectGasKernelPoint.hpp"
#include "Cauchy/Discretization/ACMKernelPoint.hpp"
#include "Cauchy/Motion/RigidFrameParameters.hpp"
#include "DNDS/Runtime/Array/Array.hpp"

namespace DNDS::Cauchy::CUDA
{
    using SourceInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using SourceOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using SourceIndices = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using SourceScalar = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using SourceStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    /// Either an interleaved cell-gradient field or one directional quadrature
    /// sample table. A complete whole-array view preserves residency and pitch.
    struct SourceGradient
    {
        SourceInput values;
        int componentStride = 1, direction = 0;
        DNDS_DEVICE_CALLABLE real operator()(index iPoint, int j) const
        {
            return values(iPoint, j * componentStride + direction);
        }
    };
    enum class SourceEvaluation
    {
        QuadratureResidual,
        MeanDestruction,
        QuadratureDestruction
    };
    /** Frozen operands for a single owned-cell source pass. Direct evaluation
     * borrows means and physical gradients. Modal evaluation reconstructs
     * from finalized coefficients and the immutable quadrature basis.
     * The optional output stores the positive RANS destruction approximation.
     * Standalone comparison fixtures can supply already sampled point views.
     */
    struct SourceRequest
    {
        Runtime::ArrayView<const PerfectGasSpatialParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> parameters;
        Runtime::ArrayView<const ACM::SpatialParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> acm;
        SourceInput means, samples, coefficients;
        VolumeSourceBasis basis;
        std::array<SourceGradient, 3> gradients;
        SourceIndices offsets;
        SourceScalar weights, lengths, wallMean;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::CSRRows> wallPoints;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> output;
        /// Optional canonical row-major dense (or diagonal-only) integrated
        /// contribution for the RHS evaluation. No temporal weight is applied.
        decltype(output) preparedDerivative;
        SourceStatus status;
        int components = 0, dimension = 0;
        SourceEvaluation evaluation = SourceEvaluation::QuadratureResidual;
        bool fullBlock = false, cellInputs = false;
        real scale = 1;
    };
    /// One independent thread per selected owned cell, quadrature in CPU order.
    /// The numerical source is committed only after all its points succeed.
    /// TODO: benchmark one CTA per modal cell with one thread per point and
    /// in-block reductions into the same RHS and optional derivative rows.
    void AddRANSSources(Runtime::ExecutionContext context, SourceRequest request, SourceIndices selected);

    void ApplyVolumeSourceDerivative(Runtime::ExecutionContext context, SourceInput source, decltype(SourceRequest::output) diagonal,
                                     index components, bool fullBlock, bool rowMajor, real scale);
    /// Add the absolute-velocity component-axis rotation source to flow5 only.
    /// Optional low-order output receives the same source. RANS components are
    /// unchanged. No field transform or geometry mutation is performed.
    void AddRotationSource(Runtime::ExecutionContext context,
                           Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells,
                           RigidFrameParameters frame, SourceInput means, SourceOutput output,
                           SourceOutput lowOutput, bool produceLow, SourceIndices selected);
}
