#pragma once

#include "Cauchy/Discretization/PerfectGasKernelPoint.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/FiniteVolumeKernelPolicy.hpp"
#include "Cauchy/Boundary/PerfectGasBoundary.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Math/ComputeScalar.hpp"

namespace DNDS::Cauchy::CUDA
{
    using RHSInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using RHSOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using RHSField = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using RHSIndices = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using RHSStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    /// Immutable borrowed numerical geometry; no field values, physics workers
    /// or callback registry. Optional basis/wall views are absent when unused.
    /// The publishing operator retains the underlying metric, stencil and
    /// cache read scopes through every launch using this descriptor.
    struct RHSGeometry
    {
        Runtime::ArrayView<const FiniteVolumeDetail::SampleRecord, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> samples;
        RHSIndices offsets, faceEntries;
        Runtime::ArrayView<const index, false, Runtime::FixedRows<1>> faceOffsets;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<7>> quadrature;
        CFV::ReconstructionPointBasisView basis;
        Runtime::ArrayView<const PerfectGasBoundaryParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> conditions;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> wallMean;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::CSRRows> wallFaces;
        bool hasBasis = false, hasWall = false;
        void ValidateDevice(Runtime::Device device) const;
    };

    /** Small immutable device publication of a large borrowed descriptor.
     * Construct during workspace setup, after all referenced arrays are placed.
     * This owns descriptor bytes only; it does NOT extend nested array lifetimes
     * or grant access. The operator retains their owners/read scopes, declares
     * changing field operands separately, and retains this publication through
     * device completion. No upload/allocation occurs in an RHS launch.
     */
    class RHSGeometryPublication
    {
        struct Data;
        std::shared_ptr<const Data> _data;

    public:
        RHSGeometryPublication(RHSGeometry geometry, Runtime::ExecutionContext context);
        const RHSGeometry &HostDescriptor() const;
        Runtime::ArrayView<const RHSGeometry, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> View() const;
        Runtime::Device Device() const;
        int MultiprocessorCount() const;
    };

    struct RHSKernelParameters
    {
        PerfectGasSpatialParameters physics;
        ThermalView thermal{};
        void Validate(int components) const;
    };
    static_assert(std::is_trivially_copyable_v<RHSKernelParameters>);
    /// Small launch controls travel by value, so host shape validation and
    /// device indexing use the same flags. Immutable physics records live in
    /// resident metadata; per-call time/frame travel by value without uploads.
    struct RHSKernelControls
    {
        FiniteVolumeDetail::FluxKernelPolicy fluxPolicy;
        RigidFrameParameters frame;
        real time = 0;
        Riemann::Solver solver = Riemann::Solver::Rusanov;
        int components = 5, modes = 0, dimension = 3;
        /// Operator-local arithmetic only; persistent inputs, output sums and
        /// PP checks remain FP64. Host dispatch resolves these before launch.
        DNDS::Math::ComputePrecision polynomialPrecision = DNDS::Math::ComputePrecision::FP64;
        DNDS::Math::ComputePrecision dissipationPrecision = DNDS::Math::ComputePrecision::FP64;
        DNDS::Math::ComputePrecision fluxPrecision = DNDS::Math::ComputePrecision::FP64;
        bool direct = false, linear = false, firstOrder = false;
        bool viscous = false, primitiveGradient = false, distinctViscous = false;
        bool positivity = false, faceFallback = false, useStencil = false, produceLow = false;
        bool warpQuadrature = false;
        int blockSize = 128;
        bool doubleBuffer = false;
        void Validate() const;
    };
    static_assert(std::is_trivially_copyable_v<RHSKernelControls>);

    /// Changing operands; geometry/physics publications are separate. Only
    /// combined uniform-row fields are supported. viscousData denotes primitive
    /// gradients when primitiveGradient is selected, otherwise optional
    /// conservative gradients/Taylor coefficients with the reconstruction shape.
    template <class MeanPolicy = Runtime::UniformTableRows, class CoefficientPolicy = Runtime::UniformTableRows>
    struct RHSFieldsT
    {
        Runtime::PairReadView<real, Runtime::AccessBoundsChecks, MeanPolicy> mean;
        Runtime::PairReadView<real, Runtime::AccessBoundsChecks, CoefficientPolicy> reconstruction, viscousData;
        RHSField waveJump, pressureMinimum;
        RHSInput preparedMean, incoming;
        bool hasPreparedMean = false, hasIncoming = false, hasWave = false, hasPressure = false;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, mean, reconstruction, viscousData);
            if (hasPreparedMean)
                preparedMean.ValidateDevice(device);
            if (hasIncoming)
                incoming.ValidateDevice(device);
            if (hasWave)
                waveJump.ValidateDevice(device);
            if (hasPressure)
                pressureMinimum.ValidateDevice(device);
        }
    };
    using RHSFields = RHSFieldsT<>;
    struct RHSFaceOutputs
    {
        RHSOutput high, low;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> wallForce;
        RHSStatus status;
        Runtime::ArrayView<index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> fallbackCounts;
    };
    /// Launch one independent thread per selected auxiliary face, integrating
    /// quadrature in CPU order. Caller uses mesh-classified face subsets and
    /// declares ghosts only for dependent work. No MPI or atomics occur here.
    void EvaluateFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                           Runtime::ArrayView<const RHSKernelParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> parameters,
                           RHSKernelControls controls, RHSFields fields, RHSFaceOutputs output, RHSIndices faces, bool requireGhost = true);
    /// Gather face contributions in canonical cell order. Opposite periodic
    /// faces rotate momentum only, including self-periodic neighbors. Outputs
    /// are integrated residuals (no division by volume). Failed face status
    /// propagates to a cell status before the cell output is published.
    void GatherFluxCells(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                         RHSInput highFaces, RHSInput lowFaces,
                         Runtime::ArrayView<const NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faceStatus,
                         RHSOutput high, RHSOutput low, RHSStatus status, RHSIndices cells, bool produceLow);
}
