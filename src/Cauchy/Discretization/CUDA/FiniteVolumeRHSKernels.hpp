#pragma once
#include "Cauchy/Discretization/EquationPhysicsProfile.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSFacePackages.hpp"

#include "Cauchy/Discretization/PerfectGasKernelPoint.hpp"
#include "Cauchy/Discretization/ACMKernelPoint.hpp"
#include "Cauchy/Boundary/ACMBoundary.hpp"
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
        FiniteVolumeStencilView stencil;
        Runtime::ArrayView<const index, false, Runtime::FixedRows<1>> faceOffsets;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<7>> quadrature;
        CFV::ReconstructionPointBasisView basis;
        Runtime::ArrayView<const PerfectGasBoundaryParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> conditions;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> wallMean;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::CSRRows> wallFaces;
        bool hasBasis = false, hasWall = false;
        Runtime::ArrayView<const ACM::SpatialParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> acm;
        Runtime::ArrayView<const ACM::BoundaryParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> acmBC;
        void ValidateDevice(Runtime::Device device) const;
        DNDS_DEVICE_CALLABLE FiniteVolumeDetail::SampleRecord FaceSample(index iAux, index iPoint) const
        {
            return stencil.RHSFaceSample(stencil.stencilFaces(iAux, 0), 0, iPoint);
        }
        DNDS_DEVICE_CALLABLE FiniteVolumeDetail::SampleRecord FaceAnchor(index iAux) const
        {
            const index face = stencil.stencilFaces(iAux, 0);
            return FaceSample(iAux, faceOffsets(face, 0));
        }
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
        RHSFaceMapping mapping = RHSFaceMapping::CTASegments;
        RHSFaceReduction reduction = RHSFaceReduction::Serial;
        int gridBlocks = 0;
        int blockSize = 128;
        bool doubleBuffer = false, hoistFaceRows = false;
        bool mechanical = false, captureBoundaryFlux = false;
        bool ransMeanFaceValues = false, ransP1Overlay = false;
        void Validate() const;
    };
    static_assert(std::is_trivially_copyable_v<RHSKernelControls>);

    /// Changing operands; geometry/physics publications are separate. Only
    /// combined uniform-row fields are supported. viscousData denotes primitive
    /// gradients when primitiveGradient is selected, otherwise optional
    /// conservative gradients/Taylor coefficients with the reconstruction shape.
    template <class MeanPolicy = Runtime::UniformTableRows, class CoefficientPolicy = Runtime::UniformTableRows,
              class ViscousPolicy = CoefficientPolicy, int Modes = 0>
    struct RHSFieldsT
    {
        static constexpr int static_modes = Modes;
        static constexpr int static_dimension = 0;
        static constexpr bool separate_roles = false;
        using IncomingPolicy = std::conditional_t<MeanPolicy::row_width == Runtime::DynamicExtent, Runtime::TableRows,
                                                  Runtime::FixedWidthRows<Runtime::TableRows, MeanPolicy::row_width>>;
        Runtime::PairReadView<real, Runtime::AccessBoundsChecks, MeanPolicy> mean;
        Runtime::PairReadView<real, Runtime::AccessBoundsChecks, CoefficientPolicy> reconstruction;
        Runtime::PairReadView<real, Runtime::AccessBoundsChecks, ViscousPolicy> viscousData;
        RHSField waveJump, pressureMinimum, ransP1Gradients;
        RHSInput preparedMean;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, IncomingPolicy> incoming;
        bool hasPreparedMean = false, hasIncoming = false, hasWave = false, hasPressure = false;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, mean, reconstruction, viscousData, ransP1Gradients);
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
    template <class Policy = Runtime::TableRows>
    struct RHSFaceOutputsT
    {
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Policy> high, low;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> wallForce;
        RHSStatus status;
        Runtime::ArrayView<index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> fallbackCounts;
    };
    using RHSFaceOutputs = RHSFaceOutputsT<>;
    /** @brief Launch the selected mechanical or perfect-gas face mapping with validated views.
     * @details ACM flow4/SA/2EQ uses FP64 thread-per-face integration and the
     * immutable mechanical parameters in geometry. The compressible path
     * supports calorically perfect-gas flow5/SA/2EQ. This is
     * not a species, chemistry or arbitrary-EOS entry: validate the immutable
     * RHSKernelParameters before publishing them to the device. Combined AoS
     * fields use fixed-width 5/6/7 row views and static P1/P2/P3 modal extents;
     * the existing uniform-layout path retains runtime access. No field data
     * is copied to obtain a static view. Caller uses mesh-classified face
     * subsets, declares ghosts only for dependent work and retains all owners
     * through stream completion. No MPI or floating-point atomic sums occur.
     */
    void EvaluateFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                           Runtime::ArrayView<const RHSKernelParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> parameters,
                           RHSKernelControls controls, RHSFields fields, RHSFaceOutputs output, RHSIndices faces, bool requireGhost = true, RHSFacePackageView packages = {});
    /// Gather face contributions in canonical cell order. Opposite periodic
    /// faces rotate momentum only, including self-periodic neighbors. Outputs
    /// are integrated residuals (no division by volume). Failed face status
    /// propagates to a cell status before the cell output is published.
    void GatherFluxCells(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                         RHSInput highFaces, RHSInput lowFaces,
                         Runtime::ArrayView<const NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faceStatus,
                         RHSOutput high, RHSOutput low, RHSStatus status, RHSIndices cells, bool produceLow);
}
