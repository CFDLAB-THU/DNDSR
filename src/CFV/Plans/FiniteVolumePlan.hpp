#pragma once

#include "DNDS/Algebra/DofField.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "Geom/Access/EntitySpace.hpp"
#include "Geom/Access/EntityWorkPartition.hpp"
#include "Geom/Access/MeshView.hpp"
#include "Geom/Geometry/Geometric.hpp"
#include "Geom/Unstructured/UnstructuredMeshFwd.hpp"
#include "DNDS/Math/HardEigen.hpp"
#include "CFV/Math/ReconstructionFunctional.hpp"
#include "CFV/Math/TaylorBasis.hpp"
#include "CFV/Plans/ReconstructionCapabilities.hpp"
#include <functional>
#include <map>
#include <optional>

namespace DNDS::Geom
{
    class MeshReadAccess;
    class UnstructuredMeshSnapshot;
}
namespace DNDS::Algebra
{
    template <class T, class Epoch>
    class DofHalo;
}

namespace DNDS::CFV
{
    enum class ReconstructionRegion
    {
        All,
        Core,
        HaloDependent
    };
    /// CPU preparation settings. Degree is polynomial degree (degree 1 is O2).
    /// Initial functional is the isotropic legacy VFV family; unsupported
    /// anisotropic/GG penalties are not silently translated to this method.
    struct ReconstructionSettings
    {
        int maxOrder = 2, intOrder = 5;
        real jacobiRelax = 1., WBAP_nStd = 10.;
        bool SORInstead = false, HQMWeights = false;
        bool coreFirst = false;
        bool cacheDiffBase = true;
        int cacheDiffBaseOrder = 1;
        ReconstructionFunctionalSettings functional;
        real bcWeight = 1., svdTolerance = 1e-13, svdAbsoluteTolerance = 0;
        std::array<real, 4> boundaryDerivativeWeights{1, 1, 1, 1};
        /// Optional zone-specific replacement for the unsquared per-order
        /// boundary weights. Higher orders impose homogeneous derivative data.
        std::map<int, std::array<real, 4>> boundaryZoneWeights;
        DNDS_DECLARE_CONFIG(ReconstructionSettings)
        {
            DNDS_FIELD(maxOrder, "Polynomial degree, 1..3");
            DNDS_FIELD(intOrder, "Supported integration degree, independent of polynomial degree; lower rules deliberately change the discrete functional");
            DNDS_FIELD(jacobiRelax, "Jacobi/SOR relaxation");
            DNDS_FIELD(SORInstead, "Use ordered two-buffer SOR with completed local predecessors from the output");
            DNDS_FIELD(coreFirst, "Use conservative node-star core before dependent cells for ordered SOR");
            DNDS_FIELD(HQMWeights, "Legacy HQM_OPT derivative and HQM_SD geometry weights");
            config.field_section(&T::functional, "functional", "Independent isotropic functional weights and length policy");
            DNDS_FIELD(cacheDiffBase, "Retain basis values and derivatives on face quadrature points");
            DNDS_FIELD(cacheDiffBaseOrder, "Highest cached derivative order, 0..3; default value and first derivatives; capped by polynomial degree");
            DNDS_FIELD(bcWeight, "Dirichlet boundary value weight");
            DNDS_FIELD(boundaryDerivativeWeights, "Boundary weights by derivative order; nonzero derivatives impose homogeneous derivative targets");
            // Integer-key maps use nlohmann's pair-array wire representation,
            // not an object with string keys. Declare that shape explicitly.
            config.field_schema(&T::boundaryZoneWeights, "boundaryZoneWeights", "Optional [zone, weights] pairs",
                                []() -> nlohmann::ordered_json
                                {
                                    using J = nlohmann::ordered_json;
                                    const J weights{{"type", "array"}, {"minItems", 4}, {"maxItems", 4}, {"items", {{"type", "number"}}}};
                                    const J pair{{"type", "array"}, {"minItems", 2}, {"maxItems", 2}, {"items", J::array({J{{"type", "integer"}}, weights})}};
                                    return {{"type", "array"}, {"items", pair}};
                                });
            DNDS_FIELD(svdTolerance, "Relative small-mode SVD cutoff against sigma_max");
            DNDS_FIELD(svdAbsoluteTolerance, "Absolute SVD cutoff, zero by default");
            DNDS_FIELD(WBAP_nStd, "WBAP center weight");
        }
        void Validate() const;
    };

    namespace Detail
    {
        struct FiniteVolumeData;
        struct ReconstructionIdentity;
        struct ReconstructionData;
        struct ReconstructionPlacementData;
        struct ReconstructionDeviceView;
    }
    using FVField = Algebra::DofField<real>;

    /// MPI-global owned-cell setup diagnostics; no ghost double counting.
    struct ReconstructionInverseDiagnostics
    {
        index cells = 0, filteredCells = 0, discardedModes = 0, rankZeroCells = 0;
        real maxCondition = 0;
        index maxConditionCell = UnInitIndex;
        std::array<index, 20> rankCounts{}; // Degree-three 3D has at most 19 modes.
    };

    /// Plain metric spans for numerical loops. FiniteVolumeRead supplies host
    /// pointers; ReconstructionDeviceRead supplies CUDA pointers. Retain the
    /// corresponding scope through every use and execute on its declared device.
    /// The historical HostView type name is retained for source compatibility.
    struct FiniteVolumeHostView
    {
        Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells;
        Runtime::ArrayView<const index, false, Runtime::FixedRows<1>> faceOffsets;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<7>> quadrature;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<3>> radialMoment;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<7>> faceMean;
        /// Host launcher validation, once before the numerical loop/launch.
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, cells, faceOffsets, quadrature, radialMoment, faceMean);
        }
        DNDS_DEVICE_CALLABLE real GetCellVol(index iCell) const { return cells(iCell, 0); }
        DNDS_DEVICE_CALLABLE Geom::tPoint GetCellCent(index iCell) const
        {
            return {cells(iCell, 1), cells(iCell, 2), cells(iCell, 3)};
        }
        DNDS_DEVICE_CALLABLE Geom::tPoint GetCellBary(index iCell) const
        {
            return {cells(iCell, 4), cells(iCell, 5), cells(iCell, 6)};
        }
        DNDS_DEVICE_CALLABLE index FaceQuadratureBegin(index iFace) const { return faceOffsets(iFace, 0); }
        DNDS_DEVICE_CALLABLE index FaceQuadratureEnd(index iFace) const { return faceOffsets(iFace + 1, 0); }
        DNDS_DEVICE_CALLABLE Geom::tPoint GetFacePoint(index iPoint) const
        {
            return {quadrature(iPoint, 0), quadrature(iPoint, 1), quadrature(iPoint, 2)};
        }
        DNDS_DEVICE_CALLABLE Geom::tPoint GetFaceNorm(index iPoint) const
        {
            return {quadrature(iPoint, 3), quadrature(iPoint, 4), quadrature(iPoint, 5)};
        }
        DNDS_DEVICE_CALLABLE real GetFaceWeight(index iPoint) const { return quadrature(iPoint, 6); }
        /// Reference-center position, normalized integral(n dS), scalar area.
        /// Geometry is integrated at setup; approximate face operators use one
        /// record regardless of reconstruction/RHS integration degree.
        DNDS_DEVICE_CALLABLE Geom::tPoint GetFaceCent(index iFace) const { return {faceMean(iFace, 0), faceMean(iFace, 1), faceMean(iFace, 2)}; }
        DNDS_DEVICE_CALLABLE Geom::tPoint GetFaceMeanNorm(index iFace) const { return {faceMean(iFace, 3), faceMean(iFace, 4), faceMean(iFace, 5)}; }
        DNDS_DEVICE_CALLABLE real GetFaceArea(index iFace) const { return faceMean(iFace, 6); }
    };
    /// Read scope pins metadata, values and the mesh generation. Construction
    /// validates once; HostView() copies only borrowed pointers/scalars.
    class FiniteVolumeRead
    {
        std::shared_ptr<const Detail::FiniteVolumeData> _data;
        std::shared_ptr<Geom::MeshReadAccess> _meshRead;
        Runtime::Array<real>::ReadLease _cells, _quadrature, _radialMoment, _faceMean;
        Runtime::Array<index>::ReadLease _offsets;
        explicit FiniteVolumeRead(std::shared_ptr<const Detail::FiniteVolumeData> data, Runtime::ExecutionContext context);
        friend class FiniteVolumePlan;

    public:
        /// Borrowed numeric view in this scope's declared address space. The
        /// retained scope owns metric leases and protects the mesh generation.
        FiniteVolumeHostView View() const;
        FiniteVolumeHostView HostView() const;
    };

    /// Setup-only metric options; frozen into the resulting publication.
    /// Face order zero follows integrationOrder, one uses reference centers and
    /// integrated mean normals/areas. Axisymmetry weights by radius=y (no 2*pi).
    /// collectivePreflight is the caller-resolved optional validation policy;
    /// it never removes routing/numerical protocols or local access checks.
    struct FiniteVolumeMetricOptions
    {
        int integrationOrder = 5;
        int faceIntegrationOrder = 0;
        bool axisymmetric = false;
        bool collectivePreflight = true;
    };

    /// Immutable metric publication; copies share Runtime buffers and mesh/order
    /// identity. Mesh edits invalidate it. No solution or time-loop ownership.
    class FiniteVolumePlan
    {
        std::shared_ptr<const Detail::FiniteVolumeData> _data;
        explicit FiniteVolumePlan(std::shared_ptr<const Detail::FiniteVolumeData> data) : _data(std::move(data)) {}
        friend class VariationalReconstructionPlan;
        friend class LimitedVRPlan;
        friend class ReconstructionBasisRead;
        friend class PolynomialSamplingPlan;

    public:
        /// axisymmetric uses radius=y>=0 and measures r*dA/r*ds, without 2*pi.
        /// Equation source terms and cylindrical component connections remain
        /// with the equation adapter; this option changes integration only.
        /// Named options avoid confusing face quadrature with validation policy.
        static FiniteVolumePlan ConstructMetrics(ssp<Geom::UnstructuredMesh> mesh, FiniteVolumeMetricOptions options);
        /// Compatibility for the unambiguous historical three-argument API.
        static FiniteVolumePlan ConstructMetrics(ssp<Geom::UnstructuredMesh> mesh, int intOrder = 5, bool axisymmetric = false)
        {
            FiniteVolumeMetricOptions options;
            options.integrationOrder = intOrder;
            options.axisymmetric = axisymmetric;
            return ConstructMetrics(std::move(mesh), options);
        }
        void Validate() const;
        FiniteVolumeRead AcquireRead(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const;
        /// Generation-only guard for queued operators whose metric arrays are
        /// declared as scheduled operands. Does not acquire metric payloads.
        Geom::MeshReadAccess AcquireMeshRead(bool wallDistance = false) const;
        /// Immutable metric operands for queued execution. Declare the required
        /// arrays with Runtime::ReadAs; keep a mesh-generation guard through
        /// operation completion. A returned handle is not an access grant.
        const Runtime::Array<real> &CellMetrics() const;
        const Runtime::Array<real> &FaceQuadrature() const;
        const Runtime::Array<real> &FaceMetrics() const;
        const Runtime::Array<real> &RadialMoments() const;
        const Runtime::Array<index> &FaceOffsets() const;
        /// Explicit setup copy of immutable metrics, independent of VR buffers.
        /// Row-space/halo/mesh identities are shared, but payload arrays belong
        /// only to the returned placement. Release the original host plan when
        /// it is no longer needed; the placed plan does not retain those arrays.
        /// Copy completion precedes return. Serialize this setup operation with
        /// lazy work-partition preparation and mesh mutation.
        FiniteVolumePlan Placed(Runtime::Device device, Runtime::ExecutionContext context) const;
        Runtime::Device Device() const;
        index NumCell() const;
        index NumCellProc() const;
        index NumFaceProc() const;
        int Dimension() const;
        bool Axisymmetric() const;
        const Geom::EntitySpace &Cells() const;
        const Algebra::DofSpace &Space() const;
        /// Preparation/transfer policy shared by this immutable plan's halos.
        /// Does not change mesh numbering, numerical collectives or access checks.
        std::shared_ptr<Runtime::CommunicationContext> Communication() const;
        /// Lazily cached conservative cell-node-cell split, shared by copies
        /// and gradient/reconstruction consumers. Purely rank-local; no halo
        /// fetch or MPI is posted. Complete published node-star adjacency is
        /// required. The returned shallow owner survives this plan handle.
        Geom::EntityWorkPartition CellWorkPartition() const;
        FVField BuildField(index width) const;
        /// Uniform physical prototype on this plan's device; derive owned/ghost
        /// pitches independently. Like the width overload, placement follows
        /// the plan, including when allocating coarse-level solver fields.
        FVField BuildField(Runtime::RowLayout prototype) const;
        /// Explicit placement override for setup/migration staging only.
        FVField BuildField(Runtime::RowLayout prototype, Runtime::Device device) const;
        /// Collective binding, retained by caller for persistent repeated pulls.
        Runtime::TransferBinding BindHalo(FVField &field, Runtime::TransferOptions options = {}) const;
        /// CPU direct transfer with completed-publication provenance. Callers
        /// provide coherent exchange epochs; no remote freshness is inferred.
        Algebra::DofHalo<real, uint64_t> BindTrackedHalo(FVField &field, Runtime::TransferOptions options = {}) const;
        void ValidateField(const FVField &field, index width) const;
        MPI_Comm Communicator() const;
        real GetCellVol(index iCell) const;
        Geom::tPoint GetCellCent(index iCell) const;
        /// Explicit CPU setup product: one maximum principal-axis cell extent
        /// for every owned/ghost metric row. Shares GetCellMaxLenScale geometry
        /// and its degeneracy-aware principal frame (relative gap 1e-8). No MPI or cache
        /// mutation; the caller owns the returned buffer and must use it with
        /// this validated metric/mesh generation. Build again after mesh edits.
        Runtime::Array<real> BuildCellMaxLenScale() const;
    };

    /// Explicit targets for boundary reconstruction equations at prepared
    /// quadrature points. The caller selects their physical convention (e.g.
    /// Euler's reflected exterior target at a wall, or a prescribed trace).
    /// Refresh after changing a state-dependent BC. No callback is invoked by
    /// reconstruction. Each object belongs to exactly one reconstruction plan.
    struct ReconstructionBoundaryValues
    {
        Runtime::Array<real> values;
        std::shared_ptr<const Detail::ReconstructionIdentity> plan;
    };
    struct BoundaryQuadraturePoint
    {
        index cell, face;
        int quadrature;
        Geom::tPoint point, normal;
    };
    /// Matrices act on column vectors in [first,first+count); all other fields
    /// are limited componentwise. count==0 is an explicit disabled descriptor:
    /// no matrix arrays are allocated or validated. Fill nonempty
    /// forward/inverse from CELL MEANS once per
    /// field stage. Geometry periodic transforms and characteristic transforms
    /// are different operations. Declare vector triples in BuildWorkspace so
    /// the periodic component map precedes characteristic limiting.
    /// Frozen transforms are finalized on CPU and guarded by completed-write
    /// versions. Queued transforms are produced on CPU or device. Caller orders
    /// producer -> limiter on one stream (or supplies
    /// cross-stream events). Completed host versions are not readiness proofs
    /// for queued writers. Optional numerical diagnostics are separate from
    /// this ordering contract and are off by default.
    enum class CharacteristicValidation
    {
        Frozen,
        Queued
    };
    struct FaceCharacteristicTransform
    {
        index first = 0, count = 0;
        Runtime::Array<real> forward, inverse;
        std::shared_ptr<const Detail::ReconstructionIdentity> plan;
        uint64_t forwardVersion = 0, inverseVersion = 0;
        uint64_t forwardIdentity = 0, inverseIdentity = 0;
        CharacteristicValidation validation = CharacteristicValidation::Frozen;
    };
    using FaceCharacteristicTransformRef = std::optional<std::reference_wrapper<const FaceCharacteristicTransform>>;
    using LimiterIndicatorRef = std::optional<std::reference_wrapper<const Runtime::Array<real>>>;

    enum class ReconstructionMatrixPrecision
    {
        FP64,
        FP32
    };
    struct ReconstructionResidencyOptions
    {
        bool mesh = false, metrics = false, basis = false;
        /// Keep the CPU preparation publication reachable from the placed
        /// handle. Disable only when all later setup/debug queries use the host
        /// plan separately; device operators retain their own immutable arrays,
        /// row space, halo route and scalar metadata.
        bool retainHostPayloads = true;
        /// Retain cell moments for later arbitrary-point sampling (e.g. moving
        /// interfaces). This is independent of face-quadrature basis caching.
        /// Disabled placements keep no otherwise-unused moment allocation.
        bool arbitraryPointSampling = false;
        /// Highest derivative needed by placed point evaluation, 0..3.
        /// Uncached requests retain moments/cell geometry; cached-only requests
        /// do not. Ignored when basis=false. Full metric publication is an
        /// independent request and must be enabled explicitly with metrics.
        int basisDerivativeOrder = 1;
        /// Quantize only the already-inverted sweep maps during placement.
        /// Means, coefficients, relaxation and WBAP maps stay in FP64 storage.
        /// Sweep arithmetic is selected independently by the device kernel.
        ReconstructionMatrixPrecision matrixPrecision = ReconstructionMatrixPrecision::FP64;
    };
    /// Explicit CUDA numerical kernel selection. Backend resolution occurs
    /// once per submitted CFV operation. Full accumulation policies require
    /// resident FP32 sweep maps and fail rather than falling back on the CPU.
    enum class ReconstructionDeviceKernel
    {
        FP64,
        FP64TensorCore,
        FP64TensorCoreShared,
        FP64TensorCoreAsync,
        /// FP32 matrix/coefficient products accumulated in FP32, then added to
        /// FP64 coefficients. SIMT FMA reference for quantized MMA experiments;
        /// this selection does not claim to use tensor cores or TF32 arithmetic.
        FP32,
        /// Experimental 16x16x8 TF32 tensor products on FP32 sweep maps.
        /// Explicitly rounds operands to TF32, accumulates in FP32, promotes
        /// the correction to FP64. This is not the FP32 reference arithmetic.
        TF32TensorCore,
        /// Experimental FP64 SIMT sweep: share one neighbor coefficient row
        /// across target modes without changing their ordered arithmetic. No
        /// tensor-core requirement; combined 5/6/7-component P1..P3 only.
        FP64Shared,
        /// FP32 sweep maps and the entire local update in FP32, including
        /// relaxation, periodic rotation, means and boundary contributions.
        /// Only the final result is promoted to durable FP64 storage.
        FP32FullAccumulation,
        /// Same FP32 maps, with the entire local update in float-float.
        /// FP64 inputs are converted directly to the high/low pair; this does
        /// not restore precision lost in the maps. Requires CCCL compute.
        FP32FloatFloatAccumulation
    };
    struct ResidentBufferInfo
    {
        std::string name;
        size_t bytes = 0;
        Runtime::Device device = Runtime::Device::Host();
    };

    /// Own iteration scratch once; kernels never grow per-cell vectors.
    struct ReconstructionWorkspace
    {
        /// Scratch output for one reconstruction/limiter pass. Durable
        /// coefficients are caller-owned orchestration state and are supplied
        /// explicitly to every operation.
        FVField temporary;
        Runtime::Array<real> scratch;
        index components;
        std::shared_ptr<const Detail::ReconstructionData> plan;
        /// Non-overlapping triples of polar-vector components. Other components
        /// are scalars. Periodic rotations act on these triples, independently
        /// of the geometric dimension (three-component flow on 2D geometry).
        std::vector<index> periodicVectorStarts;
        /// CUDA-only execution resources. They remain empty for host plans;
        /// Cauchy and other callers still own one public workspace type.
        Runtime::Array<index> devicePeriodicVectorStarts;
        Runtime::Array<int> deviceStatus;
        index deviceBlocks = 0;
        uint64_t devicePeriodicIdentity = 0, devicePeriodicVersion = 0;
        std::shared_ptr<const Detail::ReconstructionPlacementData> devicePlan;
    };

    /// Borrowed cache pointers for owned-cell face stencils. All derivatives
    /// are in the selected parent's reference frame. Retain the read scope.
    struct ReconstructionBasisHostView
    {
        Runtime::ArrayView<const real, false, Runtime::TableRows> face;
        Runtime::ArrayView<const index, false, Runtime::FixedRows<1>> faceOffsets;
        int modes, derivatives;
        /// Column-major derivatives-by-modes view, borrowing this scope's cache.
        /// Rows retain their component stride; no Eigen or host-only owner is
        /// carried into a kernel. The parent stencil validates iPoint and side.
        DNDS_DEVICE_CALLABLE auto FaceMatrix(index iPoint, int side) const
        {
            return Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::ColMajorTable>{
                face[2 * iPoint + side].data(), nullptr, derivatives, modes, derivatives, nullptr, Runtime::RowLayoutKind::ColMajorTable, face.device};
        }
        /// Column-major derivatives-by-modes matrix; derivative includes order0.
        DNDS_DEVICE_CALLABLE real FaceDerivative(index iPoint, int side, int derivative, int mode) const
        {
            return FaceMatrix(iPoint, side)(derivative, mode);
        }
    };
    /// Frozen operands for cached or on-demand Taylor evaluation. The enclosing
    /// ReconstructionBasisRead retains mesh, metric, moment and cache leases.
    struct ReconstructionBasisEvaluationView
    {
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::TableRows>
            cached, moments;
        FiniteVolumeHostView geometry;
        Geom::UnstructuredMeshView mesh;
        int dimension = 0, degree = 0, modes = 0, cachedDerivatives = 0;
    };
    /// Compact resident point basis for a prepared face incidence. The caller
    /// supplies its physical point in the selected parent cell's frame, so no
    /// mesh owner, periodic lookup or host geometry service enters evaluation.
    /// Cached matrices already use that same parent frame. mode omits the
    /// constant term; derivative uses the canonical Taylor ordinal.
    struct ReconstructionPointBasisView
    {
        Runtime::ArrayView<const real, false, Runtime::TableRows> cached, moments;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells;
        Math::TaylorBasisParameters polynomial;
        int cachedDerivatives = 0, availableDerivatives = 0;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, cached, moments, cells);
        }
        /// Prepare only when requested derivatives exceed cachedDerivatives.
        /// Fully cached placements need no cell-geometry allocation at all.
        template <class Scalar = real>
        DNDS_DEVICE_CALLABLE Math::TaylorPointEvaluationT<Scalar> PreparePoint(
            index iCell, const std::array<real, 3> &point) const noexcept
        {
            return Math::TaylorPointEvaluationT<Scalar>::FromPoint(
                point, {cells(iCell, 1), cells(iCell, 2), cells(iCell, 3)}, cells(iCell, 7));
        }
        /// Reuse this point context across all modes and field components;
        /// cached derivative rows still take precedence over direct evaluation.
        template <class Scalar>
        DNDS_DEVICE_CALLABLE Scalar Evaluate(index iCell, index iPoint, int side, const Math::TaylorPointEvaluationT<Scalar> &point,
                                             int derivative, int mode) const noexcept
        {
            DNDS_HD_assert(derivative >= 0 && derivative < availableDerivatives && mode >= 0 && mode < polynomial.size - 1);
            if (derivative < cachedDerivatives)
                return DNDS::Math::ScalarCast<Scalar>(cached(2 * iPoint + side, mode * cachedDerivatives + derivative));
            const Scalar value = polynomial.Derivative(derivative, mode + 1, point);
            return derivative == 0 ? value - DNDS::Math::ScalarCast<Scalar>(moments(iCell, mode + 1)) : value;
        }
        DNDS_DEVICE_CALLABLE real Evaluate(index iCell, index iPoint, int side, const std::array<real, 3> &point,
                                           int derivative, int mode) const noexcept
        {
            DNDS_HD_assert(derivative >= 0 && derivative < availableDerivatives && mode >= 0 && mode < polynomial.size - 1);
            if (derivative < cachedDerivatives)
                return cached(2 * iPoint + side, mode * cachedDerivatives + derivative);
            return Evaluate(iCell, iPoint, side, PreparePoint(iCell, point), derivative, mode);
        }
    };
    /// Pins the geometry generation and Runtime cache allocations once outside
    /// a hot loop. Face queries must be in an owned-cell stencil and select a
    /// real parent. Generic/debug access can use checked GetCachedFaceBasis.
    class ReconstructionBasisRead
    {
        std::shared_ptr<const Detail::ReconstructionData> _data;
        FiniteVolumeRead _geometry;
        std::optional<Runtime::Array<real>::ReadLease> _basis;
        Runtime::Array<real>::ReadLease _moments;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> _basisView, _momentView;
        std::shared_ptr<Geom::MeshReadAccess> _meshRead;
        ReconstructionBasisEvaluationView _evaluation;
        explicit ReconstructionBasisRead(std::shared_ptr<const Detail::ReconstructionData> data);
        friend class VariationalReconstructionPlan;

    public:
        ReconstructionBasisHostView HostView() const;
        /// Frozen cached/on-demand operands; retains no owner or lease itself.
        const ReconstructionBasisEvaluationView &EvaluationView() const;
        /// Global quadrature ordinal, real parent in an owned-cell stencil.
        /// Uses cached rows when available; otherwise evaluates into scratch.
        Math::ReconstructionBasisView FaceBasis(index face, int side, index point,
                                                int derivativeOrder, Math::SmallMatrix &scratch) const;
    };

    /// Retained view of a placed reconstruction publication. This is a scoped
    /// low-level view, not a second plan/workspace hierarchy. Its plain views
    /// may be passed to backend kernels only while this owner remains alive.
    class ReconstructionResidentRead
    {
        std::shared_ptr<const Detail::ReconstructionPlacementData> _data;
        explicit ReconstructionResidentRead(std::shared_ptr<const Detail::ReconstructionPlacementData> data)
            : _data(std::move(data)) {}
        friend class VariationalReconstructionPlan;

    public:
        Detail::ReconstructionDeviceView DeviceView() const;
        FiniteVolumeHostView MetricView() const;
        ReconstructionBasisHostView BasisView() const;
        /// Cached or on-demand point evaluation up to the placed derivative
        /// order. Unlike BasisView(), this also supports cacheDiffBase=false.
        ReconstructionPointBasisView PointBasisView() const;
    };

    /// Composition of metrics, basis/moments and factored VR systems on Runtime
    /// storage. CPU kernels consume explicit arrays; no legacy VR owner exists.
    class VariationalReconstructionPlan
    {
        friend class PolynomialSamplingPlan;
        std::shared_ptr<const Detail::ReconstructionData> _data;
        std::shared_ptr<const Detail::ReconstructionPlacementData> _device;
        std::shared_ptr<const Detail::ReconstructionIdentity> _identity;
        explicit VariationalReconstructionPlan(std::shared_ptr<const Detail::ReconstructionData> data,
                                               std::shared_ptr<const Detail::ReconstructionPlacementData> device = {});
        void DoReconstructionIterImpl(const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                      const ReconstructionBoundaryValues &boundary, ReconstructionRegion region,
                                      const FVField *target, const Runtime::Array<real> *alpha) const;
        ReconstructionWorkspace BuildDeviceWorkspace(index components, Runtime::RowLayout coefficientPrototype,
                                                     std::vector<index> periodicVectorStarts) const;
        void SubmitDeviceReconstructionIterTo(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                              const FVField &, const FVField &, ReconstructionWorkspace &,
                                              const ReconstructionBoundaryValues &, ReconstructionRegion,
                                              ReconstructionDeviceKernel) const;
        void SubmitDeviceReconstructionIterLimitedTo(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                                     const FVField &, const FVField &, ReconstructionWorkspace &,
                                                     const ReconstructionBoundaryValues &, const FVField &,
                                                     const Runtime::Array<real> &, ReconstructionRegion) const;
        void SubmitDeviceCommitReconstruction(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                              FVField &, ReconstructionWorkspace &) const;
        void SubmitDeviceLimiterWBAP_3OrderTo(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                              const FVField &, ReconstructionWorkspace &, int, FaceCharacteristicTransformRef,
                                              ReconstructionRegion, const Runtime::Array<real> *, real) const;
        const ReconstructionSettings &PlacementSettings() const;
        ReconstructionCapabilities PlacementCapabilities() const;
        ReconstructionStorageReport PlacementStorageReport() const;
        ReconstructionInverseDiagnostics PlacementInverseDiagnostics() const;
        int PlacementNRecDOF() const;
        index PlacementOwnedRows() const;
        index PlacementBoundaryPoints() const;
        FVField BuildDeviceField(Runtime::RowLayout) const;
        void ValidatePlacement() const;

    public:
        /// Collective CPU setup. The immutable capabilities must agree on all
        /// ranks. Default support preserves the standalone API; reduced profiles
        /// omit unused tables and fail explicitly on unprepared operations.
        static VariationalReconstructionPlan ConstructRecCoeff(FiniteVolumePlan metrics, ReconstructionSettings settings = {},
                                                               ReconstructionCapabilities capabilities = {});
        /// Return the same public CFV plan placed on one backend. CUDA setup is
        /// synchronous; hot submissions are stream ordered. Set
        /// retainHostPayloads=false after preparing host geometry queries;
        /// the returned frozen placement can allocate device boundary/workspace
        /// operands directly, but rejects host-only geometry setup queries.
        /// No device-specific plan or workspace type is exposed to callers.
        VariationalReconstructionPlan Place(Runtime::Device, Runtime::ExecutionContext,
                                            ReconstructionResidencyOptions = {}) const;
        Runtime::Device Device() const;
        bool RetainsHostPayloads() const { return bool(_data); }
        /// Immutable shared face sampling allocation on this plan's backend.
        /// Rows are 2*globalQuadratureOrdinal+side, columns are mode-major
        /// derivative values. Empty when caching is disabled. A device plan
        /// with caching enabled must have requested basis residency at placement.
        /// Copy the handle to retain the allocation beyond this plan's lifetime.
        const Runtime::Array<real> &SamplingFaceBasis() const;
        /// Immutable normalization moments and cell geometry used by on-site
        /// evaluation. On a placed plan these are available when direct basis
        /// evaluation or arbitraryPointSampling was requested. Copies retain
        /// the same allocations used by RHS point evaluation, without matrices.
        const Runtime::Array<real> &SamplingMoments() const;
        const Runtime::Array<real> &SamplingCellGeometry() const;
        /// Stable field row identity, also available after CPU payload release.
        const Algebra::DofSpace &Space() const;
        bool SupportsKernel(ReconstructionDeviceKernel) const;
        std::vector<ResidentBufferInfo> ResidentBuffers() const;
        ReconstructionResidentRead AcquireResidentRead() const;
        const Geom::UnstructuredMeshSnapshot &MeshSnapshot() const;
        const FiniteVolumePlan &Metrics() const;
        const ReconstructionSettings &Settings() const;
        ReconstructionCapabilities Capabilities() const;
        /// Validate requested support at operator entry, before output mutation
        /// or communication. This checks capabilities only; operators still
        /// pin geometry and validate its generation before using prepared data.
        void Require(ReconstructionOperation operation) const;
        ReconstructionStorageReport StorageReport() const;
        int NRecDOF() const;
        ReconstructionInverseDiagnostics GetInverseDiagnostics() const;
        /// Owned-cell spectrum diagnostics from setup, without refactorization.
        HardEigen::SVDInverseDiagnostics GetCellInverseDiagnostics(index iCell) const;
        void Validate() const;
        ReconstructionWorkspace BuildWorkspace(index components, std::vector<index> periodicVectorStarts = {}) const;
        ReconstructionWorkspace BuildWorkspace(index components, Runtime::RowLayout coefficientPrototype,
                                               std::vector<index> periodicVectorStarts = {}) const;
        /// Allocate durable coefficient storage independently of scratch.
        /// Solver orchestration owns the returned field and may reuse a
        /// compatible scratch workspace across serialized states.
        FVField BuildCoefficients(index components) const;
        FVField BuildCoefficients(index components, Runtime::RowLayout coefficientPrototype) const;
        ReconstructionBoundaryValues BuildBoundaryValues(index components) const;
        ReconstructionBoundaryValues PlaceBoundaryValues(const ReconstructionBoundaryValues &) const;
        index NumBoundaryQuadrature() const;
        BoundaryQuadraturePoint GetBoundaryQuadrature(index iPoint) const;
        FaceCharacteristicTransform BuildFaceTransform(index first, index count) const;
        /// Backend-owned mutable transform buffers. CPU seeds unused faces
        /// with identity; device contents are unspecified until produced.
        /// A queued producer must supply finite inverse matrices
        /// before first use. All WBAP orders may reuse them until mutation.
        /// count=0 requires no producer or matrix allocation. Numerical inverse
        /// checking is an explicit diagnostic, never an automatic launch.
        FaceCharacteristicTransform BuildQueuedFaceTransform(index first, index count) const;
        /// Opt-in numerical diagnostic, normally queued once after a producer.
        /// Not needed for stream ordering and not called by limiter operators.
        /// Failure sets workspace status; observe CheckStatus after draining.
        /// Caller supplies valid matrices even when diagnostics are disabled.
        void SubmitValidateFaceTransform(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                         const FaceCharacteristicTransform &, ReconstructionWorkspace &) const;
        FaceCharacteristicTransform PlaceFaceTransform(const FaceCharacteristicTransform &) const;
        /// Publish identity/version after filling. Shape checks always apply;
        /// finite/inverse scans are opt-in diagnostics. Later writes invalidate
        /// the frozen publication. No payload read/copy is needed by default.
        void FinalizeFaceTransform(FaceCharacteristicTransform &transform, bool checkInverses = false) const;
        /// One local sweep. Caller refreshes mean/coefficient halos first.
        /// Both methods write temporary then publish owned output. SOR follows
        /// logical cell order using new local predecessors and old other values.
        /// With distributed OpenMP enabled, each mesh subpartition is an ordered
        /// block. Other blocks always use frozen input; OMP=1 retains the global
        /// serial sweep. Injected meshes need prepared subpartitions for speed.
        /// No implicit MPI, allocation in the cell loop, or callbacks.
        void DoReconstructionIter(const FVField &mean, FVField &coefficients, ReconstructionWorkspace &work,
                                  const ReconstructionBoundaryValues &boundary,
                                  ReconstructionDeviceKernel kernel = ReconstructionDeviceKernel::FP64) const;
        /// Writes selected rows of temporary only; input is immutable for both
        /// methods. SOR reads already produced local predecessors from temporary.
        /// Split SOR automatically uses core-first logical ordering; All uses
        /// that same ordering when settings.coreFirst is enabled.
        /// OpenMP workers never consume another subpartition's changing output.
        /// Core acquires no input ghost leases. Finish both phases before
        /// CommitReconstruction; never update a live direct MPI send buffer.
        void DoReconstructionIterTo(const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                    const ReconstructionBoundaryValues &boundary,
                                    ReconstructionRegion region = ReconstructionRegion::All) const;
        /// A-weighted LVR penalty: blend the unrelaxed VR row toward target by
        /// alpha, then apply the configured Jacobi/SOR relaxation. Alpha is an
        /// owned scalar table in [0,1]; target has the same coefficient schema.
        /// Freeze both through the reconstruction solve. Alpha0 is unlimited
        /// VR; alpha1 skips neighbor terms. Inputs/target/alpha never alias the
        /// output. Target values must be finite. Only owned target rows are read.
        /// Core/Dependent and Commit have the same contract as the ordinary
        /// two-buffer sweep. Physical sensing and target preparation are external.
        void DoReconstructionIterLimitedTo(const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                           const ReconstructionBoundaryValues &boundary, const FVField &target,
                                           const Runtime::Array<real> &alpha,
                                           ReconstructionRegion region = ReconstructionRegion::All) const;
        /// Managed backend submission variants. Each operation record retains its
        /// shallow field/workspace owners and declares mutable and immutable
        /// Array operands to the eager access graph before queueing. Numerical
        /// semantics are identical to the synchronous kernels above.
        void SubmitReconstructionIterTo(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                        const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                        const ReconstructionBoundaryValues &boundary,
                                        ReconstructionRegion region = ReconstructionRegion::All,
                                        ReconstructionDeviceKernel kernel = ReconstructionDeviceKernel::FP64) const;
        void SubmitReconstructionIterLimitedTo(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                               const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                               const ReconstructionBoundaryValues &boundary, const FVField &target,
                                               const Runtime::Array<real> &alpha,
                                               ReconstructionRegion region = ReconstructionRegion::All) const;
        /// Copy the completed temporary into owned coefficients. The caller
        /// certifies every owned row was produced and all old sends completed.
        void CommitReconstruction(FVField &coefficients, ReconstructionWorkspace &work) const;
        void SubmitCommitReconstruction(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                        FVField &coefficients, ReconstructionWorkspace &work) const;
        /// One descending-order 3WBAP pass. Call degree..1 and exchange the
        /// coefficient halo after EACH pass, matching the legacy schedule.
        void DoLimiterWBAP_3Order(FVField &coefficients, ReconstructionWorkspace &work, int order,
                                  FaceCharacteristicTransformRef transform = std::nullopt) const;
        /// Out-of-place selected-row pass. Call Core and HaloDependent against
        /// the same immutable coefficients, then commit/exchange before the
        /// next descending order. Inactive rows copy input unchanged.
        void DoLimiterWBAP_3OrderTo(const FVField &coefficients, ReconstructionWorkspace &work, int order,
                                    FaceCharacteristicTransformRef transform = std::nullopt,
                                    ReconstructionRegion region = ReconstructionRegion::All,
                                    LimiterIndicatorRef indicator = std::nullopt, real threshold = 0) const;
        void SubmitLimiterWBAP_3OrderTo(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                        const FVField &coefficients, ReconstructionWorkspace &work, int order,
                                        FaceCharacteristicTransformRef transform = std::nullopt,
                                        ReconstructionRegion region = ReconstructionRegion::All,
                                        LimiterIndicatorRef indicator = std::nullopt, real threshold = 0) const;
        Runtime::TransferBinding BindHalo(FVField &, Runtime::TransferTransport transport = Runtime::TransferTransport::Direct,
                                          Runtime::TransferOptions options = {}) const;
        /// Observe device validation status after the producing stream prefix
        /// has completed. Host plans have no deferred device status.
        void CheckStatus(const ReconstructionWorkspace &) const;
        /// Arbitrary physical point sampling from cell means plus zero-mean
        /// coefficients. Cell IDs are current local IDs, including ready ghosts.
        void Evaluate(const FVField &mean, const FVField &coefficients,
                      const Runtime::Array<index> &cells, const Runtime::Array<real> &points,
                      Runtime::Array<real> &values) const;
        /// Spatial derivative of the polynomial, in the cell reference frame.
        /// derivative is a central Taylor multi-index ordinal (0 means value).
        /// For derivative>0 the cell mean contributes zero.
        void EvaluateDerivative(const FVField &mean, const FVField &coefficients,
                                const Runtime::Array<index> &cells, const Runtime::Array<real> &points,
                                Runtime::Array<real> &values, int derivative) const;
        /// Diagnostic copy of one cached basis matrix, rows=retained derivative modes,
        /// columns=nonconstant polynomial modes, in that parent's own frame.
        /// Throws when caching was disabled, the parent is exterior, or the
        /// face lies outside every owned cell stencil.
        Eigen::MatrixXd GetCachedFaceBasis(index iFace, int side, int iQuadrature) const;
        ReconstructionBasisRead AcquireBasisRead() const;
        /// Unscaled assembled linear system A*x - sum(B*x_neighbor). Callers
        /// refresh coefficient ghosts explicitly. Distinct output is required.
        /// Geometry/BC weights are frozen with this plan; no relaxation enters
        /// these operators. General GMRES is valid; symmetry/SPD and nullspace
        /// conditions for PCG must be established by the caller.
        void ApplyLinearOperator(const FVField &coefficients, FVField &output,
                                 const std::vector<index> &periodicVectorStarts = {}) const;
        /// Affine mean/Dirichlet forcing in unscaled coordinates. Homogeneous
        /// perturbations use ApplyLinearOperator without injecting BC values.
        void BuildRightHandSide(const FVField &mean, const ReconstructionBoundaryValues &boundary, FVField &output,
                                const std::vector<index> &periodicVectorStarts = {}) const;
        void ComputeResidual(const FVField &rhs, const FVField &coefficients, FVField &output,
                             const std::vector<index> &periodicVectorStarts = {}) const;
        /// Cell-diagonal truncated SVD inverse, including its minimum-norm
        /// projection when the local matrix has a null space. Owned rows only.
        void ApplyLocalPreconditioner(const FVField &input, FVField &output) const;
    };
}
