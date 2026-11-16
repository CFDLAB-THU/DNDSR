#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::CFV
{
    namespace Detail
    {
        struct GradientIdentity
        {
        };
    }
    enum class GradientMethod
    {
        GreenGauss,
        LeastSquares,
        GGMP
    };

    /// Optional equation-variable transform for one-level physical-gradient
    /// WBAP. Spatial gradient-frame rotation remains a separate mesh operation.
    /// `count == 0` is a valid disabled descriptor and owns no matrix buffers.
    struct GradientFaceCharacteristicTransform
    {
        index first = 0, count = 0;
        Runtime::Array<real> forward, inverse;
        std::shared_ptr<const Detail::GradientIdentity> plan;
        uint64_t forwardVersion = 0, inverseVersion = 0;
        uint64_t forwardIdentity = 0, inverseIdentity = 0;
        CharacteristicValidation validation = CharacteristicValidation::Frozen;
    };
    using GradientFaceCharacteristicTransformRef =
        std::optional<std::reference_wrapper<const GradientFaceCharacteristicTransform>>;

    /// Reusable submission metadata for one gradient component layout. No
    /// durable gradients or coefficients live here. The periodic triples are
    /// published once on the chosen backend; rebuild if that layout changes.
    struct GradientWorkspace
    {
        std::shared_ptr<const Detail::GradientIdentity> plan;
        Runtime::Array<index> periodicVectorStarts;
        Runtime::Array<real> empty;
        index components = 0;
        uint64_t vectorIdentity = 0, vectorVersion = 0, vectorShapeVersion = 0;
        /// Optional CPU worker scratch, allocated only for limiter workspaces.
        /// CUDA uses bounded local scratch; this table has zero rows there.
        Runtime::Array<real> limiterScratch;
        bool withLimiter = false;
        /// Sticky device validation status, only allocated for CUDA limiting.
        Runtime::Array<int> deviceStatus;
    };

    /// Independent cell-centered FV gradient preparation. BC inputs are physical
    /// face-center values, not reflected ghost states. Each owned cell-face
    /// incidence has one weight and boundary-ID entry, independent of the RHS
    /// quadrature degree. LS minimizes residuals
    /// with inverse-distance-squared weights using the design matrix SVD (not
    /// normal equations). GGMP eliminates face values from the coupled corrected
    /// Green-Gauss equations. No legacy VR owner or boundary callback is used.
    class GradientPlan
    {
        struct Data
        {
            std::shared_ptr<const Detail::GradientIdentity> identity =
                std::make_shared<const Detail::GradientIdentity>();
            FiniteVolumePlan metrics;
            Geom::EntityWorkPartition partition;
            Runtime::Array<index> offsets, boundaryIds;
            Runtime::Array<index> core, dependent;
            Runtime::Array<real> weights;
            std::vector<BoundaryQuadraturePoint> boundary;
            index boundaryCount = 0;
            Data(FiniteVolumePlan plan) : metrics(std::move(plan)), partition(metrics.CellWorkPartition()) {}
        };
        std::shared_ptr<const Data> _data;
        explicit GradientPlan(std::shared_ptr<const Data> data) : _data(std::move(data)) {}
        void ValidateEvaluation(const FVField &mean, const Runtime::Array<real> &boundaryValues,
                                const FVField &gradient, ReconstructionRegion region) const;
        void ValidateWorkspace(const GradientWorkspace &workspace, index components) const;
        void ValidateLimiter(const FVField &input, const FVField &output, GradientFaceCharacteristicTransformRef transform,
                             ReconstructionRegion region, LimiterIndicatorRef indicator, real threshold, real centerWeight) const;
        void ValidateConversion(const FVField &source, const FVField &destination,
                                const VariationalReconstructionPlan &reconstruction, bool toCoefficients, bool ghost) const;
        void ConvertFields(const FVField &source, FVField &destination, const VariationalReconstructionPlan &reconstruction,
                           bool toCoefficients, bool ghost) const;
        void ConvertFields(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                           const FVField &source, FVField &destination, const VariationalReconstructionPlan &reconstruction,
                           bool toCoefficients, bool ghost) const;

    public:
        static GradientPlan Prepare(Geom::UnstructuredMesh &mesh, FiniteVolumePlan metrics,
                                    GradientMethod method = GradientMethod::LeastSquares, real relativeCutoff = 1e-13);
        Runtime::Device Device() const { return _data->metrics.Device(); }
        /// Synchronous setup placement. The overload accepting placed metrics
        /// shares that publication with the caller instead of duplicating it.
        /// Device plans retain no CPU boundary-coordinate vector; obtain those
        /// setup values before placement. All numerical field storage is resident.
        GradientPlan Placed(Runtime::Device device, Runtime::ExecutionContext context) const;
        GradientPlan Placed(FiniteVolumePlan metrics, Runtime::ExecutionContext context) const;
        GradientWorkspace BuildWorkspace(index components, const std::vector<index> &periodicVectorStarts = {}, bool withLimiter = false) const;
        /// Number of face-center BC rows; retained API spelling does not imply
        /// expansion by the metric's high-order face quadrature rule.
        index NumBoundaryQuadrature() const;
        BoundaryQuadraturePoint GetBoundaryQuadrature(index iPoint) const;
        Runtime::Array<real> BuildBoundaryValues(index components) const;
        GradientFaceCharacteristicTransform BuildFaceTransform(index first, index count) const;
        /// Mutable backend matrices. Caller orders producer then limiting.
        /// Numerical inverse checking is a separate opt-in diagnostic.
        GradientFaceCharacteristicTransform BuildQueuedFaceTransform(index first, index count) const;
        /// Opt-in diagnostic, not called automatically by a limiter.
        void SubmitValidateFaceTransform(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                         const GradientFaceCharacteristicTransform &, GradientWorkspace &) const;
        void CheckStatus(const GradientWorkspace &workspace) const;
        void FinalizeFaceTransform(GradientFaceCharacteristicTransform &transform, bool checkInverses = false) const;
        /// Copy an already validated host transform to this plan's placement.
        /// Shape/value identities are frozen until a new validated publication.
        GradientFaceCharacteristicTransform PlaceFaceTransform(const GradientFaceCharacteristicTransform &transform,
                                                               Runtime::ExecutionContext context) const;
        /// Convert owned gradients to linear Taylor coefficients and zero all
        /// higher modes. Zero-mean basis moments preserve cell averages. This
        /// is an explicit overwrite, not a limiter or second reconstruction.
        void ConvertToCoefficients(const FVField &gradient, const VariationalReconstructionPlan &reconstruction,
                                   FVField &coefficients) const;
        void ConvertToCoefficients(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                                   const FVField &gradient, const VariationalReconstructionPlan &reconstruction, FVField &coefficients) const;
        /// Explicit compatibility projection of linear modes into physical
        /// gradients, without basis evaluation. Higher modes are ignored: the
        /// caller must supply a linear polynomial, e.g. the accepted direct-FV
        /// limiter result. ghost=false overwrites owned rows; ghost=true only
        /// ghost rows, after the coefficient exchange has completed. No MPI.
        void ExtractLinearGradients(const FVField &coefficients, const VariationalReconstructionPlan &reconstruction,
                                    FVField &gradient, bool ghost = false) const;
        void ExtractLinearGradients(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                                    const FVField &coefficients, const VariationalReconstructionPlan &reconstruction,
                                    FVField &gradient, bool ghost = false) const;
        /// Owned output rows hold [component*dimension+direction]. Core reads
        /// owned means only, so an input ghost write may be active. Complete
        /// the input halo before HaloDependent/All. BC values must be ready.
        /// Partial calls overwrite only their selected output rows; the caller
        /// completes both phases before consuming a full gradient. Calls sharing
        /// an output are serialized. Output must be distinct from all inputs.
        void Evaluate(const FVField &mean, const Runtime::Array<real> &boundaryValues, FVField &gradient,
                      const std::vector<index> &periodicVectorStarts = {}, ReconstructionRegion region = ReconstructionRegion::All) const;
        /// Backend-neutral queued evaluation. Only HaloDependent/All declares
        /// a ghost read. The caller orders communication and serializes writes
        /// to the output; this method performs neither transfers nor waits.
        void Evaluate(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                      const FVField &mean, const Runtime::Array<real> &boundaryValues, FVField &gradient,
                      const GradientWorkspace &workspace, ReconstructionRegion region = ReconstructionRegion::All) const;
        /// Apply one physical-gradient 3WBAP level. Neighbor gradients are first
        /// mapped into the center cell's periodic spatial/component frame. A
        /// nonempty characteristic range is transformed per spatial direction;
        /// all remaining components are scalar-limited. Output owned rows are
        /// replaced and remain halo-stale. No polynomial modes or secondary
        /// reconstruction matrices are constructed.
        void LimitWBAP3(const FVField &input, FVField &output,
                        const std::vector<index> &periodicVectorStarts = {},
                        GradientFaceCharacteristicTransformRef transform = std::nullopt,
                        ReconstructionRegion region = ReconstructionRegion::All,
                        LimiterIndicatorRef indicator = std::nullopt,
                        real threshold = 0, real centerWeight = 10) const;
        /// Enqueue without halo transfer or a host wait. Workspace must have
        /// withLimiter=true. Activation values must be finite; queued leaves
        /// assert this precondition rather than copying them to host to inspect.
        void LimitWBAP3(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                        const FVField &input, FVField &output, GradientWorkspace &workspace,
                        GradientFaceCharacteristicTransformRef transform = std::nullopt,
                        ReconstructionRegion region = ReconstructionRegion::All,
                        LimiterIndicatorRef indicator = std::nullopt,
                        real threshold = 0, real centerWeight = 10) const;
    };
}
