#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace DNDS::CFV
{
    /// Scoped setup-only placement memo. One publication uses one memo so
    /// aliases of immutable query/basis buffers remain aliases on the target
    /// backend. Entries retain their sources only until this memo is destroyed;
    /// the resulting plans retain their target allocations independently.
    /// This is neither a global cache nor an asynchronous transfer operation.
    class SamplingPlacementMemo
    {
        template <class T>
        struct Entry
        {
            Runtime::Array<T> source, placed;
            uint64_t version, shapeVersion, mutationEpoch;
        };
        Runtime::Device _device;
        Runtime::ExecutionContext _context;
        std::vector<Entry<real>> _real;
        std::vector<Entry<index>> _indices;
        template <class T>
        Runtime::Array<T> PlaceImpl(const Runtime::Array<T> &source, std::vector<Entry<T>> &entries)
        {
            for (const auto &entry : entries)
                if (entry.source.Identity() == source.Identity() && entry.source.Storage().SameRange(source.Storage()) &&
                    entry.source.Layout() == source.Layout())
                {
                    DNDS_check_throw_info(entry.version == source.Version() && entry.shapeVersion == source.ShapeVersion() && entry.mutationEpoch == source.MutationEpoch(),
                                          "sampling placement source mutated during publication");
                    return entry.placed;
                }
            auto placed = source.GetDevice() == _device ? source : source.To(_device, _context);
            entries.push_back({source, placed, source.Version(), source.ShapeVersion(), source.MutationEpoch()});
            return placed;
        }
        template <class T>
        void SeedImpl(const Runtime::Array<T> &source, const Runtime::Array<T> &placed, std::vector<Entry<T>> &entries)
        {
            DNDS_check_throw_info(placed.GetDevice() == _device && source.Layout() == placed.Layout(),
                                  "sampling placement seed layout/device mismatch");
            for (const auto &entry : entries)
                if (entry.source.Identity() == source.Identity() && entry.source.Storage().SameRange(source.Storage()) &&
                    entry.source.Layout() == source.Layout())
                {
                    DNDS_check_throw_info(entry.placed.Identity() == placed.Identity() && entry.placed.Storage().SameRange(placed.Storage()) &&
                                              entry.version == source.Version() &&
                                              entry.shapeVersion == source.ShapeVersion() && entry.mutationEpoch == source.MutationEpoch(),
                                          "conflicting sampling placement seed");
                    return;
                }
            entries.push_back({source, placed, source.Version(), source.ShapeVersion(), source.MutationEpoch()});
        }

    public:
        SamplingPlacementMemo(Runtime::Device device, Runtime::ExecutionContext context) : _device(device), _context(context) {}
        Runtime::Device Device() const { return _device; }
        Runtime::Array<real> Place(const Runtime::Array<real> &source) { return PlaceImpl(source, _real); }
        Runtime::Array<index> Place(const Runtime::Array<index> &source) { return PlaceImpl(source, _indices); }
        /// Seed a previously placed owner buffer (e.g. face RHS basis). The
        /// caller guarantees that placed is the unchanged copy of source.
        void Seed(const Runtime::Array<real> &source, const Runtime::Array<real> &placed) { SeedImpl(source, placed, _real); }
        void Seed(const Runtime::Array<index> &source, const Runtime::Array<index> &placed) { SeedImpl(source, placed, _indices); }
    };

    /// Caller-owned point values and physical gradients. Null entries omit an
    /// output; gradient slots refer to mesh-reference Cartesian directions.
    struct PolynomialSampleOutputs
    {
        Runtime::Array<real> *values = nullptr;
        std::array<Runtime::Array<real> *, 3> gradients{};
    };

    /// Immutable point-sampling cache: snapshots local cell IDs and direct
    /// points, and either computes its basis or retains a shared immutable bank.
    /// Optional query-to-basis and query-to-output maps preserve consumer order
    /// without duplicating numerical basis payloads.
    /// Copies share only the required Runtime buffers and row-space identity;
    /// the reconstruction matrix/workspace owner is not retained. Geometry or
    /// ordering changes invalidate the captured stamp. A moved interface needs
    /// a new cache. Cached-only plans retain no direct-evaluation geometry.
    /// Outputs remain in each donor's reference/component frame.
    class PolynomialSamplingPlan
    {
        ssp<Geom::UnstructuredMesh> _mesh;
        Runtime::VersionStamp _stamp;
        Geom::EntitySpace _cellSpace;
        Algebra::DofSpace _space;
        Runtime::Array<index> _cells, _basisRows, _outputRows;
        Runtime::Array<real> _points, _basis, _moments, _cellGeometry, _empty;
        int _dimension = 0, _degree = 0, _modes = 0, _cachedDerivatives = 0;
        bool _needsGhost = false;
        bool _ownsBasis = true, _basisFrozen = false;
        int _maximumDerivative = 0;
        struct FrozenArrayState
        {
            uint64_t version = 0, shapeVersion = 0, mutationEpoch = 0;
            void Capture(const Runtime::Array<real> &array)
            {
                version = array.Version();
                shapeVersion = array.ShapeVersion();
                mutationEpoch = array.MutationEpoch();
            }
            bool Matches(const Runtime::Array<real> &array) const
            {
                return version == array.Version() && shapeVersion == array.ShapeVersion() && mutationEpoch == array.MutationEpoch();
            }
        };
        FrozenArrayState _basisState, _momentsState, _cellGeometryState;
        index _outputSize = -1;
        std::string _name = "CFV/quadrature/arbitrary";
        struct SharedBasisInput
        {
            Runtime::Array<real> basis;
            Runtime::Array<index> rows;
            int derivatives;
        };
        PolynomialSamplingPlan(FiniteVolumePlan metrics, VariationalReconstructionPlan reconstruction,
                               const Runtime::Array<index> &cells, const Runtime::Array<real> &points, bool cacheDerivatives,
                               Runtime::ExecutionContext sourceContext, std::optional<SharedBasisInput> sharedBasis);
        void NameOwnedBuffers() const;
        void FreezeBasis();
        void ValidateField(const FVField &field, index width) const;
        void ValidateOutput(const FVField &mean, const FVField &coefficients, const Runtime::Array<real> &output, int derivative) const;
        void EvaluateDeviceValueGradient(const FVField &mean, const FVField &coefficients, const PolynomialSampleOutputs &outputs) const;

    public:
        /// cacheDerivatives=false permits value queries only (e.g. PP).
        /// Otherwise all derivatives may be queried; orders outside the selected
        /// cache coverage are evaluated on demand. Global cache-off stores none.
        PolynomialSamplingPlan(VariationalReconstructionPlan reconstruction,
                               const Runtime::Array<index> &cells, const Runtime::Array<real> &points, bool cacheDerivatives = true,
                               Runtime::ExecutionContext sourceContext = Runtime::ExecutionContext::Host());
        /// Explicit metric owner permits rebuilding queries after a resident
        /// reconstruction has released its host setup matrices. Device-only
        /// reconstruction must retain moments via arbitraryPointSampling or
        /// direct-basis evaluation at placement.
        PolynomialSamplingPlan(FiniteVolumePlan metrics, VariationalReconstructionPlan reconstruction,
                               const Runtime::Array<index> &cells, const Runtime::Array<real> &points, bool cacheDerivatives,
                               Runtime::ExecutionContext sourceContext);
        /// Construction remains CPU setup. With a resident reconstruction,
        /// sourceContext explicitly orders the temporary moments/metric download;
        /// it never downloads matrices or solution fields. Placed() publishes
        /// the resulting sampling cache on the desired backend afterward.
        /// Required output extent; a scattered segment may write fewer rows.
        index Size() const { return _outputSize < 0 ? QueryNumber() : _outputSize; }
        index QueryNumber() const { return _cells.RowCount(); }
        /// Construct queries over an existing immutable cache without allocating
        /// another basis table. rows maps query ordinal to cache row; an empty
        /// mapping means identity. The caller establishes semantic point/frame
        /// identity (never merely coordinate equality). All handles are retained.
        /// With derivatives=0 the plan retains only direct-evaluation metadata.
        static PolynomialSamplingPlan FromSharedBasis(FiniteVolumePlan metrics, VariationalReconstructionPlan reconstruction,
                                                      const Runtime::Array<index> &cells, const Runtime::Array<real> &points,
                                                      const Runtime::Array<real> &basis, const Runtime::Array<index> &rows,
                                                      int derivatives, bool cacheDerivatives = false,
                                                      Runtime::ExecutionContext sourceContext = Runtime::ExecutionContext::Host());
        /// Copy with unique output row destinations. Rows outside this segment
        /// remain unchanged. Setup snapshots and validates the map; queued calls
        /// retain it as an ordinary native-array read operand.
        PolynomialSamplingPlan WithOutputRows(const Runtime::Array<index> &rows, index totalRows) const;
        /// Restrict queries to derivatives through order (0 for values). This
        /// never widens the original contract. When the existing cache covers
        /// every permitted derivative, release direct points/moments/geometry;
        /// the shared basis payload and row mappings are unchanged.
        PolynomialSamplingPlan WithDerivativeOrder(int order) const;
        /// Diagnostic naming for owned payloads; borrowed basis/geometry owners
        /// retain their original names. Copies continue sharing the allocations.
        PolynomialSamplingPlan Named(std::string name) const;
        const Runtime::Array<real> &BasisStorage() const { return _basis; }
        /// Immutable direct-point coordinates retained only when a requested
        /// derivative is uncached. Cell-major consumers with an independently
        /// proven row order may share this allocation without another sampler.
        /// Fully cached plans expose an empty point allocation.
        const Runtime::Array<real> &PointStorage() const { return _points; }
        int CachedDerivativeNumber() const { return _cachedDerivatives; }
        /// Checks geometry/order generations and frozen basis/normalization provenance before
        /// execution. Aliased writes, including queued write intent, invalidate
        /// the publication. Serialize setup/mutation with submission; retained
        /// operation reads then protect the data through completion.
        void Validate() const;
        Runtime::Device Device() const { return _cells.GetDevice(); }
        /// Synchronous setup placement. The copy retains no source payloads;
        /// discard the original to release its host caches. No VR matrices are
        /// copied. Direct geometry is present only for uncached derivatives.
        PolynomialSamplingPlan Placed(Runtime::Device device, Runtime::ExecutionContext context) const;
        /// Use one scoped memo across all consumers in a publication to preserve
        /// shared immutable buffers on the target backend, including face caches.
        PolynomialSamplingPlan Placed(SamplingPlacementMemo &memo) const;
        /// derivative is the central Taylor derivative-mode ordinal. Zero
        /// samples value; higher ordinals sample derivatives without the mean.
        /// Only caches containing ghost cells acquire input ghost leases.
        void Evaluate(const FVField &mean, const FVField &coefficients, Runtime::Array<real> &output, int derivative = 0) const;
        /// Unified queued sampling. Backend dispatch and borrowed-view building
        /// stay here; the caller supplies explicit stream dependency ordering.
        /// Operation retention pins plan geometry, input/output allocations and
        /// access through CPU work or device completion. No host synchronization,
        /// transfer or MPI occurs inside the numerical leaf. Input ghosts are
        /// declared only when the prepared query cells actually require them.
        void Evaluate(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                      const FVField &mean, const FVField &coefficients, Runtime::Array<real> &output, int derivative = 0) const;
        /// Value-only counterpart with the same shape, placement, geometry and
        /// lease contract. Common cached owned AoS queries use bounded CPU
        /// kernels; all other queries call Evaluate(..., 0). Keeping this
        /// separate leaves derivative evaluation independent of specialization.
        void EvaluateValues(const FVField &mean, const FVField &coefficients, Runtime::Array<real> &output) const;
        /// Synchronous sampling of values and/or first derivatives. All
        /// supplied outputs must have Size() rows and mean.Width() components,
        /// with no input/output or output/output overlap. At least one output
        /// is required. No output allocation, halo exchange or hidden layout conversion
        /// occurs. Only ghost queries acquire input ghost leases. On failure,
        /// discard the output bundle; publication belongs to the caller.
        void EvaluateValueGradient(const FVField &mean, const FVField &coefficients, const PolynomialSampleOutputs &outputs) const;
    };
}
