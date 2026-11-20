#pragma once
#include "Geom/Unstructured/UnstructuredMeshFwd.hpp"

#include "Geom/Unstructured/UnstructuredMeshSnapshot.hpp"
#include "Geom/Access/EntitySpace.hpp"
#include <map>

namespace DNDS::Geom
{
    class MeshAdjacencyRead;

    /// Immutable native query authority owned by the public mesh facade.
    /// All map/array ownership is private and immutable after publication.
    class UnstructuredMeshState
    {
        friend struct UnstructuredMesh;
        friend class MeshEdit;
        friend class NativeMeshTransition;
        friend class NativeMeshAssembler;
        struct Space
        {
            EntitySpace identity;
            index owned, ghost;
            std::shared_ptr<GlobalOffsetsMapping> global;
            std::shared_ptr<OffsetAscendIndexMapping> ghosts;
        };
        struct NamedRelation
        {
            NativeNamedRelation data;
            std::shared_ptr<Runtime::Array<index>::ReadLease> owned, ghost;
        };
        std::map<AdjacencyDescriptor, NamedRelation> _namedRelations;
        MeshReadRequirements _requirements;
        MeshAuditOptions _audit;
        std::optional<GhostSpec> _ghostSpec;
        MeshIdentity _meshIdentity;
        std::optional<NativeMeshProvenance> _provenance;
        std::vector<std::shared_ptr<Runtime::Array<index>::ReadLease>> _provenanceReads;
        std::map<EntityKind, Space> _spaces;
        std::map<EntityKind, Runtime::Array<index>> _originalIds;
        std::map<EntityKind, std::shared_ptr<Runtime::Array<index>::ReadLease>> _originalReads;
        std::map<std::pair<Runtime::MemorySpace, int>, UnstructuredMeshSnapshot> _placements;
        std::shared_ptr<MeshReadAccess> _hostRead;
        UnstructuredMeshView _hostView;
        Runtime::VersionTracker _versions;
        Runtime::VersionStamp _epoch;
        std::vector<index> _localPartStarts;
        tLocalMatStruct _localCellNeighbors;

        UnstructuredMeshState() = default;
        void BindPreparedEpoch(Runtime::VersionTracker versions);
        static auto PlacementKey(Runtime::Device device)
        {
            return std::make_pair(device.space, device.ordinal);
        }

    public:
        /// Collective native-only publication assembly; all inputs use the same entity
        /// domain/order. ids contains one owned/appended persistent-ID column
        /// for each published family. The default full audit validates owned-ID
        /// uniqueness across ranks and ghost IDs against owners using sparse
        /// routes; data.audit selects explicit producer-trust diagnostics. Each rank must pass
        /// the same communicator, requirements and entity families. No legacy
        /// allocation or global-entity-sized local storage is constructed.
        static std::shared_ptr<const UnstructuredMeshState> FromNative(
            const MPIInfo &mpi, NativeMeshData data, std::map<EntityKind, EntitySpace> spaces,
            std::map<EntityKind, Runtime::Array<index>> ids, MeshReadRequirements requirements);
        /// Bind only after a successful publication transaction has committed.
        std::shared_ptr<const UnstructuredMeshState> BindEpoch(Runtime::VersionTracker versions) const;
        std::shared_ptr<const UnstructuredMeshState> WithPlacement(Runtime::Device device, Runtime::ExecutionContext context) const;
        /// Setup metadata only: retain current part boundaries and owned local
        /// neighbors without changing entity order, buffers or their epochs.
        std::shared_ptr<const UnstructuredMeshState> WithLocalCellSetup(std::vector<index> partStarts,
                                                                        tLocalMatStruct neighbors) const;
        /// Owned face-neighbor graph in current face-slot order. Requires faces;
        /// optionally excludes neighbors outside each current local partition.
        tLocalMatStruct BuildLocalCellNeighbors(bool onLocalPartition = false) const;
        /// Detached setup metadata for a newly prepared cell order. Entity maps
        /// and payloads stay unchanged; neighbors derive from exact face slots.
        /// Primary-only states retain boundaries with an empty neighbor cache;
        /// final facial publication rebuilds the graph in its final slot order.
        std::shared_ptr<const UnstructuredMeshState> WithLocalPartitions(std::vector<index> partStarts) const;
        /// Replace wall vectors while retaining the exact coordinate allocation.
        std::shared_ptr<const UnstructuredMeshState> WithWallDistance(Runtime::ArrayPair<real> walls) const;
        /// Detached geometry-only replacement with identical entity order and
        /// shared topology/identity. Both owned and ghost coordinates must be
        /// ready and coherent; callers arrange any owner/ghost refresh. Clearing
        /// wallVectors explicitly invalidates the derived wall-distance data.
        /// Device placements are discarded and must be rebuilt for the new state.
        std::shared_ptr<const UnstructuredMeshState> WithGeometry(
            Runtime::ArrayPair<real> coordinates, std::optional<Runtime::ArrayPair<real>> wallVectors,
            const Periodicity &periodicInfo) const;
        void Validate() const { _epoch.Validate(); }
        MeshAuditOptions AuditOptions() const
        {
            Validate();
            return _audit;
        }
        /// Final declared coverage, independent of partition-local mappings.
        /// Absent only for low-level/old-format states without a saved policy.
        const std::optional<GhostSpec> &GhostSpecification() const
        {
            Validate();
            return _ghostSpec;
        }
        /// Collective checkpoint audit of this exact immutable publication.
        /// Default options re-run full persistent identity and incidence checks,
        /// even if construction explicitly trusted the producer. This never
        /// rebuilds/reorders payloads or changes the publication's stored policy.
        /// Every rank supplies matching flags and invokes audits in the same order.
        void Audit(MeshAuditOptions options = {}) const;
        /// Rank-local capacity of retained Runtime array allocations, including
        /// every placement and host/device row metadata. Shared allocation IDs
        /// count once, including named/fixed aliases, IDs and ancestry. Excludes
        /// STL containers, entity mappings, access controllers and route caches.
        size_t ArrayBytes() const;
        /// Exact descriptor lookup over host setup storage. Device placement
        /// does not move this registry or change its canonical entity spaces.
        MeshAdjacencyRead Relation(const AdjacencyDescriptor &key) const;
        std::vector<AdjacencyDescriptor> RelationKeys() const;
        void CheckRequirements(MeshReadRequirements requested) const;
        const UnstructuredMeshSnapshot &Snapshot(Runtime::Device device) const;
        const UnstructuredMeshView &BorrowHostView() const { return _hostView; }
        bool HasEntities(EntityKind kind) const { return _spaces.find(kind) != _spaces.end(); }
        index NumOwned(EntityKind kind) const;
        index NumGhost(EntityKind kind) const;
        index NumGlobal(EntityKind kind) const;
        EntitySpace Entities(EntityKind kind) const;
        /// Immutable execution metadata, available without validating geometry
        /// or acquiring a mesh read scope. Null means enabled compatibility.
        std::shared_ptr<Runtime::CommunicationContext> PreparationCommunication() const;
        MeshReadRequirements Requirements() const { return _requirements; }
        index OriginalEntityId(EntityKind kind, index row) const;
        /// Retain the immutable combined owned/ghost ID allocation. Consumers
        /// coupling these IDs to other mesh rows also retain a MeshReadAccess.
        std::shared_ptr<const Runtime::Array<index>::ReadLease> ReadOriginalEntityIds(EntityKind kind) const
        {
            Validate();
            auto found = _originalReads.find(kind);
            DNDS_check_throw_info(found != _originalReads.end(), "original entity ID family is unavailable");
            return found->second;
        }
        const std::string &MeshNamespace() const { return _meshIdentity.meshNamespace; }
        /// Shared owned-row metadata. Array reads retain their own leases;
        /// publication prevents writes for this state's lifetime.
        const std::optional<NativeMeshProvenance> &Provenance() const
        {
            Validate();
            return _provenance;
        }
        const std::map<EntityKind, index> &IdHighWater() const { return _meshIdentity.nextIds; }
        index NextEntityId(EntityKind kind) const
        {
            auto found = _meshIdentity.nextIds.find(kind);
            DNDS_check_throw_info(found != _meshIdentity.nextIds.end(), "persistent-ID high-water mark is unavailable");
            return found->second;
        }
        int NLocalParts() const { return _localPartStarts.empty() ? 1 : int(_localPartStarts.size() - 1); }
        index LocalPartStart(int part) const { return _localPartStarts.empty() ? 0 : _localPartStarts.at(size_t(part)); }
        index LocalPartEnd(int part) const { return _localPartStarts.empty() ? NumOwned(EntityKind::Cell) : _localPartStarts.at(size_t(part + 1)); }
        const tLocalMatStruct &LocalCellNeighbors() const { return _localCellNeighbors; }
    };
}
