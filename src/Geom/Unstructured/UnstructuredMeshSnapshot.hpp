#pragma once
#include "Geom/Unstructured/UnstructuredMeshFwd.hpp"

#include "Geom/Access/MeshAccess.hpp"
#include "Geom/Connectivity/MeshRelationTypes.hpp"
#include "Geom/Unstructured/MeshIdentity.hpp"
#include "Geom/Unstructured/NativeMeshProvenance.hpp"
#include "Geom/Connectivity/MeshTopologyTypes.hpp"
#include <optional>
#include <map>
#include <functional>

namespace DNDS::Geom
{
    /// Host setup relation, keyed separately by its complete AdjacencyDescriptor definition.
    /// Fixed facade slots may share these allocations without duplicating rows.
    struct NativeNamedRelation
    {
        Runtime::ArrayPair<index> rows;
        MeshAdjState state = Adj_PointToGlobal;
    };

    /// Native host construction input. Values and metadata are adopted without
    /// copying. Mutable aliases must obey the Array access contract; publication
    /// retains read leases. No legacy mesh or transformer is required.
    struct NativeMeshData
    {
        /// Consumed by UnstructuredMeshState; setup relations remain host-only even
        /// when facade placements are prepared on a device.
        std::map<AdjacencyDescriptor, NativeNamedRelation> namedRelations;
        int dimension = -1;
        bool periodic = false;
        Periodicity periodicInfo;
        std::array<std::optional<Runtime::ArrayPair<index>>, static_cast<size_t>(MeshRelation::Count)> relations;
        std::optional<Runtime::ArrayPair<real>> coords, nodeWallDist;
        std::optional<Runtime::ArrayPair<ElemInfo>> cellInfo, bndInfo, faceInfo, edgeInfo;
        std::optional<Runtime::ArrayPair<NodePeriodicBits>> cellNodesPbi, bndNodesPbi, cellFacesPbi, faceNodesPbi, cellEdgesPbi, edgeNodesPbi;
        /// Durable UUID and non-reuse marks, consumed by UnstructuredMeshState.
        MeshIdentity identity;
        /// Owned source-ID payload, consumed by UnstructuredMeshState (host setup).
        std::optional<NativeMeshProvenance> provenance;
        /// Effective publication policy; producer trust is explicit, never global.
        MeshAuditOptions audit;
        /// Declarative final closure policy; placements do not reinterpret it.
        std::optional<GhostSpec> ghostSpec;
    };

    /// Immutable native backing used by authoritative placements or detached
    /// snapshots. Explicit Capture copies current native rows once. Repeated
    /// AcquireRead retains values, shape and controllers. Source replacement
    /// cannot alter a detached snapshot; its source stamp is provenance only.
    class UnstructuredMeshSnapshot
    {
        friend class MeshReadAccess;
        friend class CellFaceIncidenceStorage;
        friend class MeshAdjacencyRead;
        friend class NativeMeshTransition;
        friend class NativeMeshAssembler;
        friend class UnstructuredMeshState;
        friend struct NativeMeshSerialOutput;
        friend class MeshEdit;
        template <class T>
        using Pair = Runtime::ArrayPair<T>;
        struct State
        {
            Runtime::VersionTracker versions;
            Runtime::VersionStamp source;
            Runtime::Device device;
            MeshAuditOptions audit;
            MeshViewStorage semantics;
            std::array<std::optional<Pair<index>>, static_cast<size_t>(MeshRelation::Count)> relations;
            std::optional<Pair<real>> coords, nodeWallDist;
            std::optional<Pair<ElemInfo>> cellInfo, bndInfo, faceInfo, edgeInfo;
            std::optional<Pair<NodePeriodicBits>> cellNodesPbi, bndNodesPbi, cellFacesPbi, faceNodesPbi, cellEdgesPbi, edgeNodesPbi;
            /// FromNative may adopt externally aliased allocations. Keep them
            /// read-protected for this immutable publication's entire lifetime.
            std::shared_ptr<MeshReadAccess> immutableRead;
        };
        std::shared_ptr<const State> _state;
        explicit UnstructuredMeshSnapshot(std::shared_ptr<const State> state) : _state(std::move(state)) {}
        const State &Live() const
        {
            DNDS_check_throw_info(_state, "native mesh snapshot was moved or released");
            return *_state;
        }
        template <class T>
        static Pair<T> ClonePair(Runtime::PairReadView<T> source);
        static UnstructuredMeshSnapshot FromRead(MeshReadAccess read);

    public:
        /// Consume validated local-index host buffers. Unsupported or missing
        /// required topology fails before returning a usable snapshot. data.audit
        /// controls diagnostic reciprocity only; all index validity remains required.
        static UnstructuredMeshSnapshot FromNative(NativeMeshData data, MeshReadRequirements requirements);
        static UnstructuredMeshSnapshot Capture(UnstructuredMesh &source, MeshReadRequirements requirements = MeshReadRequirements::Primary());
        UnstructuredMeshSnapshot To(Runtime::Device device, Runtime::ExecutionContext context) const;
        UnstructuredMeshSnapshot To(Runtime::Device device) const { return To(device, {device == Runtime::Device::Host() ? GetDevice() : device, nullptr}); }
        /// Enumerate retained payload and metadata allocations. Aliases may
        /// repeat an identity; consumers deduplicate when computing totals.
        /// Capacities exclude native allocator bookkeeping/alignment overhead.
        void VisitAllocations(const std::function<void(uint64_t, size_t, Runtime::Device)> &visitor) const;
        Runtime::Device GetDevice() const { return Live().device; }
        Runtime::VersionStamp SourceStamp() const { return Live().source; }
        MeshAuditOptions AuditOptions() const { return Live().audit; }
        MeshReadAccess AcquireRead(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const
        {
            return MeshReadAccess(*this, context);
        }
    };
}
