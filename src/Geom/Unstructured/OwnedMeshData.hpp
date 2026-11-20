#pragma once

#include "Geom/Unstructured/UnstructuredMeshState.hpp"
#include "Geom/Connectivity/MeshTopologyTypes.hpp"

namespace DNDS::Geom
{
    /// Authoritative owned topology in current distributed GLOBAL numbering.
    /// Temporary remote inputs used by construction algorithms do not become
    /// canonical ghosts here. Final ghost materialization is an explicit,
    /// separate assembler operation with a caller-supplied GhostSpec.
    struct NativeOwnedMeshData
    {
        /// Optional immutable safety policy for this preparation pipeline.
        /// Null retains enabled compatibility behavior. Not mesh identity/I/O.
        /// Keep all spaces on the same policy; a context change on this detached
        /// record also requires WithCommunication on its entity descriptors.
        std::shared_ptr<Runtime::CommunicationContext> communication;
        template <class F>
        void LocalStage(const MPIInfo &mpi, F &&run, const char *message = "mesh preparation failed on a rank") const
        {
            Runtime::CommunicationDetail::LocalStage(mpi.comm, !communication || communication->ChecksCollectivePreflight(),
                                                     std::forward<F>(run), message);
        }
        MeshIdentity identity;
        MeshReadRequirements requirements;
        /// True only when Cell2Cell was built as the exact Cell->Node->Cell
        /// union and retained through partitioning/reordering. Detached or
        /// deliberately truncated graphs leave this false so setup uses the
        /// authoritative Node2Cell inverse instead.
        bool cellCellsComplete = false;
        int dimension = -1;
        bool periodic = false;
        Periodicity periodicInfo;
        std::map<EntityKind, EntitySpace> spaces;
        std::array<std::optional<Runtime::Array<index>>, size_t(MeshRelation::Count)> relations;
        std::map<AdjacencyDescriptor, Runtime::Array<index>> namedRelations;
        std::optional<Runtime::Array<real>> coords, nodeWallDist;
        std::optional<Runtime::Array<ElemInfo>> cellInfo, bndInfo, faceInfo, edgeInfo;
        std::optional<Runtime::Array<NodePeriodicBits>> cellNodesPbi, bndNodesPbi, cellFacesPbi, faceNodesPbi, cellEdgesPbi, edgeNodesPbi;
        std::map<EntityKind, Runtime::Array<index>> ids;
        std::optional<NativeMeshProvenance> provenance;
        std::vector<std::shared_ptr<Runtime::Array<index>::ReadLease>> retainedProvenance;
        /// Protect shared input companion allocations after CaptureOwned even
        /// when the originating mesh state is released.
        std::shared_ptr<const MeshReadAccess> retainedRead;
        MeshAuditOptions audit;
        /// Declarative policy only; no pointers, routes or local row numbers.
        std::optional<GhostSpec> ghostSpec;
        /// Optional owned-cell partition ranges, installed at final assembly.
        /// Redistribution clears these; a rank-local permutation supplies new
        /// ranges. No mesh publication is needed to carry setup scheduling.
        std::vector<WorkerRowRanges> localPartRanges;
    };
}
