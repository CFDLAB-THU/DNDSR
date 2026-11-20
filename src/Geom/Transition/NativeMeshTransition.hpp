#pragma once
#include "Geom/Unstructured/UnstructuredMeshState.hpp"
#include "Geom/Unstructured/OwnedMeshData.hpp"
#include "Geom/Transition/MeshReorderTypes.hpp"
#include "DNDS/Legacy/PermutationTransfer.hpp"
#include <unordered_map>

namespace DNDS::Geom
{
    struct LocalCellOrderingSettings;
    /// Boundary placement follows the cell in its physical image. The public
    /// Bnd2CellCoupled relation carries the periodic partner in its second slot;
    /// that support must not change which side owns the boundary. The legacy
    /// Bnd2Cell facade retains own/self slots for a periodic boundary.
    inline AdjacencyDescriptor NativeBoundaryPlacementRelation(const UnstructuredMeshState &source)
    {
        if (!source.BorrowHostView().IsPeriodic())
            return Adj::Bnd2Cell;
        return AdjacencyDescriptor::Composition(Adj::Bnd2Node, Adj::Cell2Node.Inverted(), RelationSelection::Exact, false,
                                                "geom.boundary.physical-frame-face", 1, RelationPeriodicPolicy::FrameMatched);
    }

    struct NativeMeshTransitionResult
    {
        std::shared_ptr<const UnstructuredMeshState> state;
        std::unordered_map<EntityKind, PermutationTransfer> transfers;
        /// Exact input associations for validating external fields before
        /// transfer. Numeric identity maps do not imply matching entity stamps.
        std::map<EntityKind, EntitySpace> sourceSpaces;
    };

    /// Owned construction output, with no finalized compute ghosts.
    struct NativeOwnedMeshTransitionResult
    {
        NativeOwnedMeshData owned;
        std::unordered_map<EntityKind, PermutationTransfer> transfers;
        std::map<EntityKind, EntitySpace> sourceSpaces;
    };

    /// Detached host-native transition. Persistent IDs and ordered companions
    /// move as payload; topology indices use sparse owner-directed remapping.
    class NativeMeshTransition
    {
        static NativeOwnedMeshTransitionResult BuildOwnedImpl(const NativeOwnedMeshData &source, const MPIInfo &mpi,
                                                              const ReorderInput &input,
                                                              const std::vector<index> *cellOld2New,
                                                              bool validateOwnedProvenance = false);
        static NativeMeshTransitionResult BuildImpl(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                                    const ReorderInput &input, int ghostLayers,
                                                    const std::vector<index> *cellOld2New);

    public:
        /// Redistribute authoritative owned global rows and companions, then
        /// allow further owned topology construction before final assembly.
        /// This stage retains all families; explicit reconstruction uses Build.
        static NativeOwnedMeshTransitionResult BuildOwned(const NativeOwnedMeshData &source, const MPIInfo &mpi,
                                                          const ReorderInput &input);
        /// Owned-only local permutation using exact temporary facial graph
        /// scratch and full node-neighbor private-cell classification. Carries
        /// the resulting local partition ranges through final assembly.
        static NativeOwnedMeshTransitionResult ReorderOwnedCells(const NativeOwnedMeshData &source, const MPIInfo &mpi,
                                                                 int nParts = 1, int nPartsInner = 1, bool coreFirst = false);
        static NativeMeshTransitionResult Build(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                                const ReorderInput &input, int ghostLayers = 1);
        /// Detached setup-only rank-local cell permutation. Other family owned
        /// order stays unchanged. All fixed/named references, companions, IDs
        /// and ancestry follow the same native transfer machinery. Existing
        /// fields/routes remain associated with source; transfers names every
        /// family (identity maps included) for explicit external field transfer.
        static NativeMeshTransitionResult ReorderLocalCells(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                                            int nParts = 1, int nPartsInner = 1, int ghostLayers = 1);
        /// New explicit conservative-core policy. subregion -> optional
        /// node-star core/noncore -> optional sender grouping -> RCM. The
        /// existing integer overload retains its historical ordering semantics.
        static NativeMeshTransitionResult ReorderLocalCells(UnstructuredMesh &source,
                                                            const struct LocalCellOrderingSettings &settings, int ghostLayers = 1);
    };
}
