#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Unstructured/NativeMeshPreparation.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/IO/Metis.hpp"
#include "Geom/Transition/NativeMeshPartitionGraph.hpp"
#include "Geom/Unstructured/ReaderPartitionGraph.hpp"
#include "Geom/Connectivity/TopologySubEntityQuery.hpp"
#include "Geom/Connectivity/ConnectivityInterpolateRows.hpp"
#include "Geom/Unstructured/NativeMeshBuilder.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include <map>
#include <cmath>
#include <limits>
#include <set>

namespace DNDS::Geom
{
    std::shared_ptr<const UnstructuredMeshState> PrepareNativeMeshTopology(
        const UnstructuredMeshState &source, const MPIInfo &mpi,
        bool faces, bool edges, int ghostLayers, const std::map<EntityKind, index> &generatedIdStarts)
    {
        const auto communication = source.PreparationCommunication();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !communication || communication->ChecksCollectivePreflight());
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(ghostLayers > 0, "topology finalization requires positive default cell layers"); });
        auto owned = NativeMeshTopology::Build(NativeMeshAssembler::CaptureOwned(source, mpi), mpi, faces, edges, generatedIdStarts);
        GhostSpec recipe;
        owned.LocalStage(mpi, [&]
                         { recipe = ExtendNativeMeshGhostSpecForTopology(owned, faces, edges, ghostLayers); });
        owned.ghostSpec = std::move(recipe);
        auto result = AssembleNativeMesh(mpi, std::move(owned), ghostLayers);
        if (!result->Requirements().faces)
            return result;
        std::vector<WorkerRowRanges> parts;
        for (int part = 0; part < source.NLocalParts(); ++part)
            parts.push_back(source.LocalPartRanges(part));
        return result->WithLocalPartitionRanges(std::move(parts));
    }
}
