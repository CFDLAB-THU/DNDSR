#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Refinement/NativeMeshExtraction.hpp"
#include "Geom/Unstructured/NativeMeshPreparation.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
#include "Geom/Compat/NativeMeshLegacyArchive.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Transition/MeshTransition.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Solver/Direct.hpp"
#include <algorithm>
#include <map>
namespace DNDS::Geom
{
    void UnstructuredMeshSerialRW::
        BuildSerialOut()
    {
        std::shared_ptr<const UnstructuredMeshState> source;
        FieldTransferDetail::LocalStage(mesh->getMPI(), [&]
                                        { source = mesh->CurrentNativeState(); });
        std::shared_ptr<NativeMeshSerialOutput> prepared;
        FieldTransferDetail::LocalStage(mesh->getMPI(), [&]
                                        { prepared = std::make_shared<NativeMeshSerialOutput>(); });
        *prepared = NativeMeshSerialOutput::Build(*source, mesh->getMPI(), mRank);
        // Assign only after successful collective gathering. A failed refresh
        // retains the previous descriptor; writers still validate its source.
        nativeSerialOutput = std::move(prepared);
        mode = SerialOutput;
        dataIsSerialIn = false;
        dataIsSerialOut = true;
    }
    bool UnstructuredMesh::IsO1()
    {
        using namespace Elem;
        int hasBad = 0;
        for (index iCell = 0; iCell < NumCellProc(); iCell++)
        {
            auto eType = GetCellInfo(iCell).getElemType();
            if (eType == ElemType::Line2 ||
                eType == ElemType::Tri3 ||
                eType == ElemType::Quad4 ||
                eType == ElemType::Tet4 ||
                eType == ElemType::Hex8 ||
                eType == ElemType::Prism6 ||
                eType == ElemType::Pyramid5)
            {
                continue;
            }
            else
            {
                hasBad = 1;
                break;
            }
        }
        int hasBadAll = 0;
        MPI::Allreduce(&hasBad, &hasBadAll, 1, MPI_INT, MPI_SUM, mpi.comm);
        return hasBadAll == 0;
    }
    bool UnstructuredMesh::IsO2()
    {
        using namespace Elem;
        int hasBad = 0;
        for (index iCell = 0; iCell < NumCellProc(); iCell++)
        {
            auto eType = GetCellInfo(iCell).getElemType();
            if (eType == ElemType::Line3 ||
                eType == ElemType::Tri6 ||
                eType == ElemType::Quad9 ||
                eType == ElemType::Tet10 ||
                eType == ElemType::Hex27 ||
                eType == ElemType::Prism18 ||
                eType == ElemType::Pyramid14)
            {
                continue;
            }
            else
            {
                hasBad = 1;
                break;
            }
        }
        int hasBadAll = 0;
        MPI::Allreduce(&hasBad, &hasBadAll, 1, MPI_INT, MPI_SUM, mpi.comm);
        return hasBadAll == 0;
    }
    void UnstructuredMesh::AssertOnN2CB()
    {

        auto requirements = MeshReadRequirements::Primary();
        requirements.inverse = true;
        auto read = AcquireRead(requirements);
        const auto view = read.HostView();
        auto cells = MeshAdjacencyRead::Current(*this, Adj::Node2Cell, requirements);
        auto boundaries = MeshAdjacencyRead::Current(*this, Adj::Node2Bnd, requirements);
        auto contains = [](auto row, index target)
        { return std::find(row.begin(), row.end(), target) != row.end(); };
        for (index node = 0; node < view.NumNodeProc(); ++node)
        {
            auto checkSupport = [&](const MeshAdjacencyRead &support, bool boundary)
            {
                for (size_t slot = 0; slot < support.RowSize(node); ++slot)
                {
                    const auto entry = support.ReadEntry(node, slot);
                    DNDS_check_throw_info(!entry.absent, "inverse support contains an absent entity");
                    if (entry.unavailable)
                        continue; // A partial node star is a valid publication.
                    const auto parent = support.Raw(node, slot);
                    const auto cone = boundary ? view.BoundaryNodes(parent) : view.CellNodes(parent);
                    DNDS_check_throw_info(contains(cone, node), "inverse support parent does not contain its source node");
                }
            };
            checkSupport(cells, false);
            checkSupport(boundaries, true);
        }
        auto checkOwnedCones = [&](index count, const MeshAdjacencyRead &support, bool boundary)
        {
            for (index parent = 0; parent < count; ++parent)
            {
                const auto cone = boundary ? view.BoundaryNodes(parent) : view.CellNodes(parent);
                for (auto node : cone)
                {
                    DNDS_check_throw_info(node >= 0 && node < view.NumNodeProc(), "owned cone node is unavailable");
                    bool found = false;
                    for (size_t slot = 0; slot < support.RowSize(node); ++slot)
                        found = found || support.Raw(node, slot) == parent;
                    DNDS_check_throw_info(found, "owned cone is missing from inverse node support");
                }
            }
        };
        checkOwnedCones(view.NumCell(), cells, false);
        checkOwnedCones(view.NumBnd(), boundaries, true);
        // Unused nodes may have empty support; no nonempty-star assertion.
        return;
    }
    void UnstructuredMesh::
        AssertOnFaces()
    {

        const auto requirements = MeshReadRequirements::CellFlux();
        auto read = AcquireRead(requirements);
        const auto view = read.HostView();
        // Typed scopes validate local/encoded-global targets against the
        // exact entity maps, including unavailable outer ghost references.
        auto faceCells = MeshAdjacencyRead::Current(*this, Adj::Face2Cell, requirements);
        auto cellFaces = MeshAdjacencyRead::Current(*this, Adj::Cell2Face, requirements);
        // Small adjacency rows are multisets of incidences, not trees. Sort
        // compact vectors once; duplicates retain self-periodic multiplicity.
        std::vector<std::vector<index>> incoming(size_t(view.NumCell()));
        for (index iCell = 0; iCell < view.NumCell(); ++iCell)
            incoming[iCell].reserve(view.CellFaces(iCell).size());
        for (index face = 0; face < view.NumFaceProc(); ++face)
        {
            const auto support = view.FaceCells(face);
            DNDS_check_throw_info(support.size() == 2, "face assertion requires two ordered cell sides");
            const bool external = FaceIDIsExternalBC(view.GetFaceInfo(face).zone);
            DNDS_check_throw_info(!faceCells.ReadEntry(face, 0).absent &&
                                      faceCells.ReadEntry(face, 1).absent == external,
                                  "face boundary zone disagrees with its ordered cell-side sentinels");
            for (size_t side = 0; side < support.size(); ++side)
            {
                const auto entry = faceCells.ReadEntry(face, side);
                if (entry.absent || entry.unavailable)
                    continue; // Outer faces need not have complete local cell closure.
                const auto cell = support[side];
                if (cell < view.NumCell())
                    incoming[cell].push_back(face);
            }
            if (face < view.NumFace())
            {
                for (index side = 0; side < index(support.size()); ++side)
                {
                    const auto cell = support[side];
                    if (cell == UnInitIndex || (side && cell == support[0]))
                        continue;
                    const auto count = std::count(support.begin(), support.end(), cell);
                    DNDS_check_throw_info(cell >= 0 && cell < view.NumCellProc(), "owned face cell is unavailable");
                    const auto reverse = view.CellFaces(cell);
                    const auto occurrences = size_t(std::count(reverse.begin(), reverse.end(), face));
                    DNDS_check_throw_info(occurrences == count,
                                          "owned face is missing or duplicated in an adjacent cell incidence");
                }
            }
        }
        for (index cell = 0; cell < view.NumCell(); ++cell)
        {
            std::vector<index> outgoing;
            outgoing.reserve(view.CellFaces(cell).size());
            for (auto face : view.CellFaces(cell))
            {
                DNDS_check_throw_info(face >= 0 && face < view.NumFaceProc(), "owned cell face is unavailable");
                outgoing.push_back(face);
            }
            // Count ordered incidences, not unique faces: a self-periodic
            // face appears twice on both sides of this equality.
            std::sort(outgoing.begin(), outgoing.end());
            std::sort(incoming[cell].begin(), incoming[cell].end());
            DNDS_check_throw_info(outgoing == incoming[cell],
                                  "owned cell and face incidences disagree in identity or multiplicity");
        }
        // An owned face may touch a ghost cell whose OTHER face slots are
        // unavailable. Do not demand completeness of those outer rows.
        return;
    }
    std::shared_ptr<UnstructuredMesh> UnstructuredMesh::ExtractBoundary() const
    {
        const auto communication = PreparationCommunication();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !communication || communication->ChecksCollectivePreflight());
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native boundary extraction requires a communicator");
        std::shared_ptr<const UnstructuredMeshState> source;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { source = CurrentNativeState(); });
        return FromNative(mpi, NativeMeshExtraction::ExtractBoundary(*source, mpi).state);
    }
    void UnstructuredMesh::ConstructBndMesh(UnstructuredMesh &bMesh)
    {
        const auto communication = PreparationCommunication();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !communication || communication->ChecksCollectivePreflight());
        std::shared_ptr<const UnstructuredMeshState> source;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(HasNativeState(), "boundary extraction requires native mesh authority");
                                          source = CurrentNativeState(); });
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            int communicator = MPI_UNEQUAL;
            if (bMesh.mpi.comm != MPI_COMM_NULL) MPI_Comm_compare(mpi.comm, bMesh.mpi.comm, &communicator);
            DNDS_check_throw_info(!versions.SameController(bMesh.versions) && bMesh.getDim() == getDim() - 1 &&
                                      (communicator == MPI_IDENT || communicator == MPI_CONGRUENT) && bMesh.versions.CanMutate(),
                                  "native boundary destination must be independent, idle and dimension/communicator compatible"); });
        auto extracted = NativeMeshExtraction::ExtractBoundary(*source, mpi);
        const auto state = std::move(extracted.state);
        const auto nodes = state->Entities(EntityKind::Node);
        const auto &ancestry = state->Provenance()->sourceIds.at(EntityKind::Node);
        EntityGhostMapping route(nodes);
        Runtime::ArrayPair<index> parents{ancestry, route.TransferRows(nodes, ancestry, Runtime::ElementSchema::Arithmetic<index>())};
        std::vector<index> sourceToBoundary, boundaryToSource, cellParents;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            std::unordered_map<index, index> sourceRows;
            sourceRows.reserve(size_t(NumNodeProc()));
            const auto nodeIds = source->ReadOriginalEntityIds(EntityKind::Node);
            const auto nodeIdView = nodeIds->View();
            for (index row = 0; row < NumNodeProc(); ++row)
                sourceRows.emplace(nodeIdView(row, 0), row);
            sourceToBoundary.assign(size_t(NumNodeProc()), UnInitIndex);
            boundaryToSource.assign(size_t(nodes.NumLocal()), UnInitIndex);
            auto read = parents.ReadCombined();
            for (index row = 0; row < nodes.NumLocal(); ++row)
            {
                auto found = sourceRows.find(read.View()(size_t(row), 0));
                if (found != sourceRows.end())
                {
                    boundaryToSource[size_t(row)] = found->second;
                    sourceToBoundary[size_t(found->second)] = row;
                }
            }
            cellParents.resize(size_t(state->NumOwned(EntityKind::Cell)));
            sourceRows.clear();
            sourceRows.reserve(size_t(NumBnd()));
            const auto boundaryIds = source->ReadOriginalEntityIds(EntityKind::Bnd);
            const auto boundaryIdView = boundaryIds->View();
            for (index row = 0; row < NumBnd(); ++row)
                sourceRows.emplace(boundaryIdView(row, 0), row);
            auto cellAncestry = state->Provenance()->sourceIds.at(EntityKind::Cell).Read();
            for (size_t row = 0; row < cellParents.size(); ++row)
            {
                auto found = sourceRows.find(cellAncestry.View()(row, 0));
                DNDS_check_throw_info(found != sourceRows.end(), "extracted boundary cell ancestry is not owned by its source");
                cellParents[row] = found->second;
            } });
        // These vectors are setup-only compatibility caches. They do not
        // change the source query authority or invalidate retained reads.
        MeshMutationToken destination(&bMesh, bMesh.versions, Runtime::Change::All);
        destination.Commit();
        bMesh._entityIdentity = nodes._identity;
        bMesh.isPeriodic = state->BorrowHostView().IsPeriodic();
        bMesh.periodicInfo = state->BorrowHostView().Periodic();
        bMesh._nativeState = state->BindEpoch(bMesh.versions);
        bMesh._nativeSmoothing.reset();
        bMesh.node2parentNode.swap(boundaryToSource);
        bMesh.cell2parentCell.swap(cellParents);
        bMesh.node2bndNode.clear();
        node2bndNode.swap(sourceToBoundary);
    }
    void UnstructuredMesh::ObtainSymmetricSymbolicFactorization(Direct::SerialSymLUStructure &symLU, Direct::DirectPrecControl control) const
    {
        if (control.useDirectPrec)
        {
            std::vector<WorkerRowRanges> workers;
            std::vector<index> partitions{0};
            for (int part = 0; part < NLocalParts(); ++part)
            {
                workers.push_back(LocalPartRanges(part));
                partitions.push_back(partitions.back() + workers.back().Size());
            }
            DNDS_check_throw_info(symLU.Num() == NumCell() && LocalCellNeighbors().size() == size_t(NumCell()),
                                  "symbolic factorization requires current local cell-neighbor setup");
            symLU.ObtainSymmetricSymbolicFactorization(
                LocalCellNeighbors(),
                std::move(partitions),
                control.getILUCode(), true, workers);
        }
    }
    void UnstructuredMesh::WriteSerialize(Serializer::SerializerBaseSSP serializer, const std::string &name)
    {
        WriteCurrentPrimaryMeshSnapshot(*this, std::move(serializer), name, 1, CurrentMeshCheckpointPolicy::DropAuxiliary);
    }
    void UnstructuredMesh::ReadSerialize(Serializer::SerializerBaseSSP serializer, const std::string &name)
    {
        CheckCollectiveMutationAvailable();
        auto state = ReadCompatibleNativeMeshArchive(mpi, std::move(serializer), name);
        ReplaceNativeForSetup(std::move(state));
    }
    void UnstructuredMesh::ReadSerializeAndDistribute(Serializer::SerializerBaseSSP serializer, const std::string &name, const PartitionOptions &options)
    {
        CheckCollectiveMutationAvailable();
        auto owned = ReadCompatibleNativeMeshArchiveOwned(mpi, std::move(serializer), name, Serializer::ArrayGlobalOffset_EvenSplit);
        NativePartitionOptions partition{options.metisType, options.metisUfactor, options.metisSeed, options.edgeWeightMethod, options.metisNcuts, 1};
        owned = NativeMeshPreparation::PartitionOwned(owned, mpi, partition).owned;
        if (!owned.requirements.faces)
        {
            owned = NativeMeshTopology::Build(std::move(owned), mpi);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { owned.ghostSpec = ExtendNativeMeshGhostSpecForTopology(owned, true, false); });
        }
        ReplaceNativeForSetup(AssembleNativeMesh(mpi, std::move(owned), 1));
    }
    void UnstructuredMesh::ReorderLocalCells(int parts, int inner, int ghostLayers)
    {
        const auto communication = PreparationCommunication();
        const bool checks = !communication || communication->ChecksCollectivePreflight();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, checks);
        CheckCollectiveMutationAvailable(checks);
        std::shared_ptr<const UnstructuredMeshState> source;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { source = CurrentNativeState(); });
        ReplaceNativeForSetup(NativeMeshTransition::ReorderLocalCells(*source, mpi, parts, inner, ghostLayers).state);
    }
    void UnstructuredMesh::ReorderLocalCells(const LocalCellOrderingSettings &settings, int ghostLayers)
    {
        const auto communication = PreparationCommunication();
        const bool checks = !communication || communication->ChecksCollectivePreflight();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, checks);
        CheckCollectiveMutationAvailable(checks);
        ReplaceNativeForSetup(NativeMeshTransition::ReorderLocalCells(*this, settings, ghostLayers).state);
    }
}
