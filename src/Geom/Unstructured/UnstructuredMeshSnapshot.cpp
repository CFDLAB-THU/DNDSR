#include "Geom/Unstructured/UnstructuredMeshSnapshot.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include <cstring>

namespace DNDS::Geom
{
    void UnstructuredMeshSnapshot::VisitAllocations(const std::function<void(uint64_t, size_t, Runtime::Device)> &visitor) const
    {
        const auto &state = Live();
        const auto pair = [&](const auto &values)
        {
            if (!values)
                return;
            values->owned.SharedStorage()->VisitAllocations(visitor);
            values->ghost.SharedStorage()->VisitAllocations(visitor);
        };
        for (const auto &adjacency : state.relations)
            pair(adjacency);
        pair(state.coords);
        pair(state.nodeWallDist);
        pair(state.cellInfo);
        pair(state.bndInfo);
        pair(state.faceInfo);
        pair(state.edgeInfo);
        pair(state.cellNodesPbi);
        pair(state.bndNodesPbi);
        pair(state.cellFacesPbi);
        pair(state.faceNodesPbi);
        pair(state.cellEdgesPbi);
        pair(state.edgeNodesPbi);
    }
    UnstructuredMeshSnapshot UnstructuredMeshSnapshot::FromNative(NativeMeshData data, MeshReadRequirements requirements)
    {
        DNDS_check_throw_info(data.namedRelations.empty(), "named setup relations require UnstructuredMeshState publication");
        DNDS_check_throw_info(data.dimension >= 1 && data.dimension <= 3, "native mesh dimension must be one, two or three");
        auto state = std::make_shared<State>();
        state->source = state->versions.Stamp();
        state->device = Runtime::Device::Host();
        state->audit = data.audit;
        state->semantics.dim = data.dimension;
        state->semantics.isPeriodic = data.periodic;
        state->semantics.periodicInfo = std::move(data.periodicInfo);
        state->semantics.hasNodeWallDist = bool(data.nodeWallDist);
        // Presence is a public query promise, so every adopted optional value
        // must belong to an enabled, fully validated publication group.
        DNDS_check_throw_info(requirements.wallDistance || !data.nodeWallDist,
                              "native wall distance must be explicitly included in publication requirements");
        DNDS_check_throw_info(requirements.faces || !data.faceInfo,
                              "native face information requires the facial publication group");
        DNDS_check_throw_info(requirements.edges || !data.edgeInfo,
                              "native edge information requires the edge publication group");
        DNDS_check_throw_info(data.periodic || (!data.cellNodesPbi && !data.bndNodesPbi &&
                                                !data.cellFacesPbi && !data.faceNodesPbi && !data.cellEdgesPbi && !data.edgeNodesPbi),
                              "native nonperiodic mesh cannot expose periodic incidence companions");
        DNDS_check_throw_info(requirements.faces || (!data.cellFacesPbi && !data.faceNodesPbi),
                              "native periodic facial companions require the facial publication group");
        DNDS_check_throw_info(requirements.edges || (!data.cellEdgesPbi && !data.edgeNodesPbi),
                              "native periodic edge companions require the edge publication group");
        auto checkPair = [](const auto &pair)
        {
            if (pair)
                DNDS_check_throw_info(pair->owned.GetDevice() == Runtime::Device::Host() &&
                                          pair->ghost.GetDevice() == Runtime::Device::Host(),
                                      "native mesh construction requires explicit host buffers");
        };
        for (size_t i = 0; i < data.relations.size(); ++i)
        {
            checkPair(data.relations[i]);
            state->semantics.relationPresent[i] = bool(data.relations[i]);
            state->semantics.relationStates[i] = data.relations[i] ? Adj_PointToLocal : Adj_Unknown;
        }
        auto required = [&](MeshRelation relation)
        { DNDS_check_throw_info(data.relations[size_t(relation)], "native mesh required relation is absent"); };
        DNDS_check_throw_info(!requirements.allFaceCellsLocal || requirements.faces,
                              "all-face cell closure requires facial relations");
        auto optionalGroup = [&](bool enabled, std::initializer_list<MeshRelation> relations)
        {
            for (auto relation : relations)
                DNDS_check_throw_info(enabled || !data.relations[size_t(relation)],
                                      "native optional relation must be explicitly included in publication requirements");
        };
        optionalGroup(requirements.faces, {MeshRelation::CellFaces, MeshRelation::FaceNodes, MeshRelation::FaceCells,
                                           MeshRelation::FaceBoundaries, MeshRelation::BoundaryFaces});
        optionalGroup(requirements.inverse, {MeshRelation::NodeCells, MeshRelation::NodeBoundaries});
        optionalGroup(requirements.edges, {MeshRelation::CellEdges, MeshRelation::EdgeNodes, MeshRelation::EdgeCells});
        optionalGroup(requirements.cellFaceNeighbors, {MeshRelation::CellFaceCells});
        required(MeshRelation::CellNodes);
        required(MeshRelation::BoundaryNodes);
        required(MeshRelation::BoundaryCells);
        required(MeshRelation::CellCells);
        DNDS_check_throw_info(data.coords && data.cellInfo && data.bndInfo, "native primary mesh values are missing");
        DNDS_check_throw_info(data.dimension != 1 || (!requirements.faces && !requirements.edges && !requirements.cellFaceNeighbors &&
                                                      data.bndInfo->owned.RowCount() == 0 && data.bndInfo->ghost.RowCount() == 0),
                              "native one-dimensional meshes support primary curves without point boundaries or facial/edge topology");
        if (requirements.faces)
        {
            for (auto relation : {MeshRelation::CellFaces, MeshRelation::FaceNodes, MeshRelation::FaceCells,
                                  MeshRelation::FaceBoundaries, MeshRelation::BoundaryFaces})
                required(relation);
            DNDS_check_throw_info(data.faceInfo, "native face information is missing");
        }
        if (requirements.inverse)
        {
            required(MeshRelation::NodeCells);
            required(MeshRelation::NodeBoundaries);
        }
        if (requirements.edges)
        {
            required(MeshRelation::CellEdges);
            required(MeshRelation::EdgeNodes);
            required(MeshRelation::EdgeCells);
            DNDS_check_throw_info(data.edgeInfo, "native edge information is missing");
        }
        if (requirements.cellFaceNeighbors)
            required(MeshRelation::CellFaceCells);
        DNDS_check_throw_info(!requirements.wallDistance || data.nodeWallDist, "native wall distance is missing");
        if (data.periodic)
            DNDS_check_throw_info(data.cellNodesPbi && data.bndNodesPbi &&
                                      (!requirements.faces || (data.cellFacesPbi && data.faceNodesPbi)) &&
                                      (!requirements.edges || (data.cellEdgesPbi && data.edgeNodesPbi)),
                                  "native periodic incidence companions are missing");
        // Physical incidence pairs promise two ordered slots. Normalize once at
        // publication, including empty ranks; named exact containment relations
        // remain separate. A new layout shares values and their access controller
        // and never requires mutable access to an already retained publication.
        auto cellPairTable = [](Runtime::Array<index> values)
        {
            const auto layout = values.Layout();
            if (!layout.RowCount() && (layout.IsVariable() || layout.Width() != 2))
                return Runtime::Array<index>::FromStorage(Runtime::RowLayout::Dense(0, 2, std::max(layout.Stride(), index(2))), values.Storage());
            if (!layout.IsVariable())
            {
                DNDS_check_throw_info(layout.Width() == 2, "physical cell incidence requires two ordered slots");
                return values;
            }
            for (index row = 0; row < layout.RowCount(); ++row)
            {
                DNDS_check_throw_info(layout.RowSize(row) == 2, "physical cell incidence requires two ordered slots");
                if (layout.IsRagged())
                    DNDS_check_throw_info(layout.Offset(row) == CheckedSize::Multiply(row, index(2)),
                                          "physical cell incidence CSR is not tightly contiguous");
            }
            if (layout.IsRagged())
                DNDS_check_throw_info(layout.Elements() == size_t(CheckedSize::Multiply(layout.RowCount(), index(2))),
                                      "physical cell incidence CSR terminal offset is invalid");
            const auto stride = layout.IsPadded() ? layout.Stride() : index(2);
            DNDS_check_throw_info(!layout.RowCount() || stride >= 2, "physical cell incidence stride is too small");
            return Runtime::Array<index>::FromStorage(Runtime::RowLayout::Dense(layout.RowCount(), 2, std::max(stride, index(2))), values.Storage());
        };
        for (auto relation : {MeshRelation::BoundaryCells, MeshRelation::FaceCells})
            if (auto &pair = data.relations[size_t(relation)]; pair)
            {
                pair->owned = cellPairTable(std::move(pair->owned));
                pair->ghost = cellPairTable(std::move(pair->ghost));
            }
        state->relations = std::move(data.relations);
        const bool trackAllocations = Runtime::AllocationDiagnostics::Enabled();
        auto adopt = [&](auto &destination, auto &source, const char *name)
        {
            checkPair(source);
            if (trackAllocations && source)
            {
                const auto path = std::string("Geom/mesh/") + name;
                source->owned.SetName(path + "/owned", DNDS_ALLOCATION_SITE());
                source->ghost.SetName(path + "/ghost", DNDS_ALLOCATION_SITE());
            }
            destination = std::move(source);
        };
        adopt(state->coords, data.coords, "coordinates");
        adopt(state->nodeWallDist, data.nodeWallDist, "nodeWallDistance");
        adopt(state->cellInfo, data.cellInfo, "cellInfo");
        adopt(state->bndInfo, data.bndInfo, "boundaryInfo");
        adopt(state->faceInfo, data.faceInfo, "faceInfo");
        adopt(state->edgeInfo, data.edgeInfo, "edgeInfo");
        adopt(state->cellNodesPbi, data.cellNodesPbi, "cellNodesPbi");
        adopt(state->bndNodesPbi, data.bndNodesPbi, "boundaryNodesPbi");
        adopt(state->cellFacesPbi, data.cellFacesPbi, "cellFacesPbi");
        adopt(state->faceNodesPbi, data.faceNodesPbi, "faceNodesPbi");
        adopt(state->cellEdgesPbi, data.cellEdgesPbi, "cellEdgesPbi");
        adopt(state->edgeNodesPbi, data.edgeNodesPbi, "edgeNodesPbi");
        UnstructuredMeshSnapshot result(state);
        auto read = std::make_shared<MeshReadAccess>(result.AcquireRead());
        read->Validate(requirements, state->audit);
        state->immutableRead = std::move(read);
        return result;
    }

    template <class T>
    UnstructuredMeshSnapshot::Pair<T> UnstructuredMeshSnapshot::ClonePair(Runtime::PairReadView<T> source)
    {
        auto clone = [](Runtime::ArrayView<const T> view)
        {
            Runtime::RowLayout layout;
            if (view.offsets)
            {
                std::vector<size_t> lengths(view.rows);
                for (size_t i = 0; i < view.rows; ++i)
                    lengths[i] = view.offsets[i + 1] - view.offsets[i];
                layout = Runtime::RowLayout::Ragged(lengths);
            }
            else if (view.sizes)
                layout = Runtime::RowLayout::Padded(std::vector<size_t>(view.sizes, view.sizes + view.rows), view.stride);
            else
                layout = Runtime::RowLayout::Dense(view.rows, view.width, view.stride);
            Runtime::Array<T> result(layout);
            if (layout.Elements())
            {
                auto write = result.Write();
                std::memcpy(write.View().data, view.data, CheckedSize::Multiply(layout.Elements(), sizeof(T)));
            }
            return result;
        };
        return {clone(source.owned), clone(source.ghost)};
    }

    UnstructuredMeshSnapshot UnstructuredMeshSnapshot::Capture(UnstructuredMesh &source, MeshReadRequirements requirements)
    {
        return FromRead(source.AcquireRead(requirements));
    }
    UnstructuredMeshSnapshot UnstructuredMeshSnapshot::FromRead(MeshReadAccess read)
    {
        auto state = std::make_shared<State>();
        if (read._nativeOwner)
            state->audit = read._nativeOwner->AuditOptions();
        state->source = read.Stamp();
        state->device = Runtime::Device::Host();
        const auto &s = read._storage;
        // Copy semantic values only. Pointer bundles are assembled from private
        // native arrays on read; no source pointer may escape into this state.
        state->semantics.dim = s.dim;
        state->semantics.isPeriodic = s.isPeriodic;
        state->semantics.periodicInfo = s.periodicInfo;
        state->semantics.hasNodeWallDist = s.hasNodeWallDist;
        for (size_t i = 0; i < state->relations.size(); ++i)
        {
            state->semantics.relationStates[i] = s.relationStates[i];
            state->semantics.relationPresent[i] = s.relationPresent[i];
            if (s.relationPresent[i])
                state->relations[i] = ClonePair(s.relations[i]);
        }
        state->coords = ClonePair(s.coords);
        state->cellInfo = ClonePair(s.cellElemInfo);
        state->bndInfo = ClonePair(s.bndElemInfo);
        if (s.relationPresent[size_t(MeshRelation::FaceNodes)])
            state->faceInfo = ClonePair(s.faceElemInfo);
        if (s.relationPresent[size_t(MeshRelation::EdgeNodes)])
            state->edgeInfo = ClonePair(s.edgeElemInfo);
        if (s.hasNodeWallDist)
            state->nodeWallDist = ClonePair(s.nodeWallDist);
        if (s.isPeriodic)
        {
            state->cellNodesPbi = ClonePair(s.cell2nodePbi);
            state->bndNodesPbi = ClonePair(s.bnd2nodePbi);
            if (s.relationPresent[size_t(MeshRelation::FaceNodes)])
            {
                state->cellFacesPbi = ClonePair(s.cell2facePbi);
                state->faceNodesPbi = ClonePair(s.face2nodePbi);
            }
            if (s.relationPresent[size_t(MeshRelation::EdgeNodes)])
            {
                state->cellEdgesPbi = ClonePair(s.cell2edgePbi);
                state->edgeNodesPbi = ClonePair(s.edge2nodePbi);
            }
        }
        return UnstructuredMeshSnapshot(std::move(state));
    }

    template <class T>
    std::optional<UnstructuredMeshSnapshot::Pair<T>> UnstructuredMeshSnapshot::Place(const std::optional<Pair<T>> &source,
                                                                                     Runtime::Device device, Runtime::ExecutionContext context)
    {
        if (!source)
            return {};
        return Pair<T>{source->owned.To(device, context), source->ghost.To(device, context)};
    }

    UnstructuredMeshSnapshot UnstructuredMeshSnapshot::To(Runtime::Device device, Runtime::ExecutionContext context) const
    {
        Live();
        DNDS_check_throw_info(context.device == (device == Runtime::Device::Host() ? GetDevice() : device),
                              "snapshot placement context mismatch");
        auto next = std::make_shared<State>();
        next->source = _state->source;
        next->device = device;
        next->semantics = _state->semantics;
        next->audit = _state->audit;
        for (size_t i = 0; i < next->relations.size(); ++i)
            next->relations[i] = Place(_state->relations[i], device, context);
        next->coords = Place(_state->coords, device, context);
        next->nodeWallDist = Place(_state->nodeWallDist, device, context);
        next->cellInfo = Place(_state->cellInfo, device, context);
        next->bndInfo = Place(_state->bndInfo, device, context);
        next->faceInfo = Place(_state->faceInfo, device, context);
        next->edgeInfo = Place(_state->edgeInfo, device, context);
        next->cellNodesPbi = Place(_state->cellNodesPbi, device, context);
        next->bndNodesPbi = Place(_state->bndNodesPbi, device, context);
        next->cellFacesPbi = Place(_state->cellFacesPbi, device, context);
        next->faceNodesPbi = Place(_state->faceNodesPbi, device, context);
        next->cellEdgesPbi = Place(_state->cellEdgesPbi, device, context);
        next->edgeNodesPbi = Place(_state->edgeNodesPbi, device, context);
        return UnstructuredMeshSnapshot(std::move(next));
    }

    template <class T>
    Runtime::PairReadView<T> MeshReadAccess::CaptureNative(const Runtime::Array<T> &owned, const Runtime::Array<T> &ghost)
    {
        auto capture = [&](const auto &array)
        {
            auto hold = std::make_unique<ArrayHold<T>>(array, _context);
            auto view = hold->read.View();
            _holds.push_back(std::move(hold));
            return view;
        };
        return {capture(owned), capture(ghost)};
    }

    MeshReadAccess::MeshReadAccess(const UnstructuredMeshSnapshot &snapshot, Runtime::ExecutionContext context)
        : _meshRead(snapshot.Live().versions.Read()), _stamp(snapshot.Live().versions.Stamp()), _context(context)
    {
        DNDS_check_throw_info(context.device == snapshot.GetDevice() &&
                                  (!context.stream || context.stream->GetDevice() == context.device),
                              "native snapshot acquisition placement mismatch");
        const auto &s = *snapshot._state;
        _storage = s.semantics;
        auto capture = [&](const auto &field, auto &destination)
        {
            if (field)
                destination = CaptureNative(field->owned, field->ghost);
        };
        for (size_t i = 0; i < s.relations.size(); ++i)
            capture(s.relations[i], _storage.relations[i]);
        capture(s.coords, _storage.coords);
        capture(s.nodeWallDist, _storage.nodeWallDist);
        capture(s.cellInfo, _storage.cellElemInfo);
        capture(s.bndInfo, _storage.bndElemInfo);
        capture(s.faceInfo, _storage.faceElemInfo);
        capture(s.edgeInfo, _storage.edgeElemInfo);
        capture(s.cellNodesPbi, _storage.cell2nodePbi);
        capture(s.bndNodesPbi, _storage.bnd2nodePbi);
        capture(s.cellFacesPbi, _storage.cell2facePbi);
        capture(s.faceNodesPbi, _storage.face2nodePbi);
        capture(s.cellEdgesPbi, _storage.cell2edgePbi);
        capture(s.edgeNodesPbi, _storage.edge2nodePbi);
    }
    MeshReadAccess::MeshReadAccess(const UnstructuredMeshSnapshot &snapshot, Runtime::ExecutionContext context,
                                   Runtime::VersionTracker::ReadLease source, Runtime::VersionStamp stamp)
        : MeshReadAccess(snapshot, context)
    {
        // Adopt the source publication lease after all native buffers were
        // captured. Detached snapshot reads retain their independent identity.
        _sourceRead.emplace(std::move(source));
        _stamp = std::move(stamp);
    }
}
