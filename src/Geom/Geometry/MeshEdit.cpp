#include "DNDS/MPI/Failure.hpp"
#include "Geom/Geometry/MeshEdit.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"

namespace DNDS::Geom
{
    namespace
    {
        constexpr unsigned Coordinates = 1, CellMetadata = 2, NeighborOrder = 4;

        template <class T>
        Runtime::Array<T> CloneArray(const Runtime::Array<T> &source)
        {
            Runtime::Array<T> result(source.Layout());
            auto destination = result.Storage();
            Runtime::Copy(source.Storage(), destination).Wait();
            return result;
        }

        template <class T>
        Runtime::ArrayPair<T> ClonePair(const Runtime::ArrayPair<T> &source)
        {
            return {CloneArray(source.owned), CloneArray(source.ghost)};
        }

        template <class T>
        void Refresh(Runtime::ArrayPair<T> &pair, const EntitySpace &space, const Runtime::ElementSchema &schema, const MPIInfo &mpi)
        {
            EntityGhostMapping route(space);
            auto ghost = route.TransferRows(space, pair.owned, schema);
            FieldTransferDetail::LocalStage(mpi, space.Communication(), [&]
                                            {
                if (!pair.owned.Layout().IsVariable())
                    ghost = Runtime::Array<T>::FromStorage(Runtime::RowLayout::Dense(ghost.RowCount(), pair.owned.Layout().Width()), ghost.Storage());
                pair.ghost = std::move(ghost); });
        }

    }

    void MeshEdit::Check() const
    {
        DNDS_check_throw_info(!Closed(), "mesh edit scope is closed or moved");
        _stamp.Validate();
    }
    void MeshEdit::CheckRow(index row, index count)
    {
        DNDS_check_throw_info(row >= 0 && row < count, "mesh edit row is outside owned entities");
    }

    MeshEdit MeshEdit::Begin(std::shared_ptr<UnstructuredMesh> source, int ghostLayers)
    {
        // Transaction ownership, clone submission and baseline list appends are
        // serialized by the edit controller; setters are single-row operations.
        DNDS_check_throw_info(source, "mesh edit requires a source owner");
        MeshEdit result;
        result._source = std::move(source);
        auto &mesh = *result._source;
        const auto communication = mesh.PreparationCommunication();
        Runtime::CommunicationDetail::AgreePolicy(mesh.mpi.comm, !communication || communication->ChecksCollectivePreflight());
        FieldTransferDetail::LocalStage(mesh.mpi, communication, [&]
                                        {
            DNDS_check_throw_info(mesh.HasNativeState(), "mesh edit requires native authority; explicitly publish/import a legacy mesh before editing");
            DNDS_check_throw_info(ghostLayers >= 1, "mesh edit requires positive ghost_layers");
            result._requirements = mesh.CurrentNativeRequirements();
            result._sourceRead = std::make_unique<MeshReadAccess>(mesh.AcquireRead(result._requirements));
            result._stamp = result._sourceRead->Stamp(); });
        FieldTransferDetail::LocalStage(mesh.mpi, communication, [&]
                                        {
            const auto &state = mesh._nativeState->Snapshot(Runtime::Device::Host()).Live();
            auto &data = result._nativeCandidate.emplace();
            data.identity = {mesh._nativeState->MeshNamespace(), mesh._nativeState->IdHighWater()};
            data.dimension = state.semantics.dim;
            data.audit = mesh.CurrentNativeState()->AuditOptions();
            data.periodic = state.semantics.isPeriodic;
            data.periodicInfo = state.semantics.periodicInfo;
            data.relations = state.relations;
            data.coords = ClonePair(*state.coords);
            data.cellInfo = ClonePair(*state.cellInfo);
            data.nodeWallDist = state.nodeWallDist;
            data.bndInfo = state.bndInfo;
            data.faceInfo = state.faceInfo;
            data.edgeInfo = state.edgeInfo;
            data.cellNodesPbi = state.cellNodesPbi;
            data.bndNodesPbi = state.bndNodesPbi;
            data.cellFacesPbi = state.cellFacesPbi;
            data.faceNodesPbi = state.faceNodesPbi;
            data.cellEdgesPbi = state.cellEdgesPbi;
            data.edgeNodesPbi = state.edgeNodesPbi;
            const auto view = result._sourceRead->HostView();
            for (index row = 0; row < view.NumCell(); ++row)
            {
                const auto neighbors = view.CellCells(row);
                result._neighbors.emplace_back(neighbors.begin(), neighbors.end());
                result._cellTypes.push_back(view.GetCellInfo(row).getElemType());
            } });
        return result;
    }

    index MeshEdit::NumNode() const
    {
        Check();
        return index(_nativeCandidate->coords->owned.RowCount());
    }
    index MeshEdit::NumCell() const
    {
        Check();
        return index(_nativeCandidate->cellInfo->owned.RowCount());
    }
    tPoint MeshEdit::Coord(index row) const
    {
        Check();
        CheckRow(row, NumNode());
        auto read = _nativeCandidate->coords->owned.Read();
        return tPoint{read.View()(size_t(row), 0), read.View()(size_t(row), 1), read.View()(size_t(row), 2)};
    }
    void MeshEdit::SetCoord(index row, const tPoint &value)
    {
        Check();
        CheckRow(row, NumNode());
        auto write = _nativeCandidate->coords->owned.Write();
        for (size_t component = 0; component < 3; ++component)
            write.View()(size_t(row), component) = value[component];
        _changes |= Coordinates;
    }
    ElemInfo MeshEdit::CellInfo(index row) const
    {
        Check();
        CheckRow(row, NumCell());
        return _nativeCandidate->cellInfo->owned.Read().View()(size_t(row), 0);
    }
    void MeshEdit::SetCellInfo(index row, const ElemInfo &value)
    {
        Check();
        CheckRow(row, NumCell());
        _nativeCandidate->cellInfo->owned.Write().View()(size_t(row), 0) = value;
        _changes |= CellMetadata;
    }
    std::vector<index> MeshEdit::CellNeighbors(index row) const
    {
        Check();
        CheckRow(row, NumCell());
        auto read = _nativeCandidate->relations[size_t(MeshRelation::CellCells)]->owned.Read();
        const auto values = read.View()[size_t(row)];
        return {values.begin(), values.end()};
    }
    void MeshEdit::SetCellNeighbor(index row, rowsize slot, index value)
    {
        Check();
        CheckRow(row, NumCell());
        auto &array = _nativeCandidate->relations[size_t(MeshRelation::CellCells)]->owned;
        DNDS_check_throw_info(slot >= 0 && size_t(slot) < array.Layout().RowSize(size_t(row)), "mesh neighbor slot outside row");
        if (!(_changes & NeighborOrder))
            array = CloneArray(array);
        array.Write().View()(size_t(row), size_t(slot)) = value;
        _changes |= NeighborOrder;
    }

    void MeshEdit::Commit()
    {
        // This validates/refreshes one transaction and publishes it atomically.
        // Access getters and CSR RowBuilder mutation deliberately stay outside
        // worker teams; distributed refresh is a controller-only sequence.
        // A moved-from scope has no communicator and is a caller error. A
        // closed/aborted scope still retains its source: agree that failure
        // before any peer enters validation or publication collectives.
        DNDS_check_throw_info(_source, "cannot commit a moved-from mesh edit");
        auto &mesh = *_source;
        const auto communication = mesh.PreparationCommunication();
        Runtime::CommunicationDetail::AgreePolicy(mesh.mpi.comm, !communication || communication->ChecksCollectivePreflight());
        FieldTransferDetail::LocalStage(mesh.mpi, communication, [&]
                                        {
            Check();
            for (index node = 0; node < NumNode(); ++node)
                DNDS_check_throw_info(Coord(node).allFinite(), "mesh edit coordinates must be finite");
            for (index cell = 0; cell < NumCell(); ++cell)
            {
                DNDS_check_throw_info(CellInfo(cell).getElemType() == _cellTypes[size_t(cell)],
                                      "cell metadata edit cannot change element type");
                auto before = _neighbors[size_t(cell)], after = CellNeighbors(cell);
                std::sort(before.begin(), before.end());
                std::sort(after.begin(), after.end());
                DNDS_check_throw_info(before == after, "neighbor-order edit must preserve the exact multiset including sentinels");
            } });
        unsigned changes = _changes;
        MPI_Allreduce(MPI_IN_PLACE, &changes, 1, MPI_UNSIGNED, MPI_BOR, mesh.mpi.comm);
        auto requirements = _requirements;
        if (changes & Coordinates)
            requirements.wallDistance = false;
        std::shared_ptr<UnstructuredMeshState> prepared;
        auto &data = *_nativeCandidate;
        if (changes & Coordinates)
        {
            Refresh(*data.coords, mesh.Entities(EntityKind::Node), Runtime::ElementSchema::Arithmetic<real>(), mesh.mpi);
            data.nodeWallDist.reset();
        }
        if (changes & CellMetadata)
        {
            const auto schema = Runtime::ElementSchema::For<ElemInfo>();
            Refresh(*data.cellInfo, mesh.Entities(EntityKind::Cell), schema, mesh.mpi);
        }
        if (changes & NeighborOrder)
        {
            const auto cells = mesh.Entities(EntityKind::Cell);
            auto &pair = *data.relations[size_t(MeshRelation::CellCells)];
            std::optional<Runtime::Array<index>> globals;
            FieldTransferDetail::LocalStage(mesh.mpi, communication, [&]
                                            {
                MeshAdjacencyRead read(Adj::Cell2Cell, cells, cells, Adj_PointToLocal, pair.owned, &pair.ghost);
                Runtime::RowBuilder rows(size_t(cells.NumOwned()), sizeof(index));
                for (index row = 0; row < cells.NumOwned(); ++row)
                {
                    rows.ResizeRow(size_t(row), read.RowSize(row));
                    auto output = rows.Row<index>(size_t(row));
                    for (size_t slot = 0; slot < output.size; ++slot)
                    {
                        const auto entry = read.ReadEntry(row, slot);
                        output[slot] = entry.absent ? UnInitIndex : entry.global;
                    }
                }
                globals.emplace(Runtime::Array<index>::FromSharedStorage(rows.Commit())); });
            EntityGhostMapping route(cells);
            auto ghost = route.TransferRows(cells, *globals, Runtime::ElementSchema::Arithmetic<index>());
            FieldTransferDetail::LocalStage(mesh.mpi, communication, [&]
                                            {
                auto read = ghost.Read();
                Runtime::RowBuilder rows(ghost.RowCount(), sizeof(index));
                for (size_t row = 0; row < ghost.RowCount(); ++row)
                {
                    rows.ResizeRow(row, read.View().RowSize(row));
                    auto output = rows.Row<index>(row);
                    for (size_t slot = 0; slot < output.size; ++slot)
                    {
                        const auto global = read.View()(row, slot);
                        if (global == UnInitIndex) output[slot] = UnInitIndex;
                        else if (auto local = cells.GlobalToLocal(global)) output[slot] = *local;
                        else
                        {
                            output[slot] = -1 - global;
                            DNDS_check_throw_info(output[slot] != UnInitIndex, "edited neighbor target collides with absent sentinel");
                        }
                    }
                }
                pair.ghost = Runtime::Array<index>::FromSharedStorage(rows.Commit()); });
        }
        FieldTransferDetail::LocalStage(mesh.mpi, communication, [&]
                                        {
            prepared = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState(*mesh._nativeState));
            // Only the explicit historical facade key is an authority alias.
            // Native sorted/set variants keep their own unchanged ordering.
            if ((changes & NeighborOrder) && prepared->_namedRelations.count(Adj::Cell2Cell))
            {
                auto &relation = prepared->_namedRelations.at(Adj::Cell2Cell);
                relation.data = {*data.relations[size_t(MeshRelation::CellCells)], Adj_PointToLocal};
                relation.owned = std::make_shared<Runtime::Array<index>::ReadLease>(relation.data.rows.owned.Read());
                relation.ghost = std::make_shared<Runtime::Array<index>::ReadLease>(relation.data.rows.ghost.Read());
            }
            prepared->_requirements = requirements;
            prepared->_placements.clear();
            prepared->_placements.emplace(-1, UnstructuredMeshSnapshot::FromNative(data, requirements));
            prepared->_hostRead = std::make_shared<MeshReadAccess>(prepared->_placements.at(-1).AcquireRead());
            prepared->_hostView = prepared->_hostRead->HostView();
            prepared->_versions.RebindPrepared(mesh.versions); });
        _closed = true;
        _sourceRead.reset();
        mesh.CheckCollectiveMutationAvailable(!communication || communication->ChecksCollectivePreflight());
        FieldTransferDetail::LocalStage(mesh.mpi, communication, [&]
                                        { _stamp.Validate(); });
        auto mask = Runtime::Change::Allocation;
        if (changes & Coordinates)
            mask = mask | Runtime::Change::Geometry | Runtime::Change::Values;
        if (changes & CellMetadata)
            mask = mask | Runtime::Change::Topology | Runtime::Change::Values | Runtime::Change::Interface;
        if (changes & NeighborOrder)
            mask = mask | Runtime::Change::Layout;
        // This publishes an independent current-state candidate. It deliberately
        // never enters or reads the source's detached legacy edit workspace.
        MeshMutationToken mutation(&mesh, mesh.versions, mask);
        mutation.Commit();
        try
        {
            // Owner containers were allocated before mutation. Only stamp and
            // shared-owner assignment remains after the generation commit.
            FieldTransferDetail::LocalStage(mesh.mpi, communication, [&]
                                            { prepared->BindPreparedEpoch(mesh.versions); });
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("mesh edit failed");
            mesh._nativeState.reset();
            mesh._nativeSmoothing.reset();
            auto failed = mesh.versions.BeginMutation(Runtime::Change::All);
            throw;
        }
        mesh._nativeState = std::move(prepared);
        mesh._nativeSmoothing.reset();
        _nativeCandidate.reset();
    }
    void MeshEdit::Abort()
    {
        _closed = true;
        _sourceRead.reset();
        _nativeCandidate.reset();
    }
}
