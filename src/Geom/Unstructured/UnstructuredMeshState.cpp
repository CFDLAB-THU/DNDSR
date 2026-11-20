#include "Geom/Unstructured/UnstructuredMeshState.hpp"
#include "Geom/Access/RelationAccess.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include <set>
#include <unordered_set>

namespace DNDS::Geom
{
    std::shared_ptr<Runtime::CommunicationContext> UnstructuredMeshState::PreparationCommunication() const
    {
        const auto cell = _spaces.find(EntityKind::Cell);
        return cell == _spaces.end() ? nullptr : cell->second.identity.Communication();
    }
    size_t UnstructuredMeshState::ArrayBytes() const
    {
        Validate();
        size_t bytes = 0;
        std::set<uint64_t> allocations;
        auto visit = [&](uint64_t identity, size_t capacity, Runtime::Device)
        {
            if (allocations.insert(identity).second)
                bytes = CheckedSize::Add(bytes, capacity);
        };
        auto array = [&](const auto &values)
        { values.SharedStorage()->VisitAllocations(visit); };
        auto pair = [&](const auto &values)
        {
            array(values.owned);
            array(values.ghost);
        };
        for (const auto &[key, snapshot] : _placements)
            snapshot.VisitAllocations(visit);
        for (const auto &[key, relation] : _namedRelations)
            pair(relation.data.rows);
        for (const auto &[kind, ids] : _originalIds)
            array(ids);
        if (_provenance)
            for (const auto &[kind, ids] : _provenance->sourceIds)
                array(ids);
        return bytes;
    }

    namespace
    {
        template <class F>
        void PublicationStage(const MPIInfo &mpi, const std::shared_ptr<Runtime::CommunicationContext> &context, F &&action)
        {
            Runtime::CommunicationDetail::LocalStage(mpi.comm, !context || context->ChecksCollectivePreflight(),
                                                     std::forward<F>(action), "native mesh publication failed validation on another rank");
        }

        void CheckPersistentIds(const MPIInfo &mpi, const EntitySpace &space, const Runtime::Array<index> &ids)
        {
            // Hash partition owner IDs, never allocate by the numeric ID range.
            // A peer stores only contributions routed to it. Full IDs, rather
            // than their hashes, decide uniqueness (hash collisions are benign).
            auto read = ids.Read();
            std::vector<std::vector<index>> buckets(size_t(mpi.size));
            for (index row = 0; row < space.NumOwned(); ++row)
            {
                const index id = read.View()(size_t(row), 0);
                uint64_t hash = uint64_t(id) + 0x9e3779b97f4a7c15ULL;
                hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ULL;
                hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebULL;
                hash ^= hash >> 31;
                buckets[hash % uint64_t(mpi.size)].push_back(id);
            }
            Runtime::RowBuilder builder(size_t(mpi.size), sizeof(index));
            std::vector<int> presence(size_t(mpi.size)), incoming(size_t(mpi.size));
            for (int peer = 0; peer < mpi.size; ++peer)
            {
                builder.ResizeRow(size_t(peer), buckets[size_t(peer)].size());
                auto row = builder.Row<index>(size_t(peer));
                for (size_t slot = 0; slot < buckets[size_t(peer)].size(); ++slot)
                    row[slot] = buckets[size_t(peer)][slot];
                presence[size_t(peer)] = !buckets[size_t(peer)].empty();
            }
            auto outgoing = Runtime::Array<index>::FromSharedStorage(builder.Commit());
            MPI_Alltoall(presence.data(), 1, MPI_INT, incoming.data(), 1, MPI_INT, mpi.comm);
            std::vector<std::vector<index>> send(size_t(mpi.size)), receive(size_t(mpi.size));
            for (int peer = 0; peer < mpi.size; ++peer)
            {
                if (presence[size_t(peer)])
                    send[size_t(peer)].push_back(index(peer));
                if (incoming[size_t(peer)])
                    receive[size_t(peer)].push_back(index(peer));
            }
            auto context = space.Communication() ? space.Communication() : std::make_shared<Runtime::CommunicationContext>(mpi.comm);
            auto route = Runtime::RowTransferPlan::Known(context, size_t(mpi.size), size_t(mpi.size), std::move(send), std::move(receive));
            Runtime::Array<index> received(route.DestinationLayout(outgoing.Layout()));
            Runtime::TransferOptions options{Runtime::TransferMode::Transient, context, 0};
            Runtime::PayloadPlan(route, outgoing.Layout(), received.Layout(), Runtime::ElementSchema::Arithmetic<index>())
                .Bind(outgoing, received, Runtime::TransferTransport::Direct, options)
                .Begin()
                .Wait();
            PublicationStage(mpi, space.Communication(), [&]
                             {
                std::unordered_set<index> unique;
                auto values = received.Read();
                for (size_t row = 0; row < values.View().RowCount(); ++row)
                    for (size_t slot = 0; slot < values.View().RowSize(row); ++slot)
                        DNDS_check_throw_info(unique.insert(values.View()(row, slot)).second,
                                              "duplicate persistent ID owned by different ranks"); });
            std::vector<Runtime::RowRequest> requests;
            for (index row = 0; row < space.NumGhost(); ++row)
            {
                auto global = space.LocalToGlobal(space.NumOwned() + row);
                requests.push_back({space.Owner(global), space.OwnerLocalRow(global), row});
            }
            Runtime::RowTransferPlan ghostRoute(context, ids.RowCount(), size_t(space.NumGhost()), requests);
            Runtime::Array<index> expected(Runtime::RowLayout::Dense(size_t(space.NumGhost()), 1));
            Runtime::PayloadPlan(ghostRoute, ids.Layout(), expected.Layout(), Runtime::ElementSchema::Arithmetic<index>())
                .Bind(ids, expected, Runtime::TransferTransport::Direct, options)
                .Begin()
                .Wait();
            PublicationStage(mpi, space.Communication(), [&]
                             {
                auto ownerIds = expected.Read();
                for (index row = 0; row < space.NumGhost(); ++row)
                    DNDS_check_throw_info(read.View()(size_t(space.NumOwned() + row), 0) == ownerIds.View()(size_t(row), 0),
                                          "ghost persistent ID differs from its owner"); });
        }
    }

    std::shared_ptr<const UnstructuredMeshState> UnstructuredMeshState::FromNative(
        const MPIInfo &mpi, NativeMeshData data, std::map<EntityKind, EntitySpace> spaces,
        std::map<EntityKind, Runtime::Array<index>> ids, MeshReadRequirements requirements)
    {
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native mesh publication requires a communicator");
        const auto cell = spaces.find(EntityKind::Cell);
        const auto communication = cell != spaces.end() ? cell->second.Communication() : nullptr;
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !communication || communication->ChecksCollectivePreflight());
        auto result = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState);
        auto requestedIdentity = std::move(data.identity);
        result->_requirements = requirements;
        result->_audit = data.audit;
        result->_ghostSpec = data.ghostSpec;
        int audits[]{int(data.audit.persistentIdentity), int(data.audit.closure),
                     -int(data.audit.persistentIdentity), -int(data.audit.closure)};
        MPI_Allreduce(MPI_IN_PLACE, audits, 4, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(audits[0] == -audits[2] && audits[1] == -audits[3],
                              "native publication audit options differ between ranks");
        PublicationStage(mpi, communication, [&]
                         {
        int rank = -1, size = 0;
        MPI_Comm_rank(mpi.comm, &rank);
        MPI_Comm_size(mpi.comm, &size);
        DNDS_check_throw_info(mpi.rank == rank && mpi.size == size, "native publication MPIInfo disagrees with communicator");
        const auto nodes = spaces.find(EntityKind::Node);
        DNDS_check_throw_info(nodes != spaces.end(), "native publication requires a node entity space");
        auto validate = [&](EntityKind kind, const auto &pair)
        {
            const auto found = spaces.find(kind);
            DNDS_check_throw_info(pair && found != spaces.end(), "native publication entity family is missing");
            const auto &space = found->second;
            space.Validate();
            DNDS_check_throw_info(space.ChecksCollectivePreflight() ==
                                      (!communication || communication->ChecksCollectivePreflight()),
                                  "native publication entity preparation policies differ");
            EntitySpace::CheckCommunicator(space._mpi.comm, mpi.comm);
            DNDS_check_throw_info(space.Kind() == kind && space.MeshIdentity() == nodes->second.MeshIdentity() &&
                                      pair->owned.Layout().RowCount() == size_t(space.NumOwned()) &&
                                      pair->ghost.Layout().RowCount() == size_t(space.NumGhost()),
                                  "native publication entity order or row extent mismatch");
            auto id = ids.find(kind);
            DNDS_check_throw_info(id != ids.end() && id->second.GetDevice() == Runtime::Device::Host() &&
                                      !id->second.Layout().IsVariable() && id->second.Layout().Width() == 1 &&
                                      id->second.Layout().RowCount() == size_t(space.NumLocal()),
                                  "native persistent-ID rows do not match their entity family");
            auto read = id->second.Read();
            std::unordered_set<index> ownedIds;
            for (index row = 0; row < space.NumLocal(); ++row)
            {
                const auto value = read.View()(size_t(row), 0);
                DNDS_check_throw_info(value >= 0 && value != UnInitIndex, "native persistent ID is invalid");
                if (result->_audit.persistentIdentity && row < space.NumOwned())
                    DNDS_check_throw_info(ownedIds.insert(value).second, "duplicate owned persistent ID");
            }
            result->_spaces.emplace(kind, Space{space, space.NumOwned(), space.NumGhost(), space._global, space._ghostMap});
        };
        validate(EntityKind::Node, data.coords);
        validate(EntityKind::Cell, data.cellInfo);
        validate(EntityKind::Bnd, data.bndInfo);
        if (requirements.faces)
            validate(EntityKind::Face, data.faceInfo);
        if (requirements.edges)
            validate(EntityKind::Edge, data.edgeInfo);
        DNDS_check_throw_info(result->_spaces.size() == spaces.size() && ids.size() == spaces.size(),
                              "native publication contains an unbound entity family");
        if (data.provenance)
        {
            (void)data.provenance->Signature();
            for (const auto &[kind, values] : data.provenance->sourceIds)
            {
                DNDS_check_throw_info(spaces.count(kind) && values.GetDevice() == Runtime::Device::Host() &&
                                          values.RowCount() == size_t(spaces.at(kind).NumOwned()), "provenance rows differ from owned destination family");
                auto read = std::make_shared<Runtime::Array<index>::ReadLease>(values.Read());
                for (size_t row = 0; row < values.RowCount(); ++row)
                    for (auto value : read->View()[row])
                        DNDS_check_throw_info(value >= 0 && value != UnInitIndex, "invalid source persistent ID in provenance");
                result->_provenanceReads.push_back(std::move(read));
            }
            result->_provenance = std::move(data.provenance);
        }
        for (auto &[key, relation] : data.namedRelations)
        {
            key.Validate();
            const auto source = spaces.find(key.from), target = spaces.find(key.to);
            DNDS_check_throw_info(source != spaces.end() && target != spaces.end(), "named relation entity family is missing");
            DNDS_check_throw_info(relation.rows.owned.RowCount() == size_t(source->second.NumOwned()) &&
                                      relation.rows.ghost.RowCount() == size_t(source->second.NumGhost()),
                                  "named relation rows differ from canonical source space");
            auto read = MeshAdjacencyRead(key, source->second, target->second, relation.state,
                                           relation.rows.owned, &relation.rows.ghost);
            // A legacy descriptor names one fixed facade definition. A named
            // copy may use different storage/index encoding, but may not change
            // ordered slots or absent sentinels behind that same exact key.
            const std::pair<AdjacencyDescriptor, MeshRelation> aliases[]{
                {Adj::Cell2Node, MeshRelation::CellNodes}, {Adj::Bnd2Node, MeshRelation::BoundaryNodes},
                {Adj::Bnd2Cell, MeshRelation::BoundaryCells}, {Adj::Cell2Cell, MeshRelation::CellCells},
                {Adj::Node2Cell, MeshRelation::NodeCells}, {Adj::Node2Bnd, MeshRelation::NodeBoundaries},
                {Adj::Cell2Face, MeshRelation::CellFaces}, {Adj::Face2Node, MeshRelation::FaceNodes},
                {Adj::Face2Cell, MeshRelation::FaceCells}, {Adj::Face2Bnd, MeshRelation::FaceBoundaries},
                {Adj::Bnd2Face, MeshRelation::BoundaryFaces}, {Adj::Cell2Edge, MeshRelation::CellEdges},
                {Adj::Edge2Node, MeshRelation::EdgeNodes}, {Adj::Edge2Cell, MeshRelation::EdgeCells},
                {Adj::Cell2CellFace, MeshRelation::CellFaceCells}};
            for (const auto &[alias, slot] : aliases)
                if (key == alias && data.relations[size_t(slot)])
                {
                    const auto &fixed = *data.relations[size_t(slot)];
                    if (relation.state == Adj_PointToLocal &&
                        relation.rows.owned.SharedStorage() == fixed.owned.SharedStorage() &&
                        relation.rows.ghost.SharedStorage() == fixed.ghost.SharedStorage())
                        break; // Exact retained layout/value aliases already checked above.
                    auto fixedRead = MeshAdjacencyRead(key, source->second, target->second, Adj_PointToLocal,
                                                        fixed.owned, &fixed.ghost);
                    DNDS_check_throw_info(read.RowCount() == fixedRead.RowCount(),
                                          "named relation duplicates a facade definition with different source rows");
                    for (index row = 0; row < read.RowCount(); ++row)
                    {
                        DNDS_check_throw_info(read.RowSize(row) == fixedRead.RowSize(row),
                                              "named relation contradicts ordered facade slots for the same definition");
                        for (size_t slot = 0; slot < read.RowSize(row); ++slot)
                            DNDS_check_throw_info(read.ConvertedIndex(row, slot, Adj_PointToGlobal) ==
                                                      fixedRead.ConvertedIndex(row, slot, Adj_PointToGlobal),
                                                  "named relation contradicts ordered facade slots for the same definition");
                    }
                    break;
                }
            NamedRelation held{std::move(relation), {}, {}};
            held.owned = std::make_shared<Runtime::Array<index>::ReadLease>(held.data.rows.owned.Read());
            held.ghost = std::make_shared<Runtime::Array<index>::ReadLease>(held.data.rows.ghost.Read());
            result->_namedRelations.emplace(key, std::move(held));
        }
        data.namedRelations.clear();
        result->_placements.emplace(UnstructuredMeshState::PlacementKey(Runtime::Device::Host()), UnstructuredMeshSnapshot::FromNative(std::move(data), requirements));
        result->_hostRead = std::make_shared<MeshReadAccess>(result->_placements.at(UnstructuredMeshState::PlacementKey(Runtime::Device::Host())).AcquireRead());
        result->_hostView = result->_hostRead->HostView();
        result->_originalIds = std::move(ids);
        for (const auto &[kind, values] : result->_originalIds)
            result->_originalReads.emplace(kind, std::make_shared<Runtime::Array<index>::ReadLease>(values.Read())); });
        const int signature = int(requirements.faces) | (int(requirements.inverse) << 1) |
                              (int(requirements.edges) << 2) | (int(requirements.cellFaceNeighbors) << 3) |
                              (int(requirements.wallDistance) << 4) | (int(requirements.allFaceCellsLocal) << 5) |
                              (int(result->_hostView.IsPeriodic()) << 6) | (result->_hostView.Dimension() << 7);
        int range[]{signature, -signature};
        MPI_Allreduce(MPI_IN_PLACE, range, 2, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(range[0] == -range[1], "native publication requirements differ between ranks");
        std::string schedule;
        PublicationStage(mpi, communication, [&]
                         {
            schedule += result->_provenance ? "p" + result->_provenance->Signature() + ";" : "p;";
            for (const auto &[key, relation] : result->_namedRelations)
            {
                const auto signature = key.Signature();
                schedule += std::to_string(signature.size()) + ":" + signature + ":" + std::to_string(int(relation.data.state)) + ";";
            }
            DNDS_check_throw_info(schedule.size() <= size_t(INT_MAX), "named relation schedule exceeds MPI range"); });
        int length = int(schedule.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string expected;
        PublicationStage(mpi, communication, [&]
                         { expected = mpi.rank == 0 ? schedule : std::string(size_t(length), '\0'); });
        MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
        PublicationStage(mpi, communication, [&]
                         { DNDS_check_throw_info(schedule == expected, "named relation definitions differ between ranks"); });
        if (result->_audit.persistentIdentity)
            for (const auto &[kind, values] : result->_originalIds)
                CheckPersistentIds(mpi, result->_spaces.at(kind).identity, values);
        std::map<EntityKind, index> localMaxima;
        PublicationStage(mpi, communication, [&]
                         {
            for (const auto &[kind, read] : result->_originalReads)
            {
                index maximum = -1;
                for (index row = 0; row < result->_spaces.at(kind).owned; ++row)
                    maximum = std::max(maximum, read->View()(size_t(row), 0));
                localMaxima.emplace(kind, maximum);
            } });
        result->_meshIdentity = PrepareMeshIdentity(mpi, std::move(requestedIdentity), localMaxima);
        result->BindPreparedEpoch(result->_versions);
        return result;
    }

    void UnstructuredMeshState::Audit(MeshAuditOptions options) const
    {
        const auto &mpi = _spaces.at(EntityKind::Node).identity._mpi;
        const auto &communication = _spaces.at(EntityKind::Node).identity.Communication();
        int flags[]{int(options.persistentIdentity), int(options.closure),
                    -int(options.persistentIdentity), -int(options.closure)};
        MPI_Allreduce(MPI_IN_PLACE, flags, 4, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(flags[0] == -flags[2] && flags[1] == -flags[3], "native checkpoint audit options differ between ranks");
        std::string schedule;
        PublicationStage(mpi, communication, [&]
                         {
            Validate();
            schedule = _meshIdentity.meshNamespace + ":" + std::to_string(_hostView.Dimension()) + ":" +
                       std::to_string(_hostView.IsPeriodic()) + ":" + std::to_string(_requirements.faces) +
                       std::to_string(_requirements.inverse) + std::to_string(_requirements.edges) +
                       std::to_string(_requirements.cellFaceNeighbors) + std::to_string(_requirements.wallDistance) +
                       std::to_string(_requirements.allFaceCellsLocal) + ";";
            for (const auto &[kind, unused] : _originalIds) schedule += std::to_string(int(kind)) + ";";
            DNDS_check_throw_info(schedule.size() <= size_t(INT_MAX), "native checkpoint audit schedule exceeds MPI range"); });
        int length = int(schedule.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string expected;
        PublicationStage(mpi, communication, [&]
                         { expected = mpi.rank == 0 ? schedule : std::string(size_t(length), '\0'); });
        MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
        PublicationStage(mpi, communication, [&]
                         {
            DNDS_check_throw_info(schedule == expected, "native checkpoint audit namespace or inventory differs between ranks");
            _hostRead->Validate(_requirements, options); });
        if (options.persistentIdentity)
            for (const auto &[kind, values] : _originalIds)
                CheckPersistentIds(mpi, _spaces.at(kind).identity, values);
    }

    std::shared_ptr<const UnstructuredMeshState> UnstructuredMeshState::BindEpoch(Runtime::VersionTracker versions) const
    {
        auto result = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState(*this));
        result->BindPreparedEpoch(std::move(versions));
        return result;
    }
    void UnstructuredMeshState::BindPreparedEpoch(Runtime::VersionTracker versions)
    {
        _versions.RebindPrepared(versions);
        _epoch = _versions.Stamp();
        for (auto &[kind, space] : _spaces)
        {
            space.identity._stamp = _versions.Stamp(Runtime::Change::Topology | Runtime::Change::Distribution | Runtime::Change::Layout);
            space.identity._versions.RebindPrepared(_versions);
        }
    }

    std::shared_ptr<const UnstructuredMeshState> UnstructuredMeshState::WithPlacement(Runtime::Device device, Runtime::ExecutionContext context) const
    {
        Validate();
        DNDS_check_throw_info(device.IsValid(),
                              "invalid native mesh placement device");
        context.Validate();
        DNDS_check_throw_info(context.device == device,
                              "native mesh placement context mismatch");
        auto result = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState(*this));
        if (!result->_placements.count(PlacementKey(device)))
            result->_placements.emplace(PlacementKey(device), _placements.at(UnstructuredMeshState::PlacementKey(Runtime::Device::Host())).To(device, context));
        return result;
    }

    tLocalMatStruct UnstructuredMeshState::BuildLocalCellNeighbors(bool onLocalPartition) const
    {
        Validate();
        CheckRequirements(MeshReadRequirements::CellFlux());
        const auto &view = BorrowHostView();
        tLocalMatStruct result(size_t(view.NumCell()));
        for (int part = 0; part < NLocalParts(); ++part)
        {
            const auto worker = LocalPartRanges(part);
            for (index pos = 0; pos < worker.Size(); ++pos)
            {
                const auto cell = worker.RowAt(pos);
                const auto faces = view.CellFaces(cell);
                auto &row = result[size_t(cell)];
                row.reserve(faces.size());
                for (size_t slot = 0; slot < faces.size(); ++slot)
                {
                    const auto face = faces[slot];
                    DNDS_check_throw_info(face >= 0 && face < view.NumFaceProc(), "native local graph requires complete owned cell faces");
                    const auto other = view.CellFaceOther(cell, face, rowsize(slot));
                    DNDS_check_throw_info(other == UnInitIndex || (other >= 0 && other < view.NumCellProc()),
                                          "native local graph requires local face-cell closure");
                    if (other == UnInitIndex || other == cell || other >= view.NumCell())
                        continue;
                    if (onLocalPartition && !worker.Contains(other))
                        continue;
                    row.push_back(other);
                }
            }
        }
        return result;
    }

    std::shared_ptr<const UnstructuredMeshState> UnstructuredMeshState::WithLocalPartitions(std::vector<index> partStarts) const
    {
        Validate();
        return WithLocalPartitionRanges(WorkerRangesFromStarts(partStarts, NumOwned(EntityKind::Cell)));
    }

    std::shared_ptr<const UnstructuredMeshState> UnstructuredMeshState::WithLocalPartitionRanges(std::vector<WorkerRowRanges> parts) const
    {
        Validate();
        ValidateWorkerRanges(parts, NumOwned(EntityKind::Cell));
        auto result = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState(*this));
        result->_localPartRanges = std::move(parts);
        if (result->Requirements().faces)
            result->_localCellNeighbors = result->BuildLocalCellNeighbors(true);
        else
            result->_localCellNeighbors.clear();
        return result;
    }

    std::shared_ptr<const UnstructuredMeshState> UnstructuredMeshState::WithLocalCellSetup(
        std::vector<index> partStarts, tLocalMatStruct neighbors) const
    {
        Validate();
        std::vector<index> current;
        for (int part = 0; part < NLocalParts(); ++part)
            current.push_back(LocalPartStart(part));
        current.push_back(NumOwned(EntityKind::Cell));
        DNDS_check_throw_info(partStarts == current,
                              "native local setup changes current cell partition or row count");
        return WithLocalCellNeighbors(std::move(neighbors));
    }

    std::shared_ptr<const UnstructuredMeshState> UnstructuredMeshState::WithLocalCellNeighbors(tLocalMatStruct neighbors) const
    {
        Validate();
        DNDS_check_throw_info(neighbors.size() == size_t(NumOwned(EntityKind::Cell)), "wrong cell-neighbor row count");
        for (size_t row = 0; row < neighbors.size(); ++row)
            for (auto other : neighbors[row])
                DNDS_check_throw_info(other >= 0 && other < NumOwned(EntityKind::Cell) && other != index(row),
                                      "native local graph contains an unavailable, ghost or diagonal cell");
        auto result = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState(*this));
        result->_localCellNeighbors = std::move(neighbors);
        return result;
    }

    std::shared_ptr<const UnstructuredMeshState> UnstructuredMeshState::WithWallDistance(Runtime::ArrayPair<real> walls) const
    {
        Validate();
        const auto &old = Snapshot(Runtime::Device::Host()).Live();
        return WithGeometry(*old.coords, std::move(walls), old.semantics.periodicInfo);
    }

    std::shared_ptr<const UnstructuredMeshState> UnstructuredMeshState::WithGeometry(
        Runtime::ArrayPair<real> coordinates, std::optional<Runtime::ArrayPair<real>> wallVectors,
        const Periodicity &periodicInfo) const
    {
        Validate();
        const auto &old = Snapshot(Runtime::Device::Host()).Live();
        auto check = [&](const Runtime::ArrayPair<real> &pair)
        {
            DNDS_check_throw_info(pair.owned.RowCount() == size_t(NumOwned(EntityKind::Node)) &&
                                      pair.ghost.RowCount() == size_t(NumGhost(EntityKind::Node)),
                                  "native geometry replacement changes node extent");
            for (const auto *array : {&pair.owned, &pair.ghost})
            {
                DNDS_check_throw_info(array->GetDevice() == Runtime::Device::Host(), "native geometry requires host buffers");
                auto read = array->Read();
                for (size_t row = 0; row < array->RowCount(); ++row)
                {
                    DNDS_check_throw_info(read.View().RowSize(row) == 3, "native geometry row requires three components");
                    for (size_t component = 0; component < 3; ++component)
                        DNDS_check_throw_info(std::isfinite(read.View()(row, component)), "native geometry value is not finite");
                }
            }
        };
        check(coordinates);
        if (wallVectors)
            check(*wallVectors);
        for (size_t frame = 0; frame < periodicInfo.rotation.size(); ++frame)
            DNDS_check_throw_info(periodicInfo.rotation[frame].map().allFinite() &&
                                      periodicInfo.translation[frame].map().allFinite() && periodicInfo.rotationCenter[frame].map().allFinite(),
                                  "native periodic geometry is not finite");
        NativeMeshData data;
        data.audit = _audit;
        data.ghostSpec = _ghostSpec;
        data.dimension = old.semantics.dim;
        data.periodic = old.semantics.isPeriodic;
        data.periodicInfo = periodicInfo;
        data.relations = old.relations;
        data.coords = std::move(coordinates);
        data.nodeWallDist = std::move(wallVectors);
        data.cellInfo = old.cellInfo;
        data.bndInfo = old.bndInfo;
        data.faceInfo = old.faceInfo;
        data.edgeInfo = old.edgeInfo;
        data.cellNodesPbi = old.cellNodesPbi;
        data.bndNodesPbi = old.bndNodesPbi;
        data.cellFacesPbi = old.cellFacesPbi;
        data.faceNodesPbi = old.faceNodesPbi;
        data.cellEdgesPbi = old.cellEdgesPbi;
        data.edgeNodesPbi = old.edgeNodesPbi;
        auto result = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState(*this));
        result->_requirements.wallDistance = bool(data.nodeWallDist);
        result->_placements.clear();
        result->_placements.emplace(UnstructuredMeshState::PlacementKey(Runtime::Device::Host()), UnstructuredMeshSnapshot::FromNative(std::move(data), result->_requirements));
        result->_hostRead = std::make_shared<MeshReadAccess>(result->_placements.at(UnstructuredMeshState::PlacementKey(Runtime::Device::Host())).AcquireRead());
        result->_hostView = result->_hostRead->HostView();
        // A detached replacement has independent validity until the facade
        // publishes it. Rebinding handles never mutates retained source maps.
        result->BindPreparedEpoch(Runtime::VersionTracker{});
        return result;
    }

    MeshAdjacencyRead UnstructuredMeshState::Relation(const AdjacencyDescriptor &key) const
    {
        Validate();
        const auto found = _namedRelations.find(key);
        DNDS_check_throw_info(found != _namedRelations.end(), "named relation definition is not published");
        const auto &relation = found->second.data;
        return MeshAdjacencyRead(key, Entities(key.from), Entities(key.to), relation.state,
                                 relation.rows.owned, &relation.rows.ghost);
    }

    std::vector<AdjacencyDescriptor> UnstructuredMeshState::RelationKeys() const
    {
        Validate();
        std::vector<AdjacencyDescriptor> result;
        result.reserve(_namedRelations.size());
        for (const auto &[key, unused] : _namedRelations)
            result.push_back(key);
        return result;
    }

    void UnstructuredMeshState::CheckRequirements(MeshReadRequirements r) const
    {
        Validate();
        DNDS_check_throw_info((!r.faces || _requirements.faces) && (!r.inverse || _requirements.inverse) &&
                                  (!r.edges || _requirements.edges) && (!r.cellFaceNeighbors || _requirements.cellFaceNeighbors) &&
                                  (!r.wallDistance || _requirements.wallDistance) &&
                                  (!r.allFaceCellsLocal || _requirements.allFaceCellsLocal),
                              "native publication lacks requested relation group or closure; republish explicitly");
    }

    const UnstructuredMeshSnapshot &UnstructuredMeshState::Snapshot(Runtime::Device device) const
    {
        Validate();
        auto found = _placements.find(PlacementKey(device));
        DNDS_check_throw_info(found != _placements.end(), "native mesh placement is not prepared; place explicitly before reading");
        return found->second;
    }

    index UnstructuredMeshState::NumOwned(EntityKind kind) const
    {
        auto found = _spaces.find(kind);
        DNDS_check_throw_info(found != _spaces.end(), "native entity family was not published");
        return found->second.owned;
    }
    index UnstructuredMeshState::NumGhost(EntityKind kind) const
    {
        auto found = _spaces.find(kind);
        DNDS_check_throw_info(found != _spaces.end(), "native entity family was not published");
        return found->second.ghost;
    }
    index UnstructuredMeshState::NumGlobal(EntityKind kind) const
    {
        auto found = _spaces.find(kind);
        DNDS_check_throw_info(found != _spaces.end(), "native entity family was not published");
        return found->second.global->globalSize();
    }
    EntitySpace UnstructuredMeshState::Entities(EntityKind kind) const
    {
        Validate();
        auto found = _spaces.find(kind);
        DNDS_check_throw_info(found != _spaces.end(), "native entity family was not published");
        return found->second.identity;
    }
    index UnstructuredMeshState::OriginalEntityId(EntityKind kind, index row) const
    {
        auto found = _originalReads.find(kind);
        DNDS_check_throw_info(found != _originalReads.end() && row >= 0 && size_t(row) < found->second->Layout().RowCount(),
                              "original entity ID outside published family");
        return found->second->View()(size_t(row), 0);
    }
}
