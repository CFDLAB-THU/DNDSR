#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Connectivity/TopologySubEntityQuery.hpp"
#include <limits>
#include "Geom/Connectivity/NativeMeshClosure.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Connectivity/NativePeriodicMatch.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Geom/Interface/BoundaryCondition.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include "DNDS/Core/OMP.hpp"
#include <set>

namespace DNDS::Geom
{
    namespace
    {
        namespace R = Runtime;
        using Array = R::Array<index>;

        Array BuildIndexRows(const std::vector<std::vector<index>> &rows, std::optional<size_t> width = {})
        {
            return R::BuildArrayRows<index>(rows.size(), [&](size_t row)
                                            { return rows[row].size(); }, [&](size_t row, auto output)
                                            { std::copy(rows[row].begin(), rows[row].end(), output.begin()); }, width, true);
        }

        Array Transfer(const R::RowTransferPlan &route, const Array &values, const std::shared_ptr<R::CommunicationContext> &context)
        {
            R::PayloadPlan payload(route, values.Layout(), R::ElementSchema::Arithmetic<index>());
            Array result(payload.DestinationLayout());
            R::TransferOptions options;
            options.mode = R::TransferMode::Transient;
            options.context = context;
            payload.Bind(values, result, R::TransferTransport::Direct, options)
                .Begin()
                .Wait();
            return result;
        }

        Array BitRows(const R::Array<NodePeriodicBits> &values)
        {
            auto read = values.Read();
            std::vector<std::vector<index>> rows(values.RowCount());
            OMP::ForSetupRows(0, index(rows.size()), [&](index iRow)
                              {
                auto &output = rows[size_t(iRow)];
                output.reserve(read.View().RowSize(iRow));
                for (auto bits : read.View()[iRow])
                    output.push_back(uint8_t(bits)); });
            return BuildIndexRows(rows);
        }

        Array Exchange(const MPIInfo &mpi, const std::vector<std::vector<index>> &messages, const std::shared_ptr<R::CommunicationContext> &communication)
        {
            auto values = BuildIndexRows(messages);
            std::vector<int> outgoing(size_t(mpi.size)), incoming(size_t(mpi.size));
            for (int peer = 0; peer < mpi.size; ++peer)
                outgoing[size_t(peer)] = !messages[size_t(peer)].empty();
            MPI_Alltoall(outgoing.data(), 1, MPI_INT, incoming.data(), 1, MPI_INT, mpi.comm);
            std::vector<std::vector<index>> send(size_t(mpi.size)), receive(size_t(mpi.size));
            for (int peer = 0; peer < mpi.size; ++peer)
            {
                if (outgoing[size_t(peer)])
                    send[size_t(peer)].push_back(index(peer));
                if (incoming[size_t(peer)])
                    receive[size_t(peer)].push_back(index(peer));
            }
            auto context = communication ? communication : std::make_shared<R::CommunicationContext>(mpi.comm);
            auto route = R::RowTransferPlan::Known(context, size_t(mpi.size), size_t(mpi.size), std::move(send), std::move(receive));
            return Transfer(route, values, context);
        }

        using detail::CanonicalRoleKey;
        using detail::Query;

        void AddIdentity(NativeOwnedMeshData &data, const EntitySpace &space, const std::map<EntityKind, index> &starts)
        {
            const auto found = starts.find(space.Kind());
            const index first = found == starts.end() ? 0 : found->second;
            DNDS_check_throw_info(first >= 0 && space.NumGlobal() <= std::numeric_limits<index>::max() - first,
                                  "generated native persistent IDs exceed high-water range");
            Array ids(R::RowLayout::Dense(size_t(space.NumOwned()), 1));
            {
                auto write = ids.Write();
                const auto output = write.View();
                OMP::ForSetupRows(0, space.NumOwned(), [&](index iRow)
                                  { output(iRow, 0) = first + space.LocalToGlobal(iRow); });
            }
            data.spaces.emplace(space.Kind(), space);
            data.ids.emplace(space.Kind(), std::move(ids));
            data.identity.nextIds[space.Kind()] = first + space.NumGlobal();
        }

        // Detached refresh maps use CURRENT global row numbers. Persistent IDs
        // remain payload; no table is sized or addressed by an ID magnitude.
        struct RefreshMap
        {
            EntitySpace oldSpace, newSpace;
            Array oldToNew, newToOld;
        };

        Array Fetch(const MPIInfo &mpi, const EntitySpace &space, const Array &values,
                    const std::vector<index> &globals)
        {
            std::vector<R::RowRequest> requests;
            space.LocalStage([&]
                             {
                requests.reserve(globals.size());
                for (size_t row = 0; row < globals.size(); ++row)
                {
                    DNDS_check_throw_info(globals[row] >= 0 && globals[row] < space.NumGlobal(),
                                          "native topology refresh references an unavailable source row");
                    requests.push_back({space.Owner(globals[row]), space.OwnerLocalRow(globals[row]), index(row)});
                } });
            auto context = space.Communication() ? space.Communication() : std::make_shared<R::CommunicationContext>(mpi.comm);
            R::RowTransferPlan route(context, size_t(space.NumOwned()), globals.size(), requests);
            return Transfer(route, values, context);
        }

        std::vector<std::vector<index>> IdentityKeys(const MPIInfo &mpi, const EntitySpace &nodes,
                                                     const Array &nodeIds, const Array &cone,
                                                     const R::Array<ElemInfo> &information,
                                                     const std::optional<R::Array<NodePeriodicBits>> &periodic)
        {
            std::vector<index> globals;
            nodes.LocalStage([&]
                             {
                auto read = cone.Read();
                std::vector<index> starts(cone.RowCount() + 1);
                const index total = OMP::ExclusivePrefixRows(
                    index(cone.RowCount()),
                    [&](index row) { return index(read.View().RowSize(size_t(row))); },
                    [&](index row, index offset) { starts[size_t(row)] = offset; });
                globals.resize(size_t(total));
                OMP::ForSetupRows(0, index(cone.RowCount()), [&](index row)
                {
                    const auto input = read.View()[size_t(row)];
                    std::copy(input.begin(), input.end(), globals.begin() + starts[size_t(row)]);
                });
                std::sort(globals.begin(), globals.end());
                globals.erase(std::unique(globals.begin(), globals.end()), globals.end()); });
            auto fetched = Fetch(mpi, nodes, nodeIds, globals);
            std::vector<std::vector<index>> keys;
            nodes.LocalStage([&]
                             {
                auto ids = fetched.Read(), rows = cone.Read();
                auto info = information.Read();
                std::optional<R::Array<NodePeriodicBits>::ReadLease> bits;
                if (periodic) bits.emplace(periodic->Read());
                DNDS_check_throw_info(information.RowCount() == cone.RowCount(), "native topology identity information extent mismatch");
                keys.resize(cone.RowCount());
                OMP::ForSetupRows(0, index(keys.size()), [&](index iRow)
                {
                    const auto row = size_t(iRow);
                    NativePeriodicNodes pattern;
                    pattern.reserve(rows.View().RowSize(row));
                    for (size_t slot = 0; slot < rows.View().RowSize(row); ++slot)
                    {
                        const auto global = rows.View()(row, slot);
                        const auto found = std::lower_bound(globals.begin(), globals.end(), global);
                        DNDS_check_throw_info(found != globals.end() && *found == global,
                                              "native topology identity fetch lost a node row");
                        pattern.emplace_back(ids.View()(size_t(found - globals.begin()), 0),
                                             bits ? uint8_t(bits->View()(row, slot)) : 0);
                    }
                    // Same full-node/image equivalence as NativePeriodicFrame:
                    // orientation may change, and a uniform XOR frame is free.
                    keys[row] = CanonicalRoleKey(Elem::Element{info.View()(row, 0).getElemType()}, pattern, bool(periodic));
                }); });
            return keys;
        }

        RefreshMap RecoverIdentity(const MPIInfo &mpi,
                                   const NativeOwnedMeshData &previous, NativeOwnedMeshData &data,
                                   const EntitySpace &space, MeshRelation coneSlot,
                                   const R::Array<ElemInfo> &information,
                                   const std::optional<R::Array<NodePeriodicBits>> &bits,
                                   std::optional<index> generatedFloor = {})
        {
            const auto kind = space.Kind();
            const auto old = previous.spaces.at(kind);
            const auto &oldInfo = kind == EntityKind::Face ? *previous.faceInfo : *previous.edgeInfo;
            const auto &oldBits = kind == EntityKind::Face ? previous.faceNodesPbi : previous.edgeNodesPbi;
            auto oldKeys = IdentityKeys(mpi, previous.spaces.at(EntityKind::Node), previous.ids.at(EntityKind::Node),
                                        *previous.relations[size_t(coneSlot)], oldInfo, oldBits);
            auto newKeys = IdentityKeys(mpi, previous.spaces.at(EntityKind::Node), previous.ids.at(EntityKind::Node),
                                        *data.relations[size_t(coneSlot)], information, bits);
            std::vector<std::vector<index>> messages(size_t(mpi.size));
            data.LocalStage(mpi, [&]
                            {
                auto ids = previous.ids.at(kind).Read();
                auto append = [&](const std::vector<std::vector<index>> &keys, const EntitySpace &family, bool isNew)
                {
                    for (size_t row = 0; row < keys.size(); ++row)
                    {
                        uint64_t hash = 1469598103934665603ULL;
                        // Routing uses the full-node part. Vertex roles remain
                        // in the exact key; sharing a hash server is harmless.
                        for (size_t slot = 0; slot < size_t(2 + 2 * keys[row][1]); ++slot)
                            hash = (hash ^ uint64_t(keys[row][slot])) * 1099511628211ULL;
                        auto &message = messages[size_t(hash % uint64_t(mpi.size))];
                        message.insert(message.end(), {int(isNew), index(row), family.LocalToGlobal(index(row)),
                                                       isNew ? UnInitIndex : ids.View()(row, 0), index(keys[row].size())});
                        message.insert(message.end(), keys[row].begin(), keys[row].end());
                    }
                };
                append(oldKeys, old, false);
                append(newKeys, space, true); });
            auto incoming = Exchange(mpi, messages, data.communication);
            std::vector<std::vector<index>> replies(size_t(mpi.size));
            data.LocalStage(mpi, [&]
                            {
                struct Entry { int rank; index row, global, id; };
                std::map<std::vector<index>, Entry> oldByKey, newByKey;
                auto read = incoming.Read();
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    const auto message = read.View()[size_t(peer)];
                    for (size_t slot = 0; slot < message.size();)
                    {
                        DNDS_check_throw_info(message.size() - slot >= 5, "malformed native topology identity record");
                        const bool isNew = message[slot] == 1;
                        const index count = message[slot + 4];
                        DNDS_check_throw_info((message[slot] == 0 || isNew) && count >= 2 && size_t(count) <= message.size() - slot - 5,
                                              "malformed native topology identity key");
                        std::vector<index> key(message.data() + slot + 5, message.data() + slot + 5 + size_t(count));
                        DNDS_check_throw_info((isNew ? newByKey : oldByKey).emplace(std::move(key), Entry{peer, message[slot + 1], message[slot + 2], message[slot + 3]}).second,
                                              "duplicate canonical key in native topology refresh");
                        slot += 5 + size_t(count);
                    }
                }
                for (const auto &[key, fresh] : newByKey)
                {
                    const auto found = oldByKey.find(key);
                    const auto oldGlobal = found == oldByKey.end() ? UnInitIndex : found->second.global;
                    const auto id = found == oldByKey.end() || generatedFloor ? UnInitIndex : found->second.id;
                    auto &reply = replies[size_t(fresh.rank)];
                    reply.insert(reply.end(), {1, fresh.row, id, oldGlobal});
                }
                for (const auto &[key, prior] : oldByKey)
                {
                    const auto found = newByKey.find(key);
                    auto &reply = replies[size_t(prior.rank)];
                    reply.insert(reply.end(), {0, prior.row, found == newByKey.end() ? UnInitIndex : found->second.global, 0});
                } });
            auto returned = Exchange(mpi, replies, data.communication);
            std::vector<index> ids(size_t(space.NumOwned()), UnInitIndex), oldGlobals(ids.size(), UnInitIndex), newGlobals(size_t(old.NumOwned()), UnInitIndex);
            index missing = 0;
            data.LocalStage(mpi, [&]
                            {
                std::vector<bool> seenOld(newGlobals.size()), seenNew(ids.size());
                auto read = returned.Read();
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    const auto message = read.View()[size_t(peer)];
                    DNDS_check_throw_info(message.size() % 4 == 0, "malformed native topology identity reply");
                    for (size_t slot = 0; slot < message.size(); slot += 4)
                    {
                        const auto row = message[slot + 1];
                        auto &seen = message[slot] == 1 ? seenNew : seenOld;
                        DNDS_check_throw_info(row >= 0 && size_t(row) < seen.size() && !seen[size_t(row)], "invalid native topology identity reply row");
                        seen[size_t(row)] = true;
                        if (message[slot] == 1) { ids[size_t(row)] = message[slot + 2]; oldGlobals[size_t(row)] = message[slot + 3]; }
                        else newGlobals[size_t(row)] = message[slot + 2];
                    }
                }
                DNDS_check_throw_info(std::all_of(seenOld.begin(), seenOld.end(), [](bool value){return value;}) &&
                                      std::all_of(seenNew.begin(), seenNew.end(), [](bool value){return value;}), "incomplete native topology identity replies");
                missing = std::count(ids.begin(), ids.end(), UnInitIndex); });
            index prefix = 0, total = 0;
            MPI_Exscan(&missing, &prefix, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            if (mpi.rank == 0)
                prefix = 0;
            MPI_Allreduce(&missing, &total, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            const index first = generatedFloor.value_or(previous.identity.nextIds.at(kind));
            data.LocalStage(mpi, [&]
                            {
                DNDS_check_throw_info(total <= std::numeric_limits<index>::max() - first, "refreshed native persistent IDs exceed high-water range");
                index next = first + prefix;
                for (auto &id : ids) if (id == UnInitIndex) id = next++;
                Array idRows(R::RowLayout::Dense(ids.size(), 1));
                {
                    auto write = idRows.Write();
                    const auto output = write.View();
                    OMP::ForSetupRows(0, index(ids.size()), [&](index row)
                    { output(row, 0) = ids[size_t(row)]; });
                }
                data.ids[kind] = std::move(idRows);
                data.identity.nextIds[kind] = first + total; });
            std::optional<RefreshMap> result;
            data.LocalStage(mpi, [&]
                            {
                Array forward(R::RowLayout::Dense(newGlobals.size(), 1));
                Array reverse(R::RowLayout::Dense(oldGlobals.size(), 1));
                {
                    auto forwardWrite = forward.Write();
                    auto reverseWrite = reverse.Write();
                    const auto forwardView = forwardWrite.View();
                    const auto reverseView = reverseWrite.View();
                    OMP::ForSetupRows(0, index(newGlobals.size()), [&](index row)
                    { forwardView(row, 0) = newGlobals[size_t(row)]; });
                    OMP::ForSetupRows(0, index(oldGlobals.size()), [&](index row)
                    { reverseView(row, 0) = oldGlobals[size_t(row)]; });
                }
                data.spaces.emplace(kind, space);
                result.emplace(RefreshMap{old, space, std::move(forward), std::move(reverse)}); });
            return std::move(*result);
        }

        Array RefreshSourceRows(const MPIInfo &mpi, const Array &values, const RefreshMap &mapping)
        {
            std::vector<index> globals;
            mapping.oldSpace.LocalStage([&]
                                        {
                auto read = mapping.newToOld.Read();
                for (size_t row = 0; row < mapping.newToOld.RowCount(); ++row)
                {
                    const auto global = read.View()(row, 0);
                    DNDS_check_throw_info(global != UnInitIndex, "new topology rows have no inferable named-relation or ancestry payload");
                    globals.push_back(global);
                } });
            return Fetch(mpi, mapping.oldSpace, values, globals);
        }

        Array RefreshTargetRows(const MPIInfo &mpi, const Array &values, const RefreshMap &mapping, bool sortedUnique)
        {
            std::vector<index> globals;
            mapping.oldSpace.LocalStage([&]
                                        {
                auto read = values.Read();
                for (size_t row = 0; row < values.RowCount(); ++row)
                    for (auto global : read.View()[row]) if (global != UnInitIndex) globals.push_back(global);
                std::sort(globals.begin(), globals.end());
                globals.erase(std::unique(globals.begin(), globals.end()), globals.end()); });
            auto fetched = Fetch(mpi, mapping.oldSpace, mapping.oldToNew, globals);
            std::vector<std::vector<index>> rows;
            mapping.oldSpace.LocalStage([&]
                                        {
                auto read = fetched.Read();
                auto original = values.Read();
                rows.resize(values.RowCount());
                OMP::ForSetupRows(0, index(rows.size()), [&](index iRow)
                {
                    const auto row = size_t(iRow);
                    for (auto global : original.View()[row])
                    {
                        if (global == UnInitIndex)
                        {
                            rows[row].push_back(global);
                            continue;
                        }
                        const auto found = std::lower_bound(globals.begin(), globals.end(), global);
                        DNDS_check_throw_info(found != globals.end() && *found == global,
                                              "named relation target fetch lost a topology row");
                        const auto target = read.View()(size_t(found - globals.begin()), 0);
                        DNDS_check_throw_info(target != UnInitIndex, "named relation references a removed topology key");
                        rows[row].push_back(target);
                    }
                    if (sortedUnique)
                    {
                        std::sort(rows[row].begin(), rows[row].end());
                        rows[row].erase(std::unique(rows[row].begin(), rows[row].end()), rows[row].end());
                    }
                }); });
            return BuildIndexRows(rows);
        }

        R::Array<ElemInfo> Information(const R::Array<ElemInfo> &original, const std::vector<t_index> &zones)
        {
            R::Array<ElemInfo> result(R::RowLayout::Dense(original.RowCount(), 1));
            auto read = original.Read();
            auto write = result.Write();
            const auto input = read.View();
            const auto output = write.View();
            OMP::ForSetupRows(0, input.RowCount(), [&](index iRow)
                              {
                output(iRow, 0) = input(iRow, 0);
                output(iRow, 0).zone = zones[iRow]; });
            return result;
        }
    }

    NativeOwnedMeshData NativeMeshTopology::Build(NativeOwnedMeshData data, const MPIInfo &mpi,
                                                  bool buildFaces, bool buildEdges, const std::map<EntityKind, index> &generatedIdStarts)
    {
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native topology requires a communicator");
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !data.communication || data.communication->ChecksCollectivePreflight());
        data.LocalStage(mpi, [&]
                        {
            for (const auto &[kind, space] : data.spaces)
                DNDS_check_throw_info(space.ChecksCollectivePreflight() ==
                                          (!data.communication || data.communication->ChecksCollectivePreflight()),
                                      "owned topology entity preparation policy differs from its pipeline");
            DNDS_check_throw_info(data.dimension >= 2 && data.dimension <= 3, "native topology construction requires dimension two or three");
            DNDS_check_throw_info(!buildEdges || data.dimension == 3, "native edge completion currently requires three dimensions"); });
        std::array<index, 2> idStarts{-1, -1}, expectedStarts;
        data.LocalStage(mpi, [&]
                        {
            for (const auto &[kind,first] : generatedIdStarts)
            {
                DNDS_check_throw_info(first >= 0 && ((kind == EntityKind::Face && buildFaces) || (kind == EntityKind::Edge && buildEdges)),
                                      "generated ID floor requires a requested target family");
                DNDS_check_throw_info(!data.identity.nextIds.count(kind) || first >= data.identity.nextIds.at(kind),
                                      "explicit topology ID floor would reuse a previous ID");
                idStarts[kind == EntityKind::Face ? 0 : 1] = first;
            } });
        expectedStarts = idStarts;
        MPI_Bcast(expectedStarts.data(), 2, DNDS_MPI_INDEX, 0, mpi.comm);
        data.LocalStage(mpi, [&]
                        { DNDS_check_throw_info(idStarts == expectedStarts, "generated native ID floors differ across ranks"); });
        int flags[]{int(buildFaces), int(buildEdges)}, minimum[2], maximum[2];
        MPI_Allreduce(flags, minimum, 2, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(flags, maximum, 2, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(std::equal(minimum, minimum + 2, maximum), "native topology options differ between ranks");
        std::optional<NativeOwnedMeshData> previousStorage;
        data.LocalStage(mpi, [&]
                        { previousStorage.emplace(data); });
        const auto &previous = *previousStorage;
        std::map<EntityKind, RefreshMap> refresh;
        std::set<AdjacencyDescriptor> regenerated;
        data.LocalStage(mpi, [&]
                        {
            for (auto kind : {EntityKind::Face, EntityKind::Edge})
            {
                if (!(kind == EntityKind::Face ? buildFaces : buildEdges) || !data.spaces.count(kind)) continue;
                data.spaces.erase(kind);
                data.ids.erase(kind);
                const auto slots = kind == EntityKind::Face
                    ? std::vector<MeshRelation>{MeshRelation::CellFaces, MeshRelation::FaceNodes, MeshRelation::FaceCells,
                                                MeshRelation::FaceBoundaries, MeshRelation::BoundaryFaces, MeshRelation::CellFaceCells}
                    : std::vector<MeshRelation>{MeshRelation::CellEdges, MeshRelation::EdgeNodes, MeshRelation::EdgeCells};
                for (auto slot : slots) data.relations[size_t(slot)].reset();
                if (kind == EntityKind::Face)
                {
                    data.faceInfo.reset(); data.cellFacesPbi.reset(); data.faceNodesPbi.reset();
                }
                else
                {
                    data.edgeInfo.reset(); data.cellEdgesPbi.reset(); data.edgeNodesPbi.reset();
                }
            }
            // Reinsert affected named rows after exact source/target remapping.
            for (auto item = data.namedRelations.begin(); item != data.namedRelations.end();)
            {
                const auto path = item->first.Path();
                if ((buildFaces && std::find(path.begin(), path.end(), EntityKind::Face) != path.end()) ||
                    (buildEdges && std::find(path.begin(), path.end(), EntityKind::Edge) != path.end()))
                    item = data.namedRelations.erase(item);
                else ++item;
            } });
        auto closure = BuildNativeCellClosure(data, mpi);
        auto coneRead = closure.cellNodes.Read();
        const auto cells = coneRead.Source(), nodes = coneRead.Target();
        auto metadata = closure.cellInfo.ReadCombined();
        const auto cellTypes = metadata.View();
        std::optional<RelationPeriodicBits> periodic;
        if (closure.cellNodesPbi)
            periodic.emplace(coneRead, closure.cellNodesPbi->owned, closure.cellNodesPbi->ghost);
        auto owner = [mpi](const std::vector<index> &, const std::vector<MPI_int> &ranks, index)
        { return OwnershipDecision{*std::min_element(ranks.begin(), ranks.end()) == mpi.rank, {}}; };

        if (buildFaces)
        {
            auto faces = AdjacencyRegistry::Interpolate(coneRead, EntityKind::Face, Query(cellTypes, coneRead, false, periodic ? &*periodic : nullptr), owner, periodic ? &*periodic : nullptr);
            auto inverse = AdjacencyRegistry::Inverse(faces.parent2entity.Read(), mpi);
            auto support = inverse.Read();
            auto occurrences = faces.entity2parent.Read();
            std::optional<Array> physicalStorage;
            std::vector<t_index> zones(size_t(faces.entities.NumOwned()));
            data.LocalStage(mpi, [&]
                            {
                DNDS_check_throw_info(occurrences.State() == Adj_PointToGlobal && support.State() == Adj_PointToGlobal,
                                      "native face incidence requires global construction rows");
                physicalStorage.emplace(R::RowLayout::Dense(size_t(faces.entities.NumOwned()), 2));
                auto write = physicalStorage->Write();
                const auto output = write.View<R::FixedRows<2, 2>>();
                const auto occurrenceRows = occurrences.View(), supportRows = support.View();
                OMP::ForSetupRows(0, faces.entities.NumOwned(), [&](index face)
                {
                    const auto parents = occurrenceRows[face], complete = supportRows[face];
                    DNDS_check_throw_info(parents.size() >= 1 && parents.size() <= 2,
                                          "native face has missing or nonmanifold parent incidences");
                    const auto first = parents[0], second = parents.size() == 2 ? parents[1] : first;
                    const index uniqueCount = parents.size() == 2 && first != second ? 2 : 1;
                    DNDS_check_throw_info((data.periodic || parents.size() == uniqueCount) &&
                                              complete.size() == uniqueCount && complete[0] == std::min(first, second) &&
                                              (uniqueCount == 1 || complete[1] == std::max(first, second)),
                                          "native face has missing, repeated or nonmanifold parent incidences");
                    zones[size_t(face)] = parents.size() == 1 ? BC_ID_NULL : BC_ID_INTERNAL;
                    output(face, 0) = first;
                    output(face, 1) = parents.size() == 1 ? UnInitIndex : second;
                }); });
            auto physical = std::move(*physicalStorage);
            auto physicalRowsRead = physical.Read();
            const auto faceCells = physicalRowsRead.View<R::FixedRows<2, 2>>();

            // Resolve each independently owned boundary through its containing
            // cell's exact face slots. Candidate cells may be entirely remote.
            std::vector<R::RowRequest> requests;
            Array types(R::RowLayout::Dense(size_t(cells.NumOwned()), 1));
            data.LocalStage(mpi, [&]
                            {
                auto write = types.Write();
                OMP::ForSetupRows(0, cells.NumOwned(), [&](index cell)
                { write.View()(size_t(cell), 0) = cellTypes(size_t(cell), 0).type; });
                auto boundaryCells = data.relations[size_t(MeshRelation::BoundaryCells)]->Read();
                const auto boundaries = data.spaces.at(EntityKind::Bnd).NumOwned();
                requests.resize(size_t(boundaries));
                OMP::ForSetupRows(0, boundaries, [&](index boundary)
                {
                    DNDS_check_throw_info(boundaryCells.View().RowSize(size_t(boundary)) == 2,
                                          "native topology boundary must have two physical slots");
                    const auto cell = boundaryCells.View()(size_t(boundary), 0);
                    requests[size_t(boundary)] = {cells.Owner(cell), cells.OwnerLocalRow(cell), boundary};
                }); });
            auto context = data.communication ? data.communication : std::make_shared<R::CommunicationContext>(mpi.comm);
            const auto nBoundary = data.spaces.at(EntityKind::Bnd).NumOwned();
            R::RowTransferPlan candidates(context, size_t(cells.NumOwned()), size_t(nBoundary), requests);
            auto candidateFaces = Transfer(candidates, faces.parent2entity.Owned(), context);
            auto candidateNodes = Transfer(candidates, *data.relations[size_t(MeshRelation::CellNodes)], context);
            auto candidateTypes = Transfer(candidates, types, context);
            std::optional<Array> candidateBits, candidateFaceFrames;
            if (data.periodic)
            {
                candidateBits = Transfer(candidates, BitRows(*data.cellNodesPbi), context);
                candidateFaceFrames = Transfer(candidates, BitRows(*faces.parent2entityPbi), context);
            }
            std::vector<std::vector<index>> boundaryFaces(static_cast<size_t>(nBoundary));
            std::vector<std::vector<index>> assignments(size_t(mpi.size));
            struct BoundaryAssignment
            {
                int owner = -1;
                std::array<index, 5> payload{};
            };
            std::vector<BoundaryAssignment> boundaryAssignments(static_cast<size_t>(nBoundary));
            data.LocalStage(mpi, [&]
                            {
                auto foundFaces = candidateFaces.Read(), foundNodes = candidateNodes.Read(), foundTypes = candidateTypes.Read();
                auto boundaryNodes = data.relations[size_t(MeshRelation::BoundaryNodes)]->Read();
                auto boundaryCells = data.relations[size_t(MeshRelation::BoundaryCells)]->Read();
                auto boundaryInfo = data.bndInfo->Read();
                std::optional<R::Array<index>::ReadLease> parentBits, parentFaceFrames;
                std::optional<R::Array<NodePeriodicBits>::ReadLease> boundaryBits;
                if (data.periodic)
                {
                    parentBits.emplace(candidateBits->Read());
                    parentFaceFrames.emplace(candidateFaceFrames->Read());
                    boundaryBits.emplace(data.bndNodesPbi->Read());
                }
                OMP::ForSetupRows(0, nBoundary, [&](index boundary)
                {
                    const auto b = size_t(boundary);
                    const auto raw = foundNodes.View()[b];
                    std::vector<index> parent(raw.begin(), raw.end());
                    const auto rawBoundary = boundaryNodes.View()[b];
                    std::vector<index> expected(rawBoundary.begin(), rawBoundary.end());
                    std::sort(expected.begin(), expected.end());
                    NativePeriodicNodes expectedImages;
                    for (size_t slot = 0; slot < rawBoundary.size(); ++slot)
                        expectedImages.emplace_back(rawBoundary[slot], boundaryBits ? uint8_t(boundaryBits->View()(b, slot)) : 0);
                    Elem::Element element{static_cast<Elem::ElemType>(foundTypes.View()(b, 0))};
                    index match = UnInitIndex;
                    index matchFrame = 0;
                    for (t_index slot = 0; slot < element.GetNumFaces(); ++slot)
                    {
                        const auto face = element.ObtainFace(slot);
                        if (face.type != boundaryInfo.View()(b, 0).type) continue;
                        std::vector<index> actual(size_t(face.GetNumNodes()));
                        element.ExtractFaceNodes(slot, parent, actual);
                        std::vector<index> frameValues(parent.size(), 0), faceFrames(actual.size());
                        if (parentBits)
                            for (size_t i = 0; i < frameValues.size(); ++i) frameValues[i] = parentBits->View()(b, i);
                        element.ExtractFaceNodes(slot, frameValues, faceFrames);
                        NativePeriodicNodes actualImages;
                        for (size_t i = 0; i < actual.size(); ++i) actualImages.emplace_back(actual[i], uint8_t(faceFrames[i]));
                        auto delta = NativePeriodicFrame(std::move(actualImages), expectedImages);
                        if (delta && *delta == 0)
                        {
                            DNDS_check_throw_info(match == UnInitIndex, "boundary matches multiple element faces");
                            match = foundFaces.View()(b, size_t(slot));
                            if (parentFaceFrames) matchFrame = parentFaceFrames->View()(b, size_t(slot));
                        }
                    }
                    DNDS_check_throw_info(match != UnInitIndex && (FaceIDIsExternalBC(boundaryInfo.View()(b, 0).zone) ||
                                                                   (data.periodic && FaceIDIsPeriodic(boundaryInfo.View()(b, 0).zone))),
                                          "boundary does not match a physical element face in its frame");
                    boundaryFaces[b] = {match};
                    auto &assignment = boundaryAssignments[b];
                    assignment.owner = faces.entities.Owner(match);
                    assignment.payload = {faces.entities.OwnerLocalRow(match),
                                          data.spaces.at(EntityKind::Bnd).LocalToGlobal(boundary),
                                          boundaryInfo.View()(b, 0).zone,
                                          boundaryCells.View()(b, 0), matchFrame};
                });
                for (const auto &assignment : boundaryAssignments)
                {
                    DNDS_assert(assignment.owner >= 0 && assignment.owner < mpi.size);
                    auto &message = assignments[size_t(assignment.owner)];
                    message.insert(message.end(), assignment.payload.begin(), assignment.payload.end());
                } });
            auto received = Exchange(mpi, assignments, data.communication);
            std::vector<std::vector<index>> faceBoundaries(size_t(faces.entities.NumOwned()), {UnInitIndex});
            data.LocalStage(mpi, [&]
                            {
                struct Assignment { index boundary, cell, frame; t_index zone; };
                std::vector<std::vector<Assignment>> labels(faceBoundaries.size());
                auto messages = received.Read();
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    DNDS_check_throw_info(messages.View().RowSize(size_t(peer)) % 5 == 0, "malformed native face-boundary assignment");
                    for (size_t slot = 0; slot < messages.View().RowSize(size_t(peer)); slot += 5)
                    {
                        const auto face = messages.View()(size_t(peer), slot), boundary = messages.View()(size_t(peer), slot + 1), zone = messages.View()(size_t(peer), slot + 2);
                        DNDS_check_throw_info(face >= 0 && face < faces.entities.NumOwned() &&
                                                  data.spaces.at(EntityKind::Bnd).Owner(boundary) == peer && zone >= BC_ID_PERIODIC_3_DONOR && zone <= std::numeric_limits<t_index>::max(),
                                              "invalid native face-boundary owner or zone");
                        labels[size_t(face)].push_back({boundary, messages.View()(size_t(peer), slot + 3), messages.View()(size_t(peer), slot + 4), t_index(zone)});
                    }
                }
                for (size_t face = 0; face < labels.size(); ++face)
                {
                    const auto &entries = labels[face];
                    if (entries.empty()) continue;
                    const Assignment *selected = nullptr;
                    if (FaceIDIsPeriodic(entries.front().zone))
                    {
                        DNDS_check_throw_info(entries.size() == 2 && faceCells[face][1] != UnInitIndex &&
                                                  std::abs(int(entries[0].zone) - int(entries[1].zone)) == 3 &&
                                                  FaceIDIsPeriodic(entries[1].zone) && entries[0].frame != entries[1].frame,
                                              "periodic face requires one complementary main/donor boundary pair");
                        for (const auto &entry : entries)
                            if (entry.cell == faceCells[face][0] && entry.frame == 0) selected = &entry;
                        DNDS_check_throw_info(selected, "periodic boundary pair lacks the ordered face reference side");
                    }
                    else
                    {
                        DNDS_check_throw_info(entries.size() == 1 && FaceIDIsExternalBC(entries[0].zone) && faceCells[face][1] == UnInitIndex,
                                              "boundary face is internal or has duplicate boundary assignments");
                        selected = &entries.front();
                    }
                    faceBoundaries[face][0] = selected->boundary;
                    zones[face] = selected->zone;
                } });
            auto parentFaces = faces.parent2entity.Read();
            std::vector<index> remote;
            data.LocalStage(mpi, [&]
                            {
                std::vector<index> counts(size_t(parentFaces.NumOwned()));
                OMP::ForSetupRows(0, parentFaces.NumOwned(), [&](index cell)
                {
                    index count = 0;
                    for (auto face : parentFaces.GlobalRow(cell))
                        count += faces.entities.Owner(face) != mpi.rank;
                    counts[size_t(cell)] = count;
                });
                std::vector<index> starts(size_t(parentFaces.NumOwned()) + 1);
                const index total = OMP::ExclusivePrefixRows(
                    parentFaces.NumOwned(),
                    [&](index cell) { return counts[size_t(cell)]; },
                    [&](index cell, index offset) { starts[size_t(cell)] = offset; });
                remote.resize(size_t(total));
                OMP::ForSetupRows(0, parentFaces.NumOwned(), [&](index cell)
                {
                    index output = starts[size_t(cell)];
                    for (auto face : parentFaces.GlobalRow(cell))
                        if (faces.entities.Owner(face) != mpi.rank)
                            remote[size_t(output++)] = face;
                });
                std::sort(remote.begin(), remote.end());
                remote.erase(std::unique(remote.begin(), remote.end()), remote.end()); });
            auto closure = faces.entities.WithGhosts(std::move(remote));
            MeshAdjacency physicalRelation(Adj::Face2Cell, closure, cells, physical);
            EntityGhostMapping faceRoute(closure);
            auto physicalComplete = physicalRelation.WithGhostRows(faceRoute);
            auto physicalRead = physicalComplete.Read();
            std::vector<std::vector<index>> neighbors(size_t(cells.NumOwned()));
            data.LocalStage(mpi, [&]
                            {
                DNDS_check_throw_info(parentFaces.State() == Adj_PointToGlobal && physicalRead.State() == Adj_PointToGlobal,
                                      "native neighbor construction requires global face rows");
                const auto parentRows = parentFaces.View(), physicalRows = physicalRead.View();
                OMP::ForSetupRows(0, cells.NumOwned(), [&](index cell)
                {
                    const auto global = cells.LocalToGlobal(cell);
                    const auto facesOfCell = parentRows[cell];
                    auto &cellNeighbors = neighbors[size_t(cell)];
                    cellNeighbors.reserve(size_t(facesOfCell.size()));
                    for (auto face : facesOfCell)
                    {
                        const auto row = closure.GlobalToLocal(face);
                        DNDS_check_throw_info(row, "owned cell face missing from native local closure");
                        const auto pair = physicalRows[*row];
                        DNDS_check_throw_info(pair.size() == 2, "native physical face must have two parent slots");
                        DNDS_check_throw_info(pair[0] == global || pair[1] == global, "native physical face omits its parent cell");
                        cellNeighbors.push_back(pair[0] == global ? pair[1] : pair[0]);
                    }
                });
                data.relations[size_t(MeshRelation::CellFaces)] = faces.parent2entity.Owned();
                data.relations[size_t(MeshRelation::FaceNodes)] = faces.entity2node.Owned();
                data.relations[size_t(MeshRelation::FaceCells)] = physical;
                data.relations[size_t(MeshRelation::FaceBoundaries)] = BuildIndexRows(faceBoundaries, 1);
                data.relations[size_t(MeshRelation::BoundaryFaces)] = BuildIndexRows(boundaryFaces, 1);
                data.relations[size_t(MeshRelation::CellFaceCells)] = BuildIndexRows(neighbors);
                data.namedRelations.emplace(inverse.Kind(), inverse.Owned());
                data.faceInfo = Information(faces.entityElemInfo, zones);
                if (data.periodic)
                {
                    data.cellFacesPbi = *faces.parent2entityPbi;
                    data.faceNodesPbi = *faces.entity2nodePbi;
                }
                data.requirements.faces = true;
                data.requirements.cellFaceNeighbors = true;
                data.requirements.allFaceCellsLocal = true; });
            data.LocalStage(mpi, [&]
                            { regenerated.insert(inverse.Kind()); });
            if (previous.spaces.count(EntityKind::Face))
            {
                auto recovered = RecoverIdentity(mpi, previous, data, faces.entities,
                                                 MeshRelation::FaceNodes, *data.faceInfo, data.faceNodesPbi,
                                                 generatedIdStarts.count(EntityKind::Face) ? std::optional<index>{generatedIdStarts.at(EntityKind::Face)} : std::optional<index>{});
                data.LocalStage(mpi, [&]
                                { refresh.emplace(EntityKind::Face, std::move(recovered)); });
            }
            else
                data.LocalStage(mpi, [&]
                                { AddIdentity(data, faces.entities, generatedIdStarts); });
        }
        if (buildEdges)
        {
            auto edges = AdjacencyRegistry::Interpolate(coneRead, EntityKind::Edge, Query(cellTypes, coneRead, true, periodic ? &*periodic : nullptr), owner, periodic ? &*periodic : nullptr);
            auto inverse = AdjacencyRegistry::Inverse(edges.parent2entity.Read(), mpi);
            data.LocalStage(mpi, [&]
                            {
                data.relations[size_t(MeshRelation::CellEdges)] = edges.parent2entity.Owned();
                data.relations[size_t(MeshRelation::EdgeNodes)] = edges.entity2node.Owned();
                data.relations[size_t(MeshRelation::EdgeCells)] = inverse.Owned();
                data.namedRelations.emplace(inverse.Kind(), inverse.Owned());
                data.edgeInfo = Information(edges.entityElemInfo, std::vector<t_index>(size_t(edges.entities.NumOwned()), BC_ID_INTERNAL));
                if (data.periodic)
                {
                    data.cellEdgesPbi = *edges.parent2entityPbi;
                    data.edgeNodesPbi = *edges.entity2nodePbi;
                }
                data.requirements.edges = true; });
            data.LocalStage(mpi, [&]
                            { regenerated.insert(inverse.Kind()); });
            if (previous.spaces.count(EntityKind::Edge))
            {
                auto recovered = RecoverIdentity(mpi, previous, data, edges.entities,
                                                 MeshRelation::EdgeNodes, *data.edgeInfo, data.edgeNodesPbi,
                                                 generatedIdStarts.count(EntityKind::Edge) ? std::optional<index>{generatedIdStarts.at(EntityKind::Edge)} : std::optional<index>{});
                data.LocalStage(mpi, [&]
                                { refresh.emplace(EntityKind::Edge, std::move(recovered)); });
            }
            else
                data.LocalStage(mpi, [&]
                                { AddIdentity(data, edges.entities, generatedIdStarts); });
        }
        for (const auto &[key, original] : previous.namedRelations)
        {
            if (data.namedRelations.count(key) || regenerated.count(key))
                continue;
            // An explicit fixed-slot alias means the current physical facade
            // relation, whose first-parent orientation may have changed.
            // Regenerate these aliases instead of preserving an obsolete frame.
            bool fixedAlias = false;
            data.LocalStage(mpi, [&]
                            {
                for (const auto &[alias, slot] : std::initializer_list<std::pair<AdjacencyDescriptor, MeshRelation>>{
                    {Adj::Cell2Face, MeshRelation::CellFaces}, {Adj::Face2Node, MeshRelation::FaceNodes},
                    {Adj::Face2Cell, MeshRelation::FaceCells}, {Adj::Face2Bnd, MeshRelation::FaceBoundaries},
                    {Adj::Bnd2Face, MeshRelation::BoundaryFaces}, {Adj::Cell2CellFace, MeshRelation::CellFaceCells},
                    {Adj::Cell2Edge, MeshRelation::CellEdges}, {Adj::Edge2Node, MeshRelation::EdgeNodes},
                    {Adj::Edge2Cell, MeshRelation::EdgeCells}})
                    if (key == alias && data.relations[size_t(slot)])
                    {
                        data.namedRelations.emplace(key, *data.relations[size_t(slot)]);
                        fixedAlias = true;
                        break;
                    } });
            if (fixedAlias)
                continue;
            const auto path = key.Path();
            data.LocalStage(mpi, [&]
                            {
                for (auto kind : {EntityKind::Face, EntityKind::Edge})
                {
                    if (refresh.count(kind) && key.from != kind && key.to != kind &&
                        std::find(path.begin(), path.end(), kind) != path.end())
                    {
                        auto read = refresh.at(kind).oldToNew.Read();
                        for (size_t row = 0; row < refresh.at(kind).oldToNew.RowCount(); ++row)
                            DNDS_check_throw_info(read.View()(row, 0) != UnInitIndex,
                                                  "named relation through a removed topology key cannot be inferred");
                        auto reverse = refresh.at(kind).newToOld.Read();
                        for (size_t row = 0; row < refresh.at(kind).newToOld.RowCount(); ++row)
                            DNDS_check_throw_info(reverse.View()(row, 0) != UnInitIndex,
                                                  "named relation through an added topology key cannot be inferred");
                    }
                } });
            Array values = original;
            if (refresh.count(key.from))
                values = RefreshSourceRows(mpi, values, refresh.at(key.from));
            if (refresh.count(key.to))
                values = RefreshTargetRows(mpi, values, refresh.at(key.to), key.order == RelationOrder::SortedUnique);
            data.LocalStage(mpi, [&]
                            { data.namedRelations.emplace(key, std::move(values)); });
        }
        if (data.provenance)
            for (auto &[kind, values] : data.provenance->sourceIds)
            {
                if (refresh.count(kind))
                    values = RefreshSourceRows(mpi, values, refresh.at(kind));
            }
        return data;
    }
}
