#include "DNDS/MPI/Failure.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Connectivity/NativePeriodicMatch.hpp"
#include "Geom/Connectivity/ConnectivityInterpolateRows.hpp"
#include "Geom/Connectivity/ConnectivitySetupRows.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include <set>

namespace DNDS::Geom
{
    namespace
    {
        template <class T>
        Runtime::Array<T> BuildRows(const std::vector<std::vector<T>> &rows, size_t first, size_t count)
        {
            return Runtime::BuildArrayRows<T>(count, [&](size_t row)
                                              { return rows[first + row].size(); }, [&](size_t row, auto output)
                                              {
                const auto &values = rows[first + row];
                std::copy(values.begin(), values.end(), output.begin()); });
        }
        Runtime::Array<index> Transfer(const Runtime::RowTransferPlan &route, const Runtime::Array<index> &source,
                                       const std::shared_ptr<Runtime::CommunicationContext> &context)
        {
            Runtime::PayloadPlan payload(route, source.Layout(), Runtime::ElementSchema::Arithmetic<index>());
            Runtime::Array<index> output(payload.DestinationLayout());
            Runtime::TransferOptions options;
            options.mode = Runtime::TransferMode::Transient;
            options.context = context;
            payload.Bind(source, output, Runtime::TransferTransport::Direct, options).Begin().Wait();
            return output;
        }
        void SortUnique(std::vector<index> &values)
        {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        }
    }

    RelationPeriodicBits::RelationPeriodicBits(const MeshAdjacencyRead &relation, Runtime::Array<NodePeriodicBits> owned,
                                               std::optional<Runtime::Array<NodePeriodicBits>> ghost)
        : _source(relation.Source()), _target(relation.Target()), _kind(relation.Kind()),
          _owned(std::move(owned)), _ghost(std::move(ghost)), _ownedRead(_owned.Read())
    {
        DNDS_check_throw_info(_owned.GetDevice() == Runtime::Device::Host() &&
                                  (!_ghost || _ghost->GetDevice() == Runtime::Device::Host()),
                              "periodic relation setup requires host placement");
        _view.owned = _ownedRead->View();
        if (_ghost)
        {
            _ghostRead.emplace(_ghost->Read());
            _view.ghost = _ghostRead->View();
        }
        Validate(relation);
    }
    void RelationPeriodicBits::Validate(const MeshAdjacencyRead &relation) const
    {
        if (_meshRead)
            (void)_meshRead->HostView();
        DNDS_check_throw_info(_source.SameOrder(relation.Source()) && _target.SameOrder(relation.Target()) && _kind == relation.Kind(),
                              "periodic companion belongs to another relation/order");
        DNDS_check_throw_info(_view.owned.RowCount() == size_t(relation.NumOwned()) &&
                                  _view.owned.RowCount() + _view.ghost.RowCount() == size_t(relation.RowCount()),
                              "periodic companion row extent differs from relation");
        OMP::ForSetupRows(0, relation.RowCount(), [&](index row)
                          { DNDS_check_throw_info(_view.RowSize(size_t(row)) == relation.RowSize(row), "periodic companion slots differ from relation"); });
    }

    void AdjacencyRegistry::ValidateCollectively(const std::function<void()> &validate) const
    {
        if (!_policyAgreed)
        {
            Runtime::CommunicationDetail::AgreePolicy(_mpi.comm, !_communication || _communication->ChecksCollectivePreflight());
            _policyAgreed = true;
        }
        Runtime::CommunicationDetail::LocalStage(_mpi.comm, !_communication || _communication->ChecksCollectivePreflight(), [&]
                                                 {
            if (_communication)
                EntitySpace::CheckCommunicator(_mpi.comm, _communication->Communicator());
            validate(); }, "native connectivity setup validation failed on a rank");
    }
    void AdjacencyRegistry::Register(MeshAdjacencyRead relation)
    {
        relation.Validate();
        relation.CheckCommunicator(_mpi.comm);
        const auto kind = relation.Kind();
        if (_spaces.empty())
        {
            DNDS_check_throw_info(!_policyAgreed || relation.Source().ChecksCollectivePreflight() ==
                                                        (!_communication || _communication->ChecksCollectivePreflight()),
                                  "registry preparation policy is already frozen; use a fresh registry");
            _communication = relation.Source().Communication();
        }
        for (const auto *space : {&relation.Source(), &relation.Target()})
            DNDS_check_throw_info(space->ChecksCollectivePreflight() == (!_communication || _communication->ChecksCollectivePreflight()),
                                  "native setup registry entity preparation policies differ");
        DNDS_check_throw_info(!_relations.count(kind), "duplicate native setup relation");
        for (const auto *space : {&relation.Source(), &relation.Target()})
        {
            auto found = _spaces.find(space->Kind());
            DNDS_check_throw_info(found == _spaces.end() || found->second.SameOrder(*space),
                                  "native setup registry entity order mismatch");
        }
        _spaces.emplace(relation.Source().Kind(), relation.Source());
        _spaces.emplace(relation.Target().Kind(), relation.Target());
        _relations.emplace(kind, std::make_shared<MeshAdjacencyRead>(std::move(relation)));
    }
    std::shared_ptr<const MeshAdjacencyRead> AdjacencyRegistry::Find(const AdjacencyDescriptor &kind) const
    {
        kind.Validate();
        const auto found = _relations.find(kind);
        return found == _relations.end() ? nullptr : found->second;
    }

    Runtime::Array<index> AdjacencyRegistry::Fetch(const MeshAdjacencyRead &relation, const std::vector<index> &globals,
                                                   const Runtime::RowTransferPlan *reusedRoute,
                                                   std::shared_ptr<Runtime::CommunicationContext> context) const
    {
        std::vector<Runtime::RowRequest> requests;
        ValidateCollectively([&]
                             {
            relation.CheckActive();
            relation.Source().Validate();
            relation.Target().Validate();
            auto &scratch = _fetchedRows[relation.Kind()];
            scratch.globals = globals;
            scratch.values.reset();
            if (!_fetchValues.count(relation.Kind()))
            {
                // Registration retains immutable input leases. Share globally
                // indexed native storage; convert local numbering once per
                // relation, rather than rebuilding every owned row per hop.
                const auto *owned = dynamic_cast<const MeshAdjacencyRead::ArrayHold *>(relation._owned.get());
                if (relation.State() == Adj_PointToGlobal && owned)
                    _fetchValues.emplace(relation.Kind(), owned->array);
                else
                {
                    const auto *mesh = dynamic_cast<const MeshAdjacencyRead::MeshHold *>(relation._owned.get());
                    DNDS_check_throw_info(owned || mesh, "adjacency fetch requires retained storage metadata");
                    const auto &shape = owned ? owned->read.Layout() : mesh->ownedLayout;
                    _fetchValues.emplace(relation.Kind(), Runtime::BuildArrayRows<index>(shape,
                        [&](size_t row, auto output)
                        {
                            relation.ConvertRowTo(index(row), Adj_PointToGlobal, output.data());
                        }, true));
                }
            }
            if (!reusedRoute)
                for (size_t row = 0; row < globals.size(); ++row)
                    requests.push_back({relation.Source().Owner(globals[row]),
                                        relation.Source().OwnerLocalRow(globals[row]), row}); });
        const auto &values = _fetchValues.at(relation.Kind());
        const auto retain = [&](Runtime::Array<index> fetched)
        {
            auto &scratch = _fetchedRows.at(relation.Kind());
            scratch.values.emplace(std::move(fetched));
            scratch.version = scratch.values->Version();
            return *scratch.values;
        };
        if (reusedRoute)
        {
            DNDS_check_throw_info(context, "reused native fetch route requires its communication context");
            return retain(Transfer(*reusedRoute, values, context));
        }
        context = _communication ? _communication : std::make_shared<Runtime::CommunicationContext>(_mpi.comm);
        Runtime::RowTransferPlan route(context, size_t(relation.NumOwned()), globals.size(), requests,
                                       {relation.Source().Stamp(), relation.Target().Stamp()});
        return retain(Transfer(route, values, context));
    }
    GhostResult AdjacencyRegistry::Evaluate(const CompiledGhostTree &tree) const
    {
        // Compare the compiled schedule before entering its per-hop collectives.
        // Equal schedules are a collective caller requirement, checked here.
        const auto &schedule = tree.Signature();
        int length = _mpi.rank == 0 ? (schedule.size() <= INT_MAX ? int(schedule.size()) : -1) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, _mpi.comm);
        DNDS_check_throw_info(length >= 0, "ghost program signature exceeds MPI count range");
        std::string canonical(size_t(length), ' ');
        if (_mpi.rank == 0)
            canonical = schedule;
        MPI_Bcast(canonical.data(), length, MPI_CHAR, 0, _mpi.comm);
        ValidateCollectively([&]
                             {
            DNDS_check_throw_info(schedule == canonical, "ghost traversal schedule differs between ranks");
            for (auto kind : tree.requiredAdjs())
            {
                auto found = _relations.find(kind);
                DNDS_check_throw_info(found != _relations.end(), "native ghost traversal relation is missing");
                found->second->Validate();
            } });
        if (tree.Levels().empty())
            return {};
        std::vector<std::vector<index>> sets(size_t(tree.NodeCount()));
        for (const auto &root : tree.Levels().front())
        {
            const auto &space = _spaces.at(root.kind);
            auto &output = sets[size_t(root.nodeId)];
            output.resize(size_t(space.NumOwned()));
            const auto first = (*space._global)(_mpi.rank, 0);
            OMP::ForSetupRows(0, index(output.size()), [&](index iRow)
                              { output[size_t(iRow)] = first + iRow; });
        }
        GhostResult result;
        for (int level = 0; level <= tree.MaxLevel(); ++level)
        {
            for (const auto &entry : tree.Levels()[size_t(level)])
            {
                if (!entry.collect)
                    continue;
                const auto &space = _spaces.at(entry.kind);
                auto &output = result.ghostIndices[entry.kind];
                // Registered adjacency scopes retain the numbering leases.
                // Collected values are validated global IDs; identifying the
                // remote subset only needs the owned rank-offset interval.
                const auto firstOwned = space._global->ROffsets()[size_t(_mpi.rank)];
                const auto endOwned = firstOwned + space.NumOwned();
                for (auto global : sets[size_t(entry.nodeId)])
                    if (global < firstOwned || global >= endOwned)
                        output.push_back(global);
                SortUnique(output);
            }
            if (level == tree.MaxLevel())
                break;
            const auto &hops = tree.Levels()[size_t(level + 1)];
            struct PendingHop
            {
                size_t entry, relation;
                std::vector<index> remote;
                index targetBegin = 0, targetEnd = 0;
                bool retainOwnedTargets = true;
            };
            std::vector<PendingHop> pending(hops.size());
            std::vector<size_t> hopRelation(hops.size(), SIZE_MAX);
            std::vector<const MeshAdjacencyRead *> relations;
            int valid = 1;
            try
            {
                std::vector<uint8_t> seenNodes(size_t(tree.NodeCount()), 0);
                for (size_t hopEntry = 0; hopEntry < hops.size(); ++hopEntry)
                {
                    const auto &entry = hops[hopEntry];
                    DNDS_check_throw_info(size_t(entry.nodeId) < seenNodes.size() && !seenNodes[size_t(entry.nodeId)],
                                          "native ghost traversal repeats a child node in one level");
                    seenNodes[size_t(entry.nodeId)] = 1;
                    const auto &relation = *_relations.at(entry.hop);
                    size_t relationSlot = 0;
                    while (relationSlot < relations.size() && relations[relationSlot] != &relation)
                        ++relationSlot;
                    if (relationSlot == relations.size())
                        relations.push_back(&relation);
                    hopRelation[hopEntry] = relationSlot;
                }
                // Expand rows within each exact branch. Parallelizing only the
                // small number of branches left a large single-hop recipe serial.
                // No worker mutates a registry, route or another branch frontier.
                for (size_t hopEntry = 0; hopEntry < hops.size(); ++hopEntry)
                {
                    const auto &entry = hops[hopEntry];
                    const auto &relation = *relations[hopRelation[hopEntry]];
                    relation.CheckActive();
                    auto &candidate = pending[hopEntry];
                    candidate.entry = hopEntry;
                    candidate.relation = hopRelation[hopEntry];
                    const auto &targetSpace = relation.Target();
                    candidate.targetBegin = targetSpace._global->ROffsets()[size_t(_mpi.rank)];
                    candidate.targetEnd = candidate.targetBegin + targetSpace.NumOwned();
                    candidate.retainOwnedTargets = entry.hasChildren;
                    auto &output = sets[size_t(entry.nodeId)];
                    const auto &parentValues = sets[size_t(entry.parentId)];
                    const auto &sourceView = relation._view.owned;
                    const size_t sourceElements = sourceView.offsets ? sourceView.offsets[sourceView.RowCount()]
                                                                     : sourceView.RowCount() * sourceView.stride;
                    const size_t averageRow = sourceView.RowCount()
                                                  ? std::max<size_t>(1, (sourceElements + sourceView.RowCount() - 1) /
                                                                            sourceView.RowCount())
                                                  : 1;
                    // Each branch starts empty. Reserve for local and remote
                    // expansions once; SortUnique later owns the exact shape.
                    const size_t reserveRows = CheckedSize::Multiply(parentValues.size(),
                                                                     CheckedSize::Multiply(averageRow, size_t(2)));
                    output.reserve(CheckedSize::Add(output.size(), reserveRows));
                    const auto &source = relation.Source();
                    const auto first = (*source._global)(_mpi.rank, 0);
                    const bool globalState = relation._state == Adj_PointToGlobal;
                    // Local incidence is already retained and validated. A DSL
                    // traversal needs its union, not another transferred copy
                    // of every local row. Only remote parents need a route.
                    ConnectivityRows::AppendSetupRows(index(parentValues.size()), output, [&](index iRow, const auto &emit)
                                                      {
                            const index global = parentValues[size_t(iRow)];
                            const auto appendTarget = [&](index value)
                            {
                                // Terminal nodes collect only remote targets;
                                // intermediates retain owned values for descendants.
                                if (value != UnInitIndex && (candidate.retainOwnedTargets ||
                                                            value < candidate.targetBegin || value >= candidate.targetEnd))
                                    emit(value);
                            };
                            if (global >= first && global - first < source.NumOwned())
                            {
                                const auto row = global - first;
                                if (globalState)
                                {
                                    const auto &view = relation._view.owned;
                                    const auto count = view.RowSize(size_t(row));
                                    const auto *values = count ? view.Row(size_t(row)) : nullptr;
                                    for (size_t slot = 0; slot < count; ++slot)
                                        appendTarget(values[slot]);
                                }
                                else
                                {
                                    for (size_t slot = 0; slot < relation.RowSize(row); ++slot)
                                    {
                                        const auto value = relation.ConvertedIndex(row, slot, Adj_PointToGlobal);
                                        appendTarget(value);
                                    }
                                }
                            } });
                    ConnectivityRows::AppendSetupRows(index(parentValues.size()), candidate.remote, [&](index iRow, const auto &emit)
                                                      {
                            const index global = parentValues[size_t(iRow)];
                            if (global < first || global - first >= source.NumOwned())
                                emit(global); });
                }
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("mesh adjacency preparation failed");
                valid = 0;
            }
            Runtime::CommunicationDetail::Preflight(valid, _mpi.comm, !_communication || _communication->ChecksCollectivePreflight(), "native ghost traversal local expansion failed on a rank");
            DNDS_check_throw_info(valid, "native ghost traversal local expansion failed on a rank");

            std::vector<std::vector<index>> requested(relations.size());
            for (const auto &hop : pending)
            {
                auto &target = requested[hop.relation];
                target.insert(target.end(), hop.remote.begin(), hop.remote.end());
            }
            std::vector<MPI_int> active(relations.size(), 0);
            for (size_t relationSlot = 0; relationSlot < relations.size(); ++relationSlot)
            {
                SortUnique(requested[relationSlot]);
                active[relationSlot] = requested[relationSlot].empty() ? 0 : 1;
            }
            if (!active.empty())
                MPI_Allreduce(MPI_IN_PLACE, active.data(), int(active.size()), MPI_INT, MPI_MAX, _mpi.comm);
            std::vector<std::vector<size_t>> parentIds(relations.size());
            for (const auto &hop : pending)
                parentIds[hop.relation].push_back(hops[hop.entry].parentId);
            for (auto &ids : parentIds)
            {
                std::sort(ids.begin(), ids.end());
                ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            }
            struct FetchGroup
            {
                EntityKind sourceKind;
                size_t parent;
                std::vector<size_t> relationSlots;
                std::shared_ptr<Runtime::CommunicationContext> context;
                std::optional<Runtime::RowTransferPlan> route;
            };
            std::vector<FetchGroup> groups;
            std::vector<size_t> relationGroup(relations.size(), SIZE_MAX);
            for (size_t relationSlot = 0; relationSlot < relations.size(); ++relationSlot)
                if (active[relationSlot])
                {
                    const auto sourceKind = relations[relationSlot]->Source().Kind();
                    const auto parent = parentIds[relationSlot].size() == 1 ? parentIds[relationSlot].front() : SIZE_MAX;
                    size_t groupSlot = 0;
                    if (parent != SIZE_MAX)
                        while (groupSlot < groups.size() &&
                               (groups[groupSlot].sourceKind != sourceKind || groups[groupSlot].parent != parent))
                            ++groupSlot;
                    else
                        groupSlot = groups.size();
                    if (groupSlot == groups.size())
                        groups.push_back({sourceKind, parent, {}, {}, {}});
                    groups[groupSlot].relationSlots.push_back(relationSlot);
                    relationGroup[relationSlot] = groupSlot;
                }
            for (auto &group : groups)
            {
                const auto &model = *relations[group.relationSlots.front()];
                const auto &source = model.Source();
                std::vector<Runtime::RowRequest> requests;
                requests.reserve(requested[group.relationSlots.front()].size());
                for (size_t row = 0; row < requested[group.relationSlots.front()].size(); ++row)
                    requests.push_back({source.Owner(requested[group.relationSlots.front()][row]),
                                        source.OwnerLocalRow(requested[group.relationSlots.front()][row]), row});
                group.context = _communication ? _communication : std::make_shared<Runtime::CommunicationContext>(_mpi.comm);
                group.route.emplace(group.context, size_t(model.NumOwned()),
                                    requested[group.relationSlots.front()].size(), requests,
                                    std::vector<Runtime::VersionStamp>{source.Stamp()});
            }
            std::vector<std::optional<Runtime::Array<index>>> fetched(relations.size());
            std::vector<std::optional<Runtime::Array<index>::ReadLease>> fetchViews(relations.size());
            for (size_t relationSlot = 0; relationSlot < relations.size(); ++relationSlot)
                if (active[relationSlot])
                {
                    const auto &group = groups[relationGroup[relationSlot]];
                    // Branches from one parent frontier reuse the owner route;
                    // their typed payloads remain distinct.
                    fetched[relationSlot].emplace(Fetch(*relations[relationSlot], requested[relationSlot],
                                                        &*group.route, group.context));
                    fetchViews[relationSlot].emplace(fetched[relationSlot]->Read());
                }
            ValidateCollectively([&]
                                 {
            for (size_t hopIndex = 0; hopIndex < pending.size(); ++hopIndex)
            {
                const auto &hop = pending[size_t(hopIndex)];
                const auto &entry = hops[hop.entry];
                auto &output = sets[size_t(entry.nodeId)];
                if (!hop.remote.empty())
                {
                    const auto &fetchView = fetchViews[hop.relation]->View();
                    ConnectivityRows::AppendSetupRows(index(hop.remote.size()), output, [&](index iRow, const auto &emit)
                    {
                        const index global = hop.remote[size_t(iRow)];
                        const auto found = std::lower_bound(requested[hop.relation].begin(),
                                                            requested[hop.relation].end(), global);
                        DNDS_check_throw_info(found != requested[hop.relation].end() && *found == global,
                                              "native ghost traversal lost a remote source row");
                        const auto row = size_t(found - requested[hop.relation].begin());
                        for (size_t slot = 0; slot < fetchView.RowSize(row); ++slot)
                        {
                            const auto value = fetchView(row, slot);
                            if (value != UnInitIndex && (hop.retainOwnedTargets ||
                                                         value < hop.targetBegin || value >= hop.targetEnd))
                                emit(value);
                        }
                    });
                }
                SortUnique(output);
            } });
        }
        int mask = 0;
        for (auto it = result.ghostIndices.begin(); it != result.ghostIndices.end();)
            if (it->second.empty())
                it = result.ghostIndices.erase(it);
            else
            {
                mask |= 1 << int(it->first);
                ++it;
            }
        MPI_Allreduce(MPI_IN_PLACE, &mask, 1, MPI_INT, MPI_BOR, _mpi.comm);
        for (int kind = 0; kind < int(EntityKind::NUM_KINDS); ++kind)
            if (mask & (1 << kind))
                result.activeKinds.insert(EntityKind(kind));
        return result;
    }

    EntitySpace AdjacencyRegistry::MakeSpace(EntityKind kind, index owned, const EntitySpace &anchor)
    {
        auto global = std::make_shared<GlobalOffsetsMapping>();
        global->setMPIAlignBcast(anchor._mpi, owned);
        std::vector<index> empty;
        auto ghosts = std::make_shared<OffsetAscendIndexMapping>((*global)(anchor._mpi.rank, 0), owned, empty, *global, anchor._mpi);
        Runtime::VersionTracker versions;
        EntitySpace result(kind, anchor._mpi, owned, 0, anchor._identity, versions.Stamp(), global, ghosts, versions);
        result._publishedMaps = true;
        result._communication = anchor._communication;
        return result;
    }

    NativeInterpolation AdjacencyRegistry::Interpolate(const MeshAdjacencyRead &cone, EntityKind entityKind,
                                                       const SubEntityQueryPbi &query, const OwnershipResolverMulti &resolver,
                                                       const RelationPeriodicBits *periodic)
    {
        const auto mpi = cone.MPI();
        AdjacencyRegistry setup(mpi);
        setup._communication = cone.Source().Communication();
        std::optional<ConnectivityRows::SubEntityRows> rows;
        std::vector<std::vector<NodePeriodicBits>> nodeBits, parentBits;

        std::vector<bool> owned;
        std::vector<index> parentGlobal;
        std::vector<MPI_int> parentOwner;
        setup.ValidateCollectively([&]
                                   {
            cone.Validate();
            DNDS_check_throw_info(cone.Target().Kind() == EntityKind::Node && entityKind != cone.Source().Kind() &&
                                      (entityKind == EntityKind::Face || entityKind == EntityKind::Edge) && bool(query.numSubEntities) && bool(query.describe) &&
                                      bool(query.extractNodes) && bool(resolver), "invalid native interpolation query/kinds");
            DNDS_check_throw_info(cone.RowCount() == cone.Source().NumLocal(), "interpolation requires every parent ghost row");
            if (periodic)
            {
                periodic->Validate(cone);
                DNDS_check_throw_info(bool(query.extractPbi), "periodic interpolation requires ordered bit extraction");
            }
            for (index row = 0; row < cone.RowCount(); ++row)
            {
                DNDS_check_throw_info(query.numSubEntities(row) >= 0, "negative subentity count");
                for (int sub = 0; sub < query.numSubEntities(row); ++sub)
                {
                    const auto desc = query.describe(row, sub);
                    DNDS_check_throw_info(desc.nVertices > 0 && desc.nVertices <= desc.nNodes && desc.nNodes <= 10,
                                          "invalid subentity node/vertex count");
                }
            }
            // Relation construction already validated raw encodings under the
            // retained source/target leases. Interpolation additionally needs
            // every target present in the local closure; check that condition
            // directly instead of re-entering public random access twice per
            // node. This is the same predicate as ReadEntry().unavailable.
            const auto requireLocalTarget = [&](const Runtime::ArrayView<const index> &view)
            {
                OMP::ForSetupRows(0, view.RowCount(), [&](index row)
                {
                    const auto count = view.RowSize(row);
                    const auto *values = count ? view.Row(row) : nullptr;
                    for (size_t slot = 0; slot < count; ++slot)
                    {
                        const auto value = values[slot];
                        DNDS_check_throw_info(value != UnInitIndex,
                                              "interpolation requires local node closure");
                        const index global = cone._state == Adj_PointToLocal && value >= 0
                                                 ? cone._target.LocalToGlobalPinned(value)
                                                 : (cone._state == Adj_PointToLocal ? -1 - value : value);
                        DNDS_check_throw_info(cone._target.GlobalToLocalPinned(global),
                                              "interpolation requires local node closure");
                    }
                });
            };
            requireLocalTarget(cone._view.owned);
            requireLocalTarget(cone._view.ghost);
            const auto localIndex = [&](index row, size_t slot)
            {
                const auto count = cone._view.RowSize(size_t(row));
                DNDS_check_throw_info(slot < count, "interpolation node slot outside cone");
                const auto value = cone._view.Row(size_t(row))[slot];
                if (cone._state == Adj_PointToLocal)
                {
                    DNDS_check_throw_info(value >= 0, "interpolation requires local node closure");
                    return value;
                }
                const auto local = cone._target.GlobalToLocalPinned(value);
                DNDS_check_throw_info(local.has_value(), "interpolation requires local node closure");
                return *local;
            };
            const auto globalIndex = [&](index row, size_t slot)
            {
                const auto count = cone._view.RowSize(size_t(row));
                DNDS_check_throw_info(slot < count, "interpolation node slot outside cone");
                const auto value = cone._view.Row(size_t(row))[slot];
                return cone._state == Adj_PointToLocal && value >= 0
                           ? cone._target.LocalToGlobalPinned(value)
                           : (cone._state == Adj_PointToLocal ? -1 - value : value);
            };
            // Extract each parent-node mapping once. Face enumeration revisits
            // a vertex for every adjacent subentity and candidate match. The
            // retained cone leases keep this local-sized cache valid throughout
            // enumeration; periodic identity still uses the original globals.
            std::vector<size_t> localStarts(size_t(cone.RowCount()) + 1, 0);
            std::vector<index> localNodes;
            for (index iRow = 0; iRow < cone.RowCount(); ++iRow)
                localStarts[size_t(iRow) + 1] = CheckedSize::Add(localStarts[size_t(iRow)], size_t(cone._view.RowSize(iRow)));
            localNodes.resize(localStarts.back());
            OMP::ForSetupRows(0, cone.RowCount(), [&](index iRow)
            {
                for (index iSlot = 0; iSlot < cone._view.RowSize(iRow); ++iSlot)
                    localNodes[localStarts[size_t(iRow)] + size_t(iSlot)] = localIndex(iRow, size_t(iSlot));
            });
            SubEntityQueryPbi enumeration = query;
            if (periodic)
                enumeration.matchExtra = [&](index parent, int sub, index entity, index originalParent, int originalSub)
                {
                    if (query.matchExtra && !query.matchExtra(parent, sub, entity, originalParent, originalSub)) return false;
                    auto extract = [&](index p, int slot)
                    {
                        const auto desc = query.describe(p, slot);
                        std::vector<index> nodes(size_t(desc.nNodes));
                        std::vector<NodePeriodicBits> bits(size_t(desc.nNodes));
                        query.extractNodes(p, slot, [&](int j) { return globalIndex(p, size_t(j)); }, nodes.data());
                        query.extractPbi(p, slot, [&](int j) { return (*periodic)(p, size_t(j)); }, bits.data());
                        std::vector<std::pair<index, uint8_t>> result;
                        for (size_t j = 0; j < nodes.size(); ++j) result.emplace_back(nodes[j], uint8_t(bits[j]));
                        return result;
                    };
                    auto current = extract(parent, sub), original = extract(originalParent, originalSub);
                    return bool(NativePeriodicFrame(std::move(current), std::move(original)));
                };
            rows.emplace(ConnectivityRows::EnumerateSubEntities([&](index row, int slot)
            {
                const auto first = localStarts[size_t(row)], end = localStarts[size_t(row) + 1];
                DNDS_check_throw_info(slot >= 0 && size_t(slot) < end - first, "node extraction slot outside cone");
                return localNodes[first + size_t(slot)];
            }, enumeration, cone.RowCount(), cone.Target().NumLocal()));
            nodeBits.resize(size_t(rows->nEntities));
            parentBits.resize(size_t(cone.RowCount()));
            if (periodic)
            {
                for (index parent = 0; parent < cone.RowCount(); ++parent)
                {
                    parentBits[size_t(parent)].resize(rows->parent2entity[size_t(parent)].size());
                    for (size_t sub = 0; sub < rows->parent2entity[size_t(parent)].size(); ++sub)
                    {
                        auto entity = rows->parent2entity[size_t(parent)][sub];
                        auto desc = query.describe(parent, int(sub));
                        std::vector<NodePeriodicBits> bits(size_t(desc.nNodes));
                        std::vector<index> nodes(size_t(desc.nNodes));
                        query.extractPbi(parent, int(sub), [&](int slot)
                        {
                            DNDS_check_throw_info(slot >= 0 && size_t(slot) < cone.RowSize(parent), "periodic extraction slot outside cone");
                            return (*periodic)(parent, size_t(slot));
                        }, bits.data());
                        query.extractNodes(parent, int(sub), [&](int slot)
                        { return localIndex(parent, size_t(slot)); }, nodes.data());
                        auto &stored = nodeBits[size_t(entity)];
                        if (stored.empty()) stored = bits;
                        const auto &entityNodes = rows->entity2node[size_t(entity)];
                        DNDS_check_throw_info(entityNodes.size() == nodes.size(), "periodic subentity node count mismatch");
                        std::vector<std::pair<index, uint8_t>> current, original;
                        for (size_t node = 0; node < nodes.size(); ++node)
                        {
                            current.emplace_back(nodes[node], uint8_t(bits[node]));
                            original.emplace_back(entityNodes[node], uint8_t(stored[node]));
                        }
                        // Repeated periodic node IDs are distinguished by their
                        // per-node frame bits, matching the legacy sorted-pair math.
                        auto uniform = NativePeriodicFrame(std::move(current), std::move(original));
                        DNDS_check_throw_info(uniform, "periodic subentity has nonuniform frame transform");
                        parentBits[size_t(parent)][sub] = NodePeriodicBits{*uniform};
                    }
                }
            }
            parentGlobal.resize(size_t(cone.RowCount()));
            parentOwner.resize(size_t(cone.RowCount()));
            for (index parent = 0; parent < cone.RowCount(); ++parent)
            {
                parentGlobal[size_t(parent)] = cone.Source().LocalToGlobal(parent);
                parentOwner[size_t(parent)] = cone.Source().Owner(parentGlobal[size_t(parent)]);
            }
            owned.resize(size_t(rows->nEntities));
            std::vector<MPI_int> ranks;
            ranks.reserve(4);
            for (index entity = 0; entity < rows->nEntities; ++entity)
            {
                const auto &parents = rows->entity2parent[size_t(entity)];
                ranks.clear();
                bool anyOwned = false, anyGhost = false;
                for (auto parent : parents)
                {
                    ranks.push_back(parentOwner[size_t(parent)]);
                    anyOwned = anyOwned || parent < cone.NumOwned();
                    anyGhost = anyGhost || parent >= cone.NumOwned();
                }
                owned[size_t(entity)] = anyOwned && (!anyGhost || resolver(parents, ranks, cone.NumOwned()).owned);
            } });
        int entityKinds[]{int(entityKind), -int(entityKind)};
        MPI_Allreduce(MPI_IN_PLACE, entityKinds, 2, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(entityKinds[0] == -entityKinds[1], "interpolation entity kind differs between ranks");
        int periodicFlags[]{int(bool(periodic)), -int(bool(periodic))};
        MPI_Allreduce(MPI_IN_PLACE, periodicFlags, 2, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(periodicFlags[0] == -periodicFlags[1], "periodic interpolation differs between ranks");
        std::vector<index> oldToOwned(size_t(rows->nEntities), UnInitIndex);
        index nOwned = 0;
        for (index entity = 0; entity < rows->nEntities; ++entity)
            if (owned[size_t(entity)])
                oldToOwned[size_t(entity)] = nOwned++;
        auto entities = MakeSpace(entityKind, nOwned, cone.Source());
        std::vector<std::vector<index>> e2n(static_cast<size_t>(nOwned)), e2p(static_cast<size_t>(nOwned));
        std::vector<std::vector<NodePeriodicBits>> e2nBits(static_cast<size_t>(nOwned));
        std::vector<std::vector<ElemInfo>> info(static_cast<size_t>(nOwned));
        std::vector<std::vector<index>> messages(size_t(mpi.size));
        std::vector<index> nodeGlobal(size_t(cone.Target().NumLocal()));
        setup.ValidateCollectively([&]
                                   {
        OMP::ForSetupRows(0, cone.Target().NumLocal(), [&](index iNode)
                          { nodeGlobal[size_t(iNode)] = cone.Target().LocalToGlobalPinned(iNode); });
        OMP::ForSetupRows(0, rows->nEntities, [&](index entity)
        {
            auto output = oldToOwned[size_t(entity)];
            if (output == UnInitIndex)
                return;
            for (auto node : rows->entity2node[size_t(entity)])
                e2n[size_t(output)].push_back(nodeGlobal[size_t(node)]);
            for (auto parent : rows->entity2parent[size_t(entity)])
                e2p[size_t(output)].push_back(parentGlobal[size_t(parent)]);
            e2nBits[size_t(output)] = nodeBits[size_t(entity)];
            info[size_t(output)].push_back(rows->entityElemInfo[size_t(entity)]);
        }); });
        // Peer batches preserve entity/parent/slot order. The shared per-peer
        // append and ownership numbering stay on the controller.
        for (index entity = 0; entity < rows->nEntities; ++entity)
        {
            const auto output = oldToOwned[size_t(entity)];
            if (output == UnInitIndex)
                continue;
            const auto global = entities.LocalToGlobal(output);
            for (auto parent : rows->entity2parent[size_t(entity)])
            {
                if (parent >= cone.NumOwned())
                {
                    const auto &slots = rows->parent2entity[size_t(parent)];
                    for (size_t slot = 0; slot < slots.size(); ++slot)
                        if (slots[slot] == entity)
                        {
                            auto &message = messages[size_t(parentOwner[size_t(parent)])];
                            message.insert(message.end(), {parentGlobal[size_t(parent)], global, index(slot), periodic ? index(uint8_t(parentBits[size_t(parent)][slot])) : 0});
                        }
                }
            }
        }
        // One ragged batch per peer, owner-directed and direct typed MPI.
        std::vector<int> sendCounts(size_t(mpi.size)), receiveCounts(size_t(mpi.size));
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            DNDS_check_throw_info(messages[size_t(peer)].size() <= size_t(INT_MAX),
                                  "interpolation owner message exceeds MPI count range");
            sendCounts[size_t(peer)] = int(messages[size_t(peer)].size());
        }
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT, receiveCounts.data(), 1, MPI_INT, mpi.comm);
        std::vector<int> sendOffsets(size_t(mpi.size) + 1, 0), receiveOffsets(size_t(mpi.size) + 1, 0);
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            sendOffsets[size_t(peer + 1)] = sendOffsets[size_t(peer)] + sendCounts[size_t(peer)];
            receiveOffsets[size_t(peer + 1)] = receiveOffsets[size_t(peer)] + receiveCounts[size_t(peer)];
        }
        std::vector<index> outgoing(size_t(sendOffsets.back())), incoming(size_t(receiveOffsets.back()));
        for (int peer = 0; peer < mpi.size; ++peer)
            std::copy(messages[size_t(peer)].begin(), messages[size_t(peer)].end(),
                      outgoing.begin() + sendOffsets[size_t(peer)]);
        MPI_Alltoallv(outgoing.data(), sendCounts.data(), sendOffsets.data(), DNDS_MPI_INDEX,
                      incoming.data(), receiveCounts.data(), receiveOffsets.data(), DNDS_MPI_INDEX, mpi.comm);
        std::vector<std::vector<index>> p2e(size_t(cone.RowCount()));
        for (index parent = 0; parent < cone.RowCount(); ++parent)
            for (auto entity : rows->parent2entity[size_t(parent)])
            {
                const auto output = oldToOwned[size_t(entity)];
                p2e[size_t(parent)].push_back(output == UnInitIndex ? UnInitIndex : entities.LocalToGlobal(output));
            }
        setup.ValidateCollectively([&]
                                   {
            for (size_t peer = 0; peer < size_t(mpi.size); ++peer)
            {
                const auto first = receiveOffsets[peer], end = receiveOffsets[peer + 1];
                DNDS_check_throw_info((end - first) % 4 == 0, "malformed interpolation owner message");
                for (int slot = first; slot < end; slot += 4)
                {
                    const auto parent = cone.Source().GlobalToLocal(incoming[size_t(slot)]);
                    const auto entity = incoming[size_t(slot + 1)], sub = incoming[size_t(slot + 2)];
                    DNDS_check_throw_info(parent && *parent < cone.NumOwned() && entity >= 0 && entity < entities.NumGlobal() &&
                                              entities.Owner(entity) == int(peer) && sub >= 0 && size_t(sub) < p2e[size_t(*parent)].size(),
                                          "interpolation owner message outside entity domain");
                    auto &destination = p2e[size_t(*parent)][size_t(sub)];
                    DNDS_check_throw_info(destination == UnInitIndex || destination == entity, "conflicting interpolation owners");
                    destination = entity;
                    if (periodic)
                    {
                        // The generated entity keeps its OWNER's first-parent
                        // frame. A peer's rank-local first parent can have a
                        // different frame, so its pre-ownership XOR is not the
                        // published parent->entity transform. Send/install the
                        // owner's XOR with the entity ID (legacy triplets sent
                        // only the ID and could leave wrong cross-rank flags).
                        const auto bits = incoming[size_t(slot + 3)];
                        DNDS_check_throw_info(bits >= 0 && bits <= 255, "invalid interpolation periodic frame bits");
                        parentBits[size_t(*parent)][size_t(sub)] = NodePeriodicBits{uint8_t(bits)};
                    }
                }
            }
            for (index parent = 0; parent < cone.NumOwned(); ++parent)
                for (auto entity : p2e[size_t(parent)])
                    DNDS_check_throw_info(entity != UnInitIndex, "interpolation owner did not resolve an owned parent slot"); });
        MeshAdjacency parentRelation({cone.Source().Kind(), entityKind}, cone.Source(), entities, BuildRows(p2e, 0, size_t(cone.NumOwned())));
        // Complete appended parent rows from their owners, including entities
        // seen only through ghost parents. An exposed row never carries a
        // fabricated boundary sentinel for a merely unresolved remote entity.
        EntityGhostMapping parentRoute(cone.Source());
        parentRelation = parentRelation.WithGhostRows(parentRoute);
        NativeInterpolation result{entities, std::move(parentRelation), MeshAdjacency({entityKind, cone.Target().Kind()}, entities, cone.Target(), BuildRows(e2n, 0, e2n.size())), MeshAdjacency({entityKind, cone.Source().Kind()}, entities, cone.Source(), BuildRows(e2p, 0, e2p.size())), {}, {}, {}, BuildRows(info, 0, info.size())};
        if (periodic)
        {
            result.parent2entityPbi.emplace(BuildRows(parentBits, 0, size_t(cone.NumOwned())));
            const auto pbiSchema = Runtime::ElementSchema::For<NodePeriodicBits>();
            result.parent2entityGhostPbi.emplace(parentRoute.TransferRows(cone.Source(), *result.parent2entityPbi, pbiSchema));
            result.entity2nodePbi.emplace(BuildRows(e2nBits, 0, e2nBits.size()));
        }
        return result;
    }
}

namespace DNDS::Geom
{
    namespace
    {
        const std::vector<std::pair<AdjacencyDescriptor, MeshRelation>> &MeshRelations()
        {
            static const std::vector<std::pair<AdjacencyDescriptor, MeshRelation>> bindings{
                {Adj::Cell2Node, MeshRelation::CellNodes}, {Adj::Bnd2Node, MeshRelation::BoundaryNodes}, {Adj::Bnd2Cell, MeshRelation::BoundaryCells}, {Adj::Cell2Cell, MeshRelation::CellCells}, {Adj::Cell2CellFace, MeshRelation::CellFaceCells}, {Adj::Node2Cell, MeshRelation::NodeCells}, {Adj::Node2Bnd, MeshRelation::NodeBoundaries}, {Adj::Cell2Face, MeshRelation::CellFaces}, {Adj::Face2Node, MeshRelation::FaceNodes}, {Adj::Face2Cell, MeshRelation::FaceCells}, {Adj::Face2Bnd, MeshRelation::FaceBoundaries}, {Adj::Bnd2Face, MeshRelation::BoundaryFaces}, {Adj::Cell2Edge, MeshRelation::CellEdges}, {Adj::Edge2Node, MeshRelation::EdgeNodes}, {Adj::Edge2Cell, MeshRelation::EdgeCells}};
            return bindings;
        }
        MeshRelation QueryRelation(AdjacencyDescriptor kind)
        {
            for (auto binding : MeshRelations())
                if (binding.first == kind)
                    return binding.second;
            DNDS_check_throw_info(false, "no mesh facade relation for requested setup kind");
            return MeshRelation::Count;
        }
    }
    MeshAdjacencyRead::MeshAdjacencyRead(AdjacencyDescriptor kind, std::shared_ptr<MeshReadAccess> meshRead,
                                         EntitySpace source, EntitySpace target)
        : _source(std::move(source)), _target(std::move(target)),
          _sourceLease(_source._versions.Read()), _targetLease(_target._versions.Read()), _kind(kind)
    {
        DNDS_check_throw_info(meshRead, "mesh relation needs a retained scope");
        auto view = meshRead->HostView();
        const auto relation = QueryRelation(kind);
        DNDS_check_throw_info(view.HasRelation(relation), "mesh scope does not contain requested setup relation");
        _view = meshRead->_storage.relations[size_t(relation)];
        _state = view.RelationState(relation);
        auto hold = std::make_unique<MeshHold>(std::move(meshRead));
        DNDS_check_throw_info(hold->read->_nativeOwner, "mesh adjacency requires retained native storage metadata");
        const auto &rows = *hold->read->_nativeOwner->Snapshot(Runtime::Device::Host()).Live().relations[size_t(relation)];
        hold->ownedLayout = rows.owned.Layout();
        hold->ghostLayout = rows.ghost.Layout();
        _owned = std::move(hold);
        Validate();
    }
    MeshAdjacencyRead MeshAdjacencyRead::Current(UnstructuredMesh &mesh, AdjacencyDescriptor kind, MeshReadRequirements requirements)
    {
        auto read = std::make_shared<MeshReadAccess>(mesh.AcquireRead(requirements));
        return MeshAdjacencyRead(kind, std::move(read), mesh.Entities(kind.from), mesh.Entities(kind.to));
    }
    void AdjacencyRegistry::RegisterMesh(UnstructuredMesh &mesh, MeshReadRequirements requirements)
    {
        if (!_policyAgreed && _relations.empty() && !_communication)
            _communication = mesh.PreparationCommunication();
        std::shared_ptr<MeshReadAccess> read;
        ValidateCollectively([&]
                             {
            EntitySpace::CheckCommunicator(mesh.mpi.comm, _mpi.comm);
            read = std::make_shared<MeshReadAccess>(mesh.AcquireRead(requirements));
            DNDS_check_throw_info(_relations.empty(), "register the current mesh into an empty setup registry"); });
        ValidateCollectively([&]
                             {
            auto view = read->HostView();
            for (auto binding : MeshRelations())
                if (view.HasRelation(binding.second))
                    Register(MeshAdjacencyRead(binding.first, read, mesh.Entities(binding.first.from), mesh.Entities(binding.first.to)));
            if (mesh.HasNativeState())
                for (const auto &key : mesh.NativeRelationKeys())
                    if (!_relations.count(key))
                        Register(mesh.ReadNativeRelation(key)); });
    }
    RelationPeriodicBits::RelationPeriodicBits(const MeshAdjacencyRead &relation)
        : _source(relation.Source()), _target(relation.Target()), _kind(relation.Kind())
    {
        const auto *hold = dynamic_cast<const MeshAdjacencyRead::MeshHold *>(relation._owned.get());
        DNDS_check_throw_info(hold, "periodic companion requires a current-mesh relation scope");
        _meshRead = hold->read;
        DNDS_check_throw_info(_meshRead && _meshRead->HostView().IsPeriodic(), "periodic metadata needs a retained periodic mesh scope");
        const auto &storage = _meshRead->_storage;
        if (_kind == Adj::Cell2Node)
            _view = storage.cell2nodePbi;
        else if (_kind == Adj::Bnd2Node)
            _view = storage.bnd2nodePbi;
        else if (_kind == Adj::Face2Node)
            _view = storage.face2nodePbi;
        else if (_kind == Adj::Edge2Node)
            _view = storage.edge2nodePbi;
        else
            DNDS_check_throw_info(false, "requested relation has no node-periodic companion");
        Validate(relation);
    }
}

namespace DNDS::Geom
{
    AdjacencyGhostPublication AdjacencyRegistry::BuildGhosted(const GhostResult &discovery) const
    {
        std::vector<AdjacencyDescriptor> payloads;
        for (const auto &entry : _relations)
            payloads.push_back(entry.first);
        return BuildGhosted(discovery, payloads);
    }
    AdjacencyGhostPublication AdjacencyRegistry::BuildGhosted(const GhostResult &discovery,
                                                              const std::vector<AdjacencyDescriptor> &payloadsToPublish) const
    {
        AdjacencyGhostPublication result;
        // The registry schedule, not local nonempty discovery, determines MPI
        // participation. Empty owners/requesters execute every route stage.
        std::vector<EntityKind> kinds;
        auto relations = payloadsToPublish;
        std::vector<AdjacencyDescriptor> registered;
        for (auto &entry : _spaces)
            kinds.push_back(entry.first);
        for (auto &entry : _relations)
            registered.push_back(entry.first);
        std::sort(kinds.begin(), kinds.end());
        std::sort(registered.begin(), registered.end());
        std::sort(relations.begin(), relations.end());
        std::string schedule;
        for (auto kind : registered)
        {
            const auto descriptor = kind.Signature();
            schedule += std::to_string(descriptor.size()) + ":" + descriptor;
        }
        schedule += "|payloads|";
        for (auto kind : relations)
        {
            const auto descriptor = kind.Signature();
            schedule += std::to_string(descriptor.size()) + ":" + descriptor;
        }
        int canonicalSize = _mpi.rank == 0 ? int(schedule.size()) : 0;
        MPI_Bcast(&canonicalSize, 1, MPI_INT, 0, _mpi.comm);
        std::string canonical(size_t(canonicalSize), ' ');
        if (_mpi.rank == 0)
            canonical = schedule;
        MPI_Bcast(canonical.data(), canonicalSize, MPI_CHAR, 0, _mpi.comm);
        ValidateCollectively([&]
                             {
            DNDS_check_throw_info(schedule == canonical, "ghost publication registry differs between ranks");
            DNDS_check_throw_info(std::adjacent_find(relations.begin(), relations.end()) == relations.end(),
                                  "ghost publication repeats a payload descriptor");
            for (const auto &kind : relations)
                DNDS_check_throw_info(_relations.count(kind), "ghost publication requests an unregistered payload");
            for (auto &entry : _relations) entry.second->Validate();
            for (auto &entry : discovery.ghostIndices)
            {
                auto found = _spaces.find(entry.first);
                DNDS_check_throw_info(found != _spaces.end(), "ghost discovery names an unregistered entity family");
                for (auto global : entry.second)
                    DNDS_check_throw_info(found->second.Owner(global) != _mpi.rank, "ghost discovery contains an owned or invalid entity");
            } });
        for (auto kind : kinds)
        {
            const auto &space = _spaces.at(kind);
            std::vector<index> globals;
            auto found = discovery.ghostIndices.find(kind);
            if (found != discovery.ghostIndices.end())
                globals = found->second;
            SortUnique(globals);
            auto ghosts = std::make_shared<OffsetAscendIndexMapping>((*space._global)(_mpi.rank, 0), space.NumOwned(),
                                                                     globals, *space._global, _mpi);
            EntitySpace next(kind, _mpi, space.NumOwned(), index(globals.size()), space._identity, space._stamp,
                             space._global, std::move(ghosts), space._versions);
            next._publishedMaps = true;
            next._communication = space._communication;
            result.spaces.emplace(kind, std::move(next));
            result.routes.emplace(kind, EntityGhostMapping(result.spaces.at(kind)));
        }
        using PayloadKey = std::tuple<EntityKind, EntityKind, MeshAdjState, Runtime::RowLayoutKind, uintptr_t, size_t, size_t, size_t, uintptr_t, uintptr_t, uint64_t>;
        struct Payload
        {
            int ordinal;
            Runtime::Array<index> owned, ghost;
        };
        std::map<PayloadKey, Payload> payloads;
        ValidateCollectively([&]
                             { DNDS_check_throw_info(relations.size() <= size_t(INT_MAX), "too many ghost relation descriptors"); });
        for (size_t ordinal = 0; ordinal < relations.size(); ++ordinal)
        {
            const auto kind = relations[ordinal];
            const auto &input = *_relations.at(kind);
            const auto &view = input._view.owned;
            const auto &ghostMapping = result.routes.at(kind.from);
            const auto *hold = dynamic_cast<const MeshAdjacencyRead::ArrayHold *>(input._owned.get());
            const auto *mesh = dynamic_cast<const MeshAdjacencyRead::MeshHold *>(input._owned.get());
            const auto storageKind = hold ? hold->read.Layout().Kind() : mesh->ownedLayout.Kind();
            // Empty tables and padded arrays can have identical null pointers,
            // width and stride. Their physical family must still distinguish aliases.
            const PayloadKey key{kind.from, kind.to, input.State(), storageKind, reinterpret_cast<uintptr_t>(view.data), view.rows,
                                 view.width, view.stride, reinterpret_cast<uintptr_t>(view.offsets), reinterpret_cast<uintptr_t>(view.sizes),
                                 ghostMapping.Identity()};
            auto cached = payloads.find(key);
            int candidate = cached != payloads.end() ? cached->second.ordinal : -1;
            // Every rank must reuse the SAME earlier transfer. Local hits on
            // different descriptors do not prove their remote payloads match.
            candidate = MPI::AgreeCandidate(_mpi, candidate, relations.size());
            if (candidate >= 0)
            {
                ValidateCollectively([&]
                                     {
                    MeshAdjacency alias(kind, result.spaces.at(kind.from), result.spaces.at(kind.to), cached->second.owned);
                    alias._ghost.emplace(cached->second.ghost);
                    auto checked = alias.Read();
                    result.relations.emplace(kind, std::move(alias)); });
                continue;
            }
            std::optional<Runtime::Array<index>> values;
            ValidateCollectively([&]
                                 {
                if (input.State() == Adj_PointToGlobal && hold)
                    values = hold->array;
                else
                {
                    DNDS_check_throw_info(hold || mesh, "adjacency publication requires retained storage metadata");
                    const auto &shape = hold ? hold->read.Layout() : mesh->ownedLayout;
                    values = Runtime::BuildArrayRows<index>(shape, [&](index iRow, auto output)
                        { input.ConvertRowTo(iRow, Adj_PointToGlobal, output.data()); }, true);
                } });
            auto fetched = _fetchedRows.find(kind);
            std::optional<Runtime::Array<index>::ReadLease> scratchRead;
            int reuse = 0;
            try
            {
                if (fetched != _fetchedRows.end() && fetched->second.values &&
                    fetched->second.globals == result.spaces.at(kind.from)._ghostMap->ghostIndex)
                {
                    // Pin before observing the version: a writer may have
                    // changed a previously published alias of this scratch.
                    scratchRead.emplace(fetched->second.values->Read());
                    const auto &ownedLayout = values->Layout(), &ghostLayout = fetched->second.values->Layout();
                    // Discovery scratch may have compacted a table/padded
                    // payload. Reuse only a shape that preserves this source's
                    // storage contract; otherwise fetch directly into it.
                    reuse = fetched->second.version == fetched->second.values->Version() &&
                            ownedLayout.Kind() == ghostLayout.Kind() &&
                            (ownedLayout.IsRagged() || ownedLayout.Stride() == ghostLayout.Stride()) &&
                            (ownedLayout.IsVariable() || ownedLayout.Width() == ghostLayout.Width());
                }
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("mesh adjacency preparation failed");
            } // Active writers or invalid scratch take the normal route.
            MPI_Allreduce(MPI_IN_PLACE, &reuse, 1, MPI_INT, MPI_MIN, _mpi.comm);
            auto completed = reuse
                                 ? MeshAdjacency::WithGhostValues(kind, result.spaces.at(kind.from), result.spaces.at(kind.to),
                                                                  std::move(*values), *fetched->second.values)
                                 : MeshAdjacency(kind, result.spaces.at(kind.from), result.spaces.at(kind.to), std::move(*values))
                                       .WithGhostRows(result.routes.at(kind.from));
            // Transfer scratch ownership into this publication. A later
            // publication gets independent ghost values; descriptor aliases
            // inside this one still share through the payload table below.
            if (reuse)
                fetched->second.values.reset();
            payloads.insert_or_assign(key, Payload{int(ordinal), completed.Owned(), *completed.Ghost()});
            result.relations.emplace(kind, std::move(completed));
        }
        return result;
    }
}
