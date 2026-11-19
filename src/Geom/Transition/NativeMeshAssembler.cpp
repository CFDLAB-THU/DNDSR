#include "DNDS/MPI/Failure.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Transition/NativeMeshPartitionGraph.hpp"
#include "Geom/Transition/MeshAdjacencyBindings.hpp"
#include "DNDS/Core/OMP.hpp"
#include <iomanip>
#include <sstream>

namespace DNDS::Geom
{
    namespace
    {
        using RArray = Runtime::Array<index>;
        using MeshTransitionDetail::Bindings;
        using MeshTransitionDetail::FacadeKey;
        template <class T>
        Runtime::ArrayPair<T> Ghost(const Runtime::Array<T> &owned, EntityKind kind,
                                    const AdjacencyGhostPublication &publication, const Runtime::ElementSchema &schema)
        {
            const auto &space = publication.spaces.at(kind);
            auto ghost = publication.routes.at(kind).TransferRows(space, owned, schema);
            return {owned, std::move(ghost)};
        }
        void Agree(const MPIInfo &mpi, const NativeOwnedMeshData &owned, const std::string &signature)
        {
            int length = int(signature.size());
            MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
            std::string expected;
            owned.LocalStage(mpi, [&]
                             { expected = mpi.rank == 0 ? signature : std::string(size_t(length), '\0'); });
            MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
            owned.LocalStage(mpi, [&]
                             { DNDS_check_throw_info(signature == expected, "native assembly inventory differs across ranks"); });
        }
        template <class T>
        void CheckRows(const Runtime::Array<T> &array, const EntitySpace &space)
        {
            DNDS_check_throw_info(array.GetDevice() == Runtime::Device::Host() && array.RowCount() == size_t(space.NumOwned()),
                                  "native assembly requires matching owned host rows");
            auto read = array.Read();
        }

    }

    GhostSpec DefaultNativeMeshGhostSpec(const NativeOwnedMeshData &owned, int cellLayers, bool completeOwnedNodeStars)
    {
        DNDS_check_throw_info(cellLayers > 0, "mesh default requires a positive cell ring");
        auto spec = GhostSpec::defaultPrimary(cellLayers);
        auto append = [&](EntityKind anchor, std::vector<AdjacencyDescriptor> hops)
        { spec.chains.push_back({anchor, hops, hops.back().to}); };
        append(EntityKind::Cell, {Adj::Cell2Node});
        append(EntityKind::Bnd, {Adj::Bnd2Node});
        append(EntityKind::Bnd, {Adj::Bnd2Cell});
        if (completeOwnedNodeStars)
        {
            append(EntityKind::Node, {Adj::Node2Cell});
            append(EntityKind::Node, {Adj::Node2Cell, Adj::Cell2Node});
            append(EntityKind::Node, {Adj::Node2Bnd});
            append(EntityKind::Node, {Adj::Node2Bnd, Adj::Bnd2Node});
        }
        for (size_t slot = size_t(MeshRelation::CellFaces); slot < owned.relations.size(); ++slot)
            if (owned.relations[slot])
                append(Bindings()[slot].from, {Bindings()[slot]});
        if (owned.relations[size_t(MeshRelation::CellFaces)])
            append(EntityKind::Cell, {Adj::Cell2Face, Adj::Face2Cell});
        return spec;
    }

    std::shared_ptr<const UnstructuredMeshState> AssembleNativeMesh(const MPIInfo &mpi, NativeOwnedMeshData owned, int cellLayers)
    {
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !owned.communication || owned.communication->ChecksCollectivePreflight());
        GhostSpec recipe;
        owned.LocalStage(mpi, [&]
                         {
            DNDS_check_throw_info(cellLayers >= 0, "mesh finalization cell depth must be nonnegative");
            recipe = owned.ghostSpec ? *owned.ghostSpec : DefaultNativeMeshGhostSpec(owned, std::max(cellLayers, 1));
            if (owned.ghostSpec && cellLayers > 0)
            {
                bool collectsCells = false;
                std::vector<GhostChain> expanded;
                std::set<std::string> seen;
                auto append = [&](GhostChain chain)
                {
                    std::string key = std::to_string(int(chain.anchor)) + "|" + std::to_string(int(chain.target)) + "|";
                    for (const auto &hop : chain.hops)
                        key += hop.Signature() + ";";
                    if (seen.insert(std::move(key)).second)
                        expanded.push_back(std::move(chain));
                };
                for (auto &chain : recipe.chains)
                {
                    size_t prefix = 0;
                    if (chain.anchor == EntityKind::Cell)
                        while (prefix < chain.hops.size() && chain.hops[prefix] == Adj::Cell2Cell) ++prefix;
                    if (!prefix)
                    {
                        collectsCells |= chain.anchor == EntityKind::Cell && chain.target == EntityKind::Cell;
                        append(std::move(chain));
                        continue;
                    }
                    std::vector<AdjacencyDescriptor> tail(chain.hops.begin() + std::ptrdiff_t(prefix), chain.hops.end());
                    for (int depth = 1; depth <= cellLayers; ++depth)
                    {
                        std::vector<AdjacencyDescriptor> hops(size_t(depth), Adj::Cell2Cell);
                        hops.insert(hops.end(), tail.begin(), tail.end());
                        collectsCells |= chain.target == EntityKind::Cell;
                        append({EntityKind::Cell, std::move(hops), chain.target});
                    }
                }
                if (!collectsCells)
                {
                    for (int depth = 1; depth <= cellLayers; ++depth)
                    {
                        std::vector<AdjacencyDescriptor> hops(size_t(depth), Adj::Cell2Cell);
                        append({EntityKind::Cell, hops, EntityKind::Cell});
                        hops.push_back(Adj::Cell2Node);
                        append({EntityKind::Cell, std::move(hops), EntityKind::Node});
                    }
                }
                recipe.chains = std::move(expanded);
            } });
        // All ranks must agree even if this recipe needs no depth override.
        Agree(mpi, owned, std::to_string(cellLayers));
        return NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    }

    NativeOwnedMeshData NativeMeshAssembler::CaptureOwned(const UnstructuredMeshState &source, const MPIInfo &mpi)
    {
        NativeOwnedMeshData result;
        const auto cell = source._spaces.find(EntityKind::Cell);
        if (cell != source._spaces.end())
            result.communication = cell->second.identity.Communication();
        std::string inventory;
        result.LocalStage(mpi, [&]
                          {
            source.Validate();
            const auto &snapshot = source.Snapshot(Runtime::Device::Host()).Live();
            result.retainedRead = std::make_shared<MeshReadAccess>(source.Snapshot(Runtime::Device::Host()).AcquireRead());
            result.identity = {source.MeshNamespace(), source.IdHighWater()};
            result.provenance = source.Provenance();
            if (result.provenance)
                for (const auto &[kind, values] : result.provenance->sourceIds)
                    result.retainedProvenance.push_back(std::make_shared<Runtime::Array<index>::ReadLease>(values.Read()));
            result.requirements = source.Requirements();
            result.audit = source.AuditOptions();
            result.ghostSpec = source.GhostSpecification();
            for (int part = 0; part < source.NLocalParts(); ++part)
                result.localPartitionStarts.push_back(source.LocalPartStart(part));
            result.localPartitionStarts.push_back(source.NumOwned(EntityKind::Cell));
            result.dimension = snapshot.semantics.dim;
            result.periodic = snapshot.semantics.isPeriodic;
            result.periodicInfo = snapshot.semantics.periodicInfo;
            // Fixed query slots and named descriptors can be exact aliases.
            // Validate each descriptor, but convert one retained payload once.
            using CaptureKey = std::tuple<uintptr_t, EntityKind, EntityKind, MeshAdjState>;
            std::map<CaptureKey, RArray> converted;
            auto global = [&](const AdjacencyDescriptor &key, const Runtime::ArrayPair<index> &rows, MeshAdjState state)
            {
                MeshAdjacencyRead read(key, source.Entities(key.from), source.Entities(key.to), state, rows.owned, &rows.ghost);
                const CaptureKey identity{reinterpret_cast<uintptr_t>(rows.owned.SharedStorage().get()), key.from, key.to, state};
                auto found = converted.find(identity);
                if (found != converted.end()) return found->second;
                const auto &shape = rows.owned.Layout();
                auto result = Runtime::BuildArrayRows<index>(shape,
                                                             [&](size_t row, auto output)
                                                             {
                    read.ConvertRowTo(index(row), Adj_PointToGlobal, output.data());
                }, true);
                converted.emplace(identity, result);
                return result;
            };
            for (size_t slot = 0; slot < snapshot.relations.size(); ++slot)
                if (snapshot.relations[slot])
                {
                    auto array = global(FacadeKey(slot), *snapshot.relations[slot], Adj_PointToLocal);
                    result.relations[slot] = std::move(array);
                }
            result.cellCellsComplete = source.Requirements().inverse &&
                                       result.relations[size_t(MeshRelation::CellCells)].has_value();
            for (const auto &[key, relation] : source._namedRelations)
                result.namedRelations.emplace(key, global(key, relation.data.rows, relation.data.state));
            auto capture = [](auto &destination, const auto &pair) { if (pair) destination = pair->owned; };
            capture(result.coords, snapshot.coords);
            capture(result.nodeWallDist, snapshot.nodeWallDist);
            capture(result.cellInfo, snapshot.cellInfo);
            capture(result.bndInfo, snapshot.bndInfo);
            capture(result.faceInfo, snapshot.faceInfo);
            capture(result.edgeInfo, snapshot.edgeInfo);
            capture(result.cellNodesPbi, snapshot.cellNodesPbi);
            capture(result.bndNodesPbi, snapshot.bndNodesPbi);
            capture(result.cellFacesPbi, snapshot.cellFacesPbi);
            capture(result.faceNodesPbi, snapshot.faceNodesPbi);
            capture(result.cellEdgesPbi, snapshot.cellEdgesPbi);
            capture(result.edgeNodesPbi, snapshot.edgeNodesPbi);
            for (const auto &[kind, family] : source._spaces)
            {
                EntitySpace::CheckCommunicator(family.identity._mpi.comm, mpi.comm);
                auto read = source._originalIds.at(kind).Read();
                RArray ids(Runtime::RowLayout::Dense(size_t(family.owned), 1));
                { auto write = ids.Write();
                  for (index row = 0; row < family.owned; ++row) write.View()(size_t(row), 0) = read.View()(size_t(row), 0); }
                result.ids.emplace(kind, std::move(ids));
                inventory += "k" + std::to_string(int(kind)) + ":";
                for (auto count : family.global->RLengths()) inventory += std::to_string(count) + ",";
            }
            inventory += "d" + std::to_string(result.dimension) + "p" + std::to_string(result.periodic) + ":" + result.identity.meshNamespace;
            for (const auto &[kind, next] : result.identity.nextIds) inventory += "i" + std::to_string(int(kind)) + ":" + std::to_string(next) + ";";
            const auto &r = result.requirements;
            inventory += "q" + std::to_string(r.faces) + std::to_string(r.inverse) + std::to_string(r.edges) +
                         std::to_string(r.cellFaceNeighbors) + std::to_string(r.wallDistance) + std::to_string(r.allFaceCellsLocal);
            for (size_t slot = 0; slot < result.relations.size(); ++slot)
                if (result.relations[slot]) inventory += "r" + std::to_string(slot) + ";";
            for (const auto &[key, unused] : result.namedRelations) inventory += key.Signature();
            inventory += result.provenance ? "p" + result.provenance->Signature() : "p;";
            DNDS_check_throw_info(inventory.size() <= size_t(INT_MAX), "native capture inventory exceeds MPI range"); });
        Agree(mpi, result, inventory);
        for (const auto &[kind, family] : source._spaces)
            result.spaces.emplace(kind, family.identity.WithGhosts({}));
        return result;
    }

    std::shared_ptr<const UnstructuredMeshState> NativeMeshAssembler::Build(const MPIInfo &mpi, NativeOwnedMeshData owned,
                                                                            const GhostSpec &specification)
    {
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !owned.communication || owned.communication->ChecksCollectivePreflight());
        std::string schedule;
        owned.LocalStage(mpi, [&]
                         {
            DNDS_check_throw_info(owned.dimension >= 1 && owned.dimension <= 3, "invalid native assembly dimension");
            DNDS_check_throw_info(owned.spaces.count(EntityKind::Node) && owned.spaces.count(EntityKind::Cell) && owned.spaces.count(EntityKind::Bnd), "native assembly primary families missing");
            DNDS_check_throw_info(owned.dimension != 1 || (owned.spaces.at(EntityKind::Bnd).NumGlobal() == 0 &&
                                                         !owned.requirements.faces && !owned.requirements.edges && !owned.requirements.cellFaceNeighbors),
                                  "native one-dimensional assembly supports primary curves without point boundaries or facial/edge topology");
            const auto identity = owned.spaces.at(EntityKind::Node).MeshIdentity();
            for (const auto &[kind, space] : owned.spaces)
            {
                space.Validate();
                DNDS_check_throw_info(space.ChecksCollectivePreflight() ==
                                          (!owned.communication || owned.communication->ChecksCollectivePreflight()),
                                      "owned assembly entity preparation policy differs from its pipeline");
                EntitySpace::CheckCommunicator(space._mpi.comm, mpi.comm);
                DNDS_check_throw_info(space.Kind() == kind && space.MeshIdentity() == identity && space.NumGhost() == 0, "native assembly spaces must be owned-only siblings");
                DNDS_check_throw_info(owned.ids.count(kind), "native assembly persistent IDs missing");
                CheckRows(owned.ids.at(kind), space);
                {
                    auto ids = owned.ids.at(kind).Read();
                    OMP::ForSetupRows(0, ids.View().RowCount(), [&](index iRow)
                                      { DNDS_check_throw_info(ids.View().RowSize(iRow) == 1, "native assembly IDs must have exactly one column"); });
                }
                schedule += "k" + std::to_string(int(kind)) + ":";
                for (auto count : space._global->RLengths()) schedule += std::to_string(count) + ",";
            }
            DNDS_check_throw_info(owned.ids.size() == owned.spaces.size() && owned.coords && owned.cellInfo && owned.bndInfo,
                                  "native assembly ID or primary companion inventory invalid");
            schedule += owned.provenance ? "p" + owned.provenance->Signature() : "p;";
            if (owned.provenance)
                for (const auto &[kind, values] : owned.provenance->sourceIds)
                    CheckRows(values, owned.spaces.at(kind));
            for (auto slot : {MeshRelation::CellNodes, MeshRelation::BoundaryNodes, MeshRelation::BoundaryCells, MeshRelation::CellCells})
                DNDS_check_throw_info(owned.relations[size_t(slot)], "native assembly primary topology missing");
            schedule += "d" + std::to_string(owned.dimension) + "p" + std::to_string(owned.periodic);
            schedule += "a" + std::to_string(owned.audit.persistentIdentity) + std::to_string(owned.audit.closure);
            for (size_t slot = 0; slot < owned.relations.size(); ++slot)
                if (owned.relations[slot])
                {
                    const auto &key = Bindings()[slot];
                    CheckRows(*owned.relations[slot], owned.spaces.at(key.from));
                    DNDS_check_throw_info(owned.spaces.count(key.to), "native assembly fixed relation target missing");
                    schedule += "r" + std::to_string(slot) + ";";
                }
            for (const auto &[key, values] : owned.namedRelations)
            {
                if (key == NativeSetupFacialCellNeighborsKey()) continue;
                key.Validate();
                CheckRows(values, owned.spaces.at(key.from));
                DNDS_check_throw_info(owned.spaces.count(key.to), "native assembly named relation target missing");
                schedule += key.Signature();
            }
            const auto &r = owned.requirements;
            schedule += "q" + std::to_string(r.faces) + std::to_string(r.inverse) + std::to_string(r.edges) +
                        std::to_string(r.cellFaceNeighbors) + std::to_string(r.wallDistance) + std::to_string(r.allFaceCellsLocal);
            auto companion = [&](const auto &array, EntityKind kind)
            {
                schedule += array ? "1" : "0";
                if (array) CheckRows(*array, owned.spaces.at(kind));
            };
            companion(owned.coords, EntityKind::Node); companion(owned.nodeWallDist, EntityKind::Node);
            companion(owned.cellInfo, EntityKind::Cell); companion(owned.bndInfo, EntityKind::Bnd);
            companion(owned.faceInfo, EntityKind::Face); companion(owned.edgeInfo, EntityKind::Edge);
            companion(owned.cellNodesPbi, EntityKind::Cell); companion(owned.bndNodesPbi, EntityKind::Bnd);
            companion(owned.cellFacesPbi, EntityKind::Cell); companion(owned.faceNodesPbi, EntityKind::Face);
            companion(owned.cellEdgesPbi, EntityKind::Cell); companion(owned.edgeNodesPbi, EntityKind::Edge);
            auto periodicCone = [&](const auto &bits, MeshRelation slot)
            {
                const auto &cone = owned.relations[size_t(slot)];
                DNDS_check_throw_info(owned.periodic ? bool(bits) == bool(cone) : !bits,
                                      "native assembly periodic cone companion inventory differs from topology");
                if (!bits) return;
                auto read = bits->Read();
                const auto shape = cone->Layout();
                for (size_t row = 0; row < read.View().RowCount(); ++row)
                {
                    DNDS_check_throw_info(read.View().RowSize(row) == shape.RowSize(row), "native assembly periodic cone slot count differs");
                    for (auto value : read.View()[row]) DNDS_check_throw_info(uint8_t(value) < 8, "native assembly periodic frame bits exceed three groups");
                }
            };
            periodicCone(owned.cellNodesPbi, MeshRelation::CellNodes); periodicCone(owned.bndNodesPbi, MeshRelation::BoundaryNodes);
            periodicCone(owned.cellFacesPbi, MeshRelation::CellFaces); periodicCone(owned.faceNodesPbi, MeshRelation::FaceNodes);
            periodicCone(owned.cellEdgesPbi, MeshRelation::CellEdges); periodicCone(owned.edgeNodesPbi, MeshRelation::EdgeNodes);
            if (owned.periodic)
            {
                std::ostringstream frames;
                frames << std::hexfloat;
                for (size_t frame = 0; frame < 4; ++frame)
                {
                    const auto rotation = owned.periodicInfo.rotation[frame].map();
                    DNDS_check_throw_info(rotation.allFinite() && (rotation.transpose() * rotation - tGPoint::Identity()).norm() < 1e-10 &&
                                              std::abs(rotation.determinant() - 1) < 1e-10 && owned.periodicInfo.translation[frame].map().allFinite() &&
                                              owned.periodicInfo.rotationCenter[frame].map().allFinite(), "invalid native assembly rigid periodic frame");
                    for (int i = 0; i < 9; ++i) frames << rotation.data()[i] << ";";
                    for (int i = 0; i < 3; ++i) frames << owned.periodicInfo.translation[frame].map()[i] << ";";
                    for (int i = 0; i < 3; ++i) frames << owned.periodicInfo.rotationCenter[frame].map()[i] << ";";
                }
                schedule += frames.str();
            }
            DNDS_check_throw_info(schedule.size() <= size_t(INT_MAX), "native assembly schedule exceeds MPI range"); });
        Agree(mpi, owned, schedule);
        AdjacencyRegistry connectivity(mpi);
        std::map<AdjacencyDescriptor, RArray> relations;
        owned.LocalStage(mpi, [&]
                         {
            for (size_t slot = 0; slot < owned.relations.size(); ++slot)
                if (owned.relations[slot]) relations.emplace(FacadeKey(slot), *owned.relations[slot]);
            for (const auto &[key, values] : owned.namedRelations)
                if (key != NativeSetupFacialCellNeighborsKey())
                    DNDS_check_throw_info(relations.emplace(key, values).second, "native assembly reserved descriptor collision"); });
        for (auto slot : {MeshRelation::NodeCells, MeshRelation::NodeBoundaries})
            if (!owned.relations[size_t(slot)])
            {
                const auto coneSlot = slot == MeshRelation::NodeCells ? MeshRelation::CellNodes : MeshRelation::BoundaryNodes;
                const auto &key = Bindings()[size_t(coneSlot)];
                MeshAdjacencyRead read(key, owned.spaces.at(key.from), owned.spaces.at(key.to), Adj_PointToGlobal, *owned.relations[size_t(coneSlot)]);
                auto inverse = AdjacencyRegistry::Inverse(read, mpi);
                relations.emplace(FacadeKey(size_t(slot)), inverse.Owned());
            }
        owned.LocalStage(mpi, [&]
                         {
            for (const auto &[key, values] : relations)
                try
                {
                    connectivity.Register(MeshAdjacencyRead(key, owned.spaces.at(key.from), owned.spaces.at(key.to), Adj_PointToGlobal, values));
                }
                catch (const std::exception &error)
                {
                    MPI::AbortIfSolverFailure("native mesh assembly failed");
                    throw std::runtime_error("native assembly registration [" + key.Signature() + "]: " + error.what());
                } });
        std::optional<CompiledGhostTree> tree;
        owned.LocalStage(mpi, [&]
                         {
            auto resolved = specification;
            // Legacy shorthand keys select fixed facade slots. Full named
            // descriptors retain their exact path/selection/periodic semantics.
            for (auto &chain : resolved.chains)
                for (auto &hop : chain.hops)
                    if (hop.IsLegacy())
                        for (size_t slot = 0; slot < Bindings().size(); ++slot)
                            if (hop == Bindings()[slot])
                            {
                                hop = FacadeKey(slot);
                                break;
                            }
            tree = CompiledGhostTree::compile(resolved); });
        auto discovery = connectivity.Evaluate(*tree);
        auto publication = connectivity.BuildGhosted(discovery);
        // Complete cones, and (when requested) cells of EVERY discovered face.
        // Only Cell and Node sets can grow here, so no extra cell rings arise.
        while (true)
        {
            int changed = 0;
            owned.LocalStage(mpi, [&]
                             {
                std::map<EntityKind, std::vector<index>> known;
                for (const auto &[kind, values] : discovery.ghostIndices)
                {
                    auto &rows = known[kind];
                    rows = values;
                    std::sort(rows.begin(), rows.end());
                    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
                }
                std::map<EntityKind, std::vector<index>> additional;
                auto collect = [&](MeshRelation slot)
                {
                    if (!owned.relations[size_t(slot)]) return;
                    auto read = publication.relations.at(FacadeKey(size_t(slot))).Read();
                    const auto kind = Bindings()[size_t(slot)].to;
                    DNDS_check_throw_info(read.State() == Adj_PointToGlobal,
                                          "ghost publication completion requires global target state");
                    const auto view = read.View();
                    // The adjacency lease pins the validated global target
                    // numbering. Ownership membership is a range test, not an
                    // owner search/version lock for every cone incidence.
                    const auto &target = read.Target();
                    const index firstOwned = target._global->ROffsets()[size_t(mpi.rank)];
                    const index endOwned = firstOwned + target.NumOwned();
                    const auto &knownRows = known[kind];
                    auto &extraRows = additional[kind];
                    const auto scan = [&](const Runtime::ArrayView<const index> &part)
                    {
                        for (index iRow = 0; iRow < part.RowCount(); ++iRow)
                        {
                            const auto count = part.RowSize(iRow);
                            const auto *values = count ? part.Row(iRow) : nullptr;
                            for (index entrySlot = 0; entrySlot < count; ++entrySlot)
                            {
                                const auto global = values[entrySlot];
                                if (global == UnInitIndex)
                                    continue;
                                if ((global < firstOwned || global >= endOwned) &&
                                    !std::binary_search(knownRows.begin(), knownRows.end(), global))
                                    extraRows.push_back(global);
                            }
                        }
                    };
                    scan(view.owned);
                    scan(view.ghost);
                };
                for (auto slot : {MeshRelation::CellNodes, MeshRelation::BoundaryNodes, MeshRelation::FaceNodes, MeshRelation::EdgeNodes}) collect(slot);
                if (owned.requirements.allFaceCellsLocal) collect(MeshRelation::FaceCells);
                for (auto &[kind, rows] : additional)
                {
                    std::sort(rows.begin(), rows.end());
                    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
                    if (!rows.empty()) changed = 1;
                    discovery.ghostIndices[kind].insert(discovery.ghostIndices[kind].end(), rows.begin(), rows.end());
                } });
            MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, mpi.comm);
            if (!changed)
                break;
            publication = connectivity.BuildGhosted(discovery);
        }
        NativeMeshData data;
        data.ghostSpec = specification;
        data.audit = owned.audit;
        data.identity = owned.identity;
        data.provenance = owned.provenance;
        data.dimension = owned.dimension;
        data.periodic = owned.periodic;
        data.periodicInfo = owned.periodicInfo;
        using LocalKey = std::tuple<uintptr_t, uintptr_t, EntityKind, EntityKind>;
        std::map<LocalKey, Runtime::ArrayPair<index>> localized;
        auto localRows = [&](const MeshAdjacency &relation)
        {
            auto read = relation.Read();
            const LocalKey key{reinterpret_cast<uintptr_t>(relation.Owned().SharedStorage().get()),
                               relation.Ghost() ? reinterpret_cast<uintptr_t>(relation.Ghost()->SharedStorage().get()) : 0,
                               relation.Source().Kind(), relation.Target().Kind()};
            auto found = localized.find(key);
            if (found != localized.end())
                return found->second;
            // Conversion changes indices, not physical row storage. Named and
            // facade aliases retain the same preserved table/padded/CSR shape.
            auto converted = MeshAdjacency::Convert(read, Adj_PointToLocal);
            DNDS_check_throw_info(converted.Ghost(), "published adjacency is missing ghost storage");
            Runtime::ArrayPair<index> result{converted.Owned(), *converted.Ghost()};
            localized.emplace(key, result);
            return result;
        };
        const bool trackAllocations = Runtime::AllocationDiagnostics::Enabled();
        const auto label = [&](const Runtime::ArrayPair<index> &rows, const AdjacencyDescriptor &descriptor)
        {
            const auto name = "Geom/adjacency/" + adjKindName(descriptor);
            rows.owned.SetName(name + "/owned", DNDS_ALLOCATION_SITE());
            rows.ghost.SetName(name + "/ghost", DNDS_ALLOCATION_SITE());
        };
        owned.LocalStage(mpi, [&]
                         {
            for (size_t slot = 0; slot < owned.relations.size(); ++slot)
                if (owned.relations[slot])
                {
                    data.relations[slot] = localRows(publication.relations.at(FacadeKey(slot)));
                    if (trackAllocations)
                        label(*data.relations[slot], Bindings()[slot]);
                }
            for (const auto &[key, unused] : owned.namedRelations)
                if (key != NativeSetupFacialCellNeighborsKey())
                {
                    auto rows = localRows(publication.relations.at(key));
                    if (trackAllocations)
                        label(rows, key);
                    data.namedRelations.emplace(key, NativeNamedRelation{std::move(rows), Adj_PointToLocal});
                } });
        auto companion = [&](auto &destination, const auto &values, EntityKind kind, const Runtime::ElementSchema &schema)
        { if (values) destination = Ghost(*values, kind, publication, schema); };
        companion(data.coords, owned.coords, EntityKind::Node, Runtime::ElementSchema::Arithmetic<real>());
        companion(data.nodeWallDist, owned.nodeWallDist, EntityKind::Node, Runtime::ElementSchema::Arithmetic<real>());
        const auto elementSchema = Runtime::ElementSchema::For<ElemInfo>();
        companion(data.cellInfo, owned.cellInfo, EntityKind::Cell, elementSchema);
        companion(data.bndInfo, owned.bndInfo, EntityKind::Bnd, elementSchema);
        companion(data.faceInfo, owned.faceInfo, EntityKind::Face, elementSchema);
        companion(data.edgeInfo, owned.edgeInfo, EntityKind::Edge, elementSchema);
        const auto bits = Runtime::ElementSchema::For<NodePeriodicBits>();
        companion(data.cellNodesPbi, owned.cellNodesPbi, EntityKind::Cell, bits);
        companion(data.bndNodesPbi, owned.bndNodesPbi, EntityKind::Bnd, bits);
        companion(data.cellFacesPbi, owned.cellFacesPbi, EntityKind::Cell, bits);
        companion(data.faceNodesPbi, owned.faceNodesPbi, EntityKind::Face, bits);
        companion(data.cellEdgesPbi, owned.cellEdgesPbi, EntityKind::Cell, bits);
        companion(data.edgeNodesPbi, owned.edgeNodesPbi, EntityKind::Edge, bits);
        std::map<EntityKind, RArray> ids;
        std::map<EntityKind, EntitySpace> spaces;
        for (const auto &[kind, values] : owned.ids)
        {
            auto pair = Ghost(values, kind, publication, Runtime::ElementSchema::Arithmetic<index>());
            owned.LocalStage(mpi, [&]
                             {
                auto read = pair.ReadCombined();
                RArray combined(Runtime::RowLayout::Dense(read.View().RowCount(), 1));
                {
                    auto write = combined.Write();
                    const auto input = read.View();
                    const auto output = write.View();
                    OMP::ForSetupRows(0, input.RowCount(), [&](index iRow)
                                      { output(iRow, 0) = input(iRow, 0); });
                }
                ids.emplace(kind, std::move(combined));
                spaces.emplace(kind, publication.spaces.at(kind)); });
        }
        auto result = UnstructuredMeshState::FromNative(mpi, std::move(data), std::move(spaces), std::move(ids), owned.requirements);
        owned.LocalStage(mpi, [&]
                         {
            if (!owned.localPartitionStarts.empty())
                result = result->WithLocalPartitions(std::move(owned.localPartitionStarts)); });
        return result;
    }
}
