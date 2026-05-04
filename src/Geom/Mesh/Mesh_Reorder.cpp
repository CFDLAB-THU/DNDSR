/// @file Mesh_Reorder.cpp
/// @brief Implementation of ReorderPlan::build, ReorderPlan::apply, and
///        UnstructuredMesh::buildReorderRegistry / ReorderEntities.

#include "ReorderPlan.hpp"
#include "Mesh.hpp"

#include <algorithm>
#include <set>

namespace DNDS::Geom
{
    // =================================================================
    // ComputeFollowMap: derive follower placement from leader
    // =================================================================

    /// Compute a follow map for `follower` based on `leader`'s explicit map.
    ///
    /// Uses the support adjacency follower->leader (e.g., node2cell) to
    /// determine: for each follower entity, go to min(leader.targetRank)
    /// over all leaders referencing it.
    ///
    /// @param followerGM   Global mapping for the follower entity kind.
    /// @param leaderGM     Global mapping for the leader entity kind.
    /// @param follower2leader  Support adj: follower -> leader (father-only,
    ///                         global entries). Must cover all follower entities.
    /// @param leaderTargetRanks  Per-leader-slot target rank (the explicit map).
    /// @param mpi          MPI communicator.
    /// @return Per-follower-slot target rank.
    /// @warning Collective.
    static std::vector<MPI_int> ComputeFollowMap(
        const ssp<GlobalOffsetsMapping> &followerGM,
        const ssp<GlobalOffsetsMapping> &leaderGM,
        const ReorderRegistry &registry,
        AdjKind follower2leaderKind,
        const std::vector<MPI_int> &leaderTargetRanks,
        const MPIInfo &mpi)
    {
        DNDS_assert(followerGM);
        DNDS_assert(leaderGM);

        index nFollower = followerGM->RLengths()[mpi.rank];

        // Step 1: Build a ghost-pullable lookup of leader's targetRanks.
        // lookup(leaderLocalSlot, 0) = leaderTargetRanks[leaderLocalSlot]
        ArrayAdjacencyPair<1> leaderLookup;
        leaderLookup.InitPair("followMap_leaderLookup", mpi);
        leaderLookup.father->Resize(static_cast<index>(leaderTargetRanks.size()));
        for (index i = 0; i < static_cast<index>(leaderTargetRanks.size()); i++)
            leaderLookup(i, 0) = static_cast<index>(leaderTargetRanks[i]);
        leaderLookup.TransAttach();
        leaderLookup.trans.createFatherGlobalMapping();

        // Step 2: Collect leader globals referenced by follower2leader
        // (need ghost-pull for off-rank leaders)
        // Find the adj in the registry
        const AdjEntry *f2lEntry = nullptr;
        for (auto &adj : registry.adjs)
            if (adj.kind == follower2leaderKind)
            {
                f2lEntry = &adj;
                break;
            }

        // If follower2leader is not in registry, we need to look at the mesh data.
        // For now, we require it to be registered. The caller should ensure node2cell
        // or bnd2cell is built and registered before calling.
        DNDS_assert_info(f2lEntry != nullptr,
                         fmt::format("ComputeFollowMap: follower2leader adj {} not in registry",
                                     adjKindName(follower2leaderKind)));

        // We cannot directly iterate the adj via the registry (callbacks are type-erased).
        // Instead, we use a different approach: the registry has globalMappings for the
        // leader, and the follower2leader adj stores leader globals. We ghost-pull the
        // leaderLookup for all off-rank leader globals referenced.
        //
        // Problem: we can't access the adj data through callbacks. We need the raw data.
        // Solution: the caller passes the adj data directly to ComputeFollowMap.
        // For now, return empty — this will be connected in buildReorderRegistry.
        //
        // Actually, we redesign: ComputeFollowMap takes a direct reference to the
        // follower2leader pair data.

        // This function should not be called from here — see the overload below.
        DNDS_assert_info(false, "ComputeFollowMap: internal error — use the pair-based overload");
        return {};
    }

    /// Overload that takes raw follower2leader data (father-only, global entries).
    template <rowsize f2l_rs>
    static std::vector<MPI_int> ComputeFollowMapFromAdj(
        const ArrayAdjacencyPair<f2l_rs> &follower2leader,
        index nFollower,
        const ssp<GlobalOffsetsMapping> &leaderGM,
        const std::vector<MPI_int> &leaderTargetRanks,
        const MPIInfo &mpi)
    {
        DNDS_assert(leaderGM);
        DNDS_assert(follower2leader.father);

        // Step 1: Build a ghost-pullable lookup of leader's targetRanks.
        ArrayAdjacencyPair<1> leaderLookup;
        leaderLookup.InitPair("followMap_leaderLookup", mpi);
        index nLeader = static_cast<index>(leaderTargetRanks.size());
        leaderLookup.father->Resize(nLeader);
        for (index i = 0; i < nLeader; i++)
            leaderLookup(i, 0) = static_cast<index>(leaderTargetRanks[i]);
        leaderLookup.TransAttach();
        leaderLookup.trans.createFatherGlobalMapping();

        // Step 2: Collect off-rank leader globals from follower2leader entries.
        std::set<index> offRankLeaderGlobals;
        for (index i = 0; i < nFollower; i++)
            for (rowsize j = 0; j < follower2leader.RowSize(i); j++)
            {
                index leaderGlobal = follower2leader(i, j);
                if (leaderGlobal == UnInitIndex)
                    continue;
                auto [found, rank, val] = leaderGM->search(leaderGlobal);
                if (found && rank != mpi.rank)
                    offRankLeaderGlobals.insert(leaderGlobal);
            }

        // Step 3: Ghost-pull leaderLookup for off-rank leaders.
        std::vector<index> pullSet(offRankLeaderGlobals.begin(), offRankLeaderGlobals.end());
        leaderLookup.trans.createGhostMapping(pullSet);
        leaderLookup.trans.createMPITypes();
        leaderLookup.trans.pullOnce();

        // Step 4: For each follower, find min leader targetRank.
        std::vector<MPI_int> followMap(nFollower, mpi.size); // init to max
        for (index i = 0; i < nFollower; i++)
        {
            MPI_int minRank = mpi.size;
            for (rowsize j = 0; j < follower2leader.RowSize(i); j++)
            {
                index leaderGlobal = follower2leader(i, j);
                if (leaderGlobal == UnInitIndex)
                    continue;
                // Resolve to local-appended index in leaderLookup
                MPI_int rank;
                index val;
                bool found = leaderLookup.trans.pLGhostMapping->search_indexAppend(
                    leaderGlobal, rank, val);
                DNDS_assert(found);
                MPI_int leaderTarget = static_cast<MPI_int>(leaderLookup(val, 0));
                minRank = std::min(minRank, leaderTarget);
            }
            // If no leaders found (should not happen for valid mesh), stay put
            followMap[i] = (minRank < mpi.size) ? minRank : mpi.rank;
        }

        return followMap;
    }

    // =================================================================
    // ReorderPlan::build
    // =================================================================

    ReorderPlan ReorderPlan::build(
        const ReorderInput &input,
        const ReorderRegistry &registry,
        const MPIInfo &mpi)
    {
        ReorderPlan plan;

        // --- Step 1: Merge explicit maps into allMaps ---
        std::unordered_map<EntityKind, std::vector<MPI_int>> allMaps;
        for (auto &em : input.explicitMaps)
            allMaps[em.kind] = em.targetRanks;

        // --- Step 2: Compute follow maps ---
        // (Follow computation requires raw adj data. This is handled by
        //  the mesh's buildReorderRegistry which populates follow maps
        //  before calling build. For now, we accept pre-computed follows
        //  in the input.follows and note that the mesh wrapper handles
        //  the actual computation.)
        //
        // Note: follow maps that are already in allMaps (explicit) take
        // precedence. Follows are skipped for kinds already explicit.

        // Follow maps will be inserted by the mesh wrapper before calling build.
        // (See UnstructuredMesh::ReorderEntities in the mesh wrapper section.)

        // --- Step 3: Build PermutationTransfer per entity kind ---
        plan.reorderedKinds.clear();
        for (auto &[kind, ranks] : allMaps)
        {
            plan.reorderedKinds.insert(kind);
            auto gm = registry.getGlobalMapping(kind);
            DNDS_assert_info(gm, fmt::format("ReorderPlan::build: no global mapping for kind {}",
                                             entityKindName(kind)));
            plan.transfers[kind] = PermutationTransfer::fromPartition(ranks, gm, mpi);
        }

        // --- Step 4: Detect global local-only ---
        plan.isLocalOnly = true;
        for (auto &[kind, transfer] : plan.transfers)
            if (!transfer.isLocalOnly)
            {
                plan.isLocalOnly = false;
                break;
            }
        int globalFlag;
        int localFlag = plan.isLocalOnly ? 1 : 0;
        MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_LAND, mpi.comm);
        plan.isLocalOnly = (globalFlag != 0);

        // --- Step 5: Collect pull sets and build lookups ---
        for (auto kind : plan.reorderedKinds)
        {
            std::set<index> pullSetCollector;
            auto gm = registry.getGlobalMapping(kind);

            // Scan all registered adjs for entries pointing to this kind
            for (auto &adjEntry : registry.adjs)
            {
                EntityKind targetKind = adjEntry.kind.isIntraLevel()
                                            ? adjEntry.kind.from
                                            : adjEntry.kind.to;
                if (targetKind != kind)
                    continue;
                // We cannot iterate raw adj data here (type-erased callbacks).
                // The pull set must be pre-collected by the mesh wrapper and
                // passed via an extended mechanism.
                //
                // Alternative: the mesh wrapper collects pull sets and stores
                // them in the registry. For now, we use a simpler approach:
                // pull ALL off-rank globals that this rank's adjs reference.
                // This is done by the mesh wrapper before calling build.
            }

            // For simplicity in the initial implementation: if the transfer
            // is local-only, no ghost-pull is needed (all lookups are local).
            // For distributed, the mesh wrapper must pre-collect the pull set
            // and store it. We accept a pull set from the registry.
            //
            // Store empty pull set for now — the mesh wrapper will populate
            // lookups directly.

            std::vector<index> pullSet(pullSetCollector.begin(), pullSetCollector.end());
            plan.lookups[kind] = plan.transfers.at(kind).buildLookup(pullSet, mpi);
        }

        return plan;
    }

    // =================================================================
    // ReorderPlan::apply
    // =================================================================

    void ReorderPlan::apply(ReorderRegistry &registry, const MPIInfo &mpi) const
    {
        // Phase 1: REMAP all adj entries
        for (auto &adj : registry.adjs)
        {
            auto action = classifyAdj(adj.kind, reorderedKinds);
            if (action == AdjAction::REMAP ||
                action == AdjAction::RELOCATE_REMAP ||
                action == AdjAction::SELF)
            {
                EntityKind targetKind = adj.kind.isIntraLevel()
                                            ? adj.kind.from
                                            : adj.kind.to;
                auto it = lookups.find(targetKind);
                DNDS_assert_info(it != lookups.end(),
                                 fmt::format("ReorderPlan::apply REMAP: no lookup for target kind {}",
                                             entityKindName(targetKind)));
                if (adj.remapFn)
                    adj.remapFn(it->second);
            }
        }

        // Phase 2: RELOCATE all adj rows
        for (auto &adj : registry.adjs)
        {
            auto action = classifyAdj(adj.kind, reorderedKinds);
            if (action == AdjAction::RELOCATE ||
                action == AdjAction::RELOCATE_REMAP ||
                action == AdjAction::SELF)
            {
                EntityKind sourceKind = adj.kind.from;
                auto it = transfers.find(sourceKind);
                DNDS_assert_info(it != transfers.end(),
                                 fmt::format("ReorderPlan::apply RELOCATE: no transfer for source kind {}",
                                             entityKindName(sourceKind)));
                if (adj.relocateFn)
                    adj.relocateFn(it->second, mpi);
            }
        }

        // Phase 3: RELOCATE all companions of reordered kinds
        for (auto &comp : registry.companions)
        {
            if (reorderedKinds.count(comp.kind))
            {
                auto it = transfers.find(comp.kind);
                DNDS_assert_info(it != transfers.end(),
                                 fmt::format("ReorderPlan::apply COMPANION: no transfer for kind {}",
                                             entityKindName(comp.kind)));
                comp.fn(it->second, mpi);
            }
        }
    }

    // =================================================================
    // UnstructuredMesh::buildReorderRegistry
    // =================================================================

    ReorderRegistry UnstructuredMesh::buildReorderRegistry(
        const std::unordered_set<EntityKind> &destroyKinds)
    {
        ReorderRegistry reg;

        auto shouldSkip = [&](AdjKind kind)
        {
            return destroyKinds.count(kind.from) || destroyKinds.count(kind.to);
        };

        // --- Helper: register a tracked adj member ---
        auto regAdj = [&](AdjKind kind, auto &trackedPair)
        {
            if (!trackedPair.father || shouldSkip(kind))
                return;

            AdjRemapFn remap = [&trackedPair](const PermutationTransfer::LookupResult &lookup)
            {
                index nRows = trackedPair.father->Size();
                for (index i = 0; i < nRows; i++)
                    for (rowsize j = 0; j < trackedPair.RowSize(i); j++)
                    {
                        index &v = trackedPair(i, j);
                        if (v != UnInitIndex)
                            v = lookup.resolve(v);
                    }
            };

            AdjRelocateFn relocate = [&trackedPair](
                                         const PermutationTransfer &t, const MPIInfo &m)
            {
                t.transferRows(trackedPair, m);
            };

            reg.registerAdj(kind, std::move(remap), std::move(relocate),
                            adjKindName(kind));
        };

        // Register tracked adj members
        regAdj(Adj::Cell2Node, cell2node);
        regAdj(Adj::Cell2Cell, cell2cell);
        regAdj(Adj::Bnd2Node, bnd2node);
        regAdj(Adj::Bnd2Cell, bnd2cell);
        regAdj(Adj::Node2Cell, node2cell);
        regAdj(Adj::Node2Bnd, node2bnd);
        regAdj(Adj::Cell2Face, cell2face);
        regAdj(Adj::Face2Node, face2node);
        regAdj(Adj::Face2Cell, face2cell);
        regAdj(Adj::Face2Bnd, face2bnd);
        regAdj(Adj::Bnd2Face, bnd2face);
        regAdj(Adj::Cell2CellFace, cell2cellFace);

        // --- Helper: register a companion ---
        auto regComp = [&](EntityKind kind, auto &pair, const char *name)
        {
            if (!pair.father || destroyKinds.count(kind))
                return;
            reg.registerCompanion(kind, [&pair](const PermutationTransfer &t, const MPIInfo &m)
                                  { t.transferRows(pair, m); }, name);
        };

        // Register companions
        regComp(EntityKind::Cell, cellElemInfo, "cellElemInfo");
        regComp(EntityKind::Cell, cell2cellOrig, "cell2cellOrig");
        regComp(EntityKind::Node, coords, "coords");
        regComp(EntityKind::Node, node2nodeOrig, "node2nodeOrig");
        regComp(EntityKind::Bnd, bndElemInfo, "bndElemInfo");
        regComp(EntityKind::Bnd, bnd2bndOrig, "bnd2bndOrig");

        if (isPeriodic)
        {
            regComp(EntityKind::Cell, cell2nodePbi, "cell2nodePbi");
            regComp(EntityKind::Bnd, bnd2nodePbi, "bnd2nodePbi");
            if (!destroyKinds.count(EntityKind::Face))
                regComp(EntityKind::Face, face2nodePbi, "face2nodePbi");
        }
        if (!destroyKinds.count(EntityKind::Face))
            regComp(EntityKind::Face, faceElemInfo, "faceElemInfo");
        if (coordsElevDisp.father)
            regComp(EntityKind::Node, coordsElevDisp, "coordsElevDisp");
        if (nodeWallDist.father)
            regComp(EntityKind::Node, nodeWallDist, "nodeWallDist");

        // --- Register global mappings ---
        auto getGM = [](const auto &pair) -> ssp<GlobalOffsetsMapping>
        {
            if (pair.father && pair.father->pLGlobalMapping)
                return pair.father->pLGlobalMapping;
            return nullptr;
        };

        auto firstValid = [](std::initializer_list<ssp<GlobalOffsetsMapping>> candidates)
            -> ssp<GlobalOffsetsMapping>
        {
            for (auto &gm : candidates)
                if (gm)
                    return gm;
            return nullptr;
        };

        if (auto gm = firstValid({getGM(cell2node), getGM(cell2cell), getGM(cell2face)}))
            reg.registerGlobalMapping(EntityKind::Cell, gm);
        if (auto gm = coords.father ? coords.father->pLGlobalMapping : nullptr)
            reg.registerGlobalMapping(EntityKind::Node, gm);
        if (auto gm = firstValid({getGM(bnd2node), getGM(bnd2cell), getGM(bnd2face)}))
            reg.registerGlobalMapping(EntityKind::Bnd, gm);
        if (auto gm = firstValid({getGM(face2node), getGM(face2cell), getGM(face2bnd)}))
            reg.registerGlobalMapping(EntityKind::Face, gm);

        return reg;
    }

    // =================================================================
    // UnstructuredMesh::buildReorderPlan
    // =================================================================

    ReorderPlan UnstructuredMesh::buildReorderPlan(const ReorderInput &input)
    {
        auto reg = buildReorderRegistry(input.destroyKinds);

        // Augment input with default follows: Node, Bnd follow Cell
        // if Cell is explicit and Node/Bnd are not.
        ReorderInput augmented = input;
        std::unordered_set<EntityKind> explicitKinds;
        for (auto &em : augmented.explicitMaps)
            explicitKinds.insert(em.kind);

        // Add default follows
        if (explicitKinds.count(EntityKind::Cell))
        {
            if (!explicitKinds.count(EntityKind::Node) && node2cell.father)
            {
                augmented.follows.push_back(
                    FollowSpec{EntityKind::Node, EntityKind::Cell, Adj::Node2Cell});
            }
            if (!explicitKinds.count(EntityKind::Bnd) && bnd2cell.father)
            {
                augmented.follows.push_back(
                    FollowSpec{EntityKind::Bnd, EntityKind::Cell, Adj::Bnd2Cell});
            }
        }

        // Compute follow maps and merge into the input
        std::unordered_map<EntityKind, std::vector<MPI_int>> allMaps;
        for (auto &em : augmented.explicitMaps)
            allMaps[em.kind] = em.targetRanks;

        for (auto &spec : augmented.follows)
        {
            if (allMaps.count(spec.follower))
                continue; // explicit takes precedence

            auto leaderIt = allMaps.find(spec.leader);
            DNDS_assert_info(leaderIt != allMaps.end(),
                             fmt::format("buildReorderPlan: follow spec references leader {} "
                                         "which has no map",
                                         entityKindName(spec.leader)));

            // Use the raw adj data for follow computation
            // We need the follower->leader adj data. Dispatch by known kinds:
            std::vector<MPI_int> followMap;
            auto followerGM = reg.getGlobalMapping(spec.follower);
            DNDS_assert_info(followerGM,
                             fmt::format("buildReorderPlan: no global mapping for follower {}",
                                         entityKindName(spec.follower)));
            index nFollower = followerGM->RLengths()[mpi.rank];

            if (spec.follower2leader == Adj::Node2Cell && node2cell.father)
            {
                followMap = ComputeFollowMapFromAdj(
                    static_cast<const tAdjPair &>(node2cell),
                    nFollower, reg.getGlobalMapping(spec.leader),
                    leaderIt->second, mpi);
            }
            else if (spec.follower2leader == Adj::Bnd2Cell && bnd2cell.father)
            {
                followMap = ComputeFollowMapFromAdj(
                    static_cast<const tAdj2Pair &>(bnd2cell),
                    nFollower, reg.getGlobalMapping(spec.leader),
                    leaderIt->second, mpi);
            }
            else if (spec.follower2leader == Adj::Node2Bnd && node2bnd.father)
            {
                followMap = ComputeFollowMapFromAdj(
                    static_cast<const tAdjPair &>(node2bnd),
                    nFollower, reg.getGlobalMapping(spec.leader),
                    leaderIt->second, mpi);
            }
            else
            {
                DNDS_assert_info(false,
                                 fmt::format("buildReorderPlan: unsupported follow adj {}",
                                             adjKindName(spec.follower2leader)));
            }

            allMaps[spec.follower] = std::move(followMap);
        }

        // Now build the plan with all maps (explicit + follow)
        // We need to pass allMaps into ReorderPlan::build.
        // Convert allMaps to explicitMaps format for build:
        ReorderInput finalInput;
        for (auto &[kind, ranks] : allMaps)
            finalInput.explicitMaps.push_back(EntityReorderMap{kind, ranks});
        finalInput.destroyKinds = input.destroyKinds;

        return ReorderPlan::build(finalInput, reg, mpi);
    }

    // =================================================================
    // UnstructuredMesh::ReorderEntities
    // =================================================================

    void UnstructuredMesh::ReorderEntities(const ReorderInput &input)
    {
        // Step 0: Validate precondition
        DNDS_assert_info(adjPrimaryState == Adj_PointToGlobal,
                         "ReorderEntities: adjPrimaryState must be Adj_PointToGlobal");

        // Step 1: Build registry (with destroy skip)
        auto reg = buildReorderRegistry(input.destroyKinds);

        // Step 2: Build plan (with follows computed)
        // We replicate the logic from buildReorderPlan but use the mutable registry
        ReorderInput augmented = input;
        std::unordered_set<EntityKind> explicitKinds;
        for (auto &em : augmented.explicitMaps)
            explicitKinds.insert(em.kind);

        if (explicitKinds.count(EntityKind::Cell))
        {
            if (!explicitKinds.count(EntityKind::Node) && node2cell.father)
                augmented.follows.push_back(
                    FollowSpec{EntityKind::Node, EntityKind::Cell, Adj::Node2Cell});
            if (!explicitKinds.count(EntityKind::Bnd) && bnd2cell.father)
                augmented.follows.push_back(
                    FollowSpec{EntityKind::Bnd, EntityKind::Cell, Adj::Bnd2Cell});
        }

        // Compute follows
        std::unordered_map<EntityKind, std::vector<MPI_int>> allMaps;
        for (auto &em : augmented.explicitMaps)
            allMaps[em.kind] = em.targetRanks;

        for (auto &spec : augmented.follows)
        {
            if (allMaps.count(spec.follower))
                continue;
            auto leaderIt = allMaps.find(spec.leader);
            DNDS_assert(leaderIt != allMaps.end());

            auto followerGM = reg.getGlobalMapping(spec.follower);
            DNDS_assert(followerGM);
            index nFollower = followerGM->RLengths()[mpi.rank];

            std::vector<MPI_int> followMap;
            if (spec.follower2leader == Adj::Node2Cell && node2cell.father)
                followMap = ComputeFollowMapFromAdj(
                    static_cast<const tAdjPair &>(node2cell),
                    nFollower, reg.getGlobalMapping(spec.leader),
                    leaderIt->second, mpi);
            else if (spec.follower2leader == Adj::Bnd2Cell && bnd2cell.father)
                followMap = ComputeFollowMapFromAdj(
                    static_cast<const tAdj2Pair &>(bnd2cell),
                    nFollower, reg.getGlobalMapping(spec.leader),
                    leaderIt->second, mpi);
            else if (spec.follower2leader == Adj::Node2Bnd && node2bnd.father)
                followMap = ComputeFollowMapFromAdj(
                    static_cast<const tAdjPair &>(node2bnd),
                    nFollower, reg.getGlobalMapping(spec.leader),
                    leaderIt->second, mpi);
            else
                DNDS_assert(false);

            allMaps[spec.follower] = std::move(followMap);
        }

        // Build plan
        ReorderInput finalInput;
        for (auto &[kind, ranks] : allMaps)
            finalInput.explicitMaps.push_back(EntityReorderMap{kind, ranks});
        finalInput.destroyKinds = input.destroyKinds;

        auto plan = ReorderPlan::build(finalInput, reg, mpi);

        // Step 3: Destroy adjacencies for destroyKinds
        auto destroyAdj = [&](auto &trackedPair)
        {
            trackedPair.father.reset();
            trackedPair.son.reset();
            trackedPair.idx = AdjIndexInfo{};
        };
        for (auto kind : input.destroyKinds)
        {
            if (kind == EntityKind::Face)
            {
                destroyAdj(cell2face);
                destroyAdj(face2node);
                destroyAdj(face2cell);
                destroyAdj(face2bnd);
                destroyAdj(bnd2face);
                destroyAdj(cell2cellFace);
                faceElemInfo.father.reset();
                faceElemInfo.son.reset();
                if (isPeriodic)
                {
                    face2nodePbi.father.reset();
                    face2nodePbi.son.reset();
                }
                adjFacialState = Adj_Unknown;
                adjC2FState = Adj_Unknown;
                adjC2CFaceState = Adj_Unknown;
            }
        }

        // Step 4: Apply plan (REMAP entries, RELOCATE rows + companions)
        plan.apply(reg, mpi);

        // Step 5: Rebuild global mappings for reordered entities
        for (auto kind : plan.reorderedKinds)
        {
            if (kind == EntityKind::Cell && cell2node.father)
            {
                cell2node.TransAttach();
                cell2node.trans.createFatherGlobalMapping();
                // Borrow to other cell-parallel arrays
                if (cell2cell.father)
                    cell2cell.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
                if (cellElemInfo.father)
                    cellElemInfo.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
                if (cell2cellOrig.father)
                    cell2cellOrig.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
            }
            else if (kind == EntityKind::Node && coords.father)
            {
                coords.TransAttach();
                coords.trans.createFatherGlobalMapping();
                if (node2nodeOrig.father)
                    node2nodeOrig.father->pLGlobalMapping = coords.father->pLGlobalMapping;
            }
            else if (kind == EntityKind::Bnd && bnd2node.father)
            {
                bnd2node.TransAttach();
                bnd2node.trans.createFatherGlobalMapping();
                if (bndElemInfo.father)
                    bndElemInfo.father->pLGlobalMapping = bnd2node.father->pLGlobalMapping;
                if (bnd2bndOrig.father)
                    bnd2bndOrig.father->pLGlobalMapping = bnd2node.father->pLGlobalMapping;
            }
            else if (kind == EntityKind::Face && face2node.father)
            {
                face2node.TransAttach();
                face2node.trans.createFatherGlobalMapping();
                if (faceElemInfo.father)
                    faceElemInfo.father->pLGlobalMapping = face2node.father->pLGlobalMapping;
            }
        }

        // Step 6: Update idx states
        auto markGlobalIfBuilt = [](auto &trackedPair)
        {
            if (trackedPair.father && trackedPair.idx.state() != Adj_Unknown)
                trackedPair.idx.markGlobal();
        };
        // For all tracked adj that were affected (non-SKIP), mark global.
        // Simplification: mark all existing adj as global since we're in
        // Adj_PointToGlobal state overall.
        if (cell2node.father)
            cell2node.idx.markGlobal();
        if (cell2cell.father)
            cell2cell.idx.markGlobal();
        if (bnd2node.father)
            bnd2node.idx.markGlobal();
        if (bnd2cell.father)
            bnd2cell.idx.markGlobal();
        if (node2cell.father)
            node2cell.idx.markGlobal();
        if (node2bnd.father)
            node2bnd.idx.markGlobal();

        // Step 7: Update mesh-level state
        adjPrimaryState = Adj_PointToGlobal;
        if (node2cell.father)
            adjN2CBState = Adj_PointToGlobal;

        // Invalidate local vectors
        cell2parentCell.clear();
        node2parentNode.clear();
        node2bndNode.clear();
        vtkCell2nodeOffsets.clear();
        vtkCellType.clear();
        vtkCell2node.clear();
        nodeRecreated2nodeLocal.clear();
        localPartitionStarts.clear();
    }

} // namespace DNDS::Geom
