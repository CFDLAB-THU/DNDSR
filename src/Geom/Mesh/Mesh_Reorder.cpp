/// @file Mesh_Reorder.cpp
/// @brief Implementation of ReorderPlan::build, ReorderPlan::apply, and
///        UnstructuredMesh::buildReorderRegistry / ReorderEntities.

#include "ReorderPlan.hpp"
#include "Mesh.hpp"
#include "Mesh_CellPermutation.hpp"

#include <algorithm>
#include <set>

namespace DNDS::Geom
{
    // =================================================================
    // ComputeFollowMapFromAdj: derive follower placement from leader
    // =================================================================

    /// Compute a follow map for `follower` based on `leader`'s explicit map.
    ///
    /// Uses the support adjacency follower->leader (e.g., node2cell) to
    /// determine: for each follower entity, go to min(leader.targetRank)
    /// over all leaders referencing it.
    ///
    /// @param follower2leader  Raw follower->leader adj (father-only, global entries).
    /// @param nFollower        Number of follower entities (father size).
    /// @param leaderGM         Global mapping for the leader entity kind.
    /// @param leaderTargetRanks  Per-leader-slot target rank (the explicit map).
    /// @param mpi              MPI communicator.
    /// @return Per-follower-slot target rank.
    /// @warning Collective.
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
        // Follow computation is handled by the mesh wrapper (resolveFollows)
        // before calling build. By the time we reach here, all entity kinds
        // should be in explicitMaps. The input.follows field is not processed
        // here — it is a mesh-wrapper-level concept.
        DNDS_assert_info(input.follows.empty(),
                         "ReorderPlan::build: input.follows must be empty; "
                         "use UnstructuredMesh::resolveFollows to compute follow maps "
                         "before calling build");

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

            // Use pre-collected pull sets from the registry
            std::vector<index> pullSet;
            auto psIt = registry.pullSets.find(kind);
            if (psIt != registry.pullSets.end())
                pullSet = psIt->second;

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

        // --- Pre-collect pull sets ---
        // For each entity kind, collect off-rank globals from adj entries targeting it.
        auto collectPS = [&](EntityKind targetKind, const auto &adjPair, auto targetGM)
        {
            if (!adjPair.father || !targetGM)
                return;
            auto &ps = reg.pullSets[targetKind];
            DNDS::index nRows = adjPair.father->Size();
            for (DNDS::index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adjPair.RowSize(i); j++)
                {
                    DNDS::index v = adjPair(i, j);
                    if (v == UnInitIndex)
                        continue;
                    auto [found, rank, val] = targetGM->search(v);
                    if (found && rank != mpi.rank)
                        ps.push_back(v);
                }
        };

        // Cell as target (from: bnd2cell, face2cell, node2cell, cell2cell)
        auto cellGM = reg.getGlobalMapping(EntityKind::Cell);
        collectPS(EntityKind::Cell, bnd2cell, cellGM);
        collectPS(EntityKind::Cell, node2cell, cellGM);
        collectPS(EntityKind::Cell, cell2cell, cellGM);
        if (!shouldSkip(Adj::Face2Cell))
            collectPS(EntityKind::Cell, face2cell, cellGM);

        // Node as target (from: cell2node, bnd2node, face2node)
        auto nodeGM = reg.getGlobalMapping(EntityKind::Node);
        collectPS(EntityKind::Node, cell2node, nodeGM);
        collectPS(EntityKind::Node, bnd2node, nodeGM);
        if (!shouldSkip(Adj::Face2Node))
            collectPS(EntityKind::Node, face2node, nodeGM);

        // Bnd as target (from: node2bnd, face2bnd)
        auto bndGM = reg.getGlobalMapping(EntityKind::Bnd);
        collectPS(EntityKind::Bnd, node2bnd, bndGM);
        if (!shouldSkip(Adj::Face2Bnd))
            collectPS(EntityKind::Bnd, face2bnd, bndGM);

        // Deduplicate and sort pull sets
        for (auto &[kind, ps] : reg.pullSets)
        {
            std::sort(ps.begin(), ps.end());
            ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
        }

        return reg;
    }

    // =================================================================
    // UnstructuredMesh::resolveFollows
    // =================================================================

    ReorderInput UnstructuredMesh::resolveFollows(
        const ReorderInput &input,
        const ReorderRegistry &reg)
    {
        ReorderInput augmented = input;
        std::unordered_set<EntityKind> explicitKinds;
        for (auto &em : augmented.explicitMaps)
            explicitKinds.insert(em.kind);

        // When follows is empty, apply default policy: all non-explicit
        // entity kinds with a support adjacency to an explicit kind follow it.
        // Currently: Node and Bnd follow Cell if Cell is explicit.
        if (augmented.follows.empty())
        {
            if (explicitKinds.count(EntityKind::Cell))
            {
                if (!explicitKinds.count(EntityKind::Node) && node2cell.father)
                    augmented.follows.push_back(
                        FollowSpec{EntityKind::Node, EntityKind::Cell, Adj::Node2Cell});
                if (!explicitKinds.count(EntityKind::Bnd) && bnd2cell.father)
                    augmented.follows.push_back(
                        FollowSpec{EntityKind::Bnd, EntityKind::Cell, Adj::Bnd2Cell});
            }
        }

        // Compute follow maps and merge into explicit maps
        std::unordered_map<EntityKind, std::vector<MPI_int>> allMaps;
        for (auto &em : augmented.explicitMaps)
            allMaps[em.kind] = em.targetRanks;

        for (auto &spec : augmented.follows)
        {
            if (allMaps.count(spec.follower))
                continue; // explicit takes precedence

            auto leaderIt = allMaps.find(spec.leader);
            DNDS_assert_info(leaderIt != allMaps.end(),
                             fmt::format("resolveFollows: follow spec references leader {} "
                                         "which has no map",
                                         entityKindName(spec.leader)));

            auto followerGM = reg.getGlobalMapping(spec.follower);
            DNDS_assert_info(followerGM,
                             fmt::format("resolveFollows: no global mapping for follower {}",
                                         entityKindName(spec.follower)));
            index nFollower = followerGM->RLengths()[mpi.rank];

            std::vector<MPI_int> followMap;
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
                                 fmt::format("resolveFollows: unsupported follow adj {}",
                                             adjKindName(spec.follower2leader)));
            }

            allMaps[spec.follower] = std::move(followMap);
        }

        // Build finalised input with all maps explicit
        ReorderInput finalInput;
        for (auto &[kind, ranks] : allMaps)
            finalInput.explicitMaps.push_back(EntityReorderMap{kind, ranks});
        finalInput.destroyKinds = input.destroyKinds;
        return finalInput;
    }

    // =================================================================
    // UnstructuredMesh::buildReorderPlan
    // =================================================================

    ReorderPlan UnstructuredMesh::buildReorderPlan(const ReorderInput &input)
    {
        auto reg = buildReorderRegistry(input.destroyKinds);
        auto finalInput = resolveFollows(input, reg);
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

        // Step 2: Resolve follows and build plan
        auto finalInput = resolveFollows(input, reg);
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
                cell2node.father->createGlobalMapping();
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
                coords.father->createGlobalMapping();
                if (node2nodeOrig.father)
                    node2nodeOrig.father->pLGlobalMapping = coords.father->pLGlobalMapping;
            }
            else if (kind == EntityKind::Bnd && bnd2node.father)
            {
                bnd2node.father->createGlobalMapping();
                if (bndElemInfo.father)
                    bndElemInfo.father->pLGlobalMapping = bnd2node.father->pLGlobalMapping;
                if (bnd2bndOrig.father)
                    bnd2bndOrig.father->pLGlobalMapping = bnd2node.father->pLGlobalMapping;
            }
            else if (kind == EntityKind::Face && face2node.father)
            {
                face2node.father->createGlobalMapping();
                if (faceElemInfo.father)
                    faceElemInfo.father->pLGlobalMapping = face2node.father->pLGlobalMapping;
            }
        }

        // Step 5b: Re-attach transformers with empty sons (prepare for ghost rebuild)
        // After transferRows, sons are null. The rebuild pipeline
        // (RecoverNode2CellAndNode2Bnd, etc.) needs TransAttach-ready pairs.
        auto reattach = [&](auto &pair)
        {
            if (!pair.father)
                return;
            using TArr = typename std::remove_reference_t<decltype(pair)>::t_arr;
            if (!pair.son)
                pair.son = make_ssp<TArr>(ObjName{"reorder.son"}, mpi);
            pair.TransAttach();
        };

        for (auto kind : plan.reorderedKinds)
        {
            if (kind == EntityKind::Cell)
            {
                reattach(cell2node);
                reattach(cell2cell);
                reattach(cellElemInfo);
                reattach(cell2cellOrig);
                if (isPeriodic)
                    reattach(cell2nodePbi);
            }
            else if (kind == EntityKind::Node)
            {
                reattach(coords);
                reattach(node2nodeOrig);
            }
            else if (kind == EntityKind::Bnd)
            {
                reattach(bnd2node);
                reattach(bnd2cell);
                reattach(bndElemInfo);
                reattach(bnd2bndOrig);
                if (isPeriodic)
                    reattach(bnd2nodePbi);
            }
        }
        // Also reattach inverse adjacencies if they exist
        if (node2cell.father)
            reattach(node2cell);
        if (node2bnd.father)
            reattach(node2bnd);

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

    // =================================================================
    // UnstructuredMesh::ReorderLocalCells (new, using ReorderEntities)
    // =================================================================

    void UnstructuredMesh::ReorderLocalCells(int nParts, int nPartsInner)
    {
        DNDS_assert(this->adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(cell2node.isLocal() && bnd2node.isLocal() &&
                    cell2cell.isLocal() && bnd2cell.isLocal());
        nParts = std::max(nParts, 1);
        nPartsInner = std::max(nPartsInner, 1);

        // --- Section A: Compute cell permutation (same as legacy) ---
        // We need the local face-adjacency graph and cell2cell.
        // Convert to global first to get the permutation computation right,
        // then convert back and redo with the framework.
        // Actually: ComputeCellPermutation works on LOCAL indices (it uses
        // cell2cell in local state). So compute the permutation first.
        auto cell2cellFaceV = this->GetCell2CellFaceVLocal();

        auto perm = detail::ComputeCellPermutation(
            cell2cellFaceV, cell2cell, NumCell(), nParts, nPartsInner);
        this->localPartitionStarts = std::move(perm.localPartitionStarts);

        MPI::AllreduceOneIndex(perm.bwOld, MPI_MAX, mpi);
        MPI::AllreduceOneIndex(perm.bwNew, MPI_MAX, mpi);
        if (mpi.rank == mRank)
            log() << fmt::format("UnstructuredMesh === ReorderLocalCells, nPart0 [{}], "
                                 "got reordering, bw [{}] to [{}]",
                                 nParts, perm.bwOld, perm.bwNew)
                  << std::endl;

        // --- Convert to global ---
        if (this->adjFacialState == Adj_PointToLocal && face2cell.isBuilt())
            this->AdjLocal2GlobalFacial();
        if (this->adjC2FState == Adj_PointToLocal && cell2face.isBuilt())
            this->AdjLocal2GlobalC2F();
        if (this->adjC2CFaceState == Adj_PointToLocal && cell2cellFace.isBuilt())
            this->AdjLocal2GlobalC2CFace();
        if (this->adjN2CBState == Adj_PointToLocal && node2cell.isBuilt())
            this->AdjLocal2GlobalN2CB();
        this->AdjLocal2GlobalPrimary();

        // --- Build cell partition map (all local, use permutation) ---
        // fromLocalPermutation expects old2new. perm.cellOld2New is that.
        std::vector<MPI_int> cellPartition(NumCell(), mpi.rank);

        // We need to communicate the permutation to ReorderEntities.
        // The framework's fromPartition computes new globals automatically,
        // but for a local permutation we want a specific ordering.
        // Use fromLocalPermutation by overriding the transfer after build.
        //
        // Actually, the simplest approach: call ReorderEntities with the
        // all-same-rank partition (which is an identity from the framework's
        // perspective), then separately apply the local permutation.
        //
        // Better: don't use ReorderEntities here. Instead, use the framework
        // components directly with fromLocalPermutation.

        // Build a PermutationTransfer from the computed permutation
        auto cellGM = cell2node.father->pLGlobalMapping;
        DNDS_assert(cellGM);
        auto cellTransfer = PermutationTransfer::fromLocalPermutation(
            perm.cellOld2New, cellGM, mpi);
        DNDS_assert(cellTransfer.isLocalOnly);

        // Build lookup for cell entry remapping
        // Collect off-rank cell globals from adj entries targeting cells
        std::set<index> cellPullSet;
        auto addCellRefs = [&](const auto &adj, index nRows)
        {
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                {
                    index v = adj(i, j);
                    if (v == UnInitIndex)
                        continue;
                    auto [found, rank, val] = cellGM->search(v);
                    if (found && rank != mpi.rank)
                        cellPullSet.insert(v);
                }
        };
        addCellRefs(cell2cell, NumCell());
        addCellRefs(bnd2cell, NumBnd());
        if (node2cell.father)
            addCellRefs(node2cell, NumNode());
        if (face2cell.father)
            addCellRefs(face2cell, NumFace());
        if (cell2cellFace.father)
            addCellRefs(cell2cellFace, NumCell());

        std::vector<index> pullVec(cellPullSet.begin(), cellPullSet.end());
        auto cellLookup = cellTransfer.buildLookup(pullVec, mpi);

        // --- REMAP: update cell indices in xxx2cell adjacencies ---
        auto remapCellEntries = [&](auto &adj, index nRows)
        {
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                {
                    index &v = adj(i, j);
                    if (v != UnInitIndex)
                        v = cellLookup.resolve(v);
                }
        };

        remapCellEntries(cell2cell, NumCell());
        remapCellEntries(bnd2cell, NumBnd());
        if (node2cell.father)
            remapCellEntries(node2cell, NumNode());
        if (face2cell.father)
            remapCellEntries(face2cell, NumFace());
        if (cell2cellFace.father)
            remapCellEntries(cell2cellFace, NumCell());

        // --- RELOCATE: permute cell2xxx rows ---
        cellTransfer.transferRows(cell2node, mpi);
        cellTransfer.transferRows(cell2cell, mpi);
        cellTransfer.transferRows(cellElemInfo, mpi);
        cellTransfer.transferRows(cell2cellOrig, mpi);
        if (cell2face.father)
            cellTransfer.transferRows(cell2face, mpi);
        if (cell2cellFace.father)
            cellTransfer.transferRows(cell2cellFace, mpi);
        if (isPeriodic && cell2nodePbi.father)
            cellTransfer.transferRows(cell2nodePbi, mpi);

        // --- Rebuild global mapping ---
        cell2node.father->createGlobalMapping();
        if (cell2cell.father)
            cell2cell.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
        if (cellElemInfo.father)
            cellElemInfo.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
        if (cell2cellOrig.father)
            cell2cellOrig.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;

        // --- Rebuild ghost mappings (local-only optimization) ---
        // Permute the ghost index list to match new cell globals
        {
            std::vector<index> ghostCellGlobalsNew;
            if (cell2node.trans.pLGhostMapping)
            {
                ghostCellGlobalsNew = cell2node.trans.pLGhostMapping->ghostIndex;
                for (index &g : ghostCellGlobalsNew)
                    g = cellLookup.resolve(g);
            }
            // Reattach son and create ghost mapping
            if (!cell2node.son)
                cell2node.son = make_ssp<tAdjPair::t_arr>(ObjName{"reorder.son"}, mpi);
            cell2node.TransAttach();
            cell2node.trans.createFatherGlobalMapping();
            cell2node.trans.createGhostMapping(ghostCellGlobalsNew);
            cell2node.trans.createMPITypes();
            cell2node.trans.pullOnce();
        }
        // Borrow ghost indexing for other cell arrays
        {
            auto borrowAndPull = [&](auto &pair)
            {
                if (!pair.father)
                    return;
                if (!pair.son)
                {
                    using TArr = typename std::remove_reference_t<decltype(pair)>::t_arr;
                    pair.son = make_ssp<TArr>(ObjName{"reorder.son"}, mpi);
                }
                pair.TransAttach();
                pair.trans.BorrowGGIndexing(cell2node.trans);
                pair.trans.createMPITypes();
                pair.trans.pullOnce();
            };
            borrowAndPull(cell2cell);
            borrowAndPull(cell2cellOrig);
            borrowAndPull(cellElemInfo);
            if (cell2face.father)
                borrowAndPull(cell2face);
            if (cell2cellFace.father)
                borrowAndPull(cell2cellFace);
            if (isPeriodic && cell2nodePbi.father)
                borrowAndPull(cell2nodePbi);
        }

        // --- Re-wire target mappings ---
        {
            auto cellGhostMap = cell2node.trans.pLGhostMapping;
            cell2cell.idx.wireTargetMapping(cellGhostMap);
            bnd2cell.idx.wireTargetMapping(cellGhostMap);
            if (cell2cellFace.father && cell2cellFace.idx.isWired())
                cell2cellFace.idx.wireTargetMapping(cellGhostMap);
            if (node2cell.father && node2cell.idx.isWired())
                node2cell.idx.wireTargetMapping(cellGhostMap);
            if (face2cell.father && face2cell.idx.isWired())
                face2cell.idx.wireTargetMapping(cellGhostMap);
        }

        // --- Pull ghost data for non-cell adjacencies (face2cell, node2cell) ---
        if (face2cell.father && face2cell.trans.pLGhostMapping)
            face2cell.trans.pullOnce();
        if (node2cell.father && node2cell.trans.pLGhostMapping)
            node2cell.trans.pullOnce();
        DNDS_check_throw_info(bnd2cell.father && bnd2cell.trans.pLGhostMapping,
                              "ReorderLocalCells: bnd2cell must have ghost mapping for pull");
        bnd2cell.trans.pullOnce();

        // --- Convert back to local ---
        if (this->adjFacialState == Adj_PointToGlobal && face2cell.isBuilt())
            this->AdjGlobal2LocalFacial();
        if (this->adjC2FState == Adj_PointToGlobal && cell2face.isBuilt())
            this->AdjGlobal2LocalC2F();
        if (this->adjC2CFaceState == Adj_PointToGlobal && cell2cellFace.isBuilt())
            this->AdjGlobal2LocalC2CFace();
        if (this->adjN2CBState == Adj_PointToGlobal && node2cell.isBuilt())
            this->AdjGlobal2LocalN2CB();
        this->AdjGlobal2LocalPrimary();

        if (mpi.rank == mRank)
            log() << fmt::format("UnstructuredMesh === ReorderLocalCells finished") << std::endl;
    }

} // namespace DNDS::Geom
