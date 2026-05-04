#pragma once
/// @file ReorderPlan.hpp
/// @brief Distributed entity reordering framework: ReorderRegistry, ReorderPlan, ReorderInput.
///
/// Two-layer architecture:
/// - ReorderRegistry: dynamic set of callbacks (adj remap/relocate + companion relocate)
/// - ReorderPlan: standalone computed transfers + lookups, applies via callbacks
///
/// UnstructuredMesh provides buildReorderRegistry() and ReorderEntities() convenience methods.

#include "DNDS/PermutationTransfer.hpp"
#include "MeshConnectivity.hpp"

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace DNDS::Geom
{
    // =================================================================
    // EntityReorderMap: per-entity target rank assignment
    // =================================================================

    /// Per-entity reorder specification: where each owned entity goes.
    struct EntityReorderMap
    {
        EntityKind kind;
        /// Per father slot: target rank after reorder. Size == father size.
        std::vector<MPI_int> targetRanks;
    };

    // =================================================================
    // FollowSpec: how one entity kind derives its placement from another
    // =================================================================

    /// Specification for follow-placement: entity `follower` derives its
    /// target rank from entity `leader` via the support adjacency
    /// `follower2leader` (e.g., Node follows Cell via node2cell).
    ///
    /// Assignment rule: follower entity goes to min(leader.targetRank)
    /// over all leaders referencing it.
    struct FollowSpec
    {
        EntityKind follower;     ///< Entity kind to derive map for.
        EntityKind leader;       ///< Explicit-map entity kind to follow.
        AdjKind follower2leader; ///< Support adj: follower -> leader.
    };

    // =================================================================
    // ReorderInput: what the caller provides
    // =================================================================

    /// Input to the reorder framework.
    struct ReorderInput
    {
        /// Explicit reorder maps (caller-provided).
        std::vector<EntityReorderMap> explicitMaps;

        /// Follow specifications (framework computes follow maps from these).
        /// Default follows (Node, Bnd follow Cell) are added automatically
        /// when Cell is in explicitMaps and Node/Bnd are not.
        std::vector<FollowSpec> follows;

        /// Entity kinds whose adjacencies should be destroyed before reorder
        /// (not reordered, not remapped -- just wiped). Typically {Face}.
        std::unordered_set<EntityKind> destroyKinds;
    };

    // =================================================================
    // Adjacency action classification
    // =================================================================

    /// Action to take on an adjacency during reorder.
    enum class AdjAction
    {
        SKIP,           ///< Neither source nor target reordered.
        RELOCATE,       ///< Source reordered, target not: move rows.
        REMAP,          ///< Target reordered, source not: update entries.
        RELOCATE_REMAP, ///< Both reordered: update entries then move rows.
        SELF,           ///< Intra-level (A==A): update entries then move rows.
    };

    /// Classify an adjacency given the set of reordered entity kinds.
    inline AdjAction classifyAdj(AdjKind adj, const std::unordered_set<EntityKind> &reorderedKinds)
    {
        if (adj.isIntraLevel())
            return reorderedKinds.count(adj.from) ? AdjAction::SELF : AdjAction::SKIP;

        bool fromReordered = reorderedKinds.count(adj.from) > 0;
        bool toReordered = reorderedKinds.count(adj.to) > 0;

        if (!fromReordered && !toReordered)
            return AdjAction::SKIP;
        if (fromReordered && !toReordered)
            return AdjAction::RELOCATE;
        if (!fromReordered && toReordered)
            return AdjAction::REMAP;
        return AdjAction::RELOCATE_REMAP;
    }

    // =================================================================
    // ReorderRegistry: dynamic set of arrays participating in reorder
    // =================================================================

    /// Callback invoked during REMAP phase for an adjacency array.
    using AdjRemapFn = std::function<void(const PermutationTransfer::LookupResult &lookup)>;

    /// Callback invoked during RELOCATE phase for an adjacency or companion array.
    using AdjRelocateFn = std::function<void(const PermutationTransfer &transfer, const MPIInfo &mpi)>;

    /// One registered adjacency entry.
    struct AdjEntry
    {
        AdjKind kind;
        AdjRemapFn remapFn;       ///< Remap entries (null if not needed).
        AdjRelocateFn relocateFn; ///< Relocate rows (null if not needed).
        std::string name;
    };

    /// One registered companion entry.
    struct CompanionEntry
    {
        EntityKind kind;  ///< Entity kind this array is parallel to.
        AdjRelocateFn fn; ///< Callback to relocate the array.
        std::string name;
    };

    /// Dynamic set of arrays that participate in a reorder operation.
    /// Built by UnstructuredMesh::buildReorderRegistry() for mesh members,
    /// extended by external code (solver, evaluator) before plan application.
    struct ReorderRegistry
    {
        /// All adjacency entries (mesh members + external).
        std::vector<AdjEntry> adjs;

        /// All companion entries (mesh members + external).
        std::vector<CompanionEntry> companions;

        /// Global offsets mappings per entity kind (for PermutationTransfer).
        std::unordered_map<EntityKind, ssp<GlobalOffsetsMapping>> globalMappings;

        /// Register an adjacency with type-erased callbacks.
        void registerAdj(AdjKind kind, AdjRemapFn remap, AdjRelocateFn relocate,
                         std::string name = {})
        {
            adjs.push_back(AdjEntry{kind, std::move(remap), std::move(relocate), std::move(name)});
        }

        /// Register a companion with a type-erased relocate callback.
        void registerCompanion(EntityKind kind, AdjRelocateFn fn, std::string name = {})
        {
            companions.push_back(CompanionEntry{kind, std::move(fn), std::move(name)});
        }

        /// Register a GlobalOffsetsMapping for an entity kind.
        void registerGlobalMapping(EntityKind kind, ssp<GlobalOffsetsMapping> gm)
        {
            globalMappings[kind] = std::move(gm);
        }

        /// Get a registered global mapping (nullptr if not registered).
        ssp<GlobalOffsetsMapping> getGlobalMapping(EntityKind kind) const
        {
            auto it = globalMappings.find(kind);
            return (it != globalMappings.end()) ? it->second : nullptr;
        }
    };

    // =================================================================
    // ReorderPlan: computed transfers + lookups, applies via callbacks
    // =================================================================

    /// Standalone plan object containing all computed PermutationTransfers
    /// and LookupResults for a set of entity kinds.
    ///
    /// After construction (via `build`), this object has no dependency on
    /// UnstructuredMesh. It can apply to any ReorderRegistry.
    struct ReorderPlan
    {
        /// Per reordered entity kind: the computed transfer.
        std::unordered_map<EntityKind, PermutationTransfer> transfers;

        /// Per reordered entity kind: the old->new global lookup.
        std::unordered_map<EntityKind, PermutationTransfer::LookupResult> lookups;

        /// Set of entity kinds being reordered.
        std::unordered_set<EntityKind> reorderedKinds;

        /// Whether all transfers are local-only (collective agreement).
        bool isLocalOnly{false};

        // -------------------------------------------------------------
        // Factory
        // -------------------------------------------------------------

        /// Build a ReorderPlan from input + registry + MPI.
        ///
        /// Steps:
        /// 1. Compute follow maps (ghost-pull leader targetRanks, min-rank rule).
        /// 2. Build PermutationTransfer per entity kind.
        /// 3. Collect pull sets per entity kind from registry adj entries.
        /// 4. Build lookups (ghost-pullable old->new global).
        ///
        /// @warning Collective.
        static ReorderPlan build(
            const ReorderInput &input,
            const ReorderRegistry &registry,
            const MPIInfo &mpi);

        // -------------------------------------------------------------
        // Application
        // -------------------------------------------------------------

        /// Apply the plan to all entries in a registry.
        ///
        /// Phase 1: REMAP all adj entries (target kind's lookup).
        /// Phase 2: RELOCATE all adj rows (source kind's transfer).
        /// Phase 3: RELOCATE all companions of reordered kinds.
        ///
        /// @warning Collective (when !isLocalOnly).
        void apply(ReorderRegistry &registry, const MPIInfo &mpi) const;

        // -------------------------------------------------------------
        // Standalone operations for external arrays
        // -------------------------------------------------------------

        /// Remap entries of an adjacency array whose target kind is `targetKind`.
        template <class TPair>
        void remapEntries(TPair &pair, EntityKind targetKind) const
        {
            auto it = lookups.find(targetKind);
            DNDS_assert_info(it != lookups.end(),
                             "remapEntries: no lookup for target kind");
            const auto &lookup = it->second;
            for (index i = 0; i < pair.father->Size(); i++)
                for (rowsize j = 0; j < pair.RowSize(i); j++)
                {
                    index &v = pair(i, j);
                    if (v == UnInitIndex)
                        continue;
                    v = lookup.resolve(v);
                }
        }

        /// Relocate rows of an array pair whose source kind is `sourceKind`.
        template <class TPair>
        void relocateRows(TPair &pair, EntityKind sourceKind, const MPIInfo &mpi) const
        {
            auto it = transfers.find(sourceKind);
            DNDS_assert_info(it != transfers.end(),
                             "relocateRows: no transfer for source kind");
            it->second.transferRows(pair, mpi);
        }
    };

} // namespace DNDS::Geom
