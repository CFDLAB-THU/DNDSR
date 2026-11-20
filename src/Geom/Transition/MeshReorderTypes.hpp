#pragma once

#include "Geom/Access/EntitySpace.hpp"
#include "DNDS/MPI/MPI.hpp"
#include <unordered_set>
#include <vector>
#include <map>

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
        EntityKind follower{};               ///< Entity kind to derive map for.
        EntityKind leader{};                 ///< Explicit-map entity kind to follow.
        AdjacencyDescriptor follower2leader; ///< Support adj: follower -> leader.
    };

    // =================================================================
    // ReorderInput: what the caller provides
    // =================================================================

    /// Local provenance of a permutation derived by traversing a published mesh.
    /// The source order includes both the traversal anchor and the permuted families.
    struct ReorderSourceOrder
    {
        Runtime::VersionStamp generation;
        std::vector<EntitySpace> families;

        template <class Lookup>
        void Validate(Lookup &&lookup) const
        {
            generation.Validate();
            for (const auto &family : families)
                DNDS_check_throw_info(family.SameOrder(lookup(family.Kind())),
                                      "local permutation was planned for another mesh row ordering");
        }
    };

    /// Input to the reorder framework.
    struct ReorderInput
    {
        /// Explicit reorder maps (caller-provided).
        std::vector<EntityReorderMap> explicitMaps;

        /// Follow specifications (framework computes follow maps from these).
        ///
        /// When empty (default), the mesh wrapper applies the default policy:
        /// all non-explicit entity kinds with a support adjacency to an
        /// explicit kind follow it (currently: Node and Bnd follow Cell).
        /// Native periodic Bnd defaults use the exact physical-side relation;
        /// an explicit Bnd2Cell follow instead takes min over both side slots.
        ///
        /// When non-empty, only the listed follows are computed (overrides
        /// the default policy — allows selective follow or custom adj routes).
        std::vector<FollowSpec> follows;

        /// Entity kinds whose adjacencies should be destroyed before reorder
        /// (not reordered, not remapped -- just wiped). Typically {Face}.
        std::unordered_set<EntityKind> destroyKinds;

        /// Optional local old-to-new row permutations, with ownership fixed.
        /// Mutually exclusive with rank-placement/follow/reconstruction inputs.
        /// Every rank names the same kinds; local row extents can differ. Native
        /// transitions remap all registered references and companions normally.
        std::map<EntityKind, std::vector<index>> localPermutations;

        /// Set by ordering plans. Explicit user-supplied maps may omit provenance.
        std::optional<ReorderSourceOrder> sourceOrder;
    };

}
