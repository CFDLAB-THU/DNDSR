/// @file MeshConnectivity_Ghost.cpp
/// @brief Ghost chain compilation and BFS evaluation.

#include "MeshConnectivity.hpp"

#include <algorithm>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <fmt/core.h>

namespace DNDS::Geom
{
    // =================================================================
    // GhostSpec::defaultPrimary
    // =================================================================

    GhostSpec GhostSpec::defaultPrimary()
    {
        using namespace Adj;
        return GhostSpec{{
            {EntityKind::Cell, {Cell2Cell}, EntityKind::Cell},
            {EntityKind::Cell, {Cell2Cell, Cell2Node}, EntityKind::Node},
            {EntityKind::Bnd, {Bnd2Node, Node2Bnd}, EntityKind::Bnd},
            {EntityKind::Bnd, {Bnd2Node, Node2Bnd, Bnd2Node}, EntityKind::Node},
        }};
    }

    // =================================================================
    // CompiledGhostTree::compile
    // =================================================================

    /// Insert a chain into a trie rooted at `roots`. Merges common prefixes.
    static void insertChainIntoTrie(
        std::vector<GhostTreeNode> &roots,
        const GhostChain &chain)
    {
        // Validate chain consistency.
        if (chain.hops.empty())
            throw std::runtime_error("GhostChain: empty hop list");
        if (chain.anchor != chain.hops.front().from)
            throw std::runtime_error(fmt::format(
                "GhostChain: anchor {} != first hop from {}",
                entityKindName(chain.anchor),
                entityKindName(chain.hops.front().from)));
        if (chain.target != chain.hops.back().to)
            throw std::runtime_error(fmt::format(
                "GhostChain: target {} != last hop to {}",
                entityKindName(chain.target),
                entityKindName(chain.hops.back().to)));
        for (size_t i = 0; i + 1 < chain.hops.size(); i++)
        {
            if (chain.hops[i].to != chain.hops[i + 1].from)
                throw std::runtime_error(fmt::format(
                    "GhostChain: hop[{}].to ({}) != hop[{}].from ({})",
                    i, entityKindName(chain.hops[i].to),
                    i + 1, entityKindName(chain.hops[i + 1].from)));
        }

        // Find or create the root node for this chain's anchor.
        GhostTreeNode *current = nullptr;
        for (auto &root : roots)
        {
            if (root.kind == chain.anchor)
            {
                current = &root;
                break;
            }
        }
        if (!current)
        {
            roots.push_back(GhostTreeNode{chain.anchor, {}, false, 0, {}});
            current = &roots.back();
        }

        // Walk down the trie, creating nodes as needed.
        for (size_t hi = 0; hi < chain.hops.size(); hi++)
        {
            const AdjKind &hop = chain.hops[hi];
            int childLevel = current->level + 1;
            EntityKind childKind = hop.to;

            // Look for existing child with same hop.
            GhostTreeNode *child = nullptr;
            for (auto &c : current->children)
            {
                if (c.hop == hop && c.kind == childKind)
                {
                    child = &c;
                    break;
                }
            }
            if (!child)
            {
                current->children.push_back(
                    GhostTreeNode{childKind, hop, false, childLevel, {}});
                child = &current->children.back();
            }

            // If this is the last hop, mark as collect.
            if (hi + 1 == chain.hops.size())
                child->collect = true;

            current = child;
        }
    }

    CompiledGhostTree CompiledGhostTree::compile(const GhostSpec &spec)
    {
        CompiledGhostTree tree;
        for (auto &chain : spec.chains)
            insertChainIntoTrie(tree.roots, chain);

        // Assign IDs and parent IDs, compute maxLevel, build per-level lists.
        int nextId = 0;
        tree.maxLevel = 0;

        // BFS to assign IDs and build levels.
        // Use a queue of (node pointer, parent ID).
        struct QEntry
        {
            GhostTreeNode *node;
            int parentId;
        };
        std::queue<QEntry> q;
        for (auto &root : tree.roots)
        {
            root.parentId = -1;
            q.push({&root, -1});
        }

        while (!q.empty())
        {
            auto [node, pid] = q.front();
            q.pop();

            node->id = nextId++;
            node->parentId = pid;

            if (node->level > tree.maxLevel)
                tree.maxLevel = node->level;

            for (auto &child : node->children)
                q.push({&child, node->id});
        }

        tree.totalNodes = nextId;

        // Build per-level lists.
        tree.levels.resize(tree.maxLevel + 1);
        std::function<void(const GhostTreeNode &)> buildLevels =
            [&](const GhostTreeNode &n)
        {
            tree.levels[n.level].push_back(LevelEntry{
                n.id,
                n.parentId,
                n.kind,
                n.hop,
                n.collect,
                !n.children.empty(),
            });
            for (auto &child : n.children)
                buildLevels(child);
        };
        for (auto &root : tree.roots)
            buildLevels(root);

        return tree;
    }

    // =================================================================
    // CompiledGhostTree helper methods
    // =================================================================

    static void collectRequiredAdjsHelper(
        const GhostTreeNode &node,
        std::unordered_set<AdjKind, AdjKindHash> &out)
    {
        for (auto &child : node.children)
        {
            out.insert(child.hop);
            collectRequiredAdjsHelper(child, out);
        }
    }

    std::unordered_set<AdjKind, AdjKindHash> CompiledGhostTree::requiredAdjs() const
    {
        std::unordered_set<AdjKind, AdjKindHash> result;
        for (auto &root : roots)
            collectRequiredAdjsHelper(root, result);
        return result;
    }

    std::vector<AdjKind> CompiledGhostTree::checkAvailable(
        const MeshConnectivity &dag) const
    {
        auto required = requiredAdjs();
        std::vector<AdjKind> missing;
        for (auto &adj : required)
        {
            if (!dag.hasAdj(adj))
                missing.push_back(adj);
        }
        return missing;
    }

    static void collectCollectedKindsHelper(
        const GhostTreeNode &node,
        std::unordered_set<EntityKind> &out)
    {
        if (node.collect)
            out.insert(node.kind);
        for (auto &child : node.children)
            collectCollectedKindsHelper(child, out);
    }

    std::unordered_set<EntityKind> CompiledGhostTree::collectedKinds() const
    {
        std::unordered_set<EntityKind> result;
        for (auto &root : roots)
            collectCollectedKindsHelper(root, result);
        return result;
    }

    // =================================================================
    // CompiledGhostTree::dump
    // =================================================================

    static void dumpNode(
        const GhostTreeNode &node,
        std::ostringstream &oss,
        int indent)
    {
        for (int i = 0; i < indent; i++)
            oss << "  ";
        if (indent > 0)
            oss << "\\-[" << adjKindName(node.hop) << "]-> ";
        oss << entityKindName(node.kind)
            << " (level=" << node.level;
        if (node.collect)
            oss << ", COLLECT";
        oss << ")\n";
        for (auto &child : node.children)
            dumpNode(child, oss, indent + 1);
    }

    std::string CompiledGhostTree::dump() const
    {
        std::ostringstream oss;
        for (auto &root : roots)
            dumpNode(root, oss, 0);
        return oss.str();
    }

    // =================================================================
    // GhostResult helpers
    // =================================================================

    // (hasGhosts and totalGhosts are inline in the header)

    // =================================================================
    // MeshConnectivity::evaluateGhostTree — Hybrid evaluation
    // =================================================================

    /// Sorted merge of two sorted vectors into dst. dst = union(dst, src).
    static void sortedMergeInto(std::vector<index> &dst, const std::vector<index> &src)
    {
        if (src.empty())
            return;
        if (dst.empty())
        {
            dst = src;
            return;
        }
        std::vector<index> merged;
        merged.reserve(dst.size() + src.size());
        std::set_union(dst.begin(), dst.end(), src.begin(), src.end(),
                       std::back_inserter(merged));
        dst = std::move(merged);
    }

    /// Traverse one hop: for each global index in parentSet, look up the
    /// adjacency row and collect all entries into a sorted, deduplicated vector.
    /// Templatized to work with any ArrayAdjacencyPair<rs> (variable or fixed-width).
    template <class TAdjPair>
    static std::vector<index> traverseHopImpl(
        const std::vector<index> &parentSet,
        const TAdjPair &adj,
        bool assertFound)
    {
        if (parentSet.empty())
            return {};

        auto *gm = adj.father->pLGlobalMapping.get();
        DNDS_assert(gm);

        index fatherSize = adj.father->Size();
        index sonSize = adj.son ? adj.son->Size() : 0;
        index totalSize = fatherSize + sonSize;
        index myOffset = gm->operator()(adj.father->getMPI().rank, 0);
        auto *ghostMapping = adj.trans.pLGhostMapping.get();

        // Use unordered_set for O(1) dedup, then sort at end.
        std::unordered_set<index> seen;
        seen.reserve(parentSet.size() * 2);

        for (auto parentGlobal : parentSet)
        {
            index localIdx = -1;

            if (parentGlobal >= myOffset && parentGlobal < myOffset + fatherSize)
            {
                localIdx = parentGlobal - myOffset;
            }
            else if (ghostMapping)
            {
                auto [found, rank, ghostIdx] = ghostMapping->search_indexAppend(parentGlobal);
                if (found && ghostIdx >= 0 && ghostIdx < totalSize)
                    localIdx = ghostIdx;
            }

            if (localIdx < 0)
            {
                DNDS_assert_info(!assertFound,
                                 fmt::format("traverseHop: global index {} not found locally "
                                             "(father=[{}, {}), son={}). Scratch pull incomplete?",
                                             parentGlobal, myOffset, myOffset + fatherSize, sonSize));
                continue;
            }

            for (auto entry : adj[localIdx])
                seen.insert(entry);
        }

        // Convert to sorted vector.
        std::vector<index> result(seen.begin(), seen.end());
        std::sort(result.begin(), result.end());
        return result;
    }

    /// Dispatch traverseHop through AdjVariant via std::visit.
    static std::vector<index> traverseHop(
        const std::vector<index> &parentSet,
        const AdjVariant &adjVar,
        bool assertFound)
    {
        return std::visit([&](const auto &adj) -> std::vector<index>
                          { return traverseHopImpl(parentSet, adj, assertFound); },
                          adjVar);
    }

    /// Filter non-owned indices from a sorted set.
    static std::vector<index> filterNonOwned(
        const std::vector<index> &set,
        index ownedStart, index ownedEnd)
    {
        std::vector<index> ghosts;
        for (auto gi : set)
        {
            if (gi < ownedStart || gi >= ownedEnd)
                ghosts.push_back(gi);
        }
        return ghosts;
    }

    GhostResult MeshConnectivity::evaluateGhostTree(
        const CompiledGhostTree &tree,
        const MPIInfo &mpi) const
    {
        // --- Pre-check all required adjacencies ---
        auto missing = tree.checkAvailable(*this);
        if (!missing.empty())
        {
            std::string names;
            for (auto &m : missing)
                names += " " + adjKindName(m);
            DNDS_assert_info(false,
                             fmt::format("evaluateGhostTree: missing adjacencies:{}", names));
        }

        // --- Per-node sets: flat vector indexed by node ID ---
        std::vector<std::vector<index>> nodeSets(tree.totalNodes);

        // --- Initialize roots (level 0) ---
        for (auto &entry : tree.levels[0])
        {
            auto gm = getGlobalMapping(entry.kind);
            DNDS_assert_info(gm,
                             fmt::format("evaluateGhostTree: no GlobalOffsetsMapping for root kind {}",
                                         entityKindName(entry.kind)));
            index myOffset = gm->operator()(mpi.rank, 0);
            index mySize = gm->RLengths()[mpi.rank];
            auto &set = nodeSets[entry.nodeId];
            set.resize(mySize);
            for (index i = 0; i < mySize; i++)
                set[i] = myOffset + i;
        }

        // --- Collect ghost results ---
        GhostResult result;

        // --- Level-by-level evaluation ---
        for (int level = 0; level <= tree.maxLevel; level++)
        {
            auto &levelEntries = tree.levels[level];

            // Step 1: Collect from COLLECT nodes at this level.
            for (auto &entry : levelEntries)
            {
                if (!entry.collect)
                    continue;
                auto gm = getGlobalMapping(entry.kind);
                DNDS_assert_info(gm,
                                 fmt::format("evaluateGhostTree: no GlobalOffsetsMapping "
                                             "for COLLECT kind {}",
                                             entityKindName(entry.kind)));
                index myOffset = gm->operator()(mpi.rank, 0);
                index myEnd = myOffset + gm->RLengths()[mpi.rank];

                auto ghosts = filterNonOwned(nodeSets[entry.nodeId], myOffset, myEnd);
                sortedMergeInto(result.ghostIndices[entry.kind], ghosts);
            }

            // Step 2 + 4: Traverse children.
            // (Step 3 — scratch pull — is the caller's responsibility for now.
            //  The evaluator asserts that all entities are locally resolvable
            //  when ghost data should be available, i.e., after level 0.)
            if (level < tree.maxLevel)
            {
                auto &nextLevelEntries = tree.levels[level + 1];
                for (auto &childEntry : nextLevelEntries)
                {
                    // childEntry.parentId points to the parent at this level.
                    auto &parentSet = nodeSets[childEntry.parentId];

                    auto adjVar = resolveAdj(childEntry.hop);
                    DNDS_assert_info(adjVar,
                                     fmt::format("evaluateGhostTree: adjacency {} not resolved",
                                                 adjKindName(childEntry.hop)));

                    // Assert ghost-not-found only after level 0 (level 0 is
                    // owned entities, always local; later levels may have
                    // ghost entities that need prior scratch pulls).
                    bool assertFound = (level > 0);

                    nodeSets[childEntry.nodeId] = traverseHop(parentSet, *adjVar, assertFound);
                }
            }
        }

        // --- Deduplicate ghost indices (already sorted from sortedMergeInto) ---
        // Remove empty entries.
        for (auto it = result.ghostIndices.begin(); it != result.ghostIndices.end();)
        {
            if (it->second.empty())
                it = result.ghostIndices.erase(it);
            else
                ++it;
        }

        // --- Collective: determine which EntityKinds have ghosts on ANY rank ---
        int localMask = 0;
        for (auto &[kind, indices] : result.ghostIndices)
            localMask |= (1 << static_cast<int>(kind));

        int globalMask = 0;
        MPI_Allreduce(&localMask, &globalMask, 1, MPI_INT, MPI_BOR, mpi.comm);

        for (int i = 0; i < static_cast<int>(EntityKind::NUM_KINDS); i++)
        {
            if (globalMask & (1 << i))
                result.activeKinds.insert(static_cast<EntityKind>(i));
        }

        return result;
    }

} // namespace DNDS::Geom
