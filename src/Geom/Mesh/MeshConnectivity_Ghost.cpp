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

    GhostSpec GhostSpec::defaultPrimary(int nLayers)
    {
        using namespace Adj;
        DNDS_assert(nLayers >= 1);

        // Cell chain: nLayers hops of Cell2Cell
        std::vector<AdjKind> cellHops(nLayers, Cell2Cell);

        // Node chain: nLayers hops of Cell2Cell + Cell2Node
        std::vector<AdjKind> nodeHops(nLayers, Cell2Cell);
        nodeHops.push_back(Cell2Node);

        return GhostSpec{{
            {EntityKind::Cell, cellHops, EntityKind::Cell},
            {EntityKind::Cell, nodeHops, EntityKind::Node},
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
            // Also mark intermediate hops whose entity kind matches the
            // chain target -- this enables cumulative ghost collection for
            // multi-hop same-kind chains (e.g., Cell2Cell -> Cell2Cell).
            if (hi + 1 == chain.hops.size() || childKind == chain.target)
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

        // --- Scratch state: per-AdjKind temporary pair with ghost data ---
        // Created lazily when a scratch pull is needed between levels.
        // The scratch pair shares the same father as the registered adj
        // but has its own son array and transformer.
        std::unordered_map<AdjKind, ssp<AdjVariant>, AdjKindHash> scratchAdjs;

        // Resolve adjacency, preferring scratch (ghost-populated) version.
        auto resolveAdjLive = [&](AdjKind kind) -> ssp<AdjVariant>
        {
            auto sit = scratchAdjs.find(kind);
            if (sit != scratchAdjs.end())
                return sit->second;
            return resolveAdj(kind);
        };

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

        // --- Intermediate ghost tracking (all nodes, not just COLLECT).
        // Scratch-pull decisions depend on this, not result.ghostIndices,
        // so that non-collected intermediate nodes still seed scratch pulls
        // for subsequent hops (e.g. Node in the Bnd→Bnd2Node→Node2Bnd→Bnd chain).
        std::unordered_map<EntityKind, std::vector<index>> intermediateGhosts;

        // --- Per-hop last-pulled ghost-set size, to avoid redundant pulls.
        std::unordered_map<AdjKind, size_t, AdjKindHash> lastScratchSize;

        // --- Level-by-level evaluation ---
        for (int level = 0; level <= tree.maxLevel; level++)
        {
            auto &levelEntries = tree.levels[level];

            // Step 1: Collect ghosts from ALL nodes into intermediateGhosts;
            // only COLLECT nodes feed the output result.ghostIndices.
            for (auto &entry : levelEntries)
            {
                auto gm = getGlobalMapping(entry.kind);
                DNDS_assert_info(gm,
                                 fmt::format("evaluateGhostTree: no GlobalOffsetsMapping "
                                             "for node kind {}",
                                             entityKindName(entry.kind)));
                index myOffset = gm->operator()(mpi.rank, 0);
                index myEnd = myOffset + gm->RLengths()[mpi.rank];

                auto ghosts = filterNonOwned(nodeSets[entry.nodeId], myOffset, myEnd);
                sortedMergeInto(intermediateGhosts[entry.kind], ghosts);

                if (entry.collect)
                    sortedMergeInto(result.ghostIndices[entry.kind], ghosts);
            }

            if (level >= tree.maxLevel)
                break;

            // Step 2: Scratch pull — create/expand scratch for hops where
            // intermediateGhosts[parentKind] has grown beyond the last pull.
            {
                std::vector<AdjKind> hopsNeedingScratch;
                for (auto &childEntry : tree.levels[level + 1])
                {
                    AdjKind hop = childEntry.hop;
                    EntityKind parentKind = hop.from;

                    bool alreadyHandled = false;
                    for (auto &h : hopsNeedingScratch)
                        if (h == hop)
                        {
                            alreadyHandled = true;
                            break;
                        }
                    if (alreadyHandled)
                        continue;

                    size_t curSize = 0;
                    auto git = intermediateGhosts.find(parentKind);
                    if (git != intermediateGhosts.end())
                        curSize = git->second.size();

                    bool needsPull = (curSize > 0 &&
                                      curSize > lastScratchSize[hop]);

                    int localNeedsIt = needsPull ? 1 : 0;
                    int globalNeedsIt = 0;
                    MPI_Allreduce(&localNeedsIt, &globalNeedsIt, 1, MPI_INT, MPI_MAX, mpi.comm);

                    if (globalNeedsIt)
                    {
                        hopsNeedingScratch.push_back(hop);
                        lastScratchSize[hop] = curSize;
                    }
                }

                for (auto &hop : hopsNeedingScratch)
                {
                    EntityKind parentKind = hop.from;
                    auto origAdj = resolveAdj(hop);
                    if (!origAdj)
                        continue;

                    std::vector<index> ghostSet;
                    auto git = intermediateGhosts.find(parentKind);
                    if (git != intermediateGhosts.end())
                        ghostSet = git->second;

                    std::visit([&](const auto &adj)
                               {
                        using TPair = std::decay_t<decltype(adj)>;
                        using TArray = typename TPair::t_arr;

                        auto tempSon = make_ssp<TArray>(
                            ObjName{"scratch_son"}, adj.father->getMPI());

                        auto livePair = makeAdjVariant<TPair>();
                        auto &live = std::get<TPair>(*livePair);
                        live.father = adj.father;
                        live.son = tempSon;
                        live.trans.setFatherSon(adj.father, tempSon);

                        DNDS_assert_info(adj.father->pLGlobalMapping,
                                         "scratch pull: father must have pLGlobalMapping");
                        live.trans.pLGlobalMapping = adj.father->pLGlobalMapping;

                        live.trans.createGhostMapping(ghostSet);
                        live.trans.createMPITypes();
                        live.trans.pullOnce();

                        scratchAdjs[hop] = std::move(livePair); }, *origAdj);
                }
            }

            // Step 3: Traverse children at level+1 using live adjacencies.
            auto &nextLevelEntries = tree.levels[level + 1];
            for (auto &childEntry : nextLevelEntries)
            {
                auto &parentSet = nodeSets[childEntry.parentId];

                auto adjVar = resolveAdjLive(childEntry.hop);
                DNDS_assert_info(adjVar,
                                 fmt::format("evaluateGhostTree: adjacency {} not resolved",
                                             adjKindName(childEntry.hop)));

                nodeSets[childEntry.nodeId] = traverseHop(parentSet, *adjVar, false);
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
