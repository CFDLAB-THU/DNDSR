#pragma once

#include "Geom/Connectivity/MeshRelationTypes.hpp"
#include "Geom/Unstructured/MeshTypes.hpp"
#include "Geom/Periodic/PeriodicTypes.hpp"
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace DNDS::Geom
{

    // =================================================================
    // GhostChain, GhostSpec: user-facing ghost specification
    // =================================================================

    /// One ghost chain: starts from owned entities of `anchor`, traverses
    /// explicit adjacency hops, collects ghost entities of `target`.
    ///
    /// Validation rules (checked by CompiledGhostTree::compile):
    ///   - anchor == hops[0].from
    ///   - target == hops.back().to
    ///   - consecutive hops: hops[i].to == hops[i+1].from
    ///   - at least one hop
    struct GhostChain
    {
        EntityKind anchor;                     ///< Owned entities to start from.
        std::vector<AdjacencyDescriptor> hops; ///< Sequence of adjacency lookups.
        EntityKind target;                     ///< Entity kind to ghost (must == hops.back().to).
    };

    /// Full ghost specification: multiple chains, possibly targeting the same
    /// EntityKind. The ghost set per kind is the union of all chains' results.
    struct GhostSpec
    {
        std::vector<GhostChain> chains;

        /// The default pipeline specification for 1 ghost layer.
        /// Cell ghost: owned cells -> Cell2Cell -> cells
        /// Node ghost: owned cells -> Cell2Cell -> Cell2Node -> nodes
        /// Boundary-node ghost: owned bnds -> Bnd2Node -> nodes
        /// Bnd ghost:  owned bnds  -> Bnd2Node -> Node2Bnd -> bnds
        /// Bnd-node ghost: owned bnds -> Bnd2Node -> Node2Bnd -> Bnd2Node -> nodes
        static GhostSpec defaultPrimary() { return defaultPrimary(1); }

        /// Parameterized default: explicit cumulative 1..nLayers Cell2Cell
        /// endpoint chains for cells and their nodes.
        static GhostSpec defaultPrimary(int nLayers);
    };

    // =================================================================
    // CompiledGhostTree: chains compiled into BFS-ordered forest
    // =================================================================

    /// One node in the compiled ghost tree.
    struct GhostTreeNode
    {
        EntityKind kind;         ///< Entity kind at this node.
        AdjacencyDescriptor hop; ///< Adjacency used to reach this node from parent.
                                 ///< Undefined (default-constructed) for root nodes.
        bool collect{false};     ///< True when at least one input chain ends here; non-owned entities become ghosts.
        int level{0};            ///< BFS depth (root = 0).
        int id{-1};              ///< Unique ID within the tree (assigned by compile).
        int parentId{-1};        ///< Parent node ID (-1 for roots).
        std::vector<GhostTreeNode> children;
    };

    /// A reference to a tree node at a specific level, with its parent.
    /// Used by the precomputed per-level lists.
    struct LevelEntry
    {
        int nodeId{};            ///< ID of the tree node.
        int parentId{};          ///< ID of the parent node (-1 for roots).
        EntityKind kind{};       ///< Entity kind of this node.
        AdjacencyDescriptor hop; ///< Hop used to reach this node.
        bool collect{};          ///< Whether to collect at this node.
        bool hasChildren{};      ///< Whether this node has children (needs pull).
    };

    /// Compiled forest of ghost traversal chains.
    ///
    /// Multiple chains sharing common prefixes are merged into a trie to
    /// avoid redundant traversals. The tree is evaluated BFS level-by-level
    /// with pull barriers between levels.
    ///
    /// Compilation publishes one immutable program, shared by copies. Roots
    /// are diagnostic; Levels() is the execution schedule and Signature() is
    /// derived from that exact schedule. Change GhostSpec and recompile instead
    /// of editing a compiled node. Inspection references borrow this program.
    class CompiledGhostTree
    {
        struct Data
        {
            std::vector<GhostTreeNode> roots;
            int maxLevel{0}, totalNodes{0};
            std::vector<std::vector<LevelEntry>> levels;
            std::string signature;
        };
        std::shared_ptr<const Data> _data;
        explicit CompiledGhostTree(Data data) : _data(std::make_shared<const Data>(std::move(data))) {}

    public:
        // Copy-only handle semantics also keep an rvalue source usable; there
        // is no publicly constructible partial or null compiled program.
        CompiledGhostTree(const CompiledGhostTree &) = default;
        CompiledGhostTree &operator=(const CompiledGhostTree &) = default;
        const std::vector<GhostTreeNode> &Roots() const { return _data->roots; }
        const std::vector<std::vector<LevelEntry>> &Levels() const { return _data->levels; }
        int MaxLevel() const { return _data->maxLevel; }
        int NodeCount() const { return _data->totalNodes; }
        const std::string &Signature() const { return _data->signature; }

        /// Compile a GhostSpec into a forest. Validates chain consistency.
        /// Assigns node IDs, builds per-level lists.
        /// Purely local metadata work: does not read adjacency or perform MPI.
        /// Prefixes merge only within the same anchor and descriptor sequence;
        /// equal entity kinds at different path positions do not merge.
        /// Throws std::runtime_error on invalid chains.
        static CompiledGhostTree compile(const GhostSpec &spec);

        /// Collect all distinct AdjacencyDescriptor values used by any hop in the tree.
        [[nodiscard]] std::unordered_set<AdjacencyDescriptor, AdjacencyDescriptorHash> requiredAdjs() const;

        // AdjacencyRegistry::Evaluate validates required definitions collectively.

        /// Collect all EntityKind values that appear at COLLECT nodes.
        [[nodiscard]] std::unordered_set<EntityKind> collectedKinds() const;

        /// Pretty-print the tree (for diagnostics).
        [[nodiscard]] std::string dump() const;
    };

    // =================================================================
    // GhostResult: output of ghost evaluation
    // =================================================================

    /// Result of evaluating a CompiledGhostTree.
    /// Contains per-EntityKind sorted, deduplicated global indices to ghost.
    struct GhostResult
    {
        /// Per EntityKind: sorted, deduplicated global indices to ghost.
        std::unordered_map<EntityKind, std::vector<index>> ghostIndices;

        /// Entity kinds that have ghosts on ANY rank (collective).
        /// Populated by evaluateGhostTree via MPI_Allreduce.
        std::unordered_set<EntityKind> activeKinds;

        /// Whether any rank has ghosts for a given kind (collective).
        /// Safe to branch on — consistent across all ranks.
        [[nodiscard]] bool hasGhosts(EntityKind kind) const
        {
            return activeKinds.count(kind) > 0;
        }

        /// Total number of ghost entities on THIS rank.
        [[nodiscard]] index totalGhosts() const
        {
            index total = 0;
            for (const auto &[k, v] : ghostIndices)
                total += static_cast<index>(v.size());
            return total;
        }
    };

    // =================================================================
    // SharedCountPredicate
    // =================================================================

    /// Predicate for ComposeFiltered: keep (A, C) pairs sharing >= minShared
    /// intermediate B-entities.
    struct SharedCountPredicate
    {
        int minShared{1};
        bool removeSelf{false};

        bool operator()(index a, index c, int nShared) const
        {
            if (removeSelf && a == c)
                return false;
            return nShared >= minShared;
        }
    };

    // =================================================================
    // SubEntityDef: user-provided sub-entity topology query
    // =================================================================

    /// Describes one sub-entity extracted from a parent entity.
    /// Returned by SubEntityQuery::describe().
    struct SubEntityDesc
    {
        int nVertices{0};   ///< Number of corner vertices (used for deduplication).
        int nNodes{0};      ///< Total number of nodes (vertices + mid-edge + ...).
        t_index typeTag{0}; ///< Element type tag to store in entityElemInfo.
                            ///< Opaque to Interpolate; only used for type-match
                            ///< during deduplication (two sub-entities with different
                            ///< typeTags are never considered duplicates).
    };

    /// User-provided callbacks that describe how to decompose parent entities
    /// into sub-entities. This decouples Interpolate from any specific element
    /// topology module.
    ///
    /// Example: to extract faces from cells, the caller would implement:
    ///   - numSubEntities(iParent): returns eCell.GetNumFaces()
    ///   - describe(iParent, iSub):  returns {nVerts, nNodes, faceElemType}
    ///   - extractNodes(iParent, iSub, parentNodes, out): calls ExtractFaceNodes
    ///   - matchExtra (optional): periodic collaborating check
    /// All callbacks must be deterministic and safe for concurrent const use.
    /// Scalable setup may evaluate `describe`/`extractNodes` more than once;
    /// callbacks must not retain scratch or expose observable side effects.
    struct SubEntityQuery
    {
        /// Number of sub-entities for parent iParent.
        std::function<int(index iParent)> numSubEntities;

        /// Describe sub-entity iSub of parent iParent.
        std::function<SubEntityDesc(index iParent, int iSub)> describe;

        /// Extract node indices of sub-entity iSub from parentNodes into out.
        /// Must write exactly desc.nNodes entries starting at out[0].
        /// parentNodes is an indexable range (operator[] returning index).
        std::function<void(index iParent, int iSub,
                           const std::function<index(int)> &parentNodes,
                           index *out)>
            extractNodes;

        /// Optional extra match predicate for deduplication.
        ///
        /// Called after vertex-set match succeeds. If set, must return true
        /// for the match to be accepted. Used for periodic meshes to implement
        /// the "collaborating" check (uniform XOR of periodic bits).
        ///
        /// @param iParent       Current parent entity index.
        /// @param iSub          Current sub-entity index within parent.
        /// @param iCandEntity   Index of the candidate entity in the result arrays.
        /// @param candidateParent  Index of the parent that created the candidate entity.
        /// @param candidateSub    Sub-entity index that created the candidate entity.
        /// @return true if the match is valid, false to reject.
        /// Must be deterministic and free of observable side effects; the
        /// number/order of rejected-candidate probes is implementation detail.
        std::function<bool(index iParent, int iSub,
                           index iCandEntity,
                           index candidateParent, int candidateSub)>
            matchExtra;

        /// Enables parallel equal-key group resolution when `matchExtra` is an
        /// equivalence predicate and does not inspect `iCandEntity`. Creating
        /// parent/subentity pairs and probe order within a key remain stable.
        /// Custom callbacks default to the append implementation unless they
        /// explicitly satisfy both requirements and opt in.
        bool matchExtraSupportsParallelGrouping{false};
    };

    // =================================================================
    // InterpolateGlobal: distributed sub-entity creation with global dedup
    // =================================================================

    /// Extended sub-entity query with periodic bit extraction.
    ///
    /// Inherits all callbacks from SubEntityQuery and adds extractPbi for
    /// periodic meshes. Used by InterpolateGlobal.
    ///
    /// extractPbi must produce node-parallel pbi matching extractNodes output:
    /// extractPbi(iParent, iSub, ...)[k] is the pbi for the node at position k
    /// in the sub-entity's node list (same ordering as extractNodes).
    struct SubEntityQueryPbi : SubEntityQuery
    {
        /// Extract pbi of sub-entity iSub from parentPbi into out.
        /// Must write exactly desc.nNodes entries starting at out[0].
        /// Ordering must match extractNodes (out[k] corresponds to node k).
        /// Only called when the mesh is periodic.
        std::function<void(index iParent, int iSub,
                           const std::function<NodePeriodicBits(int)> &parentPbi,
                           NodePeriodicBits *out)>
            extractPbi;
    };

    /// Ownership decision for a globally-deduplicated B entity.
    /// Returned by the OwnershipResolverMulti callback.
    struct OwnershipDecision
    {
        bool owned; ///< true if this rank owns the entity.
        /// Peer ranks that need this entity pushed (for owned entities).
        /// Empty if fully local or single-sided.
        /// For non-owned entities, this is unused.
        std::vector<MPI_int> peerRanks;
    };

    /// Callback for ownership resolution during distributed interpolation.
    ///
    /// Called for each locally-enumerated entity with its parent information.
    /// Faces have exactly 2 parents; edges can have N >= 2.
    ///
    /// @param parents       All parent indices (local-appended) of this entity.
    ///                      [0, nLocalParents) = owned, [nLocal, nTotal) = ghost.
    /// @param nLocalParents Number of owned (father) parents on this rank.
    /// @param parentRanks   Rank owning each parent (parallel to parents vector).
    /// @return OwnershipDecision: whether this rank owns the entity and which
    ///         peers need it pushed.
    /// The callback is setup-pure and must be safe for concurrent const calls;
    /// it must not invoke MPI or retain references to the temporary vectors.
    using OwnershipResolverMulti = std::function<OwnershipDecision(
        const std::vector<index> &parents,
        const std::vector<MPI_int> &parentRanks,
        index nLocalParents)>;

    /// Legacy 2-parent ownership resolver (used by InterpolateDistributed).
    using OwnershipResolver2 = std::function<OwnershipDecision(
        index parentL, index parentR, index nLocalParents)>;
}
