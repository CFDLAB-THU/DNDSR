#pragma once
/// @file MeshConnectivity.hpp
/// @brief Layered DAG of mesh adjacency relations with composable DSL operations.
///
/// MeshConnectivity manages adjacency (connectivity) tables between entity strata
/// of different topological depths (cells, faces, edges, nodes). It provides three
/// core operations:
///   - Inverse:         cone (A→B) → support (B→A)
///   - Compose:         A→B + B→C → A→C
///   - ComposeFiltered: A→B + B→C → A→C with on-the-fly predicate filtering
///
/// Cone adjacencies (downward: higher→lower depth) are ordered by element topology.
/// Support adjacencies (upward: lower→higher depth) are ordered by creation method
/// (typically the order entities were discovered during inversion).
///
/// Periodic bits (pbi) are only stored on cones whose target depth is 0 (nodes),
/// since pbi tracks how each node's coordinates transform under periodicity.

#include "DNDS/ArrayPair.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "Mesh_DeviceView.hpp"

#include <variant>
#include <unordered_map>
#include <unordered_set>

namespace DNDS::Geom
{
    // Forward declaration for registerAdj overload
    template <class TPair>
    struct AdjPairTracked;

    // =================================================================
    // EntityKind: logical entity roles
    // =================================================================

    /// Logical entity roles in the mesh. Topological depth depends on the
    /// mesh dimension (2D or 3D).
    ///
    /// In 3D: Cell=3, Face=2, Edge=1, Node=0.
    /// In 2D: Cell=2, Face=1, Edge=1 (==Face), Node=0.
    /// Bnd shares Face's depth but is stored separately (zone-labeled subset).
    enum class EntityKind : int8_t
    {
        Cell = 0,
        Face = 1,
        Edge = 2,
        Node = 3,
        Bnd = 4,
        NUM_KINDS = 5,
    };

    /// Resolve EntityKind to topological depth for a given mesh dimension.
    /// In 2D, Edge and Face both resolve to dim-1 = 1.
    /// Bnd resolves to dim-1 (same depth as Face).
    inline int entityDepth(EntityKind kind, int dim)
    {
        switch (kind)
        {
        case EntityKind::Cell:
            return dim;
        case EntityKind::Face:
            return dim - 1;
        case EntityKind::Edge:
            // NOLINTNEXTLINE(bugprone-branch-clone) — Edge depth is always 1 in both 2D and 3D
            return (dim == 2) ? 1 : 1;
        case EntityKind::Node:
            return 0;
        case EntityKind::Bnd:
            return dim - 1; // same depth as Face
        default:
            return -1;
        }
    }

    /// String name for an EntityKind (for diagnostics).
    inline const char *entityKindName(EntityKind kind)
    {
        switch (kind)
        {
        case EntityKind::Cell:
            return "Cell";
        case EntityKind::Face:
            return "Face";
        case EntityKind::Edge:
            return "Edge";
        case EntityKind::Node:
            return "Node";
        case EntityKind::Bnd:
            return "Bnd";
        default:
            return "Unknown";
        }
    }

    // =================================================================
    // AdjKind: named adjacency hop
    // =================================================================

    /// Identifies a specific adjacency relation in the DAG.
    ///
    /// Direct adjacencies (from != to): cone or support between two entity strata.
    ///   e.g., AdjKind(Cell, Node) = cell2node cone.
    ///   e.g., AdjKind(Node, Cell) = node2cell support.
    ///
    /// Intra-level adjacencies (from == to): composed adjacency traversing through
    /// a lower-level intermediary. Default intermediary is Node.
    ///   e.g., AdjKind(Cell, Cell)       = cell2cell via Node (node-neighbor).
    ///   e.g., AdjKind(Cell, Cell, Face) = cell2cell via Face (face-neighbor).
    ///   e.g., AdjKind(Bnd, Bnd)         = bnd2bnd via Node.
    struct AdjKind
    {
        EntityKind from{EntityKind::Cell};
        EntityKind to{EntityKind::Node};
        EntityKind via{EntityKind::Node}; ///< Intermediary for intra-level (from==to).
                                          ///< Ignored for direct (from!=to).

        constexpr AdjKind() = default;

        /// Direct adjacency: from != to. `via` is ignored.
        constexpr AdjKind(EntityKind from_, EntityKind to_)
            : from(from_), to(to_)
        {
        }

        /// Intra-level adjacency: from == to, with explicit intermediary.
        constexpr AdjKind(EntityKind from_, EntityKind to_, EntityKind via_)
            : from(from_), to(to_), via(via_)
        {
        }

        /// Whether this is an intra-level (composed) adjacency.
        [[nodiscard]] constexpr bool isIntraLevel() const { return from == to; }

        /// Whether this is a direct (inter-level) adjacency.
        [[nodiscard]] constexpr bool isDirect() const { return from != to; }

        /// Equality comparison (for use in hash maps).
        constexpr bool operator==(const AdjKind &o) const
        {
            if (from != o.from || to != o.to)
                return false;
            if (isIntraLevel())
                return via == o.via;
            return true; // direct: via is ignored
        }

        constexpr bool operator!=(const AdjKind &o) const { return !(*this == o); }
    };

    /// Hash for AdjKind (for use in unordered containers).
    struct AdjKindHash
    {
        std::size_t operator()(const AdjKind &k) const noexcept
        {
            // Combine from, to, and (if intra-level) via into a single hash.
            auto h = static_cast<std::size_t>(k.from) * 31 +
                     static_cast<std::size_t>(k.to);
            if (k.isIntraLevel())
                h = h * 31 + static_cast<std::size_t>(k.via);
            return h;
        }
    };

    /// Convenience constants for common adjacency kinds.
    namespace Adj
    {
        // Direct cones (downward)
        inline constexpr AdjKind Cell2Node{EntityKind::Cell, EntityKind::Node};
        inline constexpr AdjKind Cell2Face{EntityKind::Cell, EntityKind::Face};
        inline constexpr AdjKind Cell2Edge{EntityKind::Cell, EntityKind::Edge};
        inline constexpr AdjKind Face2Node{EntityKind::Face, EntityKind::Node};
        inline constexpr AdjKind Face2Edge{EntityKind::Face, EntityKind::Edge};
        inline constexpr AdjKind Edge2Node{EntityKind::Edge, EntityKind::Node};
        inline constexpr AdjKind Bnd2Node{EntityKind::Bnd, EntityKind::Node};

        // Direct supports (upward)
        inline constexpr AdjKind Node2Cell{EntityKind::Node, EntityKind::Cell};
        inline constexpr AdjKind Node2Face{EntityKind::Node, EntityKind::Face};
        inline constexpr AdjKind Node2Edge{EntityKind::Node, EntityKind::Edge};
        inline constexpr AdjKind Node2Bnd{EntityKind::Node, EntityKind::Bnd};
        inline constexpr AdjKind Face2Cell{EntityKind::Face, EntityKind::Cell};
        inline constexpr AdjKind Edge2Face{EntityKind::Edge, EntityKind::Face};
        inline constexpr AdjKind Edge2Cell{EntityKind::Edge, EntityKind::Cell};
        inline constexpr AdjKind Bnd2Cell{EntityKind::Bnd, EntityKind::Cell};
        inline constexpr AdjKind Bnd2Face{EntityKind::Bnd, EntityKind::Face};
        inline constexpr AdjKind Face2Bnd{EntityKind::Face, EntityKind::Bnd};

        // Intra-level (composed), default via Node
        inline constexpr AdjKind Cell2Cell{EntityKind::Cell, EntityKind::Cell, EntityKind::Node};
        inline constexpr AdjKind Bnd2Bnd{EntityKind::Bnd, EntityKind::Bnd, EntityKind::Node};
        inline constexpr AdjKind Face2Face{EntityKind::Face, EntityKind::Face, EntityKind::Node};

        // Intra-level with explicit intermediary
        inline constexpr AdjKind Cell2CellFace{EntityKind::Cell, EntityKind::Cell, EntityKind::Face};
    } // namespace Adj

    /// Format an AdjKind as a diagnostic string, e.g. "Cell2Node", "Cell2Cell(Node)".
    std::string adjKindName(const AdjKind &kind);

    // Forward declaration for CompiledGhostTree::checkAvailable.
    struct MeshConnectivity;

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
        EntityKind anchor;         ///< Owned entities to start from.
        std::vector<AdjKind> hops; ///< Sequence of adjacency lookups.
        EntityKind target;         ///< Entity kind to ghost (must == hops.back().to).
    };

    /// Full ghost specification: multiple chains, possibly targeting the same
    /// EntityKind. The ghost set per kind is the union of all chains' results.
    struct GhostSpec
    {
        std::vector<GhostChain> chains;

        /// The default pipeline specification for 1 ghost layer.
        /// Cell ghost: owned cells -> Cell2Cell -> cells
        /// Node ghost: owned cells -> Cell2Cell -> Cell2Node -> nodes
        /// Bnd ghost:  owned bnds  -> Bnd2Node -> Node2Bnd -> bnds
        /// Bnd-node ghost: owned bnds -> Bnd2Node -> Node2Bnd -> Bnd2Node -> nodes
        static GhostSpec defaultPrimary() { return defaultPrimary(1); }

        /// Parameterized default: nLayers hops of Cell2Cell for ghost cells.
        static GhostSpec defaultPrimary(int nLayers);
    };

    // =================================================================
    // CompiledGhostTree: chains compiled into BFS-ordered forest
    // =================================================================

    /// One node in the compiled ghost tree.
    struct GhostTreeNode
    {
        EntityKind kind;     ///< Entity kind at this node.
        AdjKind hop;         ///< Adjacency used to reach this node from parent.
                             ///< Undefined (default-constructed) for root nodes.
        bool collect{false}; ///< If true, non-owned entities here become ghosts.
        int level{0};        ///< BFS depth (root = 0).
        int id{-1};          ///< Unique ID within the tree (assigned by compile).
        int parentId{-1};    ///< Parent node ID (-1 for roots).
        std::vector<GhostTreeNode> children;
    };

    /// A reference to a tree node at a specific level, with its parent.
    /// Used by the precomputed per-level lists.
    struct LevelEntry
    {
        int nodeId{};       ///< ID of the tree node.
        int parentId{};     ///< ID of the parent node (-1 for roots).
        EntityKind kind{};  ///< Entity kind of this node.
        AdjKind hop;        ///< Hop used to reach this node.
        bool collect{};     ///< Whether to collect at this node.
        bool hasChildren{}; ///< Whether this node has children (needs pull).
    };

    /// Compiled forest of ghost traversal chains.
    ///
    /// Multiple chains sharing common prefixes are merged into a trie to
    /// avoid redundant traversals. The tree is evaluated BFS level-by-level
    /// with pull barriers between levels.
    ///
    /// After compilation, `levels[L]` contains all tree nodes at BFS depth L,
    /// with parent references for efficient evaluation (no recursive scans).
    struct CompiledGhostTree
    {
        std::vector<GhostTreeNode> roots;
        int maxLevel{0};   ///< Maximum BFS depth across all nodes.
        int totalNodes{0}; ///< Total number of nodes (for flat array sizing).

        /// Precomputed per-level node lists. `levels[L]` contains all tree
        /// nodes at BFS depth L. Level 0 = roots.
        std::vector<std::vector<LevelEntry>> levels;

        /// Compile a GhostSpec into a forest. Validates chain consistency.
        /// Assigns node IDs, builds per-level lists.
        /// Throws std::runtime_error on invalid chains.
        static CompiledGhostTree compile(const GhostSpec &spec);

        /// Collect all distinct AdjKind values used by any hop in the tree.
        [[nodiscard]] std::unordered_set<AdjKind, AdjKindHash> requiredAdjs() const;

        /// Pre-check that all required adjacencies exist in the DAG.
        /// Returns the set of missing AdjKind values (empty = all available).
        [[nodiscard]] std::vector<AdjKind> checkAvailable(const MeshConnectivity &dag) const;

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
    // Type-erased adjacency storage
    // =================================================================

    /// Variant of all supported fixed-width adjacency pair types.
    /// tAdjPair (NonUniformSize) is the general variable-width CSR;
    /// tAdj1Pair..tAdj8Pair are compile-time fixed-width for performance
    /// on hot-path adjacencies (face2cell = Adj2, bnd2face = Adj1, etc.).
    using AdjVariant = std::variant<
        tAdjPair,   // variable-width (cell2node, cell2face, node2cell, cell2cell, ...)
        tAdj1Pair,  // 1 per row (bnd2face, face2bnd, cell2cellOrig, ...)
        tAdj2Pair,  // 2 per row (face2cell, bnd2cell)
        tAdj3Pair,  // 3 per row
        tAdj4Pair,  // 4 per row
        tAdj8Pair>; // 8 per row

    /// Convenience: create a shared AdjVariant from a specific pair type.
    /// Usage: `auto adj = makeAdjVariant(myTAdj2Pair);`
    /// or:    `auto adj = makeAdjVariant<tAdj2Pair>();` (empty)
    template <class TPair>
    static ssp<AdjVariant> makeAdjVariant(TPair &&pair)
    {
        return make_ssp<AdjVariant>(std::forward<TPair>(pair));
    }

    /// Create a shared AdjVariant holding a default-constructed TPair.
    template <class TPair>
    static ssp<AdjVariant> makeAdjVariant()
    {
        return make_ssp<AdjVariant>(TPair{});
    }

    // =================================================================
    // narrowAdjToFixed: variable-width → fixed-width conversion
    // =================================================================

    /// Copy a variable-width adjacency (father-only) into a fixed-width
    /// target. Each row is truncated or padded with `fill` to exactly
    /// `target_rs` entries.
    ///
    /// @tparam target_rs  Compile-time row width of the target.
    /// @tparam source_rs  Row width of the source (typically NonUniformSize).
    /// @param source      Source adjacency (father-only or father+son).
    /// @param target      Target fixed-width adjacency. Father must be pre-Resize'd
    ///                    to at least `nRows`.
    /// @param nRows       Number of rows to copy.
    /// @param fill        Value for missing entries (default: UnInitIndex).
    template <rowsize target_rs, rowsize source_rs = NonUniformSize>
    static void narrowAdjToFixed(
        const ArrayAdjacencyPair<source_rs> &source,
        ArrayAdjacencyPair<target_rs> &target,
        index nRows,
        index fill = UnInitIndex)
    {
        static_assert(target_rs > 0, "narrowAdjToFixed: target must be fixed-width");
        for (index i = 0; i < nRows; i++)
        {
            auto row = source.father->operator[](i);
            for (rowsize j = 0; j < target_rs; j++)
                target.father->operator()(i, j) =
                    (j < static_cast<rowsize>(row.size())) ? row[j] : fill;
        }
    }

    // =================================================================
    // ConeAdj: downward adjacency (higher depth → lower depth)
    // =================================================================

    /// A cone (downward) adjacency from entities at `fromDepth` to entities at
    /// `toDepth`. Row order is defined by element topology and must not be
    /// permuted (e.g., cell2node ordering determines element shape functions).
    ///
    /// Periodic bits (`pbi`) are only meaningful when `toDepth == 0` (nodes).
    /// For non-periodic meshes or non-node targets, `pbi` remains uninitialized.
    ///
    /// The adjacency data is stored via `ssp<AdjVariant>` for shared ownership.
    /// Both the DAG and legacy mesh members can share the same allocation.
    struct ConeAdj
    {
        int fromDepth{-1};   ///< Source stratum (e.g., dim for cells, dim-1 for faces)
        int toDepth{-1};     ///< Target stratum (e.g., 0 for nodes, dim-1 for faces)
        ssp<AdjVariant> adj; ///< Shared adjacency pair (typed by row width)
        tPbiPair pbi;        ///< Periodic bits per entry (only for toDepth==0, optional)

        /// Access as variable-width tAdjPair. Throws if adj holds a fixed-width type.
        tAdjPair &asAdj() { return std::get<tAdjPair>(*adj); }
        [[nodiscard]] const tAdjPair &asAdj() const { return std::get<tAdjPair>(*adj); }

        /// Access as fixed-width tAdj2Pair (e.g., for cell2face with 2 entries).
        tAdj2Pair &asAdj2() { return std::get<tAdj2Pair>(*adj); }
        [[nodiscard]] const tAdj2Pair &asAdj2() const { return std::get<tAdj2Pair>(*adj); }

        tAdj1Pair &asAdj1() { return std::get<tAdj1Pair>(*adj); }
        [[nodiscard]] const tAdj1Pair &asAdj1() const { return std::get<tAdj1Pair>(*adj); }

        /// Typed access: `as<tAdj2Pair>()` etc.
        template <class TPair>
        TPair &as() { return std::get<TPair>(*adj); }
        template <class TPair>
        [[nodiscard]] const TPair &as() const { return std::get<TPair>(*adj); }

        /// Number of local (father) rows.
        [[nodiscard]] index fatherSize() const
        {
            if (!adj)
                return 0;
            return std::visit([](const auto &p) -> index
                              { return p.father ? p.father->Size() : 0; }, *adj);
        }

        /// Check if the adjacency pair is initialized (father is non-null).
        [[nodiscard]] bool initialized() const
        {
            if (!adj)
                return false;
            return std::visit([](const auto &p) -> bool
                              { return bool(p.father); }, *adj);
        }

        /// Check if pbi is attached (only valid for toDepth==0).
        [[nodiscard]] bool hasPbi() const { return bool(pbi.father); }
    };

    // =================================================================
    // SupportAdj: upward adjacency (lower depth → higher depth)
    // =================================================================

    /// A support (upward) adjacency from entities at `fromDepth` to entities at
    /// `toDepth`. Row order is determined by the creation method (typically
    /// Inverse), and is stable but not semantically ordered.
    ///
    /// Supports never carry periodic bits — pbi is a property of cones to nodes.
    ///
    /// The adjacency data is stored via `ssp<AdjVariant>` for shared ownership.
    struct SupportAdj
    {
        int fromDepth{-1};   ///< Source stratum (e.g., 0 for nodes)
        int toDepth{-1};     ///< Target stratum (e.g., dim for cells)
        ssp<AdjVariant> adj; ///< Shared adjacency pair (typed by row width)

        /// Access as variable-width tAdjPair.
        tAdjPair &asAdj() { return std::get<tAdjPair>(*adj); }
        [[nodiscard]] const tAdjPair &asAdj() const { return std::get<tAdjPair>(*adj); }

        tAdj2Pair &asAdj2() { return std::get<tAdj2Pair>(*adj); }
        [[nodiscard]] const tAdj2Pair &asAdj2() const { return std::get<tAdj2Pair>(*adj); }

        tAdj1Pair &asAdj1() { return std::get<tAdj1Pair>(*adj); }
        [[nodiscard]] const tAdj1Pair &asAdj1() const { return std::get<tAdj1Pair>(*adj); }

        /// Typed access: `as<tAdj2Pair>()` etc.
        template <class TPair>
        TPair &as() { return std::get<TPair>(*adj); }
        template <class TPair>
        [[nodiscard]] const TPair &as() const { return std::get<TPair>(*adj); }

        [[nodiscard]] index fatherSize() const
        {
            if (!adj)
                return 0;
            return std::visit([](const auto &p) -> index
                              { return p.father ? p.father->Size() : 0; }, *adj);
        }

        [[nodiscard]] bool initialized() const
        {
            if (!adj)
                return false;
            return std::visit([](const auto &p) -> bool
                              { return bool(p.father); }, *adj);
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
        std::function<bool(index iParent, int iSub,
                           index iCandEntity,
                           index candidateParent, int candidateSub)>
            matchExtra;
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
    using OwnershipResolverMulti = std::function<OwnershipDecision(
        const std::vector<index> &parents,
        const std::vector<MPI_int> &parentRanks,
        index nLocalParents)>;

    /// Legacy 2-parent ownership resolver (used by InterpolateDistributed).
    using OwnershipResolver2 = std::function<OwnershipDecision(
        index parentL, index parentR, index nLocalParents)>;

    /// Result of distributed interpolation with globally-unique B entities.
    ///
    /// Given A→C cone (global C indices, A is ghosted via A→C→A), creates
    /// globally-unique B entities (faces or edges) with ownership resolution.
    ///
    /// ## Outputs
    ///
    ///   - **parent2entity**: A→B mapping in global B indices. Father = local A,
    ///     son = ghost A. Slot ordering matches element topology: slot j is
    ///     face/edge j as defined by Element::ObtainFace(j)/ObtainEdge(j).
    ///     Some son entries may be UnInitIndex (fully-ghost B not resolved by
    ///     the push protocol — resolved later by the caller's ghost B pull).
    ///
    ///   - **parent2entityPbi**: Parallel to parent2entity. One NodePeriodicBits
    ///     per (A, B) slot — the uniform XOR between A's sub-entity node-pbi
    ///     and B's stored entity2nodePbi. To get B's node coords in A's frame:
    ///     apply parent2entityPbi to entity2nodePbi, then GetCoordByBits.
    ///     For faces, this is at most 1-bit (a face crosses one periodic
    ///     boundary). For edges, can be multi-bit (e.g., P1|P2 for a corner
    ///     edge). Zero for the first parent. Empty if not periodic.
    ///     Computed locally — no MPI push needed (depends only on cell2nodePbi
    ///     and entity2nodePbi, both available after Step 2b).
    ///
    ///   - **entity2node**: B→C mapping in global C indices. Father-only (owned B).
    ///     Node ordering is from the first-discovered parent's ExtractFaceNodes/
    ///     ExtractEdgeNodes call — this is the entity's own canonical ordering.
    ///
    ///   - **entity2nodePbi**: B→C pbi. Father-only. Parallel to entity2node.
    ///     Pbi from the first-discovered parent's perspective (the entity's own
    ///     frame). Other parents' perspectives differ by parent2entityPbi.
    ///     Empty if not periodic.
    ///
    ///   - **entity2parent**: B→A mapping in global A indices. Father-only.
    ///     Variable-width: 1 (boundary) or 2 (internal) for faces, 1..N for edges.
    ///     Complete under A→C→A ghosting (all parents sharing any C-vertex with
    ///     a local A are present as ghosts, so all parents of any locally-visible
    ///     B entity are enumerated).
    ///
    ///   - **entityElemInfo**: Per-entity element info. Father-only.
    ///
    /// ## Ghost B entities
    ///
    /// NOT included. Use evaluateGhostTree(A → A2B → B) to pull them, then
    /// BorrowAndPull on entity2node / entity2nodePbi / entityElemInfo.
    /// parent2entityPbi for ghost A parents is already populated (the push
    /// only carries the global B index; pbi is local). After ghost B pull,
    /// entity2parent for ghost B can be obtained via Inverse(parent2entity)
    /// or by ghost-pulling entity2parent itself.
    /// @tparam e2p_rs  Row-size of entity2parent (NonUniformSize = variable,
    ///                 2 = fixed for faces, etc.).
    template <rowsize e2p_rs = NonUniformSize>
    struct InterpolateGlobalResultT
    {
        tAdjPair parent2entity;                   ///< A → B (global B indices). Father = local A,
                                                  ///< son = ghost A. Slot j = face/edge j per topology.
        tPbiPair parent2entityPbi;                ///< A → B pbi (parallel to parent2entity).
                                                  ///< Uniform XOR: A's sub-entity node-pbi vs B's stored
                                                  ///< entity2nodePbi. Faces: at most 1 bit. Edges: multi-bit.
                                                  ///< 0 for B's first parent. Empty if not periodic.
                                                  ///< No push needed — computed locally in Step 2b.
        tAdjPair entity2node;                     ///< B → C (global C indices). Father-only (owned B).
                                                  ///< Node order: first-discovered parent's extraction order.
        ArrayAdjacencyPair<e2p_rs> entity2parent; ///< B → A (global A indices). Father-only (owned B).
                                                  ///< Width 1-2 for faces, 1-N for edges.
                                                  ///< Complete under A→C→A ghosting.
        tPbiPair entity2nodePbi;                  ///< B → C pbi. Father-only (owned B).
                                                  ///< First-discovered parent's perspective (entity's own frame).
                                                  ///< Parallel to entity2node. Empty if not periodic.
        tElemInfoArrayPair entityElemInfo;        ///< Per-B elem info. Father-only (owned B).
        index nOwnedEntities{0};                  ///< Number of owned B entities (father size).
    };

    /// Default: all fields variable-width.
    using InterpolateGlobalResult = InterpolateGlobalResultT<>;

    // =================================================================
    // InterpolateResult: output of LOCAL sub-entity interpolation
    // =================================================================

    /// Result of interpolating (extracting) sub-entities from parent→node connectivity.
    ///
    /// Given parent→node (e.g., cell→node), Interpolate creates intermediate entities
    /// (e.g., faces or edges) by extracting sub-entities from element topology,
    /// deduplicating by sorted vertex comparison, and building both parent→entity
    /// and entity→node adjacencies.
    ///
    /// All indices are local (0-based within the input arrays). No MPI communication
    /// is performed — the caller is responsible for providing a complete view
    /// (local + ghost cells) and for subsequent ownership resolution / ghost exchange.
    /// Result of local (rank-only) sub-entity interpolation.
    ///
    /// Given parent→node (e.g., cell→node), Interpolate creates intermediate entities
    /// (e.g., faces or edges) by extracting sub-entities from element topology,
    /// deduplicating by sorted vertex comparison (+ optional matchExtra for periodic),
    /// and building both parent→entity and entity→node adjacencies.
    ///
    /// All indices are local (0-based within the input arrays). No MPI communication
    /// is performed. The caller provides a complete view (local + ghost cells) and
    /// handles subsequent ownership resolution / ghost exchange.
    ///
    /// ## Ordering guarantees
    ///
    ///   - parent2entity[iParent][j] corresponds to sub-entity j of parent iParent
    ///     as defined by query.numSubEntities / query.describe / query.extractNodes.
    ///   - entity2node[iEnt] stores nodes in the order extracted by the **first parent**
    ///     that created entity iEnt. Later parents that find the same entity do not
    ///     alter the node order.
    ///   - entity2parent[iEnt] lists parents in discovery order (first parent first).
    ///
    /// @tparam p2e_rs  Row-size of parent2entity (NonUniformSize = variable).
    /// @tparam e2n_rs  Row-size of entity2node (NonUniformSize = variable).
    /// @tparam e2p_rs  Row-size of entity2parent (NonUniformSize = variable).
    template <rowsize p2e_rs = NonUniformSize,
              rowsize e2n_rs = NonUniformSize,
              rowsize e2p_rs = NonUniformSize>
    struct InterpolateResultT
    {
        ArrayAdjacencyPair<p2e_rs> parent2entity; ///< parent → entities. Father-only. Slot j = sub-entity j.
        ArrayAdjacencyPair<e2n_rs> entity2node;   ///< entity → nodes. Father-only. First-parent extraction order.
        ArrayAdjacencyPair<e2p_rs> entity2parent; ///< entity → parents. Father-only. Variable-width:
                                                  ///< 1 for boundary faces, 2 for internal faces, N for edges.
                                                  ///< Discovery order (first parent first).
        std::vector<ElemInfo> entityElemInfo;     ///< Per-entity element info (zone=0, type from SubEntityDesc::typeTag).
        std::vector<std::vector<NodePeriodicBits>> parent2entityPbi;
        ///< Per-parent, per-sub pbi. Parallel to parent2entity.
        ///< Populated by InterpolateGlobal Step 2b, not by
        ///< InterpolateLocal itself. Empty if not periodic or
        ///< if InterpolateLocal was called directly.
        index nEntities{0}; ///< Total number of unique entities created.
    };

    /// Default InterpolateResult: all fields are variable-width (NonUniformSize).
    using InterpolateResult = InterpolateResultT<>;

    // (InterpolateDistributedResult and legacy types moved above)

    // =================================================================
    // InterpolateDistributedResult: output of distributed interpolation
    // =================================================================

    /// Result of legacy distributed interpolation (2-parent only).
    ///
    /// Extends InterpolateResult with ghost communication: entity arrays have
    /// father (owned) + son (ghost) populated. parent2entity entries use
    /// local-appended entity indices that span both father and son.
    ///
    /// **Legacy**: superseded by InterpolateGlobal for production use.
    /// Retained for backward compatibility with InterpolateDistributed.
    /// Limitations: entity2parent is fixed-2 (tAdj2Pair), does not support
    /// N-parent edges, does not produce parent2entityPbi.
    ///
    /// @tparam p2e_rs  Row-size of parent2entity (NonUniformSize = variable).
    /// @tparam e2n_rs  Row-size of entity2node (NonUniformSize = variable).
    /// @tparam e2p_rs  Row-size of entity2parent (2 = fixed face-to-parent).
    template <rowsize p2e_rs = NonUniformSize,
              rowsize e2n_rs = NonUniformSize,
              rowsize e2p_rs = 2>
    struct InterpolateDistributedResultT
    {
        ArrayAdjacencyPair<p2e_rs> parent2entity; ///< parent → entities. Father = local parents,
                                                  ///< son = ghost parents. Local-appended entity indices.
        ArrayAdjacencyPair<e2n_rs> entity2node;   ///< entity → nodes. Father = owned, son = ghost.
        ArrayAdjacencyPair<e2p_rs> entity2parent; ///< entity → (parentL, parentR).
                                                  ///< Father = owned, son = ghost. Local-appended parent indices.
                                                  ///< parentR = UnInitIndex for boundary entities.
        tElemInfoArrayPair entityElemInfo;        ///< Per-entity element info. Father = owned, son = ghost.
        index nOwnedEntities{0};                  ///< Number of owned entities (father size).
    };

    /// Default InterpolateDistributedResult: entity2parent is fixed-2 (as before).
    using InterpolateDistributedResult = InterpolateDistributedResultT<>;

    // =================================================================
    // MeshConnectivity: the layered DAG
    // =================================================================

    /// Manages the layered DAG of mesh adjacency relations.
    ///
    /// Cones (downward adjacencies) and supports (upward adjacencies) are stored
    /// in separate vectors. Each is identified by a `(fromDepth, toDepth)` pair
    /// using dynamic depth tags (e.g., `(dim, 0)` for cell→node).
    ///
    /// Adjacency data is stored via `ssp<AdjVariant>` for shared ownership.
    /// Both the DAG's ConeAdj/SupportAdj and the legacy mesh members can share
    /// the same underlying arrays (shallow copy of ssp<Array> pointers).
    ///
    /// The adjacency registry (`adjRegistry`) maps AdjKind tags to
    /// `ssp<AdjVariant>` for use by the ghost traversal system. No raw pointers.
    /// Only a restricted set of adjacencies may be registered:
    ///   - Direct cones/supports (inter-level, e.g., Cell2Node, Node2Cell)
    ///   - Intra-level adjacencies via Node or Face (e.g., Cell2Cell, Cell2CellFace)
    /// More complex composed adjacencies are NOT stored in the registry.
    struct MeshConnectivity
    {
        int meshDim{0};
        std::vector<ConeAdj> cones;
        std::vector<SupportAdj> supports;

        // -----------------------------------------------------------------
        // Adjacency registry (restricted set for ghost traversal)
        // -----------------------------------------------------------------

        /// Maps AdjKind to the shared AdjVariant used by ghost chain evaluation.
        /// Only direct cones/supports and intra-level (via Node/Face)
        /// adjacencies are allowed. Registered via registerAdj().
        /// Owns shared references (ssp) — no raw pointers.
        std::unordered_map<AdjKind, ssp<AdjVariant>, AdjKindHash> adjRegistry;

        /// Per-EntityKind global offsets mapping (for ownership determination).
        /// Must be registered for every EntityKind that appears as a root anchor
        /// or COLLECT target in any ghost chain.
        std::unordered_map<EntityKind, ssp<GlobalOffsetsMapping>> globalMappings;

        /// Register an adjacency for ghost traversal.
        /// The ssp shares ownership with the ConeAdj/SupportAdj that holds the data.
        /// Overwrites any existing registration for the same AdjKind.
        void registerAdj(AdjKind kind, ssp<AdjVariant> adjPtr);

        /// Convenience: register a specific pair type (tAdjPair, tAdj2Pair, etc.).
        /// Stores a father-only shallow reference. The evaluator creates its
        /// own scratch transformers internally for any ghost data it needs.
        template <class TPair>
        void registerAdj(AdjKind kind, TPair &pair)
        {
            auto adjVar = makeAdjVariant<TPair>();
            auto &stored = std::get<TPair>(*adjVar);
            stored.father = pair.father;
            // son and ghost mapping are NOT copied. The evaluator's scratch
            // pull creates its own temporary son/transformer when needed.
            registerAdj(kind, std::move(adjVar));
        }

        /// Const overload: same as above but accepts a const reference.
        /// Safe because we only copy the father shared_ptr (no mutation).
        template <class TPair>
        void registerAdj(AdjKind kind, const TPair &pair)
        {
            auto adjVar = makeAdjVariant<TPair>();
            auto &stored = std::get<TPair>(*adjVar);
            stored.father = pair.father;
            registerAdj(kind, std::move(adjVar));
        }

        /// Overload for AdjPairTracked<TPair>: unwrap to base TPair.
        template <class TPair>
        void registerAdj(AdjKind kind, AdjPairTracked<TPair> &pair)
        {
            registerAdj(kind, static_cast<TPair &>(pair));
        }

        /// Const overload for AdjPairTracked<TPair>.
        template <class TPair>
        void registerAdj(AdjKind kind, const AdjPairTracked<TPair> &pair)
        {
            registerAdj(kind, static_cast<const TPair &>(pair));
        }

        /// Register a GlobalOffsetsMapping for an EntityKind.
        void registerGlobalMapping(EntityKind kind, const ssp<GlobalOffsetsMapping> &gm);

        /// Resolve an AdjKind to the registered AdjVariant.
        /// Returns nullptr if not registered.
        ssp<AdjVariant> resolveAdj(AdjKind kind) const;

        /// Resolve a GlobalOffsetsMapping for an EntityKind.
        /// Returns nullptr if not registered.
        const ssp<GlobalOffsetsMapping> &getGlobalMapping(EntityKind kind) const;

        /// Check whether an AdjKind is registered.
        [[nodiscard]] bool hasAdj(AdjKind kind) const;

        // -----------------------------------------------------------------
        // Cone management
        // -----------------------------------------------------------------

        /// Add a new cone for (fromDepth, toDepth). Returns reference.
        /// Asserts no duplicate exists.
        ConeAdj &addCone(int fromDepth, int toDepth);

        /// Find a cone by (fromDepth, toDepth). Returns nullptr if not found.
        ConeAdj *findCone(int fromDepth, int toDepth);
        const ConeAdj *findCone(int fromDepth, int toDepth) const;
        bool hasCone(int fromDepth, int toDepth) const;

        // -----------------------------------------------------------------
        // Support management
        // -----------------------------------------------------------------

        /// Add a new support for (fromDepth, toDepth). Returns reference.
        /// Asserts no duplicate exists.
        SupportAdj &addSupport(int fromDepth, int toDepth);

        /// Find a support by (fromDepth, toDepth). Returns nullptr if not found.
        SupportAdj *findSupport(int fromDepth, int toDepth);
        const SupportAdj *findSupport(int fromDepth, int toDepth) const;
        bool hasSupport(int fromDepth, int toDepth) const;

        // -----------------------------------------------------------------
        // Core DSL operations (static, templated on adjacency row-size)
        // -----------------------------------------------------------------
        // Input adjacencies can be any ArrayAdjacencyPair<rs> (fixed or variable).
        // Output adjacencies are always tAdjPair (NonUniformSize) because the
        // result width is generally unpredictable at compile time.
        // Default template args = NonUniformSize for backward compatibility.

        /// Invert a cone to get its support (distributed, MPI push-back).
        ///
        /// Given a cone adjacency A→B (for each A-entity, list of B-entities),
        /// compute the support adjacency B→A (for each B-entity, list of A-entities
        /// that reference it). Result is globally complete via MPI push-back.
        ///
        /// The input cone can be any row-size (fixed or variable). The output
        /// support is always variable-width (tAdjPair) because fan-in is not
        /// known at compile time.
        ///
        /// @tparam cone_rs      Row-size of the input cone (NonUniformSize or fixed).
        /// @param cone         Adjacency: from → to (global indices).
        /// @param nToLocal     Local number of "to" entities on this rank.
        /// @param mpi          MPI communicator.
        /// @param fromLocal2Global  Maps local from-entity index to global index.
        /// @param toLocal2Global    Maps local to-entity index to global index.
        /// @param toGlobalMapping   Global mapping for to-entities (ownership lookup).
        /// @return             CSR adjacency: to → from (global, complete). Father-only.
        template <rowsize cone_rs = NonUniformSize>
        static tAdjPair Inverse(
            const ArrayAdjacencyPair<cone_rs> &cone,
            index nToLocal,
            const MPIInfo &mpi,
            const std::function<index(index)> &fromLocal2Global,
            const std::function<index(index)> &toLocal2Global,
            const ssp<GlobalOffsetsMapping> &toGlobalMapping);

        /// Compose two adjacencies: A→B + B→C → A→C (delegates to ComposeFiltered
        /// with SharedCountPredicate{1}).
        ///
        /// @tparam rs_AB  Row-size of AB adjacency.
        /// @tparam rs_BC  Row-size of BC adjacency.
        /// @tparam out_rs Row-size of output (NonUniformSize or fixed).
        template <rowsize rs_AB = NonUniformSize, rowsize rs_BC = NonUniformSize,
                  rowsize out_rs = NonUniformSize>
        static ArrayAdjacencyPair<out_rs> Compose(
            const ArrayAdjacencyPair<rs_AB> &AB,
            const ArrayAdjacencyPair<rs_BC> &BC,
            index nALocal,
            const std::unordered_map<index, index> &bGlobal2Local,
            const std::function<index(index)> &aLocal2Global,
            bool removeSelf = false);

        /// Compose two adjacencies with on-the-fly predicate filtering.
        ///
        /// For each row a in AB, iterates b in AB[a], collects c in BC[b].
        /// Counts shared B-entities per candidate, applies predicate.
        ///
        /// ## Ghost prerequisite
        ///
        /// **BC must contain (father+son) all B-entities referenced by AB.**
        /// The caller must ghost-pull BC for every global B-index that appears
        /// in any row of AB. The `bGlobal2Local` map must cover all such
        /// B-globals. If a B-global from AB is not in `bGlobal2Local`, the
        /// method asserts (fatal error indicating incomplete ghost pull).
        ///
        /// Typical pattern: before calling ComposeFiltered, the caller builds
        /// BC (e.g., node2cell via Inverse), then ghost-pulls it for all
        /// off-rank B-entities referenced by local AB rows.
        ///
        /// @tparam rs_AB        Row-size of AB adjacency.
        /// @tparam rs_BC        Row-size of BC adjacency.
        /// @tparam Predicate    Predicate type.
        /// @param AB           A → B (father only, global B indices).
        /// @param BC           B → C (father+son, global C indices).
        /// @param nALocal      Number of local A-entities.
        /// @param bGlobal2Local Maps global B-index to local-appended index in BC.
        ///                      Must contain every B-global that appears in AB.
        /// @param aLocal2Global Maps local A-index to global A-index.
        /// @param pred         Predicate(a_global, c_global, nShared) → keep?
        /// @param matchExtra   Optional second predicate called after pred passes.
        ///                     Receives (aLocal, cGlobal, sharedBGlobals) where
        ///                     sharedBGlobals is the list of B-entities through
        ///                     which A and C are connected. If set and returns
        ///                     false, the (A, C) pair is rejected.
        ///                     Used for periodic pbi filtering.
        /// @return             A → C adjacency, father-only. Variable or fixed-width
        ///                     depending on out_rs.
        /// @tparam out_rs       Row-size of output (NonUniformSize = variable,
        ///                      or fixed: rows shorter than out_rs are padded with
        ///                      UnInitIndex, rows longer trigger assert).
        template <rowsize rs_AB = NonUniformSize, rowsize rs_BC = NonUniformSize,
                  rowsize out_rs = NonUniformSize, class Predicate = SharedCountPredicate>
        static ArrayAdjacencyPair<out_rs> ComposeFiltered(
            const ArrayAdjacencyPair<rs_AB> &AB,
            const ArrayAdjacencyPair<rs_BC> &BC,
            index nALocal,
            const std::unordered_map<index, index> &bGlobal2Local,
            const std::function<index(index)> &aLocal2Global,
            Predicate &&pred,
            const std::function<bool(index aLocal, index cGlobal,
                                     const std::vector<index> &sharedBGlobals)>
                &matchExtra = nullptr);

        // -----------------------------------------------------------------
        // InterpolateLocal: rank-local sub-entity extraction (no MPI)
        // -----------------------------------------------------------------

        /// Extract sub-entities (faces or edges) from parent→node connectivity.
        /// This is a **local-only** operation — no MPI communication.
        ///
        /// Enumerates and deduplicates B entities within a contiguous block of
        /// parent A entities (typically local + ghost cells). Produces local-indexed
        /// parent→entity, entity→node, and entity→parent adjacencies.
        ///
        /// Used internally by InterpolateGlobal (Step 1). Can also be used
        /// directly for rank-local analysis or testing.
        ///
        /// ## Deduplication
        ///
        /// Two sub-entities are considered the same B entity if:
        ///   1. They have the same typeTag (from SubEntityDesc).
        ///   2. Their sorted vertex sets (first nVertices nodes) are equal.
        ///   3. query.matchExtra (if set) returns true — used for periodic meshes
        ///      to implement the collaborating pbi check.
        ///
        /// ## Ordering guarantees
        ///
        ///   - parent2entity[iParent][j]: slot j = sub-entity j per query callbacks.
        ///   - entity2node[iEnt]: nodes in first-discovered parent's extraction order.
        ///   - entity2parent[iEnt]: parents in discovery order (first parent first).
        ///
        /// ## Ghost note
        ///
        /// The caller typically passes all A parents (father+son) so that
        /// shared sub-entities are deduplicated within the local view. If ghost
        /// A parents are omitted, shared B entities at rank boundaries will
        /// appear as separate single-sided entities.
        ///
        /// ## Pbi note
        ///
        /// InterpolateLocal does not handle pbi. parent2entityPbi in the result
        /// is left empty. InterpolateGlobal computes it in Step 2b after calling
        /// InterpolateLocal, using the parent2nodePbi input and extractPbi callback.
        ///
        /// @param parent2node    Parent → nodes (father-only or father+son).
        ///                       Accessed via operator[] for indices [0, nParent).
        /// @tparam p2n_rs       Row-size of parent2node (NonUniformSize or fixed).
        /// @param query          User-provided callbacks describing sub-entity topology.
        /// @param nParent        Number of parent entities to process.
        /// @param nNode          Total number of nodes (for reverse-index sizing).
        /// @param mpi            MPI info (only for array allocation, no communication).
        /// @return               InterpolateResult with all adjacencies (local indices).
        template <rowsize p2n_rs = NonUniformSize>
        static InterpolateResult Interpolate(
            const ArrayAdjacencyPair<p2n_rs> &parent2node,
            const SubEntityQuery &query,
            index nParent,
            index nNode,
            const MPIInfo &mpi);

        /// Legacy distributed interpolation (2-parent only, no pbi output).
        ///
        /// Superseded by InterpolateGlobal for production use. Retained for
        /// backward compatibility. Limitations vs InterpolateGlobal:
        ///   - entity2parent is fixed-2 (tAdj2Pair), cannot represent N-parent edges.
        ///   - No parent2entityPbi output.
        ///   - No entity2nodePbi output.
        ///   - Uses push-based ghost exchange (not pull via evaluateGhostTree).
        ///
        /// Extends Interpolate by:
        ///   1. Calling the local Interpolate on all parents (local + ghost).
        ///   2. Using the OwnershipResolver callback to decide which rank owns
        ///      each entity.
        ///   3. Compacting owned entities and assigning global IDs.
        ///   4. Push-based ghost exchange so non-owning ranks receive the
        ///      entities they need.
        ///   5. Resolving parent2entity so all entries (for both local and ghost
        ///      parents) use local-appended entity indices.
        ///
        /// @param parent2node       CSR: parent → nodes (father + son).
        /// @param parentGhostMapping Ghost mapping for parent entities (for global
        ///                           index lookups). Must map local-appended index
        ///                           to global parent index.
        /// @param query             Sub-entity topology callbacks.
        /// @param nLocalParents     Number of local (owned/father) parents.
        /// @param nTotalParents     Total parents (father + son).
        /// @param nNode             Total number of nodes.
        /// @param resolver          Ownership callback.
        /// @param nodeGhostMapping  Ghost mapping for nodes (local-appended → global).
        ///                          Used to convert entity2node to global before push
        ///                          and back to local on the receiver.
        /// @param mpi               MPI communicator.
        /// @return InterpolateDistributedResult with owned + ghost entities.
        template <rowsize p2n_rs = NonUniformSize>
        static InterpolateDistributedResult InterpolateDistributed(
            const ArrayAdjacencyPair<p2n_rs> &parent2node,
            const OffsetAscendIndexMapping &parentGhostMapping,
            const SubEntityQuery &query,
            index nLocalParents,
            index nTotalParents,
            index nNode,
            const OwnershipResolver2 &resolver,
            const OffsetAscendIndexMapping &nodeGhostMapping,
            const MPIInfo &mpi);

        /// Distributed interpolation producing globally-unique B entities.
        ///
        /// Given an A→C cone with pbi (where A is ghosted via A→C→A), creates
        /// globally-unique B entities (faces or edges). B entities are created
        /// only from **local (non-ghost) A parents**. Ghost A parents are used
        /// solely for deduplication — when a local A's sub-entity shares C-vertices
        /// with a ghost A's sub-entity, they are recognized as the same B entity.
        ///
        /// ## Ghost prerequisites (caller's responsibility)
        ///
        /// **A must be ghosted by at least the A→C→A 1-hop node-neighbor set.**
        /// Every A entity sharing at least one C-vertex with a local A must be
        /// present as a ghost. This guarantees:
        ///   - All B entities at rank boundaries are locally deduplicated.
        ///   - entity2parent is complete (all parents of any locally-visible B
        ///     are enumerated).
        ///
        /// If the ghost set is insufficient, shared B entities across ranks will
        /// appear as single-sided. This is a silent error the method cannot detect
        /// (boundary faces/edges are legitimately single-sided).
        ///
        /// **C (nodes) must be ghosted** to cover all C-vertices referenced by
        /// ghost A parents. Typically guaranteed by the A→C→A ghost chain.
        ///
        /// ## Algorithm
        ///
        ///   1. Local enumeration + dedup (InterpolateLocal) on all A (father+son).
        ///   2. Extract B→C pbi via extractPbi. Compute parent2entityPbi (uniform
        ///      XOR per matched node between each parent's view and entity's stored
        ///      first-parent view).
        ///   3. Classify: fully local → owned; fully ghost → discard; straddling →
        ///      ownership resolver callback.
        ///   4. Compact owned entities, convert indices to global.
        ///   5. Assign global B indices via createFatherGlobalMapping.
        ///   6. Push (globalA, globalB, subIdx) triplets to peer ranks so they can
        ///      fill parent2entity. parent2entityPbi is NOT pushed — it is a local
        ///      property of (cell, slot) computed entirely from cell2nodePbi and
        ///      entity2nodePbi.
        ///
        /// Ghost B entities are NOT produced. Use evaluateGhostTree(A→A2B→B)
        /// after this call to pull them.
        ///
        /// ## Ordering guarantees
        ///
        /// parent2entity[iParent][j] corresponds to sub-entity j per element topology
        /// (ObtainFace(j) / ObtainEdge(j)). entity2node stores nodes in the first-
        /// discovered parent's extraction order. All pbi arrays are parallel to their
        /// adjacency counterparts.
        ///
        /// @param parent2node       CSR: A → C (father+son, local C indices).
        /// @param parent2nodePbi    A → C pbi (father+son). Pass empty pair if not periodic.
        /// @param parentGhostMapping Ghost mapping for A (local-appended → global).
        /// @param parentGlobalMapping Global offsets mapping for A (for rank lookup).
        /// @param nodeGhostMapping   Ghost mapping for C (local-appended → global).
        /// @param query             Sub-entity topology + pbi extraction callbacks.
        /// @param nLocalParents     Number of owned A.
        /// @param nTotalParents     Total A (father + son).
        /// @param nNode             Total C (father + son).
        /// @param resolver          N-parent ownership callback.
        /// @param mpi               MPI communicator.
        /// @return InterpolateGlobalResultT<e2p_rs> with owned B entities in global indices.
        /// @tparam p2n_rs  Row-size of parent2node input.
        /// @tparam e2p_rs  Row-size of entity2parent output (NonUniformSize = variable,
        ///                 2 = fixed for faces). Fixed-width rows shorter than e2p_rs
        ///                 are padded with UnInitIndex.
        template <rowsize p2n_rs = NonUniformSize, rowsize e2p_rs = NonUniformSize>
        static InterpolateGlobalResultT<e2p_rs> InterpolateGlobal(
            const ArrayAdjacencyPair<p2n_rs> &parent2node,
            const tPbiPair &parent2nodePbi,
            const OffsetAscendIndexMapping &parentGhostMapping,
            const GlobalOffsetsMapping &parentGlobalMapping,
            const OffsetAscendIndexMapping &nodeGhostMapping,
            const SubEntityQueryPbi &query,
            index nLocalParents,
            index nTotalParents,
            index nNode,
            const OwnershipResolverMulti &resolver,
            const MPIInfo &mpi);

        // -----------------------------------------------------------------
        // Ghost evaluation
        // -----------------------------------------------------------------

        /// Evaluate a compiled ghost tree to determine which entities to ghost.
        ///
        /// BFS level-by-level evaluation with scratch pulls between levels.
        /// Produces a GhostResult containing per-EntityKind ghost index sets
        /// (union of all COLLECT nodes).
        ///
        /// @param tree          Compiled ghost tree (from CompiledGhostTree::compile).
        /// @param mpi           MPI communicator.
        /// @return              Per-EntityKind sorted, deduplicated ghost indices.
        ///
        /// Preconditions:
        ///   - All AdjKind values in the tree must be registered via registerAdj().
        ///   - The registered tAdjPair arrays must be in Adj_PointToGlobal state.
        ///   - GlobalOffsetsMapping must be available for entity kinds at root
        ///     level (to determine ownership).
        GhostResult evaluateGhostTree(
            const CompiledGhostTree &tree,
            const MPIInfo &mpi) const;
    };

    // =====================================================================
    // Template implementation: ComposeFiltered
    // =====================================================================

    template <rowsize rs_AB, rowsize rs_BC, rowsize out_rs, class Predicate>
    ArrayAdjacencyPair<out_rs> MeshConnectivity::ComposeFiltered(
        const ArrayAdjacencyPair<rs_AB> &AB,
        const ArrayAdjacencyPair<rs_BC> &BC,
        index nALocal,
        const std::unordered_map<index, index> &bGlobal2Local,
        const std::function<index(index)> &aLocal2Global,
        Predicate &&pred,
        const std::function<bool(index aLocal, index cGlobal,
                                 const std::vector<index> &sharedBGlobals)>
            &matchExtra)
    {
        const auto &mpi = AB.father->getMPI();
        const bool hasMatchExtra = bool(matchExtra);
        ArrayAdjacencyPair<out_rs> result;
        result.InitPair("ComposeFiltered_result", mpi);
        result.father->Resize(nALocal);

        for (index iA = 0; iA < nALocal; iA++)
        {
            index aGlobal = aLocal2Global(iA);

            // Collect all candidate C-entities with their shared-B count
            // and optionally the list of shared B-entities.
            std::unordered_map<index, int> candidateSharedCount;
            std::unordered_map<index, std::vector<index>> candidateSharedBs;
            for (auto iB : AB.father->operator[](iA))
            {
                // Fixed-width adjacencies may have UnInitIndex padding — skip.
                if constexpr (rs_AB > 0)
                {
                    if (iB == UnInitIndex)
                        continue;
                }

                auto it = bGlobal2Local.find(iB);
                DNDS_assert_info(it != bGlobal2Local.end(),
                                 fmt::format("ComposeFiltered: B-entity global {} (referenced by A-entity "
                                             "local {} / global {}) not found in bGlobal2Local. "
                                             "Ghost pull of B→C is incomplete — the caller must "
                                             "pull BC for all B-globals referenced by AB rows.",
                                             iB, iA, aGlobal));
                index bLocal = it->second;
                for (auto iC : BC[bLocal])
                {
                    // Fixed-width adjacencies may have UnInitIndex padding — skip.
                    if constexpr (rs_BC > 0)
                    {
                        if (iC == UnInitIndex)
                            continue;
                    }

                    candidateSharedCount[iC]++;
                    if (hasMatchExtra)
                        candidateSharedBs[iC].push_back(iB);
                }
            }

            // Apply predicate and collect passing entries
            std::vector<index> accepted;
            for (auto &[cGlobal, nShared] : candidateSharedCount)
            {
                if (!pred(aGlobal, cGlobal, nShared))
                    continue;
                if (hasMatchExtra)
                {
                    if (!matchExtra(iA, cGlobal, candidateSharedBs[cGlobal]))
                        continue;
                }
                accepted.push_back(cGlobal);
            }

            // Sort for deterministic output
            std::sort(accepted.begin(), accepted.end());

            if constexpr (out_rs == NonUniformSize)
            {
                result.father->ResizeRow(iA, accepted.size());
                for (rowsize j = 0; j < static_cast<rowsize>(accepted.size()); j++)
                    result.father->operator()(iA, j) = accepted[j];
            }
            else
            {
                // Fixed-width output: fill accepted entries, pad remainder with UnInitIndex.
                DNDS_assert_info(static_cast<rowsize>(accepted.size()) <= out_rs,
                                 fmt::format("ComposeFiltered: row {} has {} entries but out_rs={}",
                                             iA, accepted.size(), out_rs));
                for (rowsize j = 0; j < out_rs; j++)
                    result.father->operator()(iA, j) =
                        (j < static_cast<rowsize>(accepted.size())) ? accepted[j] : UnInitIndex;
            }
        }

        return result;
    }

    // =====================================================================
    // Template implementation: Compose
    // =====================================================================

    template <rowsize rs_AB, rowsize rs_BC, rowsize out_rs>
    ArrayAdjacencyPair<out_rs> MeshConnectivity::Compose(
        const ArrayAdjacencyPair<rs_AB> &AB,
        const ArrayAdjacencyPair<rs_BC> &BC,
        index nALocal,
        const std::unordered_map<index, index> &bGlobal2Local,
        const std::function<index(index)> &aLocal2Global,
        bool removeSelf)
    {
        return ComposeFiltered<rs_AB, rs_BC, out_rs>(
            AB, BC, nALocal, bGlobal2Local, aLocal2Global,
            SharedCountPredicate{.minShared = 1, .removeSelf = removeSelf});
    }

    // =====================================================================
    // Template implementation: Inverse
    // =====================================================================

    template <rowsize cone_rs>
    tAdjPair MeshConnectivity::Inverse(
        const ArrayAdjacencyPair<cone_rs> &cone,
        index nToLocal,
        const MPIInfo &mpi,
        const std::function<index(index)> &fromLocal2Global,
        const std::function<index(index)> &toLocal2Global,
        const ssp<GlobalOffsetsMapping> &toGlobalMapping)
    {
        // ----- Step 1: Local inversion -----
        std::unordered_map<index, std::unordered_set<index>> to2fromRecord;
        std::vector<index> ghostToIndices;
        std::unordered_set<index> ghostToSet;

        index nFromLocal = cone.father->Size();
        for (index iFrom = 0; iFrom < nFromLocal; iFrom++)
        {
            index fromGlobal = fromLocal2Global(iFrom);
            for (auto iTo : cone.father->operator[](iFrom))
            {
                // Fixed-width adjacencies may have UnInitIndex padding — skip.
                if constexpr (cone_rs > 0)
                {
                    if (iTo == UnInitIndex)
                        continue;
                }

                to2fromRecord[iTo].insert(fromGlobal);

                auto [ret, rank, val] = toGlobalMapping->search(iTo);
                DNDS_assert_info(ret, fmt::format("Inverse: to-entity {} not found in global mapping", iTo));
                if (rank != mpi.rank && ghostToSet.find(iTo) == ghostToSet.end())
                {
                    ghostToIndices.push_back(iTo);
                    ghostToSet.insert(iTo);
                }
            }
        }

        // ----- Step 2: Build initial support pair -----
        tAdjPair support;
        support.InitPair("Inverse_support", mpi);
        support.father->Resize(nToLocal);

        for (index iTo = 0; iTo < nToLocal; iTo++)
        {
            index toGlobal = toLocal2Global(iTo);
            auto it = to2fromRecord.find(toGlobal);
            if (it != to2fromRecord.end())
            {
                support.father->ResizeRow(iTo, it->second.size());
                rowsize j = 0;
                for (auto fromG : it->second)
                    support.father->operator()(iTo, j++) = fromG;
            }
        }

        // Set up ghost communication
        support.TransAttach();
        support.trans.createFatherGlobalMapping();
        support.trans.createGhostMapping(ghostToIndices);

        support.son->Resize(support.trans.pLGhostMapping->ghostIndex.size());
        for (auto &[toGlobal, fromSet] : to2fromRecord)
        {
            MPI_int rank{-1};
            index val{-1};
            if (!support.trans.pLGhostMapping->search(toGlobal, rank, val))
                DNDS_check_throw_info(false, "Inverse: ghost search failed");
            if (rank >= 0)
            {
                support.son->ResizeRow(val, fromSet.size());
                rowsize j = 0;
                for (auto fromG : fromSet)
                    support.son->operator()(val, j++) = fromG;
            }
        }

        // ----- Step 3: Reverse-push -----
        using tSupportArr = typename decltype(support.son)::element_type;
        ssp<tSupportArr> supportPast = make_ssp<tSupportArr>(ObjName{"Inverse_supportPast"}, mpi);

        typename DNDS::ArrayTransformerType<tSupportArr>::Type supportPastTrans;
        supportPastTrans.setFatherSon(support.son, supportPast);
        supportPastTrans.createFatherGlobalMapping();

        std::vector<index> pushSonSeries(support.son->Size());
        for (index i = 0; i < support.son->Size(); i++)
            pushSonSeries[i] = i;
        supportPastTrans.createGhostMapping(pushSonSeries, support.trans.pLGhostMapping->ghostStart);
        supportPastTrans.createMPITypes();
        supportPastTrans.pullOnce();

        // ----- Step 4: Merge remote contributions -----
        DNDS_assert(DNDS::size_to_index(support.trans.pLGhostMapping->ghostIndex.size()) == support.son->Size());
        DNDS_assert(DNDS::size_to_index(support.trans.pLGhostMapping->pushingIndexGlobal.size()) == supportPast->Size());

        for (index i = 0; i < supportPast->Size(); i++)
        {
            index toGlobal = support.trans.pLGhostMapping->pushingIndexGlobal[i];
            for (auto fromG : (*supportPast)[i])
                to2fromRecord[toGlobal].insert(fromG);
        }

        // ----- Step 5: Rebuild father -----
        tAdjPair result;
        result.InitPair("Inverse_result", mpi);
        result.father->Resize(nToLocal);

        for (index iTo = 0; iTo < nToLocal; iTo++)
        {
            index toGlobal = toLocal2Global(iTo);
            auto it = to2fromRecord.find(toGlobal);
            if (it != to2fromRecord.end())
            {
                std::vector<index> fromVec(it->second.begin(), it->second.end());
                std::sort(fromVec.begin(), fromVec.end());
                result.father->ResizeRow(iTo, fromVec.size());
                for (rowsize j = 0; j < static_cast<rowsize>(fromVec.size()); j++)
                    result.father->operator()(iTo, j) = fromVec[j];
            }
        }

        return result;
    }

} // namespace DNDS::Geom

// Template implementations for Interpolate family (too large for inline)
#include "MeshConnectivity_Interpolate.hxx"
