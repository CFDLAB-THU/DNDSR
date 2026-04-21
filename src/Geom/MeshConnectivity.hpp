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
            return (dim == 2) ? 1 : 1; // always 1; coincides with Face in 2D
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
            : from(from_), to(to_), via(EntityKind::Node)
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

        /// The current default pipeline specification.
        /// Cell ghost: owned cells -> Cell2Cell -> cells
        /// Node ghost: owned cells -> Cell2Cell -> Cell2Node -> nodes
        /// Bnd ghost:  owned bnds  -> Bnd2Node -> Node2Bnd -> bnds
        /// Bnd-node ghost: owned bnds -> Bnd2Node -> Node2Bnd -> Bnd2Node -> nodes
        static GhostSpec defaultPrimary();
    };

    // =================================================================
    // CompiledGhostTree: chains compiled into BFS-ordered forest
    // =================================================================

    /// One node in the compiled ghost tree.
    struct GhostTreeNode
    {
        EntityKind kind;        ///< Entity kind at this node.
        AdjKind hop;            ///< Adjacency used to reach this node from parent.
                                ///< Undefined (default-constructed) for root nodes.
        bool collect{false};    ///< If true, non-owned entities here become ghosts.
        int level{0};           ///< BFS depth (root = 0).
        int id{-1};             ///< Unique ID within the tree (assigned by compile).
        int parentId{-1};       ///< Parent node ID (-1 for roots).
        std::vector<GhostTreeNode> children;
    };

    /// A reference to a tree node at a specific level, with its parent.
    /// Used by the precomputed per-level lists.
    struct LevelEntry
    {
        int nodeId;             ///< ID of the tree node.
        int parentId;           ///< ID of the parent node (-1 for roots).
        EntityKind kind;        ///< Entity kind of this node.
        AdjKind hop;            ///< Hop used to reach this node.
        bool collect;           ///< Whether to collect at this node.
        bool hasChildren;       ///< Whether this node has children (needs pull).
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
        int maxLevel{0};        ///< Maximum BFS depth across all nodes.
        int totalNodes{0};      ///< Total number of nodes (for flat array sizing).

        /// Precomputed per-level node lists. `levels[L]` contains all tree
        /// nodes at BFS depth L. Level 0 = roots.
        std::vector<std::vector<LevelEntry>> levels;

        /// Compile a GhostSpec into a forest. Validates chain consistency.
        /// Assigns node IDs, builds per-level lists.
        /// Throws std::runtime_error on invalid chains.
        static CompiledGhostTree compile(const GhostSpec &spec);

        /// Collect all distinct AdjKind values used by any hop in the tree.
        std::unordered_set<AdjKind, AdjKindHash> requiredAdjs() const;

        /// Pre-check that all required adjacencies exist in the DAG.
        /// Returns the set of missing AdjKind values (empty = all available).
        std::vector<AdjKind> checkAvailable(const MeshConnectivity &dag) const;

        /// Collect all EntityKind values that appear at COLLECT nodes.
        std::unordered_set<EntityKind> collectedKinds() const;

        /// Pretty-print the tree (for diagnostics).
        std::string dump() const;
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
            for (auto &[k, v] : ghostIndices)
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

    // =================================================================
    // ConeAdj: downward adjacency (higher depth → lower depth)
    // =================================================================

    /// A cone (downward) adjacency from entities at `fromDepth` to entities at
    /// `toDepth`. Row order is defined by element topology and must not be
    /// permuted (e.g., cell2node ordering determines element shape functions).
    ///
    /// Periodic bits (`pbi`) are only meaningful when `toDepth == 0` (nodes).
    /// For non-periodic meshes or non-node targets, `pbi` remains uninitialized.
    struct ConeAdj
    {
        int fromDepth{-1}; ///< Source stratum (e.g., dim for cells, dim-1 for faces)
        int toDepth{-1};   ///< Target stratum (e.g., 0 for nodes, dim-1 for faces)
        AdjVariant adj;    ///< The adjacency pair (typed by row width)
        tPbiPair pbi;      ///< Periodic bits per entry (only for toDepth==0, optional)

        /// Access as variable-width tAdjPair. Throws if adj holds a fixed-width type.
        tAdjPair &asAdj() { return std::get<tAdjPair>(adj); }
        [[nodiscard]] const tAdjPair &asAdj() const { return std::get<tAdjPair>(adj); }

        /// Access as fixed-width tAdj2Pair (e.g., for cell2face with 2 entries).
        tAdj2Pair &asAdj2() { return std::get<tAdj2Pair>(adj); }
        [[nodiscard]] const tAdj2Pair &asAdj2() const { return std::get<tAdj2Pair>(adj); }

        tAdj1Pair &asAdj1() { return std::get<tAdj1Pair>(adj); }
        [[nodiscard]] const tAdj1Pair &asAdj1() const { return std::get<tAdj1Pair>(adj); }

        /// Number of local (father) rows.
        [[nodiscard]] index fatherSize() const
        {
            return std::visit([](const auto &p) -> index
                              { return p.father ? p.father->Size() : 0; }, adj);
        }

        /// Check if the adjacency pair is initialized (father is non-null).
        [[nodiscard]] bool initialized() const
        {
            return std::visit([](const auto &p) -> bool
                              { return bool(p.father); }, adj);
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
    struct SupportAdj
    {
        int fromDepth{-1}; ///< Source stratum (e.g., 0 for nodes)
        int toDepth{-1};   ///< Target stratum (e.g., dim for cells)
        AdjVariant adj;    ///< The adjacency pair (typed by row width)

        /// Access as variable-width tAdjPair.
        tAdjPair &asAdj() { return std::get<tAdjPair>(adj); }
        [[nodiscard]] const tAdjPair &asAdj() const { return std::get<tAdjPair>(adj); }

        tAdj2Pair &asAdj2() { return std::get<tAdj2Pair>(adj); }
        [[nodiscard]] const tAdj2Pair &asAdj2() const { return std::get<tAdj2Pair>(adj); }

        tAdj1Pair &asAdj1() { return std::get<tAdj1Pair>(adj); }
        [[nodiscard]] const tAdj1Pair &asAdj1() const { return std::get<tAdj1Pair>(adj); }

        [[nodiscard]] index fatherSize() const
        {
            return std::visit([](const auto &p) -> index
                              { return p.father ? p.father->Size() : 0; }, adj);
        }

        [[nodiscard]] bool initialized() const
        {
            return std::visit([](const auto &p) -> bool
                              { return bool(p.father); }, adj);
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
        int nVertices{0};  ///< Number of corner vertices (used for deduplication).
        int nNodes{0};     ///< Total number of nodes (vertices + mid-edge + ...).
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
    // InterpolateResult: output of sub-entity interpolation
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
    struct InterpolateResult
    {
        tAdjPair parent2entity; ///< CSR: parent → entities (e.g., cell2face). Father-only.
        tAdjPair entity2node;   ///< CSR: entity → nodes (e.g., face2node). Father-only.
        tAdj2Pair entity2parent; ///< Fixed-2: entity → (parentL, parentR). Father-only.
                                 ///< parentR = UnInitIndex for single-sided (boundary) entities.
        std::vector<ElemInfo> entityElemInfo; ///< Per-entity element info (zone=0, type from SubEntityDesc::typeTag).
        index nEntities{0};     ///< Total number of unique entities created.
        bool duplicateOverflow{false}; ///< Set if a sub-entity matched an entity that already has
                                       ///< two parents (would need a third). Indicates incorrect
                                       ///< deduplication — typically a missing matchExtra on
                                       ///< periodic meshes.
    };

    // =================================================================
    // MeshConnectivity: the layered DAG
    // =================================================================

    /// Manages the layered DAG of mesh adjacency relations.
    ///
    /// Cones (downward adjacencies) and supports (upward adjacencies) are stored
    /// in separate vectors. Each is identified by a `(fromDepth, toDepth)` pair
    /// using dynamic depth tags (e.g., `(dim, 0)` for cell→node).
    ///
    /// The adjacency registry (`adjRegistry`) maps AdjKind tags to tAdjPair
    /// pointers for use by the ghost traversal system. Only a restricted set
    /// of adjacencies may be registered:
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

        /// Maps AdjKind to the tAdjPair used by ghost chain evaluation.
        /// Only direct cones/supports and intra-level (via Node/Face)
        /// adjacencies are allowed. Registered via registerAdj().
        std::unordered_map<AdjKind, tAdjPair *, AdjKindHash> adjRegistry;

        /// Per-EntityKind global offsets mapping (for ownership determination).
        /// Must be registered for every EntityKind that appears as a root anchor
        /// or COLLECT target in any ghost chain.
        std::unordered_map<EntityKind, ssp<GlobalOffsetsMapping>> globalMappings;

        /// Register an adjacency for ghost traversal.
        /// The pointer must remain valid for the lifetime of the registry entry.
        /// Overwrites any existing registration for the same AdjKind.
        void registerAdj(AdjKind kind, tAdjPair &pair);

        /// Register a GlobalOffsetsMapping for an EntityKind.
        void registerGlobalMapping(EntityKind kind, const ssp<GlobalOffsetsMapping> &gm);

        /// Resolve an AdjKind to the registered tAdjPair.
        /// Returns nullptr if not registered.
        tAdjPair *resolveAdj(AdjKind kind);
        const tAdjPair *resolveAdj(AdjKind kind) const;

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
        // Core DSL operations (static, pure-functional on tAdjPair)
        // -----------------------------------------------------------------
        // These operate on variable-width tAdjPair. Fixed-width adjacencies
        // must be widened before use or the caller extracts tAdjPair from
        // the variant.

        /// Invert a cone to get its support (distributed, MPI push-back).
        ///
        /// Given a cone adjacency A→B (CSR: for each A-entity, list of B-entities),
        /// compute the support adjacency B→A (for each B-entity, list of A-entities
        /// that reference it). Result is globally complete via MPI push-back.
        ///
        /// @param cone         CSR adjacency: from → to (global indices).
        /// @param nToLocal     Local number of "to" entities on this rank.
        /// @param mpi          MPI communicator.
        /// @param fromLocal2Global  Maps local from-entity index to global index.
        /// @param toLocal2Global    Maps local to-entity index to global index.
        /// @param toGlobalMapping   Global mapping for to-entities (ownership lookup).
        /// @return             CSR adjacency: to → from (global, complete). Father-only.
        static tAdjPair Inverse(
            const tAdjPair &cone,
            index nToLocal,
            const MPIInfo &mpi,
            const std::function<index(index)> &fromLocal2Global,
            const std::function<index(index)> &toLocal2Global,
            const ssp<GlobalOffsetsMapping> &toGlobalMapping);

        /// Compose two adjacencies: A→B + B→C → A→C (delegates to ComposeFiltered
        /// with SharedCountPredicate{1}).
        static tAdjPair Compose(
            const tAdjPair &AB,
            const tAdjPair &BC,
            index nALocal,
            const std::unordered_map<index, index> &bGlobal2Local,
            const std::function<index(index)> &aLocal2Global,
            bool removeSelf = false);

        /// Compose two adjacencies with on-the-fly predicate filtering.
        ///
        /// For each row a in AB, iterates b in AB[a], collects c in BC[b].
        /// Counts shared B-entities per candidate, applies predicate.
        ///
        /// @param AB           CSR: A → B (father only, global B indices).
        /// @param BC           CSR: B → C (father+son, global C indices).
        /// @param nALocal      Number of local A-entities.
        /// @param bGlobal2Local Maps global B-index to local-appended index in BC.
        /// @param aLocal2Global Maps local A-index to global A-index.
        /// @param pred         Predicate(a_global, c_global, nShared) → keep?
        /// @param matchExtra   Optional second predicate called after pred passes.
        ///                     Receives (aLocal, cGlobal, sharedBGlobals) where
        ///                     sharedBGlobals is the list of B-entities through
        ///                     which A and C are connected. If set and returns
        ///                     false, the (A, C) pair is rejected.
        ///                     Used for periodic pbi filtering.
        /// @return             CSR: A → C (global C indices), father-only.
        template <class Predicate>
        static tAdjPair ComposeFiltered(
            const tAdjPair &AB,
            const tAdjPair &BC,
            index nALocal,
            const std::unordered_map<index, index> &bGlobal2Local,
            const std::function<index(index)> &aLocal2Global,
            Predicate &&pred,
            const std::function<bool(index aLocal, index cGlobal,
                                     const std::vector<index> &sharedBGlobals)>
                &matchExtra = nullptr);

        // -----------------------------------------------------------------
        // Interpolate: extract sub-entities from parent→node connectivity
        // -----------------------------------------------------------------

        /// Extract sub-entities (faces or edges) from parent→node connectivity.
        ///
        /// For each parent entity, queries the SubEntityQuery to enumerate its
        /// sub-entities, extracts node indices, and deduplicates by sorted-vertex
        /// comparison. Produces parent→entity, entity→node, and entity→parent
        /// (2-wide) adjacencies.
        ///
        /// This is a local-only operation: no MPI communication. The caller
        /// provides a contiguous block of parent entities (typically local + ghost
        /// cells) and receives local-indexed results.
        ///
        /// @param parent2node    CSR: parent → nodes (father-only or father+son).
        ///                       Accessed via operator[] for indices [0, nParent).
        /// @param query          User-provided callbacks describing sub-entity topology.
        /// @param nParent        Number of parent entities to process.
        /// @param nNode          Total number of nodes (for reverse-index sizing).
        /// @param mpi            MPI info (only for array allocation, no communication).
        /// @return               InterpolateResult with all adjacencies.
        static InterpolateResult Interpolate(
            const tAdjPair &parent2node,
            const SubEntityQuery &query,
            index nParent,
            index nNode,
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

    template <class Predicate>
    tAdjPair MeshConnectivity::ComposeFiltered(
        const tAdjPair &AB,
        const tAdjPair &BC,
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
        tAdjPair result;
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
                auto it = bGlobal2Local.find(iB);
                DNDS_assert_info(it != bGlobal2Local.end(),
                                 fmt::format("ComposeFiltered: B-entity {} not found in bGlobal2Local", iB));
                index bLocal = it->second;
                for (auto iC : BC[bLocal])
                {
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

            result.father->ResizeRow(iA, accepted.size());
            for (rowsize j = 0; j < static_cast<rowsize>(accepted.size()); j++)
                result.father->operator()(iA, j) = accepted[j];
        }

        return result;
    }

} // namespace DNDS::Geom
