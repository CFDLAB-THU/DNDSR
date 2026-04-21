# MeshConnectivity Implementation Plan {#mesh_connectivity_impl}

**Status:** Implementation plan
**Date:** 2026-04-21
**Depends on:** [Mesh DAG Design](MeshDAGDesign.md)

---

## 1. Scope

Implement `MeshConnectivity` as a standalone class in `src/DNDS/` (or
`src/Geom/`) with its own unit tests, independent of `UnstructuredMesh`. The
class provides a small DSL of composable operations on layered adjacency
graphs.

---

## 2. Ownership Model

### 2.1. Why Not Move

`tAdjPair` wraps `ssp<ParArray>` (father) and `ssp<ParArray>` (son). A C++
move would null the source pair's shared_ptrs, breaking any existing code
that holds a reference to the legacy member. Moving back after DAG operations
is fragile and error-prone.

### 2.2. Shared Ownership via ssp<>

Instead, `MeshConnectivity` stores adjacency data as `ssp<tAdjPair>`. The
legacy members in `UnstructuredMesh` share the same `ssp<>`. Both sides see
the same data. No move, no invalidation.

```cpp
// In MeshConnectivity:
ssp<tAdjPair> cone;     // e.g., cell2node
ssp<tAdjPair> support;  // e.g., node2cell

// In UnstructuredMesh (legacy):
tAdjPair &cell2node = *dag.findAdj(dim, 0)->cone;
// OR: cell2node.father = dag.findAdj(dim, 0)->cone->father; (share the ssp)
```

The simplest approach: `MeshConnectivity::InterLayerAdj` stores `tAdjPair`
by value. When `UnstructuredMesh` adopts arrays, it shallow-copies the
`ssp<>` pointers (father, son) from its legacy members into the DAG's slots.
After adoption, both the legacy member and the DAG slot point to the same
`ParArray` objects. Mutations (resize, ghost build) are visible to both.

---

## 3. Core Operations (DSL)

The DSL is a set of free functions (or methods on `MeshConnectivity`) that
compose adjacency relations. All operations work on `tAdjPair` or equivalent
CSR structures.

### 3.1. `inverse(cone) → support`

Given a cone adjacency `A → B` (CSR: for each A-entity, list of B-entities),
compute the support adjacency `B → A` (for each B-entity, list of A-entities
that reference it).

This is the distributed node-inversion currently done by
`RecoverNode2CellAndNode2Bnd` (with MPI push-back for cross-rank completeness).

```
inverse(cell2node) → node2cell
inverse(face2node) → node2face
inverse(cell2face) → face2cell   // always exactly 1-2 cells per face
```

**Signature:**
```cpp
/// Build the inverse (support) of a cone adjacency.
/// @param cone      CSR adjacency: from → to (global indices).
/// @param nTo       Global number of "to" entities across all ranks.
/// @param mpi       MPI communicator.
/// @return          CSR adjacency: to → from (global indices, complete).
tAdjPair inverse(const tAdjPair &cone, index nTo, const MPIInfo &mpi);
```

**Key property:** The result is globally complete — each "to" entity's row
contains ALL "from" entities across all ranks (via MPI push-back).

### 3.2. `compose(A→B, B→C) → A→C`

Given two adjacencies, compose them (flatten one hop through B):

```
compose(cell2node, node2cell) → cell2cell_node  (node-neighbor)
compose(cell2face, face2cell) → cell2cell_face  (face-neighbor, incl self)
compose(bnd2node, node2cell)  → bnd2cell_raw    (all cells sharing any vertex with bnd)
```

**Signature:**
```cpp
/// Compose two adjacencies: for each A-entity, collect all C-entities
/// reachable via any intermediate B-entity.
/// @param AB   CSR: A → B
/// @param BC   CSR: B → C
/// @param removeSelf  If true, remove diagonal (A==C) entries.
/// @return     CSR: A → C (deduplicated per row).
tAdjPair compose(const tAdjPair &AB, const tAdjPair &BC, bool removeSelf = false);
```

**Behavior:** For each row `a` in AB, iterate over all `b` in AB[a], collect
all `c` in BC[b], deduplicate, optionally remove `a` itself. This is a sparse
matrix-matrix product with boolean semiring.

### 3.3. `filter` (on-the-fly during compose, not post-processing)

Filtering should happen INSIDE the compose operation, not as a separate pass
over the result. When composing `A→B + B→C → A→C`, the filter predicate
evaluates each candidate C-entity as it is discovered through B, before adding
it to the result row. This avoids materializing a large unfiltered intermediate.

The filter predicate is general: it receives the A-entity, the candidate
C-entity, and the set of intermediate B-entities through which the connection
was established. The predicate decides whether to keep the (A, C) pair.

For the common case of "shared sub-entity filtering," the predicate checks
whether A and C share a sub-entity of a specific type (face, edge, or vertex).
The sharing test uses element topology, not just vertex counting — though
vertex counting is a valid fast-path for the common case:

- **Vertex share (current `cell2cell` node-neighbor):** A and C share ≥ 1 vertex
- **Face share (`cell2cellFace`):** A and C share ≥ `dim` vertices
  forming a recognized face sub-element
- **Edge share (future):** A and C share ≥ 2 vertices forming a recognized
  edge sub-element

The share judgment currently relies on counting shared vertices (≥ dim for
face-share, ≥ 2 for edge-share, ≥ 1 for vertex-share), which is correct for
standard simplex/hex elements. Future extension: use element topology tables
to verify the shared vertices actually form a sub-element of the requested
type (handles degenerate cases like pyramids where 4 shared vertices might
form a face or two triangular faces depending on topology).

**Signature:**
```cpp
/// Compose two adjacencies with on-the-fly filtering.
/// @param AB     CSR: A → B
/// @param BC     CSR: B → C
/// @param pred   Predicate(A-entity, C-entity, shared B-entities) → keep?
/// @return       CSR: A → C (only entries passing the predicate).
template <class Predicate>
static tAdjPair ComposeFiltered(
    const tAdjPair &AB, const tAdjPair &BC,
    Predicate &&pred);

/// Common predicate: keep (A, C) pairs sharing ≥ minShared B-entities.
/// Suitable for vertex-share (min=1), edge-share (min=2), face-share (min=dim).
struct SharedCountPredicate
{
    int minShared;
    bool removeSelf = false;
    bool operator()(index a, index c, int nShared) const
    {
        if (removeSelf && a == c) return false;
        return nShared >= minShared;
    }
};
```

**Usage:**
```cpp
// cell2cell (node-neighbor): compose + keep any shared vertex
auto cell2cell = ComposeFiltered(cell2node, node2cell,
    SharedCountPredicate{.minShared = 1, .removeSelf = true});

// cell2cellFace (face-neighbor): compose + keep face-shared only
auto cell2cellFace = ComposeFiltered(cell2node, node2cell,
    SharedCountPredicate{.minShared = dim, .removeSelf = true});

// bnd2cell: compose + keep face-connected only
auto bnd2cell = ComposeFiltered(bnd2node, node2cell,
    SharedCountPredicate{.minShared = dim});
```

### 3.4. `interpolate(A→C) → A→B, B→C`

Given a direct adjacency `A → C` (e.g., cell → node), create an intermediate
entity type B (e.g., faces) by extracting sub-entities from A's element
topology, deduplicating, and building both `A → B` and `B → C` adjacencies.

This is the generalization of `InterpolateFace`:
- `interpolate(cell2node, depth=dim-1)` → creates `cell2face` and `face2node`
- `interpolate(face2node, depth=1)` → creates `face2edge` and `edge2node` (3D)

**Signature:**
```cpp
struct InterpolationResult
{
    tAdjPair parentToEntity;     // A → B (cell2face)
    tAdjPair entityToNode;       // B → C (face2node)
    tElemInfoArrayPair elemInfo; // element type per new entity
    // For periodic meshes:
    tPbiPair entityToNodePbi;    // orientation per B→C entry
};

/// Create intermediate entities between two strata.
/// @param parent2node   CSR: parent → node connectivity
/// @param parentElemInfo  Element types of parent entities
/// @param entityDepth   Topological dimension of entities to create
/// @param periodicInfo  Periodic geometry (nullable)
/// @return              The new cone/support pairs
InterpolationResult interpolate(
    const tAdjPair &parent2node,
    const tElemInfoArrayPair &parentElemInfo,
    int entityDepth,
    const Periodicity *periodicInfo = nullptr);
```

### 3.5. Operation Composition for Current Pipeline

The current mesh build pipeline expressed as DSL operations:

```
Given: cell2node (source of truth, cone: depth dim → depth 0)

Step 1: node2cell = Inverse(cell2node)
Step 2: cell2cell = ComposeFiltered(cell2node, node2cell,
                        SharedCountPredicate{1, removeSelf=true})
        bnd2cell  = ComposeFiltered(bnd2node, node2cell,
                        SharedCountPredicate{dim})
Step 3: BuildGhost using traversal chain [cell → node → cell]
Step 4: (cell2face, face2node) = Interpolate(cell2node, dim-1)
        face2cell = Inverse(cell2face)   // or derived from interpolation
Step 5: node2cell_ghost = Inverse(cell2node) on ghosted mesh
```

For future node-based FV with edges:

```
Step 4b: (face2edge, edge2node) = Interpolate(face2node, 1)
         node2edge = Inverse(edge2node)
         node2node = ComposeFiltered(node2edge, edge2node,
                        SharedCountPredicate{1, removeSelf=true})
```

---

## 4. MeshConnectivity Class Design

```cpp
/// @file MeshConnectivity.hpp
#pragma once
#include "DNDS/ArrayPair.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "Geom/Elements.hpp"

namespace DNDS::Geom
{
    // Forward declarations
    using tAdjPair = ArrayAdjacencyPair<NonUniformSize>;

    /// An adjacency relation between two entity strata.
    struct InterLayerAdj
    {
        int fromDepth;   ///< Source stratum depth (e.g., dim for cells)
        int toDepth;     ///< Target stratum depth (e.g., 0 for nodes)
        tAdjPair cone;   ///< CSR: from-entity → to-entities (downward)
        tAdjPair support;///< CSR: to-entity → from-entities (upward, optional)
        tPbiPair pbi;    ///< Periodic bits per cone entry (optional)
    };

    /// Manages the layered DAG of mesh adjacency relations.
    struct MeshConnectivity
    {
        int meshDim = 0;
        std::vector<InterLayerAdj> layers;

        // --- Layer management ---
        InterLayerAdj &addLayer(int fromDepth, int toDepth);
        InterLayerAdj *findLayer(int fromDepth, int toDepth);
        const InterLayerAdj *findLayer(int fromDepth, int toDepth) const;
        bool hasLayer(int fromDepth, int toDepth) const;

        // --- Core DSL operations (static, pure-functional on pairs) ---

        /// Invert a cone to get its support (distributed, MPI push-back).
        static tAdjPair Inverse(
            const tAdjPair &cone, index nToGlobal, const MPIInfo &mpi);

        /// Compose two adjacencies: A→B + B→C = A→C.
        static tAdjPair Compose(
            const tAdjPair &AB, const tAdjPair &BC,
            bool removeSelf = false);

        /// Filter a composed adjacency by minimum shared-node count.
        static tAdjPair FilterBySharedNodes(
            const tAdjPair &AC,
            const tAdjPair &Anodes, const tAdjPair &Cnodes,
            int minShared);

        // --- Higher-level operations (use mesh element topology) ---

        /// Interpolate: create intermediate entities from parent→node.
        /// Adds a new layer (fromDepth=parentDepth, toDepth=entityDepth)
        /// and a new layer (fromDepth=entityDepth, toDepth=0).
        void Interpolate(
            int parentDepth, int entityDepth,
            const tElemInfoArrayPair &parentElemInfo,
            const Periodicity *periodicInfo = nullptr);
    };
}
```

---

## 5. Implementation Phases

### Phase 0: Scaffold + Inverse (test-first)

**Goal:** Implement `MeshConnectivity` struct, `Inverse()`, and unit tests.

**Status:** Implemented and migrated. `RecoverNode2CellAndNode2Bnd()` now uses
DSL `Inverse` internally.

**Files:**
- `src/Geom/MeshConnectivity.hpp` — class definition
- `src/Geom/MeshConnectivity.cpp` — `Inverse()` implementation
- `test/cpp/Geom/test_MeshConnectivity.cpp` — unit tests

**Tests:**
- Serial inverse: small hand-crafted cell2node → verify node2cell
- MPI inverse: partition a small mesh, verify globally-complete node2cell
- Round-trip: `inverse(inverse(cell2node))` ⊇ cell2node (every cell appears in
  the re-inverted result, possibly with extra entries from other cells sharing
  the same nodes)

**Implementation notes:**
- The serial case is a simple histogram + scatter.
- The MPI case reuses the existing `GeneralCell2NodeToNode2Cell` pattern from
  `RecoverNode2CellAndNode2Bnd` but as a standalone function.
- Input/output are `tAdjPair` in `Adj_PointToGlobal` state (global indices).

### Phase 1: Compose + Filter

**Goal:** Implement `Compose()` and `FilterBySharedNodes()` with tests.

**Status:** Implemented and migrated. `RecoverCell2CellAndBnd2Cell()` now uses
DSL `ComposeFiltered` for `cell2cell` internally. The `bnd2cell` part retains
its periodic pbi filter logic.

**Tests:**
- Compose cell2node + node2cell → cell2cell (verify against known mesh)
- Compose with removeSelf → no diagonal entries
- FilterBySharedNodes: compose(bnd2node, node2cell) filtered by dim shared
  nodes → matches known bnd2cell
- Filter edge case: bnd with all nodes on one cell → 1 result

### Phase 2: Interpolate

**Goal:** Implement `Interpolate()` — the generalized face/edge generator.

**Status:** Implemented and migrated. `InterpolateFace()` now uses DSL internally.

**Implementation:** Standalone `MeshConnectivity::Interpolate` in
`MeshConnectivity_Interpolate.cpp`. Decoupled from `Element` module via
user-provided `SubEntityQuery` callbacks. Includes periodic-aware dedup via
optional `matchExtra` predicate.

**Tests:**
- 2D quads (4-cell), 2D tris (2-cell), 2D mixed (tri+quad)
- 3D tet (1-cell, 2-cell shared face), 3D hex (1-cell)
- Edge extraction from 3D tets
- 2×2 doubly-periodic quad mesh (corner case, requires collaborating check)
- Regression: DSL matches legacy `InterpolateFace` on all 5 mesh configs
- DSL-vs-Legacy periodic pbi comparison, recreate counts

---

## 9. Periodic Face Deduplication: The Collaborating Check

### 9.1. Problem

After periodic node deduplication (`Deduplicate1to1Periodic`), cells on
opposite sides of a periodic boundary share the same node **indices** but
carry different `NodePeriodicBits` (pbi) in `cell2nodePbi`. The pbi bits
record which periodic translations must be applied to recover each node's
physical coordinates as seen from a given cell.

When extracting faces (sub-entities) from cells, face deduplication compares
sorted vertex indices. On a doubly- or triply-periodic mesh, two distinct
physical faces can share **identical vertex index sets** — because periodic
node deduplication merged nodes at intersections of periodic boundaries
(corners in 2D, edges/corners in 3D).

**Example: 2×2 doubly-periodic quad mesh.**

After dedup, 9 original nodes collapse to 4. All 4 cells reference the same
4 nodes. Every cell has the same 4 edge vertex sets: `{0,1}`, `{1,3}`,
`{2,3}`, `{0,2}`. Without disambiguation, cell 0's edges would be falsely
merged with cell 3's edges (the diagonally opposite cell), producing a
sub-entity with 3+ parents — which is topologically impossible for a
manifold mesh.

### 9.2. Solution: Uniform XOR Check

Two faces with the same vertex indices are the same physical face **if and
only if** the periodic-bit difference between the two cells' views of the
face nodes is **uniform** across all node pairs.

Concretely, for candidate face match between cell A (sub-entity `iSub`) and
cell B (sub-entity `jSub`):

1. Extract `(nodeIndex, pbi)` pairs from both cells for their respective
   face nodes: `{(n_k, pbi_A_k)}` and `{(n_k, pbi_B_k)}`.

2. Sort both lists by `(nodeIndex, pbi)`.

3. Compute `v0 = pbi_A_0 XOR pbi_B_0`.

4. For all `k > 0`, check `pbi_A_k XOR pbi_B_k == v0`.

If the XOR is uniform, the faces are **collaborating** — they represent the
same physical face, possibly viewed through a single periodic translation
(`v0 != 0`) or from the same side (`v0 == 0`). If non-uniform, the faces are
on different periodic images and must remain distinct entities.

### 9.3. Implementation in the DSL

`MeshConnectivity::Interpolate` is topology-agnostic — it does not know about
periodic bits. Instead, the `SubEntityQuery` struct has an optional
`matchExtra` callback:

```cpp
std::function<bool(index iParent, int iSub,
                   index iCandEntity,
                   index candidateParent, int candidateSub)>
    matchExtra;
```

After a vertex-set match, if `matchExtra` is set, Interpolate calls it to
validate the match. The callback receives both the current sub-entity's
origin `(iParent, iSub)` and the candidate entity's creating origin
`(candidateParent, candidateSub)`, allowing the caller to extract pbi from
both and perform the uniform XOR check.

In `UnstructuredMesh::InterpolateFace`, the callback is wired up for
periodic meshes (`isPeriodic == true`):

```cpp
faceQuery.matchExtra = [this](index iParent, int iSub,
                               index, index candidateParent, int candidateSub) -> bool
{
    // Extract (nodeIndex, pbi) for both faces, sort, check uniform XOR
    // ... (see Mesh.cpp InterpolateFace implementation)
};
```

For non-periodic meshes, `matchExtra` is left unset (null), and Interpolate
performs vertex-only dedup — which is correct since non-periodic meshes never
have two distinct faces sharing the same vertex set.

### 9.4. Overflow Detection

If the collaborating check is omitted on a periodic mesh that requires it,
a face may be falsely matched to an entity that already has two parents.
Instead of aborting, `Interpolate` sets `result.duplicateOverflow = true`
and creates a new entity for the unmatched face. This allows tests to detect
the failure gracefully.

### 9.5. Test Coverage

| Test | What it proves |
|------|----------------|
| 2×2 periodic quad, no `matchExtra` | `duplicateOverflow == true` (false merge detected) |
| 2×2 periodic quad, with `matchExtra` | 8 faces, all internal, correct cell pairs, no self-connections |
| Pipeline collaborating check (IV10_10, IV10U_10) | Uniform XOR holds for all internal faces |
| Pipeline pbi consistency | `face2nodePbi` matches owner cell's `cell2nodePbi` |
| DSL-vs-Legacy pbi | Identical `(node, pbi)` sets per face |
| RecreatePeriodicNodes counts | DSL and Legacy produce identical recreated node counts |

### Phase 3: Ghost Build via Traversal Chains

**Goal:** Implement generic ghost building using explicit adjacency chains
compiled into a BFS tree with pull barriers between levels.

**Status:** Core types implemented and tested. BFS evaluation skeleton done.
Scratch pull orchestration deferred to Phase 4 integration.

**Files:**
- `src/Geom/MeshConnectivity.hpp` — `EntityKind`, `AdjKind`, `GhostChain`,
  `GhostSpec`, `CompiledGhostTree`, `GhostResult`
- `src/Geom/MeshConnectivity_Ghost.cpp` — compilation + evaluation

#### 3.1. EntityKind

```cpp
/// Logical entity roles. Depth depends on mesh dimension.
enum class EntityKind : int8_t
{
    Cell,   // depth = dim
    Face,   // depth = dim-1
    Edge,   // depth = 1 (== Face in 2D)
    Node,   // depth = 0
    Bnd,    // depth = dim-1 (separate storage, zone-labeled subset of faces)
};

/// Resolve EntityKind to topological depth.
/// In 2D, Edge == Face == dim-1 = 1.
inline int entityDepth(EntityKind kind, int dim);
```

#### 3.2. AdjKind — Named Adjacency Hops

Each hop names a specific adjacency relation in the DAG. `AdjKind` is a
struct with `(from, to, via)` fields rather than a flat enum — this gives a
compact, extensible representation without combinatorial explosion.

- **Direct adjacencies** (`from != to`): `via` is ignored. E.g.,
  `AdjKind(Cell, Node)` = cell2node cone. All direct cones and supports
  are allowed.
- **Intra-level adjacencies** (`from == to`): `via` specifies the
  intermediary. E.g., `AdjKind(Cell, Cell, Node)` = cell2cell via node,
  `AdjKind(Cell, Cell, Face)` = cell2cellFace.

```cpp
struct AdjKind
{
    EntityKind from, to, via;
    constexpr AdjKind(EntityKind from, EntityKind to);              // direct
    constexpr AdjKind(EntityKind from, EntityKind to, EntityKind via); // intra-level
    bool isIntraLevel() const { return from == to; }
    bool operator==(const AdjKind &o) const;  // via ignored for direct
};
```

Predefined constants in `namespace Adj`:

```cpp
// Direct cones/supports (all allowed in registry):
Adj::Cell2Node, Adj::Cell2Face, Adj::Cell2Edge,
Adj::Face2Node, Adj::Face2Edge, Adj::Edge2Node, Adj::Bnd2Node,
Adj::Node2Cell, Adj::Node2Face, Adj::Node2Edge, Adj::Node2Bnd,
Adj::Face2Cell, Adj::Edge2Face, Adj::Edge2Cell, Adj::Bnd2Cell,
// Intra-level (allowed in registry):
Adj::Cell2Cell, Adj::Cell2CellFace, Adj::Bnd2Bnd, Adj::Face2Face,
```

**Registry policy:** The adjacency registry (`MeshConnectivity::adjRegistry`)
stores only:
1. Direct cones/supports (inter-level): `Cell2Node`, `Node2Cell`, `Cell2Face`,
   `Face2Cell`, `Bnd2Node`, `Node2Bnd`, `Bnd2Cell`, etc.
2. Intra-level adjacencies via Node or Face: `Cell2Cell`, `Cell2CellFace`,
   `Bnd2Bnd`.

More complex composed adjacencies (multi-hop, filtered) are NOT stored in the
registry and cannot be used as ghost chain hops.

#### 3.3. GhostChain and GhostSpec

```cpp
struct GhostChain
{
    EntityKind anchor;           ///< Owned entities to start from.
    std::vector<AdjKind> hops;   ///< Sequence of adjacency lookups.
    EntityKind target;           ///< Must == hops.back().to.
};

struct GhostSpec
{
    std::vector<GhostChain> chains;
    static GhostSpec defaultPrimary();  ///< Current pipeline spec.
};
```

**Current pipeline expressed as GhostSpec:**
```cpp
GhostSpec defaultPrimary = {{
    {Cell, {Cell2Cell},            Cell},  // 1-ring cell neighbors
    {Cell, {Cell2Cell, Cell2Node}, Node},  // nodes of ghost cells
    {Bnd,  {Bnd2Node, Node2Bnd},  Bnd},   // 1-ring bnd neighbors
    {Bnd,  {Bnd2Node, Node2Bnd, Bnd2Node}, Node},  // nodes of ghost bnds
}};
```

#### 3.4. Compiled Ghost Tree

Chains are compiled into a forest of BFS-level-ordered trees. Chains sharing
common prefixes merge into shared trie paths to avoid redundant traversals.

```cpp
/// One node in the compiled ghost tree.
struct GhostTreeNode
{
    EntityKind kind;        ///< Entity kind at this tree node.
    AdjKind hop;            ///< Adjacency used to reach this node from parent.
                            ///< Undefined for root nodes.
    bool collect{false};    ///< If true, non-owned entities here become ghosts.
    int level{0};           ///< BFS depth (root = 0). Pull barriers between levels.
    std::vector<GhostTreeNode> children;
};

/// The compiled forest: one root per distinct anchor EntityKind.
/// Also stores validation results and the resolved AdjKind → tAdjPair mapping.
struct CompiledGhostTree
{
    std::vector<GhostTreeNode> roots;

    /// Compile chains into the tree. Validates:
    ///   - anchor == adjFrom(hops[0])
    ///   - target == adjTo(hops.back())
    ///   - consecutive hops: adjTo(hops[i]) == adjFrom(hops[i+1])
    ///   - no empty chains
    static CompiledGhostTree compile(const GhostSpec &spec);

    /// Pre-check that all required adjacencies exist in the DAG.
    /// Returns list of missing AdjKind values.
    std::vector<AdjKind> checkAvailable(const MeshConnectivity &dag) const;
};
```

**Example compiled tree for current pipeline:**
```
Root: Cell (level 0, no collect)
 └─[Cell2Cell]→ Cell (level 1, COLLECT)
     └─[Cell2Node]→ Node (level 2, COLLECT)

Root: Bnd (level 0, no collect)
 └─[Bnd2Node]→ Node (level 1, no collect)
     └─[Node2Bnd]→ Bnd (level 2, COLLECT)
         └─[Bnd2Node]→ Node (level 3, COLLECT)
```

Tip merging: `Node (level 2, COLLECT)` from the Cell subtree and
`Node (level 3, COLLECT)` from the Bnd subtree both contribute to
`ghostResult[Node]` via union.

#### 3.5. BFS Evaluation: Scratch Pulls + Definitive Pull

Evaluation has two phases: **BFS traversal** with conservative scratch pulls
to enable adjacency lookups on non-local entities, then a **definitive pull**
using the exact computed ghost set.

##### Phase A: BFS Traversal (scratch pulls)

The tree is evaluated level-by-level (BFS). At each level, all tree nodes at
that level traverse their hop adjacency to produce an entity set. After each
level, the evaluator checks which entity kinds have sets containing non-local
entries AND have children in the tree. For those kinds, a **scratch pull** is
performed: a temporary ghost mapping is built and adjacency data is pulled so
that the next level can traverse from those ghost entities.

Scratch pulls are conservative — they may pull data for entities that no
COLLECT node ultimately needs. This is acceptable because they are temporary;
the definitive pull replaces them.

```
Level 0: Initialize index sets from owned entities.
         Cell-root: ownedCells.   Bnd-root: ownedBnds.

Level 1: Traverse hops from level 0.
         Cell→[Cell2Cell]→ cells.   Bnd→[Bnd2Node]→ nodes.
         COLLECT cells.
         Scratch-pull cell data (Cell has children, set grew).

Level 2: Traverse hops from level 1.
         Cell→[Cell2Node]→ nodes (using owned+ghost cells).
         Node→[Node2Bnd]→ bnds.
         COLLECT nodes (partial), COLLECT bnds.
         Scratch-pull bnd data (Bnd has children, set grew).

Level 3: Traverse hops from level 2.
         Bnd→[Bnd2Node]→ nodes (using owned+ghost bnds).
         COLLECT nodes (more).
         No more levels — done.
```

##### Phase B: Definitive Pull (final ghost assembly)

After BFS completes, the final ghost set per EntityKind is the **union** of
all COLLECT nodes' non-owned entities for that kind:

```cpp
for each EntityKind with collected ghosts:
    ghostIndices[kind] = sorted, deduplicated union of all COLLECT results
```

Then, for each entity kind with ghost indices, build the **definitive** ghost
mapping and pull all associated arrays:

1. `createFatherGlobalMapping()` (if not already done)
2. `createGhostMapping(ghostIndices[kind])` — the exact, minimal set
3. `createMPITypes()` + `pullOnce()` on the primary array
4. `BorrowAndPull()` on all secondary arrays sharing the same ghost layout

This replaces any scratch pull state. The definitive pull is what persists
in the `ArrayTransformer` for later use (persistent pull/push in solvers).

**Scratch pull optimization:** During BFS, scratch pulls only need the
adjacency arrays required by subsequent hops — NOT all arrays for that
entity kind. The compiled tree knows which `AdjKind` hops follow from each
kind, so it can determine the minimal scratch set. E.g., if the only hop
from `Cell` at level 2 is `Cell2Node`, the scratch pull for `Cell` at
level 1 only needs `cell2node` ghost data — not `cellElemInfo`, not
`cell2nodePbi`. This keeps scratch pulls lightweight.

**Pull groups:** All arrays sharing the same entity kind are pulled together.
The caller provides a registry mapping EntityKind to the list of ArrayPairs:

```cpp
/// Registry of arrays to ghost-pull per entity kind.
/// The first array in each list is the "primary" (owns the ghost mapping);
/// remaining arrays BorrowAndPull from it.
using PullRegistry = std::unordered_map<EntityKind, std::vector<ArrayPairBase*>>;
```

E.g., for entity kind `Cell`:
- Primary: `cell2cell` (full 4-step setup)
- Secondary: `cell2node`, `cellElemInfo`, `cell2nodePbi`, `cell2cellOrig`

For `Node`:
- Primary: `coords`
- Secondary: `node2nodeOrig`

For `Bnd`:
- Primary: `bnd2cell`
- Secondary: `bnd2node`, `bndElemInfo`, `bnd2nodePbi`, `bnd2bndOrig`

#### 3.6. GhostResult

```cpp
struct GhostResult
{
    /// Per EntityKind: sorted, deduplicated global indices to ghost.
    std::unordered_map<EntityKind, std::vector<index>> ghostIndices;

    /// Whether any ghost was collected for a given kind.
    bool hasGhosts(EntityKind kind) const;
};
```

#### 3.7. Ownership Transfer (ssp<> swap)

`MeshConnectivity` does NOT need full `ArrayPair` move semantics.
Ownership transfer between `UnstructuredMesh` legacy members and the DAG
uses `ssp<>` (shared_ptr) swap on father/son pointers, which is O(1):

```cpp
std::swap(mesh.cell2node.father, dag_slot_father);
std::swap(mesh.cell2node.son,    dag_slot_son);
```

`ArrayTransformer` stays on the `ArrayPair` in `UnstructuredMesh` and is
(re)initialized during ghost building. Full move semantics on `Array`,
`ParArray`, `ArrayTransformer`, `ArrayPair` is deferred to a separate effort
requiring DNDS-level unit test coverage.

#### 3.8. Adjacency Registry and Resolution

`MeshConnectivity` holds a restricted registry mapping `AdjKind` to `tAdjPair*`:

```cpp
std::unordered_map<AdjKind, tAdjPair*, AdjKindHash> adjRegistry;

void registerAdj(AdjKind kind, tAdjPair &pair);
tAdjPair *resolveAdj(AdjKind kind);
bool hasAdj(AdjKind kind) const;
```

The registry stores raw pointers to adjacency pairs owned elsewhere (typically
by `UnstructuredMesh`). The caller is responsible for ensuring pointer validity.

During `evaluateGhostTree`, each hop in the BFS tree is resolved via
`resolveAdj`. If any hop is unresolved, the evaluator aborts with a
diagnostic message listing the missing adjacencies.

#### 3.10. Flat List Formulation and Correctness Proof

The forest of ghost chains can be unified into a single **flat state-vector
propagation** that is mathematically equivalent (for non-cross-contaminating
specs) and simpler to implement efficiently.

##### Definitions

**State vector.** $\mathbf{S}^L$ is a vector indexed by `EntityKind`, where
$\mathbf{S}^L(K)$ is the set of global indices of kind $K$ at BFS level $L$.

**Initialization (level 0):**
$$\mathbf{S}^0(K) = \begin{cases}
\text{Owned}(K) & \text{if } K \text{ is an anchor of any chain} \\
\emptyset & \text{otherwise}
\end{cases}$$

**Transfers.** Each chain $C_k$ with hops $h_k^1, \ldots, h_k^{n_k}$ generates
transfers $(h_k^j, \text{level} = j)$ for $j = 1, \ldots, n_k$. Let $T_L$
be all transfers at level $L$.

**Update rule:**
$$\mathbf{S}^L(K) = \mathbf{S}^{L-1}(K) \;\cup\; \bigcup_{\substack{(h, L) \in T_L \\ \text{to}(h) = K}} h\bigl(\mathbf{S}^{L-1}(\text{from}(h))\bigr)$$

where $h(S) = \bigcup_{e \in S} h[e]$ (row lookup and union).

**Ghost result:**
$$G_{\text{flat}}(K) = \mathbf{S}^{L_{\max}}(K) \setminus \text{Owned}(K)$$

##### Monotonicity

**Lemma 1.** $\mathbf{S}^L(K) \subseteq \mathbf{S}^{L+1}(K)$ for all $K$, $L$.

*Proof.* The update rule takes a union with $\mathbf{S}^{L-1}(K)$, so the
state can only grow. $\square$

**Lemma 2.** If $A \subseteq B$ then $h(A) \subseteq h(B)$.

*Proof.* $h(A) = \bigcup_{e \in A} h[e] \subseteq \bigcup_{e \in B} h[e] = h(B)$. $\square$

##### Correctness Theorem

**Theorem.** $G_{\text{flat}}(K) \supseteq G_{\text{forest}}(K)$ for all
collected kinds $K$. Equality holds when the spec has no cross-chain
contamination (defined below).

*Proof of $\supseteq$.*

For chain $C_k$ with hops $h_k^1, \ldots, h_k^{n_k}$, anchor $a_k$, target
$t_k = K$, define the chain's intermediate sets:

$$P_k^0 = \text{Owned}(a_k), \qquad P_k^j = h_k^j(P_k^{j-1})$$

The chain result is $R_k = P_k^{n_k}$.

*Claim:* $P_k^j \subseteq \mathbf{S}^j(\text{to}(h_k^j))$ for all $j$.

Induction on $j$:
- Base $j = 0$: $P_k^0 = \text{Owned}(a_k) = \mathbf{S}^0(a_k)$. $\checkmark$
- Step: Assume $P_k^{j-1} \subseteq \mathbf{S}^{j-1}(\text{from}(h_k^j))$.
  Transfer $(h_k^j, j) \in T_j$ and $\text{to}(h_k^j) = K_j$. By the update rule:
  $$\mathbf{S}^j(K_j) \supseteq h_k^j(\mathbf{S}^{j-1}(\text{from}(h_k^j))) \supseteq h_k^j(P_k^{j-1}) = P_k^j$$
  using Lemma 2 and the inductive hypothesis. $\checkmark$

Therefore $R_k \subseteq \mathbf{S}^{n_k}(K) \subseteq \mathbf{S}^{L_{\max}}(K)$
(by Lemma 1). Taking the union over all chains targeting $K$ and subtracting
owned: $G_{\text{forest}}(K) \subseteq G_{\text{flat}}(K)$. $\square$

*Proof of $\subseteq$ (when no cross-chain contamination).*

**Cross-chain contamination** occurs when a transfer $(h, L)$ from chain $B$
reads $\mathbf{S}^{L-1}(\text{from}(h))$ which contains entities injected by
a different chain $A$ at an earlier level, and the resulting entities are not
reachable by any single chain.

Formally, an entity $e \in \mathbf{S}^{L_{\max}}(K) \setminus \text{Owned}(K)$
is **cross-contaminated** if every reachability path from owned entities to $e$
passes through transfers from two or more distinct chains in an order that
does not correspond to any single chain's hop sequence.

When no cross-contamination exists, every entity in $\mathbf{S}^{L_{\max}}(K)$
is either owned or reachable through a single chain's hop sequence, hence
belongs to some $R_k$. Therefore $G_{\text{flat}}(K) \subseteq G_{\text{forest}}(K)$.

**Sufficient condition for no cross-contamination:** For every pair of
transfers $(h_A, L)$ from chain $A$ and $(h_B, L')$ from chain $B$ with
$L' > L$ and $\text{from}(h_B) = \text{to}(h_A)$, there exists a chain $C$
whose hop sequence includes both $h_A$ at position $L$ and $h_B$ at position
$L'$ (i.e., the cross-chain path is also a valid single-chain path).

##### Verification for Default Spec

The default spec chains:
```
C1: Cell →[Cell2Cell]→ Cell                  (Cell2Cell@1)
C2: Cell →[Cell2Cell]→[Cell2Node]→ Node      (Cell2Cell@1, Cell2Node@2)
C3: Bnd  →[Bnd2Node]→[Node2Bnd]→ Bnd        (Bnd2Node@1, Node2Bnd@2)
C4: Bnd  →[Bnd2Node]→[Node2Bnd]→[Bnd2Node]→ Node  (Bnd2Node@1, Node2Bnd@2, Bnd2Node@3)
```

Transfers by level:
- Level 1: Cell2Cell(Cell→Cell), Bnd2Node(Bnd→Node)
- Level 2: Cell2Node(Cell→Node), Node2Bnd(Node→Bnd)
- Level 3: Bnd2Node(Bnd→Node)

Check cross-contamination pairs:
- Cell2Cell@1 produces Cell. Cell2Node@2 reads Cell. Both from C2. $\checkmark$
- Bnd2Node@1 produces Node. Node2Bnd@2 reads Node. Both from C3/C4. $\checkmark$
- Node2Bnd@2 produces Bnd. Bnd2Node@3 reads Bnd. Both from C4. $\checkmark$
- Cell2Cell@1 produces Cell. No transfer reads Cell at level 2+ except
  Cell2Node@2 which is from C2 (includes Cell2Cell@1). $\checkmark$
- Bnd2Node@1 produces Node. Node2Bnd@2 reads Node — from C3/C4. No
  cross-chain mixing. $\checkmark$
- Cell2Node@2 produces Node. No transfer reads Node at level 3+. $\checkmark$

No cross-contamination. $G_{\text{flat}} = G_{\text{forest}}$ exactly. $\square$

##### Note on Safe Superset

Even when cross-contamination exists, $G_{\text{flat}} \supseteq G_{\text{forest}}$.
Extra ghost entities are safe — they waste some memory but do not cause
incorrect results. The definitive pull uses the exact computed set. For ghost
building, this superset property is acceptable and often desirable (simpler
implementation, no need to track per-chain provenance).

#### 3.11. Hybrid Evaluation: Per-Node Traversal with Union Pulls

The flat formulation (Section 3.10) is efficient for MPI pulls (one pull per
EntityKind per level) but may produce a strict superset of the forest result
due to cross-chain contamination. The hybrid approach combines the best of
both: exact per-tree-node traversal with efficient union-based pulls.

##### Algorithm

**Data structures.** All indices are global throughout.

- Per-tree-node set: `S[n]` — sorted vector of global indices for tree node `n`.
- Pull union: `Pull[K]` — sorted vector of global indices for EntityKind `K`,
  computed as the union of all tree nodes' sets of that kind at the current level.
  Used solely for ghost communication (scratch pull).

**Evaluation.**

```
Level 0 (initialization):
  For each root r: S[r] = Owned(r.kind)

For level L = 0 to maxLevel:

  Step 1 — Collect:
    For each tree node n at level L with n.collect = true:
      ghostIndices[n.kind] ∪= S[n] \ Owned(n.kind)

  Step 2 — Compute pull unions:
    For each EntityKind K:
      Pull[K] = ∪{ S[n] : n.kind = K, n.level = L, n has children }
    Only compute for kinds that have children at this level.

  Step 3 — Scratch pull (MPI):
    For each kind K where Pull[K] contains non-local indices:
      Build ghost mapping from Pull[K] \ Owned(K)
      Pull adjacency data for all registered arrays of kind K
    After this step, every index in Pull[K] has locally-available
    adjacency data (father or son rows).

  Step 4 — Traverse:
    For each child c of a level-L node n:
      S[c] = c.hop.apply(S[n])
    Where apply(S) iterates each global index in S, looks up its row
    in the adjacency (father or son), and collects all entries.
    DNDS_assert: every index in S[n] must be locally resolvable
    (father range or ghost mapping). Failure = pull was incomplete.

Final:
  Deduplicate and sort ghostIndices[K] for each K.
  MPI_Allreduce bitmask for activeKinds (collective hasGhosts).
```

##### Correctness

**Claim:** $G_{\text{hybrid}}(K) = G_{\text{forest}}(K)$ exactly.

*Proof.* Each tree node $n$ at level $L$ with parent $p$ computes
$S[n] = n.hop(S[p])$, using only the parent's per-node set $S[p]$, not the
full union $\text{Pull}[\text{from}(n.hop)]$. This is identical to the forest
formulation. The pull union is only used for MPI communication — it ensures
that the adjacency data is available for $S[p]$, since $S[p] \subseteq
\text{Pull}[p.kind]$ (by construction of Pull as a union containing $S[p]$).
Therefore every index in $S[p]$ is resolvable after the scratch pull.

No cross-chain contamination occurs because each tree node's traversal is
driven by its own per-node set, which was produced by its own chain's hop
sequence. The pull union is a superset used only for communication, not for
traversal.

The COLLECT step reads from per-node sets, producing exactly the forest's
ghost sets. Union across COLLECT nodes of the same kind gives the same
result as the forest's union of chain results. $\square$

##### Pull Efficiency

The pull union $\text{Pull}[K]$ may contain more entities than any single
tree node needs. This means the scratch pull fetches some adjacency data that
no tree node will actually traverse. The overhead is:

- Extra MPI data transferred: proportional to the difference between the
  pull union and the largest per-node set of that kind.
- Extra local memory: the ghost (son) arrays are sized for the pull union.

In practice, for the default spec, the pull unions match the per-node sets
(no overlap between tree nodes of the same kind at the same level), so there
is zero overhead.

##### Ghost-Not-Found Assertion

After a scratch pull at level $L$, every global index in every per-node set
$S[n]$ at level $L$ (where $n$ has children) must be locally resolvable in
the adjacency used by the child's hop. If an index is not found in the
father range and not in the ghost mapping, this indicates a bug in the pull
union computation or the MPI communication. The traversal asserts this with
`DNDS_assert` rather than silently skipping.

The only exception is level 0 (before any pull), where the per-node sets
are owned entities — always locally available by definition.

##### Global Index Convention

All sets, adjacency lookups, and ghost results operate on **global indices**
throughout the evaluation. The adjacency arrays are in `Adj_PointToGlobal`
state. Local-to-global and global-to-local conversions happen only at the
boundary between the evaluator and the `ArrayTransformer` (for pull setup
and for resolving row indices in the adjacency arrays).

#### 3.12. Tests (implemented)
|------|----------------|
| `defaultPrimary compiles` | Tree structure: 2 roots, correct levels, COLLECT flags, prefix merging |
| `prefix merging` | Two chains `{Cell2Cell}` and `{Cell2Cell,Cell2Cell}` share prefix |
| `invalid chain detection` | Empty hops, anchor mismatch, target mismatch, consecutive mismatch |
| `checkAvailable` | Missing adjacencies correctly identified |
| `dump` | Diagnostic output contains expected entity names |
| `face ghost chain` | `Cell→Cell2Face→Face` compiles correctly |
| `AdjKind equality and hash` | Direct ignores via; intra-level uses via |
| `entityDepth` | Correct depths for 2D and 3D, Edge==Face in 2D |
| `adjKindName` | Formatted names: "Cell2Node", "Cell2Cell(Node)" |
| Evaluate on partitioned mesh | TODO (Phase 4 integration) |

### Phase 4: Integration with UnstructuredMesh

**Goal:** Wire `MeshConnectivity` into `UnstructuredMesh` as the internal
organizer. Legacy members transfer ownership via `ssp<>` pointer swap.

**Approach:**
- `UnstructuredMesh` gains a `MeshConnectivity dag` member.
- `AdoptIntoDAG()` swaps `ssp<>` father/son pointers from legacy `ArrayPair`
  members into the DAG's cone/support slots. The legacy `ArrayPair::trans`
  stays in place; the `father`/`son` shared_ptrs become null on the legacy side.
- `ReturnFromDAG()` swaps back: DAG slots → legacy members.
- Ghost building uses `GhostSpec` + compiled tree to orchestrate pulls.
  After evaluation, the legacy `ArrayPair::trans` is (re)initialized with
  the ghost mapping produced by the tree.
- `convertAllToLocal()` / `convertAllToGlobal()` iterate DAG cone/support
  vectors instead of hardcoded member lists.

**Tests:**
- Full pipeline test: build mesh through DAG path, compare all adjacency
  arrays with the current pipeline's output (bit-for-bit match).
- Round-trip: adopt → convertToGlobal → convertToLocal → verify unchanged.

### Phase 5: Replace Legacy Build Methods

**Goal:** Rewrite `RecoverNode2CellAndNode2Bnd`, `RecoverCell2CellAndBnd2Cell`,
`InterpolateFace` as DSL operation sequences.

**Approach:** The legacy methods become thin wrappers that call DSL operations:

```cpp
void UnstructuredMesh::RecoverNode2CellAndNode2Bnd()
{
    auto *layer = dag.findLayer(dim, 0);
    layer->support = MeshConnectivity::Inverse(layer->cone, nNodeGlobal, mpi);
    // (similarly for bnd2node → node2bnd)
}
```

**Tests:** Existing `test_MeshPipeline.cpp` must pass unchanged.

---

## 6. Boundary Entities in the Framework

Boundaries are NOT a separate entity stratum. A boundary is a codim-1 face
with a non-internal zone label. In the current code, `bnd2node` is a separate
array; in the DAG, it's simply a subset of `face2node` selected by zone ID.

However, for backward compatibility, the DAG can store `bnd2node` as a
separate `InterLayerAdj` with `fromDepth = dim - 1` (boundary faces) and
`toDepth = 0` (nodes). The "boundary" stratum overlaps with the face stratum
in entity depth but represents a subset.

`bnd2cell` is derived:
```
bnd2cell = filterBySharedNodes(
    compose(bnd2node, node2cell),
    bnd2node, cell2node,
    dim)    // face-connected: shared nodes >= dim
```

Or more directly: after face interpolation, each boundary face has a face ID,
and `face2cell` (the support of the cell→face cone) directly gives the 1-2
cells. So `bnd2cell[iBnd] = face2cell[bnd2face[iBnd]]`.

---

## 7. Testing Strategy

All tests are MPI-aware (np=1, 2, 4) using doctest + mpirun.

| Phase | Test File | Key Assertions |
|-------|-----------|----------------|
| 0 | `test_MeshConnectivity.cpp` | Inverse correctness, MPI completeness |
| 1 | same | Compose correctness, filter correctness |
| 2 | same | Interpolation face count, cell2face consistency |
| 3 | same | Ghost set matches known results |
| 4 | `test_MeshPipeline.cpp` (existing) | Full pipeline regression |

Small test meshes (hand-crafted 4-cell, 9-cell quads) for Phases 0-3.
Existing CGNS meshes (UniformSquare_10, IV10_10, NACA0012_H2) for Phase 4.

---

## 8. File Layout

```
src/Geom/
  MeshConnectivity.hpp      — class definition, InterLayerAdj, GhostRequirement
  MeshConnectivity.cpp      — Inverse, Compose, FilterBySharedNodes
  MeshConnectivity_Interpolate.cpp — Interpolate (extracted from Mesh.cpp)
  MeshConnectivity_Ghost.cpp      — BuildGhost with traversal chains

test/cpp/Geom/
  test_MeshConnectivity.cpp — standalone unit tests (Phases 0-3)
```
