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

**Status:** Fully implemented. Three tiers:

- `InterpolateLocal` (rank-only, no MPI): extracts and deduplicates sub-entities
  from parent→node. Used directly for local analysis/testing and as Step 1 of
  InterpolateGlobal.
- `InterpolateDistributed` (legacy, 2-parent only): extends InterpolateLocal with
  push-based ghost exchange. Superseded by InterpolateGlobal. Retained for backward
  compatibility.
- `InterpolateGlobal` (production): distributed sub-entity creation with global
  dedup, N-parent edges, pbi extraction, and `parent2entityPbi` output. Used by
  `InterpolateFace()` in `Mesh.cpp`.

**Key design decisions:**
- `entity2parent` is variable-width `tAdjPair` (not fixed-2), supporting N-parent edges.
- `entity2nodePbi` stores pbi from the first-discovered parent's perspective.
- `parent2entityPbi` stores the uniform XOR between each parent's view and the entity's
  stored pbi. Computed locally (no push needed). Faces: at most 1 bit. Edges: multi-bit.
- Ghost B entities are NOT produced. The caller pulls them via `evaluateGhostTree`.
- The push protocol uses `(globalA, globalB, subIdx)` triplets — `subIdx` disambiguates
  which slot of a parent with multiple non-owned sub-entities receives which global B ID.

**Tests:**
- 2D quads (4-cell), 2D tris (2-cell), 2D mixed (tri+quad)
- 3D tet (1-cell, 2-cell shared face), 3D hex (1-cell)
- Edge extraction from 3D tets (single, two-tet shared)
- 2×2 doubly-periodic quad mesh (corner case, requires collaborating check)
- 2×2×2 triply-periodic hex mesh (faces: 24, all 2-parent; edges: 24, all 4-parent)
- Regression: DSL matches legacy `InterpolateFace` on UniformSquare_10
- Distributed InterpolateGlobal on 4×4×4 hex: non-periodic + X-periodic (faces + edges)
- Distributed InterpolateGlobal on 4×4×4 triply-periodic hex (faces + edges)
- entity2nodePbi value verification (faces + edges, triply-periodic)
- parent2entityPbi value verification (faces + edges, triply-periodic)
- Coordinate-based center verification: face/edge center via entity2nodePbi + parent2entityPbi
  matches cell's direct computation (triply-periodic, np=2,4,8)
- parent2entityPbi 1-bit check for faces (triply-periodic)

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

### 9.4. N-Parent Entity Handling (Variable-Width entity2parent)

With variable-width `entity2parent` (`tAdjPair`), there is no overflow.
If the collaborating check is omitted on a periodic mesh, false merges
produce entities with >2 parents (faces) or >4 parents (edges). Tests
detect this by checking `entity2parent.RowSize(i)` bounds.

In 2D doubly-periodic: without `matchExtra`, at least one face gets 3+ parents.
In 3D triply-periodic: without `matchExtra`, at least one face/edge gets extra parents.
With `matchExtra`: faces have exactly 1-2 parents, edges have exactly 1-4 parents.

### 9.5. parent2entityPbi: Relative Periodic Transform

`entity2nodePbi` is stored from the first-discovered parent's perspective (the
entity's own frame). When parent cell A accesses entity B, A needs to know the
relative periodic transform between its own frame and B's stored frame.

`parent2entityPbi[iParent][iSub]` is a single `NodePeriodicBits` value — the
**uniform XOR** between parent A's sub-entity node-pbi and entity B's stored
`entity2nodePbi`, matched by node identity (not position).

**Why match by node identity:** Different parents may enumerate the same face/edge
nodes in different local orderings. The per-position XOR can be non-uniform even
when the per-node XOR is uniform. Sorting `(node, pbi)` pairs by node index before
XORing handles this correctly.

**Properties:**
- For the first parent (B's stored perspective): always `{0}`.
- For faces: at most 1 bit (P1, P2, or P3). A face crosses at most one periodic boundary.
- For edges: can be multi-bit (e.g., P1|P2 for a corner edge crossing two periodic boundaries).
- Computed locally in Step 2b of InterpolateGlobal — no MPI push needed. The XOR depends
  only on `cell2nodePbi` and `entity2nodePbi`, both available after InterpolateLocal.

**Usage:** To get entity B's node coordinates in parent A's frame:
```cpp
NodePeriodicBits relPbi = parent2entityPbi(iCell, iSub);
for (int k = 0; k < nFaceNodes; k++)
{
    NodePeriodicBits entPbi = entity2nodePbi(iFace, k);
    // XOR relPbi to transform from entity frame to cell frame:
    NodePeriodicBits cellPbi{uint8_t(uint8_t(entPbi) ^ uint8_t(relPbi))};
    coord = periodicInfo.GetCoordByBits(coords[entity2node(iFace, k)], cellPbi);
}
```

### 9.6. Test Coverage

| Test | What it proves |
|------|----------------|
| 2×2 periodic quad, no `matchExtra` | >2-parent entity detected (false merge) |
| 2×2 periodic quad, with `matchExtra` | 8 faces, all internal, correct cell pairs, no self-connections |
| 2×2×2 triply-periodic hex, faces | 24 Quad4, all 2-parent, 3 distinct face-neighbors per cell |
| 2×2×2 triply-periodic hex, edges | 24 Line2, all 4-parent, 12 edges per cell |
| Distributed 4×4×4 hex, non-periodic | Exact face/edge counts, boundary/internal split |
| Distributed 4×4×4 hex, X-periodic | No X-boundary faces, adjusted counts |
| Distributed 4×4×4 hex, triply-periodic | 3*np*N^3 faces/edges, all internal |
| entity2nodePbi value verification | Stored pbi matches first-parent extraction (faces + edges) |
| parent2entityPbi value verification | Uniform XOR matches cell-to-entity pbi difference |
| parent2entityPbi 1-bit for faces | At most 1 bit set per face slot |
| Coordinate center verification | Face/edge center via entity2nodePbi + parent2entityPbi matches cell's direct computation |
| Regression: InterpolateFace | DSL matches legacy on UniformSquare_10 |

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

**Goal:** Wire `MeshConnectivity` DSL operations into `UnstructuredMesh`.

**Status:** Implemented. All ghost operations migrated to `evaluateGhostTree`.
`InterpolateFace` migrated to `InterpolateGlobal` + pull-based ghost faces.

**What was done:**
- `BuildGhostPrimary`: node ghosts via `evaluateGhostTree(Cell→Cell2Cell→Cell2Node→Node)`,
  bnd ghosts via `evaluateGhostTree(Node→Node2Bnd→Bnd)`.
- `RecoverCell2CellAndBnd2Cell`: N2CB ghost pull via
  `evaluateGhostTree(Cell→Cell2Node→Node ∪ Bnd→Bnd2Node→Node)`.
- `ReadSerializeAndDistribute`: ghost cell collection via
  `evaluateGhostTree(Cell→Cell2Cell→Cell)`.
- `InterpolateFace`: uses `InterpolateGlobal` + pull-based ghost faces via
  `createGhostMapping`, not push protocol inside the DSL.

**Not done (kept as manual code):**
- `BuildCell2CellFace`: 9 lines of local code, uses `tAdj2Pair` not supported by
  `ComposeFiltered`.
- Serial-path adjacency ops (`BuildCell2Cell`, `BuildBnd2CellSerial`): rank-0 only.
- `ReadSerializeAndDistribute` facial filtering: needs vertex-only counting.
- `InterpolateGlobal` for edges in production: API and tests ready, not yet called
  by the mesh pipeline (no edge adjacency in current solver).

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

All tests are MPI-aware (np=1, 2, 4, 8) using doctest + mpirun.

| Phase | Test File | Key Assertions |
|-------|-----------|----------------|
| 0 | `test_MeshConnectivity.cpp` | Inverse correctness, MPI completeness |
| 1 | same | Compose correctness, filter correctness, periodic pbi filter |
| 2 | `test_MeshConnectivity_Interpolate.cpp` | Face/edge counts, dedup, periodic collab check, distributed InterpolateGlobal, pbi value verification, coordinate center verification |
| 3 | `test_MeshConnectivity_Ghost.cpp` | Ghost set matches known results, chain merging, performance benchmarks |
| 4 | `test_MeshPipeline.cpp` (existing) | Full pipeline regression |

Small test meshes (hand-crafted 4-cell quads, periodic 2×2, 2×2×2) for local tests.
Distributed NxNxN hex meshes (N=4) for InterpolateGlobal tests.
Existing CGNS meshes (UniformSquare_10) for regression.

Shared builders in `SyntheticMeshBuilders.hpp`:
`HandCraftedMesh`, `Periodic2x2Mesh`, `Periodic2x2x2Mesh`, `DistributedHex3D`,
`makeFaceQuery`, `makeEdgeQuery`, `makePeriodicMatchExtra`,
`makeHex8FaceQueryPbi`, `makeHex8EdgeQueryPbi`.

---

## 8. File Layout

```
src/Geom/
  MeshConnectivity.hpp              — types, result structs, method declarations
  MeshConnectivity.cpp              — Inverse, Compose, ComposeFiltered, registry
  MeshConnectivity_Interpolate.cpp  — InterpolateLocal, InterpolateDistributed, InterpolateGlobal
  MeshConnectivity_Ghost.cpp        — GhostChain, CompiledGhostTree, evaluateGhostTree

test/cpp/Geom/
  SyntheticMeshBuilders.hpp         — shared mesh builders (4-quad, periodic 2x2/2x2x2, DistributedHex3D)
  test_MeshConnectivity.cpp         — Inverse, Compose, management tests (11 tests)
  test_MeshConnectivity_Ghost.cpp   — ghost chain and evaluateGhostTree tests (16 tests)
  test_MeshConnectivity_Interpolate.cpp — InterpolateLocal + InterpolateGlobal tests (17 tests)
```

---

## 9. Templatization of MeshConnectivity (Row-Size Parametric)

**Date:** 2026-04-23
**Goal:** Make every adjacency in MeshConnectivity capable of using any
`ArrayAdjacencyPair<rs>` directly, instead of forcing `tAdjPair` (NonUniformSize)
everywhere. This enables fixed-width adjacencies (face2cell=2, bnd2face=1, etc.)
to flow through the DSL without conversion, and supports custom mesh topologies.

### 9.1. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| DSL method dispatch | Function templates (`template<int rs_...>`) | Compile-time specialization; `if constexpr` for ResizeRow differences |
| Result structs | Per-field template params | `InterpolateResult<p2e_rs, e2n_rs, e2p_rs>` lets callers choose output width |
| ConeAdj/SupportAdj storage | `ssp<AdjVariant>` shared ownership | Both DAG and legacy mesh members share the same allocation |
| adjRegistry | Owns `ssp<AdjVariant>`, no raw pointers | Safe shared ownership; `std::visit` dispatches at ghost traversal time |
| Backward compat | Default template params = NonUniformSize | Existing callers compile unchanged |

### 9.2. Key Type Changes

**Before:**
```cpp
using AdjVariant = std::variant<tAdjPair, tAdj1Pair, tAdj2Pair, ...>;

struct ConeAdj {
    AdjVariant adj;     // by-value, not shared
    tPbiPair pbi;
    // ...
};

struct MeshConnectivity {
    std::unordered_map<AdjKind, tAdjPair*, AdjKindHash> adjRegistry;
    // ...
};
```

**After:**
```cpp
// AdjVariant unchanged (std::variant of all pair widths)

struct ConeAdj {
    ssp<AdjVariant> adj;   // shared ownership
    tPbiPair pbi;
    // Typed accessor: get<tAdj2Pair>() etc.
    template<class TPair> TPair& as() { return std::get<TPair>(*adj); }
    // ...
};

struct MeshConnectivity {
    // Registry owns shared references to the AdjVariant stored in ConeAdj/SupportAdj.
    // No raw pointers. Ghost traversal dispatches via std::visit.
    std::unordered_map<AdjKind, ssp<AdjVariant>, AdjKindHash> adjRegistry;
    // ...
};
```

### 9.3. DSL Method Templates

All DSL methods become function templates parameterized on input/output row sizes.
Default template args = NonUniformSize for backward compatibility.

```cpp
// Inverse: input cone rs → output support (always NonUniformSize, since
// the inverse of a fixed-width cone is variable-width)
template<int cone_rs = NonUniformSize>
static tAdjPair Inverse(
    const ArrayAdjacencyPair<cone_rs>& cone,
    index nToLocal,
    const MPIInfo& mpi,
    ...);

// ComposeFiltered: AB and BC can have different rs.
// Output is always NonUniformSize (composed adjacency has unpredictable width).
template<int rs_AB = NonUniformSize, int rs_BC = NonUniformSize, class Predicate>
static tAdjPair ComposeFiltered(
    const ArrayAdjacencyPair<rs_AB>& AB,
    const ArrayAdjacencyPair<rs_BC>& BC,
    index nALocal,
    ...);

// InterpolateLocal: parent2node can be any rs.
// Outputs have caller-chosen rs (defaulting to NonUniformSize).
template<int p2n_rs = NonUniformSize,
         int p2e_rs = NonUniformSize,
         int e2n_rs = NonUniformSize,
         int e2p_rs = NonUniformSize>
static InterpolateResult<p2e_rs, e2n_rs, e2p_rs> Interpolate(
    const ArrayAdjacencyPair<p2n_rs>& parent2node,
    const SubEntityQuery& query,
    index nParent,
    index nNode,
    const MPIInfo& mpi);
```

### 9.4. Result Struct Templates

```cpp
template<int p2e_rs = NonUniformSize,
         int e2n_rs = NonUniformSize,
         int e2p_rs = NonUniformSize>
struct InterpolateResult {
    ArrayAdjacencyPair<p2e_rs> parent2entity;
    ArrayAdjacencyPair<e2n_rs> entity2node;
    ArrayAdjacencyPair<e2p_rs> entity2parent;
    std::vector<ElemInfo> entityElemInfo;
    std::vector<std::vector<NodePeriodicBits>> parent2entityPbi;
    index nEntities{0};
};

// Similar for InterpolateGlobalResult, InterpolateDistributedResult
```

### 9.5. if-constexpr Differences

For fixed-width arrays (`rs > 0`), `ResizeRow` is not supported (the width
is compile-time fixed). The DSL methods must use `if constexpr` to skip
`ResizeRow` calls when the output is fixed-width (the rows are already
the correct size after `Resize(n)`).

```cpp
// In Interpolate, when building parent2entity:
if constexpr (p2e_rs == NonUniformSize)
    result.parent2entity.father->ResizeRow(iParent, nSubs);
else
    DNDS_assert(p2e_rs == nSubs); // compile-time width must match

// In Inverse, output is always NonUniformSize (variable fan-out), but
// input iteration over cone rows uses:
if constexpr (cone_rs == NonUniformSize)
    for (auto iTo : cone.father->operator[](iFrom)) ...
else
    for (rowsize j = 0; j < cone_rs; j++) ...
// (Actually both cases work via operator[] since AdjacencyRow handles both.)
```

Key `if constexpr` sites:
- `ResizeRow`: skip for fixed-width outputs, assert width matches instead
- `Compress`: skip for fixed-width (no CSR to compress)
- Row iteration: works uniformly via `operator[]` (AdjacencyRow handles both)
- `RowSize`: compile-time constant for fixed-width, runtime for NonUniformSize

### 9.6. adjRegistry with Shared Ownership

```cpp
struct MeshConnectivity {
    std::unordered_map<AdjKind, ssp<AdjVariant>, AdjKindHash> adjRegistry;

    // Register: shares the ssp from ConeAdj/SupportAdj
    void registerAdj(AdjKind kind, ssp<AdjVariant> adjPtr) {
        adjRegistry[kind] = adjPtr;
    }

    // Resolve: returns the shared variant
    ssp<AdjVariant> resolveAdj(AdjKind kind) const;
};
```

Ghost traversal (`evaluateGhostTree`) dispatches via `std::visit`:

```cpp
void traverseHop(const ssp<AdjVariant>& adjVar, ...) {
    std::visit([&](auto& pair) {
        // pair is tAdjPair, tAdj1Pair, tAdj2Pair, etc.
        // All have the same operator[] / RowSize interface.
        for (index i = 0; i < nOwned; i++)
            for (auto target : pair[i])
                collectGhost(target);
    }, *adjVar);
}
```

### 9.7. Migration Plan

**Phase A: Shared ownership for ConeAdj/SupportAdj/Registry**
1. Change `ConeAdj::adj` from `AdjVariant` to `ssp<AdjVariant>`.
2. Change `SupportAdj::adj` from `AdjVariant` to `ssp<AdjVariant>`.
3. Change `adjRegistry` from `map<AdjKind, tAdjPair*>` to `map<AdjKind, ssp<AdjVariant>>`.
4. Update `registerAdj` to accept `ssp<AdjVariant>`.
5. Update `addCone`/`addSupport` to allocate `make_ssp<AdjVariant>(...)`.
6. Update all `asAdj()`, `asAdj2()`, etc. to dereference through ssp.
7. Update `evaluateGhostTree` to use `std::visit` on `*resolveAdj(kind)`.
8. Tests must pass unchanged.

**Phase B: Templatize DSL methods**
1. Convert `Inverse` to `template<int cone_rs>`. Keep return as `tAdjPair`.
2. Convert `ComposeFiltered` to `template<int rs_AB, int rs_BC, class Pred>`.
3. Convert `Interpolate` (local) to `template<int p2n_rs, int p2e_rs, int e2n_rs, int e2p_rs>`.
4. Convert `InterpolateGlobal` similarly.
5. Add `if constexpr` for ResizeRow/Compress differences.
6. All default template args = NonUniformSize → existing callers compile unchanged.
7. Move template implementations to header (or explicit instantiation).

**Phase C: Templatize result structs**
1. Parameterize `InterpolateResult<p2e_rs, e2n_rs, e2p_rs>`.
2. Parameterize `InterpolateGlobalResult<p2e_rs, e2n_rs, e2p_rs>`.
3. Parameterize `InterpolateDistributedResult<p2e_rs, e2n_rs, e2p_rs>`.
4. Update callers that destructure result fields.

**Phase D: Production usage with fixed-width**
1. InterpolateFace: use `e2p_rs=2` for entity2parent (faces always have ≤2 parents).
2. face2cell stored as `tAdj2Pair` already — wire through the DAG.
3. bnd2face/face2bnd as `tAdj1Pair` through the DAG.

### 9.8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Template bloat (many instantiations) | Only instantiate used combinations; explicit instantiation in .cpp |
| Compile time increase | Templates in headers are unavoidable for if-constexpr; mitigate with explicit instantiation |
| AdjVariant size grows | Currently 6 alternatives; no change needed |
| ssp overhead vs raw ptr | Negligible — these are large arrays; one extra indirection per access |
| evaluateGhostTree perf with std::visit | Visit dispatch is one virtual call per hop, not per element; negligible |

---

## 10. Per-Adjacency Index State Tracking

### 10.1. Problem Statement

The current global/local index state management has five group-level state
variables (`adjPrimaryState`, `adjFacialState`, `adjC2FState`, `adjN2CBState`,
`adjC2CFaceState`) that each govern 1-4 adjacency arrays. This creates three
problems:

1. **Wrong granularity.** `adjPrimaryState` covers `cell2node` (points to
   nodes), `cell2cell` (points to cells), `bnd2node` (points to nodes), and
   `bnd2cell` (points to cells). Converting any one forces converting all four,
   even when only one needs to change. The "ForBnd" variants
   (`AdjGlobal2LocalPrimaryForBnd`) are ad-hoc workarounds.

2. **Implicit mapping dependencies.** Each conversion hard-codes which
   `pLGhostMapping` to use: `cell2cell` uses `cellElemInfo.trans.pLGhostMapping`,
   `cell2node` uses `coords.trans.pLGhostMapping`. This coupling is scattered
   across 10+ methods and is easy to break during refactoring.

3. **Bounce conversions.** Ghost pulls require global indices; computation
   requires local. Operations like `MatchFaceBoundary`, `BuildCell2CellFace`,
   and `ReorderLocalCells` do expensive local->global->local round-trips on
   entire groups because the group state forces all-or-nothing conversion.

### 10.2. Design: `AdjIndexInfo` and `AdjWithState<TPair>`

#### 10.2.1. `AdjIndexInfo`

A lightweight struct recording what an adjacency array points to and its
current state:

```cpp
// In Mesh_DeviceView.hpp or a new header (e.g., AdjIndexInfo.hpp)

struct AdjIndexInfo
{
    /// Current global/local state of the entries in this adjacency.
    MeshAdjState state{Adj_Unknown};

    /// Ghost mapping of the **target** entity kind.
    /// - cell2node → coords.trans.pLGhostMapping (node ghost mapping)
    /// - cell2cell → cellElemInfo.trans.pLGhostMapping (cell ghost mapping)
    /// - face2cell → cellElemInfo.trans.pLGhostMapping (cell ghost mapping)
    /// - node2cell → cellElemInfo.trans.pLGhostMapping (cell ghost mapping)
    /// - face2node → coords.trans.pLGhostMapping (node ghost mapping)
    /// nullptr until the target entity's ghost layer has been built.
    t_pLGhostMapping targetMapping;

    /// Global-offsets mapping of the target entity kind.
    /// Used for _NoSon conversion paths (father-only, no ghost lookup).
    t_pLGlobalMapping targetGlobalMapping;

    /// Convert all entries in [0, nRows) from global to local.
    template <class TAdj>
    void toLocal(TAdj &adj, index nRows)
    {
        DNDS_assert(state == Adj_PointToGlobal);
        DNDS_assert(targetMapping);
        for (index i = 0; i < nRows; i++)
            for (rowsize j = 0; j < adj.RowSize(i); j++)
            {
                index &v = adj(i, j);
                if (v == UnInitIndex)
                    continue;
                MPI_int rank;
                index val;
                auto ret = targetMapping->search_indexAppend(v, rank, val);
                DNDS_assert(ret);
                v = val;
            }
        state = Adj_PointToLocal;
    }

    /// Convert all entries in [0, nRows) from local to global.
    template <class TAdj>
    void toGlobal(TAdj &adj, index nRows)
    {
        DNDS_assert(state == Adj_PointToLocal);
        DNDS_assert(targetMapping);
        for (index i = 0; i < nRows; i++)
            for (rowsize j = 0; j < adj.RowSize(i); j++)
            {
                index &v = adj(i, j);
                if (v == UnInitIndex)
                    continue;
                if (v < 0) // "not-found" encoding from prior G2L
                    v = -1 - v;
                else
                    v = targetMapping->operator()(-1, v);
            }
        state = Adj_PointToGlobal;
    }

    /// OMP-parallelized variant for large arrays.
    template <class TAdj>
    void toLocalOMP(TAdj &adj, index nRows);
    template <class TAdj>
    void toGlobalOMP(TAdj &adj, index nRows);
};
```

#### 10.2.2. `AdjWithState<TPair>`

A thin wrapper that pairs an `ArrayPair` with its `AdjIndexInfo`:

```cpp
template <class TPair>
struct AdjWithState
{
    TPair pair;
    AdjIndexInfo idx;

    // ---- Forwarding accessors (transparent to callers) ----
    decltype(auto) operator[](index i) { return pair[i]; }
    decltype(auto) operator[](index i) const { return pair[i]; }
    decltype(auto) operator()(index i, rowsize j) { return pair(i, j); }
    decltype(auto) operator()(index i, rowsize j) const { return pair(i, j); }
    auto RowSize(index i) const { return pair.RowSize(i); }
    auto Size() const { return pair.Size(); }

    // Access the underlying pair directly
    TPair *operator->() { return &pair; }
    const TPair *operator->() const { return &pair; }

    // Implicit conversion to TPair& for legacy callers
    operator TPair &() { return pair; }
    operator const TPair &() const { return pair; }

    // Convenience: convert this adjacency in-place
    void toLocal() { idx.toLocal(pair, pair.Size()); }
    void toGlobal() { idx.toGlobal(pair, pair.Size()); }
    void toLocalOMP() { idx.toLocalOMP(pair, pair.Size()); }
    void toGlobalOMP() { idx.toGlobalOMP(pair, pair.Size()); }

    // State queries
    MeshAdjState state() const { return idx.state; }
    bool isLocal() const { return idx.state == Adj_PointToLocal; }
    bool isGlobal() const { return idx.state == Adj_PointToGlobal; }
    bool isBuilt() const { return idx.state != Adj_Unknown; }

    // Delegation to TPair
    void InitPair(const std::string &name, const MPIInfo &mpi)
    { pair.InitPair(name, mpi); }
    void TransAttach() { pair.TransAttach(); }
};
```

### 10.3. Target Entity Mapping Table

Each adjacency points to a specific entity kind. This table shows which
`pLGhostMapping` and `pLGlobalMapping` each adjacency needs:

| Adjacency | Points to | targetMapping source | targetGlobalMapping source |
|-----------|-----------|---------------------|---------------------------|
| `cell2node` | Node | `coords.trans.pLGhostMapping` | `coords.father->pLGlobalMapping` |
| `bnd2node` | Node | `coords.trans.pLGhostMapping` | `coords.father->pLGlobalMapping` |
| `face2node` | Node | `coords.trans.pLGhostMapping` | `coords.father->pLGlobalMapping` |
| `cell2cell` | Cell | `cellElemInfo.trans.pLGhostMapping` | `cell2node.father->pLGlobalMapping` |
| `bnd2cell` | Cell | `cellElemInfo.trans.pLGhostMapping` | `cell2node.father->pLGlobalMapping` |
| `face2cell` | Cell | `cellElemInfo.trans.pLGhostMapping` | `cell2node.father->pLGlobalMapping` |
| `node2cell` | Cell | `cellElemInfo.trans.pLGhostMapping` | `cell2node.father->pLGlobalMapping` |
| `cell2cellFace` | Cell | `cellElemInfo.trans.pLGhostMapping` | `cell2node.father->pLGlobalMapping` |
| `node2bnd` | Bnd | `bndElemInfo.trans.pLGhostMapping` | `bnd2node.father->pLGlobalMapping` |
| `face2bnd` | Bnd | `bndElemInfo.trans.pLGhostMapping` | `bnd2node.father->pLGlobalMapping` |
| `cell2face` | Face | `face2node.trans.pLGhostMapping` | `face2node.father->pLGlobalMapping` |
| `bnd2face` | Face | `face2node.trans.pLGhostMapping` | `face2node.father->pLGlobalMapping` |

### 10.4. Wiring Protocol

Mappings are wired at well-defined points in the mesh pipeline:

1. **After `PartitionReorderToMeshCell2Cell` / `ReadDistributed_Redistribute`:**
   - Cell and node `pLGlobalMapping` exist (from `createFatherGlobalMapping`).
   - Ghost mappings do NOT exist yet.
   - Set `state = Adj_PointToGlobal` on primary adjacencies.
   - Wire `targetGlobalMapping` where available.

2. **After `BuildGhostPrimary`:**
   - `cellElemInfo.trans.pLGhostMapping` and `coords.trans.pLGhostMapping` exist.
   - Wire `targetMapping` for all adjacencies that point to cells or nodes:
     ```cpp
     auto cellGhostMap = cellElemInfo.trans.pLGhostMapping;
     auto nodeGhostMap = coords.trans.pLGhostMapping;
     cell2node.idx.targetMapping = nodeGhostMap;
     cell2cell.idx.targetMapping = cellGhostMap;
     bnd2node.idx.targetMapping  = nodeGhostMap;
     bnd2cell.idx.targetMapping  = cellGhostMap;
     ```

3. **After `InterpolateFace` builds face ghosts:**
   - `face2node.trans.pLGhostMapping` exists.
   - Wire `cell2face.idx.targetMapping` and `bnd2face.idx.targetMapping`.
   - Wire face2cell and face2node targetMappings.

4. **After `BuildGhostN2CB` builds node2cell/node2bnd ghosts:**
   - Wire `face2bnd.idx.targetMapping` and `node2bnd.idx.targetMapping`
     (if bnd ghost mapping is available).

Each wiring step is a single shared_ptr copy per adjacency — O(1) cost.

### 10.5. Group State Variables as Derived Queries

The 5 legacy group state variables become read-only computed properties:

```cpp
// In UnstructuredMesh:
MeshAdjState adjPrimaryState() const
{
    // All primary adjacencies must agree (or be Unknown).
    MeshAdjState s = cell2node.state();
    DNDS_assert(cell2cell.state() == s || cell2cell.state() == Adj_Unknown);
    DNDS_assert(bnd2node.state() == s || bnd2node.state() == Adj_Unknown);
    DNDS_assert(bnd2cell.state() == s || bnd2cell.state() == Adj_Unknown);
    return s;
}
// Similarly for adjFacialState(), adjC2FState(), adjN2CBState(), adjC2CFaceState().
```

This maintains backward compatibility: callers that read `m->adjPrimaryState`
continue to compile (now as a function call) and get the same semantics.

For the DeviceView, the state values are still plain members (POD for GPU),
populated from the derived queries at construction time:
```cpp
// In UnstructuredMeshDeviceView constructor:
adjPrimaryState = mesh.adjPrimaryState();
adjFacialState  = mesh.adjFacialState();
```

### 10.6. Convenience Wrappers (Backward Compat)

The grouped conversion methods remain as thin wrappers:

```cpp
void AdjGlobal2LocalPrimary()
{
    cell2cell.toLocal();
    bnd2cell.toLocal();
    cell2node.toLocal();
    bnd2node.toLocal();
}

void AdjLocal2GlobalPrimary()
{
    cell2cell.toGlobal();
    bnd2cell.toGlobal();
    cell2node.toGlobal();
    bnd2node.toGlobal();
}
```

The `ForBnd` variants become trivially:
```cpp
void AdjGlobal2LocalPrimaryForBnd()
{
    cell2node.toLocal();
    // Only cell2node needs conversion for bnd mesh objects
}
```

### 10.7. Phased Migration Plan

**Phase A: Introduce `AdjIndexInfo` + `AdjWithState<TPair>` (non-breaking)**

1. Add `AdjIndexInfo` struct (new header `src/Geom/AdjIndexInfo.hpp`
   or within `Mesh_DeviceView.hpp`).
2. Add `AdjWithState<TPair>` wrapper template.
3. Ensure `AdjWithState` has implicit conversion `operator TPair&()` so
   existing code that passes the pair to functions compiles unchanged.
4. Unit tests for `AdjIndexInfo::toLocal` / `toGlobal` standalone.

**Phase B: Replace member types on `UnstructuredMesh`**

1. Change `tAdjPair cell2node;` → `AdjWithState<tAdjPair> cell2node;`
   and similarly for all adjacency pairs.
2. Because `AdjWithState` has implicit conversion and forwarding operators,
   most callers should compile unchanged. Fix any that break.
3. The 5 group state variables become `MeshAdjState adjPrimaryState() const`
   methods. The old data members are removed.
4. Update `device_array_list_primary()` etc. to return references to
   `pair` member of each `AdjWithState`.

**Phase C: Wire target mappings**

1. After `BuildGhostPrimary`, wire `targetMapping` for all primary adjacencies.
2. After `InterpolateFace`, wire for facial/C2F adjacencies.
3. After `BuildGhostN2CB`, wire for N2CB adjacencies.
4. After `BuildCell2CellFace`, wire for C2CFace.

**Phase D: Simplify conversion methods**

1. Replace the body of each `AdjGlobal2Local*` / `AdjLocal2Global*` with
   calls to `adj.toLocal()` / `adj.toGlobal()`.
2. The group methods remain as convenience wrappers.
3. Remove the hard-coded mapping lookups from each method body.

**Phase E: Enable fine-grained conversions (optional, future)**

1. Call sites that currently bounce an entire group can now convert
   individual adjacencies. E.g., `BuildCell2CellFace` only needs
   `cell2cell` in local state — no need to convert `bnd2node`.
2. `ReorderLocalCells` can convert only the adjacencies it touches.
3. `MatchFaceBoundary` can convert only `cell2face` without touching
   `bnd2face`.

### 10.8. Key Type Changes

```
// Before:
tAdjPair cell2node;
MeshAdjState adjPrimaryState{Adj_Unknown};

// After:
AdjWithState<tAdjPair> cell2node;
// adjPrimaryState is a derived query method
```

### 10.9. `AdjWithState` Implicit Conversion Challenges

Potential breakage sites for implicit `operator TPair&()`:

1. **Template argument deduction.** When a template function takes
   `template<class T> void f(T&)`, passing `AdjWithState<tAdjPair>` deduces
   `T = AdjWithState<tAdjPair>`, not `T = tAdjPair`. If the function body
   accesses `.father`, `.son`, `.trans`, it breaks.
   **Mitigation:** Add forwarding members: `auto &father()`, `auto &son()`,
   `auto &trans()` on `AdjWithState`, or use `.pair.father` at those sites.

2. **`ConvertAdjEntries` and `ConvertAdjEntriesOMP`.** These templates use
   `TAdj::RowSize()` and `TAdj::operator()()` — both forwarded by
   `AdjWithState`, so they should work.

3. **`PermuteRows`.** Uses `pair.father` directly. Will need `.pair.father`
   or forwarding.

4. **`device_array_list_*()`.** The `DNDS_MAKE_1_MEMBER_REF` macro takes
   a member name. Will need to reference `cell2node.pair` or adapt the macro.

5. **Serialization (`ReadSerialize`/`WriteSerialize`).** Accesses
   `.father`, `.son`, `.trans` directly. Will need `.pair.` prefix.

### 10.10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Implicit conversion not deduced in templates | Add forwarding members or use `.pair` at breakage sites |
| `device_array_list_*` macros break | Adapt macros to access `.pair` member |
| DeviceView constructor copies state | Populate from derived query methods — same semantics |
| Serialization/deserialization breaks | Access `.pair` explicitly at serialization sites |
| Python bindings reference member types | pybind11 bindings need `.pair` access; update `Mesh_bind.hpp` |
| `evaluateGhostTree` still needs global state | Unchanged; the per-adj state just makes the precondition checkable per-adjacency |
| OMP parallel conversion safety | `toLocalOMP`/`toGlobalOMP` use same `#pragma omp parallel for` pattern as existing code |

### 10.11. Non-Goals (This Phase)

- **Lazy/automatic conversion.** The system does NOT automatically convert
  indices on access. Callers still explicitly call `toLocal()` / `toGlobal()`.
  The improvement is that the mapping dependency is recorded (not hard-coded)
  and conversion is per-adjacency (not per-group).

- **Eliminating bounce conversions.** Phase E can reduce bounces by converting
  individual adjacencies, but we do NOT change the ghost pull protocol (which
  fundamentally requires global indices).

- **Moving state into ArrayPair.** The user chose `AdjWithState<TPair>`
  wrapper over modifying `ArrayPair` directly, keeping DNDS core types
  unchanged.
