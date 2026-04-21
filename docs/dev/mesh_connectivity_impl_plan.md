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

**Goal:** Implement generic ghost building using the chain model.

**Files:**
- `src/Geom/MeshConnectivity_Ghost.cpp`

**Implementation:**
- `BuildGhost(GhostRequirement)` evaluates each chain, collects non-local
  entities, builds ghost mappings for all affected layers.
- Complements: for each ghost cell at a complement depth, pull cone
  sub-entities.

**Tests:**
- Ghost with `cell → node → cell` chain → matches current `BuildGhostPrimary`
  results on known meshes
- Ghost with `cell → face → cell` chain → fewer ghosts than node-neighbor
- Ghost with 2-layer chain → more ghosts than 1-layer
- Verify complement: every ghost cell has all its nodes locally available

### Phase 4: Integration with UnstructuredMesh

**Goal:** Wire `MeshConnectivity` into `UnstructuredMesh` as the internal
organizer. Legacy members become shared references.

**Approach:**
- `UnstructuredMesh` gains a `MeshConnectivity dag` member.
- After the build pipeline completes, `AdoptIntoDAG()` shares the ssp<>
  pointers between legacy members and DAG layers.
- New `convertAllToLocal()` / `convertAllToGlobal()` methods iterate DAG layers.
- Existing methods (`AdjGlobal2LocalPrimary`, etc.) become thin wrappers.

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
