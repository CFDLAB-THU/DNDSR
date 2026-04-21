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

**Tests:**
- Compose cell2node + node2cell → cell2cell (verify against known mesh)
- Compose with removeSelf → no diagonal entries
- FilterBySharedNodes: compose(bnd2node, node2cell) filtered by dim shared
  nodes → matches known bnd2cell
- Filter edge case: bnd with all nodes on one cell → 1 result

### Phase 2: Interpolate

**Goal:** Implement `Interpolate()` — the generalized face/edge generator.

**Implementation:** Extract the core algorithm from the existing
`EnumerateFacesFromCells` / `CollectFaces` / `CompactFacesAndRemapCell2Face`
anonymous-namespace helpers in `Mesh.cpp` into `MeshConnectivity::Interpolate`.

**Tests:**
- Interpolate a 2D quad mesh → verify face count, face2node, cell2face
- Interpolate a 2D tri mesh → verify face count
- Interpolate a 3D tet mesh → verify face count
- Verify: for each cell, the number of faces matches element topology
- Verify: every face has exactly 1 or 2 parent cells

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
