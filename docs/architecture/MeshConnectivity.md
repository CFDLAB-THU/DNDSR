# Mesh Connectivity and Ghost Management {#mesh_connectivity}

**Status:** Current architecture reference
**Date:** 2026-04-26

---

**TL;DR:** A distributed mesh is built from a small set of source arrays
(`cell2node`, `bnd2node`, `coords`, element info). Everything else is derived
via a fixed pipeline: invert node adjacency, compose cell-cell neighbours,
gather ghost cells by node-sharing rings, convert to local indices, then
interpolate faces. Each adjacency now carries its own state (`AdjIndexInfo`)
so the five legacy group flags are gradually being replaced. The document also
describes the `MeshConnectivity` DSL (Inverse, Compose, Interpolate,
evaluateGhostTree) and a future DAG-based design inspired by DMPlex.

---

## 1. Overview

DNDSR's `UnstructuredMesh` manages distributed mesh topology through a system
of explicit adjacency arrays, state flags, and a fixed ghost-creation pipeline.
This document describes how the current system works, its assumptions and
limitations, and a design direction toward configurable ghost connectivity
inspired by PETSc's DMPlex.

---

## 2. Source of Truth

The minimal representation of a distributed mesh is:

| Array | Description | Created by |
|-------|-------------|------------|
| `coords` | Node coordinates | Reader / Partition |
| `cell2node` | Cell-to-vertex connectivity | Reader / Partition |
| `bnd2node` | Boundary-face-to-vertex connectivity | Reader / Partition |
| `cellElemInfo` | Per-cell element type and zone | Reader / Partition |
| `bndElemInfo` | Per-boundary element type and zone | Reader / Partition |
| `cell2nodePbi` | Periodic bits per cell-node pair | Reader (if periodic) |
| `bnd2nodePbi` | Periodic bits per bnd-node pair | Reader (if periodic) |

After partitioning (`PartitionReorderToMeshCell2Cell` or
`ReadSerializeAndDistribute`), each rank owns a subset of cells, nodes, and
boundaries. All adjacency indices point to **global** numbering
(`adjPrimaryState == Adj_PointToGlobal`).

`cell2cell` is NOT stored as source-of-truth. It is always rebuilt from
`cell2node` via the node-inversion path.

For serialization/deserialization, only the source-of-truth arrays are
read/written. Everything else is derived.

---

## 3. Current Build Pipeline

The canonical sequence to build a fully-operational distributed mesh is:

```
PartitionReorderToMeshCell2Cell()   [or ReadSerializeAndDistribute()]
  adjPrimaryState = Adj_PointToGlobal
  Source arrays: coords, cell2node, bnd2node, cellElemInfo, bndElemInfo
  (father-only, global indices)

RecoverNode2CellAndNode2Bnd()
  adjN2CBState = Adj_PointToGlobal
  Builds: node2cell.father, node2bnd.father
  (globally complete for each owned node — MPI push-back collects from all ranks)

RecoverCell2CellAndBnd2Cell()
  Builds: cell2cell.father, bnd2cell.father
  (node-neighbor: cells sharing at least one vertex)
  Uses: node2cell to find all cells reachable via cell → node → cell

BuildGhostPrimary()
  Builds: .son (ghost) for cells, nodes, bnds
  Ghost criterion: one or more rings of node-neighbors from cell2cell
  Supports multi-layer ghost via nGhostLayers parameter (default 1)
  Also ghosts: all nodes touched by ghost cells; all bnds touching owned nodes

AdjGlobal2LocalPrimary()
  adjPrimaryState = Adj_PointToLocal
  Converts all global indices to local (father+son appended)

InterpolateFace()
  adjFacialState = Adj_PointToLocal
  adjC2FState = Adj_PointToLocal
  Builds: face2node, face2cell, cell2face, faceElemInfo, bnd2face, face2bnd
  Enumerates faces from cell2node over all cells (father+son)
  Face ownership: lower-rank-wins for cross-partition faces

[AdjLocal2GlobalN2CB → BuildGhostN2CB → AdjGlobal2LocalN2CB]
  adjN2CBState = Adj_PointToLocal
  Builds: node2cell.son, node2bnd.son
  Ghost criterion: same as coords ghost (all nodes referenced by father+son cells)

BuildCell2CellFace()
  adjC2CFaceState = Adj_PointToGlobal → Adj_PointToLocal (after conversion)
  Builds: cell2cellFace (face-neighbor cell adjacency, distinct from node-neighbor cell2cell)
```

### 3.1. Implicit Connectivity Depths

The current pipeline encodes specific connectivity depth assumptions:

| What | Connectivity Depth | Meaning |
|------|--------------------|---------|
| `cell2cell` | cell → node → cell | All cells sharing any vertex (point-complete) |
| Ghost cells | N rings of `cell2cell` | All cells reachable by N hops (default N=1) |
| Ghost nodes | cell2cell2node | All nodes of ghost cells (so local+ghost cells have all their nodes) |
| Ghost bnds | node2bnd (owned nodes) | All boundaries touching any owned node |
| `node2cell` ghost | Same as coords ghost | Every ghost node has its full cell list |
| `cell2cellFace` | cell → face → cell | Only cells sharing a face (subset of cell2cell) |
| Face ghost | Cross-partition faces | A face is ghost on the non-owning side of a partition boundary |

**Key invariant after `BuildGhostPrimary`:** Starting from any local (father)
cell and traversing `cell2node`, every referenced node is present as father or
ghost. Starting from any such node and traversing `node2cell` (after
`BuildGhostN2CB`), all listed cells that are present in father+son can be
accessed. Some `node2cell` entries may reference cells NOT in father+son
(encoded as negative indices after local conversion).

### 3.2. What "cell2cell2node Complemented" Means

"Cell2cell2node complemented" means: for every ghost cell (in `cell2node.son`),
ALL its nodes are present in the local node set (coords father+son). This is
guaranteed because `BuildGhostPrimary` iterates over ALL cells (father+son)
when collecting ghost nodes (see `Mesh.cpp`, `BuildGhostPrimary`).

Since `cell2cell = cell → node → cell`, and ghost cells = one ring of cell2cell,
this means:
- Starting from any local cell
- Traverse to any neighbor cell via shared vertex (one cell2cell hop)
- That neighbor's full node set is available

This is exactly the stencil needed for compact finite volume (VFV)
reconstruction which uses face-neighbor cell values but needs the full
geometric information (all nodes) of those neighbor cells.

### 3.3. Node2cell Complementation

After `BuildGhostN2CB`, every ghost node has its `node2cell` data pulled from
its owning rank. This data is the GLOBALLY COMPLETE cell adjacency list. However,
not all those cells are present in the local father+son cell set. After
`AdjGlobal2LocalN2CB`, cells not in the local set get the "not-found" encoding
`(-1 - iGlobal)`.

`AssertOnN2CB` verifies that every ghost node has at least one cell that IS
present in the local father+son (i.e., at least one adjacent cell is local or
ghost).

---

## 4. State Management

Five `MeshAdjState` flags track the state of different adjacency groups:

| Flag | Arrays Governed | Valid States |
|------|----------------|--------------|
| `adjPrimaryState` | cell2node, cell2cell, bnd2node, bnd2cell | Unknown, PointToGlobal, PointToLocal |
| `adjFacialState` | face2cell, face2node, face2bnd | Unknown, PointToGlobal, PointToLocal |
| `adjC2FState` | cell2face, bnd2face | Unknown, PointToGlobal, PointToLocal |
| `adjN2CBState` | node2cell, node2bnd | Unknown, PointToGlobal, PointToLocal |
| `adjC2CFaceState` | cell2cellFace | Unknown, PointToGlobal, PointToLocal |

State transitions are guarded by assertions at the start of each
`AdjGlobal2Local*` / `AdjLocal2Global*` method. The states form a linear
sequence: `Unknown → PointToGlobal → PointToLocal`, with the ability to
toggle back and forth between Global and Local.

### 4.1. Weaknesses of Current State Management

1. **No distinction between "father-only" and "father+son".** After
   `PartitionReorderToMeshCell2Cell`, the arrays have only fathers (no ghost),
   but after `BuildGhostPrimary` they have both. The state flag is the same
   (`Adj_PointToGlobal`) in both cases.

2. **No machine-checked preconditions.** Methods like `InterpolateFace`
   require `adjPrimaryState == Adj_PointToLocal` AND that ghost (son) arrays
   exist, but only the state flag is asserted. If someone calls
   `AdjGlobal2LocalPrimary` without first calling `BuildGhostPrimary`, the
   state flag would be correct but the son arrays would be null.

3. **State flags don't encode which arrays are initialized.** `adjPrimaryState`
   covers cell2node, cell2cell, bnd2node, bnd2cell simultaneously. But
   cell2cell doesn't exist until `RecoverCell2CellAndBnd2Cell`, while cell2node
   exists from the partition step. They share the same state flag.

### 4.2. Per-Adjacency State Tracking (`AdjPairTracked`)

To address the weaknesses above, each adjacency array now carries its own
`AdjIndexInfo` recording per-adjacency state and a shared pointer to the
target entity's ghost mapping. This sits alongside (not replacing) the five
group flags.

**Files:**
- `src/Geom/Mesh/AdjIndexInfo.hpp` — `AdjIndexInfo` struct + `AdjPairTracked<TPair>`
- `src/Geom/Mesh/MeshConnectivity_StateChecked.hpp` — checked DSL wrappers

#### `AdjIndexInfo`

All fields are private. State transitions go through methods:

```cpp
struct AdjIndexInfo
{
private:
    MeshAdjState    _state{Adj_Unknown};
    t_pLGhostMapping _targetMapping;   // ghost mapping of the TARGET entity

public:
    // Queries
    MeshAdjState state() const;
    bool isLocal() const;
    bool isGlobal() const;
    bool isBuilt() const;
    bool isWired() const;
    const t_pLGhostMapping &mapping() const;

    // State transitions
    void markGlobal();                              // Unknown|Global -> Global
    void markLocal();                               // Unknown -> Local (requires wired)
    void wireTargetMapping(const t_pLGhostMapping&);// rejects Local state

    // Conversion (requires wired)
    void toLocal(adj, nRows);                       // Global -> Local
    void toGlobal(adj, nRows);                      // Local -> Global
    void toLocalOMP(adj, nRows);                    // Global -> Local (OMP)
    void toGlobalOMP(adj, nRows);                   // Local -> Global (OMP)

    // Bootstrap: wire + convert in one call (solves chicken-and-egg)
    void bootstrapToLocal(mapping, adj, nRows);     // Unknown|Global -> Local
    void bootstrapToLocalOMP(mapping, adj, nRows);
};
```

- `_targetMapping` is the ghost mapping of the entity kind the indices
  **point to**. For `cell2node`, that is the node ghost mapping
  (`coords.trans.pLGhostMapping`). The target mapping always exists on
  some other array pair's transformer — `wireTargetMapping` just stores
  a shared pointer to it.
- `wireTargetMapping()` asserts `_state != Adj_PointToLocal` — wiring a
  mapping while indices are local is a logic error.
- `markLocal()` is for adjacencies populated directly with local indices
  (bypassing the global phase), e.g. `ConstructBndMesh`. Requires the
  mapping to be wired first.
- `toLocal()` encodes not-found entries as `(-1 - globalIndex)`, matching
  `IndexGlobal2Local`.

#### `AdjPairTracked<TPair>`

Uses **inheritance** (not composition) from `TPair`:

```cpp
template <class TPair>
struct AdjPairTracked : public TPair
{
    AdjIndexInfo idx;
    void toLocal();             // idx.toLocal(*this, Size())
    void toGlobal();
    void toLocalOMP();
    void toGlobalOMP();
    void bootstrapToLocal(mapping);
    void bootstrapToLocalOMP(mapping);
    MeshAdjState state() const;
    bool isLocal() const;
    bool isGlobal() const;
    bool isBuilt() const;
    bool isWired() const;
    const t_pLGhostMapping &mapping() const;
};
```

Inheritance keeps all existing callers (`.father`, `.son`, `.trans`,
`BorrowAndPull`, `operator[]`, etc.) working unchanged. All 12 adjacency
members on `UnstructuredMesh` use `AdjPairTracked<T>`:

| Member | Type | Target Entity |
|--------|------|---------------|
| `cell2node` | `AdjPairTracked<tAdjPair>` | node |
| `bnd2node` | `AdjPairTracked<tAdjPair>` | node |
| `bnd2cell` | `AdjPairTracked<tAdj2Pair>` | cell |
| `cell2cell` | `AdjPairTracked<tAdjPair>` | cell |
| `node2cell` | `AdjPairTracked<tAdjPair>` | cell |
| `node2bnd` | `AdjPairTracked<tAdjPair>` | bnd |
| `cell2face` | `AdjPairTracked<tAdjPair>` | face |
| `face2node` | `AdjPairTracked<tAdjPair>` | node |
| `face2cell` | `AdjPairTracked<tAdj2Pair>` | cell |
| `bnd2face` | `AdjPairTracked<tAdj1Pair>` | face |
| `face2bnd` | `AdjPairTracked<tAdj1Pair>` | bnd |
| `cell2cellFace` | `AdjPairTracked<tAdjPair>` | cell |

#### Wiring Protocol

Target mappings are wired at these points in the mesh build pipeline:

1. **After `BuildGhostPrimary`:** cell2node, bnd2node (target=node);
   cell2cell, bnd2cell, node2cell (target=cell); node2bnd (target=bnd).

2. **In `BuildGhostFace`:** face2node (target=node); face2cell
   (target=cell); cell2face, bnd2face (target=face).

3. **In `MatchFaceBoundary`:** face2bnd (target=bnd).

4. **After `BuildCell2CellFace`:** cell2cellFace (target=cell).

`markGlobal()` is called wherever `adjXState = Adj_PointToGlobal` is
set. Conversion methods (`AdjGlobal2Local*` / `AdjLocal2Global*`)
delegate to `adj.toLocal()` / `adj.toGlobal()` and update both the
per-adj state and the group flag in parallel.

Legacy methods (`BuildGhostPrimaryLegacy`, `InterpolateFaceLegacy`)
mirror the same wiring and `markGlobal`/`markLocal` calls as their
DSL counterparts.

#### Three-Layer Architecture

| Layer | File | Knows About State? |
|-------|------|--------------------|
| **DSL** | `MeshConnectivity.hpp` | No — operates on bare `ArrayAdjacencyPair<rs>` |
| **Checked wrappers** | `MeshConnectivity_StateChecked.hpp` | Yes — asserts `idx.state`, then forwards to DSL |
| **Mesh methods** | `Mesh.cpp` | Yes — owns `AdjPairTracked` members, calls checked wrappers |

`CheckedInverse`, `CheckedComposeFiltered`, and `CheckedInterpolateGlobal`
are stateless free function templates. They static-cast `AdjPairTracked<T>&`
to `T&` before forwarding.

#### Current Status

The five group state variables still exist as real data members and are
updated in parallel with per-adj states. They have NOT been converted to
derived queries yet. All code paths (DSL and legacy) now maintain per-adj
`idx` state consistently — `markGlobal()`, `wireTargetMapping()`,
`markLocal()`, `toLocal()`/`toGlobal()` are called at every site where
adjacency data or state changes. The `setStateUnchecked` escape hatch has
been eliminated; all state transitions go through the `AdjIndexInfo` API.

DeviceView does not read per-adj state yet (still uses group state
variables).

Python bindings expose `AdjPairTracked<T>` (with `idx` member),
`AdjIndexInfo` (query methods only: `state()`, `isLocal()`, `isGlobal()`,
`isBuilt()`, `isWired()`), and the `MeshAdjState` enum (`Unknown`,
`PointToGlobal`, `PointToLocal`). State-mutation methods (`markGlobal`,
`wireTargetMapping`, `toLocal`, etc.) are intentionally not exposed; Python
code should not mutate adjacency state directly.

---

## 5. MeshConnectivity DSL

The `MeshConnectivity` struct (in `src/Geom/Mesh/MeshConnectivity.hpp`)
provides a composable DSL for adjacency operations, independent of
`UnstructuredMesh`. It stores adjacency data as shared pointers and
provides the following static operations:

| Operation | Description |
|-----------|-------------|
| `Inverse<rs>` | Invert a cone to a support (distributed MPI push-back) |
| `Compose<AB,BC,out>` | Compose A->B + B->C -> A->C |
| `ComposeFiltered<AB,BC,out,Pred>` | Compose with predicate filtering |
| `Interpolate<p2n_rs>` | Local sub-entity extraction (no MPI) |
| `InterpolateGlobal<p2n_rs,e2p_rs>` | Distributed interpolation with global dedup |
| `evaluateGhostTree(tree, mpi)` | BFS ghost evaluation from compiled ghost spec |

### 5.1. Ghost Specification System

Ghost requirements are specified as chains of adjacency hops:

```cpp
// GhostChain: anchor --hop1--> --hop2--> ... --> target
GhostSpec spec;
spec.chains = {
    { Cell, {Cell2Cell, Cell2Cell}, Cell },  // 2-layer cell ghost
    { Cell, {Cell2Cell, Cell2Cell, Cell2Node}, Node },  // nodes of 2-layer cells
    { Bnd,  {Bnd2Node, Node2Bnd}, Bnd },    // bnds touching owned nodes
};
```

`GhostSpec::defaultPrimary(nLayers)` builds the standard ghost spec:
- Cell chain: `nLayers` hops of `Cell2Cell`
- Node chain: `nLayers` hops of `Cell2Cell` + 1 hop of `Cell2Node`
- Bnd chain: `Bnd2Node -> Node2Bnd`
- Bnd-node chain: `Bnd2Node -> Node2Bnd -> Bnd2Node`

### 5.2. Compiled Ghost Tree

`CompiledGhostTree::compile(spec)` merges chains sharing common prefixes
into a trie-forest and flattens it into BFS-ordered levels. The evaluator
(`evaluateGhostTree`) then processes each level:

1. **Collect** non-owned indices into intermediate ghost sets.
2. **Scratch pull** any adjacency arrays whose ghost sets have grown (collective
   MPI decision via `Allreduce`). This creates temporary transformers.
3. **Traverse** the hop to populate the next level's entity sets.

The result is a `GhostResult` containing per-`EntityKind` sorted,
deduplicated global ghost indices, plus a collective `activeKinds` bitmask.

### 5.3. Adjacency Registry

`MeshConnectivity` maintains an `adjRegistry` mapping `AdjKind` keys to
type-erased `AdjVariant` values. Mesh build methods register their
adjacencies (e.g., `Cell2Node`, `Cell2Cell`) so `evaluateGhostTree` can
look them up by kind. A `globalMappings` registry similarly stores
per-`EntityKind` global offset mappings.

### 5.4. Mesh Helpers

`src/Geom/Mesh/Mesh_Helpers.hpp` provides high-level inline functions
that compose multiple mesh-build steps:

| Helper | Description |
|--------|-------------|
| `BuildGhostPrimary(mesh, nLayers)` | 5-step: recover N2C/C2C + ghost + G2L |
| `ReadMeshFromCGNS(...)` | Full CGNS read + partition + elevation + bisection |
| `ReadMeshFromH5(...)` | H5 distributed read with ParMetis repartition |
| `ReadMeshFromH5Parallel(...)` | H5 pre-partitioned read (exact np match) |
| `PrepareMesh(...)` | Cell reorder + face interp + ghost N2CB + serial out |
| `BuildBndMesh(...)` | Extract boundary surface mesh |
| `SerializeMesh(...)` | Write partitioned mesh to H5 |
| `MeshH5Path(...)` | Build conventional H5 filename |

---

## 6. Limitations and Inflexibility

### 6.1. Configurable Ghost Depth

`BuildGhostPrimary` now accepts an `nGhostLayers` parameter (default 1),
and uses `GhostSpec::defaultPrimary(nLayers)` + `evaluateGhostTree` to
support N-layer node-neighbor ghost cells. The ghost tree evaluator performs
BFS level-by-level with scratch pulls between levels, so intermediate hops
can fetch remote data as needed.

This addresses the compact FV use case (1 layer) and higher-order stencils
(2+ layers). However, the adjacency definition is still node-neighbor
(`cell2cell` via vertex-sharing). The remaining limitations are:

- **FEM** only needs node2cell complemented (cell2node2cell stencil for
  assembly), which is a smaller ghost set than cell2cell node-neighbors.
  The current ghost set is larger than necessary.

- **Node-based FV** needs two layers of node-neighbors
  (node → cell → node → cell → node), which the current single ring does NOT
  provide.

- **High-order VFV** with wide stencils may need 2+ rings of face-neighbors.

### 6.2. No Edge Entities

The current mesh only has 4 entity types: Node, Face (codimension 1), Cell
(codimension 0), and Boundary (a special subset of faces). There are no edge
entities. For 3D node-based FV, we need:

```
3D:  Node - Edge - Face - Cell
2D:  Node - Edge(=Face) - Cell
```

Adding edges would require new arrays (`edge2node`, `cell2edge`, `node2edge`,
etc.) and new ghost/state management for them — significant code duplication
under the current explicit-array approach.

### 6.3. Face Generation is Not Configurable

Faces are generated by `InterpolateFace` which enumerates topological faces from
cell connectivity. This is fixed to the cell-face covering relation. To support
edge generation for node-based FV, a separate `InterpolateEdge` would be needed
with nearly identical logic (enumerate edges from faces or cells, deduplicate,
assign ownership, build ghost). The same pattern would repeat for any new
inter-entity relation.

---

## 7. Design Direction: Configurable Ghost Connectivity

### 7.1. Connectivity Requirement Specification

Instead of a hardcoded ghost pipeline, users should specify their ghost
requirements as a set of **connectivity chains**:

```cpp
// Current implicit requirement (compact FV):
ghostSpec.require("cell2cell");           // ghost cells = node-neighbors
ghostSpec.require("cell2cell.cell2node"); // ghost cells have all their nodes

// FEM requirement (smaller ghost):
ghostSpec.require("cell2node.node2cell"); // only cells sharing nodes with local cells

// Node-based FV requirement (2 layers):
ghostSpec.require("node2cell.cell2node.node2cell.cell2node.node2cell");
```

Each chain specifies a traversal path through the adjacency graph. The ghost set
is the union of all entities reachable by the specified chains from the local
(father) entities.

### 7.2. Inclusive Relations

Some connectivity chains form inclusive relations:

- `cell2cell` (node-neighbor) ⊇ `cell2cellFace` (face-neighbor)
- `cell2cell.cell2node` ⊇ `cell2node` (trivially)
- `cell2cell.cell2node.node2cell` ⊇ `cell2cell` (2-ring via nodes includes 1-ring)

Understanding these inclusions helps avoid redundant ghost communication.

### 7.3. Current Ghost as a Configuration

The current pipeline's ghost criterion can be expressed as:

```
Ghost cells:  cell → node → cell  (1 ring node-neighbors, i.e. cell2cell)
Ghost nodes:  (local cells ∪ ghost cells) → node  (all nodes of all cells)
Ghost bnds:   local nodes → bnd  (all bnds touching owned nodes)
```

Face ghost is determined by the face interpolation step and follows naturally
from cell ghost: faces between a local cell and a ghost cell become ghost faces
on the non-owning side.

### 7.4. Abstract DAG Approach (PETSc DMPlex Inspiration)

PETSc's DMPlex represents the entire mesh topology as a single Directed Acyclic
Graph (DAG) where every entity (cell, face, edge, vertex) is a "point" with a
unique integer ID. The only stored relations are:

- **Cone** (downward/covering): a cell's cone is its bounding faces; a face's
  cone is its bounding edges or vertices.
- **Support** (upward/covered-by): a face's support is the 1-2 cells on either
  side; a vertex's support is all edges or cells touching it.

All other adjacencies are derived by transitive closure queries:
- `cell2node` = transitive cone closure of a cell, filtered to depth-0 (vertices)
- `face2cell` = support of a face, filtered to max-depth (cells)
- `node2cell` = transitive support closure of a vertex, filtered to max-depth
- `cell2cell` (face-neighbor) = for each cone point (face) of cell, get its support

**Key DMPlex design principles relevant to DNDSR:**

1. **Unified point numbering.** All entities share a single contiguous ID space.
   Entities are grouped by "depth" (distance from vertices in the DAG) or
   "height" (distance from cells). Depth 0 = vertices, depth max = cells.

2. **Parameterized adjacency.** Two boolean flags control adjacency queries:
   - `useCone`: traverse downward (cell → boundary faces → neighbors) vs. upward
   - `useClosure`: use transitive closure (all sub-entities) vs. single level
   
   | Setting | Coupling Pattern | Use Case |
   |---------|-----------------|----------|
   | `useCone=F, useClosure=T` | All cells sharing any vertex | FEM, compact FV |
   | `useCone=T, useClosure=F` | Only face-adjacent cells | Standard FV |
   | `useCone=T, useClosure=T` | All cells sharing any sub-entity | Wide-stencil FV |

3. **N-layer overlap.** Ghost layers are computed by N iterations of adjacency
   expansion on the DAG, using the parameterized adjacency definition. One
   "layer" = expand the partition boundary by one adjacency hop.

4. **PetscSF for communication.** A Star Forest data structure encodes which
   points are shared and who owns them, replacing explicit ghost mappings.

5. **PetscSection for data layout.** Data is NOT stored inside the mesh.
   A Section maps `(point) → (ndof, offset)` into a flat vector. This
   decouples topology from discretization.

### 7.5. Practical Evolution Path for DNDSR

A full DMPlex-style DAG rewrite is a major undertaking. A practical evolution:

**Phase A: Configurable ghost depth (near-term)**

Add a `GhostRequirement` configuration that parameterizes `BuildGhostPrimary`:

```cpp
struct GhostRequirement
{
    int cellRings = 1;        // number of cell2cell rings to ghost
    bool nodeNeighbor = true; // cell2cell means node-neighbor (vs face-neighbor)
    bool complementNodes = true; // ghost cells get all their nodes
    bool complementBnds = true;  // owned nodes get all their bnds
};
```

This replaces the hardcoded ghost criterion without changing the array structure.
The existing pipeline works unchanged for `{1, true, true, true}`.

**Phase B: Edge entity support (medium-term)**

Add `edge2node`, `cell2edge`, `node2edge` arrays following the same pattern as
the existing face arrays. Extract the face interpolation logic into a generic
"interpolate codimension-K entities" template.

**Phase C: Abstract adjacency layer (long-term)**

Introduce a `MeshDAG` abstraction that stores cone/support relations for all
entity types. The existing explicit arrays (`cell2node`, `face2cell`, etc.)
become views into the DAG. Ghost management operates on the DAG directly.

```
MeshDAG
  points: [0, nCell + nFace + nEdge + nNode)
  cones[point] → list of covered points
  supports[point] → list of covering points

  stratum(depth) → [start, end)  // depth 0 = nodes, depth dim = cells
  transitiveClosureCone(point) → all descendants
  transitiveClosureSupport(point) → all ancestors

UnstructuredMesh (facade)
  dag: MeshDAG
  coords: data on depth-0 points
  cellElemInfo: data on depth-dim points
  // cell2node, face2cell, etc. become convenience accessors
```

---

## 8. Face and Edge Interpolation Pattern

The current `InterpolateFace` implements a general pattern that could be
reused for edges:

1. **Enumerate** sub-entities from cells by extracting topological sub-elements
   (faces from cells, or edges from faces/cells).
2. **Deduplicate** by sorted vertex comparison (with periodic-aware matching).
3. **Assign ownership** across partitions (lower-rank-wins).
4. **Build ghost** sub-entities for cross-partition interfaces.
5. **Build inverse** relations (cell2face from face2cell, etc.).

For edges, the same algorithm applies with "face" replaced by "edge":
- A face's edges are its topological sub-elements (already defined per element type)
- Deduplication by sorted vertex pair
- Ownership assignment and ghost communication follow the same pattern

This argues for extracting the interpolation logic into a generic template:

```cpp
// Pseudo-code for generic entity interpolation
template <int codim>
void InterpolateEntities(
    UnstructuredMesh &mesh,
    const tAdjPair &parent2node,     // parent-entity to node connectivity
    const tElemInfoArrayPair &parentElemInfo,
    tAdjPair &entity2node,           // output: new entity to node
    tAdj2Pair &entity2parent,        // output: new entity to parent(s)
    tAdjPair &parent2entity);        // output: inverse relation
```

---

## 9. Glossary

| Term | Meaning |
|------|---------|
| **Father** | Locally-owned (non-ghost) portion of an array |
| **Son** | Ghost (remote-owned) portion of an array, pulled from other ranks |
| **Ghost mapping** | `pLGhostMapping` on an ArrayTransformer — maps global indices to local father+son indices |
| **Target mapping** | Ghost mapping of the entity kind an adjacency's indices point to (e.g., for `cell2node`, the node ghost mapping) |
| **AdjPairTracked** | Wrapper that inherits from an ArrayPair and adds per-adjacency `AdjIndexInfo` (state + target mapping) |
| **Node-neighbor** | Two cells that share at least one vertex |
| **Face-neighbor** | Two cells that share a codimension-1 face (>= dim shared vertices) |
| **Complemented** | A ghost entity has all its sub-entity data available locally |
| **Ring** | One hop of adjacency expansion. "1 ring of node-neighbors" = all cells reachable from any vertex of local cells |
| **MeshConnectivity** | Standalone DSL struct providing composable adjacency operations (Inverse, Compose, Interpolate, evaluateGhostTree) |
| **GhostSpec** | Collection of `GhostChain`s specifying ghost requirements as adjacency hop sequences |
| **CompiledGhostTree** | Trie-forest compiled from GhostSpec, used by `evaluateGhostTree` for BFS ghost evaluation |
| **EntityKind** | Scoped enum: `Cell`, `Face`, `Edge`, `Node`, `Bnd` |
| **AdjKind** | Identifier for an adjacency relation (from, to, optional via entity) |
| **Cone** (DMPlex) | Downward covering relation in the DAG (cell → faces → edges → vertices) |
| **Support** (DMPlex) | Upward covering relation (vertex → edges → faces → cells) |
| **Transitive closure** | All points reachable by repeated cone/support traversal |
| **Stratum** | Set of points at a given depth in the DAG (depth 0 = vertices, depth dim = cells) |
| **PetscSF** | Star Forest — PETSc's communication structure for shared/ghost point data exchange |
| **PetscSection** | Maps mesh points to DOF counts and offsets in a flat vector |

---

## References

- PETSc DMPlex manual: https://petsc.org/release/manual/dmplex/
- PETSc PetscSection manual: https://petsc.org/release/manual/section/
- Knepley, M.G. and Karpeev, D.A., "Mesh algorithms for PDE with Sieve I: Mesh Distribution," Scientific Programming, 2009
- Lange, M. et al., "Efficient mesh management in Firedrake using PETSc DMPlex," SIAM J. Sci. Comput., 2016
