# Mesh DAG Connectivity: Design Proposal {#mesh_dag_design}

**Status:** Proposal (partially implemented — see notes below)
**Date:** 2026-04-21 (updated 2026-04-26)
**Depends on:** [Mesh Connectivity Architecture](MeshConnectivity.md)

> **Implementation status:** Phases A-B of the migration path (Section 7)
> have been partially implemented. The `MeshConnectivity` DSL class exists
> with `Inverse`, `Compose`, `ComposeFiltered`, `InterpolateGlobal`, and
> `evaluateGhostTree`. Shared ownership via `ssp<>` is in use. Phases C-D
> (full DAG abstraction, unified point numbering) remain future work. Some
> design choices described below (e.g., move semantics in Section 3.2, label
> system in Section 3.5, DAGState enum in Section 6) were superseded by the
> actual implementation — see [MeshConnectivity.md](MeshConnectivity.md)
> for current architecture.

**TL;DR:** This is the original design proposal for replacing ~15 explicit
adjacency arrays and 12 global/local conversion methods with a single DAG
abstraction. The DAG stores cone (cell→face→edge→node) and support
(face→cell, node→cell) relations as shared `tAdjPair` slots. Ghost
generation becomes configurable traversal chains (e.g. `cell→face→cell` for
face-neighbours instead of hard-coded `cell→node→cell`). Phases A–B
(MeshConnectivity DSL + `AdjPairTracked` state) are implemented; Phases
C–D (unified point numbering, full DAG abstraction) are future work.

---

## 1. Motivation

The current `UnstructuredMesh` maintains ~15 explicit adjacency arrays, each as
a separate `tAdjPair` with its own ghost mapping, state flag, and
global-to-local conversion method. This leads to:

- **Duplication:** 12 conversion methods (Adj*Global2Local*, Adj*Local2Global*)
  with overlapping logic.
- **Rigidity:** Ghost depth is hardcoded to one ring of node-neighbors. FEM
  users pay for unneeded ghosts; wide-stencil FV users cannot get enough.
- **No edge entities:** Adding edges would require ~8 new arrays and ~4 new
  conversion methods.
- **Fragile state management:** 5 independent state flags that don't compose or
  enforce ordering.

A DAG-based abstraction can eliminate these problems while providing the same
query performance (or better) that the solver hot paths require.

---

## 2. What the Hot Paths Actually Need

An audit of all solver code (CFV, Euler, EulerP) reveals that **runtime
performance depends on exactly two adjacency queries**:

| Query | Occurrences | Pattern |
|-------|-------------|---------|
| `face → (cellL, cellR)` | ~60 | Constant-time, 2 entries per face |
| `cell → [faces]` | ~62 | Variable-length row (3-8 entries) |

Everything else is derived:
- `CellFaceOther(iCell, iFace)` reads `face2cell`
- `CellIsFaceBack(iCell, iFace)` reads `face2cell`
- `face → zone_ID` reads `faceElemInfo`

**Key finding:** `cell2cell` (node-neighbor adjacency) is **never accessed**
at runtime. It exists only to determine the ghost set during mesh setup, then
is superseded by face-based traversal.

Setup-time queries (metric construction, initialization) also use:
- `cell → [nodes]` (for coordinate gathering)
- `face → [nodes]` (for face metric computation)
- `bnd → (face, cell, zone)` (for boundary identification)
- `node → [cells]` (for inverse topology during ghost build)

---

## 3. Design: MeshConnectivity as a DAG

### 3.1. Entity Model

Every mesh entity gets a unique ID in a contiguous range. Entities are grouped
by **depth** (topological dimension):

```
Depth 0:  Nodes       [nodeStart, nodeEnd)
Depth 1:  Edges       [edgeStart, edgeEnd)       ← new, optional
Depth 2:  Faces       [faceStart, faceEnd)       ← codim-1
Depth 3:  Cells       [cellStart, cellEnd)       ← codim-0
```

For 2D meshes, depth 1 = edges = faces (codim-1), and depth 2 = cells.
For 3D meshes, all four strata exist.

Boundaries are not a separate entity type — they are **labels** on
codim-1 faces (see Section 3.5).

### 3.2. Stored Relations: Per-Depth Adjacency Pairs

Unlike DMPlex's single monolithic cone/support over all points, we store
**one adjacency pair per inter-layer edge**. This maps naturally to DNDSR's
existing `tAdjPair` infrastructure: each pair IS the legacy array, just
organized by the DAG.

```cpp
struct MeshConnectivity
{
    int meshDim;              // 2 or 3
    MPIInfo mpi;

    // --- Strata (entity ranges) ---
    // Each stratum is identified by depth (topological dimension).
    // Depth 0 = nodes, depth dim = cells. Intermediate depths may or
    // may not exist (depending on interpolation).
    struct Stratum
    {
        int depth;
        index count;          // number of owned (father) entities
        // The father/son sizes come from the pair arrays below.
    };
    std::vector<Stratum> strata; // ordered by depth, 0..dim

    // --- Inter-layer adjacency (the DAG edges) ---
    // Each InterLayerAdj stores the covering relation between two
    // adjacent strata. Direction is always "down" (higher-depth →
    // lower-depth), matching the cone direction.
    //
    // The arrays are the SAME tAdjPair objects that currently exist
    // in UnstructuredMesh — they are moved into the DAG at setup time,
    // not copied. After the move, the legacy member becomes a reference
    // or alias into the DAG's storage.
    struct InterLayerAdj
    {
        int fromDepth;        // e.g., dim (cells)
        int toDepth;          // e.g., 0 (nodes) or dim-1 (faces)
        tAdjPair cone;        // CSR: from-entity → to-entities
        tAdjPair support;     // CSR: to-entity → from-entities (inverse)
        // Orientation per cone entry (for periodicity, relative orientation)
        tPbiPair coneOrientation; // optional; null if non-periodic
    };
    std::vector<InterLayerAdj> layers;

    // --- Lookup helpers ---
    // Find the inter-layer adjacency between two depths (nullptr if absent)
    InterLayerAdj *findAdj(int fromDepth, int toDepth);
    const InterLayerAdj *findAdj(int fromDepth, int toDepth) const;
};
```

After face interpolation, a 2D mesh has these inter-layer adjacencies:

```
layers[0]: depth 2 → depth 1  (cell → face)    cone = cell2face, support = face2cell
layers[1]: depth 1 → depth 0  (face → node)    cone = face2node, support = node2face
```

A 3D mesh with edges would have:

```
layers[0]: depth 3 → depth 2  (cell → face)
layers[1]: depth 2 → depth 1  (face → edge)
layers[2]: depth 1 → depth 0  (edge → node)
```

The **non-interpolated** source state (before face generation) has a single
direct layer:

```
layers[0]: depth dim → depth 0  (cell → node)  cone = cell2node, support = node2cell
```

#### Why per-layer storage (not a single global CSR)?

1. **Legacy array compatibility.** Each `tAdjPair` IS one of the existing
   arrays (`cell2node`, `face2cell`, etc.), moved in place. No data copy.
   Code holding a reference to `face2cell` continues to work after the pair
   is moved into the DAG.

2. **Independent ghost lifecycles.** Different layers can be ghosted at
   different times. `cell2node` ghost exists from `BuildGhostPrimary`;
   `face2cell` ghost exists from `InterpolateFace`; edge ghost may be added
   later. A monolithic CSR would require all-or-nothing ghosting.

3. **Selective interpolation.** You can interpolate faces without edges, or
   edges without faces. Each interpolation step adds one new `InterLayerAdj`.

4. **Device transfer granularity.** Only the layers needed by the solver
   kernel are transferred to GPU.

#### Legacy array move semantics

The existing arrays are **moved** (C++ move semantics) into the DAG at the
point where the DAG takes ownership. The old member becomes a reference:

```cpp
// During DAG initialization
auto &cellFaceLayer = dag.addLayer(dim, dim - 1);
cellFaceLayer.cone = std::move(cell2face);     // cell2face is now empty
cellFaceLayer.support = std::move(face2cell);   // face2cell is now empty

// Legacy aliases (references into DAG storage)
auto &cell2face = dag.findAdj(dim, dim - 1)->cone;
auto &face2cell = dag.findAdj(dim, dim - 1)->support;
```

After the move, `cell2face` and `face2cell` point to the same memory, just
organized under the DAG. Existing solver code that accesses `mesh->cell2face`
or `mesh->face2cell` sees no change.
```

### 3.3. Derived Queries (Zero Extra Storage)

All traditional adjacency arrays become queries against the DAG layers:

```cpp
// --- Stratum accessors ---
auto cells()  -> Range  { return strata[meshDim]; }
auto faces()  -> Range  { return strata[meshDim - 1]; }
auto edges()  -> Range  { return strata[1]; }  // only if interpolated
auto nodes()  -> Range  { return strata[0]; }

// --- Direct adjacency (single layer hop) ---
// These are O(1) CSR row lookups — identical cost to current arrays.
auto cellFaces(index iCell) -> RowView
{
    return findAdj(dim, dim-1)->cone[iCell];
}
auto faceCells(index iFace) -> RowView
{
    return findAdj(dim, dim-1)->support[iFace];
}
auto faceNodes(index iFace) -> RowView
{
    return findAdj(dim-1, 0)->cone[iFace];
    // or findAdj(dim-1, 1)->cone → findAdj(1, 0)->cone for 3D with edges
}

// --- Multi-hop traversal (for setup-time queries) ---
auto cellNodes(index iCell) -> SmallVec
{
    // Before face interpolation: direct cell→node layer exists
    if (auto *adj = findAdj(dim, 0))
        return adj->cone[iCell];
    // After interpolation: traverse cell → faces → nodes
    SmallVec result;
    for (auto iFace : cellFaces(iCell))
        for (auto iNode : faceNodes(iFace))
            result.insertUnique(iNode);
    return result;
}

// --- Convenience (performance-critical, inlined) ---
index cellFaceOther(index iCell, index iFace)
{
    auto cells = faceCells(iFace);
    return cells[0] == iCell ? cells[1] : cells[0];
}

bool cellIsFaceBack(index iCell, index iFace)
{
    return faceCells(iFace)[0] == iCell;
}
```

### 3.4. cellNodes: Before and After Interpolation

An important subtlety: before face interpolation, the DAG has a direct
`cell → node` layer (depth dim → depth 0). After interpolation, this direct
layer is replaced by `cell → face → node` (or `cell → face → edge → node`).

The `cellNodes()` query must handle both states. Two strategies:

1. **Keep the direct cell→node layer.** After interpolation, the cell→face
   and face→node layers are ADDED but cell→node is NOT removed. This is
   redundant but means `cellNodes()` is always O(1) CSR.

2. **Replace and cache.** After interpolation, cell→node is removed and
   derived on demand (or cached at setup time). Saves memory at the cost of
   a multi-hop query.

For DNDSR, strategy (1) is preferred: cell→node is used extensively in
metric construction and is small relative to total mesh data. Keeping it
alongside the face layers costs only ~5% extra memory.

### 3.5. Labels Replace Entity-Type Arrays

Instead of `cellElemInfo`, `faceElemInfo`, `bndElemInfo` as separate arrays per
entity type, use **labels** (named integer maps on the point set):

```cpp
struct MeshLabels
{
    // Element type per point (cells, faces, edges all share one array)
    tElemInfoArrayPair elemInfo;  // indexed by point ID

    // Zone/BC IDs per point
    // Replaces the separate zone queries per entity type
    // Boundaries = faces whose zone != BC_ID_INTERNAL

    // Named label groups (like DMPlex's DMLabel)
    std::map<std::string, tAdj1Pair> labels;
};
```

The boundary concept becomes: "faces with a non-internal zone ID."
`NumBnd()` becomes a count query on the zone label. `bnd2face` and `bnd2cell`
become derived lookups, not stored arrays.

### 3.6. Data on Points: Section-Like Layout

Following DMPlex's `PetscSection` concept, field data (coordinates, solution
variables, etc.) is laid out over the point set via a "section":

```cpp
struct MeshSection
{
    // For each point: number of DOFs and offset into flat storage
    std::vector<int> ndof;       // per point
    std::vector<index> offset;   // per point, into data Vec

    // The actual data lives in a separate flat array (Vec)
};
```

Coordinates are a section on depth-0 (node) points with ndof=3.
Cell-centered solution is a section on depth-max (cell) points.
This replaces the convention of separate `tCoordPair coords` that is indexed
by node IDs.

---

## 4. Ghost Management on the DAG

### 4.1. Ghost as DAG Point Ownership

Each point has an owner rank. Local points are "father" (owned), ghost points
are "son" (remote-owned). This is exactly the current father/son model, but
applied uniformly to ALL entity types via a single ownership map:

```cpp
struct PointOwnership
{
    // For each point in [chartStart, chartEnd): owning rank
    // Father points: owner == mpi.rank
    // Ghost (son) points: owner != mpi.rank
    std::vector<MPI_int> owner;

    // PetscSF-like structure for communication
    // Maps (local ghost point) → (remote rank, remote local index)
    // Used for ghost pull/push
    GhostMapping ghostMap;
};
```

### 4.2. Configurable Ghost Requirements: Traversal Chains

The ghost requirement is specified as a sequence of **inter-layer edge
traversals** — NOT as "N rings of same-type adjacency." This is the key
flexibility: each hop in the chain can traverse a different layer.

#### The Traversal Chain Model

A traversal chain is a sequence of (fromDepth, toDepth) hops through the DAG:

```cpp
// A single hop: traverse from one stratum to another via an InterLayerAdj
struct AdjHop
{
    int fromDepth;
    int toDepth;
    enum Direction { Cone, Support } dir;  // Cone = downward, Support = upward
};

// A ghost requirement is a sequence of hops starting from a seed stratum
struct GhostTraversalChain
{
    int seedDepth;               // starting entity depth (typically dim = cells)
    std::vector<AdjHop> hops;   // sequence of traversals
    // The LAST hop's toDepth entities are the ones that become ghost.
    // All intermediate entities traversed are also ghosted if needed.
};

struct GhostRequirement
{
    std::vector<GhostTraversalChain> chains;

    // Which depths to "complement" — for every ghost entity at this depth,
    // also pull all its cone sub-entities so their full connectivity is available.
    std::set<int> complementDepths;
};
```

#### Examples: Current and Future Configurations

**Current DNDSR (compact FV, 1 ring of node-neighbors):**

```cpp
// cell → node → cell  (one ring of node-neighbors)
GhostRequirement current;
current.chains = {{
    .seedDepth = dim,
    .hops = {
        {dim, 0, Cone},     // cell → node (downward)
        {0, dim, Support},  // node → cell (upward)
    }
}};
current.complementDepths = {0};  // ghost cells get all their nodes
```

This reads: "starting from local cells, traverse down to their nodes, then up
to all cells touching those nodes. The resulting non-local cells become ghost.
Complement at depth 0 means those ghost cells also pull their full node sets."

**Face-neighbor only (standard FV):**

```cpp
// cell → face → cell  (one ring of face-neighbors)
GhostRequirement faceNeighbor;
faceNeighbor.chains = {{
    .seedDepth = dim,
    .hops = {
        {dim, dim-1, Cone},     // cell → face
        {dim-1, dim, Support},  // face → cell
    }
}};
faceNeighbor.complementDepths = {0};
```

**Two rings of face-neighbors (wide stencil FV):**

```cpp
// cell → face → cell → face → cell
GhostRequirement wideStencil;
wideStencil.chains = {{
    .seedDepth = dim,
    .hops = {
        {dim, dim-1, Cone},
        {dim-1, dim, Support},
        {dim, dim-1, Cone},
        {dim-1, dim, Support},
    }
}};
wideStencil.complementDepths = {0};
```

**Node-based FV (2 layers of node-neighbors):**

```cpp
// cell → node → cell → node → cell
GhostRequirement nodeFV;
nodeFV.chains = {{
    .seedDepth = dim,
    .hops = {
        {dim, 0, Cone},      // cell → node
        {0, dim, Support},   // node → cell
        {dim, 0, Cone},      // cell → node (2nd ring)
        {0, dim, Support},   // node → cell
    }
}};
nodeFV.complementDepths = {0, 1};  // nodes and edges
```

**Mixed: face-neighbor cells + node complement (FEM-like):**

```cpp
// Ghost cells via faces, but complement nodes via node2cell
GhostRequirement femLike;
femLike.chains = {
    // Chain 1: ghost cells via face-neighbor
    {dim, {{dim, dim-1, Cone}, {dim-1, dim, Support}}},
    // Chain 2: complement — for each local+ghost cell, pull all nodes
    //          (this is implicit via complementDepths = {0})
};
femLike.complementDepths = {0};
```

#### Non-Uniform Chains: The Key Insight

The chain model's power is that each hop traverses a DIFFERENT inter-layer
edge. This enables specifications that cannot be expressed as "N rings of
uniform adjacency":

```
cell → face → cell → node → cell
```

This says: "first, find face-neighbors of local cells. Then, from those
face-neighbors, find all cells sharing any vertex." This is a 1-ring
face-neighbor core plus a 1-ring node-neighbor expansion — a non-symmetric
stencil that might be useful for certain reconstruction schemes.

The traversal engine doesn't need to understand the semantics — it just
follows the hops:

```
BuildGhost(dag, requirement):
  for each chain in requirement.chains:
    frontier = set of local entities at chain.seedDepth
    for each hop in chain.hops:
      nextFrontier = {}
      adj = dag.findAdj(hop.fromDepth, hop.toDepth)
      for entity in frontier:
        if hop.dir == Cone:
          for target in adj.cone[entity]:
            nextFrontier.add(target)
        else:  // Support
          for target in adj.support[entity]:
            nextFrontier.add(target)
      frontier = nextFrontier

    // frontier now contains all entities reached by this chain
    // Mark non-local entities as ghost
    for entity in frontier:
      if not local(entity):
        ghostSet[depthOf(entity)].add(entity)

  // Complement: for each ghost at a complement depth, pull cone sub-entities
  for depth in requirement.complementDepths:
    for ghostEntity in ghostSet[depth+1..dim]:  // higher-depth ghosts
      for sub in transitiveCone(ghostEntity, depth):
        if not local(sub):
          ghostSet[depth].add(sub)
```

### 4.3. Inclusive Relations Between Chains

Some traversal chains produce ghost sets that are subsets of others:

```
cell → face → cell  ⊆  cell → node → cell
```

This is because every face-neighbor shares at least `dim` vertices, so it's
always reachable via node-neighbor. The ghost builder can detect such
inclusions and skip redundant communication.

More generally:

```
cell → face → cell → face → cell  ⊆  cell → node → cell → node → cell
                                   (in general, not true: depends on mesh topology)
```

The traversal engine should NOT try to prove general inclusions at runtime.
Instead, it should:
1. Evaluate all chains independently.
2. Union the ghost sets.
3. Deduplicate.

This is simple and correct. Optimizing redundant chains is a future concern.

---

## 5. Interpolation (Entity Generation) on the DAG

Currently, face generation is hardcoded in `InterpolateFace()`. With a DAG, it
becomes generic: "insert intermediate entities at a given depth."

### 5.1. Generic Interpolation

`DMPlexInterpolate` creates all intermediate entities. For DNDSR, we separate
this into configurable steps:

```cpp
// Insert codim-1 entities (faces) between cells and nodes
void InterpolateDepth(MeshConnectivity &dag, int depth);

// For 3D:
InterpolateDepth(dag, 2);  // faces: cell → face → node
InterpolateDepth(dag, 1);  // edges: face → edge → node

// For 2D:
InterpolateDepth(dag, 1);  // edges (= faces): cell → edge → node
```

Each step follows the same algorithm (currently implemented for faces only):
1. For each cell, extract sub-entities from element topology tables
2. Deduplicate by sorted vertex set
3. Assign ownership across partitions
4. Build cones (new entity → nodes) and supports (new entity → cells)
5. Build ghost for cross-partition entities

### 5.2. Edge Interpolation Falls Out Naturally

With the DAG, adding edges requires:
1. Element topology tables that list edge-to-node connectivity (already present
   for most element types via `GetFaceElement` — edges are just lower-dimensional
   faces of faces)
2. Calling `InterpolateDepth(dag, 1)` for 3D meshes

No new arrays, no new state flags, no new conversion methods.

---

## 6. State Management on the DAG

Replace the 5 independent `MeshAdjState` flags with a single state machine:

```cpp
enum class DAGState : uint8_t
{
    Empty,            // No topology loaded
    SourceOnly,       // cell2node (depth-max → depth-0) loaded, global indices
    Interpolated,     // Intermediate entities (faces, edges) generated
    GhostBuilt,      // Ghost points added
    LocalIndexed,     // All indices converted to local (father+son)
};
```

Transitions:

```
Empty → SourceOnly: ReadMesh / PartitionReorder
SourceOnly → Interpolated: InterpolateDepth(all requested depths)
Interpolated → GhostBuilt: BuildGhost(requirement)
GhostBuilt → LocalIndexed: ConvertToLocal()
LocalIndexed → GhostBuilt: ConvertToGlobal()  (reversible)
```

A single flag covers the entire DAG. The intermediate-entity generation
(currently `InterpolateFace`) becomes a state transition, not a side effect.

---

## 7. Migration Path

### 7.1. Phase A: MeshConnectivity as an Internal Organizer (non-breaking)

The DAG does NOT replace the legacy arrays — it **shares** them. Since
`tAdjPair` already wraps `ssp<ParArray>` (shared_ptr) for father and son,
the DAG and the legacy members can point to the same underlying `ParArray`
objects without any move or copy.

```cpp
struct UnstructuredMesh : public DeviceTransferable<UnstructuredMesh>
{
    MeshConnectivity dag;  // NEW: the DAG organizer

    // Legacy members remain as tAdjPair values (not references).
    // They share the same ssp<ParArray> pointers as the DAG's InterLayerAdj slots.
    tAdjPair cell2node;    // cell2node.father == dag.findLayer(dim, 0)->cone.father
    tAdjPair cell2face;    // cell2face.father == dag.findLayer(dim, dim-1)->cone.father
    tAdj2Pair face2cell;   // face2cell.father == dag.findLayer(dim, dim-1)->support.father
    tAdjPair face2node;    // face2node.father == dag.findLayer(dim-1, 0)->cone.father
    tAdjPair node2cell;    // node2cell.father == dag.findLayer(dim, 0)->support.father
    // ... etc.

    // Adoption: share ssp<> pointers between legacy members and DAG
    void AdoptIntoDAG()
    {
        auto &cellNodeLayer = dag.addLayer(dim, 0);
        // Share (not move!) the underlying ParArray shared_ptrs
        cellNodeLayer.cone.father = cell2node.father;
        cellNodeLayer.cone.son = cell2node.son;
        cellNodeLayer.support.father = node2cell.father;
        cellNodeLayer.support.son = node2cell.son;
        // ... etc. for all layers

        // Both legacy members and DAG now point to the same data.
        // Mutations via either path are visible to both.
    }
};
```

After `AdoptIntoDAG()`:
- All existing solver code that reads `mesh->cell2face[iCell]` works unchanged.
- The DAG can operate on the arrays collectively (global-to-local conversion
  iterates all layers instead of calling 12 separate methods).
- No move, no invalidation, no need to "move back."

### 7.2. Phase B: Migrate Build Pipeline to DAG Operations

The build pipeline methods are replaced with DAG operations. The legacy method
names remain as thin wrappers.

| Current Method | DAG Equivalent |
|----------------|---------------|
| `RecoverNode2CellAndNode2Bnd` | `dag.buildSupport(dim, 0)` — derive support from cone for the cell→node layer |
| `RecoverCell2CellAndBnd2Cell` | `dag.buildReachability(dim, 0, dim)` — compute cell→node→cell reachability for ghost determination |
| `BuildGhostPrimary` | `dag.buildGhost(ghostReq)` — generic ghost build using traversal chains |
| `AdjGlobal2LocalPrimary` | `dag.convertToLocal()` — single method for all layers |
| `InterpolateFace` | `dag.interpolateDepth(dim - 1)` — generic entity generation |
| `BuildGhostN2CB` | Subsumed by `dag.buildGhost` — node2cell ghost is part of the traversal chain |
| `BuildCell2CellFace` | Not needed — `cell → face → cell` traversal is native in the DAG |
| All 12 `AdjGlobal2Local*` / `AdjLocal2Global*` | `dag.convertToLocal()` / `dag.convertToGlobal()` — iterates over ALL layers |

The 12 conversion methods collapse to 2. The 5 state flags collapse to 1.

### 7.3. Phase C: Migrate Solver Code (Optional, Low Priority)

The solver hot paths use only `cell2face` and `face2cell`, which already work
as references into the DAG. No migration needed for correctness. Optional
improvements:

```cpp
// New API (preferred, more explicit)
auto mesh->cellFaces(iCell) -> RowView;
auto mesh->faceCells(iFace) -> RowView;

// Old API (still works, delegates to same storage)
mesh->cell2face[iCell]
mesh->face2cell(iFace, j)
```

### 7.4. Phase D: Edge Support and Extended Ghost

Once the DAG is operational:
1. Add edge interpolation: `dag.interpolateDepth(1)` for 3D meshes.
   This inserts a new `InterLayerAdj{depth=2, toDepth=1}` (face→edge) and
   `InterLayerAdj{depth=1, toDepth=0}` (edge→node).
2. Enable 2-layer ghost for node-based FV via the chain model.
3. Support `node → edge → face → cell` traversal for node-based schemes.
4. The old `cell2cell` array can be removed entirely — it was never used
   at runtime, and the DAG's traversal chains subsume its role in ghost build.

---

## 8. Device (GPU) Considerations

The current `DeviceTransferable` mixin transfers each `tAdjPair` individually.
With a DAG, the transfer becomes:

```cpp
// Transfer the entire DAG topology to device
dag.cone.to_device(backend);
dag.support.to_device(backend);
dag.coneOrientation.to_device(backend);
dag.elemInfo.to_device(backend);
```

The `UnstructuredMeshDeviceView` would hold device views of cone and support
CSR arrays, providing the same `faceCells`/`cellFaces` queries on GPU:

```cpp
template <DeviceBackend B>
struct MeshConnectivityDeviceView
{
    ArrayAdjacencyView<B> cone;    // cell → faces
    ArrayAdjacencyView<B> support; // face → cells
    // Same query API as host
};
```

This is simpler than the current approach of transferring 15+ individual arrays.

---

## 9. Periodicity in the DAG

The current `cell2nodePbi` / `face2nodePbi` arrays store per-connectivity-entry
periodic transformation bits. In the DAG, this maps naturally to
**cone orientations**:

- Each cone entry already has an orientation (sign + group element)
- Periodic bits can be encoded as additional orientation data
- The `periodicInfo.GetCoordByBits(coord, pbi)` pattern becomes
  `dag.getCoordWithOrientation(iNode, orientation)`

This unifies the periodic handling with the general orientation system.

---

## 10. Summary

| Aspect | Current | DAG-based |
|--------|---------|-----------|
| Storage | ~15 separate tAdjPair members | Same pairs, organized in `InterLayerAdj` slots |
| Legacy API | Direct member access | Same — references into DAG storage after move |
| State flags | 5 independent | 1 unified `DAGState` |
| Conversion methods | 12 (per-array-group) | 2 (iterate all layers) |
| Ghost specification | Hardcoded 1-ring node-neighbor | Traversal chains: arbitrary inter-layer hops |
| Ghost example | `cell → node → cell` only | `cell → face → cell`, `cell → node → cell → node → cell`, mixed |
| Edge support | Not possible | `interpolateDepth(1)` adds edge layer |
| Entity creation | `InterpolateFace()` only | Generic `interpolateDepth(d)` for any depth |
| Device transfer | 15+ individual transfers | Iterate layers, transfer cone+support per layer |
| Hot-path query cost | O(1) CSR row | O(1) CSR row (identical — same underlying data) |
| Data ownership | UnstructuredMesh owns all arrays | Shared ssp<> — DAG and legacy members point to same ParArrays |
