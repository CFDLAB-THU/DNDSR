# Node-Primal Mesh: Design & Migration Plan

**Status:** Draft  
**Date:** 2026-05-10  
**Scope:** Allow the mesh to be partitioned and distributed using **nodes**
as the primary entity, with cells deriving their partition from node
ownership.  This enables node-based loading, node-centric repartitioning,
and node-primal redistribution without rewriting the serial CGNS reader.

---

## 1. Motivation

The current mesh pipeline is **cell-primal** throughout:

| Stage | Cell-primal assumption |
|---|---|
| CGNS serial read | Metis partitions cells; nodes/bnds derive from cell partition |
| H5 distributed read | Cells are even-split; nodes/bnds derive from cell partition |
| `GhostSpec::defaultPrimary()` | All chains start from `Cell` (or `Bnd`, which is cell-adjacent) |
| `fillRegistry()` | `Cell` is the first global-mapping source tried |
| `ReorderEntities()` / `resolveFollows()` | When reordering `Cell`, automatically follows `Node` and `Bnd` |

For applications that want node-based load balancing (e.g., vertex-centred
FVM, DG with node patches, node-level AMR), we need the ability to:
1. Load a mesh where **nodes determine the partition** (node-primal read)
2. Repartition nodes (e.g., Metis on node-neighbor graph) and redistribute
   cells/bnds accordingly
3. Use the **distributed-read path** (already DSL-based) rather than
   rewriting the legacy serial-reader push pipeline

---

## 2. Feasibility: What Already Works

### 2.1 The distributed-read path is mostly entity-agnostic

`ReadDistributed_Redistribute()` (`Mesh_ReadSerializeDistributed.cpp:450-476`)
calls `ReorderEntities()` with an explicit partition map per entity kind.
If we provide a **node-based partition** (instead of deriving it from cells),
the reorder framework should handle the rest.

The DSL operations (`Inverse`, `Compose`, `ComposeFiltered`, `evaluateGhostTree`,
`InterpolateGlobal`) are entity-agnostic at the algorithmic level — they take
`EntityKind` as parameters and operate on any registered adjacency.

### 2.2 Ghost chains are user-definable

`GhostSpec` accepts arbitrary `GhostChain` definitions.  A node-primal
ghost specification could be:

```
Node → Node2Cell → Cell             (ghost cells from owned nodes)
Node → Node2Cell → Cell2Node → Node (ghost nodes from owned nodes via cells)
Node → Node2Bnd  → Bnd              (ghost bnds from owned nodes)
```

The compiled-tree evaluator and scratch-pull mechanism work for any
registered adjacency — no code changes needed in the ghost subsystem.

### 2.3 The serial-reader path stays cell-primal

`PartitionReorderToMeshCell2Cell()` (`Mesh.cpp:91-190`) is the only path
that performs **initial partition + scatter from serial data**.  It is
hardwired cell-primal (Metis on `cell2cell`, cell partition, derived
node/bnd partitions).  We do **not** propose to change this path.
Node-primal loading will use the distributed-read path exclusively.

---

## 3. Changes Required

### 3.1 Distributed read: node-based even-split

**File:** `src/Geom/Mesh/Mesh_ReadSerializeDistributed.cpp`

`ReadDistributed_EvenSplitRead()` currently splits primary arrays by
X-slice.  For node-primal, we need to split by **node range** instead.
The H5 file stores node coordinates as a contiguous array; a simple
node-count even split (or X/Y/Z geometric partition) is sufficient.

**Work items:**

| # | Description | Est. LOC |
|---|---|---|
| N1 | Add `ReadDistributed_EvenSplitReadNodePrimal()` or a `split_by_nodes` flag to the existing function | ~20 |
| N2 | After even-split read, `cell2node` rows may reference off-rank nodes — handle via ghost pull (same pattern as cell-primal) | ~10 |

### 3.2 Node partition derivation

**File:** `src/Geom/Mesh/Mesh_ReadSerializeDistributed.cpp`

In `ReadDistributed_DeriveEntityPartitions()`, the partition-derivation
logic is currently:

```
cell partition → (Metis on cell2cellFace)
node partition → min(cell partition) per node via node2cell
bnd partition  → owner cell's partition via bnd2cell(:,0)
```

For node-primal, we invert this:

```
node partition → (Metis on node-neighbor graph via ComposeFiltered)
cell partition → min(node partition) per cell via cell2node
bnd partition  → owner cell's partition via bnd2cell(:,0)
```

The node-neighbor graph can be built with the DSL:
`ComposeFiltered(node2cell, cell2node)` with `minShared=dim`.

**Work items:**

| # | Description | Est. LOC |
|---|---|---|
| N3 | Build node-neighbor graph via `ComposeFiltered` (or Compose with `minShared=dim` for edge-neighbor) | ~15 |
| N4 | Call ParMetis on node graph instead of cell graph | ~5 |
| N5 | Derive cell partition from node partition via `cell2node` (min node partition) | ~10 |

### 3.3 `fillRegistry()` — node-primal global mapping source

**File:** `src/Geom/Mesh/Mesh.cpp:1749-1834`

`fillRegistry()` currently tries `Cell` first for the global mapping and
assumes `Cell` always has a mapping.  In node-primal mode, `Node` is the
primary entity and `Cell` may not have a mapping until after reorder/derive.

**Work items:**

| # | Description | Est. LOC |
|---|---|---|
| N6 | Make the global-mapping source order configurable or detect which kind has a valid mapping first | ~10 |

### 3.4 `ReorderEntities()` — follow semantics

**File:** `src/Geom/Mesh/Mesh_Reorder.cpp:227-476`

`resolveFollows()` currently hardcodes: when reordering `Cell`,
automatically follows `Node` and `Bnd`.  We need the inverse: when
reordering `Node`, follow `Cell` and `Bnd`.

The follow computation uses `ComputeFollowMapFromAdj` which is already
generic — it takes a source `EntityKind` and walks adjacency to find
which entities reference it.  Adding node→cell follow should be
straightforward.

**Work items:**

| # | Description | Est. LOC |
|---|---|---|
| N7 | Add node-primal follow path in `resolveFollows()`: Node → Cell (via node2cell), Node → Bnd (via node2bnd) | ~20 |

### 3.5 Reorder hardcoded adj list

**File:** `src/Geom/Mesh/Mesh_Reorder.cpp:266-382`

The reorder code hardcodes 12 adjacency names, companion arrays, global
mapping sources, and pull-set collection per entity kind.  This works
for both cell-primal and node-primal as long as the entity kinds exist
in the mesh.  No changes needed *unless* we introduce a new entity kind.

### 3.6 Ghost specification for node-primal

**File:** `src/Geom/Mesh/MeshConnectivity_Ghost.cpp:18-36` (or caller)

`GhostSpec::defaultPrimary()` is cell-anchored.  We need a node-anchored
variant:

```cpp
static GhostSpec defaultNodePrimal(int nLayers)
{
    GhostSpec spec;
    // Ghost cells via node2cell
    spec.chains.push_back(makeChain("Node", nLayers, {Adj::Node2Cell}, "Cell"));
    // Ghost nodes via node2cell → cell2node
    spec.chains.push_back(makeChain("Node", nLayers, {Adj::Node2Cell, Adj::Cell2Node}, "Node"));
    // Ghost bnds via node2bnd
    spec.chains.push_back(makeChain("Node", nLayers, {Adj::Node2Bnd}, "Bnd"));
    return spec;
}
```

This is a purely additive change — the current `defaultPrimary()` stays.

**Work items:**

| # | Description | Est. LOC |
|---|---|---|
| N8 | Add `GhostSpec::defaultNodePrimal(nLayers)` | ~15 |

### 3.7 `fillRegistry()` — no changes needed

The current implementation already handles missing adjs gracefully via
`tryAdj` (skips null fathers) and `regMap` (skips missing global mappings
if no adj needs them).  If `Face`-related adjs are not yet built, they
are simply skipped.  This is correct for both cell-primal and node-primal.

---

## 4. What Stays Cell-Primal

| Component | Status | Rationale |
|---|---|---|
| `PartitionReorderToMeshCell2Cell()` | Unchanged | Serial-reader path; acceptable to keep cell-primal |
| `BuildSerialOut()` | Unchanged | Serial output; entity-agnostic |
| `ConstructBndMesh()` | Unchanged | Domain-specific surgery; cell→face→bnd topology |
| `BuildCell2CellFace()` | Unchanged | Trivial 1:1 cell2face→face2cell map |
| `InterpolateFace()` / `InterpolateGlobal` | Unchanged | Entity-agnostic; takes parent2node as input |
| Per-adjacency state tracking | Unchanged | `AdjIndexInfo` is entity-agnostic |

---

## 5. Implementation Order

```
Phase 1: Node-primal distributed read       (N1-N5, ~60 LOC)
    → ReadDistributed with node-based split
    → Node partition via ParMetis on node-neighbor graph
    → Derive cell/bnd partitions from node partition

Phase 2: Node-primal ghost spec             (N8, ~15 LOC)
    → GhostSpec::defaultNodePrimal()

Phase 3: fillRegistry / follow semantics    (N6-N7, ~30 LOC)
    → Node-primary global mapping source
    → resolveFollows() for node→cell/bnd

Phase 4: Test & validate
    → C++ test: node-primal read + partition + reorder + ghost
    → Python integration test
    → Verify round-trip: node-primal → cell-primal → node-primal
```

---

## 6. Test Plan

### 6.1 C++ unit tests

| Test | What it verifies |
|---|---|
| `NodePrimal: even-split read` | H5 distributed read produces correct node/cell counts per rank |
| `NodePrimal: node partition` | ParMetis on node-neighbor graph via ComposeFiltered |
| `NodePrimal: reorder entities` | `ReorderEntities(Node)` with cell/bnd follows |
| `NodePrimal: ghost tree` | `evaluateGhostTree` with `defaultNodePrimal()` on tiled grid |
| `NodePrimal: round-trip` | Node-primal reorder then cell-primal reorder recovers original |

### 6.2 Python integration test

- Load `UniformSquare_10.cgns` via `read_mesh(mode="distributed", partition="node")`
- Verify cell/node counts, ghost completeness
- Run one solver step to verify no segfault / NaN

---

## 7. Open Questions

1. **Node-neighbor vs edge-neighbor for partitioning:** In 2D, node-neighbor
   (minShared=1) includes all diagonal neighbors; edge-neighbor (minShared=2)
   gives standard face-adjacent partitioning.  Which should be the default?

2. **ParMetis node weights:** Should we weight nodes by e.g. number of
   incident cells for load-balanced node-primal partitioning?

3. **Mixed cell/node-primal in the same run:**  Should we support a mode
   where the mesh is node-primal for I/O and repartitioning, then converted
   to cell-primal for the solver?  This would require `ReorderEntities(Cell)`
   with a node-derived cell partition after the node-primal phase.
