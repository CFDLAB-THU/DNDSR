# Distributed Entity Reordering — Design Document

> **Status:** Proposal (not yet implemented). The existing
> `ReorderLocalCells` and `ReadDistributed_Redistribute` remain the
> active code paths.
>
> **Last reviewed:** 2026-04-27 against PR #6 merge (commit 13b4e7b1).
> All 4 plans re-evaluated; no fundamental changes needed. See
> "Re-evaluation Notes" section at bottom.

## Motivation

The codebase has two separate reordering mechanisms:

1. **`ReorderLocalCells`** — rank-local cell permutation for cache locality.
   Converts to global, replaces cell refs in xxx2cell, permutes cell2xxx rows,
   rebuilds ghost mappings, converts back to local.

2. **`ReadDistributed_Redistribute`** — cross-rank redistribution during
   distributed mesh read.  Uses the "father=old, son=new" ArrayTransformer
   push trick.  Converts adjacency entries to new global numbering via
   ghost-pulled lookup arrays, then transfers data between ranks.

Both follow the same logical pattern but share no code.  They are also
both hardcoded to cells.  We need a general mechanism that can:

- Reorder any entity kind (Cell, Node, Bnd, Face)
- Handle rank-local and cross-rank reorderings uniformly
- Reorder multiple entity kinds simultaneously (e.g., cells and nodes
  together during repartitioning)
- Detect all affected adjacencies automatically via the mesh registry

## Scope

### In scope

- General `ReorderEntities` method on `UnstructuredMesh`
- Multi-entity reordering (e.g., Cell + Node in one call)
- Local-only fast path (no MPI when all entities stay on same rank)
- Dedicated `PermutationTransfer` utility for push/pull index construction
- Registry-based adjacency discovery (`fillRegistry`)
- Companion array handling (same row layout as the reordered entity)

### Out of scope (left to caller)

- Ghost layer rebuild (caller calls `BuildGhostPrimary`, etc. after)
- Global-to-local conversion (caller calls `AdjGlobal2Local*` after)
- Partition computation (Metis, ParMetis, etc. — separate concern)
- Face/edge reconstruction after cell/node reorder

## Key Design Decisions

### Decision 1: Two-level permutation vs arbitrary entity reordering

**Question:** Should we allow arbitrary combinations of entity reorderings,
or restrict to a "primary" two-level pattern (Cell + Node, with Face/Edge
fully reconstructed)?

**Answer:** Support arbitrary combinations, but document the common patterns.

Rationale:
- Cell-only reorder (local partitioning) is already needed
- Cell + Node reorder (repartitioning) is needed for distributed read
- Node-only reorder is conceivable (e.g., RCM on the node graph)
- Face reorder would be needed if face-based solvers are added
- Restricting to two levels would require special-casing or
  re-implementing when a new pattern emerges

The common patterns are:
1. **Cell-only local** — cache-locality reorder (current `ReorderLocalCells`)
2. **Cell + Node + Bnd** — repartitioning (current `ReadDistributed_Redistribute`)
3. **All entities** — full mesh migration

For pattern (2), faces are destroyed before redistribution and rebuilt
after.  This remains the recommended approach: reorder the "primary"
entities (Cell, Node, Bnd), then reconstruct derived entities
(Face, Edge) from scratch.  The framework supports reordering faces
directly, but it is not the expected usage.

### Decision 2: Reorder map representation

Each reordered entity needs a map from old slot to new identity.

```cpp
/// Per-entity reorder specification.
struct EntityReorderMap
{
    EntityKind kind;

    /// For each father slot i: (new_rank, new_global_index).
    /// Size must equal the father size of the canonical array for this kind.
    /// For local-only reorder: new_rank == mpi.rank for all i.
    std::vector<std::pair<MPI_int, index>> map;
};
```

Alternatively, the reverse map (for each new slot: old identity) is useful
for the push-based transfer.  However, the forward map is more natural
for the caller (Metis/ParMetis produce partition assignments indexed by
current local entity).  The implementation can derive push indices from
the forward map.

### Decision 3: MPI collectivity for local-only detection

```cpp
bool localOnly = true;
for (auto &[rank, gidx] : entityMap.map)
    if (rank != mpi.rank) { localOnly = false; break; }
// Must agree across all ranks (asymmetric decisions cause MPI hangs)
int globalLocalOnly;
MPI_Allreduce(&localOnly, &globalLocalOnly, 1, MPI_INT, MPI_LAND, mpi.comm);
```

When all ranks agree it is local-only, skip all MPI communication in the
data transfer step.  Use in-place `PermuteRows` + direct entry replacement.

### Decision 4: Ghost rebuild is the caller's responsibility

After `ReorderEntities`:
- All adjacencies are in `Adj_PointToGlobal` state
- Ghost mappings (`pLGhostMapping`) on reordered entities are stale/null
- Ghost mappings on non-reordered entities that *target* reordered entities
  are stale

The caller must rebuild ghosts.  This is correct because:
- Ghost spec may differ (multi-layer, different adjacency set)
- Ghost rebuild is heavyweight and should be explicit
- The reorder may be an intermediate step before other transforms

## Algorithm

### Input

```
ReorderEntities(
    mesh,
    reorderMaps: vector<EntityReorderMap>,   // one per entity to reorder
    registry: MeshConnectivity               // from fillRegistry()
)
```

### Precondition

All adjacencies must be in `Adj_PointToGlobal` state.  (The caller
converts to global before calling.)

### Step 1: Classify adjacencies

For each registered adjacency A→B in the registry:

| A reordered? | B reordered? | Category | Action |
|---|---|---|---|
| yes | no | SOURCE_ONLY | relocate rows |
| no | yes | TARGET_ONLY | update entries |
| yes | yes | BOTH | relocate rows + update entries |
| no | no | NONE | skip |
| yes, A==B | — | SELF | relocate rows + update entries |

Also classify companion arrays (same row layout as a reordered entity):
`cellElemInfo` companions Cell, `bndElemInfo` companions Bnd,
`faceElemInfo` companions Face, `coords` companions Node.
Periodic-bits arrays (`cell2nodePbi`, etc.) companion their parent adj.
`*Orig` arrays (`cell2cellOrig`, etc.) companion their entity.

### Step 2: Build lookup arrays (old → new global)

For each reordered entity kind E:

1. Create `tAdj1Pair lookupE` where `lookupE(i, 0)` = new global index
   for father slot `i`.
2. `createFatherGlobalMapping()` on `lookupE`.
3. Collect all old-global E references from TARGET_ONLY and BOTH
   adjacencies (entries that point to E from non-E sources).
4. Ghost-pull `lookupE` so every rank can resolve any referenced E global.

This is the same pattern as `ReorderLocalCells` uses for `cellOld2NewArr`.

For local-only reorder, step 3-4 can be skipped — all lookups are local.

### Step 3: Update entries in TARGET_ONLY and BOTH adjacencies

For each adj X→E where E is reordered:
```
for each row i in [0, adj.father->Size()):
    for each entry j:
        old_global = adj(i, j)
        new_global = lookupE.search_indexAppend(old_global) → lookupE(val, 0)
        adj(i, j) = new_global
```

For SELF adjacencies (E→E), update entries first, then relocate rows.

If the adj also has a son (ghost data), pull the updated entries after
all father entries are rewritten:
```
adj.trans.pullOnce()   // if ghost comm exists
```

For local-only reorder, the lookup is a direct local array access.

### Step 4: Relocate rows in SOURCE_ONLY, BOTH, and SELF adjacencies

For each adj E→X where E is reordered:

**Local-only path:**
```
PermuteRows(adj, NumE(), old2newLocal)
```

**Distributed path:**
Uses `PermutationTransfer` (see below):
```
auto transfer = PermutationTransfer::fromForwardMap(entityMap, globalMapping, mpi);
transfer.execute(adj.father, mpi);
// adj.father is now the redistributed array
```

Same for companion arrays.

### Step 5: Rebuild global mappings

For each reordered entity E:
```
canonicalPair.father->createGlobalMapping()   // new contiguous global numbering
```

Other pairs for the same entity borrow via `pLGlobalMapping = canonical->pLGlobalMapping`.

### Step 6: Update idx states

For all affected adjacencies:
```
adj.idx.markGlobal()   // state = Adj_PointToGlobal
// Target mapping is now stale — clear it
// (caller will re-wire after ghost rebuild)
```

Clearing stale wiring: We need a `unwireTargetMapping()` or simply allow
`wireTargetMapping` to overwrite from Global state (which it already does).
The existing `wireTargetMapping` precondition is `_state != Adj_PointToLocal`,
so rewiring from Global is already permitted.

### Post-condition

- All adjacencies in `Adj_PointToGlobal`
- Father arrays for reordered entities have new row layout
- Entries pointing to reordered entities use new global indices
- Ghost mappings are stale (caller rebuilds)
- Global mappings are fresh

## `PermutationTransfer` — Dedicated Utility

### Motivation

Both the serial-read distribution and the distributed reorder need to:
1. Convert a forward map (old_slot → new_rank) into push indices (CSR)
2. Compute new global numbering (prefix sums across ranks)
3. Transfer array data between ranks

This is currently done by ad-hoc helpers in `Mesh_PartitionHelpers.hpp`
(`Partition2LocalIdx`, `Partition2Serial2Global`, `TransferDataSerial2Global`,
`ConvertAdjSerial2Global`).  These should be unified into a reusable tool.

### API

```cpp
/// Encapsulates a distributed permutation: moving rows of arrays
/// between ranks according to a forward map.
struct PermutationTransfer
{
    /// Per father slot: target rank.
    std::vector<MPI_int> targetRanks;

    /// New global index for each father slot.
    std::vector<index> newGlobalIndices;

    /// Push-mode CSR indices (derived from targetRanks).
    std::vector<index> pushIndex;      // flat, local indices to push
    std::vector<index> pushStart;      // [nRanks+1] prefix sums

    /// Whether this is a rank-local permutation (no MPI needed).
    bool isLocalOnly;

    /// Build from a forward map: for each father slot, (new_rank, new_global).
    static PermutationTransfer fromForwardMap(
        const std::vector<std::pair<MPI_int, index>> &forwardMap,
        const ssp<GlobalOffsetsMapping> &oldGlobalMapping,
        const MPIInfo &mpi);

    /// Build from partition assignment only (new globals computed automatically).
    /// This is the pattern used by serial-read distribution.
    static PermutationTransfer fromPartition(
        const std::vector<MPI_int> &partition,
        const ssp<GlobalOffsetsMapping> &oldGlobalMapping,
        const MPIInfo &mpi);

    /// Transfer array data: moves rows from old father to new father.
    /// After return, pair.father contains the redistributed data.
    /// Uses the "father=old, son=new" ArrayTransformer trick for
    /// distributed transfers, or PermuteRows for local-only.
    template <class TPair>
    void transferRows(TPair &pair, const MPIInfo &mpi) const;

    /// Build a ghost-pullable lookup array for old→new global conversion.
    /// The returned pair has: lookup(i, 0) = newGlobalIndices[i].
    /// Ghost-pulled for all globals in pullSet.
    tAdj1Pair buildLookup(
        const std::vector<index> &pullSet,
        const ssp<GlobalOffsetsMapping> &oldGlobalMapping,
        const MPIInfo &mpi) const;
};
```

### Usage in existing code

**Serial-read distribution** (`PartitionReorderToMeshCell2Cell`):
```cpp
auto cellTransfer = PermutationTransfer::fromPartition(cellPartition, cellGlobal, mpi);
auto nodeTransfer = PermutationTransfer::fromPartition(nodePartition, nodeGlobal, mpi);

// Update cell2node entries: old node globals → new node globals
auto nodeLookup = nodeTransfer.buildLookup(referencedNodeGlobals, nodeGlobal, mpi);
ConvertAdjEntries(cell2node, ..., [&](index g) { return nodeLookup.resolve(g); });

// Transfer rows
cellTransfer.transferRows(cell2node, mpi);
cellTransfer.transferRows(cellElemInfo, mpi);
nodeTransfer.transferRows(coords, mpi);
```

**Local cell reorder** (`ReorderLocalCells`):
```cpp
auto cellTransfer = PermutationTransfer::fromForwardMap(cellForwardMap, cellGlobal, mpi);
assert(cellTransfer.isLocalOnly);  // all ranks local

auto cellLookup = cellTransfer.buildLookup(referencedCellGlobals, cellGlobal, mpi);
// Update face2cell, node2cell, bnd2cell entries
// Permute cell2node, cell2face, cellElemInfo rows
```

## Multi-Entity Reordering

### Problem

When repartitioning, both cells and nodes move between ranks.  The order
matters:

1. Node indices in `cell2node` must be updated to new node globals
   **before** `cell2node` rows are relocated (otherwise the entries
   are stale after relocation).

2. Cell indices in `bnd2cell` must be updated to new cell globals.

3. Both `cell2node` and `bnd2node` have entries pointing to nodes
   (update needed) and rows belonging to cells/bnds (relocate needed).

### Ordering rule

For an adjacency A→B:
- If B is reordered: **update entries first** (before A is relocated)
- If A is reordered: **relocate rows second** (after B entries are updated)

When both A and B are reordered:
- Update entries (B's new globals) first
- Relocate rows (A's new layout) second

This is safe because entry update is a value-level operation (replacing
global indices) while row relocation is a structural operation (moving
rows between ranks).  The entry update reads old globals and writes new
globals; the row relocation moves the already-updated rows.

### Algorithm for multi-entity reorder

```
ReorderEntities(mesh, maps: {Cell→mapC, Node→mapN, Bnd→mapB}, registry):

  1. Build all lookup arrays:
     cellLookup = mapC.buildLookup(...)
     nodeLookup = mapN.buildLookup(...)
     bndLookup  = mapB.buildLookup(...)

  2. Update all entries (order doesn't matter between kinds):
     // cell2node entries: node globals → new node globals
     // bnd2node entries:  node globals → new node globals
     // bnd2cell entries:  cell globals → new cell globals
     // node2cell entries: cell globals → new cell globals (if N2CB exists)
     // node2bnd entries:  bnd globals → new bnd globals  (if N2CB exists)
     // cell2cell entries: cell globals → new cell globals

  3. Relocate all rows (order doesn't matter between kinds):
     // cell2node rows: move with cell
     // cell2cell rows: move with cell
     // cellElemInfo rows: move with cell
     // bnd2node rows: move with bnd
     // bnd2cell rows: move with bnd
     // bndElemInfo rows: move with bnd
     // coords rows: move with node
     // node2cell rows: move with node (if exists)
     // node2bnd rows: move with node (if exists)

  4. Rebuild global mappings for Cell, Node, Bnd

  5. Destroy derived adjacencies (face2cell, face2node, cell2face, etc.)
     — caller will reconstruct from the redistributed primary data
```

### Face handling

Faces are derived from cells and nodes.  After Cell + Node + Bnd
reorder, face adjacencies are meaningless (face globals no longer
correspond to the right cells/nodes).  The recommended pattern:

1. Destroy facial adjacencies before reorder
2. Reorder primary entities (Cell, Node, Bnd)
3. Rebuild: `RecoverNode2CellAndNode2Bnd` → `RecoverCell2CellAndBnd2Cell` →
   `BuildGhostPrimary` → `InterpolateFace` → etc.

This avoids reordering faces at all.  If face reordering is ever needed
(e.g., face-based solver with face-local partitioning), the framework
supports it — just add Face to the reorder maps.

## Implementation Plan

### Phase 1: `PermutationTransfer` utility

- Extract `Partition2LocalIdx`, `Partition2Serial2Global`,
  `TransferDataSerial2Global` into a standalone `PermutationTransfer`
  struct in `src/DNDS/PermutationTransfer.hpp`
- Add `fromForwardMap` and `fromPartition` factory methods
- Add `transferRows` (dispatches to local PermuteRows or distributed
  ArrayTransformer push)
- Add `buildLookup` for old→new global conversion
- Unit tests in `test/cpp/`

### Phase 2: `fillRegistry` on UnstructuredMesh

- Implement `fillRegistry(MeshConnectivity&, set<AdjKind> skip)` (Plan 2)
- Canonical global mapping sources documented

### Phase 3: Single-entity `ReorderEntity`

- Implement the classify/lookup/update/relocate/rebuild pipeline
  for a single entity kind
- Migrate `ReorderLocalCells` to use it (local-only path)
- Test: verify local cell reorder produces identical results

### Phase 4: Multi-entity `ReorderEntities`

- Extend to accept multiple `EntityReorderMap`s
- Implement the multi-entity ordering rule (update entries first,
  relocate rows second)
- Migrate `ReadDistributed_Redistribute` to use it
- Test: verify distributed mesh read produces identical results

### Phase 5: Cleanup

- Remove or deprecate the old ad-hoc helpers in `Mesh_PartitionHelpers.hpp`
- Remove redundant code in `ReorderLocalCells` and
  `ReadDistributed_Redistribute` (they become thin wrappers)

---

## Re-evaluation Notes (2026-04-27, post PR #6)

Reviewed against the merged PR #6 changes (InterpolateGlobal, face pipeline
split, templatization, pybind11 bindings, file reorganization into
`src/Geom/Mesh/`).

### What changed

1. **`InterpolateFace` split** into `InterpolateFace` + `BuildGhostFace` +
   `MatchFaceBoundary`. Ghost faces now via `evaluateGhostTree(Cell ->
   Cell2Face -> Face)`. This creates a 5th `registerAdj` call site.

2. **`InterpolateGlobal`** added: N-parent distributed interpolation with
   pbi, ownership resolution, and template `e2p_rs` (face2cell uses rs=2).
   `parent2entityPbi` computed locally (no MPI push needed).

3. **File reorganization**: All mesh source moved to `src/Geom/Mesh/`.
   `MeshConnectivity.cpp` created for non-template implementations.

4. **`AdjPairTracked_bind.hpp`** added: Python bindings for tracked pairs
   via pybind11 inheritance from base `ArrayAdjacencyPair`.

5. **Implementation plan updated** (1786 lines): Sections 9 (templatization)
   and 10 (per-adj state tracking) with detailed migration plans.

### Impact on Plan 1 (makeFatherOnlyMapping)

**No changes.** `AdjIndexInfo.hpp` is unchanged. `wireTargetMapping` still
accepts null. The factory and non-null assertion are still needed.

### Impact on Plan 2 (fillRegistry)

**Minor adjustment.** Now 5 call sites (was 4). The new `BuildGhostFace`
site borrows `cell2face`'s global mapping from `cell2node`, confirming the
need to document canonical global mapping sources per entity kind.

### Impact on Plan 3 (device view)

**Confirmed.** The inheritance pattern used for `AdjPairTracked` and
`AdjPairTracked_bind.hpp` validates the device view inheritance approach.
The `AdjStateView` struct for the mesh device view remains the pragmatic
choice (avoids changing 12 member type declarations).

### Impact on Plan 4 (distributed reordering)

**Mostly unchanged.** Key updates:

- **Face reconstruction** after reorder should use the modular
  `InterpolateFace -> BuildGhostFace -> MatchFaceBoundary` pipeline
  (which internally uses `InterpolateGlobal`), not `InterpolateFaceLegacy`.

- **`parent2entityPbi`** is computed locally from `cell2nodePbi` +
  `entity2nodePbi` — no need to preserve across reorder. Destroyed with
  faces, recomputed after.

- **`PermutationTransfer`** should handle `AdjVariant` via `std::visit`
  for type-erased transfer (same pattern as `evaluateGhostTree`).

- The list of arrays destroyed before reorder and reconstructed after
  should include: `cell2face`, `face2cell`, `face2node`, `face2bnd`,
  `bnd2face`, `faceElemInfo`, `face2nodePbi`, `cell2facePbi` (when
  periodic), and any `parent2entityPbi` stored on the mesh.
