# Distributed Entity Reordering — Design Document (v2)

> **Status:** Detailed design, not yet implemented.
>
> **Supersedes:** The original v1 design (same file, git history).
> v2 adds: placement-follow semantics, automatic adj conversion rules
> with formal classification, `PermutationTransfer` internal design,
> companion propagation, and full integration with `AdjPairTracked` /
> `fillRegistry` / checked-wrapper infrastructure.
>
> **Last updated:** 2026-04-28.

## Table of Contents

1. [Motivation](#motivation)
2. [Scope](#scope)
3. [Concepts and Terminology](#concepts-and-terminology)
4. [Entity Dependency Graph and Propagation](#entity-dependency-graph-and-propagation)
5. [Adjacency Conversion Rules](#adjacency-conversion-rules)
6. [PermutationTransfer Utility](#permutationtransfer-utility)
7. [ReorderEntities — Top-Level Algorithm](#reorderentities--top-level-algorithm)
8. [Companion Array Handling](#companion-array-handling)
9. [Integration with AdjPairTracked and fillRegistry](#integration-with-adjpairtracked-and-fillregistry)
10. [Local-Only Fast Path](#local-only-fast-path)
11. [Concrete Use Cases](#concrete-use-cases)
12. [Implementation Plan](#implementation-plan)
13. [Appendix: Re-evaluation Notes (v1)](#appendix-re-evaluation-notes-v1)

---

## Motivation

The codebase has two separate reordering mechanisms:

1. **`ReorderLocalCells`** — rank-local cell permutation for cache locality.
   Converts to global, replaces cell refs in xxx2cell, permutes cell2xxx
   rows, rebuilds ghost mappings, converts back to local.

2. **`ReadDistributed_Redistribute`** — cross-rank redistribution during
   distributed mesh read. Uses the "father=old, son=new" ArrayTransformer
   push trick. Converts adjacency entries to new global numbering via
   ghost-pulled lookup arrays, then transfers data between ranks.

Both follow the same logical pattern but share no code. They are also
both hardcoded to specific entity kinds. We need a general mechanism
that can:

- Reorder any entity kind (Cell, Node, Bnd, Face)
- Handle rank-local and cross-rank reorderings uniformly
- Reorder multiple entity kinds simultaneously with automatic follow
  propagation (e.g., cell reorder makes node placement follow)
- Detect all affected adjacencies and classify the required conversion
  (entry remapping, row relocation, or both) automatically via the
  mesh registry

## Scope

### In scope

- General `ReorderEntities` method on `UnstructuredMesh`
- Multi-entity reordering with **explicit** and **follow** maps
- Automatic adjacency conversion: for each adj A→B, determine whether
  entries need remapping, rows need relocation, or both
- `PermutationTransfer` utility for both local and distributed transfer
- Companion array handling (same row layout as a reordered entity)
- Local-only fast path (no MPI when all entities stay on same rank)
- Registry-based adjacency discovery via `fillRegistry`
- Integration with `AdjPairTracked` idx state

### Out of scope (left to caller)

- Ghost layer rebuild (caller calls `BuildGhostPrimary`, etc. after)
- Global-to-local conversion (caller calls `AdjGlobal2Local*` after)
- Partition computation (Metis, ParMetis, etc. — separate concern)
- Face/edge reconstruction after cell/node reorder

---

## Concepts and Terminology

### Reorder map

A per-entity specification that says where each owned entity goes:

```cpp
struct EntityReorderMap
{
    EntityKind kind;
    /// For each father slot i: target rank after reorder.
    /// Size == father size of the canonical array for this kind.
    std::vector<MPI_int> targetRanks;
};
```

For local-only reorder: `targetRanks[i] == mpi.rank` for all i. The new
local ordering is determined by a separate permutation vector (see below).

For distributed reorder: `targetRanks[i]` may differ from `mpi.rank`.
New global indices are computed by `PermutationTransfer` (contiguous
prefix-sum across ranks).

### Explicit vs follow reorder

- **Explicit reorder**: the caller provides an `EntityReorderMap` directly.
  The caller chose this entity's new placement. Example: Metis assigns
  each cell to a rank.

- **Follow reorder**: the entity is not directly reordered by the caller.
  Instead, its placement is **derived** from an explicitly-reordered entity.
  Example: when cells are redistributed, each node follows the cell that
  "owns" it (e.g., lowest-rank cell referencing the node). The framework
  computes the follow map automatically.

Both explicit and follow maps produce the same `EntityReorderMap` struct
internally. The distinction is in how the map is produced, not how it is
consumed.

### Canonical array

For each `EntityKind`, one array pair serves as the "canonical" source of:
- Father size (= number of local entities of that kind)
- `pLGlobalMapping` (global offset mapping for this entity)

Current canonicals:
| Kind | Canonical pair | Fallback |
|------|---------------|----------|
| Cell | `cell2node` | `cell2cell`, `cell2face` |
| Node | `coords` | — |
| Bnd  | `bnd2node` | `bnd2cell`, `bnd2face` |
| Face | `face2node` | `face2cell`, `face2bnd` |

The canonical array is what `fillRegistry` sources `globalMappings[kind]`
from. After reorder, the canonical array's global mapping is rebuilt first,
and other arrays for the same kind borrow it.

### Adjacency role classification

For a specific reorder operation, each registered adjacency `A→B` falls
into exactly one category:

| A reordered? | B reordered? | A == B? | Category | Actions needed |
|:---:|:---:|:---:|---|---|
| no  | no  | —   | `SKIP`         | nothing |
| yes | no  | no  | `RELOCATE`     | relocate rows (A moves) |
| no  | yes | no  | `REMAP`        | remap entries (B indices change) |
| yes | yes | no  | `RELOCATE_REMAP` | remap entries, then relocate rows |
| yes | yes | yes | `SELF`         | remap entries, then relocate rows |

"Relocate" = move rows between ranks (or permute locally).
"Remap" = replace entry values with new global indices.

### Companion array

An array whose rows are parallel to an entity kind but is not an
adjacency. Examples:
- `cellElemInfo` companions Cell
- `bndElemInfo` companions Bnd
- `faceElemInfo` companions Face
- `coords` companions Node
- `cell2nodePbi` companions Cell (same row count, parallel to `cell2node`)
- `*Orig` arrays (`cell2cellOrig`, `node2nodeOrig`, `bnd2bndOrig`) companion
  their entity kind

Companions need row relocation when their entity is reordered, but have
no entries to remap (they don't store entity indices — they store
element types, coordinates, periodic bits, etc.).

---

## Entity Dependency Graph and Propagation

### The entity reference graph

Each registered adjacency A->B means "A-entities store B-entity
global indices in their entries". The full reference graph for a
typical built mesh:

```
Cell --cell2node--> Node <--bnd2node-- Bnd
  |                   ^                  |
  |--cell2face-->  Face --face2node--+   |
  |                  |                   |
  |                face2cell --> Cell     |--bnd2cell--> Cell
  |                face2bnd --> Bnd      |
  |--cell2cell--> Cell (self)            |
                                         |
Node --node2cell--> Cell (support)    Bnd --bnd2face--> Face
Node --node2bnd-->  Bnd  (support)
```

### Follow propagation rules

When entity A is explicitly reordered, other entities may need to
**follow** (derive their own reorder map from A's).

**Rule 1: Downward follow (referenced entity follows referencing entity)**

When a "primary" entity is redistributed, referenced "secondary"
entities should follow to maintain locality. The follow assignment for
entity B is derived from A->B (or equivalently, from B->A support):

```
for each local B entity b:
    candidateRanks = { A.targetRank : A references b via A->B }
    b.targetRank = min(candidateRanks)
```

Using min is deterministic and consistent with the existing
`ReadDistributed_DeriveEntityPartitions` implementation.

Typical follow chains:
- Cell explicitly reordered -> Node follows (via cell2node / node2cell)
- Cell explicitly reordered -> Bnd follows (via bnd2cell, bnd goes to
  its owner cell's rank)

**Rule 2: No upward follow**

Reordering nodes does NOT automatically reorder cells. Support
adjacencies (node2cell, node2bnd) have their entries remapped but
their source entity does not follow. If the caller wants both Cell
and Node reordered, both must be in the explicit set.

**Rule 3: Derived entities are destroyed, not followed**

Face entities are derived from Cell + Node. After a Cell or Node
reorder, face adjacencies become invalid. The recommended pattern:
destroy face-related arrays before reorder and reconstruct after.

### Follow computation

```
ComputeFollowMap(mesh, explicitMap_A, adjKind_B2A, mpi):
    // Need B->A support. Either use an existing support (node2cell)
    // or invert A->B on the fly.

    1. Ghost-pull A's targetRanks array:
       - Create tAdj1Pair lookupA with lookupA(i,0) = targetRanks[i]
       - createFatherGlobalMapping + ghost-pull for all A globals
         referenced by B->A entries.

    2. For each local B entity b:
       minRank = INT_MAX
       for each a in B->A[b]:
           rank = lookupA.resolve(a)  // local or ghost
           minRank = min(minRank, rank)
       followMap[b] = minRank

    3. Return EntityReorderMap{kind_B, followMap}
```

### Which entities can follow which?

| Explicit | Follow | Via | Assignment rule |
|----------|--------|-----|-----------------|
| Cell | Node | node2cell (support) | min rank of referencing cells |
| Cell | Bnd | bnd2cell | rank of bnd's owner cell (slot 0) |
| (rare) Node | -- | -- | no natural follower |
| (rare) Face | -- | -- | no natural follower |

### FollowSpec struct

```cpp
struct FollowSpec
{
    EntityKind follower;       // entity kind to derive map for
    EntityKind leader;         // explicit-map entity kind to follow
    AdjKind follower2leader;   // support adj: follower -> leader
    // Assignment: follower goes to min(leader.targetRank) over
    // all leaders referencing this follower entity.
};
```

### Order of operations

1. Caller provides explicit `EntityReorderMap`s.
2. Framework computes follow maps using `ComputeFollowMap`.
3. All maps (explicit + follow) are merged into a uniform set.
4. From this point, the main reorder algorithm treats all maps
   identically.

---

## Adjacency Conversion Rules

This is the core of the design. Given a set of entity kinds being
reordered (explicit + follow), every registered adjacency must be
classified and converted.

### Classification algorithm

```cpp
enum class AdjAction { SKIP, RELOCATE, REMAP, RELOCATE_REMAP, SELF };

AdjAction classifyAdj(AdjKind adj, const set<EntityKind> &reordered)
{
    bool fromReordered = reordered.count(adj.from);
    bool toReordered   = reordered.count(adj.to);

    if (adj.isIntraLevel())  // A == A (e.g. cell2cell)
        return fromReordered ? AdjAction::SELF : AdjAction::SKIP;

    if (!fromReordered && !toReordered) return AdjAction::SKIP;
    if (fromReordered  && !toReordered) return AdjAction::RELOCATE;
    if (!fromReordered && toReordered)  return AdjAction::REMAP;
    return AdjAction::RELOCATE_REMAP;  // both reordered
}
```

### What each action does

**SKIP**: Nothing. The adjacency's entries are still valid, its rows
are in the right place. Example: `face2node` when only Face is
NOT reordered and Node is NOT reordered.

**RELOCATE** (source reordered, target not): Rows of the adjacency
must be moved to match the new row layout of the source entity.
Entries remain unchanged (they still point to valid global indices
of the non-reordered target). Example: `cell2node` when only Cell
is reordered (not Node). Cell rows move; node globals in the
entries are still correct.

**REMAP** (target reordered, source not): Entries must be replaced
with new global indices of the target entity. Rows stay in place.
Example: `face2cell` when only Cell is reordered. Face rows don't
move, but cell globals in the entries must be updated.

**RELOCATE_REMAP** (both reordered): First remap entries (target's
new globals), then relocate rows (source moves). Order matters:
remap before relocate. Example: `cell2node` when both Cell and
Node are reordered. Node indices are remapped first, then cell
rows are relocated.

**SELF** (intra-level, e.g. cell2cell): Same as RELOCATE_REMAP
but source == target. Entries point to the same entity kind as
the rows. Remap entries first (replace old cell globals with new
cell globals), then relocate rows (move to new cell layout).

### Concrete classification for common reorder patterns

#### Pattern 1: Cell-only local reorder

Reordered: {Cell}

| Adjacency | from | to | Action |
|-----------|------|----|--------|
| cell2node | Cell | Node | RELOCATE |
| cell2cell | Cell | Cell | SELF |
| cell2face | Cell | Face | RELOCATE |
| bnd2cell | Bnd | Cell | REMAP |
| face2cell | Face | Cell | REMAP |
| node2cell | Node | Cell | REMAP |
| bnd2node | Bnd | Node | SKIP |
| face2node | Face | Node | SKIP |
| node2bnd | Node | Bnd | SKIP |
| bnd2face | Bnd | Face | SKIP |
| face2bnd | Face | Bnd | SKIP |

This matches what `ReorderLocalCells` does today:
- Replaces cell indices in face2cell, node2cell, bnd2cell, cell2cell
  (= REMAP + SELF entry part)
- Permutes rows of cell2node, cell2cell, cell2face, cellElemInfo
  (= RELOCATE + SELF row part)

#### Pattern 2: Cell + Node + Bnd redistribution

Reordered: {Cell, Node, Bnd}

| Adjacency | from | to | Action |
|-----------|------|----|--------|
| cell2node | Cell | Node | RELOCATE_REMAP |
| cell2cell | Cell | Cell | SELF |
| bnd2node | Bnd | Node | RELOCATE_REMAP |
| bnd2cell | Bnd | Cell | RELOCATE_REMAP |
| node2cell | Node | Cell | RELOCATE_REMAP |
| node2bnd | Node | Bnd | RELOCATE_REMAP |
| face2* | Face | * | destroyed before reorder |

This matches what `ReadDistributed_Redistribute` does:
- Converts node indices in cell2node, bnd2node (= REMAP part)
- Transfers cell2node/cellElemInfo rows with cell, bnd2node/bndElemInfo
  with bnd, coords with node (= RELOCATE part)

### Ordering constraint within a single adjacency

For RELOCATE_REMAP and SELF:

```
1. REMAP entries  (replace old target globals with new target globals)
2. RELOCATE rows  (move rows to new source layout)
```

This ordering is safe because:
- REMAP reads old target globals and writes new target globals. This is
  a value-level operation that does not change row structure.
- RELOCATE moves rows (including the already-remapped entries) to new
  positions. It is a structural operation.
- Doing RELOCATE first would scatter rows to new ranks before entries
  are remapped, making it impossible to resolve old target globals
  (the lookup arrays are indexed by old globals).

### Ordering between adjacencies

All REMAPs can proceed in parallel (they read from lookup arrays and
write to different adjacency arrays). All RELOCATEs can proceed in
parallel (they move rows of different arrays independently).

The only constraint: all REMAPs for a given entity kind must complete
before any RELOCATE that moves rows for that same kind. But since
REMAP and RELOCATE operate on different adjacencies (REMAP targets
adjacencies where the entity is the *target*, RELOCATE targets
adjacencies where the entity is the *source*), they never conflict.

Therefore the global ordering is simply:

```
Phase 1: REMAP all entries (all adjacencies, all entity kinds)
Phase 2: RELOCATE all rows (all adjacencies, all entity kinds)
```

### Ghost data during reorder

**Precondition**: All adjacencies must be in `Adj_PointToGlobal` state.
Son arrays (ghost data) exist but may be stale after reorder.

**During REMAP**: We need to remap entries in father rows only.
Son rows (ghost data) will be rebuilt after ghost rebuild. However,
for `ReorderLocalCells` the existing code also pulls son data after
REMAP (`face2cell.trans.pullOnce()`) to keep ghost entries consistent
for the subsequent local-to-global conversion. This is an optimization
for the local-only path where ghost mappings are NOT rebuilt.

**During RELOCATE**: For the distributed path, the father=old/son=new
ArrayTransformer trick replaces the entire father. Son is discarded.
For the local-only path, `PermuteRows` operates on father only.

**Post-condition**: Son arrays are stale. Ghost mappings (`pLGhostMapping`)
on reordered entities are stale. The caller must rebuild ghosts.

---

## PermutationTransfer Utility

### Motivation

Both the serial-read distribution (`ReadDistributed_Redistribute`) and
the local cell reorder (`ReorderLocalCells`) need to:

1. Convert a partition assignment or forward map into push/permute indices
2. Compute new global numbering (prefix sums across ranks)
3. Transfer or permute array data

Currently this is done by ad-hoc helpers in `Mesh_PartitionHelpers.hpp`:
`Partition2LocalIdx`, `Partition2Serial2Global`, `TransferDataSerial2Global`,
`ConvertAdjSerial2Global`. These should be unified into a reusable tool.

### Data members

```cpp
struct PermutationTransfer
{
    /// Per father slot: target rank after reorder.
    std::vector<MPI_int> targetRanks;

    /// New global index for each father slot.
    /// Computed by prefix-sum across ranks grouped by target rank.
    std::vector<index> newGlobalIndices;

    /// Push-mode CSR indices: pushIndex[pushStart[r]..pushStart[r+1])
    /// are the local father indices that go to rank r.
    std::vector<index> pushIndex;
    std::vector<index> pushStart;   // size = nRanks + 1

    /// Local permutation: localOld2New[i] = new local index for old
    /// local index i. Only valid when isLocalOnly == true.
    /// For distributed transfers, this is empty.
    std::vector<index> localOld2New;

    /// Whether this is a pure rank-local permutation (no cross-rank).
    bool isLocalOnly{false};

    /// New global offsets: newGlobalOffsets[r] = first global index
    /// owned by rank r after reorder. Size = nRanks + 1.
    std::vector<index> newGlobalOffsets;
};
```

### Factory methods

**`fromPartition`**: Used by redistribution (ReadDistributed, ParMetis).
Caller provides only target ranks. New global indices are computed
automatically.

```cpp
static PermutationTransfer fromPartition(
    const std::vector<MPI_int> &partition,
    const ssp<GlobalOffsetsMapping> &oldGlobalMapping,
    const MPIInfo &mpi)
{
    PermutationTransfer pt;
    pt.targetRanks = partition;

    // 1. Compute push CSR (same as Partition2LocalIdx)
    Partition2LocalIdx(partition, pt.pushIndex, pt.pushStart, mpi);

    // 2. Compute new global indices (same as Partition2Serial2Global)
    Partition2Serial2Global(partition, pt.newGlobalIndices, mpi, mpi.size);

    // 3. Detect local-only
    pt.isLocalOnly = true;
    for (auto r : partition)
        if (r != mpi.rank) { pt.isLocalOnly = false; break; }
    int globalLocal;
    MPI_Allreduce(&pt.isLocalOnly, &globalLocal, 1, MPI_INT, MPI_LAND, mpi.comm);
    pt.isLocalOnly = globalLocal;

    // 4. Build local permutation if local-only
    if (pt.isLocalOnly)
    {
        // newGlobalIndices maps old local -> new global.
        // Convert to old local -> new local using offset.
        index myOffset = oldGlobalMapping->operator()(mpi.rank, 0);
        pt.localOld2New.resize(partition.size());
        for (size_t i = 0; i < partition.size(); i++)
            pt.localOld2New[i] = pt.newGlobalIndices[i] - myOffset;
    }

    // 5. Compute new global offsets
    // (count per rank, exclusive scan)
    std::vector<index> localCounts(mpi.size, 0);
    for (auto r : partition) localCounts[r]++;
    std::vector<index> totalCounts(mpi.size);
    MPI_Allreduce(localCounts.data(), totalCounts.data(),
                  mpi.size, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
    pt.newGlobalOffsets.resize(mpi.size + 1);
    pt.newGlobalOffsets[0] = 0;
    for (int r = 0; r < mpi.size; r++)
        pt.newGlobalOffsets[r + 1] = pt.newGlobalOffsets[r] + totalCounts[r];

    return pt;
}
```

**`fromLocalPermutation`**: Used by local cell reorder. Caller provides
a local old-to-new permutation. All entities stay on the same rank.

```cpp
static PermutationTransfer fromLocalPermutation(
    const std::vector<index> &old2new,
    const ssp<GlobalOffsetsMapping> &oldGlobalMapping,
    const MPIInfo &mpi)
{
    PermutationTransfer pt;
    pt.isLocalOnly = true;
    pt.localOld2New = old2new;
    pt.targetRanks.assign(old2new.size(), mpi.rank);

    index myOffset = oldGlobalMapping->operator()(mpi.rank, 0);
    pt.newGlobalIndices.resize(old2new.size());
    for (size_t i = 0; i < old2new.size(); i++)
        pt.newGlobalIndices[i] = myOffset + old2new[i];

    // Push CSR is trivial for local-only (not needed but fill for consistency)
    pt.pushStart.assign(mpi.size + 1, 0);
    pt.pushStart[mpi.rank] = 0;
    pt.pushStart[mpi.rank + 1] = old2new.size();
    // ... (fill remaining)

    // Global offsets unchanged for local-only
    pt.newGlobalOffsets.resize(mpi.size + 1);
    for (int r = 0; r <= mpi.size; r++)
        pt.newGlobalOffsets[r] = oldGlobalMapping->operator()(r, 0);

    return pt;
}
```

### Core operations

**`transferRows`**: Move rows of an array pair according to this transfer.

```cpp
template <class TPair>
void transferRows(TPair &pair, const MPIInfo &mpi) const
{
    if (isLocalOnly)
    {
        // Use PermuteRows (in-place, no MPI)
        UnstructuredMesh::PermuteRows(pair, pair.father->Size(),
            [&](index i) { return localOld2New[i]; });
    }
    else
    {
        // Distributed: father=old, son=new ArrayTransformer push trick
        // (same as TransferDataSerial2Global)
        auto oldFather = pair.father;
        using TArr = typename decltype(pair.father)::element_type;
        pair.father = make_ssp<TArr>(ObjName{"transfer.new"}, mpi);

        typename ArrayTransformerType<TArr>::Type trans;
        trans.setFatherSon(oldFather, pair.father);
        trans.createFatherGlobalMapping();
        trans.createGhostMapping(pushIndex, pushStart);
        trans.createMPITypes();
        trans.pullOnce();
        // pair.father is now the new array with transferred data
    }
}
```

**`buildLookup`**: Build a ghost-pullable old-global -> new-global
lookup array. Used for REMAP operations.

```cpp
struct LookupResult
{
    tAdj1Pair pair;           // pair(i, 0) = new global for local slot i
    // Ghost-pulled: pair.son available for off-rank lookups.

    /// Resolve an old global index to its new global index.
    index resolve(index oldGlobal) const
    {
        MPI_int rank;
        index val;
        bool found = pair.trans.pLGhostMapping->search_indexAppend(
            oldGlobal, rank, val);
        DNDS_assert(found);
        return pair(val, 0);
    }
};

LookupResult buildLookup(
    const std::vector<index> &pullSet,
    const ssp<GlobalOffsetsMapping> &oldGlobalMapping,
    const MPIInfo &mpi) const
{
    LookupResult result;
    result.pair.InitPair("reorder_lookup", mpi);
    result.pair.father->Resize(newGlobalIndices.size());
    for (index i = 0; i < (index)newGlobalIndices.size(); i++)
        result.pair(i, 0) = newGlobalIndices[i];

    result.pair.TransAttach();
    result.pair.trans.createFatherGlobalMapping();
    result.pair.trans.createGhostMapping(
        std::vector<index>(pullSet.begin(), pullSet.end()));
    result.pair.trans.createMPITypes();
    result.pair.trans.pullOnce();

    return result;
}
```

### Collecting the pull set

Before `buildLookup`, we need to know which old-global indices of
entity E are referenced by non-E adjacencies (for REMAP). This set
is collected by scanning all REMAP and RELOCATE_REMAP adjacencies:

```
collectPullSet(entityKind E, registry, mesh, mpi):
    pullSet = {}
    for each adj A->B in registry where B == E and A != E:
        for each row i in adj.father:
            for each entry j:
                global = adj(i, j)
                if global != UnInitIndex and not owned by this rank:
                    pullSet.insert(global)
    // Also from SELF adjacencies (E->E):
    for each adj E->E:
        for each row i in adj.father:
            for each entry j:
                global = adj(i, j)
                if global != UnInitIndex and not owned:
                    pullSet.insert(global)
    return sorted_unique(pullSet)
```

For the local-only path, the pull set only includes globals from
ghost (son) data — entries in father rows that point to off-rank
entities of the same kind.

### Relationship to existing helpers

| Existing helper | PermutationTransfer equivalent |
|-----------------|-------------------------------|
| `Partition2LocalIdx` | `fromPartition` (push CSR computation) |
| `Partition2Serial2Global` | `fromPartition` (new global computation) |
| `TransferDataSerial2Global` | `transferRows` (distributed path) |
| `ConvertAdjSerial2Global` | `buildLookup` + `ConvertAdjEntries` |
| `PermuteRows` (in ReorderLocalCells) | `transferRows` (local path) |
| `cellOld2NewArr` build + ghost-pull | `buildLookup` |

After migration, the existing helpers become dead code.

---

## ReorderEntities -- Top-Level Algorithm

### API

```cpp
struct ReorderInput
{
    /// Explicit reorder maps (caller-provided).
    std::vector<EntityReorderMap> explicitMaps;

    /// Follow specifications (framework computes follow maps from these).
    std::vector<FollowSpec> follows;

    /// Entity kinds whose adjacencies should be destroyed before reorder
    /// (not reordered, not remapped -- just wiped). Typically {Face}.
    std::unordered_set<EntityKind> destroyKinds;
};

/// Post-condition: all adj in Adj_PointToGlobal, ghost mappings stale.
void UnstructuredMesh::ReorderEntities(const ReorderInput &input);
```

### Precondition

- All adjacencies in `Adj_PointToGlobal` state.
- For the local-only path: adjacencies may be in `Adj_PointToLocal`;
  the method converts to global first, then back to local after.
  (Or: caller converts beforehand. TBD.)

### Full algorithm

```
ReorderEntities(input):

  // ============================================================
  // Step 0: Validate and prepare
  // ============================================================

  0a. Assert all adj in Adj_PointToGlobal (or convert if local).
  0b. Build registry: fillRegistry(dag, skip=destroyKinds' adjs).
  0c. Destroy adjacencies for destroyKinds:
      for kind in input.destroyKinds:
          for each adj where adj.from == kind or adj.to == kind:
              adj.father.reset(); adj.son.reset();
              adj.idx = AdjIndexInfo{};  // back to Adj_Unknown

  // ============================================================
  // Step 1: Compute follow maps
  // ============================================================

  1a. For each FollowSpec in input.follows:
      followMap = ComputeFollowMap(mesh, explicitMaps[spec.leader],
                                   spec.follower2leader, mpi)
      allMaps[spec.follower] = followMap

  1b. Merge explicit maps into allMaps.
      (Explicit maps take precedence over follow maps if both exist
       for the same entity kind -- but this should not happen.)

  1c. reorderedKinds = set of entity kinds in allMaps.

  // ============================================================
  // Step 2: Build PermutationTransfer per entity kind
  // ============================================================

  for each kind in reorderedKinds:
      transfers[kind] = PermutationTransfer::fromPartition(
          allMaps[kind].targetRanks,
          dag.getGlobalMapping(kind),
          mpi)

  // ============================================================
  // Step 3: Classify all registered adjacencies
  // ============================================================

  for each (adjKind, adjVariant) in dag.adjRegistry:
      action = classifyAdj(adjKind, reorderedKinds)
      classified[adjKind] = action

  // ============================================================
  // Step 4: Build lookup arrays for REMAP
  // ============================================================

  for each kind in reorderedKinds:
      pullSet = collectPullSet(kind, dag, classified, mpi)
      lookups[kind] = transfers[kind].buildLookup(
          pullSet, dag.getGlobalMapping(kind), mpi)

  // ============================================================
  // Step 5: Phase 1 -- REMAP all entries
  // ============================================================

  for each (adjKind, action) in classified:
      if action == REMAP or action == RELOCATE_REMAP or action == SELF:
          targetKind = adjKind.to
          // For SELF: targetKind == adjKind.from
          std::visit([&](auto &adj) {
              ConvertAdjEntries(adj, adj.father->Size(),
                  [&](index oldGlobal) -> index {
                      if (oldGlobal == UnInitIndex) return UnInitIndex;
                      return lookups[targetKind].resolve(oldGlobal);
                  });
          }, *dag.resolveAdj(adjKind));

  // Also remap entries in companion adjacencies that store entity
  // indices (rare -- most companions store non-index data).

  // ============================================================
  // Step 6: Phase 2 -- RELOCATE all rows
  // ============================================================

  for each (adjKind, action) in classified:
      if action == RELOCATE or action == RELOCATE_REMAP or action == SELF:
          sourceKind = adjKind.from
          std::visit([&](auto &adj) {
              transfers[sourceKind].transferRows(adj, mpi);
          }, *dag.resolveAdj(adjKind));

  // Relocate companion arrays:
  for each companion of each reordered kind:
      transfers[companionKind].transferRows(companion, mpi);

  // ============================================================
  // Step 7: Rebuild global mappings
  // ============================================================

  for each kind in reorderedKinds:
      // Create new contiguous global mapping on the canonical array
      canonicalPair(kind).TransAttach();
      canonicalPair(kind).trans.createFatherGlobalMapping();
      // Other pairs for the same kind borrow:
      for each otherPair of kind:
          otherPair.father->pLGlobalMapping =
              canonicalPair(kind).father->pLGlobalMapping;

  // ============================================================
  // Step 8: Update idx states and clear stale wiring
  // ============================================================

  for each (adjKind, action) in classified:
      if action != SKIP:
          // The adj is now in global state (entries are new globals)
          mesh.getTrackedAdj(adjKind).idx.markGlobal();
          // Target mapping is stale -- will be re-wired after ghost rebuild

  // ============================================================
  // Step 9: Update mesh-level state variables
  // ============================================================

  adjPrimaryState = Adj_PointToGlobal;
  if any facial adj was affected:
      adjFacialState = Adj_PointToGlobal;
  // etc. for adjC2FState, adjN2CBState, adjC2CFaceState
```

### Post-condition

- All (non-destroyed) adjacencies in `Adj_PointToGlobal`
- Father arrays for reordered entities have new row layout
- Entries pointing to reordered entities use new global indices
- Ghost mappings stale (caller must rebuild)
- Global mappings fresh on reordered entities
- Destroyed adjacencies (e.g., facial) have null father/son

---

## Companion Array Handling

### Which arrays are companions?

| Entity kind | Companion arrays |
|-------------|-----------------|
| Cell | `cellElemInfo`, `cell2cellOrig`, `cell2nodePbi` (if periodic) |
| Node | `coords`, `node2nodeOrig`, `coordsElevDisp` (if elevation) |
| Bnd | `bndElemInfo`, `bnd2bndOrig`, `bnd2nodePbi` (if periodic) |
| Face | `faceElemInfo`, `face2nodePbi` (if periodic) |

### Treatment

Companions are treated as RELOCATE-only: their rows are moved with
their entity kind, but they have no entries to remap (they don't
store entity indices).

```
for each reordered kind:
    for each companion of that kind:
        transfers[kind].transferRows(companion, mpi);
```

### Pbi arrays as companions

`cell2nodePbi` has the same row count as `cell2node` and its rows
are parallel to `cell2node` rows. When cells are relocated,
`cell2nodePbi` must be relocated identically. The framework treats
pbi arrays as companions of their parent adjacency's source kind.

### The `cell2parentCell`, `node2parentNode`, `node2bndNode` arrays

These are local mapping vectors (not ArrayPairs). They become invalid
after reorder. The framework nullifies them (clear the vector). The
caller can rebuild them if needed (they are only used by elevation
and VTK output, which are called later).

---

## Integration with AdjPairTracked and fillRegistry

### How the framework discovers adjacencies

`fillRegistry` registers all built adjacencies into a `MeshConnectivity`
registry. The reorder algorithm uses this registry to enumerate every
adjacency that exists, classify it, and operate on it.

The registry stores `ssp<AdjVariant>` (type-erased father-only shallow
copy). But the reorder algorithm needs to operate on the actual mesh
member (to modify the father in-place and update the idx state). So the
registry is used only for **discovery and classification**. The actual
operations are performed on the mesh's `AdjPairTracked` members.

### Mapping from AdjKind to mesh member

The framework needs a mapping:

```cpp
AdjKind -> AdjPairTracked<T> & (reference to mesh member)
```

This can be implemented as a visitor pattern or a dispatch table:

```cpp
// In UnstructuredMesh:
void visitAdj(AdjKind kind, auto &&visitor)
{
    if (kind == Adj::Cell2Node) visitor(cell2node);
    else if (kind == Adj::Cell2Cell) visitor(cell2cell);
    else if (kind == Adj::Bnd2Node) visitor(bnd2node);
    // ... etc for all 12 tracked adj members
}
```

Alternatively, `fillRegistry` can store references (but the current
design uses shared_ptr copies, not references). The simplest approach
is a `switch`/`if` dispatch in `ReorderEntities` since the set of
adjacencies is fixed and small.

### Idx state transitions during reorder

Before reorder:
- All adj must be `Adj_PointToGlobal` (or `Adj_PointToLocal` for
  local-only, converted to global first).

During reorder:
- No idx transitions happen. Entries are remapped (still global, just
  different globals). Rows are relocated (still global).

After reorder:
- `idx.markGlobal()` is called on all affected adj (idempotent if
  already global, resets from Unknown if the adj was destroyed and
  recreated).
- `idx._targetMapping` is stale (the ghost mapping of the target
  entity was invalidated by the reorder). It will be re-wired after
  the caller rebuilds ghosts.

### fillRegistry skip set

When `destroyKinds` is specified (e.g., {Face}), the fillRegistry call
should skip adjacencies involving Face:

```cpp
std::unordered_set<AdjKind, AdjKindHash> skip;
for (auto kind : input.destroyKinds)
{
    // Skip all adjacencies where kind is source or target
    for (auto &adjKind : allAdjKinds)
        if (adjKind.from == kind || adjKind.to == kind)
            skip.insert(adjKind);
}
fillRegistry(dag, skip);
```

This prevents the registry from containing stale or soon-to-be-destroyed
adjacencies.

---

## Local-Only Fast Path

### Detection

```cpp
bool localOnly = true;
for (auto &[kind, transfer] : transfers)
    localOnly = localOnly && transfer.isLocalOnly;
// Must be collective:
int globalLocal;
MPI_Allreduce(&localOnly, &globalLocal, 1, MPI_INT, MPI_LAND, mpi.comm);
localOnly = globalLocal;
```

### Optimizations for local-only

1. **No distributed transfer**: `transferRows` uses `PermuteRows`
   (in-place, no MPI).

2. **Lookup arrays are local**: `buildLookup` still creates the
   `tAdj1Pair` but with no ghost entries. Resolve is a direct
   local array access.

3. **Ghost mapping rebuild can be optimized**: Instead of rebuilding
   from scratch, the existing ghost set can be permuted (replace old
   globals with new globals in the ghost index list). This is what
   `ReorderLocalCells` does today (Section F).

4. **Son data can be preserved**: After permuting father rows and
   remapping entries, the existing son data can be updated by
   pulling once (the ghost mapping is rewired with permuted globals).
   This avoids the full ghost rebuild overhead.

### Local-only ghost mapping rewire (optional optimization)

For the local-only path, the framework can optionally rewire ghost
mappings in-place instead of leaving them stale:

```
for each reordered kind:
    for each adj whose source == kind:
        // Permute the ghost index list
        newGhostGlobals = [lookup.resolve(g) for g in ghostIndex]
        adj.trans.createGhostMapping(newGhostGlobals)
        adj.trans.createMPITypes()
        adj.trans.pullOnce()
```

This is an optimization that keeps the mesh in a fully-local state
after `ReorderLocalCells`. The caller still needs to re-wire target
mappings (`idx.wireTargetMapping`) to point to the new ghost mappings.

---

## Concrete Use Cases

### Use case 1: Local cell reorder (ReorderLocalCells replacement)

```cpp
// Caller computes cell permutation via Metis
auto perm = ComputeCellPermutation(...);

// Build explicit map
EntityReorderMap cellMap{EntityKind::Cell, /*targetRanks all = mpi.rank*/};
// (localOld2New is embedded in PermutationTransfer)

// No follows (only cells move, nodes/bnds/faces stay)
ReorderInput input;
input.explicitMaps = {cellMap};
// No follows, no destroyKinds (faces are already built and will have
// their entries remapped)

mesh.ReorderEntities(input);
// All adj now in Adj_PointToGlobal, ghost mappings rewired (local-only opt)
```

### Use case 2: Distributed redistribution (ReadDistributed replacement)

```cpp
// Caller computes cell partition via ParMetis
auto cellPartition = ReadDistributed_PartitionParMetis(...);

EntityReorderMap cellMap{EntityKind::Cell, cellPartition};

// Follows: Node and Bnd follow Cell
ReorderInput input;
input.explicitMaps = {cellMap};
input.follows = {
    {EntityKind::Node, EntityKind::Cell, Adj::Node2Cell},
    {EntityKind::Bnd,  EntityKind::Cell, Adj::Bnd2Cell},
};
// Destroy faces (they will be rebuilt from scratch)
input.destroyKinds = {EntityKind::Face};

mesh.ReorderEntities(input);
// adjPrimaryState = Adj_PointToGlobal
// Caller proceeds: RecoverNode2CellAndNode2Bnd -> ... -> InterpolateFace
```

### Use case 3: Node-only reorder (hypothetical RCM on node graph)

```cpp
EntityReorderMap nodeMap{EntityKind::Node, /*all local*/};

ReorderInput input;
input.explicitMaps = {nodeMap};
// No follows (cells don't follow nodes)
// Faces must be destroyed (face2node entries become stale)
input.destroyKinds = {EntityKind::Face};

mesh.ReorderEntities(input);
// cell2node entries remapped (RELOCATE_REMAP if cells were also
// reordered, but here only REMAP since only Node is reordered)
// Actually: cell2node is Cell->Node, Cell not reordered, Node reordered
// => REMAP. bnd2node same. node2cell => RELOCATE. coords => companion RELOCATE.
```

---

## Implementation Plan

### Phase 1: PermutationTransfer utility

**File**: `src/DNDS/PermutationTransfer.hpp` (+ `.cpp` if needed)

- `PermutationTransfer` struct with data members
- `fromPartition` factory (wraps Partition2LocalIdx + Partition2Serial2Global)
- `fromLocalPermutation` factory
- `transferRows<TPair>` (local PermuteRows or distributed push)
- `LookupResult` struct with `resolve()`
- `buildLookup` (ghost-pullable old->new global)
- Unit tests: `test/cpp/dnds_test_permutation_transfer.cpp`
  - Local-only: create N entities, permute, verify
  - Distributed: create N entities across 2 ranks, redistribute, verify

### Phase 2: ReorderEntities framework

**File**: `src/Geom/Mesh/Mesh_Reorder.cpp` (new, or extend Mesh.cpp)

- `EntityReorderMap`, `FollowSpec`, `ReorderInput` structs (in Mesh.hpp)
- `ComputeFollowMap` free function
- `classifyAdj` free function
- `collectPullSet` free function
- `ReorderEntities` method on UnstructuredMesh
  - Step 0: validate + fill registry
  - Step 1: compute follows
  - Step 2: build transfers
  - Step 3: classify
  - Step 4: build lookups
  - Step 5: remap entries
  - Step 6: relocate rows + companions
  - Step 7: rebuild global mappings
  - Step 8: update idx states
  - Step 9: update mesh state variables
- `visitAdj` dispatch helper

### Phase 3: Migrate ReorderLocalCells

- Replace Sections B-G of `ReorderLocalCells` with:
  ```cpp
  auto cellMap = EntityReorderMap{EntityKind::Cell, ...};
  ReorderInput input;
  input.explicitMaps = {cellMap};
  this->ReorderEntities(input);
  ```
- Keep Section A (ComputeCellPermutation) unchanged
- Verify: 74/74 C++ tests pass, 50/50 Python tests pass

### Phase 4: Migrate ReadDistributed_Redistribute

- Replace the manual push logic with:
  ```cpp
  auto cellMap = EntityReorderMap{EntityKind::Cell, partitions.cellPartition};
  ReorderInput input;
  input.explicitMaps = {cellMap};
  input.follows = {
      {EntityKind::Node, EntityKind::Cell, Adj::Node2Cell},
      {EntityKind::Bnd,  EntityKind::Cell, Adj::Bnd2Cell},
  };
  this->ReorderEntities(input);
  ```
- The follow computation replaces `ReadDistributed_DeriveEntityPartitions`
- Verify: distributed mesh read produces identical results

### Phase 5: Cleanup

- Deprecate `Partition2LocalIdx`, `Partition2Serial2Global`,
  `TransferDataSerial2Global`, `ConvertAdjSerial2Global`
- Remove redundant manual code in old methods
- Update AGENTS.md with new API

---

## Appendix: Re-evaluation Notes (v1)

> Preserved from v1 for historical context. The v1 design proposed the
> same general direction but lacked: follow semantics, formal adj
> classification, concrete PermutationTransfer internals, and integration
> with AdjPairTracked/fillRegistry infrastructure (which did not exist at
> the time of v1).

### 2026-04-27, post PR #6

Reviewed against the merged PR #6 changes (InterpolateGlobal, face pipeline
split, templatization, pybind11 bindings, file reorganization into
`src/Geom/Mesh/`).

**Key updates for v2:**

- Face reconstruction after reorder uses the modular
  `InterpolateFace -> BuildGhostFace -> MatchFaceBoundary` pipeline.

- `parent2entityPbi` is computed locally -- no need to preserve across
  reorder. Destroyed with faces, recomputed after.

- `PermutationTransfer` should handle `AdjVariant` via `std::visit`
  for type-erased transfer (same pattern as `evaluateGhostTree`).

- Arrays destroyed before reorder and reconstructed after:
  `cell2face`, `face2cell`, `face2node`, `face2bnd`, `bnd2face`,
  `faceElemInfo`, `face2nodePbi`, `cell2facePbi` (when periodic).
