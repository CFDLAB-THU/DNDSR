# Distributed Entity Reordering — Design Document (v2) {#distributed_reorder_design}

> **Status:** Implemented (Phases 1–4 complete). See
> [Implementation Notes](#implementation-notes) for divergences from the
> original design pseudocode.
>
> **Supersedes:** The original v1 design (same file, git history).
> v2 adds: placement-follow semantics, automatic adj conversion rules
> with formal classification, `PermutationTransfer` internal design,
> companion propagation, and full integration with `AdjPairTracked` /
> `fillRegistry` / checked-wrapper infrastructure.
>
> **Last updated:** 2026-05-05.

**TL;DR:** When mesh entities are reordered (local permutation for cache
locality, or cross-rank redistribution after partitioning), every adjacency
array must be updated: entries pointing to reordered entities are remapped to
new global indices, and rows belonging to reordered entities are relocated.
The framework classifies each adjacency into one of five actions (`SKIP`,
`RELOCATE`, `REMAP`, `RELOCATE_REMAP`, `SELF`), builds `PermutationTransfer`
objects for both local and distributed cases, and handles companion arrays
(solution DOFs, element info) via a callback-based registry. Solver code can
register its own arrays so they participate in the same reorder operation.

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
14. [Implementation Notes](#implementation-notes)

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

### API (two-layer design)

```cpp
// =================================================================
// Companion callback: type-erased row relocation for non-adj arrays
// =================================================================

/// Callback invoked during the RELOCATE phase for a companion array.
/// The framework passes the PermutationTransfer for the entity kind
/// that this companion belongs to. The callback calls transferRows
/// on its own array.
using CompanionRelocateFn = std::function<void(
    const PermutationTransfer &transfer, const MPIInfo &mpi)>;

/// One registered companion entry.
struct CompanionEntry
{
    EntityKind kind;          // entity kind this array is parallel to
    CompanionRelocateFn fn;   // callback to relocate the array
    std::string name;         // diagnostic name (optional)
};

// =================================================================
// Adj entry: type-erased adjacency operation
// =================================================================

/// Callback invoked during the REMAP phase for an adjacency array.
/// The framework passes the LookupResult for the target entity kind.
/// The callback calls ConvertAdjEntries on its own array.
using AdjRemapFn = std::function<void(
    const LookupResult &lookup)>;

/// Callback invoked during the RELOCATE phase for an adjacency array.
using AdjRelocateFn = std::function<void(
    const PermutationTransfer &transfer, const MPIInfo &mpi)>;

/// One registered adjacency entry.
struct AdjEntry
{
    AdjKind kind;             // adjacency kind (from -> to)
    AdjRemapFn remapFn;      // remap entries (may be null if no remap needed)
    AdjRelocateFn relocateFn; // relocate rows (may be null if no relocate needed)
    std::string name;         // diagnostic name (optional)
};

// =================================================================
// ReorderRegistry: the dynamic set of arrays to operate on
// =================================================================

/// Collects all adjacencies and companions that participate in a reorder.
/// Built by UnstructuredMesh::buildReorderRegistry() for mesh members,
/// and extended by external code for solver/evaluator arrays.
struct ReorderRegistry
{
    /// All adjacency entries (mesh members + external).
    std::vector<AdjEntry> adjs;

    /// All companion entries (mesh members + external).
    std::vector<CompanionEntry> companions;

    /// Global offsets mappings per entity kind (for PermutationTransfer).
    std::unordered_map<EntityKind, ssp<GlobalOffsetsMapping>> globalMappings;

    /// Register an adjacency with type-erased callbacks.
    void registerAdj(AdjKind kind, AdjRemapFn remap, AdjRelocateFn relocate,
                     std::string name = {});

    /// Register a companion with a type-erased relocate callback.
    void registerCompanion(EntityKind kind, CompanionRelocateFn fn,
                           std::string name = {});

    /// Register a GlobalOffsetsMapping for an entity kind.
    void registerGlobalMapping(EntityKind kind, ssp<GlobalOffsetsMapping> gm);
};

// =================================================================
// ReorderInput and ReorderPlan
// =================================================================

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

// Layer 1: Standalone plan (no mesh dependency after construction)
struct ReorderPlan
{
    std::unordered_map<EntityKind, PermutationTransfer> transfers;
    std::unordered_map<EntityKind, LookupResult> lookups;
    std::unordered_set<EntityKind> reorderedKinds;
    bool isLocalOnly{false};

    /// Build plan from input + registry.
    static ReorderPlan build(const ReorderInput &input,
                             const ReorderRegistry &registry,
                             const MPIInfo &mpi);

    /// Apply the plan to all entries in the registry.
    void apply(ReorderRegistry &registry, const MPIInfo &mpi) const;

    /// Standalone operations for external arrays:
    template <class TPair>
    void remapEntries(TPair &pair, EntityKind targetKind) const;

    template <class TPair>
    void relocateRows(TPair &pair, EntityKind sourceKind,
                      const MPIInfo &mpi) const;
};

// Layer 2: Mesh methods
class UnstructuredMesh
{
    /// Build a ReorderRegistry containing all mesh members.
    /// Registers all built adj arrays (as callbacks) and all
    /// companion arrays (coords, cellElemInfo, pbi arrays, etc.).
    /// Skips adjacencies involving destroyKinds.
    ReorderRegistry buildReorderRegistry(
        const std::unordered_set<EntityKind> &destroyKinds = {}) const;

    /// Build plan from input (uses buildReorderRegistry internally).
    ReorderPlan buildReorderPlan(const ReorderInput &input) const;

    /// Build plan AND apply to all mesh members.
    void ReorderEntities(const ReorderInput &input);
};
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
| Node | `coords`, `node2nodeOrig`, `coordsElevDisp` (if elevation), `nodeWallDist` |
| Bnd | `bndElemInfo`, `bnd2bndOrig`, `bnd2nodePbi` (if periodic) |
| Face | `faceElemInfo`, `face2nodePbi` (if periodic) |

### Callback-based registration

Companions are registered into the `ReorderRegistry` via type-erased
callbacks. Each callback captures a reference to its array and calls
`transferRows` on it when invoked:

```cpp
// Inside UnstructuredMesh::buildReorderRegistry():
auto reg = ReorderRegistry{};

// Register cell companions
reg.registerCompanion(EntityKind::Cell,
    [&](const PermutationTransfer &t, const MPIInfo &m) {
        t.transferRows(cellElemInfo, m);
    }, "cellElemInfo");

reg.registerCompanion(EntityKind::Cell,
    [&](const PermutationTransfer &t, const MPIInfo &m) {
        t.transferRows(cell2cellOrig, m);
    }, "cell2cellOrig");

if (isPeriodic && cell2nodePbi.father)
    reg.registerCompanion(EntityKind::Cell,
        [&](const PermutationTransfer &t, const MPIInfo &m) {
            t.transferRows(cell2nodePbi, m);
        }, "cell2nodePbi");

// Register node companions
reg.registerCompanion(EntityKind::Node,
    [&](const PermutationTransfer &t, const MPIInfo &m) {
        t.transferRows(coords, m);
    }, "coords");

reg.registerCompanion(EntityKind::Node,
    [&](const PermutationTransfer &t, const MPIInfo &m) {
        t.transferRows(node2nodeOrig, m);
    }, "node2nodeOrig");

// ... etc for Bnd, Face
```

### External companion registration (solver arrays)

External code appends its own companions to the registry before
the reorder is applied:

```cpp
auto registry = mesh.buildReorderRegistry(input.destroyKinds);

// Solver registers its cell-parallel DOF arrays:
registry.registerCompanion(EntityKind::Cell,
    [&](const PermutationTransfer &t, const MPIInfo &m) {
        t.transferRows(cellSolution, m);
    }, "cellSolution");

registry.registerCompanion(EntityKind::Cell,
    [&](const PermutationTransfer &t, const MPIInfo &m) {
        t.transferRows(cellGradient, m);
    }, "cellGradient");

// Now build plan and apply:
auto plan = ReorderPlan::build(input, registry, mpi);
plan.apply(registry, mpi);
```

This is the **recommended pattern for solver participation**.
The mesh's `ReorderEntities` convenience method does the same
thing internally (builds registry, builds plan, applies).

### Treatment during apply

During `ReorderPlan::apply`:

```
Phase 1 (REMAP): invoke adjEntry.remapFn for all classified REMAP/SELF adjs
Phase 2 (RELOCATE): invoke adjEntry.relocateFn for all classified RELOCATE/SELF adjs
Phase 3 (COMPANIONS): invoke companion.fn for all companions of reordered kinds
```

Companions always run after adj relocation (Phase 3), but since they
have no ordering dependency on adj arrays, they could also run in
parallel with Phase 2.

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

## Dynamic Reorder Registry (Decoupled from UnstructuredMesh)

### Problem

A naive design ties the reorder operation to `UnstructuredMesh`'s
hardcoded member list. The solver, CFV evaluator, or other subsystems
may own arrays parallel to mesh entities (solution DOFs per cell,
gradient arrays per node, etc.) that also need relocation or remapping.

### Solution: ReorderRegistry + callback-based participation

The `ReorderRegistry` (defined in the API section above) is the
dynamic set of arrays that participate in a reorder. It is built by
`UnstructuredMesh::buildReorderRegistry()` for mesh members, and
extended by external code before plan application.

### How `buildReorderRegistry` works

```cpp
ReorderRegistry UnstructuredMesh::buildReorderRegistry(
    const std::unordered_set<EntityKind> &destroyKinds) const
{
    ReorderRegistry reg;

    // ---- Adjacency registration (with callbacks) ----
    auto shouldSkip = [&](AdjKind kind) {
        return destroyKinds.count(kind.from) || destroyKinds.count(kind.to);
    };

    // Helper: register a tracked adj member with remap + relocate callbacks.
    auto regAdj = [&](AdjKind kind, auto &trackedPair) {
        if (!trackedPair.father || shouldSkip(kind)) return;

        AdjRemapFn remap = [&trackedPair](const LookupResult &lookup) {
            ConvertAdjEntries(trackedPair, trackedPair.father->Size(),
                [&](index g) -> index {
                    if (g == UnInitIndex) return UnInitIndex;
                    return lookup.resolve(g);
                });
        };

        AdjRelocateFn relocate = [&trackedPair](
            const PermutationTransfer &t, const MPIInfo &m) {
            t.transferRows(trackedPair, m);
        };

        reg.registerAdj(kind, remap, relocate,
                        adjKindName(kind));
    };

    // Register all 12 tracked adj members
    regAdj(Adj::Cell2Node, cell2node);
    regAdj(Adj::Cell2Cell, cell2cell);
    regAdj(Adj::Bnd2Node,  bnd2node);
    regAdj(Adj::Bnd2Cell,  bnd2cell);
    regAdj(Adj::Node2Cell, node2cell);
    regAdj(Adj::Node2Bnd,  node2bnd);
    regAdj(Adj::Cell2Face, cell2face);
    regAdj(Adj::Face2Node, face2node);
    regAdj(Adj::Face2Cell, face2cell);
    regAdj(Adj::Face2Bnd,  face2bnd);
    regAdj(Adj::Bnd2Face,  bnd2face);
    regAdj(Adj::Cell2CellFace, cell2cellFace);

    // ---- Companion registration (with callbacks) ----
    auto regComp = [&](EntityKind kind, auto &pair, const char *name) {
        if (!pair.father) return;
        reg.registerCompanion(kind,
            [&pair](const PermutationTransfer &t, const MPIInfo &m) {
                t.transferRows(pair, m);
            }, name);
    };

    regComp(EntityKind::Cell, cellElemInfo, "cellElemInfo");
    regComp(EntityKind::Cell, cell2cellOrig, "cell2cellOrig");
    regComp(EntityKind::Node, coords, "coords");
    regComp(EntityKind::Node, node2nodeOrig, "node2nodeOrig");
    regComp(EntityKind::Bnd,  bndElemInfo, "bndElemInfo");
    regComp(EntityKind::Bnd,  bnd2bndOrig, "bnd2bndOrig");

    if (isPeriodic) {
        regComp(EntityKind::Cell, cell2nodePbi, "cell2nodePbi");
        regComp(EntityKind::Bnd,  bnd2nodePbi, "bnd2nodePbi");
    }
    if (faceElemInfo.father && !destroyKinds.count(EntityKind::Face))
        regComp(EntityKind::Face, faceElemInfo, "faceElemInfo");
    if (coordsElevDisp.father)
        regComp(EntityKind::Node, coordsElevDisp, "coordsElevDisp");
    if (nodeWallDist.father)
        regComp(EntityKind::Node, nodeWallDist, "nodeWallDist");

    // ---- Global mappings ----
    // Source from adj arrays (same as fillRegistry logic)
    auto getGM = [](const auto &pair) -> ssp<GlobalOffsetsMapping> {
        if (pair.father && pair.father->pLGlobalMapping)
            return pair.father->pLGlobalMapping;
        return nullptr;
    };
    if (auto gm = getGM(cell2node)) reg.registerGlobalMapping(EntityKind::Cell, gm);
    if (auto gm = coords.father ? coords.father->pLGlobalMapping : nullptr)
        reg.registerGlobalMapping(EntityKind::Node, gm);
    if (auto gm = getGM(bnd2node)) reg.registerGlobalMapping(EntityKind::Bnd, gm);
    if (auto gm = getGM(face2node)) reg.registerGlobalMapping(EntityKind::Face, gm);

    return reg;
}
```

### External code extends the registry

```cpp
// Solver extends the mesh's registry with its own arrays:
auto registry = mesh.buildReorderRegistry(input.destroyKinds);

// Add solver cell-parallel arrays:
registry.registerCompanion(EntityKind::Cell,
    [&](const PermutationTransfer &t, const MPIInfo &m) {
        t.transferRows(cellSolution, m);
    }, "solver::cellSolution");

registry.registerCompanion(EntityKind::Cell,
    [&](const PermutationTransfer &t, const MPIInfo &m) {
        t.transferRows(cellGradient, m);
    }, "solver::cellGradient");

// Add solver node-parallel arrays:
registry.registerCompanion(EntityKind::Node,
    [&](const PermutationTransfer &t, const MPIInfo &m) {
        t.transferRows(nodeWallDist, m);
    }, "solver::nodeWallDist");

// Add an external adjacency (e.g., a DOF connectivity array):
registry.registerAdj(
    AdjKind{EntityKind::Cell, EntityKind::Node},  // same kind as cell2node
    [&](const LookupResult &lookup) {
        ConvertAdjEntries(myDOFConn, myDOFConn.father->Size(),
            [&](index g) { return g == UnInitIndex ? g : lookup.resolve(g); });
    },
    [&](const PermutationTransfer &t, const MPIInfo &m) {
        t.transferRows(myDOFConn, m);
    },
    "solver::dofConnectivity"
);

// Build plan from the extended registry:
auto plan = ReorderPlan::build(input, registry, mpi);
plan.apply(registry, mpi);
```

### How ReorderPlan::apply works with the registry

```cpp
void ReorderPlan::apply(ReorderRegistry &registry, const MPIInfo &mpi) const
{
    // Phase 1: REMAP all adj entries
    for (auto &adj : registry.adjs)
    {
        auto action = classifyAdj(adj.kind, reorderedKinds);
        if (action == REMAP || action == RELOCATE_REMAP || action == SELF)
        {
            EntityKind targetKind = adj.kind.isIntraLevel()
                ? adj.kind.from : adj.kind.to;
            if (adj.remapFn)
                adj.remapFn(lookups.at(targetKind));
        }
    }

    // Phase 2: RELOCATE all adj rows
    for (auto &adj : registry.adjs)
    {
        auto action = classifyAdj(adj.kind, reorderedKinds);
        if (action == RELOCATE || action == RELOCATE_REMAP || action == SELF)
        {
            EntityKind sourceKind = adj.kind.from;
            if (adj.relocateFn)
                adj.relocateFn(transfers.at(sourceKind), mpi);
        }
    }

    // Phase 3: RELOCATE all companions
    for (auto &comp : registry.companions)
    {
        if (reorderedKinds.count(comp.kind))
            comp.fn(transfers.at(comp.kind), mpi);
    }
}
```

### Relationship to existing fillRegistry

The existing `fillRegistry(MeshConnectivity &dag)` method remains
for ghost tree evaluation (it fills a `MeshConnectivity` DAG for
`evaluateGhostTree`). The new `buildReorderRegistry` is a separate
method for reorder operations. They share the same discovery logic
(which adj arrays exist) but produce different output types:

| Method | Output | Purpose |
|--------|--------|---------|
| `fillRegistry(dag)` | `MeshConnectivity` with `ssp<AdjVariant>` | Ghost tree evaluation |
| `buildReorderRegistry()` | `ReorderRegistry` with callbacks | Reorder operations |

Both use the same underlying member-existence checks. The reorder
registry additionally captures companions and type-erased callbacks.

### Summary: decoupling layers

```
   Caller (solver, evaluator, etc.)
      |
      | extends ReorderRegistry with own arrays
      v
   ReorderRegistry (dynamic set of callbacks)
      |
      | consumed by ReorderPlan::build + apply
      v
   ReorderPlan (standalone, computed transfers + lookups)
      |
      | invokes callbacks during apply()
      v
   PermutationTransfer + LookupResult (MPI primitives)
      |
      | wraps ArrayTransformer / PermuteRows
      v
   DNDS array infrastructure
```

The mesh is just one contributor to the registry. Any code that owns
entity-parallel arrays can register callbacks and participate in the
same reorder operation.

---

## Integration with AdjPairTracked and fillRegistry

### How the framework discovers adjacencies

`buildReorderRegistry()` replaces the role previously played by
`fillRegistry` for reorder operations. It iterates all mesh members,
checks if their father is non-null, and registers them with type-erased
callbacks. The callbacks capture references to the actual
`AdjPairTracked` members, so the plan operates on live mesh data.

The existing `fillRegistry(MeshConnectivity &dag)` remains unchanged
for ghost tree evaluation. The two methods coexist:
- `fillRegistry` -> `MeshConnectivity` (for `evaluateGhostTree`)
- `buildReorderRegistry` -> `ReorderRegistry` (for `ReorderEntities`)

### No explicit AdjKind-to-member dispatch needed

Because `buildReorderRegistry` registers callbacks that already
capture member references, there is no need for a `visitAdj` dispatch
table. The plan invokes callbacks directly during `apply()`. This
eliminates the coupling between `ReorderPlan` and `UnstructuredMesh`'s
specific member layout.

### Idx state transitions during reorder

Before reorder:
- All adj must be `Adj_PointToGlobal` (or `Adj_PointToLocal` for
  local-only, converted to global first).

During reorder:
- No idx transitions happen. Entries are remapped (still global, just
  different globals). Rows are relocated (still global).

After reorder (handled by mesh's `ReorderEntities` wrapper):
- `idx.markGlobal()` is called on all affected adj (idempotent if
  already global, resets from Unknown if the adj was destroyed and
  recreated).
- `idx._targetMapping` is stale (the ghost mapping of the target
  entity was invalidated by the reorder). It will be re-wired after
  the caller rebuilds ghosts.

The mesh wrapper also updates the group state variables
(`adjPrimaryState`, `adjFacialState`, etc.) to `Adj_PointToGlobal`.

### Post-reorder idx state update (mesh wrapper responsibility)

```cpp
// Inside UnstructuredMesh::ReorderEntities, after plan.apply():
auto updateIdx = [&](auto &trackedPair) {
    if (trackedPair.father)
        trackedPair.idx.markGlobal();
};
updateIdx(cell2node);
updateIdx(cell2cell);
updateIdx(bnd2node);
updateIdx(bnd2cell);
// ... etc for all tracked adj members

adjPrimaryState = Adj_PointToGlobal;
// adjFacialState, adjC2FState, adjN2CBState: set to Unknown if destroyed,
// or Adj_PointToGlobal if still present.
```

### buildReorderRegistry skip logic

When `destroyKinds` is specified (e.g., {Face}), `buildReorderRegistry`
skips adjacencies involving that kind:

```cpp
auto shouldSkip = [&](AdjKind kind) {
    return destroyKinds.count(kind.from) || destroyKinds.count(kind.to);
};
```

This prevents registering callbacks for adjacencies that will be
destroyed before the reorder runs.

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

### Use case 4: Solver-participates redistribution (external arrays)

```cpp
// Solver owns DOF arrays parallel to cells:
tDOFPair cellSolution;   // father->Size() == mesh.NumCell()
tDOFPair cellGradient;

// Step 1: Build plan (does MPI communication for lookups)
ReorderInput input;
input.explicitMaps = {cellMap};
input.follows = {
    {EntityKind::Node, EntityKind::Cell, Adj::Node2Cell},
    {EntityKind::Bnd,  EntityKind::Cell, Adj::Bnd2Cell},
};
input.destroyKinds = {EntityKind::Face};

auto plan = mesh.buildReorderPlan(input);

// Step 2: Reorder mesh
mesh.ReorderEntities(input);

// Step 3: Reorder solver arrays using the same plan
plan.relocateRows(cellSolution, EntityKind::Cell, mpi);
plan.relocateRows(cellGradient, EntityKind::Cell, mpi);

// Step 4: If solver has node-parallel arrays:
plan.relocateRows(nodeWallDist, EntityKind::Node, mpi);

// Step 5: Rebuild global mappings on solver arrays
cellSolution.TransAttach();
cellSolution.trans.createFatherGlobalMapping();
// ... etc
```

### Use case 5: Node-only reorder with solver participation

```cpp
// Hypothetical: RCM reorder on node graph for FEM bandwidth reduction

EntityReorderMap nodeMap{EntityKind::Node, /*all local*/};
ReorderInput input;
input.explicitMaps = {nodeMap};
input.destroyKinds = {EntityKind::Face};

auto plan = mesh.buildReorderPlan(input);
mesh.ReorderEntities(input);

// Solver has node-based arrays:
plan.relocateRows(nodeSolution, EntityKind::Node, mpi);
plan.relocateRows(nodeRHS, EntityKind::Node, mpi);

// Mesh's cell2node entries are already remapped (REMAP action)
// so cells now reference new node globals. No cell rows moved.
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

### Phase 2: ReorderPlan + ReorderEntities framework

**Files**:
- `src/Geom/Mesh/ReorderPlan.hpp` — `ReorderPlan`, `ReorderInput`,
  `EntityReorderMap`, `FollowSpec`, `AdjAction`, classification logic
- `src/Geom/Mesh/Mesh_Reorder.cpp` — `UnstructuredMesh::ReorderEntities`,
  `buildReorderPlan`, mesh-member dispatch

Contents:
- `EntityReorderMap`, `FollowSpec`, `ReorderInput` structs
- `ReorderPlan` with `build()`, `remapEntries()`, `relocateRows()`
- `classifyAdj` free function
- `ComputeFollowMap` free function
- `collectPullSet` free function
- `UnstructuredMesh::ReorderEntities` (wrapper that builds plan + applies)
- `UnstructuredMesh::buildReorderPlan` (builds plan without applying)
- `visitAdj` dispatch helper (AdjKind -> mesh member reference)

Unit tests:
- Single-entity remap (node-only): verify cell2node entries updated
- Single-entity relocate (cell-only): verify cell rows permuted
- Multi-entity (cell + node + bnd): verify full redistribution

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

---

## Implementation Notes

> **Added 2026-05-05** after full implementation of Phases 1–4.

### Divergences from Design Pseudocode

| Aspect | Design Doc | Actual Implementation | Rationale |
|--------|-----------|----------------------|-----------|
| `fromPartition` | Delegates to `Partition2LocalIdx` helper | Self-contained: inline prefix-sum + scatter loop | Avoids extra abstraction; single function is easier to reason about |
| `transferRows` local path | Uses `PermuteRows` (in-place cycle-follow) | Deep-copies father, then writes permuted entries | Simpler for CSR arrays where in-place row reordering is complex; trades memory for correctness |
| `buildLookup` | Takes `oldGlobalMapping` param | Builds mapping internally via `createFatherGlobalMapping` | Encapsulates ghost mapping construction; caller doesn't need to precompute |
| Follow computation | Done inside `ReorderPlan::build` | Done in mesh wrapper (`buildReorderPlan`/`ReorderEntities`) before `build` | `ReorderPlan::build` has no access to adjacency data; follows require adj traversal |
| `ReorderLocalCells` (Phase 3) | Full migration to `ReorderEntities` | Uses `PermutationTransfer::fromLocalPermutation` directly + manual remap | Partial migration only; avoids destroy/rebuild cycle for local-only case |
| `ReadDistributed_Redistribute` (Phase 4) | Use `ReorderEntities` | Calls `ReorderEntities` with explicit `EntityReorderMap`s | Fully migrated as designed |

### Key Architectural Observations

1. **`ReorderPlan::build` ignores `input.follows`** — the `follows` field in
   `ReorderInput` is consumed by the mesh wrapper to compute follow maps; by
   the time `build` is called, all entities have explicit `EntityReorderMap`s.
   The `follows` field on `ReorderInput` is effectively a mesh-wrapper-level
   concept, not a plan-level concept.

2. **`fromLocalPermutation` is NOT MPI-collective** — unlike `fromPartition`,
   which uses `MPI_Allreduce` to detect all-local status and `Alltoall` for
   distributed transfer, `fromLocalPermutation` performs zero MPI.

3. **`ReorderLocalCells` vs `ReorderEntities`** — these have opposite
   preconditions: `ReorderLocalCells` requires `Adj_PointToLocal` (converts
   to global internally), while `ReorderEntities` requires
   `Adj_PointToGlobal`. This is because `ReorderLocalCells` is called after
   the full mesh pipeline (faces built, ghosts attached), while
   `ReorderEntities` is called during redistribution before faces exist.

4. **Pull-set collection** — `buildReorderRegistry` collects pull sets (ghost
   entries that reference reordered entities) by iterating ghost rows of
   relevant adjacencies. These are passed to `buildLookup` for ghost mapping
   reconstruction after the plan is applied.

### Known Limitations

- **`cell2cellFace` not handled by `ReorderLocalCells`** — the manual remap
  block does not include `cell2cellFace`. An assertion should guard that it
  is not built when entering this path.
- **No periodic mesh test coverage** — the `isPeriodic` branches in
  `buildReorderRegistry` and `ReorderEntities` are untested.
- **No 3D or non-identity real-mesh reorder test** — all real-mesh tests use
  `UniformSquare_10.cgns` (2D) with identity partitions.
- **Code duplication** — ~80 lines of follow-computation logic are duplicated
  between `buildReorderPlan` and `ReorderEntities`.

### Recommended Next Steps

1. Deduplicate follow logic into a shared helper.
2. Add debug-mode permutation validation in `fromLocalPermutation`.
3. Add non-identity, 3D, and periodic mesh reorder tests.
4. Add external-companion participation test (solver arrays).
5. Consider full `ReorderLocalCells` migration to `ReorderEntities` (would
   unify preconditions but requires handling the faces-already-built case).
