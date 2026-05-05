# Multi-Layer Ghost Cell Support {#multi_layer_ghost_design}

> **Status:** Implemented. `evaluateGhostTree` supports multi-hop chains
> with scratch pulls between BFS levels. `BuildGhostPrimary(nGhostLayers)`
> passes through to `GhostSpec::defaultPrimary(nLayers)`.

## Current State

`BuildGhostPrimary` supports N layers of ghost cells by traversing
`cell2cell` N times via `evaluateGhostTree`:

```
GhostSpec::defaultPrimary(nLayers)
  Cell chain: nLayers hops of Cell2Cell
  Node chain: nLayers hops of Cell2Cell + Cell2Node
  Bnd chain:  Bnd2Node -> Node2Bnd (unchanged)
```

`cell2cell` uses node-sharing (minShared=1). Nodes, boundaries, and N2CB
are derived from the ghost cells.

## Goal

Upgrade `evaluateGhostTree` to handle ANY multi-hop chain with internal
scratch pulls between levels, without mutating any input arrays. Use this
to support N-layer ghost cells.

## Resolved decisions

- **Node-sharing** for cell2cell: kept as-is.
- **Face ghost range**: unchanged; separate concern.
- **No input mutation**: evaluator must not modify the caller's arrays.

## Design

### Core principle: evaluator-owned scratch transformers

The evaluator accepts adjacency arrays as **read-only father-only** inputs
(global-indexed). When a multi-hop chain requires ghost data at an
intermediate level, the evaluator:

1. Creates a **temporary son array** (same type as the father).
2. Creates a **temporary `ArrayTransformer`** with `setFatherSon(father, tempSon)`.
3. Calls `createFatherGlobalMapping` (reuses father's existing mapping),
   `createGhostMapping(ghostIndices)`, `createMPITypes`, `pullOnce`.
4. Now the temporary pair (father + tempSon) has ghost data available.
5. `traverseHop` at the next level reads from this temporary pair.

The input arrays are never touched. The temporary son arrays are discarded
after evaluation.

### Registration: father-only, global-indexed

`registerAdj` stores a reference to the father array only. The adjacency
entries must be **global indices** (the evaluator needs global indices to
identify non-owned entities and to build ghost mappings).

```cpp
// In MeshConnectivity:
template <class TPair>
void registerAdj(AdjKind kind, const TPair &pair)
{
    // Store a shallow reference: just father + pLGlobalMapping.
    // No son, no ghost mapping -- the evaluator builds its own.
    auto adjVar = makeAdjVariant<TPair>();
    auto &stored = std::get<TPair>(*adjVar);
    stored.father = pair.father;
    // pLGlobalMapping lives on father->pLGlobalMapping
    registerAdj(kind, std::move(adjVar));
}
```

### evaluateGhostTree: internal scratch pull loop

```cpp
GhostResult MeshConnectivity::evaluateGhostTree(
    const CompiledGhostTree &tree,
    const MPIInfo &mpi) const
{
    // Per-node entity sets (global indices), indexed by tree node ID.
    std::vector<std::vector<index>> nodeSets(tree.totalNodes);

    // Per-AdjKind scratch state: temporary transformer + son for pulls
    // done between levels. Lazily created on first scratch pull.
    struct ScratchState
    {
        // The "live" pair with ghost data after scratch pull.
        // father = shared with input; son = temp (owned by evaluator).
        ssp<AdjVariant> liveAdj;
    };
    std::unordered_map<AdjKind, ScratchState, AdjKindHash> scratchStates;

    // Helper: resolve an adjacency, preferring the scratch (ghost-populated)
    // version if a scratch pull has been done for this AdjKind.
    auto resolveAdjLive = [&](AdjKind kind) -> ssp<AdjVariant>
    {
        auto sit = scratchStates.find(kind);
        if (sit != scratchStates.end())
            return sit->second.liveAdj;
        return resolveAdj(kind);  // original father-only
    };

    // Cumulative ghost indices per EntityKind.
    GhostResult result;

    // Initialize roots (level 0): owned entities.
    for (auto &entry : tree.levels[0])
    {
        auto gm = getGlobalMapping(entry.kind);
        index myOffset = gm->operator()(mpi.rank, 0);
        index mySize = gm->RLengths()[mpi.rank];
        auto &set = nodeSets[entry.nodeId];
        set.resize(mySize);
        for (index i = 0; i < mySize; i++)
            set[i] = myOffset + i;
    }

    for (int level = 0; level <= tree.maxLevel; level++)
    {
        // 1. Collect ghost indices from COLLECT nodes at this level.
        for (auto &entry : tree.levels[level])
        {
            if (!entry.collect)
                continue;
            auto gm = getGlobalMapping(entry.kind);
            index myOffset = gm->operator()(mpi.rank, 0);
            index myEnd = myOffset + gm->RLengths()[mpi.rank];
            auto ghosts = filterNonOwned(nodeSets[entry.nodeId], myOffset, myEnd);
            sortedMergeInto(result.ghostIndices[entry.kind], ghosts);
        }

        if (level >= tree.maxLevel)
            break;

        // 2. Scratch pull: for each adjacency used at level+1, if its
        //    parent entity kind has ghost indices, pull ghost data into a
        //    temporary son array.
        for (auto &childEntry : tree.levels[level + 1])
        {
            AdjKind hop = childEntry.hop;
            EntityKind parentKind = hop.from;

            // Check if there are ghost indices for the parent entity kind
            // that need to be resolvable for this hop.
            auto git = result.ghostIndices.find(parentKind);
            if (git == result.ghostIndices.end() || git->second.empty())
                continue;
            if (scratchStates.count(hop))
                continue;  // already pulled for this adjacency

            auto origAdj = resolveAdj(hop);
            if (!origAdj)
                continue;

            // Create scratch: temp son + transformer, pull ghost data.
            std::visit([&](auto &adj)
            {
                using TPair = std::decay_t<decltype(adj)>;
                using TArray = typename TPair::t_pArray::element_type;

                // Temp son array (same type as father)
                auto tempSon = make_ssp<TArray>(
                    ObjName{"scratch_son"}, adj.father->getMPI());

                // Build a temporary pair with its own transformer.
                auto livePair = makeAdjVariant<TPair>();
                auto &live = std::get<TPair>(*livePair);
                live.father = adj.father;
                live.son = tempSon;
                live.trans.setFatherSon(adj.father, tempSon);

                // Reuse existing global mapping.
                if (adj.father->pLGlobalMapping)
                    live.trans.pLGlobalMapping = adj.father->pLGlobalMapping;
                else
                    live.trans.createFatherGlobalMapping();

                // Ghost set = cumulative ghosts for the parent entity kind.
                auto ghostCopy = git->second;
                live.trans.createGhostMapping(ghostCopy);
                live.trans.createMPITypes();
                live.trans.pullOnce();

                scratchStates[hop] = ScratchState{std::move(livePair)};
            }, *origAdj);
        }

        // 3. Traverse children at level+1 using live (ghost-populated) adjacencies.
        for (auto &childEntry : tree.levels[level + 1])
        {
            auto &parentSet = nodeSets[childEntry.parentId];
            auto adjVar = resolveAdjLive(childEntry.hop);
            nodeSets[childEntry.nodeId] = traverseHop(parentSet, *adjVar, false);
        }

        // 4. After traversal, update scratch states for adjacencies that
        //    now have NEW ghost indices (from the level+1 collect that will
        //    happen at the top of the next iteration). We need to re-pull
        //    with the expanded ghost set.
        //
        //    This is handled naturally: at the top of the next iteration,
        //    the COLLECT step merges new ghosts. Then the scratch pull step
        //    checks if the ghost set grew and re-creates the scratch if needed.
        //
        //    To enable this, clear scratch states that need expansion:
        //    we'll re-pull with the larger cumulative set next iteration.
        scratchStates.clear();  // simple: re-pull everything each level.
    }

    // ... activeKinds collective check (same as current) ...
    return result;
}
```

### Key properties

- **No input mutation.** The original `AdjPairTracked` members (father,
  son, transformer, ghost mapping, `idx` state) are never modified. All
  pulls happen on evaluator-owned temporary transformers with temporary
  son arrays. `registerAdj` already unwraps `AdjPairTracked<TPair>` to
  the base `TPair` -- the evaluator never sees `idx`.

- **Shared father, independent son.** Scratch transformers share the
  same `father` shared_ptr as the input pair, but create their own
  temporary `son`. `setFatherSon` on the scratch transformer doesn't
  affect the input pair's transformer.

- **No `createFatherGlobalMapping`.** The scratch transformer must reuse
  `father->pLGlobalMapping` directly (via `trans.pLGlobalMapping =
  father->pLGlobalMapping`), never call `createFatherGlobalMapping()`
  which would replace the shared father's mapping pointer (the same
  `BuildSerialOut` side-effect we already fixed).

- **Global-indexed inputs.** The evaluator expects adjacency entries to
  be global indices. `BuildGhostPrimary` is called when
  `adjPrimaryState == Adj_PointToGlobal`, so this is satisfied. The
  tracked pairs' `idx.state()` is `Adj_PointToGlobal` at that point.

- **Any chain depth.** The mechanism works for any number of hops.
  Each level that needs ghost data gets a scratch pull. The
  `scratchStates.clear()` at the end of each level ensures re-pull
  with the expanded cumulative ghost set.

- **Correct ghost accumulation.** COLLECT at intermediate levels adds
  to the cumulative ghost set. The scratch pull at the next level uses
  the full cumulative set, so all ghost entries are resolvable.

- **Backward compatible.** For 1-hop chains (current usage), level 0
  is roots, level 1 is the single hop. No scratch pull is needed at
  level 0->1 because the parent set is owned entities (always in
  father). The evaluator produces the same result as today.

### Intermediate COLLECT marking

For multi-hop same-kind chains (e.g., `Cell2Cell -> Cell2Cell`), the
intermediate Cell node must also COLLECT so the ghost set accumulates
between layers.

In `insertChainIntoTrie`, when creating/finding a child node, set
`collect = true` if the child's entity kind matches the chain's target
AND the child is not the last hop:

```cpp
// Mark intermediate COLLECT for multi-hop same-kind chains.
// This ensures cumulative ghost sets are available for scratch pulls.
if (childKind == chain.target)
    child->collect = true;
```

This is safe because COLLECT only adds to the ghost set -- it never
removes or changes traversal behavior.

### GhostSpec for N-layer cells

```cpp
GhostSpec GhostSpec::defaultPrimary(int nLayers)
{
    using namespace Adj;
    DNDS_assert(nLayers >= 1);

    // Cell chain: nLayers hops of Cell2Cell
    std::vector<AdjKind> cellHops(nLayers, Cell2Cell);

    // Node chain: nLayers hops of Cell2Cell + Cell2Node
    std::vector<AdjKind> nodeHops(nLayers, Cell2Cell);
    nodeHops.push_back(Cell2Node);

    return GhostSpec{{
        {EntityKind::Cell, cellHops, EntityKind::Cell},
        {EntityKind::Cell, nodeHops, EntityKind::Node},
        {EntityKind::Bnd,  {Bnd2Node, Node2Bnd}, EntityKind::Bnd},
        {EntityKind::Bnd,  {Bnd2Node, Node2Bnd, Bnd2Node}, EntityKind::Node},
    }};
}
```

### Compiled tree for N=2

```
Cell (root, level 0)
  \-[Cell2Cell]-> Cell (level 1, COLLECT)     <-- layer 1 ghost cells
    \-[Cell2Cell]-> Cell (level 2, COLLECT)   <-- layer 2 ghost cells
    \-[Cell2Node]-> Node (level 2, COLLECT)   <-- nodes of outermost layer

Bnd (root, level 0)
  \-[Bnd2Node]-> Node (level 1)
    \-[Node2Bnd]-> Bnd (level 2, COLLECT)     <-- ghost bnds
      \-[Bnd2Node]-> Node (level 3, COLLECT)  <-- nodes of ghost bnds
```

Evaluation for N=2:

1. Level 0: roots = owned Cells, owned Bnds.
2. Level 1: traverse Cell2Cell for owned cells -> cell set with ghosts.
   COLLECT ghost cells (layer 1). Traverse Bnd2Node for owned bnds ->
   node set. No scratch pull needed (parents are all owned).
3. **Scratch pull**: create temp transformer on cell2cell.father, ghost
   mapping = layer-1 ghost cells, pullOnce. Now layer-1 ghost cells'
   cell2cell rows are available in temp son.
4. Level 2: traverse Cell2Cell for layer-1 cells (including ghosts, now
   resolvable via temp son) -> cell set with layer-2 ghosts. COLLECT
   ghost cells (layers 1+2). Traverse Cell2Node for outermost cells ->
   node set. COLLECT ghost nodes. Traverse Node2Bnd -> bnd set. COLLECT
   ghost bnds.
5. Level 3 (bnd chain): traverse Bnd2Node for ghost bnds -> node set.
   COLLECT ghost nodes (merged with cell-derived nodes).

### Caller code in BuildGhostPrimary

```cpp
void UnstructuredMesh::BuildGhostPrimary(int nLayers)
{
    DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
    // ... existing assertions ...

    cell2cell.TransAttach();
    cell2node.TransAttach();
    // ... etc ...

    cell2cell.trans.createFatherGlobalMapping();

    MeshConnectivity dag;
    dag.meshDim = dim;
    dag.registerAdj(Adj::Cell2Cell, cell2cell);
    dag.registerAdj(Adj::Cell2Node, cell2node);
    dag.registerAdj(Adj::Bnd2Node, bnd2node);
    dag.registerAdj(Adj::Node2Bnd, node2bnd);
    dag.registerGlobalMapping(EntityKind::Cell, cell2cell.trans.pLGlobalMapping);
    dag.registerGlobalMapping(EntityKind::Node, coords.trans.pLGlobalMapping);
    dag.registerGlobalMapping(EntityKind::Bnd, bnd2node.trans.pLGlobalMapping);

    auto spec = GhostSpec::defaultPrimary(nLayers);
    auto tree = CompiledGhostTree::compile(spec);
    auto result = dag.evaluateGhostTree(tree, mpi);
    // evaluateGhostTree did internal scratch pulls on temp arrays.
    // The real mesh arrays are unchanged.

    // Now apply the ghost sets to the real mesh arrays:
    auto &ghostCells = result.ghostIndices[EntityKind::Cell];
    cell2cell.trans.createGhostMapping(ghostCells);
    cell2cell.trans.createMPITypes();
    cell2cell.trans.pullOnce();
    cell2node.BorrowAndPull(cell2cell);
    // ... other co-indexed arrays ...

    auto &ghostNodes = result.ghostIndices[EntityKind::Node];
    coords.trans.createGhostMapping(ghostNodes);
    // ... etc (same as current) ...
}
```

The caller code is almost identical to today. The only change is passing
`nLayers` to `defaultPrimary`. The evaluator handles the complexity
internally.

## Implementation order

1. Mark intermediate COLLECT in `insertChainIntoTrie`.
2. Add scratch pull mechanism to `evaluateGhostTree` (temp transformers).
3. Add `GhostSpec::defaultPrimary(int nLayers)`.
4. Update `BuildGhostPrimary(int nLayers = 1)`.
5. Propagate through `Mesh_Helpers.hpp`, Python `read_mesh`.
6. Tests.

## Files to modify

| File | Change |
|------|--------|
| `MeshConnectivity_Ghost.cpp` | Scratch pull in evaluateGhostTree, intermediate COLLECT |
| `MeshConnectivity.hpp` | registerAdj simplification (father-only) |
| `Mesh.cpp` | `BuildGhostPrimary(int nLayers)` |
| `Mesh.hpp` | Update declaration |
| `Mesh_Helpers.hpp` | Propagate nLayers |
| `python/DNDSR/Geom/utils.py` | Add ghost_layers parameter |
| `test/` | New multi-layer tests |
