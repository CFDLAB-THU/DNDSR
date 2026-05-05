# UnstructuredMesh Refactoring Plan {#mesh_refactoring}

**Branch:** `dev/harry`
**Baseline commit:** `cd065ad` (on `dev/harry`)
**Date:** 2026-04-21

---

## 1. Current State

The Geom mesh module (~11,600 lines across 14 files) centres on two structs:

```
UnstructuredMesh           (Mesh.hpp: ~810 lines header, ~2,585 lines Mesh.cpp)
  ~50 data members, ~60 methods, 9 impl files
  Inherits DeviceTransferable<UnstructuredMesh>

UnstructuredMeshSerialRW   (Mesh.hpp: ~250 lines header, shared Mesh.cpp)
  16 declared-but-unused adjacency fields
  Mixes reading, partitioning, serial output
```

Implementation files:

| File | Lines | Responsibility |
|------|-------|----------------|
| Mesh.cpp | 2,585 | Topology building, ghost, face interpolation, reordering, serialization, bnd mesh |
| Mesh_Elevation.cpp | 1,120 | O1→O2 elevation, O2→O1 bisection, boundary smooth |
| Mesh_Elevation_SmoothSolver.cpp | 1,782 | 4 smooth solver variants (dispatch, V1Old, V1, V2) |
| Mesh_Plts.cpp | 1,685 | VTK/PLT/CGNS/VTKHDF output |
| Mesh_Serial_ReadFromCGNS.cpp | 726 | CGNS + OpenFOAM readers |
| Mesh_Serial_BuildCell2Cell.cpp | 702 | Cell2Cell, Deduplicate1to1Periodic |
| Mesh_ReadSerializeDistributed.cpp | 512 | Distributed H5 read + ParMetis repartition |
| Mesh_Serial_Partition.cpp | 414 | Metis partitioning |
| Mesh_WallDist.cpp | 224 | Wall distance (CGAL-based) |

### Prior Work (Phases 1–3, done on `dev/harry_refac1`)

- **Phase 1 (Eliminate duplication):** Template index conversions (12→4+12 wrappers),
  `ConvertAdjEntries`, `PermuteRows` — done.
- **Phase 2 (Break up massive methods):** Extracted helpers from `ReadFromCGNSSerial`,
  `InterpolateFace`, `ReorderLocalCells`, `PrintSerialPartVTKDataArray` — done.
- **Phase 3 (Coupling/cleanup):** Moved `BuildNodeWallDist`, broke Geom→Solver
  dependency, removed `auto mesh = this`, deleted dead code — done.

### Remaining Pain Points

1. **Giant functions** (>200 lines, hard to understand/test individually):

   | Function | File | Lines | Issue |
   |----------|------|-------|-------|
   | `ElevatedNodesSolveInternalSmoothV2` | SmoothSolver.cpp | 864 | Monolithic: assertions, setup, FEM assembly, limiters, GMRES, apply — all in one |
   | `PrintSerialPartVTKDataArray` | Plts.cpp | 560 | Repeated binary/ascii encoding pattern for every data array |
   | `BuildO2FromO1Elevation` | Elevation.cpp | 450 | 5 distinct phases jammed together |
   | `ElevatedNodesSolveInternalSmoothV1` | SmoothSolver.cpp | 448 | Similar structure to V2 but RBF-based |
   | `ElevatedNodesGetBoundarySmooth` | Elevation.cpp | 427 | Extended-face building, normals, Hermite interp |
   | `Deduplicate1to1Periodic` | BuildCell2Cell.cpp | 312 | 5 distinct phases |
   | `RecoverCell2CellAndBnd2Cell` | Mesh.cpp | 258 | Non-periodic (simple) + periodic (complex) fused |

2. **Code duplication across smooth solver variants** (~80 lines copied 3×):
   - State assertion + early exit block (identical in V1Old, V1, V2)
   - `nodesBoundInterpolated` identification loop (identical)
   - `boundInterpCoo`/`boundInterpVal` init + populate + ghost pull (identical)
   - KD-tree construction from boundary coords (identical in V1Old, V1)
   - Interior displacement evaluation via radius search (identical in V1Old, V1)

3. **Adj conversion methods partially unified** — `ConvertAdjEntries` used for
   Primary/PrimaryForBnd/C2CFace but NOT for Facial, C2F, N2CB (which use
   manual OMP-parallel loops or lambdas).

4. **Deprecated smooth solver variant** — `V1Old` (235 lines) has raw
   `std::cout << iN` debug prints, serial-only solve, and is superseded by V1.

5. **Disabled code blocks in V2** — ~175 lines of `#if 0` / commented-out "fem
   method" and "spring method" blocks inside the `AssembleRHSMat` lambda.

---

## 2. Refactoring Phases

### Phase 0: Extend Baseline Tests (DO FIRST)

**Goal:** Ensure elevation/bisection + smooth solver correctness is testable
before touching any code.

**Tests to add** (in `test/cpp/Geom/test_MeshPipeline.cpp` or new file):

| Test | Assertion |
|------|-----------|
| Elevation + BoundarySmooth: node movement count | `nTotalMoved > 0` for a curved mesh |
| Elevation + BoundarySmooth: all O2 nodes valid coords | No NaN/Inf in coords after smooth |
| Elevation + InternalSmooth(V2): Jacobian positive | All cell Jacobians remain > 0 |
| Elevation + InternalSmooth(V2): coords change | coords differ from pre-smooth state |
| Bisection preserves O2 node positions | Bisected mesh nodes are a subset of O2 coords |

**Build and run:**
```bash
cmake --build build -t geom_test_mesh_pipeline -j8
ctest --test-dir build -R geom_mesh_pipeline --output-on-failure
```

### Phase 1: Extract Shared Smooth-Solver Setup

**Goal:** Deduplicate the ~80-line setup block copied across 3 smooth solver variants.

**Implementation:** Extract a file-local helper struct + function in
`Mesh_Elevation_SmoothSolver.cpp`:

```cpp
namespace {
    struct SmoothSolverSetup
    {
        std::set<index> nodesBoundInterpolated;
        tCoordPair boundInterpCoo;
        tCoordPair boundInterpVal;
        // + nanoflann KD-tree (V1Old, V1 only)
    };

    /// Shared setup: assertions, identify boundary nodes, gather boundary
    /// coords/values with MPI ghost pull.
    /// Returns nullopt if nTotalMoved == 0 (nothing to do).
    std::optional<SmoothSolverSetup> PrepareSmoothSolverSetup(
        UnstructuredMesh &mesh);
}
```

**Lines saved:** ~160 (80 lines × 3 copies → 1 copy + 3 calls).

| # | Item | Lines Saved |
|---|------|-------------|
| 1.1 | State assertions + early exit | ~30 |
| 1.2 | `nodesBoundInterpolated` identification | ~20 |
| 1.3 | `boundInterpCoo`/`boundInterpVal` init + populate + ghost pull | ~90 |
| 1.4 | KD-tree construction (V1Old, V1 only) | ~20 |

### Phase 2: Decompose `ElevatedNodesSolveInternalSmoothV2` (864 → ~150 + helpers)

**Goal:** Break the monolithic function into focused helpers.

| # | Extract | Current Lines | Description |
|---|---------|--------------|-------------|
| 2.1 | `BuildNode2NodeFromCells()` | ~15 | Build node-to-node connectivity from cell2node |
| 2.2 | `BuildSmoothDOFVectors()` | ~40 | Initialize DOF arrays (`dispO2`, `bO2`, etc.) |
| 2.3 | `AssembleElasticityRHSAndMatrix()` | ~200 | The active "bisect fem method" block from the `AssembleRHSMat` lambda |
| 2.4 | `ApplyDirichletBCToSparse()` | ~40 | The "find nDiag, set identity row" pattern (repeated 3× → 1×) |
| 2.5 | `LimitDisplacementIncrement()` | ~95 | The `LimitDisp` lambda |
| 2.6 | `BuildSparsePreconditioner()` | ~40 | Eigen sparse matrix construction from block matrix A |
| 2.7 | Delete disabled code blocks | ~175 | Remove `#if 0` "fem method" (lines 1054-1228) and "spring method" (lines 1427-1488) |

After extraction, `ElevatedNodesSolveInternalSmoothV2` becomes a ~150-line
orchestrator calling the helpers in sequence.

### Phase 3: Unify Adj Conversion Methods

**Goal:** Reduce boilerplate in the 12 Adj methods without losing OMP parallelism.

| # | Item | Description | Lines Saved |
|---|------|-------------|-------------|
| 3.1 | Add `FaceIndexGlobal2Local` / `FaceIndexLocal2Global` wrappers | Analogous to Cell/Node/Bnd wrappers, using `face2node` as the pair | 0 (enables 3.2-3.3) |
| 3.2 | Add OMP variant: `ConvertAdjEntriesOMP` | Same as `ConvertAdjEntries` with `#pragma omp parallel for` | ~5 |
| 3.3 | Unify C2F pair with `ConvertAdjEntries` | Replace inline lambda + loops with 2 calls each | ~40 |
| 3.4 | Unify N2CB pair with `ConvertAdjEntriesOMP` | Replace manual loops with 2 calls each | ~20 |

**Not unified (deliberate):**
- **Facial pair:** Fused OMP loop over 3 arrays per face + asymmetric `face2bnd`
  conversion + stricter assertions. Splitting would regress cache locality and
  require assertion wrappers for no readability gain.
- **`bnd2cell` in Primary:** Per-entry assertion depends on row+column index,
  which `ConvertAdjEntries`'s `index(index)` signature cannot express.

### Phase 4: Decompose `RecoverCell2CellAndBnd2Cell` (258 → ~80 + helpers)

**Goal:** Separate the simple non-periodic logic from the complex periodic filter.

| # | Extract | Lines | Description |
|---|---------|-------|-------------|
| 4.1 | `FindCellsSharingFace()` | ~30 | Node-intersection algorithm to find cells sharing a face |
| 4.2 | `FilterPeriodicBnd2Cell()` | ~120 | Ghost-pull candidate cells, pbi-match filter, donor/receiver classification |

The main function becomes: (1) setup, (2) build cell2cell via node neighbors,
(3) build bnd2cell via `FindCellsSharingFace`, (4) if periodic, call
`FilterPeriodicBnd2Cell`.

### Phase 5: Cleanup

| # | Item | Lines Removed | Risk |
|---|------|--------------|------|
| 5.1 | Deprecate `ElevatedNodesSolveInternalSmoothV1Old` | 235 | Add `[[deprecated]]` attribute; later delete |
| 5.2 | Remove debug prints from V1 (`HEre0`, `HEre1`, etc.) | ~10 | Trivial |
| 5.3 | Remove debug prints from V1Old (`std::cout << iN`) | ~5 | Trivial |
| 5.4 | Remove commented-out `InterpolateTopology` (Mesh.cpp:26-42) | ~17 | Dead code |
| 5.5 | Remove dead code after `continue` in BuildCell2Cell.cpp:416-438 | ~22 | Dead code |
| 5.6 | Deduplicate OMP progress reporting in `BuildCell2Cell` | ~20 | Extract `ReportProgress` lambda |

### Phase 6 (future): Decompose into Focused Components

This is the TODO.md Phase 4, deferred to a future pass because it requires
touching the public API (Python bindings, downstream Euler/EulerP/CFV code):

- **`MeshTopology`**: adjacency arrays + state flags + state transitions
- **`MeshIO`**: serialization, VTK/PLT/CGNS/VTKHDF output
- **`MeshElevation`**: O1→O2 elevation, bisection, smooth solvers
- **`MeshReordering`**: local cell reordering, partition start tracking
- `UnstructuredMesh` becomes a thin facade

### Phase 7 (future): Reduce Periodic Branching

`if (isPeriodic)` appears 58 times across Mesh files. Consider extracting
periodic-specific behavior into a policy so non-periodic paths stay clean.
Deferred because the branching is in hot paths where template specialization
may be needed for performance.

---

## 3. Implementation Order and Risk Assessment

| Phase | Risk | Effort | Prerequisite | Lines Changed |
|-------|------|--------|-------------|--------------|
| Phase 0: Tests | None | 0.5 day | — | +150 test |
| Phase 1: Shared Setup | Low | 0.5 day | Phase 0 | ~160 removed |
| Phase 2: V2 Decomposition | Medium | 1-2 days | Phase 0, 1 | ~175 removed (dead), ~400 restructured |
| Phase 3: Adj Unification | Low | 0.5 day | Phase 0 | ~60 removed |
| Phase 4: RecoverC2C Decomposition | Medium | 0.5 day | Phase 0 | ~150 restructured |
| Phase 5: Cleanup | Trivial | 0.5 day | Any time | ~310 removed |
| Phase 6: Components | High | 3-5 days | All above | Major restructure |
| Phase 7: Periodic Policy | High | 2-3 days | Phase 6 | 58 sites |

Each phase must pass the full geom test suite after completion:
```bash
cmake --build build -t geom_unit_tests -j8
ctest --test-dir build -R geom_ --output-on-failure
```

---

## 4. Smooth Solver Variant Status

| Variant | Lines | Method | Status | Recommendation |
|---------|-------|--------|--------|----------------|
| `ElevatedNodesSolveInternalSmooth` | 139 | Dispatch to V1Old/V1/V2 | Active | Keep as dispatcher |
| `ElevatedNodesSolveInternalSmoothV1Old` | 235 | Serial RBF direct solve | **Deprecated** | `[[deprecated]]`, then delete |
| `ElevatedNodesSolveInternalSmoothV1` | 448 | Distributed RBF GMRES | Intermediate | Keep (used by some configs) |
| `ElevatedNodesSolveInternalSmoothV2` | 864 | FEM elasticity GMRES | **Active/Default** | Decompose in Phase 2 |

Evidence V1Old is deprecated:
- Serial-only solve (unscalable for production meshes)
- Has raw `std::cout << iN` debug prints in production code
- Superseded by V1 (parallel GMRES) which was itself superseded by V2 (FEM)
- No case file references `nIter=0` (V1Old path) in production

---

## 5. Adj Conversion Method Summary

After Phase 3:

| Pair | Before | After |
|------|--------|-------|
| Primary | `ConvertAdjEntries` (3/4 arrays) + manual `bnd2cell` | Unchanged (bnd2cell needs positional assertion) |
| PrimaryForBnd | `ConvertAdjEntries` | Unchanged |
| **C2F** | Inline lambda + manual loops | `ConvertAdjEntries` + `FaceIndex` wrappers |
| **N2CB** | Manual OMP loops | `ConvertAdjEntriesOMP` |
| Facial | Fused OMP loop | Unchanged (fused loop is intentional optimization) |
| C2CFace | `ConvertAdjEntries` | Unchanged |

---

## 6. Test Coverage After Refactoring

Existing tests that guard correctness:

| Test File | Tests | np | Coverage |
|-----------|-------|-----|---------|
| `test_MeshPipeline.cpp` | 50 test cases | 1,2,4,8 | Full pipeline, Adj round-trips, elevation, bisection |
| `test_MeshIndexConversion.cpp` | 24 test cases | 1,2,4,8 | All 12 named wrappers |
| `test_MeshDistributedRead.cpp` | 18 test cases | 1,2,4,8 | ReadSerializeAndDistribute |

New tests added in Phase 0 specifically guard elevation + smooth solver paths.
