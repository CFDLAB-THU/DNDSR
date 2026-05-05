# Unified Mesh Helper Design {#mesh_helpers_design}

> **Status:** Implemented. C++ helpers are in `src/Geom/Mesh/Mesh_Helpers.hpp`.
> Python helpers are in `python/DNDSR/Geom/utils.py` as `read_mesh`,
> `prepare_mesh`, `build_bnd_mesh`, `build_fv`, `serialize_mesh`, and
> `mesh_h5_path`. The legacy `create_mesh_from_CGNS` wrapper is preserved
> for backward compatibility.

## Problem

Mesh assembly logic is duplicated across at least 6 sites:

| Site | Language | Lines |
|------|----------|-------|
| `python/DNDSR/Geom/utils.py` `create_mesh_from_CGNS` | Python | ~230 |
| `python/DNDSR/EulerP/EulerP_Solver.py` `ReadMesh` | Python | thin wrapper |
| `src/Euler/EulerSolver_Init.hxx` `ReadMeshAndInitialize` | C++ | ~370 |
| `app/Geom/partitionMeshSerial.cpp` | C++ | ~60 |
| `app/Geom/meshSerial_Test.cpp` | C++ | ~100 |
| `test/Geom/draw_meshes_2D.py` | Python | ~50 (manual) |

Every test that builds a mesh (`test_basic_geom`, `test_basic_fv`,
`test_basic_cfv`, `test_cfv_dissdisp`, `test_basic_eulerP`, `test_solver`,
`test_fv_correctness`, `test_vr_correctness`) calls `create_mesh_from_CGNS`
followed by ad-hoc FV setup.

The problems:
1. The read phase (CGNS/H5/distributed), elevation/bisection, and
   partitioning are tangled with the solver-prep phase (faces, ghost N2CB,
   reorder, VTK) inside one 230-line function.
2. FV construction (14 method calls) is copy-pasted between
   `EulerP_Solver.BuildFV` and `test_basic_fv.get_fv`.
3. The C++ solver does extra post-read steps (elevation smoothing, wall
   distance, coord transforms, serialization) that have no Python equivalent.
4. `name2ID` is only defined in the Serial branch of `create_mesh_from_CGNS`
   but is returned unconditionally (bug: `UnboundLocalError` in
   Parallel/Distributed modes).
5. Partition options (metisType, ufactor, seed, ncuts) are hardcoded in each
   call site rather than passed as a struct.

## Design

Five helpers, each doing one job. They compose linearly:

```
read_mesh  -->  prepare_mesh  -->  build_bnd_mesh
                                   build_fv
                                   serialize_mesh
```

All helpers live in `python/DNDSR/Geom/utils.py` (Python side). The C++
solver (`EulerSolver_Init.hxx`) keeps its own pipeline but should converge
on the same structure over time.

### 1. `read_mesh` -- any source to distributed mesh

```python
def read_mesh(
    mesh_file: str,
    mpi: DNDS.MPIInfo,
    dim: int,
    *,
    # Periodic geometry
    periodic_geometry: dict | None = None,
    periodic_tolerance: float = 1e-9,
    # Read mode (auto-detected from extension if not given)
    read_mode: str | None = None,  # "cgns", "h5"
    # Partition options (for CGNS and h5 distributed modes)
    partition_options: dict | None = None,
    # Elevation / bisection (only for CGNS mode)
    elevation: str = "",   # "" or "O2"
    bisect: int = 0,       # 0..4
    # Serializer factory (for H5 modes)
    serializer_factory: SerializerFactory | None = None,
) -> MeshReadResult:
```

**Returns** a `MeshReadResult` dataclass:

```python
@dataclasses.dataclass
class MeshReadResult:
    mesh: Geom.UnstructuredMesh
    reader: Geom.UnstructuredMeshSerialRW
    name_to_id: Geom.AutoAppendName2ID | None  # None for H5 modes
```

**Auto-detection:**

If `read_mode` is None, infer from file extension:
- `.cgns` -> `"cgns"` (serial CGNS read + Metis partition)
- `.dnds.h5` -> `"h5"` (distributed read: even-split + ParMetis)

Explicit `read_mode` override is still available for edge cases.

**Behavior by mode:**

1. **cgns mode:**
   - ReadFromCGNSSerial -> Deduplicate1to1Periodic -> BuildCell2Cell
   - MeshPartitionCell2Cell (partition_options) -> PartitionReorderToMeshCell2Cell
   - `_build_ghost_primary(mesh)`
   - Optional elevation (O2), optional bisection (0..4), each re-runs
     `_build_ghost_primary`

2. **h5 mode:**
   - Open H5 via serializer_factory (default: `default_serializer_factory()`)
   - `ReadSerializeAndDistribute(serializer, "meshPart", partition_options)`
     -- even-split read + ParMetis repartition, works with any MPI size
   - `_build_ghost_primary(mesh)`

The internal helper `_build_ghost_primary` encapsulates the 5-step
connectivity+ghost sequence identical across all branches:

```python
def _build_ghost_primary(mesh):
    mesh.RecoverNode2CellAndNode2Bnd()
    mesh.RecoverCell2CellAndBnd2Cell()
    mesh.BuildGhostPrimary()
    mesh.AdjGlobal2LocalPrimary()
    mesh.AdjGlobal2LocalN2CB()
```

**What it does NOT do:**
- No faces, no ghost N2CB, no reorder, no VTK, no serial output, no
  wall distance, no coord transforms, no elevation smoothing. The mesh
  is distributed with ghost cells and local indices, but not solver-ready.

### 2. `prepare_mesh` -- distributed mesh to solver-ready

```python
def prepare_mesh(
    mesh: Geom.UnstructuredMesh,
    reader: Geom.UnstructuredMeshSerialRW,
    *,
    # Cell reordering
    reorder_parts: int = 1,
    reorder_inner_parts: int = 1,
    # Serial output
    build_serial_out: bool = True,
    # Wall distance
    wall_dist_predicate: Callable[[int], bool] | None = None,
    wall_dist_options: dict | None = None,
    # Coordinate transforms (default no-op; callers reading from H5
    # should generally omit these since the stored mesh may already
    # have been transformed before serialization)
    coord_scale: float = 1.0,
    coord_rot_z_deg: float = 0.0,
    rectify_near_plane: int = 0,
    rectify_threshold: float = 1e-12,
) -> None:
```

**Mutates `mesh` in place.** Steps:

1. `mesh.ReorderLocalCells(nParts=reorder_parts, nPartsInner=reorder_inner_parts)`
2. `mesh.InterpolateFace()`
3. `mesh.AssertOnFaces()`
4. Ghost N2CB cycle: `AdjLocal2GlobalN2CB` -> `BuildGhostN2CB` -> `AdjGlobal2LocalN2CB`
5. Wall distance: `mesh.BuildNodeWallDist(wall_dist_predicate, wall_dist_options)` if predicate given
6. Serial output: `AdjLocal2GlobalPrimary` -> `reader.BuildSerialOut` -> `AdjGlobal2LocalPrimary` if `build_serial_out`
7. `mesh.RecreatePeriodicNodes()`
8. `mesh.BuildVTKConnectivity()`
9. Coordinate transforms (scale, rotation, rectify) -- applied after
   everything else, matching C++ order. All default to no-op.

Note: elevation smoothing is NOT included here. It is a separate stage
between `prepare_mesh` and `build_bnd_mesh` (see Resolved Questions).

### 3. `build_bnd_mesh` -- boundary mesh extraction

```python
def build_bnd_mesh(
    mesh: Geom.UnstructuredMesh,
    *,
    build_serial_out: bool = True,
) -> BndMeshResult:
```

```python
@dataclasses.dataclass
class BndMeshResult:
    mesh_bnd: Geom.UnstructuredMesh
    reader_bnd: Geom.UnstructuredMeshSerialRW
```

Steps (same as current `create_bnd_mesh` but with explicit serial_out control):
1. `mesh.ConstructBndMesh(mesh_bnd)`
2. Optional serial output for boundary mesh
3. `mesh_bnd.RecreatePeriodicNodes()`
4. `mesh_bnd.BuildVTKConnectivity()`

### 4. `build_fv` -- finite volume construction

```python
def build_fv(
    mpi: DNDS.MPIInfo,
    mesh: Geom.UnstructuredMesh,
    settings: dict | None = None,
) -> CFV.FiniteVolume:
```

Encapsulates the 14-step FV construction currently duplicated in
`EulerP_Solver.BuildFV` and `test_basic_fv.get_fv`:

1. Create `CFV.FiniteVolume(mpi, mesh)`
2. Merge `settings` into defaults via `fv.GetSettings().update(settings)`
3. `fv.ParseSettings(merged)`
4. Cell construction: `SetCellAtrBasic`, `ConstructCellVolume`,
   `ConstructCellBary`, `ConstructCellCent`, `ConstructCellIntJacobiDet`,
   `ConstructCellIntPPhysics`, `ConstructCellAlignedHBox`,
   `ConstructCellMajorHBoxCoordInertia`
5. Face construction: `SetFaceAtrBasic`, `ConstructFaceCent`,
   `ConstructFaceArea`, `ConstructFaceIntJacobiDet`,
   `ConstructFaceIntPPhysics`, `ConstructFaceUnitNorm`,
   `ConstructFaceMeanNorm`
6. `ConstructCellSmoothScale`

### 5. `serialize_mesh` -- write partitioned mesh

```python
def serialize_mesh(
    mesh: Geom.UnstructuredMesh,
    output_path: str,
    mpi: DNDS.MPIInfo,
    *,
    serializer_factory: SerializerFactory | None = None,
) -> None:
```

Steps:
1. `mesh.AdjLocal2GlobalPrimary()`
2. Open serializer, `mesh.WriteSerialize(serializer, "meshPart")`
3. `mesh.AdjGlobal2LocalPrimary()`

This is the Python equivalent of the `partitionMeshOnly` path in C++.

## Composition examples

### Typical solver setup (replaces EulerP_Solver.ReadMesh + BuildFV)

```python
result = read_mesh("wing.cgns", mpi, dim=3, elevation="O2", bisect=1)
prepare_mesh(result.mesh, result.reader, reorder_parts=4,
             wall_dist_predicate=is_wall, wall_dist_options={"method": 1})

# Elevation smoothing -- separate stage, needs BC handler
result.mesh.ElevatedNodesGetBoundarySmooth(wall_predicate)
result.mesh.ElevatedNodesSolveInternalSmooth()

bnd = build_bnd_mesh(result.mesh)
fv = build_fv(mpi, result.mesh, {"intOrder": 3, "maxOrder": 3})
```

### Pre-partition and serialize for later parallel read

```python
result = read_mesh("wing.cgns", mpi, dim=3, elevation="O2")
prepare_mesh(result.mesh, result.reader)
serialize_mesh(result.mesh, mesh_h5_path("wing.cgns", mpi.size, "O2"), mpi)
```

### Later distributed read (any MPI size)

```python
result = read_mesh("wing.cgns_part_4_elevated.dnds.h5", mpi, dim=3)
prepare_mesh(result.mesh, result.reader)
```

### Minimal test (no wall dist, no transforms)

```python
result = read_mesh("Uniform_3x3.cgns", mpi, dim=2,
                   periodic_geometry={"translation1": [3,0,0], "translation2": [0,3,0]})
prepare_mesh(result.mesh, result.reader)
fv = build_fv(mpi, result.mesh)
```

## Migration plan

### Phase 1: Implement helpers

1. Add `MeshReadResult` and `BndMeshResult` dataclasses to `utils.py`.
2. Extract `_build_ghost_primary` from current code.
3. Implement `read_mesh`, `prepare_mesh`, `build_bnd_mesh`, `build_fv`,
   `serialize_mesh` in `utils.py`.
4. Fix the `name2ID` bug (set to `None` for H5 modes, include in
   `MeshReadResult`).

### Phase 2: Migrate callers

5. Rewrite `create_mesh_from_CGNS` as a thin wrapper that calls
   `read_mesh` + `prepare_mesh` with the same signature for backward
   compatibility. Mark it deprecated.
6. Rewrite `create_bnd_mesh` as a thin wrapper around `build_bnd_mesh`.
7. Update `EulerP_Solver.ReadMesh` to use the new helpers.
8. Update `EulerP_Solver.BuildFV` to use `build_fv`.
9. Update tests one by one.

### Phase 3: Remove old wrappers

10. Remove deprecated `create_mesh_from_CGNS` and `create_bnd_mesh` once
    all callers are migrated.

## Resolved questions

1. **H5 file path convention:** `read_mesh` auto-detects the read mode
   from the file path with predefined conventions. When `read_mode` is
   None:
   - `.cgns` -> CGNS serial read
   - `.dnds.h5` -> H5 distributed read (even-split + ParMetis). This is
     the safest default: works with any MPI size regardless of how the
     file was written.
   - Explicit `read_mode` override is still available.

   For the H5 path naming convention used by `serialize_mesh`, a utility
   function `mesh_h5_path(base, mpi_size, elevation, bisect)` is provided:

   ```python
   def mesh_h5_path(
       base: str, mpi_size: int, elevation: str = "", bisect: int = 0
   ) -> str:
       """Build the conventional H5 filename for a partitioned mesh.

       Convention: ``{base}_part_{mpi_size}[_elevated][_bisectN].dnds.h5``
       """
       name = f"{base}_part_{mpi_size}"
       if elevation == "O2":
           name += "_elevated"
       if bisect > 0:
           name += f"_bisect{bisect}"
       name += ".dnds.h5"
       return name
   ```

   This keeps the convention in one place rather than scattered across
   callers.

2. **Elevation smoothing:** Kept as a **separate stage** between
   `prepare_mesh` and `build_bnd_mesh` / `build_fv`. The solver calls
   the smoothing methods directly on the mesh:

   ```python
   result = read_mesh(...)
   prepare_mesh(result.mesh, result.reader)
   # Elevation smoothing -- solver responsibility, needs BC handler
   mesh.ElevatedNodesGetBoundarySmooth(wall_predicate)
   mesh.ElevatedNodesSolveInternalSmooth()  # or V1/V2
   bnd = build_bnd_mesh(result.mesh)
   fv = build_fv(mpi, result.mesh)
   ```

   This avoids coupling `prepare_mesh` to the BC handler and keeps the
   smoothing logic visible at the call site.

3. **Coordinate transforms:** Included in `prepare_mesh` with optional
   parameters defaulting to no-op. When `read_mode` was `"h5_distributed"`
   or `"h5_parallel"` (i.e. reading an already-partitioned mesh), the
   transforms default to being omitted since the stored mesh may already
   have been transformed before serialization. Callers can still pass
   explicit transform parameters if needed.

4. **Symmetry boundary rectification:** Stays in the solver. Requires a
   BC handler to identify symmetry boundaries, so it doesn't belong in
   `prepare_mesh`.
