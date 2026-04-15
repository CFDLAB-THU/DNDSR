# Python Geom Module - Mesh Reader Guide

This document describes how to use the DNDSR Python Geom module for reading and manipulating computational meshes.

## Overview

The `DNDSR.Geom` module provides Python bindings to the C++ geometry library, enabling:
- Reading CGNS mesh files
- Mesh partitioning and distribution
- Order elevation (O1 → O2)
- Mesh bisection for h-refinement
- Boundary mesh extraction
- VTK output generation
- Wall distance computation
- CUDA device offloading

## Setup

Import the module and initialize MPI:

```python
from DNDSR import Geom, DNDS
```

MPI is initialized automatically when `DNDSR.DNDS` is imported (via
`MPI.Init_thread`).  If you need to control MPI initialization yourself
(e.g., to set the thread level), import and initialize `mpi4py` **before**
importing `DNDSR.DNDS`.

## Core Classes

### UnstructuredMesh

The main mesh container class.

```python
from DNDSR import Geom, DNDS

mpi = DNDS.MPIInfo()
mpi.setWorld()
mesh = Geom.UnstructuredMesh(mpi, dim=3)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `NumCell()` | Number of local cells |
| `NumCellGlobal()` | Global cell count (collective) |
| `NumNode()` | Number of local nodes |
| `NumNodeGlobal()` | Global node count (collective) |
| `NumFace()` | Number of local faces |
| `NumFaceGlobal()` | Global face count (collective) |
| `NumBnd()` | Number of local boundary faces |
| `NumBndGlobal()` | Global boundary face count (collective) |
| `NumCellGhost()` | Number of ghost cells |
| `NumNodeGhost()` | Number of ghost nodes |
| `NumFaceGhost()` | Number of ghost faces |
| `NumBndGhost()` | Number of ghost boundary faces |
| `getDim()` | Mesh dimension (2 or 3) |
| `getMPI()` | Return the `MPIInfo` object |
| `RecoverNode2CellAndNode2Bnd()` | Build node connectivity |
| `RecoverCell2CellAndBnd2Cell()` | Build cell/boundary connectivity |
| `BuildGhostPrimary()` | Build ghost cell communication |
| `AdjGlobal2LocalPrimary()` | Convert global indices to local |
| `AdjLocal2GlobalPrimary()` | Convert local indices to global |
| `AdjGlobal2LocalPrimaryForBnd()` | Global→local for boundary mesh |
| `AdjLocal2GlobalPrimaryForBnd()` | Local→global for boundary mesh |
| `AdjGlobal2LocalN2CB()` | Global→local for node-to-cell/bnd |
| `AdjLocal2GlobalN2CB()` | Local→global for node-to-cell/bnd |
| `BuildGhostN2CB()` | Build ghost node-to-cell/bnd comm |
| `InterpolateFace()` | Build face interpolation data |
| `AssertOnFaces()` | Validate face data |
| `BuildVTKConnectivity()` | Prepare for VTK output |
| `RecreatePeriodicNodes()` | Reconstruct periodic node mapping |
| `ConstructBndMesh(bndMesh)` | Extract (dim-1) boundary mesh |
| `ReorderLocalCells(nParts, nPartsInner)` | Reorder cells for cache locality |
| `BuildO2FromO1Elevation(meshO1)` | Elevate O1→O2 in-place |
| `BuildBisectO1FormO2(meshO2)` | Bisect O2→O1 sub-mesh in-place |
| `BuildNodeWallDist(fBndIsWall, options)` | Compute wall distance field |
| `to_device(backend)` | Offload arrays to device (`"CUDA"`) |
| `to_host()` | Pull arrays back to host |
| `getArrayBytes()` | Total memory used by arrays |

**Key Read-Only Members:**

| Member | Type | Description |
|--------|------|-------------|
| `coords` | coordinate pair | Node coordinates |
| `cell2node` | adjacency pair | Cell-to-node connectivity |
| `bnd2node` | adjacency pair | Boundary-to-node connectivity |
| `bnd2cell` | adjacency pair | Boundary-to-cell connectivity |
| `cell2cell` | adjacency pair | Cell-to-cell connectivity |
| `cellElemInfo` | ElemInfo pair | Per-cell element type and zone |
| `bndElemInfo` | ElemInfo pair | Per-bnd element type and zone |
| `cell2face` | adjacency pair | Cell-to-face (after `InterpolateFace`) |
| `face2cell` | adjacency pair | Face-to-cell (after `InterpolateFace`) |
| `face2node` | adjacency pair | Face-to-node (after `InterpolateFace`) |
| `faceElemInfo` | ElemInfo pair | Per-face element type and zone |

> **Note:** `GetCellElement()`, `GetBndElement()`, `IsO1()`, and `IsO2()` are
> C++ methods that are **not** exposed in the Python bindings.  To query element
> types from Python, use `mesh.cellElemInfo[iCell][0].getElemType()` which
> returns a `Geom.Elem.ElemType` enum value.

### ElemInfo

Per-element metadata, accessible via `mesh.cellElemInfo` and `mesh.bndElemInfo`.

```python
info = mesh.cellElemInfo[iCell][0]
elem_type = info.getElemType()   # returns Geom.Elem.ElemType enum
zone = info.zone                 # zone index
```

### UnstructuredMeshSerialRW

Serial mesh reader/writer for CGNS files.

```python
reader = Geom.UnstructuredMeshSerialRW(mesh, 0)
name2ID = reader.ReadFromCGNSSerial("mesh.cgns")
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `ReadFromCGNSSerial(path)` | Read CGNS file; returns `AutoAppendName2ID` |
| `Deduplicate1to1Periodic(tol)` | Merge periodic 1-to-1 connections |
| `BuildCell2Cell()` | Build serial cell-to-cell adjacency |
| `MeshPartitionCell2Cell(options)` | Partition with Metis |
| `PartitionReorderToMeshCell2Cell()` | Reorder to match partition |
| `BuildSerialOut()` | Prepare serial output data |

**Key Members:**

| Member | Access | Description |
|--------|--------|-------------|
| `mesh` | read/write | The `UnstructuredMesh` object |
| `dataIsSerialOut` | read-only | Whether serial output is built |
| `dataIsSerialIn` | read-only | Whether serial input is ready |

### AutoAppendName2ID

Returned by `ReadFromCGNSSerial`. Maps boundary/zone names to integer IDs.

```python
name2ID = reader.ReadFromCGNSSerial(meshFile)
n2id = name2ID.n2id_map   # dict-like: name → ID
```

### ElemType Enum

Available as `Geom.Elem.ElemType`:

| Value | Description |
|-------|-------------|
| `UnknownElem` | Unknown / uninitialized |
| `Line2`, `Line3` | 1D line elements |
| `Tri3`, `Tri6` | Triangle elements |
| `Quad4`, `Quad9` | Quadrilateral elements |
| `Tet4`, `Tet10` | Tetrahedron elements |
| `Hex8`, `Hex27` | Hexahedron elements |
| `Prism6`, `Prism18` | Prism elements |
| `Pyramid5`, `Pyramid14` | Pyramid elements |

> **Note:** The C++ `Elem::Element` struct (with methods `GetDim()`, `GetOrder()`,
> `GetNumNodes()`, `IsO1()`, `GetParamSpace()`, etc.) is **not** exposed in the
> Python bindings.  Only the `ElemType` enum is available.

### WallDistOptions

Nested class `UnstructuredMesh.WallDistOptions` for configuring wall distance
computation:

```python
opts = Geom.UnstructuredMesh.WallDistOptions()
opts.method = 1
opts.subdivide_quad = 5
opts.verbose = 10
opts.wallDistExecution = 4
opts.minWallDist = 1e-10
mesh.BuildNodeWallDist(lambda bnd_id: bnd_id == wall_id, opts)
```

## Mesh Reading Workflow

### Using the High-Level API (Recommended)

The `create_mesh_from_CGNS` function handles the complete mesh pipeline:

```python
from DNDSR.Geom.utils import create_mesh_from_CGNS
from DNDSR import DNDS

mpi = DNDS.MPIInfo()
mpi.setWorld()

mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
)
```

**Full parameter list:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `meshFile` | `str` | required | Path to CGNS mesh file |
| `mpi` | `MPIInfo` | required | MPI context |
| `dim` | `int` | `2` | Mesh dimension (2 or 3) |
| `periodic_tolerance` | `float` | `1e-9` | Tolerance for periodic dedup |
| `inner_process_parts` | `int` | `1` | Cell reordering partitions |
| `second_level_parts` | `int` | `1` | Second-level reordering partitions |
| `periodic_geometry` | `dict` | see below | Periodic transform parameters |
| `readMeshMode` | `str` | `"Serial"` | `"Serial"`, `"Parallel"`, or `"Distributed"` |
| `meshElevation` | `str` | `""` | `""` or `"O2"` for quadratic elevation |
| `meshDirectBisect` | `int` | `0` | Number of bisection levels (0–4) |
| `serializerFactory` | `SerializerFactory` | H5 factory | Serializer for Parallel/Distributed mode |
| `outPltMode` | `str` | `"Serial"` | `"Serial"` to build serial output |

**Periodic geometry default:**

```python
periodic_geometry={
    "translation1": [1, 0, 0],
    "rotationCenter1": [0, 0, 0],
    "eulerAngles1": [0, 0, 0],
    "translation2": [0, 1, 0],
    "rotationCenter2": [0, 0, 0],
    "eulerAngles2": [0, 0, 0],
    "translation3": [0, 0, 1],
    "rotationCenter3": [0, 0, 0],
    "eulerAngles3": [0, 0, 0],
}
```

These are passed directly to `mesh.SetPeriodicGeometry()`, which accepts up to
three periodic direction triples (translation, rotationCenter, eulerAngle).

### readMeshMode Options

| Mode | Description |
|------|-------------|
| `"Serial"` | Read CGNS on rank 0, partition with Metis, distribute. Returns `name2ID`. |
| `"Parallel"` | Read pre-partitioned H5 files (one per rank). `name2ID` may be `None`. |
| `"Distributed"` | Read H5 with even-split + ParMetis repartition. Works with any MPI rank count. `name2ID` may be `None`. |

> **Warning:** In `"Parallel"` and `"Distributed"` modes, `name2ID` is not read
> from the serializer and will be `None`.  Only `"Serial"` mode returns a valid
> `AutoAppendName2ID`.

### Common Usage Patterns

```python
# Basic read
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
)

# With order elevation (O1 → O2)
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
    meshElevation="O2",
)

# With bisection (h-refinement)
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
    meshDirectBisect=1,
)

# Combined elevation + bisection
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
    meshElevation="O2",
    meshDirectBisect=2,
)

# With cell reordering for cache locality
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
    inner_process_parts=4,
    second_level_parts=4,
)

# Pre-partitioned H5 read
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="mesh.cgns",
    mpi=mpi,
    dim=3,
    readMeshMode="Parallel",
)

# Distributed H5 read with auto-repartitioning
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="mesh.cgns",
    mpi=mpi,
    dim=3,
    readMeshMode="Distributed",
)
```

### What `create_mesh_from_CGNS` Does

The function runs the complete mesh pipeline so you do not need to call these
steps manually.  For reference, the Serial-mode pipeline is:

1. `reader.ReadFromCGNSSerial(meshFile)` — read CGNS on rank 0
2. `reader.Deduplicate1to1Periodic(tolerance)` — merge periodic connections
3. `reader.BuildCell2Cell()` — build serial cell adjacency
4. `reader.MeshPartitionCell2Cell({...})` — partition with Metis
5. `reader.PartitionReorderToMeshCell2Cell()` — reorder to match partition
6. `mesh.RecoverNode2CellAndNode2Bnd()` — build node→cell/bnd
7. `mesh.RecoverCell2CellAndBnd2Cell()` — build cell→cell/bnd
8. `mesh.BuildGhostPrimary()` — ghost cell communication
9. `mesh.AdjGlobal2LocalPrimary()` — global→local index
10. `mesh.AdjGlobal2LocalN2CB()` — global→local node→cell/bnd
11. (If `meshElevation == "O2"`) Build O2 mesh and swap
12. (If `meshDirectBisect > 0`) Elevate→bisect loop
13. `mesh.ReorderLocalCells(nParts, nPartsInner)` — cell reordering
14. `mesh.InterpolateFace()` — build face data
15. `mesh.AssertOnFaces()` — validate faces
16. `mesh.AdjLocal2GlobalN2CB()` / `BuildGhostN2CB()` / `AdjGlobal2LocalN2CB()`
17. (If `outPltMode == "Serial"`) Build serial output
18. `mesh.RecreatePeriodicNodes()` — reconstruct periodic mapping
19. `mesh.BuildVTKConnectivity()` — prepare VTK output

### Low-Level Serial Read (Manual Pipeline)

If you need fine-grained control, you can run the pipeline manually:

```python
from DNDSR import Geom, DNDS
import os

mpi = DNDS.MPIInfo()
mpi.setWorld()

mesh = Geom.UnstructuredMesh(mpi, dim=2)
reader = Geom.UnstructuredMeshSerialRW(mesh, 0)

meshFile = "path/to/mesh.cgns"
assert os.path.isfile(meshFile)
name2ID = reader.ReadFromCGNSSerial(meshFile)

reader.Deduplicate1to1Periodic(1e-9)
reader.BuildCell2Cell()
reader.MeshPartitionCell2Cell({
    "metisType": "KWAY",
    "metisUfactor": 5,
    "metisSeed": 0,
    "metisNcuts": 3,
})
reader.PartitionReorderToMeshCell2Cell()

mesh.RecoverNode2CellAndNode2Bnd()
mesh.RecoverCell2CellAndBnd2Cell()
mesh.BuildGhostPrimary()
mesh.AdjGlobal2LocalPrimary()
mesh.AdjGlobal2LocalN2CB()
```

## Order Elevation (O1 → O2)

Elevation converts linear elements to quadratic elements:
- Quad4 → Quad9
- Tri3 → Tri6
- Hex8 → Hex27
- Tet4 → Tet10
- Prism6 → Prism18
- Pyramid5 → Pyramid14

```python
meshO2 = Geom.UnstructuredMesh(mpi, dim)
meshO2.BuildO2FromO1Elevation(meshO1)

meshO2.RecoverNode2CellAndNode2Bnd()
meshO2.RecoverCell2CellAndBnd2Cell()
meshO2.BuildGhostPrimary()
meshO2.AdjGlobal2LocalPrimary()
meshO2.AdjGlobal2LocalN2CB()
```

**Node count increase for UniformSquare_10 (10×10 Quad4):**
- O1: 121 nodes (11×11 grid)
- O2: 441 nodes (+320 new nodes)
  - 220 edge midpoints (110 horizontal + 110 vertical)
  - 100 cell centers

## Mesh Bisection (h-refinement)

Bisection splits O2 elements into O1 sub-elements:

| O2 Element | Sub-elements | O1 Type |
|------------|--------------|---------|
| Quad9 | 4 | Quad4 |
| Tri6 | 4 | Tri3 |
| Hex27 | 8 | Hex8 |
| Tet10 | 8 | Tet4 |
| Prism18 | 8 | Prism6 |
| Pyramid14 | 12 | Pyramid5 |

```python
meshO2 = Geom.UnstructuredMesh(mpi, dim)
meshO2.BuildO2FromO1Elevation(meshO1)

meshO2.RecoverNode2CellAndNode2Bnd()
meshO2.RecoverCell2CellAndBnd2Cell()
meshO2.BuildGhostPrimary()

meshO1B = Geom.UnstructuredMesh(mpi, dim)
meshO1B.BuildBisectO1FormO2(meshO2)

meshO1B.RecoverNode2CellAndNode2Bnd()
meshO1B.RecoverCell2CellAndBnd2Cell()
meshO1B.BuildGhostPrimary()
meshO1B.AdjGlobal2LocalPrimary()
meshO1B.AdjGlobal2LocalN2CB()
```

**Cell count progression for UniformSquare_10:**
- Original: 100 Quad4 cells
- Elevated: 100 Quad9 cells
- Bisected: 400 Quad4 cells (100 × 4)

## Periodic Meshes

For meshes with periodic boundaries:

```python
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="IV10_10.cgns",
    mpi=mpi,
    dim=2,
    periodic_geometry={
        "translation1": [10, 0, 0],
        "rotationCenter1": [0, 0, 0],
        "eulerAngles1": [0, 0, 0],
    },
    periodic_tolerance=1e-9,
)
```

`SetPeriodicGeometry` accepts up to three periodic direction triples:
`translation1`/`rotationCenter1`/`eulerAngles1` through
`translation3`/`rotationCenter3`/`eulerAngles3`.

## Boundary Mesh Extraction

```python
from DNDSR.Geom.utils import create_bnd_mesh

meshBnd, readerBnd = create_bnd_mesh(mesh)

# Access boundary information
n2id = name2ID.n2id_map
id2name = {v: k for k, v in n2id.items()}

for iBnd in range(mesh.NumBnd()):
    elem_type = mesh.bndElemInfo[iBnd][0].getElemType()
    zone = mesh.bndElemInfo[iBnd][0].zone
    print(f"Boundary {iBnd}: type={elem_type}, zone={zone}")
```

## Wall Distance Computation

```python
# Identify wall boundaries by name
n2id = name2ID.n2id_map
id2name = {v: k for k, v in n2id.items()}

def id_is_wall(bnd_id):
    if bnd_id in id2name:
        name = id2name[bnd_id].upper()
        return "WALL" in name
    return False

opts = Geom.UnstructuredMesh.WallDistOptions()
opts.method = 1
opts.subdivide_quad = 5
opts.verbose = 10
opts.wallDistExecution = 4
mesh.BuildNodeWallDist(id_is_wall, opts)
```

## CUDA Device Offloading

```python
mesh.coords.to_device("CUDA")
mesh.to_device("CUDA")

# ... GPU computation ...

mesh.to_host()
```

## VTK Output

VTK connectivity is built automatically by `create_mesh_from_CGNS`.  For
manual control:

```python
mesh.BuildVTKConnectivity()

# For serial output
mesh.AdjLocal2GlobalPrimary()
reader.BuildSerialOut()
mesh.AdjGlobal2LocalPrimary()
```

## Querying Mesh State

### Element Information

```python
# Get element type for a cell
elem_type = mesh.cellElemInfo[iCell][0].getElemType()  # Geom.Elem.ElemType enum

# Compare with known types
if elem_type == Geom.Elem.Quad4:
    print("Linear quad")
elif elem_type == Geom.Elem.Quad9:
    print("Quadratic quad")

# Get zone index
zone = mesh.cellElemInfo[iCell][0].zone
```

### Mesh Properties

```python
# Global counts
nCellGlobal = mesh.NumCellGlobal()
nNodeGlobal = mesh.NumNodeGlobal()

# Local counts
nCellLocal = mesh.NumCell()
nNodeLocal = mesh.NumNode()
nGhost = mesh.NumCellGhost()
```

> **Note:** The C++ methods `IsO1()` and `IsO2()`, as well as the
> `MeshElevationState` enum (`Elevation_Untouched`, `Elevation_O1O2`), are
> **not** exposed in the Python bindings.  To check whether a mesh uses O2
> elements, inspect the element types via `cellElemInfo[i][0].getElemType()`.

> **Note:** The `MeshAdjState` enum values (`Adj_Unknown`, `Adj_PointToLocal`,
> `Adj_PointToGlobal`) are **not** exposed in the Python bindings.  The
> `adjPrimaryState` member is accessible but its integer values should be
> compared against the C++ enum: 0 = Unknown, 1 = Local, 2 = Global.

## References

- See `test/Geom/test_basic_geom.py` for Python usage examples
- See `python/DNDSR/Geom/utils.py` for the Python API implementation
- See `src/Geom/Mesh_bind.hpp` for the complete list of Python-exposed methods
- See `src/Geom/Elements_bind.hpp` for the `ElemType` enum bindings
- See `docs/Paradigm.md` for overall design philosophy
