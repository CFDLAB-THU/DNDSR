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

## Core Classes

### UnstructuredMesh

The main mesh container class.

```python
from DNDSR import Geom, DNDS

# Create mesh with MPI context
mpi = DNDS.MPIInfo()
mpi.setWorld()
mesh = Geom.UnstructuredMesh(mpi, dim=3)  # 3D mesh
```

**Key Methods:**
- `NumCell()` - Number of local cells
- `NumCellGlobal()` - Global cell count (across all MPI ranks)
- `NumNode()` - Number of local nodes
- `NumNodeGlobal()` - Global node count
- `GetCellElement(i)` - Get element type for cell i
- `RecoverNode2CellAndNode2Bnd()` - Build node connectivity
- `BuildGhostPrimary()` - Build ghost cell communication
- `BuildVTKConnectivity()` - Prepare for VTK output

### UnstructuredMeshSerialRW

Serial mesh reader/writer for CGNS files.

```python
reader = Geom.UnstructuredMeshSerialRW(mesh, 0)
name2ID = reader.ReadFromCGNSSerial("mesh.cgns")
```

## Mesh Reading Workflow

### Basic Serial Read

```python
from DNDSR import Geom, DNDS
import os

# Initialize MPI
mpi = DNDS.MPIInfo()
mpi.setWorld()

# Create mesh and reader
mesh = Geom.UnstructuredMesh(mpi, dim=2)
reader = Geom.UnstructuredMeshSerialRW(mesh, 0)

# Read CGNS file
meshFile = "path/to/mesh.cgns"
assert os.path.isfile(meshFile)
name2ID = reader.ReadFromCGNSSerial(meshFile)

# Process mesh
reader.Deduplicate1to1Periodic(tolerance=1e-9)
reader.BuildCell2Cell()
reader.MeshPartitionCell2Cell({
    "metisType": "KWAY",
    "metisUfactor": 5,
    "metisSeed": 0,
    "metisNcuts": 3,
})
reader.PartitionReorderToMeshCell2Cell()

# Build connectivity
mesh.RecoverNode2CellAndNode2Bnd()
mesh.RecoverCell2CellAndBnd2Cell()
mesh.BuildGhostPrimary()
mesh.AdjGlobal2LocalPrimary()
mesh.AdjGlobal2LocalN2CB()
```

### Using the High-Level API

For convenience, use the `create_mesh_from_CGNS` function:

```python
from DNDSR.Geom.utils import create_mesh_from_CGNS
from DNDSR import DNDS

mpi = DNDS.MPIInfo()
mpi.setWorld()

# Basic usage
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
    meshElevation="O2",  # Elevate to quadratic elements
)

# With bisection (h-refinement)
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
    meshDirectBisect=1,  # One level of bisection
)

# Combined elevation + bisection
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
    meshElevation="O2",
    meshDirectBisect=2,  # Two levels of bisection
)
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
# Create O2 mesh from O1 mesh
meshO1 = ...  # existing O1 mesh
meshO2 = Geom.UnstructuredMesh(mpi, dim)
meshO2.BuildO2FromO1Elevation(meshO1)

# Rebuild connectivity
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
# Elevate to O2, then bisect back to O1
meshO2 = Geom.UnstructuredMesh(mpi, dim)
meshO2.BuildO2FromO1Elevation(meshO1)

meshO2.RecoverNode2CellAndNode2Bnd()
meshO2.RecoverCell2CellAndBnd2Cell()
meshO2.BuildGhostPrimary()

# Must be in global state for bisection
meshO2.AdjLocal2GlobalPrimary()

# Bisect O2 → O1 sub-mesh
meshO1B = Geom.UnstructuredMesh(mpi, dim)
meshO1B.BuildBisectO1FormO2(meshO2)

# Rebuild connectivity
meshO1B.RecoverNode2CellAndNode2Bnd()
meshO1B.RecoverCell2CellAndBnd2Cell()
meshO1B.BuildGhostPrimary()
meshO1B.AdjGlobal2LocalPrimary()
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
        "eulerAngles3": [0, 0, 0],
    },
    periodic_tolerance=1e-9,
)
```

## Boundary Mesh Extraction

```python
from DNDSR.Geom.utils import create_bnd_mesh

# Create boundary mesh (dim-1)
meshBnd, readerBnd = create_bnd_mesh(mesh)

# Access boundary information
for iBnd in range(mesh.NumBnd()):
    elem = mesh.GetBndElement(iBnd)
    bndId = mesh.bndElemInfo[iBnd].getBndID()
    print(f"Boundary {iBnd}: type={elem.type}, ID={bndId}")
```

## VTK Output

```python
# Build VTK connectivity for visualization
mesh.BuildVTKConnectivity()

# For serial output
mesh.AdjLocal2GlobalPrimary()
reader.BuildSerialOut()
mesh.AdjGlobal2LocalPrimary()
```

## Parallel Mesh Reading

For pre-partitioned meshes (H5 format):

```python
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="mesh.cgns",
    mpi=mpi,
    dim=3,
    readMeshMode="Parallel",  # Read from H5 partition files
)
```

## Common Operations

### Get Element Information

```python
elem = mesh.GetCellElement(iCell)
print(f"Type: {elem.type}")  # ElemType enum
print(f"Dimension: {elem.GetDim()}")
print(f"Order: {elem.GetOrder()}")
print(f"Num nodes: {elem.GetNumNodes()}")

# Element traits
from DNDSR.Geom.Elem import Element
e = Element(Elem.Quad4)
print(f"Is O1: {e.IsO1()}")
print(f"Param space: {e.GetParamSpace()}")
```

### Check Mesh Properties

```python
# Global counts
nCellGlobal = mesh.NumCellGlobal()
nNodeGlobal = mesh.NumNodeGlobal()

# Local counts
nCellLocal = mesh.NumCell()
nNodeLocal = mesh.NumNode()
nGhost = mesh.NumCellGhost()

# Check if O2
isO2 = mesh.IsO2()

# Check elevation state
# MeshElevationState: Elevation_Untouched, Elevation_O1O2, etc.
```

## Error Handling

Common assertions and checks:

```python
from DNDS import DNDS_assert, DNDS_check_throw

# Check mesh state
DNDS_check_throw(mesh.adjPrimaryState == DNDS.Adj_PointToLocal)

# Check before elevation
DNDS_check_throw(meshO1.IsO1())  # Must be O1

# Check before bisection
DNDS_check_throw(meshO2.adjPrimaryState == DNDS.Adj_PointToGlobal)
```

## References

- See `test/cpp/Geom/test_MeshPipeline.cpp` for C++ usage examples
- See `python/DNDSR/Geom/utils.py` for Python API implementation
- See `docs/Paradigm.md` for overall design philosophy
