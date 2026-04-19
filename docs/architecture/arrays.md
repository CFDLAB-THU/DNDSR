# Array Infrastructure {#array_infrastructure}

@tableofcontents

CFD on unstructured meshes requires storing per-cell data where the "width"
of each row may differ (e.g. the number of neighbor cells varies).  At the
same time, the data must be distributed across MPI ranks with halo (ghost)
exchange.  DNDSR addresses both problems with a single family of array
classes that unifies dense and sparse storage, MPI communication, and
Eigen interoperability.

This page explains the design from bottom to top.  For working code
examples, see @ref array_usage.

## Design Goals

1. **One container for all per-entity data** -- cell volumes (fixed 1
   column), conservative variables (fixed 5 columns), cell-to-node
   connectivity (variable columns).  The same `Array` template handles
   all three by choosing different template parameters.
2. **Zero-copy Eigen access** -- returning `Eigen::Map` from `operator[]`
   so linear algebra works directly on array memory without copies.
3. **Transparent MPI ghost exchange** -- a single `startPersistentPull()`
   / `waitPersistentPull()` pair refreshes all ghost data using
   persistent MPI requests.
4. **Checkpoint / restart with redistribution** -- write from 4 ranks,
   read back on 8, without rewriting the solver.

## Class Hierarchy

```
Array<T, rs, rm>                  Core 2D container (5 memory layouts)
  └── ParArray<T, rs, rm>         + MPI context, global mapping, serialization
        ├── ArrayAdjacency        CSR index connectivity
        ├── ArrayEigenVector       Per-row Eigen::Vector
        ├── ArrayEigenMatrix       Per-row Eigen::Matrix
        └── ArrayEigenUniMatrixBatch   Per-row batch of uniform matrices

ArrayTransformer<T, rs, rm>       Ghost communication engine

ArrayPair<TArray>                 Bundle: father + son + transformer

ArrayDof<n_m, n_n>               ArrayEigenMatrixPair + MPI vector-space ops
```

Each layer adds one concern.  Code that only needs local storage uses
`Array`; code that needs MPI uses `ParArray`; code that needs ghosts
uses `ArrayPair`.

## Array -- Core 2D Container

**Source:** `DNDS/Array.hpp`, class at line 60; layout logic in
`DNDS/ArrayBasic.hpp`, line 73.

```cpp
template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size,
          rowsize _align = NoAlign>
class Array;
```

### Why Three Size Parameters?

Unstructured-mesh data falls into three categories:

| Category | Example | `_row_size` | `_row_max` |
|---|---|---|---|
| Every row is the same width, known at compile time | Cell volume (1 real), Euler state (5 reals) | `>=0` (e.g. `5`) | (ignored) |
| Every row is the same width, but the width is decided at runtime | Reconstruction coefficients where the polynomial order is a runtime setting | `DynamicSize` | (ignored) |
| Each row has a different width | `cell2node` connectivity (a triangle has 3 nodes, a quad has 4) | `NonUniformSize` | controls padding vs. CSR |

The third parameter `_row_max` only matters when `_row_size = NonUniformSize`.
It selects between a padded table (fast random access, wastes some memory)
and compressed sparse row (no waste, but row-start indirection):

| `_row_max` | Layout | Trade-off |
|---|---|---|
| `>=0` (compile-time max) | `TABLE_StaticMax` | Rows padded to max; O(1) access; memory = n * max |
| `DynamicSize` (runtime max) | `TABLE_Max` | Same but max is set at `Resize` time |
| `NonUniformSize` | `CSR` | Flat buffer + `pRowStart[n+1]`; no waste; needs `Compress()` before MPI |

The full layout table is at `DNDS/ArrayBasic.hpp:46`.

### CSR Compressed vs. Decompressed

CSR arrays have two internal modes (see `Array.hpp:219`):

- **Decompressed**: a `vector<vector<T>>` -- each inner vector is one row.
  Allows `ResizeRow()` to grow or shrink individual rows incrementally.
  Used during mesh construction when connectivity is built row by row.
- **Compressed**: a single flat `vector<T>` plus `pRowStart`.  Required
  before MPI communication or serialization.

Call `Compress()` to pack and `Decompress()` to unpack.  Most code
compresses once after construction and never decompresses again.

## ParArray -- MPI-Aware Array

**Source:** `DNDS/ArrayTransformer.hpp`, line 35.

`ParArray` inherits from `Array` and adds two things:

1. **`MPIInfo mpi`** -- communicator handle, rank, size.
2. **`pLGlobalMapping`** -- a table mapping local row indices to a global
   index space.  Each rank owns a contiguous slice:
   rank 0 owns `[0, n0)`, rank 1 owns `[n0, n0+n1)`, etc.

`createGlobalMapping()` is a collective call that gathers all ranks'
sizes and builds this table.  After that, `pLGlobalMapping->operator()(localRow)`
returns the global index, and `globalSize()` returns the total across
all ranks.

`ParArray` also overrides `WriteSerializer` / `ReadSerializer` to add
MPI-collective HDF5 I/O (see @ref serialization).

## ArrayTransformer -- Ghost Communication

**Source:** `DNDS/ArrayTransformer.hpp`, line 342.

### The Father / Son Model

When a mesh is partitioned, each rank owns some cells (the **father**
array).  Stencil-based schemes (finite volume, reconstruction) also need
data from neighboring cells on other ranks.  These are stored in a
**son** (ghost) array.

A unified index treats them as one array:

```
index:  0 ... fatherSize-1 | fatherSize ... fatherSize+sonSize-1
        ─────── father ──── ──────────── son ────────────────────
        (locally owned)      (copies from other ranks)
```

`ArrayTransformer` manages this pair.  It holds two `ParArray` shared
pointers (`father`, `son`) plus the MPI machinery for exchanging data.

### Setup: Four Steps

The ghost mapping is built once during mesh construction.  After that,
the data can be pulled (or pushed) any number of times without rebuilding.

**Step 1: Attach.**  `setFatherSon(father, son)` connects the two arrays
and copies the MPI context.

**Step 2: Global mapping.**  `createFatherGlobalMapping()` is collective.
Every rank learns where every other rank's rows sit in the global index
space.

**Step 3: Ghost mapping.**  `createGhostMapping(pullIndexGlobal)` takes
a vector of global row indices this rank needs.  The transformer
determines which ranks own them and builds push/pull tables.
(See `ArrayTransformer.hpp:498` for the pull-based variant.)

**Step 4: MPI types.**  `createMPITypes()` builds `MPI_Type_create_hindexed`
datatypes that describe the scattered memory layout of the rows to send
and receive.  It also resizes the son array to match the ghost count.
(See `ArrayTransformer.hpp:541`.)

### Communication: Persistent Requests

After setup, ghost data is refreshed with persistent MPI:

```cpp
trans.initPersistentPull();     // MPI_Send_init + MPI_Recv_init (once)
// ...each iteration:
trans.startPersistentPull();    // MPI_Startall
trans.waitPersistentPull();     // MPI_Waitall
// ...at cleanup:
trans.clearPersistentPull();
```

Persistent requests survive across multiple start/wait cycles, avoiding
the overhead of re-posting sends and receives every time step.  The
push direction (`initPersistentPush`, etc.) works the same way but moves
data from son back to father (used by e.g. assembly-style operations).

### Borrowing Ghost Structure

When multiple arrays share the same ghost structure (e.g. the DOF array
and the gradient array both live on the same cell partition),
`BorrowGGIndexing(otherTransformer)` copies the ghost/global mapping
without repeating the expensive collective setup.  Only `createMPITypes()`
needs to run again because the MPI datatypes depend on the element size.
See `ArrayTransformer.hpp:466`.

## ArrayPair -- Convenience Bundle

**Source:** `DNDS/ArrayPair.hpp`, line 120.

Most application code does not use `ArrayTransformer` directly.
`ArrayPair` bundles father + son + transformer:

```cpp
template <class TArray = ParArray<real, 1>>
struct ArrayPair {
    ssp<TArray> father;
    ssp<TArray> son;
    TTrans       trans;
};
```

`operator[](i)` routes to father if `i < father->Size()`, son otherwise.
This lets stencil loops iterate over `[0, pair.Size())` without caring
which cells are local and which are ghosts.

The most common pattern is `BorrowAndPull(primary)`: given a primary
pair whose ghost mapping is already set up (typically `cell2cell` or
`coords`), borrow its mapping, build MPI types for the new element size,
and pull data in one call.  See `ArrayPair.hpp:269`.

### Type Aliases

The project defines convenience aliases for the most common array-pair
combinations (see `ArrayPair.hpp:624`):

| Alias | Underlying | Use |
|---|---|---|
| `ArrayAdjacencyPair<rs, rm>` | `ArrayPair<ArrayAdjacency<rs, rm>>` | Mesh connectivity |
| `ArrayEigenVectorPair<N>` | `ArrayPair<ArrayEigenVector<N>>` | Per-entity vectors (coords) |
| `ArrayEigenMatrixPair<M, N>` | `ArrayPair<ArrayEigenMatrix<M, N>>` | Per-entity matrices |
| `ArrayEigenUniMatrixBatchPair<M, N>` | `ArrayPair<ArrayEigenUniMatrixBatch<M, N>>` | Quadrature-point data |

## ArrayDerived -- Typed Wrappers

Each derived class inherits from `ParArray` and overrides `operator[]` to
return a typed view instead of a raw `T*`.

### ArrayAdjacency

**Source:** `DNDS/ArrayDerived/ArrayAdjacency.hpp`, line 24.

Stores index connectivity (cell-to-node, cell-to-cell, etc.).
`operator[](i)` returns an `AdjacencyRow` -- a lightweight span that
supports `size()`, range-based `for`, and assignment from `std::vector`.
This is the workhorse for all mesh topology arrays.

### ArrayEigenVector

**Source:** `DNDS/ArrayDerived/ArrayEigenVector.hpp`, line 19.

Each row is an `Eigen::Vector<real, N>`.  `operator[](i)` returns
`Eigen::Map<Vector>`, so Eigen operations work directly on the array's
memory.  Used for node coordinates (`ArrayEigenVectorPair<3>`).

### ArrayEigenMatrix

**Source:** `DNDS/ArrayDerived/ArrayEigenMatrix.hpp`, line 25.

Each row stores an `M x N` matrix.  The underlying `ParArray` row size
is `M * N`.  `operator[](i)` returns `Eigen::Map<Matrix>`.  Used for
per-cell Jacobians, gradient matrices, and the DOF arrays (via
`ArrayDof`).

### ArrayEigenUniMatrixBatch

**Source:** `DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp`, line 19.

A CSR array where each row stores a *batch* of identically-sized
`M x N` matrices.  `RowSize(i)` returns the batch count (not the raw
element count).  `operator()(i, j)` returns the `j`-th matrix in row
`i`'s batch.  Used for per-quadrature-point data such as Jacobians and
basis function coefficients inside `FiniteVolume` and
`VariationalReconstruction`.

## ArrayDof -- DOF Arrays with Vector-Space Operations

**Source:** `DNDS/ArrayDOF.hpp`, line 134.

```cpp
template <int n_m, int n_n>
class ArrayDof : public ArrayEigenMatrixPair<n_m, n_n>;
```

`ArrayDof` is the primary data structure for solver state.  It inherits
the father/son/transformer machinery from `ArrayEigenMatrixPair` and adds
**MPI-collective vector-space operations**: `norm2()`, `dot()`, `+=`,
`addTo(R, alpha)` (AXPY), `setConstant()`.  These are implemented in
`DNDS/ArrayDOF_op.hxx` with specializations for both CPU and CUDA
backends.

### CFV DOF Typedefs

Defined in `CFV/VRDefines.hpp` (lines 79-85):

| Typedef | Template | Per-cell shape | Purpose |
|---|---|---|---|
| `tUDof<N>` | `ArrayDof<N, 1>` | `N x 1` | Cell-mean conservative variables (e.g. rho, rho*u, rho*v, rho*w, rho*E) |
| `tURec<N>` | `ArrayDof<DynamicSize, N>` | `nDOF x N` | High-order reconstruction coefficients.  `nDOF` depends on the polynomial order and is set at runtime. |
| `tUGrad<N, dim>` | `ArrayDof<dim, N>` | `dim x N` | Spatial gradients of the N variables |

If you are writing a new PDE solver on top of DNDSR, these are the
arrays you will use for your unknowns.  `FiniteVolume::BuildUDof`,
`BuildURec`, and `BuildUGrad` allocate them with the correct sizes and
ghost mappings from the mesh.
