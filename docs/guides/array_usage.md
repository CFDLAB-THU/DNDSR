# Array Usage Guide {#array_usage}

@tableofcontents

This guide walks through the DNDSR array infrastructure from the
perspective of someone writing a new module or solver on top of it.
For the design rationale see @ref array_infrastructure.

## When to Use Which Array Type

| You need... | Use | Why |
|---|---|---|
| Local scratch storage (no MPI) | `Array<T, rs>` | Lightest weight; no MPI overhead |
| Per-entity data on a distributed mesh | `ArrayPair<ParArray<T, rs>>` | Father+son+transformer in one object |
| Mesh connectivity (variable row width) | `ArrayAdjacencyPair<NonUniformSize>` | CSR storage + `AdjacencyRow` iteration |
| Per-cell Eigen vectors (coords, fluxes) | `ArrayEigenVectorPair<N>` | `operator[]` returns `Eigen::Map<Vector>` |
| Solver DOFs with norm/dot/AXPY | `ArrayDof<M, N>` (or `tUDof`/`tURec`/`tUGrad`) | Adds MPI-collective vector ops |

## Creating and Filling a Basic Array

The simplest case is a local array with a fixed row width.
(Source reference: `DNDS/Array.hpp:60` for the class,
`DNDS/Array.hpp:346` for `Resize`.)

```cpp
#include "DNDS/Array.hpp"
using namespace DNDS;

// 100 rows, 5 columns.  The "5" is a compile-time constant so the
// compiler can optimize indexing into a stride-5 multiply.
Array<real, 5> a;
a.Resize(100);

// operator[] returns T* (pointer to row start).  Fast but untyped.
a[0][0] = 1.0;
a[0][4] = 5.0;

// operator()(row, col) is equivalent but bounds-checkable in debug.
a(42, 3) = 3.14;
```

When the row width is not known until runtime (e.g. reconstruction order
is a user setting), use `DynamicSize`:

```cpp
// Row width decided at runtime.  Every row has the same width.
Array<real, DynamicSize> b;
b.Resize(50, /*rowWidth=*/7);
```

### CSR Arrays (Variable Row Width)

Mesh connectivity is the classic CSR case: a triangle has 3 nodes, a
quad has 4, a hex has 8.  Use `NonUniformSize` for both `_row_size`
and `_row_max`:

```cpp
Array<index, NonUniformSize, NonUniformSize> adj;

// Resize with a callable that returns each row's width.
// Row i gets (i + 2) elements.
adj.Resize(4, [](index i) -> rowsize {
    return static_cast<rowsize>(i + 2);
});
// Row 0: 2 elements, row 1: 3, row 2: 4, row 3: 5.

// After Resize the array is in compressed (flat) form.
// To modify individual row sizes, decompress first:
adj.Decompress();
adj.ResizeRow(2, 6);   // grow row 2 from 4 to 6 elements
adj.Compress();        // pack back into flat buffer
```

**Why compress/decompress?**  The decompressed form uses a
`vector<vector<T>>` which allows arbitrary per-row resizing but is
scattered in memory.  The compressed form is a single contiguous
allocation, which is required for MPI communication (the MPI datatype
describes offsets into one buffer) and for CUDA device transfer.

## Distributing Data with ParArray

`ParArray` adds MPI awareness to `Array`.  Every rank allocates its own
portion; the global index mapping tells each rank where its rows sit in
the global index space.
(Source: `DNDS/ArrayTransformer.hpp:35`.)

```cpp
#include "DNDS/ArrayTransformer.hpp"
using namespace DNDS;

MPIInfo mpi;
mpi.setWorld();

auto father = std::make_shared<ParArray<real, 5>>(mpi);
father->Resize(100);   // this rank owns 100 rows

// Build the global offset table (collective).
// After this call, father->pLGlobalMapping->operator()(localRow)
// returns the global index.
father->createGlobalMapping();
```

You rarely use `ParArray` alone -- the ghost layer (`ArrayTransformer`)
is almost always needed.

## Ghost Communication with ArrayTransformer

Finite volume schemes need data from cells on neighboring ranks.
`ArrayTransformer` manages a **father** (owned) and **son** (ghost)
array pair and the MPI machinery to exchange data between them.
(Source: `DNDS/ArrayTransformer.hpp:342`.)

The setup has four steps.  Each step is explained with *why* it exists:

```cpp
auto father = std::make_shared<ParArray<real, 5>>(mpi);
auto son    = std::make_shared<ParArray<real, 5>>(mpi);
father->Resize(nLocal);

ArrayTransformer<real, 5> trans;

// Step 1: Attach father and son.
// The transformer needs both pointers so it can build MPI types
// that describe memory layouts in both arrays.
trans.setFatherSon(father, son);

// Step 2: Build global index mapping (collective).
// Every rank learns the global offset of every other rank's data.
// This is needed to convert the "global indices I need" in step 3
// into "rank + local offset" pairs.
trans.createFatherGlobalMapping();

// Step 3: Specify which global rows this rank needs as ghosts.
// For a mesh solver, these come from cell2cell adjacency:
// for each local cell, its neighbor cells may live on other ranks.
std::vector<index> pullGlobal = computeNeighborGlobalIndices();
trans.createGhostMapping(pullGlobal);
// Internally this sorts and deduplicates the vector, determines
// which ranks own which rows, and builds send/recv tables.

// Step 4: Build MPI derived datatypes and resize son.
// This allocates son to hold exactly the number of ghost rows,
// and creates MPI_Type_create_hindexed types that describe the
// non-contiguous memory layout for sends and receives.
trans.createMPITypes();
```

After setup, pulling ghost data is two lines:

```cpp
// One-time initialization of persistent MPI requests.
// Persistent requests avoid re-posting Send/Recv each iteration.
trans.initPersistentPull();

// ... in the time-stepping loop:
trans.startPersistentPull();   // non-blocking start
trans.waitPersistentPull();    // block until complete
// Now son[0..sonSize-1] contains fresh copies from other ranks.
```

## ArrayPair: the Typical Pattern

In practice, most code uses `ArrayPair` which bundles
father + son + transformer.  The most common pattern is:

1. One "primary" pair (e.g. `cell2cell`) is set up with the full
   four-step process above.
2. All other pairs on the same partition (e.g. `coords`, `u`, `uRec`)
   **borrow** the primary's ghost mapping and only rebuild the MPI types
   (because the element size differs).

(Source: `DNDS/ArrayPair.hpp:269` for `BorrowAndPull`.)

```cpp
// Primary pair: cell-to-cell adjacency (built during mesh construction).
// Its ghost mapping is already set up.
ArrayAdjacencyPair<NonUniformSize> cell2cell;

// Secondary pair: solution DOF (5 reals per cell).
ArrayPair<ParArray<real, 5>> u;
u.InitPair("u", mpi);
u.father->Resize(nLocal);

// BorrowAndPull = TransAttach + BorrowGGIndexing + createMPITypes + pullOnce
u.BorrowAndPull(cell2cell);

// Now u has ghost data.  Unified access:
for (index i = 0; i < u.Size(); i++)  // iterates father + son
    u(i, 0) = 1.0;  // density = 1.0 everywhere
```

## ArrayDof: Solver State Arrays

`ArrayDof` is what you use for your PDE unknowns.  It inherits
everything from `ArrayEigenMatrixPair` (father + son + transformer +
per-row `Eigen::Map<Matrix>` access) and adds MPI-collective vector-space
operations.
(Source: `DNDS/ArrayDOF.hpp:134`.)

```cpp
#include "CFV/VRDefines.hpp"
using namespace DNDS;

// tUDof<5> = ArrayDof<5, 1> -- per-cell 5-component state vector.
// Built by the VariationalReconstruction object:
CFV::tUDof<5> u;
vr->BuildUDof(u, 1);    // allocates father+son, sets up ghost mapping

// Fill initial condition.  u[iCell] returns Eigen::Map<Vector<real,5>>.
for (index iCell = 0; iCell < u.father->Size(); iCell++)
    u[iCell] << 1.0, 0.0, 0.0, 0.0, 2.5;   // quiescent air

// Pull ghost data so stencil computations see neighbor values.
u.trans.startPersistentPull();
u.trans.waitPersistentPull();

// MPI-collective operations (all ranks participate):
real residual = u.norm2();       // global L2 norm
u.addTo(du, 1.0);               // u += 1.0 * du  (AXPY)
u *= 0.5;                        // scale all entries
```

If you are writing a new solver, the typical startup is:

1. Build the mesh (see @ref geom_usage).
2. Construct a `FiniteVolume` and `VariationalReconstruction`.
3. Call `BuildUDof`, `BuildURec`, `BuildUGrad` to get your state arrays.
4. Initialize, then loop: evaluate RHS, update, pull ghosts, repeat.

See the Euler solver at `Euler/EulerSolver.hxx` for the full pattern.

## Python Array Interface

The Python bindings (built with pybind11) mirror the C++ API.  The naming
convention encodes the element type, row size, row max, and alignment:
`<Base>_<type>_<rs>_<rm>_<align>` where `d`=double, `q`=int64,
`D`=DynamicSize, `I`=NonUniformSize, `N`=NoAlign.

| Python class | C++ equivalent |
|---|---|
| `DNDS.ParArray_d_5_5_N` | `ParArray<real, 5>` |
| `DNDS.ParArray_q_I_I_N` | `ParArray<index, NonUniformSize, NonUniformSize>` |
| `DNDS.ParArrayPair_d_5_5_N` | `ArrayPair<ParArray<real, 5>>` |

```python
from DNDSR import DNDS
import numpy as np

mpi = DNDS.MPIInfo()
mpi.setWorld()

# Create a distributed array of 10 rows x 5 columns.
arr = DNDS.ParArray_d_5_5_N(mpi)
arr.Resize(10)

# Element access via __getitem__ / __setitem__:
arr[0, 0] = 3.14
val = arr[0, 0]

# Zero-copy numpy view of the raw buffer:
buf = np.array(arr.data(), copy=False)   # shape: (50,) = 10*5 flat
buf[0:5] = [1, 2, 3, 4, 5]              # writes directly to arr

# ArrayPair bundles father + son + transformer:
pair = DNDS.ParArrayPair_d_5_5_N()
pair.InitPair("myPair", mpi)
pair.father.Resize(10)
# ... BorrowAndPull from a primary pair ...
print(pair.Size())   # father + son
```

For DOF arrays, the CFV module provides `tUDof_D` (dynamic-size) and
fixed-size variants.  See `test/CFV/test_fv_correctness.py` for a
complete working example.
