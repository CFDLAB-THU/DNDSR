# Example Playground {#example_playground}

@tableofcontents

The `examples/` directory contains compilable C++ programs that exercise
the code patterns described in the documentation guides.  Drop a new
`.cpp` file into `examples/` and it is automatically picked up by CMake
-- no manual registration needed.

## Building Examples

### Via CMake (recommended)

```sh
# Build all examples at once:
cmake --build build -t examples -j32

# Build a single example:
cmake --build build -t ex_array_basic -j32

# Run (most examples need MPI):
mpirun -np 2 build/examples/ex_array_basic
mpirun -np 1 build/examples/ex_geom_mesh
mpirun -np 1 build/examples/ex_cfv_solver
```

Examples are `EXCLUDE_FROM_ALL`, so they are not built by a plain
`cmake --build build`.  You must name them explicitly or use the
`examples` aggregate target.

### Adding a New Example

Create a file in `examples/` matching one of these naming patterns:

| Filename prefix | Linked libraries | Use when |
|---|---|---|
| `ex_array_*` | `dnds` | Array, ParArray, ArrayTransformer |
| `ex_geom_*` | `geom` + `dnds` | Mesh, elements, quadrature |
| `ex_cfv_*` | `cfv` + `geom` + `dnds` | FiniteVolume, VR, ModelEvaluator |
| `ex_euler_*` | `euler_library_NS` + `cfv` + `geom` + `dnds` | Euler evaluator, gas dynamics |

The `examples/CMakeLists.txt` glob discovers `ex_*.cpp` files
automatically via `CONFIGURE_DEPENDS`.

### Equivalent g++ Commands

If you want to compile an example outside of CMake (e.g. for a quick
test on a cluster), here are the equivalent commands.  Adjust paths to
match your installation.

**Paths** (set these for your environment):

```sh
DNDSR_SRC=/path/to/DNDSR/src
DNDSR_BUILD=/path/to/DNDSR/build
DNDSR_EXT=/path/to/DNDSR/external
CFD_EXT=$DNDSR_EXT/cfd_externals/install
MPI_DIR=$(dirname $(dirname $(which mpicxx)))
```

**ex_array_basic** (links only `dnds`):

```sh
mpicxx -std=c++17 -O2 \
    -I$DNDSR_SRC \
    -I$DNDSR_EXT/eigen \
    -I$DNDSR_EXT/nlohmann \
    -I$DNDSR_EXT/boost \
    -I$DNDSR_EXT/fmt/include \
    -I$DNDSR_EXT/nanoflann \
    -I$DNDSR_EXT/cppcodec \
    -I$DNDSR_EXT/argparse/include/argparse \
    -I$DNDSR_EXT/CGAL/include \
    -I$CFD_EXT/include \
    -DNINSERT -D__DNDS_REALLY_COMPILING__ \
    -fopenmp \
    examples/ex_array_basic.cpp \
    -L$DNDSR_BUILD/src/DNDS -ldnds \
    -L$DNDSR_BUILD/fmt -lfmt \
    -L$CFD_EXT/lib -lcgns -lhdf5 -lparmetis -lmetis -lz \
    -lstdc++fs -ldl \
    -o ex_array_basic
```

**ex_geom_mesh** (links `geom` + `dnds`):

```sh
mpicxx -std=c++17 -O2 \
    -I$DNDSR_SRC \
    -I$DNDSR_EXT/eigen -I$DNDSR_EXT/nlohmann -I$DNDSR_EXT/boost \
    -I$DNDSR_EXT/fmt/include -I$DNDSR_EXT/nanoflann \
    -I$DNDSR_EXT/cppcodec -I$DNDSR_EXT/argparse/include/argparse \
    -I$DNDSR_EXT/CGAL/include -I$CFD_EXT/include \
    -DNINSERT -D__DNDS_REALLY_COMPILING__ -fopenmp \
    examples/ex_geom_mesh.cpp \
    -L$DNDSR_BUILD/src/Geom -lgeom \
    -L$DNDSR_BUILD/src/DNDS -ldnds \
    -L$DNDSR_BUILD/fmt -lfmt \
    -L$CFD_EXT/lib -lcgns -lhdf5 -lparmetis -lmetis -lz \
    -lstdc++fs -ldl \
    -o ex_geom_mesh
```

**ex_cfv_solver** (links `cfv` + `geom` + `dnds`):

```sh
mpicxx -std=c++17 -O2 \
    -I$DNDSR_SRC \
    -I$DNDSR_EXT/eigen -I$DNDSR_EXT/nlohmann -I$DNDSR_EXT/boost \
    -I$DNDSR_EXT/fmt/include -I$DNDSR_EXT/nanoflann \
    -I$DNDSR_EXT/cppcodec -I$DNDSR_EXT/argparse/include/argparse \
    -I$DNDSR_EXT/CGAL/include -I$CFD_EXT/include \
    -DNINSERT -D__DNDS_REALLY_COMPILING__ -fopenmp \
    examples/ex_cfv_solver.cpp \
    -L$DNDSR_BUILD/src/CFV -lcfv \
    -L$DNDSR_BUILD/src/Geom -lgeom \
    -L$DNDSR_BUILD/src/DNDS -ldnds \
    -L$DNDSR_BUILD/fmt -lfmt \
    -L$CFD_EXT/lib -lcgns -lhdf5 -lparmetis -lmetis -lz \
    -lstdc++fs -ldl \
    -o ex_cfv_solver
```

**Notes:**
- Replace `mpicxx` with your MPI C++ wrapper (e.g. `mpicxx`, `mpiCC`).
- Add `-I/usr/local/cuda-XX/include` and `-lcudart` if your build has
  CUDA enabled.
- The static libraries (`libdnds.a`, `libgeom.a`, `libcfv.a`) must be
  built first with CMake.  The g++ commands link against the *already
  built* libraries in the build tree.

## Available Examples

### ex_array_basic

**Guide:** @ref array_usage

Demonstrates the core array infrastructure:
- Fixed-row `Array` (compile-time width)
- Dynamic-row `Array` (runtime width)
- CSR `Array` with compress/decompress
- `ParArray` ghost communication with `ArrayTransformer`
- `ArrayPair` with `BorrowAndPull`

```sh
mpirun -np 2 build/examples/ex_array_basic
```

### ex_geom_mesh

**Guide:** @ref geom_usage

Demonstrates mesh building and geometric queries:
- CGNS file reading and mesh construction pipeline
- Element type queries (`GetCellElement`)
- Node coordinate access
- Face traversal and boundary detection
- Quadrature integration to compute cell volumes

```sh
mpirun -np 1 build/examples/ex_geom_mesh
```

### ex_cfv_solver

**Guides:** @ref geom_usage (Parts 5--9)

Demonstrates the full CFV solver pipeline:
- Mesh building from a periodic CGNS file
- `VariationalReconstruction` settings and construction
- `ConstructMetrics`, `ConstructBaseAndWeight`, `ConstructRecCoeff`
- DOF array allocation (`BuildUDof`, `BuildURec`)
- Initial condition from cell barycenters
- Iterative VR reconstruction (5 sweeps)
- RHS evaluation via `ModelEvaluator`

```sh
mpirun -np 1 build/examples/ex_cfv_solver
```
