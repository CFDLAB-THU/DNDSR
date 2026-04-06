# TODO

## Miscellaneous

- DPW: see separation size
- Outer tangential functional (1 3 3 1 -> 1 9 9 1)

- 2EQ: other equations ?

## Primary Features

- Rotational source and velocity
- Rotational bc
- Multi Zone Support
- Forcing Source
- LES
- DES

## Supplementary Features

- CGNS writer for plt data
- H5 (parallel) serializer
- CGNS parallel read/write
- parallel mesh partitioning
- true serial mesh distribution

## Build System

### CMake modernization

- ~~Uncomment `CMAKE_CXX_STANDARD_REQUIRED`~~ (done)
- ~~Fix `BUILD_SHARED_LIBS` corruption: restore value after fmt `add_subdirectory`~~ (done)
- Replace global `add_compile_options` / `add_definitions` (50+ calls) with per-target
  `target_compile_options`, `target_compile_definitions`, `target_compile_features`.
  Current global flags leak into third-party subdirectories (fmt, SuperLU, pybind11)
- ~~Remove redundant `cmake_minimum_required(VERSION 3.1)` from subdirectory
  CMakeLists files (resets CMake policies, the root already requires 3.21)~~ (done)
- ~~Remove deprecated `CMAKE_USE_RELATIVE_PATHS`~~ (done)
- ~~Fix syntax error on line 47 (mismatched quotes in Clang compiler check)~~ (done)
- ~~Remove the single `include(cmakeCommonUtils.cmake)` per subdirectory;
  one `include()` in the root suffices since function definitions persist~~ (done)
- ~~Remove redundant `-std=c++17` flag; `CMAKE_CXX_STANDARD 17` already handles it~~ (done)
- ~~Remove duplicate `-Wall` warning block~~ (done)
- ~~Create `dnds_external_deps` INTERFACE library target to replace raw
  `DNDS_EXTERNAL_LIBS`/`DNDS_EXTERNAL_INCLUDES` variable lists~~ (done)
- ~~Refactor `cmakeCommonUtils.cmake`: remove dead CMake version checks~~ (done)
  (`LLVM` variable in `add_fast_flags` is kept: used for LLVM Clang on Windows)
- ~~Remove unused shared Euler library variants (`euler_library_*_shared`)~~ (done)
- ~~Use `CONFIGURE_DEPENDS` with `file(GLOB)` calls~~ (done)
- Replace raw `find_library`/`find_path` with `find_package` for external deps
  (ZLIB, HDF5, CGNS, Metis, ParMetis, Eigen, Boost, CGAL, nlohmann_json).
  Create imported targets with proper transitive properties

### Build time reduction

- Stop building every library twice (static + shared) for DNDS/Geom/CFV/EulerP.
  Consider building only shared libraries and linking executables against those
- Consolidate explicit-instantiation translation units: group multiple model
  instantiations per .cpp file to cut the 58 Euler + 24 CFV tiny compilation
  units (each pulls in massive template headers)
- Enable PCH by default (`DNDS_USE_PRECOMPILED_HEADER ON`) now that
  infrastructure exists

### CTest and test infrastructure

- ~~Add `enable_testing()` and register C++ unit tests with CTest~~ (done)
- ~~Add `DNDS_BUILD_TESTS` cache option~~ (done)
- ~~Register pytest tests via `add_test()` so `ctest` runs Python tests too~~ (done)
- ~~Add a `CMakePresets.json` with Debug, Release, CUDA, and CI configurations~~ (done)

### Install and packaging

- ~~Stop forcing `CMAKE_INSTALL_PREFIX`; let users override it~~ (done)
- Stop installing pybind11 modules into the source tree
  (`CMAKE_CURRENT_SOURCE_DIR/_internal`); use a proper staging area
- Remove configure-time `file(INSTALL ...)` calls; use `install()` rules only
- Generate CMake config/export files so downstream projects can
  `find_package(DNDSR)` if needed

### Euler build simplification

- Consolidate the 9 identical solver entry points (`app/Euler/euler*.cpp`)
  into a single executable with a runtime model-selection argument or a
  single `main.cpp.in` with `configure_file`
- Refactor the explicit-instantiation macros in `EulerEvaluator.hpp`
  (lines 1440-1671, ~230 lines of manual method signature listings) into
  a less fragile mechanism

### Python package build

- ~~Fix scikit-build-core `cmake.args` (split `"-G Ninja"` into `["-G", "Ninja"]`)~~ (done)
- ~~Change `build.tool-args = ["-j0"]` to bounded `["-j8"]` to prevent OOM~~ (done)
- ~~Uncomment `"DNDSR" = "src/DNDSR"` in `[tool.scikit-build.wheel.packages]`~~ (done)
- ~~Add `[tool.pytest.ini_options]` with testpaths, markers, timeout~~ (done)
- Add `cmake.define` entries for essential build options

## Refactoring

### EulerEvaluator decomposition (6,855 lines, ~162 methods)

- Extract flux computation (inviscid, viscous, RANS turbulence) into a
  dedicated class or set of free functions
- Extract boundary condition handling (`EulerEvaluator.hpp` lines 889-1045)
  into a separate BC evaluator class
- Extract Jacobian operations (lines 647-799) into their own unit
- Extract positive-preserving limiters (`CompressRecPart`, `CompressInc`,
  `AddFixedIncrement`, lines 1047-1393) into a limiter class
- Move large inline implementations (50-130 lines each) out of the `.hpp`
  header and into the `.hxx` implementation files
- Remove ~50 lines of commented-out dead code (lines 1342-1393)

### EulerSolver decomposition (4,936 lines)

- Extract the 12 nested `Configuration` structs (lines 114-610) into a
  separate `EulerSolverConfig.hpp` with its own JSON serialization
- Separate I/O (init, restart, print) from solver loop logic

### RANS model branching duplication

- The patterns `if constexpr (model == NS_SA || model == NS_SA_3D)` and
  `if constexpr (model == NS_2EQ || model == NS_2EQ_3D)` each appear 14 times
  across Euler .hpp/.hxx files. Extract model-specific behavior into traits
  or policy classes to eliminate duplication

### Element and Quadrature organization

- Split `Elements.hpp` (2,642 lines) into:
  `ElementTypes.hpp` (enums + topology tables),
  `ElementGeometry.hpp` (shape functions + Jacobians),
  `ElementIO.hpp` (CGNS/VTK conversion utilities)
- Split `Quadrature.hpp` (1,103 lines) into:
  `QuadratureData.hpp` (raw point/weight tables, 636 lines of static data),
  `Quadrature.hpp` (class and integration logic)

### UnstructuredMesh decomposition (~13,370 lines across 13 files)

`UnstructuredMesh` (`Mesh.hpp:25-806`) is a god struct: ~50 data members,
~60+ methods, and 9 `.cpp` implementation files spanning topology building,
ghost communication, face interpolation, I/O, elevation, reordering, and
output.  `UnstructuredMeshSerialRW` (`Mesh.hpp:821-1058`) has 16 declared-
but-unused adjacency fields (lines 862-877) and mixes reading, partitioning,
and serial output in a single struct.

#### Phase 1: Eliminate duplication (low risk, high reward) — done

- ~~**Template the 12 index conversion methods** (`Mesh.hpp`):
  Replaced 12 identical methods (168 lines) with 4 generic templates
  (`IndexGlobal2Local<TPair>`, `IndexLocal2Global<TPair>`,
  `IndexLocal2Global_NoSon<TPair>`, `IndexGlobal2Local_NoSon<TPair>`)
  plus 12 one-line named wrappers.  Zero call-site changes required~~ (done)
- ~~**Extract `ConvertAdjEntries` helper** (`Mesh.hpp` + `Mesh.cpp`):
  Added `ConvertAdjEntries<TAdj, TFn>(adj, nRows, fn)` template.  Applied
  to 8 of 12 adjacency methods (Primary, PrimaryForBnd, C2CFace).  N2CB
  methods preserved verbatim (they use `#pragma omp parallel for`).
  Facial and C2F methods preserved (inline ghost mapping lookups)~~ (done)
- ~~**Extract `PermuteRows` helper** (`Mesh.hpp` + `Mesh.cpp`):
  Added `PermuteRows<TPair, TFn>(pair, nRows, old2new)` template that
  handles both CSR (`Decompress`/`ResizeRow`/`Compress`) and fixed-size
  arrays via `if constexpr (TPair::IsCSR())`.  Replaced 6 permutation
  blocks (~70 lines) in `ReorderLocalCells()` with 6 one-line calls~~ (done)
- **Dead fields** in `UnstructuredMeshSerialRW` (lines 862-877):
  16 adjacency arrays declared "not used for now" — preserved intentionally;
  may be used in future serial I/O paths

#### Phase 2: Break up massive methods

- **Split `InterpolateFace()`** (~416 lines, `Mesh.cpp:1520-1936`) into:
  `CreateLocalFaces()` (face enumeration from cells),
  `DeduplicateGhostFaces()` (MPI rank comparison),
  `MapBoundariesToFaces()` (bnd-to-face matching),
  `CommunicateGhostFaces()` (MPI ghost exchange)
- **Split `ReorderLocalCells()`** (~370 lines, `Mesh.cpp:2318-2687`) into:
  partition computation, permutation application (using the new
  `PermuteArrayRows` helper), and ghost re-indexing
- **Split `PrintSerialPartVTKDataArray()`** (~700 lines,
  `Mesh_Plts.cpp:535-1235`) into per-format writers
- **Split `ReadFromCGNSSerial()`** (~550 lines,
  `Mesh_Serial_ReadFromCGNS.cpp:13-551`) into zone reading, zone assembly,
  and interface-node deduplication

#### Phase 3: Decompose into focused components

- **`MeshTopology`**: adjacency arrays, state flags, state transitions
  (the 5 `MeshAdjState` fields and their assertions)
- **`MeshIO`**: serialization (`WriteSerialize`/`ReadSerialize`),
  VTK/PLT/CGNS output (currently in `Mesh_Plts.cpp`), CGNS/OpenFOAM
  reading (currently in `Mesh_Serial_ReadFromCGNS.cpp`)
- **`MeshElevation`**: O1-to-O2 elevation, bisection, smooth solvers
  (currently in `Mesh_Elevation.cpp` + `Mesh_Elevation_SmoothSolver.cpp`).
  Factor shared setup (KD-tree, boundary gathering) out of the 4 smooth
  solver variants; remove `V1Old` if deprecated
- **`MeshReordering`**: local cell reordering, partition start tracking
- Existing `UnstructuredMesh` becomes a thin facade holding a
  `MeshTopology` plus coordinate data, with component access methods

#### Phase 4: Fix coupling and cleanup

- **Break the Geom-to-Solver dependency**: `Mesh.hpp:11` includes
  `Solver/Direct.hpp` solely for `SerialSymLUStructure` and
  `DirectPrecControl` used in two methods (`ObtainLocalFactFillOrdering`,
  `ObtainSymmetricSymbolicFactorization` at lines 439-440).  Move these
  methods to a bridge file (e.g., `Mesh_DirectPrec.cpp`) or forward-declare
  the solver types
- **Consolidate `__GetCoords` overloads** (`Mesh.hpp:500-611`): 4 template
  overloads x 2 (periodic/non-periodic) can become a single template with
  optional coordinate source and optional PBI source
- **Replace `auto mesh = this;`** pattern (`Mesh.cpp:641, 1417, 2046`)
  with direct member access
- **Delete dead/commented code**: commented `InterpolateTopology`
  (`Mesh.cpp:26-42`), dead code after `continue`
  (`Mesh_Serial_BuildCell2Cell.cpp:416-438`), debug print blocks
- **Standardize logging**: replace raw `std::cout`/`std::cerr` calls
  (found in `Mesh_Serial_ReadFromCGNS.cpp`, `Mesh.cpp`) with `DNDS::log()`
- **Fix OpenMP progress reporting**: `Mesh_Serial_BuildCell2Cell.cpp:386-404`
  uses `#pragma omp critical` for progress output inside a parallel loop.
  Use thread-local counters or atomic operations instead

#### Phase 5: Reduce periodic branching

Throughout the mesh code, `if (isPeriodic)` appears in every major method
(25+ sites across `Mesh.cpp` and `Mesh.hpp`).  Consider extracting
periodic-specific behavior into a policy or strategy pattern so that
non-periodic mesh paths remain clean.

### Encapsulation

- Convert `UnstructuredMesh` from `struct` (all ~30+ data members public)
  to `class` with private members and public accessor methods
  (can be done incrementally as part of UnstructuredMesh Phase 3 above)
- Reduce `EulerEvaluator` public surface: make internal buffers
  (`lambdaFace*`, `uGradBuf*`, `fluxWallSum`, `symLU`, etc.) private;
  expose only through accessors where needed
- Audit the 9 access-specifier toggles in `EulerEvaluator.hpp` and
  establish clear public API vs. internal state boundaries

### Error handling consistency

- Add `DNDS_check_throw` for user-facing validation in Euler and CFV
  modules (currently 286 combined `DNDS_assert` calls but zero
  `DNDS_check_throw`; no runtime validation in release builds)
- Replace 6 raw `assert()` calls in `Solver/ODE.hpp`
  (lines 400, 429, 501, 643, 666, 1034) with `DNDS_assert`
- Replace 149 raw `std::cout` calls in Euler with the project's
  `log()` function or a proper logging mechanism

### Miscellaneous cleanup

- Remove duplicate `#include` directives: `DNDS/JsonUtil.hpp` is included
  twice in `EulerEvaluator.hpp`, `EulerSolver.hpp`, and
  `EulerEvaluatorSettings.hpp`; `Solver/Linear.hpp` is included twice in
  `EulerSolver.hpp`
- Remove test source (`test_FiniteVolume.cpp`) from the CFV production
  library (line 27 of `src/CFV/CMakeLists.txt`)
- Move `test/Geom/OversetCart/` (7-file library package) out of `test/`
  into `src/` since it is library code, not tests
- Delete `src/Geom/GeomUtils.py` (outdated duplicate of `utils.py` with bugs)
- Delete `src/check.py` (stale integration test, superseded by `test/`)

## Unit Test

### C++ test framework (doctest) — done

Doctest v2.4.11 is integrated under `external/doctest/`. Tests live in `test/cpp/`.
Built with `cmake -DDNDS_BUILD_TESTS=ON`, run with `ctest -R dnds_`.
MPI tests registered at np=1, np=2, np=4.

### Completed DNDS core C++ unit tests (`test/cpp/DNDS/`)

- `test_Array.cpp` — all 5 layouts (StaticFixed, Fixed, StaticMax, Max, CSR),
  resize, compress/decompress, clone, copy, swap, edge cases (zero-size,
  single-element), JSON serialization round-trip, array signature, hash
- `test_MPI.cpp` — MPIInfo, Allreduce (SUM/MAX, real/index),
  AllreduceOneReal/OneIndex, Scan, Allgather, Bcast, Barrier, Alltoall,
  BasicType_To_MPIIntType, CommStrategy
- `test_ArrayTransformer.cpp` — ParArray basics, pull-based ghost comm
  (StaticFixed, Fixed, CSR, std::array elements), persistent pull,
  BorrowGGIndexing, push communication
- `test_ArrayDerived.cpp` — ArrayAdjacency (basics, ghost comm, clone,
  fixed-size), ArrayEigenVector (basics, static/dynamic, ghost comm),
  ArrayEigenMatrix (static, dynamic, NonUniform rows, ghost comm),
  ArrayEigenMatrixBatch (basics, ghost comm), ArrayEigenUniMatrixBatch
  (static, dynamic)
- `test_ArrayDOF.cpp` — setConstant (scalar, matrix), operator+= (scalar,
  array, matrix), operator-=, operator*= (scalar, array element-wise,
  matrix), operator/=, addTo, norm2, norm2 (difference), dot, min, max,
  sum, componentWiseNorm1, operator= (copy), clone, scalar-array multiply
- `test_IndexMapping.cpp` — GlobalOffsetsMapping (uniform, non-uniform,
  search), OffsetAscendIndexMapping (pull-based, search_indexAppend,
  empty ghost set)
- `test_Serializer.cpp` — SerializerJSON scalar round-trip, vector
  round-trip, uint8 (with/without codec), path operations, shared pointer
  deduplication. ~~SerializerH5 tests present but skipped pending API fixes~~
  SerializerH5 scalar, vector, distributed vector, uint8 (two-pass),
  path operations, string round-trips (done)

### Test infrastructure improvements remaining

- Create a shared `test/conftest.py` with an MPI fixture, replacing the
  per-file `DNDS.MPIInfo()` boilerplate duplicated across all test files
- Add `[tool.pytest.ini_options]` to `pyproject.toml`:
  `testpaths = ["test"]`, custom markers (`mpi`, `cuda`, `slow`),
  and a default timeout
- Remove the `sys.path.append` hack in `test/DNDS/test_basic.py`
  (unnecessary with a proper editable install)
- Fix `test/Geom/test_basic_geom.py`: remove the `while True: pass`
  infinite loop (line 85) that makes the test hang

### Convert existing print-only tests to assertion-based tests

- `test/CFV/test_basic_cfv.py`: add assertions on reconstruction
  coefficient values, metric correctness, RHS conservation
- `test/CFV/test_basic_fv.py`: add assertions on cell/face metric norms,
  gradient operator accuracy
- `test/EulerP/test_basic_eulerP.py`: add assertions on primitive variable
  conversion, eigenvalue positivity, flux conservation
- `test/EulerP/test_solver.py`: add assertions on conservation property
  preservation, time integration convergence
- `test/DNDS/test_arraydof.py`: convert to pytest-compatible structure,
  add CUDA skip marker

### Missing C++ test coverage — DNDS module

- ~~SerializerH5 round-trip tests (currently skipped; need MPI-collective
  open/close fixes)~~ (done)
- DeviceStorage host/CUDA transfer correctness
- ExprtkWrapper expression evaluation
- ObjectPool allocation/reclaim

### Missing test coverage — Geom module

- Element shape function delta property and standard element volumes
  (port the logic from `app/Geom/elements_Test.cpp` to doctest)
- Quadrature accuracy: integrate known polynomials, verify exactness
  for the claimed order
- Mesh CGNS read: load a small mesh, verify cell/face/node counts,
  connectivity, and coordinate values
- Mesh partitioning: partition a mesh across N ranks, verify all cells
  are assigned and ghost layers are consistent
- Periodic geometry handling: verify periodic face matching
- Boundary mesh construction correctness

### Missing test coverage — CFV module

- Variational reconstruction polynomial exactness: set DOF values from
  a known polynomial, reconstruct, verify face values match exactly up
  to the scheme's order
- FiniteVolume metric conservation: sum of face area vectors per cell
  should be zero (divergence theorem)
- Limiter behavior: verify monotonicity preservation on discontinuous data
- Gradient operator accuracy: compare computed gradients against
  analytical gradients of known functions

### Missing test coverage — Solver module

- Add pybind11 bindings for `Solver/Linear.hpp` (GMRES, PCG) and
  `Solver/ODE.hpp` (ImplicitEuler, SDIRK4, BDF2) to enable Python testing
- GMRES convergence test: solve a known linear system, verify residual
  reduction
- PCG convergence test: solve SPD system, verify convergence
- ODE integrator accuracy: integrate a simple ODE (e.g. exponential decay),
  verify error order matches the scheme's theoretical order

### Missing test coverage — Euler / EulerP module

- Gas model unit tests: verify Roe/HLLC/HLLEP flux on known Riemann
  problems against analytical or reference solutions
- Boundary condition tests: verify wall BC produces zero normal velocity,
  inlet/outlet produce correct state
- Conservation test: verify global conservation of mass/momentum/energy
  over several time steps on a periodic domain
- Convergence order verification: run on a sequence of refined meshes,
  verify spatial convergence rate matches scheme order

## Python Integration — Package Restructure Plan

### Problem summary

The current layout mixes C++ headers, pybind11 bindings, compiled `.so` files,
Python `__init__.py`, and `.pyi` stubs all inside `src/DNDS/`, `src/Geom/`, etc.
This causes:

- `cmake --install` places `.so` files into the **source tree** (`src/*/_internal/`)
- Two conflicting `__init__.py` at `src/__init__.py` and `src/DNDSR/__init__.py`
- Four duplicated `_pre_import()` functions for CDLL preloading
- Stubs generated by concatenation (appending submodule `.pyi` into package `.pyi`)
  producing duplicate `from __future__` imports and stale artifacts
- Tests need `sys.path.append` hacks to find the package

### Target layout

```
python/                            # NEW: all Python code lives here
└── DNDSR/                         # the pip-installable package root
    ├── __init__.py                # imports DNDS, Geom, CFV, EulerP
    ├── py.typed                   # PEP 561 typed package marker
    ├── _loader.py                 # single shared CDLL preload logic
    ├── DNDS/
    │   ├── __init__.py            # calls _loader, imports from _ext
    │   ├── _ext/                  # cmake installs .so here
    │   │   └── (dnds_pybind11.cpython-*.so)
    │   └── Debug_Py.py ...        # pure-Python helpers
    ├── Geom/
    │   ├── __init__.py
    │   ├── _ext/
    │   └── utils.py
    ├── CFV/
    │   ├── __init__.py
    │   └── _ext/
    └── EulerP/
        ├── __init__.py
        ├── _ext/
        └── EulerP_Solver.py
stubs/                             # NEW: generated .pyi stubs
└── DNDSR/
    ├── DNDS/__init__.pyi, MPI.pyi, ...
    ├── Geom/__init__.pyi, Elem.pyi
    ├── CFV/__init__.pyi
    └── EulerP/__init__.pyi
```

C++ sources (`*.hpp`, `*.cpp`, `*_pybind.cpp`) remain in `src/` unchanged.

### Implementation steps (incremental, each step buildable/testable)

#### Step 1: Create `python/DNDSR/` tree with `_loader.py`

- Create `python/DNDSR/__init__.py` (imports DNDS, Geom, CFV, EulerP)
- Create `python/DNDSR/_loader.py` (consolidated CDLL preload: one function
  `preload_libs(module_name)` that loads the correct shared libs based on
  the calling module, using a lib search path relative to the install prefix)
- Create `python/DNDSR/DNDS/__init__.py` (calls `_loader`, then
  `from ._ext.dnds_pybind11 import *`, factory functions, MPI init)
- Create `python/DNDSR/Geom/__init__.py`, `CFV/__init__.py`,
  `EulerP/__init__.py` (same pattern)
- Copy pure-Python files: `Debug_Py.py`, `utils.py`, `EulerP_Solver.py`
- Create `python/DNDSR/py.typed` marker
- Create empty `_ext/__init__.py` in each submodule

#### Step 2: Update CMake install rules

- Change pybind11 install destinations from `${CMAKE_CURRENT_SOURCE_DIR}/_internal`
  to `${PROJECT_SOURCE_DIR}/python/DNDSR/<Module>/_ext`
- Change shared library install from `DNDSR/lib` to
  `${PROJECT_SOURCE_DIR}/python/DNDSR/_lib` (or keep in install prefix)
- Update `run-pybind11-stubgen.sh` to output to `stubs/` instead of
  in-tree concatenation

#### Step 3: Update `pyproject.toml`

- Change `[tool.scikit-build.wheel.packages]` to `"DNDSR" = "python/DNDSR"`
- Remove all the individual `DNDSR/DNDS = src/DNDS` mappings
- scikit-build-core installs `.so` into the wheel automatically since
  the cmake install targets land inside `python/DNDSR/`

#### Step 4: Update test imports

- Remove `sys.path.append` hacks from test files
- Tests use `from DNDSR import DNDS` (works with `pip install -e .`)
- Update CI/Makefile to `pip install -e .` before running tests

#### Step 5: Fix stub generation

- Rewrite `run-pybind11-stubgen.sh` to output clean per-file stubs
  into `stubs/DNDSR/` (no concatenation, no append)
- Copy stubs into `python/DNDSR/` for PEP 561 compliance, or configure
  `pyproject.toml` to include `stubs/` in the wheel
- Remove stale `placeholder_submodule.pyi` files
- Strip machine-specific paths from generated stubs

#### Step 6: Clean up old `src/` Python files

- Remove `src/__init__.py`, `src/DNDSR/__init__.py`
- Remove `src/DNDS/__init__.py`, `src/Geom/__init__.py`, etc.
- Remove `src/*/_internal/` directories (`.so` files, `.pyi` files)
- Remove `src/DNDS/__init__.pyi` and other in-tree stubs
- Keep `src/run-pybind11-stubgen.sh` (updated) or move to `scripts/`

### Build mode behavior after restructure

| Mode | What happens |
|------|-------------|
| **Pure C++ build** | `cmake --build . -t euler` — only compiles C++ under `src/`, no Python artifacts |
| **Editable install** | `pip install -e .` — builds pybind11 targets, installs `.so` into `python/DNDSR/*/_ext/`, pip links `python/DNDSR/` |
| **Editable C++ rebuild** | `cmake --build build_py -t dnds_pybind11 ... && cmake --install build_py` — `.so` lands in `python/`, Python picks it up immediately |
| **Package build** | `pip install .` — wheel contains `DNDSR/` with `.so` files, pure Python, and stubs |

### Items carried forward (not part of restructure)

- API design: configurable physics params, lazy MPI init
- Documentation: Sphinx/MkDocs from stubs
- Enable ccache for pip builds
- Add Makefile/justfile for common workflows
