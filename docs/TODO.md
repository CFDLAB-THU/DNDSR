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
- Register pytest tests via `add_test()` so `ctest` runs Python tests too
- Add a `CMakePresets.json` with Debug, Release, CUDA, and CI configurations

### Install and packaging

- Stop forcing `CMAKE_INSTALL_PREFIX` (line 29); let users override it
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

- Fix scikit-build-core `cmake.args = ["-G Ninja"]` (space issue;
  should be `["-GNinja"]` or `["-G", "Ninja"]`)
- Change `build.tool-args = ["-j0"]` to a bounded value to prevent OOM
  on template-heavy builds
- Uncomment `"DNDSR" = "src/DNDSR"` in `[tool.scikit-build.wheel.packages]`
  so the top-level `DNDSR/__init__.py` is included in the wheel
- Add `cmake.define` entries for essential build options
  (`DNDS_FAST_BUILD_FAST`, `DNDS_USE_OMP`, etc.)
- Add `[tool.pytest.ini_options]` to `pyproject.toml` with `testpaths`,
  markers (e.g. `mpi`, `cuda`), and timeout settings

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

### Encapsulation

- Convert `UnstructuredMesh` from `struct` (all ~30+ data members public)
  to `class` with private members and public accessor methods
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

## Python Integration

### Package structure

- Include `DNDSR/__init__.py` in the wheel (uncomment the mapping in
  `pyproject.toml`)
- Add `EulerP` to the re-exports in `src/DNDSR/__init__.py`
  (currently only DNDS, Geom, CFV are listed)
- Remove debug `print(f"module load: {__file__}")` from all four
  `__init__.py` files
- Fix misleading `__all__` in each module: either list all public names
  or remove `__all__` entirely to match the `import *` behavior
- Add `py.typed` marker for PEP 561 typed package compliance
- Add `__init__.py` to all `_internal/` directories for consistency
- Fix missing `import os` in `src/DNDS/Debug_Py.py`

### Type stubs

- Fix the stub concatenation script (`run-pybind11-stubgen.sh`) to avoid
  duplicate `from __future__ import annotations` and duplicate `__all__`
  in the generated package-level `.pyi` files
- Strip hardcoded local paths from stubs (e.g. `DNDSR_bin_dir` leaks
  a machine-specific path into `Geom/__init__.pyi`)
- Remove stale `placeholder_submodule.pyi` files if the placeholders
  are no longer needed

### API design

- Make `EulerP_Solver.Solver.BuildEval` configurable: physics parameters
  (gamma, mu, Pr, etc.) are currently hardcoded; accept them as arguments
  or from a config dict
- Make `EulerP_Solver.Solver.BuildDataArray` generic: currently hardcoded
  for 5-component Euler (`tUDof_5`); should support SA (6) and
  k-omega (7) component counts
- Make boundary conditions configurable through the Solver interface
  (currently `bcInputs = []` is empty/hardcoded)
- Reduce MPI auto-initialization surprise: consider lazy initialization
  or requiring explicit `DNDS.init()` instead of calling `MPI_Init_thread`
  at import time

### Build workflow

- Enable ccache for pip/scikit-build-core builds (currently explicitly
  disabled at CMakeLists.txt line 179)
- Document the editable-install rebuild workflow:
  `pip install -e .` once, then
  `cmake --build build_py -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j N && cmake --install build_py`
- Add a top-level Makefile or `justfile` with shortcuts for common
  build/test/format workflows

### Documentation

- Add Python API documentation (Sphinx or MkDocs) generated from
  the `.pyi` stubs and docstrings
- Add usage examples for the main workflows: mesh loading, FV setup,
  solver execution, CUDA device transfer
