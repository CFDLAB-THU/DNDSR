# AGENTS.md — DNDSR

DNDSR is a C++17 / Python CFD (Computational Fluid Dynamics) research code implementing
Compact Finite Volume methods with MPI parallelism and optional CUDA GPU support.

## Project Structure

- `src/` — C++ and Python source, organized by module:
  - `DNDS/` — Core: MPI arrays, serialization (JSON, HDF5), profiling, CUDA
    - `Config/` — Runtime configuration enums, parameters, registry
    - `Device/` — Host-device memory transfer, CUDA utilities
    - `Serializer/` — JSON and HDF5 serialization framework
    - `ArrayDerived/` — Specialized array types (adjacency, Eigen matrices)
  - `Geom/` — Unstructured mesh, CGNS I/O, partitioning (Metis/ParMetis)
    - `Mesh/` — Mesh data structures, connectivity, ghost management, state tracking
    - `Elements/` — Per-element-type shape functions
    - `Quadratures/` — Numerical integration rules
  - `CFV/` — Compact Finite Volume, variational reconstruction
  - `Euler/` — Compressible N-S solvers (2D/3D, SA, k-omega RANS)
  - `EulerP/` — Alternative evaluator with CUDA GPU support
- `app/` — C++ application entry points (solver executables)
- `test/` — Python tests (pytest + pytest-mpi + pytest-timeout)
- `cases/` — JSON configuration files for solver runs
- `external/` — Git submodule (`cfd_externals`) and header-only libraries

## Build Commands

> **For humans:** The canonical build guide with full explanations,
> troubleshooting, and platform notes lives in `docs/guides/building.md`.
> This section is a condensed agent reference.

### C++ (CMake)

```bash
# Configure (from project root)
mkdir build && cd build
CC=mpicc CXX=mpicxx cmake ..
# ^ Use CC=mpicc CXX=mpicxx when unsure which MPI CMake will find.

# Or use CMake presets (see CMakePresets.json)
cmake --preset release-test   # Release with tests enabled
cmake --preset debug          # Debug with tests enabled
cmake --preset cuda           # Release with CUDA and tests

# Build a specific target (-j for parallel)
cmake --build . -t euler -j 8
# Available targets: euler, euler3D, eulerSA, eulerSA3D, euler2EQ, euler2EQ3D
# Python modules: dnds_pybind11, geom_pybind11, cfv_pybind11, eulerP_pybind11
```

### Python Package (scikit-build-core)

```bash
# Full install
CC=/usr/bin/gcc CXX=/usr/bin/g++ CMAKE_BUILD_PARALLEL_LEVEL=16 pip install . --verbose

# Editable install
CC=/usr/bin/gcc CXX=/usr/bin/g++ CMAKE_BUILD_PARALLEL_LEVEL=16 pip install -e . --verbose

# Rebuild C++ pybind11 targets only (from build_py/)
cmake --build . -t dnds_pybind11 geom_pybind11 cfv_pybind11 -j32 && cmake --install .
```

### Using the DNDSR Python Module (from build/)

To use the pybind11-based Python module from a CMake build directory:

1. **Build the pybind11 targets:**

   ```bash
   cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
   ```

2. **Install the Python component** (copies `.so` files into `python/`):

   ```bash
   cmake --install build --component py
   ```

3. **Run with the project venv and `PYTHONPATH`:**

   ```bash
   source venv/bin/activate
   PYTHONPATH=<project_root>/python python my_script.py
   ```

The `.so` files are built against the venv's Python (3.12). Always use
`venv/bin/python`, not the system Python.

### External Dependencies

```bash
git submodule update --init --recursive --depth=1
cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
```

## Test Commands

### Python Tests

**IMPORTANT: Before running ANY Python test, you MUST build the pybind11
shared libraries AND install them.** The Python modules load `.so` files
from `python/DNDSR/`, which are only placed there by `cmake --install`.
Running Python tests against stale or missing `.so` files will produce
misleading crashes (segfaults, aborts, wrong results) that look like code
bugs but are actually stale-binary problems. **Every time C++ source
changes, repeat both steps before running Python tests:**

```bash
# Step 1: Build pybind11 targets
cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32

# Step 2: Install into python/ (MANDATORY — do not skip)
cmake --install build --component py

# Step 3: Now run Python tests
source venv/bin/activate
PYTHONPATH=<project_root>/python pytest test/
```

If you switch git branches or checkout different commits, you MUST rebuild
and reinstall before running Python tests. A `git checkout` changes source
files but does NOT rebuild binaries — the installed `.so` files will be
from the previous build and will silently produce wrong behavior.

**CRITICAL: Before running ANY `git checkout`, `git switch`, `git restore`,
or `git reset` command, ALWAYS run `git status` first.** Verify the working
tree is clean or that all valuable changes are committed/stashed. These
commands silently overwrite uncommitted modifications and delete untracked
files, discarding work with no way to recover it.

Tests use **pytest** with **pytest-mpi** and **pytest-timeout**. Test files live under `test/`. A default 120-second timeout is configured in `pyproject.toml` to prevent hung MPI tests from blocking CI.

```bash
# Run all tests
pytest test/

# Run a single test file
pytest test/DNDS/test_basic.py

# Run a single test function
pytest test/DNDS/test_basic.py::test_all_reduce_scalar

# Run with MPI (multiple ranks)
mpirun -np 4 python -m pytest test/DNDS/test_basic.py

# Run a test file as a script (some tests support this)
python test/DNDS/test_basic.py
mpirun -np 2 python test/DNDS/test_basic.py
```

### C++ Unit Tests (doctest)

C++ tests live under `test/cpp/` and use the [doctest](https://github.com/doctest/doctest)
framework. They are built when `DNDS_BUILD_TESTS=ON` and registered with CTest.
MPI tests are registered at np=1, np=2, np=4, and np=8 by default (configurable via
`DNDS_TEST_NP_LIST` environment variable at configure time). All tests run with
`OMP_NUM_THREADS=2` by default (configurable via `DNDS_TEST_OMP_THREADS` environment
variable at configure time).

```bash
# Configure with tests enabled (from build directory)
cmake .. -DDNDS_BUILD_TESTS=ON

# Configure with custom OMP threads (optional)
DNDS_TEST_OMP_THREADS=4 cmake .. -DDNDS_BUILD_TESTS=ON

# Build all C++ unit tests (all categories)
cmake --build . -t all_unit_tests -j8

# Build only specific category
cmake --build . -t dnds_unit_tests -j8   # DNDS/ tests only
cmake --build . -t geom_unit_tests -j8   # Geom/ tests only
cmake --build . -t cfv_unit_tests -j8    # CFV/ tests only
cmake --build . -t euler_unit_tests -j8  # Euler/ tests only
cmake --build . -t solver_unit_tests -j8 # Solver/ tests only

# Run all C++ tests via CTest
ctest --test-dir . --output-on-failure

# Run with aggregated doctest summary (shows total test cases + assertions)
python scripts/ctest_summary.py --output-on-failure
python scripts/ctest_summary.py -R "^dnds_"   # filter by category

# Run tests by category prefix
ctest --test-dir . -R "^dnds_" --output-on-failure   # DNDS tests
ctest --test-dir . -R "^geom_" --output-on-failure   # Geom tests
ctest --test-dir . -R "^cfv_" --output-on-failure    # CFV tests
ctest --test-dir . -R "^euler_" --output-on-failure  # Euler tests
ctest --test-dir . -R "^solver_" --output-on-failure # Solver tests

# Run only np=2 MPI tests across all categories
ctest --test-dir . -R "_np2$" --output-on-failure

# Run a single test executable directly
./test/cpp/dnds_test_array
mpirun -np 4 ./test/cpp/dnds_test_mpi
```

**Test categories and targets:**

- **DNDS:** `dnds_test_array`, `dnds_test_mpi`, `dnds_test_array_transformer`,
  `dnds_test_array_derived`, `dnds_test_array_dof`, `dnds_test_index_mapping`,
  `dnds_test_serializer`, `dnds_test_permutation_transfer`
- **Geom:** `geom_test_elements`, `geom_test_quadrature`, `geom_test_mesh_index_conversion`,
  `geom_test_mesh_pipeline`, `geom_test_mesh_distributed_read`, `geom_test_mesh_connectivity`,
  `geom_test_mesh_connectivity_ghost`, `geom_test_mesh_connectivity_interpolate`,
  `geom_test_mesh_reorder`
- **CFV:** `cfv_test_reconstruction`, `cfv_test_limiters`, `cfv_test_reconstruction3d`,
  `cfv_test_device_transferable` (CUDA only)
- **Euler:** `euler_test_gas_thermo`, `euler_test_riemann_solvers`, `euler_test_rans`,
  `euler_test_evaluator_pipeline`
- **Solver:** `solver_test_ode`, `solver_test_linear`, `solver_test_direct`, `solver_test_scalar`

**Note:** When writing new C++ tests with `using namespace DNDS;`, always qualify
`DNDS::index`, `DNDS::real`, and `DNDS::rowsize` in declarations to avoid ambiguity
with POSIX `index()` from `<strings.h>` (pulled in by doctest).

### Python Geom Module

The Python `DNDSR.Geom` module provides mesh reading and manipulation capabilities.
See the comprehensive guide at `docs/guides/python_geom_guide.md` for full API
details, including all parameters, read modes, and notes on which C++ methods
are (and are not) exposed in the Python bindings.

**Quick Example:**

```python
from DNDSR.Geom.utils import read_mesh, prepare_mesh
from DNDSR import DNDS

mpi = DNDS.MPIInfo()
mpi.setWorld()

# Read mesh with elevation and bisection
result = read_mesh(
    "data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
    elevation="O2",         # Elevate O1→O2
    bisect=1,               # Bisect once
)
prepare_mesh(result.mesh, result.reader)
```

The legacy `create_mesh_from_CGNS` wrapper is still available for backward
compatibility.

**Key Features:**

- CGNS mesh reading (`ReadFromCGNSSerial`)
- H5 distributed reading with ParMetis repartition
- Order elevation: Quad4→Quad9, Hex8→Hex27, etc. (`BuildO2FromO1Elevation`)
- Mesh bisection for h-refinement (`BuildBisectO1FormO2`)
- Boundary mesh extraction (`build_bnd_mesh`)
- Multi-layer ghost cells via `BuildGhostPrimary(nGhostLayers)`
- VTK output generation
- Wall distance computation (`BuildNodeWallDist`)
- CUDA device offloading (`to_device` / `to_host`)

## Code Style

Full style guide (naming, formatting, includes, error handling, Doxygen,
Python conventions): **`docs/guides/style_guide.md`**

Quick reference for C++:

- **Braces:** Allman (opening brace on its own line)
- **Naming:** `PascalCase` classes/methods, `_` prefix for private members,
  `DNDS_ALL_CAPS` macros, `t_` prefix type aliases
- **Headers:** `#pragma once`, preserve include order (no auto-sort)
- **Errors:** `DNDS_assert` (debug) / `DNDS_check_throw` (release) from
  `DNDS/Errors.hpp`; never raw `assert()`
- **Core types:** `real = double`, `index = int64_t`, `rowsize = int32_t`,
  `ssp<T> = std::shared_ptr<T>`

Quick reference for Python:

- `snake_case` functions/variables; C++ wrapper classes match C++ name
- Plain `assert`; `@pytest.fixture` for MPI; numpy for array comparisons

### Clang-tidy sanitation

DNDS is clean as of 2026-04-29 (26-pass cleanup, 24 597 → 1
diagnostics; the remaining one is an unrelated Eigen PCH
`omp.h` include issue). Full per-pass record, `.clang-tidy`
disable rationale, and NOLINT placement gotchas:
**`docs/dev/clang_tidy_plan.md`**.

Other modules (`Solver`, `Geom`, `CFV`, `Euler`, `EulerP`) are
not yet sanitised. Apply the same recipe in that order. Run
`scripts/run_clang_tidy.py <module>` to get the per-check
histogram; the `.clang-tidy` disables carry forward unchanged.

## Geom Module Architecture

Mesh connectivity, ghost management, and the build pipeline are documented
in **`docs/architecture/MeshConnectivity.md`**.

Key concepts agents should know:

- **Adjacency state:** Each adjacency array (e.g. `cell2node`, `face2cell`)
  is wrapped in `AdjPairTracked<TPair>` (inherits from `TPair`, adds an
  `AdjIndexInfo idx` member). All `idx` fields are private; state
  transitions go through `markGlobal()`, `markLocal()`,
  `wireTargetMapping()`, `toLocal()`/`toGlobal()`, and
  `bootstrapToLocal()`. See `src/Geom/Mesh/AdjIndexInfo.hpp`.

- **Three-layer architecture:**
  1. `MeshConnectivity` (DSL) -- bare `ArrayAdjacencyPair<rs>`, no state
  2. `Mesh/MeshConnectivity_StateChecked.hpp` -- asserts `idx.state()`, forwards to DSL
  3. `UnstructuredMesh` (Mesh/Mesh.cpp) -- owns `AdjPairTracked` members, calls checked wrappers

- **Conversion methods:** `AdjGlobal2Local*` / `AdjLocal2Global*` delegate
  to `adj.toLocal()` / `adj.toGlobal()`, which use the stored target
  mapping. The five group state variables (`adjPrimaryState`, etc.)
  still exist and are updated in parallel with per-adj states.

- **State discipline:** Every site that sets a group state variable
  (`adjXState = ...`) must also call the corresponding `idx` method
  (`markGlobal()`, `markLocal()`) on the governed adjacencies. Every
  site that builds ghost mappings must wire them to the relevant
  adjacencies via `wireTargetMapping()`. Both DSL and legacy code
  paths maintain this invariant.

## Key Dependencies

- **Compiler:** GCC 9+ / Clang 8+, C++17 required
- **MPI:** MPI-3 compatible (OpenMPI or MPICH)
- **CMake:** >= 3.21
- **C++ libs:** Eigen, Boost, CGAL, nlohmann_json, fmt, pybind11, HDF5, CGNS, Metis, ParMetis
- **Python:** >= 3.10, numpy, scipy, pytest, pytest-mpi, mpi4py, h5py
- **Optional:** CUDA toolkit, SuperLU_dist

## GitHub CLI (`gh`) Policy

**Read-only by default.** You may use `gh` freely for read operations (viewing
issues, PRs, checks, releases, diffs, comments). **Do NOT use `gh` for any
write operation** (creating/closing issues, creating/merging PRs, posting
comments, approving reviews, creating releases, editing labels, deleting
caches, etc.) **unless the user explicitly requests that specific write
action.** One-time explicit permission does not carry over to other write
actions — ask each time.

**Operations requiring explicit user authorization (non-exhaustive):**
- `gh pr create/merge/close/edit`
- `gh issue create/close/edit`
- `gh pr comment` / `gh issue comment`
- `gh pr review`
- `gh release create/delete`
- `gh cache delete`
- `gh api` with non-GET methods (POST, PUT, PATCH, DELETE)
- `git push --force` / `git push --force-with-lease`

**Draft PR by default** You must use --draft on new prs.
