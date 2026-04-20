# AGENTS.md — DNDSR

DNDSR is a C++17 / Python CFD (Computational Fluid Dynamics) research code implementing
Compact Finite Volume methods with MPI parallelism and optional CUDA GPU support.

## Project Structure

- `src/` — C++ and Python source, organized by module:
  - `DNDS/` — Core: MPI arrays, serialization (JSON, HDF5), profiling, CUDA
  - `Geom/` — Unstructured mesh, CGNS I/O, partitioning (Metis/ParMetis)
  - `CFV/` — Compact Finite Volume, variational reconstruction
  - `Euler/` — Compressible N-S solvers (2D/3D, SA, k-omega RANS)
  - `EulerP/` — Alternative evaluator with CUDA GPU support
- `app/` — C++ application entry points (solver executables)
- `test/` — Python tests (pytest + pytest-mpi)
- `cases/` — JSON configuration files for solver runs
- `external/` — Git submodule (`cfd_externals`) and header-only libraries

## Build Commands

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
   cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j8
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

Tests use **pytest** with **pytest-mpi**. Test files live under `test/`.

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
MPI tests are registered at np=1, np=2, np=4, and np=8.

```bash
# Configure with tests enabled (from build directory)
cmake .. -DDNDS_BUILD_TESTS=ON

# Build all C++ unit tests
cmake --build . -t dnds_unit_tests -j8

# Run all C++ tests via CTest
ctest --test-dir . -R dnds_ --output-on-failure

# Run a specific test (e.g., only np=2 MPI tests)
ctest --test-dir . -R dnds_mpi_np2 --output-on-failure

# Run a single test executable directly
./test/cpp/dnds_test_array
mpirun -np 4 ./test/cpp/dnds_test_mpi
```

Available test targets: `dnds_test_array`, `dnds_test_mpi`, `dnds_test_array_transformer`,
`dnds_test_array_derived`, `dnds_test_array_dof`, `dnds_test_index_mapping`,
`dnds_test_serializer`.

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
from DNDSR.Geom.utils import create_mesh_from_CGNS
from DNDSR import DNDS

mpi = DNDS.MPIInfo()
mpi.setWorld()

# Read mesh with elevation and bisection
mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
    meshElevation="O2",         # Elevate O1→O2
    meshDirectBisect=1,         # Bisect once
)
```

**Key Features:**
- CGNS mesh reading (`ReadFromCGNSSerial`)
- Order elevation: Quad4→Quad9, Hex8→Hex27, etc. (`BuildO2FromO1Elevation`)
- Mesh bisection for h-refinement (`BuildBisectO1FormO2`)
- Boundary mesh extraction (`create_bnd_mesh`)
- VTK output generation
- Wall distance computation (`BuildNodeWallDist`)
- CUDA device offloading (`to_device` / `to_host`)
- Three read modes: Serial, Parallel, Distributed

## Formatting and Linting

### C++ — clang-format (config in `src/.clang-format`)

- **Style:** LLVM-based with Allman braces
- **Indent:** 4 spaces, no tabs
- **Column limit:** none (unlimited line length)
- **Includes:** not sorted by clang-format (`SortIncludes: Never`)
- **Namespace indentation:** all content indented
- **Access modifiers:** outdented 4 spaces from body

```bash
clang-format -i src/DNDS/SomeFile.hpp    # format a file
```

### C++ — clang-tidy (config in `src/.clang-tidy`)

Enabled check groups: `modernize-*`, `readability-*`, `bugprone-*`, `performance-*`,
`cppcoreguidelines-*`, `google-build-using-namespace`, `mpi-*`, `openmp-*`.

## Code Style — C++

### Naming Conventions

| Element               | Convention       | Example                              |
|-----------------------|------------------|--------------------------------------|
| Namespace             | `PascalCase`     | `DNDS`, `DNDS::Geom`, `DNDS::CFV`   |
| Class / Struct        | `PascalCase`     | `MPIInfo`, `VariationalReconstruction` |
| Public method         | `PascalCase`     | `Resize()`, `ConstructMetrics()`     |
| Private/protected member | `_` prefix    | `_size`, `_data`, `_pRowStart`       |
| Type alias            | `t_` prefix or `using` | `t_IndexVec`, `tDiFj`          |
| Template parameter    | `T`-prefix or PascalCase | `T`, `TOut`, `_row_size`      |
| Constant              | `PascalCase`     | `UnInitReal`, `DynamicSize`          |
| Macro                 | `DNDS_ALL_CAPS`  | `DNDS_INDEX_MAX`, `DNDS_MPI_REAL`    |
| Enum value            | `PascalCase`     | `UnknownElem`, `Line2`, `Roe`        |

### Core Type Aliases (from `Defines.hpp`)

```cpp
using real = double;
using index = int64_t;
using rowsize = int32_t;
template <typename T> using ssp = std::shared_ptr<T>;
```

### Header Guards

Always use `#pragma once` — no `#ifndef` guards.

### Include Order

1. Project macros (`"DNDS/Macros.hpp"`, `"DNDS/Defines.hpp"`)
2. Standard library headers
3. External library headers (Eigen, fmt, nlohmann_json, etc.)
4. Project headers (quoted, relative to `src/`): `"DNDS/Array.hpp"`, `"Geom/Mesh.hpp"`

Includes are NOT auto-sorted. Preserve existing order.

### Error Handling

Use the project's assert/check macros from `DNDS/Errors.hpp`:

```cpp
DNDS_assert(expr);                         // debug-only, calls std::abort()
DNDS_assert_info(expr, info_string);       // debug-only with message
DNDS_assert_infof(expr, fmt_string, ...);  // debug-only with printf format
DNDS_check_throw(expr);                    // always active, throws std::runtime_error
DNDS_check_throw_info(expr, info_string);  // always active with message
```

Do NOT use raw `assert()`. Use `DNDS_assert` for debug checks and
`DNDS_check_throw` for runtime validation that must remain in release builds.

### Templates and Eigen

- Heavy use of Eigen matrices; prefer `Eigen::Matrix<real, ...>` with project's `real` type
- Use `if constexpr` for compile-time branching on template parameters
- Explicit template instantiation goes in `_explicit_instantiation/` subdirectories

### Brace Style

Allman style — opening brace on its own line:

```cpp
if (condition)
{
    // ...
}
```

## Code Style — Python

### Naming

- Functions/variables: `snake_case` (`test_all_reduce_scalar`, `meshFile`)
- Classes wrapping C++ types: match C++ name (`MPIInfo`, `VariationalReconstruction_2`)
- Private helpers: `_` prefix (`_pre_import`, `_init_mpi`)

### Imports

```python
from __future__ import annotations      # when used
import sys, os
from DNDSR import DNDS, Geom, CFV, EulerP
import numpy as np
import pytest
```

Test files append `src/` to `sys.path` to find the `DNDSR` package:

```python
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "src"))
```

### Testing Patterns

- Use `@pytest.fixture` for MPI setup
- Use plain `assert` statements (not unittest-style)
- Tests may also run standalone: `if __name__ == "__main__":` block calling test functions
- Use numpy for array comparisons: `assert np.all(...)`, `assert (arr == val).all()`

## Key Dependencies

- **Compiler:** GCC 9+ / Clang 8+, C++17 required
- **MPI:** MPI-3 compatible (OpenMPI or MPICH)
- **CMake:** >= 3.21
- **C++ libs:** Eigen, Boost, CGAL, nlohmann_json, fmt, pybind11, HDF5, CGNS, Metis, ParMetis
- **Python:** >= 3.10, numpy, scipy, pytest, pytest-mpi, mpi4py, h5py
- **Optional:** CUDA toolkit, SuperLU_dist
