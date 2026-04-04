# Project Structure {#project_structure}

## Directory Layout

```
DNDSR/
├── src/                        C++ source code
│   ├── DNDS/                   Core: MPI arrays, serialization, profiling, CUDA
│   ├── Geom/                   Unstructured mesh, CGNS I/O, partitioning
│   ├── CFV/                    Compact Finite Volume, variational reconstruction
│   ├── Euler/                  Compressible N-S solvers (2D/3D, SA, k-omega)
│   ├── EulerP/                 Alternative evaluator with CUDA GPU support
│   └── Solver/                 ODE integrators, Krylov solvers (GMRES, PCG)
│
├── python/DNDSR/               Python package (pip-installable)
│   ├── __init__.py             Top-level: imports DNDS, Geom, CFV, EulerP
│   ├── _loader.py              Shared native library preloader (ctypes.CDLL)
│   ├── py.typed                PEP 561 typed package marker
│   ├── DNDS/                   Python bindings + factory functions for core
│   │   ├── __init__.py         Preload, import bindings, MPI init, factories
│   │   ├── _ext/               Compiled pybind11 .so (cmake --install target)
│   │   ├── Debug_Py.py         Pure-Python debug helpers
│   │   └── Wrapper.py          Python introspection utilities
│   ├── Geom/                   Mesh bindings
│   │   ├── __init__.py
│   │   ├── _ext/
│   │   └── utils.py            Mesh creation utilities
│   ├── CFV/                    FV bindings
│   │   ├── __init__.py
│   │   └── _ext/
│   └── EulerP/                 Euler evaluator bindings
│       ├── __init__.py
│       ├── _ext/
│       └── EulerP_Solver.py    High-level solver wrapper
│
├── app/                        C++ application entry points
│   ├── Euler/                  Solver executables (euler, eulerSA, euler2EQ, ...)
│   ├── DNDS/                   Old standalone test apps
│   ├── Geom/                   Mesh tool apps
│   └── CFV/                    FV test apps
│
├── test/                       Tests
│   ├── cpp/DNDS/               C++ unit tests (doctest, registered with CTest)
│   ├── DNDS/                   Python tests (pytest)
│   ├── Geom/                   Python Geom tests
│   ├── CFV/                    Python CFV tests
│   └── EulerP/                 Python Euler tests
│
├── external/                   Third-party dependencies
│   ├── cfd_externals/          C libraries submodule (HDF5, CGNS, Metis, ...)
│   ├── doctest/                C++ test framework (header-only)
│   ├── eigen/                  Eigen linear algebra (header-only)
│   ├── fmt/                    fmt formatting library
│   ├── pybind11/               Python binding framework
│   ├── nlohmann/               JSON for Modern C++ (header-only)
│   └── ...                     boost, CGAL, nanoflann, exprtk, etc.
│
├── docs/                       Documentation sources
│   ├── Doxyfile                Doxygen configuration (cmake-configured)
│   ├── main.dox                Doxygen main page
│   └── *.md                    Documentation pages (rendered by Doxygen)
│
├── cases/                      JSON configuration files for solver runs
├── scripts/                    Utility scripts
│   └── generate-stubs.sh       Type stub generator (pybind11-stubgen)
├── stubs/                      Generated .pyi stubs (intermediate output)
│
├── CMakeLists.txt              Root CMake build file
├── CMakePresets.json            CMake presets (debug, release-test, cuda, ci)
├── cmakeCommonUtils.cmake       Shared CMake helper functions
├── pyproject.toml              Python package build configuration
└── AGENTS.md                   Agentic coding guide
```

## C++ Source Organization (`src/`)

Each module under `src/` contains:

- `*.hpp` / `*.cpp` — C++ headers and implementation files.
- `*_pybind.cpp` / `*_bind.cpp` — Pybind11 binding definitions.
- `_explicit_instantiation/` — Explicit template instantiation files
  (one per model variant, compiled as separate translation units).
- `CMakeLists.txt` — Module-level build rules.

### Module dependency graph

Each module depends on those above it:

```
DNDS        (no dependencies within DNDSR)
  ↑
Geom        (depends on DNDS)
  ↑
CFV         (depends on Geom, DNDS)
  ↑
Euler       (depends on CFV, Geom, DNDS, Solver)
EulerP      (depends on CFV, Geom, DNDS)
Solver      (header-only, depends on DNDS)
```

## Python Package Organization (`python/DNDSR/`)

The Python package is separate from C++ sources.  Compiled pybind11
extension modules (`.so` / `.pyd`) are installed into `_ext/`
subdirectories by `cmake --install`.

### Import chain

```python
from DNDSR import DNDS
  # → python/DNDSR/__init__.py
  #     → from . import DNDS
  #       → python/DNDSR/DNDS/__init__.py
  #         → _loader.preload("dnds")           # ctypes.CDLL for native libs
  #         → from ._ext.dnds_pybind11 import *  # load C++ bindings
  #         → _init_mpi()                        # MPI_Init_thread at import time
```

### Native Library Preloader (`_loader.py`)

Before pybind11 extensions can be imported, their shared library
dependencies must be loaded.  `_loader.py` provides a single
`preload(module)` function that:

1. Locates the library directory (`python/DNDSR/_lib/` or legacy paths).
2. Loads external dependencies (libstdc++, zlib, HDF5, CGNS, Metis,
   ParMetis) via `ctypes.CDLL` with `RTLD_GLOBAL`.
3. Loads module-specific shared libraries (libdnds_shared, libgeom_shared,
   etc.).

## Test Organization

| Directory        | Framework | Runner            | What it tests          |
|------------------|-----------|-------------------|------------------------|
| `test/cpp/DNDS/` | doctest   | CTest (np=1,2,4)  | DNDS core C++ classes  |
| `test/DNDS/`     | pytest    | pytest             | DNDS Python bindings   |
| `test/Geom/`     | pytest    | pytest             | Geom Python bindings   |
| `test/CFV/`      | pytest    | pytest             | CFV Python bindings    |
| `test/EulerP/`   | pytest    | pytest             | EulerP Python bindings |

See @ref dnds_unit_tests for the full C++ test suite documentation.

## CMake Build Targets

| Target              | Description                                 |
|---------------------|---------------------------------------------|
| `euler`             | Euler N-S solver (2D)                       |
| `euler3D`           | Euler N-S solver (3D)                       |
| `eulerSA`           | Spalart-Allmaras RANS solver (2D)           |
| `eulerSA3D`         | Spalart-Allmaras RANS solver (3D)           |
| `euler2EQ`          | k-omega RANS solver (2D)                    |
| `euler2EQ3D`        | k-omega RANS solver (3D)                    |
| `dnds_pybind11`     | DNDS Python binding module                  |
| `geom_pybind11`     | Geom Python binding module                  |
| `cfv_pybind11`      | CFV Python binding module                   |
| `eulerP_pybind11`   | EulerP Python binding module                |
| `dnds_unit_tests`   | All C++ unit test executables (aggregate)   |
