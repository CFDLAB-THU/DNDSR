# Building DNDSR {#building}

## Prerequisites

| Requirement     | Version       | Notes                                  |
|-----------------|---------------|----------------------------------------|
| C++ compiler    | GCC 9+ / Clang 8+ | Must support C++17                |
| MPI             | MPI-3         | OpenMPI or MPICH                       |
| CMake           | >= 3.21       |                                        |
| Python          | >= 3.10       | System Python recommended (not conda)  |
| Ninja           | any           | Optional but recommended for speed     |

C++ libraries (managed via `external/cfd_externals` submodule):
Eigen, Boost, CGAL, nlohmann_json, fmt, pybind11, HDF5, CGNS,
Metis, ParMetis, ZLIB.  Optional: CUDA toolkit, SuperLU_dist.

## Building External Dependencies

DNDSR requires two sets of external dependencies: binary libraries built
from the `cfd_externals` submodule, and header-only libraries shipped as a
tarball.

### Binary libraries (HDF5, CGNS, Metis, ParMetis, ZLIB)

```bash
git submodule update --init --recursive --depth=1
cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
cd ../..
```

This installs all binary libraries into `external/cfd_externals/install/`.

### Header-only libraries (Eigen, Boost, CGAL, fmt, pybind11, ...)

Download the latest release tarball from GitHub and extract it into
the `external/` directory:

```bash
curl -L -o external/external_headeronlys.tar.gz \
  https://github.com/harryzhou2000/cfd_externals_headeronlys/releases/latest/download/external_headeronlys.tar.gz
cd external
tar -xzf external_headeronlys.tar.gz
cd ..
```

After extraction, directories such as `external/eigen/`,
`external/boost/`, `external/CGAL/`, etc. should exist.

## CMake Module Architecture

The build system is split into focused modules under `cmake/`.  The
main `CMakeLists.txt` (~100 lines) orchestrates them in dependency
order:

| #  | Module                    | Purpose                                                  |
|----|---------------------------|----------------------------------------------------------|
| 1  | `DndsStdlibSetup.cmake`  | Detect and bundle libstdc++ or libc++ for the Python package |
| 2  | `DndsOptions.cmake`      | All user-facing cache options, commit recording, ccache  |
| 3  | `DndsCudaSetup.cmake`    | CUDA language enable, toolkit discovery, CCCL include path |
| 4  | `DndsCompilerFlags.cmake`| LTO, MPI discovery, platform flags, OpenMP               |
| 5  | `DndsExternalDeps.cmake` | find_library/find_path, pybind11/fmt/superlu subdirs, `dnds_external_deps` target |
| 6  | `cmakeCommonUtils.cmake`  | Helper functions: `add_fast_flags`, `dnds_add_lib`, `dnds_add_py_module` |
| 7  | `DndsTests.cmake`        | CTest registration for doctest C++ tests and pytest      |
| 8  | `DndsApps.cmake`         | Application executables and `ADD_EXE_APP` function       |
| 9  | `DndsDocs.cmake`         | Doxygen documentation target                             |
| 10 | `DndsTooling.cmake`      | compile_commands.json post-processing, automatic stub generation |

Between modules 6 and 7, the five core library subdirectories are
added: `src/DNDS`, `src/Geom`, `src/CFV`, `src/Euler`, `src/EulerP`.

## Building C++ (Solvers and Libraries)

### Using CMake Presets

```bash
cmake --preset release-test        # Release with unit tests enabled
cmake --build --preset tests -j32  # Build all C++ unit tests
ctest --preset unit                # Run unit tests
```

Available presets (defined in `CMakePresets.json`):

| Preset          | Build Type | Tests | Build Dir       | Notes                         |
|-----------------|------------|-------|-----------------|-------------------------------|
| `default`       | Release    | OFF   | `build/`        | Minimal solver build          |
| `debug`         | Debug      | ON    | `build-debug/`  | Full debug symbols            |
| `release-test`  | Release    | ON    | `build/`        | Main development preset       |
| `cuda`          | Release    | ON    | `build-cuda/`   | Enables CUDA GPU support      |
| `ci`            | Release    | ON    | `build-ci/`     | CI (no ccache)                |

### Manual CMake Configuration

```bash
cmake -S . -B build -G Ninja -DDNDS_BUILD_TESTS=ON
cmake --build build -t euler -j32           # Build a solver
cmake --build build -t dnds_unit_tests -j32 # Build C++ tests
ctest --test-dir build -R dnds_ --output-on-failure
```

Let CMake detect the system default compiler; do not set `CC`/`CXX`
unless you have a specific reason.

### Solver Targets

Each Euler model variant generates a separate executable:

| Target        | Model      | Dimension |
|---------------|------------|-----------|
| `euler`       | Navier-Stokes | 2D/3D auto |
| `euler2D`     | Navier-Stokes | 2D only    |
| `euler3D`     | Navier-Stokes | 3D only    |
| `eulerSA`     | Spalart-Allmaras | 2D/3D |
| `eulerSA3D`   | Spalart-Allmaras | 3D only |
| `euler2EQ`    | k-omega 2-equation | 2D/3D |
| `euler2EQ3D`  | k-omega 2-equation | 3D only |
| `eulerEX`     | Extended   | 2D/3D     |
| `eulerEX3D`   | Extended   | 3D only   |

### CMake Cache Options

Key options (set via `-D<OPTION>=<VALUE>` or in a preset):

| Option                       | Default | Description                            |
|------------------------------|---------|----------------------------------------|
| `DNDS_BUILD_TESTS`          | OFF     | Build C++ unit tests (doctest)         |
| `DNDS_USE_CUDA`             | OFF     | Enable CUDA GPU support                |
| `DNDS_USE_OMP`              | ON      | Enable OpenMP                          |
| `DNDS_FAST_BUILD_FAST`      | ON      | Use -O3 -g0 on core library modules   |
| `DNDS_LTO`                  | OFF     | Link-time optimization                 |
| `DNDS_LTO_THIN`             | OFF     | Use -flto=thin (Clang)                 |
| `DNDS_PYBIND11_NO_LTO`     | OFF     | Disable LTO for pybind11 modules only  |
| `DNDS_NATIVE_ARCH`          | OFF     | Use -march=native                      |
| `DNDS_UNSAFE_MATH_OPT`     | OFF     | Use -funsafe-math-optimizations        |
| `DNDS_USE_CCACHE`           | auto    | Use ccache (auto-detected, off for pip)|
| `DNDS_USE_RDYNAMIC`        | ON      | Use -rdynamic on POSIX                 |
| `DNDS_GENERATE_COMPILE_COMMANDS` | OFF | Generate compile_commands.json for clangd |
| `DNDS_USE_CLANG_TIDY`      | OFF     | Run clang-tidy during build            |
| `DNDS_RECORD_COMMIT`       | ON      | Record git commit hash at configure    |
| `DNDS_USE_PRECOMPILED_HEADER` | OFF  | Use precompiled headers                |
| `DNDS_EIGEN_USE_BLAS`      | OFF     | Use external BLAS in Eigen             |
| `DNDS_EIGEN_USE_LAPACK`    | OFF     | Use external LAPACK in Eigen           |

## Building the Python Package

### Creating a Virtual Environment

Use the system Python (not conda) to avoid libstdc++ version conflicts:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install numpy scipy pytest pytest-mpi pytest-timeout mpi4py \
            pybind11 pybind11-stubgen scikit-build-core ninja
```

> **Why not conda?** Conda Python binaries embed an `RPATH` pointing to
> conda's bundled libstdc++, which may be too old for the compiler used
> to build DNDSR.  Using the system Python avoids this entirely.

### Three Ways to Build the Python Package

#### 1. In-place build (no pip install)

Build the pybind11 `.so` modules via the main CMake build, then
install them (and auto-generate type stubs) into `python/DNDSR/`:

```bash
source venv/bin/activate
cmake -S . -B build -G Ninja
cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
cmake --install build --component py
```

Use the package by setting `PYTHONPATH`:

```bash
PYTHONPATH=python pytest test/
# or:
export PYTHONPATH=$PWD/python
python -c "from DNDSR import DNDS, Geom, CFV, EulerP"
```

After making C++ changes, rebuild and reinstall:

```bash
cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
cmake --install build --component py   # reinstalls .so AND regenerates stubs
```

#### 2. Editable install (development with pip)

Uses scikit-build-core.  Builds into a separate `build_py/` directory:

```bash
source venv/bin/activate
CMAKE_BUILD_PARALLEL_LEVEL=32 pip install -e . --no-build-isolation
```

This configures and builds all four pybind11 targets, installs `.so`
files into `python/DNDSR/<Module>/_ext/`, generates `.pyi` stubs, and
registers the package as editable in the venv.

After the initial install, rebuild only the C++ parts:

```bash
cmake --build build_py -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
cmake --install build_py --component py
```

#### 3. Full wheel install

```bash
CMAKE_BUILD_PARALLEL_LEVEL=32 pip install . --verbose
```

Builds a wheel with `.so` files, bundled shared libraries, and `.pyi`
stubs included.

### Controlling Build Parallelism

The default is `-j0` (all available cores).  Override via environment:

```bash
SKBUILD_BUILD_TOOL_ARGS="-j8" pip install -e . --no-build-isolation
```

### Pybind11 Module Targets

| Target           | Module   | C++ Source               | Output .so location             |
|------------------|----------|--------------------------|---------------------------------|
| `dnds_pybind11`  | DNDS     | `src/DNDS/*_bind*.cpp`   | `python/DNDSR/DNDS/_ext/`      |
| `geom_pybind11`  | Geom     | `src/Geom/*_bind*.cpp`   | `python/DNDSR/Geom/_ext/`      |
| `cfv_pybind11`   | CFV      | `src/CFV/*_bind*.cpp`    | `python/DNDSR/CFV/_ext/`       |
| `eulerP_pybind11`| EulerP   | `src/EulerP/*_bind*.cpp` | `python/DNDSR/EulerP/_ext/`    |

Each pybind11 module links against a corresponding `*_shared` library
(`dnds_shared`, `geom_shared`, `cfv_shared`, `eulerP_shared`) which
contains the compiled C++ code.

## Type Stub Generation

Type stubs (`.pyi`) provide IDE autocompletion and type checking for
the pybind11 bindings.  Stubs are generated automatically during
install and are not tracked in git.

### How it works

`cmake/DndsTooling.cmake` registers an `install(CODE ...)` step on
the `py` component that runs `scripts/generate-stubs.sh` after all
`.so` files are installed.  This happens in both workflows:

- `cmake --install build --component py` (in-place build)
- `pip install -e .` (scikit-build-core editable install)

The script runs `pybind11-stubgen` for each submodule (DNDS, Geom,
CFV, EulerP), writes raw output to `stubs/`, and copies `.pyi` files
into `python/DNDSR/` for PEP 561 compliance.

### Manual stub regeneration

If you only changed Python code (no C++ binding changes), you can
regenerate stubs without rebuilding:

```bash
PYTHONPATH=python ./scripts/generate-stubs.sh
```

### Stubs in wheels

The `.pyi` files under `python/DNDSR/` are included in sdist/wheel
builds via `pyproject.toml` `sdist.include`.  They are generated at
build time and packaged into the wheel, so end users get type hints
without running stubgen themselves.

## CUDA Support

### Enabling CUDA

```bash
cmake -S . -B build-cuda -G Ninja -DDNDS_USE_CUDA=ON
cmake --build build-cuda -t euler -j32
```

Or use the `cuda` preset:

```bash
cmake --preset cuda
cmake --build build-cuda -j32
```

### CUDA 13.1 (CCCL 3.x) Compatibility

CUDA 13.1 moved thrust, cub, and libcudacxx headers into a `cccl/`
subdirectory under the CUDA toolkit include path.  `nvcc` adds this
path automatically, but the host C++ compiler (g++) does not.

`DndsCudaSetup.cmake` detects `${CUDAToolkit_INCLUDE_DIRS}/cccl` and
exposes it as `DNDS_CUDA_CCCL_INCLUDE_DIR`.  `DndsExternalDeps.cmake`
appends it to `DNDS_EXTERNAL_INCLUDES`, so `#include <thrust/...>`
works from both `.cu` and `.cpp` files.

This is backward compatible: on CUDA 12.x the `cccl/` path does not
exist, so nothing is added.

### CUDA-specific targets

GPU-accelerated versions of the test apps are built when
`DNDS_USE_CUDA=ON`:

- `cuda_test`, `array_cuda_Test`, `array_cuda_Bench`, `arrayDOF_test_cuda`
- `eulerP_pybind11` (Python bindings with GPU evaluator)

## Running Tests

### C++ Unit Tests

C++ tests use [doctest](https://github.com/doctest/doctest) and live
under `test/cpp/`.  They are registered with CTest at np=1, np=2, and
np=4.

```bash
cmake --build build -t dnds_unit_tests -j32
ctest --test-dir build -R dnds_ --output-on-failure

# Run a single test executable directly
./build/test/cpp/dnds_test_array
mpirun -np 4 ./build/test/cpp/dnds_test_mpi
```

Available test executables: `dnds_test_array`, `dnds_test_mpi`,
`dnds_test_array_transformer`, `dnds_test_array_derived`,
`dnds_test_array_dof`, `dnds_test_index_mapping`,
`dnds_test_serializer`.

### Python Tests

Python tests use pytest and live under `test/`.  The root
`test/conftest.py` adds `python/` to `sys.path` so tests work with
both `PYTHONPATH=python` and `pip install -e .`.

```bash
pytest test/DNDS/test_basic.py -v

# With MPI
mpirun -np 4 python -m pytest test/DNDS/test_basic.py

# All tests
pytest test/ -x --timeout=120
```

CTest also registers pytest suites when `DNDS_BUILD_TESTS=ON`:

```bash
ctest --test-dir build -R pytest_ --output-on-failure
```

## Build Mode Summary

| Mode                     | Command                                               | Build Dir   | Stubs         |
|--------------------------|-------------------------------------------------------|-------------|---------------|
| **Pure C++ build**       | `cmake --build build -t euler -j32`                   | `build/`    | N/A           |
| **C++ unit tests**       | `cmake --build build -t dnds_unit_tests -j32`         | `build/`    | N/A           |
| **In-place Python**      | `cmake --install build --component py`                | `build/`    | Auto-generated|
| **Editable install**     | `pip install -e . --no-build-isolation`                | `build_py/` | Auto-generated|
| **Editable C++ rebuild** | `cmake --build build_py -t ... && cmake --install ...`| `build_py/` | Auto-generated|
| **Full wheel**           | `pip install .`                                       | `build_py/` | Included in wheel|

## Developer Tooling

### compile_commands.json for clangd

clangd needs a `compile_commands.json` at the project root for C++
code intelligence.  CMake generates one in the build directory; the
`compdb` tool post-processes it to include header-only translation
units.

```bash
cmake -S . -B build -DDNDS_GENERATE_COMPILE_COMMANDS=ON -G Ninja
cmake --build build -j32
cmake --build build -t process-compile-commands
```

This creates `build/compile_commands_processed.json` and symlinks it
to `compile_commands.json` at the project root.

The build system uses a shipped, modified version of compdb at
`scripts/compdb/` (invoked as `PYTHONPATH=scripts python -m compdb`).
If the shipped version is not found, it falls back to a
system-installed `compdb` executable (`pip install compdb`).

### Doxygen Documentation

```bash
cmake --build build -t docs
```

Output goes to `build/docs/html/`.

### CMake Utility Targets

| Target                     | Description                                      |
|----------------------------|--------------------------------------------------|
| `process-compile-commands` | Post-process compile_commands.json for clangd    |
| `docs`                     | Build Doxygen HTML documentation                 |
| `dnds_unit_tests`          | Build all C++ unit test executables               |

### Shared Library Bundling

Both the in-place and pip install workflows copy external shared
libraries (MPI, HDF5, CGNS, Metis, ParMetis, zlib, libstdc++) into
`python/DNDSR/_lib/dndsr_external/`.  The pybind11 `.so` files have
`RPATH` set to find these bundled copies, so the package works without
`LD_LIBRARY_PATH` manipulation.

`DndsStdlibSetup.cmake` detects whether the compiler uses libstdc++
or libc++ and bundles the appropriate runtime.  RPATH uses
`--disable-new-dtags` to ensure the bundled libstdc++ takes precedence
over any `LD_LIBRARY_PATH` (important when conda is active).

### pyproject.toml Configuration

The Python package is built with
[scikit-build-core](https://scikit-build-core.readthedocs.io/).  Key
settings in `pyproject.toml`:

| Setting                | Value                   | Purpose                       |
|------------------------|-------------------------|-------------------------------|
| `build-dir`           | `build_py`              | Separate from the C++ build   |
| `build.targets`       | 4 pybind11 targets      | Only build Python bindings    |
| `cmake.args`          | `["-G", "Ninja"]`       | Use Ninja generator           |
| `build.tool-args`     | `["-j0"]`               | All cores by default          |
| `wheel.packages`      | `python/DNDSR`          | Package root                  |

When scikit-build-core configures CMake it sets `SKBUILD_PROJECT_NAME`,
which the build system uses to skip ccache and in-source symlink
creation (these are only relevant for the developer C++ build).
