# Building DNDSR {#building}

## Quick Start (Step-by-Step)

If you just want to build and run as fast as possible, follow these steps.
Detailed explanations for each step follow in the sections below.

**1. System dependencies**
```bash
sudo apt install build-essential cmake ninja-build openmpi-bin libopenmpi-dev doxygen
```

**2. Python virtual environment**
```bash
python3.12 -m venv venv
source venv/bin/activate
```

**3. Clone and get external dependencies**
```bash
git clone --recursive https://github.com/CFDLAB-THU/DNDSR.git
cd DNDSR

# Cantera build deps (must be installed before cfd_externals_build.py)
pip install -r external/cfd_externals/requirements.txt

# Header-only libs first
curl -L -o external/external_headeronlys.tar.gz \
  https://github.com/harryzhou2000/cfd_externals_headeronlys/releases/download/v0.3.0/external_headeronlys.tar.gz
echo "00692e1581f900d76c43432f4712b8dc2a53408d968fbf656bd3dd3561e2514e  external/external_headeronlys.tar.gz" | sha256sum -c -
cd external && tar -xzf external_headeronlys.tar.gz && cd ..

# Then binary libs (needs header-onlys already extracted)
cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
cd ../..

# Install remaining Python dependencies (h5py against the project's HDF5, etc.)
bash scripts/install_python_deps.sh
```

**4. Build solvers**
```bash
cmake --preset release-test
cmake --build build -t euler -j32
```

**5. Run a case**
```bash
./build/app/euler.exe cases/your_config.json
# or parallel:
mpirun -np 4 ./build/app/euler.exe cases/your_config.json
```

**6. Install Python package (optional)**
```bash
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=32 pip install -e .
```

**7. Run tests (optional)**
```bash
# C++ tests
cmake --build build -t euler_unit_tests -j32
ctest --test-dir build --output-on-failure

# Python tests
pytest test/ -v
```

---

## Prerequisites

| Requirement     | Version       | Notes                                  |
|-----------------|---------------|----------------------------------------|
| C++ compiler    | GCC 9+ / Clang 8+ | Must support C++17                |
| MPI             | MPI-3         | OpenMPI or MPICH                       |
| CMake           | >= 3.21       |                                        |
| Python          | >= 3.10       | System Python recommended (not conda)  |
| Ninja           | any           | Optional but recommended for speed     |
| Doxygen         | any           | Required for Cantera CLib codegen      |

OpenMPI **4.1.0 is rejected by default** because observed installations can
fail at large scales. CMake checks that exact release using the selected MPI headers;
choose another MPI release with matching compiler wrappers and libraries.
The check compiles but does not launch MPI, so it also works when cross-compiling.

For an explicitly validated 4.1.0 installation and transport, configure a
separate build directory with `-DDNDS_ALLOW_OPENMPI_410=ON`. The option defaults
to `OFF` and prints a warning when enabled. It only bypasses the exact-release
guard; it does not suppress MPI header/link failures or repair runtime bugs.
Keep compiler wrappers, libraries and runtime launcher from the intended MPI
installation together. Set the option back to `OFF` to restore the guard.

C++ libraries (managed via `external/cfd_externals` submodule):
Eigen, Boost, CGAL, nlohmann_json, fmt, pybind11, HDF5, CGNS,
Metis, ParMetis, ZLIB. Optional: CUDA toolkit, SuperLU_dist. CUDA builds enable
NCCL by default; provide its headers/runtime or explicitly opt out as below.

## Building External Dependencies

DNDSR requires two sets of external dependencies: header-only libraries
shipped as a tarball, and binary libraries built from the `cfd_externals`
submodule.

### Header-only libraries (Eigen, Boost, CGAL, fmt, spdlog, pybind11, ...)

Download the pinned v0.3.0 release tarball from GitHub and extract it into
the `external/` directory:

```bash
curl -L -o external/external_headeronlys.tar.gz \
  https://github.com/harryzhou2000/cfd_externals_headeronlys/releases/download/v0.3.0/external_headeronlys.tar.gz
echo "00692e1581f900d76c43432f4712b8dc2a53408d968fbf656bd3dd3561e2514e  external/external_headeronlys.tar.gz" | sha256sum -c -
cd external
tar -xzf external_headeronlys.tar.gz
cd ..
```

After extraction, directories such as `external/eigen/`,
`external/boost/`, `external/CGAL/`, etc. should exist.

### Binary libraries (HDF5, CGNS, Metis, ParMetis, ZLIB, Caliper)

```bash
git submodule update --init --recursive --depth=1
cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
cd ../..
```

The binary helper always includes CPU Caliper. CUDA profiling services and the
modified single-device AMGX fork are explicit options; see the
[AMGX dependency guide](amgx.md). The v0.3.0 header bundle also provides the
coherent `external/cccl` headers used by experimental compute arithmetic.

This installs all binary libraries into `external/cfd_externals/install/`.

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

## System Build Dependencies

Ensure MPI and basic build tools are available before proceeding:

```bash
# Debian/Ubuntu
sudo apt install build-essential cmake ninja-build openmpi-bin libopenmpi-dev doxygen

# RHEL/Fedora
sudo dnf install gcc-c++ cmake ninja-build openmpi-devel
```

## Python Virtual Environment

The Python package and tests need a virtual environment. Create it first,
then install Cantera build dependencies before building cfd_externals,
and the full Python environment after:

```bash
python3.12 -m venv venv
source venv/bin/activate

# First: install packages needed to build cfd_externals (Cantera)
pip install -r external/cfd_externals/requirements.txt

# Then build cfd_externals (HDF5, CGNS, Metis, ParMetis, Cantera)
# (see "Binary libraries" section above)

# Finally: install full Python environment (h5py, mpi4py, numpy, etc.)
bash scripts/install_python_deps.sh
```

> **Why not conda?** Conda Python binaries embed an `RPATH` pointing to
> conda's bundled libstdc++, which may be too old for the compiler used
> to build DNDSR.  Using the system Python avoids this entirely.

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

> **Note:** Each preset writes to its own build directory (`build/`, `build-debug/`, `build-cuda/`, etc.).
> If you switch presets, use the corresponding directory in subsequent `cmake --build` and `ctest` commands.

### Manual CMake Configuration

Prefer **Ninja for new build directories** when it is installed, unless a user
or cluster profile requires another generator. Use `-G Ninja` explicitly; it
can also accompany a configure preset, for example
`cmake --preset release-test -G Ninja`. Existing build directories keep their
configured generator. Do not delete their cache or compiled objects just to
change generators; use a separate directory if a migration is needed.

Ninja schedules independent targets in one dependency graph. With CMake's Unix
Makefiles generator, listing multiple targets can process them sequentially,
even with `-j16`; parallelism then applies only inside the current target. For
an existing Makefiles build, prefer a suitable aggregate target such as
`all_unit_tests` when building a group. Keep the requested parallelism bounded
by the local or cluster allocation.

```bash
cmake -S . -B build -G Ninja -DDNDS_BUILD_TESTS=ON
cmake --build build -t euler -j32           # Build a solver
cmake --build build -t dnds_unit_tests -j32 # Build C++ tests
ctest --test-dir build -R dnds_ --output-on-failure
```

Let CMake detect the system default compiler. Use `CC=mpicc CXX=mpicxx`
when unsure which MPI implementation CMake will find.

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
| `DNDS_USE_NCCL`             | CUDA setting | Enable NCCL payload provider; explicit OFF supported |
| `DNDS_USE_OMP`              | ON      | Enable OpenMP                          |
| `DNDS_DIST_MT_USE_OMP`      | OFF     | Enable native distributed row teams; see [CPU threading](../architecture/CPUThreading.md) |
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
| `cauchy_pybind11`| Cauchy   | `src/Cauchy/Bindings/*.cpp` | `python/DNDSR/Cauchy/_ext/`   |
| `eulerP_pybind11`| EulerP   | `src/EulerP/*_bind*.cpp` | `python/DNDSR/EulerP/_ext/`    |

Each pybind11 module links against a corresponding `*_shared` library
(`dnds_shared`, `geom_shared`, `cfv_shared`, `cauchy_shared`, `eulerP_shared`) which
contains the compiled C++ code.

For in-tree C++ consumers, `cfv` links through `geom` to `dnds`.
The shared counterparts publish the same native dependency chain.
`physics_shared` exposes its DNDS dependency, and the DNDS targets expose
MPI for their public headers. Consumers still need the project's configured
include directories. Solver time-equation templates are headers; native
algebra and provider implementations are in DNDS.

On Unix builds with precompiled headers disabled, the static and shared
Cauchy libraries consume the same private position-independent objects for
their generated CPU and CUDA flux catalogs. Both CUDA consumers still perform
their own device link. The generated kernels retain their per-source device
assert and register options. Builds using precompiled headers or a non-Unix
platform compile the catalogs within each consumer target.

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

NCCL is enabled by default with CUDA. Point `NCCL_ROOT` at an installation with
`include/nccl.h` and `lib/libnccl.so` (a versioned `libnccl.so.2` also works), or
set `DNDS_NCCL_INCLUDE_DIR` and `DNDS_NCCL_LIBRARY` directly. No Torch dependency
or automatic SDK download is introduced. Missing enabled NCCL dependencies are
a configuration error. To build CUDA without NCCL, pass `-DDNDS_USE_NCCL=OFF`.
An existing cached choice is preserved; clear that cache entry to apply the new
default. See [NCCL transfers](../architecture/NCCLTransfer.md) for runtime selection.

```bash
cmake -S . -B build-cuda -G Ninja -DDNDS_USE_CUDA=ON
cmake --build build-cuda -t euler -j32
```

Or use the `cuda` preset:

```bash
cmake --preset cuda
cmake --build build-cuda -j32
```

### Explicit toolkit selection

The Cauchy CUDA optimization work targets **CUDA 13.2** for new builds while
retaining **CUDA 12.6** compatibility. Select the compiler explicitly and use
a separate build directory for each toolkit; changing `PATH` does not replace
the CUDA compiler already recorded in an existing CMake cache:

```bash
cmake -S . -B build-cuda13 -G Ninja -DDNDS_USE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.2
# Compatibility build, with independent cache and artifacts:
cmake -S . -B build-cuda12 -G Ninja -DDNDS_USE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.6
```

Use an installed toolkit path appropriate for the host. CUDA-dependent binary
providers must match that selection too: in particular, rebuild AmgX into a
separate prefix because CUDA13 changes cuBLAS/cuSOLVER library major versions.
Check the resulting executable's resolved libraries, not only `nvcc --version`.
See [AmgX](amgx.md) and the [optimization record](../dev/plan612_cuda_optimization.md)
for qualification scope and the optional source-CCCL arithmetic dependency.

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

### Harness Regression Tests

These tests cover the commit hook, CTest summary reporting, build parallelism,
timeout selection, preset compatibility, shared skill paths, and MPI version
rejection. They use temporary fixtures and do not import DNDSR or require
building/installing its Python extensions:

```bash
venv/bin/python -m unittest discover -s scripts/tests -v
# Also registered when DNDS_BUILD_TESTS=ON:
ctest --test-dir build -R '^harness_tooling$' --output-on-failure
```

Set `DNDS_HARNESS_CMAKE=/path/to/cmake` for the standalone command to test a
different CMake installation. Preset schema 3 is the lowest version that
preserves automatic generator selection; it is supported by CMake 3.21.

### C++ Unit Tests

C++ tests use [doctest](https://github.com/doctest/doctest) and live
under `test/cpp/`.  MPI tests are registered with CTest at np=1, np=2,
and np=4 (DNDS and Geom tests additionally at np=8).

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
`dnds_test_serializer`, `dnds_test_permutation_transfer`.

By default, no per-test CTest `TIMEOUT` property is set, so use
`ctest --timeout <seconds>` to bound tests at runtime. Setting
`DNDS_TEST_TIMEOUT=<seconds>` during configuration explicitly enables the
existing per-category/rank timeout values, which take precedence over that
command-line default. The independent pytest timeout fallback remains 1800
seconds when invoked through CTest.

### Python Tests

Python tests use pytest with pytest-mpi and pytest-timeout, and live under `test/`.  The root
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

The repository VS Code settings start clangd with
`--compile-commands-dir=.`. The extension launches clangd with the opened
workspace folder as its working directory, so this pins compilation database
lookup to that checkout. This matters for linked worktrees stored below another
DNDSR checkout: without the pin, clangd may also discover the ancestor
checkout's database. The settings also allow clangd to query common MPI C++
wrapper names for their implicit MPI and system include paths; no particular
MPI installation path is embedded.

After configuring or changing build targets, refresh the processed database
and restart clangd:

```bash
cmake --build build -t process-compile-commands
# VS Code command palette: clangd: Restart language server
```

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

### SYCL Runtime Array and mesh placement

Configure a separate build with a DPC++ C++ compiler and `-DDNDS_USE_SYCL=ON`.
Runtime assigns a local placement to each selected device and owns one USM
context per device. `SelectSYCLDevice(SYCLTarget::CPU)` selects a CPU with
device USM. `SYCLTarget::CUDA` selects the NVIDIA CUDA adapter. Multiple queues
from `CreateSYCLQueue(device)` share the allocation context. Submit kernels
through `SYCLQueue(context)` and attach `Completion::Record(context)` to their
operand leases before releasing them.

For local qualification, enable tests, build `geom_test_sycl_migration`, and run
`ctest -R '^geom_sycl_(cpu|cuda)_migration_np2$' --output-on-failure`.
The probe compiles a SPIR image and an NVIDIA image. Set the semicolon-separated
CMake list `DNDS_SYCL_TEST_FLAGS` to select the installed CUDA toolkit and SM.
For example, CUDA 12.6 and an A100 use
`-fsycl-targets=spir64,nvptx64-nvidia-cuda;--cuda-path=/usr/local/cuda-12.6;-Xsycl-target-backend=nvptx64-nvidia-cuda;--offload-arch=sm_80`.
Both MPI ranks must be able to access the selected device.
