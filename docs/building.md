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

```bash
git submodule update --init --recursive --depth=1
cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
```

## Building C++ (Solvers and Libraries)

### Using CMake Presets

```bash
cmake --preset release-test        # Release with unit tests enabled
cmake --build --preset tests -j32  # Build all C++ unit tests
ctest --preset unit                # Run unit tests
```

Available presets (defined in `CMakePresets.json`):

| Preset          | Build Type | Tests | Notes                         |
|-----------------|------------|-------|-------------------------------|
| `default`       | Release    | OFF   | Minimal solver build          |
| `debug`         | Debug      | ON    | Full debug symbols            |
| `release-test`  | Release    | ON    | Main development preset       |
| `cuda`          | Release    | ON    | Enables CUDA GPU support      |
| `ci`            | Release    | ON    | CI (no ccache)                |

### Manual CMake Configuration

```bash
mkdir build && cd build
CC=mpicc CXX=mpicxx cmake .. -DDNDS_BUILD_TESTS=ON
cmake --build . -t euler -j32           # Build a solver
cmake --build . -t dnds_unit_tests -j32 # Build C++ tests
ctest -R dnds_ --output-on-failure      # Run C++ tests
```

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

### Editable Install (Development)

```bash
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=32 \
    pip install -e . --no-build-isolation
```

This runs cmake, builds the pybind11 targets, installs the `.so` files
into `python/DNDSR/<Module>/_ext/`, and registers the package as editable.

After the initial install, rebuild only the C++ parts without re-running pip:

```bash
cmake --build build_py -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
cmake --install build_py --component py
```

### Package Install

```bash
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=32 \
    pip install . --verbose
```

### Controlling Build Parallelism

The default is `-j0` (all available cores).  Override via environment:

```bash
SKBUILD_BUILD_TOOL_ARGS="-j8" pip install -e . --no-build-isolation
```

## Generating Type Stubs

After installing the package (editable or full), generate `.pyi` stubs:

```bash
PATH="venv/bin:$PATH" bash scripts/generate-stubs.sh
```

This runs `pybind11-stubgen` for each submodule, writes raw output to
`stubs/`, and copies the `.pyi` files into `python/DNDSR/` for PEP 561
compliance.  The generated stubs should be committed to git.

## Running Tests

### C++ Unit Tests

```bash
# Build and run all (via CTest)
cmake --build build -t dnds_unit_tests -j32
ctest --test-dir build -R dnds_ --output-on-failure

# Run a single test executable directly
mpirun -np 4 ./build/test/cpp/dnds_test_mpi
```

### Python Tests

```bash
pytest test/DNDS/test_basic.py -v

# With MPI
mpirun -np 4 python -m pytest test/DNDS/test_basic.py
```

## Build Mode Summary

| Mode                     | Command                                          | What happens                                    |
|--------------------------|--------------------------------------------------|-------------------------------------------------|
| **Pure C++ build**       | `cmake --build . -t euler`                       | Compiles solver under src/, no Python artifacts  |
| **C++ unit tests**       | `cmake --build . -t dnds_unit_tests`             | Builds doctest executables under test/cpp/        |
| **Editable install**     | `pip install -e .`                               | Builds pybind11 .so, installs into python/_ext/  |
| **Editable C++ rebuild** | `cmake --build build_py` + `cmake --install ...` | Rebuilds only C++ bindings, updates .so in-place |
| **Package build**        | `pip install .`                                  | Full wheel build with .so files included         |
