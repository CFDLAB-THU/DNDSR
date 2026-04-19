# DNDSR

DNDSR is a C++17 / Python CFD research code implementing Compact Finite Volume
methods with MPI parallelism and optional CUDA GPU support.

**Documentation**: [cfdlab-thu.github.io/DNDSR](https://cfdlab-thu.github.io/DNDSR/)
(includes guides, architecture, API reference, and embedded
[Doxygen C++ docs](https://cfdlab-thu.github.io/DNDSR/doxygen/))

## Features

- **Solvers**: Euler / Navier-Stokes (2D/3D), SA-IDDES, k-omega RANS
- **Numerics**: Compact Finite Volume with variational reconstruction,
  Roe / HLLE+ Riemann solvers, ESDIRK / HM3 time integration, p-Multigrid
- **Parallelism**: MPI with ghost communication, optional CUDA GPU support
- **Python bindings**: pybind11 modules for mesh, CFV, and solver access
- **Config system**: typed JSON configs with schema validation

Solver executables:

| Executable | Model |
|------------|-------|
| `euler` / `euler3D` | Navier-Stokes |
| `eulerSA` / `eulerSA3D` | Spalart-Allmaras RANS (IDDES) |
| `euler2EQ` / `euler2EQ3D` | k-omega two-equation RANS |

## Quick Start

### 1. Clone and fetch dependencies

```bash
git clone --recursive https://github.com/CFDLAB-THU/DNDSR.git
cd DNDSR

# Build binary external libraries (HDF5, CGNS, Metis, ParMetis, ...)
cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
cd ../..

# Download and extract header-only libraries (Eigen, Boost, CGAL, fmt, pybind11, ...)
curl -L -o external/external_headeronlys.tar.gz \
  https://github.com/harryzhou2000/cfd_externals_headeronlys/releases/latest/download/external_headeronlys.tar.gz
cd external && tar -xzf external_headeronlys.tar.gz && cd ..
```

### 2. Build C++ solvers

```bash
# Using CMake presets (recommended)
cmake --preset release-test
cmake --build build -t euler -j32

# Or manually
mkdir build && cd build
CC=mpicc CXX=mpicxx cmake .. -DDNDS_BUILD_TESTS=ON
cmake --build . -t euler -j32
```

### 3. Run a solver

```bash
# Serial
./build/app/euler.exe cases/your_config.json

# Parallel
mpirun -np 4 ./build/app/euler.exe cases/your_config.json
```

Input parameters are defined in JSONC config files.
See [cases/euler_default_config_commented.json](cases/euler_default_config_commented.json)
for documentation of all options.

### 4. Install the Python package

Use the system Python (not conda) to avoid libstdc++ version conflicts:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install numpy scipy pytest pytest-mpi pytest-timeout mpi4py \
            pybind11 pybind11-stubgen scikit-build-core ninja

CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=32 \
    pip install -e . --no-build-isolation
```

> **Why system Python?** Conda/Anaconda Python embeds an RPATH to conda's
> bundled libstdc++, which may be older than what the MPI compiler produces.
> System Python uses the system libstdc++ and avoids this conflict.

After the initial install, rebuild only the C++ bindings without re-running pip:

```bash
cmake --build build_py \
    -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
cmake --install build_py --component py
```

### 5. Run tests

```bash
# C++ unit tests (doctest, via CTest)
cmake --build build -t dnds_unit_tests -j32
ctest --test-dir build -R dnds_ --output-on-failure

# Python tests
pytest test/DNDS/test_basic.py -v
```

### 6. Build and serve documentation locally

```bash
pip install -r docs/sphinx/requirements.txt
cmake --build build -t serve-docs
# Then: cd build/docs/sphinx && python3 -m http.server 8000
```

## Documentation

The full documentation is hosted at
**[cfdlab-thu.github.io/DNDSR](https://cfdlab-thu.github.io/DNDSR/)**
and includes:

- **Guides**: [Building](https://cfdlab-thu.github.io/DNDSR/guides/building.html),
  [Style Guide](https://cfdlab-thu.github.io/DNDSR/guides/style_guide.html),
  [Array Usage](https://cfdlab-thu.github.io/DNDSR/guides/array_usage.html),
  [Geometry + CFV](https://cfdlab-thu.github.io/DNDSR/guides/geom_usage.html),
  [Doc Authoring](https://cfdlab-thu.github.io/DNDSR/guides/doc_authoring.html)
- **Architecture**: Array infrastructure, Serialization, Paradigm
- **Theory**: Variational Reconstruction, Shape Functions
- **C++ API**: via [Breathe](https://cfdlab-thu.github.io/DNDSR/sphinx/api_cpp.html)
  and [Doxygen](https://cfdlab-thu.github.io/DNDSR/doxygen/) (with class diagrams)
- **Python API**: [autodoc](https://cfdlab-thu.github.io/DNDSR/sphinx/api_python.html)
- **Unit Tests**: test suite overview, per-module documentation

For local-only references:
- [Project Structure](docs/guides/project_structure.md)
- [Building](docs/guides/building.md)
- [Doc Authoring Guide](docs/guides/doc_authoring.md)

## Type Stubs

Type stubs (`.pyi`) are generated automatically during `cmake --install`
(and therefore during `pip install`). They are placed in `python/DNDSR/`
for PEP 561 compliance and do not need to be committed.

## VS Code Setup

For Pylance / clangd to work correctly, the project ships `.vscode/`
configs. Set the Python interpreter to `venv/bin/python` via the
VS Code Python extension.

## License

See [LICENSE](LICENSE).
