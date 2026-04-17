# DNDSR

DNDSR is a C++17 / Python CFD research code implementing Compact Finite Volume
methods with MPI parallelism and optional CUDA GPU support.

Solver executables:

- `euler` / `euler3D` — 2D/3D Navier-Stokes
- `eulerSA` / `eulerSA3D` — Spalart-Allmaras RANS
- `euler2EQ` / `euler2EQ3D` — k-omega two-equation RANS

See [docs/project_structure.md](docs/project_structure.md) for the full
directory layout, module dependencies, and build targets.

## Quick Start

### 1. Clone and fetch dependencies

```bash
git submodule update --init --recursive --depth=1

# Build binary external libraries (HDF5, CGNS, Metis, ParMetis, ...)
cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
cd ../..

# Download and extract header-only libraries (Eigen, Boost, CGAL, fmt, pybind11, ...)
# from the GitHub release:
curl -L -o external/external_headeronlys.tar.gz \
  https://github.com/harryzhou2000/cfd_externals_headeronlys/releases/latest/download/external_headeronlys.tar.gz
cd external
tar -xzf external_headeronlys.tar.gz
cd ..
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

## Type Stubs

After installing the Python package, generate `.pyi` type stubs for IDE
support:

```bash
PATH="venv/bin:$PATH" bash scripts/generate-stubs.sh
```

This produces stubs in `stubs/` and copies them into `python/DNDSR/` for
PEP 561 compliance. The stubs should be committed to git.

## VS Code / Pylance Setup

For Pylance to resolve `from DNDSR import DNDS` correctly, add to
`.vscode/settings.json`:

```jsonc
{
    "python.analysis.extraPaths": ["${workspaceFolder}/python"],
    "python.analysis.packageIndexDepths": [
        {"name": "DNDSR", "depth": 10, "includeAllSymbols": true}
    ]
}
```

Also set the Python interpreter to the project venv
(`venv/bin/python`) via the VS Code Python extension.

## Documentation

- [Project Structure](docs/project_structure.md) — directory layout, module
  dependencies, build targets.
- [Building](docs/building.md) — full build instructions, CMake presets,
  Python packaging.
- [TODO](docs/TODO.md) — planned improvements.
- Doxygen: `cmake --build build -t docs` (output in `docs/html/`).

## License

See [LICENSE](LICENSE).
