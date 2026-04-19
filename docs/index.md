(index_page)=
# DNDSR Documentation

DNDSR is a C++17 / Python CFD (Computational Fluid Dynamics) research code
implementing Compact Finite Volume methods with MPI parallelism and optional
CUDA GPU support.

## Core Modules

| Module     | Directory    | Description                                              |
|------------|-------------|----------------------------------------------------------|
| **DNDS**   | `src/DNDS`  | MPI arrays, serialization (JSON, HDF5), profiling, CUDA  |
| **Geom**   | `src/Geom`  | Unstructured mesh, CGNS I/O, partitioning (Metis/ParMetis) |
| **CFV**    | `src/CFV`   | Compact Finite Volume, variational reconstruction        |
| **Euler**  | `src/Euler` | Compressible Navier-Stokes solvers (2D/3D, SA, k-omega)  |
| **EulerP** | `src/EulerP`| Alternative evaluator with CUDA GPU support              |
| **Solver** | `src/Solver`| ODE integrators and Krylov solvers (GMRES, PCG)          |

## Quick Start

```sh
# 1. Build external dependencies
git submodule update --init --recursive --depth=1
cd external/cfd_externals && CC=mpicc CXX=mpicxx python cfd_externals_build.py && cd ../..

# 2. Build C++ solvers
cmake --preset release-test
cmake --build build -t euler -j32

# 3. Install Python package (editable)
python3.12 -m venv venv && source venv/bin/activate
pip install numpy scipy pytest pytest-mpi pytest-timeout mpi4py pybind11 pybind11-stubgen scikit-build-core ninja
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=32 pip install -e . --no-build-isolation

# 4. Run tests
ctest --test-dir build -R dnds_ --output-on-failure   # C++
pytest test/DNDS/test_basic.py -v                      # Python

# 5. Build and serve documentation
cmake --build build -t serve-docs
```

## C++ API (Doxygen)

The Sphinx C++ API pages provide namespace outlines and key class documentation
via Breathe. For the full Doxygen HTML documentation with interactive
**call graphs**, **class inheritance diagrams**, and **include dependency graphs**,
build and serve the Doxygen output separately:

```sh
cmake --build build -t serve-doxygen
```

Once built, the full Doxygen reference is at
<a href="doxygen/index.html">Doxygen C++ API Reference</a> within the same site.

```{toctree}
:maxdepth: 2
:caption: Contents

/sphinx/guides
/sphinx/architecture
/sphinx/theory
/sphinx/tests
/sphinx/dev
/sphinx/api_cpp
/sphinx/api_python
```
