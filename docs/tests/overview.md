# Unit Test Suite Overview {#test_overview}

@tableofcontents

DNDSR uses [doctest](https://github.com/doctest/doctest) for C++ unit
tests and [pytest](https://docs.pytest.org/) (with
[pytest-mpi](https://pypi.org/project/pytest-mpi/)) for Python tests.
MPI-aware C++ tests are registered with CTest at multiple process counts
to verify parallel correctness.

## Module Test Pages

| Module | Page | C++ tests | Python tests | Assertions |
|---|---|---|---|---|
| **DNDS** (core) | @ref dnds_unit_tests | 7 executables | 2 pytest files (~10 tests) | ~400 |
| **Geom** | @ref geom_unit_tests | 5 executables | 1 pytest file (1 test) | — |
| **CFV** | @ref cfv_unit_tests | 3 executables | 2 pytest files (32 tests) | ~340 |
| **Euler** | @ref euler_unit_tests | 4 executables | — | ~310 |
| **Solver** | @ref solver_unit_tests | 4 executables | — | ~65 |

## Quick Start

```sh
# 1. Configure with tests enabled
cmake -B build -DDNDS_BUILD_TESTS=ON

# 2. Build all test executables
cmake --build build -t all_unit_tests -j8

# 3. Run the full C++ test suite
ctest --test-dir build --output-on-failure

# 4. Run Python tests (requires pybind11 shared libraries)
cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
cmake --install build --component py
source venv/bin/activate
PYTHONPATH=python pytest test/ -v
```

## Aggregate CMake Targets

| Target | Contents |
|---|---|
| `dnds_unit_tests` | All DNDS core test executables |
| `geom_unit_tests` | All Geom test executables |
| `cfv_unit_tests` | All CFV test executables |
| `euler_unit_tests` | All Euler test executables |
| `solver_unit_tests` | All Solver test executables |
| `all_unit_tests` | All of the above |

All test executables are `EXCLUDE_FROM_ALL` and must be built explicitly.

## MPI Test Registration

MPI-aware tests are registered at multiple process counts.  The CTest
name encodes the count:

| Module | Process counts | Timeout |
|---|---|---|
| DNDS | np = 1, 2, 4, 8 | 120-240 s |
| Geom | np = 1, 2, 4, 8 | 120-240 s |
| CFV | np = 1, 2, 4 | 120-180 s |
| Euler | np = 1, 2, 4 | 600 s |

Serial tests have a single CTest entry with a 60-120 s timeout.

## Naming Conventions

### C++ (CTest)

```
<module>_<test_name>             # serial
<module>_<test_name>_np<N>       # MPI at N ranks
```

Examples: `cfv_limiters`, `euler_evaluator_pipeline_np4`,
`solver_ode`.

### Python (pytest)

```
test/<Module>/test_<name>.py::TestClass::test_method
```

Examples: `test/CFV/test_fv_correctness.py::TestCellVolumes::test_wall_mesh_cell_volumes`.

## Golden Values and Regression

Many tests compare computed results against pre-captured **golden
values** with a relative tolerance (typically 1e-6 to 1e-8).  Golden
values are deterministic because:

- Iterative VR uses **Jacobi iteration** (not SOR) to avoid
  partition-dependent update ordering.
- Euler pipeline tests use **Jacobi update** (not LU-SGS) for the same
  reason.
- Metis partitioning uses a fixed seed (`metisSeed = 42`).

When a golden value has not yet been captured, the sentinel `1e300`
(or `0.0` in older tests) is stored.  In this mode the test only prints
the value and checks it is finite and non-negative — no regression
assertion.

## POSIX `index()` Ambiguity

Because doctest includes `<cstring>`, which transitively pulls in the
POSIX `index()` function from `<strings.h>`, the bare name `index` is
ambiguous when `using namespace DNDS;` is active.  All C++ test files
qualify the DNDS type aliases as `DNDS::index`, `DNDS::real`, and
`DNDS::rowsize` in variable declarations.

## Rebuilding Before Testing

Before running Python tests, always rebuild and reinstall the pybind11
shared libraries:

```sh
cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
cmake --install build --component py
```

If C++ source changed since the last build, stale `.so` files produce
misleading crashes (segfaults, aborts, wrong results) that look like code
bugs.
