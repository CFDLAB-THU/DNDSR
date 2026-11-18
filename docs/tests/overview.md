# Unit Test Suite Overview {#test_overview}

@tableofcontents

DNDSR uses [doctest](https://github.com/doctest/doctest) for C++ unit
tests and [pytest](https://docs.pytest.org/) (with
[pytest-mpi](https://pypi.org/project/pytest-mpi/) and
[pytest-timeout](https://pypi.org/project/pytest-timeout/)) for Python tests.
MPI-aware C++ tests are registered with CTest at multiple process counts
to verify parallel correctness.

## Module Test Pages

| Module | Page | Aggregate target | Python suite |
|---|---|---|---|
| DNDS/Runtime/Algebra | @ref dnds_unit_tests | `dnds_unit_tests` | `test/DNDS` |
| Geom | @ref geom_unit_tests | `geom_unit_tests` | `test/Geom` |
| CFV | @ref cfv_unit_tests | `cfv_unit_tests` | `test/CFV` |
| Euler/shared Physics | @ref euler_unit_tests | `euler_unit_tests` | `test/Euler` |
| Solver | @ref solver_unit_tests | `solver_unit_tests` | — |
| Cauchy | [audit execution](../dev/plan53_audit_execution.md) | `cauchy_unit_tests` | `test/Cauchy` |

Executable/registration counts depend on CUDA, Cantera, Python and configured
MPI sizes. Inspect the configured suite with `ctest -N` or
`ctest --show-only=json-v1`; historical assertion totals are not a coverage gate.

## Quick Start

```sh
# 1. Configure with tests enabled
cmake -B build -DDNDS_BUILD_TESTS=ON

# 2. Build all test executables
cmake --build build -t all_unit_tests -j8

# 3. Run the full C++ test suite
ctest --test-dir build -L cpp --output-on-failure

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
| `cauchy_unit_tests` | All Cauchy test executables and frontend |
| `all_unit_tests` | All of the above |

Test executables are `EXCLUDE_FROM_ALL` and must be built explicitly.
Registration helpers record their targets; configuration rejects one that is
not reachable through `all_unit_tests`. Tooling tests exercise transitive
aggregate coverage and a deliberately missing dependency without compiling C++.

## Execution tiers and Python ownership

C++ registrations have `cpp`, module (`dnds`, `geom`, `cfv`, `euler`, `solver`,
`cauchy`) and `routine`/`integration` labels. `mpi-high` identifies np4/8;
these ranks remain registered. Use `ctest -L routine -LE mpi-high` for a bounded
np1/2 contract pass, or `ctest -L integration` for retained large-mesh and golden
numerical tests. This classification is not a measured whole-suite time budget.

The 3D CFV binary has a generated 64-cell constant/halo contract and a separate
32^3 periodic golden/convergence/order registration. Mesh pipeline/conversion
fixtures are lazy, so doctest `--count`, test and subcase filters do not import
unselected meshes. The realistic NACA/Ball2 suites remain integration tests;
existing small native smoothing tests retain independent Hermite boundary and
interior constraints in the routine tier. No large numerical tolerance or
iteration cap was reduced.

The topology-incidence integration case runs once by default. Repeated timing
is opt-in through `geom_benchmark_topology_np2`, or
`DNDS_TEST_TOPOLOGY_REPETITIONS=3 mpiexec -np N geom_test_mesh_pipeline
--test-case="Native inverse compose and face topology match independent incidences"`.
Use the runner's normal allocation and affinity policy for measurements.

CI excludes the CTest `python` label because its complete serial pytest step
owns extension coverage; explicit two-rank Runtime Python integration remains
separate. Standalone CTest users can still run the serial Python registrations
after building and installing extensions. `harness_tooling` needs no extensions.

## MPI Test Registration

The CI workflow has three cumulative test levels. Use `/ci-run` for the
critical gate, `/ci-run 1` for extended coverage, or `/ci-run full` for all
levels. `/ci-run-self-hosted` accepts the same suffixes. The manual workflow
uses the numeric level.

From a configured CPU test build, use `ci_critical_tests` and
`ci_python_tests` to build Level 0. Build `all_unit_tests` and
`ci_python_tests` for either higher level. Install the `py` component and
run `bash scripts/run_ci_tests.sh 0 <build> <venv>/bin/python` from this
checkout. Use the pinned CI meshes when validating a clean checkout.
The C++ registrations carry their minimum level as a CTest label.
`ci_critical_tests` builds the executables selected by the `level0` label.
Some np4 registrations split a named critical case from the rest of the
executable so the full run executes each case once.

MPI-aware tests are registered at multiple process counts.  The CTest
name encodes the count:

| Module | Process counts | Timeout |
|---|---|---|---|
| DNDS | np = 1, 2, 4, 8 | 120-240 s |
| Geom | np = 1, 2, 4, 8 | 120-240 s |
| CFV | np = 1, 2, 4, 8 | 120-180 s |
| Euler | np = 1, 2, 4, 8 | 600 s |

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
