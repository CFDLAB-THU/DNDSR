# AGENTS.md — DNDSR

> **CRITICAL RULE — NEVER COMMIT WITHOUT AUTHORIZATION.**
> `git commit`, `git commit --amend`, and `git push` require an explicit
> "proceed", "go ahead", "authorized", or similar directive from the user.
> "Looks good", "ok", or passive approval is NOT sufficient.  If in doubt,
> ask: "Proceed with commit?"

DNDSR is a C++17 / Python CFD (Computational Fluid Dynamics) research code implementing
Compact Finite Volume methods with MPI parallelism and optional CUDA GPU support.

Start with the [documentation index](docs/index.md), then the
[mesh connectivity and construction guide](docs/architecture/MeshConnectivity.md)
before editing mesh setup, relation operations, closure or ownership. The guide
distinguishes owned construction, temporary algorithm closure and final ghost
assembly; test counts alone do not establish legacy semantic compatibility.

## Temporary Source Trees and Cleanup

By default, put task-specific DNDSR worktrees, standalone source checkouts,
and source-transfer bundles under the canonical checkout's
`workspace/worktrees/`, with a descriptive task name. Do not create temporary
`DNDSR-*` or `dndsr-*` siblings in `~/projects/`. An explicit user-selected
location takes precedence. Existing active session workspaces stay where they
are unless the user requests migration.

Create an isolated source tree only when the task needs it, and keep its build
directory paired with that source tree. Reuse verified canonical dependencies
rather than duplicating them. Record the source revision and purpose.

Before retiring a tree, check for active users/processes, unique commits,
uncommitted changes, and untracked or ignored research artifacts. Preserve
necessary patches, base revisions, logs and evidence under
`workspace/worktrees/archives/` or the relevant study. Verify transfer archives
before removing unpacked duplicates. Remove linked worktrees through Git;
never discard changes or remove a tree without the required user authorization.

## CodeGraph in Separate Checkouts

When working in a standalone checkout or linked Git worktree, an agent may
initialize a checkout-local CodeGraph index if needed and the `codegraph` CLI
is available, including when `.codegraph/` does not yet exist. Run
`codegraph init .` from that checkout's root. This is optional, not a requirement
to create a separate worktree.

If CodeGraph reports that its index belongs to another checkout, do not treat
that index as authoritative for the active checkout. Use a checkout-local index
or verify against the active checkout's source. Keep initialization scoped to
the intended checkout; do not reinitialize another worktree's index.

## Shared Agent Skills

Keep project skills in `.opencode/skills/`, shared by OpenCode and Codex.
`.agents/skills` and the legacy `.codex/skills` are symlinks to that same
directory; edit the shared source, not separate harness-specific copies.

## Project Structure

- `src/` — C++ and Python source, organized by module:
  - `DNDS/` — Core: MPI arrays, serialization (JSON, HDF5), profiling, CUDA
    - `Config/` — Runtime configuration enums, parameters, registry
    - `Device/` — Host-device memory transfer, CUDA utilities
    - `Serializer/` — JSON and HDF5 serialization framework
    - `Runtime/` — Native shared arrays, layouts, access leases, and communication
    - `Algebra/` — Native DOF ownership and numerical operators
    - `Legacy/` — Historical arrays, transformers, DOFs, and `ArrayDerived/` wrappers
  - `Geom/` — Unstructured mesh, CGNS I/O, partitioning (Metis/ParMetis)
    - `Unstructured/` — Mesh owner, immutable state/snapshot, builders, and preparation
    - `Access/` — Read leases, mesh views, entity spaces, and fields
    - `Connectivity/` — Incidence operations, topology, and ghost closure
    - `Transition/` — Assembly, transitions, and reordering
    - `Geometry/` — Geometry edits, wall distance, and smoothing
    - `Refinement/` — Elevation, refinement, and extraction
    - `IO/` — Archives, CGNS/OpenFOAM import, and output
    - `Compat/` — Legacy views and adapters
    - `Mesh/` — Forwarding headers for historical include paths
    - `Elements/` — Per-element-type shape functions
    - `Quadratures/` — Numerical integration rules
  - `CFV/` — Compact Finite Volume, variational reconstruction
  - `Physics/` — Shared thermodynamics, transport, chemistry workers and RANS kernels
    - See `docs/architecture/Physics.md` for units, energy conventions and ownership
  - `Cauchy/` — Native FV solver, equation state, domain scheduling and implicit solves
  - `Euler/` — Compressible N-S solvers (2D/3D, SA, k-omega RANS)
  - `EulerP/` — Alternative evaluator with CUDA GPU support
- `app/` — C++ application entry points (solver executables)
- `test/` — Python tests (pytest + pytest-mpi + pytest-timeout)
- `cases/` — JSON configuration files for solver runs
- `external/` — Git submodule (`cfd_externals`) and header-only libraries

## Build Commands

> **For humans:** The canonical build guide with full explanations,
> troubleshooting, and platform notes lives in `docs/guides/building.md`.
> This section is a condensed agent reference.

### C++ (CMake)

For a **new build directory**, prefer the Ninja generator whenever `ninja` is
available, unless the user or selected cluster profile requires another
generator. Pass `-G Ninja` explicitly, including with a configure preset.
Preserve the generator of an existing build directory; never delete its cache
or rebuild it from scratch solely to switch generators. For existing Unix
Makefiles builds, use one aggregate target when building several independent
targets: a list of separate targets may serialize them despite `-j`.

```bash
# Configure (from project root)
mkdir build && cd build
CC=mpicc CXX=mpicxx cmake .. -G Ninja # Omit -G Ninja if unavailable.
# ^ Use CC=mpicc CXX=mpicxx when unsure which MPI CMake will find.

# Or use CMake presets (see CMakePresets.json)
cmake --preset release-test -G Ninja # Fresh Release build with tests enabled
cmake --preset debug -G Ninja        # Fresh Debug build with tests enabled
cmake --preset cuda -G Ninja         # Fresh CUDA build with tests

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
   cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
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

**IMPORTANT: Before running Python tests under `test/`, you MUST build the pybind11
shared libraries AND install them.** The Python modules load `.so` files
from `python/DNDSR/`, which are only placed there by `cmake --install`.
Running Python tests against stale or missing `.so` files will produce
misleading crashes (segfaults, aborts, wrong results) that look like code
bugs but are actually stale-binary problems. **After functional C++ changes,
repeat both steps before running Python tests. Formatter-only changes do not
require a rebuild or reinstall:**

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

Tooling-only regression tests do not import DNDSR or require its C++ libraries:

```bash
venv/bin/python -m unittest discover -s scripts/tests -v
```

### HARD RULE — never discard changes you did not make

**Every modified tracked file belongs to the user** (or another agent).
The agent is a visitor in the user's working tree.  It may edit files,
stage changes, or create new files, but it must **never** destroy the
user's uncommitted work.

**Prohibited operations on files/state the agent did NOT modify:**
- `git checkout -- <file>`
- `git restore <file>`
- `git checkout <file>`
- `git reset --hard` / `git clean -f`
- overwriting a file with `git show <rev>:<path> > <path>`
- any command whose effect is to discard or revert the user's changes

If a clean working tree is genuinely needed (e.g. for a benchmark
baseline, or to isolate the agent's own edits from pre-existing noise),
**push a stash with a brief message first**. Record its identity and what it
contains; plain `stash push` does not include untracked or ignored files:

    git stash push -m "baseline: user changes before <task description>"

After applying or popping, inspect the exit status, `git status`, the diff,
and `git stash list`. A conflicted pop can partially apply changes and leave
conflict markers while retaining the stash. Do not assume a retained stash
means nothing was applied, and do not retry blindly. Preserve the stash until
the restored changes have been verified; never discard conflicts or user work.

**Before running ANY `git checkout`, `git switch`, `git restore`,
`git reset`, `git stash`, `git stash pop`, or `git checkout -- <file>`
command, ALWAYS run `git status` first.** Verify the working tree is
clean or that all valuable changes are committed/stashed.
    Do not guess — a "clang-format: N file(s)" pre-commit message does
    not guarantee the changes are formatting-only.

### Running Solver Executables

Solver executables (`euler`, `eulerEX`, `eulerSA`, etc.) are run from the
`build/` directory.  **All file paths in JSON config files (mesh, output,
mechanism) are relative to the CWD at invocation time** — typically
`build/`.  Use `../` to reach the project root:

```bash
# From build/ — typical invocation
cd build
DNDS_MECH_PATH=../external/cfd_externals/install/data \
  ./app/eulerEX.exe 14 ../cases/eulerEX/react_test.json
```

**Path conventions in configs:**
- `meshFile`: relative to CWD (e.g. `../data/mesh/IV10_10.cgns`)
- `outPltName`: relative to CWD (e.g. `../data/out/react_test/react_`)
- `mechanismFile` (in `reactiveFlow`): relative to `DNDS_MECH_PATH` env var or CWD

**Editing solver JSON configs:** must use the edit tool.  Read and
understand the file first, then edit in place — never rewrite the
whole config with Python `json.dump`.  Python is acceptable for
read-only queries (`json.load`, `json.tool` validation).  Configs may
contain hand-maintained `caseNotes["/**/"]` sections and inline
comments that must be preserved across edits.

**Generated Euler defaults:** every Euler-series solver regenerates its adjacent
`<solver>_default_config.json` and writes a leading `"/**/"` notice declaring
it generated input. Do not edit, stage, or track these files. Keep a
hand-maintained case JSON separately; the solver merges it over the generated
default and then applies CLI `-k`/`-v` overlays.

**Updating JSON schemas:** never edit generated schema files by hand. Rebuild
the corresponding executable and regenerate the schema with `--emit-schema`.

Tests use **pytest** with **pytest-mpi** and **pytest-timeout**. Test files live under `test/`. A default 120-second timeout is configured in `pyproject.toml` to prevent hung MPI tests from blocking CI.

MPI-marked Python tests require `--with-mpi`, including when pytest is launched
under `mpirun`. Multiple processes alone do not enable those tests. Inspect
skip reasons (`-rs`) before crediting MPI coverage; rank-count-specific skips
can still be expected at np=1.

```bash
# Run all tests
pytest test/

# Run a single test file
pytest test/DNDS/test_basic.py

# Run a single test function
pytest test/DNDS/test_basic.py::test_all_reduce_scalar

# Run with MPI (multiple ranks)
mpirun -np 4 python -m pytest --with-mpi test/DNDS/test_basic.py

# Run a test file as a script (some tests support this)
python test/DNDS/test_basic.py
mpirun -np 2 python test/DNDS/test_basic.py
```

### C++ Unit Tests (doctest)

C++ tests live under `test/cpp/` and use the [doctest](https://github.com/doctest/doctest)
framework. They are built when `DNDS_BUILD_TESTS=ON` and registered with CTest.
MPI tests are registered at np=1, np=2, np=4, and np=8 by default (configurable via
`DNDS_TEST_NP_LIST` environment variable at configure time). All tests run with
`OMP_NUM_THREADS=2` by default (configurable via `DNDS_TEST_OMP_THREADS` environment
variable at configure time).

```bash
# Configure with tests enabled (from build directory)
cmake .. -DDNDS_BUILD_TESTS=ON

# Configure with custom OMP threads (optional)
DNDS_TEST_OMP_THREADS=4 cmake .. -DDNDS_BUILD_TESTS=ON

# Build all C++ unit tests (all categories)
cmake --build . -t all_unit_tests -j8

# Build only specific category
cmake --build . -t dnds_unit_tests -j8   # DNDS/ tests only
cmake --build . -t geom_unit_tests -j8   # Geom/ tests only
cmake --build . -t cfv_unit_tests -j8    # CFV/ tests only
cmake --build . -t euler_unit_tests -j8  # Euler/ tests only
cmake --build . -t solver_unit_tests -j8 # Solver/ tests only

# Run all C++ tests via CTest
ctest --test-dir . --output-on-failure

# Run with aggregated doctest summary (shows total test cases + assertions)
python scripts/ctest_summary.py --output-on-failure
python scripts/ctest_summary.py -R "^dnds_"   # filter by category

# Run tests by category prefix
ctest --test-dir . -R "^dnds_" --output-on-failure   # DNDS tests
ctest --test-dir . -R "^geom_" --output-on-failure   # Geom tests
ctest --test-dir . -R "^cfv_" --output-on-failure    # CFV tests
ctest --test-dir . -R "^euler_" --output-on-failure  # Euler tests
ctest --test-dir . -R "^solver_" --output-on-failure # Solver tests

# Run only np=2 MPI tests across all categories
ctest --test-dir . -R "_np2$" --output-on-failure

# Run a single test executable directly
./test/cpp/dnds_test_array
mpirun -np 4 ./test/cpp/dnds_test_mpi
```

**Test categories and targets:**

- **DNDS:** `dnds_test_array`, `dnds_test_mpi`, `dnds_test_array_transformer`,
  `dnds_test_array_derived`, `dnds_test_array_dof`, `dnds_test_index_mapping`,
  `dnds_test_serializer`, `dnds_test_permutation_transfer`
- **Geom:** `geom_test_elements`, `geom_test_quadrature`, `geom_test_mesh_index_conversion`,
  `geom_test_mesh_pipeline`, `geom_test_mesh_distributed_read`, `geom_test_mesh_connectivity`,
  `geom_test_mesh_connectivity_ghost`, `geom_test_mesh_connectivity_interpolate`,
  `geom_test_mesh_reorder`
- **CFV:** `cfv_test_reconstruction`, `cfv_test_limiters`, `cfv_test_reconstruction3d`,
  `cfv_test_device_transferable` (CUDA only)
- **Euler:** `euler_test_gas_thermo`, `euler_test_riemann_solvers`, `euler_test_rans`,
  `euler_test_evaluator_pipeline`
- **Solver:** `solver_test_ode`, `solver_test_linear`, `solver_test_direct`, `solver_test_scalar`

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
from DNDSR.Geom.utils import read_mesh, prepare_mesh
from DNDSR import DNDS

mpi = DNDS.MPIInfo()
mpi.setWorld()

# Read mesh with elevation and bisection
result = read_mesh(
    "data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
    elevation="O2",         # Elevate O1→O2
    bisect=1,               # Bisect once
)
prepare_mesh(result.mesh, result.reader)
```

The legacy `create_mesh_from_CGNS` wrapper is still available for backward
compatibility.

**Key Features:**

- CGNS mesh reading (`ReadFromCGNSSerial`)
- H5 distributed reading with ParMetis repartition
- Order elevation: Quad4→Quad9, Hex8→Hex27, etc. (`BuildO2FromO1Elevation`)
- Mesh bisection for h-refinement (`BuildBisectO1FormO2`)
- Boundary mesh extraction (`build_bnd_mesh`)
- Multi-layer ghost cells via `BuildGhostPrimary(nGhostLayers)`
- VTK output generation
- Wall distance computation (`BuildNodeWallDist`)
- CUDA device offloading (`to_device` / `to_host`)

## Code Style

Full style guide (naming, formatting, includes, error handling, Doxygen,
Python conventions): **`docs/guides/style_guide.md`**

Formatting is handled by the pre-commit hook; do not run formatters manually.
If an explicit formatting run is needed, use `scripts/run-clang-format.sh`.

When reporting audit or review findings to a human user, use numbered findings
(`1.`, `2.`, `3.`) rather than bullets so each issue can be referenced
unambiguously in follow-up discussion.

Quick reference for C++:

- **Braces:** Allman (opening brace on its own line)
- **Naming:** `PascalCase` classes/methods, `_` prefix for private members,
  `DNDS_ALL_CAPS` macros, `t_` prefix type aliases
- **Headers:** `#pragma once`, preserve include order (no auto-sort)
- **Errors:** `DNDS_assert` / `DNDS_check_throw` from `DNDS/Errors.hpp`; never raw
  `assert()`.
- DNDS currently builds with `DNDS_assert*` active by default at the maximum
  assertion level, including release-style builds. Still prefer
  `DNDS_check_throw_info` for non-hot user config, file input, CLI, and runtime
  validation that should report recoverable errors rather than aborting.
- **Core types:** `real = double`, `index = int64_t`, `rowsize = int32_t`,
  `ssp<T> = std::shared_ptr<T>`

Quick reference for Python:

- `snake_case` functions/variables; C++ wrapper classes match C++ name
- Plain `assert`; `@pytest.fixture` for MPI; numpy for array comparisons

### Clang-tidy sanitation

DNDS is clean as of 2026-04-29 (26-pass cleanup, 24 597 → 1
diagnostics; the remaining one is an unrelated Eigen PCH
`omp.h` include issue). Full per-pass record, `.clang-tidy`
disable rationale, and NOLINT placement gotchas:
**`docs/dev/clang_tidy_plan.md`**.

Other modules (`Solver`, `Geom`, `CFV`, `Euler`, `EulerP`) are
not yet sanitised. Apply the same recipe in that order. Run
`scripts/run_clang_tidy.py <module>` to get the per-check
histogram; the `.clang-tidy` disables carry forward unchanged.

## Geom Module Architecture

Mesh connectivity, ghost management, and the build pipeline are documented
in **`docs/architecture/MeshConnectivity.md`**.

Key concepts agents should know:

- **Native ownership:** `UnstructuredMesh` is the public semantic facade over
  one immutable `NativeMeshState`. Owned/ghost values and row metadata use
  Runtime Arrays. Mesh building, topology, closure and transitions use
  `NativeMeshBuilder`, `NativeMeshTopology` and `NativeMeshAssembler`; there is
  no legacy mesh construction workspace. Standalone legacy DNDS Arrays and
  solver-field transformers remain supported.

- **Access:** Acquire one `MeshReadAccess` per operator, request the required
  groups, and use its borrowed host/device view in loops. The scope retains
  native state, value/shape buffers and leases; keep it through asynchronous
  execution or join declared streams before release. Unrequested groups are
  hidden. Serialize facade setup/publication with access acquisition.

- **Relations:** `NativeConnectivity` evaluates full `AdjKind` descriptors.
  Physical slot relations, inverses and compositions are distinct definitions.
  `MeshConnectivity` provides native mathematical compatibility spellings.
  Compute publications use local indices; detached `NativeRelation::Convert`
  and `ConvertedIndex` perform global/local conversion with explicit source
  and target spaces, preserving absence/unavailable encodings. Legacy group
  flags and in-place whole-mesh conversion APIs have been retired.

- **Identity and mutation:** Entity spaces describe current numbering/order;
  persistent IDs survive reorder, redistribution and ghost construction.
  New-mesh refinement may allocate a new namespace with explicit provenance.
  Stage and validate replacements before publication, preserve version masks,
  and reject conflicting readers. Pure placement/auxiliary cache setup does
  not change geometry generations; retained scopes pin their acquired state.

- **Sparse closure:** Ghost discovery and remapping must not allocate
  global-entity-sized local scratch. Storage scales with local requested/served
  incidences plus ranks. Explicit serial file ingress/output may gather on its
  designated reader; this exception does not apply to ghost construction.

## Key Dependencies

- **Compiler:** GCC 9+ / Clang 8+, C++17 required
- **MPI:** MPI-3 compatible (OpenMPI or MPICH); OpenMPI 4.1.0 is rejected
  at configure time because of large-scale transport hangs.
- **CMake:** >= 3.21
- **C++ libs:** Eigen, Boost, CGAL, nlohmann_json, fmt, pybind11, HDF5, CGNS, Metis, ParMetis
- **Python:** >= 3.10, numpy, scipy, pytest, pytest-mpi, mpi4py, h5py
- **Optional:** CUDA toolkit, SuperLU_dist

## GitHub CLI (`gh`) Policy

## Git Commit Messages

Prefer verbose commit messages for non-trivial changes. Keep the subject concise
and conventional, then use the body to explain the problem, the root cause, the
fix, and the verification performed. Mention subtle behavioral changes,
edge-cases, and test coverage so future readers can understand why the change
was made without reconstructing the investigation from the diff.

For very small mechanical changes, a short message is acceptable.

Every commit message must end with a separate trailing line:

    committed by <agent harness> with model <model name>

Use the actual agent harness name (for example, `codex` or `opencode`) and model
identifier. This line must appear after a blank line separating it from the body
(or subject, if no body).

**Read-only by default.** You may use `gh` freely for read operations (viewing
issues, PRs, checks, releases, diffs, comments). **Do NOT use `gh` for any
write operation** (creating/closing issues, creating/merging PRs, posting
comments, approving reviews, creating releases, editing labels, deleting
caches, etc.) **unless the user explicitly requests that specific write
action.** One-time explicit permission does not carry over to other write
actions — ask each time.

**Operations requiring explicit user authorization (non-exhaustive):**
- `git commit` / `git commit --amend`
- `git push` / `git push --force` / `git push --force-with-lease`
- `gh pr create/merge/close/edit`
- `gh issue create/close/edit`
- `gh pr comment` / `gh issue comment`
- `gh pr review`
- `gh release create/delete`
- `gh cache delete`
- `gh api` with non-GET methods (POST, PUT, PATCH, DELETE)

Explicit authorization means a direct imperative like "commit", "proceed",
"go ahead", or "authorized".  Passive or ambiguous statements ("looks good",
"ok", "fine", "nice") are NOT authorization to commit.  A single-commit
authorization is consumed by the first commit — do NOT commit again without
a fresh authorization, even for follow-up fixes.

**Draft PR by default** You must use --draft on new prs.
