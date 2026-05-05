# 🚀 DNDSR v0.2.0 — Infrastructure & Quality Release

155 commits · 385 files changed · 56,227 insertions · 10,506 deletions

📖 **Documentation**: [cfdlab-thu.github.io/DNDSR](https://cfdlab-thu.github.io/DNDSR/)

---

## 🔗 Geometry DSL & Reorder Framework

The centerpiece of this release is a complete declarative adjacency graph engine that replaces imperative element-traversal loops with composable DAG operations — the foundation for mesh reordering, ghost exchange, and multi-layer connectivity.

### 🌳 MeshConnectivity DAG DSL (`src/Geom/Mesh/MeshConnectivity.hpp`, 1,497 lines)

A layered framework of adjacency relations between entity strata (cells ↔ faces ↔ edges ↔ nodes) with three composable operations:

- **`Inverse`** — cone (A→B) → support (B→A), computing upward adjacencies from downward topology
- **`Compose`** — A→B + B→C → A→C, chaining adjacencies across strata
- **`ComposeFiltered`** — filtered composition with on-the-fly predicate matching (including PBI containment for periodic boundaries)

Periodic bits (PBI) are stored only on cones whose target depth is 0 (nodes), tracking how node coordinates transform under periodicity. The DSL is templated on adjacency row-size for optimal memory use.

### 👻 Ghost Chain Evaluator (`src/Geom/Mesh/MeshConnectivity_Ghost.cpp`, 583 lines)

Hybrid BFS evaluator for ghost cell construction supporting:
- Five ghost chain types: cell2node, face2node, cell2cell through faces/edges/nodes
- `evaluateGhostTree` with declarative chain definitions
- Multi-layer ghost cells (`BuildGhostPrimary(nGhostLayers)`) — arbitrary depth
- Standalone unit tests with analytical verification on synthetic grids

### 🔀 Reorder Framework (`src/Geom/Mesh/ReorderPlan.hpp`, 267 lines + `Mesh_Reorder.cpp`, 877 lines)

Two-layer architecture for distributed entity reordering:
- **`ReorderRegistry`** — dynamic callback container holding adjacency remapping, relocation, and companion-relocate callbacks
- **`ReorderPlan`** — standalone computed transfers + lookups, with forward/backward index mapping
- **`buildReorderRegistry`** — automatic callback population from mesh state
- **`ReorderEntities`** — complete reorder on real meshes: pull-set collection, son entity reattachment, cell2cellFace vertical face handling

### 🔄 PermutationTransfer (`src/DNDS/PermutationTransfer.hpp`, 363 lines)

Reusable distributed/local row permutation utility encapsulating the common pattern: given a partition assignment or forward map for a set of entities, compute new global indices and transfer array rows to target ranks. Supports both MPI push (distributed) and in-place permutation (local) paths.

### 🏷️ AdjPairTracked Infrastructure (`src/Geom/Mesh/AdjIndexInfo.hpp`, 355 lines)

Per-adjacency index state tracking with encapsulated `AdjIndexInfo` fields:
- `markGlobal()` / `markLocal()` / `toLocal()` / `toGlobal()` / `bootstrapToLocal()` API
- `wireTargetMapping()` for ghost mapping routing
- `fillRegistry()` for automatic reorder callback registration
- Device views for CUDA offloading
- Full contract coverage in state-checked wrapper contract
- Pybind11 exports for Python introspection

### 📐 Mesh Helpers & Pipeline

- **`mesh_helpers`** — unified C++ API for read/prepare/build workflows, migrated from inline Euler solver boilerplate
- **`Python Geom utils`** — `read_mesh()`, `prepare_mesh()`, `build_bnd_mesh()`, `build_fv()` with CGNS/H5 read modes, elevation, bisection
- **Distributed mesh read** — `ReadDistributed_Redistribute` and `ReorderLocalCells` migrated to declarative framework
- **Synthetic mesh builders** — 2D tiled synthetic grids with analytical ghost formulas for rigorous connectivity testing

### 🧬 Element Generation Tools

- **CGNS topology data** — 1-based reference tables mapping CGNS face/node numbering to DNDSR conventions
- **Element diagrams** — automated 3D edge-group visualization with tuned viewing angles
- **Traits emitter** — `traits_emitter.py` codegen for element shape function traits from CGNS data, including edge topology
- **Prism18** shape functions expanded with edge group metadata

---

## 🧹 Code Quality Crusade

A foundational rework of the DNDS/ core layer achieving near-zero warning diagnostics.

### DNDS/ Clang-Tidy Sanitation (26 passes)

From **24,597 warnings → 1** (remaining: an unrelated Eigen PCH `omp.h` include issue). Key transformations:

| Category | Check | Description |
|---|---|---|
| Modernization | `modernize-use-nullptr` | `NULL` → `nullptr` (entire codebase) |
| Modernization | `modernize-use-equals-default` | `{}` → `= default` for special members |
| Modernization | `modernize-use-emplace` | `push_back(T(...))` → `emplace_back(...)` |
| Modernization | `modernize-use-nodiscard` | `[[nodiscard]]` on all const getters |
| Modernization | `modernize-loop-convert` | Raw loops → range-based for where safe |
| C++ Core | `cppcoreguidelines-pro-type-cstyle-cast` | C-style casts → `static_cast` / `reinterpret_cast` |
| C++ Core | `cppcoreguidelines-avoid-c-arrays` | `T arr[N]` → `std::array<T, N>` |
| C++ Core | `cppcoreguidelines-pro-type-member-init` | Zero-init all raw members |
| C++ Core | `cppcoreguidelines-special-member-functions` | Close rule-of-five gaps |
| C++ Core | `cppcoreguidelines-init-variables` | Initialize locals before write |
| Bugprone | `bugprone-reserved-identifier` | Rename leading-underscore identifiers |
| Bugprone | `bugprone-unhandled-self-assignment` | Guard `AdjacencyRow::operator=` |
| Performance | `performance-unnecessary-value-param` | Pass by const-ref where copy unused |
| Readability | `readability-qualified-auto` | `auto *` / `const auto &` for clarity |
| Readability | `readability-simplify-boolean-expr` | Simplify boolean logic |
| Readability | `readability-named-parameter` | Annotate unused parameters |
| Readability | `readability-redundant-casting` | Drop unnecessary casts |

### 🎨 clang-format Consistency

Unified `.clang-tidy` and `.clang-format` configs with Python drivers (`run_clang_tidy.py`, `run_clang_format.py`) supporting:
- `--fix` serialization for incremental correction
- Per-check histogram reporting
- Cross-module consistent configuration

Formatting applied across all DNDS/ headers and drifted Euler/Solver files.

### 🔧 Move Semantics & Memory Safety

- Proper move constructors/assignment operators for all array types
- `ArrayPair::clone` latent bug fix (incorrect resource sharing)
- `rvalue-ref-parameter-not-moved` fix in `ArrayDofDeviceView`
- Default-disable LTO in pybind11 builds to prevent link-time ODR violations
- Stop bundling `libstdc++`/`libmpi` into `dndsr_external` shared library

---

## ⚙️ CI/CD Infrastructure

A full GitHub Actions pipeline replacing the previous stub-only workflow.

### 🏭 CI Workflow (`.github/workflows/ci.yml`, 375 lines)

- **Runner selection**: GitHub-hosted (`ubuntu-latest`) or self-hosted Docker containers via a lightweight "resolve" job → single "build-and-test" job (no wasted matrix entries)
- **Trigger system**: `/ci-run` and `/ci-run-self-hosted` PR comments with write-access gating — no label management needed
- **3-layer caching**: restore/save split for externals (submodule SHA keys) and Python venv, with runtime sentinel checking for cache prefix mismatches
- **ccache** with cross-branch cache persistence on both runner types
- **h5py/mpi4py** source builds to avoid HDF5 version conflicts with cfd_externals
- **pytest-timeout** integration for hung MPI test prevention
- **MPI oversubscribe** enforcement for GitHub runners
- **Cache save on failure** — ccache persists even when build or tests fail
- **OMP thread pinning** (`OMP_NUM_THREADS=2` by default)

### 📖 Docs Deployment (`.github/workflows/docs.yml`)

Manual Pages deployment workflow with 3-layer caching for incremental Sphinx/Doxygen builds.

### 📊 CTest Summary

`scripts/ctest_summary.py` — aggregated doctest statistics (total test cases + assertions) across all test categories with regex filtering.

---

## 📖 Documentation System

### 📚 Architecture Documentation (New)

- **MeshConnectivity Architecture** (`docs/architecture/MeshConnectivity.md`, 685 lines) — full state tracking model, three-layer architecture, conversion methods, group state invariants
- **Mesh DAG Design** (`docs/architecture/MeshDAGDesign.md`, 774 lines) — DAG DSL specification, ghost chain types, interpolate design, BFS evaluation algorithm

### 🎞️ Marp Slide Deck (New)

Complete DNDSR overview presentation pipeline (`docs/presentations/DNDSR_overview/`):

- **10 sections**: Title, Opening, Architecture, Geometry, Numerics, Parallelism, I/O Interop, Solvers, Engineering, Roadmap
- **Chinese translation** — full parallel `zh/` directory with `_no_zh` slide-level filter
- **Mermaid pre-render** — `render_mermaid.py` for DAG diagrams before Marp processing
- **Auto-fit & overflow check** — `auto_fit.py` and `check_overflow.js` for slide quality assurance
- **CI deployment** — `build.sh` pipeline in CI, tracked in repo (not gitignored)
- **TGV weak-scaling benchmark** — BSSCA series CSV + plot script + Marp slide
- **Purple accent** theme refresh for v0.2.0

### 📝 Content Restructure

- **Guides** — `style_guide.md` expanded (+82 lines), `building.md` updated, new `python_geom_guide.md` (comprehensive API reference for Python Geom module, 215-line rewrite)
- **Solver guide** — three new docs: `user_guide.md`, `solver_config.md`, `troubleshooting.md`
- **Theory** — Pyramid shape function diagrams, updated `Shape_Functions.md`
- **Tests** — new `geom_unit_tests.md`, renamed files for consistency (`cfv_unit_tests.md`, `euler_unit_tests.md`, `solver_unit_tests.md`)
- **Sphinx** — new API sections for C++ solver API and Python bindings, presentations page, solver-guide page
- **Contributing** — new `CONTRIBUTING.md`

### 🛠️ Development Docs

- `distributed_reorder_design.md` — v2 distributed reorder design: dynamic registry, callback companions, follow semantics
- `clang_tidy_plan.md` — complete 26-pass sanitation plan with per-pass checklist, disable rationale table, NOLINT placement guidelines
- `mesh_connectivity_impl_plan.md` — implementation plan for state-checked layer
- `mesh_helpers_design.md` — unified C++ mesh helpers design
- `mesh_refactoring_plan.md` — Geom module refactoring strategy
- `multi_layer_ghost_design.md` — multi-layer ghost algorithm design
- `InitialReport.md`, `doc_backlog.md` — documentation audit and backlog
- `audit/2026-04-26_range-13b4e7b-to-a075bb2.md` — comprehensive audit for merge range

---

## ✅ Testing

### 🧪 C++ Test Suite (29 executables, 416 test cases)

| Module | New Test Files | Lines | Coverage |
|---|---|---|---|
| **Geom** | `test_MeshConnectivity.cpp` | 821 | DAG DSL: Inverse, Compose, ComposeFiltered |
| **Geom** | `test_MeshConnectivity_Ghost.cpp` | 1,061 | Ghost chains, BFS evaluator, multi-layer |
| **Geom** | `test_MeshConnectivity_Interpolate.cpp` | 1,931 | Face/edge interpolation, multi-parent periodic |
| **Geom** | `test_MeshReorder.cpp` | 1,086 | Full reorder pipeline on real meshes |
| **Geom** | `test_MeshPipeline.cpp` | +1,383 | Expanded pipeline with ghost and reorder phases |
| **Geom** | `test_Elements.cpp` | +357 | Prism18 edge group tests |
| **Geom** | `SyntheticMeshBuilders.hpp` | 940 | 2D tiled synthetic grids with analytical formulas |
| **DNDS** | `test_PermutationTransfer.cpp` | 376 | Distributed + local row permutation |
| **CFV** | `test_Reconstruction.cpp` | +351 | Expanded reconstruction coverage |

### 🔁 Test Infrastructure

- **doctest per-case CTest registration** — individual test case names in CTest output for pinpointed failure identification
- **`OMP_NUM_THREADS=2`** default for all C++ unit tests (configurable via `DNDS_TEST_OMP_THREADS`)
- **`DNDS_TEST_NP_LIST`** environment variable for MPI test ranks at configure time
- **`ctest_summary.py`** — aggregated doctest statistics across all categories

### 🐍 Python Tests

- **Geom** — `test_basic_geom.py` expanded (+110 lines) for new mesh helpers, distributed read, elevation with bisection
- **Euler** — `test_restart_redistribute.py` updated for mesh helper migration
- **EulerP** — `test_basic_eulerP.py` & `test_solver.py` CUDA guard updates

---

## 🔧 Developer Tooling

### 🧹 Static Analysis

- **`run_clang_tidy.py`** (559 lines) — per-check histogram, `--fix` serialization, module-scoped iteration
- **`run_clang_format.py`** (341 lines) — multi-module formatting with cross-module config uniformity
- **Unified `.clang-tidy`** (178 lines) — single source of truth for CLI and IDE (clangd discovers automatically)
- **Unified `.clang-format`** — consistent brace/indent/wrap rules across all modules
- **CI gate** — advisory (not build-blocking); clang-tidy runs produce reports, not errors

### 🧬 Element Code Generation

- `traits_emitter.py` (227 lines) — generates C++ element trait source from CGNS topology data
- `element_data.py` (552 lines) — comprehensive CGNS element reference: node numbering, face maps, topology
- `gen_diagrams.py` (478 lines) — 3D element visualization with PyVista, structured edge groups, tuned viewing angles

### 🐳 Pre-commit

- Updated hook scripts for clang-tidy/format integration
- Stub generation script for pybind11 modules
- Deprecated legacy shell scripts, replaced with Python drivers

---

## 🐛 Notable Bug Fixes

### 🧵 Mesh State Tracking

- **Per-adjacency state conditions** now enforced alongside group state guards — every site that sets `adjXState` also calls `markGlobal()`/`markLocal()` on governed adjacencies
- **Target mapping re-wiring** — after `ReorderLocalCells` ghost rebuilds, stale per-adjacency target mappings now correctly re-wired
- **`MatchFaceBoundary`** — `face2bnd` now correctly converted to global before ghost pull
- **Ghost tree evaluation** — scratch-pull gating separated from output collection, fixing stale data reads
- **`faceElemInfo`** — re-pulled after `MatchBoundariesToFaces` in DSL path

### 🔁 Interpolate & Periodic BCs

- **Multi-parent handling** — fix edge and face interpolation on 2×2×2 periodic hex meshes where an entity has multiple periodic parents
- **Double-periodic BCs** — `InterpolateFace` wrong face identity on doubly-periodic boundary configurations (fixed in earlier range, carried forward)

### 🔒 Build & Runtime

- **`BuildSerialOut`** side-effects — `pLGlobalMapping` no longer silently modified during serial export
- **MPI `mpicxx.h`** — deprecated header inclusion prevented across all MPI implementations (not just MPICH)
- **`libstdc++`/`libmpi` bundling** — removed from `dndsr_external` shared library, eliminating version conflicts
- **Pybind11 LTO** — disabled by default to prevent link-time ODR violations
- **Python MPI init** — switched to `mpi4py` initialization to avoid OpenMPI singleton hang
- **Stub generation** — MPI init skipped during stub generation to prevent double-free crash in MPI_Finalize
- **`Python_EXECUTABLE`** — passed to CMake configure on self-hosted runners where `python3` ≠ target venv

### 🧪 Test Infrastructure

- **CTest registration** — uses `Python_EXECUTABLE -m pytest` to avoid PATH-dependent python resolution
- **MPI oversubscribe** — `--oversubscribe` always passed on GitHub runners for `np > cores` scenarios
- **Euler config** — `euler_default_config.json` removed from test dependency chain (was gitignored, causing CI failures)

### ⚡ Performance & Correctness

- **Array move semantics** — proper move constructors/assignments prevent accidental copies in template instantiation paths
- **`ArrayPair::clone`** — fix latent bug where clone incorrectly shared internal resources
- **SA-DES `lLES` clamping** — bypass for pure RANS mode, resolving hybridization edge case; `SAVersion` option added
- **Merge range audit** — comprehensive fixes for range 13b4e7b..HEAD (Euler SA-DES, state tracking, documentation)

---

## 📊 By the Numbers

| Metric | v0.1.0 → v0.2.0 |
|---|---|
| Commits | 155 |
| Files changed | 385 |
| Insertions | 56,227 |
| Deletions | 10,506 |
| New C++ files | 25 |
| Test cases (C++) | 416 (across 29 executables) |
| CTest registrations | 82 (np=1,2,4,8 for MPI tests) |
| Clang-tidy warnings resolved | 24,596 |
| New documentation pages | 30+ |
| Slide deck sections | 10 (×2 for Chinese) |
| CI workflow LOC | 375 |
| Mesh DSL LOC | 3,000+ |
