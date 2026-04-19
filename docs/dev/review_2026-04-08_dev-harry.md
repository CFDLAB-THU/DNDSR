# Code Review: dev/harry (commits de53aa6..fb8c367) {#review_dev_harry}

**Date:** 2026-04-08
**Branch:** dev/harry
**Reviewer:** Automated review
**Commits reviewed:** 7 commits from `de53aa6` to `fb8c367` (after merge `fd9d2fc`)
**Scope:** ~11,300 lines added, ~2,400 lines removed across 221 files

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Commit-by-Commit Overview](#2-commit-by-commit-overview)
3. [Feature: Array Redistribution & HDF5 Checkpoint](#3-feature-array-redistribution--hdf5-checkpoint)
4. [Feature: C++ Unit Tests (doctest)](#4-feature-c-unit-tests-doctest)
5. [Refactor: CMake Modularization](#5-refactor-cmake-modularization)
6. [Refactor: Python Package Restructure](#6-refactor-python-package-restructure)
7. [Refactor: Script Directory Rename](#7-refactor-script-directory-rename)
8. [Bug Fixes & Minor Changes](#8-bug-fixes--minor-changes)
9. [Issues Found](#9-issues-found)
10. [Recommendations](#10-recommendations)

---

## 1. Executive Summary

This review covers a substantial series of commits on `dev/harry` that introduce:

- **Array redistribution for HDF5 checkpoints** — the marquee feature: read checkpoint data written with `np=N` using `np=M`, enabling flexible restart across different MPI decomposition. This is a well-designed feature with comprehensive C++ and Python test coverage.
- **Comprehensive C++ unit tests** — 7 doctest-based test files (~4,300 lines) covering Array, ArrayTransformer, ArrayDerived, ArrayDOF, IndexMapping, MPI, and Serializer, with parametric MPI rank counts (np=1,2,4,8).
- **CMake modularization** — the 1,100-line monolithic `CMakeLists.txt` split into 9 focused modules under `cmake/`, plus `CMakePresets.json`.
- **Python package restructure** — package moved from `src/` to `python/DNDSR/`, consolidated `_loader.py`, auto stub generation, `.pyi` files removed from git tracking.
- **Script directory rename** — `script/` → `scripts/`.
- **Documentation** — new `docs/Serialization.md`, `docs/building.md`, `docs/project_structure.md`, and extensive Doxygen annotations.

The overall quality is high. The redistribution feature is the most complex addition and is generally well-implemented with a sound rendezvous pattern, but there are several issues and improvement opportunities detailed below.

---

## 2. Commit-by-Commit Overview

| Commit | Message | Size | Risk |
|--------|---------|------|------|
| `a54e587` | recording configs | 134+/59- | Low — VS Code, case configs |
| `de53aa6` | C++ unit tests, build system, Doxygen | +5,152/-73 | Medium — build system changes |
| `e95aaac` | separate Python package into python/ | +152,292/-1,202 | High — massive restructure, 73K-line .pyi files |
| `4a2c843` | modular CMake, auto stubgen, CUDA 13.1 | +1,508/-152,153 | High — CMake rewrite, .pyi removal |
| `903d23a` | script path change; compdb use shipped | +305/-86 | Low — directory rename |
| `62e90b5` | redistributable Array serialization (HDF5) | +3,223/-129 | High — core serialization changes |
| `fb8c367` | Doxyfile template path; numpy compat | +10/-10 | Low — minor fixes |

---

## 3. Feature: Array Redistribution & HDF5 Checkpoint

### 3.1 Architecture

The redistribution system is built in layers:

1. **`EvenSplitRange`** (`Defines.hpp:114-119`) — utility for computing fair partition ranges when `nGlobal < nRanks`.
2. **`ArrayGlobalOffset_EvenSplit`** (`SerializerBase.hpp:85`) — new offset type for "each rank reads ~N/nRanks rows."
3. **`BuildRedistributionPullingIndex`** (`ArrayRedistributor.hpp:68-168`) — three-round MPI rendezvous via `Alltoallv` to build a global directory mapping `origIdx → globalReadIdx`.
4. **`RedistributeArrayWithTransformer`** (`ArrayRedistributor.hpp:193-289`) — uses `ArrayTransformer::pullOnce()` to fetch data, then copies into output array.
5. **`ArrayPair::ReadSerializeRedistributed`** (`ArrayPair.hpp:401-533`) — top-level orchestrator handling same-np and different-np cases.
6. **Euler solver integration** (`EulerSolver_PrintData.hxx`) — `PrintRestart` writes `origIndex`, `ReadRestart`/`ReadRestartOtherSolver` use `ReadSerializeRedistributed`.

### 3.2 Strengths

- **Sound rendezvous design**: The 3-round `Alltoallv` pattern (send entries → build directory → send queries → reply) is correct and avoids the need for a central coordinator. The block-partitioned directory rank assignment (`origIdx * nRanks / nGlobal`) provides good locality.
- **Zero-size partition handling**: The fix for `nGlobal < nRanks` is thorough — every `Read*Vector` caller passes a dummy non-null pointer when `size==0`, ensuring all ranks participate in collective `H5Dread`. This is a subtle but critical fix.
- **`pullingIndexGlobal` sorting warning**: The `createGhostMapping` documentation now explicitly warns that the input vector is sorted in-place. The redistribution code correctly saves a copy (`pullingIndexOrig`) before the sort.
- **CSR support**: The redistribution handles CSR layout by setting output row sizes from pulled son data before copying element values. This is non-trivial and appears correct.
- **Comprehensive test coverage**: `test_Serializer.cpp` (574 lines added), `test/Euler/test_restart_redistribute.py` (403 lines), and the parametric tests across rank counts are excellent.

### 3.3 Issues

#### 3.3.1 `BuildRedistributionPullingIndex` — Memory overhead and scalability

**Severity:** Medium

The function builds three `unordered_map` objects (`directoryMap`, `globalIdx2SonPos`) and multiple temporary vectors (`sendBuf`, `recvBuf`, `querySendBuf`, `queryRecvBuf`, `queryReplyBuf`, `replyRecvBuf`). For a large mesh (e.g., 100M cells), the memory footprint is significant:

- `directoryMap`: O(nGlobal/nRanks) entries per rank (rendezvous partition).
- `sendBuf` + `recvBuf`: 2 × O(nLocal) `index` values each.
- `querySendBuf` + `queryRecvBuf` + `queryReplyBuf` + `replyRecvBuf`: 4 × O(nLocal) `index` values.

**Recommendation:** Consider adding a `reserve()` hint for `directoryMap` based on `recvDisps[nRanks]` (already done) and documenting the expected memory scaling. For very large meshes, an out-of-core or chunked approach may be needed.

#### 3.3.2 `RedistributeArrayWithTransformer` — Unnecessary global mapping on read father

**Severity:** Low

Line 215: `readFather->createGlobalMapping()` creates a `GlobalOffsetsMapping` on the read father. This is needed for `BuildRedistributionPullingIndex` and `ArrayTransformer::createFatherGlobalMapping`. However, `readFather` is a temporary that is discarded after redistribution. If `readFather` already had a global mapping (e.g., from `ParArray::ReadSerializer`), this is redundant.

**Recommendation:** Document that `readFather` must not have a pre-existing global mapping, or check and reuse it.

#### 3.3.3 `ArrayPair::ReadSerializeRedistributed` — `ReadIndexVector` with `EvenSplit` for `origIndex`

**Severity:** Low

Lines 483-485: The `origIndex` is read with `ArrayGlobalOffset_EvenSplit`, which works correctly. However, the `ReadIndexVector` API requires a two-pass pattern (nullptr query, then data read). The even-split logic in `SerializerH5.cpp` handles this, but the interaction between `EvenSplit` and the `size` query is subtle. If the first pass sets `offset` to `isDist`, the second pass must use that resolved offset — which it does, but this is fragile.

**Recommendation:** Add an assertion after the first `ReadIndexVector` call that `offsetOrigIdx.isDist()`, to catch future breakage.

#### 3.3.4 `Array::ReadSerializer` — `dataOffset` parameter complexity

**Severity:** Medium

The `ReadSerializer` signature now takes `offset` and `dataOffset` as in/out parameters. The logic for resolving these is spread across `ParArray::ReadSerializer` (which resolves `EvenSplit` and CSR offsets via `MPI_Scan`) and `Array::ReadSerializer` (which resolves `dataOffset` from `pRowStart` and `DataStride`). The flow is:

1. `ParArray::ReadSerializer` resolves `offset` from `EvenSplit` → `isDist`.
2. `Array::ReadSerializer` reads structural data, computes `dataOffset`.
3. `Array::__ReadSerializerData` reads flat data using `dataOffset`.
4. `dataOffset` is propagated back to the caller.

This is correct but hard to follow. The relationship between `offset` (row-level) and `dataOffset` (element-level) is documented in the docstring, but the code itself has many conditional branches.

**Recommendation:** Consider extracting the offset resolution logic into a helper method (e.g., `resolveDataOffset`) to reduce the cognitive load.

#### 3.3.5 `Array::DataStride()` — CSR assertion

**Severity:** Low

`DataStride()` asserts with a message "not valid for CSR." This is good — it prevents misuse. However, the method is called from `ReadSerializer` only when `_dataLayout != CSR`, so the assertion is unreachable in practice. Consider making it `static_assert` or removing it.

#### 3.3.6 `EulerSolver_PrintData.hxx::ReadRestartOtherSolver` — `storedRowDynamic` peek

**Severity:** Medium

Lines 1039-1050: The code peeks at `array_sig` and `row_size_dynamic` from the H5 file to determine the stored `nVars`, then resizes `readBuf.father` before calling `ReadSerializeRedistributed`. This is fragile because:

- It reaches into the H5 path structure (`u/father/array/`) directly, which is an implementation detail of the serializer.
- If the array signature format changes, the peek will silently fail or produce incorrect results.
- The `Resize(mesh->NumCell(), storedRowDynamic, 1)` call may not be correct for all `ArrayDOF` variants.

**Recommendation:** Add a dedicated `PeekArraySignature` method to the serializer that returns the signature without fully reading the data. Or at minimum, add a comment explaining the H5 path structure dependency.

#### 3.3.7 `EulerSolver_PrintData.hxx::ReadRestart` — Missing `u.trans.startPersistentPull()` in H5 path

**Severity:** **High**

Line 987 (H5 path): After `u.ReadSerializeRedistributed()`, the code does NOT call `u.trans.startPersistentPull()` / `u.trans.waitPersistentPull()` before returning. The JSON path also skips this, but the original code (before this commit) had it after `paste_read_restart_with_cell_ordering`. Looking at the diff more carefully, the `startPersistentPull/waitPersistentPull` calls ARE present after the `if/else` block at line 989 in the new code — they were moved to after the conditional. This is correct. **On closer inspection, this is fine.**

---

## 4. Feature: C++ Unit Tests (doctest)

### 4.1 Strengths

- **Parametric test design**: Tests are parameterized over data types (`real`, `index`, `rowsize`), layouts (StaticFixed, Fixed, CSR, Max, StaticMax), and MPI rank counts (1, 2, 4, 8). This is excellent for catching layout-specific and rank-specific bugs.
- **Comprehensive coverage**: The `test_Serializer.cpp` file alone is 574 new lines (1,270 total) covering JSON and H5 round-trips, distributed reads, and redistribution.
- **CTest integration**: MPI tests are registered at np=1, 2, 4, and 8 with `PROCESSORS` property set correctly.

### 4.2 Issues

#### 4.2.1 Test naming: `dnds_` prefix collision with CTest

**Severity:** Low

The test names are `dnds_array`, `dnds_serializer_np1`, etc. The `ctest -R dnds_` filter in `AGENTS.md` works, but if other targets are added with the `dnds_` prefix, they will be inadvertently included.

**Recommendation:** Consider using a more specific prefix like `dnds_ut_` (unit test) to distinguish from integration tests.

#### 4.2.2 `EXCLUDE_FROM_ALL` on test executables

**Severity:** Low

Test executables have `EXCLUDE_FROM_ALL ON`, meaning they won't be built by a plain `cmake --build .`. They require `cmake --build . -t dnds_unit_tests`. This is intentional (avoids building tests in production), but should be documented more prominently in `AGENTS.md`.

---

## 5. Refactor: CMake Modularization

### 5.1 Strengths

- **Clean separation**: The monolithic 1,100-line `CMakeLists.txt` is now ~100 lines, with 9 focused modules: `DndsStdlibSetup`, `DndsOptions`, `DndsCudaSetup`, `DndsCompilerFlags`, `DndsExternalDeps`, `DndsTests`, `DndsApps`, `DndsDocs`, `DndsTooling`.
- **CMakePresets.json**: Five configure presets (`default`, `debug`, `release-test`, `cuda`, `ci`) with build and test presets make the build accessible to new contributors.
- **No logic changes**: The modularization preserves existing behavior; only structure changed.

### 5.2 Issues

#### 5.2.1 `cmake/DndsCompilerFlags.cmake` — Typos in warning messages

**Severity:** Low

Multiple instances of `complier` should be `compiler`:
- Line ~165: `message(WARNING "${CMAKE_CXX_COMPILER_ID} complier not using DNDS_USE_PARALLEL_MACRO")`
- Line ~172: Same
- Line ~185: Same
- Line ~195: Same

#### 5.2.2 `cmake/DndsCudaSetup.cmake` — CUDA 13.1 CCCL path detection

**Severity:** Low

The code detects `${CUDAToolkit_INCLUDE_DIRS}/cccl` and appends it to `DNDS_EXTERNAL_INCLUDES`. This is backward-compatible (path won't exist on CUDA 12.x). However, it uses a filesystem path check at configure time, which may not work with cross-compilation or when CUDA is in a non-standard location.

**Recommendation:** Add a comment explaining the CUDA version dependency.

#### 5.2.3 `--disable-new-dtags` linker flag

**Severity:** Medium

`cmake/DndsCompilerFlags.cmake` adds `-Wl,--disable-new-dtags` globally. This changes the dynamic linker behavior from RUNPATH (search after dependent libraries) to RPATH (search before). While this fixes the conda/libstdc++ conflict, it has security implications: RPATH allows a library's own directories to override `LD_LIBRARY_PATH`, which could be used for library injection attacks in multi-user environments.

**Recommendation:** Document the security trade-off. Consider making this configurable via a CMake option.

---

## 6. Refactor: Python Package Restructure

### 6.1 Strengths

- **Consolidated `_loader.py`**: Replaces four duplicated `_pre_import()` functions with a single `preload()` function. The GLIBCXX error message is helpful.
- **PEP 561 compliance**: `py.typed` marker and `.pyi` stubs enable type checking.
- **Auto stub generation**: `scripts/generate-stubs.sh` is called automatically during `cmake --install`, removing `.pyi` files from git tracking.
- **`conftest.py`**: Root conftest adds `python/` to `sys.path`, enabling `pytest test/` without `pip install -e .`.

### 6.2 Issues

#### 6.2.1 `_loader.py` — `RTLD_DEEPBIND` may cause issues

**Severity:** Medium

`_load_so` uses `os.RTLD_DEEPBIND` when available. This flag makes the loaded library prefer its own dependencies over already-loaded ones, which helps with the conda libstdc++ conflict. However:

- `RTLD_DEEPBIND` is Linux-only and not available on macOS or some container environments.
- It can cause issues with libraries that expect to share symbols (e.g., MPI libraries that use `dlopen`).
- The `getattr(os, "RTLD_DEEPBIND", 0)` fallback silently disables it on unsupported platforms.

**Recommendation:** Add a comment explaining the trade-offs. Consider adding a `DNDSR_NO_DEEPBIND` environment variable override for debugging.

#### 6.2.2 `_loader.py` — Hard-coded `.so` suffix

**Severity:** Low

All library names are hard-coded with `.so` suffix (e.g., `libdnds_shared.so`). This won't work on macOS (`.dylib`) or Windows (`.dll`). The `preload()` function has an early return for `os.name != "posix"`, which skips the entire mechanism on Windows, but macOS is POSIX and will attempt to load `.so` files.

**Recommendation:** Use `sysconfig.get_config_var('EXT_SUFFIX')` or platform detection to handle macOS.

#### 6.2.3 `_loader.py` — Missing `libhdf5` version matching

**Severity:** Low

The external libs list includes `libhdf5.so` but not `libhdf5_hl.so` or `libhdf5_fortran.so`. Some HDF5 installations may require these additional libraries. If they are not loaded, the CGNS library (which depends on HDF5) may fail to find symbols.

**Recommendation:** Add `libhdf5_hl.so` to `_EXTERNAL_LIBS` if CGNS links against it.

#### 6.2.4 `python/DNDSR/DNDS/__init__.py` — `_init_mpi()` called at import time

**Severity:** Medium

`_init_mpi()` calls `MPI.Init_thread(sys.argv)` and registers `MPI.Finalize` via `atexit`. This means importing `DNDSR.DNDS` always initializes MPI, even if the user only needs the array types without MPI. This can cause issues in environments where MPI initialization is not desired (e.g., Jupyter notebooks, documentation builds, or when the user wants to control MPI initialization themselves).

**Recommendation:** Consider lazy initialization — only call `_init_mpi()` when an `MPIInfo` object is created or when the user explicitly calls it. Alternatively, check if MPI is already initialized before calling `Init_thread`.

#### 6.2.5 `python/DNDSR/DNDS/__init__.py` — Factory function naming conflicts

**Severity:** ~~Medium~~ False alarm (resolved)

Investigation confirmed that pybind11 exports names like `Array_d_1_1_N`, `ParArray_d_I_I_N`, etc. — never a bare `Array`. The Python factory `def Array(...)` does NOT shadow any C++ class name. No fix needed.

#### 6.2.6 `_row_size_to_name` — Inconsistent mapping for `None`

**Severity:** Low

`_row_size_to_name(None)` returns `"None"`, but the pybind11 bindings use `"N"` for `NoAlign` and `"I"` for `NonUniformSize`. The `"None"` string would fail to match any pybind11 class name. This may be dead code (no caller passes `None`), but it's misleading.

---

## 7. Refactor: Script Directory Rename

### 7.1 Assessment

The `script/` → `scripts/` rename is clean. All 100 files are simple `git mv` operations with no content changes. The `cmake/DndsTooling.cmake` and `docs/building.md` references are updated accordingly.

**No issues found.**

---

## 8. Bug Fixes & Minor Changes

### 8.1 Empty-partition hang fix (`SerializerH5.cpp`)

**Critical fix.** Previously, ranks with 0 local elements would skip the collective `H5Dread` call (because `v.data()` returned `nullptr`), causing other ranks to hang indefinitely. The fix passes a dummy stack variable (`uint8_t dummy{}`) when `size == 0`.

This pattern is applied consistently to all 6 `Read*Vector`/`ReadShared*Vector` callers in `SerializerH5.cpp` and to `Array::__ReadSerializerData`. This is correct and well-documented in the `SerializerBase` docstring.

### 8.2 `test_basic.py` — `np.concat` → `np.concatenate`

The `np.concat` function was introduced in NumPy 2.0 and is not available in older versions. The fix changes it to `np.concatenate`, which is available in all NumPy versions.

### 8.3 `_Unknown` → `_N` alignment suffix

The `Align_To_PySnippet()` function now returns `"N"` for `NoAlign` instead of `"Unknown"`, making the pybind11 class names more readable (e.g., `Array_d_I_I_N` instead of `Array_d_I_I_Unknown`). All test files are updated accordingly.

### 8.4 `ArrayAdjacency_bind.hpp` etc. — Updated alignment suffix

The bind files for `ArrayAdjacency`, `ArrayEigenMatrix`, `ArrayEigenVector`, and `Array` all use `Align_To_PySnippet(_align)` instead of `RowSize_To_PySnippet(_align)`. This is correct because alignment is semantically different from row size.

### 8.5 `Defines_bind.hpp` — Conditional `#include <omp.h>`

The `#include <omp.h>` is now guarded by `#ifdef DNDS_USE_OMP`, fixing compilation when OpenMP is disabled. This was a latent bug.

---

## 9. Issues Found

### Critical

None found.

### High

| # | File | Issue |
|---|------|-------|
| 1 | `EulerSolver_PrintData.hxx:1039-1050` | Fragile H5 path peek for `storedRowDynamic` — depends on internal serializer layout |

### Medium

| # | File | Issue |
|---|------|-------|
| 2 | `ArrayRedistributor.hpp:68-168` | Memory overhead of rendezvous pattern for large meshes (3+ temporary buffers of O(nLocal)) |
| 3 | `Array.hpp:754-857` | `ReadSerializer` offset/dataOffset resolution logic is complex and hard to follow |
| 4 | `cmake/DndsCompilerFlags.cmake` | `--disable-new-dtags` has security implications (RPATH vs RUNPATH) |
| 5 | `_loader.py:91` | `RTLD_DEEPBIND` may cause symbol resolution issues with MPI |
| 6 | `python/DNDSR/DNDS/__init__.py:64` | `MPI.Init_thread()` called at import time, preventing user control |
| 7 | `python/DNDSR/DNDS/__init__.py` | Factory functions (`Array()`, etc.) shadow C++ class names from `import *` |

### Low

| # | File | Issue |
|---|------|-------|
| 8 | `cmake/DndsCompilerFlags.cmake` | Typo: `complier` → `compiler` (4 instances) |
| 9 | `_loader.py` | Hard-coded `.so` suffix won't work on macOS |
| 10 | `_loader.py` | Missing `libhdf5_hl.so` in external libs |
| 11 | `Array.hpp:143` | `DataStride()` CSR assertion is unreachable |
| 12 | `__init__.py:_row_size_to_name` | `None` → `"None"` mapping may be dead code |
| 13 | `test/cpp/CMakeLists.txt` | `EXCLUDE_FROM_ALL` requires explicit target build, not documented prominently |

---

## 10. Recommendations

### Architecture

1. **Extract offset resolution** in `Array::ReadSerializer` into a helper method to reduce complexity.
2. **Add a `PeekArraySignature`** method to the serializer API to avoid reaching into H5 path internals from the Euler solver.
3. **Consider lazy MPI initialization** in the Python package to support non-MPI use cases.

### Performance

4. **Profile the redistribution** on large meshes (>10M cells) to verify the memory overhead is acceptable. Consider chunked rendezvous for very large cases.
5. **Document the expected memory scaling** of `BuildRedistributionPullingIndex` in the code comments.

### Robustness

6. **Add `RTLD_DEEPBIND` escape hatch** via environment variable (`DNDSR_NO_DEEPBIND`).
7. **Fix macOS library loading** in `_loader.py` — use platform-appropriate suffixes.
8. **Add `libhdf5_hl.so`** to the external libs list in `_loader.py`.

### Code Quality

9. **Fix typos** in `cmake/DndsCompilerFlags.cmake` (`complier` → `compiler`).
10. **Consider renaming Python factory functions** to avoid shadowing C++ names (e.g., `make_array()`).
11. **Remove or document dead code** in `_row_size_to_name(None)` path.

### Documentation

12. **Document the `--disable-new-dtags` security trade-off** in `cmake/DndsCompilerFlags.cmake`.
13. **Document the `EXCLUDE_FROM_ALL`** behavior for test targets in `AGENTS.md`.
14. **Add a comment** in `cmake/DndsCudaSetup.cmake` explaining the CUDA 13.1 / CCCL dependency.

---

## Appendix A: File Change Summary

### Core C++ (highest-impact changes)

| File | Lines Changed | Description |
|------|--------------|-------------|
| `src/DNDS/ArrayRedistributor.hpp` | +289 | **New**: Redistribution rendezvous + pull |
| `src/DNDS/Array.hpp` | +197/-27 | Serialization protocol, DataStride, CopyRowFrom, ResizeRowsAndCompress |
| `src/DNDS/ArrayPair.hpp` | +184/-9 | WriteSerialize with origIndex, ReadSerializeRedistributed |
| `src/DNDS/ArrayTransformer.hpp` | +149/-6 | ParArray WriteSerializer/ReadSerializer, docstring |
| `src/DNDS/SerializerBase.hpp` | +52/-7 | ArrayGlobalOffset_EvenSplit, docstrings, GetMPIRank/Size |
| `src/DNDS/SerializerH5.cpp` | +61/-17 | EvenSplit read, zero-size partition fix |
| `src/DNDS/SerializerH5.hpp` | +22 | GetMPIRank/Size, docstrings |
| `src/DNDS/Defines.hpp` | +38/-6 | EvenSplitRange, NoAlign, Align_To_PySnippet |
| `src/Euler/EulerSolver_PrintData.hxx` | +105/-20 | PrintRestart origIndex, ReadRestart/ReadRestartOtherSolver redistribution |

### Python Package

| File | Lines Changed | Description |
|------|--------------|-------------|
| `python/DNDSR/_loader.py` | +132 | **New**: Consolidated native library preloader |
| `python/DNDSR/__init__.py` | +17 | **New**: Package root |
| `python/DNDSR/DNDS/__init__.py` | +221 | **New**: Factory functions, MPI init |
| `python/DNDSR/Geom/__init__.py` | +12 | **New**: Module init |
| `python/DNDSR/CFV/__init__.py` | +11 | **New**: Module init |
| `python/DNDSR/EulerP/__init__.py` | +11 | **New**: Module init |

### C++ Unit Tests

| File | Lines | Description |
|------|-------|-------------|
| `test/cpp/DNDS/test_Array.cpp` | 744 | All 5 layouts, resize, compress, clone, JSON serialization |
| `test/cpp/DNDS/test_Serializer.cpp` | 1,270 | JSON + H5 round-trips, redistribution (same + different np) |
| `test/cpp/DNDS/test_ArrayDerived.cpp` | 1,046 | Adjacency, EigenVector, EigenMatrix, ghost comm |
| `test/cpp/DNDS/test_ArrayTransformer.cpp` | 669 | Pull/push ghost, persistent pull, BorrowGGIndexing |
| `test/cpp/DNDS/test_IndexMapping.cpp` | 629 | GlobalOffsetsMapping, OffsetAscendIndexMapping |
| `test/cpp/DNDS/test_MPI.cpp` | 439 | MPIInfo, Allreduce, Scan, Bcast, types |
| `test/cpp/DNDS/test_ArrayDOF.cpp` | 691 | setConstant, norm2, dot, CUDA |

### Python Tests

| File | Lines | Description |
|------|-------|-------------|
| `test/Euler/test_restart_redistribute.py` | 403 | End-to-end redistribution: np=2→2, np=2→3, np=4→4..8 |
| `test/conftest.py` | 19 | **New**: Root conftest for PYTHONPATH |
| `test/DNDS/test_basic.py` | +5/-11 | Remove sys.path hack, fix `_N` suffix, `np.concatenate` |
| `test/DNDS/test_arraydof.py` | +12/-8 | `pytest.importorskip("cupy")`, `_N` suffix |

### Build System

| File | Lines | Description |
|------|-------|-------------|
| `CMakeLists.txt` | 1,059→103 | Monolithic → 9 modular cmake/ files |
| `cmake/DndsApps.cmake` | +163 | **New**: Application targets |
| `cmake/DndsCompilerFlags.cmake` | +230 | **New**: Flags, warnings, OpenMP |
| `cmake/DndsCudaSetup.cmake` | +48 | **New**: CUDA + CCCL detection |
| `cmake/DndsExternalDeps.cmake` | +438 | **New**: External dependency management |
| `cmake/DndsOptions.cmake` | +86 | **New**: CMake option definitions |
| `cmake/DndsStdlibSetup.cmake` | +93 | **New**: stdlib detection and bundling |
| `cmake/DndsTests.cmake` | +26 | **New**: Test registration |
| `cmake/DndsDocs.cmake` | +33 | **New**: Doxygen |
| `cmake/DndsTooling.cmake` | +89 | **New**: ccache, stubgen, compdb |
| `CMakePresets.json` | +93 | **New**: Configure/build/test presets |

### Documentation

| File | Lines | Description |
|------|-------|-------------|
| `docs/Serialization.md` | +174 | **New**: Serialization subsystem documentation |
| `docs/building.md` | +416 | **New**: Build instructions |
| `docs/project_structure.md` | +157 | **New**: Directory tree, module dependencies |
| `docs/TODO.md` | +384 | Expanded with concrete improvement items |
| `docs/main.dox` | +24/-4 | Updated links, \@mainpage |

### Removed

| File | Lines | Description |
|------|-------|-------------|
| `python/DNDSR/DNDS/__init__.pyi` | -36,381 | Auto-generated stubs (now generated at install time) |
| `stubs/DNDSR/DNDS/__init__.pyi` | -36,381 | Duplicate stubs removed from tracking |
| `src/DNDS/__init__.py` | -347 | Old Python package init |
| `src/Geom/GeomUtils.py` | -214 | Stale duplicate |
| `src/check.py` | -168 | Obsolete utility |

---

## Appendix B: Fixes Applied (2026-04-08)

| Issue | Fix | Files Changed |
|-------|-----|---------------|
| HIGH #1: Fragile H5 path peek | Added `Array::ReadSerializerMeta()` with overridable result struct; `ArrayEigenMatrix` and `ArrayEigenUniMatrixBatch` override to add derived-type metadata and handle the `"array"` sub-path; Euler solver uses `readBuf.father->ReadSerializerMeta(serializerP, "u/father")` instead of raw path navigation | `src/DNDS/Array.hpp`, `src/DNDS/ArrayDerived/ArrayEigenMatrix.hpp`, `src/DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp`, `src/Euler/EulerSolver_PrintData.hxx` |
| MEDIUM #2: Memory scaling undocumented | Added `@par Memory scaling` section to `BuildRedistributionPullingIndex` docstring with concrete O(nLocal) formula and example numbers for 100M-cell mesh | `src/DNDS/ArrayRedistributor.hpp` |
| MEDIUM #3: Complex offset resolution | Extracted `__ReadSerializerStructuralAndResolveDataOffset` and `__ReadSerializerDataAndPropagateOffset` helpers from `ReadSerializer`; `ReadSerializer` now calls them as 3 clear phases. All 25 C++ unit tests pass. | `src/DNDS/Array.hpp` |
| MEDIUM #4: `--disable-new-dtags` rationale | Expanded comment explaining that RPATH is safer than LD_LIBRARY_PATH for this use case (package-controlled directories vs. environment variable), not a security concern | `cmake/DndsCompilerFlags.cmake` |
| MEDIUM #5: `RTLD_DEEPBIND` risks | Added `DNDSR_NO_DEEPBIND` environment variable override; documented the trade-off and escape hatch inline | `python/DNDSR/_loader.py` |
| MEDIUM #6: MPI init at import time | Added docstring to `_init_mpi()` explaining singleton mode behavior and mpi4py conflict; noted that users should import mpi4py first if they need to control MPI thread level | `python/DNDSR/DNDS/__init__.py` |
| MEDIUM #7: Factory function shadowing | Verified false alarm — no C++ class is named `Array` (all have parameter suffixes like `Array_d_1_1_N`). No fix needed. | — |
