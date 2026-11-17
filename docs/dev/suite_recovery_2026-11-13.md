# Full-suite setup and collective admission corrections

The full run exposed two distinct communication admission defects alongside
checkout-specific missing test assets. Original failures, process snapshots
and gdb stacks are preserved under canonical
`data/out/dndsr-full-suite-20261113/` and `data/out/dndsr-suite-fix-20261113/`.
The hung driver and its exact MPI descendants were stopped before rebuilding;
no numerical campaign or unrelated GPU workload was stopped.

## Per-rank archive parsing

`ReadNativeMeshFromArchive` now agrees on per-rank JSON `OpenFile`/parse success
before entering schema collectives. A malformed rank-zero JSON previously
threw while peers entered the schema protocol. The coordinated retry fixture
could then match an unrelated agreement and strand Allreduce against Bcast.
JSON parsing is rank-local, so a local-stage guard is valid here. HDF5 opening
may itself communicate and stays outside that recoverable local-stage guard.
Production solver catches continue to terminate; this does not authorize
exception recovery around collective I/O.

## Transition descriptor admission

The existing alias fixture intentionally supplies an invalid SortedUnique row.
The serial-owned transition deferred that descriptor's validation until after
earlier moves had posted. Throwing then correctly triggered the transfer
destructor's active-MPI fail-fast policy, preventing the expected preflight
rejection. All input descriptors are now validated in the existing admission
stage before the first move. Validated input scopes are retained and reused;
the change adds no numerical algorithm or transfer recovery.

The previously unfinished wall-distance missing-authority fixture also mixed
communicators: its rank-zero empty facade used WORLD while the native ranks
used their retained preparation context. The fixture now constructs the empty
facade on that exact context communicator and keeps the original owner alive.
It still verifies collective rejection of missing native authority. No
wall-distance arithmetic or validation policy was weakened.

## Test assets and Python environment

Separate checkouts need the canonical mesh assets and populated dependencies.
For this rerun, missing entries in `data/mesh/` link to verified canonical
assets without overwriting existing files. The expected CGNS shared-library
install directory similarly shares the canonical populated install.

Build and install bindings before running Python tests. The test-only
`runtime_array_bridge_test` module also requires `<build>/test/cpp` on
`PYTHONPATH`; CTest now sets this automatically. Manual pytest uses both that
directory and `<checkout>/python`, with the canonical Python interpreter and
mechanism path. `all_unit_tests` now includes the test bridge extension as a
dependency. The restart Python fixture accepts `DNDS_BUILD_DIR` so a separate
build directory is not mistaken for a missing solver. CUDA was reconfigured with the same Python executable,
include directory and library to avoid a cached 3.14/3.12 mismatch.
`test_basic_cfv.py` now explicitly imports the `os` module it uses.

Use a finite CTest controller limit, e.g. `ctest --timeout 300`, when
configure-time per-test timeouts are unset. Timeout means failure; it is not a
successful skipped test. Rerun receipts separately enumerate failed and
unfinished members; earlier passing results are retained rather than erased.

The Python mixed-authority fixture had the same communicator mismatch as its
C++ wall-distance counterpart; it now creates its empty facade using the
native facade's communicator, with the native owner retained. The serial
single-free-cell wave fixture now applies its existing single-rank requirement
to both entry points; distributed reconstruction tests retain their coverage.

The large restart integration test performs one writer and five MPI restart
comparisons, so it receives a bounded 300-second case budget. When invoking
pytest itself under MPI, use `mpiexec --bind-to none` for the outer driver;
the child solvers control their own rank placement. The previously timed-out
case passed in approximately 3.5 seconds with this outer binding, with every
np4/5/6/7/8 comparison exactly equal. The role of binding in the timing difference
was not independently benchmarked.

## Validation

- Both builds passed archive/transition tests at np1/2/4/8 (8/8 each).
- CPU reran the 96 previously failed/unfinished cases. Its only three failures
  were the old wall-distance fixture processes; after rebuilding, wall-distance
  passed np1/2/4/8. Combined with the 395 preserved passes, all 491 CPU names
  have passing evidence. This is combined coverage, not one fresh unfiltered run.
- CUDA reran 180 cases, initially passing 177. The missing bridge and two
  interrupted wall tests were rerun successfully after their fixes (5/5 focused
  follow-ups including all wall rank counts). All 515 eligible names have
  passing evidence across the preserved runs. The separate two-GPU legacy DOF
  test passed; four/eight-GPU variants remain unsupported on two devices.
- Full Python with MPI tests enabled: np1 passed 309 with 85 skips. The np2
  run passed 308 with 85 skips and one restart timeout per rank; that remaining
  test passed in the focused rerun on both ranks with exact restart comparisons.
  The corrected helper file separately passed all 13 tests per rank.
- Initial failures, timeouts, interrupted processes, gdb stacks, exact name
  inventories and subsequent successes remain in the evidence directories.
  Skips for optional unavailable capabilities are not credited as passes.

No CFD campaign was rerun and no 00–03 result was modified by these repairs.
