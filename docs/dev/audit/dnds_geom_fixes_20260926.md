# DNDS/Geom audit repairs — 2026-09-26

Ten findings repaired in seven sequential implementation commits on
`fix/dnds-geom-audit-20260926`, based on `dev/harry` at `ca518f83`.
Validated implementation tip: `24d435da5a35deb560c012a44bd9d0b2f487665c`.
This follow-up is documentation only; implementation and tests are unchanged.

The independent review repository is
`/home/harry/projects/DNDSR/workspace/review_gpt6astra_2026-09-26`, committed at
`276a226`. Its numbered findings remain the reference for this batch.
Work was performed in the separate `workspace/fix_dnds_geom_2026-09-26`
worktree, with external dependencies linked for reuse. No commits were made
on `dev/harry`, and the original checkout remained clean at `b7cb11e7`.

## Repairs and C++ reproductions

1. Finding 7: CSR hashing now reads its row-offset vector (`e35f9f71`).
   Regression checks copied, empty and differently partitioned CSR arrays.
2. Finding 18: Host-backend mirroring republishes the current allocation after
   creating its alias (`e35f9f71`). Tests cover remirroring, resize and clear.
3. Finding 19: transformer copies retain pull/push backend selection
   (`511b44e4`). Tests cover copy construction and assignment, both directions,
   default/Host backends and empty communication graphs.
4. Finding 21: persistent reinitialization permits absent directions and
   retains the selected backend without duplicate waits (`511b44e4`).
   Tests cover uninitialized and repeatedly reinitialized partial setups.
5. Finding 13: dynamic difference norms retain component shape on empty ranks
   (`da96ffb5`). Tests include wholly empty and mixed partitions.
6. Finding 9: periodic row rvalue assignment copies values into the destination
   view rather than rebinding it (`1874a784`). Tests check destination identity.
7. Finding 22: OpenFOAM block-comment scanning handles empty comments and
   overlapping stars (`1874a784`). Tests check the following token is retained.
8. Finding 1: CGNS boundary PointList and PointRange inputs receive their own
   index/cardinality checks (`0784e604`). A generated one-cell fixture exercises
   a single-entry PointList; the reader-instrumented baseline fails, and the
   repaired memory safety regression passes all 16 assertions.
9. Finding 4: cell reordering also transfers periodic face companion rows and
   refreshes their ghost mapping (`77e78bb2`). Tests compare owned and ghost
   rows by original cell identity after a nontrivial periodic-mesh permutation.
10. Finding 6: matrix row-count metadata detaches before clone mutation
    (`24d435da`). Tests cover independent CSR clones and padded-row assignment.

Public signatures are unchanged. Production edits are confined to nine files;
the repository pre-commit hook also reformatted touched legacy test files.
Larger ownership, layout, lifetime safety and CUDA work remains outside this
batch. No claim is made that all review findings have been repaired.

## Verification

The Release CPU build completed all unit-test targets, the four Python binding
targets and `euler`; Python components were freshly installed before testing.
All reported results below use the implementation tip above.

| Gate | Result |
| --- | --- |
| Full CTest, serial scheduling, MPI np=1,2,4,8 | 90/90 passed, 315.49 s |
| Whole Python `test/` suite | 49 passed, 1 CUDA-only skip, 20.95 s |
| Focused C++ regressions | All 25 serial/MPI runs passed, OMP=1 |
| Instrumented CGNS reader regression | 1 case, 16 assertions passed |
| Periodic Euler smoke, np=2, OMP=1 | Exit 0, two physical steps, 0.516 s |

CTest used OMP=2 and at most 8 MPI ranks, within the 16-core limit. Builds used
8 jobs; BLAS threads were limited to one. No CUDA validation is claimed.
The initial CTest run passed 89/90: one existing Euler test could not find
its JSON case from the nested build directory. The local untracked link
`build/cases -> ../cases` resolved this without source changes. The entire
suite was then rerun without exclusions. `build/app -> fix-cpu-ninja/app`
supplied the existing Python restart tests' executable path.

The numerical-experiment skill governed the bounded smoke and evidence layout.
The 100-cell periodic Euler case exercises CGNS input, cell reordering, periodic
metadata, DOF operations and halo exchange. It completed to `t=0.0002` within
the 120-second wall limit. Output fields were finite, density and pressure
positive, and the equal-cell density sum changed by relative `2.22e-16`.
The eight-inner-iteration cap prevented convergence to the requested residual
threshold in some stages: this is an execution smoke, not an accuracy,
convergence or performance result. Dedicated unit tests cover the corner
cases and alternative paths not reached by this solver smoke.

## Evidence and reproduction

Full local evidence is retained in the independent workspace
`workspace/audit_fix_validation/` inside the fix worktree. Start with its
`RESULTS.md`, `PLAN.md`, `final_state.json`, `final_focused/results.json`,
`logs/full/ctest_rerun.xml`, `logs/full/pytest_all.xml` and
`smoke/{manifest,checks}.json`. Raw solver output is in `data/audit_fix_smoke`.
The initial failures and superseded test-setup attempts are retained and
explicitly distinguished from authoritative results in `RESULTS.md`.

Run from the fix worktree root with its linked dependencies and mesh fixtures:

```bash
cmake --build build/fix-cpu-ninja --target all_unit_tests \
  dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 euler -j8
cmake --install build/fix-cpu-ninja --component py
export TMPDIR="$PWD/workspace/audit_fix_validation/tmp"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
timeout 1800 ctest --test-dir build/fix-cpu-ninja --output-on-failure -j1
PYTHONPATH="$PWD/python" timeout 1800 venv/bin/python -m pytest test/ --timeout=300
venv/bin/python workspace/audit_fix_validation/run_focused.py new_focused_run
```

Exact configure/build/test commands are in `logs/full/*_command.txt`.
The smoke manifest contains the executable/config/mesh hashes and invocation;
its runner deliberately refuses to replace an existing result directory.
