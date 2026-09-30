# DNDS/Geom audit repairs: second batch

Ten additional findings from the independent review at commit `276a226` are
repaired on `fix/dnds-geom-audit-20260926`. Validated implementation tip:
`675392e63659562edf7f9cea224950d57d199209`. This document adds no source changes.
The starting tip `27de9019` was pushed before this batch. Work remained in
`workspace/fix_dnds_geom_2026-09-26`, with linked external dependencies.
The original checkout stayed clean at `b7cb11e7`; `dev/harry` stayed at `ca518f83`.

## Repairs and C++ evidence

1. Finding 2: shared-read caches own typed pointers instead of addresses of
   caller variables (`f93f7966`). JSON/HDF5 tests cover rebind/reset, both integer
   vector types, reopen and cached-type mismatch.
2. Finding 3: HDF5 cache entries distinguish resolved regions and local offsets;
   cache decisions preserve collective participation; CSR consumers detach
   global row starts before normalization (`f93f7966`). Tests cover slices,
   unknown offsets, asymmetric hits, empty local results and repeated CSR reads.
3. Finding 5: padded in-situ exchanges use storage stride consistently in both
   directions, but copy only active values into initialized transport padding
   (`a0e6251d`). Tests cover five layouts, sparse peers and zero-length rows.
4. Finding 8: moved arrays reset structural dimensions; vectors lazily recreate
   missing storage managers for ordinary reuse (`c062c22c`). Moves stay noexcept;
   tests cover construction, assignment, self-move, clear, copy and reuse.
5. Finding 10: empty in-situ graphs, empty peer spans and repeated completion
   calls no longer index absent buffers (`a0e6251d`). Repeated push/wait/clear
   and zero-payload controls are included.
6. Finding 11: integer config conversion validates numeric representability and
   integrality before narrowing in both builder APIs (`0e591bee`). Follow-up
   `675392e6` preserves existing boolean aliases for int mode fields, following
   the JSON library's destination-type rules. Integer extrema, oversized,
   fractional, string and legacy boolean inputs are tested.
7. Finding 12: one-time registration publishes a complete staged descriptor,
   discards partial state on failure and rejects same-type recursion
   (`0e591bee`). Controlled concurrency, retry and recursion tests pass.
8. Finding 15: smoothing implementations share one canonical helper definition
   (`be475178`). Structural verification and linked no-displacement controls
   replace a source-confirmed one-definition defect; no runtime miscompile is
   claimed, and nonzero-displacement convergence was not tested.
9. Finding 16: the wall-distance point-collection callback initializes its
   quadrature contribution (`be475178`). Analytic square distances, ghosts and
   empty-wall sentinels pass; this is a source-confirmed fix, not a reproduced
   instrumentation failure.
10. Finding 20: obey the user's choice to require identical row/matrix shapes
    before swapping (`c062c22c`). Derived matrices and both batch forms are
    checked; father and son are prevalidated together. Tests cover unequal
    shapes with equal storage and matching-shape controls.

These are CFD correctness, memory safety, object safety and lifetime safety
repairs. Five commits group related finding pairs; a sixth preserves a legacy
configuration convention exposed by the full test suite. Config/serializer
interfaces and file formats remain unchanged. Typed derived swap overloads
enforce the selected stricter shape contract.

## Final validation

| Gate at `675392e6` | Result |
| --- | --- |
| Full CPU build: all unit tests, four Python bindings, Euler | 153 steps, exit 0, 645.05 s |
| Python installation before tests | Exit 0 |
| Full CTest, MPI np1/2/4/8, OMP2, serial scheduling | 99/99 passed, 323.29 s |
| Whole Python `test/` suite | 49 passed, 1 CUDA-only skip, 22.51 s |
| New focused C++ matrix, np1/2, OMP1 | 25/25 passed, exactly one selected case each |
| First-batch focused matrix | 25/25 passed, selected counts checked from logs |
| Periodic Euler smoke, np2/OMP1 | Two physical steps, exit 0, 0.4656 s |

Builds used eight jobs; CTest used at most 16 cores; BLAS threads were limited
to one. All 16 recorded build/final file identities match, including Euler and
freshly installed libraries. No CUDA validation is claimed.

The first complete CTest attempt passed 95/99: four Euler pipeline entries
rejected an existing boolean `meshReorderCells` value under the new integer
validator. The compatibility follow-up repairs that regression without changing
existing cases or excluding tests. The complete suite was rerun. An interim
test draft incorrectly expected native uint64 boolean conversion too; it was
corrected, and its premature build was stopped. The final config executable
passes four cases and 59 assertions. All attempts remain explicitly classified.

The numerical-experiment skill governed the bounded smoke; GPT-5.6 Luna
monitored compilation and tests. The 100-cell periodic Euler case reached
`t=0.0002`; every checked initial/final field matched the first-batch control
exactly. Density/pressure remained positive and finite, and the relative
equal-cell density-sum change was `2.22e-16`. This is an execution smoke only:
some stages did not reach their residual threshold within eight inner iterations.
Dedicated tests cover serializer cache, in-situ, empty-rank, swap and smoothing
corner cases not reached by this solver configuration.

## Records and reproduction

Detailed before/after snippets for all 22 original findings, including the 20
repairs and two remaining findings, are in the independent review repository:
`workspace/review_gpt6astra_2026-09-26/BUGS_VS_FIXES.md` under the original root.
The report is separate from this source branch.

Local evidence inside the fix worktree:
`workspace/audit_fix_validation/batch2/RESULTS.md`, `full_final/results.json`,
`full_final/{ctest,pytest}.xml`, `final_checked/results.json`,
`final_batch1/results.json`, `full_final_final_state.json`, and
`smoke/{manifest,checks}.json`. Raw output is in `data/audit_fix_smoke_batch2`.
The checked runner retains exact commands, bounds, status, elapsed time and hashes.

```bash
cmake --build build/fix-cpu-ninja --target all_unit_tests \
  dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 euler -j8
cmake --install build/fix-cpu-ninja --component py
export TMPDIR="$PWD/workspace/audit_fix_validation/tmp"
export OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
timeout 1800 ctest --test-dir build/fix-cpu-ninja --output-on-failure -j1
PYTHONPATH="$PWD/python" timeout 1800 venv/bin/python -m pytest test/ --timeout=300
venv/bin/python workspace/audit_fix_validation/batch2/run_cases.py all new_run
```

Remaining review findings: **14**, Python-view allocation lifetime policy, and
**17**, CUDA-only kernel bounds. Neither is counted as repaired or validated.
