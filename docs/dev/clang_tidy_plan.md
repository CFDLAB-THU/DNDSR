# clang-tidy Cleanup Plan

Living document driving the check-by-check cleanup of the DNDSR C++
tree. Each "pass" below is a unit of work resulting in one commit; each
bucketed check is addressed to completion before the next starts.

Driver: `scripts/run_clang_tidy.py`. Config: `/.clang-tidy` (reporting)
and `/.clang-tidy-fix` (auto-fix profile). See
`docs/guides/style_guide.md` for usage.

## Status snapshot

- Current module: **DNDS**. **All planned passes complete.**
  Pass 6 (special-member-functions), pass 7 (macro-usage),
  pass 8 (redundant-casting), pass 9 (member-init),
  pass 10 (nullptr), pass 11 (emplace), pass 12 (equals-default),
  pass 13 (qualified-auto), pass 14 (named-parameter),
  pass 15 (simplify-boolean), pass 16 (unnecessary-value-param),
  pass 17 (loop-convert), pass 18 (prefer-member-init),
  pass 19 (cstyle-cast), pass 20 (non-const-global — KA disable),
  pass 21 (c-arrays), pass 22 (unhandled-self-assign),
  pass 23 (branch-clone), pass 24 (implicit-widening),
  pass 25 (rvalue-ref-not-moved), pass 26 (long-tail sweep).
- **Latest total: 1 diagnostic** (a `clang-diagnostic-error` from
  `omp.h` inside Eigen's PCH — unrelated to DNDS code; cannot be
  fixed without installing `libomp-<v>-dev` matched to the
  clang-tidy / clangd binary version).
- **Original baseline: 24 597 diagnostics across 51 distinct checks**
  (captured in `build/clang-tidy-logs/baseline-dnds.txt`).
- **End-state delta: –99.996 %** (or "effectively zero") on total
  diagnostics; 51 → 1 distinct checks.

Summary counts after each pass (DNDS only, all-TU):

| After | Total | Distinct | Delta |
|---|---:|---:|---:|
| Baseline | 24 597 | 51 | — |
| Pass 1 (macro-parentheses NOLINT) | 24 067 | 50 | −530 |
| Pass 2 (nodiscard --fix) | 22 495 | 49 | −1 572 |
| Pass 3 (init-variables --fix) | 21 297 | 49 | −1 198 |
| Pass 4 (missing-std-forward disable + YAML trap) | 20 794 | 47 | −503 |
| Pass 5 (reserved-identifier rename) | 19 390 | 46 | −1 404 |
| KA bucket disables | 7 341 | 35 | −12 049 |
| Pass 6 (special-member-functions) | 5 691 | 34 | −1 650 |
| Pass 7 (macro-usage disable) | 3 581 | 33 | −2 110 |
| Pass 8 (redundant-casting --fix) | 2 805 | 32 | −776 |
| Pass 9 (member-init --fix) | 2 330 | 31 | −475 |
| Pass 10 (use-nullptr --fix) | 2 099 | 30 | −231 |
| Pass 11 (use-emplace --fix) | 1 987 | 29 | −112 |
| Pass 12 (use-equals-default --fix) | 1 928 | 28 | −59 |
| Pass 13 (qualified-auto --fix) | 1 876 | 27 | −52 |
| Pass 14 (named-parameter --fix) | 1 654 | 26 | −222 |
| Pass 15 (simplify-boolean --fix + manual) | 1 499 | 25 | −155 |
| Pass 16 (unnecessary-value-param) | 1 180 | 24 | −319 |
| Pass 17 (loop-convert --fix + NOLINT) | 1 031 | 23 | −149 |
| Pass 18 (prefer-member-initializer --fix) | 971 | 23 | −60 |
| Pass 19 (cstyle-cast manual) | 847 | 22 | −124 |
| Pass 20 (non-const-global disable) | 523 | 21 | −324 |
| Pass 21 (avoid-c-arrays) | 455 | 20 | −68 |
| Pass 22 (unhandled-self-assign) | 402 | 19 | −53 |
| Pass 23 (branch-clone NOLINTBEGIN/END) | 80 | 18 | −322 |
| Pass 24 (implicit-widening NOLINT) | 14 | 14 | −66 |
| Pass 25 (rvalue-ref move) | 13 | 13 | −1 |
| Pass 26 (long-tail sweep) | 1 | 1 | −12 |

## Table of contents

1. [Scope and ground rules](#scope-and-ground-rules)
2. [Baseline — DNDS](#baseline--dnds)
3. [Triage table](#triage-table)
4. [Pass log](#pass-log)
   1. [Pass 1 — bugprone-macro-parentheses](#pass-1--bugprone-macro-parentheses)
   2. [Pass 2 — modernize-use-nodiscard](#pass-2--modernize-use-nodiscard)
   3. [Pass 3 — cppcoreguidelines-init-variables](#pass-3--cppcoreguidelines-init-variables)
   4. [Pass 4 — cppcoreguidelines-missing-std-forward](#pass-4--cppcoreguidelines-missing-std-forward)
   5. [Pass 5 — bugprone-reserved-identifier](#pass-5--bugprone-reserved-identifier)
   6. [Pass 6 — cppcoreguidelines-special-member-functions](#pass-6--cppcoreguidelines-special-member-functions)
   7. [Pass 7 — cppcoreguidelines-macro-usage](#pass-7--cppcoreguidelines-macro-usage)
   8. [Pass 8 — readability-redundant-casting](#pass-8--readability-redundant-casting)
   9. [Pass 9 — cppcoreguidelines-pro-type-member-init](#pass-9--cppcoreguidelines-pro-type-member-init)
   10. [Pass 10 — modernize-use-nullptr](#pass-10--modernize-use-nullptr)
   11. [Pass 11 — modernize-use-emplace](#pass-11--modernize-use-emplace)
   12. [Pass 12 — modernize-use-equals-default](#pass-12--modernize-use-equals-default)
   13. [Pass 13 — readability-qualified-auto](#pass-13--readability-qualified-auto)
   14. [Pass 14 — readability-named-parameter](#pass-14--readability-named-parameter)
   15. [Pass 15 — readability-simplify-boolean-expr](#pass-15--readability-simplify-boolean-expr)
   16. [Pass 16 — performance-unnecessary-value-param](#pass-16--performance-unnecessary-value-param)
   17. [Pass 17 — modernize-loop-convert](#pass-17--modernize-loop-convert)
   18. [Pass 18 — cppcoreguidelines-prefer-member-initializer](#pass-18--cppcoreguidelines-prefer-member-initializer)
   19. [Pass 19 — cppcoreguidelines-pro-type-cstyle-cast](#pass-19--cppcoreguidelines-pro-type-cstyle-cast)
   20. [Pass 20 — cppcoreguidelines-avoid-non-const-global-variables](#pass-20--cppcoreguidelines-avoid-non-const-global-variables)
   21. [Pass 21 — cppcoreguidelines-avoid-c-arrays](#pass-21--cppcoreguidelines-avoid-c-arrays)
   22. [Pass 22 — bugprone-unhandled-self-assignment](#pass-22--bugprone-unhandled-self-assignment)
   23. [Pass 23 — bugprone-branch-clone](#pass-23--bugprone-branch-clone)
   24. [Pass 24 — bugprone-implicit-widening-of-multiplication-result](#pass-24--bugprone-implicit-widening-of-multiplication-result)
   25. [Pass 25 — cppcoreguidelines-rvalue-reference-param-not-moved](#pass-25--cppcoreguidelines-rvalue-reference-param-not-moved)
   26. [Pass 26 — long-tail sweep](#pass-26--long-tail-sweep)
5. [Disables applied](#disables-applied)
6. [NOLINT markers in the tree](#nolint-markers-in-the-tree)
7. [Next modules](#next-modules)

---

## Scope and ground rules

1. **One check per commit.** Mixing checks makes review impossible and
   regressions hard to bisect.
2. **Triage first, fix second.** Before a pass starts, decide whether
   the check fits the project: *Keep & fix*, *Keep & accept* (add to
   disables), *Keep but silence locally* (per-site `// NOLINT(...)`),
   or *Re-read later*.
3. **Verify after every pass.** `cmake --build build -j32` must
   succeed. Relevant ctests (currently `ctest -R '^dnds_'`) must pass.
   `scripts/run_clang_format.py --check <scope>` must pass.
4. **Spot-check every auto-fix.** After any `--fix` run, sample at
   least a handful of changed sites (random spread, not just the
   first) before committing.
5. **Never skip the pre-commit hook.**
6. **No CUDA TUs.** `.cu` files are excluded by the driver (nvcc
   flags in `compile_commands.json` break clang's CUDA frontend).
   CUDA-included headers are still tidied via their `.cpp` includers.

## Baseline — DNDS

Captured with:

```bash
scripts/run_clang_tidy.py --summary --top-checks 50 DNDS \
  > build/clang-tidy-logs/baseline-dnds.txt
```

Top-20 (after `f1fb769` / `2d5257c`, with the summary regex fixed,
68 TUs, **24 597 total diagnostics across 51 distinct checks**):

| # | Check | Hits |
|---|---|---|
| 1 | `cppcoreguidelines-non-private-member-variables-in-classes` | 3 412 |
| 2 | `cppcoreguidelines-avoid-magic-numbers` | 2 690 |
| 3 | `cppcoreguidelines-pro-bounds-pointer-arithmetic` | 2 119 |
| 4 | `cppcoreguidelines-macro-usage` | 2 110 |
| 5 | `cppcoreguidelines-special-member-functions` | 1 645 |
| 6 | `modernize-use-nodiscard` | 1 572 |
| 7 | `bugprone-reserved-identifier` | 1 404 |
| 8 | `cppcoreguidelines-pro-type-vararg` | 1 402 |
| 9 | `cppcoreguidelines-init-variables` | 1 198 |
| 10 | `cppcoreguidelines-pro-type-reinterpret-cast` | 842 |
| 11 | `readability-redundant-casting` | 776 |
| 12 | `readability-redundant-access-specifiers` | 537 |
| 13 | `bugprone-macro-parentheses` | 530 |
| 14 | `cppcoreguidelines-missing-std-forward` | 503 |
| 15 | `cppcoreguidelines-pro-type-member-init` | 475 |
| 16 | `cppcoreguidelines-pro-type-const-cast` | 371 |
| 17 | `cppcoreguidelines-pro-bounds-array-to-pointer-decay` | 334 |
| 18 | `cppcoreguidelines-avoid-non-const-global-variables` | 324 |
| 19 | `bugprone-branch-clone` | 322 |
| 20 | `performance-unnecessary-value-param` | 319 |

> The summary regex originally matched bracketed tokens like
> `[[nodiscard]]`, `<loc>`, `<pos>`, `<name>` inside clang-tidy note
> lines, inflating the `modernize-use-nodiscard` row to 3 621 and
> adding four phantom "checks". The regex was tightened (require a
> `-` or `.` inside the name) before the table above was captured.
> Delta between the two runs: 3 740 spurious hits removed, real
> totals unchanged.

Goal: after all passes in this plan, the top-20 should drop by more
than 50 % of total volume, with the "Keep & fix" rows going to
near-zero.

## Triage table

Bucket legend:

- **KF** — *Keep & fix.* Schedule a pass.
- **KA** — *Keep & accept.* Add to `.clang-tidy` disables.
- **KS** — *Keep but silence locally* at specific call sites with
  `// NOLINT(check-name)` + a one-line reason.
- **RL** — *Re-read later.* Decide after noisy checks are drained.

| Check | Bucket | Rationale |
|---|---|---|
| `cppcoreguidelines-non-private-member-variables-in-classes` | KA | Project uses `struct` with public fields as data bags extensively (Array storage, MPI views, config sections). Enforcing private+getters would be a massive architectural change with no safety gain at our scale. |
| `cppcoreguidelines-avoid-magic-numbers` | KA | Overlaps with already-disabled `readability-magic-numbers`. Identical hit set, identical reasoning. |
| `cppcoreguidelines-pro-bounds-pointer-arithmetic` | KA | CSR storage and MPI byte buffers are fundamentally pointer-arithmetic. Fixing means migrating to `std::span`/`gsl::span` — a design move, not a tidy pass. |
| `cppcoreguidelines-macro-usage` | RL | Project uses macros for device portability, Eigen workarounds, config registration. Load-bearing. Revisit once other noise is gone. |
| `cppcoreguidelines-special-member-functions` | KF | Missing rule-of-five declarations. Can catch real bugs around implicit moves/copies. **Pass 6.** |
| `modernize-use-nodiscard` | KF | Mechanical, mostly auto-fixable. Spot-check needed (some functions are legitimately called for side effects). **Pass 2.** |
| `bugprone-reserved-identifier` | KF | `_Tp`, `__DNDS_str`, etc. are reserved-name patterns. Mechanical rename, not auto-fixable across TUs. **Pass 5.** |
| `cppcoreguidelines-pro-type-vararg` | KA | MPI and printf-family APIs require varargs. Silencing globally is correct. |
| `cppcoreguidelines-init-variables` | KF | Mechanical, mostly auto-fixable. Spot-check for cases where default-init is intentional (e.g. `int rank;` right before `MPI_Comm_rank`). **Pass 3.** |
| `cppcoreguidelines-pro-type-reinterpret-cast` | KA | Used for MPI byte buffers and serialization. No idiomatic replacement. |
| `readability-redundant-casting` | RL | Some are style-only, some indicate real type drift. Revisit after pass 7. |
| `readability-redundant-access-specifiers` | KA | Project explicitly repeats `public:` for readability in large classes. Documented convention. |
| `bugprone-macro-parentheses` | KF | Cheap, mechanical, catches real macro-expansion bugs. Some hits are design-intentional (token-paste operands, argument types) and need `NOLINT`. **Pass 1.** |
| `cppcoreguidelines-missing-std-forward` | KF | Mechanical, auto-fixable. **Pass 4.** |
| `cppcoreguidelines-pro-type-member-init` | RL | Overlaps with `cppcoreguidelines-init-variables` (pass 3) on members. Revisit delta. |
| `cppcoreguidelines-pro-type-const-cast` | KA | Needed for interop with C APIs (MPI, CGNS, HDF5) that take non-const pointers. |
| `cppcoreguidelines-pro-bounds-array-to-pointer-decay` | KA | Same reasoning as pointer-arithmetic. |
| `cppcoreguidelines-avoid-non-const-global-variables` | RL | Globals for signal handlers, log streams, MPI info are intentional; there may be stragglers worth trimming. Revisit after pass 2. |
| `bugprone-branch-clone` | RL | Sometimes a real duplication, sometimes stylistic (exhaustive `if/else` cascades for enum handling). Needs eyes. |
| `performance-unnecessary-value-param` | KF candidate later | Would flag things like `ssp<T>` taken by value where const-ref or move would do. Needs care — some by-value params are deliberate (MPI type, POD). Defer until pass 6 is done. |

## Pass log

Each pass below records: objective, scope, commands, sample reviews,
verification (build + any ctests), commit hash, post-pass delta.

### Pass 1 — bugprone-macro-parentheses

**Outcome.** 530 hits → 0. Total diagnostics 24 597 → 24 067.

All 530 hits collapsed to **17 unique source locations in 3 files**:

| File | Lines (col) | Distinct sites | Dupes per site |
|---|---|---|---|
| `src/DNDS/Defines.hpp` | 86:28, 88:39, 93:28, 95:39 | 4 | 66 |
| `src/DNDS/ArrayDOF.hpp` | 68..79:5 | 12 | 22 |
| `src/DNDS/Config/ConfigParam.hpp` | 700:60 | 1 | 2 |

**Decision.** All 17 are false positives: every flagged token is a
C++ *type name* or *storage-class specifier* in a context where
parenthesization is not valid syntax:

- `DNDS_DEVICE_TRIVIAL_COPY_DEFINE(T, T_Self)` — `T` and `T_Self` appear
  as parameter types in constructor and assignment-operator signatures.
- `DNDS_ARRAY_DOF_OP_FUNC_LIST(..., spec)` — `spec` is always passed
  `static`; it's the leading storage-class specifier of a function
  declaration.
- `DNDS_DECLARE_CONFIG(Type_)` — `Type_` is used as a parameter type
  and as a template argument.

**Fix.** Three `NOLINTBEGIN` / `NOLINTEND` pairs around the macro
definitions, each with a one-sentence rationale comment. 13 lines
of comments replace 530 diagnostics.

**Verification.** `cmake --build build -t dnds -j32` succeeds; the
summary drop of 530 exactly matches the decrement in the total
(no secondary effects).

Commit: see `git log -- docs/dev/clang_tidy_plan.md`.

### Pass 2 — modernize-use-nodiscard

**Outcome.** 1 572 hits → 0. Total diagnostics 24 067 → 22 495.

**Method.** Single-check override config at `/tmp/pass2.clang-tidy`:

```yaml
Checks: "-*,modernize-use-nodiscard"
WarningsAsErrors: ''
HeaderFilterRegex: '.*/DNDSR/(src|app|test/cpp)/.*'
ExtraArgs: [-UDNDS_USE_OMP, -Wno-unknown-warning-option, -Wno-unused-command-line-argument]
FormatStyle: file
```

Driven by `scripts/run_clang_tidy.py --fix --config-file /tmp/pass2.clang-tidy DNDS`.

**Incident & fix.** The first `--fix` attempt ran with 64 parallel
workers and produced syntax-error corruption in `Vector.hpp`,
`Array.hpp`, `ArrayPair.hpp`, and several other headers because the
parallel workers race when the same header is edited from multiple
TUs (observed: interleaved `[[nodiscard]]` insertions in mid-token
positions).

Reverted via `git checkout HEAD -- src/DNDS/` and patched the driver
to force `jobs=1` whenever `--fix` is set (with
`--unsafe-parallel-fix` escape hatch). Serialized run completed
cleanly. Committed as a separate `fix(tooling)` commit so the pass's
diff stays pure.

**Diff footprint.** 11 files, 32 line-replacements (one `[[nodiscard]]`
prefix each). 1 572 repeated hits collapsed to 32 unique decls
because each header declaration is reported once per including TU.

**Sample review.** 4 spot-checks:

- `Array.hpp:570` `at() const` — const getter, value return.
- `ArrayBasic.hpp:423` `at_compressed(...) const` with
  `DNDS_DEVICE_CALLABLE` prefix — attribute order is valid.
- `EigenUtil.hpp:278..294` `rows()/cols()/size() const` — Eigen
  dimension wrappers, pure reads.
- `SerializerH5.cpp:180` `get_indent() const` — local helper,
  string-builder, no side effects.

All good; no call sites in DNDS were found that call these functions
for side effects.

**Verification.** `cmake --build build -t dnds --clean-first -j32`
succeeds; `euler` target builds. Pre-commit clang-format ran on 11
modified files, no drift.

Commits: `7502b8d` (driver serialize-on-fix) + pass 2 commit.

### Pass 3 — cppcoreguidelines-init-variables

**Outcome.** 1 198 hits → 0. Total diagnostics 22 495 → 21 297.

**Method.** Same recipe as Pass 2; single-check config at
`/tmp/pass3.clang-tidy`, `--fix`, serialized. One full run fixed all
1 198 hits in one shot (no stragglers).

**Diff footprint.** 8 files, ~30 line-replacements. Most are the
classic "declare-then-immediately-MPI-writes" pattern; `int x; MPI_*(&x)`
now `int x = 0; MPI_*(&x)`.

**Sample review found two corrections needed:**

1. `ArrayTransformer.hpp:838,867` — clang-tidy initialised
   `MPI_Datatype dtype` to `nullptr`. That's only valid on MPI
   implementations where `MPI_Datatype` is a pointer typedef
   (OpenMPI). MPICH defines it as `int`, so `nullptr` would not
   compile. Manually corrected to `MPI_DATATYPE_NULL`, the canonical
   sentinel that works on both.
2. `ArrayDOF_op.hxx` — clang-tidy inserted `#include <math.h>` and
   initialised `real sqrSumAll = NAN;`. The sibling function on
   line 228 initialises the same variable to `0`. Corrected to
   match the sibling and removed the unnecessary include.

**Verification.** `cmake --build build -t dnds -j32` succeeds after
both corrections. Pre-commit clang-format re-ran, no drift.

**Lesson learned.** `cppcoreguidelines-init-variables` with `--fix`
picks unusual sentinel values (`nullptr` based on typedef, `NAN` for
floats). Always review auto-fix output for semantic appropriateness,
not just build success.

Commit: see `git log`.

### Pass 4 — cppcoreguidelines-missing-std-forward

**Outcome.** 503 hits → 0, by reclassifying as **KA (keep & accept)**
and adding to `.clang-tidy` disables.

**Method.** Single-check `--fix` attempt produced 0 edits: this check
does not implement auto-fix. Switched to manual review.

**Sample analysis (all 9 unique sites in DNDS):**

1. `Array.hpp:477` — `Resize(index, TFRowSize&& FRowSize)`. FRowSize
   is a **functor called in-place**. Forwarding-ref was chosen only
   to accept both lvalue and rvalue callables; `std::forward` on a
   functor that the body keeps calling directly would move-from on
   first call and break subsequent ones.
2. `Array.hpp:723` — `ResizeRowsAndCompress(TRowSizeFunc&&)`. Same
   pattern (functor called inside a loop).
3. `ArrayPair.hpp:249` — `runFunctionAppendedIndex(index, TF&& F)`.
   Same pattern (`F(*father, i)` inside the body).
4. `ArrayEigenUniMatrixBatch.hpp:110` — `Resize(..., TFRowSize&& rsf)`.
   Functor called twice inside a lambda capturing it by reference.
5. `Defines.hpp:834` (×2 columns) — hash functor
   `operator()(TBegin&& begin, TEnd&& end)`. Iterators used for
   random access; no forwarding semantics apply.
6. `IndexMapping.hpp:215` — `OffsetAscendIndexMapping(...)` taking
   `TpullSet&& pullingIndexGlobal`. The body mutates the collection
   in place (`sort`, `unique`, `erase`, `shrink_to_fit`) but never
   moves it elsewhere. Should be `TpullSet&`, but the author's
   commented-out `// std::forward<...>(...); // might delete` shows
   they considered it and decided against.
7. `IndexMapping.hpp:298-299` — analogous ctor overload taking two
   collections, used read-only.

**Decision.** All nine sites are either functor-called-in-place
(cannot be forwarded without breaking repeated calls) or
collection-used-by-reference (should be `T&`/`const T&`, not `T&&`).
The latter is an API change that ripples through call sites and is
outside the scope of "tidy passes." The check cannot distinguish
legitimate forwarding-template usage from these patterns, so it
produces a steady stream of false positives.

**Action.** Bucket moved from KF → KA. Added to `.clang-tidy`
disables; Checks list now has an `-cppcoreguidelines-missing-std-forward`
line.

**Secondary fix: YAML folded-scalar trap.** During this pass I
discovered that rationale comments placed *inside* the `Checks: >`
folded scalar are parsed as text (YAML `#` is only a comment at
line start, not inside a folded block), which silently concatenated
my commentary into a malformed check name and the subsequent
disables were ignored. Fixed by moving all rationale comments into
the file header as a table; the Checks: block is kept comment-free.
A warning note was added to the header explaining this.

Commit: see `git log`.

### Pass 5 — bugprone-reserved-identifier

**Outcome.** 1 404 hits → 0. Total diagnostics 20 794 → 19 390.

**Method.** Manual; this check has no auto-fix. The 1 404 hits
collapsed to ~25 unique identifiers, each flagged because it starts
with `__` or `_[A-Z]` or contains `__` (all reserved-name patterns
per [lex.name]/3.2).

**Renames applied (leading underscores dropped):**

| Before | After | Notes |
|---|---|---|
| `_Tp` | `Tp` | Template parameter in 2 sites (Defines.hpp). |
| `__DNDS_str` | `DNDS_str` | Token-stringize helper macro. |
| `__DNDS__json_to_config` | `DNDS_json_to_config` | Both leading and internal `__` removed. |
| `__DNDSToMPITypeInt`, `__DNDSToMPITypeFloat` | drop leading `__`. |  |
| `__EndTimerType` | `EndTimerType` | Timer callback type. |
| `__InSituPackStartPull`, `__InSituPackStartPush` | drop leading `__`. |  |
| `__OneMatGetRowSize` | `OneMatGetRowSize` | Template helper. |
| `__ReadSerializerData`, `__ReadSerializerDataAndPropagateOffset`, `__ReadSerializerStructuralAndResolveDataOffset`, `__WriteSerializerData` | drop leading `__`. | Serializer internals. |
| `__Row_size` | `Row_size` | Template metafunction. |
| `__p_indices` | `p_indices` | Helper in bind module. |
| `__start_timer`, `__stop_timer` | `start_timer`, `stop_timer` | Timer API. |
| `__pybind11_callBind*s_rowsizes_sequence` (×8) | drop leading `__`. | Our own template helpers that happened to live next to pybind11 code. |
| `__EigenPCH`, `__ExprtkPCH` | `EigenPCH_tag`, `ExprtkPCH_tag` | Module-tag strings; avoided colliding with the class / filename. |

**Collision-handled:**

| Before | After | Rationale |
|---|---|---|
| `__size`, `__offset` (SerializerBase.hpp:40) | `sz`, `ofs` | Ctor params; `size`/`offset` have 521 / 302 existing uses. |
| `_GetDataLayout` (ArrayBasic.hpp / Array.hpp) | `ComputeDataLayout` | `GetDataLayout` already exists as a different member. |

**Verification.** `cmake --build build -t dnds -j32` succeeds;
`cmake --build build -t euler -j32` succeeds (catches cross-module
consumers in Euler/CFV); clang-tidy re-summary shows 0 hits for
`bugprone-reserved-identifier`.

**Lesson.** A plain leading-underscore strip is not enough when the
identifier also has an internal `__`; re-running the check after each
bulk sed pass catches the residue quickly.

Commit: see `git log`.

### Pass 6 — cppcoreguidelines-special-member-functions

**Outcome.** 1 645 → 0. Completed as one commit (`1880d48`).

**Per-class audit.** The 1 645 warnings collapsed to ~20 distinct
class declarations, each bucketed into one of four categories with
an explicit rule-of-five closure:

1. **Value-semantic classes** (members are all `shared_ptr`,
   `host_device_vector`, POD, or `std::vector`): add
   `= default` move ctor, move assign, and destructor alongside
   the existing custom copy. Default move is a shallow transfer
   of the shared handles — correct and observably identical to
   "copy source + reset source" on the moved-from side.
   Classes: `Array`, `ArrayAdjacency`, `ArrayDof`, `ArrayEigenMatrix`,
   `ArrayEigenMatrixBatch`, `ArrayEigenUniMatrixBatch`,
   `ArrayEigenVector`, `ArrayTransformer`, `ParArray`,
   `AdjacencyRow`, `RowView`, `EmptyNoDefault`,
   `host_device_vector_r0`, `host_device_vector_r1`, and their
   nested `iterator` classes.
2. **Polymorphic RAII bases** (virtual dtor, owns file handle /
   MPI handle / opaque exprtk pointer): `= delete` copy and move
   to prevent slicing and double-close.
   Classes: `SerializerBase`, `SerializerJSON`, `SerializerH5`,
   `DeviceStorageBase`, `DeviceHostSingleAllocationBase`,
   `DeviceHostSingleAllocationDirect`, `ExprtkWrapperEvaluator`.
3. **Classic singletons** (old pre-C++11 private-unimplemented
   idiom): replace with `= delete` copy/move + `= default` dtor.
   Classes: `CommStrategy`, `MPIBufferHandler`, `ResourceRecycler`,
   `PerformanceTimer`.
4. **Resource-registry holders** (register `this` with
   `ResourceRecycler` by raw pointer): `= delete` copy/move to
   prevent registering the same `this` twice.
   Classes: `MPIReqHolder`, `MPITypePairHolder`.

Every declaration carries a one-line rationale comment explaining
which bucket the class falls in and why the chosen semantics are
correct.

### Pass 7 — cppcoreguidelines-macro-usage

**Outcome.** 2 110 → 0. Config-only disable commit (`9c71e4b`).

All 41 distinct macros in DNDS are legitimate uses that a
`constexpr` template function cannot express:

- Assertions / checks capturing `__FILE__` / `__LINE__`
  (`DNDS_assert*`, `DNDS_check_throw*`, `DNDS_HD_assert*`).
- Code-generation DSLs that declare class members, static data,
  or JSON binding glue (`DNDS_DECLARE_CONFIG`, `DNDS_FIELD`,
  `DNDS_json_to_config`, `DNDS_NLOHMANN_DEFINE_*`,
  `pybind11_bind_*`, `DNDS_DEVICE_TRIVIAL_COPY_DEFINE*`).
- Platform probes / define-before-include switches
  (`MPICH_SKIP_MPICXX`, `OMPI_SKIP_MPICXX`,
  `EIGEN_DONT_PARALLELIZE`).
- Intrinsic / attribute wrappers (`DNDS_likely`, `DNDS_unlikely`,
  `DNDS_FORCEINLINE`, `DISABLE_WARNING` via `_Pragma`).
- CMake-injected constants (`DNDS_VERSION_STRING`).

The full list is in the `.clang-tidy` header disables table.

### Pass 8 — readability-redundant-casting

**Outcome.** 776 → 0 via single-check `--fix` run (`742119d`).

Three patterns auto-fixed:
- `MPI_Datatype(MPI_FLOAT)` etc. (OpenMPI constants are already
  `MPI_Datatype`, so the functional cast is a no-op) — 11 sites
  in `MPI.hpp`.
- `reinterpret_cast<uint8_t *>(uint8_t *)` identity casts in
  `Device/DeviceStorage.cpp` — 3 sites.
- `index(nSend)` where `nSend` is already `index` in
  `ArrayTransformer.hpp`.

### Pass 9 — cppcoreguidelines-pro-type-member-init

**Outcome.** 475 → 0 via `--fix` + 1 manual fix (`173ad1a`).

Raw class members (`index _size`, `rowsize Row_size`,
`MPI_Aint pushSendSize`, `ConfigTypeTag typeTag`, `tStart`
array) gained `{}` default initializers. Local `std::array` scratch
buffers in `MPI.cpp`, `SerializerH5.cpp`, `Defines.cpp`,
`ArrayEigenMatrix.hpp`, `ArrayTransformer.hpp` zero-init'd the same way.

Manual fix: auto-fix emitted `struct winsize w { };` on two lines;
reformatted inline to `struct winsize w{};`.

### Pass 10 — modernize-use-nullptr

**Outcome.** 231 → 0 via `--fix` (`6c494f4`).

`(T *)(NULL)` → `(T *)nullptr` in the past-the-end row inquiry
(`ArrayBasic.hpp`), `getenv()` comparisons (`MPI.hpp`, `MPI.cpp`),
and HDF5 handle probes (`SerializerH5.cpp`).

### Pass 11 — modernize-use-emplace

**Outcome.** 112 → 0 via `--fix` (`56c43af`).

`push_back(std::make_pair(r, dtype))` → `emplace_back(r, dtype)`
in MPI type-pair vector appends (2 sites in
`ArrayTransformer.hpp`) and HDF5 dimension vectors
(`SerializerH5.cpp`), plus `push_back(std::string(...))` →
`emplace_back(...)` in `MPI.cpp` / `MPI_bind.cpp`.

### Pass 12 — modernize-use-equals-default

**Outcome.** 59 → 0 via `--fix` (`9231ae6`).

Two sites (duplicated across TUs):
`DeviceHostSingleAllocationBase::~DeviceHostSingleAllocationBase() {}` and
`DeviceHostSingleAllocationDirect::~DeviceHostSingleAllocationDirect()
override {}` → `= default`.

### Pass 13 — readability-qualified-auto

**Outcome.** 52 → 0 via `--fix` (`7586d2f`).

`auto ptr = reinterpret_cast<T*>(...)` → `auto *ptr = ...` in
pybind11 binding helpers; `for (auto &[k, v] : map.items())` →
`for (const auto &[k, v] : map.items())` in `SerializerJSON.cpp`.

### Pass 14 — readability-named-parameter

**Outcome.** 222 → 0 via `--fix` (`0bf9edd`).

Added `/*unused*/` on tag-dispatch parameters
(`std::index_sequence<Is...> /*unused*/`) in the pybind11
binding machinery.

### Pass 15 — readability-simplify-boolean-expr

**Outcome.** 155 → 0 via `--fix` + 1 manual (`e07c315`).

`!(a && b)` / `!(a || b)` DeMorgan expansions in `ArrayDOF.hpp`
(SFINAE `enable_if`), `ArrayDOF_bind.hpp` (`if constexpr`),
`EigenUtil.hpp` (ternary condition), and
`Defines.hpp::checkedIndexTo32` (manual — auto-fix emitted 66
duplicates into header-include paths).

### Pass 16 — performance-unnecessary-value-param

**Outcome.** 319 → 0 via `--fix` + manual sed (`b2019a7`).

`py::buffer row` by value → `const py::buffer &row` in 15
pybind11 `setitem` / operator overloads across the 5
`_bind.hpp` headers. Two additional sites in `Serializer_bind.hpp`
(`py::object options_in`) and `MPI_bind.cpp` (`Allreduce`
`py_sendbuf` / `py_recvbuf`).

### Pass 17 — modernize-loop-convert

**Outcome.** 149 → 0 via `--fix` + NOLINTBEGIN/END (`9a88b36`).

Auto-fix converted two index-based loops over `std::vector<index>`
in `ArrayRedistributor.hpp` to range-based form. One site in
`Array_bind.hpp` (`for (ssize_t i = 0; i < pullIndexGlobal.size();
i++) pullIndexVec.push_back(pullIndexGlobal.at(i));`) carries
`NOLINTBEGIN / NOLINTEND` — pybind11 `array_t` iterators yield
`pybind11::handle`, not `long`, and the explicit index-based form
is required for the numpy-to-long conversion.

A plain NOLINTNEXTLINE is insufficient: `--fix` rewrites the
`for` line itself, erasing the preceding comment. Block-form
NOLINT survives.

### Pass 18 — cppcoreguidelines-prefer-member-initializer

**Outcome.** 60 → 0 via `--fix` (`a3cb4a5`).

One unique site: `MPIInfo::MPIInfo(MPI_Comm ncomm)` — moved
`comm = ncomm` from the body into the member-initializer list.

### Pass 19 — cppcoreguidelines-pro-type-cstyle-cast

**Outcome.** 124 → 0 via 6 manual edits (`226ead9`).

Four sites in `SerializerH5.cpp` / `SerializerJSON.cpp`:
`(ssp<tValue> *)(pth_2_ssp[refPath])` → explicit
`reinterpret_cast<ssp<tValue> *>(...)` on the type-erased dedup
registry (matches the author TODO).

Two sites in `MPI.hpp`: `MPI_IN_PLACE` expands to the
OpenMPI-defined `((void *)1)` sentinel. `NOLINTNEXTLINE` placed
*immediately* above the offending `Allreduce(...)` call — multi-
line rationale comments between the NOLINT and the offending
line break the suppression (same trap as Pass 17).

### Pass 20 — cppcoreguidelines-avoid-non-const-global-variables

**Outcome.** 324 → 0 via config disable (`c407cff`).

Every global mutable in DNDS is intentional and cannot be made
`const`, `thread_local`, or class-scoped without wider redesign:
`logStream`, `useCout`, `outputDelim`, `HDF_mutex`, `isDebugging`,
`EigenPCH_tag`, `ExprtkPCH_tag`. Full rationale in the
`.clang-tidy` header table.

### Pass 21 — cppcoreguidelines-avoid-c-arrays

**Outcome.** 68 → 0 via 2 manual edits (`1d60a57`).

Two sites, both stack scratch buffers for printf-family calls:
- `Errors.hpp::genFatalErrorMessage`: `char format_buf[1024*512]`
  → `std::array<char, 1024*512> format_buf{}`.
- `SerializerFactory.hpp`: `char BUF[512]` → `std::array<char, 512>`.

### Pass 22 — bugprone-unhandled-self-assignment

**Outcome.** 53 → 0 via 1 manual edit (`dd31508`).

`AdjacencyRow::operator=(const AdjacencyRow &r)` called
`std::copy(r.cbegin(), r.cend(), p_indices)`. On self-assign,
source and destination ranges are fully overlapping — UB.
Added `if (this == &r) return;` early-return guard with a
comment explaining the UB.

### Pass 23 — bugprone-branch-clone

**Outcome.** 322 → 0 via NOLINTBEGIN / NOLINTEND blocks and
rationale comments (`10f5305`).

All 7 unique sites are intentional — the diagnostic fires where
two logically distinct branches happen to produce the same code:
- `ArrayBasic.hpp`, `Array.hpp` — `if constexpr` cascades over
  `_dataLayout`. `TABLE_Fixed` and `TABLE_Max` currently both
  compute `iRow * _row_size_dynamic + iCol`; the layouts are
  conceptually distinct (padded rows may diverge).
- `ArrayEigenUniMatrixBatch.hpp`, `ArrayEigenUniMatrixBatch_DeviceView.hpp`,
  `EigenUtil.hpp::MatrixFMTSafe` — ternary for the Eigen
  `options` template parameter; both non-row-vector arms
  intentionally select `ColMajor`.
- `Config/ConfigParam.hpp` — switch over `ConfigTypeTag` → JSON
  Schema type strings; several enums collapse to the same built-in
  (`Enum`→"string", `ArrayOfObjects`→"array",
  `MapOfObjects`→"object").

NOLINTBEGIN/END is required because clang-tidy reports the
diagnostic at the first clone of the pair; NOLINTNEXTLINE on one
line of the pair was insufficient.

### Pass 24 — bugprone-implicit-widening-of-multiplication-result

**Outcome.** 66 → 0 via 1 NOLINT (`583ab45`).

One source site: `std::array<char, 1024 * 512>` in `Errors.hpp`.
Compile-time constant `524 288` trivially fits in `int32_t`; the
widening to `size_t` happens at compile time during template
argument deduction. Spurious diagnostic; NOLINTNEXTLINE with
rationale.

### Pass 25 — cppcoreguidelines-rvalue-reference-param-not-moved

**Outcome.** 44 → 0 via 1 edit (`6c90a62`).

`ArrayDofDeviceView(t_base &&base_view) : t_base(base_view) {}`
and the `Const` variant: `base_view` inside the body is an
lvalue, so `t_base(base_view)` silently copied. Added
`std::move(base_view)` on the base-init to perform the intended
move.

### Pass 26 — long-tail sweep

**Outcome.** 12 warnings across 10 distinct checks → 0 via 13
edits (`0d6d0d9`). Drains every remaining check to zero:

- `performance-move-const-arg` (3 sites): removed no-op
  `std::move` of `const py::buffer &` / `const py::module_ &`.
- `readability-avoid-return-with-void-value` (5 sites): changed
  pybind11 setitem helpers from `auto` (deduced void) to
  explicit `void`, dropped `return` from the wrapping lambdas.
- `readability-make-member-function-const` (2 sites): made
  `SerializerFactory::BuildSerializer` / `::ModifyFilePath`
  `const`.
- `bugprone-empty-catch` (4 sites): four env-var parse catches in
  `CommStrategy::CommStrategy` guarded with NOLINTBEGIN/END.
- `readability-redundant-member-init`: dropped `: SerializerBase()`
  in `SerializerH5.hpp`.
- `modernize-pass-by-value`: `SerializerFactory(const std::string &)`
  → `SerializerFactory(std::string _type) : type(std::move(_type)) {}`.
- `modernize-use-auto` (2 sites): `py::buffer buf = v.cast<...>()`
  → `auto buf = ...`; `TraverseData *data = static_cast<...>`
  → `auto *data`.
- `readability-container-size-empty`: `ver.length()` →
  `!ver.empty()` in `Defines.cpp`.
- `bugprone-exception-escape`: `~SerializerJSON()` wraps
  `CloseFileNonVirtual()` in `try/catch` — destructors mustn't
  throw (NOLINTBEGIN/END guards the empty catch).
- `modernize-return-braced-init-list`: kept `return std::string(n, ch)`
  in `SerializerH5.cpp::get_indent` with NOLINTNEXTLINE — brace
  init is ambiguous with `initializer_list<char>` and triggers
  `-Wnarrowing`.
- `performance-unnecessary-copy-initialization`: NOLINT on
  `T vV = v` in `SerializerH5.cpp` (clang-tidy misses that the
  variable is addressed via `&vV` in the non-string `if constexpr`
  branch).
- `cppcoreguidelines-avoid-const-or-ref-data-members`: NOLINT on
  `H5Contents &contents` in the `TraverseData` per-call aggregate.
- `performance-no-int-to-ptr`: NOLINT on `MPI_Comm(pComm)` —
  Python side passes an opaque `uintptr_t`.
- `performance-inefficient-vector-operation`: added
  `pArgvOut.reserve(*pargc)` before the `emplace_back` loop in
  `MPI_bind.cpp`.
- `bugprone-multi-level-implicit-pointer-conversion`: NOLINT on
  `H5Aread(attr_id, dtype_id, &attr_value)` where `attr_value`
  is `char *` — HDF5 wants `void *buf` and explicit
  `static_cast<void*>` does not silence the check.
- `cppcoreguidelines-owning-memory` (7 sites): NOLINT on
  shared-pointer deleter callback (`DeviceStorage.cpp`), opaque
  exprtk pointers (`ExprtkWrapper.cpp`), and MPI_Init argv
  allocation (`MPI_bind.cpp`) with rationale.

### NOLINT placement: a repeated gotcha

Every auto-fix pass in this session re-taught the same lesson:

- `NOLINTNEXTLINE(check)` applies to the line *immediately*
  following the directive. Rationale comments must come *before*
  the directive, not between it and the offending code, otherwise
  the NOLINT applies to the rationale line.
- When `--fix` can rewrite the flagged line (e.g. `modernize-loop-convert`,
  `modernize-return-braced-init-list`), use
  `NOLINTBEGIN(check) ... NOLINTEND(check)` around the
  preserved block — the directive line survives the rewrite.
- For `switch` / `?:` / chained `if / else if` with
  `bugprone-branch-clone`, the diagnostic is reported at the first
  clone of the pair, which may not match the line the edit would
  change. NOLINTBEGIN/END is the safe form.

## Disables applied

Added to `.clang-tidy` during this effort (all motivated by the
triage table above):

| Check | Reason |
|---|---|
| `cppcoreguidelines-missing-std-forward` | Functor `T&&` called in-place (Pass 4). |
| `cppcoreguidelines-non-private-member-variables-in-classes` | Project uses struct-of-fields data bags. |
| `cppcoreguidelines-avoid-magic-numbers` | Duplicate of already-disabled `readability-magic-numbers`. |
| `cppcoreguidelines-pro-bounds-pointer-arithmetic` | CSR / MPI buffer idioms. |
| `cppcoreguidelines-pro-bounds-array-to-pointer-decay` | Same. |
| `cppcoreguidelines-pro-bounds-constant-array-index` | Same. |
| `cppcoreguidelines-pro-type-vararg` | MPI / printf-family. |
| `cppcoreguidelines-pro-type-reinterpret-cast` | MPI byte buffers, serialization. |
| `cppcoreguidelines-pro-type-const-cast` | C-API interop (MPI, CGNS, HDF5). |
| `readability-redundant-access-specifiers` | Repeated `public:` is a project convention. |
| `modernize-use-transparent-functors` | Eigen expression templates break with `std::less<>` etc. |
| `cppcoreguidelines-c-copy-assignment-signature` | Duplicate of `misc-unconventional-assign-operator`. |
| `cppcoreguidelines-macro-usage` | All 41 DNDS macros require `__FILE__`/`__LINE__` capture, token pasting, code generation, or define-before-include semantics (Pass 7). |
| `cppcoreguidelines-avoid-non-const-global-variables` | Every DNDS global mutable (`logStream`, `useCout`, `outputDelim`, `HDF_mutex`, `isDebugging`, PCH tags) is intentional; redesign out of scope for a tidy pass (Pass 20). |

Rationale comments live in the `.clang-tidy` file header, not inside
the `Checks: >` folded scalar (see Pass 4 / YAML trap).

## NOLINT markers in the tree

The tidy session left ~70 targeted `NOLINT` markers. All of them
pair with a rationale comment. Representative breakdown:

| Check | Count | Notable sites |
|---|---:|---|
| `bugprone-branch-clone` | 5 | `ArrayBasic.hpp`, `Array.hpp`, Eigen options ternaries, `ConfigParam.hpp` switch |
| `cppcoreguidelines-owning-memory` | 4 | `Device/DeviceStorage.cpp` deleter, `ExprtkWrapper.cpp` opaque new, `MPI_bind.cpp` argv alloc |
| `bugprone-empty-catch` | 5 | 4x `CommStrategy::CommStrategy` env-var parse, 1x `~SerializerJSON` |
| `modernize-loop-convert` | 1 | `Array_bind.hpp` pybind11 array_t iteration |
| `cppcoreguidelines-pro-type-cstyle-cast` | 2 | `MPI_IN_PLACE` (OpenMPI `(void*)1` sentinel) |
| `bugprone-implicit-widening-of-multiplication-result` | 1 | `Errors.hpp` compile-time `1024*512` |
| `modernize-return-braced-init-list` | 1 | `SerializerH5.cpp` `std::string(n, ch)` vs `initializer_list` overload |
| `performance-unnecessary-copy-initialization` | 2 | `SerializerH5.cpp` `T vV = v` used in non-string `if constexpr` branch |
| `cppcoreguidelines-avoid-const-or-ref-data-members` | 1 | `TraverseData` H5Literate callback state |
| `performance-no-int-to-ptr` | 1 | `MPI_Comm(pComm)` in pybind11 ctor |
| `bugprone-multi-level-implicit-pointer-conversion` | 1 | `H5Aread(..., &char_ptr)` |
| `bugprone-macro-parentheses` | 3 blocks | Unparenthesizable type-name / storage-class args |

## Next modules

**DNDS is now clean.** Every check with >= 1 actionable instance
has been driven to zero, either by fixing the code or by adding a
rationale-commented `NOLINT` / `.clang-tidy` disable. The sole
remaining diagnostic is a `clang-diagnostic-error` on `omp.h`
inside Eigen's PCH, which is an Eigen-internal include path issue
unrelated to DNDS source.

Repeat the recipe for the other modules in this order:

1. `src/Solver/` — small, limited blast radius; good next target.
2. `src/Geom/` — largest module; expect a new set of checks specific
   to mesh connectivity loops. Start with
   `clang-analyzer-optin.cplusplus.VirtualCall` per the historical
   note in `.clang-tidy`.
3. `src/CFV/` — follows Geom closely.
4. `src/Euler/`, `src/EulerP/` — solver layer; `bugprone-branch-clone`
   will matter here (exhaustive `if / else` cascades for enum handling).

The `.clang-tidy` disables and NOLINT placement lessons carry
forward unchanged. Any new module-specific disables should be
appended to the header table, not inside the `Checks:` block
(YAML folded scalars eat `#` as literal text).

The `.clang-tidy` disables carry forward unchanged. Any new module-
specific disables should be appended in the same table at the top of
`.clang-tidy`, not inside the `Checks:` block.
