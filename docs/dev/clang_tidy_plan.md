# clang-tidy Cleanup Plan

Living document driving the check-by-check cleanup of the DNDSR C++
tree. Each "pass" below is a unit of work resulting in one commit; each
bucketed check is addressed to completion before the next starts.

Driver: `scripts/run_clang_tidy.py`. Config: `/.clang-tidy` (reporting)
and `/.clang-tidy-fix` (auto-fix profile). See
`docs/guides/style_guide.md` for usage.

## Status snapshot

- Current module: **DNDS** (all others scheduled after DNDS is clean).
- Baseline: `build/clang-tidy-logs/baseline-dnds.txt`.

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
   7. [Pass 7 — bugprone-implicit-widening-of-multiplication-result](#pass-7--bugprone-implicit-widening-of-multiplication-result)
5. [Disables applied](#disables-applied)
6. [Next modules](#next-modules)

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

*TODO*

### Pass 4 — cppcoreguidelines-missing-std-forward

*TODO*

### Pass 5 — bugprone-reserved-identifier

*TODO*

### Pass 6 — cppcoreguidelines-special-member-functions

*TODO*

### Pass 7 — bugprone-implicit-widening-of-multiplication-result

*TODO*

## Disables applied

*TODO: table of checks moved to `.clang-tidy` disables during this effort,
each with one-sentence reason.*

## Next modules

*TODO: Solver, Geom, CFV, Euler, EulerP — order and open questions.*
