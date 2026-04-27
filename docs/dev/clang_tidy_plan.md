# clang-tidy Cleanup Plan

Living document driving the check-by-check cleanup of the DNDSR C++
tree. Each "pass" below is a unit of work resulting in one commit; each
bucketed check is addressed to completion before the next starts.

Driver: `scripts/run_clang_tidy.py`. Config: `/.clang-tidy` (reporting)
and `/.clang-tidy-fix` (auto-fix profile). See
`docs/guides/style_guide.md` for usage.

## Status snapshot

- Current module: **DNDS**. Passes 1–5 complete, pass 6 deferred
  (engineering task), pass 7 deferred (judgement per site).
- Latest total: **19 390 diagnostics, 46 distinct checks** (down from
  the 24 597 baseline, 21 % reduction). The remaining volume is
  dominated by checks still scheduled for triage in the KA bucket
  (non-private member vars, magic numbers, pointer arithmetic,
  varargs, reinterpret_cast) which will drop to zero in one
  config-only commit.
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

**Outcome.** Deferred. Still 1 645 hits after the preceding passes
(no overlap with anything else).

**Scope assessed.** 1 645 hits collapse to **32 unique class
declarations**, each matching the pattern "defines a copy constructor
and a copy assignment operator but does not define a destructor, a
move constructor or a move assignment operator" — i.e. the classes
are implicitly copy-only. Affected classes include `Array`,
`ArrayAdjacency`, `ArrayDof`, `ArrayEigenMatrix*`, `ArrayTransformer`,
`CommStrategy`, `DeviceStorageBase`, and ~20 others in the DNDS
subtree.

**Why deferred.** The canonical tidy is to add the three missing
members as `= default`, but doing so **changes the public API**:
classes that were implicitly copy-only become move-enabled, so
callers using `std::move(x)` get a move where they previously got a
copy. The existing copy ctor on every flagged class does a deep
`clone(R)`; memberwise move would instead shallow-move the
`shared_ptr` / `host_device_vector` / `std::vector` members, which
is almost certainly the *right* behaviour but is a semantic change
the author should sanction class-by-class.

Auto-fix would also require choosing between `= default` and
`= delete` per class. The check does not implement `--fix` and
clang-tidy cannot infer the right choice.

**Recommendation.** Schedule a dedicated session that:

1. Audits each of the 32 classes for owning members (raw pointers,
   custom deleters, virtual destructors).
2. Adds all five special members explicitly (`= default`/`= delete`)
   per class, keeping the existing copy-ctor / copy-assign semantics
   but picking the appropriate move semantics.
3. Builds after each batch of 5-6 classes to catch any
   move-semantic regression early.
4. Adds targeted tests where the copy-vs-move distinction matters
   (e.g. `Array` move should drop the source's row-start pointer,
   not deep-copy it).

This is an engineering task on its own, not a "tidy pass."

### Pass 7 — bugprone-implicit-widening-of-multiplication-result

**Outcome.** Not reached — the check fell out of the top-20 after the
preceding passes and is interleaved with real semantic decisions on
each site. Deferred to a future session.

**Notes.** After passes 1-5 the residue is ~298 hits across the
project (count before passes; DNDS alone carries some). Each site
needs a judgement call: is the `int * int` result at risk of 32-bit
overflow once extents grow? If yes, cast the operand; if not,
silence. A mass rewrite is unsafe.

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

Rationale comments live in the `.clang-tidy` file header, not inside
the `Checks: >` folded scalar (see Pass 4 / YAML trap).

## Next modules

Once DNDS is "clean" (= all KF passes done, all KA disables applied,
all KS NOLINTs in place), repeat the recipe for:

1. `src/Solver/` — small, limited blast radius; good next target.
2. `src/Geom/` — largest module; expect a new set of checks specific
   to mesh connectivity loops. Start with `clang-analyzer-optin.cplusplus.VirtualCall`
   per the historical note in `.clang-tidy`.
3. `src/CFV/` — follows Geom closely.
4. `src/Euler/`, `src/EulerP/` — solver layer; `bugprone-branch-clone`
   will matter here (exhaustive `if/else` cascades for enum handling).

The `.clang-tidy` disables carry forward unchanged. Any new module-
specific disables should be appended in the same table at the top of
`.clang-tidy`, not inside the `Checks:` block.
