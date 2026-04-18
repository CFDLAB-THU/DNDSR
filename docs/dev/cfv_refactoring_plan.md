# CFV Module Refactoring Plan {#cfv_refactoring}

**Branch:** `dev/harry_refac1`
**Baseline commit:** `c774b89` (on `dev/harry_refac1`)
**Date:** 2026-04-15

---

## 1. Current State

The CFV module (~8,000 lines in `src/CFV/`) implements physics-agnostic Compact
Finite Volume discretizations. Its core is a two-level class hierarchy:

```
FiniteVolume         (1,262 lines) -- geometric metrics, array factories, device mgmt
    ^  (inheritance)
VariationalReconstruction<dim>  (3,538 lines) -- reconstruction, limiters, base functions
```

Supporting files: `Limiters.hpp` (789 lines), `VRDefines.hpp` (types),
`VRSettings.hpp` / `FiniteVolumeSettings.hpp` (configuration),
`ModelEvaluator` (test physics), `test_FiniteVolume` (benchmark kernels).

Physics is injected via callbacks (`TFBoundary`, `tFGetBoundaryWeight`,
`TFTrans`), which is a clean Strategy pattern. The implementation has
accumulated structural debt that this plan addresses.

### Module Dependency Graph (current)

```
DNDS (Core)
  ^
Geom (Mesh, Elements, Quadrature, Base Functions)
  ^
CFV (FiniteVolume, VariationalReconstruction, Limiters, ModelEvaluator)
  ^           ^
  |       Euler  -- uses full VR + VRDefines types (deep structural dep)
EulerP        -- uses only FiniteVolume (clean composition)
```

---

## 2. Refactoring Phases

### Phase 0: Unit Tests for Correctness (DO FIRST) -- DONE

**Goal:** Establish baseline correctness tests *before* touching any CFV code.
Golden reference values captured from commit `c774b89`.

**Implementation:** C++ doctest in `test/cpp/CFV/test_Reconstruction.cpp`.
CMake target: `cfv_test_reconstruction`. CTest names:
`cfv_reconstruction_np1`, `cfv_reconstruction_np2`, `cfv_reconstruction_np4`.

All iterative VR uses Jacobi iteration (`SORInstead=false`) to ensure
deterministic results across MPI partitions. Golden values are constant
for all np.

**Error metric:** L1 pointwise error at cell quadrature points (intOrder >= 5),
normalized by domain volume, so its dimension matches the field function.

**Meshes:**
- `Uniform_3x3_wall.cgns` (9 quads, wall BC, [-1,2]^2) -- polynomial tests
- `IV10_10.cgns` (100 quads, periodic, [0,10]^2) + bisections 0,1,2
  -> 100, 400, 1600 cells -- convergence series on structured mesh
- `IV10U_10.cgns` (322 tris, periodic, [0,10]^2) + bisections 0,1,2
  -> 322, ~1288, ~5152 cells -- convergence series on unstructured mesh

**Reconstruction methods tested:**
- Gauss-Green (explicit 2nd-order gradient, `DoReconstruction2ndGrad`)
- VFV P1, P2, P3 with HQM weights (dirWeightScheme=HQM_OPT, geomWeightScheme=HQM_SD)
- VFV P1, P3 with default Factorial weights

**Test cases (17 total, 50 assertions):**

| Category | Tests | Assertion |
|----------|-------|-----------|
| Wall/constant (5 methods) | u=1 on wall mesh | L1 err < 1e-12 (exact) |
| Wall/linear (5 methods) | u=x+2y on wall mesh | Sanity check < 10 |
| Wall/quadratic (3 methods) | u=x^2+y^2 | Sanity check < 10 |
| Wall/cubic (2 methods) | u=x^3+xy^2 | Sanity check < 10 |
| Periodic convergence (11 configs x 3 bisections) | sin*cos and cos+cos on IV10/IV10U | Golden value match (1e-6 rel) |
| Convergence check | P2-HQM on IV10 base | Converges in < 200 Jacobi iters |

**Convergence series (golden, representative):**

| Mesh | Method | bis=0 | bis=1 | bis=2 | Rate |
|------|--------|-------|-------|-------|------|
| IV10 | GG | 1.56e-02 | 3.42e-03 | 7.89e-04 | ~2 |
| IV10 | P1-HQM | 4.66e-02 | 9.30e-03 | 1.53e-03 | ~2.5 |
| IV10 | P2-HQM | 3.05e-03 | 2.31e-04 | 2.44e-05 | ~3.5 |
| IV10 | P3-HQM | 1.91e-03 | 4.67e-05 | 1.49e-06 | ~5 |
| IV10 | P3-Def | 1.88e-03 | 2.50e-05 | 8.37e-07 | ~5 |
| IV10U | P3-HQM | 1.59e-04 | 9.88e-06 | 4.34e-07 | ~4.5 |

**Verified passing:** np=1, np=2, np=4. Total CTest time ~10s.

**Build and run:**
```bash
cmake --build build -t cfv_test_reconstruction -j8
ctest --test-dir build -R cfv_reconstruction --output-on-failure
```

### Phase 1: Safety Fixes -- DONE

Fix latent bugs that could cause silent incorrect results.

| # | Issue | Location | Fix |
|---|---|---|---|
| 1.1 | Thread-unsafe `static` local in template | `Reconstruction.hxx:418` | Replace with local variable |
| 1.2 | Raw `new[]/delete[]` in limiter helper | `Limiters.hpp:315-331` | Use `Eigen::VectorXd` or stack array |
| 1.3 | Settings slicing (two divergent copies) | `VR.hpp:44-48` | Single settings instance; base reads via accessor |
| 1.4 | Missing `[[fallthrough]]` annotations | `VR.cpp:386-397, 405-416` | Add annotations |

### Phase 2: Code Deduplication -- DONE (4 of 6 items; 2 skipped)

~480 lines removed. Items 2.2 and 2.3 were skipped after analysis showed the
structural differences between the paired functions are too deep for a clean
merge; forcing unification would reduce readability without meaningful
maintenance benefit.

| # | Duplication | Status | Fix | Lines Saved |
|---|---|---|---|---|
| 2.1 | Periodic transform block (4 copies) | **Done** | `ApplyPeriodicTransform()` variadic template method | ~40 |
| 2.2 | `DoLimiterWBAP_C` vs `DoLimiterWBAP_3` | **Skipped** | Sweep strategies differ fundamentally (successive SR vs. three-sweep) | -- |
| 2.3 | `GetBoundaryRHS` vs `GetBoundaryRHSDiff` | **Skipped** | Diff version has different callback signature, no GG1 block, no u subtraction | -- |
| 2.4 | Polynomial norm in `Limiters.hpp` (8 copies) | **Done** | `PolynomialSquaredNorm<dim>()` + `PolynomialDotProduct<dim>()` | ~300 |
| 2.5 | `LimStart/LimEnd` DOF index lookup (2 copies) | **Done** | `GetRecDOFRange<dim>()` in `VRDefines.hpp` | ~60 |
| 2.6 | Biway limiter dispatch switch (2 copies) | **Done** | `DispatchBiwayLimiter<dim, nVarsFixed>()` | ~30 |

**New helpers introduced:**
- `VRDefines.hpp`: `GetRecDOFRange<dim>(pOrder)` -- returns `{LimStart, LimEnd}` pair
- `Limiters.hpp`: `PolynomialSquaredNorm<dim>(theta)`, `PolynomialDotProduct<dim>(t1, t2)`
- `VariationalReconstruction.hpp`: `ApplyPeriodicTransform(if2c, faceID, data...)` -- variadic CRTP
- `LimiterProcedure.hxx`: `DispatchBiwayLimiter<dim, nVarsFixed>(alter, u1, u2, out, n)`

**Verified:** Phase 0 tests pass at np=1,2,4. Euler target compiles clean.

### Phase 3: Module Boundary Corrections -- DONE (1 of 2 items; 1 deferred)

| # | Item | Status | Action |
|---|---|---|---|
| 3.1 | `ModelEvaluator` (concrete physics model inside CFV) | **Deferred** | Added placement comment; actual move to `src/Model/` blocked by Python binding coupling (`CFV.ModelEvaluator`) |
| 3.2 | `test_FiniteVolume*` (benchmark kernels with `test_` prefix) | **Done** | Renamed to `BenchmarkFiniteVolume*`; updated CMake, includes, pybind11 C++ function names; Python-visible names unchanged |

### Phase 4: Extract `FFaceFunctional` -- DONE

- **File:** `VariationalReconstruction.hpp`
- **What was done:**
  - Extracted `AccumulateDiffOrderContributions<dim, powV>()` free function template
    that replaces 4 copies of the `switch(cnDiffs)` fallthrough pattern (~160 lines removed)
  - Replaced `#define __POWV` / `#undef __POWV` macro with `constexpr int powV`
  - Uses compile-time `Eigen::segment<N>()` instead of dynamic `{6,7,8,9}` initializer-list slicing
  - Replaced `std::pow(faceL, order*2)` with integer multiplications
  - FFaceFunctional body remains inline in `VariationalReconstruction.hpp` (moving
    to separate `.hxx` provides no functional benefit given the above improvements)

### Phase 5: Decompose `VariationalReconstruction` into Composed Sub-Objects -- DONE (2 of 4 items; 2 skipped)

Split the monolithic class's data members into two nested sub-objects:

| Component | Status | Owns |
|---|---|---|
| `VRBaseWeight` | **Done** | `cellBaseMoment`, `faceWeight`, diff-base caches (4), `faceAlignedScales`, `faceMajorCoordScale`, `bndVRCaches` |
| `VRCoefficients` | **Done** | `matrixAB`, `vectorB`, `matrixAAInvB`, `vectorAInvB`, `matrixSecondary`, `matrixAHalf_GG`, `matrixA`, Cholesky caches (4) |
| `VRReconstructor` (free functions) | **Skipped** | Methods already in separate `_Reconstruction.hxx`; making free functions would require rewriting all `_explicit_instantiation/` files for no practical benefit |
| `VRLimiter` (free functions) | **Skipped** | Same reason; already in separate `_LimiterProcedure.hxx` |

`VariationalReconstruction<dim>` holds `baseWeight_` and `coefficients_` members
and delegates through them. The public API is preserved unchanged. All methods
continue to live on the VR class but access data through the sub-objects.

### Phase 6: Separate `FiniteVolume` Concerns -- DONE

| Concern | Current State | Refactored |
|---|---|---|
| Geometric metrics | 18 protected array pairs + `Construct*()` | Keep in `FiniteVolume` (core responsibility) |
| Array factory / DOF builder | `MakePairDefaultOnCell`, `BuildUDof`, etc. | DOF logic extracted to free functions in `DOFFactory.hpp`; FV methods are thin wrappers |
| Device management | `to_host()`, `to_device()`, `device()`, `deviceView()` | Extracted to `DeviceTransferable<Derived>` CRTP mixin (see Phase 6a) |

**Additional improvements:**
- `MakePairDefaultOnCell`/`MakePairDefaultOnFace` now require a `name` parameter (33 call sites updated with descriptive names like `"FV::volumeLocal"`, `"VR::matrixA"`)
- `Array::to_device()`/`to_host()`/`deviceView()` error messages now include the array's object identity
- `ArrayPair::deviceView()` error messages include the array's object identity

#### Phase 6a: Unified Device Management Mixin -- DONE

**Implemented:** `DNDS/DeviceTransferable.hpp` — CRTP mixin providing `to_device()`,
`to_host()`, `device()`, `getDeviceArrayBytes()`. Derived class provides
`for_each_device_member(F&&)`.

**Classes migrated:**
- `CFV::FiniteVolume` — inherits `DeviceTransferable<FiniteVolume>`, delegates via `for_each_device_member` → `device_array_list()`
- `Geom::UnstructuredMesh` — inherits `DeviceTransferable<UnstructuredMesh>`, delegates via `for_each_device_member` → `op_on_device_arrays()` (conditional 4-group iteration preserved)

**Classes skipped (documented rationale):**
- `EulerP::Evaluator`, `BCHandler`, `Physics`, `BC` — use **manual delegation** pattern (calling `to_device` on heterogeneous sub-objects of different types: `shared_ptr<FV>`, `shared_ptr<BCHandler>`, `host_device_vector`, individual `ArrayPair`s). The mixin's tuple-based `MemberRef` iteration doesn't fit without significant restructuring. The manual delegation is appropriate here since these classes compose other `DeviceTransferable` objects.

**CUDA unit tests:** `test/cpp/CFV/test_DeviceTransferable.cu` — 5 tests:
1. `test_device_view_trivially_copyable` — compile-time `static_assert` + host view correctness
2. `test_device_state_tracking` — `device()` returns correct backend through transfer lifecycle
3. `test_host_device_transfer` — Host backend preserves data access
4. `test_cuda_round_trip` — host→CUDA kernel read→host verification of cell volumes, barycenters, face areas
5. `test_cuda_multiple_round_trips` — 3 round-trips with bitwise value stability

### Phase 7: API Hygiene and Encapsulation -- SKIPPED

All three items were evaluated and skipped. The cost/benefit ratio does not
justify the changes given the current codebase state after Phases 0-6+8.

| # | Item | Status | Rationale |
|---|---|---|---|
| 7.1 | `BoundaryContext<nVarsFixed>` param struct | **Skipped** | Stable interface (8/9 params), 6 invocation sites + 8 lambda definitions to update atomically; Euler lambdas lack Phase 0 test coverage; low value for a cosmetic change |
| 7.2 | Const accessors for VR→FV protected members | **Skipped** | 11 protected members accessed directly by VR (23 reads, all read-only except 1 settings write); protected access is the intended C++ pattern for inheritance; only worthwhile if switching to composition, which was already evaluated and skipped in Phase 5 |
| 7.3 | Make `mesh`/`mpi` private | **Skipped** | `mesh->` appears ~897 times across the codebase; making private requires ~60 external call-site changes for a `getMesh()` wrapper that provides no real encapsulation gain (the `shared_ptr` grants full mutability anyway) |

**Detailed findings (preserved for future reference):**

- `TFBoundary`: 8 params `(UBL, UMEAN, iCell, iFace, ig, Norm, pPhy, fType)`.
  `TFBoundaryDiff`: 9 params (adds `dUMEAN` as 2nd arg). Defined in
  `VariationalReconstruction.hpp:885-904`. Invoked in 6 sites (all in
  `_Reconstruction.hxx`). Lambda definitions: 4 in `src/Euler/`, 1 in
  `ModelEvaluator.hpp`, 1 in `EulerSolver.hxx` (diff), 2 in test code.
- VR directly accesses 11 FV protected members (23 total reads). Heaviest:
  `axisSymmetric` (5), `faceMeanNorm` (4), `cellAlignedHBox` (2),
  `cellMajorHBox` (2), `cellMajorCoord` (2). 5 members lack any public
  accessor: `cellAlignedHBox`, `cellMajorCoord`, `periodicity`, `cellCent`,
  `faceCent`. External code (Euler/EulerP) never touches protected members.
- `mesh` is public `ssp<UnstructuredMesh>`; `mpi` is public `MPIInfo` value.
  External `mesh` access: ~25 sites in EulerP (`fv->mesh->`), ~35 in
  tests/app. Euler keeps its own `mesh` shared_ptr copy and never goes through
  `fv->mesh`. External `mpi` access through FV: essentially zero.

### Phase 8: Code Cleanup -- DONE (3 of 4 items; 1 preserved)

Removed ~170 lines of pure debug output across 7 files. Deleted the incomplete
`FWBAP_LE_Multiway` limiter stub. Fixed `bcWeight` condition for periodic faces.

| # | Item | Status | Notes |
|---|---|---|---|
| 8.1 | Delete debug prints (`std::cout`/`printf`/`std::abort`) | **Done** | ~170 lines removed from 7 files; all commented-out experimental/alternative code preserved per policy |
| 8.2 | Replace `#define __POWV`/`#undef` | **Done** | Completed in Phase 4 (`constexpr int powV`) |
| 8.3 | Remove `DNDS_SWITCH_INTELLISENSE` hack | **Preserved** | Useful for IDE support; deliberately kept |
| 8.4 | Delete incomplete `FWBAP_LE_Multiway` | **Done** | Empty TODO body with no callers; 11 lines removed |

**Behavior change:** `ConstructBaseAndWeight()` face weight condition changed
from `FaceIDIsExternalBC(...) || FaceIDIsPeriodic(...)` to
`FaceIDIsExternalBC(...)` only — periodic faces are now treated as internal
faces for weight settings, not assigned `bcWeight`.

---

## 3. Implementation Order and Risk Assessment

| Phase | Risk | Effort | Prerequisite | Status |
|---|---|---|---|---|
| Phase 0: Unit Tests | None | 1 day | -- | **DONE** |
| Phase 1: Safety Fixes | Low | 0.5 day | Phase 0 | **DONE** |
| Phase 2: Deduplication | Low | 2-3 days | Phase 0 | **DONE** (4/6) |
| Phase 3: Module Boundaries | Low | 1 day | Phase 0 | **DONE** (1/2) |
| Phase 4: FFaceFunctional | Medium | 1-2 days | Phase 2 | **DONE** |
| Phase 5: VR Decomposition | Medium-High | 3-5 days | Phase 4 | **DONE** (2/4) |
| Phase 6: FV Separation + Device Mixin | High | 4-6 days | Phase 0 | **DONE** |
| Phase 7: API Hygiene | Medium | 2-3 days | Phase 5, 6 | **SKIPPED** |
| Phase 8: Cleanup | Trivial | 0.5 day | Any time | **DONE** (3/4) |

Each phase must pass the full test suite (`pytest test/` + C++ `ctest`) after
completion. Phase 0 tests serve as the regression guard for all subsequent
phases.

---

## 4. Dependency Graph After Refactoring

```
DNDS (Core)
  DeviceTransferable<T>     -- CRTP mixin for device management
  ^
Geom (Mesh, Elements, Quadrature, Base Functions)
  ^
CFV/
  FiniteVolume              -- geometric metrics (core)
  DOFFactory (free fns)     -- array allocation helpers
  VRBaseWeight              -- base functions, caches, face weights
  VRCoefficients            -- reconstruction matrices
  Limiters (free fns)       -- individual limiter functions
  VariationalReconstruction<dim>  -- facade composing above
  ModelEvaluator            -- advection-diffusion test physics (stays in CFV)
  BenchmarkFiniteVolume     -- benchmark kernels (renamed from test_FiniteVolume)
  ^           ^
  |       Euler (uses full VR facade + DOFFactory)
  |
EulerP (uses only FiniteVolume)
```

---

## 5. Typical VR Configurations (for Testing Reference)

Based on analysis of ~90 case files, two configurations dominate:

**HQM Production** (~85% of real cases):
```json
{
    "maxOrder": 3, "intOrder": 5, "intOrderVR": 5,
    "cacheDiffBase": true, "SORInstead": false, "jacobiRelax": 1.0,
    "smoothThreshold": 0.001, "WBAP_nStd": 10.0,
    "subs2ndOrder": 1, "subs2ndOrderGGScheme": 0,
    "baseSettings": {"localOrientation": false, "anisotropicLengths": false},
    "functionalSettings": {
        "scaleType": "BaryDiff",
        "dirWeightScheme": "HQM_OPT",
        "geomWeightScheme": "HQM_SD",
        "geomWeightPower": 0.5, "geomWeightBias": 1
    }
}
```

**Default Factorial** (~15% of cases, code default):
```json
{
    "maxOrder": 3, "intOrder": 5,
    "SORInstead": true, "jacobiRelax": 1.0,
    "smoothThreshold": 0.01, "WBAP_nStd": 10.0,
    "subs2ndOrder": 0,
    "baseSettings": {"localOrientation": false, "anisotropicLengths": false},
    "functionalSettings": {
        "scaleType": "BaryDiff",
        "dirWeightScheme": "Factorial",
        "geomWeightScheme": "GWNone"
    }
}
```

---

## 6. Final Summary

**Refactoring completed** on branch `dev/harry_refac1` (9 commits from
baseline `c774b89`). All Phase 0 regression tests pass at np=1,2,4.

### Commits

| # | Hash | Phase | Summary |
|---|------|-------|---------|
| 1 | `6537b94` | 0 | 17 test cases, 50 assertions in `test_Reconstruction.cpp` |
| 2 | `5523abe` | 1 | Thread-unsafe static, raw new/delete, `[[fallthrough]]`, settings slicing |
| 3 | `700ed4f` | 2 | `GetRecDOFRange`, `PolynomialSquaredNorm`, `PolynomialDotProduct`, `ApplyPeriodicTransform`, `DispatchBiwayLimiter` |
| 4 | `54d9de9` | 3 | `test_FiniteVolume*` -> `BenchmarkFiniteVolume*`; ModelEvaluator placement comment |
| 5 | `048fe4d` | 4 | `AccumulateDiffOrderContributions` with compile-time `segment<N>()`, integer powers, `constexpr powV` |
| 6 | `959f751` | 5 | `VRBaseWeight` (9 members) + `VRCoefficients` (12 members) nested sub-objects |
| 7 | `217fc09` | 6 | `DeviceTransferable<Derived>` CRTP mixin, DOF factory extraction, named arrays, 5 CUDA tests |
| 8 | `f119cbc` | 8 | ~170 lines debug prints removed, `FWBAP_LE_Multiway` deleted, periodic face weight fix |
| 9 | (this)   | doc | Final documentation with Phase 7 skip rationale |

### What was achieved

- **Safety:** Thread-unsafe statics, raw memory management, missing fallthrough annotations, settings slicing -- all fixed.
- **Deduplication:** ~480 lines removed via 6 extracted helpers.
- **Structure:** VR data split into `VRBaseWeight` + `VRCoefficients`; DOF allocation extracted to `DOFFactory.hpp`; device management unified in `DeviceTransferable<T>` CRTP mixin.
- **Naming:** Benchmark files renamed; 33 array pairs given descriptive names; error messages include object identity.
- **Cleanup:** ~170 lines of debug prints removed; dead stub deleted; periodic face weight bug fixed.
- **Testing:** 17 regression test cases (50 assertions) at np=1,2,4; 5 CUDA device transfer tests.

### What was deliberately skipped (with rationale)

- **Phase 2.2/2.3:** `DoLimiterWBAP_C` vs `_3` and `GetBoundaryRHS` vs `GetBoundaryRHSDiff` -- structural differences too deep for clean merge.
- **Phase 3.1:** ModelEvaluator move to `src/Model/` -- blocked by Python binding coupling; stays in CFV.
- **Phase 5c/5d:** Reconstruction/limiter methods as free functions -- already in separate `.hxx` files; rewriting explicit instantiation infrastructure for no functional benefit.
- **Phase 6a EulerP:** DeviceTransferable mixin for EulerP classes -- manual delegation pattern is appropriate for heterogeneous sub-object composition.
- **Phase 7.1:** `BoundaryContext` param struct -- stable 8-param interface, Euler lambdas lack test coverage, cosmetic-only.
- **Phase 7.2:** Const accessors for VR->FV -- protected access is the intended C++ pattern for inheritance; only worthwhile if switching to composition.
- **Phase 7.3:** Private `mesh`/`mpi` -- ~897 `mesh->` sites; `getMesh()` wrapper provides no real encapsulation gain.
- **Phase 8.3:** `DNDS_SWITCH_INTELLISENSE` hack -- useful for IDE support, deliberately preserved.
