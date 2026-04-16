# CFV Module Refactoring Plan

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

### Phase 3: Module Boundary Corrections

| # | Item | Action |
|---|---|---|
| 3.1 | `ModelEvaluator` (concrete physics model inside CFV) | Move to `src/Model/` or `app/CFV/` |
| 3.2 | `test_FiniteVolume*` (benchmark kernels with `test_` prefix) | Rename to `benchmark_*`; conditionally compile when `DNDS_BUILD_TESTS=ON` |

### Phase 4: Extract `FFaceFunctional` (405 inline lines)

- **File:** `VariationalReconstruction.hpp:308-712`
- **Action:** Move to a free function template in `VRFaceFunctional.hxx`. Replace
  the `switch(cnDiffs)` fallthrough pattern with an `if constexpr` chain.
  Eliminate `#define __POWV` / `#undef __POWV` macros.

### Phase 5: Decompose `VariationalReconstruction` into Composed Sub-Objects

Split the monolithic 3,538-line class into:

| Component | Owns | Methods Moved |
|---|---|---|
| `VRBaseWeight` | `cellBaseMoment`, `faceWeight`, diff-base caches, `faceAlignedScales`, `faceMajorCoordScale`, `bndVRCaches` | `ConstructBaseAndWeight()`, `FDiffBaseValue()`, `GetIntPointDiffBaseValue()`, `FFaceFunctional()` |
| `VRCoefficients` | `matrixAB`, `vectorB`, `matrixAAInvB`, `vectorAInvB`, `matrixSecondary`, `matrixAHalf_GG`, `matrixA`, Cholesky caches | `ConstructRecCoeff()`, `GetCellRecMatAInv()`, `GetMatrixSecondary()`, `MatrixAMult()`, `WriteSerializeRecMatrix()` |
| `VRReconstructor` | (stateless) | `DoReconstruction2ndGrad()`, `DoReconstruction2nd()`, `DoReconstructionIter()`, `DoReconstructionIterDiff()`, `DoReconstructionIterSOR()` |
| `VRLimiter` | (stateless) | `DoCalculateSmoothIndicator()`, `DoCalculateSmoothIndicatorV1()`, `DoLimiterWBAP_C()`, `DoLimiterWBAP_3()` |

`VariationalReconstruction<dim>` becomes a facade that holds and delegates to
these components. The public API is preserved.

### Phase 6: Separate `FiniteVolume` Concerns

| Concern | Current State | Refactored |
|---|---|---|
| Geometric metrics | 18 protected array pairs + `Construct*()` | Keep in `FiniteVolume` (core responsibility) |
| Array factory / DOF builder | `MakePairDefaultOnCell`, `BuildUDof`, etc. | Extract to free functions in `DOFFactory.hpp` |
| Device management | `to_host()`, `to_device()`, `device()`, `deviceView()` | Extract to `DeviceTransferable<Derived>` CRTP mixin (see Phase 6a) |

#### Phase 6a: Unified Device Management Mixin

**Current state:** Every composite class (`UnstructuredMesh`, `FiniteVolume`,
`Evaluator`, `BCHandler`, `Physics`) manually implements the same four
methods: `to_device()`, `to_host()`, `device()`, `deviceView()`. Two
implementation patterns exist:

1. **Reflection-based** (Mesh, FV): `device_array_list()` + `for_each_member_list`
2. **Manual delegation** (Evaluator, BCHandler): explicit calls per sub-object

**Proposed unified mixin:**

```cpp
template <typename Derived>
class DeviceTransferable
{
public:
    void to_device(DeviceBackend B)
    {
        static_cast<Derived*>(this)->for_each_device_member(
            [B](auto &v) { v.ref.to_device(B); });
    }

    void to_host()
    {
        static_cast<Derived*>(this)->for_each_device_member(
            [](auto &v) { v.ref.to_host(); });
    }

    DeviceBackend device() const
    {
        DeviceBackend B = DeviceBackend::Unknown;
        bool first = true;
        static_cast<const Derived*>(this)->for_each_device_member(
            [&](const auto &v) {
                auto memberB = device_of(v.ref);
                if (first && memberB != DeviceBackend::Unknown) {
                    B = memberB;
                    first = false;
                }
                // consistency check in debug
                DNDS_assert(memberB == B || memberB == DeviceBackend::Unknown);
            });
        return B;
    }

    size_t getArrayBytes() const { /* sum via for_each_device_member */ }
    void clear_device()         { /* clear via for_each_device_member */ }
};
```

**The Derived class provides** a `for_each_device_member(F&&)` method that
iterates over its device-transferable members. This works for both the
reflection pattern (tuples) and heterogeneous composition (shared_ptrs,
vectors).

**Classes to migrate:**
- `Geom::UnstructuredMesh` -- already uses `device_array_list()`, adapt
- `CFV::FiniteVolume` -- same pattern
- `EulerP::Evaluator` -- manually delegates, adapt to mixin
- `EulerP::BCHandler`, `EulerP::Physics`, `EulerP::BC` -- simple wrappers
- `DNDS::EigenMatrixHolder` -- standalone, adapt

### Phase 7: API Hygiene and Encapsulation

| # | Item | Action |
|---|---|---|
| 7.1 | `TFBoundary` takes 8 params, `TFBoundaryDiff` takes 9 | Introduce `BoundaryContext<nVarsFixed>` param struct |
| 7.2 | VR accesses ~10 protected FV members directly | Add const accessors; VR uses accessors |
| 7.3 | `mesh` and `mpi` are public on `FiniteVolume` | Make private; add forwarding accessors |

### Phase 8: Code Cleanup

| # | Item |
|---|---|
| 8.1 | Delete ~180 lines of commented-out debug code (`.cpp`, `Limiters.hpp`) |
| 8.2 | Replace `#define __POWV`/`#undef` with constexpr lambdas |
| 8.3 | Remove `DNDS_SWITCH_INTELLISENSE` hack from `.hxx` files |
| 8.4 | Delete incomplete `FWBAP_LE_Multiway` (empty TODO body) |

---

## 3. Implementation Order and Risk Assessment

| Phase | Risk | Effort | Prerequisite |
|---|---|---|---|
| Phase 0: Unit Tests | None | 1 day | -- |
| Phase 1: Safety Fixes | Low | 0.5 day | Phase 0 |
| Phase 2: Deduplication | Low | 2-3 days | Phase 0 |
| Phase 3: Module Boundaries | Low | 1 day | Phase 0 |
| Phase 4: FFaceFunctional | Medium | 1-2 days | Phase 2 |
| Phase 5: VR Decomposition | Medium-High | 3-5 days | Phase 4 |
| Phase 6: FV Separation + Device Mixin | High | 4-6 days | Phase 0 |
| Phase 7: API Hygiene | Medium | 2-3 days | Phase 5, 6 |
| Phase 8: Cleanup | Trivial | 0.5 day | Any time |

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
  VRReconstructor           -- reconstruction algorithms
  VRLimiter                 -- limiter procedures
  Limiters (free fns)       -- individual limiter functions
  VariationalReconstruction<dim>  -- facade composing above
  ^           ^
  |       Euler (uses full VR facade + DOFFactory)
  |
EulerP (uses only FiniteVolume)

Model/ (moved from CFV)
  ModelEvaluator            -- advection-diffusion test physics
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
