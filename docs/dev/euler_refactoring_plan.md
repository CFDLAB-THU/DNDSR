# Euler Module Refactoring Plan {#euler_refactoring}

**Branch:** (current)
**Date:** 2026-04-18

---

## 1. Current State

The Euler module (~16,500 lines in `src/Euler/`) implements compressible
Navier-Stokes solvers with Spalart-Allmaras, k-omega SST, k-omega Wilcox,
and Realizable k-epsilon turbulence models. A separate GPU-capable module
(`src/EulerP/`, ~2,400 lines) reimplements base NS with CUDA support but
lacks RANS, viscous flux, and working BCs.

### Architecture

The module is template-parameterized on `EulerModel model` (an enum with 9
variants). Two main classes are instantiated for all 9:

```
EulerEvaluator<model>   -- physics evaluator (RHS, time step, LUSGS, wall dist)
EulerSolver<model>      -- top-level driver (init, time march, I/O)
```

Supporting types: `BoundaryHandler<model>`, `EulerEvaluatorSettings<model>`,
`ArrayDOFV<nVarsFixed>`, `ArrayRECV<nVarsFixed>`, `ArrayGRADV<nVarsFixed,gDim>`,
`JacobianDiagBlock<nVarsFixed>`, `JacobianLocalLU<nVarsFixed>`.

### Model Variants

| Model      | nVarsFixed | dim | gDim | Description                |
|------------|-----------|-----|------|----------------------------|
| NS         | 5         | 3   | 2    | 3D physics on 2D mesh      |
| NS_2D      | 4         | 2   | 2    | True 2D                    |
| NS_3D      | 5         | 3   | 3    | Full 3D                    |
| NS_SA      | 6         | 3   | 2    | + Spalart-Allmaras (2D)    |
| NS_SA_3D   | 6         | 3   | 3    | + Spalart-Allmaras (3D)    |
| NS_2EQ     | 7         | 3   | 2    | + 2-equation RANS (2D)     |
| NS_2EQ_3D  | 7         | 3   | 3    | + 2-equation RANS (3D)     |
| NS_EX      | Dynamic   | 3   | 2    | Extended/dynamic nVars     |
| NS_EX_3D   | Dynamic   | 3   | 3    | Extended/dynamic nVars 3D  |

Explicit instantiation produces 54 `.cpp` files (9 models x 6 groups).

---

## 2. Identified Refactoring Points

### HIGH priority

| # | Finding | Location | Impact |
|---|---------|----------|--------|
| 1 | Roe-average preamble duplicated 4x | `Gas.hpp` ~280,415,680,860 | 4 functions share ~40 identical lines; `scaleHartenYee=0.05` declared 4x |
| 2 | 36+ `if constexpr` model-dispatch chains | `EulerEvaluator*.hpp/hxx` | Adding a model requires editing 36+ sites |
| 3 | LUSGS Forward/Backward/SGS copy-paste | `EulerEvaluator.hxx` 317-604 | Same flux-Jacobian product call appears 7x |
| 4 | `GetWallDist()` 800-line monolith | `EulerEvaluator_EvaluateDt.hxx` 21-813 | 4 algorithms in one function, dead code, magic numbers |

### MEDIUM priority

| # | Finding | Location | Impact |
|---|---------|----------|--------|
| 5 | `generateBoundaryValue()` 660-line if-else | `EulerEvaluator_EvaluateDt.hxx` 1662-2327 | Hardcoded test data, no extensibility |
| 6 | Dual Riemann solver dispatchers | `Gas.hpp` 992-1090 | Adding a solver requires updating both |
| 7 | 56 TODO/FIXME comments, 4 stub methods | Throughout | `JacobianValue` has 4 unimplemented methods |
| 8 | Magic numbers scattered across files | Throughout | `0.05`, `0.2`, `800`, `1e-6`, `7.1`, `16` |
| 9 | Euler vs EulerP physics duplication ~400 lines | `Gas.hpp` vs `EulerP_ARS.hpp/Physics.hpp` | Formulas must stay in sync |

### LOW priority

| #  | Finding | Location | Impact |
|----|---------|----------|--------|
| 10 | Operator forwarding boilerplate | `Euler.hpp` 24-333 | ~30 trivial forwards in 3 array classes |
| 11 | Macro-based Eigen sequence injection | `Euler.hpp` 11-21 | Invoked at top of every method |

---

## 3. Refactoring Phases

### Phase 1: Quick Wins (Low Risk, Localized)

**Goal:** Reduce duplication and improve readability in `Gas.hpp` and
`Euler.hpp` without changing any public API or template structure.

**Validation:** All 6 Euler CTest tests must pass after each change:
```bash
cmake --build build -t euler_test_gas_thermo euler_test_riemann_solvers euler_test_rans euler_test_evaluator_pipeline -j32
ctest --test-dir build -R euler_ --output-on-failure
```

#### 1a. Centralize Roe preamble in Gas.hpp

Unify the four Riemann flux functions (`HLLEPFlux_IdealGas`,
`HLLCFlux_IdealGas_HartenYee`, `RoeFlux_IdealGas_HartenYee`,
`RoeFlux_IdealGas_HartenYee_Batch`) to use the existing `GetRoeAverage()`
helper or a new lightweight struct for the shared preamble (L/R velocity
extraction, IdealGasThermal, Roe-averaged state).

#### 1b. Centralize magic numbers in Gas.hpp

Replace the four `static real scaleHartenYee = 0.05` and three
`static const real scaleLD = 0.2` declarations with a single set of
named constants at namespace scope.

#### 1c. Centralize Euler model constants

Move SA model constants (`cnu1=7.1`, `cn1=16`) and wall-omega coefficient
(`800`) from inline literals to named constants.

#### 1d. Replace operator forwarding with using declarations

In `ArrayDOFV`, `ArrayRECV`, `ArrayGRADV` replace trivial forwarding
operators (e.g., `operator+=(t_self&)`) with `using t_base::operator+=`.
Keep only operators that add behavior.

#### 1e. Move Eigen sequence macro to class-scope constants

Replace `DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS` macro invocations
with `static constexpr` or `static const` definitions at class/namespace
scope. (Deferred if Eigen `seq` objects are not trivially constexpr.)

---

### Phase 2: Medium Effort, High Value

**Goal:** Extract helpers to reduce dangerous duplication where a bug fix
must be replicated 7x.

#### 2a. Extract LUSGS off-diagonal flux helper

Create `computeOffDiagFluxInc()`. Merge `UpdateLUSGSForward` and
`UpdateLUSGSBackward` into a single direction-parameterized function.

#### 2b. Split `GetWallDist()` into 3 functions

`GetWallDist_AABB()`, `GetWallDist_Batched()`, `GetWallDist_Poisson()`.
Remove dead commented-out MPI code. Extract shared CGAL triangle setup.

---

### Phase 3: Structural (High Effort)

**Goal:** Long-term maintainability through traits-based dispatch,
strategy patterns, and a shared physics library.

#### 3a. Introduce `RANSModelTraits<model>`

Encode `nRANSVars`, positivity constraints, and flux behavior as traits.
Replace per-model `if constexpr` chains with trait-based generic code.

#### 3b. BC strategy pattern

Refactor `generateBoundaryValue()` into per-BC-type handlers. Move
test-case boundary data to configuration.

#### 3c. Shared gas physics library

Extract ~400 lines of shared thermodynamics and Roe decomposition into
`src/Common/GasPhysics.hpp` with `DNDS_DEVICE_CALLABLE` annotations.
Both Euler and EulerP consume the common implementation.

---

## 4. EulerP Comparison Notes

| Criterion | Euler | EulerP | Winner |
|---|---|---|---|
| Feature completeness | Full NS + RANS + viscous | NS-only, inviscid Roe only | Euler |
| GPU support | None | CUDA via `DeviceBackend` | EulerP |
| Python bindings | None | Full pybind11 | EulerP |
| Code architecture | Monolithic templates, 54 inst. files | Clean kernel/impl/evaluator layers | EulerP |
| Build scalability | O(models x functions) | O(backends) | EulerP |
| Arg passing | Ad-hoc member variables | CRTP Arg structs with validation | EulerP |

EulerP has the better architecture but is feature-incomplete. The top
cross-module win is extracting shared gas physics (Phase 3c).

---

## 5. Completed Work Log

### Phase 1 (completed 2026-04-18)

**1a+1b: Roe preamble and constants dedup** (`Gas.hpp`)
- Added `RoePreamble<dim>` struct and `ComputeRoePreamble()` function.
- Three scalar Riemann solvers (HLLEP, HLLC, Roe) now call `ComputeRoePreamble`
  instead of duplicating ~22 lines of Roe averaging.
- Added `kScaleHartenYee`, `kScaleLD`, `kScaleHFix` namespace-scope constants,
  replacing 4+3 duplicate `static` local declarations.
- `Roe_EntropyFixer` uses the namespace constants.

**1c: SA/RANS constants** (`RANS_ke.hpp`, `EulerEvaluator.hpp`, `.hxx` files)
- Added `RANS::SA::cnu1`, `RANS::SA::cn1`, `RANS::SA::sigma` constants.
- Added `RANS::kWallOmegaCoeff = 800.0`.
- Migrated 4 sites across 3 files.

**1d, 1e: Cancelled** (operator forwarding and Eigen macro -- too risky for benefit).

### Phase 2 (completed 2026-04-18)

**2a: LUSGS/SGS unification** (`EulerEvaluator.hpp`, `EulerEvaluator.hxx`)
- Added `bool uIncIsZero = false` to `UpdateSGS`. When true, skips flux
  computation for not-yet-processed neighbours (LUSGS optimisation).
- Marked `UpdateLUSGSForward` and `UpdateLUSGSBackward` as `[[deprecated]]`.
- Migration path: replace Forward+pull+Backward with two `UpdateSGS` calls
  (forward then backward) with `uIncIsZero=true` and separate buffers.

**2b: Split GetWallDist** (`EulerEvaluator_EvaluateDt.hxx`, `EulerEvaluator.hpp`)
- Split 800-line monolith into 5 private helpers:
  `GetWallDist_CollectTriangles`, `GetWallDist_AABB`, `GetWallDist_BatchedAABB`,
  `GetWallDist_Poisson`, `GetWallDist_ComputeFaceDistances`.
- `GetWallDist()` is now a 10-line dispatcher.
- Removed ~50 lines of dead commented-out MPI code.
- Eliminated duplicate triangle collection between AABB and BatchedAABB.

### Phase 3a (completed 2026-04-18)

**EulerModelTraits** (`Euler.hpp`, all Evaluator files)
- Added `EulerModelTraits<model>` struct in `Euler.hpp` with:
  `hasSA`, `has2EQ`, `hasRANS`, `nRANSVars`, `isExtended`, `isPlainNS`,
  `isGeom2D`, `isGeom3D`, plus forwarded `nVarsFixed`, `dim`, `gDim`.
- Added `using Traits = EulerModelTraits<model>` in `EulerEvaluator` and
  `EulerEvaluatorSettings`.
- Migrated 35 `if constexpr` dispatch sites across 5 files to use traits:
  - `Traits::hasSA` (14 sites)
  - `Traits::has2EQ` (13 sites)
  - `Traits::hasRANS` (1 site)
  - `Traits::isExtended` (1 site)
  - `Traits::isPlainNS` (1 site)
- Test-case-specific initializer dispatch (5 sites) kept as direct model
  checks since they are problem-specific, not model-trait-based.

### Phases 3b, 3c: Remaining

### Phase 3b (completed 2026-04-18)

**BC strategy pattern** (`EulerEvaluator_EvaluateDt.hxx`, `EulerEvaluator.hpp`)
- Split 570-line `generateBoundaryValue` into a dispatcher + 7 handlers:
  `generateBV_FarField`, `generateBV_SpecialFar`, `generateBV_InviscidWall`,
  `generateBV_ViscousWall`, `generateBV_Outflow`, `generateBV_Inflow`,
  `generateBV_TotalConditionInflow`.
- Removed dead `BCFar` check in `BCIn` branch.
- Used `Traits::hasSA`/`has2EQ` in viscous wall handler.

### Phase 3c (completed 2026-04-18)

**Shared gas physics library** (`DNDS/IdealGasPhysics.hpp`)
- Created `src/DNDS/IdealGasPhysics.hpp` with `DNDS_DEVICE_CALLABLE` functions
  in a new `DNDS::IdealGas` namespace:
  - `IdealGasThermal` (p, a², H from conservative state)
  - `Pressure_From_InternalEnergy` / `InternalEnergy_From_Pressure`
  - `Enthalpy` / `SpeedOfSoundSqr`
  - `Cons2PrimEnergy<PrimVariable>` / `Prim2ConsEnergy<PrimVariable>`
  - `PrimE2Pressure<PrimVariable>`
  - `RoeSpeedOfSoundSqr` / `RoeAlphaDecomposition`
  - `EntropyFix_HCorrHY`
  - Constants: `kScaleHartenYee`, `kScaleLD`, `kScaleHFix`
- Introduced `PrimVariable` enum (`Pressure` vs `InternalEnergy`) to
  parameterize conservative-primitive conversion.
- Migrated `Euler/Gas.hpp`:
  - `IdealGasThermal` delegates to shared function
  - Constants delegate to shared `IdealGas::` constants
- Migrated `EulerP/EulerP_Physics.hpp`:
  - `Prim2Pressure` -> `IdealGas::Pressure_From_InternalEnergy`
  - `Pressure2Enthalpy` -> `IdealGas::Enthalpy`
  - `Prim2GammaAcousticSpeed` -> `IdealGas::SpeedOfSoundSqr`
- Migrated `EulerP/EulerP_ARS.hpp`:
  - `RoeEigenValueFixer` -> `IdealGas::EntropyFix_HCorrHY`
  - `RoeAverageNS` -> `IdealGas::RoeSpeedOfSoundSqr`
  - `RoeFluxFlow` alpha decomposition -> `IdealGas::RoeAlphaDecomposition`
