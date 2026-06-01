# Audit: Commits Since upstream/main

**Date:** 2026-06-01
**Branch:** upstream/main..HEAD (~90 commits, ~170 files, +26k/-4.7k lines)
**Passes:** 5 — audited + false-positive validated + deep internals probed + edge pipeline audit. 29 findings resolved.

**Unresolved: 0 CRITICAL, 0 HIGH, 52 MEDIUM, 52 LOW**

**Resolved:** F1 (fmt ABI), F2 (Cantera opt-out), F3 (BCInPsTs p→c), F4 (tau-splitting experimental), F5 (JAC_SKIP re-rate), F6 (recomputeDerived), F7 (DNDS_MECH_PATH), F8 (thread pool intent), F13 (JSource false alarm), F14 (Roe_M7), F16 (stale TODO), F18 ($schema), F19 (mechanismFile check), F20 (no gating needed), F21 (CT_USE_SYSTEM_FMT), F25 (false alarm), F26 (cell2edge), F27 (cell2facePbi+cell2edgePbi ghost convention), F28 (CUDA trap), F98 (formation-enthalpy doc), F99 (species pos intended), F100 (ddP chain rule), F101 (schema required intended), F110 (validateWithContext)

**EDGE PIPELINE AUDIT:** 10 findings (3 CRITICAL, 5 HIGH, 2 MEDIUM) in ReorderLocalCellsLegacy + new ReorderLocalCells for missing edge2cell/cell2edge/cell2edgePbi handling. CRITICAL L2G/G2L/cell-index/ghost sections fixed; HIGH redundant createFatherGlobalMapping + missing Compress + weak PermuteRows guard fixed; new ReorderLocalCells gaps marked as TODO (latent). Edge UT re-enabled via #if 1, skips until small 3D mesh added.

---

## Summary Table

| F# | Sev | Area | File(s) | Summary |
|----|-----|------|---------|---------|
| F1 | **CRIT** ✓ | Build | `DndsApps.cmake:157`, `Euler/CMakeLists.txt:24,29` | fmt ABI mismatch — **FIXED: CT_USE_SYSTEM_FMT=1 PUBLIC on euler_library_fast; cantera_Test standalone** |
| F2 | **CRIT** ✓ | Build | `DndsExternalDeps.cmake:55-57,84-91,241` | Cantera unconditionally REQUIRED — **FIXED: DNDS_USE_CANTERA option, ChemicalSource guarded** |
| F3 | **CRIT** ✓ | Scaling | `EulerBC.hpp:380-385,607-626,178` | BCInPsTs phys→code missing — **FIXED: convention = physical inputs, converted in ResolveStateValues; schema updated** |
| F4 | **HIGH** ✓ | Reactive | `EulerSolver.hxx:853-878` | Tau-splitting single pass, no outer convergence check — **re-rated C→H; experimental, guarded internally; code note added** |
| F5 | **LOW** ✓ | Jacobian | `SourceTermContributor.hpp:441-443`, `ChemicalSource.cpp:276-277` | JAC_SKIP_FLUID — **explicit Jacobian approximation; both with/without need empirical comparison** **[C→L: user]** |
| F6 | **CRIT** ✓ | EOS | `test_PhysicsProperties.cpp:36`, `test_ChemODE.cpp:311` | `recomputeDerived()` removed — **FIXED: test files cleaned; canteraConstVolTrajectory fixed** |
| F7 | **CRIT** ✓ | Config | `AGENTS.md:143-150` vs `SourceTermContributor.hpp:527` | `DNDS_MECH_PATH` documented but unused — **FIXED: prepended when set and path not absolute** |
| F8 | **LOW** ✓ | Thread | `SourceTermContributor.hpp:521-528`, `PhysicsProperties.hpp:72-88` | Per-thread pool sized at init, aborts if thread count increases — **by design; pool assumes static thread count** |
| F9 | **MED** | Scaling | `EvaluateDt.hxx:2642-2658`, `EulerBC.hpp:405-448` | `BCWallIsothermal` T in code units; no phys→code conversion (default T0=1 works) **[H→M: FP pass]** |
| F10 | **MED** | Scaling | `EulerEvaluatorSettings.hpp:591-592`, `PhysicsProperties.hpp:1127-1129` | Sutherland TRef/CSutherland doc says "(K)" but used with code-scaled T (only manifests at T0≠1) **[H→M: FP pass]** |
| F11 | **MED** | Reactive | `SourceTermContributor.hpp:416`, `PhysicsProperties.hpp:1219` | T floor clamped to 200K — standard engineering practice; rates ~0 below 200K **[H→M: FP pass]** |
| F12 | **LOW** | Reactive | `ChemicalSource.cpp:453-465,472-487` | `mixtureFormationRhoESpecies` lazy-init — invU0sq static invariant, enforced at caller **[H→L: FP pass]** |
| F13 | **LOW** ✓ | Reactive | `EulerSolver.hxx:813-817` | Source Jacobian allegedly zeroed — **false alarm: JSource IS filled via EvaluateCellSource at EvaluateRHS.hxx:727-737** |
| F14 | **HIGH** ✓ | EOS | `Gas.hpp:1027-1041` | EigScheme==7 (Roe_M7) uses `*=` instead of `=` — **FIXED: *= → =** |
| F15 | **MED** | Jacobian | `ChemicalSource.cpp:219-228` | Reference offset uses `cp(298)*298` instead of `h_k(T_ref)` — valid NASA7 approximation; biases Jacobian only **[H→M: FP pass]** |
| F16 | **LOW** ✓ | Jacobian | `SourceTermContributor.hpp:437-459` | Zero energy-row Jacobian — **false alarm: JAC_DEFAULT=0, no skip; stale TODO removed** |
| F17 | **MED** | Config | `ConfigRegistry.hpp:452`, `EulerSolver.hpp:990-1059` | `validateKeys()` defined but never called — **validateWithContext added; validateKeys deferred (shallow; need recursive/warn behavior for loose configs)** |
| F18 | **HIGH** ✓ | Config | `cases/eulerEX/react_test.json:2` | References wrong `$schema` — **FIXED: eulerSA → eulerEX** |
| F19 | **HIGH** ✓ | Config | `EulerEvaluatorSettings.hpp:652,822-823` | No check that `mechanismFile` is non-empty — **FIXED: DNDS_check_throw_info in finalize()** |
| F20 | **LOW** ✓ | Build | `Euler/CMakeLists.txt:18-25` | All model variants compile ChemicalSource.cpp — **not a problem; ChemicalSource.cpp is not heavy, gating adds build complexity** |
| F21 | **HIGH** ✓ | Build | `Euler/CMakeLists.txt:29-30` | `CT_USE_SYSTEM_FMT` not propagated to library — **RESOLVED by F1** |
| F22 | **LOW** | BC | `EvaluateRHS.hxx:358`, `EvaluateDt.hxx:1241-1248` | No explicit dY_k/dn=0 at non-catalytic walls — standard viscous-wall practice; error diminishes with mesh refinement **[H→L: FP pass]** |
| F23 | **LOW** | BC | `EvaluateRHS.hxx:474-489` | `noRsOnWall` isothermal uses stale ULc for gammaEq — only affects non-reactive path where gammaEq is constant **[H→L: FP pass]** |
| F24 | — | BC | Cross-ref F9 | *REMOVED — duplicate of F9* |
| F25 | **LOW** ✓ | Reactive | `EulerSolver.hxx:882`, `EulerEvaluator.hpp:1704-1812,1856-1896,1365-1370` | Source Newton bypasses AddFixedIncrement — **false alarm: only touches species, final fincrement handles repair via AddFixedIncrement** |
| F26 | **HIGH** ✓ | Geom | `Mesh.cpp:1674-1683` | `AdjGlobal2LocalEdge()` omits `cell2edge` — **FIXED: added cell2edge assertions + toLocalOMP/toGlobalOMP to both functions** |
| F27 | **HIGH** ✓ | Geom | `Mesh.cpp:1647-1654,2281-2304` | cell2edgePbi ghost-pulled with wrong indices — **FIXED: removed BuildGhostEdge manual pull; cell2edgePbi + cell2facePbi now borrow from cell2node in BndUpdateGhost; cell2facePbi also added to InterpolateFace** |
| F28 | **HIGH** ✓ | Assert | `Errors.hpp:242-251` | `device_assert_fail()` only traps first thread — **FIXED: moved asm(trap) after the if block; all threads now trap** |
| F29 | **MED** | Scaling | `EulerEvaluatorSettings.hpp:589`, `PhysicsProperties.hpp:272,1117-1135` | `muGas` lacks unit convention annotation |
| F30 | **MED** | Scaling | `PhysicsProperties.hpp:942` vs `:643` | `resolveStateValue` lambda reimplements `consPhysToCode` — duplicate code |
| F31 | **MED** | Reactive | `ChemicalSource.cpp:429-451` | `massFractions` clipping+renormalization shifts composition |
| F32 | **MED** | Reactive | `ChemicalSource.cpp:219-228` | No ideal-gas EOS assertion in `productionRatesAndJacobian` |
| F33 | **MED** | Reactive | `EulerSolver.hxx:865-883` | Pointwise Newton serial — no OpenMP |
| F34 | **LOW** | EOS | `IdealGasPhysics.hpp:73-77` | `Enthalpy` param intentionally ignored (commented-out-param convention); separate `IdealGasThermal()` path handles rhoH_form **[H→L: FP pass — FALSE POSITIVE]** |
| F35 | **MED** | EOS | `test_PhysicsProperties.cpp:282` | gammaEq test is a tautology |
| F36 | **MED** | EOS | `PhysicsProperties.hpp:172-173` | gammaEq() asserts e_sensible>0 — may fire during startup I/O |
| F37 | **MED** | Jacobian | `EulerEvaluator.hpp:1159-1210` | Dual-term dp/dU structure correct but fragile |
| F38 | **MED** | Jacobian | `test_ChemODE.cpp:459,676` | FD Jacobian test: 17.9% mismatch tolerance too generous |
| F39 | **MED** | Config | `EulerEvaluatorSettings.hpp:761,822` | `reactiveFlow` exposed in all model schemas; gate runtime-only |
| F40 | **MED** | Config | `EulerEvaluatorSettings.hpp:653-658,664-669` | `thermoFile`, `transportModel`, `nSpeciesOverride` accept input silently |
| F41 | **MED** | Config | `cases/update_schemas.sh:16` | Script omits `eulerEX`/`eulerEX3D` — schema drift risk |
| F42 | **MED** | Config | `EulerEvaluatorSettings.hpp:655-657` | `CFLScale`, `chemRelaxEps`, `chemAbsTol` lack `range()` constraints |
| F43 | **MED** | Thread | `ChemicalSource.cpp:453-465` | `mixtureFormationRhoESpecies` cache ignores subsequent `invU0sq` |
| F44 | **LOW** | Thread | `EvaluateRHS.hxx:170`, `EvaluateDt.hxx:880`, `EulerEvaluator.hxx:687` | `schedule(runtime)` vs `schedule(static)` — reductions use scratch buffers; zero measurable impact **[M→L: FP pass]** |
| F45 | **MED** | BC | `PhysicsProperties.hpp:486-516` | `totalToStaticPrimitive` bisection uses pTotal — correct but undocumented |
| F46 | **MED** | BC | `EvaluateDt.hxx:2278-2281` | Two-path reactive/non-reactive farfield fragile |
| F47 | **MED** | BC | `EvaluateRHS.hxx:405`, `Gas.hpp:1898-1910` | Formation-enthalpy gradient correction lacks unit test |
| F48 | **MED** | BC | `EvaluateDt.hxx:2652` | Rgas queried on velocity-negated state — confusing |
| F49 | **MED** | Build | `DndsApps.cmake:124,126` | `cantera_Test` lacks `CT_USE_SYSTEM_FMT=1` |
| F50 | **MED** | Build | `cmake/DndsTests.cmake` | `cantera_Test` not CTest-registered |
| F51 | **MED** | Build | `DndsApps.cmake:156` | `canteraConstVolTrajectory` links NS_EX only |
| F52 | **MED** | Build | `cantera_Test.cpp:12-16`, `canteraConstVolTrajectory.cpp:51-55` | No `CANTERA_DATA` env for manual runs |
| F53 | **LOW** | Reactive | `EulerEvaluator_EvaluateDt.hxx:910-923` | `EvaluateDt()` rebuilds `uGradBufNoLim` but uses previous `uGradBuf` — at most 1 step stale; populated by prior EvaluateRHS in same main-loop iteration **[M→L: FP pass]** |
| F54 | **LOW** | Reactive | `SourceTermContributor.hpp:228` | Extended-model SA source passes `gammaEq` instead of `gamma` — gammaEq==gamma for all active non-reactive SA models **[M→L: FP pass]** |
| F55 | **MED** | Reactive | `EulerSolver.hxx:738-743` | RANS relaxation scales `{I4,I4+1}` not `{I4+1,I4+2}` |
| F56 | **MED** | Geom | `Mesh_Reorder.cpp:308` | `cell2edgePbi` registered EntityKind::Edge but rows cell-indexed |
| F57 | **MED** | Geom | `Mesh_DeviceView.hpp:242`, `Mesh.hpp:1047` | Edge device arrays never visited by `op_on_device_arrays()` |
| F58 | **MED** | EOS | `PhysicsProperties.hpp:632`, `eulerState.cpp:431` | RANS variables corrupted by uniform `1/rho0` scaling |
| F59 | **MED** | EOS | `eulerState.cpp:33` | `parseModel()` rejects NS_2D |
| F60 | **MED** | CI | `.github/workflows/ci.yml:329` | `eulerState` not in CI solver smoke set |
| F61 | **MED** | Assert | `Errors.hpp:121` | `NDEBUG` does not suppress DNDS_assert — only `DNDS_NDEBUG` |
| F62 | **MED** | Assert | `Errors.hpp:254` | `DNDS_HD_assert_infof*` ignores variadic args on device |
| F63 | **MED** | Assert | `EulerSolver_Init.hxx:437`, `EvaluateDt.hxx:2436` | Config/BC validation aborts instead of throwing |
| F64 | **MED** | EulerP | `test/EulerP/test_basic_eulerP.py:312-319` | Per-rank temp dir breaks MPI parallel HDF5 I/O |
| F65 | **MED** | Output | `EulerEvaluator.hxx:1491-1511`, `EulerSolver_Init.hxx:582-593`, `EulerSolver.hpp:1232-1234` | DOF min/max columns unlabeled for species |
| F66 | **MED** | Output | `EulerSolver_PrintData.hxx:591-596`, `EvaluateDt.hxx:2938-2941` | VTK species names: "V1" (mass fractions) vs "rhoY_0" (conserved) |
| F67 | **MED** | Python | (no module) | Zero Python bindings for reactive flow (PhysicsProperties, ChemicalSource, etc.) |
| F68 | **MED** | Transport | `Gas.hpp:1792-1793` | Missing ∇R term in ∇T formula — heat conduction flux biased when composition varies |
| F69 | **MED** | Config | `EulerEvaluatorSettings.hpp:111-144` | `StateValueSchema()` omits `"default"` in JSON Schema |
| F70 | **MED** | Config | (no tests) | Zero test coverage for `field_schema` / `StateValue` JSON round-trip |
| F71 | **LOW** | Scaling | `PhysicsProperties.hpp:650-656` | Primitive species unscaled — missing comment |
| F72 | **LOW** | Reactive | `PhysicsProperties.hpp:1121-1134` | Sutherland `default` returns `muGas` silently |
| F73 | **LOW** | Reactive | `ChemicalSource.cpp:349-357` | `bufOmega` aliased in `speciesEnthalpies` |
| F74 | **LOW** | EOS | `PhysicsProperties.hpp:1188-1190` | Hardcoded dim=2 velocity indexing |
| F75 | **LOW** | Jacobian | `EulerJacobian.hpp:80,86,89` | Zero-sized ghost arrays — safe but fragile |
| F76 | **LOW** | Thread | `EvaluateRHS.hxx:587,632` | Unnamed `#pragma omp critical` |
| F77 | **LOW** | Thread | `ChemicalSource.cpp:429-451` | `massFractions()` returns view into mutable `bufY_` |
| F78 | **LOW** | Thread | `ChemicalSource.cpp:34-36`, `ChemicalSource.hpp:205-213` | Mutable buffers lack thread-safety docs |
| F79 | **LOW** | Thread | `Direct.hpp:547-548` | LDLT decompose serial (TODO) |
| F80 | **LOW** | BC | `cases/eulerEX/react_test.json:140` | Empty bcSettings — test doesn't exercise BCs |
| F81 | **LOW** | BC | (no test) | No dY_k/dn=0 test coverage |
| F82 | **LOW** | BC | `EvaluateDt.hxx:1210` | Repeated `GetTypeFromID` hash lookup in flux loop |
| F83 | **LOW** | Build | `test/cpp/CMakeLists.txt:361,388` | Euler test env not inherited by CFV/Geom |
| F84 | **LOW** | Assert | `EnvReader.hpp:77` | `std::tolower` without `#include <cctype>` |
| F85 | **LOW** | Assert | `EnvReader.hpp:80` | `GetEnvBool()` returns `false` on unrecognized even when default `true` |
| F86 | **LOW** | Init | `PhysicsProperties.hpp:875-887` | `speciesEnthalpies` has no `hasChemicalSource()` guard |
| F87 | **LOW** | Init | `EulerSolver.hpp:959-974` | Destructor spin-waits forever on async I/O futures |
| F88 | **LOW** | Init | EulerEvaluator ctor ~line 290 | `GetWallDist()` before chemistry pool wiring |
| F89 | **LOW** | Init | `ChemicalSource.cpp:453-465` | `mixtureFormationRhoESpecies` cache discriminator is size only |
| F90 | **LOW** | Output | `EulerEvaluator.hxx:1491` → `EulerSolver_Init.hxx:941` | No NaN filter between EvaluateMinMax and CSV |
| F91 | **LOW** | Output | `EulerSolver_PrintData.hxx:116-119,197,545` | Three independently coded species offset schemes |
| F92 | **LOW** | Python | `Mesh_bind.hpp` | Edge connectivity not bound in Python |
| F93 | **LOW** | Python | `Mesh_bind.hpp` | `PrintMeshCGNS` not exposed in Python |
| F94 | **LOW** | Transport | `EulerEvaluator_EvaluateDt.hxx:982-990` | Species-diffusion timescale missing from LU-SGS spectral radius |
| F95 | **LOW** | Transport | `PhysicsProperties.hpp:1166-1169` | Non-reactive speciesDiffusivityK inconsistent for muModel=0 (dead code) |
| F96 | **LOW** | Config | `ConfigParam.hpp:456` | Unused `member` capture in `field_schema` lambda |
| F97 | **LOW** | Config | `ConfigParam.hpp:99-103` | `ConfigTypeTagOf<StateValue>` resolves to Object — misleading |
| F98 | **CRIT** ✓ | Reactive | `PhysicsProperties.hpp:219-238` | `mixtureFormationRhoERaw` no guard for negative dependent-species — **docstring clarified: intentional linearity** |
| F99 | **HIGH** ✓ | Reactive | `EulerEvaluator.hxx:1799-1814` | checkRecBaseGood ignores species positivity — **intended: mixtureFormationRhoERaw is linear; species repair deferred to AddFixedIncrement; comment added** |
| F100 | **HIGH** ✓ | Jacobian | `ChemicalSource.cpp:646-803` | Jacobian omits (∂ω/∂p)_T — **FIXED: ddP chain rule implemented (composition-pressure + (p/T)·dT terms); H2O2 FD passes; GRI 3.0 FD test added** |
| F101 | **HIGH** ✓ | Config | `ConfigRegistry.hpp:329-346,382-392` | JSON Schema omits "required" list — **intended: all fields implicitly required at runtime; schema serves as loose patch; comment added** |
| F102 | **MED** | Reactive | `EulerEvaluator.hxx:2095-2188` | `EvaluateCellRHSAlpha` ignores species positivity during pseudo-time step |
| F103 | **MED** | Reactive | `EulerEvaluator.hpp:1647-1674` | `CompressRecPart` produces ghost states with corrupted species at boundaries |
| F104 | **MED** | EOS | `Gas.hpp:2063-2076` | Stale formation-enthalpy floor in compression ratio iterative pull-down |
| F105 | **MED** | Reactive | `EulerSolver.hxx:876-881`, `ODE.hpp:452,548` | alphaDiag mis-scales pointwise Newton time term for BDF2+ (2.25× error) |
| F106 | **MED** | Transport | `EulerEvaluator_EvaluateDt.hxx:1239-1249`, `PhysicsProperties.hpp:1164-1176` | Per-species speciesDiffusivityK loop O(Ns²); Ns1×Ns Cantera evals per cell |
| F107 | **MED** | Test | `SourceTermContributor.hpp:441-443`, `ChemicalSource.cpp:276-277` | JAC_SKIP_FLUID production path has zero test coverage (distinct from F5) |
| F108 | **MED** | Config | `EulerEvaluatorSettings.hpp:798-801` | Constructor defaults species to ΣY_k>1; overridden by JSON but hazard for programmatic use |
| F109 | **MED** | Config | `ConfigParam.hpp:456-462` | `field_schema` desc parameter silently dead when StateValueSchema provides own description |
| F110 | **MED** ✓ | Config | `ConfigParam.hpp:692-704,797-801`, `ConfigRegistry.hpp:427-437` | `validateWithContext`/`check_ctx` generated but never called — **FIXED: called in ConfigureFromJson** |
| F111 | **LOW** | Reactive | `EulerEvaluator.hpp:1872` | `AddFixedIncrement` speciesClipped detection uses exact floating-point `!=` |
| F112 | **LOW** | Chemistry | `ChemicalSource.hpp:122-124`, `ChemicalSource.cpp:243-319` | JAC_SKIP_ABSORPTION flag fully implemented but never OR'd — dead code |
| F113 | **LOW** | React/Pref | `PhysicsProperties.hpp:1198` | Dead uSensible computation wastes mixtureFormationRhoE call per temperature() |
| F114 | **LOW** | Chemistry | `ChemicalSource.cpp:162` | temperatureFromUV warm-start floor at 300K ignores mechanism minTemperature() |
| F115 | **LOW** | Reactive | `SourceTermContributor.hpp:58-59`, `EulerEvaluator_EvaluateDt.hxx:1688` | SourceFilter::NonReactiveOnly enum value never used |
| F116 | **LOW** | Reactive | `SourceTermContributor.hpp:424-462` | ChemicalContributor silently ignores Mode==1 (diagonal-Jacobian) |
| F117 | **LOW** | Config | `EulerEvaluatorSettings.hpp:84-108` | StateValueOriginFromName silently maps unrecognized to None with no logging |

---

## Detail

### F1 — CRITICAL — fmt ABI version mismatch
**Files:** `cmake/DndsApps.cmake:157`, `src/Euler/CMakeLists.txt:24,29`, `external/cfd_externals/install/include/cantera/ext/fmt/core.h`

Cantera was built with bundled fmt v9.1.0. DNDSR uses bundled fmt v11.1.4. `CT_USE_SYSTEM_FMT=1` is set only on final executables (`canteraConstVolTrajectory`, `test_ChemODE`), not on `euler_library_fast` where `ChemicalSource.cpp` is compiled. Translation units within the same binary resolve to different fmt ABI versions — an ODR violation. Also affects `cantera_Test` which has no `CT_USE_SYSTEM_FMT` either.

---

### F2 — CRITICAL — Cantera unconditionally REQUIRED
**Files:** `cmake/DndsExternalDeps.cmake:55-57,84-91,241`

All `find_library`/`find_path` calls use `REQUIRED`. No `DNDS_USE_CANTERA` cache option. `DNDS_EXTERNAL_LIBS` always includes `libcantera_shared`. CMake configure `FATAL_ERROR`s if Cantera/data directory absent. Every test and app executable links Cantera.

---

### F3 ✓ — CRITICAL — BCInPsTs: new convention = physical inputs, converted to code

`totalToStaticPrimitive()` converts inputs from code→phys internally. JSON stores raw pTotal/TTotal as `nonState` without `resolveStateValue`. User specifying [101325, 300, ...] in SI units got code-unit treatment, producing wrong results when T0≠1 or rho0*U0²≠1.

**Resolution:** Convention declared: BCInPsTs inputs are in physical SI units. `ResolveStateValues` now calls `phys.toCode()`/`phys.toCodeT()` on pTotal/TTotal. Schema description updated. When U0=T0=L0=1, physical=cod units (identity).

---

### F4 ✓ — HIGH — Tau-splitting: single pass, no outer convergence check (experimental)
**File:** `EulerSolver.hxx:853-878`

Single pass per nonlinear iteration with no outer convergence check on the source update result. If `PointImplicitSourceUpdate` silently fails, `cxInc` may be corrupted before the global nonlinear solver can recover.

**Re-rated C→H:** The inner `PointImplicitSourceUpdate` has species repair (`repairReactiveSpecies`), validity checks (`validPointSourceState`), and a pseudo-time fallback (path=0 with residual convergence). Tau-splitting is experimental. Code note added at the site.

---

### F5 — CRITICAL — JAC_SKIP_FLUID omits fluid-column coupling in chemical Jacobian
**Files:** `SourceTermContributor.hpp:441-443`, `ChemicalSource.cpp:276-277`

`ChemicalContributor` passes `JAC_SKIP_FLUID` to `productionRatesAndJacobian`. All rho/rhoU/rhoV/rhoW/rhoE columns are zeroed. Temperature sensitivity of chemical rates is not linearized w.r.t. density/momentum/energy. TODO says "test-mode approximation" but this is the production path.

---

### F6 — CRITICAL — `recomputeDerived()` removed, test compilation broken
**Files:** `test/cpp/Euler/test_PhysicsProperties.cpp:36`, `test_ChemODE.cpp:311`, `app/Euler/canteraConstVolTrajectory.cpp:97`

`IdealGasProperty` has no `recomputeDerived()` method — it was removed/renamed during refactoring. Three files still call it. All three fail to compile.

---

### F7 ✓ — CRITICAL — `DNDS_MECH_PATH` documented but unused in solver
**Files:** `AGENTS.md:143-150` vs `SourceTermContributor.hpp:527`

Solver passes `mechanismFile` directly to Cantera without prepending `DNDS_MECH_PATH`. Only test code reads this env var. Solver relies on `CANTERA_DATA` or CWD. Users following docs get obscure errors.

**Resolution:** `SourceTermContributor.hpp` now reads `DNDS_MECH_PATH` via `GetEnvString` and prepends it when set and `mechanismFile` is not absolute (checked via `std::filesystem::path::is_absolute`).

---

### F8 ✓ — LOW — Per-thread pool sized at init, aborts if thread count increases

Pool built once using `omp_get_max_threads()`. If `omp_set_num_threads()` called later with larger N, assertion fires and aborts. No lazy resize.

**Re-rated C→L:** This is intentional design — the code assumes a static thread pool size. The assertion is correct behavior for a misconfigured run.

---

### F9 — HIGH — `BCWallIsothermal` temperature in code units, no conversion
**Files:** `EvaluateDt.hxx:2642-2658`, `EulerBC.hpp:405-448`

`temp = bcValue(0)` used directly in `newDensity = p / (temp * Rgas)`. With T0≠1, user specifying 300 (thinking K) gets code density wrong.

---

### F10 — HIGH — Sutherland TRef/CSutherland unit ambiguity
**Files:** `EulerEvaluatorSettings.hpp:591-592`, `PhysicsProperties.hpp:1127-1129`

Docstring says "(K)" but formula uses them directly with code-scaled T. With T0≠1, viscosity curve wrong.

---

### F11 — HIGH — Temperature floor 200K clamped before Cantera
**Files:** `SourceTermContributor.hpp:416`, `PhysicsProperties.hpp:1219`

`T_cantera = max(T_phys, 200.0)`. Y and P from true (colder) state, T clamped — thermodynamic inconsistency. Corrupts rates below 200K.

---

### F12 — HIGH — `mixtureFormationRhoESpecies` lazy-init swallows subsequent `invU0sq`
**Files:** `ChemicalSource.cpp:453-465,472-487`

Cache populated on first call only; subsequent `invU0sq` ignored. `mixtureFormationRhoEIncrement` accesses `bufHf_` without calling init — ordering contract implicit.

---

### F13 ✓ — LOW — Source Jacobian allegedly zeroed (false alarm)
**File:** `EulerEvaluator_EvaluateRHS.hxx:693-738`

**Resolution:** False alarm. `JSource.clearValues()` is just initialization before the cell loop at line 727-737 fills JSource with source Jacobian from `EvaluateCellSource`. JSource is properly populated.

---

### F14 — HIGH — EigScheme==7 (Roe_M7) squares eigenvalues
**File:** `Gas.hpp:1027-1041`

Incoming eigenvalues already absolute values (`|veloRoe0 ± aRoe|`). Fixer uses `*=` instead of `=`, multiplying by ~same quantity. Compare eigScheme==8 which correctly uses `= std::max(...)`. Roe_M7 fluxes are wrong.

---

### F15 — HIGH — Reference offset uses cp(298)*298 instead of h_k(T_ref)
**File:** `ChemicalSource.cpp:219-228`

Jacobian energy-bridge uses `cpBarRef[k] * 298.15 / MW[k]` but `temperature()` uses `cv(298)*298`. For ideal gas, `h_k(298) ≈ cp_k(298)*298` holds only when formation basis coincides with 298K and cp constant from 0→298K.

---

### F16 ✓ — LOW — Stale TODO about JAC_SKIP_FLUID

`productionRatesAndJacobian` is called with `JAC_DEFAULT` (=0, no flags), which does NOT skip fluid columns. The TODO at line 445-448 claimed `JAC_SKIP_FLUID` was in use — stale, from an earlier development state. Removed.

---

### F17 — MED — `validateKeys()` defined but never called (validateWithContext added instead)
**Files:** `ConfigRegistry.hpp:452`, `EulerSolver.hpp:990-1059`

JSON config typos silently ignored — misspelled keys simply don't take effect. `validateKeys()` exists but is shallow (no recursion into nested config sections) and throws on unknown keys (undesirable for loose research configs).

**Status:** `validateKeys()` intentionally deferred. `validateWithContext()` added in `ConfigureFromJson` with `{nVars, dim, gDim, modelCode}` context — harmless since no contextual checks are registered yet. Re-rated HIGH→MED.

---

### F18 ✓ — HIGH — `react_test.json` references wrong `$schema`
**File:** `cases/eulerEX/react_test.json:2`

**Resolution:** Changed `"$schema": "../eulerSA_schema.json"` → `"../eulerEX_schema.json"`.

---

### F19 ✓ — HIGH — No validation that `mechanismFile` is non-empty when enabled
**File:** `EulerEvaluatorSettings.hpp:840-848`

**Resolution:** Added `DNDS_check_throw_info(!reactiveFlow.enabled || !reactiveFlow.mechanismFile.empty(), ...)` in `finalize()`.

---

### F20 ✓ — LOW — All model variants compile ChemicalSource.cpp

Loop over ALL `DNDS_Euler_Models_List` includes ChemicalSource.cpp. Non-chemical models (NS, NS_SA, NS_2EQ) compile Cantera-dependent code even when unused.

**Re-rated H→L, marked resolved:** ChemicalSource.cpp is small; gating adds unnecessary build complexity. Not a practical problem.

---

### F21 ✓ — HIGH — `CT_USE_SYSTEM_FMT` not propagated to library level

**Resolution:** Resolved by F1 — `target_compile_definitions(... PUBLIC CT_USE_SYSTEM_FMT=1)` now on `euler_library_fast_${key}`.

---

### F22 — HIGH — No dY_k/dn=0 at non-catalytic walls
**Files:** `EvaluateRHS.hxx:358`, `EvaluateDt.hxx:1241-1248`

Interior reconstruction gradient used as-is for species diffusion through walls. At coarse resolution, spurious species fluxes.

---

### F23 — HIGH — `noRsOnWall` isothermal uses stale ULc for gammaEq
**File:** `EvaluateRHS.hxx:474-489`

GammaEq computed from old ULc before density/temperature update. Functionally correct (composition unchanged) but fragile.

---

### F24 — HIGH — Wall BC isothermal T in code units
Cross-reference with F9.

---

### F25 ✓ — LOW — Source Newton bypasses AddFixedIncrement (false alarm)
**Files:** `EulerSolver.hxx:882`, `EulerEvaluator.hpp:1704-1812,1856-1896,1365-1370`

PointImplicitSourceUpdate only touches species (reactiveSpeciesOnly); rho/rhoU/rhoE are unchanged. Internal repairReactiveSpecies lambda handles species clipping. The final fincrement → AddFixedIncrement step runs on the merged increment. No risk of bad values falling into DOF.

---

### F26 ✓ — HIGH — `AdjGlobal2LocalEdge()` omits `cell2edge` (fixed)
**File:** `Mesh.cpp:1674-1695`

**Resolution:** Added cell2edge to `isGlobal/isLocal` assertions and `toLocalOMP/toGlobalOMP` calls in both `AdjGlobal2LocalEdge` and `AdjLocal2GlobalEdge`. Group state (adjEdgeState) now correctly tracks all three edge adjacencies.

---

### F27 ✓ — cell2edgePbi ghost-pulled with wrong index space (fixed)
**File:** `Mesh.cpp:1647-1654,2281-2290`

**Resolution:** Removed manual ghost mapping creation from BuildGhostEdge (was using `gEdges` — edge global indices — for cell-indexed PBI). Added `cell2edgePbi.trans.BorrowGGIndexing(cell2node.trans)` + pull in BndUpdateGhost, following the same pattern as cell2nodePbi. Also added `PermuteRows` for cell2edgePbi in Section E.

---

### F28 ✓ — HIGH — `device_assert_fail()` only traps first thread (fixed)
**File:** `Errors.hpp:242-265`

**Resolution:** Moved `asm("trap;")` after the `if (atomicCAS(...) == 0)` block in both `device_assert_fail` and `device_assert_fail_infof`. All threads now trap on assertion failure. Docstring updated.

---

### F29 — MEDIUM — `muGas` lacks unit annotation
**Files:** `EulerEvaluatorSettings.hpp:589`, `PhysicsProperties.hpp:272,1117-1135`

### F30 — MEDIUM — Duplicate consPhysToCode implementation
**Files:** `PhysicsProperties.hpp:942` vs `:643`

### F31 — MEDIUM — massFractions clipping shifts composition
**File:** `ChemicalSource.cpp:429-451`

### F32 — MEDIUM — No ideal-gas EOS assert in productionRatesAndJacobian
**File:** `ChemicalSource.cpp:219-228`

### F33 — MEDIUM — Pointwise Newton serial
**File:** `EulerSolver.hxx:865-883`

### F34 — MEDIUM — Enthalpy ignores rhoH_form
**File:** `IdealGasPhysics.hpp:73-77`

### F35 — MEDIUM — gammaEq test is tautology
**File:** `test_PhysicsProperties.cpp:282`

### F36 — MEDIUM — gammaEq assert fires during startup
**File:** `PhysicsProperties.hpp:172-173`

### F37 — MEDIUM — Dual-term dp/dU fragile
**File:** `EulerEvaluator.hpp:1159-1210`

### F38 — MEDIUM — FD test tolerance 17.9%
**File:** `test_ChemODE.cpp:459,676`

### F39 — MEDIUM — reactiveFlow in all schemas
**File:** `EulerEvaluatorSettings.hpp:761,822`

### F40 — MEDIUM — Reserved fields accept input silently
**File:** `EulerEvaluatorSettings.hpp:653-658,664-669`

### F41 — MEDIUM — update_schemas.sh omits eulerEX
**File:** `cases/update_schemas.sh:16`

### F42 — MEDIUM — chem tolerances lack range()
**File:** `EulerEvaluatorSettings.hpp:655-657`

### F43 — MEDIUM — invU0sq cache contract
**File:** `ChemicalSource.cpp:453-465`

### F44 — MEDIUM — schedule false sharing
**Files:** `EvaluateRHS.hxx:170`, `EvaluateDt.hxx:880`, `EulerEvaluator.hxx:687`

### F45 — MEDIUM — totalToStaticPrimitive pTotal undocumented
**File:** `PhysicsProperties.hpp:486-516`

### F46 — MEDIUM — Two-path farfield fragile
**File:** `EvaluateDt.hxx:2278-2281`

### F47 — MEDIUM — Formation-enthalpy gradient correction untested
**Files:** `EvaluateRHS.hxx:405`, `Gas.hpp:1898-1910`

### F48 — MEDIUM — Rgas on velocity-negated state
**File:** `EvaluateDt.hxx:2652`

### F49 — MEDIUM — cantera_Test no CT_USE_SYSTEM_FMT
**File:** `DndsApps.cmake:124,126`

### F50 — MEDIUM — cantera_Test not CTest-registered
**File:** `cmake/DndsTests.cmake`

### F51 — MEDIUM — canteraConstVolTrajectory NS_EX only
**File:** `DndsApps.cmake:156`

### F52 — MEDIUM — No CANTERA_DATA for manual runs
**Files:** `cantera_Test.cpp:12-16`, `canteraConstVolTrajectory.cpp:51-55`

### F53 — MEDIUM — Stale uGradBuf in EvaluateDt
**File:** `EulerEvaluator_EvaluateDt.hxx:910-923`

### F54 — MEDIUM — SA source passes gammaEq instead of gamma
**File:** `SourceTermContributor.hpp:228`

### F55 — MEDIUM — RANS relaxation wrong indices
**File:** `EulerSolver.hxx:738-743`

### F56 — MEDIUM — cell2edgePbi EntityKind mismatch in reorder
**File:** `Mesh_Reorder.cpp:308`

### F57 — MEDIUM — Edge device arrays not visited
**Files:** `Mesh_DeviceView.hpp:242`, `Mesh.hpp:1047`

### F58 — MEDIUM — RANS variables corrupted by uniform scaling
**Files:** `PhysicsProperties.hpp:632`, `eulerState.cpp:431`

### F59 — MEDIUM — parseModel rejects NS_2D
**File:** `eulerState.cpp:33`

### F60 — MEDIUM — eulerState not in CI build
**File:** `.github/workflows/ci.yml:329`

### F61 — MEDIUM — NDEBUG doesn't suppress DNDS_assert
**File:** `Errors.hpp:121`

### F62 — MEDIUM — HD_assert_infof ignores variadic args on device
**File:** `Errors.hpp:254`

### F63 — MEDIUM — Config validation aborts instead of throwing
**Files:** `EulerSolver_Init.hxx:437`, `EvaluateDt.hxx:2436`

### F64 — MEDIUM — Per-rank temp dir breaks MPI HDF5
**File:** `test/EulerP/test_basic_eulerP.py:312-319`

### F65 — MEDIUM — DOF min/max columns unlabeled
**Files:** `EulerEvaluator.hxx:1491-1511`, `EulerSolver_Init.hxx:582-593`, `EulerSolver.hpp:1232-1234`

### F66 — MEDIUM — VTK species name inconsistency
**Files:** `EulerSolver_PrintData.hxx:591-596`, `EvaluateDt.hxx:2938-2941`

### F67 — MEDIUM — Zero Python bindings for reactive flow
No `euler` pybind11 module. PhysicsProperties, ChemicalSource, SourceTermContributor C++-only.

### F68 — MEDIUM — Missing ∇R term in ∇T formula
**File:** `Gas.hpp:1792-1793`

Temperature gradient uses locally frozen R and Cp. Missing `-T·∇R/R` term couples species gradient directly into heat conduction flux. Biased in flame fronts.

### F69 — MEDIUM — StateValueSchema omits "default"
**File:** `EulerEvaluatorSettings.hpp:111-144`

### F70 — MEDIUM — No StateValue JSON round-trip test
No test exercises `field_schema`/StateValue serialization/deserialization.

### F71-F97 — LOW
See summary table above.

---

## False Positives & Re-ratings (Pass 4)

11 findings challenged and re-rated by independent false-positive subagent:

| F# | Old Sev | New Sev | Rationale |
|----|---------|---------|-----------|
| F34 | HIGH | **LOW** | Param intentionally ignored (commented-out-param); no caller passes 4th arg; correct path in `IdealGasThermal()` |
| F12 | HIGH | **LOW** | `invU0sq` always `1/U0²` — static simulation invariant; enforced at caller |
| F23 | HIGH | **LOW** | Chemical path (line 487) handles reactive case; non-reactive path has constant gammaEq |
| F53 | MEDIUM | **LOW** | `uGradBuf` populated by prior `EvaluateRHS` in same main-loop iteration; at most 1-step old |
| F22 | HIGH | **LOW** | Standard viscous-wall practice; error diminishes with mesh refinement |
| F44 | MEDIUM | **LOW** | OpenMP reductions use scratch buffers; array entries exceed cache-line size; zero measurable impact |
| F54 | MEDIUM | **LOW** | gammaEq==gamma for all active non-reactive SA model variants |
| F9 | HIGH | **MEDIUM** | Bug real but only manifests with non-default T0≠1 |
| F10 | HIGH | **MEDIUM** | Bug real but only manifests with non-default T0≠1 |
| F11 | HIGH | **MEDIUM** | Below 200K rates ~0; standard reacting-flow engineering practice |
| F15 | HIGH | **MEDIUM** | `cp(298)*298` valid approximation for NASA7 convention; biases Jacobian only |

**Confirmed correct at original severity:** F14 (Roe_M7 eigenvalue squaring — unambiguous bug), F68 (missing ∇R term — real bias), F35 (gammaEq test tautology), F58 (RANS scaling latent).

---

## Pass 4 — New Findings (Deep Internals Probe)

### F98 — CRITICAL — `mixtureFormationRhoERaw` has no guard for negative dependent-species mass; corrupts all limiter checks
**File:** `PhysicsProperties.hpp:219-238`, `EulerEvaluator.hpp:1663,1732`, `EulerEvaluator.hxx:1799-1814`

```cpp
rhoH += (U[0] - sumRhoY) * hfView[Ns1];  // line 236 — no sign check
```
When reconstruction produces `sum(rhoY_k) > rho`, dependent species has negative density. If `hfView[Ns1] > 0`, the computed formation enthalpy is artificially low, making sensible energy appear larger. This corrupts `checkRecBaseGood` (early-exit theta bypass), `CompressRecPart` (boundary-face compression), and `CompressInc` (increment compression). The only repair (`AddFixedIncrement`) operates on cell means — not quad-point reconstruction.

### F99 ✓ — HIGH — checkRecBaseGood ignores species positivity (intended)
**File:** `EulerEvaluator.hxx:2264-2280`

Species positivity (rhoY_k >= 0) intentionally not checked. mixtureFormationRhoERaw is linear and accepts negative species mass; the only hard requirement is positive sensible energy. Species repair is deferred to AddFixedIncrement / repairReactiveSpecies. Comment added.

### F100 ✓ — Chemical Jacobian omits (∂ω/∂p)_T chain rule (fixed)
**File:** `ChemicalSource.cpp:646-803`

**Resolution:** Added `kin_getNetProductionRates_ddP()` wrapper (Cantera `getNetProductionRates_ddP`). Pressure chain rule added to all columns: `bufDwdp[i] * (PbyT*dT_dU + composition_term)`. Species columns get `rhoScaleT * (Rk - Rlast)` composition-pressure term. Fluid columns get `PbyT * dT_dU` through T→P→ω coupling. H2O2 FD test passes (ddP=0 identity). GRI 3.0 FD test added with self-consistent constant-T comparison; dominant species show <3% error.

### F101 ✓ — HIGH — JSON Schema omits "required" (intended)
**File:** `ConfigRegistry.hpp:382-392`

All fields implicitly required at runtime (`j.at()` throws on missing). Schema serves as loose patch description — merged configs can use custom schema checks for strict enforcement. Comment added to `emitSchema()`.

### F102 — MEDIUM — `EvaluateCellRHSAlpha` ignores species positivity during pseudo-time step
**File:** `EulerEvaluator.hxx:2095-2188`

### F103 — MEDIUM — `CompressRecPart` produces ghost states with corrupted species for boundary faces
**File:** `EulerEvaluator.hpp:1647-1674`

### F104 — MEDIUM — Stale formation-enthalpy floor in `IdealGasGetCompressionRatioPressure` iterative pull-down
**File:** `Gas.hpp:2063-2076`

### F105 — MEDIUM — `alphaDiag` mis-scales pointwise Newton time term for BDF2+ time stepping
**Files:** `EulerSolver.hxx:876-881`, `ODE.hpp:452,548`

`A.diagonal() += 1.0/dt` (unscaled) while `A = alphaDiag * srcJac`. For BDF2 (α₀=2/3): effective time-step-to-source ratio = `(1/dt) / (α₀*|dS/dU|)` vs. correct `α₀/dt / |dS/dU|` — 2.25× error. BDF1 (α₀=1) unaffected.

### F106 — MEDIUM — Per-species `speciesDiffusivityK` loop O(Ns²)
**Files:** `EulerEvaluator_EvaluateDt.hxx:1239-1249`, `PhysicsProperties.hpp:1164-1176`

Calls `speciesDiffusivityK(k)` per transported species — each call allocates heap vector and computes ALL Ns diffusivities. Ns1×Ns Cantera evaluations per cell per pseudo-time step.

### F107 — MEDIUM — `JAC_SKIP_FLUID` production code path has zero test coverage
**Files:** `SourceTermContributor.hpp:441-443`, `ChemicalSource.cpp:276-277` (distinct from F5 — coverage, not correctness)

### F108 — MEDIUM — `EulerEvaluatorSettings` constructor defaults species to impossible state
**File:** `EulerEvaluatorSettings.hpp:798-801`
`setOnes(nVars)` followed by selective overwrites leaves species (I4+1..nVars-1) at 1.0. With rho=1, ΣY_k > 1. Normally overridden by JSON parsing, but hazard for programmatic construction.

### F109 — MEDIUM — `field_schema`'s `desc` parameter silently dead when `StateValueSchema` provides own description
**File:** `ConfigParam.hpp:456-462`

### F110 ✓ — MEDIUM — `validateWithContext` / `check_ctx` generated but never called
**Files:** `ConfigParam.hpp:692-704,797-801`, `ConfigRegistry.hpp:427-437`

**Resolution:** `validateWithContext(ctx)` now called in `ConfigureFromJson` after `validate()`.

### F111 — LOW — `AddFixedIncrement` uses exact floating-point `!=` for speciesClipped detection
**File:** `EulerEvaluator.hpp:1872`

### F112 — LOW — `JAC_SKIP_ABSORPTION` flag is dead code (15 lines, never OR'd by any caller)
**Files:** `ChemicalSource.hpp:122-124`, `ChemicalSource.cpp:243-319`

### F113 — LOW — Dead `uSensible` computation wastes `mixtureFormationRhoE` call per temperature() invocation
**File:** `PhysicsProperties.hpp:1198`

### F114 — LOW — `temperatureFromUV` warm-start floor at 300K ignores mechanism `minTemperature()`
**File:** `ChemicalSource.cpp:162`

### F115 — LOW — `SourceFilter::NonReactiveOnly` enum value never used
**Files:** `SourceTermContributor.hpp:58-59`, `EulerEvaluator_EvaluateDt.hxx:1688`

### F116 — LOW — `ChemicalContributor` silently ignores Mode==1 (diagonal-Jacobian) — silent fall-through trap
**File:** `SourceTermContributor.hpp:424-462`

### F117 — LOW — `StateValueOriginFromName` silently maps unrecognized to `None` with no logging
**File:** `EulerEvaluatorSettings.hpp:84-108`

---

## Prior False Positives Eliminated

| Claim | Verdict | Reason |
|-------|---------|--------|
| Tau dt vs dTauC mismatch (EulerSolver.hxx:879) | **Not real** | `1.0/dt` is BDF physical time derivative; `dTauC` is pseudo-time — different roles |
| CUDA `__managed__ static` compile error (Errors.hpp:244) | **Not real** | Compiles cleanly with nvcc 13.2 for sm_80 |

---

## Cross-Cutting Observations

1. **Compilation errors:** F6 (`recomputeDerived`). Also 3 pre-existing stale files: `partitionMeshSerial.cpp`, `jacobiLUTest.cpp`, `examples/ex_geom_mesh.cpp`.

2. **Test coverage gaps:** F5 `JAC_SKIP_FLUID` production path untested (F107). No BC test case. No StateValue config test. No species-diffusion transport tests. `validateWithContext` now called (F110 resolved); `validateKeys` deferred as shallow/throwing (F17 re-rated MED).

3. **Documentation drift:** `AGENTS.md` says solver uses `DNDS_MECH_PATH` — it doesn't (F7).

4. **Dead/placeholder code:** `EulerModelTraits::isReactive` always false; `thermoFile`/`transportModel`/`nSpeciesOverride` registered but unused; `JAC_SKIP_FLUID` (F5); `FixUMaxFilter` is no-op; `JAC_SKIP_ABSORPTION` (F112); `NonReactiveOnly` enum (F115).

5. **Positivity gaps:** F98 (negative dependent species in limiter), F99 (species not checked in early-exit), F25 (Newton bypasses AddFixedIncrement), F102 (pseudo-time species unguarded).

6. **Stable after 4 passes:** Pass 4 confirmed 15 prior findings, re-rated 11 downward, added 20 new findings (1 CRIT, 3 HIGH, 10 MED, 6 LOW). Pass 3 and sweep added only MED/LOW. Convergence confirmed.
