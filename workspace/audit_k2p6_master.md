# k2p6 Audit Master Report — All 11 Categories

## SEVERE (18 findings)

| # | Audit | File:Line | Issue | Status |
|---|-------|-----------|-------|--------|
| 1 | 2 | `IdealGasPhysics.hpp:97-99` | `Cons2PrimEnergy<Pressure>` ignores `rhoH_form` — returns `p = (γ-1)*(E-KE)` without subtracting formation | **DONE** |
| 2 | 2 | `IdealGasPhysics.hpp:118-119` | `Prim2ConsEnergy<Pressure>` omits `rhoH_form` — reconstructed `E` lacks formation offset | **DONE** |
| 3 | 2 | `Gas.hpp:253` | `ComputeRoePreamble`: `asqrRoe = (γ-1)*(HRoe - ½v² - rhoH_roe/ρ)` double-subtracts rhoH_form because HRoe already excludes it | **DONE** |
| 4 | 2 | `Gas.hpp:1184` | `GetRoeAverage`: same double-subtraction bug in `asqrRoe` | **DONE** |
| 5 | 2 | `Gas.hpp:1345` | `RoeFlux_IdealGas_HartenYee_Batch`: same double-subtraction bug in `asqrRoe` | **DONE** |
| 6 | 2 | `Gas.hpp:1191` | `GetRoeAverage` reconstructs `UOut(I4) = p/(γ-1) + KE`, omitting `rhoH_roe` | **DONE** |
| 7 | 2 | `Gas.hpp:473` | `IdealGasUIncrement` takes `rhoH_form` but `(void)`s it — dp correction missing | **DONE** |
| 8 | 5 | `EulerEvaluator.hxx:2251-2285` | `EvaluateCellRHSAlphaExpansion`: `nLimLocal` never incremented; `nLim` always 0 | **DONE** |
| 9 | 5 | `EulerEvaluator.hxx:2254-2265` | `EvaluateCellRHSAlphaExpansion`: iterates ALL cells, overwrites `cellRHSAlpha` with `alphaMin` whenever full inc violates positivity — destroys already-limited cells | **DONE** |
| 10 | 5 | `EulerEvaluator.hxx:2212-2213` | `EvaluateCellRHSAlphaExpansion` ignores `ppEpsIsRelaxed` — always uses fixed thresholds | **DONE** |
| 11 | 6 | `PhysicsProperties.hpp:58,62-65` | `useOMP()`: `pool_->size()>1` gate — when pool size=1 in OMP region, all threads race on `pool[0]` | **DONE** |
| 12 | 6 | `SourceTermContributor.hpp:360-368` | `ChemicalContributor::threadIdx()`: same `pool_->size()>1` race condition | **DONE** |
| 13 | 7 | `PhysicsProperties.hpp:364` | `temperature()` subtracts only `pVAtReference` — missing 0K→298K sensible energy offset; systematic ~298K T overestimate | **DONE** |
| 14 | 8 | `EulerEvaluator.hpp:162` | `int nVars = 5` default — wrong for NS_2D (4), NS_SA (6), NS_2EQ (7) | **FALSE_ALARM** (always set by constructor; default changed to -1 for safety) |
| 15 | 8 | `SourceTermContributor.hpp:37` | `constexpr int ExDim = 3` — hardcodes dim=3 for ALL source contributors, OOB on NS_2D | **DONE** |
| 16 | 8 | `EulerEvaluator.hpp:1112` | `fluxJacobian0_Right`: hardcoded 3D/5-var indices; runtime assert on compile-time constant | **DONE** (removed — dead code; the actually-used `_Times_du`/`_Times_du_AsMatrix` wrappers are dim-agnostic) |
| 17 | 9 | `EulerEvaluator.hxx:1327-1328` | `InitializeUDOF`: double-counts `rhoH_form` for exprtk-initialized cells | **DONE** |
| 18 | 11 | `ChemicalSource.cpp:210-212` | Jacobian `du = (hRT[k]-1)*Ru*T/M_k` — ideal-gas u_k formula; should use Cantera `getPartialMolarIntEnergies` | **DONE** |

## MEDIUM (24 findings)

| # | Audit | File:Line | Issue |
|---|-------|-----------|-------|
| 1 | 1 | `ChemicalSource.cpp:385` | `mixtureFormationRhoE(double rho, Y)` dead method — missing `invU0sq` scaling |
| 2 | 2 | `IdealGasPhysics.hpp:68-70` | `Enthalpy(E,rho,p)` omits `rhoH_form` parameter |
| 3 | 2 | `Gas.hpp:1738` | `GradientCons2Prim_IdealGas` skips pHf correction for single-species (hfSpecies.size()==1) |
| 4 | 2 | `Gas.hpp:534,561` | Eigenvector helpers pass sensible H to standard eigenvectors (offset by h_form) |
| 5 | 4 | `EvaluateDt.hxx:1456-1458` | Rotated Riemann scheme overwrites blended eigenvalues with N1-direction values |
| 6 | 4 | `Gas.hpp:1634-1635` | `ViscousFlux_IdealGas` ∇T formula assumes constant R_mix; missing ∇R_mix term |
| 7 | 5 | `EulerEvaluator.hxx:1918-1927` | Species reconstruction scaled by `uRecBeta` (θ1·θP) instead of `θ1` — over-compressed |
| 8 | 5 | `EulerEvaluator.hxx:1780-1796` | `checkRecBaseGood()` lacks species positivity check |
| 9 | 5 | `EulerEvaluator.hxx:1931-1948` | Validation loop lacks species bounds assertion |
| 10 | 5 | `EulerEvaluator.hxx:2092-2096` | `EvaluateCellRHSAlpha` `ppEpsIsRelaxed` zeroes thresholds instead of using field minima |
| 11 | 5 | `EulerEvaluator.hpp:1739-1742` | `CompressInc` `ppEpsIsRelaxed` uses `verySmallReal` instead of field minima |
| 12 | 5 | `EulerEvaluator.hpp:1755-1768` | `CompressInc` uses `rhoH_form_old` instead of `rhoH_form_new` in quadratic energy solve |
| 13 | 5 | `EulerEvaluator.hxx:2110-2141` | `EvaluateCellRHSAlpha` limits density only — species increments can go negative |
| 14 | 6 | `PhysicsProperties.hpp:375-376` | `static int cnt` in `temperature()` — unprotected race in OMP |
| 15 | 6 | `ChemicalSource.cpp:315-329` | `clone()` dereferences `I.sol`/`I.solT` without null check |
| 16 | 7 | `SourceTermContributor.hpp:437` | `JAC_SKIP_FLUID` zeroes energy→species coupling in source Jacobian |
| 17 | 8 | `EulerEvaluator.hpp:292` | Runtime `if(model==NS_2EQ)` should be `if constexpr` |
| 18 | 9 | `PhysicsProperties.hpp:370-371` | Fallback T-guess can go ≤0 for reactive mixtures near 298K |
| 19 | 9 | `ChemicalSource.cpp:298-304` | `mixtureFormationEnergy` name — returns enthalpy, not energy |
| 20 | 9 | `EulerEvaluator.hxx:1402,1406` | `MeanValuePrim2Cons` passes `rhoH_form=0` — round-trip drops formation |
| 21 | 11 | `ChemicalSource.cpp:107,113,119` | `mixtureCp/Cv/Gamma` hardcode `p=101325` — real gases need actual pressure |
| 22 | 11 | `ChemicalSource.cpp:125-128` | `speedOfSound` implements `a=√(γRT)` — dead ideal-gas code |
| 23 | 11 | `PhysicsProperties.hpp:140-141` | `gammaEq` computes `p_exact = rho·Rmix·T` — ideal-gas assumption |
| 24 | 11 | `PhysicsProperties.hpp:370-379` | `temperature()` invalid-state fallback returns `p/(ρ·Rgas)` (ideal-gas) |

## LOW (14 findings)

| # | Audit | File:Line | Issue |
|---|-------|-----------|-------|
| 1 | 1 | `ChemicalSource.cpp:125` | `speedOfSound` dead code — offers no code-scaled variant |
| 2 | 1 | `PhysicsProperties.hpp:84` | `invR0()` naming inverted — returns R0, not 1/R0 |
| 3 | 1 | `SourceTermContributor.hpp:47-48` | `SourceCellAux::p=101325` labeled "code pressure" but is physical Pa |
| 4 | 3 | `EvaluateRHS.hxx:406` | `gamma` computed before wall-fix overwrites ULxy/URxy |
| 5 | 3 | `PhysicsProperties.hpp:138` | `e_sensible<=0` silently falls back to cp/cv gamma |
| 6 | 5 | `EulerEvaluator.hxx:2204-2286` | Dead lambdas `cellIsHalfAlpha`/`cellAdjAlphaMin` never called |
| 7 | 5 | `EulerEvaluator.hpp:1751-1753` | Dead exponential calculation in `CompressInc` |
| 8 | 8 | `EvaluateDt.hxx:2774` | `outMap["RV"]` inserted twice — first is dead code |
| 9 | 8 | `EvaluateDt.hxx:2815` | Runtime `if(model==NS_2EQ)` should be `if constexpr` |
| 10 | 8 | `EvaluateDt.hxx:1023,1025,1044` | Hardcoded `Vector<real,5>`/`Vector<real,4>` in special-field initializers |
| 11 | 8 | `SourceTermContributor.hpp:436` | `uM3 = (I4>=4) ? U[3] : 0` — hardcodes dim inference |
| 12 | 10 | `ChemicalSource.cpp:247` | `dT_drho` omits KE term (masked by JAC_SKIP_FLUID) |
| 13 | 11 | `ChemicalSource.hpp:86` | Docstring: "perfect gas, variable γ" — misleading; only mixtureR/speedOfSound assume perfect |
| 14 | 11 | `PhysicsProperties.hpp:231` | Comment: `h_k = e_sensible + h_f + R·T` — ideal-gas decomposition of Cantera output |

## CLEAN AUDITS (no bugs)

| Audit | Notes |
|-------|-------|
| 3 (Gamma) | All 33 gammaEq call sites verified; formula 1+p/(ρ·e_sensible) correct; debug shortcut commented out |
| 10 (Unit scaling) | All nondimensionalization chains verified: invS0, rhoScale, transport mu0/k0/D0, speciesEnthalpies, rhoH_form/U0² |

## AUDIT SUMMARY

| Audit | SEVERE | MEDIUM | LOW | Verdict |
|-------|--------|--------|-----|---------|
| 1. Dimensionality | 0 | 1 | 3 | PASS |
| 2. EOS rhoE sensible | 7 | 3 | 0 | **FAIL** |
| 3. Gamma consistency | 0 | 0 | 2 | PASS |
| 4. Flux correctness | 0 | 2 | 0 | PASS |
| 5. PP correctness | 3 | 7 | 2 | **FAIL** |
| 6. Thread safety | 2 | 2 | 0 | **FAIL** |
| 7. Staged diff bugs | 1 | 1 | 0 | FIX |
| 8. Hardcoded dims | 3 | 1 | 4 | **FAIL** |
| 9. Energy/enthalpy | 1 | 3 | 0 | **FAIL** |
| 10. Unit scaling | 0 | 1 | 1 | PASS |
| 11. Ideal gas assumptions | 1 | 4 | 2 | **FAIL** |
| **Total** | **18** | **24** | **14** | |

## Priority Fix Order

1. Thread safety (#11,12) — OMP race on pool[0] when size=1
2. Roe asqrRoe double-subtraction (#3-5) — affects ALL reactive face fluxes
3. EvaluateCellRHSAlphaExpansion (#8,9,10) — PP limiter expansion broken
4. IdealGasPhysics Cons2Prim/Prim2Cons (#1,2) — EulerP path ignores formation
5. GetRoeAverage UOut(I4) missing rhoH_roe (#6) — Roe state inconsistent
6. ExDim=3 hardcoded (#15) — OOB for 2D models
7. temperature() 0K→298K offset (#13) — systematic ~298K T error
8. Jacobian du ideal-gas formula (#18) — real-gas correctness
