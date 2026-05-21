# Reactive-Flow Boundary Condition Audit

**Date:** 2026-05-21  
**Source:** `src/Euler/EulerEvaluator_EvaluateDt.hxx`, `src/Euler/EulerEvaluateRHS.hxx`, `src/Euler/EulerBC.hpp`

## Summary

| BC Type | Species handling | Formation convention | γ source | Issues |
|---------|-----------------|---------------------|----------|--------|
| `BCFar` / `BCOutP` | Preserved (U/U(0) scaling) | `far` is sensible, interior is total; all correct | From interior `ULxyStatic` (minor) | None |
| `BCWall` (adiabatic no-slip) | Copied from interior | Energy untouched (total preserved) | N/A | None |
| `BCWallIsothermal` | Preserved (mass fractions constant) | ❌ rhoH_form uses old density | From ghost `URxy` | Bug #1 (SEVERE), Bug #2 (MEDIUM) |
| `BCWallInvis` / `BCSym` | Copied from interior | Total energy unchanged | N/A | None |
| `BCOut` (extrapolation) | Preserved from interior | Total energy unchanged | N/A | None |
| `BCIn` (direct inflow) | From user input | Sensible→total on line 2682 | N/A | None (convention documented) |
| `BCInPsTs` (total-condition inflow) | From user input | ❌ Uses interior rhoH_form, Cp, Rgas | From interior | Bug #3, #4, #5 (MEDIUM) |
| `BCSpecial` (DMR/RT/IV/2DR/Noh) | Mixed | DMR/RT/Noh correct; IV broken (#6) | From interior or far | Bug #6 (MEDIUM) |

---

## Bug #1 — SEVERE: BCWallIsothermal ignored when `noRsOnWall=true`

**File:** `EulerEvaluator_EvaluateRHS.hxx:487-490`  
**Issue:** When `noRsOnWall=true`, the isothermal ghost state computed on line 487 is overwritten by the adiabatic state on lines 489-490. The wall thermal condition is silently ignored — wall treated as adiabatic.

```
// Line 475-490: Inside BCWall/BCWallIsothermal block
if (faceBCType == EulerBCType::BCWallIsothermal)
{
    real temp = pBCHandler->GetValueFromID(mesh->GetFaceZone(iFace))(0);
    // ... compute isothermal ULc ...
    ULxy = ULc;   // fixes only ULxy
}
// Lines 489-490: BLIND overwrite of BOTH ULxy and URxy
URxy = ULc;      // ULc from adiabatic-block, not isothermal-block
ULxy = ULc;      // destroys the isothermal ULxy set above
```

**Fix:** Move the `URxy = ULc; ULxy = ULc;` lines inside each branch (adiabatic and isothermal) separately.

---

## Bug #2 — MEDIUM: BCWallIsothermal rhoH_form uses old density

**File:** `EulerEvaluator_EvaluateDt.hxx:2564-2565`  
**Issue:** After computing `newDensity` from the isothermal condition, `Prim2Cons` is called with `mixtureFormationRhoE(URxy)` where `URxy` still has the **old** density. The formation energy density in `URxy` corresponds to ρ_old, but after `Prim2Cons` the new state has ρ_new. Mass fractions are preserved (Y_k unchanged), but `rhoH_form_new = rhoH_form_old · newDensity / oldDensity` — the code uses `rhoH_form_old`.

```
// Line 2557-2566:
URxy(0) = newDensity;
URxy(I4) = p;  // pressure, independent of density
Gas::IdealGasThermalPrimitive2Conservative<dim>(
    URxy, out, gamma,
    phys_.mixtureFormationRhoE(URxy));  // <-- URxy has old density here!
```

**Fix:** Compute `rhoH_form_corrected = phys_.mixtureFormationRhoE(URxy_copy_with_new_density)` before calling Prim2Cons.

---

## Bug #3 — MEDIUM: BCInPsTs uses interior Cp instead of inflow Cp

**File:** `EulerEvaluator_EvaluateDt.hxx:2731`  
**Issue:** `Cp` is taken from `phys_.Cp(T, ULxyStatic)` — the **interior** composition — to compute static temperature from total temperature. If the inflow species differ from interior, the wrong Cp is used.

**Fix:** Build a temporary conservative vector with `farPrimitive`'s species and interior density, call `phys_.Cp(T, dummyState)`.

---

## Bug #4 — MEDIUM: BCInPsTs uses interior Rgas instead of inflow Rgas

**File:** `EulerEvaluator_EvaluateDt.hxx:2744`  
**Issue:** `Rgas = phys_.Rgas(ULxyStatic)` (interior composition) to compute static density. Wrong if inflow composition differs.

**Fix:** Use `phys_.Rgas(dummyState)` with inflow species.

---

## Bug #5 — MEDIUM: BCInPsTs uses interior rhoH_form in Prim2Cons

**File:** `EulerEvaluator_EvaluateDt.hxx:2748`  
**Issue:** `Prim2Cons(farPrimitive, URxy, gamma, mixtureFormationRhoE(ULxyStatic))` uses interior formation instead of inflow formation. Total energy is wrong when inflow species differ.

**Fix:** Compute `rhoH_form_inflow` from inflow species before Prim2Cons.

---

## Bug #6 — MEDIUM: BCSpecial Isentropic Vortex lacks reactive support

**File:** `EulerEvaluator_EvaluateDt.hxx:2349-2373`  
**Issue:** `URxy.setZero()` zeros all species. No `mixtureFormationRhoE(URxy)` is added when `reactiveFlow.enabled`. The IV BC is broken for reactive flows.

**Fix:** Copy species from `settings.farFieldStaticValue` (or interior) and add formation energy.

---

## Observations (not bugs)

1. **`farFieldStaticValue` convention**: All BC config vectors store **sensible** energy. `rhoH_form=0` on `Cons2Prim` calls for these states is intentional. Formation is added later (e.g., `BCIn` line 2682, `DMR` line 2279) via `+= mixtureFormationRhoE(URxy)`.

2. **Gamma approximation**: `BCFar`/`BCOutP`/`BCInPsTs` compute `gamma` from the interior state `ULxyStatic` and reuse it for boundary conversions. If far-field/inflow temperature/composition differs significantly, `gamma` should ideally be recomputed. Minor approximation — harmless for most flows.

3. **Species diffusion at walls**: No explicit zero-normal-gradient enforcement for species at impermeable walls. High-order reconstructions could produce small unphysical diffusive flux. Not urgent for current verification phase.

4. **Species mechanical preservation**: All standard BCs preserve species via the `prim = U/U(0)` and `U = prim*prim(0)` pattern. Mass fractions Y_k are never explicitly manipulated — only density scaling changes rhoY_k. Clean design.

5. **No illicit `rhoH_form=0`**: Every BC Prim2Cons/Cons2Prim call on total-energy states uses correct `phys_.mixtureFormationRhoE(...)`. Only config/sensible states use 0.

---

## Recommended Fix Order

1. ✅ **Bug #1** (SEVERE): Fix BCWallIsothermal noRsOnWall overwrite in EvaluateRHS.hxx
2. ✅ **Bug #3, #4, #5** (MEDIUM): Fix BCInPsTs composition-dependent properties
3. ✅ **Bug #2** (MEDIUM): Fix BCWallIsothermal rhoH_form density scaling
4. ✅ **Bug #6** (MEDIUM): Add reactive shield to BCSpecial Isentropic Vortex

## Secondary Audit (2026-05-21) — gammaEq Usage

A second pass audited all 14 `gammaEq` calls in EvaluateDt.hxx for reactive-flow consistency.

| Line | Context | gammaEq from | Verdict |
|------|---------|--------------|---------|
| 2151 | `BCFar`/`BCOutP` | `ULxyStatic` (interior) | Acceptable — used for characteristic switching and far↔interior conversions. Interior gamma ≈ far-field gamma for same mixture. |
| 2286 | `BCSpecialFar` DMR | `ULxy` (interior) | Acceptable — DMR shock-tube has uniform composition. |
| 2356 | `BCSpecial` IV | `ULxy` (interior) | Acceptable — isentropic vortex is constant-γ analytic flow. |
| 2380 | `BCSpecial` 2D Riemann | `ULxy` (interior) | Acceptable — Riemann problem has uniform composition. |
| 2561 | `BCWallIsothermal` | `URxy` (ghost, post-reversal) | ✓ Correct — gamma of the state being converted. |
| 2736 | `BCInPsTs` | `ULxyStatic` (interior) | ⚠️ Approximation — isentropic relation `p=Pstag*(T/Tstag)^(γ/(γ−1))` uses interior γ, should ideally use inflow γ. Circular for variable-γ mixtures (inflow γ depends on inflow T, which depends on γ). Not fixed — negligible for same-mixture boundaries. |
| 2857 | output scalar | `ULMeanXy` | Irrelevant to BCs. |

**Result:** All `gammaEq` usage is appropriate. No new bugs found beyond the 6 already fixed.

## Additional Note — `BCInPsTs` gamma approximation

The isentropic total→static relation at line 2759 uses `gamma` from `ULxyStatic` (interior). For inflow boundaries with different species, the correct gamma would be from the inflow composition. However:

1. Computing inflow γ requires inflow T, which requires the isentropic relation, which requires γ — circular.
2. For the same gas mixture, γ varies slowly with T (~1.35-1.40 for H₂/O₂/N₂ from 300-3000K). The interior γ is within ~1% of inflow γ.
3. Tests with `eulerState` show `gamma_eq ≈ 1.388` for the reactive test case at both 845K (interior) and far-field conditions — the difference is in the third decimal.

**Decision:** Not fixing. The approximation is accurate to ~1% and the circular dependency makes exact computation impractical without an iterative approach.
