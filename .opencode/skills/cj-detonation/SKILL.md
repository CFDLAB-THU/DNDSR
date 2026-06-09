---
name: cj-detonation
description: Use when estimating Chapman-Jouguet (CJ) detonation properties, induction lengths, or ZND structure for a given mixture and mechanism. Trigger on CJ, detonation speed, induction length, von Neumann spike, ZND, post-shock state, or when designing detonation simulations.
---

# CJ Detonation Estimation

## Purpose

When a user asks about Chapman-Jouguet (CJ) detonation properties, induction
lengths, or ZND structure for a given mixture and mechanism, use the bundled
scripts to compute:

* CJ detonation speed U_CJ
* von Neumann (post-shock) state: T_VN, P_VN, ρ_VN
* CJ (equilibrium) state: T_CJ, P_CJ, ρ_CJ
* Ignition delay and induction length
* Mesh resolution requirements (cells across induction zone)

## Scripts

All scripts are in the skill directory alongside this file.

### `estimate_cj.py` — CJ speed and thermodynamic states

```bash
python .opencode/skills/cj-detonation/estimate_cj.py \
    --mechanism h2o2.yaml \
    --composition "H2:0.2, O2:0.1, AR:0.7" \
    --basis mole \
    --temperature 300 --pressure 101325
```

Outputs: U_CJ, T_VN, P_VN, ρ_VN, T_CJ, P_CJ, ρ_CJ, Mach numbers.

Supports `--composition` with molar (`--basis mole`) or mass
(`--basis mass`) fractions.  Default mechanism is Cantera's built-in
search path; use an absolute or relative path otherwise.

### `estimate_induction.py` — Ignition delay and induction length

```bash
python .opencode/skills/cj-detonation/estimate_induction.py \
    --mechanism h2o2.yaml \
    --composition "H2:0.2, O2:0.1, AR:0.7" \
    --basis mole \
    --temperature 300 --pressure 101325 \
    --dx 1e-5
```

Uses Cantera constant-pressure reactor at the von Neumann state to
compute ignition delay (OH threshold) and induction length.
Optional `--dx` reports cells across the induction zone.

## When to use this skill

* User asks about CJ parameters for any mixture
* User wants to know if their mesh resolves the induction zone
* User is designing a detonation simulation and needs dt, dx estimates
* User is comparing simulation results with CJ theory

## Typical workflow

1. Run `estimate_cj.py` to get U_CJ and post-shock state
2. Run `estimate_induction.py` to get induction length
3. Compare induction length with mesh dx to assess resolution
4. Recommend dt so the detonation advances ~1 cell/step:
   dt_code = dx / U_CJ * U0 (where U0 is the code-unit velocity scale)

## Physics notes

* The CJ condition is the minimum-speed detonation solution where
  products are sonic relative to the shock (M_product = 1).
* The von Neumann spike is the frozen-shock state before any reaction.
* Induction length ≈ u_VN × t_ignition, where u_VN is the particle
  velocity behind the shock (lab frame, U_CJ × (1 − ρ₀/ρ_VN)).
* For H2/O2 mixtures, induction is extremely sensitive to post-shock T.
  Even small differences in U_CJ produce large changes in t_ignition.
* Dilution (Ar, N2) increases induction length and slows CJ speed,
  making detonations easier to resolve numerically.

## Integration with DNDSR

* Use U_CJ to set dt: dt_code ≈ dx / U_CJ * U0
* Use induction_length / dx to verify mesh resolution
* Compare DNDSR-measured shock speed with estimate_cj.py output
* The ZND structure (shock → induction → reaction → products) should
  be visible in DNDSR VTU profiles
