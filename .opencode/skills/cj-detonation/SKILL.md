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
* ZND induction length and exothermic length
* Mesh resolution requirements (cells across induction zone)

## Backend: SDToolbox

The scripts use the Caltech **Shock and Detonation Toolbox (SDToolbox)** —
https://shepherd.caltech.edu/EDL/PublicResources/sdt/

SDToolbox provides the standard "minimum wave speed" method for CJ speed and
a robust ZND ODE integrator for detonation structure.  It is the canonical
tool for detonation property estimates.

Install SDToolbox from the pip-installable mirror:

```bash
pip install "sdtoolbox @ git+https://github.com/harryzhou2000/sdtoolbox.git#subdirectory=Python3"
```

Docs and examples are at the source repo (same URL) under `Python3/demo/`.

## Scripts

All scripts are in the skill directory alongside this file.

### `estimate_cj.py` — CJ speed and thermodynamic states

```bash
python .opencode/skills/cj-detonation/estimate_cj.py \
    --mechanism gri30.yaml \
    --composition "H2:2, O2:1" \
    --basis mole \
    --temperature 300 --pressure 101325
```

Outputs: U_CJ, T_CJ, P_CJ, ρ_CJ, T_VN, P_VN, ρ_VN, u_VN, Mach numbers.

Supports `--composition` with molar (`--basis mole`) or mass (`--basis mass`)
fractions.  Mechanism is resolved via Cantera's search path; use a relative or
absolute path for project-local mechanisms.  Default mechanism is `gri30.yaml`.

### `estimate_induction.py` — ZND structure and induction length

```bash
python .opencode/skills/cj-detonation/estimate_induction.py \
    --mechanism gri30.yaml \
    --composition "H2:2, O2:1, AR:7" \
    --basis mole \
    --temperature 300 --pressure 101325 \
    --dx 1e-5
```

Uses SDToolbox's `zndsolve` to integrate the reactive Euler equations through
the detonation wave (shock → induction → reaction → equilibrium products).
The induction length is the distance from the shock to the point of maximum
thermicity gradient (dσ̇/dx).

Options:
* `--overdrive 1.001` — U/U_CJ > 1 to avoid the sonic singularity (default 1.001)
* `--t-end` — max integration time (auto-selected based on pressure)
* `--tolerance 1e-8` — integration tolerance
* `--dx 1e-5` — report cells across the induction zone for mesh resolution
* `--znd-output profile.npz` — save full ZND profile (distance, T, P, ρ,
  species, thermicity, Mach number)
* `--plot-output profile.png` — save a 5-panel ZND profile plot (T, P,
  velocity in both frames, thermicity, major species mass fractions vs
  distance from shock; supports png/pdf/svg)

The `--znd-output` npz file also stores metadata: detonation speed `D`,
`cj_speed`, `species_names`, initial state (`T1`, `P1`, `rho1`, `Y1`),
and `ind_len`/`exo_len` for downstream use by `generate_znd_exprtk.py`.

### `generate_znd_exprtk.py` — exprtk initial condition from ZND profile

```bash
python .opencode/skills/cj-detonation/generate_znd_exprtk.py \
    --znd-profile data/out/znd_H2O2_6667Pa.npz \
    --Ly 0.1 --shock-pert-amp 1e-3 --v-pert-amp 40 \
    --output data/out/znd_exprtk_init.json
```

Generates `exprtkInitializers` JSON for injecting a 1D ZND detonation profile
as the initial condition in DNDSR solver configs.  The output JSON can be
copy-pasted into the `eulerSettings.exprtkInitializers` field.

The generated exprtk expression:
1. Defines inline data vectors (50 points, adaptively sampled) for distance,
   T, P, lab-frame velocity, and all species mass fractions
2. Computes a perturbed shock position: `x_shock(y) = x_shock_0 + A * cos(2πy/Ly)`
3. Uses piecewise linear interpolation with for-loop + break to map
   `dist = x_shock - x[0]` to ZND state variables
4. Falls back to unreacted gas (farfield) for `dist < 0` and CJ equilibrium
   for `dist > L_CJ`
5. Adds transverse velocity perturbation `v = A_v * sin(2πy/Ly)` near the
   shock, decaying exponentially over ~2 induction lengths

Options:
* `--n-points 50` — number of interpolation points (adaptive spacing)
* `--Ly 0.1` — periodic domain width in y [m]
* `--shock-pert-amp 1e-3` — shock position perturbation amplitude [m]
* `--v-pert-amp 40` — transverse velocity perturbation amplitude [m/s]
* `--x-shock` — override shock position [m] (default: 1.5 × L_CJ)
* `--cj-tol 0.01` — tolerance for detecting CJ distance (P within tol of final)

## When to use this skill

* User asks about CJ parameters for any mixture
* User wants to know if their mesh resolves the induction zone
* User is designing a detonation simulation and needs dt, dx estimates
* User is comparing simulation results with CJ theory
* User wants to inject a ZND profile as initial conditions for a solver run

## Typical workflow

1. Run `estimate_cj.py` to get U_CJ and thermodynamic states
2. Run `estimate_induction.py` to get induction length, ZND profile, and plot
3. Compare induction length with mesh dx to assess resolution
4. Compute dt so the detonation advances ~1 cell/step:
   dt_code = dx / U_CJ * U0 (where U0 is the code-unit velocity scale)
5. (Optional) Run `generate_znd_exprtk.py` to produce exprtk initial conditions
   from the ZND profile for solver injection

## Physics notes

* The CJ condition is the minimum-speed detonation solution where products
  are sonic relative to the shock (M_product = 1).  SDToolbox finds it via
  the minimum wave speed method: U_CJ = argmin_U w(U), where w is the
  particle velocity at the equilibrium end state for each trial U.
* The von Neumann spike is the frozen-shock state before any reaction
  (PostShock_fr).
* Induction length ≈ u_VN × t_induction, where u_VN is the particle velocity
  behind the shock in the lab frame: U_CJ × (1 − ρ₀/ρ_VN).
* For H2/O2 mixtures, induction is extremely sensitive to post-shock T.
  Even small differences in U_CJ produce large changes in induction length.
* Ar dilution **increases** T_VN (Ar reduces γ → stronger shock compression
  at a given Mach number), which *shortens* induction.  The CJ speed drops
  because the mixture has higher density (higher MW), but the induction
  zone can become very short (~50 μm for H2/O2/Ar 2:1:7 at 1 atm).
* The ZND solver needs U > U_CJ (overdrive > 1) to avoid division by zero
  at the sonic point (1 − M² → 0).  Use `--overdrive 1.001` as default;
  increase to 1.005 if integration fails.
* If the ZND solve fails, try: (a) increase `--overdrive`, (b) use a simpler
  mechanism, (c) increase `--tolerance` to 1e-6, or (d) increase `--t-end`.

## Integration with DNDSR

* Use U_CJ to set dt: dt_code ≈ dx / U_CJ * U0
* Use ℓ_induction / dx to verify mesh resolution
* Compare DNDSR-measured shock speed with estimate_cj.py output
* The ZND structure (shock → induction → reaction → products) should be visible
  in DNDSR VTU profiles
* Save full ZND profiles with `--znd-output` for comparison with 1D code output

## Reference H2/O2 values (GRI-Mech 3.0, T1=300 K)

| Mixture | P1 | U_CJ [m/s] | T_VN [K] | ℓ_ind [mm] |
|---|---|---|---|---|
| H2/O2 2:1 | 1 atm | 2836 | 1765 | 0.05 |
| H2/O2 2:1 | 6.67 kPa | 2689 | 1626 | 0.98 |
| H2/O2/Ar 2:1:7 | 1 atm | 1692 | 2055 | 0.08 |
| H2/O2/Ar 2:1:7 | 6.67 kPa | 1617 | 1903 | 1.51 |
