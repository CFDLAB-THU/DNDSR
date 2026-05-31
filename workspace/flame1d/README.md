1-D Stoichiometric Premixed H2/Air Flame Workspace
===================================================

Cantera reference: Su = 2.2593 m/s (h2o2.yaml, T_u = 300 K, p = 101325 Pa).

Strategy
--------

The goal is to validate the DNDSR Euler/CFV reactive-flow solver against
the well-known 1-D freely propagating premixed flame benchmark.

Correct Su is obtained by tracking the midpoint-temperature marker
(T_mid ≈ 1329 K) and subtracting the unburned gas velocity ahead of the
flame:

    Su = marker_speed − u_unburned

Contents
--------

* `plot_output_digest.py` — reusable helper that auto-detects a DNDSR
  output directory, reads VTU snapshots and CSV logs, generates profile
  and history plots, and writes a machine-readable `digest.json`.
  Usage: `python workspace/flame1d/plot_output_digest.py <output-dir>`

* `cantera_freeflame_profile.csv` — Cantera FreeFlame solution (flipped
  so burned gas is on the left).

* `strang_vs_nostrang_comparison.md` — detailed comparison of the
  Strang-split and non-Strang coupled runs (np = 8, dt = 1e-3).

* `strang_*.png` / `strang_digest.json` — digest outputs for the Strang
  run (sourceStrangSplitting = 1).

* `nostrang_*.png` / `nostrang_digest.json` — digest outputs for the
  non-Strang coupled run (sourceStrangSplitting = 0).

* `cantera_exprtk_exprs.json` — ExprTk piecewise-linear interpolation
  expressions generated from the Cantera profile (experimental).

* `cantera_profile_samples.json` — down-sampled Cantera profile data
  (experimental).

Key Results
-----------

| Method      | dt      | np | Su        | vs Cantera |
|-------------|---------|----|-----------|------------|
| Non-Strang  | 1e-3    | 8  | 2.39 m/s  | 1.06×      |
| Non-Strang  | 1e-4    | 4  | 3.68 m/s  | 1.63×      |
| Strang      | 1e-3    | 8  | 4.0 m/s   | 1.8×       |

The **non-Strang fully coupled** solver agrees with Cantera to within
6 %.  Strang splitting over-estimates Su by ~80 % at dt = 1e-3.

Both flames hit the right BCIn boundary partway through the run
(domain too short at 2 cm).

Current Config
--------------

`cases/eulerEX/config_1d_premixed_stoichiometric.json`

See `caseNotes["/**/"]` in that file for the latest setup and
audit-follow-up list.
