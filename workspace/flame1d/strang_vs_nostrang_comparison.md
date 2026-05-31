Strang vs Non-Strang Premixed Flame Comparison
===============================================

Cantera reference: `scripts/reaction/h2o2_free_flame_mixture_averaged.py`,
h2o2.yaml stoichiometric H2/air, mixture-averaged transport, no Soret
(matching DNDSR transport).  T_u = 300 K, p = 101325 Pa,
Su = 2.2540 m/s, burned T = 2360.02 K, rho_u/rho_b = 6.77.

Shared Configuration
--------------------

* Mesh: Uniform_01_400.cgns, 400 quad cells, meshScale = 0.02 → 2 cm domain
* BC: BCIn both sides — burned on left, unburned on right, zero velocity
* Initial: tanh profile centred at x = 0.005 m, width 1 mm
* ODE: ESDIRK2 (odeCode = 204), CFL = 10 (pseudo-time only)
* Pseudo-time: nTimeStepInternal = 200, rhsThreshold = 1e-3
* Limiter: PPRec, nPartialLimiterStart = 1e6 (off)
* Linear solver: SGS/Jacobi (jacobiCode = 1), ILU direct preconditioner
* Reactive source scale = 1, species diffusion enabled
* MPI: np = 8
* Output: every 10 steps

Differs by
----------

| Setting                | Strang          | Non-Strang      |
|------------------------|-----------------|-----------------|
| sourceStrangSplitting  | 1               | 0               |
| Reactive source in RHS | off (flow-only) | on (fully coupled) |
| Chemistry integration  | Cantera CVODE   | point-implicit pseudo-time |

Flame Speed
-----------

Front: x where T = T_mid = 0.5 × (300 + 2360.02) = 1330.01 K.

Marker speed is obtained from a sliding 5-pt linear fit over the
later half of the free-propagation phase with t_code < 0.6.
All fits have R² > 0.99998.

Su = marker_speed − u_gas(unburned ahead of flame).

All flames hit the right BCIn boundary before reaching tEnd.

### dt = 1e-3 (1000 physical steps, tEnd = 1.0)

|                          | Non-Strang    | Strang        |
|--------------------------|---------------|---------------|
| Free steps / pinned at   | 88 / step 870 | 83 / step 820 |
| Marker speed (sliding)   | 6.37 m/s      | 6.88 m/s      |
| u_unburned (mid-fit)     | 3.91 m/s      | 4.22 m/s      |
| **Derived Su**           | **2.46 m/s**  | **2.66 m/s**  |
| Ratio vs Cantera         | **1.09×**     | **1.18×**     |

### dt = 2e-3 (500 physical steps, tEnd = 0.5)

|                          | Non-Strang    | Strang        |
|--------------------------|---------------|---------------|
| Free steps / pinned at   | 26 / step 250 | 26 / step 250 |
| Marker speed (sliding)   | 6.25 m/s      | 7.66 m/s      |
| u_unburned (mid-fit)     | 3.83 m/s      | 4.70 m/s      |
| **Derived Su**           | **2.41 m/s**  | **2.97 m/s**  |
| Ratio vs Cantera         | **1.07×**     | **1.32×**     |

### Summary Table

| Method      | dt    | Su        | R²       | vs Cantera |
|-------------|-------|-----------|----------|------------|
| Non-Strang  | 1e-3  | 2.46 m/s  | 0.999999 | 1.09×      |
| Non-Strang  | 2e-3  | 2.41 m/s  | 1.000000 | 1.07×      |
| Strang      | 1e-3  | 2.66 m/s  | 1.000000 | 1.18×      |
| Strang      | 2e-3  | 2.97 m/s  | 1.000000 | 1.32×      |

Key Observations
----------------

1. **Non-Strang coupled agrees with Cantera to within 7–9 %** at both
   dt = 1e-3 and dt = 2e-3.  This is excellent for a 400-cell mesh
   at dx = 0.05 mm.

2. **Strang splitting systematically over-predicts Su** by 18–32 %,
   with the error growing at larger dt — consistent with O(dt²)
   splitting error accumulation.

3. **Non-Strang does not degrade with dt** over this range
   (2.46 → 2.41 m/s from dt=1e-3 to 2e-3), suggesting temporal
   truncation error is below other error sources at these timesteps.

4. **The 2 cm domain is too short.**  All flames hit the right BCIn
   boundary during the run.  The sliding-window fit uses only the
   early portion (t_code < 0.6) to avoid boundary influence.

5. **BCIn on both ends creates a non-free-flame flow field.**  Gas
   expands away from the flame in both directions; Su must be derived
   as marker_speed − u_unburned.

6. **CFL = 10 (pseudo-time) does not add physical diffusion.**
   Physical-time accuracy is governed by dtImplicit and the ODE method.

Plots and Data
--------------

* Front marker vs time:      `workspace/flame1d/front_marker_time.png`
* Flame speed analysis:      `workspace/flame1d/analyze_flame_speed.py`
* Output digest helper:      `workspace/flame1d/plot_output_digest.py`
* Strang dt=1e-3 profiles:   `workspace/flame1d/strang_profiles_thermo.png`, `_species.png`
* Strang dt=1e-3 digest:     `workspace/flame1d/strang_digest.json`
* Non-Strang dt=1e-3 profiles:`workspace/flame1d/nostrang_profiles_thermo.png`, `_species.png`
* Non-Strang dt=1e-3 digest: `workspace/flame1d/nostrang_digest.json`
* Cantera reference CSV:     `workspace/flame1d/cantera_freeflame_profile.csv`
* Cantera reference summary: `workspace/flame1d/cantera_reference_summary.json`
