1-D H2/Air Detonation Configuration
==================================

Physics Summary
---------------

**Reactants (unburned, right side):**
  Stoichiometric H2/air: Y_H2=0.028, Y_O2=0.222, Y_N2=0.75
  T = 300 K, P = 101325 Pa
  rho = 1.144 kg/m³, a = 352 m/s

**CJ detonation (estimated for stoichiometric H2/air):**
  U_CJ ~ 1970 m/s
  P_CJ ~ 15-20 bar (from rho*U² scaling)
  T_CJ ~ 2800-3000 K
  Density ratio rho_CJ/rho_u ~ 5-6

**Domain and mesh:**
  Mesh: Uniform_01_5000.cgns (5000 cells on [0,1])
  meshScale = 0.05 → physical domain [0, 0.05] m = 5 cm
  dx = 10 μm per cell
  At U_CJ ~ 1970 m/s: crossing time ~ 25 μs → t_code ~ 0.0095
  dt = 1e-4 → ~95 steps to cross, reasonable

**Boundary conditions:**
  Left (x=0): BCWallInvis — adiabatic reflecting wall
  Right (x=0.05): BCIn — unburned mixture at rest

**Initial condition (ExprTk):**
  Left spark region (x < 0.001 m): high T/P to initiate detonation
  Right region: unburned mixture

**Solver settings:**
  Non-Strang coupled (sourceStrangSplitting=0) — better accuracy
  dt = 1e-4, nTimeStep = 200, tEnd ≈ 0.02 (52.8 μs physical)
  RecScheme = 0 (Green-Gauss), CFL = 10, 400 pseudo-time steps

**Expected behavior:**
  Spark at left wall → shock forms → shock-induced combustion →
  detonation wave propagating rightward at ~1970 m/s.
  Front should reach ~halfway across domain by tEnd.

**Config file:** `cases/eulerEX/config_1d_detonation.json`
**Output dir:** `data/out/react_1d_detonation_dt1e4_np8/`
