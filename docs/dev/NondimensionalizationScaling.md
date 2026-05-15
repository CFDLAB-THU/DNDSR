# Nondimensionalization and Unit Scaling

This document describes the nondimensionalization system used by the Euler
solvers, including reactive (multi-species) flow.  All conversions between
physical (SI) and code-scaled units happen inside `PhysicsProperties` at
the Cantera boundary.

## Fundamental reference scales

Four user-specified scales define the system.  They live in
`EulerEvaluatorSettings::IdealGasProperty`:

| Symbol | Config key | Default | SI unit | Description |
|--------|-----------|---------|---------|-------------|
| `L0`   | `L0`      | 1       | m       | Reference length |
| `U0`   | `U0`      | 1       | m/s     | Reference velocity |
| `rho0` | `rho0`    | 1       | kg/m^3  | Reference density |
| `T0`   | `T0`      | 1       | K       | Reference temperature |

When all four are 1 (the default), code units equal SI units.

## Derived scales

Mechanical dimensions (M, L, T) are fully determined by `rho0`, `U0`, `L0`.
Temperature adds an independent scale `T0`.

| Scale | Formula | SI unit | Purpose |
|-------|---------|---------|---------|
| `t0`  | `L0 / U0` | s | Time |
| `p0`  | `rho0 * U0^2` | Pa | Pressure, energy density |
| `R0`  | `U0^2 / T0` | J/(kg K) | Gas constant, heat capacity |
| `mu0` | `rho0 * U0 * L0` | Pa s | Dynamic viscosity |
| `k0`  | `rho0 * U0^3 * L0 / T0` | W/(m K) | Thermal conductivity |
| `D0`  | `U0 * L0` | m^2/s | Mass diffusivity |
| `S0`  | `rho0 * U0 / L0` | kg/(m^3 s) | Volumetric source rate |

All derived scales are available as methods on `PhysicsProperties`:
`p0()`, `t0()`, `mu0()`, `k0()`, `D0()`, `S0()`, `invR0()` (= R0).

## Code-unit conversions

Every physical quantity `x_phys` is stored in code as
`x_code = x_phys / x0`, where `x0` is the appropriate reference scale.

| Quantity | Code form | Conversion |
|----------|-----------|------------|
| Position | `x_code = x_phys / L0` | `dx_code = L0 * dx_phys` for gradients |
| Velocity | `v_code = v_phys / U0` | |
| Density | `rho_code = rho_phys / rho0` | |
| Pressure | `p_code = p_phys / p0` | `toPhysP(p_code)` |
| Temperature | `T_code = T_phys / T0` | `toPhysT(T_code)`, `toCodeT(T_phys)` |
| Energy/mass | `e_code = e_phys / U0^2` | Specific energy, enthalpy |
| Energy/vol | `(rho*e)_code = (rho*e)_phys / (rho0 * U0^2)` | rhoE, rhoH_form |
| R, Cp, Cv | `R_code = R_phys / R0` | `toCode(R_phys)` |
| Viscosity | `mu_code = mu_phys / mu0` | |
| Conductivity | `k_code = k_phys / k0` | |
| Diffusivity | `D_code = D_phys / D0` | |
| Source rate | `S_code = S_phys / S0` | Chemical production rate |
| Mass fraction | `Y_k` | Dimensionless -- no scaling |

## Conservation equations in code units

The NS equations in code units:

```
d(rho_code)/dt_code + div_code(rho_code * v_code) = 0

d(rho_code * v_code)/dt_code + div_code(rho_code * v v + p I) = div_code(tau) + f_code

d(rhoE_code)/dt_code + div_code((rhoE + p) * v) = div_code(tau*v + k*gradT + Sigma h_k J_k)

d(rhoY_k_code)/dt_code + div_code(rhoY_k * v) = div_code(rho D_k gradY_k) + S_k_code
```

All terms are in code units. The time derivative uses code time `t_code`.

## Cantera boundary

All Cantera functions expect physical SI inputs and return physical SI
outputs.  `PhysicsProperties` handles all conversions:

| Cantera function | Inputs | Output | Conversion |
|-----------------|--------|--------|------------|
| `temperatureFromUV` | `u_phys = e_internal_code * U0^2`, `v_phys = 1/(rho_code * rho0)` | T [K] | `toCodeT(T_phys)` |
| `productionRates` | `toPhysT(T)`, `toPhysP(p)`, Y | omega [kmol/(m^3 s)] | `omega * MW / S0` |
| `productionRatesAndJacobian` | `T_phys`, `p_phys`, rho_code, rhoE_code, velScale=U0, rhoScale=rho0 | J [omega/U_code] | `MW * J * invS0` |
| `viscosity` | `toPhysT(T)`, `toPhysP(p)`, Y | mu [Pa s] | `/ mu0()` |
| `thermalConductivity` | `toPhysT(T)`, `toPhysP(p)`, Y | k [W/(m K)] | `/ k0()` |
| `speciesDiffusivity` | `toPhysT(T)`, `toPhysP(p)`, Y | D [m^2/s] | `/ D0()` |
| `speciesEnthalpies` | `toPhysT(T)`, `toPhysP(p)`, Y | h_k [J/kg] | `/ U0^2` |
| `mixtureGamma` | `toPhysT(T)`, Y | gamma [-] | dimensionless |
| `mixtureR` | Y | R [J/(kg K)] | `toCode(R)` |
| `mixtureCp` | `toPhysT(T)`, Y | Cp [J/(kg K)] | `toCode(Cp)` |
| `mixtureFormationEnergy` | Y | e_f [J/kg] | `* rho_code / U0^2` |

## Chemical source Jacobian

The Jacobian `productionRatesAndJacobian` computes `d(omega_i)/d(U_j_code)`
where `omega_i` is in molar rate [kmol/(m^3 s)] and `U_j` is in code units.

Two chain-rule contributions for species columns:

1. **Concentration chain rule**: `dC_k/d(rhoY_k)_code = rho0 / MW_k`.
   Requires `rhoScale = rho0` parameter.

2. **Temperature chain rule**: `dT/d(rhoY_k)_code = -(u_k - u_last) / (rho_code * cv)`.
   The `rho0` factors cancel: `dT/d(rhoY_k)_phys` has `1/rho_phys` and
   `d(rhoY_k)_phys/d(rhoY_k)_code = rho0`, giving `rho0/rho_phys = 1/rho_code`.

The caller (`ChemicalContributor`) applies `MW_k * invS0` to convert from
molar Jacobian to code-unit mass-source Jacobian with correct sign
(`jac -= val`, storing `-dS/dU`).

## Formation enthalpy convention

The state vector stores **total** rhoE = sensible + formation + kinetic.
Configuration input vectors store **sensible** rhoE; formation is added at
initialization and BC assignment via `mixtureFormationRhoE(U)`.

Formation enthalpy per species: `h_f_k = H_f_298(k) / MW_k` [J/kg].
For H2, O2, N2: `h_f = 0`.  For H2O: `h_f = -13.4 MJ/kg`.

Code-unit formation energy density:
`rhoH_form_code = rho_code * (Sigma Y_k * h_f_k) / U0^2`.

## Species diffusion enthalpy transport

The energy equation includes `div(Sigma h_k J_k)` where `J_k = -rho D_k gradY_k`
is the Fickian diffusion flux and `h_k` is the total specific enthalpy
(sensible + formation).

In code units (implemented in `fluxFace`):
```
VisFlux(energy) += Sigma_{k<Ns-1} (h_k/U0^2 - h_last/U0^2) * (-rho_code * D_code * gradY_k . n)
```

The `(h_k - h_last)` form enforces the constraint `Sigma J_k = 0` for the
dependent last species (N2).

## Equivalent gamma

`gammaEq = 1 + p / (rho * e_sensible)` where `p = rho * R_mix * T` (exact
ideal-gas EOS).  This ensures `(gammaEq - 1) * rho * e_sensible = p` exactly,
eliminating the thermodynamic inconsistency between the UV-based temperature
and the `(gamma-1)` pressure formula for variable-property mixtures.

For non-reactive (constant gamma) gas, `gammaEq = gamma` identically.

All evaluator-side `gamma` computations use `gammaEq`.  The Gas.hpp
Riemann solver functions receive gamma as a parameter from evaluator
callers, so they also use the corrected value.

## Relevant source files

| File | Role |
|------|------|
| `EulerEvaluatorSettings.hpp` | `IdealGasProperty` struct: L0, U0, rho0, T0 |
| `Physics/PhysicsProperties.hpp` | Scale methods, Cantera boundary conversions |
| `SourceTermContributor.hpp` | `ChemicalContributor`: source + Jacobian with invS0 |
| `Chemistry/ChemicalSource.cpp` | `productionRatesAndJacobian`: rhoScale for dC/d(rhoY) |
| `EulerEvaluator_EvaluateDt.hxx` | `fluxFace`: species diffusion + enthalpy transport |
| `Gas.hpp` | `ViscousFlux_IdealGas`, `IdealGasThermal` |
