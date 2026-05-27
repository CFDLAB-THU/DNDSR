# State Vector, Nondimensionalization, and Unit Scaling

This document describes the state-vector layout, the nondimensionalization
system, and the energy-convention bridge to Cantera used by the Euler solvers
(including reactive multi-species flow).

## Nondimensionalization

### Fundamental reference scales

Four user-specified scales live in `EulerEvaluatorSettings::IdealGasProperty`:

| Symbol | Config key | Default | SI unit | Description |
|--------|-----------|---------|---------|-------------|
| `L0`   | `L0`      | 1       | m       | Reference length |
| `U0`   | `U0`      | 1       | m/s     | Reference velocity |
| `rho0` | `rho0`    | 1       | kg/m^3  | Reference density |
| `T0`   | `T0`      | 1       | K       | Reference temperature |

When all four are 1 (default), code units equal SI units.

### Derived scales

| Scale | Formula | SI unit | Purpose |
|-------|---------|---------|---------|
| `t0`  | `L0 / U0` | s | Time |
| `p0`  | `rho0 * U0^2` | Pa | Pressure, energy density |
| `R0`  | `U0^2 / T0` | J/(kg·K) | Gas constant, heat capacity |
| `mu0` | `rho0 * U0 * L0` | Pa·s | Dynamic viscosity |
| `k0`  | `rho0 * U0^3 * L0 / T0` | W/(m·K) | Thermal conductivity |
| `D0`  | `U0 * L0` | m^2/s | Mass diffusivity |
| `S0`  | `rho0 * U0 / L0` | kg/(m^3·s) | Volumetric source rate |

### Code-unit conversions

Every physical quantity `x_phys` is stored as `x_code = x_phys / x0`:

| Quantity | Code form | Conversion |
|----------|-----------|------------|
| Density | `rho_code = rho_phys / rho0` | |
| Velocity | `v_code = v_phys / U0` | |
| Pressure | `p_code = p_phys / p0` | |
| Temperature | `T_code = T_phys / T0` | |
| Specific energy | `e_code = e_phys / U0^2` | |
| Volumetric energy | `rhoE_code = rhoE_phys / (rho0 * U0^2)` | |
| R, Cp, Cv | `R_code = R_phys / R0` | = `R_phys * T0 / U0^2` |
| Momentum density | `(rho*u)_code = (rho*u)_phys / (rho0 * U0)` | |
| Mass fraction | `Y_k` | Dimensionless |

## State Vector Layout

### Conservative (dim=3, Ns=10, nVars=14)

`U = [rho, rho*u, rho*v, rho*w, rho*E, rho*Y_0, ..., rho*Y_{Ns-2}]`

- `U[0]`: density `rho`
- `U[1..dim]`: momentum density `rho*u_j`
- `U[dim+1]` (I4): total volumetric energy `rho*E`
- `U[dim+2 ..]`: species densities `rho*Y_k` (first `Ns-1` species)
- Last species (N2) is derived: `rho*Y_last = rho - sum(rho*Y_k)`

`rhoE_total` includes **sensible + kinetic + formation-enthalpy**:
```
rhoE_total = rho * (e_sensible + 1/2 * |v|^2 + Sum_k Y_k * h_f_k)
```

### Primitive (dim=3, Ns=10, nVars=14)

`W = [rho, u, v, w, p, Y_0, ..., Y_{Ns-2}]`

- `W[0]`: density `rho`
- `W[1..dim]`: velocity `u_j`
- `W[dim+1]` (I4): pressure `p`
- `W[dim+2 ..]`: mass fractions `Y_k` (first `Ns-1` species, dimensionless)
- Last species mass fraction is derived: `Y_last = 1 - sum(Y_k)`

### Variants (I/O only)

Two additional primitive layouts are supported for input/output convenience:

**`prim-rhoT`**: `[rho, u, v, w, T, Y_k]` — temperature at I4 instead of pressure.
`p = rho * Rmix * T`.

**`prim-TP`**: `[T, u, v, w, p, Y_k]` — temperature at index 0 instead of density.
`rho = p / (Rmix * T)`.

## Formation Enthalpy Convention

The conservative `rhoE` includes formation enthalpy at reference T=298.15 K.
Configuration input vectors (JSON `farFieldStaticValue`) store **sensible** rhoE;
formation is added at initialization.

Formation enthalpy per species: `h_f_k = H_f_298(k) / M_k` [J/kg].
For elements (H2, O2, N2): `h_f = 0`. For H2O: `h_f ≈ −13.4 MJ/kg`.

Code-unit formation energy density:
```
rhoH_form_code = rho_code * Sum_k(Y_k * h_f_k) / U0^2
```

## Energy Convention: DNDSR ↔ Cantera Bridge

### DNDSR's calorically-perfect convention

DNDSR stores energy measured from **0 K** assuming a calorically-perfect ideal
gas.  The sensible internal energy is:

```
e_sensible(T) = cv_stored * T    where    cv_stored = R / (gamma_stored − 1)
```

`cv_stored` is a **chord slope** — the straight line from (0, 0) to (T, e_sensible).
It is constant for a given gamma_stored; it does **not** vary with temperature.
This is the defining assumption of a calorically-perfect gas:

```
p = (gamma_stored − 1) * rho * e_sensible
  = (gamma_stored − 1) * rho * cv_stored * T
  = rho * R * T                                        [ideal gas EOS]
```

### Cantera's NASA-polynomial convention

Cantera measures thermal energy from the reference temperature **T_ref = 298.15 K**.
At T_ref the thermal part is zero by definition.  Formation internal energy is
stored separately:

```
u_k(T) = u_f_k(T_ref) + Integral(T_ref -> T) cv_k(T') dT'
```

Cantera's `cv_mass(T)` is the **local slope** — the temperature derivative
of the `u(T)` curve at the given T.  It varies with T because rotational and
vibrational modes become active at different temperatures (notably H2 near
1000 K).

### The three gammas

| Name | Formula | Source | Meaning |
|------|---------|--------|---------|
| `gamma_stored` | `1 + R / cv_stored` | User config `IdealGasProperty.gamma` | Non-reactive constant-gamma closure coefficient |
| `gammaEq` | `1 + rho*Rmix*T / (rho*e_sensible)` | `PhysicsProperties::gammaEq` | Pressure/energy closure coefficient used by primitive/conservative conversion and pressure gradients |
| `cp/cv` | `cp_mass(T,Y) / cv_mass(T,Y)` | `PhysicsProperties::gamma`, Cantera NASA polynomials | Frozen-composition acoustic coefficient used by wave speeds and Mach number |

**Why `gammaEq` ≠ `cp/cv`**: DNDSR's energy convention `e_sensible = c_v_stored * T`
uses a constant (chord) cv, while Cantera's `cv_mass(T)` varies with T. The
difference is the gap between a straight-line chord and a curved EOS:

```
cv_stored  = e_sensible / T                    (chord slope, 0K -> T)
cv_mass(T) = du/dT                              (local slope, at T)
cp/cv      = (cv_mass + R) / cv_mass           (real ratio at T)
gammaEq    = (cv_stored + R) / cv_stored       (stored ratio = gamma_stored)
```

At 845 K for H2/O2/N2, `cv_stored ≈ 1020`, `cv_local ≈ 1103` J/(kg·K),
`gammaEq ≈ 1.4`, and `cp/cv ≈ 1.359`.

### Cantera temperature bridge

To convert DNDSR's internal energy (measured from 0 K) to Cantera's convention
(measured from 298.15 K), `PhysicsProperties::temperature()` subtracts two
quantities before calling `setState_UV`:

```
u_sent = u_DNDSR_code * U0^2 − pVAtReference(Y) − e_sens_ref(Y)
```

- `pVAtReference(Y)` = `h(T_ref) − u(T_ref)` = `Rmix * T_ref` for ideal gas.
  Converts formation enthalpy → formation internal energy at T_ref.
- `e_sens_ref(Y)` = `cv_mass(T_ref,Y) * T_ref`.  Subtracts the 0K→T_ref
  sensible energy that DNDSR includes but Cantera starts counting after.

After these subtractions `u_sent` matches Cantera's `intEnergy_mass(T)` for
the same T.  This conversion is guarded by `isIdealGas()` — non-ideal phases
crash with an assertion.

## PhysicsProperties State-Conversion API

All conversion methods operate in **code units**.  Methods that iterate
gamma (prim → cons) are marked *for I/O only, not tight loops*.

### Conservative ↔ Sensible

| Method | Description |
|--------|-------------|
| `consSensibleToTotal<dim>(sens, total)` | Add formation to `U[I4]` |
| `consTotalToSensible<dim>(total, sens)` | Subtract formation from `U[I4]` |

### Primitive ↔ Conservative

| Method | Description |
|--------|-------------|
| `primToConservative<dim>(prim, cons)` | Iterates gammaEq; cfg.gamma as initial guess |
| `conservativeToPrimitive<dim>(cons, prim)` | Uses gammaEq from cons state |
| `primRhoTToConservative<dim>(primRhoT, cons)` | Converts via `p = rho*Rmix*T` |
| `conservativeToPrimRhoT<dim>(cons, primRhoT)` | Replaces p with T |
| `primTPToConservative<dim>(primTP, cons)` | Converts via `rho = p/(Rmix*T)` |
| `conservativeToPrimTP<dim>(cons, primTP)` | Replaces rho with T |

### Code ↔ Physical (I/O only)

| Method | Description |
|--------|-------------|
| `consCodeToPhys<dim>(code, phys)` | Conservative code → physical |
| `consPhysToCode<dim>(phys, code)` | Conservative physical → code |
| `primCodeToPhys<dim>(code, phys)` | Primitive code → physical |
| `primPhysToCode<dim>(phys, code)` | Primitive physical → code |
| `primRhoTCodeToPhys<dim>(code, phys)` | prim-rhoT code → physical |
| `primRhoTPhysToCode<dim>(phys, code)` | prim-rhoT physical → code |
| `primTPCodeToPhys<dim>(code, phys)` | prim-TP code → physical |
| `primTPPhysToCode<dim>(phys, code)` | prim-TP physical → code |

### Ideal-gas guard

Every conversion method that invokes the ideal-gas EOS
(`p = rho*R*T` or `p = (gammaEq−1)*rho*e_sensible`) asserts
`chem().isIdealGas()` before proceeding.  Non-ideal Cantera phases
crash with a clear message.

## Acoustic-Speed Convention

`gammaEq` is not used as the acoustic coefficient for reactive mixtures. The
Euler module computes pressure with `gammaEq` and frozen-composition sound speed
with `cp/cv`:

```
p  = (gammaEq - 1) * rho * e_sensible
a² = (cp/cv) * p / rho
```

Roe averages carry both values. `gammaEqRoe` appears in the pressure-wave
strength decomposition; `gammaRoe = cp/cv` appears in `aRoe`.

## IdealGasProperty::Rgas Convention

`IdealGasProperty::Rgas` stores the **physical** gas constant in J/(kg·K),
defaulting to 287 (dry air).  All consumption passes through
`PhysicsProperties::toCode(Rgas_phys)` which divides by `R0 = U0^2/T0`.
The reactive path uses Cantera's `mixtureR()` instead — `Rgas` only
serves as the non-reactive fallback.

## State-Convert CLI Tool (`eulerState`)

`app/eulerState.exe` converts a single Euler state between all
representations using the PhysicsProperties API:

```
eulerState --model NS_EX --nVars 14 --from cons-sensible --scaling code \
  --config "gamma=1.4,Rgas=287,U0=379,rho0=1" --mechanism h2o2.yaml \
  --state "[1.0,0,0,0,6.0,0.028,0,0,0.222,0,0,0,0,0]"
```

**Input formats** (`--from`):
- `cons-total`, `cons-sensible`: conservative, with/without formation
- `prim`: standard primitive `[rho, u, v, w, p, Y_k]`
- `prim-rhoT`: `[rho, u, v, w, T, Y_k]` — temperature at energy slot
- `prim-TP`: `[T, u, v, w, p, Y_k]` — temperature at density slot

**Output**: prints all state representations (conservative total+sensible,
primitive, prim-rhoT, prim-TP), derived quantities (T, p, gamma_stored,
gammaEq, cp/cv, Rmix, rhoH_form), reference scales with SI units, Cantera state
(intEnergy, enthalpy, cv, cp, speed_of_sound), and energy-consistency
check (u_sent − u_cantera = 0).  All arrays include JSON versions with
full precision.

## Relevant Source Files

| File | Role |
|------|------|
| `EulerEvaluatorSettings.hpp` | `IdealGasProperty` struct: L0, U0, rho0, T0, gamma, Rgas |
| `Physics/PhysicsProperties.hpp` | Scale methods, Cantera boundary, state conversion API |
| `Chemistry/ChemicalSource.hpp/cpp` | PIMPL Cantera wrapper, isIdealGas, pVAtReference, sensibleInternalEnergyAtReference |
| `Gas.hpp` | `IdealGasThermal`, Roe flux, Prim2Cons / Cons2Prim |
| `app/Euler/eulerState.cpp` | CLI state-conversion tool |
