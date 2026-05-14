# ThermoPhase API Detailed Reference

Comprehensive reference for the Cantera `ThermoPhase` class API, covering state
setting, property access, species queries, and equilibrium. Based on the Cantera
user guide (`cxx-tutorial.md`, `demo1a.cpp`, `thermodemo.cpp`, `demoequil.cpp`)
and the DNDSR `ChemicalSource` wrapper.

## Object Creation

```cpp
#include "cantera/core.h"

// Via Solution (3.0+)
auto sol = Cantera::newSolution("gri30.yaml", "gri30");
auto gas = sol->thermo();  // std::shared_ptr<Cantera::ThermoPhase>
```

The `ThermoPhase` pointer returned by `sol->thermo()` is non-owning. The
`Solution` object must outlive it.

## Header Files

| Header | Status |
|--------|--------|
| `"cantera/core.h"` | Current (3.0+); includes ThermoPhase |
| `"cantera/thermo.h"` | **Deprecated** since 3.2, removed in 3.3 |

## State Setting Methods

### The Imperative Model

Cantera uses an imperative state-setting model: call a setter first, then query
properties. The state is stored internally; queries read from that cached state.

### `setState_TP(T, P)` — Set Temperature and Pressure

```cpp
gas->setState_TP(1500.0, 2.0 * Cantera::OneAtm);
```
- `T` in K, `P` in Pa
- Does NOT change composition — only T and P
- Use when composition was already set (e.g. via `setMassFractions_NoNorm`)
- **Most common for CFD:** set mass fractions first, then `setState_TP`

### `setState_TPX(T, P, composition_string)` — Set T, P, and Mole Fractions

```cpp
gas->setState_TPX(500.0, 2.0 * Cantera::OneAtm, "H2O:1.0, H2:8.0, AR:1.0");
```
- String format: `"SPEC:mol, SPEC2:mol, ..."`
- Mole fractions are normalized internally — they do NOT need to sum to 1
- Unspecified species get zero
- Convenient for test/initial conditions, but use `setMassFractions_NoNorm` + `setState_TP` in CFD loops (avoids string parsing)

### `setMassFractions_NoNorm(Y_data)` — Set Mass Fractions Without Normalization

```cpp
gas->setMassFractions_NoNorm(Y.data());  // Y must already sum to 1
```
- **Caller is responsible** for providing normalized mass fractions
- Faster than `setMassFractions()` which re-normalizes internally
- Must call `setState_TP(T, P)` or equivalent afterward to finalize state

### `setState_UV(u, v)` — Solve T from Internal Energy and Specific Volume

```cpp
gas->setMassFractions_NoNorm(Y.data());
gas->setState_TP(T_guess, p_guess);  // warm-start for Newton solver
gas->setState_UV(u, v);              // u = J/kg, v = m³/kg
double T = gas->temperature();
```
- Performs a Newton solve — needs a reasonable initial guess
- **Critical gotcha:** `setState_UV` mutates internal state during the solve.
  If the same `ThermoPhase` object is also used for TP-based queries on other
  calls, use a **separate dedicated `Solution`** to avoid state corruption.
  DNDSR does this (see `ChemicalSource.cpp:58-60`).

### `equilibrate(flag)` — Chemical Equilibrium

```cpp
gas->equilibrate("TP");   // constant T, P
gas->equilibrate("HP");   // constant H, P
gas->equilibrate("UV");   // constant U, V
gas->equilibrate("SV");   // constant S, V
gas->equilibrate("SP");   // constant S, P
```
- Finds equilibrium composition at the specified constraints
- The two constrained properties must already be set
- After call, composition is replaced with equilibrium composition

## Scalar Property Access

All scalar methods take **no arguments** — they read the internally stored state.

| Method | Returns | Units | Notes |
|--------|---------|-------|-------|
| `temperature()` | T | K | |
| `pressure()` | P | Pa | |
| `density()` | ρ | kg/m³ | |
| `cp_mass()` | cp | J/kg/K | Specific (per unit mass) |
| `cv_mass()` | cv | J/kg/K | Specific |
| `enthalpy_mole()` | H | J/kmol | Molar |
| `enthalpy_mass()` | h | J/kg | Specific |
| `entropy_mole()` | S | J/kmol/K | Molar |
| `entropy_mass()` | s | J/kg/K | Specific |
| `gibbs_mole()` | G | J/kmol | Molar |
| `gibbs_mass()` | g | J/kg | Specific |
| `meanMolecularWeight()` | M | kg/kmol | Mixture-averaged |
| `sumOfConcentrations()` | C_total | kmol/m³ | |
| `isothermalCompressibility()` | β_T | 1/Pa | |
| `thermalExpansionCoeff()` | α | 1/K | |

### Naming Convention

- Methods ending in `_mole` → molar property
- Methods ending in `_mass` → per-unit-mass property
- No suffix → dimensionless or uses natural units (K, Pa, kg/m³)

## Array Property Access

Methods starting with `get` write into a **user-allocated** output array.

| Method | Output | Units | Size |
|--------|--------|-------|------|
| `getMolecularWeights(data)` | M_k per species | kg/kmol | `nSpecies()` |
| `getChemPotentials(data)` | μ_k per species | J/kmol | `nSpecies()` |
| `getPartialMolarEnthalpies(data)` | h̄_k per species | J/kmol | `nSpecies()` |
| `getPartialMolarEntropies(data)` | s̄_k per species | J/kmol/K | `nSpecies()` |
| `getPartialMolarCp(data)` | cp̄_k per species | J/kmol/K | `nSpecies()` |
| `getMassFractions(data)` | Y_k | — | `nSpecies()` |
| `getMoleFractions(data)` | X_k | — | `nSpecies()` |
| `getConcentrations(data)` | C_k | kmol/m³ | `nSpecies()` |
| `getEnthalpy_RT(data)` | h_k / RT | — | `nSpecies()` |
| `getEntropy_R(data)` | s_k / R | — | `nSpecies()` |
| `getCp_R(data)` | cp_k / R | — | `nSpecies()` |

### Pattern

```cpp
size_t ns = gas->nSpecies();
std::vector<double> mu(ns);
gas->getChemPotentials(mu.data());
```

## Species Queries

| Method | Returns |
|--------|---------|
| `nSpecies()` | Number of species |
| `speciesName(k)` | Species name (string) |
| `speciesIndex("CO2")` | Species index (0-based) |
| `molecularWeight(k)` | Single-species M_k [kg/kmol] |
| `Hf298(k)` | Formation enthalpy at 298 K [J/kmol] |
| `charge(k)` | Species charge (in elementary charges) |
| `elementIndex("C")` | Element index |
| `nAtoms(kSpec, mElem)` | Number of atoms of element m in species k |

## Species-Specific Property Functions

`ThermoPhase` also provides per-species (standard-state) property functions:

```cpp
gas->getStandardChemPotentials(data);   // μ°_k  [J/kmol]
gas->getEnthalpy_RT(data);              // h_k/RT
gas->getEntropy_R(data);                // s_k/R
gas->getCp_R(data);                     // cp_k/R
```

## The `report()` Function

```cpp
std::cout << gas->report() << std::endl;
```

Prints a formatted summary including:
- Temperature, pressure, density
- Phase name and number of species
- Table of mole fractions (X), mass fractions (Y), and non-dimensional chemical potentials
- Useful for debugging and state inspection

## Error Handling

All Cantera operations throw `Cantera::CanteraError`:

```cpp
try {
    auto sol = Cantera::newSolution("gri30.yaml");
    auto gas = sol->thermo();
    gas->setState_TP(300, 101325);
} catch (Cantera::CanteraError& err) {
    std::cerr << err.what() << std::endl;
}
```

## Full Cantera User Guide Examples

Source files at `external/cfd_externals/repos/cantera/doc/sphinx/userguide/`:

- `demo1a.cpp` — basic gas creation, `setState_TPX`, `report()`
- `thermodemo.cpp` — scalar and array property access patterns
- `demoequil.cpp` — `equilibrate("TP")`, equilibrium verification via chemical potentials
