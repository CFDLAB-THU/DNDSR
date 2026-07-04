# Kinetics API Detailed Reference

Comprehensive reference for the Cantera `Kinetics` class API, covering
production rates, Jacobians, reaction access, and integration patterns for CFD
source terms. Based on the Cantera user guide (`kinetics_transport.cpp`,
`cxx-tutorial.md`) and the DNDSR `ChemicalSource` wrapper.

## Object Creation

```cpp
#include "cantera/core.h"
#include "cantera/kinetics/Reaction.h"  // needed for rxn->equation()

auto sol = Cantera::newSolution("gri30.yaml", "gri30");
auto kin = sol->kinetics();  // std::shared_ptr<Cantera::Kinetics>
```

The `Kinetics` pointer is non-owning — the `Solution` must outlive it.

## Header Files

| Header | Status |
|--------|--------|
| `"cantera/core.h"` | Current (3.0+); includes Kinetics |
| `"cantera/kinetics/Reaction.h"` | Needed for `Reaction::equation()` access |
| `"cantera/kinetics.h"` | **Deprecated** since 3.2, removed in 3.3 |

## Reaction Count and Access

```cpp
size_t nRxn = kin->nReactions();

// Per-reaction access
for (size_t i = 0; i < nRxn; i++) {
    auto rxn = kin->reaction(i);          // shared_ptr<Reaction>
    std::string eqn = rxn->equation();     // "H2 + O <=> H + OH"
    // Note: requires #include "cantera/kinetics/Reaction.h"
}
```

### Stoichiometric Coefficients

```cpp
size_t kCO2 = gas->speciesIndex("CO2");
for (size_t i = 0; i < kin->nReactions(); i++) {
    double nu_r = kin->reactantStoichCoeff(kCO2, i);   // reactant side
    double nu_p = kin->productStoichCoeff(kCO2, i);    // product side

    // Find reactions involving a given species
    if (nu_r != 0 || nu_p != 0) {
        // reaction i involves CO2
    }
}
```

Stoichiometric coefficients for reactants are positive (consumed); for products
they are positive (produced). To get the net stoichiometric coefficient for a
species in a reaction: `nu_net = nu_p - nu_r`.

### Filtering Reactions by Species Involvement

```cpp
// Find all reactions involving CO2
size_t kCO2 = gas->speciesIndex("CO2");
for (size_t i = 0; i < kin->nReactions(); i++) {
    if (kin->reactantStoichCoeff(kCO2, i) || kin->productStoichCoeff(kCO2, i)) {
        auto rxn = kin->reaction(i);
        std::cout << i << "  " << rxn->equation() << std::endl;
    }
}
```

## Production Rates

### `getNetProductionRates(wdot)` — Species Net Production Rates

```cpp
std::vector<double> wdot(nSpecies);
kin->getNetProductionRates(wdot.data());
```
- `wdot[k]` = net molar production rate of species k [kmol/m³/s]
- Defined as: ω_k = Σ_i ν_{k,i} · q_i  where q_i is the rate of progress of reaction i
- Equivalent to `getCreationRates() - getDestructionRates()`

### `getCreationRates(cdot)` and `getDestructionRates(ddot)`

```cpp
kin->getCreationRates(cdot.data());       // creation rates [kmol/m³/s]
kin->getDestructionRates(ddot.data());    // destruction rates [kmol/m³/s]
```

### `getNetRatesOfProgress(qdot)` — Per-Reaction Rates

```cpp
std::vector<double> qdot(nReactions);
kin->getNetRatesOfProgress(qdot.data());  // rate of progress per reaction [kmol/m³/s]
```
- `qdot[i]` = net rate of progress of reaction i
- Relationship: ω_k = Σ_i (ν_{k,i}^{prod} - ν_{k,i}^{reac}) · qdot_i

## Jacobian / Sensitivity Derivatives

Cantera provides analytical (or numerically-augmented) derivatives of production
rates with respect to temperature, pressure, and species concentrations.

### Temperature Derivative ∂ω/∂T

```cpp
std::vector<double> dwdt(nSpecies);
kin->getNetProductionRates_ddT(dwdt.data());
```
- `dwdt[k]` = ∂ω_k / ∂T  [kmol/m³/s/K]
- Used in CFD to compute ∂ω/∂(ρE) = ∂ω/∂T · (∂T/∂(ρE))

### Pressure Derivative ∂ω/∂P

```cpp
std::vector<double> dwdp(nSpecies);
kin->getNetProductionRates_ddP(dwdp.data());
```
- `dwdp[k]` = ∂ω_k / ∂P  [kmol/m³/s/Pa]
- Used in CFD to compute ∂ω/∂ρ contribution

### Concentration Jacobian ∂ω_i/∂C_k

```cpp
auto dWdC = kin->netProductionRates_ddCi();  // Eigen::SparseMatrix<double>
double val = dWdC.coeff(i, k);               // ∂ω_i / ∂C_k  [s⁻¹]
```
- Returns an `Eigen::SparseMatrix<double>` of size Ns × Ns
- `dWdC.coeff(i, k)` = ∂ω_i / ∂C_k where C_k is concentration [kmol/m³]
- Access via `.coeff()` (sparse, may be zero if no direct dependence)

### Full Jacobian Assembly for CFD (from ChemicalSource.cpp:170-213)

The conservative-variable Jacobian ∂ω/∂U chains Cantera's composition-sensitivity
derivatives through the chain rule. Here's the complete pattern:

```cpp
// U = [ρ, ρu, ρv, ρw, ρE, ρY_0..ρY_{Ns-2}]
// ω_i in kmol/m³/s, converted to kg/m³/s by multiplying by M_k

kin->getNetProductionRates(wdot.data());
kin->getNetProductionRates_ddT(dwdt.data());
kin->getNetProductionRates_ddP(dwdp.data());
auto dWdC = kin->netProductionRates_ddCi();

double cv = gas->cv_mass();
double cp = gas->cp_mass();
double rho = /* mixture density */;

// ∂ω/∂(ρY_k) = ∂ω/∂C_k · 1/M_k
// Column index: 5 + k  (5 = 1(density) + dim(momentum) + 1(energy))
for (int k = 0; k < Ns - 1; k++) {
    double invMk = 1.0 / Mw[k];
    for (int i = 0; i < Ns; i++)
        J(i, 5 + k) = dWdC.coeff(i, k) * invMk;
}

// ∂ω/∂(ρE) = ∂ω/∂T · 1/(ρ·cv)
double dT_drhoe = 1.0 / (rho * cv);
for (int i = 0; i < Ns; i++)
    J(i, 4) = dwdt[i] * dT_drhoe;

// ∂ω/∂ρ = ∂ω/∂p · ∂p/∂ρ + Σ_k ∂ω/∂C_k · Y_k/M_k
double dp_drho = cp / cv * p / rho;
for (int i = 0; i < Ns; i++) {
    double d = dwdp[i] * dp_drho;
    for (int kk = 0; kk < Ns; kk++)
        d += dWdC.coeff(i, kk) * Y[kk] / Mw[kk];
    J(i, 0) = d;
}
```

**Important:** The chain-rule conversion requires:
- `cv = cp - R` (perfect gas assumption)
- `∂p/∂ρ = cp/cv · p/ρ` (isentropic relationship)
- `∂T/∂(ρE) = 1/(ρ·cv)` (from e = cv·T for perfect gas)

## Conversion from Cantera to CFD Source Terms

Production rates from Cantera are in **kmol/m³/s**:

```
ω_i^code [kg/m³/s] = ω_i^Cantera [kmol/m³/s] × M_i [kg/kmol]
```

The Jacobian conversion:

```
∂ω_i/∂(ρY_k) = ∂ω_i/∂C_k · 1/M_k
```

where `∂ω_i/∂C_k` is from `dWdC.coeff(i, k)` and `1/M_k` converts per-kmol to
per-kg.

## Full Cantera User Guide Reference

Source file at `external/cfd_externals/repos/cantera/doc/sphinx/userguide/`:
- `kinetics_transport.cpp` — complete example: rate computation, species
  filtering via stoichiometric coefficients, transport property loop
