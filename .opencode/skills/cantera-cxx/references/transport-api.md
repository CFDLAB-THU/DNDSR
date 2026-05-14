# Transport API Detailed Reference

Comprehensive reference for the Cantera `Transport` class API, covering
viscosity, thermal conductivity, diffusion coefficients, and usage patterns
for CFD.

## Object Creation

```cpp
#include "cantera/core.h"

auto sol = Cantera::newSolution("gri30.yaml", "gri30", "default");
auto trn = sol->transport();  // std::shared_ptr<Cantera::Transport>
```

The transport model is selected by the third argument to `newSolution()`:

| String | Transport Model | Description |
|--------|----------------|-------------|
| `"default"` | MixTransport | Mixture-averaged diffusion (default) |
| `"mixture-averaged"` | MixTransport | Same as default |
| `"multicomponent"` | MultiTransport | Multicomponent diffusion (more accurate, heavier) |
| `"none"` or `""` | None | No transport — thermo-only phase |

The `Transport` pointer is non-owning — the `Solution` must outlive it.

## Header Files

| Header | Status |
|--------|--------|
| `"cantera/core.h"` | Current (3.0+); includes Transport |
| `"cantera/transport.h"` | **Deprecated** since 3.2, removed in 3.3 |

## Pre-requisite: State Must Be Set

Transport properties depend on the current thermodynamic state and composition
of the associated `ThermoPhase`. **Always call state setters before querying
transport**:

```cpp
gas->setMassFractions_NoNorm(Y.data());
gas->setState_TP(T, P);
// Now query transport:
double mu = trn->viscosity();
```

## Scalar Transport Properties

### `viscosity()` — Mixture Dynamic Viscosity

```cpp
double mu = trn->viscosity();  // [Pa·s]
```

### `thermalConductivity()` — Mixture Thermal Conductivity

```cpp
double lambda = trn->thermalConductivity();  // [W/m/K]
```

### `electricalConductivity()` — Mixture Electrical Conductivity

```cpp
double sigma = trn->electricalConductivity();  // [S/m]
```

## Array Transport Properties

### `getMixDiffCoeffs(D)` — Mixture-Averaged Diffusion Coefficients

```cpp
std::vector<double> D(nSpecies);
trn->getMixDiffCoeffs(D.data());  // D[k] = D_{k,m} [m²/s]
```

Returns the mixture-averaged diffusion coefficient for each species. These
satisfy: `Σ_k Y_k · D_{k,m} ≠ 0` in general — a correction velocity is needed
for mass conservation.

In CFD, these are typically used as species diffusivities:
- D_i = D_{i,m} (mixture-averaged approach)
- The last species is often transported implicitly via `Σ Y_k = 1`

### `getThermalDiffCoeffs(Dt)` — Thermal Diffusion Coefficients

```cpp
std::vector<double> Dt(nSpecies);
trn->getThermalDiffCoeffs(Dt.data());  // Dt[k] [kg/m/s] — thermal diffusion
```

### `getMixDiffCoeffsMass(D)` — Mass-Based Diffusion Coefficients

```cpp
trn->getMixDiffCoeffsMass(D.data());  // ρ·D_{k,m} [kg/m/s]
```

### `getMixDiffCoeffsMole(D)` — Mole-Based Diffusion Coefficients

```cpp
trn->getMixDiffCoeffsMole(D.data());  // C·D_{k,m} [kmol/m/s]
```

### `getBinaryDiffCoeffs(nSpecies, Dbin)` — Binary Diffusion Coefficients

```cpp
int nPairs = nSpecies * (nSpecies - 1) / 2;  // or nSpecies * nSpecies for full matrix
// ...
```

## Transport Model Selection and Gotchas

### Mixture-Averaged (Default)

- Used with `"default"` or `"mixture-averaged"` transport model string
- Good accuracy-to-cost ratio for most CFD applications
- `getMixDiffCoeffs()` provides D_{k,m} — the diffusivity of species k in the
  mixture

### Multicomponent

- Selected with `"multicomponent"` transport model
- Significantly more expensive but more accurate for:
  - H₂-rich mixtures (light species)
  - Laminar flames with strong differential diffusion
  - High-temperature plasma

### No Transport

- Selected with `""` or `"none"` transport model
- `sol->transport()` returns `nullptr`
- Use this for thermo-only phases (like the dedicated UV-solve phase in DNDSR)

## Temperature Dependence Pattern

Transport properties are strong functions of temperature. The canonical pattern
for evaluating at multiple states:

```cpp
auto trans = sol->transport();
for (double T = 300; T <= 1500; T += 200) {
    gas->setState_TP(T, gas->pressure());
    double mu = trans->viscosity();
    double lambda = trans->thermalConductivity();
    // use mu, lambda...
}
```

Note: `setState_TP` is called in the loop because transport properties depend on
T via molecular collision integrals.

## CFD Usage Pattern (from ChemicalSource.cpp)

In CFD, transport is queried at every cell at every iteration:

```cpp
void ChemicalSource::Impl::setTPY(double T, double p, ConstSpeciesBufferView Y) {
    gas->setMassFractions_NoNorm(Y.data);
    gas->setState_TP(T, p);
}

double ChemicalSource::viscosity(double T, double p, ConstSpeciesBufferView Y) const {
    impl_->setTPY(T, p, Y);
    return impl_->trn->viscosity();
}

double ChemicalSource::thermalConductivity(double T, double p, ConstSpeciesBufferView Y) const {
    impl_->setTPY(T, p, Y);
    return impl_->trn->thermalConductivity();
}

void ChemicalSource::speciesDiffusivity(double T, double p,
                                        ConstSpeciesBufferView Y,
                                        SpeciesBufferView D) const {
    auto &I = *impl_;
    I.setTPY(T, p, Y);
    I.trn->getMixDiffCoeffs(I.bufD.data());
    for (int k = 0; k < I.Ns; ++k)
        D[k] = I.bufD[k];
}
```

### Unit Scaling for CFD

Transport parameters from Cantera are in **physical SI units**. They must be
scaled to code units before use in the flow solver. DNDSR conventions (from
`PhysicsProperties.hpp:8-15`):

```
μ_code  = μ_phys / (rho0 · U0)          // viscosity
κ_code  = κ_phys / (rho0 · U0³)         // thermal conductivity
D_code  = D_phys / U0                   // diffusivity
```

When reference scales are unset (`T0 = rho0 = U0 = 0`), scaling factors default
to 1 (no conversion).

### Transport inside Viscous Fluxes

The mixture viscosity and conductivity feed into the Navier-Stokes viscous
fluxes. Species diffusivities feed into the species diffusion fluxes:

```
J_k = -ρ · D_{k,m} · ∇Y_k   (Fick's law, mixture-averaged)
```

Note that with mixture-averaged diffusion, the sum of diffusive fluxes is not
zero — a correction velocity must be applied for mass conservation.

## Full Cantera User Guide Reference

Source file at `external/cfd_externals/repos/cantera/doc/sphinx/userguide/`:
- `kinetics_transport.cpp` — complete example with transport property evaluation
  across a temperature range
