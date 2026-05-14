---
name: cantera-cxx
description: Use when writing or modifying C++ code that uses Cantera for chemical kinetics, thermodynamics, and transport. Covers Solution creation, ThermoPhase state setting, Kinetics/Transport access, the established PIMPL isolation pattern, Jacobian assembly, and project-specific linkage/includes. Trigger on Cantera, ChemicalSource, chemical kinetics, reaction rates, setState_TP, newSolution, or YAML mechanisms.
license: MIT
compatibility: opencode
---

# Cantera C++ API

Cantera C++ API usage in DNDSR. This skill gives you the high-level map;
full API details live in the reference files linked below.

## Quick Start

```cpp
#include "cantera/core.h"
// Never include cantera headers in .hpp files — see [PIMPL pattern](references/pimpl-integration.md)

auto sol = Cantera::newSolution("gri30.yaml", "gri30", "default");
auto gas = sol->thermo();    // non-owning ThermoPhase*
auto kin = sol->kinetics();  // non-owning Kinetics*
auto trn = sol->transport(); // non-owning Transport*

// Set state (mass fractions + T, P) — the CFD hot-path pattern
gas->setMassFractions_NoNorm(Y_data);  // caller ensures sum=1
gas->setState_TP(T, P);

// Query
double mu = trn->viscosity();                         // [Pa·s]
kin->getNetProductionRates(wdot.data());              // ω [kmol/m³/s]
auto dWdC = kin->netProductionRates_ddCi();           // ∂ω_i/∂C_k sparse matrix
```

## Project Linkage

Already wired globally — include `"cantera/core.h"` and link. CMake provides `DNDS_CANTERA_DATA_DIR` for YAML mechanism paths (see `cmake/DndsExternalDeps.cmake:55-92`).

## Reference Files

For comprehensive API details see:

| File | Content |
|------|---------|
| [references/thermo-api.md](references/thermo-api.md) | Full ThermoPhase API — all state setters, scalar/array property access, species queries, equilibrium, naming conventions |
| [references/kinetics-api.md](references/kinetics-api.md) | Full Kinetics API — production rates, all Jacobian derivatives (∂ω/∂T, ∂ω/∂P, ∂ω_i/∂C_k), stoichiometric coefficients, Jacobian assembly for CFD |
| [references/transport-api.md](references/transport-api.md) | Full Transport API — viscosity, thermal conductivity, mixture-averaged vs multicomponent diffusion, binary coefficients, CFD scaling |
| [references/pimpl-integration.md](references/pimpl-integration.md) | DNDSR PIMPL pattern — public header design, Impl struct, construction, work buffers, Y clamping, unit scaling, UV-solve dedicated phase, thread safety |

## Key Design Rules

1. **PIMPL only** — Cantera headers never appear in `.hpp` files. Always follow `src/Euler/Chemistry/ChemicalSource.hpp` pattern.
2. **`_NoNorm` in hot paths** — use `setMassFractions_NoNorm()` (caller normalizes), not `setMassFractions()`.
3. **Dedicated phase for UV** — `setState_UV` does a Newton solve; use a separate `Solution` to avoid state corruption.
4. **Work buffers** — pre-allocate mutable `std::vector` in `Impl`; reuse across calls.
5. **Clamp Y** — clamp to [0,1] + renormalize before every call (done by caller, not by ChemicalSource).
6. **Clamp T ≥ 200 K** — NASA polynomial lower bound.
7. **Physical pressure** — pass `aux.pPhys` (SI Pa), not code-scaled pressure, to Cantera.
8. **Legacy headers banned** — never use `thermo.h`, `kinetics.h`, `transport.h` (deprecated 3.2).

## Constants

| Constant | Value |
|----------|-------|
| `Cantera::GasConstant` | 8314.462... J/kmol/K |
| `Cantera::OneAtm` | 101325 Pa |

## Unwired Configuration Fields

These `ReactiveFlowSettings` fields are declared but **not wired** to Cantera:
`thermoFile`, `transportModel` (hardcoded to `"default"`/`""`), `CFLScale`,
`chemRelaxEps`, `chemAbsTol`, `nSpeciesOverride`. Check before relying on them.

## Upstream User Guide

Source at `external/cfd_externals/repos/cantera/doc/sphinx/userguide/`:
`cxx-tutorial.md`, `demo1a.cpp`, `thermodemo.cpp`, `demoequil.cpp`,
`kinetics_transport.cpp`, `compiling-cxx.md`, `creating-mechanisms.md`,
`input-tutorial.md`, `ck2yaml-tutorial.md`
