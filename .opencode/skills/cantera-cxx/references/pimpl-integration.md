# PIMPL Integration Pattern for Cantera

The established pattern for wrapping Cantera in DNDSR is full **PIMPL
(Pointer-to-Implementation) isolation**. This reference documents the pattern
from `src/Euler/Chemistry/ChemicalSource.hpp/.cpp` and shows how to follow it
for new Cantera integration.

## Why PIMPL?

1. **Compile-time isolation** — Cantera headers require Eigen, fmt, yaml-cpp,
   and other heavy dependencies. PIMPL prevents these from polluting every
   translation unit.
2. **Link-time control** — Cantera is a large library. Only the `.cpp` file
   that owns the PIMPL needs to link against it.
3. **API stability** — The public header exposes only plain buffer-view
   structs. Internal Cantera API changes don't require recompilation of
   consumers.
4. **Thread safety** — All Cantera state is hidden behind the PIMPL boundary,
   making it easier to reason about thread safety.

## Architecture

```
Public Header (ChemicalSource.hpp)
├── Zero Cantera includes
├── Buffer-view structs (ConstSpeciesBufferView, SpeciesBufferView, JacobianBufferView)
├── ChemicalSource class (PIMPL — unique_ptr<Impl>)
└── Clean, self-documenting API

Implementation (ChemicalSource.cpp)
├── #include "cantera/core.h"          ← sole Cantera include
├── struct ChemicalSource::Impl {     ← all Cantera types here
│       shared_ptr<Solution> sol;
│       ThermoPhase *gas, *kin, *trn;
│       Work buffers...
│       void setTPY(T, p, Y) { ... }
│   }
└── All methods delegate to impl_
```

## Step-by-Step Pattern

### 1. Public Header: Buffer-View Structs

These are plain `{double*, int}` structs for zero-overhead interop with
`Eigen::Map`:

```cpp
struct ConstSpeciesBufferView {
    const double *data = nullptr;
    int nSpecies = 0;
    double operator[](int i) const { return data[i]; }
};

struct SpeciesBufferView {
    double *data = nullptr;
    int nSpecies = 0;
    double &operator[](int i) { return data[i]; }
    double operator[](int i) const { return data[i]; }
};

struct JacobianBufferView {
    double *data = nullptr;
    int rows = 0;  // Ns
    int cols = 0;  // nVars
    int ld = 0;    // leading dimension (== rows for dense ColMajor)
    double &operator()(int i, int j) { return data[i + j * ld]; }
    double operator()(int i, int j) const { return data[i + j * ld]; }
};
```

### 2. Public Header: PIMPL Class

```cpp
class ChemicalSource {
public:
    ChemicalSource(const std::string &mechanismFile,
                   const std::string &phaseName = "");
    ~ChemicalSource();

    // Non-copyable, movable
    ChemicalSource(const ChemicalSource &) = delete;
    ChemicalSource &operator=(const ChemicalSource &) = delete;
    ChemicalSource(ChemicalSource &&) noexcept;
    ChemicalSource &operator=(ChemicalSource &&) noexcept;

    // Accessors
    int nSpecies() const;
    int nReactions() const;
    const std::vector<double> &molecularWeights() const;

    // Evaluation methods — take T, p, Y and write to caller buffers
    double mixtureCp(double T, ConstSpeciesBufferView Y) const;
    void productionRates(double T, double p, ConstSpeciesBufferView Y,
                         SpeciesBufferView omega) const;
    double viscosity(double T, double p, ConstSpeciesBufferView Y) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

### 3. Implementation: Impl Struct

```cpp
struct ChemicalSource::Impl {
    std::shared_ptr<Cantera::Solution> sol;    // owns everything
    Cantera::ThermoPhase *gas = nullptr;        // non-owning
    Cantera::Kinetics    *kin = nullptr;        // non-owning
    Cantera::Transport   *trn = nullptr;        // non-owning

    int Ns = 0;
    std::vector<std::string> speciesNames;
    std::vector<double> mw;      // molecular weights [kg/kmol]
    std::vector<double> Rk;      // species gas constants R/M_k [J/kg/K]

    // mutable work buffers (allocate once, reuse across calls)
    mutable std::vector<double> bufOmega;   // Ns
    mutable std::vector<double> bufDwdt;    // Ns
    mutable std::vector<double> bufDwdp;    // Ns
    mutable std::vector<double> bufDwdc;    // Ns × Ns
    mutable std::vector<double> bufD;       // Ns

    void setTPY(double T, double p, ConstSpeciesBufferView Y) {
        gas->setMassFractions_NoNorm(Y.data);
        gas->setState_TP(T, p);
    }
};
```

### 4. Implementation: Constructor

```cpp
ChemicalSource::ChemicalSource(const std::string &mechanismFile,
                               const std::string &phaseName)
    : impl_(std::make_unique<Impl>())
{
    auto &I = *impl_;

    // Main phase with transport
    I.sol = Cantera::newSolution(mechanismFile, phaseName, "default");
    I.gas = &(*I.sol->thermo());
    I.kin = &(*I.sol->kinetics());
    I.trn = &(*I.sol->transport());

    // Cache species metadata (avoid Cantera calls in hot path)
    I.Ns = static_cast<int>(I.gas->nSpecies());
    I.speciesNames.resize(I.Ns);
    I.mw.resize(I.Ns);
    I.Rk.resize(I.Ns);
    I.gas->getMolecularWeights(I.mw.data());
    for (int k = 0; k < I.Ns; ++k) {
        I.speciesNames[k] = I.gas->speciesName(k);
        I.Rk[k] = Cantera::GasConstant / I.mw[k];
    }

    // Pre-allocate work buffers
    I.bufOmega.resize(I.Ns);
    I.bufDwdt.resize(I.Ns);
    I.bufDwdp.resize(I.Ns);
    I.bufDwdc.resize(I.Ns * I.Ns);
    I.bufD.resize(I.Ns);
}
```

### 5. Implementation: Evaluation Methods

Each method follows the same pattern: set TPY → query Cantera → copy to caller
buffer.

```cpp
void ChemicalSource::productionRates(double T, double p,
                                     ConstSpeciesBufferView Y,
                                     SpeciesBufferView omega) const
{
    auto &I = *impl_;
    I.setTPY(T, p, Y);  // set state
    I.kin->getNetProductionRates(I.bufOmega.data());  // query
    for (int k = 0; k < I.Ns; ++k)
        omega[k] = I.bufOmega[k];  // copy to caller buffer
}

double ChemicalSource::viscosity(double T, double p,
                                 ConstSpeciesBufferView Y) const
{
    impl_->setTPY(T, p, Y);
    return impl_->trn->viscosity();
}
```

## Dedicated Phase for UV Solves

The DNDSR `ChemicalSource` creates a **second `Solution`** specifically for
`temperatureFromUV` calls:

```cpp
// In constructor:
I.solT = Cantera::newSolution(mechanismFile, phaseName, "");  // no transport
I.gasT = &(*I.solT->thermo());

// In temperatureFromUV:
impl_->gasT->setMassFractions_NoNorm(Y.data);
impl_->gasT->setState_TP(Tinit, p_init);  // warm-start
impl_->gasT->setState_UV(u, v);           // Newton solve
return impl_->gasT->temperature();
```

**Why:** `setState_UV` performs an internal Newton solve that mutates state. If
shared with the main phase (which is also used for TP-based thermo/kinetics
queries), state can become corrupted between calls. A dedicated phase ensures
isolation.

## Work Buffer Pattern

Pre-allocate mutable buffers in `Impl` and reuse them:

```cpp
mutable std::vector<double> bufOmega;    // reusable across evaluate() calls
mutable std::vector<double> bufD;        // reusable
```

These are `mutable` to allow writes from `const` evaluation methods —
thermodynamic evaluation should not need mutable semantic state, only mutable
scratch space.

## Consumer-Side: Y Clamping and Renormalization

The **caller** (not `ChemicalSource`) is responsible for numerical robustness:

```cpp
// In SourceTermContributor.hpp and PhysicsProperties.hpp:

// 1. Extract Y from conservative state
for (int k = 0; k < Ns1; ++k)
    bufY[k] = U[Isp + k] * rhoInv;
bufY[Ns1] = 1.0 - sumY;

// 2. Clamp to [0,1]
for (int k = 0; k < Ns; ++k) {
    if (bufY[k] < 0) bufY[k] = 0;
    if (bufY[k] > 1) bufY[k] = 1;
}

// 3. Renormalize to sum=1
double ySum = 0;
for (int k = 0; k < Ns; ++k) ySum += bufY[k];
if (ySum > 0)
    for (int k = 0; k < Ns; ++k) bufY[k] /= ySum;

// 4. Clamp T ≥ baseTemperature() (per-mechanism species lower bound)
double Tcantera = std::max(T, chem->baseTemperature());

// 5. Use physical pressure (pPhys, not code-scaled p)
chem->productionRates(Tcantera, aux.pPhys, Yv, omega);
```

## Consumer-Side: Unit Scaling

All unit conversion between code-scaled and physical (SI) values happens at the
`PhysicsProperties` boundary, not inside `ChemicalSource`:

```cpp
// PhysicsProperties.hpp — the scaling layer

real toPhysP(real pCode) const { return pCode * p0(); }    // p0 = rho0·U0²
real toPhysT(real TCode) const { return TCode * T0; }      // T0 = reference temp
real toCodeT(real TPhys) const { return TPhys / T0; }

// Usage: convert before Cantera call
real muPhys = chemSrc_->viscosity(T, toPhysP(p), massFractions(U));
return muPhys / (rho0 * U0);  // convert result back to code units
```

`ChemicalSource` always works in SI — it's unaware of code scaling.

## Thread Safety Note

The evaluation methods on `ChemicalContributor` are **not thread-safe** due to
mutable work buffers:

```cpp
// Pre-allocated buffers — allocated once, reused every evaluate() call.
// Thread-unsafe (callers serialise via the SGS sweep over cells).
mutable std::vector<double> bufY;
mutable std::vector<double> bufOmega;
```

Callers achieve parallelism through domain decomposition (each process owns its
cells), not through shared-memory threading within a cell loop.

## Error Handling

The PIMPL wrapper should NOT catch Cantera exceptions internally — let them
propagate to the solver's outer try/catch:

```cpp
// In main solver loop:
try {
    chemSrc_->productionRates(T, p, Y, omega);
} catch (Cantera::CanteraError& err) {
    // Log and handle (e.g., fall back to constant-gamma, abort step, etc.)
}
```

## When to Create a New PIMPL Wrapper

Create a new PIMPL-wrapped class when:
1. Adding a new Cantera feature not covered by `ChemicalSource` (e.g., surface
   chemistry, electrochemistry, or 1D flame integration)
2. The existing wrapper would need to expose Cantera types through its public
   interface
3. You need different lifetime management (e.g., per-thread Cantera instances)

Do NOT add Cantera includes to headers — always encapsulate in a `.cpp` file.

## Reference Files

- `src/Euler/Chemistry/ChemicalSource.hpp` — public header (buffer views, PIMPL class)
- `src/Euler/Chemistry/ChemicalSource.cpp` — implementation (Cantera code)
- `src/Euler/Physics/PhysicsProperties.hpp` — scaling layer (code ↔ SI)
- `src/Euler/SourceTermContributor.hpp` — consumer (ChemicalContributor struct)
