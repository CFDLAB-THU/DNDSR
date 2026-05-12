# Reactive Flow Fully-Implicit Solver Design

**Status:** Design proposal  
**Target:** Extend DNDSR Euler solvers to support multi-species reactive flow
with coupled fully-implicit time integration using full chemical Jacobian blocks.  
**Branch:** `eulerEX` / `eulerEX3D` (extensible EulerModel variants with `Eigen::Dynamic` nVars)

> **Scope note:** This document describes the full-block coupled implicit approach
> for immediate implementation. Partial-decoupling (point-implicit chemistry) is
> described as a **future extension** (§3.4.3) for when species counts exceed
> ~10 and the full Jacobian block becomes the performance bottleneck.

---

## 1. Problem Statement

The current Euler solver family (NS, NS_SA, NS_2EQ and their 3D variants) solves
compressible Navier-Stokes with turbulence transport equations. The `eulerEX`
and `eulerEX3D` variants (`EulerModel::NS_EX = 101`, `NS_EX_3D = 102`) already
support runtime-determined numbers of conservation variables
(`getnVarsFixed(model) == Eigen::Dynamic`), designed as the extensible backbone
for multi-species and other advected scalars.

### Current state of `eulerEX`/`eulerEX3D`

1. **Dynamic variable count.** `EulerSolver<NS_EX>` takes `nVars` at construction.
   The base 5 variables (ρ, ρu, ρv, ρw, E) occupy indices [0..I4], and extra
   scalars occupy indices [I4+1..nVars-1]. The Eigen sequence `SeqI52Last`
   indexes the extension region.

2. **Passive scalar advection.** Extra variables are advected by the inviscid
   flux (treated as passive scalars transported at velocity un in
   `fluxJacobian0_Right_Times_du`, `EulerEvaluator.hpp:1154-1158`). The
   `passiveDiscardSource` flag (optional) suppresses source terms for them.

3. **No chemistry.** Source terms are computed by `EulerEvaluator::source()`
   (`EulerEvaluator_EvaluateDt.hxx:1532`), which dispatches via `if constexpr
   (Traits::hasSA)` / `if constexpr (Traits::has2EQ)` / `else DNDS_assert(false)`.
   There is no branch for NS_EX — the fallthrough `else` triggers an abort.

4. **Jacobian is block-diagonal.** The implicit solver uses
   `JacobianDiagBlock<nVarsFixed>` in scalar-diagonal or matrix-block mode
   (`EulerJacobian.hpp`). Source term Jacobians are added **diagonally** only:
   `jacobian += retInc.asDiagonal()` (lines 1653, 1705). Full chemical coupling
   (off-diagonal ∂ω_i/∂Y_j and ∂ω_i/∂T) is absent.

5. **No multi-species thermodynamics.** Gas properties use constant γ, R, μ
   (`IdealGasProperty`, `EulerEvaluatorSettings.hpp`). No mixture EOS,
   species-dependent transport, or reaction rate evaluation exists.

### What we must add

| Capability | Current | Target |
|---|---|---|
| Species transport | Passive scalar advection | Active species with chemical production |
| Thermodynamics | Single-species ideal gas (constant γ, R) | Multi-species mixture EOS (NASA polynomials) |
| Transport | Constant μ, Pr | Mixture-averaged μ, κ, D_i (Wilke, Mathur) |
| Chemistry source | None | Arrhenius kinetics, reduced mechanisms |
| Source Jacobian | Diagonal only | Full block (∂ω/∂U) with off-diagonal coupling |
| Implicit coupling | Block-diagonal, SGS | Full Jacobian blocks, SGS + FGMRES |
| BCs | Single-species farfield/wall/inflow | Multi-species with mass fractions |

---

## 2. Design Strategy: Graceful Extension

The guiding principle is **minimal structural change to the existing solver**
while enabling reactive flow as a configuration-time (not compile-time) choice.
We leverage three existing extensibility points:

1. **`EulerModel::NS_EX` / `NS_EX_3D`** — already handles `Eigen::Dynamic` nVars,
   all arrays are allocated dynamically, and MPI/distribution works unchanged.

2. **`source()` callback signature** — already supports 3 modes (source vector,
   diagonal Jacobian, full Jacobian block). We populate the full-block mode (2)
   with actual chemical Jacobian data.

3. **`JacobianDiagBlock::isBlock()`** — when in matrix-block mode, stores full
   nVars×nVars matrices per cell. The SGS and FGMRES solvers already operate on
   these blocks via `getBlock(iCell)`. Enlarging the block from 5×5 to
   nVars×nVars and filling in the chemical coupling changes no solver code.

### Non-goals (for this design)
- Finite-rate wall chemistry (catalytic walls)
- Multi-phase / multi-fluid models
- GPU acceleration of chemistry (stays on CPU like the rest of the Euler module)
- Operator-split or fractional-step methods (we target fully-coupled implicit)
- Point-implicit / partial-decoupling chemistry — **deferred to a future phase** (§3.4.3)

---

## 3. Architecture Changes by Layer

### 3.1 Model & Traits Layer (`Euler.hpp`)

**Add new `EulerModel` enumerants** (or reuse `NS_EX` with a configuration flag):

```cpp
enum EulerModel {
    // ... existing ...
    NS_EX = 101,    // Extended NS, 2D geom, dynamic nVars (existing)
    NS_EX_3D = 102, // Extended NS, 3D geom, dynamic nVars (existing)
};
```

We propose **reusing `NS_EX`** and distinguishing reactive mode via a new config
flag `eulerSettings.reactiveFlow.enabled`. This avoids duplicating all the
explicit template instantiations.

**Add reactive traits to `EulerModelTraits`:**

```cpp
template <EulerModel model>
struct EulerModelTraits {
    // ... existing traits ...
    static constexpr bool isReactive = false; // override for reactive models
    static constexpr int nBaseSpecies = 0;    // override for reactive models
};
```

A separate reactive-specialized traits struct can be used when the reactive flag
is set:

```cpp
template <EulerModel model, bool Reactive = false>
struct ExtendedModelTraits : EulerModelTraits<model> {
    static constexpr bool isReactive = Reactive;
    static constexpr int numThermoVar = Reactive ? 1 : 0; // T or ρe
};
```

**Update `getnVarsFixed` / `getNVars`** logic so that:
- When reactive mode is active, `nVars = 5 + nSpecies` (conservative: ρ, ρu, ρv, ρw, E, ρY_1..ρY_{Ns-1})
- Species mass fractions sum to 1, so one species is redundant (Y_{Ns} = 1 - sum Y_i)

The `nVars` value is passed from the CLI argument `field_n_variables` (existing
for NS_EX) and validated against the chemistry mechanism at configuration time.

#### 3.1.1 RANS as Dynamic Species (unified eulerEX)

Currently, RANS turbulence models (SA, k-omega SST, k-omega Wilcox, RKE) are
tied to dedicated `EulerModel` enumerants (`NS_SA`, `NS_2EQ`, etc.) via
compile-time `if constexpr (Traits::hasSA)` dispatch in `source()`. This
prevents `eulerEX` from running RANS, and prevents reactive flow from being
combined with turbulence (reactive RANS).

**Observation:** A RANS field is structurally identical to a transported
species with a special source term:
- **SA** adds 1 scalar (ρν̃) at index I4+1 with source S_SA(U, ∇U) and
  eddy viscosity contribution μ_t(ρν̃)
- **2-EQ** adds 2 scalars (ρk, ρω) at indices I4+1, I4+2 with coupled sources
  and eddy viscosity contribution μ_t(k, ω)
- **Chemistry** adds Ns−1 scalars at indices I4+1+nRANS..nVars−1 with
  Arrhenius sources and no direct viscosity coupling

All follow the same pattern: extra transported scalar + per-field source term +
optional viscosity augmentation.

**Design: RANS models become `SourceTermContributor`s** (see 3.3.1). At
configuration time, the RANS model is selected by
`eulerSettings.ransModel = "SA" | "SST" | "Wilcox" | "RKE"`. The appropriate
`RANSSourceContributor` is registered, which:
1. Computes eddy viscosity μ_t and adds it to the effective viscosity used in
   the viscous flux (via a shared `EddyViscosityProvider` interface)
2. Computes the RANS source term (production, destruction, diffusion) via the
   existing `RANS::GetSource_*` functions
3. Computes the source Jacobian (diagonal or full, depending on block mode)

**Index layout for combined reactive RANS (example: SA + 9-species methane):**

```
U indices:    0      1     2     3     4        5         6     7  ... 14
Variable:     ρ    ρu    ρv    ρw     E      ρν̃(SA)    ρY_CH4  ρY_O2  ...
Block:        [---- fluid 5×5 ---]  [scalar]  [--- chemical species (Ns-1) ---]
```

The RANS variable sits between the fluid block and the chemical species. Its
source Jacobian couples to the fluid variables (∂S_SA/∂T, ∂S_SA/∂ρu) and to
itself (∂S_SA/∂ρν̃), but not to the chemical species. This natural separation
is captured by the block_scalar Jacobian mode where the RANS entry is treated
as a diagonal scalar plus optional fluid-coupling rows.

**Backward compatibility:** The existing dedicated models (`NS_SA`, `NS_2EQ`)
continue to work unchanged via the existing `if constexpr` dispatch path. The
source contributor approach is activated only when `model == NS_EX` (or
`NS_EX_3D`) is selected, making the existing solvers unaffected by the
refactoring.

```cpp
// In EulerEvaluator constructor, when isExtended:
if (settings.ransModel != RANS_None) {
    sourceContributors.push_back(
        std::make_unique<RANSSourceContributor>(settings.ransModel, ...));
    eddyViscosityProvider = sourceContributors.back()->getMuTProvider();
}
if (settings.reactiveFlow.enabled) {
    sourceContributors.push_back(
        std::make_unique<ChemicalSourceContributor>(mechanism, thermo, transport));
}
```

### 3.2 Thermodynamics & Chemistry Subsystem (`src/Euler/Chemistry/`)

A new sub-directory decouples chemistry from the numeric solver.

#### 3.2.1 Thermodynamics (`ThermoModels.hpp`)

Polymorphic EOS interface:

```cpp
struct ThermoModel {
    virtual ~ThermoModel() = default;
    /// Compute T from ρe, Y_i (conservative → temperature)
    virtual real Temperature(const TU &Ucons, real Rmix) const = 0;
    /// Compute p, c, h from ρ, T, Y_i
    virtual void State(const TU &Ucons, real &p, real &c, real &h, real &gamma) const = 0;
    /// Mixture gas constant
    virtual real RMix(const TU &Ucons) const = 0;
    /// Species enthalpies and heat capacities
    virtual void SpeciesCp(real T, Eigen::Ref<Eigen::VectorXd> cp) const = 0;
    virtual void SpeciesH(real T, Eigen::Ref<Eigen::VectorXd> h) const = 0;
};

struct MultiSpeciesIdealGas : ThermoModel {
    int nSpecies;
    std::vector<Nasa7Polynomial> nasa; // NASA-7 polynomial coefficients
    // ... implementation
};
```

The `MultiSpeciesIdealGas` class encapsulates:
- NASA-7 (or NASA-9) polynomial evaluation for c_p(T), h(T), s(T) per species
- Mixture rules: R_mix = Σ Y_i R_i, h_mix = Σ Y_i h_i(T)
- Newton iteration T ← T - (e_mix(T) - e_target) / c_v,mix(T) for temperature inversion
- All functions return the analytic Jacobian dT/dU for the implicit solver

#### 3.2.2 Transport Properties (`TransportModels.hpp`)

```cpp
struct TransportModel {
    virtual ~TransportModel() = default;
    /// Mixture viscosity
    virtual real Viscosity(real T, const Eigen::Ref<const Eigen::VectorXd> &Y) const = 0;
    /// Mixture thermal conductivity
    virtual real Conductivity(real T, const Eigen::Ref<const Eigen::VectorXd> &Y) const = 0;
    /// Species mass diffusivities D_i
    virtual void Diffusivity(real T, real p,
                             const Eigen::Ref<const Eigen::VectorXd> &Y,
                             Eigen::Ref<Eigen::VectorXd> D) const = 0;
};
```

At least two implementations:
- **`ConstantTransport`** — fixed Pr, Sc, Le = const (for verification)
- **`MixtureAveragedTransport`** — Wilke's rule for μ, Mathur for κ,
  Hirschfelder-Curtiss for D_i, with CHEMKIN-compatible fits

#### 3.2.3 Reaction Mechanism (`ReactionMechanism.hpp`)

```cpp
struct ReactionMechanism {
    int nSpecies;
    int nReactions;
    std::vector<std::string> speciesNames;

    /// Compute chemical production rates ω_i (kmol/m³·s or kg/m³·s)
    /// Inputs: T, p or ρ, Y_i (mass fractions)
    virtual void ProductionRates(real T, real p_or_rho,
                                  const Eigen::Ref<const Eigen::VectorXd> &Y,
                                  Eigen::Ref<Eigen::VectorXd> omega,
                                  bool useSI = true) const = 0;
    /// Compute production rates AND their Jacobian ∂ω/∂U
    /// J_omega[i, j] = ∂ω_i/∂U_j where U_j are the conservative variables
    virtual void ProductionRatesAndJacobian(
        real T, real p_or_rho,
        const Eigen::Ref<const Eigen::VectorXd> &Y,
        Eigen::Ref<Eigen::VectorXd> omega,
        Eigen::Ref<Eigen::MatrixXd> J_omega_rhoY, // ∂ω/∂(ρY_j)
        Eigen::Ref<Eigen::VectorXd> J_omega_T,    // ∂ω/∂T
        bool useSI = true) const = 0;
};
```

Implementations:
- **`ArrheniusMechanism`** — parses CHEMKIN-format mechanism files (species,
  reactions, Arrhenius parameters, third-body efficiencies). Computes forward
  and reverse rates with analytic partial derivatives.
- **`ReducedMechanism`** — quasi-steady-state / partial-equilibrium reduced
  mechanisms wrapping the full mechanism.

**Mechanism I/O:** Use the existing DNDS serializer infrastructure to load
mechanism data from JSON or a custom format. A `DNDS_DECLARE_CONFIG` struct
captures mechanism file paths.

#### 3.2.4 Chemistry Plug-in Registration

Use the existing `DNDS::Config` parameter framework for runtime selection:

```cpp
DNDS_DECLARE_CONFIG(ReactiveFlowSettings) {
    bool enabled = false;
    std::string mechanismFile;      // path to CHEMKIN mechanism
    std::string thermoFile;         // path to NASA polynomial database
    std::string transportFile;      // path to transport data file
    std::string transportModel = "MixtureAveraged";
    std::string thermoModel = "NASA7";
    Real CFLScale = 1.0;           // CFL reduction factor for stiff chemistry
    Real chemRelaxEps = 1e-3;      // pseudo-transient relaxation for chemistry
    Real chemAbsTol = 1e-10;       // absolute tolerance for species
    int nSpeciesOverride = 0;      // 0 = read from mechanism
};
```

This lives under `eulerSettings.reactiveFlow` in the JSON config. Validation
(`post_read` hook) checks that `nVars == 5 + nSpecies`.

### 3.3 Source Term Extension (`EulerEvaluator`)

The critical change is in `EulerEvaluator::source()` and the RHS evaluation.

#### 3.3.1 Source Function Refactor

Replace the monolithic `if constexpr` chain with a **composable source term stack**:

```cpp
// In EulerEvaluator:
std::vector<std::shared_ptr<SourceTermContributor>> sourceContributors;
```

Each contributor implements:

```cpp
struct SourceTermContributor {
    virtual ~SourceTermContributor() = default;
    /// Evaluate source at a quadrature point
    virtual void evaluate(
        const TU &U, const TDiffU &GradU, const Geom::tPoint &pPhy,
        int mode, // 0=value, 1=diag jac, 2=full block jac
        TU &sourceVal,
        TJacobianU &jacobian) const = 0;
};
```

Existing sources become contributors:
- `ConstantMassForceContributor`
- `RotatingFrameContributor`
- `AxisymmetricContributor`
- `RANSSourceContributor` (wraps the SA / 2-EQ dispatch)

New for reactive flow:
- `ChemicalSourceContributor` — wraps the `ReactionMechanism`, computes ω(T,Y)
  and its full Jacobian, maps it to the conservative-variable Jacobian

The `source()` method iterates over contributors:

```cpp
TU EulerEvaluator::source(const TU &U, const TDiffU &GradU,
                          const Geom::tPoint &pPhy, TJacobianU &jac,
                          index iCell, index ig, int Mode)
{
    TU ret = TU::Zero(U.size());
    jac.setZero(U.size(), U.size());
    for (auto &c : sourceContributors)
        c->evaluate(U, GradU, pPhy, Mode, ret, jac);
    return ret;
}
```

This is backward-compatible: if `reactiveFlow.enabled == false`, the chemistry contributor is not registered, and the code path is identical.

#### 3.3.2 Full Jacobian Block for Chemistry

The key improvement: **when `Mode == 2`, the `ChemicalSourceContributor` fills
the entire `jacobian` matrix**, not just the diagonal. This requires mapping
the chemical Jacobian from (T, Y) space to the conservative variable indices:

```
U = [ρ, ρu, ρv, ρw, E, ρY₁, ..., ρY_{Ns-1}]
```

The chemical production rates ω_i depend on (T, ρY_j). The chain rule gives:

```
∂ω_i/∂(ρY_j) = (1/ρ) · (∂ω_i/∂Y_j) - (∂ω_i/∂Y_k · Y_k) / ρ  [mass fraction constraint]
∂ω_i/∂(ρe)   = (∂ω_i/∂T) / (ρ · c_v,mix)                     [temperature sensitivity]
∂ω_i/∂ρ      = (∂ω_i/∂T) · (∂T/∂ρ)                            [density sensitivity]
∂ω_i/∂(ρu_k) = 0                                               [velocity decoupling]
```

These populate **dense blocks** in the lower-right (species-species) region of
the Jacobian, plus coupling columns for ρ and E. The fluid variables
(ρ, ρu, ρv, ρw, E) get contribution rows from the total energy source
(sum of species formation enthalpies × ω_i).

#### 3.3.3 Species Diffusion Flux

Add a Fickian diffusion flux contribution to `EvaluateRHS` for the species
equations:

```
F_diff,i = -ρ D_i ∇Y_i
```

where D_i is the mixture-averaged diffusivity. The flux Jacobian for these
terms is diagonal in species space (since D_i depends weakly on composition).
This is handled alongside the existing viscous flux, with the Jacobian
contribution added to the face flux Jacobian in `LUSGSMatrixInit`.

The Soret effect (thermal diffusion) and Dufour effect can be added as optional
contributors.

### 3.4 Jacobian Assembly (`LUSGSMatrixInit` / `UpdateSGS`)

The existing `LUSGSMatrixInit` (`EulerEvaluator.hxx:37`) computes:

```
J_diag[i] = (V/Δτ + 1/Δt) * I
          + α * Σ_faces(0.5 * Area_f / V * (spectral_radius + R_jacobian))
          + α * J_source[i]
```

Currently `J_source` is the **diagonal** of the source Jacobian. For reactive
flow, we upgrade to the **full block**:

```cpp
if (settings.reactiveFlow.enabled)
    JDiag.getBlock(iCell) += alphaDiag * JSource.getBlock(iCell);
// JSource.getBlock(iCell) now contains the full nVars×nVars source Jacobian
```

**No changes to `UpdateSGS` or the FGMRES solver** are needed, because:
- `UpdateSGS` already operates on full matrix blocks via `getBlock(iCell)` when `isBlock()` is true
- The matrix-vector product `LUSGSMatrixVec` (FGMRES matvec) already multiplies by the full block
- The direct LU preconditioner already factorizes the full block

The only change: ensure `SetModeAndInit(mode=1, ...)` is called (matrix-block mode)
when reactive flow is enabled, and the block size is `nVars × nVars`.

#### 3.4.1 Jacobian Storage Mode 2: Hybrid Block-Scalar (block_scalar) — **future feature**

> **Deferred.** The block_scalar mode is described here for completeness and will
> be needed when the point-implicit decoupling (§3.4.3) is implemented. The
> **immediate implementation uses full-block mode only** (Mode 1, existing), since
> initial targets have nSpecies ≤ 5.

For completeness, the design sketch is:

The fully-coupled approach (section 3.4) stores a dense nVars×nVars matrix per
cell. For realistic mechanisms with 20–50+ species this is costly: a 35×35
block is 35²/5² = 49× larger than the base 5×5 fluid block, and the solver
per-cell inversion of the diagonal block scales as O(nVars³). Memory and FLOPs
grow cubically with species count.

We introduce a **third Jacobian storage mode** in `JacobianDiagBlock` that
stores a hybrid:

```
Mode 0: scalar-diagonal  – one scalar per variable per cell (existing)
Mode 1: full-block       – one nVars×nVars dense matrix per cell (existing)
Mode 2: block-scalar     – [I4+1 × I4+1] dense block for fluid +
                           (nVars − (I4+1)) scalars for transported species
```

The per-cell storage shrinks from nVars² to (I4+1)² + (nVars − (I4+1)).

| nSpecies | nVars | Full-block entries | Block-scalar entries | Savings |
|----------|-------|-------------------|---------------------|---------|
| 5 | 10 | 100 | 25+5 = 30 | 70% |
| 10 | 15 | 225 | 25+10 = 35 | 84% |
| 20 | 25 | 625 | 25+20 = 45 | 93% |
| 50 | 55 | 3025 | 25+50 = 75 | 98% |

**Implementation in `JacobianDiagBlock`:**

The `SetModeAndInit` method gains a tri-state `_mode` field:

```cpp
enum JacobianMode { ScalarDiag = 0, FullBlock = 1, BlockScalar = 2 };

void SetModeAndInit(JacobianMode mode, int nVarsC, ArrayDOFV<nVarsFixed> &mock)
{
    _mode = mode;
    int fluidSize = getDim_Fixed(model) + 1; // I4+1 = 5 for dim=3
    if (_mode == FullBlock) {
        _data.InitPair(...);
        _data.father->Resize(mock.father->Size(), nVarsC, nVarsC);
    } else if (_mode == BlockScalar) {
        _fluidData.InitPair(...);     // ArrayEigenMatrix<I4+1, I4+1>
        _scalarData.InitPair(...);    // ArrayDOFV<nVarsFixed - (I4+1)>
        _fluidData.father->Resize(mock.father->Size(), fluidSize, fluidSize);
        _scalarData.father->Resize(mock.father->Size(), nVarsC - fluidSize, 1);
    } else { /* ScalarDiag, existing */ }
}
```

**Matrix-vector operations** dispatch on mode:

```
MatVecLeft(iCell, v):
  Mode 0 (scalar):    result = diag * v
  Mode 1 (full):      result = block * v
  Mode 2 (hybrid):    result[0..I4]     = fluidBlock * v[0..I4]
                                        + fluid_species_coupling * v[I4+1..end]
                      result[I4+1..end] = scalarDiag * v[I4+1..end]
                                          + species_fluid_coupling * v[0..I4]
```

The off-diagonal coupling (fluid→species, species→fluid) can optionally be
zeroed, which gives pure operator splitting. When retained, it provides
one-way coupling in the SGS sweep.

#### 3.4.2 Jacobian Scaling for Stiff Chemistry

Chemical timescales are often much smaller than fluid timescales. We introduce
a **pseudo-transient continuation** (Ψtc) approach:

```
J_chem = (1/Δτ_chem) * I + ∂ω/∂U
```

where `Δτ_chem = CFL_chem × (characteristic chemical time)`. This is added to
the existing diagonal contribution. As the solution converges, `Δτ_chem → ∞`
and the Jacobian becomes the exact Newton Jacobian.

Configuration parameters:
- `reactiveFlow.CFLScale` — multiplier on the fluid CFL for chemical pseudo-time
- `reactiveFlow.chemRelaxEps` — minimum chemical pseudo-time step floor

#### 3.4.3 Partial Decoupling: Point-Implicit Chemistry — **future feature**

> **Deferred.** This approach is described for the long-term roadmap when species
> counts exceed ~10 and the full-block SGS/GMRES becomes performance-limited.
> The **immediate implementation does NOT include point-implicit chemistry.**
> All solver paths use full-block Jacobian with all chemical couplings active.

**Motivation (future):** For mechanisms with many species (10+), the
block_scalar storage — still imposes a chemical source Jacobian that couples
every species pair inside each cell's diagonal block. This becomes the dominant
cost in the SGS/FGMRES solver, where every sweep involves O(nSpecies²) BLAS
operations per cell *per face*. For a 35-species mechanism, this is ~50× more
expensive per cell than the fluid-only solver.

**Idea: pseudo-time operator splitting between spatial transport and chemical
source.** The implicit linear system at each SGS sweep is factored into two
decoupled solves:

```
Step 1 — Flow + passive-species transport (existing SGS/GMRES):
    J_flow  * ΔU_flow  = -R_flow
    J_passive * Δ(ρY)  = -R_species_transport

    // Fluid block is [I4+1]×[I4+1], species are N diagonal scalars.
    // Jacobian storage: block_scalar mode with species coupling zeroed.
    // This is the existing implicit solver with no chemistry Jacobian.

Step 2 — Point-implicit chemical relaxation (per cell, independent):
    (I/Δτ_chem - ∂ω/∂(ρY)) * Δ(ρY)_chem = ω(U_after_step1)
    U_final = U_after_step1 + [0, 0, 0, 0, 0, Δ(ρY)_chem]
```

**What this achieves:**

1. The SGS/GMRES solve operates on a system where the Jacobian block for fluid
   is at most 5×5 (or 6×6 for SA model) and species are scalar diagonals. This
   is identical in cost to the non-reactive solver, regardless of nSpecies.

2. After each SGS sweep (or after the full linear solve), each cell solves a
   small (Ns-1)×(Ns-1) dense linear system for the chemical increment. This is
   an **embarrassingly parallel** per-cell operation: no face stencil, no MPI
   communication, just a local LU solve.

3. The chemical Jacobian ∂ω/∂(ρY) is the same analytic Jacobian used in the
   fully-coupled approach — but only the species-species block is needed. The
   ∂ω/∂T and ∂ω/∂ρ rows (coupling to fluid) are discarded in this scheme.

4. For N_s species, the per-cell solve is O((N_s−1)³) in the worst case, but
   since it's local (no stencil, no comms), it parallelizes trivially with
   OpenMP. In practice this is negligible compared to the face loop.

**SGS integration — three strategies:**

```
Strategy A (after each SGS sweep):
    for sgs_sweep in 1..nSgsIter:
        UpdateSGS(cell_inc, species_inc, J_flow_diag)  // block_scalar
        for each cell i:
            PointImplicitChemistry(u[i])  // local (Ns-1)×(Ns-1) solve

Strategy B (after full linear solve, before PP limiting):
    solve FGMRES with block_scalar Jacobian  // converges spatial part
    for each cell i:
        PointImplicitChemistry(u[i])

Strategy C (half chem → flow → half chem) is not needed: pseudo-time iteration
only requires convergence, not temporal accuracy. The standard Strategies A and B
cover the stability requirements without the overhead of double chemistry evaluation.

**Configuration control:**

```json
{
    "eulerSettings": {
        "reactiveFlow": {
            "pointImplicit": {
                "enabled": true,
                "strategy": "afterSGS",    // "afterSGS" or "afterLinear"
                "subCycles": 1,            // N inner chemical iterations per SGS sweep
                "solverType": "directLU",  // "directLU" or "gmres"
                "gmresRestart": 5
            }
        }
    }
}
```

**When to use which approach:**

| Scenario | Recommended approach |
|----------|---------------------|
| nSpecies ≤ 5, detonation | Full block Jacobian with SGS |
| nSpecies ≤ 5, flame | Full block Jacobian with FGMRES |
| 5 < nSpecies ≤ 15 | Block_scalar + point-implicit (Strategy A) |
| nSpecies > 15 | Block_scalar + point-implicit (Strategy B) |

**Chemical sub-cycling:**

When `subCycles > 1`, the point-implicit step is itself split into N sub-steps
with reduced Δτ_chem/N, effectively solving the stiff ODE system ẏ = ω(y)
with a Rosenbrock-like method. This provides additional stability for
mechanisms with radical species whose timescales span many orders of magnitude.

```
for k in 1..subCycles:
    Δτ_sub = Δτ_chem / subCycles
    (I/Δτ_sub - ∂ω/∂y) · Δy = ω(y)
    y ← y + Δy
```

### 3.5 Boundary Conditions (`EulerBC.hpp`)

Multi-species BCs require species mass fractions at boundaries:

| BC Type | Fluid (unchanged) | Species (new) |
|---------|-------------------|---------------|
| Farfield | Characteristic-based | Fixed Y_i = farfield composition |
| Inflow | Sub/supersonic inflow | Fixed Y_i = inflow composition |
| Outflow | Extrapolation | Extrapolation (zero gradient) |
| Wall (inviscid) | Slip | Zero normal gradient (catalytic: fixed Y_i) |
| Wall (viscous) | No-slip, Twall | Zero mass flux: ∂Y_i/∂n = 0 |
| Symmetry | Mirror | Zero gradient |

**Implementation:** The `generateBoundaryValue` method in `EulerEvaluator`
receives an additional parameter packet for the species state. Since `NS_EX`
already handles the extension region in boundary values (all boundary handlers
copy or set the extension variables), the existing BC dispatch works with
the `bcSettings` vector containing per-BC-zone species mass fractions.

A new BC type `BCInSpecies` could specify fixed species mass fractions for
fuel/oxidizer injection boundaries.

### 3.6 Initial Conditions

Species mass fraction fields must be initializable. Extend the existing
`SpecialBuiltinInitializer` mechanism (`EulerEvaluatorSettings.hpp`) to
include multi-species profiles:

- **Uniform composition:** all cells get the same Y_i (e.g., premixed flame)
- **Box/Plane initializers:** existing mechanism, with Y_i fields added
- **Exprtk initializers:** existing mechanism, with Y_i accessible as `UPrim[5..nVars-1]`
- **Burke-Schumann / equilibrium:** compute equilibrium composition from mixture fraction (for diffusion flames)

### 3.7 Output

Species mass fractions must be outputtable. The existing VTK/HDF5/Tecplot
output infrastructure already handles arbitrary field names via
`outCellScalarNames`. We add automatic registration of species names from
the mechanism:

- `rhoY_H2`, `rhoY_O2`, `rhoY_H2O`, ... (conservative, dimensional)
- `Y_H2`, `Y_O2`, `Y_H2O`, ... (mass fraction, non-dimensional)
- `T` (temperature, computed from EOS)
- `omega_H2`, ... (chemical production rates)

### 3.8 Configuration Example

A reactive flow case configuration extends the existing eulerEX config:

```json
{
    "eulerSettings": {
        "reactiveFlow": {
            "enabled": true,
            "mechanismFile": "data/chem/h2_o2.yaml",
            "thermoFile": "data/chem/thermo.dat",
            "transportFile": "data/chem/transport.dat",
            "transportModel": "MixtureAveraged",
            "CFLScale": 0.1,
            "chemRelaxEps": 0.001
        },
        "farFieldStaticValue": [
            1.0,     // ρ
            0.0,     // ρu
            0.0,     // ρv
            0.0,     // ρw
            2.5,     // E
            0.028,   // ρY_H2
            0.222    // ρY_O2
            // ρY_H2O = 1 - (Y_H2 + Y_O2) = 0.75 (redundant, computed)
        ]
    },
    "bcSettings": [
        {
            "name": "inflow",
            "type": "BCInPsTs",
            "values": [ 1.0, 300.0, 1.0, 0.0,  0.028, 0.222 ]
        }
    ]
}
```

---

## 4. Data Flow: Implicit Reactive Step

For each implicit time step (pseudo-time iteration):

```
1. Reconstruction
   └─ Same as non-reactive: VFV reconstructs all nVars components
      as a monolithic system. Species gradients contribute to
      diffusion and are used in the chemical source Jacobian.

2. EvaluateDt
   └─ Spectral radii account for multi-species speed of sound c_mix.
      Chemical timescale constraint (optional):
           Δτ_chem = min_i( |Y_i / ω_i| ) × safety_factor
      Enforced via additional dTau clamping.

3. LUSGSMatrixInit
   └─ Jacobian blocks are full nVars×nVars per cell (Mode 1).
      - Fluid-fluid block: same as non-reactive (Roe Jacobian or
        spectral-radius diagonal)
      - Source Jacobian: full reactive block (∂ω/∂U) plus diagonal
        contributions from existing source terms (body force, etc.)
      Off-diagonal face contributions remain as-is (scalar advection
      handled by the existing flux Jacobian for passive scalars).

4. EvaluateRHS
   └─ Inviscid flux: species are advected as passive scalars (existing).
      Viscous flux: adds Fickian diffusion flux for species equations.
      Source term: chemicalProductionRates(T, ρY) evaluated at each
      quadrature point, multiplied by integration weights.

5. Solve (SGS or FGMRES)
   └─ The linear system A·ΔU = R has A = diag(V/Δτ) + flux_jacobian + source_jacobian.
      The source Jacobian provides the critical species-species and
      species-temperature coupling that makes the chemistry implicit.
      SGS sweeps and FGMRES operate on the full nVars×nVars blocks
      without any code changes.

6. Convergence Check
   └─ Species residual norms are monitored separately.
      ρY_i tolerances (absolute, relative) can be more stringent
      than momentum/continuity tolerances.
```

> **Future (point-implicit, see §3.4.3):** For large mechanisms, a decoupled
> path would solve J_flow · ΔU = −R with scalar-diagonal species in the SGS
> sweep, followed by a per-cell dense (Ns−1)×(Ns−1) chemical solve. This is
> described in §3.4.3 but is not part of the immediate implementation.

---

## 5. Implementation Roadmap

### Phase 1: Infrastructure (no chemistry, backward-compatible)

1. Add `ReactiveFlowSettings` to `EulerEvaluatorSettings` with `DNDS_DECLARE_CONFIG`
2. Add `enableReactive` flag to `EulerSolver::Configuration`
3. **Refactor `source()` into composable `SourceTermContributor` stack**
   - Extract existing body-force, rotating-frame, axisymmetric sources into contributors
   - Convert RANS source (SA, SST, Wilcox, RKE) dispatch into `RANSSourceContributor`
   - Add `EddyViscosityProvider` interface (shared by RANS contributor, used in viscous flux)
   - Keep existing `if constexpr` dispatch path for dedicated models (NS_SA, NS_2EQ, etc.)
   - Verify: run existing euler/eulerSA/euler2EQ test cases — zero diff
4. Extend `EulerModelTraits` with `isReactive`, `hasRANSFromConfig` traits
5. Wire `nSpecies` from config to `nVars` validation

### Phase 2: Chemistry Subsystem

7. Implement `ThermoModels.hpp`:
   - `MultiSpeciesIdealGas` with NASA-7 polynomials
   - Temperature-from-energy Newton solver
   - Analytic derivatives dT/dρe, dT/d(ρY_i)
8. Implement `TransportModels.hpp`:
   - `ConstantTransport` (fixed Pr, Sc, Le)
   - `MixtureAveragedTransport` (Wilke, Mathur, Hirschfelder-Curtiss)
9. Implement `ReactionMechanism.hpp`:
   - CHEMKIN-format parser (species, thermo, reactions, transport)
   - `ArrheniusMechanism::ProductionRatesAndJacobian()` with analytic ∂ω/∂T and ∂ω/∂Y_j
   - Jacobian assembly: conservative-variable chain rule mapping
10. Implement `ChemicalSourceContributor`

### Phase 3: Solver Integration

11. Modify `LUSGSMatrixInit` to accept full source Jacobian blocks
    - When `isReactive`, switch `JacobianDiagBlock` to full-block mode (Mode 1)
    - Add source block without diagonalization: `JDiag.block[i] += α * JSource.block[i]`
12. Modify `EvaluateRHS` to accept and propagate full source Jacobian
    - The `sourceV` matrix in the quadrature loop already reserves space
      for full blocks when `JSource.isBlock()` — fill in off-diagonals
13. Configure SGS sweeps and FGMRES for larger state vectors
    - Existing values (sgsIter=5, nGmresIter=10) are good starting points
    - Add configurable chemistry-specific SGS sub-iterations if needed

### Phase 4: Verification & Validation

14. Zero-dimensional autoignition (constant-volume reactor)
    - Compare against Cantera reference solutions
    - Verify temporal order of accuracy (1st, 2nd order via BDF/VBDF)
15. One-dimensional laminar premixed flame
    - Compare flame speed against Cantera 1-D free-flame
    - Grid convergence study
16. Two-dimensional lifted flame / diffusion flame
    - Qualitative comparison against literature
17. Reacting RANS (SA + chemistry)
    - Test eulerEX with ransModel=SA + reactiveFlow
    - Verify eddy viscosity enhances mixing and flame spreading

### Phase 5: Future — Large-Mechanism Optimizations

18. **block_scalar Jacobian mode** (Mode 2 in `JacobianDiagBlock`, §3.4.1)
    - Hybrid storage: dense nFluid×nFluid block + nScalar diagonal entries
    - Needed as prerequisite for point-implicit chemistry
19. **Point-implicit chemistry** (§3.4.3)
    - Per-cell (Ns-1)×(Ns-1) dense linear solves after each SGS sweep
    - Chemical sub-cycling for stiff mechanisms
    - Strategies: afterSGS, afterLinear
20. **GPU-accelerated chemistry**
    - Batch-similar cells for GPU-parallelized reaction rate evaluation
    - CUDA kernel for ProductionRates() on `EulerP`-style device arrays

---

## 6. Key Design Principles

### 6.1 Changes localized, not distributed

- The `EulerEvaluator::source()` function is the only place where chemistry
  logic is injected into the flow solver.
- `LUSGSMatrixInit` needs a one-line change to use full blocks instead of
  diagonals for source Jacobian.
- The SGS/FGMRES solvers, ODE integrators, reconstruction, mesh I/O, and
  output all work unchanged.
- Boundary conditions need species mass fractions but use existing BC
  infrastructure (per-zone value vectors).

### 6.2 Existing Jacobian infrastructure is sufficient

The `JacobianDiagBlock` already supports full nVars×nVars blocks via Mode 1.
The SGS sweep and FGMRES matvec already operate on full blocks. The direct LU
preconditioner already factorizes full blocks. **We are not inventing new
linear algebra — we are populating existing structures more densely.**
The new block_scalar mode (Mode 2) is an additive extension; Mode 0 and Mode 1
continue to work identically for non-reactive flows.

### 6.3 RANS is just another transported scalar

SA and 2-equation RANS fields are structurally identical to chemical species:
they add one or two extra conservation equations with specialized source terms
and a viscosity augmentation. By making RANS models into
`SourceTermContributor`s, `eulerEX` becomes the universal solver that can
combine flow + RANS + chemistry in any combination. The dedicated models
(`NS_SA`, `NS_2EQ`) remain as optimized compile-time specializations.

### 6.4 Chemistry is a configuration-time choice

No compile-time template explosion. A single flag
(`reactiveFlow.enabled` in JSON) activates the chemistry subsystem. The same
`eulerEX` binary serves inert, RANS, reactive, and reactive-RANS cases,
avoiding the proliferation of solver executables.

### 6.5 Fallback to existing paths

When `reactiveFlow.enabled == false`:
- The source contributor list is identical to the current hard-coded chain
- The source Jacobian remains diagonal (Mode 1 or Mode 2 with `.asDiagonal()`)
- `JacobianDiagBlock` uses the same mode as before
- CPU cost is identical (zero-overhead abstraction via devirtualized function calls)

### 6.6 Analytic chemical Jacobian

We compute ∂ω/∂U analytically (not by finite differences). Each
`ArrheniusReaction` computes its contribution to the Jacobian via the chain
rule through concentration → temperature → conservative variables. This is
essential for the stiff chemistry implicit solver — FD Jacobians are both
expensive (nVars+1 RHS evaluations) and noisy near equilibrium.

---

## 7. Summary of Changes

| File/Layer | Change | Impact |
|---|---|---|
| `Euler.hpp` | Add `isReactive` to `EulerModelTraits`, reactive-aware nVars logic | ~30 lines |
| `EulerEvaluator.hpp` | Composable `SourceTermContributor` registry (incl. RANS), reactive flow member data | ~80 lines |
| `EulerEvaluator.hxx` | Refactor `source()`, upgrade `LUSGSMatrixInit` for full blocks | ~100 lines |
| `EulerEvaluator_EvaluateRHS.hxx` | Fill full source Jacobian, add species diffusion flux | ~50 lines |
| `EulerEvaluatorSettings.hpp` | Add `ReactiveFlowSettings` sub-struct | ~40 lines |
| `EulerJacobian.hpp` | No changes (existing Mode 1 full-block storage is sufficient) | 0 lines |
| `EulerSolver.hpp/hxx` | Wire reactive config, adjust linear solver hints | ~50 lines |
| `src/Euler/Chemistry/` | New: ThermoModels, TransportModels, ReactionMechanism, ChemSourceContributor | ~2000 lines |
| `src/Euler/SourceTerms/` | New: Extract existing RANS + body-force sources into contributors, add `EddyViscosityProvider` | ~500 lines |
| `cases/eulerEX/` | New example configs (autoignition, premixed flame, RANS+chemistry) | ~200 lines |
| `test/` | New tests (0-D reactor, Cantera cross-validation) | ~500 lines |

**Total estimated lines:** ~3500 lines across ~14 files. The critical paths
(EulerSolver, SGS/GMRES solvers, reconstruction, mesh I/O, output) are
essentially untouched.

### Where changes are NOT needed

| Component | Reason |
|-----------|--------|
| SGS sweeps (`UpdateSGS`) | Already operates on `getBlock(iCell)` — full blocks with nVars×nVars require zero changes |
| FGMRES (`GMRES_LeftPreconditioned`) | Already dispatches via callbacks `FA`/`FML` — unchanged |
| Direct LU preconditioner | Already factorizes nVars×nVars blocks — unchanged |
| ODE integrators (`ODE/`) | Time integration is callback-based, unaware of nVars — unchanged |
| Variational reconstruction (CFV/) | Treats all variables identically — unchanged |
| Mesh I/O (`Geom/Mesh/`) | Mesh is field-agnostic — unchanged |
| MPI distribution | `ArrayDof`/`ArrayPair` already handle DynamicSize distributions — unchanged |
| VTK/HDF5/Tecplot output | Already supports arbitrary field names via `outCellScalarNames` |

---

## 8. Compatibility & Risk Assessment

| Risk | Mitigation |
|------|------------|
| Stiff chemistry causes SGS/FGMRES divergence | Pseudo-transient continuation (Δτ_chem ramping), adaptive CFL |
| Large state vectors slow FGMRES | Chemistry ILU preconditioner block on the species sub-block; outer SGS sweeps unchanged |
| Mechanism parsing errors | Validate at load time, fail with clear error message referencing the problematic line |
| Memory: nVars×nVars blocks for all cells when nSpecies is large | Initial targets have nSpecies ≤ 5; future block_scalar mode (§3.4.1) will address larger mechanisms |
| Multi-species thermodynamics are slow at face quadrature points | Cache mixture properties per face; evaluate chemistry only at cell centers |
| RANS + chemistry interaction (turbulence-chemistry) | Eddy-viscosity provider is a shared interface; RANS μ_t feeds into species diffusion via turbulent Schmidt number |

---

## 9. Example Simulation Configurations

### 9.1 0-D Constant-Volume Autoignition

```json
{
    "timeMarchControl": { "odeCode": 102, "dtImplicit": 1e-6, "tEnd": 0.001 },
    "convergenceControl": { "nTimeStepInternal": 100, "rhsThresholdInternal": 1e-12 },
    "eulerSettings": {
        "reactiveFlow": {
            "enabled": true,
            "mechanismFile": "data/chem/h2_o2.yaml",
            "CFLScale": 10.0
        },
        "specialBuiltinInitializer": 0,
        "idealGasProperty": { "gamma": 1.3, "Rgas": 0.0 },
        "initialConditions": {
            "T": 1200.0, "p": 1e5,
            "Y_H2": 0.028, "Y_O2": 0.222, "Y_N2": 0.75
        }
    }
}
```

### 9.2 2-D Laminar Premixed Flame

```json
{
    "dataIOControl": { "meshFile": "data/mesh/flame_channel.cgns" },
    "eulerSettings": {
        "reactiveFlow": {
            "enabled": true,
            "mechanismFile": "data/chem/ch4_bfer.yaml",
            "transportModel": "MixtureAveraged",
            "CFLScale": 0.5
        },
        "bcSettings": [
            { "name": "inlet", "type": "BCInPsTs", "values": [1.0, 500.0, 1.0, 0.0, "...", 0.055, 0.22, 0.0, 0.0] },
            { "name": "outlet", "type": "BCOutP", "values": [101325] }
        ]
    }
}
```

---

*Design prepared: 2026-05-12. Subject to revision based on feasibility studies.*
