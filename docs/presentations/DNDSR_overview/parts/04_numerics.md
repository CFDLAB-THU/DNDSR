---
<!-- _class: chapter -->
<!-- _paginate: false -->

<div class="ch-num">CHAPTER 4</div>

# Numerics

## CFV · VR · flux · limiters · ODE · Krylov

---
<!-- _footer: "src/CFV/VRDefines.hpp:27 · docs/theory/Variational_Reconstruction.md:21-30" -->

## Compact Finite Volume — the reconstruction

Reconstruct a piecewise polynomial from cell means with a **zero-mean basis** per cell:

$$
u_i(\mathbf{x})
= \overline{u}_i
+ \sum_{l=1}^{N_{\text{base}}} u^l_i\, \varphi^l_i(\mathbf{x})
$$

- $\overline{u}_i$ — cell-mean, lives in `tUDof<N> = ArrayDof<N,1>`.
- $u^l_i$ — reconstruction coefficients, in `tURec<N> = ArrayDof<Dyn,N>`.
- Basis $\varphi^l_i$ is orthogonalized locally, normalized by cell scale; degree chosen at runtime per cell.

**Supported polynomial orders: 1 – 3** (linear, quadratic, cubic).

```cpp
// Static capacities (src/CFV/VariationalReconstruction.hpp:1051-1054)
maxRecDOFBatch = (dim == 2) ?  4 : 10;
maxRecDOF      = (dim == 2) ?  9 : 19;
maxNDiff       = (dim == 2) ? 10 : 20;
maxNeighbour   = 7;
```

The stencil is **one ring of node-neighbors** — which is exactly what `BuildGhostPrimary(1)` provides by default. Wider stencils (`nGhostLayers ≥ 2`) are available for higher-order variants.

---
<!-- _footer: "docs/theory/Variational_Reconstruction.md:33-106" -->
<!-- _class: dense -->

## Variational Reconstruction — the functional

Minimize the jumps of **all derivatives up to order k** across each face:

$$
I_f
= w_g(f)\int_f \sum_{p=0}^{k}
  w_d(p)^2\,
  \bigl\|\mathcal{D}_p u_L - \mathcal{D}_p u_R\bigr\|_{\langle\,,\,\rangle_{f,p}}^{2}\, d\Gamma
$$

<div class="cols">
<div>

**Weights**

- $w_g(f)$ — geometric weight; default $w_g = S_f^{-1}$ (area-inverse).
- $w_d(p)$ — dimensionless derivative weight; selects how aggressively each derivative order contributes.
- $\mathcal{D}_p u$ — the $p$-th derivative tensor (covariant only under linear coordinate changes).

**Local system**

$$
A^{i}_{mn} u^{n}_{i}
= \sum_{j \in S_i}
  \bigl(B^{i{\leftarrow}j}_{mn} u^{n}_{j}
        + b^{i{\leftarrow}j}_{m}(\overline{u}_j-\overline{u}_i)\bigr)
$$

Solved iteratively — options below.

</div>
<div>

**Three inner-product choices**

- **Wang (normal):** $\langle\mathcal{D}_3 u,\mathcal{D}_3 v\rangle = d_f^6\,\partial_{nnn}u\,\partial_{nnn}v$
- **Pan (X-Y aligned):** $\sum (\Delta_x^a\Delta_y^b\partial^{\cdot}_{xy} u)\, (\Delta_x^a\Delta_y^b\partial^{\cdot}_{xy} v)$
- **Huang (pre-isotropic):** $d_f^{2p}$ weighting, directionally isotropic.

**Reconstruction iteration schemes** (`VariationalReconstruction.hpp:938-1031`)

- `DoReconstructionIter` — Jacobi / SOR sweep (tests use Jacobi).
- `DoReconstructionIterDiff` — Jacobian-vector product (GMRES inner).
- `DoReconstructionIterSOR` — SOR with optional reverse pass.
- Fallbacks: `DoReconstruction2nd`, `DoReconstruction2ndGrad`.

</div>
</div>

---
<!-- _footer: "src/CFV/VariationalReconstruction.hpp:282-289" -->
<!-- _class: denser -->

## VR setup — the three `Construct*` calls

```cpp
template <int dim = 2>
class VariationalReconstruction : public FiniteVolume {
public:
    void ConstructMetrics();                                                      // via FiniteVolume
    void ConstructBaseAndWeight(tFGetBoundaryWeight id2faceDircWeight = …);      // basis + cached diff values
    void ConstructRecCoeff();                                                    // A, B, A^-1 B, secondary
    // …
};
```

<div class="cols">
<div>

### What `ConstructMetrics` builds

- Cell volumes, face areas, unit normals, quadrature Jacobians.
- Inertia tensors, major-axis frames, bounding-box scales.
- Physical coords of every quadrature point.
- Smoothness scales for each cell.

### What `ConstructBaseAndWeight` builds

- `cellBaseMoment` — basis moments per cell.
- `faceAlignedScales`, `faceMajorCoordScale`.
- `cellDiffBaseCache`, `faceDiffBaseCache` — cached derivative values at all quadrature points, for every neighbour in the stencil.
- `bndVRCaches` — boundary-face caches for BC-weighted VR.

</div>
<div>

### What `ConstructRecCoeff` builds

- `matrixAB`, `vectorB` — per-neighbor RHS blocks.
- `matrixAAInvB`, `vectorAInvB` — precomputed $A^{-1}B$ to accelerate Jacobi / SOR iterations.
- `matrixSecondary`, `matrixAHalf_GG` — auxiliary reconstruction systems.
- `matrixA`, `matrixACholeskyL`, `volIntCholeskyL` — full system + Cholesky factor for dense local solves.

All arrays are `ArrayEigenMatrix*` or `ArrayEigenUniMatrixBatch*` — i.e., Eigen maps over an MPI-aware distributed memory block.

</div>
</div>

---
<!-- _footer: "src/CFV/FiniteVolume.hpp:38-86" -->
<!-- _class: dense -->

## `FiniteVolume` — the metric cache

```cpp
class FiniteVolume : public DeviceTransferable<FiniteVolume> {
    real sumVolume, minVolume{veryLargeReal}, maxVolume, volGlobal;

    tScalarPair  volumeLocal;         // per-cell volume
    tScalarPair  faceArea;            // per-face area
    tRecAtrPair  cellAtr,  faceAtr;   // (NDOF, NDIFF, Order, intOrder)
    tCoeffPair   cellIntJacobiDet, faceIntJacobiDet;
    t3VecsPair   faceUnitNorm;        // normal at each face quadrature pt
    t3VecPair    faceMeanNorm;
    t3VecPair    cellBary,  faceCent,  cellCent;
    t3VecsPair   cellIntPPhysics, faceIntPPhysics;
    t3VecPair    cellAlignedHBox, cellMajorHBox;
    t3MatPair    cellMajorCoord, cellInertia;
    tScalarPair  cellSmoothScale;

    int axisSymmetric = 0;            // wedge axisymmetry
    std::set<index> axisFaces;

    // CRTP: to_device(), to_host(), device(), deviceView<B>()
};
```

<div class="callout callout-ok">

**CUDA-transferable.** `FiniteVolume` (and therefore `VariationalReconstruction`) inherits from `DeviceTransferable<FiniteVolume>`. One call to `fv.to_device()` migrates the entire metric cache to the GPU as a device-side view.

</div>

---
<!-- _footer: "src/Euler/Gas.hpp:61-95,230" -->
<!-- _class: tight -->

## 13 Riemann solvers

```cpp
enum RiemannSolverType {
    UnknownRS = 0,
    Roe       = 1, HLLC     = 2, HLLEP    = 3, HLLEP_V1 = 21,
    Roe_M1    = 11, Roe_M2  = 12, Roe_M3  = 13, Roe_M4  = 14, Roe_M5 = 15,
    Roe_M6    = 16, Roe_M7  = 17, Roe_M8  = 18, Roe_M9  = 19,
};
```

| Variant | Entropy-fix / eigenvalue scheme |
|---------|---------------------------------|
| `Roe`    | standard Roe + Harten–Yee |
| `Roe_M1` | cLLF (central + Local Lax–Friedrichs) |
| `Roe_M2` | Lax–Friedrichs |
| `Roe_M3` | LD Roe (low-dissipation) |
| `Roe_M4` | ID Roe (intermediate dissipation) |
| `Roe_M5` | LD cLLF |
| `Roe_M6` | H-correction only |
| `Roe_M7` | Harten–Yee only, no H-correction |
| `Roe_M8` | H-correction + Harten–Yee |
| `Roe_M9` | Reserved (eigScheme 9, currently asserts false) |
| `HLLC`   | Harten–Lax–van Leer–Contact |
| `HLLEP`  | HLLE with pressure fix |
| `HLLEP_V1` | HLLEP variant 1 |

```cpp
// Shared helper
template <int dim>
RoePreamble<dim> ComputeRoePreamble(ULm, URm, gamma, dumpInfo);
```

---
<!-- _footer: "src/Euler/Gas.hpp:200-230" -->
<!-- _class: dense -->

## `RoePreamble` — the shared middle

```cpp
template <int dim>
struct RoePreamble {
    TVec veloLm, veloRm;                     // primitive velocities
    real rhoLm, rhoRm, pLm, pRm, HLm, HRm;   // primitive state
    real veloLm0, veloRm0;                   // normal velocity components

    TVec veloRoe;                            // Roe-averaged velocity
    real sqrtRhoLm, sqrtRhoRm;
    real vsqrRoe, HRoe, asqrRoe, rhoRoe, aRoe;
};
```

<div class="cols">
<div>

### Flux signature

```cpp
template <int dim, int eigScheme>
void RoeFlux(UL, UR, ULm, URm, n, vgN,
             /*out*/ flux,
             /*out*/ dLambda,
             fixScale, gamma, dumpInfo);

template <int dim, int type>
void HLLEPFlux_IdealGas(UL, UR, ULm, URm, n, vgN,
                        flux, …, gamma, dumpInfo);

template <int dim>
void HLLCFlux(UL, UR, ULm, URm, n, vgN, …);
```

</div>
<div>

### Why this factoring

All 13 variants share `ComputeRoePreamble` — the Roe average, $H_{\text{Roe}}$, $a_{\text{Roe}}$, etc. The `eigScheme` template parameter then selects the dissipation / entropy-fix strategy.

- **One template instantiation per (`dim`, `eigScheme`)** keeps code size bounded.
- **Compile-time dispatch** — no indirect calls in the flux kernel.
- **Same interface** for inviscid and full Navier-Stokes flux: `NSFluxInvis<dim>`, `NSFluxVis<dim>(U, gradU, T, mu, n, flux, adiabaticWall, useQCR)`.

</div>
</div>

---
<!-- _footer: "src/CFV/Limiters.hpp:28-577" -->
<!-- _class: dense -->

## Limiters — the FWBAP L2 family

<div class="cols">
<div>

### Multi-way (≥ 2 directions)

- `FWBAP_L2_Multiway` — generic Eigen arrays.
- `FWBAP_L2_Multiway_Polynomial2D` — 2D polynomial-weighted norm.
- `FWBAP_L2_Multiway_PolynomialOrth` — orthogonal variant.
- `FMEMM_Multiway_Polynomial2D` — Modified Extremum-Monotone Mixer.

**Power parameter:** `p = 4`; `verySmallReal_pDiP = std::pow(verySmallReal, 1.0/p)` stabilises near zero.

</div>
<div>

### Biway (pair)

- `FWBAP_L2_Biway`
- `FWBAP_L2_Cut_Biway` — sign-cutoff
- `FMINMOD_Biway`
- `FVanLeer_Biway`
- `FWBAP_L2_Biway_PolynomialNorm<dim, nVarsFixed>`
- `FMEMM_Biway_PolynomialNorm<dim, nVarsFixed>`
- `FWBAP_L2_Biway_PolynomialOrth`

**Configuration**

```jsonc
"limiterProcedure":  0   // WBAP (V2)
"limiterProcedure":  1   // CWBAP (V3)  ← recommended
"usePPRecLimiter":   true
```

</div>
</div>

> **Positivity preservation** — `LimiterUGrad` (Euler side) clamps gradients; `EvaluateURecBeta` enforces cell-mean positivity on reconstructed values; `EvaluateCellRHSAlpha` enforces CFL-consistent per-cell RHS scaling.

---
<!-- _footer: "src/CFV/VariationalReconstruction.hpp:1071-1086" -->
<!-- _class: dense -->

## VR's own limiter — WBAP with characteristic transform

```cpp
template <int nVarsFixed>
void DoLimiterWBAP_C(tUDof<nVarsFixed>  &u,
                     tURec<nVarsFixed>  &uRec,
                     tURec<nVarsFixed>  &uRecNew,
                     tURec<nVarsFixed>  &uRecBuf,
                     tSmoothIndicator   &si,
                     bool                ifAll,
                     tFM   FM,                  // cons → char transform
                     tFMI  FMI,                 // char → cons transform
                     bool  putIntoNew = false);

template <int nVarsFixed>
void DoLimiterWBAP_3(...);                      // 3-mode variant
```

<div class="cols">
<div>

### Flow

1. Compute per-face smoothness indicator `si`.
2. Transform reconstruction coefficients to characteristic variables (`FM`).
3. Apply WBAP limiter per characteristic, across the multi-way neighborhood.
4. Transform back (`FMI`).
5. Optionally write into `uRecNew` (double-buffer for iterative schemes).

</div>
<div>

### Smoothness indicators

- `DoCalculateSmoothIndicator<nVarsFixed, nVarsSee=2>(si, uRec, u, varsSee)` — classical indicator over a subset of variables.
- `DoCalculateSmoothIndicatorV1<nVarsFixed>(si, uRec, u, varsSee, FPost)` — V1 with user-provided post-processing.

</div>
</div>

---
<!-- _footer: "src/Solver/ODE.hpp · RELEASE_NOTES.md:11,14" -->
<!-- _class: dense -->

## Time integration — the ODE zoo

All integrators descend from:

```cpp
template <class TDATA, class TDTAU>
class ImplicitDualTimeStep {
    using Frhs     = std::function<void(TDATA&, TDATA&, TDTAU&, int, real, int)>;
    using Fdt      = std::function<void(TDATA&, TDTAU&, real, int)>;
    using Fsolve   = std::function<void(TDATA&, TDATA&, TDATA&, TDTAU&, real, real, TDATA&, int, real, int)>;
    using Fstop    = std::function<bool(int, TDATA&, int)>;
    using Fincrement = std::function<void(TDATA&, TDATA&, real, int)>;
    virtual void Step(TDATA &x, TDATA &xinc, const Frhs&, const Fdt&, const Fsolve&,
                      int maxIter, const Fstop&, const Fincrement&, real dt) = 0;
};
```

| `odeCode` | Class                                             | Scheme                |
|-----------|---------------------------------------------------|-----------------------|
| `103`     | `ImplicitEulerDualTimeStep`                       | Backward Euler        |
| `0`       | `ImplicitBDFDualTimeStep`                         | BDF2 / BDF-k          |
| —         | `ImplicitVBDFDualTimeStep`                        | Variable-step BDF-k   |
| `1`       | `ImplicitSDIRK4DualTimeStep` (`schemeCode` 0…4)   | SDIRK-4 · ESDIRK2/3 · Trapezoidal |
| `101`     | (alias for `1`)                                    | (backward-compat `odeCode`)      |
| **`401`** | `ImplicitHermite3SimpleJacobianDualStep`          | **HM3 + p-Multigrid** |
| `2`       | `ExplicitSSPRK3TimeStepAsImplicitDualTimeStep`    | SSP-RK3               |

`SetExtraParams(json)` exposes scheme-specific knobs (e.g. `nMG`, `incFScale`).

---
<!-- _footer: "src/Solver/ODE.hpp:123-363,917-1438" -->
<!-- _class: denser -->
## HM3 + p-Multigrid

**HM3** (Hermite-3) is a 3rd-order A-stable implicit scheme with three modes:

- **U2R2** — 2 solution states + 2 residual states.
- **U2R1** — 2 solution states + 1 residual state.
- **U3R1** — 3 solution states + 1 residual state.

<div class="cols">
<div>

### p-MG inside the time step

Inside `ImplicitHermite3SimpleJacobianDualStep::Step()` a **nonzero `nMG`** triggers p-multigrid smoothing cycles:

```cpp
// pseudocode inside the inner solve (lines 1250-1251)
fdt (xMG, dTau, 1.0, /*upos=*/2);      // lower-order pseudo-timestep
frhs(rhsbuf[1], xMG, dTau, iter, 1.0, /*upos=*/2);
```

The `upos=2` argument tells the evaluator to evaluate at a **lower polynomial order** (level-transition). VR provides `DownCastURecOrder(curOrder, iCell, uRec, downCastMethod)` to project reconstruction coefficients between orders.

</div>
<div>

### Companions

- **`tpMG`** — toggle for multigrid in the outer dual-time loop.
- **`incFScale`** — incremental flux scaling on lower MG levels; integrated into the entropy fix path (`RELEASE_NOTES.md`).
- **Positivity-preserving limiters in `LimiterUGrad`** — prevent the lower-order coarse-grid correction from producing negative density / pressure.

### Other `SDIRK4` codes

- `schemeCode = 0` — Nørsett 3-stage SDIRK-4
- `schemeCode = 1` — 6-stage ARK-family SDIRK
- `schemeCode = 2` — Kennedy–Carpenter ESDIRK3
- `schemeCode = 3` — Trapezoidal
- `schemeCode = 4` — ESDIRK2, `γ = 1 − √2/2`

</div>
</div>

---
<!-- _footer: "src/Solver/Linear.hpp · src/Euler/EulerEvaluator.hpp:427-580" -->
<!-- _class: dense -->

## Linear solvers — Krylov + LU-SGS preconditioner

<div class="cols">
<div>

### Krylov methods

```cpp
template <class TDATA>
class GMRES_LeftPreconditioned {
public:
    GMRES_LeftPreconditioned(index dofSize);
    void setSpace(int kSpace);
    bool solve(const TDATA &rhs, TDATA &x,
               FMatVec Ax, FPCApply PC,
               int maxIter, real tol);
};

template <class TDATA, class TScalar>
class PCG_PreconditionedRes { … };
```

Matrix-free: the caller supplies `Ax` and `PC` functors.

</div>
<div>

### Matrix-free LU-SGS preconditioner

Provided by `EulerEvaluator`:

```cpp
void LUSGSMatrixInit(JDiag, JSource, dTau, dt, alphaDiag, u, uRec, jacCode, t);
void LUSGSMatrixVec(alphaDiag, t, u, uInc, JDiag, AuInc);
void LUSGSMatrixToJacobianLU(alphaDiag, t, u, JDiag, jacLU);
void UpdateSGS(alphaDiag, t, rhs, u, uInc, uIncNew, JDiag,
               forward, gsUpdate, sumInc, uIncIsZero = false);
void LUSGSMatrixSolveJacobianLU(alphaDiag, t, rhs, u, uInc, uIncNew,
                                bBuf, JDiag, jacLU,
                                uIncIsZero, sumInc);
void UpdateSGSWithRec(alphaDiag, t, rhs, u, uRec, uInc, uRecInc,
                      JDiag, forward, sumInc);
```

</div>
</div>

**Selector**

```jsonc
"gmresCode": 0  // LUSGS only      (cheap, robust)
"gmresCode": 1  // GMRES            (matrix-free Krylov)
"gmresCode": 2  // LUSGS + GMRES    (LUSGS as PC for GMRES)
```

Direct path for small blocks: `src/Solver/Direct.hpp` (LU / LDLT). Optional **SuperLU_dist** via the `cfd_externals` submodule.
