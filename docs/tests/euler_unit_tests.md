# Euler Module Unit Tests {#euler_unit_tests}

@tableofcontents

Tests for the compressible Navier-Stokes solver module (`src/Euler/`).
All C++ tests use [doctest](https://github.com/doctest/doctest).

## Building and Running

```sh
# Build all Euler C++ test executables
cmake --build build -t euler_unit_tests -j8

# Run every Euler CTest (serial + MPI np=1,2,4,8)
ctest --test-dir build -R euler_ --output-on-failure

# Run a single test
ctest --test-dir build -R euler_gas_thermo --output-on-failure
```

## Target Summary

| CMake target | CTest names | Source file | Type |
|---|---|---|---|
| `euler_test_gas_thermo` | `euler_gas_thermo` | test_GasThermo.cpp | Serial |
| `euler_test_riemann_solvers` | `euler_riemann_solvers` | test_RiemannSolvers.cpp | Serial |
| `euler_test_rans` | `euler_rans` | test_RANS.cpp | Serial |
| `euler_test_evaluator_pipeline` | `euler_evaluator_pipeline_np{1,2,4}` | test_EulerEvaluator.cpp | MPI (600 s) |

---

## Gas Thermodynamics and Eigenvectors (test_GasThermo.cpp) {#euler_test_gas_thermo}
@see test_GasThermo.cpp

Serial tests for ideal-gas thermodynamics and Euler eigenvector routines
in `Gas.hpp`.  22 test cases.  No MPI or mesh; all
functions are pure.

### IdealGasThermal

| Test case | Description |
|---|---|
| `standard quiescent air` | rho=1, p=1/gamma: checks a=1, H=gamma/(gamma-1), internal energy. |
| `Mach 2 flow` | Supersonic state: verifies p, a, H, total energy. |

### Conservative / Primitive Round-Trip

| Test case | Description |
|---|---|
| `Cons2Prim and Prim2Cons round-trip: 3D` | 5-component state round-trips to 1e-14. |
| `Cons2Prim and Prim2Cons round-trip: 2D` | 4-component state round-trips to 1e-14. |
| `Prim2Cons: known state verification` | Manually computed state matches exactly. |

### Stagnation Quantities

| Test case | Description |
|---|---|
| `PrimitiveGetP0T0: quiescent gas` | At rest: p0 = p, T0 = T. |
| `PrimitiveGetP0T0: p0 > p for moving gas` | Stagnation pressure exceeds static pressure. |

### Eigenvectors

Verifies L * R = I (orthogonality) for the Euler flux Jacobian
eigenvectors.

| Test case | Description |
|---|---|
| `EulerGas eigenvectors: L*R = I for 3D` | Quiescent state, x-normal. |
| `EulerGas eigenvectors: L*R = I for non-trivial velocity` | Moving flow, oblique normal. |
| `IdealGas convenience wrappers produce L*R=I` | Same check via `IdealGas_EulerGasRight/LeftEigenVector`. |

### Inviscid Flux

| Test case | Description |
|---|---|
| `GasInviscidFlux: x-direction, quiescent gas` | F = [0, p, 0, 0, 0] at rest. |
| `GasInviscidFlux: x-direction, moving gas` | Full flux against hand-computed values. |
| `GasInviscidFlux_XY: n=(1,0,0) equals GasInviscidFlux` | Rotated-normal variant matches direct formula. |

### Conservative Increments

| Test case | Description |
|---|---|
| `IdealGasUIncrement: zero increment gives zero` | delta_U = 0 produces delta_{u,p} = 0. |
| `IdealGasUIncrement: finite-difference verification` | Increment matches prim(U+dU) - prim(U) to O(dU^2). |

### Roe Average

| Test case | Description |
|---|---|
| `GetRoeAverage: identical states give same state` | Roe(U,U) = U. |
| `GetRoeAverage: density is geometric mean` | rho_Roe = sqrt(rhoL * rhoR). |

### Gradient Transformation

| Test case | Description |
|---|---|
| `GradientCons2Prim: zero gradient produces zero` | Pure sanity check. |
| `GradientCons2Prim: finite-difference verification` | Transformed gradient matches numerical differentiation. |

### Compression Ratio and Viscous Flux

| Test case | Description |
|---|---|
| `CompressionRatio: zero increment gives alpha=0` | No compression needed for zero perturbation. |
| `CompressionRatio: alpha in [0,1]` | Output is bounded for random perturbations. |
| `ViscousFlux: zero gradient produces zero flux` | Sanity check for viscous flux routine. |

---

## Riemann Solvers (test_RiemannSolvers.cpp) {#euler_test_riemann}
@see test_RiemannSolvers.cpp

Serial tests for Roe, HLLC, and HLLEP Riemann solvers in `Gas.hpp`.
11 test cases.

### Consistency (F(U,U) = exact flux)

| Test case | Description |
|---|---|
| `Roe consistency: identical states give exact flux` | F_Roe(U,U,n) == F_exact(U,n). |
| `HLLC consistency` | Same check for HLLC. |
| `HLLEP consistency` | Same check for HLLEP. |
| `Roe consistency: diagonal normal` | Oblique normal n = (1,1,1)/sqrt(3). |

### Variant Consistency

| Test case | Description |
|---|---|
| `Roe variants M1-M8 consistency` | eigScheme 1,3,4,5,6,7,8 pass consistency; 2 and 9 are unimplemented. |

### Symmetry (F(UL,UR,n) = -F(UR,UL,-n))

| Test case | Description |
|---|---|
| `Roe symmetry` | Verified for 3 state pairs. |
| `HLLC symmetry` | Same check with 3 pairs. |

### Sod Shock Tube

| Test case | Description |
|---|---|
| `Sod shock tube: flux is finite and bounded` | UL=(1,0,0,0,2.5), UR=(0.125,0,0,0,0.25): flux has finite components. |

### Golden Values

| Test case | Description |
|---|---|
| `Golden flux values for mixed-state test vector` | Roe, HLLC, HLLEP flux components against captured golden values (tolerance 1e-8). |

### Additional Checks

| Test case | Description |
|---|---|
| `All solvers: quiescent gas produces same flux` | Roe, HLLC, HLLEP agree on a zero-velocity state. |
| `Roe eigenvalue output: lam0 < lam123 < lam4 for subsonic` | Wave speed ordering: u-a < u < u+a. |

---

## RANS Turbulence Models (test_RANS.cpp) {#euler_test_rans}
@see test_RANS.cpp

Serial tests for k-omega Wilcox 2006, k-omega SST, and Realizable k-epsilon
model functions in `RANS_ke.hpp`.  26 test cases.

The SA model is excluded because `GetSource_SA` references
`EulerEvaluator::settings` (evaluator context) and cannot be tested
standalone.  SA coverage is provided through the EulerEvaluator pipeline
test on the NACA0012 case.

### Turbulent Viscosity (GetMut)

| Test case | Description |
|---|---|
| `GetMut_KOWilcox: non-negative` | mut >= 0. |
| `GetMut_KOWilcox: bounded by 1e5 * muLam` | CFL3D limiting. |
| `GetMut_KOWilcox: golden value` | Regression against captured value. |
| `GetMut_SST: non-negative` | mut >= 0. |
| `GetMut_SST: bounded by 1e5 * muLam` | CFL3D limiting. |
| `GetMut_SST: golden value` | Regression against captured value. |
| `GetMut_RealizableKe: non-negative` | mut >= 0. |
| `GetMut_RealizableKe: bounded by 1e5 * muLam` | CFL3D limiting. |
| `GetMut_RealizableKe: golden value` | Regression against captured value. |

### Mut Sensitivity

| Test case | Description |
|---|---|
| `GetMut_KOWilcox: mut increases with k` | Higher k produces higher mut. |
| `GetMut_SST: mut increases with k` | Same check for SST. |
| `GetMut_KOWilcox: very small k/omega produces finite mut` | Robustness near zero. |
| `GetMut_SST: very small k/omega produces finite mut` | Robustness near zero. |
| `GetMut_RealizableKe: very small k/eps produces finite mut` | Robustness near zero. |

### Source Terms (GetSource, mode=0: full)

| Test case | Description |
|---|---|
| `GetSource_KOWilcox: zero gradient finite and has destruction` | Zero velocity gradient: production=0, destruction from omega. |
| `GetSource_SST: zero gradient finite` | Same check for SST. |
| `GetSource_RealizableKe: zero gradient finite` | Same check for Realizable k-epsilon. |
| `GetSource_KOWilcox: shear gradient golden` | Non-zero dU/dx: golden source vector regression. |
| `GetSource_SST: shear gradient golden` | Same check for SST. |
| `GetSource_RealizableKe: shear gradient golden` | Same check for Realizable k-epsilon. |

### Source Terms (mode=1: implicit Jacobian diagonal)

| Test case | Description |
|---|---|
| `GetSource_KOWilcox mode=1: implicit diagonal non-negative` | Diagonal terms are >= 0 (stabilizing). |
| `GetSource_SST mode=1: implicit diagonal non-negative` | Same check for SST. |

### Viscous Flux (GetVisFlux)

| Test case | Description |
|---|---|
| `GetVisFlux_KOWilcox: zero gradient -> zero flux` | No gradient, no transport. |
| `GetVisFlux_SST: zero gradient -> zero flux` | Same for SST. |
| `GetVisFlux_RealizableKe: zero gradient -> zero flux` | Same for Realizable k-epsilon. |
| `GetVisFlux_KOWilcox: k-gradient produces k-flux` | Non-zero dk/dx generates flux in the k-equation. |

---

## EulerEvaluator Pipeline (test_EulerEvaluator.cpp) {#euler_test_evaluator}
@see test_EulerEvaluator.cpp

MPI integration test (np=1,2,4, 600 s timeout) exercising the full
evaluator pipeline: config → mesh → initialize DOF → EvaluateDt →
EvaluateRHS → Jacobi solve (Forward + Backward).

For each case the test measures:
- **RHS L1 norm** (golden regression)
- **Jacobi increment L1 norm** (golden regression)

Jacobi update (not LU-SGS) is used for MPI-deterministic increment norms.

### Test Cases

| Test case | Config | Physics | Mesh | Notes |
|---|---|---|---|---|
| `IV (NS, P1, 2D)` | euler_config_IV.json | Inviscid, isentropic vortex | IV10_10 (100 quads, periodic) | 2D Euler |
| `NACA0012 (NS_SA, P1)` | eulerSA_config.json | Viscous + SA turbulence | NACA0012_H2 (external, wall) | Tests SA model in-situ |
| `Box (NS_3D, P1)` | euler3D_config_Box.json | Inviscid, 3D periodic | Uniform32_3D_Periodic (32768 hexes) | 3D Euler |

### Pipeline Steps

For each case:

1. Write default config with `ConfigureFromJson(path, false)`, then
   merge-patch with the case JSON and test overrides (Jacobi update, P1,
   absolute mesh paths).
2. `ReadMeshAndInitialize` — reads mesh, builds VR.
3. `eval.InitializeUDOF(u)` — sets initial condition.
4. `EvaluateDt` — computes local time steps.
5. `EvaluateRHS` — computes the spatial residual.
6. `LUSGSMatrixInit + Forward + Backward` — one Jacobi-style iteration
   (despite the LU-SGS name, the code path uses Jacobi when configured).
7. Check RHS and increment norms against golden values (tolerance 1e-6).
