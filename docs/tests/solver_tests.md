# Solver Module Unit Tests {#solver_unit_tests}

@tableofcontents

Tests for the ODE integrators, iterative solvers, and direct factorization
routines in `src/Solver/`.  All C++ tests use
[doctest](https://github.com/doctest/doctest) and run serially (no MPI).

## Building and Running

```sh
# Build all Solver C++ test executables
cmake --build build -t solver_unit_tests -j8

# Run every Solver CTest
ctest --test-dir build -R solver_ --output-on-failure

# Run a single suite
ctest --test-dir build -R solver_ode --output-on-failure
```

## Target Summary

| CMake target | CTest name | Source file | Timeout |
|---|---|---|---|
| `solver_test_ode` | `solver_ode` | test_ODE.cpp | 60 s |
| `solver_test_linear` | `solver_linear` | test_Linear.cpp | 60 s |
| `solver_test_direct` | `solver_direct` | test_Direct.cpp | 60 s |
| `solver_test_scalar` | `solver_scalar` | test_Scalar.cpp | 60 s |

---

## ODE Time Integrators (test_ODE.cpp) {#solver_test_ode}
@see test_ODE.cpp

13 test cases, 25 assertions.  Tests use the harmonic oscillator

    du/dt = A * u,  A = [[0, -w], [w, 0]],  u(0) = [1, 0]

with exact solution u(t) = [cos(wt), sin(wt)].  This oscillatory
(non-dissipative) system reveals phase and amplitude errors that
exponential-decay tests hide.

### Convergence Order Methodology

Each integrator is run at two step counts (N, 2N) over one full period
T = 2*pi/w.  The order is computed as log2(err_N / err_2N).  A tolerance
of 0.15 is used (expected_order - 0.15 < measured < expected_order + 0.85).

For implicit methods, Newton iterations use threshold-based stopping
(resNorm < 1e-13 or iter >= 50).  ESDIRK methods require at least 2
Newton iterations so that `rhsbuf[iB]` holds f(x_converged) for
subsequent stages.  Fresh ODE instances are constructed per convergence
run to avoid FSAL state carryover.

### Explicit Methods

| Test case | Integrator | Expected order |
|---|---|---|
| `SSPRK3: 3rd-order on oscillator` | ExplicitSSPRK3 | 3 |

### Implicit Single-Step Methods

| Test case | Integrator | Expected order |
|---|---|---|
| `ImplicitEuler: 1st-order on oscillator` | ImplicitEuler | 1 |
| `SDIRK4 (sc=0): 4th-order on oscillator` | SDIRK4, subCode=0 | 4 |
| `ESDIRK4 (sc=1): 4th-order on oscillator` | ESDIRK, subCode=1 | 4 |
| `ESDIRK3 (sc=2): 3rd-order on oscillator` | ESDIRK, subCode=2 | 3 |
| `Trapezoid (sc=3): 2nd-order on oscillator` | ESDIRK, subCode=3 (trapezoidal rule) | 2 |
| `ESDIRK2 (sc=4): 2nd-order on oscillator` | ESDIRK, subCode=4 | 2 |

**Note on ESDIRK4 step count:** N is kept at 10-20 for clean 4th-order
measurement.  At large N the error approaches machine precision and
roundoff dominates, showing only 2nd order.

### Multistep Methods (VBDF)

| Test case | Integrator | Expected order |
|---|---|---|
| `VBDF k=1: 1st-order on oscillator` | VBDF, maxOrder=1 | 1 |
| `VBDF k=2: 2nd-order on oscillator` | VBDF, maxOrder=2 | 2 |

### DITR (Implicit Hermite3 Dual-Step)

The DITR family uses `ImplicitHermite3SimpleJacobianDualStep` with
mask and alpha parameters:

| Test case | Mask | Alpha | Expected order | Notes |
|---|---|---|---|---|
| `DITR U2R2 (mask=0, alpha=0.5)` | 0 | 0.5 | 4 | 4th order at alpha=0.5 exactly |
| `DITR U2R2 (mask=0, alpha=0.55)` | 0 | 0.55 | 3 | 3rd order for alpha != 0.5 |
| `DITR U2R1 (mask=1)` | 1 | 0.5 | 3 | L-stable, 3rd order |

Theory reference: `HM3Draft/HM3Draft_Content.tex` (U2R2, U2R1, U3R1
methods in the Hermite-Multistep family).

### Golden Value

| Test case | Description |
|---|---|
| `SSPRK3: golden value on oscillator` | Error at N=100 against captured golden value. |

---

## Iterative Linear Solvers (test_Linear.cpp) {#solver_test_linear}
@see test_Linear.cpp

4 test cases, 8 assertions.  Tests use a thin `DVec` wrapper around
`Eigen::VectorXd` satisfying the TDATA interface.

### GMRES

| Test case | Description |
|---|---|
| `GMRES: solve 3x3 SPD system exactly` | A = [[4,1,0],[1,3,1],[0,1,2]], b = [1,2,3]. Residual < 1e-12 after at most 3 iterations. |
| `GMRES: solve 10x10 random SPD system` | Random A'A + 10I system. Residual < 1e-10 within 10 iterations. |
| `GMRES: diagonal preconditioner improves convergence` | Preconditioned GMRES converges in fewer iterations than unpreconditioned. |

### PCG (Preconditioned Conjugate Gradient)

| Test case | Description |
|---|---|
| `PCG: solve 5x5 SPD system` | Tridiagonal (2,-1) matrix. Exact solution recovery to 1e-12. |

---

## Block-Sparse Direct Solvers (test_Direct.cpp) {#solver_test_direct}
@see test_Direct.cpp

7 test cases, 26 assertions.  Uses a 2D periodic 4x4 Laplacian with 2x2
blocks (16 cells, 5-point stencil with wrap-around).

### Problem Setup

Each cell (i,j) on a 4x4 periodic grid connects to its 4 neighbors
(i+/-1, j+/-1) with periodic wrap-around.  The diagonal block is
4*I + delta*diag(i, 2i) (asymmetric to exercise full LU, not LDLT).
Off-diagonal blocks are -I.  This gives a 32x32 system (16 cells x 2x2
blocks) with a diverse sparse pattern including fill-in during
factorization.

### Tests

| Test case | Description |
|---|---|
| `symbolic factorization has fill-in` | `SerialSymLUStructure` with iluCode=-1 produces more non-zeros than the original adjacency. |
| `MatMul is correct` | A*x == b verified against dense reference for random x. |
| `complete LU solve is exact` | Full LU decompose + solve recovers x to 1e-12. |
| `ILU(0) is approximate but reduces residual` | ILU(0) single-step solve has residual > 1e-10 (not exact) but < initial residual. |
| `ILU(1) converges faster than ILU(0) as preconditioner` | Fixed-point iteration x += M^{-1}(b-Ax): ILU(1) reaches tolerance in fewer iterations. Only original-adjacency entries are filled with -I; fill-in positions stay zero. |
| `LDLT: MatMul matches LU MatMul for symmetric system` | Symmetric variant produces identical A*x. |
| `LDLT: complete decompose + solve is exact` | LDLT on symmetric system recovers x to 1e-12. |

### ILU Preconditioned Iteration Detail

The ILU(0) vs ILU(1) comparison uses 200 iterations of stationary
iteration x_{k+1} = x_k + M^{-1}(b - A*x_k).  The test verifies that
ILU(1) reaches a residual threshold (1e-8) in strictly fewer iterations
than ILU(0).  Fill-in positions from higher ILU levels are identified via
`cell2cellFaceVLocal2FullRowPos` and left at zero so only original
adjacency entries carry -I blocks.

---

## Scalar Utilities (test_Scalar.cpp) {#solver_test_scalar}
@see test_Scalar.cpp

5 test cases, 6 assertions.  Tests `BisectSolveLower` from
`Solver/Scalar.hpp`, a bisection root-finder for monotonic functions.

| Test case | Description |
|---|---|
| `x^2 = 4 => x = 2` | Quadratic, tolerance 1e-10. |
| `sin(x) = 0.5 => x ~ 0.5236` | Trigonometric, tolerance 1e-10. |
| `linear f(x)=x, target=0.75` | Trivial case. |
| `exp(x) = e => x = 1` | Exponential. |
| `few iterations gives lower accuracy` | 5 iterations: error < 0.1 but > 1e-6 (verifying iteration count affects precision). |
