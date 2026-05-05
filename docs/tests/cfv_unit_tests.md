# CFV Module Unit Tests {#cfv_unit_tests}

@tableofcontents

Tests for the Compact Finite Volume module (`src/CFV/`).  C++ tests use
[doctest](https://github.com/doctest/doctest); Python tests use pytest.

## Building and Running

```sh
# Build all CFV C++ test executables
cmake --build build -t cfv_unit_tests -j8

# Run every CFV CTest (serial + MPI np=1,2,4,8)
ctest --test-dir build -R cfv_ --output-on-failure

# Python tests (requires pybind11 .so to be installed first)
cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 -j32
cmake --install build --component py
PYTHONPATH=python pytest test/CFV/ -v
```

## Target Summary

| CMake target | CTest names | Source file | Type |
|---|---|---|---|---|
| `cfv_test_limiters` | `cfv_limiters` | test_Limiters.cpp | Serial |
| `cfv_test_reconstruction` | `cfv_reconstruction_np{1,2,4,8}` | test_Reconstruction.cpp | MPI |
| `cfv_test_reconstruction3d` | `cfv_reconstruction3d_np{1,2,4,8}` | test_Reconstruction3D.cpp | MPI |
| `cfv_test_device_transferable` | `cfv_device_transferable_np1` | test_DeviceTransferable.cu | MPI (CUDA) |
| — | — | test_fv_correctness.py | pytest |
| — | — | test_vr_correctness.py | pytest |
| — | — | test_basic_fv.py | pytest |
| — | — | test_basic_cfv.py | pytest |
| — | — | test_cfv_dissdisp.py | pytest |

---

## Limiter Functions (test_Limiters.cpp) {#cfv_test_limiters}
@see test_Limiters.cpp

Serial tests for every standalone limiter function in `CFV/Limiters.hpp`.
36 test cases.  No mesh or MPI required — pure Eigen array
computations.

### PolynomialSquaredNorm

Weighted squared norms used internally by polynomial-aware limiters.

| Test case | Description |
|---|---|
| `PolynomialSquaredNorm<2> with 2 rows (P1)` | Uniform weight (w=1); result[j] = sum(theta(i,j)^2). |
| `PolynomialSquaredNorm<2> with 3 rows (P2)` | Mixed weights (1, 0.5, 1); checks exact values. |
| `PolynomialSquaredNorm<2> with 4 rows (P3)` | Weights (1, 1/3, 1/3, 1); single-column check. |
| `PolynomialSquaredNorm<3> with 3 rows (P1)` | 3D uniform weight; verifies column-wise sums. |
| `PolynomialSquaredNorm<3> with 6 rows (P2)` | 3D mixed weights (1,1,1, 0.5,0.5,0.5). |
| `PolynomialSquaredNorm<3> with 10 rows (P3)` | 3D full weight vector (1,1/3,1/6). |

### PolynomialDotProduct

| Test case | Description |
|---|---|
| `PolynomialDotProduct<2> self-product equals PolynomialSquaredNorm` | dot(theta, theta) == norm(theta) for nRows in {2,3,4}. |
| `PolynomialDotProduct<2> linearity` | dot(alpha*a, b) == alpha * dot(a, b). |

### FMINMOD_Biway (classical minmod)

| Test case | Description |
|---|---|
| `same-sign inputs` | Returns min-magnitude: minmod(1,2)=1, minmod(3,6)=3, etc. |
| `opposite-sign inputs produce zero` | Opposite signs always give zero output. |
| `zero input` | Any zero input produces zero output. |

### FVanLeer_Biway (Van Leer limiter)

| Test case | Description |
|---|---|
| `same-sign inputs` | VanLeer(2,4) = 8/3; VanLeer(-3,-6) = -4. |
| `opposite-sign produces zero` | Opposite signs give zero. |
| `equal inputs return same value` | VanLeer(a,a) = a. |

### FWBAP_L2_Biway (weighted-bounded averaging, L2 norm)

| Test case | Description |
|---|---|
| `identical inputs pass through` | WBAP(u,u) = u for any u. |
| `no NaN for random inputs` | Robustness: random arrays produce finite results. |
| `output bounded by inputs for same-sign` | All-positive inputs yield non-negative output. |

### FWBAP_L2_Cut_Biway

| Test case | Description |
|---|---|
| `opposite-sign cut to zero` | Opposite-sign pairs are hard-zeroed. |
| `no NaN` | Random inputs produce finite results. |

### Multiway Limiters

| Test case | Description |
|---|---|
| `FWBAP_L2_Multiway: all identical pass through` | 4 identical inputs produce the same output. |
| `FWBAP_L2_Multiway: no NaN` | 5 random stencils produce finite results. |
| `FWBAP_L2_Multiway_Polynomial2D: all identical pass through` | Polynomial-norm variant preserves identical inputs. |
| `FWBAP_L2_Multiway_Polynomial2D: no NaN (nRows=2,3,4)` | Robust across P1/P2/P3 row counts. |
| `FMEMM_Multiway_Polynomial2D: center unchanged when smallest` | MEMM leaves the minimum-norm candidate untouched. |
| `FMEMM_Multiway_Polynomial2D: no NaN` | Random inputs produce finite results. |
| `FWBAP_L2_Multiway_PolynomialOrth: all identical pass through` | Orthogonal variant preserves identical inputs. |
| `FWBAP_L2_Multiway_PolynomialOrth: no NaN` | Robustness check. |

### Biway Polynomial-Norm Limiters

| Test case | Description |
|---|---|
| `FWBAP_L2_Biway_PolynomialNorm<2,1>: identical inputs pass through` | Fixed-1-var variant preserves identical inputs. |
| `FWBAP_L2_Biway_PolynomialNorm<2,Dynamic>: no NaN` | Dynamic-var variant is NaN-free. |
| `FWBAP_L2_Biway_PolynomialNorm<3,Dynamic>: no NaN for 3D dims` | Tests nRows in {3,6,10}. |
| `FMEMM_Biway_PolynomialNorm<2,Dynamic>: u2 smaller returns u2` | MEMM reduces toward the smaller-norm input. |
| `FMEMM_Biway_PolynomialNorm<2,Dynamic>: no NaN` | Robustness check. |
| `FWBAP_L2_Biway_PolynomialOrth: identical inputs pass through` | Orthogonal biway preserves identical inputs. |
| `FWBAP_L2_Biway_PolynomialOrth: no NaN` | Robustness check. |

### Cross-Consistency and Edge Cases

| Test case | Description |
|---|---|
| `Limiter ordering: FMINMOD <= FWBAP for same-sign` | Minmod is the most restrictive; 100-element check. |
| `All limiters handle near-zero inputs without NaN` | Inputs at 1e-300: 6 subcases (MINMOD, VanLeer, WBAP, Cut, PolyNorm, PolyOrth). |

---

## Variational Reconstruction 2D (test_Reconstruction.cpp) {#cfv_test_reconstruction}
@see test_Reconstruction.cpp

MPI tests (np=1,2,4,8) exercising the full VR pipeline for scalar fields on
2D meshes.  Uses Jacobi iteration (SOR disabled) for MPI-deterministic
golden values.

### Meshes

| Mesh | Cells | Domain | Boundary |
|---|---|---|---|
| Uniform_3x3_wall.cgns | 9 quads | [-1, 2]^2 | Wall (Dirichlet) |
| IV10_10.cgns | 100 quads | [0, 10]^2 | Periodic |
| IV10U_10.cgns | 322 tris | [0, 10]^2 | Periodic |

Periodic meshes are bisected 0/1/2 times via O2-elevation + bisection
(yielding 100/400/1600 quads and 322/~1288/~5152 tris).

### Reconstruction Methods

| Label | Description |
|---|---|
| GG | Explicit 2nd-order Gauss-Green gradient |
| P1-HQM | Iterative VFV, maxOrder=1, HQM_OPT + HQM_SD weights |
| P2-HQM | Iterative VFV, maxOrder=2, HQM weights |
| P3-HQM | Iterative VFV, maxOrder=3, HQM weights |
| P1-Def | Iterative VFV, maxOrder=1, Factorial + GWNone weights |
| P2-Def | Iterative VFV, maxOrder=2, default weights |
| P3-Def | Iterative VFV, maxOrder=3, default weights |

### Polynomial Exactness (Wall Mesh)

The following tests verify exact recovery of polynomial fields on the
9-cell wall mesh with Dirichlet boundary conditions:

| Polynomial | Methods verified exact (error < 1e-12) |
|---|---|
| Constant (f = 1) | GG, P1-HQM, P2-HQM, P3-HQM, P1-Def |
| Linear (f = x + 2y) | GG, P1-HQM, P2-HQM, P3-HQM, P1-Def |
| Quadratic (f = x^2 + y^2) | P2-HQM, P3-HQM, P2-Def |
| Cubic (f = x^3 + xy^2) | P3-HQM, P3-Def |

### Convergence Series (Periodic Meshes)

L1/volume errors at bisection levels 0, 1, 2 are compared against
golden values captured from commit c774b89 (tolerance 1e-6).

11 parameter combinations:

- IV10 quad + sin*cos: GG, P1-HQM, P2-HQM, P3-HQM, P1-Def, P3-Def
- IV10U tri + sin*cos: GG, P1-HQM, P2-HQM, P3-HQM
- IV10 quad + cos+cos: P3-HQM

### Iteration Convergence

`VFV P2 HQM converges on IV10 base mesh` — verifies the Jacobi
iteration reaches an increment below 1e-14 within 200 iterations.

### Limiter Procedure Tests

After iterative reconstruction converges, tests apply
`DoCalculateSmoothIndicator` followed by `DoLimiterWBAP_C` or
`DoLimiterWBAP_3`, then measure the post-limiter L1 error against
golden values.

5 parameter combinations:

| Mesh | Method | Limiter | Note |
|---|---|---|---|
| IV10 quad | P2-HQM | CWBAP | ifAll=true, 3 bisection levels |
| IV10 quad | P3-HQM | CWBAP | ifAll=true, 3 bisection levels |
| IV10U tri | P2-HQM | CWBAP | ifAll=true, 3 bisection levels |
| IV10 quad | P2-HQM | 3WBAP | ifAll=true, 3 bisection levels |
| IV10 quad | P3-HQM | 3WBAP | ifAll=true, 3 bisection levels |

The eigenvalue transform (FM/FMI) is set to identity since the test
field is scalar (nVars=1).

---

## Variational Reconstruction 3D (test_Reconstruction3D.cpp) {#cfv_test_reconstruction3d}
@see test_Reconstruction3D.cpp

MPI tests (np=1,2,4,8) exercising the 3D VR template instantiation.
Uses `Uniform32_3D_Periodic.cgns` (32768 hex cells on the unit cube).
7 test cases.

### Constant Exactness

All methods reproduce f=1 with error < 1e-12:

- Gauss-Green
- VFV P1-HQM
- VFV P2-HQM

### Golden-Value Regression

6 method/function combinations:

| Method | Function | Golden error |
|---|---|---|
| GG | sin*cos*cos | 1.510e-03 |
| P1-HQM | sin*cos*cos | 3.248e-03 |
| P2-HQM | sin*cos*cos | 7.482e-05 |
| GG | cos+cos+cos | 1.493e-03 |
| P1-HQM | cos+cos+cos | 2.426e-03 |
| P2-HQM | cos+cos+cos | 3.701e-05 |

### Structural Tests

| Test case | Description |
|---|---|
| `VFV P1 HQM converges on sinCos3D` | Iteration reaches inc < 1e-14 within 200 iters. |
| `VFV P2 HQM error < P1 on sinCos3D` | Higher order gives lower error. |
| `cell volumes sum to 1.0 (unit cube)` | Sum of all cell volumes equals domain volume. |

---

## FiniteVolume Python Tests (test_fv_correctness.py) {#cfv_test_fv_python}
@see test_fv_correctness.py

Pytest tests (16 tests) for the `CFV.FiniteVolume` Python bindings.
Uses the Uniform_3x3_wall mesh (9 quads, [-1,2]^2) and IV10_10
periodic mesh (100 quads, [0,10]^2).

### Test Classes

**TestCellVolumes** — Cell volumes are positive and match known values
(1.0 per quad for 3x3 wall; sum = 100 for IV10 periodic).

**TestGlobalVolume** — Global volume sums correctly: 9.0 for wall mesh,
100.0 for periodic mesh.

**TestFaceAreas** — Face areas are positive; internal face areas equal
1.0 for a uniform grid.

**TestCellBarycenters** — Barycenters lie within the domain; for the
3x3 wall mesh, barycenters are at half-integer coordinates.

**TestDOFArrays** — `BuildUDof` (dynamic and fixed-5), `BuildUGrad`
allocate arrays with correct sizes; DOF write-then-read round-trips.

**TestArrayBytes** — `DataSizeBytes()` returns a positive value for
both DOF and adjacency arrays.

**TestSmoothScale** — `GetSmoothScale` returns a sensible ratio
(between 0.1 and 10 for uniform grids).

**TestJacobiDet** — Cell and face Jacobian determinants are positive.

---

## VR and ModelEvaluator Python Tests (test_vr_correctness.py) {#cfv_test_vr_python}
@see test_vr_correctness.py

Pytest tests (16 tests) for `CFV.VariationalReconstruction` and
`CFV.ModelEvaluator` through Python bindings.  Uses the Uniform_3x3
periodic mesh (9 quads, [0,3]^2).

### Test Classes

**TestVRConstruction** — Global volume matches 9.0; barycenters lie in
[0,3]^2; cell volumes and face areas are positive.

**TestVRDOFBuilding** — `BuildURec`, `BuildUDof`, `BuildUGrad` produce
arrays with correct shapes.

**TestConstantExactness** — A constant field (u=1 everywhere) produces
zero uRec coefficients after reconstruction.

**TestLinearFieldReconstruction** — Linear field f(x)=x produces uRec
coefficients with a dominant first component (approximating df/dx).

**TestModelEvaluatorRHS** — `EvaluateRHS` produces finite results; a
non-uniform field gives nonzero RHS; a constant field gives near-zero RHS.

**TestReconstructionConvergence** — Iterative VR converges (increment
decreases monotonically).

**TestMatrixAccess** — `matrixAAInvB` and `vectorAInvB` accessors
return arrays with correct shapes.

**TestBoundaryCallback** — `getFBoundary` returns a callable.
