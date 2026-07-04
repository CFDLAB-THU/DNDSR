# Roe-Averaged Eigensystems for Reactive Ideal-Gas Flows

## State vector convention

The conservative state vector stores **total** `ρE`. Reactive runs also define a
bookkeeping-only base internal-energy offset used by the sensible-energy limiter:

```
U = [ρ, ρu, ρv, (ρw), ρE_total]
ρE_total = ρ·(e_sensible + ½|v|² + e_base)
```

where `e_base = Σ Y_k · e_base,k` and `e_base,k` is the per-species internal
energy at the configured base temperature `TBase`.

## Sensible vs. total enthalpy in the Roe Jacobian

The Roe Jacobian `A = ∂F/∂U` of the Euler equations has the same form
regardless of whether `E` includes `e_base` — the base-energy offset is a
linear composition-dependent bookkeeping term. The
eigenvalues `{u−a, u, u+a}` depend only on the frozen-composition acoustic speed.
For reactive ideal-gas mixtures DNDSR computes it as `a² = γ_cp/cv·p/ρ`, while
pressure still comes from the closure `p = (γ_eq−1)·ρe_sensible`.

The eigenvectors, however, include the specific enthalpy `H` in their
energy component:

```
r₀ (λ = u−a):  [1,  u−a,  H − u·a]ᵀ
r₁ (λ = u):    [1,  u,    ½|u|²]ᵀ
r₄ (λ = u+a):  [1,  u+a,  H + u·a]ᵀ
```

DNDSR uses the **sensible** specific enthalpy `H_sensible` throughout
the Roe eigensystem. This choice is both correct (pressure propagation
is purely thermo-mechanical) and convenient (`H_sensible` is computed
directly by `IdealGasThermal` with the base internal energy already
subtracted).

## Base internal energy in the α decomposition

The base internal-energy offset in `ρE_total` is handled entirely by the
α decomposition — the contact/entropy wave `α₁` adjusts to account for
the difference between the total energy jump and the sensible
eigenvectors. The decomposition is exact regardless of which H
convention is used, **as long as the same H appears in both the
decomposition and the recombination**.

## Consistency conditions

Let `Γ = γ_eq−1` be the pressure-energy derivative and let `γ = cp/cv` be the
thermodynamic acoustic coefficient. For a Roe-averaged state,

```
p/ρ = Γ/γ_eq · (H_sensible − ½|u|²)
a²  = γ · Γ/γ_eq · (H_sensible − ½|u|²)
```

This reduces to the standard perfect-gas formula `a² = (γ−1)(H−½|u|²)` when
`γ_eq == γ` and both are constant. All quantities below are Roe-averaged unless
subscripted L or R.

### α decomposition

```
α1 = Γ/a² · [Δρ·(H − u_n²)  +  u_n·Δ(ρu_n)  −  incU₄^b]

α0 = [Δρ·(u_n + a)  −  Δ(ρu_n)  −  a·α1] / (2a)

α4 = Δρ − α0 − α1
```

where `incU₄^b = Δ(ρE_total) − α₂₃^VT·u` removes the tangential
kinetic-energy coupling from the energy jump.

### α derivation from the energy equation

The energy decomposition identity is:

```
Δ(ρE) = (H − u_n·a)·α0  +  ½|u|²·α1  +  (H + u_n·a)·α4  +  α₂₃^VT·u
```

Expanding with `α0 + α4 = Δρ − α1` and substituting the momentum
identity `a·(α4 − α0) = Δ(ρu_n) − u_n·Δρ`:

```
Δ(ρE) = H·(Δρ − α1) + u_n·a·(α4 − α0) + ½|u|²·α1 + α₂₃^VT·u
      = H·Δρ + α1(½|u|² − H) + u_n·[Δ(ρu_n) − u_n·Δρ] + α₂₃^VT·u
      = Δρ·(H − u_n²) + u_n·Δ(ρu_n) + α1(½|u|² − H) + α₂₃^VT·u
```

Solving for α1 using `a² = γ·Γ/γ_eq·(H − ½|u|²)` and the pressure-energy
closure coefficient `Γ = γ_eq−1`:

```
α1 = Γ/a² · [Δρ·(H − u_n²) + u_n·Δ(ρu_n) − (Δ(ρE) − α₂₃^VT·u)]
   = Γ/a² · [Δρ·(H − u_n²) + u_n·Δ(ρu_n) − incU₄^b]
```

This matches the code exactly and holds for any H convention.

### Recombination (dissipation flux)

```
incF(0)      = α0 + α1 + α4                          = Δρ
incF(1:dim)  = (u − a·n)·α0 + u·α1 + (u + a·n)·α4 + α₂₃^VT
             = u·Δρ + a·n·(α4 − α0) + α₂₃^VT        = Δ(ρu)
incF(I4)     = (H − u_n·a)·α0 + ½|u|²·α1 + (H + u_n·a)·α4 + α₂₃^VT·u
             = Δ(ρE_total)
```

The final Roe flux is `F = ½(F_L + F_R) − ½·Σ |λ_k|·α_k·r_k`.

## Sound-speed formula

```
a² = γ_cp/cv · p/ρ
p/ρ = (γ_eq−1)/γ_eq · (H_sensible − ½|u|²)
a² = γ_cp/cv · (γ_eq−1)/γ_eq · (H_sensible − ½|u|²)
a  = √(a²)
```

For a calorically perfect gas, `γ_eq = γ_cp/cv = γ`, so this becomes the usual
Roe expression:

```
H_sensible = e_sensible + ½|u|² + p/ρ
H_sensible − ½|u|² = e_sensible + p/ρ = γ·e_sensible
a² = (γ−1)·γ·e_sensible = γp/ρ        ✓
```

For a perfect gas, using `H_total = H_sensible + e_base` would add the spurious
term `(γ−1)e_base`; this is incorrect.

## Code trace

| Step | File | Function | What happens |
|------|------|----------|--------------|
| 1 | `Gas.hpp` | `ComputeRoePreamble` | `IdealGasThermal(E,ρ,v²,gammaEq,gammaCpCv, p,asqr,H, rhoE_base)` → pressure from `gammaEq`, sound speed from `gammaCpCv`, H sensible |
| 2 | `Gas.hpp` | `ComputeRoePreamble` | `HRoe = (√ρL·HLm + √ρR·HRm)/(√ρL+√ρR)` → sensible |
| 3 | `Gas.hpp` | `ComputeRoePreamble` | `gammaEqRoe` and `gammaRoe` are √ρ-weighted separately |
| 4 | `Gas.hpp` | `ComputeRoePreamble` | `asqrRoe = gammaRoe·(gammaEqRoe−1)/gammaEqRoe·(HRoe − ½v²)` |
| 5 | `Gas.hpp` | `RoeFlux_*` / `RoeFluxIncFDiff` | α1 uses `(gammaEqRoe−1)/asqrRoe`; eigenvalues use `sqrt(asqrRoe)` |
| 6 | `Gas.hpp` | `RoeFlux_*` | Energy flux uses HRoe (sensible) in eigenvector components |

All consumption sites use the same sensible H convention derived at step 1.
The standalone eigenvector extractors `IdealGas_EulerGasRightEigenVector` /
`IdealGas_EulerGasLeftEigenVector` take both `gammaEq` and `gammaCpCv`; callers
must pass `rhoE_base` for states that include base internal energy.

## EulerP note

The EulerP module (`EulerP_ARS.hpp`) computes `HLm/HRm` via `IdealGas::Enthalpy(U(I4),U(0),p)`
which returns **total** H (no formation subtraction). For non-reactive
flows this is identical to sensible H. Reactive EulerP would need a
convention alignment.
