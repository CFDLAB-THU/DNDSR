# Roe-Averaged Eigensystems for Reactive Ideal-Gas Flows

## State vector convention

The conservative state vector stores **total** `ρE`, which includes formation enthalpy:

```
U = [ρ, ρu, ρv, (ρw), ρE_total]
ρE_total = ρ·(e_sensible + ½|v|² + h_f)
```

where `h_f = Σ Y_k · h_{f,k}` is the specific formation enthalpy, and
`h_f = 0` for H₂, O₂, N₂ (elemental reference species).

## Sensible vs. total enthalpy in the Roe Jacobian

The Roe Jacobian `A = ∂F/∂U` of the Euler equations has the same form
regardless of whether `E` includes `h_f` — the formation energy is a
constant offset per cell that does not affect the flux derivatives. The
eigenvalues `{u−a, u, u+a}` depend only on `a² = γp/ρ`, which is
determined by sensible internal energy alone.

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
directly by `IdealGasThermal` with the formation enthalpy already
subtracted).

## Formation enthalpy in the α decomposition

The formation enthalpy offset in `ρE_total` is handled entirely by the
α decomposition — the contact/entropy wave `α₁` adjusts to account for
the difference between the total energy jump and the sensible
eigenvectors. The decomposition is exact regardless of which H
convention is used, **as long as the same H appears in both the
decomposition and the recombination**.

## Consistency conditions

Let `a² = (γ−1)(H_sensible − ½|u|²)` be the Roe-averaged speed of sound
squared. All quantities below are Roe-averaged unless subscripted L or R.

### α decomposition

```
α1 = (γ−1)/a² · [Δρ·(H − u_n²)  +  u_n·Δ(ρu_n)  −  incU₄^b]

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

Solving for α1 using `a² = (γ−1)(H − ½|u|²)`:

```
α1 = (γ−1)/a² · [Δρ·(H − u_n²) + u_n·Δ(ρu_n) − (Δ(ρE) − α₂₃^VT·u)]
   = (γ−1)/a² · [Δρ·(H − u_n²) + u_n·Δ(ρu_n) − incU₄^b]
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
a² = (γ−1)(H_sensible − ½|u|²)     // valid only for ideal/perfect gas
a  = √(a²)
```

This is correct for `H_sensible` because:

```
H_sensible = e_sensible + ½|u|² + p/ρ
H_sensible − ½|u|² = e_sensible + p/ρ = γ·e_sensible
a² = (γ−1)·γ·e_sensible = γp/ρ        ✓
```

Using `H_total = H_sensible + h_f` would give `a² = γp/ρ + (γ−1)h_f`,
which is incorrect.

## Code trace

| Step | File | Function | What happens |
|------|------|----------|--------------|
| 1 | `Gas.hpp:241` | `ComputeRoePreamble` | `IdealGasThermal(E,ρ,½v²,γ, p,asqr,H, rhoH_form)` → H = (E−ρh_f+p)/ρ = sensible |
| 2 | `Gas.hpp:249` | `ComputeRoePreamble` | `HRoe = (√ρL·HLm + √ρR·HRm)/(√ρL+√ρR)` → sensible |
| 3 | `Gas.hpp:257` | `ComputeRoePreamble` | `asqrRoe = (γ−1)(HRoe − ½v²)` → correct sound speed |
| 4 | `Gas.hpp:662` | `RoeFlux_HartenYee` | α1 uses HRoe (sensible) in `(HRoe − u_n²)` |
| 5 | `Gas.hpp:718` | `RoeFlux_HartenYee` | Energy flux uses HRoe (sensible) in eigenvector components |

All consumption sites use the same sensible H convention derived at step 1.
The standalone eigenvector extractors `IdealGas_EulerGasRightEigenVector` /
`IdealGas_EulerGasLeftEigenVector` (Gas.hpp:540,567) also pass `rhoH_form=0`
to compute sensible H — intentionally, for diagnostic use and consistency
with the Roe eigensystem convention.

## EulerP note

The EulerP module (`EulerP_ARS.hpp`) computes `HLm/HRm` via `IdealGas::Enthalpy(U(I4),U(0),p)`
which returns **total** H (no formation subtraction). For non-reactive
flows this is identical to sensible H. Reactive EulerP would need a
convention alignment.
