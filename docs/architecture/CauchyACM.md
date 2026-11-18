# Constant-density artificial compressibility

`equation.system = "incompressibleACM"` selects three Cartesian velocities
and relative mechanical pressure in that order. Code density is one.
`primUP` accepts a vector in code units. `primUP_phy` divides velocities
by the explicit reference velocity and pressure by `rho0 * U0²`. Both use
the same vector-valued expression and region facilities as other Cauchy
inputs. The material requires explicit positive `rho0, U0, L0, T0`,
`eos = "none"`, and either inviscid transport or Newtonian transport with
constant viscosity. A pressure offset can be negative.

## Stage equations and pressure gauge

For state `Q = (u,v,w,p,s₁,…)`, physical accumulation is
`M Q = (u,v,w,0,s₁,…)`. Cached stage **rates** retain the complete
spatial continuity residual. In volume-normalized units the nonlinear defect
is `resOther + a R(Q) - m M Q`. Each approximate correction uses the
integrated block matrix `V Gamma/dTau + m V M - a dR_integrated/dQ`.
The first three entries of the pressure column of Gamma are
`(1+alpha) velocity/beta2`, and the fourth is `1/beta2`. In dual time each
transported primitive scalar contributes `s/beta2` in this column. The
steady scalar pseudo-equations use the identity scalar diagonal. Use
`implicit.diagonal = "block"` and `jacobianAction = "assembled"`.

For closed and periodic domains the correction removes its volume-weighted
mean pressure change, retaining the input or restarted gauge. A pressure
outlet or far-field boundary supplies the pressure reference. Select `p`
(the continuity equation) alongside the differential components for nonlinear
convergence, or select all components. Histories distinguish the normalized
time-equation residual from `continuity_rms`, the volume-weighted RMS of the
accepted state's spatial continuity residual. `mean_pressure` records the gauge.
The default resolved ACM controls reject capped stages. Plain SSPRK3 cannot
advance this singular physical system. The one-cycle vortex campaigns qualify
individual implicit time methods separately.

An analytical solenoidal velocity can still have a nonzero discrete continuity
residual because the numerical flux includes pressure dissipation. Trapz and
ESDIRK require the explicit initial stage and each subsequent stage to satisfy
`implicit.algebraicRateTolerance` (default `1e-8` in inverse code time).
DITR also tests the midpoint and endpoint spatial continuity separately.
Its combined update residual can otherwise hide cancellation between them.
A failed starting-state check leaves the accepted state and time unchanged.
Subtracting a constant pressure changes the gauge and cannot remove a spatial
continuity residual. BDF1 and BDF2 impose continuity inside the implicit
equation, so their initial data need no explicit-stage check.

For the supplied 20×20 periodic vortex, the separate
`cauchy_experiment_acm_vortex_projection` target prepares discretely
consistent time-zero means. Run it from the mesh checkout with the case JSON
and a fresh checkpoint filename. It holds the analytical pressure fixed,
corrects the horizontal velocity, and verifies the resulting rate with the
same finite-volume operator used by the solver. Supply this checkpoint with
`--restart` to use identical initial data across integrators. A time-zero
snapshot containing only means carries no active nonlinear iteration. An
internal checkpoint still requires its original method and update policy.

## Face operators and turbulence

Scheme A uses the arithmetic mean velocity and the `acmAlpha` and `acmBeta2`
controls. The Roe absolute wave operator supports a relative Harten smoothing
width (`acmEntropyFixRatio`, default 0.05). Rusanov uses a spectral-radius
bound on the preconditioned state jump. Both conserve the face flux, including
the scalar transport upwinded by the numerical continuity flux. The assembled
matrix differentiates the first-order inviscid face flux, including its boundary
or periodic mapping, and adds frozen viscous and source approximations.
The reference formulation comes from [Kirabo's GPUACM branch](https://github.com/Kirabo167/DNDSR-Kirabo/tree/GPUACM).

SA appends the normalized working variable (`nuTilde / viscosityReference` in
code units). A dimensional `primUP_phy` SA tail supplies kinematic viscosity
in m²/s. Divide it by `U0 * L0 * viscosityReference` to obtain the stored
value. Wilcox and SST append `k, omega`, with physical scales `U0²` and
`U0 / L0`. The velocity gradient drives
the shared RANS closures with constant density, molecular viscosity and the
mesh-derived wall distance. The two-equation momentum flux includes the
isotropic `2 k/3` contribution. At a stationary no-slip wall the trace has
zero velocity and zero SA working variable or k. Omega uses the existing
viscosity and cell wall-distance law. The wall shear columns give the
tangential viscous traction of the fluid on the wall per unit area from the
last completed full RHS. Summed numerical momentum flux and per-area shear are
distinct observables.

The initial scope uses stationary single domains, unlimited reconstruction
and FP64 ACM face fluxes. CUDA requires combined AoS and the resident Jacobi
or supported Krylov path. See the periodic [vortex](../../cases/cauchy/acm_translating_vortex_20.json)
and the [SA](../../cases/cauchy/acm_plate_sa.json) and
[SST](../../cases/cauchy/acm_plate_sst.json) plate inputs for complete configurations.

BDF1/2, trapezoidal and ESDIRK2/3/4 have completed three-timestep
translation studies on this mesh. DITR-U2R2 with its default fraction
gave velocity and pressure self-orders of 3.23 at dt=2/3, 1/3 and 1/6,
with independent stage-continuity checks. At the default fraction
and theta, the sequential DITR-U2R1 corrections diverged on this
fixture even with consistent initial data. The invertibility check on the
coupled time equations establishes independence of the two constraints.
Convergence of the alternating corrections requires separate verification.
For U2R1, setting `time.ditr.fraction` to 1/3 gives the two-stage Radau IIA
tableau. With the original theta and nonlinear tolerances, the velocity and
pressure self-orders were both 2.96 over full cycles at dt=0.4, 0.2 and 0.1.

On the supplied stretched plate mesh the SA fixture uses Green–Gauss P1 face
states. The SST fixture uses cell means at faces and the normal jump for face
diffusion, retaining gradients for the turbulence source. Both fixtures require
convergence of the turbulence variables along with momentum and continuity.
These choices are specific to this grid. Inspect wall friction and profiles
separately from the nonlinear stopping criterion.
