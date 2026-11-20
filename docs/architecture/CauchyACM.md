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
is `resOther + rhsScale R(Q) - (M Q)/dt`, where `R(Q) = R_integrated(Q)/V`.
The pressure row retains the continuity rate even when its correction is masked.
Each approximate correction uses the integrated block matrix
`V Gamma/dTau + V M/dt - rhsScale dR_integrated/dQ`. The time method supplies
`inverseDt = 1/dt` for a finite physical step and zero for a steady solve.
The first three entries of the pressure column of Gamma are
`(1+alpha) velocity/beta2`, and the fourth is `1/beta2`. In dual time each
transported primitive scalar contributes `s/beta2` in this column. The
steady scalar pseudo-equations use the identity scalar diagonal. Use
`implicit.diagonal = "block"` and `jacobianAction = "assembled"`.

Closed and periodic domains project pressure to one fixed volume-weighted
mean. Set `equation.acmPressureMean` to override the initialized mean in
code units. Accepted and internal checkpoints carry the target. Older
checkpoints derive it from the loaded accepted field. A pressure outlet
or farfield zone present on the mesh supplies its own pressure reference.
The default nonlinear convergence selection is momentum `u,v,w`.
Continuity also has its own absolute volume-weighted RMS criterion in all
implicit methods, including steady and BDF. Histories distinguish the normalized
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
continuity residual. BDF1 and BDF2 start directly with the implicit solve and
check continuity there.

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
mesh-derived wall distance. SST uses SST-2003m production and Cauchy's
mean momentum flux omits the isotropic `2 k/3` term. Slip walls retain
inviscid pressure flux with zero viscous and RANS scalar diffusion on CPU
and CUDA. At a stationary no-slip wall the trace has zero velocity and zero
SA working variable or k. Omega uses the existing viscosity and cell
wall-distance law. Opt-in boundary VTKHDF stores the face average of the
tangential numerical momentum flux from the last observation RHS.

The default `implicit.boundaryJacobian = "physical"` differentiates the
mean-state ACM flux with the boundary exterior-state rule included.
`"eulerScalar"` instead contributes half the face area times the sum of
the prepared convective and viscous spectral radii, scaled by the residual
coefficient, to each diagonal entry. This changes the approximate matrix
at physical boundary faces on CPU and CUDA. It keeps the full ACM time
block, the interior face linearization, and the centered source diagonal.
The RHS and its wall states use the same flux in both modes.

During RANS setup, Cauchy reports the domain-wide minimum cell-center wall
distance before applying `numerics.wallDistanceFloor`. The distance is the
nodal Geom wall field interpolated with the element shape functions at
the same reference center as `FiniteVolumePlan::GetCellCent`.
`numerics.firstLayerCenterWallDistance = true` selects one distance for
all volume quadrature points in each cell incident to a no-slip wall.
`numerics.firstLayerWallDistanceMethod` defaults to `cellCenter` for the
interpolated center value. Set it to `halfMaxNodal` for half the maximum
nodal wall distance of the cell. Both rules apply the configured floor
and supply the SST wall \(\omega\) condition and other mean-distance
consumers. Setup also reports the global minimum of the applied first-layer
distance. Face quadrature evaluates the nodal interpolant at each face
point. With the feature disabled, each volume quadrature point evaluates
the nodal interpolant and mean-distance consumers use the unweighted
mean of those samples.

Set `numerics.cellCenterWallDistanceForQuadrature = true` to reuse
each cell's interpolated center distance at its volume quadrature points.
Each interior face uses the arithmetic mean of its two adjacent center
distances at every point. A physical boundary face uses its only adjacent
center. The floor applies to each center before averaging. Outer ghost
faces beyond the local flux stencil retain nodal samples if their parent
cells are outside the halo. The first-layer mean-distance override still
feeds the omega boundary condition and other mean-distance consumers.
The option defaults to false and applies on CPU and CUDA through the same setup.

A positive `numerics.wallDistanceFloor` clips every volume and face
quadrature sample and either first-layer replacement to that minimum in
solver units. The default value `0` uses the existing positive denominator
safeguard of `1e-10` for RANS, since wall values can be exactly zero.

With variational reconstruction, `numerics.ransP1Overlay` forces the RANS
coefficients after reconstruction and limiting, before coefficient exchange.
For each RANS scalar, the linear coefficients are the cell length scale times
its physical mean gradient and the higher coefficients are zero. The ordinary
face and volume samplers consume this polynomial. Flow coefficients retain
their VR order, including the velocity gradients in the SST volume source.
With `numerics.limiter = "Barth"`, the RANS coefficients use the Barth-limited
physical gradient. Cell-center turbulent viscosity and the centered source
Jacobian continue to use the unlimited mean gradient.
`ransMeanFaceValues` independently selects cell means for the RANS face value.
Both switches default to false.

`numerics.ransCellCenterViscosity` requests a separate scalar auxiliary
Array on owned and ghost cells. The auxiliary pass computes eddy viscosity
using the cell mean and center wall distance, with the mean gradient for
two-equation closures. The viscous flux averages the adjacent cell values on
interior, periodic and MPI faces. Physical and domain-interface boundaries
use their pointwise closure. This switch defaults to false.

ACM uses stationary single domains and FP64 face fluxes. For direct P1
reconstruction, set `numerics.reconstruction = "gradient"` and
`numerics.limiter = "Barth"` to limit the gradients used in the RHS. The
limiter bounds each velocity, pressure and turbulence component by cell
means. It rotates only the velocity triplet for periodic neighbor reads.
Positivity limiting is disabled for ACM. CUDA requires combined AoS and the resident Jacobi
or supported Krylov path. Both CUDA face mappings have generated polynomial
specializations for flow4, SA and two-equation profiles, in 2D and 3D for
degrees 1–3. On A100 the automatic policy uses CTA segments for 3D P2 with
quadrature order 4 or higher and 3D P3 with order 2 or higher. Other ACM cases
use one thread per face. The policy is resolved at placement and explicit
`execution.kernels.rhsMapping` choices take precedence. A CTA workspace
retains its mesh- and rule-dependent face packages across RHS evaluations.
An order-one `numerics.cfv.intOrder` sets one-point VR/RHS quadrature while
retaining integrated cell and face metrics.

See the periodic [vortex](../../cases/cauchy/acm_translating_vortex_20.json)
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
