# Native limited variational reconstruction

The native LVR path is a penalty inside the variational reconstruction solve,
not a final WBAP pass on its answer. It preserves the Euler v2/v3 family with
explicit buffers on the new Array, mesh, gradient and reconstruction facilities.
The physical adapter supports CPU and CUDA compressible perfect-gas flow5.
Unsupported equation/device selections must fail at capability resolution.

```text
FiniteVolume immutable data                 Per-evaluation workspace
  Selected GradientPlan (GG/LS/GGMP)                   target gradient + O2 coefficients
  LimitedVRPlan (face-neighbor stencils)       mean pressure/density + scalar gates
  shared CFV face-center geometry             base / seeds / layer scratch
  shared canonical cell-face incidence         retained shock/contact history
                     |                                   |
mean + independent center traces -> selected gradient -> order1 WBAP -> center PP
                                                        |
          mean and finalized-O2 face sensors -> extremum correction / expansion
                                                        |
                       frozen O2 target + alpha = max(shock, contact)
                                                        |
                   VR Jacobi / ordered two-buffer SOR with A-weighted penalty
                                                        |
                                existing final high-order positivity limiting
```

## Responsibility and indexing

CFV owns generic scalar sensor mathematics, immutable neighbor stencils and
the reconstruction penalty. `CFV/Math/LimitedVR.hpp` centralizes settings and
pure ramps/extremum calculations. `LimitedVRPlan` consumes canonical mesh
incidences and exposes owned-only core and halo-dependent phases; it performs
no MPI. `DoReconstructionIterLimitedTo` receives explicit target/alpha arrays,
with the same two-buffer, row-layout and SOR ordering rules as the ordinary
iteration. For existing relaxation `r`, its update is

```text
new = (1-r)*old + r*((1-alpha)*unlimited_VR_update + alpha*O2_target).
```

Alpha zero follows the unlimited arithmetic exactly; alpha one omits neighbor
reconstruction terms. Inputs and outputs must not alias. Validation precedes
writer acquisition, and input/target values must be finite. The generic CFV
operation does not interpret density, pressure, sound speed or equation width.

Cauchy's private `Discretization/LimitedVRInternal.hpp` and `FiniteVolume_LimitedVR*`
provide the physical adapter and persistent scratch. They prepare mean-based
characteristic transformations and Euler's common-beta O2 positivity bound,
then evaluate density, pressure, normal velocity and sound speed. There is no
dependency on Euler storage or its evaluator owner. Face-center sampling uses
the mapped reference center, including the preserved parent periodic frame;
it does not substitute an area-weighted centroid. Neighbor momentum is rotated
into the selected parent frame before sensing. Published face incidences retain
multiplicity, including periodic self-neighbors.

Only cell-face-cell neighbors enter the LVR sensor propagation. The cached mesh
work partition remains conservatively based on the prepared node-star halo.
Each propagation round reads a frozen previous layer through its ordinary
native halo binding. Core rows read owned input only, then dependent rows run
after transfer completion. There is no global-sized local scratch or special
single-rank algorithm. The public mean/reconstruction DOFs support AoS, SoA
and AoSoA through row views; the small private gate/history buffers use tables.

## Configuration and gate order

Select `numerics.limiter="LVR"` with variational degree at least two. Controls
live under `numerics.limitedVR`; selecting a `variant` in an input layer resets
that section to its explicit preset before applying overrides from that layer.
Later configuration layers may override those values. Typed C++ callers can use
`LimitedVRSettings::ForVariant` to construct the same preset.

| Preset | Shock cap | Contact cap | Extra contact seed layers |
|---|---:|---:|---:|
| v2 | 0.5 | 0 | 1 (inactive by default) |
| v3 | 1 | 1 | 2 |

Pressure, compression and density jumps use configurable cubic start/full
threshold pairs. `detectExpansion=false` uses compressive normal-velocity jump;
true admits the absolute jump. Mean-pressure extremum support augments the
shock branch. v3 additionally uses the mean-density/contact extremum branch.
Boundary/equal-neighbor and support rules follow the legacy family. The previous
final shock/contact fractions supply the configured retained extremum support.

Expand the pressure correction seed by one face-neighbor layer. Expand the v3
contact correction seed by `contactLayers`. Merge each unexpanded base gate
only after that branch's own correction expansion. Finally apply
`neighborLayers` extra shock rounds, optionally only propagating saturated
values (`saturatedOnly`). Thus different branch depths do not accidentally
expand the base sensor or the other branch. The frozen alpha and O2 target are
then reused through that RHS evaluation's VR sweeps. There is no post-LVR
3WBAP step. Setting both caps to zero omits the target/gate workspace and
recovers ordinary unlimited VR.

The weak-shock experiment, alternate Barth target, anisotropic/tree LVR and
new physical models are not advertised by this implementation. LVR with the
AMG reconstruction solver is also deferred: the existing CUDA LVR capability
belongs to the relaxation solver. The proposed true-residual formulation and
base-hierarchy reuse are recorded in the
[AMG deferred integration guide](../guides/amgx.md#deferred-lvr-integration-preserve-the-base-hierarchy).

## Numerical history, observation and restart

Retained LVR fractions are numerical warm state alongside reconstruction
coefficients. A physical trial saves both before its first stage; discard or
explicit rejection restores both before retry. Observation owns separate
scratch initialized from accepted warm state, so history/plot cadence cannot
change the production trajectory. A pending-trial checkpoint writes accepted
means, clock and the saved accepted warm state.

Default checkpoints omit both coefficients and LVR history. With reconstruction
enabled, the optional owned `limited_vr_history` array has width two (shock,
contact), schema marker one, and the same exact persistent-cell-ID redistribution
as other checkpoint fields. A compatible warm load restores both together.
Missing optional history or incompatible polynomial metadata cold-starts both
at zero; a file declaring missing/malformed history is an error. Allocation,
shape, bounds and access checks complete before either warm destination changes.

Warm restart does not require writer layout or local row order to match the
reader. Native Array I/O uses component datasets for SoA/AoSoA without an AoS
normalization buffer at write time; Cauchy maps read shards into the configured
target storage. See [solver contracts](Cauchy.md) for conversion details.

An iteration cap follows Euler's default advance-at-cap control flow, retaining
an unconverged report. Only `implicit.rejectUnconvergedStages=true` rejects the
whole physical attempt on a cap. Invalid states/nonfinite operations remain
errors. This policy does not establish physical or inner convergence.

## Validation scope

`CFV/test_LimitedVR.cpp` compares pure sensors to legacy v2/v3, independently
checks ramps/extrema and distributed layer distances, and exercises penalty
limits, SOR/Jacobi, layouts, active halo writes and bad inputs.
`Cauchy/test_LimitedVR.cpp` exercises shock/expansion sensing, exact zero-cap
recovery, rank-inconsistent controls, warm-state access errors and periodic
split/unsplit execution. `Cauchy/test_LimitedVRCheckpoint.cpp` uses nonzero gates
to test restart, local permutation, degree change, output cadence and retry.
These are focused implementation gates; full DM240U completion and numerical
qualification are separate experiment records in the plan repository.
