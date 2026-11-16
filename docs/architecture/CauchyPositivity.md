# Cauchy positivity policy

The implicit CPU policy preserves Euler's implemented numerical sequence. A thermal
criterion and the components multiplied by its factor are different concepts.
RANS positivity does not determine the thermal factor, but Euler applies that
factor to the entire nonlinear increment/RHS. Reconstruction has a different
scope. Do not replace these policies with a common generic limiter.

CUDA modal PP samples each lane's current point into registers and checks it
before a warp/block acceptance reduction. The common unchanged case needs no
shared sampled-state table or lane-zero point scan. Refresh first stores and
reloads the candidate under the operator's exclusive output access; therefore
that sampling pass verifies actual stored coefficients. Only cells requiring
limiting materialize cell-local shared samples for the existing base/beta
algorithm. Modified coefficients are stored, reloaded and checked cooperatively
after every recovery attempt. No candidate becomes visible to another operator
until successful completion; a failure leaves output unspecified. Species/RANS
scope, thermal bounds and the lower-base sequence remain unchanged.

## State and energy constraints

Write the conservative state as `U=(rho,m,rhoE,z,r)`, where `z_k=rho*Y_k`
contains independent species densities and `r` contains RANS variables. Logical
block locations are resolved separately from physical AoS/SoA/AoSoA storage.
The implicit last species is `z_last=rho-sum(z_k)`.

For fixed reference temperature `T*`, the raw energy reference is affine:

```text
B_T*(U) = rho*e_last(T*) + sum_k z_k*[e_k(T*) - e_last(T*)]
Q_T*(U) = rhoE - dot(m,m)/(2*rho) - B_T*(U)
```

Require `rho >= densityFloor` and both prepared `Q >= floor` constraints.
For a perfect gas these are the pressure/internal-energy floor and the
temperature floor. For mixtures they are the base-reference sensible-energy
margin and the energy at `max(temperatureFloor, material.BaseTemperature())`.
No equivalent gamma is used. The equation owner prepares the species energies;
the numerical view performs arithmetic without chemistry callbacks.

`AdmissibilityPolicy` is the owning mean-state contract. It combines
the prepared thermal/species coefficients with the resolved RANS scalar policy
and publishes one layout-independent `AdmissibilityView`. `Domain`,
standalone finite-volume checks, restart validation and nonlinear acceptance
therefore use the same row predicate. A mixture policy acquires a chemistry
worker only during preparation; row checks perform no EOS inversion, worker
acquisition or allocation. The policy retains the equation property key so a
same-width state from another material cannot be validated accidentally.
Specialized fixed-flow kernels may retain a five-component affine bound, but it
is an internal leaf and not a competing orchestration-level admissibility rule.

With positive density and the **raw affine reference**, Q is concave along a
line in conservative state space. Its endpoint chord supplies Euler's
scheme-one admissible compression estimate. Species must move with flow along
that line: independently changing composition can violate the energy bound
even when the flow increment is zero. The accepted design retains common
flow/species thermal compression rather than introducing a species backoff.

Property evaluation can use a clipped/normalized temporary composition. That
reference is not globally affine; the same concavity argument is not a proof
for that repaired map. Euler nevertheless uses its tolerant property
composition for the **forecast** RHS/dTau and provisional increment factors.
Cauchy retains that compatibility: those factors guard a trial direction and
do not accept a stored state.

Reconstruction has two simultaneous representations. Its raw, possibly
out-of-simplex `rhoY` polynomial is retained and supplies conservative species
advection and diffusion. A temporary clipped/normalized copy supplies only the
pointwise EOS, transport and temperature solver. The raw affine reference gives
the analytic beta; final sampled-point acceptance checks the temporary clipped
reference actually seen by physics. Euler's compatibility policy falls back to
the validated base on failure; `backtracking` searches an intermediate common
flow/species beta. Neither policy writes the clipped composition back into
coefficients, means, halos or flux states.

In short, the reconstruction coefficients and the conservative flux state keep
the raw polynomial; clipping is a read-only point-property adapter. By contrast,
final increment repair publishes the simplex-repaired species densities into
the cell mean itself. There is no raw out-of-simplex mean hidden behind that
accepted state.

Final mean repair is different: it constructs the exact simplex-repaired species
target before bounded raw-affine pull-up, and the resulting clipped `rhoY` is the
actual state stored, exchanged and supplied to EOS/T. Both endpoints of the
pull-up line lie in the simplex, so the raw affine reference equals the clipped
composition everywhere on that line. The raw arithmetic is therefore an
efficient proof of the final clipped-state invariant, not a second raw-species
acceptance policy. No later species projection is permitted. An exhausted or
nonfinite repair is an explicit failure, not a successful PP update.

> **TODO(PP-RECONSTRUCTION-CLIPPED-RHOY):** evaluate with matched Euler/Cauchy
> oracles whether the analytic reconstruction thermal beta should use the
> clipped/normalized temporary-composition reference instead of the current raw
> affine reference. The clipped map is heuristic and non-convex; retain final
> endpoint checking/backtracking for either choice.

## Operations and their scopes

RHS FPP means **RHS forecasted positivity preserving**. Its factor can be
applied to either the implicit solve defect or the pseudo-step; it is not a
selection of the spatial RHS algorithm. Current configuration separates the
three operators:

```json
{
  "time": {
    "positivity": {"enabled": false, "relax": 0.8, "scale": 1, "increaseLimit": 2}
  },
  "defaults": {
    "implicit": {
      "positivity": {
        "rhsFPPMode": "rhs", "rhsFPPReduction": "cell",
        "rhsFPPScale": 1, "rhsFPPRelax": 0.5,
        "incrementRelax": 1, "compressionPower": 1
      },
      "multigrid": {"coarseRHSFPP": false}
    },
    "numerics": {
      "positivity": true, "positivityReconstruction": "Euler",
      "positivityCompressToMean": true,
      "positivityBetaPower": 1, "positivityBetaCutoff": 0.001
    }
  }
}
```

Domain settings can override the `defaults` sections. Physical dt PP has its
own enable flag. Interior reconstruction PP uses the same quadrature as volume
source integration, regardless of whether the source is active; face PP uses
the RHS face points. There is no separately configurable PP quadrature.
`numerics.positivity` currently gates reconstruction and
implicit thermal PP. `rhsFPPMode` selects none, complete-defect scaling (`rhs`)
or scalar pseudo-step compression (`dTau`). `rhsFPPScale` multiplies only the
trial increment; `rhsFPPRelax` sets its Euler relaxed density/energy threshold.
`incrementRelax` separately controls the post-linear correction.

`rhsFPPReduction` preserves Euler's factor sharing. For original cell factors
`[1,.8,.3]`, `cell` applies `[1,.8,.3]`; `minimumLimited` applies `[1,.3,.3]`;
`minimumAll` applies `[.3,.3,.3]`. The minimum spans the domain communicator.
The trigger count records original limited cells. Even cell mode retains the
required global diagnostic statistics; disabled RHS FPP performs neither its
factor calculation nor these reductions. This scope does not select a new
spatial flux blend or a global increment-repair method.

The former development keys `rhsMode`, `rhsReduction`, `rhsScale`, `rhsRelax`
and `coarseRHSPositivity` are renamed to the RHS-FPP keys above. Unknown config
keys fail explicitly. Archived run configs retain their original source/schema
provenance; their immutable records are not rewritten.

| Operation | Factor calculation | Application |
| --- | --- | --- |
| Reconstruction beta | Raw-affine analytic beta followed by final clipped-reference energy/temperature-floor checking at every configured cell/face/source point | Flow5 and raw species coefficient difference from the selected reference polynomial; clipped species exist only as temporary EOS/T input; not RANS |
| Implicit RHS compression | Property-projected thermal forecast `U + rhsFPPScale*dTau*defect/V` | Entire solve defect, as Euler's `cres *= alpha` |
| Implicit dTau compression | Same property-projected forecast | Replace scalar dTau by `max(alpha,smallReal)*dTau`; no second component-wise RHS scaling |
| Implicit increment compression | Property-projected forecast with the selected nonlinear correction | Entire correction, including RANS and species, before final repair |
| Implicit final increment repair | Raw affine density/thermal bound of the exact stored-species target | Entire correction, then separate RANS repair and addition; no later species projection |
| Physical dt restriction | Property-projected forecast with `dt*scale*R/V` | A scalar global timestep restriction; no independent RANS/species positivity scale |

Coarse pMG defaults follow Euler's omission of the RHS/dTau limiter:
`implicit.multigrid.coarseRHSFPP=false` forces coarse rhsFPPMode to none
after level overrides. True enables the inherited/per-level RHS/dTau policy.
This switch does not disable finest RHS/dTau limiting, reconstruction beta or
final increment compression/repair. It changes numerical policy, independently
of optional collective diagnostics and MPI launch validation.

Thermal factors depend on species through energy, even though there is no
additional minimum over individual species or RANS positivity ratios. A scalar
time restriction naturally affects all coupled equations. CPU physical-dt PP
also accepts the logical extended state; it uses the same mixture reference
energies and `compress=1`, with no independent species/RANS positivity minimum.
This is a timestep proposal, not a proof that every explicit stage is positive.
Extended explicit flux blending remains unavailable.

Equation reconstruction PP uses `CFV::PolynomialSamplingPlan::EvaluateValues`
for its value-only samples. This retains the same geometry/shape validation and
read/write leases as generic `Evaluate(..., 0)`. Cached owned AoS fields with
5/6/7 components and 2D P1/P3 mode counts use bounded CPU kernels; every other
layout, width, ghost or cache configuration retains generic evaluation. Source
derivative sampling stays in the original evaluator translation unit, avoiding
changes to its generated loop when value kernels are specialized. This changes
execution only, not positivity points, policy or reconstruction arithmetic.

## Explicit stage acceptance and retry

SSPRK3 uses the uncompressed conservative residual for flow5 and species in each
Shu-Osher stage. It never applies cellwise thermal increment compression or
species mean repair to make that stage acceptable. The user permits Euler's
separate RANS exponential increment repair and model scalar mean bounds; those
act only on the RANS block and never rescale flow5/species. Reconstruction
limiting and temporary property-state validation remain spatial operations.
The raw species candidate must satisfy nonnegative independent densities and
their sum no greater than rho (the implicit last species is the remainder).
Failing this check rejects the trial; no clipping/normalization of accepted
explicit species values is allowed.

The established perfect-gas flow5 path may blend low/high residuals with one
globally shared factor. Both sides of a face use the same factor, retaining
flux cancellation. Extended equations currently have no such flux-blend adapter:
an inadmissible stage rejects the complete physical-step trial. `numerics.positivity`
still selects their reconstruction PP; it does not enable increment repair.

For each cell, form the complete low/high SSPRK stage candidates with the
same history weights and dt. `AdmissibleFraction(low, high)` bisects their
conservative-state segment for 56 iterations against density and thermal
interior bounds, retaining a small numerical safety margin. If every high
candidate is admissible, alpha is one. Otherwise the low candidates must be
admissible everywhere and alpha is the minimum local fraction over all cells
and MPI ranks. Apply that single alpha to the residuals, not a per-cell factor
to updates. The search re-evaluates state predicates, never the face fluxes.

**Deferred direction:** replace the domain-global factor with a conservative
local flux-correction scheme. A face must use one shared correction coefficient
for its two adjacent cells, including across MPI partitions, and satisfy both
cells' coupled density/thermal/species constraints. The admissible low-order
update, source and boundary treatment, and multi-stage guarantee need a separate
design and validation. No local algorithm is implemented here; independently
compressing cell increments is not a substitute. The global blend remains a
simple baseline until that explicit PP work is undertaken.

The caller retries from the accepted state with smaller dt. `Simulation::Advance`
discards every domain's failed trial without changing accepted values, time or
step count. The production C++ frontend halves dt up to `time.maxRetries`
(default zero, meaning report the rejection). Direct Domain/Python callers
perform the same discard/retry explicitly; use `DiscardStep()` after a failed
`PrepareStep()` and supply the reduced dt. The next trial restores the accepted
reconstruction guess and restarts all three stages. Backend/allocation failures
can poison private scratch and require rebuilding the Domain; they are not
ordinary positivity rejections. No timestep restriction proves every later
stage admissible.

Physical Strang splitting also accepts SSPRK3 transport: chemical half-step,
source-free SSPRK3 transport, then chemical half-step. The external CVODE
reactor retains its separate Euler-compatible species-output convention; it
does not apply a compressed increment to the explicit transport stages. Overall
Strang composition is second order regardless of the transport method's order.

The extended trial uses `EquationMeanWorkspace`, separate from the normal VR
workspace, over the same FV plans and CPU engine. Its coefficients stay zero,
VR/WBAP/rec-PP are skipped, and viscous face fluxes are ignored. Configured
chemical/RANS volume sources and frame sources remain present, following
Euler's mean-trial sequence. RANS source/wall producers still receive actual
molecular properties and geometry; ignoring viscosity does not disable sources.
No accepted owned values, warm polynomial, physical clock or history changes.

`FiniteVolume::LimitPhysicalStep` reads ready, nonaliased owned logical rows;
a mean ghost transfer may be active. Its caller retains the matching frozen
thermal-policy buffers and coordinates the same model/policy on domain ranks.
The wrapper validates schema/mapping and finite increments before the existing
global minimum/count reductions. Domain lazily owns the independent trial
workspace/residual and applies growth capping from the last committed dt.

## Implicit compression details

The relaxed density target is
`rhoTarget = relax*densityFloor + (1-relax)*rho`. Density compression is applied
first. For a decreasing sensible energy the relaxed energy target is
`floor + (1-relax)*(Q_old-floor)`. After the scheme-one energy estimate, Euler's
additional compression is `(alphaRho*alphaEnergy)^(compress*power)*(1-1e-5)`
when the unpowered factor is below one. Internal/increment control uses
`compress=2`; physical-dt control uses `compress=1`. These forecast factors use
the tolerant property composition at their old/trial endpoints, matching
`EulerEvaluator::EvaluateCellRHSAlpha`. They are deliberately distinguished
from the raw affine proof used after the linear solve. A diagnostic must report
RHS and final-increment limiter counts separately: an unconstrained final
increment does not imply that the pre-solve RHS forecast was inactive.

Implicit final increment repair first forms and simplex-repairs the exact species
target. It then uses the unrelaxed raw-affine bound, the legacy `.99` safety
factor and, if needed, successive `.9` pull-ups with a finite iteration cap.
The repaired rhoY is the actual stored/exchanged mean. Since the repaired target
and the old mean lie in the complete simplex, their connecting line also lies in
the simplex; along it the raw species reference and clipped EOS/T composition
are identical. The accepted-state invariant is explicitly the **clipped,
simplex-valid rhoY state consumed by EOS/T**. The raw-affine predicate is only a
cheaper line-search proof of that clipped invariant in this operation; it does
not authorize retaining a raw out-of-simplex mean. Afterwards RANS uses the
model-specific exponential
negative-update repair (and the SA floor where enabled), then its mean bounds.
No later species projection may invalidate the line search. The combined state
is validated after these repairs. Scalar repairs must not silently change a
component frozen by a partial solve. Exact species-target projection is not a
common thermal alpha: it does not reduce `minimumIncrementAlpha` unless the
subsequent density/energy line search itself contracts the coupled update.

Reconstruction optionally chooses a lower-order admissible reference before
compression. This order fallback can discard higher modes in **all** blocks,
as in Euler; it is distinct from multiplying the retained polynomial by beta.
The unlimited VR warm guess remains unchanged; PP updates the flux polynomial.
Before physical reconstruction BC evaluation, the warm polynomial's sampled
point is checked separately when positivity or face fallback is enabled.
An inadmissible temporary point falls back to the cell mean, following Euler's
`FBoundary -> CompressRecPart` role. The thermal predicate uses a temporary
clipped composition; raw species coefficients remain unchanged. A reported
EOS/T failure may also retry from the mean; invalid BC requests and backend
failures remain errors. This boundary input repair does not replace beta
limiting, alter the warm guess or repair a mean state.
Thermal admissibility at sampled points is not a proof of positivity at every
point inside a cell.

## Legacy correspondence and validation limits

- `EulerEvaluator.hxx::EvaluateURecBeta`: reference/order fallback,
  flow/species beta, and separate two-equation scalar coefficient damping.
- `EulerEvaluator.hxx::EvaluateCellRHSAlpha`: relaxed thermal ratio, powers,
  safety factor and global reduction options.
- `EulerSolver.hxx::fsolve/fincrement`: whole-defect scaling or scalar dTau
  reduction, then whole-increment scaling before `AddFixedIncrement`.
- `EulerEvaluator.hpp::CompressInc/AddFixedIncrement`: final density/energy
  repair, scalar RANS repair, addition and species mean repair.

RANS reconstruction is independent of flow/species base selection and beta.
For nonnegative cell-mean scalar q and polynomial increment d(q,x), choose
`theta = min(1, q / (-d(q,x)))` over negative reconstructed values. Scale the
entire scalar coefficient columns by `0.9*theta` only when theta is below one.
The floor is zero; tiny positive constant states remain unchanged. Two-equation
models share one theta across the pair. SA follows this rule only with
`USE_NS_SA_NUT_REDUCED_ORDER`; otherwise its polynomial is untouched.

The scalar correction runs before flow's full-polynomial early exit. Rejected
flow bases and their coefficient writes operate only on flow/species columns,
so they cannot erase RANS coefficients. Scalar-only repairs do not affect
thermal beta or its diagnostic count. Species stay coupled to thermal PP.

Reconstruction no longer needs primitive far-field scalar floors or a reference
density conversion. `numerics.ransState.referenceConservative` remains the
explicit physical-state reference for mean repair (including the conservative
omega floor). It is separate from nondimensionalization scales, and this change
introduces no unit-density fallback. Point-value RANS fallback in property/flux
operators and temporal increment/mean repair keep their existing policies.

The homogeneous mixture fixture demonstrates convergence of a short discrete
reacting step, with selected H2/H2O residuals and actual species evolution. It
does not establish equilibrium, flame accuracy, or robustness for arbitrary
chemistry steps. See the matched physical studies in the active phase plan.

## Point-local execution and geometry

Modal PP is fused per cell on CPU and CUDA. It keeps bounded per-worker/shared
samples, not full-domain sampled/base fields. Face and volume points retain the
RHS/source rule. Physical-gradient PP and LVR's linear target instead check
incident face centers only, validating the mean separately. This declared
gradient point-set change does not lower modal PP or RHS quadrature.

Reference selection, independent RANS scaling, local recovery and the actual
post-storage thermal predicate retain the policies above. CUDA producers emit
compact status/count/minimum block summaries, finalized without a cell-sized
rescan. No host-observed global retry occurs between local recovery trials.
See [compact geometry and ownership](CauchyCompactGeometry.md) for boundary
center packets, shared volume basis and queued lifetimes.
