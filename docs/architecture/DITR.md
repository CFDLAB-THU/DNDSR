# Coupled DITR time equations

`Solver/TimeIntegration/DITRMethod.hpp` provides U2R2/U2R1 arithmetic without a
nonlinear loop. `StateDITR.hpp` is a forwarding alias. `FieldDITR` and Cauchy's
`EquationDITR` select storage/algebra adapters. Legacy Euler ODE is unchanged.

For initial, intermediate and endpoint states U0, Um, Ue, spatial rates
F0, Fm, Fe, and physical mass map M, the two equations are

```text
H = M(c0 U0+c1 Ue-Um)/dt + c2 F0+c3 Fe
I = M(U0-Ue)/dt + w0 F0+w1 Fm+w2 Fe
U2R2: c = [1-3a²+2a³, 3a²-2a³, a-2a²+a³, -a²+a³]
U2R1: c = [(1-a)², 2a-a², 0, -a+a²]
w = [1/2-1/(6a), -1/(6a(a-1)), 1/2+1/(6(a-1))]
```

Each common nonlinear iteration makes **one** correction to H+theta*I for Um,
refreshes Fm, makes **one** correction to I for Ue, refreshes Fe, then observes
I+H. This matches Euler's default thetaM2=0/thetaMMG=1; it is not a maximum of
separate H/I norms and does not contain completed inner nonlinear solves.
Defaults a=.55/theta=.9146 follow Euler. Cauchy's positive-scale correction
profile requires 0<a<2/3. U2R1 is available for reaction-dominated problems.
Temporal pMG and U3R1 are deferred.

## Ownership and execution contract

The method owns initial/intermediate state, initial/intermediate/endpoint rates
and resOther. The controller owns the endpoint trial. Begin copies initial
state and seeds both rate guesses with F(U0,t0). PrepareCoupledStage/Complete-
CoupledStage enforce intermediate/endpoint order. AssembleStoppingResidual
closes each iteration. The method does not choose convergence, mutate accepted
state, advance a clock or retain accepted-step history.

Prepared equations use resOther + alpha*F(U,t) - M*U/dt. resOther excludes the
current unknown and its RHS; the other state/rate and forcing remain frozen
through the whole correction, including a spatial V-cycle. CurrentRate exposes
the initial guess or last refreshed rate for that stage. Correction operators
can consume it while separately refreshing spectral auxiliary values and
chemical-Jacobian samples, without another reconstruction or chemical-rate pass.

Callbacks may refresh ghosts but preserve owned rate-input values, bindings,
other-stage inputs and forcing. Version/binding checks stay outside hot loops.
The controller serializes calls, agrees ordering and coordinates failures before
MPI. Callbacks complete outputs before returning and retain no temporary raw
views. Outputs guarantee owned values only. The equation operations use the
domain compute stream. A host observation must wait for its producing work.

The midpoint and endpoint have independent states, rates, source derivatives
and durable reconstruction coefficients. Their serialized corrections share
temporary FV scratch. On a new trial the midpoint copies the endpoint's warm
reconstruction, and both rates start at F(U0,t0). Exact internal restart
restores the saved midpoint and endpoint reconstructions and the active
method fields. Accepted-step restart uses the configured checkpoint contents.

Python can assemble the same two sources and the stopping residual with
`Cauchy.DITREquations`. It uses the native interpolation and integration
weights and queues owned row arithmetic on the supplied Runtime stream.
The pressure component of an ACM state has zero physical mass. Its rate
still contributes to the midpoint and endpoint equations.

## Common control and current validation

`DITRSequential2StagePolicy` observes after both corrections, once per iteration.
LastCoupledTargetRatio is the same selected-component I+H ratio as the single
nonlinear report. All component norms remain observable. Compressible flow defaults to density.
ACM uses its configured velocity components and independently checks continuity
at both stages. Reacting domains can select species by name. A finite admissible
capped iterate may be accepted by default, retaining converged=false; explicit
implicit.rejectUnconvergedStages rejects it.

Iteration limits and tolerances default to the domain's common `implicit`
controls. Explicit numeric `time.ditr` iteration/minimum/tolerance keys are
deprecated overrides; omitted/null keys inherit. Explicit zero is retained for
minimum iterations and tolerances. JSON serialization preserves inheritance
with null, so writing defaults does not turn them into overrides. Inherited and
overridden bounds must be consistent. They do not create an outer loop.
Named components, norm/reference and pseudo-CFL come from implicit.
There is no inner convergence tolerance for each individual correction.

Tests independently check two incomplete Euler-form iterations, both variants'
dense-system solutions, nonautonomous temporal order and retry/publication.
Public method and distributed implicit/FAS tests pass. Reacting-Domain
convergence qualification is in progress; the earlier nested-driver ratios are
not validation of this implementation. See [implicit contracts](CauchyImplicit.md).
