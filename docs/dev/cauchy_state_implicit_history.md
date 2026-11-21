# Cauchy equation-state and implicit implementation history

These excerpts record earlier implementation stages and their validation limits. For current behavior, consult the [equation state](../architecture/CauchyState.md), [implicit operators](../architecture/CauchyImplicit.md), and [dispatch catalog](../architecture/CauchyDispatch.md).

## November 6 CPU capability checkpoint

Current CPU scope (November 6): configured flow5, SA/Wilcox/SST/realizable-k-epsilon
and optional ideal-mixture chemistry share logical DOF bindings, implicit time
equations, native archives and observation. Perfect-gas extended profiles use the
same flow5 Riemann dispatcher, including Roe_M1; general-mixture preparation
currently supports Rusanov. Extended wave/pressure-stencil producers remain
unavailable, so such flux selections need their explicit supplied scalar inputs
(`riemann.useStencil=false`) or fail at setup. Extended profiles use
polynomial viscosity, component WBAP and Euler-compatible implicit/rec PP;
missing characteristic/LVR/interface adapters
fail explicitly. QCR and the existing SA/SST DES formulas are connected below.
The short flame comparison is a converged discrete sanity run; the NACA pair
is capped and does not establish converged aerodynamics. CUDA remains deferred.
Both coupled chemistry and Euler's physical Strang/CVODE composition have
focused distributed gates and short matched spatial flame checks.
Earlier implementation checkpoints in this document describe their then-current
limits; they do not override the current profile summary or runtime guards.

## Initial CUDA kernel design

For Cauchy CUDA, the chosen direction is dedicated kernels for expected layouts
and physical profiles, not interpretation of a dynamically registered DOF
bundle on device. Host dispatch resolves the CPU registry and passes component
views directly, or a statically composed POD holding inline flow/RANS/species
views with absent segments represented by a type. Kernels may have different
loop mappings/staging for AoS, SoA, tiled or split storage; sharing the CPU
generic blocked-view loop is optional. Keep supported choices bounded and reject
missing kernels explicitly. Flow5 is initial; flow4 is a later incompressible
profile. RANS/species presence, widths and layout choices are explicit dispatch
inputs, not inferred physics from total width. A CUDA compile-only probe of
generic descriptor-based views proves technical device-callability, not adoption
of that interface or automatic GPU registry upload. Solver CUDA implementation
is deferred while the current goal concerns CPU correctness/performance.

## Early logical linear solver coverage

The logical driver is exercised by the distributed known-solution matrix fixture,
the actual conservative chemical-reactor Newton test and a coupled physical
mixture matrix. Nonlinear/frontend and checkpoint adapters remain separate. Matrix dimensions
alone do not validate a physical model.
The source/flux producer must supply the matching logical derivative ordering;
physical RANS/species acceptance remains separately gated.

## Scalar-expanded SA preconditioner comparison

The previous SA experiment used scalar expansion; that changed block Jacobi
into scalar Jacobi and allowed component-mixing aggregation. It therefore did
not assess the requested coupled-block preconditioner. This verified algorithm
difference does not alone prove the cause of that run's nonlinear residual
growth. The new entry keeps fixed V-cycle/smoother/coarse sweep counts. A separate
configuration-scope bug is corrected: Jacobi now receives the requested
relaxation (default 0.75), whereas the old string-valued smoother silently used
AmgX's default 0.9. Actual-operator residual diagnostics and corrected fixed-work experiments are
needed before making a causal or convergence claim.

## Time-equation recurrence tests

Focused tests compare two incomplete DITR iterations against independently
written Euler-form scalar equations for both variants, compare a single CPU
correction byte-for-byte with the first ordinary-solve update, and exercise
distributed FAS with explicit finest/coarse work accounting. This does not yet
complete the remaining whole-phase qualification. BDF/DIRK history ownership
and the first tracked mean-halo integration are described below.

## Multidomain nonlinear policy validation

The distributed recurrence fixture now invokes the production single-stage
policy with shared/reversed and disjoint domain rank groups, both refresh
cadences and inline/worker execution. It checks the independent two-variable
recurrence and live owned-read/MPI overlap, with domain norms reduced on their
subset communicators. The DITR policy ordering fixture includes idle control
ranks. Transfer rejection tests retain source/packet access checks
without preserving the removed prototype schedule API. These are focused
integration fixtures, not evidence of a completed fluid campaign. Their current
source migration must pass the grouped build/test gate.

## November 4 DITR stopping correction

The November 4 nested-control defect is corrected in code: each DITR iteration
now applies one intermediate correction and one endpoint correction, refreshing
each rate after its update. The DITR policy observes the method-provided
`I+H` residual after both updates. There is one report and one CFL/reference/cap
policy, not separate inner and outer acceptance criteria. The reacting-Domain
validation remains in progress; earlier nested-driver ratios are not evidence
for this corrected control.

## Initial implicit flow5 positivity qualification

The requested implicit flow5 PP measures have distributed operator/transaction
tests. Full physical qualification, high-order viscous equivalence and the
remaining phase implementation/performance work are still outstanding.

## Distributed stage and retry qualification

Distributed tests recover a manufactured constant solution from nonconstant
means, check unusable/bad correction handling, and exercise the connected
BDF1/2, Trapz and ESDIRK2/3/4 paths. The P3 transport test checks periodic
conservation and exact discard/retry versus fresh-run agreement. They are
focused implementation gates, not the final IV/cylinder qualification matrix.

## CGNS periodic checkpoint repartition

The original strict envelope passed generated-mesh tests but rejected a real
CGNS IV20 archive when read at np4 after writing at np8. Diagnostics identified
complementary periodic tags (-1/-4, -2/-5), while coordinates matched exactly.
A permanent CGNS repartition test was proven failing before this correction.
It now passes at np1/2/4/8. The actual old format3 archive also loads and takes
its next ESDIRK4 step after repartition, preserving loaded means and diagnostics.

## Early extended-state nonlinear adapter

This is a usable full-equation stage operator, not yet the production extended
Domain frontend. Boundary/checkpoint/frontend/RANS adapters and the requested
NACA0012/flame studies remain required. CUDA kernels remain deferred.

Focused tests compare finite-sweep flow5 corrections/norms against the existing
adapter, recover a mixture manufactured state with separate/permuted storage,
and advance a distributed reacting FV state through the actual nonlinear and
BDF drivers toward an independent Cantera reactor trajectory. The latter
checks temporal refinement and preserved full-energy flow components; it is
not the pending spatial flame sanity study. Critical errors cover aliasing and
negative-species candidate rejection without changing input generations.
