# State-based FAS cycles

`Solver/StateFAS.hpp` provides one synchronous V-cycle over explicit
`StateFASLevel<State>` operators and `StateFASTransfer<State>` edges.
`StateFASCycle<State, Operations>` delegates storage and owned algebra to a small
operations adapter. `Solver/FAS.hpp` retains the existing CPU field API as
`FASLevel`, `FASTransfer` and `FASCycle` aliases; Cauchy's logical-state adapter
uses the same recursion. Cauchy polynomial multigrid supplies spatial
levels and physics; the cycle does not construct meshes, own time history,
choose convergence criteria or perform hidden MPI. A future agglomerated
hierarchy supplies the same three transfer operations.

Operations provide allocation and compatibility checks, a retained
`Binding::Check(state, unchangedOwned)`, owned Copy/LinearCombination/Axpy,
component count, frozen-component checks, correction projection and selected
addition. They must explicitly reject unavailable backends. Neither recursion
nor the operation contract requires a single physical array, a particular row
layout or any knowledge of equation physics. The standalone field adapter executes on CPU. Cauchy supplies CPU and CUDA
operations for its supported equation profiles. All physical allocation and row-metadata owners survive callback access.

```text
FASCycle
  levels[i] -> retained state + apply F(U) + bounded smoother + update policy
  edges[i]  -> state restriction / residual restriction / prolong correction
  private scratch per level -> source, F(U), defect, entrance state, correction
```

For `F_f(U_f)=b_f`, the coarse equation is
`F_c(U_c)=F_c(I U_f)+R(b_f-F_f(U_f))`. After the coarse cycle, prolong
`U_c-I U_f`, apply the level's optional physical update policy, then post-smooth.
Both native recursion and caller-supplied schedules use these restriction and
prolongation operations. For a caller-supplied schedule, all ranks must agree
on the operation order. The cursor starts and ends at the finest level.
The incoming coarse reference remains unchanged throughout its recursive child
cycle. State and residual restriction need not be the same operator.

`FASCycleSettings::preSmoothByLevel` and `postSmoothByLevel` optionally supply
one nonnegative count per noncoarsest level, indexed from the finest. Empty
vectors retain the scalar pre/post counts. The bottom uses `coarseSmooth`.
Each level already owns its independent smoother callback; these counts change
only how often it is called, without introducing a level convergence loop or
changing its linear solver/physics. Cauchy resolves its named coarse-level
solver controls into those callbacks separately.

An optional `cancelIdentityLinearTerm` edge is reserved for a caller-proven
identity state restriction and the same removable linear operator on both
levels. Both levels must supply `applyForIdentityTau`; generic default edges
still evaluate their complete operators. For Cauchy's same-cell pMG this removes
the common `U/dt` analytically from tau differences. Smoothing and finest
convergence continue to use the complete time equation. This prevents a small
spatial correction from being lost when subtracting two large time terms;
it is not permission to omit time terms in a different-mesh hierarchy.

Callbacks write provided allocations without rebinding them, and return only
after their work and required communication completes. An operator may refresh
input ghosts but may not modify owned input values. Distinct levels have
nonoverlapping mutable state. Fixed bindings pin exact ArrayStorage objects,
not merely allocation IDs; this rejects rebinding to a disjoint range of the
same allocation. Temporary numerical views remain local to callbacks.

All ranks call the same hierarchy/cycle order. Callbacks coordinate local errors
before their communication; an optional LocalStage coordinates local vector
algebra errors. Caller serializes operations and releases conflicting leases.
The cycle reports work, not convergence: the physical controller must check its
original finest equation. A failed cycle can leave partial level updates, marks
the instance failed, and requires state restoration and hierarchy reconstruction
before retry. It is an in-place primitive, not an accepted-state transaction.

`FASCycleSettings::updateComponents` is an optional set of component indices;
empty means all. A complete explicit set is normalized to the same full-state
path. A proper subset requires equal component widths on all levels and each
level's `smoothSelected` callback. That callback receives the normalized set and
must solve its principal equations while holding the remaining owned components
fixed. Solving a full coupled system then discarding some entries is a different
method and violates the contract. Distributed callers agree the selection and
hierarchy before entering; the generic cycle performs no hidden MPI agreement.

Restriction still transfers the complete background state and residual for tau
construction. Each coarse level freezes its own restricted background, rather
than zeroing components excluded from updates. Prolongated corrections are
projected onto the selected components before the physical update. Optional
physical update callbacks must preserve the other means, including through PP.
Subset mode allocates one retained snapshot per level and checks frozen owned
values bit for bit after smooth/update callbacks, outside numerical kernels.
A violation fails the cycle; it does not silently restore an invalid update.
Full-state mode has no snapshot allocation or additional validation collective.
The outer controller still measures its original complete finest residual;
update selection and convergence selection are independent decisions.

Tests use independent two- and three-level numeric forcing/correction oracles,
a nonlinear fixed point, differing state/residual restrictions and mixed native
layouts. Negative tests exercise aliases, callback mutation, output rebinding
and same-allocation/different-view replacement. These are building-block tests;
they do not qualify Cauchy's pMG convergence or performance.
Subset tests additionally use an independent coupled 3-component/2-component
principal inverse, incompatible frozen equations, strided and empty fields,
prolongation projection, signed-zero preservation and rejected callback writes.
