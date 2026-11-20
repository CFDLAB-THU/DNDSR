# Cauchy audit fixes for recent Euler corrections

## Scope and provenance

Worktree: `/home/harry/projects/DNDSR/workspace/worktrees/DNDSR-cauchy-audit-fix`.
Branch: `codex/cauchy-audit-fix`, based on refactor commit
`aec8d0e4194aa3ab39758ead263e0929fd1b9135`.
The reference Euler corrections are `14c4fd47`, `477218a0`, `bbff4412` on
`dev/harry`, following merge base `46be01186330617786943438e28ccdc1fa9f0ebf`.
They are audited here without merging the dev branch into the refactor.
MG/IV jobs, artifacts, and campaign configurations belong to another agent
and are excluded from this work.

## Implemented corrections

1. **Shared SST omega production.** `Physics/RANS/RANSModels.hpp` now blends
   `gamma1*F1 + gamma2*(1-F1)`. Cauchy's SST source dispatcher uses this shared
   kernel. The implicit destruction diagonal and existing model variant remain
   the same. An independent constant-density shear oracle checks inner and
   outer production coefficients, rather than comparing two calls to the
   same erroneous kernel.
2. **Independent RANS reconstruction positivity.** Flow/species admissibility
   determines the reference polynomial and flow beta. Scalar values no longer
   reject a flow base; rejected-base writes affect only flow/species columns.
   RANS coefficients are compressed separately about their cell means with a
   zero floor. For mean q and reconstructed value h < 0, the bound is
   `theta <= q/(q-h)`; the entire scalar polynomial is scaled by `0.9*theta`
   only when theta < 1. Two-equation models share one factor. SA uses the same
   rule only with `USE_NS_SA_NUT_REDUCED_ORDER`; otherwise it is untouched.

The scalar correction executes inside the existing coefficient write pass,
before flow's early exit, even when flow beta equals one. It introduces no
extra array traversal or halo exchange. It uses logical component views, so
reversed packed block ordering and separate SoA/AoSoA fields remain supported.

Reconstruction no longer derives scalar floors from density or a prescribed
reference. The setup policy now depends only on the model. Physical input
reference conversions and conservative omega mean floors retain their separate
contracts, with no guessed unit density. Species remain coupled to thermal PP;
temporal increment and mean-repair policies are unchanged.

## Existing correct paths

Cauchy's turbulent normal stress already projects all momentum components
along the physical normal and uses absolute fluid velocity for energy work.
Its conservative derivative has an existing oblique-normal finite-difference
test. No corresponding x-only flux correction is needed here.

Auxiliary mean gradients are explicit fields with source/time/storage/shape
validation and completed halo ownership. The hidden stale `uGradBuf` defect in
Euler's old `EvaluateDt` does not directly apply. This fix does not change the
chosen gradient limiting convention or add implicit reuse permissions.

## Focused checks

- Shared RANS source: independent SST inner/outer shear expectation, plus
  existing property, diffusion and rotation checks.
- Scalar policy: mean-based compression, tiny positive constants, and the
  separately compiled reduced-order SA header policy.
- Distributed reconstruction: SA and SST, mean and hierarchical flow bases,
  reversed packed AoS and split SoA/AoSoA storage. Scalar-negative, tiny-positive
  and flow-fallback cases check the two directions of independence. Existing
  thermal reconstruction and RANS implicit-update cases remain in the suite.
- Physical state input: continue checking scaled conservative reference values
  directly, rather than through obsolete reconstruction floor fields.

The paired `build-audit` directory uses canonical dependencies and eight
compiler workers; no shared source/build or cluster work is used.

Local validation passed **7/7 CTest entries** (3.68 seconds), with the default
two OpenMP threads per rank:

```sh
ctest --test-dir build-audit --output-on-failure \
  -R '^(euler_rans_model|dnds_cauchy_rans_state|dnds_cauchy_state_input|dnds_cauchy_rans_reconstruction_reduced|dnds_cauchy_rans_fv_np[124])$'
```

The reduced-order SA executable checks the header policy independently; it does
not claim a full solver build with that feature flag. No CUDA or production
flow campaign is part of this fix's validation.

## Historical Euler mean-gradient input

Euler's `EvaluateDt` computes/exchanges `uGradBufNoLim`
but reads the separate `uGradBuf` for turbulent viscosity. This mismatch originated
in `e5b00928cc97` (2025-06-15), “EvaluateDt now does not rely on uRec (for getting
gradient)”, and entered modern history through squash `90490801` (“Initial
Release (#1)”). Cauchy's explicit-current input avoids reproducing that stale
buffer behavior.
