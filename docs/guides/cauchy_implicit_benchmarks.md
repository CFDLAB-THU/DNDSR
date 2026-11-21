# Cauchy frozen-state benchmarks

Build the opt-in `cauchy_benchmark_implicit` target with tests enabled. It uses
the public DNDS/Geom/CFV/Cauchy facilities on a unit-square CGNS mesh, including
ordinary physical boundary labels (mapped to farfield) or unit-periodic labels.
Pass the mesh, repetition count, trial count, scheduled-access checking, overlap
and semi-matrix-free selection in that order. Two optional trailing switches
enable extra controller phases and restrict the run to polynomial multigrid.

```sh
OMP_NUM_THREADS=1 mpiexec -n 32 ./test/cpp/cauchy_benchmark_implicit \
  ../data/mesh/Uniform256.cgns 10 3 1 1 0
```

Run timings on an exclusive allocated node. Uniform256 has 65,536 cells, giving
2,048 cells per rank at np=32. The separate `cfv_benchmark_fv_facility` compares native
and legacy VR preparation/sweeps/WBAP on the same mesh, means and halo schedule.
Local small runs validate benchmark execution; they are not performance evidence.

`CAUCHY_PHASE_BENCH` lines report seconds per call, using a barrier outside
each batch and the maximum elapsed time among ranks. Import, topology, combined
FV preparation and ILU symbolic setup are separate one-shot measurements.
Diagonal inversion is timed once after a real Jacobian fill because repeated
calls otherwise measure cached no-ops. Numeric factorization is repeated;
its elapsed time includes copying native matrix blocks into the local factor.

`cauchy_benchmark_block_lu_comparison` isolates numeric block factorization or
triangular solve against Euler's actual `JacobianLocalLU<6>`, with identical
local graph, symbolic fill, matrix and RHS. Choose exactly one mode per timed
invocation: `factor`, `solve` (physical Array entry), or `logical-solve`
(EquationDofs-style component access). Symbolic preparation and MPI payloads
are excluded. Factor mode resets both numeric matrices each repetition.
Solutions and RHS generation are checked before and after each timing batch.
This synthetic six-component matrix benchmark does not measure a physical
Jacobian or distributed linear convergence.

### Observing CPU halo overlap

With Caliper enabled at build time and `DNDS_TIMELINE=1`, the queued CPU
callbacks mark work on the thread that executes it. `Cauchy/CPU/FluxRHS`,
`VolumeSource`, `RotationSource`, `LinearApply`, `Jacobi` (under the same
prefix), and `CFV/CPU/ReconstructionSweep` record their row selection as
an integer attribute at both boundaries. Values 0, 1 and 2 mean all, core
and halo-dependent. Auxiliary preparation records its own phase enumeration.

Compare a worker interval with MPI observations on the **same rank**.
A successful post and a later observed completion give an upper bound on
the time a request was outstanding. For a receive, a singleton MPI_Testall
returning false also gives a lower bound from posting to the start of that
test. A delayed peer send can make this overlap reproducible. Use the unsplit
execution as a control and compare final fields by persistent cell ID.
