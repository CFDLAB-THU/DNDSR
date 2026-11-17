# Pinned AmgX native cell blocks

`amgx-v2.5.0-native-blocks.patch` applies only to upstream revision
`cc1cebdbb32b14d33762d4ddabcb2e23c1669f47` (v2.5.0). Build using
`scripts/build_amgx.py`; rerunning accepts either a clean pin or exactly this
patch, and refuses other edits. The install manifest records source, patch and
library hashes. CMake verifies the selected library, including a cached path.

The audited DNDS path is single-device dDDI, AGGREGATION/SIZE_2, LOW_DEG,
BLOCK_JACOBI, V-cycle, NOSOLVER, fixed cycles/sweeps. Cell blocks remain native
5/6/7 throughout; no scalarization or padding. Scalar width one remains valid
for genuinely scalar generic DNDS systems. This is not a claim that arbitrary
AmgX smoothers/coarse solvers support six/seven: e.g. MULTICOLOR_DILU still has
its own restricted dispatch and is not selected here.

The patch fixes:

1. LOW_DEG and residual restriction lacked 6/7 dispatch. SIZE_2, prolongation,
   vector operations and cuSPARSE BSR actions already accept square blocks.
2. LOW_DEG's large-block kernel loaded 32 column indices but advanced by N*N
   and shuffled N*N indices. It now advances/shuffles 32 and loops over block
   entry tiles separately (two tiles for 6/7/8, four for 10). Separate diagonals
   now retain every tile and never reduce different matrix entries together.
3. Generic >5 Jacobi inversion allocated scratch by grid-limit count but indexed
   it per launched thread. It now allocates `numBlocks*512*tileCount²` values.
   Sixteen cooperating lanes use explicit subgroup barriers between the shared
   matrix phases; the active condition is uniform within each 16-lane group,
   including a partial final CTA. The >5 Schur-update parentheses also wrongly
   negated `A_ij` along with the product. The patch restores `A_ij - A_ik*A_kj`,
   matching the existing <=5 implementation. Native6/7 support requires this
   correction, not just adding dispatch cases.
4. The independent upstream host/GPU Galerkin test covers 6/7/8/10, more than
   32 fine and coarse neighbors, and both inline and separate diagonals. Its
   separate-diagonal comparison previously read A for both sides and only one
   coefficient; it now compares every dense coefficient with B. This oracle is
   included in the otherwise explicitly built `amgx_tests_launcher` target.

Focused validation (not performed merely by source review):

```sh
cmake --build <amgx-build> --target amgx_tests_launcher -j16
CUDA_VISIBLE_DEVICES=1 <amgx-build>/src/amgx_tests_launcher --mode dDDI AggregatesCoarseGeneratorTest
cmake --build <dndsr-build> --target dnds_test_amgx_preconditioner dnds_test_cauchy_linear_solver_cuda dnds_test_cauchy_config -j16
CUDA_VISIBLE_DEVICES=1 <dndsr-build>/test/cpp/dnds_test_amgx_preconditioner
CUDA_VISIBLE_DEVICES=1 compute-sanitizer --tool memcheck --error-exitcode=99 <dndsr-build>/test/cpp/dnds_test_amgx_preconditioner
```

The DNDS provider tests check native 5/6/7 multilevel residual contraction,
value refresh, simultaneous owner lifetimes, and a nonsymmetric dense-block
exact solve at 1/63/5003 rows for native6/7. The distributed Cauchy test checks
known solutions and exactly two stationary corrections, with no Krylov products
or tolerance-dependent early stopping. Use the normal MPI harness for that
last test; the provider memcheck is deliberately MPI-free.

A successful provider setup is not evidence of nonlinear CFD convergence.
The earlier SA run used scalar expansion and GMRES, so it is not a qualification
of this native-block, fixed-count preconditioner-only path. Actual SA operator
residuals and corrected case runs remain separate experimental evidence.


## Plan 61-6 extension

The current patch additionally qualifies native widths 2/3/4/5/6/7/8/9/19
(with scalar 1 and the existing width-10 reference), adds restricted dDDI
numeric attachment and per-level/allocation diagnostics. See the fork's
`doc/NativeStorage.md` and `docs/guides/amgx.md`. New grouped validation is
pending; historical 6/7 results above must not be credited to these new paths.
A dedicated `amgx-plan616` install protects existing reference artifacts.
The builder's `--source-tree` checks the public fork's compiled source against
the exact bundled patch without resetting or rewriting that checkout.
