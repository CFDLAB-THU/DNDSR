# Automatic A100 kernel selection

CUDA operator placement resolves `auto` selectors once, using the actual
device and equation/reconstruction shape. CPU arithmetic and scheduling are
unchanged. Kernel launches perform no hardware-name queries or tuning.

For the measured combined AoS, 3D P3 flow5 profile with quadrature order at most
five, A100 defaults are:

| Operator | Automatic selection |
| --- | --- |
| RHS | `ctaSegments`, deterministic segmented reduction, 128 threads by default |
| VR Jacobi | `fp64Pipelined`, neighbor-buffered SIMT, default 32-thread block per cell |
| Polynomial PP | Warp per cell, 512 threads per block |
| Diagonal inverse | `guardedLU`, retaining the condition guard and filtered SVD fallback |
| Linear Jacobi | Existing thread-per-cell, compact matrix storage |

The VR matrix representation remains independent: raw face-shared B/full or
triangular AInv and precomputed AInvB are not interchanged. No FP32, float-float,
tensor-core arithmetic, storage padding, reordering or overlap is enabled by
this policy. Those choices are separately configurable.

Explicit selectors retain their arithmetic/mapping meaning. Plan 61-6 bounds
the comparison-only selectors, including `"fp64MatrixShared"`, to their
single-grid flow5/19-mode precomputed non-LVR fixture. Other shapes use the
general `"fp64"` or production `"fp64Pipelined"` path; an explicit unavailable
comparison request fails. `"auto"` allows the measured A100 default.
`execution.kernels.rhsMapping`, `rhsReduction`,
and `polynomialPPThreads` override their respective automatic choices. Existing
`DNDS_CAUCHY_CUDA_MODAL_PP_*` environment overrides still take precedence.
`implicit.linear.cudaDiagonalInverse` accepts explicit `batchedSVD` or
`guardedLU`; `auto` selects guarded LU on A100 and batched SVD elsewhere.

Outside the measured A100 shape, FP64 RHS uses the same CTA mapping because it
is the sole consolidated mapping, not because every shape was benchmarked.
Reduced-precision flow5 comparisons use thread mapping. VR uses general FP64,
and PP uses 128 threads. Automatic LVR also uses
general FP64; an explicit `fp64Pipelined` LVR selection is supported separately.
Unsupported explicit
selections still fail; automatic selection never falls back to CPU.

`FiniteVolume::Settings()` on the placed plan exposes its concrete spatial
choices. The canonical input/resolved configuration retains the requested
`auto` intent; archived explicit benchmark configurations remain reproducible.
Already-running processes keep their original executable and configuration.

Fresh builds with `DNDS_USE_CUDA=ON` also default `DNDS_DIST_MT_USE_OMP=ON`.
An explicit cache value, including OFF, is preserved. Existing build caches do
not change implicitly. This controls CPU setup/host-kernel compilation; CUDA
kernels do not execute OpenMP threads.

## Plan 61-6 RHS source contract

The FP64 CTA-segment catalog uses one shared quadrature-point arithmetic body.
It supports only combined calorically perfect-gas flow5,
flow5+SA and flow5+2EQ, with no species or chemical source. The immutable
`RHSKernelParameters` must be validated before device publication; mixtures
cannot enter this profile by supplying another component width.

The host boundary projects combined AoS mean/incoming/output rows to static
5/6/7 extents and polynomial rows to C*M extents for M=2/5/9 in 2D and
M=3/9/19 in 3D. M=0 denotes non-VR constant/direct-gradient evaluation. Actual
field rows must be compact; padded input rows are rejected by the wrapper.
Direct-gradient and primitive-gradient views retain their appropriate fixed
widths. The typed views reach numerical leaves and shared staging without
decaying to a generic Array view. Dimension also specializes the derivative
arithmetic and indexing. Optional viscous inputs select a typed role on the
host, without another instantiation axis or an input copy.

One manifest generates 24 one-kernel CUDA TUs in the build directory. Block
width remains runtime; it does not multiply the catalog. The private common
body and CMake generator dependency contract are documented in
[kernel contracts](../architecture/CauchyKernelContracts.md#cuda-rhs-shape-catalog).
The local register cap keeps 512-thread launches within the register file and
does not constrain reduced-precision comparison TUs.
Cooperative reductions stay deterministic and retain complete-face selection,
tail-lane/barrier and scheduled-lifetime contracts. Polynomial PP retains its
runtime shape dispatch and numerical policy.
