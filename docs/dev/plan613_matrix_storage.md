# Experimental matrix storage for FV and VR

These choices are experimental. The default FV block representation and
precomputed VR Jacobi representation are unchanged. All layout selections
change physical storage; they do not change the equation unknown ordering.

## FV linear blocks

`implicit.linear.matrixStorage` separately selects the diagonal, inverse and
off-diagonal matrix payloads and the native sparse-index width. Each payload
supports `aos`, `soa` and `aosoa32`. The AoS `rowStridePaddingMultiple` is a
multiple of scalar elements, not bytes; `allocationAlignmentBytes` controls
the allocation base independently. These settings may increase capacity without
changing the logical block dimensions. Matrix actions must use the matching
layout-aware view. Compact AoS retains the static CPU map path.

The single-device AmgX adapter requires native compact row-major AoS diagonal
and off-diagonal blocks and native int32 CSR indices. The inverse layout is
independent. Incompatible storage is rejected; DNDS does not repack the matrix.
See [the AmgX guide](../guides/amgx.md) for provider-specific restrictions.

## Variational reconstruction

`numerics.cfv.storage` selects the sweep representation. VR matrix, inverse,
mean and boundary maps use contiguous AoS rows; losing experimental VR-map
SoA/AoSoA choices were retired in the stage-once follow-up, with their evidence
retained in plan history. Runtime/DOF and FV matrix layout options are unchanged.
In `precomputed` mode, a sweep
uses the previously assembled A-inverse-times-B maps. In `raw` mode, it first
accumulates the B products and mean/boundary terms, then applies the local
inverse. Both use the same functional and SVD cutoff, but reassociation can
change floating-point rounding.

For raw sweeps, `inverseTriangle` packs the validated symmetric local inverse.
`faceCoupling` stores one full B block per local face and transposes the
opposite incidence. A B block is not assumed symmetric. Periodic vector
transformations still apply in the corresponding incidence frame, including
self-periodic faces. `aosRowPadding` controls AoS pitch in scalar elements;
`allocationAlignmentBytes` independently requests base alignment.

The retained CUDA families are the general `fp64` sweep, existing
`fp64MatrixShared`, neighbor-buffered SIMT `fp64Pipelined`, and neighbor-buffered
MMA `fp64TensorAsync`. `floatFloatPipelined` adds complete float-float arithmetic
to the same SIMT body. General quantized-map `fp32`, `fp32Accum`,
`fp32FloatFloatAccum` and `tf32` policies remain separate; the latter float-float
policy has already lost matrix precision and is not the new experiment.

`numerics.cfv.cudaSweep` controls only `threads` and `gridBlocks`. Zero threads
select 32 for block-team SIMT and 128 for static warp-team MMA (32 for dynamic component counts); zero blocks selects
a wide grid. Both pipelines stage a whole neighbor matrix and every component
coefficient once, using two cp.async slots. Full-cell-row buffering, warp-team
SIMT, synchronous staged alternatives and duplicate wide/static/tensor selectors
were retired after measurements; their configurations belong to their historical
source revision. Ordinary
FP64/FP32 retain general component support. The staged families specialize
5/6/7 components and use shared accumulation for dynamic counts; all five
modal sizes 2/3/5/9/19 cover P1/P2/P3 in 2D and 3D. Dynamic counts are limited
by checked shared-memory capacity, rather than a fixed seven-component array.
FP64 MMA requires supporting hardware.

`floatFloatPipelined` requires `DNDS_USE_CCCL_COMPUTE` and the pinned CCCL
`cuda::fp32mp2` provider (`DNDS::Math::real_f32f32`). ALL numerical sweep
arithmetic uses the pair: matrix products, reductions, periodic rotation,
mean/BC terms, raw inverse application and relaxation. FP64 is used only at
input/output conversion. Matrix/state storage stays FP64; inputs are converted
directly into both limbs, without first rounding to a single float. Shared
staging transports FP64 bytes then converts each matrix/coefficient once.
The narrower exponent range can overflow. A device-only result check sets the
sticky workspace status before guarded commit; no host drain or FP64 fallback
is inserted. This changes arithmetic precision, not the reconstruction weights.

Quantized FP32-map placement also supports `fp32Pipelined` (full FP32 SIMT
arithmetic) and `tf32Pipelined` (TF32-rounded tensor products and FP32 update
arithmetic), through the same neighbor-buffered implementation. Raw matrix
representations remain FP64-only. TF32 has a separately looser accuracy budget;
it is not an ordinary-FP32 alias. The general FP32/TF32 policies remain
available as numerical/comparison anchors. Matrix, state and output component
meaning is identical across those choices. All use the common queued CFV
operator with explicit placement/capability checks.

The shared staged numerical implementation is in
`Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh`. The paired `_FP64`,
`_FloatFloat` and `_FP32` translation units instantiate mode/component variants
independently; the small `.cu` wrapper checks arguments and selects a provider.
The general sweep uses the same mode/component dispatch discipline via
`ReconstructionDevice_General.cuh` and its precision-specific translation units.
Kernels receive only numeric views and POD parameters. Host dispatch descriptors
never become device kernel arguments or asynchronous owners.

Raw sweeps gather B products then apply the local inverse, loaded once per
cell and reused across variables. Periodic component transformations and face
transpose conventions are preserved. Distinct inputs/outputs, scheduled leases
and immutable plan retention remain responsibilities of the common queued
operator. Numerical kernels contain no allocation, lease acquisition, MPI or
host synchronization. Unsupported device/build/shape combinations reject.

## Interpreting measurements

Compare matched mesh, polynomial and quadrature choices, solver iterations,
precision, and placement. Report model/effective bandwidth based on useful
nonduplicated bytes separately from measured physical DRAM traffic. Include
capacity, alignment padding and host/device placement in memory comparisons.
Runtime allocation snapshots count registered allocations; provider-private,
allocator, driver and other-process memory are not implied by that total.
