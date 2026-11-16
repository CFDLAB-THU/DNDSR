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

`numerics.cfv.storage` selects the sweep representation and independent matrix,
inverse, vector and boundary payload layouts. In `precomputed` mode, a sweep
uses the previously assembled A-inverse-times-B maps. In `raw` mode, it first
accumulates the B products and mean/boundary terms, then applies the local
inverse. Both use the same functional and SVD cutoff, but reassociation can
change floating-point rounding.

For raw sweeps, `inverseTriangle` packs the validated symmetric local inverse.
`faceCoupling` stores one full B block per local face and transposes the
opposite incidence. A B block is not assumed symmetric. Periodic vector
transformations still apply in the corresponding incidence frame, including
self-periodic faces. `aosRowPadding` controls AoS pitch in scalar elements;
`allocationAlignmentBytes` independently requests base alignment. SoA uses
`rowPadding`; AoSoA uses a fixed tile width of 32.

The CUDA `fp64Tensor` and `fp64TensorWide` choices can evaluate raw sweeps on
devices with FP64 tensor cores: one matrix multiplication gathers B products,
and another applies the local inverse. The former limits the grid to the
historical mapping; the latter permits a wider grid. Kernel selection remains
explicit and unsupported combinations fail instead of silently switching.
The scalar/SIMT and tensor paths need not produce bitwise identical results.

## Interpreting measurements

Compare matched mesh, polynomial and quadrature choices, solver iterations,
precision, and placement. Report model/effective bandwidth based on useful
nonduplicated bytes separately from measured physical DRAM traffic. Include
capacity, alignment padding and host/device placement in memory comparisons.
Runtime allocation snapshots count registered allocations; provider-private,
allocator, driver and other-process memory are not implied by that total.
