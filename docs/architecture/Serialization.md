# Serialization {#serialization}

This page describes the serialization subsystem used for HDF5 checkpoint
I/O and restart redistribution across different MPI partition counts.

## Overview

DNDSR serializes `ArrayPair` data (father + optional son/ghost arrays)
through a layered design:

| Layer              | Header                | Responsibility                                    |
|--------------------|-----------------------|---------------------------------------------------|
| `SerializerBase`   | `SerializerBase.hpp`  | Abstract read/write interface for scalars, vectors, byte arrays. |
| `SerializerH5`     | `SerializerH5.hpp/cpp`| MPI-parallel HDF5 implementation (collective I/O). |
| `SerializerJSON`   | `SerializerJSON.hpp`  | Per-rank JSON implementation (no MPI coordination). |
| `Array`            | `Array.hpp`           | Reads/writes a single array (metadata, structure, data). |
| `ParArray`         | `ArrayTransformer.hpp`| MPI-aware wrapper: resolves EvenSplit, CSR global offsets. |
| `ArrayPair`        | `ArrayPair.hpp`       | Father-son pair: WriteSerialize, ReadSerialize, ReadSerializeRedistributed. |
| `ArrayRedistributor` | `ArrayRedistributor.hpp` | Rendezvous redistribution via ArrayTransformer. |

## Serializer interface

### Offset modes

`ArrayGlobalOffset` describes a rank's portion of a global dataset:

| Sentinel             | Meaning                                                    |
|----------------------|------------------------------------------------------------|
| `Unknown`            | Auto-detect from `::rank_offsets` dataset in the H5 file.  |
| `Parts`              | Compute offset via `MPI_Scan` over local sizes at write time. |
| `One`                | Rank 0 writes/reads the entire dataset; others write/read nothing. |
| `EvenSplit`          | Read only: each rank reads `~nGlobal/nRanks` rows.        |
| `isDist()` (offset >= 0) | Explicit `{localSize, globalStart}`.                  |

`EvenSplit` is resolved by `ParArray::ReadSerializer` into an `isDist()` offset
before calling `Array::ReadSerializer`. `Array::ReadSerializer` asserts that it
never receives `EvenSplit` directly.

### Collective semantics (HDF5)

All `Read*Vector`, `ReadShared*Vector`, `ReadUint8Array`, and their `Write`
counterparts are **MPI-collective** in `SerializerH5`. Every rank must call
them in the same order, even when its local element count is 0. Failing to
participate causes a hang.

### Two-pass read pattern

`SerializerH5` reads vectors in two passes internally:

1. **Pass 1** (`buf == nullptr`): queries dataset size and resolves the offset.
2. **Pass 2** (`buf != nullptr`): performs the collective `H5Dread`.

The `Read*Vector` and `ReadShared*Vector` methods handle both passes internally
and are single-call for the user.

`ReadUint8Array` exposes the two-pass pattern to the caller: the first call
with `data == nullptr` returns the size; the second call reads data.

### Zero-size partitions

When `nGlobal < nRanks` (e.g., 5 elements across 8 ranks), `EvenSplitRange`
assigns 0 rows to some ranks. This is valid and handled throughout the stack:

- **`ReadDataVector`**: accepts `size == 0` in the `isDist()` second-pass
  branch. The `H5_ReadDataset` call proceeds with a 0-count hyperslab
  selection (selects nothing, but the rank participates in the collective).

- **Callers** (`ReadIndexVector`, `ReadRealVector`, etc.): when `size == 0`,
  `std::vector<>::data()` may return nullptr on an empty vector. A nullptr
  `buf` would skip the `H5Dread` block (guarded by `if (buf != nullptr)`)
  and hang other ranks. Each caller passes a dummy stack pointer when
  `size == 0`:
  ```cpp
  index dummy;
  ReadDataVector<index>(name, size == 0 ? &dummy : v.data(), ...);
  ```

- **`ReadUint8Array`** exposes the two-pass pattern to the caller directly.
  When the queried size is 0, the caller must pass a non-null pointer on the
  second call. `Array::__ReadSerializerData`'s `treatAsBytes` lambda does this:
  ```cpp
  uint8_t dummy;
  serializerP->ReadUint8Array("data", bufferSize == 0 ? &dummy : (uint8_t*)_data.data(), ...);
  ```

## Array serialization

### Write path

`ParArray::WriteSerializer` (called by `ArrayPair::WriteSerialize`):

1. Delegates to `Array::WriteSerializer` for metadata, structural data, and the
   flat data buffer.
2. Writes `sizeGlobal` (sum of all ranks' sizes) as a scalar.
3. For CSR arrays: computes global data offsets via `MPI_Scan` and writes
   `pRowStart` in global data coordinates as a contiguous `(nRowsGlobal + 1)`
   dataset. Non-last ranks write `nRows` entries; last rank writes `nRows + 1`.

### Read path (same partition)

`ParArray::ReadSerializer` with `Unknown` offset:

1. Reads per-rank size from the `size` dataset (auto-detected via `::rank_offsets`).
2. For CSR: reads per-rank size, computes row offset via `MPI_Scan`, resolves to
   `isDist()`.
3. Delegates to `Array::ReadSerializer`.

### Read path (different partition / EvenSplit)

`ParArray::ReadSerializer` with `EvenSplit` offset:

1. Reads `sizeGlobal` from the file.
2. Computes `EvenSplitRange(rank, nRanks, sizeGlobal)` to get
   `{localRows, globalRowStart}`.
3. Resolves to `isDist()` and delegates to `Array::ReadSerializer`.

Some ranks may get `localRows == 0`. The read proceeds with 0-count hyperslab
selections in all collective HDF5 calls.

## ArrayPair serialization

### WriteSerialize

Writes under a sub-path `name`:

| Dataset              | Content                                          |
|----------------------|--------------------------------------------------|
| `MPIRank`            | Per-rank serializer only.                        |
| `MPISize`            | Number of MPI ranks at write time.               |
| `father/`            | Father array via `ParArray::WriteSerializer`.     |
| `son/`               | Son (ghost) array, if `includeSon`.              |
| `pullingIndexGlobal` | Ghost pull indices, if `includePIG`.             |

The origIndex overload additionally writes:

| Dataset              | Content                                          |
|----------------------|--------------------------------------------------|
| `origIndex`          | Partition-independent key per row (e.g., CGNS cell index). |
| `redistributable`    | Integer flag (1).                                |

### ReadSerialize

Reads data written by `WriteSerialize` with the **same** MPI size. Resizes
father (and optionally son) arrays internally. If `includePIG` is true, the
caller must call `trans.createMPITypes()` afterward.

### ReadSerializeRedistributed

Handles three cases:

1. **No origIndex, same np**: falls back to `ReadSerialize`.
2. **Has origIndex, same np**: reads father normally, reads origIndex, then
   redistributes via `RedistributeArrayWithTransformer`.
3. **Has origIndex, different np**: reads father and origIndex via EvenSplit,
   then redistributes via `RedistributeArrayWithTransformer`.

In case 3, the redistribution uses a rendezvous pattern
(`BuildRedistributionPullingIndex`) with three rounds of `MPI_Alltoallv` to
build a directory mapping `origIdx -> globalReadIdx`, then an
`ArrayTransformer` pull to move data to the correct ranks.

Ranks with 0 rows from EvenSplit participate in all collective calls with
empty arrays and empty Alltoallv buffers.

## Restart redistribution (Euler solver)

`EulerSolver::PrintRestart` writes checkpoint data with origIndex (from
`cell2cellOrig`) for H5 serializers. `EulerSolver::ReadRestart` calls
`ReadSerializeRedistributed` to load the data, handling both same-np and
cross-np restarts transparently.

`ReadRestartOtherSolver` peeks the DOF count (`nVars`) from the file,
constructs a temporary ArrayPair with matching layout, reads via
`ReadSerializeRedistributed`, and copies the data.
