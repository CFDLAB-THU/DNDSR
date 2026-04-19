# Serialization Usage Guide {#serialization_usage}

@tableofcontents

This guide explains how to checkpoint and restart solver state using
DNDSR's serialization system.  The key feature is **redistribution**:
a checkpoint written with 4 MPI ranks can be read back on 8 without
any special handling in the solver code.

For the internal design see @ref serialization.

## The Problem

A CFD solver periodically writes its solution to disk so that the
computation can resume after a crash or be continued on a different
machine.  Two difficulties arise:

1. **Parallel I/O**: each MPI rank owns a slice of the data.  Writing
   one file per rank is simple but creates thousands of files at scale.
   A single shared file via MPI-IO is cleaner.
2. **Partition independence**: the user may restart on a different number
   of ranks.  The checkpoint must store enough information to redistribute
   the data.

DNDSR solves both with a two-layer system:

- **`SerializerH5`** writes a single `.h5` file using MPI-parallel HDF5.
  Each array is stored as a contiguous global dataset with per-rank
  offset metadata (the `rank_offsets` companion dataset).
- **`ArrayPair::WriteSerialize`** stores an `origIndex` vector alongside
  the data -- a partition-independent global cell ID.  On read,
  `ReadSerializeRedistributed` uses this to pull each rank's cells
  regardless of how the file was partitioned.

## Writing a Checkpoint

The solver builds a serializer, opens a file, and writes its DOF array.
The `origIndex` vector enables redistribution on read.

(Source: `Euler/EulerSolver_PrintData.hxx:844` for the real implementation.)

```cpp
#include "DNDS/SerializerFactory.hpp"

void MyPDESolver::WriteCheckpoint(const std::string &path)
{
    // 1. Build the serializer.
    //    SerializerFactory reads the config to choose H5 or JSON
    //    and applies settings (chunking, deflate, collective I/O).
    auto ser = Serializer::SerializerFactory::BuildSerializer(
        "H5", mpi, /*collectiveRW=*/true);
    ser->OpenFile(path, /*read=*/false);

    // 2. Build a partition-independent cell ID.
    //    cell2cellOrig(i, 0) is a global ID that stays the same
    //    regardless of how the mesh is partitioned.  This is what
    //    makes redistribution possible.
    std::vector<index> origIdx(mesh->NumCell());
    for (index i = 0; i < mesh->NumCell(); i++)
        origIdx[i] = mesh->cell2cellOrig(i, 0);

    // 3. Write the DOF with origIndex.
    //    includePIG=false: don't store ghost pull indices (they are
    //    rebuilt from the mesh on restart).
    //    includeSon=false: don't store ghost data (pulled after read).
    u.WriteSerialize(ser, "u", origIdx,
                     /*includePIG=*/false, /*includeSon=*/false);

    ser->CloseFile();
}
```

**What goes into the H5 file** (under path `"u"`):

| Dataset | Content |
|---|---|
| `u/father/data` | The raw DOF values (real vector) |
| `u/father/size` | Per-rank row count |
| `u/father/data\:\:rank_offsets` | Cumulative offsets for each rank's data slice |
| `u/origIndex` | Partition-independent cell IDs |
| `u/redistributable` | Flag = 1 (enables redistribution on read) |

## Reading a Checkpoint (Same Rank Count)

When the rank count matches the checkpoint, the read is straightforward:

```cpp
void MyPDESolver::ReadCheckpoint(const std::string &path)
{
    auto ser = std::make_shared<Serializer::SerializerH5>(mpi);
    ser->OpenFile(path, /*read=*/true);

    u.ReadSerialize(ser, "u", /*PIG=*/false, /*son=*/false);

    ser->CloseFile();

    // After reading, ghost data is stale -- pull it.
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();
}
```

## Reading on a Different Rank Count (Redistribution)

This is the main feature.  The solver code is nearly identical; only
the read call changes from `ReadSerialize` to `ReadSerializeRedistributed`.

(Source: `Euler/EulerSolver_PrintData.hxx:942` for the real implementation;
`DNDS/ArrayPair.hpp:479` for the redistribution logic.)

```cpp
void MyPDESolver::ReadCheckpointRedistributed(const std::string &path)
{
    auto ser = std::make_shared<Serializer::SerializerH5>(mpi);
    ser->OpenFile(path, /*read=*/true);

    // Build newOrigIndex from the CURRENT mesh partition.
    // This tells the reader "I need cells with these global IDs."
    std::vector<index> newOrigIdx(mesh->NumCell());
    for (index i = 0; i < mesh->NumCell(); i++)
        newOrigIdx[i] = mesh->cell2cellOrig(i, 0);

    // ReadSerializeRedistributed handles the np mismatch.
    u.ReadSerializeRedistributed(ser, "u", newOrigIdx);

    ser->CloseFile();

    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();
}
```

### How Redistribution Works Internally

When the file was written with `np_old = 4` and is read with `np_new = 8`:

1. **Even-split read**: each of the 8 ranks reads `~N_global / 8` rows
   of both the data and the `origIndex` vector.  This is a balanced but
   incorrect distribution -- the rows don't correspond to this rank's
   mesh cells yet.  (See `ParArray::ReadSerializer` at
   `DNDS/ArrayTransformer.hpp:165`.)

2. **Rendezvous pull**: `RedistributeArrayWithTransformer` (at
   `DNDS/ArrayRedistributor.hpp:235`) creates a temporary
   `ArrayTransformer`.  Each rank announces which `origIndex` values it
   needs (from `newOrigIdx`).  The transformer's ghost-pull mechanism
   fetches the corresponding rows from whichever rank holds them after
   the even-split read.

3. **Reorder**: the pulled data is placed into the output array in the
   order dictated by `newOrigIdx`.

This works because `origIndex` is partition-independent -- it does not
change between runs or between different rank counts.

## Cross-Solver Restart

DNDSR supports reading a checkpoint from a *different* solver variant.
For example, reading an Euler (5-variable) checkpoint into an SA
(6-variable) solver.  The key is `ReadSerializerMeta`, which peeks at
the stored dimensions without reading data.

(Source: `Euler/EulerSolver_PrintData.hxx:1001`.)

```cpp
// Peek at the stored variable count.
auto probe = std::make_shared<ParArray<real, DynamicSize>>(mpi);
Serializer::ArrayGlobalOffset off, doff;
probe->ReadSerializerMeta(ser, "u/father", off);
int storedNVars = probe->RowSize();   // e.g. 5 for Euler

// Allocate a temporary buffer with the stored dimensions.
TDof readBuf;
vfv->BuildUDof(readBuf, storedNVars);
readBuf.ReadSerializeRedistributed(ser, "u", newOrigIdx);

// Copy the common variables into the solver's DOF.
for (index i = 0; i < mesh->NumCell(); i++)
    for (int v = 0; v < std::min(storedNVars, myNVars); v++)
        u(i, v) = readBuf(i, v);

// Initialize the extra variables (e.g. nu_tilde for SA) to defaults.
```

## Low-Level Array Serialization

For writing/reading individual arrays without the ArrayPair wrapper
(e.g. a custom output not tied to solver DOFs):

```cpp
// Write
auto ser = std::make_shared<Serializer::SerializerH5>(mpi);
ser->OpenFile("output.h5", false);
Serializer::ArrayGlobalOffset off = Serializer::ArrayGlobalOffset_Parts;
myParArray.WriteSerializer(ser, "myData", off);
ser->CloseFile();

// Read
ser->OpenFile("output.h5", true);
Serializer::ArrayGlobalOffset readOff = Serializer::ArrayGlobalOffset_Unknown;
Serializer::ArrayGlobalOffset dataOff = Serializer::ArrayGlobalOffset_Unknown;
myParArray.ReadSerializer(ser, "myData", readOff, dataOff);
ser->CloseFile();
```

`ArrayGlobalOffset_Parts` tells the writer to compute per-rank offsets
via `MPI_Scan`.  `ArrayGlobalOffset_Unknown` tells the reader to
auto-detect this rank's slice from the stored `rank_offsets` companion.
See `DNDS/SerializerBase.hpp:20` for all offset modes.
