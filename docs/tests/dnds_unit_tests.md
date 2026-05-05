# DNDS Core Unit Tests {#dnds_unit_tests}

@tableofcontents

@section dnds_tests_overview Overview

The DNDSR project uses the [doctest](https://github.com/doctest/doctest)
(v2.4.11) framework for C++ unit testing.  Test sources live under
`test/cpp/DNDS/` and are built only when `DNDS_BUILD_TESTS=ON`.

Every MPI-aware test is registered with CTest at four process counts
(np = 1, 2, 4, 8) so that correctness under parallel execution is
verified automatically.

@subsection test_build Building and Running

```sh
# Configure with tests
cmake -B build -DDNDS_BUILD_TESTS=ON

# Build all test executables at once
cmake --build build -t dnds_unit_tests -j8

# Run every registered test (serial + MPI np=1,2,4,8)
ctest --test-dir build -R dnds_ --output-on-failure

# Run a single test suite
ctest --test-dir build -R dnds_mpi_np2 --output-on-failure

# Run a single executable directly
mpirun -np 4 ./build/test/cpp/dnds_test_array_transformer
```

@subsection test_naming Naming Conventions

| CMake target              | CTest names                              | Source file                        |
|---------------------------|------------------------------------------|------------------------------------|
| `dnds_test_array`         | `dnds_array`                             | test_Array.cpp                     |
| `dnds_test_mpi`           | `dnds_mpi_np{1,2,4}`                     | test_MPI.cpp                       |
| `dnds_test_array_transformer` | `dnds_array_transformer_np{1,2,4}`   | test_ArrayTransformer.cpp          |
| `dnds_test_array_derived` | `dnds_array_derived_np{1,2,4}`           | test_ArrayDerived.cpp              |
| `dnds_test_array_dof`     | `dnds_array_dof_np{1,2,4}`               | test_ArrayDOF.cpp                  |
| `dnds_test_index_mapping` | `dnds_index_mapping_np{1,2,4}`           | test_IndexMapping.cpp              |
| `dnds_test_serializer`    | `dnds_serializer_np{1,2,4}`              | test_Serializer.cpp                |
| `dnds_test_permutation_transfer` | `dnds_permutation_transfer_np{1,2,4}` | test_PermutationTransfer.cpp   |

@subsection test_note Note on POSIX `index()` Ambiguity

Because doctest includes `<cstring>` which transitively pulls in the POSIX
`index()` function from `<strings.h>`, the bare name `index` is ambiguous
when `using namespace DNDS;` is active.  All test files therefore qualify
the DNDS type aliases as `DNDS::index`, `DNDS::real`, and `DNDS::rowsize`
in declarations.

---

@section test_array Array Tests (test_Array.cpp)

@see test_Array.cpp

Serial-only tests covering every @ref DNDS::Array data layout:

- **TABLE_StaticFixed** — `Array<real, 3>`: compile-time fixed row
  size.  Validates `Size()`, `RowSize()`, element read/write via
  `operator()` and `operator[]`, `DataSize()`, `DataSizeBytes()`.
- **TABLE_Fixed** — `Array<real, DynamicSize>`: runtime-uniform row
  size.
- **TABLE_StaticMax** — `Array<real, NonUniformSize, 4>`: per-row
  variable sizes up to a compile-time maximum.
- **TABLE_Max** — `Array<real, NonUniformSize, DynamicSize>`:
  per-row variable sizes up to a runtime maximum.
- **CSR** — `Array<real, NonUniformSize>`: compressed sparse row.
  Tests lambda-based `Resize`, `Compress`/`Decompress` round-trips,
  `ResizeRow` in decompressed mode.
- **Clone and copy** — `clone()`, `CopyData()`, copy constructor,
  `SwapData()` independence checks.
- **Edge cases** — zero-size arrays, single-element arrays, CSR
  rows of length zero.
- **Serialization** — JSON round-trip for TABLE_Fixed, CSR, and
  TABLE_StaticFixed via @ref DNDS::Serializer::SerializerJSON.
- **Signature** — `GetArraySignature()`, `ParseArraySignatureTuple()`,
  `ArraySignatureIsCompatible()`.
- **Hash** — identical arrays produce equal hashes; modified arrays
  differ.

---

@section test_mpi MPI Wrapper Tests (test_MPI.cpp)

@see test_MPI.cpp

Run at np = 1, 2, 4, 8.  All expected values are formulated in
terms of `mpi.size` so they hold for any rank count.

- **MPIInfo basics** — `setWorld()`, field validity, equality.
- **Allreduce** — `MPI_SUM` and `MPI_MAX` for `real` and `index`;
  `AllreduceOneReal`, `AllreduceOneIndex`.
- **Scan** — inclusive prefix sum.
- **Allgather** — single and multi-element.
- **Bcast** — from rank 0 and from last rank.
- **Barrier** — returns `MPI_SUCCESS`.
- **BasicType_To_MPIIntType** — scalar types, `std::array`,
  `Eigen::Matrix`.
- **CommStrategy** — get/set `HIndexed`/`InSituPack`.
- **Alltoall** — single and multi-element exchange.

---

@section test_trans ArrayTransformer Tests (test_ArrayTransformer.cpp)

@see test_ArrayTransformer.cpp

Exercises ghost (halo) communication at np = 1, 2, 4, 8.

- **ParArray basics** — `createGlobalMapping()`, `globalSize()`,
  `AssertConsistent()`.
- **Pull – TABLE_StaticFixed** — 100 elements/rank, pulls
  first 5 from each remote rank, verifies ghost data.
- **Pull – TABLE_Fixed** — complete replication pull.
- **Pull – CSR** — variable row sizes, verifies both data
  and row sizes in ghosts.
- **Pull – std::array elements** — `std::array<real,9>` as
  the element type.
- **Persistent pull** — `initPersistentPull` / `startPersistentPull` /
  `waitPersistentPull`, then modify father and re-pull.
- **BorrowGGIndexing** — second array shares the ghost mapping of
  the first, independently pulls correct data.
- **Push** — write to son, `pushOnce()`, verify father received
  values.

---

@section test_derived ArrayDerived Tests (test_ArrayDerived.cpp)

@see test_ArrayDerived.cpp

Covers every derived array type at np = 1, 2, 4, 8.

- **ArrayAdjacency** — basics (resize, row resize, compress,
  `operator[]`, `rowPtr()`), ghost communication, clone, fixed-size
  variant.
- **ArrayEigenVector** — basics (static and dynamic sizes), ghost
  communication.
- **ArrayEigenMatrix** — static sizes, dynamic sizes, NonUniform
  rows, ghost communication.
- **ArrayEigenMatrixBatch** — `InitializeWriteRow`, `BatchSize`,
  `operator()`, ghost communication.
- **ArrayEigenUniMatrixBatch** — static and dynamic sizes,
  `ResizeBatch`.

---

@section test_dof ArrayDOF Tests (test_ArrayDOF.cpp)

@see test_ArrayDOF.cpp

Tests every vector-space operation on @ref DNDS::ArrayDof at
np = 1, 2, 4, 8.  All MPI-global reductions (`norm2`, `dot`,
`min`, `max`, `sum`, `componentWiseNorm1`) use rank-aware expected
values.

- `setConstant` (scalar and matrix)
- `operator+=` (scalar, array, matrix)
- `operator-=`, `operator*=` (scalar, element-wise, matrix)
- `operator/=`
- `addTo`
- `norm2`, `norm2(other)` (L2 distance)
- `dot`
- `min`, `max`, `sum`
- `componentWiseNorm1`, `componentWiseNorm1(other)`
- `operator=` (value copy), `clone` (deep copy)
- scalar-array multiply (`ArrayDof<N,1> *= ArrayDof<1,1>`)
- identity: `dot(x, x) == norm2(x)^2`

---

@section test_idx IndexMapping Tests (test_IndexMapping.cpp)

@see test_IndexMapping.cpp

Tests global/local index mapping at np = 1, 2, 4, 8.

- **GlobalOffsetsMapping** — uniform distribution, non-uniform
  distribution, `search()` for first/last/middle/out-of-range indices.
- **OffsetAscendIndexMapping** — pull-based construction,
  `searchInMain`, `searchInGhost`, `searchInAllGhost`, `search`,
  `search_indexAppend`, `operator()` reverse mapping, empty ghost set.

---

@section test_serial Serializer Tests (test_Serializer.cpp)

@see test_Serializer.cpp

Round-trip write/read verification at np = 1, 2, 4, 8.

- **SerializerJSON scalar** — `WriteInt`/`ReadInt`,
  `WriteIndex`/`ReadIndex`, `WriteReal`/`ReadReal`,
  `WriteString`/`ReadString`.
- **SerializerJSON vector** — `WriteRealVector`,
  `WriteIndexVector`, `WriteRowsizeVector`.
- **SerializerJSON uint8** — with and without base64 codec.
- **SerializerJSON paths** — `CreatePath`, `GoToPath`,
  `GetCurrentPath`, `ListCurrentPath`.
- **SerializerJSON pointer dedup** — shared pointer written twice
  resolves to same object on read.
- **SerializerH5 scalar** — `WriteInt`/`ReadInt`,
  `WriteIndex`/`ReadIndex`, `WriteReal`/`ReadReal`,
  `WriteString`/`ReadString` (HDF5 attributes).
- **SerializerH5 vector** — `WriteRealVector`, `WriteIndexVector`
  with explicit `ArrayGlobalOffset`; read with `ArrayGlobalOffset_Unknown`
  auto-detection from the `rank_offsets` companion dataset.
- **SerializerH5 distributed vector** — non-uniform per-rank sizes,
  read with both `ArrayGlobalOffset_Unknown` and explicit offset.
- **SerializerH5 uint8** — two-pass read (nullptr size query, then
  actual read with offset from pass 1).
- **SerializerH5 paths** — `GoToPath`, `GetCurrentPath`,
  `ListCurrentPath` (groups materialized by writing content).
- **SerializerH5 string** — fixed-length HDF5 string attributes.

---

@section test_permutation_transfer PermutationTransfer Tests (test_PermutationTransfer.cpp)

@see test_PermutationTransfer.cpp

Tests for @ref DNDS::PermutationTransfer, the MPI primitive underlying the
distributed entity reordering framework (see
@ref distributed_reorder_design "Distributed Reorder Design" for the
full architecture). Runs at np = 1, 2, 4, 8.

@subsection pt_construction Construction

- **`fromLocalPermutation` reverse** — builds from an `old2new` local
  permutation (reverse), verifies `isLocalOnly == true`,
  `newGlobalIndices` reflect the permutation, and `transferRows` places
  source data at the permuted slot.
- **`fromLocalPermutation` identity** — no-op permutation; data unchanged.
- **`fromPartition` all-local** — partition where every entity stays on
  its current rank; validates `isLocalOnly == true` via `MPI_Allreduce`.
- **`fromPartition` round-robin** — cross-rank partition (`i % size`);
  validates counts received per rank and that all tags arrive uniquely.

@subsection pt_lookup buildLookup and resolve

- **`buildLookup` resolve** — construct lookup from a transfer;
  `resolve()` for local globals returns correct new globals;
  `resolve(UnInitIndex)` passes through.
- **Distributed `buildLookup` cross-rank** — with non-empty `pullSet`,
  `resolve()` correctly maps off-rank old globals to their new globals
  after ghost-pull.

@subsection pt_transfer transferRows

- **CSR local permutation** — variable-row-size CSR array, reverse
  permutation preserves every row's size and contents at the permuted
  slot.
- **Distributed transfer tag tracking** — each entry tagged with a
  unique `TAG_BASE + globalIdx` sentinel; after round-robin transfer,
  all received tags are within the valid range, unique on each rank,
  and the total count across ranks matches the original global size.
