# Shared CSR row starts: interface and serialization follow-up

Date: 2026-09-28. Branch: `fix/dnds-geom-audit-20260926`.
Baseline: `2f831618d79fcad5b826194b8ddc62d457f7c810`, including `f93f7966`.
This follow-up implements the explicitly agreed shared row-start API. It does
not modify `dev/harry` or change generic vector slicing semantics.

## 1. Const iterator interface

Previously, const iterator methods inferred a mutable pointer:

```cpp
auto begin() const { return host_ptr; } // T*, not const T*
auto cbegin() const { return host_ptr; }
```

They now return `const T*`; `end()` and `cend()` receive the same correction.
Mutable `begin()/end()` continue to return `T*`. This makes const-pointee access
meaningfully read-only without changing storage or iterator complexity.

Source: [Vector.hpp:300](../../../src/DNDS/Vector.hpp#L300).

## 2. Public CSR structural accessor

Before:

```cpp
t_pRowStart getRowStart() { return _pRowStart; }
```

After:

```cpp
ssp<const t_RowStart> getRowStart() const { return _pRowStart; }
```

The returned handle retains ownership, but cannot change offsets. Resetting a
caller's copy cannot rebind the array's member. Internal array code retains its
mutable structural handle. CSR decompression releases only that array's handle;
recompression/direct structural resize constructs new offsets. No general
copy-on-write mechanism is introduced.

The Python binding uses pybind11's const-buffer overload. Its existing read-only
view behavior is preserved. Keeping the array object alive is not equivalent to
retaining an old allocation after structural replacement; that separate Python
view lifetime safety issue is not claimed fixed here.

Sources: [Array.hpp:146](../../../src/DNDS/Array.hpp#L146),
[Array_bind.hpp:126](../../../src/DNDS/Array_bind.hpp#L126).

## 3. Explicit CSR read/write API

The serializer now exposes specialized entry points alongside unchanged generic
vector APIs:

```cpp
void WriteSharedRowStartVector(
    const std::string &name,
    const ssp<const host_device_vector<index>> &local,
    ArrayGlobalOffset data);

ArrayGlobalOffset ReadSharedRowStartVector(
    const std::string &name,
    ssp<host_device_vector<index>> &local,
    ArrayGlobalOffset rows);
```

`rows` is a resolved `{local row count, global row start}`. ParArray still resolves
Unknown/EvenSplit requests before invoking this API. JSON requires the full local
row count and row start zero. `data` describes flat array elements, not row-start
dataset entries; JSON uses Unknown. HDF5 callers supply the contiguous flat-data
region computed by ParArray's prefix sum.

### Write: retain original in-memory identity

The old ParArray path constructed a fresh `prsGlobal` shared pointer for every
write, then asked the ordinary serializer to deduplicate that temporary pointer:

```cpp
auto prsGlobal = std::make_shared<host_device_vector<index>>(nWrite);
prsGlobal->at(i) = _pRowStart->at(i) + globalDataStart;
serializerP->WriteSharedIndexVector("pRowStart", prsGlobal, Parts);
```

Two arrays sharing `_pRowStart` therefore acquired independent stored datasets.

The new API registers the original local vector and retains its owning handle
for the session. On first HDF5 encoding, a temporary still converts local starts
to global starts; it is never used as the sharing identity and never modifies
the live Array. Subsequent equivalent writes emit a reference to the original
dataset. Non-last ranks omit their redundant terminal offset; the last rank
includes it, preserving the existing contiguous global file representation.

All ranks must both have a hit and agree on the same reference path before
writing an alias. Mixed local identities or differing candidate paths cause a
new collective dataset write, not inconsistent per-rank reference metadata.
Reusing one pointer with changed encoding parameters is rejected collectively.
Callers must not modify source structural values during the writer session;
same-size direct element mutations are not detected by content hashing.

### Read: normalize once and retain the original data region

`f93f7966` protected the global read cache by copying before normalization:

```cpp
index base = _pRowStart->at(0);
_pRowStart = std::make_shared<t_RowStart>(*_pRowStart);
// subtract base from each entry
```

This was correct for values but lost structural sharing. The new specialized
read cache owns a decoded tuple:

```cpp
struct RowStartReadEntry
{
    ArrayGlobalOffset rows;
    ArrayGlobalOffset data;
    ssp<RowStartVector> local;
};
```

For file values `[20,23,27]`, the first read saves `{count=7,start=20}`, normalizes
its newly read allocation to `[0,3,7]`, and publishes both. Later equivalent reads
receive exactly the same local pointer and the saved data region. No per-array
normalization copy is needed; the original base is never recovered from an
already-normalized vector.

The specialized cache is separate from generic raw-vector caches. Ordinary
`ReadSharedIndexVector` continues returning global file values. Explicitly using
both APIs therefore creates two representations rather than mutating a raw
result behind its caller.

### Session slicing and MPI participation

All aliases resolve to their canonical stored path before lookup. The first
specialized read binds that path to one resolved row slice per rank. A conflicting
slice on any rank throws on all ranks before entering the data-read phase.
Reopening the file clears the binding and permits another slice.

Data I/O is skipped only if every rank has a cached decoded result. On an
asymmetric miss, every rank participates; ranks with existing published results
retain their allocations. An empty CSR rank still reads one terminal offset.
CloseFile clears both specialized caches without invalidating returned owners.

Sources: [SerializerBase.hpp:168](../../../src/DNDS/Serializer/SerializerBase.hpp#L168),
[SerializerBase.cpp:39](../../../src/DNDS/Serializer/SerializerBase.cpp#L39),
[SerializerH5.cpp:953](../../../src/DNDS/Serializer/SerializerH5.cpp#L953),
[SerializerJSON.cpp:237](../../../src/DNDS/Serializer/SerializerJSON.cpp#L237),
[ArrayTransformer.hpp:97](../../../src/DNDS/ArrayTransformer.hpp#L97).

### Boundary between Array and Serializer

Array still owns row counts, element values, compression, and structural
detachment. ParArray resolves the requested row partition and computes the
flat-data prefix sum for writes. The specialized serializer owns the translation
between local row starts and the stored representation, reference identity,
decoded-buffer lifetime, and the same-slice session contract. Generic vector
serialization does not acquire CSR semantics.

Normalization remains mathematically idempotent, but the original global base
is not recoverable from its normalized result. Therefore the cache retains the
base alongside the shared local vector; callers neither renormalize it nor try
to infer the data slice again. The new API avoids the sharing-breaking read
copy, not every temporary allocation on the write path.

## 4. Compatibility and limits

1. Dataset names, `::ref` / JSON `ref` encoding, and stored global/local row-start
   representations are unchanged. The new API can read existing representations.
2. Existing Array read/write signatures and generic vector APIs remain unchanged.
   The new row-start API deliberately restricts its own per-session slicing.
3. `getRowStart()` intentionally no longer permits mutable external handles.
   Third-party code that expected one must adopt read-only access.
4. The new virtual methods require external SerializerBase subclasses to provide
   their explicit row-start implementation; both in-repository backends do so.
5. Constness protects the public Array accessor, not deliberate casts, protected
   member access in derived implementations, or independently held mutable aliases.
6. No GPU execution, broad ownership redesign, or performance speedup is claimed.

## 5. Reproduction and validation

Evidence directory in the isolated checkout:
`workspace/audit_fix_validation/csr_shared_2026-09-28/`.

Before implementation, the added existing-API reproductions failed:

1. Array test: five failed type assertions (const iterator/accessor contracts).
2. Serializer round trip at two ranks: four failed pointer-identity assertions
   per rank, while the checked numerical contents remained correct.

After implementation, focused checks passed: array test 7 assertions;
four serializer cases at np=1 and np=2; complete serializer executable at np=2
with 90 cases per rank. Tests cover pointer identity, JSON/HDF5 round trips,
nonzero global offsets, raw/decoded separation, aliases, changed-slice rejection,
reopen, empty ranks, mixed writer identities, unequal cache histories, and
decompress/recompress detachment.

Tests: [test_Array.cpp](../../../test/cpp/DNDS/test_Array.cpp),
[test_Serializer.cpp](../../../test/cpp/DNDS/test_Serializer.cpp).

The first full run passed the 282-step CPU build, Python installation, all 59
CTest entries (np=1,2/OMP=1; all categories), and Python tests. Its two-rank Euler
partition-only write and HDF5-mesh-read/two-step run both exited zero. The initial
smoke comparison failed because it compared cell-array positions across different
mesh reorderings. A checker matching unique cell centers and cell types verified
exactly equal initial/final physical fields, with only `1.11e-16` maximum final
RHS difference. Original failed logs are retained; tolerances were not relaxed.

Additional final coverage explicitly enables collective HDF5 data I/O, tests
all-empty partitions, and reads the legacy raw-vector file representation using
the new API. Final post-format validation passed:

1. Complete configured CPU target build: 279 steps; fresh Python installation.
2. Full CTest: 59/59 entries, all categories, np=1,2 and OMP=1, 274.48 seconds.
   The count differs from the earlier 99-entry audit run because np=4,8 are
   intentionally not registered under the current two-rank allocation.
3. Full Python suite: 49 passed, one CUDA-only skip, 25.82 seconds including
   process overhead. The row-start memoryview remains read-only.
4. Focused final C++ tests: five serializer cases at np=1 (84 assertions) and
   np=2 (83/82 assertions per rank), plus one Array case (seven assertions).
5. Euler partition-only HDF5 write and HDF5-mesh-read/two-step smoke: both
   exited zero with np=2/OMP=1; combined driver 1.12 seconds. All fields finite,
   density/pressure positive, exact center/type correspondence, physical fields
   equal to the control and maximum RHS difference `1.11e-16`.

Evidence: `final/results.json`, `final/ctest.xml`, `final/pytest.xml`,
`smoke_final/manifest.json`, `smoke_final/checks.json`, and
`focused_final_*.log`, relative to the evidence directory above. `validate.py`
records commands, source state, limits, timings, and gate exit statuses;
`smoke.py` records executable/config/mesh hashes and raw-output paths.
The source/test diff SHA-256 before commit is
`24529baece4020c42700488ea55bec46a24446befda0ce12664dd0c8474c0c5f`.
Only this report changed during the final gates; source and tests did not.

The original checkout remains clean on `dev/harry` at
`f6ece7e91e97c6337ef0761c30017b2ddad6b24b`, matching the initial validation
manifest. External installation reuse remains via the isolated checkout's
`external/cfd_externals/install` link. No CUDA execution or long-horizon
numerical accuracy claim is made.
