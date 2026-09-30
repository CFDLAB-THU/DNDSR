# Array views and allocation lifetime

DNDS Array `data()`, row/matrix/batch getters, `getRowStart()` and
`getRowSizes()` return zero-copy Python memoryviews. Each exported view retains
its backing allocation, not the Array object. NumPy views and memoryview slices
retain that same allocation through the Python buffer protocol.

**A small view can keep a large allocation alive.** For example, one matrix in
a compressed CSR array retains that array's entire flat host buffer. Use
`np.asarray(view).copy()` when a small long-lived result should be independent,
then release all original views. Retention until the last view is released is
intentional, not an allocation leak. Uncompressed CSR views retain their row.

## Mutation contract

- Ordinary element writes are shared: writing a writable view updates the
  corresponding Array while both still refer to the same allocation.
- A storage-replacing operation (resize, compression/decompression, assignment,
  or deletion) does not invalidate an existing exported view. The view stays
  attached to the old allocation, with its original shape and strides; it does
  **not** follow the Array's replacement storage.
- Uncompressed row resizing/reserving detaches a leased row when it changes
  its structure/allocation. A same-size resize or reserve within capacity may
  leave storage unchanged. A no-op operation need not detach.
- Move/swap transfers storage to another holder; an existing view still points
  to those same bytes. It is not an immutable snapshot.
- Row-start and row-size metadata views are read-only. Writable data views
  remain writable after their Array is deleted.
- Shared ownership does not serialize simultaneous element writes or structural
  mutation; callers must synchronize those operations themselves.

```python
view = array.data()
old = np.asarray(view)          # shares the current allocation
array.Resize(new_size)         # replaces the Array's allocation
old[0] = 1                     # updates the old allocation, not new Array data
saved = old.copy()             # independent result
del old, view                  # releases these references to the old allocation
```

Each memoryview owns a private buffer exporter containing an allocation lease
and stable format/shape/stride metadata. It does not retain the Array's MPI
state, unrelated rows, or a device mirror. Individual matrix views returned in
a list each own their lease, so retaining one list item is sufficient.

This contract covers Array and its adjacency, Eigen-vector, Eigen-matrix, and
matrix-batch bindings, including pair getters forwarding to those bindings.
Unrelated bindings exporting ordinary `std::vector` objects are not changed.

## C++ and device views

`host_device_vector` preserves its cached `host_ptr` and `device_ptr` access
paths. No reference-count operation is added to element access. `hostLease()`
and `deviceLease()` explicitly return aliasing shared pointers retaining the
specific allocation; only constructing/copying/destroying leases affects its
ownership count. Deep copies of the vector remain deep copies.

Only the owning vector can replace its storage. A lease exposes elements, not
the allocation manager's allocate/free interface. Host-backend device leases
retain the actual host allocation, because that backend is only an alias.
CUDA leases retain the CUDA allocation independently of the host allocation.
Vector resize creates a fresh allocation manager; this adds structural-operation
allocation overhead, not an extra indirection to the existing hot accessors.
Uncompressed CSR uses a lease-aware row wrapper with cached pointer and size.

`Array::rowLease(i)` and `dataLease()` supply allocation-lifetime tokens for
long-lived host consumers. Raw C++ pointers, Eigen maps, and `deviceView()`
remain borrowed and non-owning. For asynchronous kernels, callers must retain
every referenced allocation (including layout metadata) until work completes.
A lease adds no implicit synchronization, stream tracking, or MPI-request
lifetime management. Do not resize/release borrowed storage while it is in use.

The public uncompressed-storage type alias now contains `RowStorage<T>` rather
than `std::vector<T>` rows. Normal Array APIs are unchanged; external code naming
or manipulating that implementation alias directly may need adaptation.
