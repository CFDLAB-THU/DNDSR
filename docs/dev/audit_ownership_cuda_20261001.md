# Allocation ownership and CUDA bounds: audit findings 14 and 17

This follow-up is based on `ad24339a`, on `fix/dnds-geom-audit-20260926` in the
isolated audit worktree. Original `dev/harry` is not modified. Linked externals
are reused; builds, installed Python modules and solver outputs stay local to
the fix worktree. Serializer implementations are unchanged in this round.

## Changes

1. `host_device_vector` shares allocation ownership through aliasing host/device
   leases. Structural replacement creates a new allocation manager instead of
   freeing leased bytes. The existing cached pointers remain the element-access
   path; deep copies remain deep. Host device leases retain their host allocation.
2. Uncompressed CSR uses lease-aware `RowStorage<T>` rows with cached pointer and
   size. Structural mutation detaches a leased row, preserving `std::vector`
   resize behavior. Hashing continues to hash elements, including non-arithmetic
   values, not ownership-wrapper bytes.
3. Array and derived Python bindings export an allocation-owning buffer object
   with stable shape, strides and format. Views survive replacement/deletion,
   retain write-through while sharing storage, and release storage after the
   final view. Row metadata stays read-only; matrix-list items own their leases.
4. Three CUDA ArrayDof kernels now reject the equality boundary using `>=`.
   Numerical expressions and launch sizes are unchanged.

See [view lifetime contract and retention warning](../guides/python_array_views.md).
Small views may retain whole flat allocations. Raw C++ maps/device views remain
borrowed; allocation leases do not add MPI-request or asynchronous-kernel
synchronization. Structural operations acquire additional ownership bookkeeping;
no performance result is claimed. External code naming the uncompressed storage
implementation alias may need to adapt from vector rows to `RowStorage<T>`.

## Regressions and evidence

- C++ host/Host-backend leases: writes, resize, clear, copy, move, swap,
  assignment, final-owner release; CSR reserve/resize/compress/decompress/deletion;
  non-arithmetic CSR hash control.
- Python: 13 new ownership cases for flat data, read-only row starts, compressed
  and uncompressed rows, vector/matrix/batch getters, NumPy/derived views, empty
  exports, Array collection, and final buffer-owner collection.
- CUDA: father/ghost extents at zero, singleton, and around 32/128 launch
  boundaries; host-result comparisons and a device lease retained across
  replacement/clear. Four cases, 7,933 assertions per rank.

Old-guard controls fail for all three CUDA kernels. These controls use the new
harness/ownership code with original guards, not a pristine baseline checkout.
The lifetime defect is source-confirmed with passing retained-view controls;
no deterministic old-allocation runtime reproduction is claimed.

CUDA 13.2 / sm80 build uses the same installed MPI include directory explicitly
for nvcc (the MPI wrapper did not propagate it). Final memory checking uses
`--report-api-errors explicit` and `OMPI_MCA_accelerator=null`; all kernels are
checked. The default extended-API attempt is retained: all cases passed, but
CUB duplicate kernel registration and OpenMPI finalization produced API reports.
This does not claim CUDA-aware MPI validation or unrestricted extended-API zero
diagnostics. The ordinary CUDA CTest run also passes with default MPI settings.

The complete terminal gate results are recorded in the independent review
repository's `OWNERSHIP_CUDA_FOLLOWUP_20261001.md`. Raw logs and reproducible
orchestration are under the fix worktree's independent
`workspace/audit_fix_validation/ownership_2026-10-01/` repository directory.
The first full run passed 59/59 CPU CTest entries and Python 62 passed/one CUDA
skip, followed by the bounded two-rank Euler HDF5 write/read smoke. Because a
generic hash compatibility edit occurred during that build, the final formatted
source was rebuilt and all gates repeated: 213 build steps (467.63 s), fresh
installation (4.27 s), CTest 59/59 (285.94 s), Python 62 passed/one skip (30.38 s
including process overhead), and smoke (1.12 s including both invocations and
checks). Final physical fields match the control exactly; maximum RHS difference
is `1.11e-16`. The three focused C++ ownership cases pass 29 assertions; Python
ownership passes all 13 cases independently on each of two MPI ranks.

The verified source/test diff against `ad24339a` has SHA-256
`10c7ff6f95b0fe1a1505232ff658a12f17b4cc52515165e43546bf24a8d8a2c7`.
`record_final.py` checks this identity, every terminal gate, the CUDA executable
hash and original checkout identity. CUDA up-to-date confirmation reports no
work; its tested executable SHA-256 is
`d2477301b23298e7c8b8b22ccfb84525c1896e1f83b884a8e4e996b99ec02e84`.

The numerical-experiment protocol bounds the Euler smoke to a 100-cell periodic
case, two physical steps of `1e-4`, eight inner iterations per stage and a
120-second cap per command. It checks finite fields, positive density/pressure
and agreement with the prior control after exact unique-cell matching. This is
execution/regression evidence, not an accuracy, convergence or performance study.

## Reproduction

From the isolated worktree (CPU tests configured for np1/2 and OMP1):

```sh
DNDS_TEST_NP_LIST='1;2' DNDS_TEST_OMP_THREADS=1 cmake -S . -B build/fix-cpu-ninja
cmake --build build/fix-cpu-ninja --target all_unit_tests dnds_pybind11 \
  geom_pybind11 cfv_pybind11 eulerP_pybind11 euler -j16
cmake --install build/fix-cpu-ninja --component py
OMP_NUM_THREADS=1 ctest --test-dir build/fix-cpu-ninja --output-on-failure -j1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python \
  venv/bin/python -m pytest test/ --timeout=300
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OMPI_MCA_accelerator=null \
  mpirun --oversubscribe -np 2 compute-sanitizer --tool memcheck \
  --report-api-errors explicit --error-exitcode 99 \
  build/fix-cuda-ownership/test/cpp/dnds_test_array_dof_cuda
```

CUDA runtime tests require an available GPU; select it explicitly. Retain logs
and use a new smoke suffix/config output directory, because the smoke scripts
refuse to overwrite earlier output.
