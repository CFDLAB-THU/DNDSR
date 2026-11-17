# Source-scoped device assertions

Configure `-DDNDS_CUDA_KERNEL_ASSERTS=OFF` to remove device assertions only
from CFV's explicitly listed `Operators/CUDA/*.cu` sources and Cauchy's
explicit CUDA kernel catalog (including generated RHS wrappers). The default
is `ON`. CMake attaches `DNDS_NDEBUG_DEVICE=1` as a **source property**;
it is not a global, directory, target or public interface definition.
Consequently a toggle rebuilds these CUDA objects and their link consumers,
without recompiling unrelated CPU, Python-binding or diagnostic CUDA sources.
Static/shared variants of a selected source receive the same setting.

The device assertion sites include:

1. `ArrayView::Row`, `RowSize`, `operator()` and `RowSpan::operator[]` bounds
   when the view's `Checked` template argument is true. Default views follow
   `DNDS_RUNTIME_ARRAY_BOUNDS_CHECKS` (currently OFF); explicit checked views
   can still contain checks. A static row shape does not inherently disable
   bounds checks.
2. `SpanView` construction/subspan invariants and optionally element bounds;
   mesh adjacency/index-state/periodic conventions; polynomial and model
   helper invariants that use `DNDS_HD_assert*`.
3. Eigen's device assertion hook, which DNDS maps to `DNDS_HD_assert`.

These all use the existing leveled error machinery. The selected source's
device pass compiles them out at every assertion level; the **host pass stays
checked**. An assertion expression must never have required side effects.
`DNDS_ASSERT_LEVEL` still provides finer global levels, but the ordinary
MAX-level invariants are deliberately not stripped by lowering verbosity.
There is no second, competing `DNDS_CUDA_DISABLE_ASSERTS` macro.

This option does not remove public host shape/residency/lease validation,
CUDA API error checks, explicit numerical error codes, finite/PP tests or
third-party libraries' independent checks. It is not a mechanism for removing
required numerical admissibility behavior. DNDS Runtime communication/algebra,
Geom, legacy CUDA, tests and separately built AMGX remain outside its scope.
To opt another source in, register it explicitly with
`dnds_cuda_kernel_assertions(...)` in its owning CMake directory.

The source-property regression tests both settings and a return to ON,
including generated source names and untouched host/test sources.
