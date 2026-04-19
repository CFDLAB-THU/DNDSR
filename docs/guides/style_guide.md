# Style Guide {#style_guide}

This page collects the detailed coding conventions for DNDSR. It complements
the short guardrails in `AGENTS.md`.

## C++

### Formatting

- Use Allman braces (opening brace on its own line).
- Use `#pragma once` in headers; no `#ifndef` guards.
- Preserve include order; do not auto-sort includes.

The clang-format configuration lives at `src/.clang-format`.

```bash
clang-format -i src/DNDS/SomeFile.hpp
```

### Include Order

1. Project macros (`"DNDS/Macros.hpp"`, `"DNDS/Defines.hpp"`)
2. Standard library headers
3. External library headers (Eigen, fmt, nlohmann_json, etc.)
4. Project headers (quoted, relative to `src/`): `"DNDS/Array.hpp"`, `"Geom/Mesh.hpp"`

### Naming Conventions

| Element                 | Convention       | Example                              |
|-------------------------|------------------|--------------------------------------|
| Namespace               | `PascalCase`     | `DNDS`, `DNDS::Geom`, `DNDS::CFV`   |
| Class / Struct          | `PascalCase`     | `MPIInfo`, `VariationalReconstruction` |
| Public method           | `PascalCase`     | `Resize()`, `ConstructMetrics()`     |
| Private/protected member| `_` prefix       | `_size`, `_data`, `_pRowStart`       |
| Type alias              | `t_` prefix or `using` | `t_IndexVec`, `tDiFj`          |
| Template parameter      | `T`-prefix or PascalCase | `T`, `TOut`, `_row_size`      |
| Constant                | `PascalCase`     | `UnInitReal`, `DynamicSize`          |
| Macro                   | `DNDS_ALL_CAPS`  | `DNDS_INDEX_MAX`, `DNDS_MPI_REAL`    |
| Enum value              | `PascalCase`     | `UnknownElem`, `Line2`, `Roe`        |

### Core Type Aliases

From `Defines.hpp`:

```cpp
using real = double;
using index = int64_t;
using rowsize = int32_t;
template <typename T> using ssp = std::shared_ptr<T>;
```

### Error Handling

Use the project's assert/check macros from `DNDS/Errors.hpp`:

```cpp
DNDS_assert(expr);                         // debug-only, calls std::abort()
DNDS_assert_info(expr, info_string);       // debug-only with message
DNDS_assert_infof(expr, fmt_string, ...);  // debug-only with printf format
DNDS_check_throw(expr);                    // always active, throws std::runtime_error
DNDS_check_throw_info(expr, info_string);  // always active with message
```

Avoid raw `assert()`. Use `DNDS_assert` for debug checks and
`DNDS_check_throw` for runtime validation that must remain in release builds.

### Templates and Eigen

- Prefer `Eigen::Matrix<real, ...>` with the project's `real` type.
- Use `if constexpr` for compile-time branching on template parameters.
- Explicit template instantiation goes in `_explicit_instantiation/` subdirectories.

### clang-tidy

The clang-tidy configuration lives at `src/.clang-tidy`.
Enabled check groups include: `modernize-*`, `readability-*`, `bugprone-*`,
`performance-*`, `cppcoreguidelines-*`, `google-build-using-namespace`,
`mpi-*`, `openmp-*`.

### C++ Docstrings (Doxygen)

Source comments are processed by **Doxygen** (for the standalone C++ API docs)
and by **Breathe** (to embed them in the Sphinx site). Both systems read the
same `///` and `/** */` comment blocks. The rules below ensure cross-references
work in both outputs.

#### Basic structure

Use `///` for single-line doc comments and `/** */` for multi-line blocks.
Always document the *why*, not just the *what*.

```cpp
/// @brief One-line summary of the function.
void simpleFunction();

/**
 * @brief Compress the CSR storage into a flat buffer.
 *
 * @details After this call, data is contiguous in `_data` and
 * indexed by `_pRowStart`.  Required before MPI communication,
 * CUDA transfer, or serialization.
 *
 * @pre  Array must be in decompressed state.
 * @post `IsCompressed()` returns true.
 *
 * @sa Decompress, ResizeRow
 */
void Compress();
```

#### Cross-referencing classes and structs

Use `@ref DNDS::ClassName "ClassName"` for fully-qualified references.
Doxygen resolves these as hyperlinks in its HTML output. Breathe (Sphinx)
also resolves them when the qualified name matches a documented symbol.

```cpp
/// @brief MPI-aware @ref DNDS::Array "Array" with global index mapping.
///
/// Layers an @ref DNDS::MPIInfo "MPIInfo" context and a
/// @ref DNDS::GlobalOffsetsMapping "GlobalOffsetsMapping" on top of
/// @ref DNDS::Array "Array".
template <class T, rowsize _row_size = 1>
class ParArray : public Array<T, _row_size> { ... };
```

**Why the `"DisplayText"` part?** Without it, Doxygen renders the full
qualified name `DNDS::MPIInfo` as the link text, which is verbose.
`@ref DNDS::MPIInfo "MPIInfo"` shows just `MPIInfo` as the link label.

#### Cross-referencing methods, macros, and enums

For members of the *same class* or symbols in the *same namespace*, use
unqualified `@ref Name`:

```cpp
/// @brief Layout-polymorphic compress: no-op for non-CSR,
/// calls @ref CSRCompress for CSR.
/// After this, @ref ResizeRow and @ref ReserveRow may be used.
void Compress();
```

Doxygen resolves unqualified names contextually (within the enclosing
class/namespace). Breathe cannot resolve these, so they render as plain
text in Sphinx — but the Doxygen HTML (served at `/doxygen/`) has
working links.

For macros, use `@ref DNDS_assert` (macros live in the global scope):

```cpp
/// Uses @ref DNDS_assert for debug-only bounds checking.
/// In release builds, use @ref DNDS_check_throw instead.
```

#### What NOT to cross-reference

Keep these as backtick inline code (no `@ref`):

- **External API names**: `` `MPI_Allreduce` ``, `` `Eigen::Matrix` ``,
  `` `std::vector` ``
- **Layout names used as prose**: `` `TABLE_Fixed` ``, `` `CSR` ``,
  `` `TABLE_StaticMax` ``
- **Template parameters**: `` `_row_size` ``, `` `T` ``
- **Code snippets**: `` `Resize(100)` ``, `` `operator()` ``

```cpp
/// @brief Width used by row `iRow` in number of `T` elements.
/// - `TABLE_*Fixed`: returns the uniform row width;
/// - `TABLE_*Max`:   returns `_pRowSizes->at(iRow)`;
/// - `CSR`:          returns `pRowStart[iRow+1] - pRowStart[iRow]`.
```

#### Common Doxygen commands

| Command | Purpose | Example |
|---------|---------|---------|
| `@brief` | One-line summary | `@brief Resize the array.` |
| `@details` | Extended description | Multi-paragraph explanation |
| `@param name` | Document a parameter | `@param mpi  The MPI context.` |
| `@tparam T` | Document a template param | `@tparam T  Element type.` |
| `@return` | Document the return value | `@return Number of rows.` |
| `@pre` / `@post` | Pre/post conditions | `@pre Array is compressed.` |
| `@sa` | "See also" links | `@sa Compress, Decompress` |
| `@ref DNDS::Foo "Foo"` | Cross-ref to class/struct | See above |
| `@ref Bar` | Cross-ref to local symbol | See above |
| `@note` | Highlighted note | `@note Thread-safe.` |
| `@warning` | Highlighted warning | `@warning Not MPI-safe.` |
| `@code{.cpp}` ... `@endcode` | Code block | Fenced example |

## Python

### Naming

- Functions/variables: `snake_case` (`test_all_reduce_scalar`, `meshFile`).
- Classes wrapping C++ types: match the C++ name (`MPIInfo`, `VariationalReconstruction_2`).
- Private helpers: `_` prefix (`_pre_import`, `_init_mpi`).

### Imports

```python
from __future__ import annotations  # when used
import sys, os
from DNDSR import DNDS, Geom, CFV, EulerP
import numpy as np
import pytest
```

### Testing Patterns

- Use `@pytest.fixture` for MPI setup.
- Use plain `assert` statements (not unittest-style).
- Tests may run standalone via `if __name__ == "__main__":` blocks.
- Use numpy for array comparisons: `assert np.all(...)`, `assert (arr == val).all()`.

### Doctest and POSIX `index()`

When writing new C++ tests with `using namespace DNDS;`, qualify
`DNDS::index`, `DNDS::real`, and `DNDS::rowsize` in declarations to avoid
ambiguity with POSIX `index()` from `<strings.h>` (pulled in by doctest).
