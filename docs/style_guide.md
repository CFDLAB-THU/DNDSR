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
