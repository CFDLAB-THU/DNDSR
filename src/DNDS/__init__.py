from __future__ import annotations

from ._internal import dnds_pybind11 as core


# List the symbols you want to expose from core
__all__ = dir(core)

# Expose symbols from core in the current module's namespace
for symbol in __all__:
    globals()[symbol] = getattr(core, symbol)


print(dir(core))
