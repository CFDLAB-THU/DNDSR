from __future__ import annotations
from ctypes import CDLL
import os

DNDSR_bin_dir = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "bin")
)
DNDSR_lib_dir = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib")
)


if os.name == "posix":
    CDLL(os.path.join(DNDSR_bin_dir, "libfmt.so"))
    
    CDLL(os.path.join(DNDSR_lib_dir, "libz.so"))
    CDLL(os.path.join(DNDSR_lib_dir, "libhdf5.so"))
    CDLL(os.path.join(DNDSR_lib_dir, "libcgns.so"))
    CDLL(os.path.join(DNDSR_lib_dir, "libmetis.so"))
    CDLL(os.path.join(DNDSR_lib_dir, "libparmetis.so"))
    
    CDLL(os.path.join(DNDSR_bin_dir, "libdnds_shared.so"))
    # os.environ["LD_LIBRARY_PATH"] = (
    #     DNDSR_bin_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    # )
    pass

if __name__ == "__main__":
    from _internal import dnds_pybind11 as core
else:
    from ._internal import dnds_pybind11 as core

# List the symbols you want to expose from core
__all__ = dir(core)

# Expose symbols from core in the current module's namespace
for symbol in __all__:
    if symbol != "__name__":
        globals()[symbol] = getattr(core, symbol)

if __name__ == "__main__":
    print(dir(core))
