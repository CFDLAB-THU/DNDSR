from __future__ import annotations

def _pre_import():
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
    elif os.name == 'nt':
        raise RuntimeError("not yet implemented")
        pass

_pre_import()

if __name__ == "__main__":
    from _internal.dnds_pybind11 import *
    from _internal.dnds_pybind11 import MPI
    from _internal.dnds_pybind11 import Debug
else:
    from ._internal.dnds_pybind11 import *
    from ._internal.dnds_pybind11 import MPI
    from ._internal.dnds_pybind11 import Debug

# List the symbols you want to expose from core
# __all__ = dir(code)

# Expose symbols from core in the current module's namespace
# for symbol in __all__:
#     if symbol != "__name__":
#         globals()[symbol] = getattr(core, symbol)

def _import_submodules():
    import sys
    sys.modules[__name__ + ".MPI"] = MPI
    sys.modules[__name__ + ".Debug"] = Debug

__all__ = ["MPI", "Debug"]

if __name__ == "__main__":
    print(__all__)
