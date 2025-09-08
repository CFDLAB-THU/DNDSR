from __future__ import annotations
from ctypes import CDLL
import os, sys

__file_dir__ = os.path.dirname(os.path.realpath(__file__))
DNDSR_bin_dir = os.path.realpath(os.path.join(__file_dir__, "..", "bin"))
DNDSR_libext_dir = os.path.realpath(
    os.path.join(__file_dir__, "..", "lib", "dndsr_external")
)
DNDSR_lib_dir = os.path.realpath(os.path.join(__file_dir__, "..", "lib"))
DNDSR_mod_dir = os.path.realpath(os.path.join(__file_dir__, ".."))

# !never do this! meddling with package name
# sys.path.append(DNDSR_mod_dir)
# import DNDS
# import CFV


def _pre_import():

    # print(DNDS.__file__)

    if os.name == "posix":

        # print(f"here {DNDSR_bin_dir}")
        # os.system(f"ls -la {DNDSR_bin_dir}")
        # name = os.path.join(DNDSR_bin_dir, "libdnds_shared.so")
        # os.system(f"ldd { name }")
        # while True:
        #     pass
        # CDLL(os.path.join(DNDSR_libext_dir, "libz.so"))
        # CDLL(os.path.join(DNDSR_libext_dir, "libhdf5.so"))
        # CDLL(os.path.join(DNDSR_libext_dir, "libcgns.so"))
        # CDLL(os.path.join(DNDSR_libext_dir, "libmetis.so"))
        # CDLL(os.path.join(DNDSR_libext_dir, "libparmetis.so"))

        # CDLL(os.path.join(DNDSR_lib_dir, "libdnds_shared.so"))
        # CDLL(os.path.join(DNDSR_lib_dir, "libgeom_shared.so"))
        CDLL(os.path.join(DNDSR_lib_dir, "libcfv_shared.so"))
        pass
        
    elif os.name == "nt":
        raise RuntimeError("not yet implemented")
        pass


_pre_import()

if __name__ == "__main__":
    from _internal.cfv_pybind11 import *
else:
    from ._internal.cfv_pybind11 import *

    print(f"module load: {__file__}")


__all__ = ["Elem"]

if __name__ == "__main__":
    print(__all__)
