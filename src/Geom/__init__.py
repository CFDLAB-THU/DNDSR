from __future__ import annotations
from multiprocessing import context
from ctypes import CDLL
import os, sys

__file_dir__ = os.path.dirname(os.path.realpath(__file__))
DNDSR_bin_dir = os.path.realpath(os.path.join(__file_dir__, "..", "bin"))
DNDSR_libext_dir = os.path.realpath(
    os.path.join(__file_dir__, "..", "lib", "dndsr_external")
)
DNDSR_lib_dir = os.path.realpath(os.path.join(__file_dir__, "..", "lib"))
DNDSR_mod_dir = os.path.realpath(os.path.join(__file_dir__, ".."))

sys.path.append(DNDSR_mod_dir)
import DNDS

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

        # print(f"here {DNDSR_bin_dir}")
        # os.system(f"ls -la {DNDSR_bin_dir}")
        # name = os.path.join(DNDSR_bin_dir, "libdnds_shared.so")
        # os.system(f"ldd { name }")
        # while True:
        #     pass
        # CDLL(os.path.join(DNDSR_lib_dir, "libdnds_shared.so"))
        CDLL(os.path.join(DNDSR_lib_dir, "libgeom_shared.so"))
        # os.environ["LD_LIBRARY_PATH"] = (
        #     DNDSR_bin_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        # )
        pass
    elif os.name == "nt":
        raise RuntimeError("not yet implemented")
        pass


_pre_import()

if __name__ == "__main__":
    from _internal.geom_pybind11 import *
    from _internal.geom_pybind11 import Elem
else:
    from ._internal.geom_pybind11 import *
    from ._internal.geom_pybind11 import Elem
    print(f"module load: {__file__}")


__all__ = ["Elem"]

if __name__ == "__main__":
    print(__all__)
